"""CP2 before/after analysis (plan Amendment A5). Offline, $0, from runs/cache + runs/embeddings.

Judges compared on the same pairs:
  before: LLM-1 = CP1 baseline LLM pairwise judge as deployed (one order per call; per-call average over both orders)
          J4    = CP1 classifier on frozen UIClip embeddings
  after:  LLM-2i / LLM-2ii = LLM in both orders (tie if the orders disagree), note arm (i) pixels only / (ii) sanitized
          J4H   = classifier retrained with build-split exploit augmentation;  J4crop = J4 on located crops (secondary)
          COMB-i / COMB-ii = combined judge (rule chosen on build pairs only)
Lift(e) = P(human loser wins | edit) - P(... | control rep 0); CIs = cluster bootstrap over both items' groups.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from . import config, harden, judges, metrics as M, stages
from .analyze import PairGroups, group_map, _jsonable

EXPLOITS = ("E1", "E2", "E3", "E4a", "E4b", "E5")
BIASES = ("R2", "R3", "R4")
IMAGE_EDITS = ("E1", "E2", "E3", "E4a", "E5", "R4")
NOTE_EDITS = ("R2", "R3", "E4b")
SETS = ("vda_clear", "vda_close", "gde_close")
RES = config.RESULTS / "cp2"


def _f(x):
    return None if x is None or (isinstance(x, float) and x != x) else round(float(x), 4)


# ---------------------------------------------------------------------------------------------
def load(stage_names=("main", "cp2", "cp2s")):
    recs = {}
    for st in stage_names:
        p = config.RUNS / "manifests" / f"{st}.json"
        if not p.exists():
            continue
        man = json.loads(p.read_text())
        for cid, h in zip(man["custom_ids"], man["request_hashes"]):
            cp = config.CACHE / f"{cid}.json"
            if cp.exists():
                rec = json.loads(cp.read_text())
                if rec.get("request_hash") == h:
                    recs[cid] = rec
    return list(recs.values())


class Calls:
    """pairwise calls: (dataset, model, test, rep, pair) -> {order: picked pair[1]?}"""

    def __init__(self, recs):
        self.d = defaultdict(dict)
        self.invalid = 0
        for r in recs:
            m = r["meta"]
            if m.get("judge") != "J2":
                continue
            v = judges.parse(r)
            if v is None:
                self.invalid += 1
                continue
            key = tuple(m["pair"])
            picked = m["a"] if v["winner"] == "A" else m["b"]
            self.d[(m["dataset"], m["model_key"], m["test"], m["rep"], key)][m["order"]] = 1.0 if picked == key[1] else 0.0

    def get(self, ds, model, test, rep=0):
        return {k[4]: v for k, v in self.d.items() if k[:4] == (ds, model, test, rep)}


def single_order(calls: dict) -> dict:
    """before: per-call P(pick pair[1]) averaged over the orders present."""
    return {k: [float(np.mean(list(v.values())))] for k, v in calls.items() if v}


def both_order(calls: dict) -> dict:
    """after: 1 if pair[1] picked in both orders, 0 if pair[0] in both, 0.5 if they disagree (needs both orders)."""
    out = {}
    for k, v in calls.items():
        if len(v) == 2:
            s = sum(v.values())
            out[k] = [1.0 if s == 2 else 0.0 if s == 0 else 0.5]
    return out


# ---------------------------------------------------------------------------------------------
def classifiers(ds, extra_emb: dict | None = None):
    """Returns per dataset: {'J4': score(item, cond), 'J4H': ..., 'J4crop': ...} and fit info.
    extra_emb (CP3): {dataset: {(item, cond): embedding}} scored with the same fitted weights; never used in fitting."""
    from . import clip_arm, embed_run
    out, info = {}, {}
    for name in ("vda", "gde"):
        items, pairs = ds[name]
        z = np.load(embed_run.emb_path(name))
        emb = {tuple(k.split("|", 1)): z[k] for k in z.files}
        base = {i: v for (i, c), v in emb.items() if c == "base"}
        groups = {i: it.group for i, it in items.items()}
        if name == "vda":
            build = [p for p in pairs if p.split == "build"]
            w, i1 = clip_arm.fit_bt(base, build, groups)
            wh, i2 = harden.fit_bt_augmented(emb, build, groups)
            s4 = {k: float(v @ w) for k, v in emb.items()}
            s4h = {k: float(v @ wh) for k, v in emb.items()}
            build_items = sorted({p.a for p in build} | {p.b for p in build})
            oof = harden.oof_bt_scores(emb, build, groups, build_items, i2["C"])
        else:
            target = {i: it.human["mean"] for i, it in items.items()
                      if it.split == "build" and i in base and it.human["mean"] == it.human["mean"]}
            (w, b), i1 = clip_arm.fit_ridge(base, target, groups)
            (wh, bh), i2 = harden.fit_ridge_augmented(emb, target, groups)
            s4 = {k: float(v @ w + b) for k, v in emb.items()}
            s4h = {k: float(v @ wh + bh) for k, v in emb.items()}
            oof = _oof_ridge(emb, target, groups, i2["alpha"])
        for k, v in (extra_emb or {}).get(name, {}).items():
            assert k not in emb, f"extra embedding {k} collides with a fitted/scored one"
            s4[k] = float(v @ w) if name == "vda" else float(v @ w + b)
            s4h[k] = float(v @ wh) if name == "vda" else float(v @ wh + bh)
        crop = {(i, c.split(":", 1)[1]): s for (i, c), s in s4.items() if c.startswith("crop:")}
        out[name] = {"J4": s4, "J4H": s4h, "J4crop": crop, "J4H_oof_build": oof}
        info[name] = {"J4": i1, "J4H": i2}
    return out, info


def _oof_ridge(emb, target, groups, alpha):
    from sklearn.linear_model import Ridge
    out = {}
    gs = sorted({groups[i] for i in target})
    for g in gs:
        tr = [i for i in target if groups[i] != g]
        X, y = [], []
        for i in tr:
            for c in harden.AUG_CONDS:
                if (i, c) in emb:
                    X.append(emb[(i, c)])
                    y.append(target[i])
        m = Ridge(alpha=alpha).fit(np.stack(X), y)
        for i in target:
            if groups[i] == g:
                out[i] = float(emb[(i, "base")] @ m.coef_ + m.intercept_)
    return out


def cls_pick_second(sc: dict, pair, cx, cy) -> list | None:
    """Deterministic classifier decision for pair (x with cx, y with cy): P(pick y)."""
    x, y = pair
    a, b = sc.get((x, cx)), sc.get((y, cy))
    if a is None or b is None or a != a or b != b:
        return None
    return [1.0 if b > a else 0.0 if b < a else 0.5]


def combined_pick_second(rule, stack, sc, pair, cx, cy, calls_pair) -> list | None:
    if calls_pair is None or len(calls_pair) != 2:
        return None
    x, y = pair
    a, b = sc.get((x, cx)), sc.get((y, cy))
    if a is None or b is None:
        return None
    vote_x = 1.0 - (sum(calls_pair.values()) / 2.0)        # fraction of orders that picked x
    vote_x = 1.0 if vote_x == 1.0 else 0.0 if vote_x == 0.0 else 0.5
    p_x = harden.decide(rule, a - b, vote_x, stack)
    return [1.0 - p_x]


# ---------------------------------------------------------------------------------------------
def choose_rules(ds, C: Calls, cls):
    out = {}
    for name in ("vda", "gde"):
        items, pairs = ds[name]
        oof = cls[name]["J4H_oof_build"]
        calls = C.get(name, "haiku", "W.build")
        rows = []
        for p in pairs:
            if p.split != "build" or (p.a, p.b) not in calls or len(calls[(p.a, p.b)]) != 2:
                continue
            if p.a not in oof or p.b not in oof:
                continue
            v = calls[(p.a, p.b)]
            vote_a = 1.0 - sum(v.values()) / 2.0
            vote_a = 1.0 if vote_a == 1.0 else 0.0 if vote_a == 0.0 else 0.5
            rows.append((oof[p.a] - oof[p.b], vote_a, p.share_a))
        out[name] = harden.choose_rule(rows) if rows else None
    return out


# ---------------------------------------------------------------------------------------------
NO_NOTE = "<no-note base pair, W.agree>"


def family_tests(set_name, aliased: bool):
    """Test names per edit: raw (CP1 baseline LLM, notes as written) and san (arm ii) edit/control tests.
    aliased (A6/D1): this dataset's sanitized control note is empty, so arm (ii) image families and the arm (ii) control
    are byte-identical to pixels-only requests and are read from the arm (i) calls."""
    ds = "gde" if set_name == "gde_close" else "vda"
    cp1 = {"vda_clear": "W.flip", "gde_close": "W.flipclose"}.get(set_name)
    f2 = f"F2.{set_name}"
    spec = {}
    for e in IMAGE_EDITS:
        ctl = "C2" if e == "E2" else "C1"
        raw = (f"{cp1}.{e}", f"{cp1}.ctl.{ctl}") if cp1 else (f"{f2}.{e}", f"{f2}.ctl.{ctl}")
        san = raw if aliased else (f"{f2}.{e}s", f"{f2}.ctl.{ctl}s")
        spec[e] = {"kind": "image", "cx": ctl, "cy": e, "raw": raw, "san": san}
    for e in NOTE_EDITS:
        if cp1 and e == "E4b":
            raw = (f"{cp1}.E4b", f"{cp1}.ctl.N0")            # both from the CP1 batch
        else:
            raw = (f"{f2}.{e}", f"{f2}.ctl.N0")               # same CP2 batch (A6/D3)
        san = (f"{f2}.{e}s", NO_NOTE if aliased else f"{f2}.ctl.N0s")
        spec[e] = {"kind": "note", "cx": "base", "cy": "base", "raw": raw, "san": san}
    return ds, spec


def _no_note_calls(C, dname, model, xy, rep=0):
    """Both-order no-note calls for each (x, y), re-oriented from W.agree's (p.a, p.b) keys (rep 0) or W.repeat (rep 1)."""
    src = C.get(dname, model, "W.agree" if rep == 0 else "W.repeat", rep)
    out = {}
    for k in xy:
        if k in src:
            out[k] = src[k]
        elif (k[1], k[0]) in src:
            out[k] = {o: 1.0 - v for o, v in src[(k[1], k[0])].items()}
    return out


def _calls(C, dname, model, test, xy, rep=0):
    if test == NO_NOTE:
        return _no_note_calls(C, dname, model, xy, rep)
    return C.get(dname, model, test, rep)


def _judge_vals(judge, calls, k, sc, cx, cy, rule):
    """P(loser wins) for one pair under one judge, from that family's calls (LLM) and/or classifier scores."""
    c = calls.get(k)
    if judge == "LLM-1":
        return single_order({k: c})[k] if c else None
    if judge in ("LLM-2i", "LLM-2ii"):
        return both_order({k: c}).get(k) if c else None
    if judge.startswith("COMB"):
        if not rule:
            return None
        return combined_pick_second(rule["rule"], rule["stack"], sc, k, cx, cy, c)
    return cls_pick_second(sc, k, cx, cy)


def _band(v0: dict, v1: dict, pg, n_target: int | None = None) -> float:
    """95th percentile |A/A effect| (rep 1 vs rep 0 of identical requests); optionally rescaled to the target n."""
    keys = [k for k in v0 if k in v1]
    if len(keys) < 5:
        return float("nan")
    aa = M.paired_effect({k: v1[k] for k in keys}, {k: v0[k] for k in keys}, pg)
    b = aa["boot"][~np.isnan(aa["boot"])]
    band = float(np.quantile(np.abs(b), 0.95))
    return band * np.sqrt(len(keys) / n_target) if n_target else band


def no_note_band(C, rule, sch, judge, pg, n_target, model="haiku"):
    """A6/N2: noise band for no-note controls from ALL W.repeat pairs (W.agree rep 0 vs W.repeat rep 1), scaled to n."""
    r0, r1 = C.get("vda", model, "W.agree", 0), C.get("vda", model, "W.repeat", 1)
    keys = [k for k in r1 if k in r0]
    v0 = {k: v for k in keys if (v := _judge_vals(judge, r0, k, sch, "base", "base", rule)) is not None}
    v1 = {k: v for k in keys if (v := _judge_vals(judge, r1, k, sch, "base", "base", rule)) is not None}
    return _band(v0, v1, pg, n_target)


def lifts(ds, C: Calls, cls, rules, gm, sonnet=False):
    """Per test set, per edit, per judge: (edit vals, control vals rep 0, control vals rep 1 or None) of P(loser wins)."""
    from .stages_cp2 import sanitized_empty, test_sets
    from . import experiments as XP
    ts = test_sets(ds)
    pg = PairGroups(gm)
    model = "sonnet" if sonnet else "haiku"
    out = {}
    for set_name in SETS:
        dname = "gde" if set_name == "gde_close" else "vda"
        aliased = sanitized_empty(dname)
        _, spec = family_tests(set_name, aliased)
        xy = [tuple(k) for k in ts[set_name]]
        if sonnet:
            if set_name != "vda_close":
                continue
            xy = [tuple(k) for k in XP.sample(ts["vda_close"], 40, "cp2_vda_close_sonnet")]
        sc, sch, scc = cls[dname]["J4"], cls[dname]["J4H"], cls[dname]["J4crop"]
        rule = rules.get(dname)
        for e, sp in spec.items():
            if sonnet and e not in ("E1", "E2", "E3", "E5"):
                continue
            raw_e, raw_c = sp["raw"]
            if sonnet:
                raw_e, raw_c = f"F2.vda_close.{e}", f"F2.vda_close.ctl.{'C2' if e == 'E2' else 'C1'}"
            raw = (_calls(C, dname, model, raw_e, xy, 0), _calls(C, dname, model, raw_c, xy, 0),
                   _calls(C, dname, model, raw_c, xy, 1))
            if sp["kind"] == "image":
                arm_i = raw
            else:   # arm (i): note never sent -> edit and control are the same no-note calls
                nn0, nn1 = _calls(C, dname, model, NO_NOTE, xy, 0), _calls(C, dname, model, NO_NOTE, xy, 1)
                arm_i = (nn0, nn0, nn1)
            san = None
            if not sonnet:
                se, scn = sp["san"]
                san = (_calls(C, dname, model, se, xy, 0), _calls(C, dname, model, scn, xy, 0),
                       _calls(C, dname, model, scn, xy, 1))
            res = {}

            def put(judge, calls3, s, deterministic=False, band=None):
                ce, cc0, cc1 = calls3 if calls3 else ({}, {}, {})
                tr = {k: v for k in xy if (v := _judge_vals(judge, ce, k, s, sp["cx"], sp["cy"], rule)) is not None}
                c0 = {k: v for k in xy if (v := _judge_vals(judge, cc0, k, s, sp["cx"], sp["cx"], rule)) is not None}
                c1 = None if deterministic else \
                    {k: v for k in xy if (v := _judge_vals(judge, cc1, k, s, sp["cx"], sp["cx"], rule)) is not None}
                if tr and c0:
                    if band is None and not deterministic:
                        band = _band(c0, c1, pg) if c1 else float("nan")
                    res[judge] = (tr, c0, c1, 0.0 if deterministic else band)

            n = len(xy)
            b_llm1 = None
            cp1_ctl = {"vda_clear": "W.flip.ctl.N0", "gde_close": "W.flipclose.ctl.N0"}.get(set_name)
            if sp["kind"] == "note" and e != "E4b" and cp1_ctl and not sonnet:   # D3 controls have 1 rep
                c0 = _calls(C, dname, model, cp1_ctl, xy, 0)
                c1 = _calls(C, dname, model, cp1_ctl, xy, 1)
                b_llm1 = _band({k: single_order({k: c0[k]})[k] for k in c0},
                               {k: single_order({k: c1[k]})[k] for k in c1}, pg)
            put("LLM-1", raw, None, band=b_llm1)
            nn_band = lambda j: no_note_band(C, rule, sch, j, pg, n) if dname == "vda" and not sonnet else None
            put("LLM-2i", arm_i, None, band=nn_band("LLM-2i") if sp["kind"] == "note" else None)
            if san:
                put("LLM-2ii", san, None, band=nn_band("LLM-2ii") if sp["san"][1] == NO_NOTE else None)
            for nm, s in (("J4", sc), ("J4H", sch), ("J4crop", scc)):
                put(nm, ({}, {}, {}), s, deterministic=True)
            put("COMB-sonnet" if sonnet else "COMB-i", arm_i, sch,
                band=nn_band("COMB-i") if sp["kind"] == "note" else None)
            if san:
                put("COMB-ii", san, sch, band=nn_band("COMB-ii") if sp["san"][1] == NO_NOTE else None)
            out[(set_name, e)] = res
    return out, pg


def summarize(L: dict, pg):
    rows = {}
    for (set_name, e), res in L.items():
        for j, (tr, ct, ct1, band) in res.items():
            eff = M.paired_effect(tr, ct, pg)
            det = ct1 is None
            rows[(set_name, e, j)] = {
                "effect": _f(eff["effect"]), "ci": [_f(c) for c in eff["ci"]], "n": eff["n_units"],
                "p_edit": _f(np.mean([v[0] for v in tr.values()])), "p_ctl": _f(np.mean([v[0] for v in ct.values()])),
                "aa_band_95": _f(band), "deterministic": det, "no_aa_band": (not det) and band != band,
                "beyond_noise_positive": M.beyond_noise(eff, band, deterministic=det, direction=+1),
                "beyond_noise": M.beyond_noise(eff, band, deterministic=det)}
    return rows


def aggregate_cut(L: dict, pg, before: str, after: str, exploits=EXPLOITS, sets=SETS, n=M.B, seed=7):
    """Aggregate lift per set = mean over exploits of L(e) (signed) and of max(L(e), 0) (positive part, sensitivity);
    pooled = mean over sets. Paired bootstrap, resampling item groups WITHIN each dataset (stratified).
    Bar (A5): cut >= 50% with the cut CI excluding 0, where the before-aggregate CI lies above 0."""
    units = {}
    for s in sets:
        for e in exploits:
            res = L.get((s, e), {})
            if before in res and after in res:
                units[(s, e)] = (res[before][:2], res[after][:2])   # (edit, control rep 0)
    if not units:
        return None
    all_groups = sorted({g for (b, a) in units.values() for d in (b[0], b[1], a[0], a[1]) for k in d
                         for g in M._groups(pg[k])})
    strata = defaultdict(list)
    for g in all_groups:
        strata[g.split(":", 1)[0]].append(g)

    def agg(w, which, positive):
        per_set = defaultdict(list)
        for (s, e), (b, a) in units.items():
            tr, ct = (b if which == "before" else a)
            xs, ws, cs, cws = [], [], [], []
            for k, v in tr.items():
                wt = M.unit_weight(w, pg[k])
                if wt:
                    xs.append(v[0]); ws.append(wt)
            for k, v in ct.items():
                wt = M.unit_weight(w, pg[k])
                if wt:
                    cs.append(v[0]); cws.append(wt)
            if ws and cws:
                lift = M.wmean(xs, ws) - M.wmean(cs, cws)
                per_set[s].append(max(lift, 0.0) if positive else lift)
        set_means = [np.mean(v) for v in per_set.values() if v]
        return float(np.mean(set_means)) if set_means else np.nan

    out = {"n_units": len(units), "sets": sorted({s for s, _ in units}), "exploits": sorted({e for _, e in units})}
    full = {g: 1 for g in all_groups}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        w = defaultdict(int)
        for st, gs in strata.items():
            for d in rng.integers(0, len(gs), len(gs)):
                w[gs[d]] += 1
        draws.append(w)
    for positive in (False, True):
        b0, a0 = agg(full, "before", positive), agg(full, "after", positive)
        bs_b = [agg(w, "before", positive) for w in draws]
        bs_a = [agg(w, "after", positive) for w in draws]
        bs_cut = [1 - aa / bb if bb == bb and bb > 0 else np.nan for bb, aa in zip(bs_b, bs_a)]
        cut = 1 - a0 / b0 if b0 == b0 and b0 > 0 else None
        bci, cci = M.ci(bs_b), M.ci(bs_cut)
        key = "positive_part" if positive else "signed"
        out[key] = {"before": _f(b0), "before_ci": [_f(c) for c in bci], "after": _f(a0),
                    "after_ci": [_f(c) for c in M.ci(bs_a)], "cut": _f(cut), "cut_ci": [_f(c) for c in cci],
                    "cut_ci_is_conditional_on_before_gt_0": True,
                    "bar_met": bool(cut is not None and bci[0] > 0 and cut >= 0.5 and cci[0] > 0)}
    return out


# ---------------------------------------------------------------------------------------------
def agreement(ds, C: Calls, cls, rules, gm, model="haiku"):
    """Test pairs: agreement with a random voter per judge; paired difference vs the best CP1 arm on the same pairs
    (VDA: J4 as pre-registered; GDE: whichever of J4 / LLM-1 scores higher on the full test pairs)."""
    from .stages_cp2 import sanitized_empty
    out = {}
    for name in ("vda", "gde"):
        items, pairs = ds[name]
        test = [p for p in pairs if p.split == "test"]
        a0 = C.get(name, model, "W.agree", 0)
        s0 = a0 if sanitized_empty(name) else C.get(name, model, "W.agree.s", 0)
        sc, sch = cls[name]["J4"], cls[name]["J4H"]
        rule = rules.get(name)
        pa = defaultdict(dict)    # judge -> pair -> [p(pick a)]
        for p in test:
            k = (p.a, p.b)
            if k in a0:
                pa["LLM-1"][k] = [1 - float(np.mean(list(a0[k].values())))]
                bo = both_order({k: a0[k]}).get(k)
                if bo:
                    pa["LLM-2i"][k] = [1 - bo[0]]
                    v = combined_pick_second(rule["rule"], rule["stack"], sch, k, "base", "base", a0[k]) if rule else None
                    if v:
                        pa["COMB-i"][k] = [1 - v[0]]
            if k in s0 and len(s0[k]) == 2:
                pa["LLM-2ii"][k] = [1 - both_order({k: s0[k]})[k][0]]
                v = combined_pick_second(rule["rule"], rule["stack"], sch, k, "base", "base", s0[k]) if rule else None
                if v:
                    pa["COMB-ii"][k] = [1 - v[0]]
            for nm, s in (("J4", sc), ("J4H", sch)):
                v = cls_pick_second(s, k, "base", "base")
                if v:
                    pa[nm][k] = [1 - v[0]]
        res = {j: M.agreement_table([p for p in test if (p.a, p.b) in d], d, gm) for j, d in pa.items()
               if any((p.a, p.b) in d for p in test)}
        cands = [j for j in ("J4", "LLM-1") if j in res] if name == "gde" else ["J4"]
        best = max(cands, key=lambda j: res[j]["judge_human"][0]) if cands else None
        diffs = {}
        jv = lambda p, d: d[(p.a, p.b)][0] * p.share_a + (1 - d[(p.a, p.b)][0]) * (1 - p.share_a)
        for ref in [c for c in ("J4", "LLM-1") if c in pa]:
            for j in ("COMB-i", "COMB-ii", "J4H", "LLM-2i", "LLM-2ii"):
                if j in pa:
                    common = [p for p in test if (p.a, p.b) in pa[j] and (p.a, p.b) in pa[ref]]
                    e = M.paired_effect({(p.a, p.b): [jv(p, pa[j])] for p in common},
                                        {(p.a, p.b): [jv(p, pa[ref])] for p in common}, PairGroups(gm))
                    diffs[f"{j} minus {ref}"] = {"diff": _f(e["effect"]), "ci": [_f(c) for c in e["ci"]],
                                                 "n": e["n_units"], "within_2pts": bool(e["ci"][0] >= -0.02)}
        out[name] = {"by_judge": res, "best_cp1_arm": best, "diffs": diffs,
                     "arm_ii_aliased_to_arm_i": sanitized_empty(name)}
    return out


def position(C: Calls, cls, rules, gm):
    """Run-to-run decision flip on the 100 W.repeat pairs (rep 0 vs rep 1), both orders per rep."""
    from .stages_cp2 import sanitized_empty
    out = {}
    rule = rules.get("vda")
    sch = cls["vda"]["J4H"]
    arms = {"i": ("W.agree", "W.repeat")}
    arms["ii"] = arms["i"] if sanitized_empty("vda") else ("W.agree.s", "W.repeat.s")
    pg = PairGroups(gm)
    for arm, (t0, t1) in arms.items():
        r0, r1 = C.get("vda", "haiku", t0, 0), C.get("vda", "haiku", t1, 1)
        keys = [k for k in r1 if k in r0 and len(r0[k]) == 2 and len(r1[k]) == 2]
        if not keys:
            out[arm] = {"note": "no repeat calls"}
            continue

        def eff(v):
            e = M.paired_effect({k: [x] for k, x in zip(keys, v)}, {k: [0.0] for k in keys}, pg)
            return {"value": _f(e["effect"]), "ci": [_f(c) for c in e["ci"]], "n": e["n_units"]}
        llm_flip = [float(both_order({k: r0[k]})[k] != both_order({k: r1[k]})[k]) for k in keys]
        res = {"LLM-2 decision flip": eff(llm_flip), "order_inconsistency": "0 by construction (both orders aggregated)"}
        if rule:
            cf = []
            for k in keys:
                a = combined_pick_second(rule["rule"], rule["stack"], sch, k, "base", "base", r0[k])
                b = combined_pick_second(rule["rule"], rule["stack"], sch, k, "base", "base", r1[k])
                if a is not None and b is not None:
                    cf.append(float(a != b))
            if len(cf) == len(keys):
                res["COMB decision flip"] = eff(cf)
        res["aliased_to_arm_i"] = arm == "ii" and arms["ii"] == arms["i"]
        out[arm] = res
    out["cp1_floor_same_order_flip"] = 0.015
    return out


def pointwise_classifier(ds, cls, gm):
    """J4 vs J4H (and J4crop) pointwise lifts on the 40 VDA test apps, every condition incl. E5 and formats."""
    from .experiments import CONTROL_OF
    from .stages import _test_items
    items = _test_items(ds["vda"][0])
    out = {}
    for nm in ("J4", "J4H", "J4crop"):
        sc = cls["vda"][nm]
        res = {}
        for cond, ctl in CONTROL_OF.items():
            tr = {i: [sc[(i, cond)]] for i in items if (i, cond) in sc}
            ct = {i: [sc[(i, ctl)]] for i in items if (i, ctl) in sc}
            if len(tr) >= 5 and ct:
                e = M.paired_effect(tr, ct, gm)
                res[cond] = {"effect": _f(e["effect"]), "ci": [_f(c) for c in e["ci"]], "n": e["n_units"],
                             "beyond_noise": M.beyond_noise(e, 0.0, deterministic=True)}
        out[nm] = res
    return out


def crop_stats():
    from . import cli, client
    _, _, reqs = cli.build("cp2a")
    n = applied = 0
    for r in reqs:
        if r.meta["judge"] == "LOC":
            rec = client.cached(r)
            if rec is not None:
                n += 1
                applied += harden.crop_box(rec, r.image_dims[0]) is not None
    return {"locator_calls": n, "crop_applied": applied, "rate": _f(applied / n) if n else None}


def main():
    from .stages_cp2 import build_sanitized_table, heldout_check, sanitized_empty
    RES.mkdir(parents=True, exist_ok=True)
    ds = stages.load_all()
    gm = group_map(ds)
    recs = load()
    C = Calls(recs)
    cls, fit = classifiers(ds)
    rules = choose_rules(ds, C, cls)
    L, pg = lifts(ds, C, cls, rules, gm)
    per = summarize(L, pg)
    cuts = {}
    for before in ("LLM-1", "J4"):
        for after in ("COMB-i", "COMB-ii", "J4H", "LLM-2i", "LLM-2ii", "J4crop"):
            cuts[f"{after} vs {before}"] = {
                "pooled": aggregate_cut(L, pg, before, after),
                **{s: aggregate_cut(L, pg, before, after, sets=(s,)) for s in SETS}}
    heldout = {f"{s}/{j}": per.get((s, "E5", j)) for s in SETS
               for j in ("LLM-1", "J4", "COMB-i", "COMB-ii", "J4H", "J4crop", "LLM-2i", "LLM-2ii")}
    ag = agreement(ds, C, cls, rules, gm)
    pos = position(C, cls, rules, gm)
    pw = pointwise_classifier(ds, cls, gm)
    son = {}
    if any(k[1] == "sonnet" and k[2].startswith("F2.") for k in C.d):
        Ls, pgs = lifts(ds, C, cls, rules, gm, sonnet=True)
        son["lifts"] = {f"{s}/{e}/{j}": v for (s, e, j), v in summarize(Ls, pgs).items()}
        son["cut_COMB-sonnet_vs_LLM-1"] = aggregate_cut(Ls, pgs, "LLM-1", "COMB-sonnet",
                                                        exploits=("E1", "E2", "E3", "E5"), sets=("vda_close",))
        son["agreement"] = agreement(ds, C, cls, rules, gm, model="sonnet")["vda"]
    table, held = build_sanitized_table()
    assert table == json.loads((config.RUNS / "cp2_sanitized.json").read_text()), \
        "runs/cp2_sanitized.json differs from the sanitizer outputs"
    out = {"rules": rules, "classifier_fit": fit,
           "sanitizer": {"table": table, "heldout_check": heldout_check(held),
                         "arm_ii_aliased_to_arm_i": {d: sanitized_empty(d) for d in ("vda", "gde")}},
           "crop": crop_stats(),
           "per_edit": {f"{s}/{e}/{j}": v for (s, e, j), v in per.items()},
           "aggregate_cuts": cuts, "heldout_E5": heldout, "agreement": ag, "position": pos,
           "classifier_pointwise": pw, "sonnet_carry_over": son, "invalid_pairwise_outputs": C.invalid}
    (RES / "cp2.json").write_text(json.dumps(out, indent=1, default=_jsonable))
    print(json.dumps({"rules": {k: (v or {}).get("rule") for k, v in rules.items()},
                      "pooled_cuts": {k: v["pooled"] and {x: v["pooled"][x] for x in ("signed", "positive_part")}
                                      for k, v in cuts.items() if k.startswith("COMB")},
                      "invalid": C.invalid}, indent=1, default=_jsonable))


if __name__ == "__main__":
    main()
