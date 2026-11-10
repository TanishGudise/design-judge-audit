"""Regenerate every reported number from the response cache + embeddings (offline, $0).

Outputs (results/, committed): spend.json, agreement.json, reliability.json, effects.json, summary.md
Pre-registered rules applied here:
- an effect is "beyond noise" if its 95% cluster-bootstrap CI excludes 0 and the CI bound nearest 0 is
  farther from 0 than the 95th percentile of the A/A (same condition re-run) effect at the same shape;
  deterministic arms (J0, J4) have no run-to-run noise, so for them CI-excludes-0 suffices
- CP1 kill rule counts the named tests R1-R5, E1-E5 (E4 as E4a and E4b) on Haiku 5.5 on the primary
  web dataset (any of J1 / J2-flip / J3); fewer than 3 -> STOP before CP2. Exploits count only when they
  raise the score (Amendment A2); biases count in either direction.
- effects, agreement and the kill rule use ONE stage's calls (main by default; `analyze pilot` for the pilot)
- pair units (flips, position) are cluster-bootstrapped on both items' groups
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from . import config, experiments as X, judges, metrics as M, stages
from .ledger import Ledger

RES = config.RESULTS
ARM_NAMES = {"J1": "J1 rubric score (LLM)", "J2": "J2 pairwise (LLM)", "J3": "J3 checklist (LLM)",
             "J0": "J0 UIClip zero-shot", "J4": "J4 classifier on frozen UIClip embeddings"}


def _f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


MISMATCHED = []


def load_records(stage_names=("pilot", "main")):
    """Cached responses for the frozen manifests' requests. A cache entry whose request hash differs
    from the manifest's (a stale response to a changed prompt) is never used."""
    recs = []
    for st in stage_names:
        p = config.RUNS / "manifests" / f"{st}.json"
        if not p.exists():
            continue
        man = json.loads(p.read_text())
        for cid, h in zip(man["custom_ids"], man["request_hashes"]):
            cp = config.CACHE / f"{cid}.json"
            if cp.exists():
                rec = json.loads(cp.read_text())
                if rec.get("request_hash") != h:
                    MISMATCHED.append(cid)
                    continue
                recs.append(rec)
    return recs


# ---------------------------------------------------------------------------------------------
def spend(recs):
    led = Ledger().summary()
    by = defaultdict(lambda: {"calls": 0, "usd": 0.0, "in": 0, "out": 0, "max_in_over_bound": 0.0,
                              "thinking_tokens": 0, "price_flags": 0, "max_out": 0})
    for r in recs:
        k = f"{r['meta']['stage']}/{r['meta']['dataset']}/{r['model']}/{r['meta']['judge']}/{r['mode']}"
        b = by[k]
        b["calls"] += 1
        b["usd"] += r["cost_usd"]
        u = r.get("usage", {})
        b["in"] += u.get("input_tokens", 0)
        b["out"] += u.get("output_tokens", 0)
        b["max_in_over_bound"] = max(b["max_in_over_bound"], u.get("input_tokens", 0) / max(1, r["input_bound"]))
        b["max_out"] = max(b["max_out"], u.get("output_tokens", 0) or 0)
        b["thinking_tokens"] += ((u.get("output_tokens_details") or {}).get("thinking_tokens") or 0)
        b["price_flags"] += bool(r.get("price_flags"))
    per = {k: {**v, "usd": round(v["usd"], 6), "usd_per_call": round(v["usd"] / v["calls"], 7),
               "in_per_call": round(v["in"] / v["calls"]), "out_per_call": round(v["out"] / v["calls"]),
               "max_in_over_bound": round(v["max_in_over_bound"], 3)} for k, v in sorted(by.items())}
    cache_total = round(sum(r["cost_usd"] for r in recs), 6)
    usage_keys = sorted({k for r in recs for k in (r.get("usage") or {})})
    return {"ledger": led, "cache_total_usd": cache_total, "by_group": per, "usage_fields_seen": usage_keys,
            "stale_cache_entries_skipped": len(MISMATCHED),
            "prices_read_on": config.PRICES_READ_ON,
            "note": "ledger 'total' includes conservative charges for calls with unknown outcome; "
                    "cache_total is the sum of usage-priced successful calls"}


# ---------------------------------------------------------------------------------------------
class Scores:
    """Parsed judge outputs for one stage, indexed for analysis."""

    def __init__(self, recs, stage: str):
        self.point = defaultdict(list)   # (ds, model, judge, test, item) -> [score]
        self.pair = defaultdict(list)    # (ds, model, test, rep, (x, y)) -> [1 if pair[1] picked else 0]
        self.first = defaultdict(list)   # (ds, model, test) -> [(pairkey, order, rep, picked_first)]
        self.invalid = defaultdict(int)
        self.n = 0
        self.parsed_by_substage = defaultdict(lambda: [0, 0])   # stage -> [parsed, total]
        for r in recs:
            m = r["meta"]
            st = m.get("stage", "")
            if not (st == stage or st.startswith(stage + ".")):
                continue
            v = judges.parse(r)
            self.parsed_by_substage[st][1] += 1
            if v is None:
                self.invalid[(st, m["dataset"], m["model_key"], m["judge"])] += 1
                continue
            self.parsed_by_substage[st][0] += 1
            if st != stage:
                continue
            self.n += 1
            mk = m["model_key"]
            if m["judge"] in ("J1", "J3"):
                self.point[(m["dataset"], mk, m["judge"], m["test"], m["item"])].append(v["score"])
            else:
                key = tuple(m["pair"])
                picked = m["a"] if v["winner"] == "A" else m["b"]
                self.pair[(m["dataset"], mk, m["test"], m["rep"], key)].append(1.0 if picked == key[1] else 0.0)
                self.first[(m["dataset"], mk, m["test"])].append((key, m["order"], m["rep"], v["winner"] == "A"))


def group_map(ds):
    return {iid: f"{name}:{it.group}" for name, (items, _) in ds.items() for iid, it in items.items()}


class PairGroups(dict):
    """unit (x, y) -> (group of x, group of y): pair units are clustered on BOTH items."""

    def __init__(self, gm):
        super().__init__()
        self.gm = gm

    def __missing__(self, key):
        return (self.gm[key[0]], self.gm[key[1]])


def human_scores(name, items):
    return {iid: it.human["mean"] for iid, it in items.items()
            if "mean" in it.human and it.human["mean"] == it.human["mean"]}


def _rows():
    rows = []
    for st in ("pilot", "main"):
        p = config.RUNS / "manifests" / f"{st}.json"
        if p.exists():
            rows += json.loads(p.read_text())["rows"]
    return rows


# ---------------------------------------------------------------------------------------------
def mf_half_panels(pairs, arm_pa: dict, rng, reps=200):
    """MF has ~4.5 raters per page, so the human 'ceiling' and the judges are compared on the same footing:
    in each resample every page's raters are split in two halves; per arm, on exactly the pairs that arm
    judged: human-human = sign agreement of the two half-panels, judge-human = agreement of the judge's pick
    with half-panel 1."""
    from .data import mf_raw_ae
    raw = mf_raw_ae()
    by_page = {f"mf/{s}": g.z.values for s, g in raw.groupby("stimulusId")}
    hh, jh, hh_arm = [], defaultdict(list), defaultdict(list)
    for _ in range(reps):
        halves = {}
        for pid, v in by_page.items():
            v = rng.permutation(v)
            h = len(v) // 2
            if h >= 1:
                halves[pid] = (v[:h].mean(), v[h:2 * h].mean())
        s = {}
        for p in pairs:
            if p.a in halves and p.b in halves:
                s1 = np.sign(halves[p.a][0] - halves[p.b][0])
                s2 = np.sign(halves[p.a][1] - halves[p.b][1])
                if s1 != 0 and s2 != 0:
                    s[(p.a, p.b)] = (s1, s2)
        hh.append(np.mean([a == b for a, b in s.values()]))
        for arm, pa in arm_pa.items():
            ks = [k for k in s if k in pa]
            if ks:
                jh[arm].append(np.mean([pa[k] if s[k][0] > 0 else 1 - pa[k] for k in ks]))
                hh_arm[arm].append(np.mean([s[k][0] == s[k][1] for k in ks]))
    q = lambda v: [float(np.mean(v)), float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    return {"human_half_vs_half_all_pairs": q(hh),
            **{arm: {"judge_vs_half": q(jh[arm]), "human_half_vs_half_same_pairs": q(hh_arm[arm]),
                     "n_pairs": len([k for k in pa if k in {(p.a, p.b) for p in pairs}])}
               for arm, pa in arm_pa.items() if jh[arm]}}


def _pairs_for(name, items, pairs, stage):
    if name != "mf":
        return [p for p in pairs if p.split == "test"]
    from .data import Pair
    keys = sorted({tuple(r["pair"]) for r in _rows() if r["dataset"] == "mf" and r["test"] == "W.agree"
                   and r.get("stage") == stage})
    return [Pair(a, b, int(items[a].human["mean"] > items[b].human["mean"]),
                 int(items[a].human["mean"] < items[b].human["mean"]), "mf", "test") for a, b in keys]


def agreement(ds, S: Scores, arms_extra: dict, gm, stage: str, return_pa: bool = False):
    """Per dataset: every arm's agreement with humans on the agreement pairs + rank correlation.
    return_pa (CP3, post-hoc paired comparisons): also return {dataset: {arm: {pair: p(pick side a)}}}."""
    out, pa_out = {}, {}
    rng = np.random.default_rng(config.SEED)
    for name, (items, pairs) in ds.items():
        test_pairs = _pairs_for(name, items, pairs, stage)
        if not test_pairs:
            continue
        rows, arm_pa = {}, {}
        hum = human_scores(name, items)
        if name == "vda":   # pre-registered: rank apps by Bradley-Terry strength from the TEST pairs' votes
            test_items = sorted(i for i, it in items.items() if it.split == "test")
            hum = M.bt_scores(test_pairs, test_items)
        for mk in ("haiku", "sonnet"):
            pa = defaultdict(list)
            for (d, m, t, rep, key), vals in S.pair.items():
                if d == name and m == mk and t == "W.agree":
                    pa[key] += [1 - v for v in vals]       # p(pick pair side a)
            if pa:
                tp = [p for p in test_pairs if (p.a, p.b) in pa]
                rows[f"J2/{mk}"] = M.agreement_table(tp, pa, gm)
                arm_pa[f"J2/{mk}"] = {k: float(np.mean(v)) for k, v in pa.items()}
            for j in ("J1", "J3"):
                sc = {it: float(np.mean(v)) for (d, m, jj, t, it), v in S.point.items()
                      if d == name and m == mk and jj == j and t == "P.base"}
                if len(sc) < 5:
                    continue
                pa2 = {(p.a, p.b): [1.0 if sc[p.a] > sc[p.b] else 0.0 if sc[p.a] < sc[p.b] else 0.5]
                       for p in test_pairs if p.a in sc and p.b in sc}
                tab = M.agreement_table([p for p in test_pairs if (p.a, p.b) in pa2], pa2, gm) if pa2 else {}
                rc, n = M.rank_corr(sc, hum, gm) if hum else ((None, None, None), 0)
                rows[f"{j}/{mk}"] = {**tab, "spearman_vs_human_mean": rc, "n_items": n}
                arm_pa[f"{j}/{mk}"] = {k: v[0] for k, v in pa2.items()}
        for arm, sc_all in arms_extra.get(name, {}).items():
            sc = {it: s for it, s in sc_all.items() if items[it].split == "test" and s == s}
            pa2 = {(p.a, p.b): [1.0 if sc[p.a] > sc[p.b] else 0.0 if sc[p.a] < sc[p.b] else 0.5]
                   for p in test_pairs if p.a in sc and p.b in sc}
            tab = M.agreement_table([p for p in test_pairs if (p.a, p.b) in pa2], pa2, gm) if pa2 else {}
            rc, n = M.rank_corr(sc, hum, gm) if hum else ((None, None, None), 0)
            rows[arm] = {**tab, "spearman_vs_human_mean": rc, "n_items": n}
            arm_pa[arm] = {k: v[0] for k, v in pa2.items()}
        ceil = {}
        if name == "vda":
            test_items = sorted(i for i, it in items.items() if it.split == "test")
            r, sb = M.split_half_votes(test_pairs, test_items, rng, reps=100)
            ceil = {"pairwise": "human_human column (two random voters of the same pair)",
                    "bt_rank_split_half_15v15": r, "bt_rank_spearman_brown_30": sb}
        elif name == "gde":
            ratings = {iid: [x for x in sum(it.human["by_principle"].values(), []) if x == x]
                       for iid, it in items.items() if it.split == "test"}
            r, sb = M.split_half_ratings(ratings, rng)
            ceil = {"pairwise": "human_human column (two random voters of the same pair)",
                    "rating_split_half": r, "rating_spearman_brown": sb}
        elif name == "mf":
            from .data import mf_raw_ae
            raw = mf_raw_ae()
            ratings = {f"mf/{s}": list(g.z.values) for s, g in raw.groupby("stimulusId")
                       if f"mf/{s}" in items and items[f"mf/{s}"].split == "test"}
            r, sb = M.split_half_ratings(ratings, rng)
            ceil = {"rating_split_half": r, "rating_spearman_brown": sb,
                    "half_panel_comparison": mf_half_panels(test_pairs, arm_pa, rng)}
        out[name] = {"arms": rows, "ceiling": ceil, "n_test_pairs": len(test_pairs)}
        pa_out[name] = arm_pa
    return (out, pa_out) if return_pa else out


# ---------------------------------------------------------------------------------------------
def position(ds, S: Scores, gm):
    out = {}
    pg = PairGroups(gm)
    for (name, mk, test), calls in S.first.items():
        if test != "W.agree":
            continue
        by_pair = defaultdict(dict)
        for key, order, rep, first in calls:
            by_pair[key][(order, rep)] = first
        pf = {k: [float(v) for v in d.values()] for k, d in by_pair.items()}
        eff = M.paired_effect(pf, {k: [0.5] for k in pf}, pg)
        incons = {k: [float(d[("ab", 0)] == d[("ba", 0)])] for k, d in by_pair.items()
                  if ("ab", 0) in d and ("ba", 0) in d}     # same slot won both orders = different designs picked
        rep_by = defaultdict(dict)
        for key, order, rep, first in S.first.get((name, mk, "W.repeat"), []):
            rep_by[key][(order, rep)] = first
        flips = {}
        for k, d in rep_by.items():
            vals = [float(d[(o, 1)] != by_pair[k][(o, 0)]) for o in ("ab", "ba")
                    if (o, 1) in d and (o, 0) in by_pair.get(k, {})]
            if vals:
                flips[k] = vals
        res = {"p_first": {"value": _f(0.5 + eff["effect"]), "ci": [_f(0.5 + c) for c in eff["ci"]],
                           "n_pairs": eff["n_units"]},
               "first_slot_bias_beyond_noise": M.beyond_noise(eff, 0.0, deterministic=True)}
        # p_first vs 0.5 has no run-to-run analogue: a judge with no position preference sits at 0.5 in
        # expectation whatever its noise, so CI-excludes-0.5 is the test (cluster bootstrap over both items).
        if incons:
            ie = M.paired_effect(incons, {k: [0.0] for k in incons}, pg)
            res["order_inconsistency"] = {"value": _f(ie["effect"]), "ci": [_f(c) for c in ie["ci"]], "n": ie["n_units"]}
        if flips:
            fe = M.paired_effect(flips, {k: [0.0] for k in flips}, pg)
            res["same_order_repeat_flip"] = {"value": _f(fe["effect"]), "ci": [_f(c) for c in fe["ci"]], "n": fe["n_units"]}
            common = [k for k in incons if k in flips]
            if common:
                d = M.paired_effect({k: incons[k] for k in common}, {k: flips[k] for k in common}, pg)
                res["excess_inconsistency_over_noise"] = {"value": _f(d["effect"]), "ci": [_f(c) for c in d["ci"]],
                                                          "n": d["n_units"],
                                                          "beyond_noise": M.beyond_noise(d, 0.0, deterministic=True)}
        out[f"{name}/{mk}"] = res
    return out


def _effect_entry(e, control, band, deterministic):
    return {"control": control, "effect": _f(e["effect"]), "ci": [_f(c) for c in e["ci"]],
            "n_units": e["n_units"], "aa_band_95": _f(band),
            "beyond_noise": M.beyond_noise(e, band, deterministic=deterministic),
            "beyond_noise_positive": M.beyond_noise(e, band, deterministic=deterministic, direction=+1)}


def pointwise_effects(ds, S: Scores, gm, extra_scores: dict):
    """Every P.<cond> vs its control, per dataset x model x judge; deterministic arms on images."""
    out = {}
    keys = {(d, m, j) for (d, m, j, t, it) in S.point}
    for (d, m, j) in sorted(keys):
        def reps(test):
            return {it: v for (dd, mm, jj, t, it), v in S.point.items() if (dd, mm, jj, t) == (d, m, j, test)}
        base = reps("P.base")
        n_base = max((len(v) for v in base.values()), default=0)   # pre-registered reps (invalid outputs
        k = n_base // 2                                             # don't shrink k; aa_band skips short units)
        band = M.aa_band(base, k, gm) if k >= 1 else float("nan")
        sds = [np.std(v, ddof=1) for v in base.values() if len(v) > 1]
        res = {"aa_band_95": _f(band), "aa_reps_per_side": k,
               "base_mean": _f(np.mean([np.mean(v) for v in base.values()])) if base else None,
               "repeat_sd_mean": _f(np.mean(sds)) if sds else None, "tests": {}}
        for cond, ctl in X.CONTROL_OF.items():
            tr, cr = reps(f"P.{cond}"), reps(f"P.{ctl}")
            if tr and cr:
                res["tests"][cond] = _effect_entry(M.paired_effect(tr, cr, gm), ctl, band, False)
        out[f"{d}/{m}/{j}"] = res
    for d, arms in extra_scores.items():
        for arm, sc in arms.items():   # sc: (item, cond) -> score
            res = {"aa_band_95": 0.0, "tests": {}}
            for cond, ctl in X.CONTROL_OF.items():
                tr = {it: [s] for (it, c), s in sc.items() if c == cond and s == s}
                cr = {it: [s] for (it, c), s in sc.items() if c == ctl and s == s}
                if len(tr) >= 5 and cr:
                    res["tests"][cond] = _effect_entry(M.paired_effect(tr, cr, gm), ctl, 0.0, True)
            out[f"{d}/-/{arm}"] = res
    return out


def flip_effects(ds, S: Scores, gm, extra_scores: dict, stage: str):
    out = {}
    pg = PairGroups(gm)
    combos = {(d, m, t.split(".")[1]) for (d, m, t, rep, key) in S.pair if t.startswith("W.flip")}
    for (d, m, fam) in sorted(combos):
        pre = f"W.{fam}"
        def get(test, rep):
            return {key: v for (dd, mm, t, r, key), v in S.pair.items() if (dd, mm, t, r) == (d, m, test, rep)}
        res = {"tests": {}, "pairs": "clear (pre-registered)" if fam == "flip" else "close 55-80% (Amendment A4)"}
        for edit, (cx, cy, nx, ny) in X.FLIP.items():
            ctl = "N0" if nx else cx
            tr, c0, c1 = get(f"{pre}.{edit}", 0), get(f"{pre}.ctl.{ctl}", 0), get(f"{pre}.ctl.{ctl}", 1)
            if not tr or not c0:
                continue
            e = M.paired_effect(tr, c0, pg)
            band = float("nan")
            if c1:
                aa = M.paired_effect(c1, c0, pg)
                band = float(np.quantile(np.abs(aa["boot"][~np.isnan(aa["boot"])]), 0.95))
            ent = _effect_entry(e, ctl, band, False)
            ent["p_loser_wins_control"] = _f(np.mean([np.mean(v) for v in c0.values()]))
            ent["p_loser_wins_edit"] = _f(np.mean([np.mean(v) for v in tr.values()]))
            res["tests"][edit] = ent
        out[f"{d}/{m}/J2" + ("" if fam == "flip" else "close")] = res
    # deterministic arms: does the loser WITH the edit outscore the winner WITH the control?
    rows = [r for r in _rows() if r.get("stage") == stage]
    for d, arms in extra_scores.items():
      for fam in ("flip", "flipclose"):
        pre = f"W.{fam}"
        flips = {(tuple(r["pair"]), r["test"]) for r in rows if r["dataset"] == d
                 and r["test"].startswith(pre + ".") and not r["test"].startswith(pre + ".ctl")}
        if not flips:
            continue
        for arm, sc in arms.items():
            res = {"tests": {}}
            for edit, (cx, cy, nx, ny) in X.FLIP.items():
                if nx:      # note edits are text; image-only arms cannot see them
                    continue
                keys = sorted({k for k, t in flips if t == f"{pre}.{edit}"})
                tr, c0 = {}, {}
                for k in keys:
                    xs, ye, yc = sc.get((k[0], cx)), sc.get((k[1], edit)), sc.get((k[1], cx))
                    if None in (xs, ye, yc) or any(v != v for v in (xs, ye, yc)):
                        continue          # missing embedding: drop the pair, never score it as a loss
                    tr[k], c0[k] = [float(ye > xs)], [float(yc > xs)]
                if tr:
                    res["tests"][edit] = _effect_entry(M.paired_effect(tr, c0, pg), cx, 0.0, True)
            out[f"{d}/-/{arm}" + ("" if fam == "flip" else "close")] = res
    return out


# ---------------------------------------------------------------------------------------------
def extra_arms(ds):
    """J0 / J4 scores. Returns (agreement scores: ds -> arm -> item -> score on base,
    condition scores: ds -> arm -> (item, cond) -> score, J4 fit info)."""
    from . import clip_arm, embed_run
    agree, conds, fits = defaultdict(dict), defaultdict(dict), {}
    # CP2 added renders to the same embedding files (exploit augmentation, close-pair test images, crops). CP1 skips
    # every image only CP2 needs, so its pointwise averages stay those of the CP1 inputs (found in CP3; checked equal to
    # the key sets committed with CP1 for vda, gde and mf).
    needed, later = embed_run.needed(ds), embed_run.cp2_needed(ds)
    for name, (items, pairs) in ds.items():
        p = embed_run.emb_path(name)
        if not p.exists():
            continue
        z = np.load(p)
        cp1 = {f"{i}|{c}" for i, c in needed.get(name, ())}
        drop = {f"{i}|{c}" for i, c in later.get(name, ())} - cp1
        emb = {k: z[k] for k in z.files if k not in drop and "|crop:" not in k}
        base = {k.split("|")[0]: v for k, v in emb.items() if k.endswith("|base")}
        keys = sorted(emb)
        zs = clip_arm.zero_shot(np.stack([emb[k] for k in keys]), name)
        j0 = {tuple(k.split("|")): float(s) for k, s in zip(keys, zs)}
        groups = {i: it.group for i, it in items.items()}
        if name == "vda":
            build_pairs = [pp for pp in pairs if pp.split == "build"]
            w, info = clip_arm.fit_bt(base, build_pairs, groups)
            j4 = {tuple(k.split("|")): float(emb[k] @ w) for k in keys}
        else:
            target = {i: it.human["mean"] for i, it in items.items()
                      if it.split == "build" and i in base and it.human["mean"] == it.human["mean"]}
            (w, b), info = clip_arm.fit_ridge(base, target, groups)
            j4 = {tuple(k.split("|")): float(emb[k] @ w + b) for k in keys}
        conds[name] = {"J0": j0, "J4": j4}
        agree[name] = {"J0": {i: s for (i, c), s in j0.items() if c == "base"},
                       "J4": {i: s for (i, c), s in j4.items() if c == "base"}}
        fits[name] = info
    return agree, conds, fits


NAMED_TESTS = ["R1", "R2", "R3", "R4", "R5", "E1", "E2", "E3", "E4a", "E4b", "E5"]


def kill_rule(eff_point, eff_flip, pos, web: str):
    """Haiku 5.5, primary web dataset. Biases (R1-R5) count in either direction; exploits (E*) count only
    when they RAISE the judge's score / the loser's win rate (Amendment A2). R5 counts once (any format)."""
    from .render import FORMAT_CONDITIONS
    hits = {}
    for t in NAMED_TESTS:
        why = []
        key = "beyond_noise_positive" if t.startswith("E") else "beyond_noise"
        if t == "R1":
            r = pos.get(f"{web}/haiku", {})
            if r.get("first_slot_bias_beyond_noise") or r.get("excess_inconsistency_over_noise", {}).get("beyond_noise"):
                why.append("J2 position")
        conds = list(FORMAT_CONDITIONS) if t == "R5" else [t]
        for c in conds:
            for j in ("J1", "J3"):
                e = eff_point.get(f"{web}/haiku/{j}", {}).get("tests", {}).get(c)
                if e and e[key]:
                    why.append(f"{j} {c}")
            e = eff_flip.get(f"{web}/haiku/J2", {}).get("tests", {}).get(c)
            if e and e[key]:
                why.append(f"J2-flip {c}")
        hits[t] = why
    n = sum(bool(v) for v in hits.values())
    return {"web_dataset": web, "tests_beyond_noise": n, "detail": hits, "stop_before_cp2": n < 3}


def _jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def main(stage: str = "main"):
    out_dir = RES if stage == "main" else RES / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = stages.load_all()
    gm = group_map(ds)
    MISMATCHED.clear()
    recs = load_records()
    S = Scores(recs, stage)
    agree_extra, cond_extra, fit_info = extra_arms(ds)
    sp = spend(recs)
    ag = agreement(ds, S, agree_extra, gm, stage)
    pos = position(ds, S, gm)
    pe = pointwise_effects(ds, S, gm, cond_extra)
    fe = flip_effects(ds, S, gm, cond_extra, stage)
    web = "vda" if "vda" in ds else "mf"
    kr = kill_rule(pe, fe, pos, web)
    for name, obj in [("spend", sp), ("agreement", {"by_dataset": ag, "j4_fit": fit_info}),
                      ("reliability_position", pos), ("effects_pointwise", pe), ("effects_flip", fe),
                      ("kill_rule", kr), ("invalid_outputs", {"by_key": {str(k): v for k, v in S.invalid.items()},
                                                              "parsed_by_substage": dict(S.parsed_by_substage)})]:
        (out_dir / f"{name}.json").write_text(json.dumps(obj, indent=1, default=_jsonable))
    print(json.dumps({"stage": stage, "records_in_stage": S.n, "spend": {"ledger": sp["ledger"],
                      "cache_total_usd": sp["cache_total_usd"]}, "kill_rule": kr,
                      "invalid": {str(k): v for k, v in S.invalid.items()},
                      "stale_cache_skipped": len(MISMATCHED)}, indent=1, default=_jsonable))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
