"""CP3 analysis (plan A7 + addendum). Offline, $0, from runs/cache + runs/embeddings.

1. Duplicate-screenshot check: the two byte-identical VDA screenshots (both a blank "Not Found" page), their split, and
   the CP2 headline numbers with the pairs that touch them removed (sensitivity only; CP2 numbers are not replaced).
2. New exploit types X1-X4 vs geometry-matched controls K1-K4 on the CP2 VDA clear + close pairs:
   J4H (hardened VDA judge = the combined judge, rule "classifier only"), J4 (CP1 classifier), Haiku pairwise
   (LLM-1 = per-call average over both orders, as the CP2 "CP1 LLM" column; LLM-2 = both-order vote, secondary).
   Lift = P(human loser wins | exploit on the loser) - P(... | control on both), cluster bootstrap over both items' apps.
3. Secondary: pointwise classifier score shift s(exploit) - s(control) over the 40 test apps; J4H - J4 lift difference.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from . import config, metrics as M, render, stages
from .analyze import PairGroups, group_map, _jsonable
from .analyze_cp2 import (Calls, _band, _f, agreement, aggregate_cut, both_order, choose_rules, classifiers,
                          cls_pick_second, lifts, load, single_order, summarize)
from .stages import _test_items
from .stages_cp2 import test_sets
from .stages_cp3 import NOT_FOUND, SETS

RES = config.RESULTS / "cp3"
JUDGES = ("J4H", "J4", "LLM-1", "LLM-2")
CLS = ("J4", "J4H")


def _r(eff, band, det):
    lo, hi = eff["ci"]
    through = M.beyond_noise(eff, band, deterministic=det, direction=+1)
    back = M.beyond_noise(eff, band, deterministic=det) and eff["effect"] < 0
    return {"effect": _f(eff["effect"]), "ci": [_f(lo), _f(hi)], "n": eff["n_units"],
            "aa_band_95": None if det else _f(band),
            "verdict": "gets through" if through else "backfires" if back else "not shown to get through"}


def _vals(judge, calls, sc, xy, cx, cy):
    out = {}
    for k in xy:
        if judge in CLS:
            v = cls_pick_second(sc[judge], k, cx, cy)
        else:
            c = calls.get(k)
            if not c:
                continue
            v = single_order({k: c})[k] if judge == "LLM-1" else both_order({k: c}).get(k)
        if v is not None:
            out[k] = v
    return out


def _xy(ds, exclude=()):
    ts = test_sets(ds)
    out = {s: [tuple(k) for k in ts[s] if not (set(k) & set(exclude))] for s in SETS}
    out["pooled"] = out["vda_clear"] + out["vda_close"]
    return out


def cp3_lifts(ds, C3: Calls, sc, pg, exclude=()):
    """{exploit: {set|pooled: {judge: result}}, plus raw per-pair values for the J4H - J4 difference."""
    out, raw = {}, {}
    for e in render.CP3_EXPLOITS:
        k = render.CP3_CONTROL[e]
        out[e] = {}
        for sname, xy in _xy(ds, exclude).items():
            sets = SETS if sname == "pooled" else (sname,)
            ce, c0, c1 = {}, {}, {}
            for s in sets:
                ce.update(C3.get("vda", "haiku", f"F3.{s}.{e}", 0))
                c0.update(C3.get("vda", "haiku", f"F3.{s}.ctl.{k}", 0))
                c1.update(C3.get("vda", "haiku", f"F3.{s}.ctl.{k}", 1))
            res = {}
            for j in JUDGES:
                det = j in CLS
                tr, ct = _vals(j, ce, sc, xy, k, e), _vals(j, c0, sc, xy, k, k)
                band = 0.0 if det else _band(ct, _vals(j, c1, sc, xy, k, k), pg)
                eff = M.paired_effect(tr, ct, pg)
                res[j] = {**_r(eff, band, det), "p_edit": _f(np.mean([v[0] for v in tr.values()])) if tr else None,
                          "p_ctl": _f(np.mean([v[0] for v in ct.values()])) if ct else None}
                raw[(e, sname, j)] = (tr, ct)
            out[e][sname] = res
    return out, raw


def hardening_gain(raw, pg):
    """Paired J4H - J4 lift difference per exploit (pooled): negative = the CP2 hardening reduced the lift."""
    out = {}
    for e in render.CP3_EXPLOITS:
        (th, ch), (t4, c4) = raw[(e, "pooled", "J4H")], raw[(e, "pooled", "J4")]
        keys = [k for k in th if k in ch and k in t4 and k in c4]
        eff = M.paired_effect({k: [th[k][0] - ch[k][0]] for k in keys}, {k: [t4[k][0] - c4[k][0]] for k in keys}, pg)
        out[e] = {"J4H_minus_J4": _f(eff["effect"]), "ci": [_f(c) for c in eff["ci"]], "n": eff["n_units"]}
    return out


def pointwise(ds, sc, exclude=()):
    """Mean classifier score shift s(item, exploit) - s(item, control) over the 40 test apps (app-cluster CI), also in
    units of the SD of that classifier's base scores over the same apps."""
    vi, _ = ds["vda"]
    apps = [i for i in _test_items(vi) if i not in exclude]
    g = {i: vi[i].group for i in apps}
    out = {}
    for j in CLS:
        s = sc[j]
        sd = float(np.std([s[(i, "base")] for i in apps]))
        out[j] = {"base_score_sd": _f(sd)}
        for e in render.CP3_EXPLOITS:
            k = render.CP3_CONTROL[e]
            eff = M.paired_effect({i: [s[(i, e)]] for i in apps}, {i: [s[(i, k)]] for i in apps}, g)
            out[j][e] = {"shift": _f(eff["effect"]), "ci": [_f(c) for c in eff["ci"]],
                         "shift_in_sd": _f(eff["effect"] / sd), "n_apps": eff["n_units"]}
    return out


def render_facts(W=1120, H=700):
    """Geometry of the CP3 conditions at the canonical VDA size (for checking the matched controls)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), (255, 255, 255))
    font, cx, cy, th, _, _, box = render.hero_layout(img)
    d = ImageDraw.Draw(img)
    cxb, cyb, R = render.badge_geometry(W, H)
    dw, dh, b, base_h, x, top = render.laptop_geometry(W, H)
    return {"canvas": [W, H],
            "hero_text_width_px": {c: round(d.textlength(t, font=font), 1) for c, t in render.HERO_TEXT.items()},
            "hero_font_px": font.size, "hero_centre": [cx, cy],
            "badge_centre_radius_px": [round(cxb, 1), round(cyb, 1), round(R, 1)],
            "laptop_design_box_px": [x, top + b, dw, dh], "laptop_bezel_px": b}


# ---------------------------------------------------------------------------------------------
def _filter_L(L, exclude):
    bad = set(exclude)
    keep = lambda d: None if d is None else {k: v for k, v in d.items() if not (set(k) & bad)}
    return {key: {j: (keep(tr), keep(c0), keep(c1), band) for j, (tr, c0, c1, band) in res.items()}
            for key, res in L.items() if key[0] != "gde_close"}


def duplicate_check(ds, C2, cls, rules, gm):
    vi, vp = ds["vda"]
    a, b = NOT_FOUND
    sha = {i: hashlib.sha256(open(vi[i].path, "rb").read()).hexdigest() for i in NOT_FOUND}
    share, rank, tot = {}, {}, {}
    for p in vp:                            # within-split pairs (cross-split pairs are used nowhere)
        for side, v, o in ((p.a, p.votes_a, p.votes_b), (p.b, p.votes_b, p.votes_a)):
            w, n = tot.get(side, (0, 0))
            tot[side] = (w + v, n + v + o)
    rates = {i: w / n for i, (w, n) in tot.items()}
    order = sorted(rates, key=rates.get)
    for i in NOT_FOUND:
        share[i] = _f(rates[i])
        rank[i] = order.index(i) + 1
    self_pair = [(p.a, p.b, p.votes_a, p.votes_b, p.split) for p in vp if {p.a, p.b} == {a, b}]
    out = {"byte_identical": sha[a] == sha[b], "sha256": sha[a][:16],
           "splits": {i: vi[i].split for i in NOT_FOUND}, "leak": vi[a].split != vi[b].split,
           "screenshot": "blank 'Not Found' page (visual check)",
           "human_vote_share_within_split_pairs": share, "rank_from_worst_of_60": rank,
           "self_comparison_pair": self_pair}
    full = agreement(ds, C2, cls, rules, gm)["vda"]["diffs"]["COMB-i minus J4"]
    ds_self = dict(ds); ds_self["vda"] = (vi, [p for p in vp if {p.a, p.b} != {a, b}])
    no_self = agreement(ds_self, C2, cls, rules, gm)["vda"]["diffs"]["COMB-i minus J4"]
    ds_nf = dict(ds); ds_nf["vda"] = (vi, [p for p in vp if not ({p.a, p.b} & set(NOT_FOUND))])
    no_nf = agreement(ds_nf, C2, cls, rules, gm)["vda"]
    out["cp2_agreement_COMB-i_minus_J4"] = {"all_780": full, "self_pair_removed": no_self,
                                           "not_found_pairs_removed": no_nf["diffs"]["COMB-i minus J4"],
                                           "not_found_pairs_removed_by_judge": {
                                               j: [_f(x) for x in no_nf["by_judge"][j]["judge_human"]]
                                               for j in ("COMB-i", "J4", "LLM-1")}}
    return out


def cp2_not_found_sensitivity(ds, C2, cls, rules, gm):
    """CP2 VDA lift numbers with every pair that touches a Not-Found screenshot removed (A7 addendum; the A/A bands are
    the full-set ones)."""
    L, pg = lifts(ds, C2, cls, rules, gm)
    Lf = _filter_L(L, NOT_FOUND)
    per = summarize(Lf, pg)
    keep = ("LLM-1", "J4", "J4H", "COMB-i")
    rows = {f"{s}/{e}/{j}": {x: v[x] for x in ("effect", "ci", "n", "beyond_noise_positive")}
            for (s, e, j), v in per.items() if j in keep}
    cuts = {}
    for s in SETS:
        for after in ("COMB-i", "J4H"):
            c = aggregate_cut(Lf, pg, "J4", after, sets=(s,))
            cuts[f"{s}: {after} vs J4"] = c and {x: c["signed"][x] for x in ("before", "before_ci", "after", "cut",
                                                                            "cut_ci", "bar_met")}
    n_pairs = {s: len(next(iter(Lf[(s, "E1")].values()))[0]) for s in SETS}
    return {"pairs_left": n_pairs, "per_edit": rows, "cuts": cuts}


def probe_vs_llm_agreement(ds, gm):
    """Post-hoc (after CP3 review): paired difference in agreement with a random voter, CP1 probe J4 minus each CP1
    Haiku judge, on the same VDA test pairs. Same per-pair values and clustering as the CP1 agreement table."""
    from . import analyze as A
    recs = A.load_records()
    S = A.Scores(recs, "main")
    agree_extra, _, _ = A.extra_arms(ds)
    _, pa = A.agreement({"vda": ds["vda"]}, S, {"vda": agree_extra["vda"]}, gm, "main", return_pa=True)
    pa = pa["vda"]
    pairs = {(p.a, p.b): p for p in ds["vda"][1] if p.split == "test"}
    pg = PairGroups(gm)
    jv = lambda k, v: v * pairs[k].share_a + (1 - v) * (1 - pairs[k].share_a)
    out = {}
    for other in ("J2/haiku", "J1/haiku", "J3/haiku"):
        common = [k for k in pairs if k in pa["J4"] and k in pa.get(other, {})]
        e = M.paired_effect({k: [jv(k, pa["J4"][k])] for k in common}, {k: [jv(k, pa[other][k])] for k in common}, pg)
        out[f"J4 minus {other}"] = {"diff": _f(e["effect"]), "ci": [_f(c) for c in e["ci"]], "n": e["n_units"]}
    return out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    from . import embed_run
    ds = stages.load_all()
    gm = group_map(ds)
    pg = PairGroups(gm)
    z3 = np.load(embed_run.emb_path("vda_cp3"))
    extra = {"vda": {tuple(k.split("|", 1)): z3[k] for k in z3.files}}
    cls, _ = classifiers(ds, extra_emb=extra)
    sc = cls["vda"]
    C2 = Calls(load())
    rules = choose_rules(ds, C2, cls)
    assert rules["vda"]["rule"] == "classifier", "CP3 assumes the CP2 VDA rule (classifier only)"
    C3 = Calls(load(("cp3",)))
    lift, raw = cp3_lifts(ds, C3, sc, pg)
    lift_nf, _ = cp3_lifts(ds, C3, sc, pg, exclude=NOT_FOUND)
    n_calls = sum(len(v) for k, v in C3.d.items())
    out = {"vda_rule": rules["vda"]["rule"], "pairs": {k: len(v) for k, v in _xy(ds).items()},
           "haiku_calls_parsed": n_calls, "invalid_pairwise_outputs": C3.invalid,
           "lifts": lift, "lifts_without_not_found_pairs": lift_nf,
           "hardening_gain_pooled": hardening_gain(raw, pg),
           "pointwise": pointwise(ds, sc), "pointwise_without_not_found": pointwise(ds, sc, exclude=NOT_FOUND),
           "post_hoc_probe_vs_llm_agreement": probe_vs_llm_agreement(ds, gm),
           "duplicate_check": duplicate_check(ds, C2, cls, rules, gm),
           "cp2_vda_sensitivity_without_not_found_pairs": cp2_not_found_sensitivity(ds, C2, cls, rules, gm)}
    out["render_facts"] = render_facts()
    (RES / "cp3.json").write_text(json.dumps(out, indent=1, default=_jsonable))
    print("pooled lifts (135 pairs):")
    for e in render.CP3_EXPLOITS:
        print(" ", e, {j: (v["effect"], v["ci"], v["verdict"]) for j, v in lift[e]["pooled"].items()})
    print("invalid:", C3.invalid, "parsed calls:", n_calls)


if __name__ == "__main__":
    main()
