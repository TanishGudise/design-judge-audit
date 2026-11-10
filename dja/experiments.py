"""Pre-registered call manifests (frozen at the CP1 freeze commit) -> Req objects.

Row fields (local metadata, never sent): judge, model_key, dataset, test, and either
  pointwise: item, cond (image condition), note, rep
  pairwise:  a, b (A/B as shown), cond_a, cond_b, note_a, note_b, order, rep, and the pair key
Tests
  P.base     pointwise J1/J3 on the canonical image, BASE_REPS reps (R6 noise + A/A band)
  P.<cond>   pointwise J1/J3 on an image condition (C1 E1 E3 E4a E5 R4 C2 E2 half jpeg40 webp pad10)
             or a note condition on the base image (N0 R2 R3 E4b), COND_REPS reps
  W.agree    pairwise J2, both orders (R1 position + human agreement), 1 rep; W.repeat: 1 more rep in each
             order on a subset (same-order run-to-run flip rate)
  W.flip.*   pairwise J2 on clear pairs: human loser Y gets the edit, winner X its matched control
"""
from __future__ import annotations

import numpy as np

from . import config, judges, render
from .data import Pair

BASE_REPS = 4      # A/A band: 2 reps vs 2 reps, same shape as a 2-rep condition vs its 2-rep control
COND_REPS = 2
STRIP_EDITS = ("E1", "E3", "E4a", "E5", "R4")
NOTE_EDITS = ("R2", "R3", "E4b")
POINT_IMAGE_CONDS = ("C1",) + STRIP_EDITS + ("C2", "E2") + render.FORMAT_CONDITIONS
POINT_NOTE_CONDS = ("N0",) + NOTE_EDITS
CONTROL_OF = {**{e: "C1" for e in STRIP_EDITS}, "E2": "C2", **{e: "N0" for e in NOTE_EDITS},
              **{f: "base" for f in render.FORMAT_CONDITIONS}}
# flip families: (control image cond, edit image cond, control note, edit note)
FLIP = {**{e: ("C1", e, None, None) for e in STRIP_EDITS}, "E2": ("C2", "E2", None, None),
        **{e: ("base", "base", "N0", e) for e in NOTE_EDITS}}
FLIP_CONTROLS = {"C1": ("C1", "C1", None, None), "C2": ("C2", "C2", None, None),
                 "N0": ("base", "base", "N0", "N0")}


def _rng(tag: str):
    return np.random.default_rng([config.SEED, *map(ord, tag)])


def sample(xs, k, tag):
    xs = list(xs)
    if k is None or k >= len(xs):
        return xs
    idx = _rng(tag).permutation(len(xs))[:k]
    return [xs[i] for i in sorted(idx)]


# ---------------------------------------------------------------------------------------------
def pointwise_rows(dataset, item_ids, model_key, conds: dict, judges_=("J1", "J3"), stage="main"):
    rows = []
    for iid in item_ids:
        for cond, reps in conds.items():
            img_cond, note = (("base", cond) if cond in POINT_NOTE_CONDS else (cond, None))
            for j in judges_:
                for r in range(reps):
                    rows.append({"judge": j, "model_key": model_key, "dataset": dataset,
                                 "test": f"P.{cond}", "item": iid, "cond": img_cond, "note": note,
                                 "rep": r, "stage": stage})
    return rows


def agree_rows(dataset, pairs: list[Pair], model_key, reps=1, test="W.agree", rep0=0, stage="main"):
    rows = []
    for p in pairs:
        for order in ("ab", "ba"):
            a, b = (p.a, p.b) if order == "ab" else (p.b, p.a)
            for r in range(rep0, rep0 + reps):
                rows.append({"judge": "J2", "model_key": model_key, "dataset": dataset, "test": test,
                             "pair": [p.a, p.b], "a": a, "b": b, "cond_a": "base", "cond_b": "base",
                             "note_a": None, "note_b": None, "order": order, "rep": r, "stage": stage})
    return rows


def flip_rows(dataset, xy: list[tuple], model_key, edits, control_reps=2, stage="main", prefix="W.flip"):
    """xy: (human winner X, human loser Y). Control families get control_reps reps (A/A), edits 1.
    prefix 'W.flip' = clear pairs (pre-registered); 'W.flipclose' = close pairs (Amendment A4)."""
    rows = []
    fams = [(f"{prefix}.ctl.{c}", FLIP_CONTROLS[c], control_reps) for c in
            sorted({FLIP[e][0] if FLIP[e][2] is None else "N0" for e in edits})]
    fams += [(f"{prefix}.{e}", FLIP[e], 1) for e in edits]
    for test, (cx, cy, nx, ny), reps in fams:
        for (x, y) in xy:
            for order in ("xy", "yx"):
                for r in range(reps):
                    a, b, ca, cb, na, nb = ((x, y, cx, cy, nx, ny) if order == "xy" else (y, x, cy, cx, ny, nx))
                    rows.append({"judge": "J2", "model_key": model_key, "dataset": dataset, "test": test,
                                 "pair": [x, y], "a": a, "b": b, "cond_a": ca, "cond_b": cb,
                                 "note_a": na, "note_b": nb, "order": order, "rep": r, "stage": stage})
    return rows


# ---------------------------------------------------------------------------------------------
def to_req(row: dict, items: dict):
    ds = row["dataset"]
    if row["judge"] in ("J1", "J3"):
        it = items[row["item"]]
        img = render.render_cached(it.id, it.path, ds, row["cond"])
        build = judges.build_j1 if row["judge"] == "J1" else judges.build_j3
        return build(ds, img, row["note"], row["model_key"], row)
    ia, ib = items[row["a"]], items[row["b"]]
    img_a = render.render_cached(ia.id, ia.path, ds, row["cond_a"])
    img_b = render.render_cached(ib.id, ib.path, ds, row["cond_b"])
    return judges.build_j2(ds, img_a, img_b, row["note_a"], row["note_b"], row["model_key"], row)


def clear_xy(dataset, pairs: list[Pair], items: dict, k=None, tag="clear"):
    """Clear test pairs as (winner, loser). VDA: >= 20/30 votes. GDE: unanimous original vs
    large perturbation. MF: |delta mean AE (z)| >= 1.0."""
    out = []
    for p in pairs:
        if p.split != "test":
            continue
        if dataset == "gde":
            if not p.b.split("/")[1].endswith("large"):
                continue
            if p.votes_b == 0 and p.votes_a >= 5:
                out.append((p.a, p.b))
        elif max(p.share_a, 1 - p.share_a) >= 2 / 3:
            out.append((p.a, p.b) if p.share_a > 0.5 else (p.b, p.a))
    return sample(out, k, tag)


def close_xy(pairs: list[Pair], lo: float = 0.55, hi: float = 0.80):
    """Amendment A4 (2026-11-09): test pairs whose human majority share is in [lo, hi], as
    (majority winner, minority loser). Where humans are split, an exploit could plausibly flip the verdict."""
    out = []
    for p in pairs:
        if p.split != "test" or p.n == 0:
            continue
        m = max(p.share_a, 1 - p.share_a)
        if lo <= m <= hi:
            out.append((p.a, p.b) if p.share_a > 0.5 else (p.b, p.a))
    return sorted(out)


def mf_pairs(items: dict, k: int, min_delta: float = 1.0, tag="mf_pairs"):
    """Test-split page pairs with a clear human aesthetics gap, as 1-0 'votes' for the higher page."""
    test = sorted(i for i, it in items.items() if it.split == "test")
    cand = []
    for x in range(len(test)):
        for y in range(x + 1, len(test)):
            a, b = test[x], test[y]
            d = items[a].human["mean"] - items[b].human["mean"]
            if abs(d) >= min_delta:
                cand.append(Pair(a, b, int(d > 0), int(d < 0), "mf", split="test"))
    return sample(cand, k, tag)


def estimate(reqs) -> dict:
    """Worst case (live price, reservation bound) and a typical-case guess (bound inputs are ~1.2x
    real; 150 output tokens) for a list of Req."""
    wc = sum(r.worst_case_usd() for r in reqs)
    typ = 0.0
    for r in reqs:
        p = config.PRICES[r.model]
        typ += (r.input_bound() / 1.2 * p["in"] + 150 * p["out"]) / 1e6
    return {"n": len(reqs), "worst_case_live": round(wc, 4), "typical_live": round(typ, 4),
            "typical_batch": round(typ * config.BATCH_DISCOUNT, 4)}
