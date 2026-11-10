"""Pre-registered stages. Frozen at the CP1 freeze commit; changes after a paid call are dated amendments.

pilot (bucket 'pilot', cap $2.00; Haiku rows live, Sonnet rows through the batch path): a slice of every
       Haiku request shape (all image conditions incl. JPEG/WebP media types, all notes, J1/J2/J3) and the
       Sonnet shapes main uses (J1, J2; PNG), to check parsing, real token counts vs the reservation bound,
       thinking tokens and cost per call. The largest pilot image is smaller than main's largest (both under
       the 2,576 px limit).
main  (batch, bucket 'cp1', cap $4.50 incl. pilot): the full CP1 measurement. Primary web dataset is
       VDA if it is on disk when main starts, else Miniukovich & Figl (MF). GDE runs in both cases.
       Sonnet volume follows SONNET_MAIN and is scaled down in SONNET_DROP_ORDER if the projected
       typical batch cost would leave < $0.50 of the cp1 cap.
"""
from __future__ import annotations

from . import data, experiments as X

ALL_POINT = {"base": X.BASE_REPS, **{c: X.COND_REPS for c in X.POINT_IMAGE_CONDS},
             **{c: X.COND_REPS for c in X.POINT_NOTE_CONDS}}



def _test_items(items, prefix=None, k=None, tag="items"):
    ids = sorted(i for i, it in items.items() if it.split == "test" and (prefix is None or i.startswith(prefix)))
    return X.sample(ids, k, tag)


def load_all():
    out = {"gde": data.load_gde(), "mf": data.load_mf()}
    if data.vda_available():
        out["vda"] = data.load_vda()
    return out


def gde_point_items(items, k_templates, tag):
    """k test templates -> their original + both large perturbations (3 items per template)."""
    orgs = _test_items(items, "gde/org/", k_templates, tag)
    out = []
    for o in orgs:
        tid = o.split("/")[-1]
        out += [o, f"gde/lefttop_large/{tid}", f"gde/size_large/{tid}"]
    return out


def pilot_rows(ds: dict) -> list[dict]:
    rows = []
    pc = {"base": 2, "C1": 1, "E1": 1, "E4a": 1, "C2": 1, "E2": 1, "N0": 1, "E4b": 1}
    shapes = {c: 1 for c in X.POINT_IMAGE_CONDS + X.POINT_NOTE_CONDS}   # every condition once
    gi, gp = ds["gde"]
    rows += X.pointwise_rows("gde", gde_point_items(gi, 4, "pilot_gde"), "haiku", pc, stage="pilot")
    rows += X.pointwise_rows("gde", gde_point_items(gi, 1, "pilot_gde_shapes"), "haiku",
                             {**shapes, "base": 1}, judges_=("J1",), stage="pilot.shapes")
    xy = X.clear_xy("gde", gp, gi, 8, "pilot_gde_xy")
    rows += X.flip_rows("gde", xy, "haiku", ("E1", "E4a", "E2"), control_reps=1, stage="pilot")
    gtest = [p for p in gp if p.split == "test"]
    rows += X.agree_rows("gde", X.sample(gtest, 10, "pilot_gde_agree"), "haiku", stage="pilot")
    rows += X.agree_rows("gde", X.sample(gtest, 8, "pilot_gde_agree_s"), "sonnet", stage="pilot")
    mi, _ = ds["mf"]
    rows += X.pointwise_rows("mf", _test_items(mi, k=8, tag="pilot_mf"), "haiku",
                             {"base": 2, "C1": 1, "E1": 1}, stage="pilot")
    rows += X.pointwise_rows("mf", _test_items(mi, k=2, tag="pilot_mf_shapes"), "haiku",
                             {**shapes, "base": 1}, judges_=("J1", "J3"), stage="pilot.shapes")
    rows += X.agree_rows("mf", X.mf_pairs(mi, 8, tag="pilot_mf_pairs"), "haiku", stage="pilot")
    rows += X.pointwise_rows("mf", _test_items(mi, k=4, tag="pilot_mf_s"), "sonnet", {"base": 1},
                             judges_=("J1",), stage="pilot")
    if "vda" in ds:
        vi, vp = ds["vda"]
        rows += X.pointwise_rows("vda", _test_items(vi, k=8, tag="pilot_vda"), "haiku", pc, stage="pilot")
        vtest = [p for p in vp if p.split == "test"]
        rows += X.agree_rows("vda", X.sample(vtest, 12, "pilot_vda_agree"), "haiku", stage="pilot")
        rows += X.flip_rows("vda", X.clear_xy("vda", vp, vi, 6, "pilot_vda_xy"), "haiku",
                            ("E1", "E4a", "E2"), control_reps=1, stage="pilot")
        rows += X.agree_rows("vda", X.sample(vtest, 10, "pilot_vda_agree_s"), "sonnet", stage="pilot")
    return rows


# ---------------------------------------------------------------------------------------------
FLIP_EDITS = X.STRIP_EDITS + ("E2", "E4b")          # every exploit/bias with an in-design or note edit
SONNET_POINT = {"base": 2, "C1": 1, "E1": 1, "E3": 1, "E4a": 1, "E5": 1, "R4": 1, "C2": 1, "E2": 1}
SONNET_DROP_ORDER = ("sonnet_agree", "sonnet_items_half")


def main_rows(ds: dict, drop: tuple = ()) -> list[dict]:
    rows = []
    # ---------- GDE (graphics): agreement on every test template + flips on clear pairs ----------
    gi, gp = ds["gde"]
    rows += X.pointwise_rows("gde", gde_point_items(gi, None, "main_gde"), "haiku", {"base": 2}, stage="main")
    gtest = [p for p in gp if p.split == "test"]
    rows += X.agree_rows("gde", gtest, "haiku", stage="main")
    rows += X.flip_rows("gde", X.clear_xy("gde", gp, gi, 60, "main_gde_xy"), "haiku", FLIP_EDITS, stage="main")
    # Amendment A4: GDE's clear pairs (original vs large perturbation) hit a ceiling (the judge never flips),
    # so exploit lift is also measured on CLOSE pairs (human majority 55-80%), all 75 test pairs.
    rows += X.flip_rows("gde", X.close_xy(gp), "haiku", FLIP_EDITS, stage="main", prefix="W.flipclose")

    # ---------- primary web dataset: VDA if on disk, else Miniukovich & Figl ----------
    if "vda" in ds:
        wname, (wi, wp) = "vda", ds["vda"]
        w_items = _test_items(wi)                                     # 40 test apps
        w_pairs = [p for p in wp if p.split == "test"]                # 780 test pairs
        w_xy = X.clear_xy("vda", wp, wi, 60, "main_vda_xy")
    else:
        wname, (wi, _) = "mf", ds["mf"]
        w_items = _test_items(wi, k=40, tag="main_mf_items")
        w_pairs = X.mf_pairs(wi, 300, tag="main_mf_pairs")
        w_xy = [(p.a, p.b) if p.votes_a else (p.b, p.a) for p in X.sample(w_pairs, 60, "main_mf_xy")]
    rows += X.pointwise_rows(wname, w_items, "haiku", ALL_POINT, stage="main")
    rows += X.agree_rows(wname, w_pairs, "haiku", stage="main")
    rows += X.agree_rows(wname, X.sample(w_pairs, 100, f"main_{wname}_rep"), "haiku", reps=1, rep0=1,
                         test="W.repeat", stage="main")
    rows += X.flip_rows(wname, w_xy, "haiku", FLIP_EDITS, stage="main")

    # ---------- Sonnet 5.5 carry-over check (web dataset) ----------
    k_items = 15 if "sonnet_items_half" in drop else 30
    s_items = X.sample(w_items, k_items, f"main_{wname}_sonnet_items")
    rows += X.pointwise_rows(wname, s_items, "sonnet", SONNET_POINT, judges_=("J1",), stage="main")
    if "sonnet_agree" not in drop:
        rows += X.agree_rows(wname, X.sample(w_pairs, 60, f"main_{wname}_sonnet"), "sonnet", stage="main")
    return rows
