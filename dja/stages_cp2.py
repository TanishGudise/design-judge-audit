"""CP2 manifests (plan Amendment A5). Three phases, each frozen before it runs:

cp2a  sanitizer calls on every unique note body (+ the A5 held-out generalization sentences), live; locator calls on
      every unique test image the crop arm needs, batch.
cp2   Haiku 5.5, batch: build pairs both orders (rule choice), flips on VDA clear / VDA close (new) / GDE close pairs for
      raw notes, sanitized notes (arm ii, incl. in-image edits with the sanitized control note on both sides), and
      agreement + repeat with sanitized notes.
cp2s  Sonnet 5.5 carry-over on the final combined judge, batch.
Pixels-only (arm i) LLM calls for in-image edits and for agreement are the CP1 calls (same requests), reused from cache.
"""
from __future__ import annotations

from . import experiments as X, harden, judges, render
from .stages import load_all, _test_items

IMAGE_EDITS = ("E1", "E2", "E3", "E4a", "E5", "R4")       # E5 = held out
NOTE_EDITS = ("R2", "R3", "E4b")
VDA_CLOSE_K = 75


def _fam_rows(dataset, xy, model_key, fams, stage, prefix):
    """fams: [(name, (cx, cy, nx, ny), reps)] -> both-order flip rows (X = human winner, Y = loser)."""
    rows = []
    for name, (cx, cy, nx, ny), reps in fams:
        for (x, y) in xy:
            for order in ("xy", "yx"):
                for r in range(reps):
                    a, b, ca, cb, na, nb = ((x, y, cx, cy, nx, ny) if order == "xy" else (y, x, cy, cx, ny, nx))
                    rows.append({"judge": "J2", "model_key": model_key, "dataset": dataset, "test": f"{prefix}.{name}",
                                 "pair": [x, y], "a": a, "b": b, "cond_a": ca, "cond_b": cb, "note_a": na,
                                 "note_b": nb, "order": order, "rep": r, "stage": stage})
    return rows


def raw_note_fams(edits):
    return [("ctl.N0", ("base", "base", "N0", "N0"), 2)] + [(e, ("base", "base", "N0", e), 1) for e in edits]


def san_note_fams():
    return [("ctl.N0s", ("base", "base", "N0s", "N0s"), 2)] + \
           [(f"{e}s", ("base", "base", "N0s", f"{e}s"), 1) for e in NOTE_EDITS]


def san_image_fams():
    return [("ctl.C1s", ("C1", "C1", "N0s", "N0s"), 2), ("ctl.C2s", ("C2", "C2", "N0s", "N0s"), 2)] + \
           [(f"{e}s", ("C2" if e == "E2" else "C1", e, "N0s", "N0s"), 1) for e in IMAGE_EDITS]


def pixel_image_fams():
    return [("ctl.C1", ("C1", "C1", None, None), 2), ("ctl.C2", ("C2", "C2", None, None), 2)] + \
           [(e, ("C2" if e == "E2" else "C1", e, None, None), 1) for e in IMAGE_EDITS]


def test_sets(ds):
    vi, vp = ds["vda"]
    gi, gp = ds["gde"]
    vda_clear = X.clear_xy("vda", vp, vi, 60, "main_vda_xy")                    # the CP1 flip pairs
    clear_keys = {tuple(sorted(k)) for k in vda_clear}
    close_all = [k for k in X.close_xy(vp) if tuple(sorted(k)) not in clear_keys]
    vda_close = X.sample(close_all, VDA_CLOSE_K, "cp2_vda_close")
    gde_close = X.close_xy(gp)                                                   # the A4 pairs
    return {"vda_clear": vda_clear, "vda_close": vda_close, "gde_close": gde_close}


# ---------------------------------------------------------------------------------------------
def build_sanitized_table() -> dict:
    """runs/cp2_sanitized.json from the cached sanitizer calls; refuses if any is missing or invalid (A6)."""
    from . import cli, client
    _, _, reqs = cli.build("cp2a")
    table, heldout = {}, {}
    for r in reqs:
        if r.meta["judge"] != "SAN":
            continue
        out = harden.apply_sanitizer(r.meta["text"], client.cached(r))
        if out is None:
            raise SystemExit(f"sanitizer output missing/invalid for {r.meta['tag']}")
        if r.meta["tag"].startswith("note:"):
            table[r.meta["text"]] = out
        else:
            heldout[r.meta["text"]] = out
    return table, heldout


def heldout_check(heldout: dict) -> dict:
    drop_ok = sum(1 for s in harden.HELDOUT_DROP
                  if heldout.get(f"{harden.HELDOUT_PREFIX} {s}") is not None and s not in heldout[f"{harden.HELDOUT_PREFIX} {s}"])
    keep_ok = sum(1 for s in harden.HELDOUT_KEEP if s in (heldout.get(f"{harden.HELDOUT_PREFIX} {s}") or ""))
    return {"dropped_adversarial": f"{drop_ok}/{len(harden.HELDOUT_DROP)}",
            "kept_factual": f"{keep_ok}/{len(harden.HELDOUT_KEEP)}"}


def cp2a_rows(ds):
    """Sanitizer rows (one per unique note body + held-out checks) and locator rows (one per unique test image)."""
    san = []
    for d in ("vda", "gde"):
        for n in judges.NOTES:
            san.append(("note", d, n, harden.note_body(n, d)))
    for i, s in enumerate(harden.HELDOUT_DROP + harden.HELDOUT_KEEP):
        san.append(("heldout", "-", f"h{i}", f"{harden.HELDOUT_PREFIX} {s}"))
    loc = set()
    vi, _ = ds["vda"]
    for iid in _test_items(vi):
        for c in ("base", "C1", "C2") + IMAGE_EDITS:
            loc.add(("vda", iid, c))
    ts = test_sets(ds)
    for (x, y) in ts["gde_close"]:
        for iid in (x, y):
            for c in ("base", "C1", "C2") + IMAGE_EDITS:
                loc.add(("gde", iid, c))
    return san, sorted(loc)


def cp2_rows(ds):
    rows = []
    ts = test_sets(ds)
    vi, vp = ds["vda"]
    gi, gp = ds["gde"]
    # build pairs, both orders, pixels only (combination-rule choice)
    rows += X.agree_rows("vda", [p for p in vp if p.split == "build"], "haiku", test="W.build", stage="cp2")
    rows += X.agree_rows("gde", [p for p in gp if p.split == "build"], "haiku", test="W.build", stage="cp2")
    # VDA close pairs (new test set): pixels-only image edits, raw notes, sanitized notes (+ image edits with N0s)
    rows += _fam_rows("vda", ts["vda_close"], "haiku", pixel_image_fams(), "cp2", "F2.vda_close")
    rows += _fam_rows("vda", ts["vda_close"], "haiku", raw_note_fams(NOTE_EDITS), "cp2", "F2.vda_close")
    # VDA clear and GDE close: CP1 has pixels-only image edits and raw N0/E4b; add raw R2, R3
    same_batch = [("ctl.N0", ("base", "base", "N0", "N0"), 1)] + [(e, ("base", "base", "N0", e), 1) for e in ("R2", "R3")]
    rows += _fam_rows("vda", ts["vda_clear"], "haiku", same_batch, "cp2", "F2.vda_clear")       # A6 (D3)
    rows += _fam_rows("gde", ts["gde_close"], "haiku", same_batch, "cp2", "F2.gde_close")
    for name, d in (("vda_clear", "vda"), ("vda_close", "vda"), ("gde_close", "gde")):
        rows += _fam_rows(d, ts[name], "haiku", san_note_fams() + san_image_fams(), "cp2", f"F2.{name}")
    # agreement and repeat with sanitized notes on both sides (arm ii)
    vtest = [p for p in vp if p.split == "test"]
    gtest = [p for p in gp if p.split == "test"]
    for d, ps in (("vda", vtest), ("gde", gtest)):
        for r in _note_agree_rows(d, ps, "haiku", "W.agree.s", 0):
            rows.append(r)
    for r in _note_agree_rows("vda", X.sample(vtest, 100, "main_vda_rep"), "haiku", "W.repeat.s", 1):
        rows.append(r)
    return [r for r in rows if not collapses_to_pixels_only(r)]


def collapses_to_pixels_only(row) -> bool:
    """A6 (D1): an arm (ii) row whose sanitized notes are empty on BOTH sides sends exactly the pixels-only (arm i)
    request, which CP1/CP2 already makes; it is not re-sent and the analysis reads the arm (i) calls instead."""
    na, nb = row.get("note_a"), row.get("note_b")
    if not (na or nb) or not all(n is None or n.endswith("s") for n in (na, nb)):
        return False
    return judges.note_text(na, row["dataset"]) is None and judges.note_text(nb, row["dataset"]) is None


def sanitized_empty(dataset: str) -> bool:
    return judges.note_text("N0s", dataset) is None


def _note_agree_rows(dataset, pairs, model_key, test, rep):
    out = X.agree_rows(dataset, pairs, model_key, reps=1, rep0=rep, test=test, stage="cp2")
    for r in out:
        r["note_a"] = r["note_b"] = "N0s"
    return out


def cp2s_rows(ds):
    """Sonnet carry-over on the final combined judge (pixels-only LLM arm)."""
    vi, vp = ds["vda"]
    vtest = [p for p in vp if p.split == "test"]
    cp1_s = X.sample(vtest, 60, "main_vda_sonnet")                           # CP1 Sonnet pairs (cached)
    rest = [p for p in vtest if p not in cp1_s]
    extra = X.sample(rest, 40, "cp2_vda_sonnet_extra")
    rows = X.agree_rows("vda", cp1_s, "sonnet", stage="main") + X.agree_rows("vda", extra, "sonnet", stage="cp2s")
    close40 = X.sample(test_sets(ds)["vda_close"], 40, "cp2_vda_close_sonnet")
    fams = [("ctl.C1", ("C1", "C1", None, None), 1), ("ctl.C2", ("C2", "C2", None, None), 1)] + \
           [(e, ("C2" if e == "E2" else "C1", e, None, None), 1) for e in ("E1", "E2", "E3", "E5")]
    rows += _fam_rows("vda", close40, "sonnet", fams, "cp2s", "F2.vda_close")
    return rows


def all_cp2_image_keys(ds):
    """(dataset, item, cond) for every image any CP2 arm scores (for embeddings)."""
    keys = set()
    vi, vp = ds["vda"]
    gi, gp = ds["gde"]
    for p in vp:
        if p.split == "build":
            for c in harden.AUG_CONDS:
                keys |= {("vda", p.a, c), ("vda", p.b, c)}
    for p in gp:
        if p.split == "build":
            for c in harden.AUG_CONDS:
                keys |= {("gde", p.a, c), ("gde", p.b, c)}
    for iid in _test_items(vi):
        for c in ("base", "C1", "C2") + IMAGE_EDITS + render.FORMAT_CONDITIONS:
            keys.add(("vda", iid, c))
    ts = test_sets(ds)
    for (x, y) in ts["gde_close"]:
        for iid in (x, y):
            for c in ("base", "C1", "C2") + IMAGE_EDITS:
                keys.add(("gde", iid, c))
    return keys
