"""CP3 manifest (plan A7): new exploit types on the 40 held-out VDA apps.

Haiku 5.5 pairwise baseline, pixels only (no note), both orders, batch: on the CP2 VDA clear (60) and VDA close (75)
pairs, the human winner gets the control and the human loser the exploit; control family = control on both sides.
Exploits 1 rep, controls 2 reps (rep 1 gives the A/A band). Classifier arms (J4, J4H) are scored offline ($0).
"""
from __future__ import annotations

from . import render
from .stages import _test_items
from .stages_cp2 import _fam_rows, test_sets

SETS = ("vda_clear", "vda_close")
# A7 addendum: these two apps' screenshots are the same blank "Not Found" page (byte-identical); every CP2/CP3 lift pair
# that touches them has one as the human loser. Reported as a pre-registered sensitivity (pairs excluded), not dropped.
NOT_FOUND = ("vda/concero", "vda/mathix___educational_app_for_admission_exams")


def fams():
    out = []
    for e in render.CP3_EXPLOITS:
        k = render.CP3_CONTROL[e]
        out.append((f"ctl.{k}", (k, k, None, None), 2))
        out.append((e, (k, e, None, None), 1))
    return out


def cp3_rows(ds):
    ts = test_sets(ds)
    rows = []
    for s in SETS:
        rows += _fam_rows("vda", ts[s], "haiku", fams(), "cp3", f"F3.{s}")
    return rows


def cp3_image_keys(ds):
    """(item, cond) for every CP3 image: all 40 test apps x the 8 new conditions (classifier arms score all of them)."""
    vi, _ = ds["vda"]
    return {(iid, c) for iid in _test_items(vi) for c in render.CP3_CONDITIONS}
