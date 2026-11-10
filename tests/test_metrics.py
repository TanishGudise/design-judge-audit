import numpy as np
import pytest

from dja import metrics as M
from dja.data import Pair


def test_hh_agree():
    assert M.hh_agree(30, 0) == 1.0
    assert M.hh_agree(15, 15) == pytest.approx((15 * 14 * 2) / (30 * 29))
    assert M.hh_agree(1, 1) == 0.0


def test_agreement_perfect_and_contrarian_judge():
    pairs = [Pair(f"a{i}", f"b{i}", 25, 5, "vda", "test") for i in range(12)]
    group = {f"a{i}": f"g{i}" for i in range(12)} | {f"b{i}": f"h{i}" for i in range(12)}
    good = M.agreement_table(pairs, {(p.a, p.b): [1.0, 1.0] for p in pairs}, group)
    bad = M.agreement_table(pairs, {(p.a, p.b): [0.0] for p in pairs}, group)
    assert good["judge_human"][0] == pytest.approx(25 / 30)
    assert bad["judge_human"][0] == pytest.approx(5 / 30)
    assert good["clear_majority_acc"][0] == 1.0 and bad["clear_majority_acc"][0] == 0.0
    assert good["human_human"][0] == pytest.approx(M.hh_agree(25, 5))


def test_bt_recovers_order():
    items = ["x", "y", "z"]
    pairs = [Pair("x", "y", 25, 5, "v"), Pair("y", "z", 25, 5, "v"), Pair("x", "z", 28, 2, "v")]
    s = M.bt_scores(pairs, items)
    assert s["x"] > s["y"] > s["z"]


def test_effect_and_noise_band():
    rng = np.random.default_rng(0)
    units = [f"u{i}" for i in range(40)]
    group = {u: u for u in units}
    ctl = {u: list(rng.normal(5, 0.3, 2)) for u in units}
    trt = {u: list(np.array(ctl[u]) + 1.0 + rng.normal(0, 0.1, 2)) for u in units}
    eff = M.paired_effect(trt, ctl, group)
    assert eff["effect"] == pytest.approx(1.0, abs=0.1)
    base = {u: list(rng.normal(5, 0.3, 4)) for u in units}
    band = M.aa_band(base, 2, group)
    assert 0 < band < 0.3
    assert M.beyond_noise(eff, band)
    null = M.paired_effect({u: list(rng.normal(5, 0.3, 2)) for u in units}, ctl, group)
    assert not M.beyond_noise(null, band)


def test_beyond_noise_deterministic():
    assert M.beyond_noise({"ci": (0.01, 0.02)}, float("nan"), deterministic=True)
    assert not M.beyond_noise({"ci": (-0.01, 0.02)}, 0.0)
    assert not M.beyond_noise({"ci": (0.01, 0.02)}, 0.05)


def test_pair_units_cluster_on_both_items_and_same_group_counts_once():
    assert M.unit_weight({"g": 2, "h": 3}, ("g", "h")) == 6
    assert M.unit_weight({"g": 2}, ("g", "g")) == 2
    assert M.unit_weight({"g": 2}, "g") == 2
    # pairs that share items: CI must be wider than clustering on the first item alone
    rng = np.random.default_rng(1)
    items = [f"i{k}" for k in range(8)]
    pairs = [(a, b) for a in items for b in items if a < b]
    item_eff = {i: rng.normal(0, 1) for i in items}
    tr = {p: [item_eff[p[1]] + rng.normal(0, 0.1)] for p in pairs}
    ct = {p: [0.0] for p in pairs}
    both = M.paired_effect(tr, ct, {p: p for p in pairs})
    first = M.paired_effect(tr, ct, {p: p[0] for p in pairs})
    assert (both["ci"][1] - both["ci"][0]) > (first["ci"][1] - first["ci"][0])


def test_llm_arm_without_band_is_not_beyond_noise():
    assert not M.beyond_noise({"ci": (0.5, 0.9)}, float("nan"))
    assert M.beyond_noise({"ci": (0.5, 0.9)}, float("nan"), deterministic=True)
    assert not M.beyond_noise({"ci": (-0.9, -0.5)}, 0.1, direction=+1)
    assert M.beyond_noise({"ci": (-0.9, -0.5)}, 0.1)


def test_aa_band_skips_short_units():
    rng = np.random.default_rng(3)
    base = {f"u{i}": list(rng.normal(5, 0.3, 4)) for i in range(30)}
    base["u0"] = base["u0"][:3]                         # one invalid output
    band = M.aa_band(base, 2, {u: u for u in base})
    assert band == band and 0 < band < 0.3


def test_close_pairs_selection_amendment_a4():
    from dja import experiments as X
    ps = [Pair("a", "b", 4, 1, "g", "test"), Pair("c", "d", 5, 0, "g", "test"), Pair("e", "f", 2, 3, "g", "test"),
          Pair("g", "h", 1, 1, "g", "test"), Pair("i", "j", 17, 13, "v", "test"), Pair("k", "l", 4, 1, "g", "build")]
    assert X.close_xy(ps) == [("a", "b"), ("f", "e"), ("i", "j")]
