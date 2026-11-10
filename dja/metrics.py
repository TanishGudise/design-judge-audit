"""Agreement with humans, human-human ceilings, effects vs the A/A noise band, cluster bootstrap.

Conventions
- A judge's pairwise output is stored as p_a in {0, 0.5, 1}: the probability it picked pair side 'a'
  (pointwise judges: compare mean scores; ties = 0.5). Calls in both orders are re-oriented to 'a'.
- Agreement with a random human (vote data): sum over calls of p_a*share_a + (1-p_a)*share_b.
  Human-human ceiling on the same pairs: [a(a-1) + b(b-1)] / [n(n-1)] (two distinct random voters agree).
- CIs: cluster bootstrap over groups (apps / templates / pages), 2,000 resamples. A pair's weight in a
  resample is m(group a) * m(group b).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

B = 2000


# ---------------------------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------------------------
def cluster_bootstrap(stat, groups: list, n: int = B, seed: int = 0) -> np.ndarray:
    """stat(weights: dict group->multiplicity) -> float. Returns n bootstrap values."""
    groups = sorted(set(groups))
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        draw = rng.integers(0, len(groups), len(groups))
        w = defaultdict(int)
        for d in draw:
            w[groups[d]] += 1
        out[i] = stat(w)
    return out


def ci(vals, alpha=0.05):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def _groups(spec):
    """A unit's cluster spec: one group, or a tuple of groups (a pair). Returns the distinct groups."""
    return set(spec) if isinstance(spec, tuple) else {spec}


def unit_weight(w, spec) -> int:
    """Resample weight of a unit = product of its distinct groups' multiplicities (a pair whose two items
    share a group counts m, not m^2)."""
    out = 1
    for g in _groups(spec):
        out *= w.get(g, 0)
    return out


def wmean(xs, ws):
    xs, ws = np.asarray(xs, float), np.asarray(ws, float)
    s = ws.sum()
    return float((xs * ws).sum() / s) if s > 0 else float("nan")


# ---------------------------------------------------------------------------------------------
# human ceilings
# ---------------------------------------------------------------------------------------------
def hh_agree(va: int, vb: int) -> float:
    n = va + vb
    return (va * (va - 1) + vb * (vb - 1)) / (n * (n - 1)) if n > 1 else float("nan")


def bt_scores(pairs, items: list, iters: int = 500) -> dict:
    """Bradley-Terry strengths (log scale) from vote counts, Hunter's MM algorithm."""
    idx = {it: i for i, it in enumerate(items)}
    k = len(items)
    wins = np.zeros(k)
    nij = np.zeros((k, k))
    for p in pairs:
        if p.a in idx and p.b in idx:
            i, j = idx[p.a], idx[p.b]
            wins[i] += p.votes_a
            wins[j] += p.votes_b
            nij[i, j] += p.n
            nij[j, i] += p.n
    g = np.ones(k)
    for _ in range(iters):
        denom = (nij / (g[:, None] + g[None, :])).sum(1)
        g = np.where(denom > 0, (wins + 0.5) / denom, g)
        g /= np.exp(np.log(g).mean())
    return {it: float(np.log(g[idx[it]])) for it in items}


def split_half_votes(pairs, items, rng, reps=200):
    """Spearman between BT rankings from two random halves of every pair's voters, and the
    Spearman-Brown estimate for the full panel."""
    from dataclasses import replace
    rs = []
    for _ in range(reps):
        h1, h2 = [], []
        for p in pairs:
            votes = np.array([1] * p.votes_a + [0] * p.votes_b)
            rng.shuffle(votes)
            m = len(votes) // 2
            a1, a2 = votes[:m].sum(), votes[m:2 * m].sum()
            h1.append(replace(p, votes_a=int(a1), votes_b=int(m - a1)))
            h2.append(replace(p, votes_a=int(a2), votes_b=int(m - a2)))
        s1, s2 = bt_scores(h1, items), bt_scores(h2, items)
        rs.append(spearmanr([s1[i] for i in items], [s2[i] for i in items]).statistic)
    r = float(np.mean(rs))
    return r, 2 * r / (1 + r)


def split_half_ratings(ratings: dict, rng, reps=200):
    """ratings: item -> list of individual ratings. Spearman between half-panel means + Spearman-Brown."""
    items = [i for i, v in ratings.items() if len(v) >= 2]
    rs = []
    for _ in range(reps):
        m1, m2 = [], []
        for i in items:
            v = np.array(ratings[i], float)
            rng.shuffle(v)
            h = len(v) // 2
            m1.append(v[:h].mean())
            m2.append(v[h:].mean())
        rs.append(spearmanr(m1, m2).statistic)
    r = float(np.mean(rs))
    return r, 2 * r / (1 + r)


# ---------------------------------------------------------------------------------------------
# agreement
# ---------------------------------------------------------------------------------------------
def agreement_table(pairs, pa_by_pair: dict, group_of: dict, clear_share: float = 2 / 3, seed=0):
    """pa_by_pair: (a, b) -> list of p_a values (calls). Returns judge-vs-human and human-human stats."""
    rows = []
    for p in pairs:
        calls = pa_by_pair.get((p.a, p.b))
        if not calls:
            continue
        pa = float(np.mean(calls))
        j = pa * p.share_a + (1 - pa) * (1 - p.share_a)
        clear = max(p.share_a, 1 - p.share_a) >= clear_share
        maj = 1.0 if p.share_a > 0.5 else 0.0 if p.share_a < 0.5 else 0.5
        acc = 1 - abs(pa - maj) if clear else np.nan
        rows.append((p, j, hh_agree(p.votes_a, p.votes_b), acc, pa, max(p.share_a, 1 - p.share_a)))

    def stat(w, col):
        xs, ws = [], []
        for r in rows:
            wt = unit_weight(w, (group_of[r[0].a], group_of[r[0].b]))
            if wt and not np.isnan(r[col]):
                xs.append(r[col])
                ws.append(wt)
        return wmean(xs, ws) if xs else np.nan

    groups = list({group_of[r[0].a] for r in rows} | {group_of[r[0].b] for r in rows})
    full = defaultdict(int, {g: 1 for g in groups})
    out = {"n_pairs": len(rows), "n_clear": int(sum(not np.isnan(r[3]) for r in rows))}
    for col, name in [(1, "judge_human"), (2, "human_human"), (3, "clear_majority_acc"), (5, "majority_oracle")]:
        est = stat(full, col)
        bs = cluster_bootstrap(lambda w: stat(w, col), groups, seed=seed + col)
        out[name] = (est, *ci(bs))
    # judge minus ceiling, paired in the same resample
    bs_gap = cluster_bootstrap(lambda w: stat(w, 1) - stat(w, 2), groups, seed=seed + 9)
    out["judge_minus_ceiling"] = (stat(full, 1) - stat(full, 2), *ci(bs_gap))
    return out


def rank_corr(judge_scores: dict, human_scores: dict, group_of: dict, seed=0):
    items = [i for i in judge_scores if i in human_scores and not np.isnan(judge_scores[i])]
    est = spearmanr([judge_scores[i] for i in items], [human_scores[i] for i in items]).statistic

    def stat(w):
        xs, ys = [], []
        for i in items:
            for _ in range(w.get(group_of[i], 0)):
                xs.append(judge_scores[i])
                ys.append(human_scores[i])
        return spearmanr(xs, ys).statistic if len(set(xs)) > 1 and len(set(ys)) > 1 else np.nan

    bs = cluster_bootstrap(stat, [group_of[i] for i in items], seed=seed)
    return (float(est), *ci(bs)), len(items)


# ---------------------------------------------------------------------------------------------
# effects vs the A/A band
# ---------------------------------------------------------------------------------------------
def paired_effect(treated: dict, control: dict, group_of: dict, seed=0):
    """treated/control: unit -> list of values (reps). Effect = mean over units of (mean T - mean C).
    group_of[unit] is a group or a tuple of groups (pairs: both items' groups)."""
    units = [u for u in treated if u in control and len(treated[u]) and len(control[u])
             and not np.isnan(np.mean(treated[u])) and not np.isnan(np.mean(control[u]))]
    d = {u: np.mean(treated[u]) - np.mean(control[u]) for u in units}

    def stat(w):
        xs, ws = [], []
        for u in units:
            wt = unit_weight(w, group_of[u])
            if wt:
                xs.append(d[u])
                ws.append(wt)
        return wmean(xs, ws)

    all_groups = sorted(set().union(*[_groups(group_of[u]) for u in units])) if units else []
    full = {g: 1 for g in all_groups}
    est = stat(full)
    bs = cluster_bootstrap(stat, all_groups, seed=seed) if units else np.array([np.nan])
    return {"effect": est, "ci": ci(bs), "n_units": len(units), "boot": bs}


def aa_band(reps_by_unit: dict, k: int, group_of: dict, seed=0, n=B) -> float:
    """95th percentile of |A/A effect|: two disjoint random k-rep subsets of the same condition,
    compared exactly like a treated-vs-control effect, resampled over groups and over the split."""
    units = [u for u, v in reps_by_unit.items() if len(v) >= 2 * k]
    if not units:
        return float("nan")
    rng = np.random.default_rng(seed)
    groups = sorted({group_of[u] for u in units})
    by_group = defaultdict(list)
    for u in units:
        by_group[group_of[u]].append(u)
    vals = np.empty(n)
    for i in range(n):
        draw = rng.integers(0, len(groups), len(groups))
        xs = []
        for gi in draw:
            for u in by_group[groups[gi]]:
                v = np.array(reps_by_unit[u], float)
                perm = rng.permutation(len(v))
                xs.append(v[perm[:k]].mean() - v[perm[k:2 * k]].mean())
        vals[i] = abs(np.mean(xs))
    return float(np.quantile(vals, 0.95))


def beyond_noise(effect: dict, band: float, deterministic: bool = False, direction: int = 0) -> bool:
    """CI excludes 0 AND its bound nearest 0 is farther from 0 than the A/A band.
    deterministic=True (J0/J4: no run-to-run noise): CI excluding 0 suffices. An LLM arm without an A/A
    band (NaN) is never 'beyond noise'. direction=+1 requires a positive effect (exploits must raise scores)."""
    lo, hi = effect["ci"]
    if lo != lo or hi != hi or lo <= 0 <= hi:
        return False
    if direction > 0 and lo <= 0:
        return False
    if deterministic:
        return True
    if band != band:
        return False
    return min(abs(lo), abs(hi)) > band
