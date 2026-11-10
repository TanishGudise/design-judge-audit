"""Dataset loaders -> one schema.

Item:  id, dataset, path (raw image), group (unit for splits/bootstrap), split ('build' | 'test'),
       human (dict of human scores for pointwise comparison)
Pair:  a, b (item ids), votes_a, votes_b (human pairwise votes), dataset, split

Splits are seeded and made at the group level, so every variant of a design (or every pair of an
app) lives in one split. Personal columns (VDA creator / location / revenue) are dropped at load.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


@dataclass
class Item:
    id: str
    dataset: str
    path: str
    group: str
    split: str = ""
    human: dict = field(default_factory=dict)
    kind: str = ""          # 'web' | 'graphic'


@dataclass
class Pair:
    a: str
    b: str
    votes_a: int
    votes_b: int
    dataset: str
    split: str = ""
    tag: str = ""           # e.g. GDE perturbation family

    @property
    def n(self):
        return self.votes_a + self.votes_b

    @property
    def share_a(self):
        return self.votes_a / self.n if self.n else 0.5


def _split_groups(groups, n_build: int, seed: int) -> dict:
    groups = sorted(set(groups))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    build = {groups[i] for i in perm[:n_build]}
    return {g: ("build" if g in build else "test") for g in groups}


# ---------------------------------------------------------------------------------------------
# GDE (Haraguchi et al., SIGGRAPH Asia 2024), Apache-2.0, commit 8666cb6
# alignment <- x-coordinate perturbation (lefttop_*); overlap, white space <- font size (size_*)
# ---------------------------------------------------------------------------------------------
GDE_DIR = config.RAW / "gde"
GDE_FAMILY = {"alignment": "lefttop", "overlap": "size", "whitespace": "size"}


def load_gde(n_build_templates: int = 33):
    csv = GDE_DIR / "csv"
    items: dict[str, Item] = {}
    ratings: dict[str, dict] = {}        # item id -> principle -> list of 5 ratings
    for principle, fam in GDE_FAMILY.items():
        d = pd.read_csv(csv / f"human_abs_{principle}.csv")
        for _, r in d.iterrows():
            folder = "org" if r.perturbation == "none" else f"{fam}_{r.perturbation}"
            iid = f"gde/{folder}/{r.id}"
            vals = [float(r[str(k)]) for k in range(5)]
            ratings.setdefault(iid, {})[principle] = [v for v in vals if v == v]   # some raters are missing
            if iid not in items:
                items[iid] = Item(iid, "gde", str(GDE_DIR / "img" / folder / f"{r.id}.png"),
                                  group=str(r.id), kind="graphic")
    for iid, it in items.items():
        rs = ratings[iid]
        means = [np.mean(v) for v in rs.values() if v]
        it.human = {"by_principle": rs, "mean": float(np.mean(means)) if means else float("nan")}
    pairs = []
    pooled: dict[tuple, list] = {}
    for principle, fam in GDE_FAMILY.items():
        d = pd.read_csv(csv / f"human_comp_{principle}.csv")
        for _, r in d.iterrows():
            org = f"gde/org/{r.id}"
            pert = f"gde/{fam}_{r.comparative}/{r.id}"
            votes = [r[f"better_design_{k}"] for k in range(5)]
            # 'none' = the unperturbed original is better; '<level>' = perturbed better; 'both' = tie
            va = sum(v == "none" for v in votes)
            vb = sum(v == r.comparative for v in votes)
            key = (org, pert)
            acc = pooled.setdefault(key, [0, 0, fam])
            acc[0] += va
            acc[1] += vb
    for (org, pert), (va, vb, fam) in pooled.items():
        pairs.append(Pair(org, pert, va, vb, "gde", tag=fam))
    split = _split_groups([it.group for it in items.values()], n_build_templates, config.SEED)
    for it in items.values():
        it.split = split[it.group]
    for p in pairs:
        p.split = items[p.a].split
    return items, pairs


# ---------------------------------------------------------------------------------------------
# Miniukovich & Figl web prototypicality, banks vertical, CC0 1.0 (Harvard Dataverse Z7KLIH v4)
# 200-page seeded sample stratified by AE decile (data/raw/mf_banks/meta/sample200.csv)
# ---------------------------------------------------------------------------------------------
MF_DIR = config.RAW / "mf_banks"


def mf_raw_ae() -> pd.DataFrame:
    """Raw aesthetics ratings: non-training, non-duplicate, sessions not flagged as cheating,
    z-scored within session (as the dataset's averages are)."""
    raw = pd.read_csv(MF_DIR / "meta" / "ratings.raw.banks.txt", sep="\t")
    demo = pd.read_csv(MF_DIR / "meta" / "demographics.banks.txt", sep="\t")
    cheat = set(demo.loc[demo.susCheat.astype(str).str.upper() == "TRUE", "sessionId"])
    raw = raw[(raw.dimension == "AE")]
    raw = raw[~raw.sessionId.isin(cheat)]
    raw = raw[raw.isTraining.astype(str).str.lower() != "true"]
    first = raw[raw.isDuplicate.astype(str).str.lower() != "true"].copy()
    g = first.groupby("sessionId")["rating"]
    first["z"] = (first.rating - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return first.dropna(subset=["z"])


def load_mf(n_build: int = 70):
    sample = pd.read_csv(MF_DIR / "meta" / "sample200.csv")
    avg = pd.read_csv(MF_DIR / "meta" / "ratings.avg.banks.txt", sep="\t").set_index("stimulusId")
    items = {}
    for s in sample.stimulusId:
        iid = f"mf/{s}"
        items[iid] = Item(iid, "mf", str(MF_DIR / "img" / s), group=s, kind="web",
                          human={"mean": float(avg.loc[s, "AE"])})
    split = _split_groups([it.group for it in items.values()], n_build, config.SEED + 1)
    for it in items.values():
        it.split = split[it.group]
    return items, []


# ---------------------------------------------------------------------------------------------
# Vibe Design Arena (datapointai), cc-by-4.0, HF-gated (accept the gate on your own HF account)
# ---------------------------------------------------------------------------------------------
VDA_DIR = config.RAW / "vda"
VDA_PERSONAL_COLUMNS = ["creator", "location", "revenue"]


def vda_available() -> bool:
    return (VDA_DIR / "apps").exists() and any((VDA_DIR / "comparisons").glob("*.parquet"))


def _img_bytes(v):
    if isinstance(v, dict):
        return v.get("bytes")
    return v


def load_vda(n_build: int = 20):
    apps = pd.read_parquet(next((VDA_DIR / "apps").glob("*.parquet")))
    apps = apps.drop(columns=[c for c in VDA_PERSONAL_COLUMNS if c in apps.columns])
    img_dir = config.DERIVED / "vda_src"
    img_dir.mkdir(parents=True, exist_ok=True)
    items = {}
    for _, r in apps.iterrows():
        aid = str(r.app_id)
        p = img_dir / f"{aid}.png"
        if not p.exists():
            from PIL import Image
            Image.open(io.BytesIO(_img_bytes(r.screenshot))).convert("RGB").save(p)
        items[f"vda/{aid}"] = Item(f"vda/{aid}", "vda", str(p), group=aid, kind="web")
    comps = pd.concat([pd.read_parquet(f) for f in sorted((VDA_DIR / "comparisons").glob("*.parquet"))])
    pairs = [Pair(f"vda/{r.app_a}", f"vda/{r.app_b}", int(r.votes_a), int(r.votes_b), "vda")
             for _, r in comps.iterrows()]
    split = _split_groups([it.group for it in items.values()], n_build, config.SEED + 2)
    for it in items.values():
        it.split = split[it.group]
    keep = []
    for p in pairs:
        sa, sb = items[p.a].split, items[p.b].split
        if sa == sb:          # cross-split pairs are used by neither side
            p.split = sa
            keep.append(p)
    return items, keep


def load(dataset: str):
    return {"gde": load_gde, "mf": load_mf, "vda": load_vda}[dataset]()
