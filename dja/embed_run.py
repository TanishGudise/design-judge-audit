"""UIClip embeddings for every image the arms need ($0, CPU, offline after the one-time download).

Needed images: every item's base render (agreement + J4 training) and every (item, condition) image
used by the pointwise and flip manifests. Stored in runs/embeddings/<dataset>.npz as key 'item|cond'.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import config, render, stages

EMB_DIR = config.RUNS / "embeddings"


def emb_path(name: str):
    return EMB_DIR / f"{name}_v{render.RENDER_VERSION}.npz"


def needed(ds) -> dict:
    keys = {name: set() for name in ds}
    for name, (items, _) in ds.items():
        for iid in items:
            keys[name].add((iid, "base"))
    for stage in ("pilot", "main"):
        p = config.RUNS / "manifests" / f"{stage}.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["rows"]:
            d = r["dataset"]
            if d not in keys:
                continue
            if r["judge"] in ("J1", "J3"):
                keys[d].add((r["item"], r["cond"]))
            else:
                keys[d].add((r["a"], r["cond_a"]))
                keys[d].add((r["b"], r["cond_b"]))
    return keys


def cp2_needed(ds) -> dict:
    """CP2 (A5): build items under the augmentation conditions, every test image any CP2 arm scores."""
    from . import stages_cp2
    keys = {name: set() for name in ds}
    if "vda" in ds:
        for d, iid, c in stages_cp2.all_cp2_image_keys(ds):
            keys[d].add((iid, c))
    return keys


def crop_keys(ds) -> dict:
    """(item, 'crop:<cond>') for every locator call that has a cached result (crop arm, A5)."""
    from . import cli, client, harden
    out = {name: {} for name in ds}
    p = config.RUNS / "manifests" / "cp2a.json"
    if not p.exists():
        return out
    _, _, reqs = cli.build("cp2a")
    items = {k: v for d in ds.values() for k, v in d[0].items()}
    for r in reqs:
        if r.meta["judge"] != "LOC":
            continue
        rec = client.cached(r)
        if rec is None:
            continue
        d, iid, c = r.meta["dataset"], r.meta["item"], r.meta["cond"]
        img = render.render_cached(iid, items[iid].path, d, c)
        out[d][(iid, f"crop:{c}")] = (img.path, harden.crop_box(rec, img.size))
    return out


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from . import clip_arm, harden
    ds = stages.load_all()
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    extra = cp2_needed(ds)
    crops = crop_keys(ds)
    for name, base_keys in needed(ds).items():
        ks = set(base_keys) | extra.get(name, set()) | set(crops.get(name, {}))
        out_p = emb_path(name)
        have = dict(np.load(out_p)) if out_p.exists() else {}
        items = ds[name][0]
        todo = sorted(k for k in ks if f"{k[0]}|{k[1]}" not in have)
        print(f"[embed] {name}: {len(ks)} images needed, {len(todo)} to embed")
        for i in range(0, len(todo), 16):
            chunk = todo[i:i + 16]
            imgs = []
            for iid, cond in chunk:
                if cond.startswith("crop:"):
                    path, box = crops[name][(iid, cond)]
                    imgs.append(harden.cropped_bytes(path, box))
                else:
                    r = render.render_cached(iid, items[iid].path, name, cond)
                    imgs.append(open(r.path, "rb").read())
            e = clip_arm.embed(imgs)
            for (iid, cond), v in zip(chunk, e):
                have[f"{iid}|{cond}"] = v.astype(np.float32)
            if (i // 16) % 20 == 0:
                np.savez(out_p, **have)
                print(f"[embed] {name}: {min(i + 16, len(todo))}/{len(todo)}", flush=True)
        np.savez(out_p, **have)
    print("[embed] done")


def cp3_main():
    """CP3 (A7): the 8 new conditions for the 40 test apps, in a separate file so CP2's inputs stay byte-identical."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from . import clip_arm, stages_cp3
    ds = stages.load_all()
    items = ds["vda"][0]
    out_p = emb_path("vda_cp3")
    have = dict(np.load(out_p)) if out_p.exists() else {}
    todo = sorted(k for k in stages_cp3.cp3_image_keys(ds) if f"{k[0]}|{k[1]}" not in have)
    print(f"[embed cp3] {len(todo)} to embed")
    for i in range(0, len(todo), 16):
        chunk = todo[i:i + 16]
        imgs = [open(render.render_cached(iid, items[iid].path, "vda", c).path, "rb").read() for iid, c in chunk]
        for (iid, c), v in zip(chunk, clip_arm.embed(imgs)):
            have[f"{iid}|{c}"] = v.astype(np.float32)
        np.savez(out_p, **have)
        print(f"[embed cp3] {min(i + 16, len(todo))}/{len(todo)}", flush=True)
    print("[embed cp3] done")


if __name__ == "__main__":
    import sys
    cp3_main() if "--cp3" in sys.argv[1:] else main()
