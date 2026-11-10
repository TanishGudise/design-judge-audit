"""Download and verify Vibe Design Arena (datapointai/vibe-design-arena, cc-by-4.0, HF-gated).

Accept the gate on your own HF account and put a read-only HF_TOKEN in .env. The token is read
by name and passed straight to huggingface_hub; it is never printed or written anywhere else.
Writes data/raw/vda/ (gitignored) and data/raw/vda/VERIFY.json (counts, license, revision; no personal data).

Checks: license cc-by-4.0 (API + card), 60 apps, 1,770 comparisons = every unordered pair once,
votes_a + votes_b == 30 on every pair, winner consistent with votes, screenshots decode (size recorded).
"""
from __future__ import annotations

import io
import itertools
import json
import os

from . import config

REPO = "datapointai/vibe-design-arena"
OUT = config.RAW / "vda"


def _token() -> str:
    env = config.ROOT / ".env"
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if line.startswith("HF_TOKEN="):
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            if v:
                return v
    raise SystemExit("HF_TOKEN not set in .env")


def main():
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import pandas as pd
    from huggingface_hub import HfApi, snapshot_download
    from PIL import Image

    tok = _token()
    info = HfApi().dataset_info(REPO, token=tok)
    rev = info.sha
    lic = (info.card_data or {}).get("license") if info.card_data else None
    path = snapshot_download(REPO, repo_type="dataset", revision=rev, token=tok, local_dir=str(OUT),
                             allow_patterns=["README.md", "apps/*.parquet", "comparisons/*.parquet"])
    del tok
    readme = (OUT / "README.md").read_text(errors="replace")
    apps = pd.read_parquet(next((OUT / "apps").glob("*.parquet")))
    comps = pd.concat([pd.read_parquet(f) for f in sorted((OUT / "comparisons").glob("*.parquet"))])
    ids = set(apps.app_id.astype(str))
    pairs = {tuple(sorted((str(a), str(b)))) for a, b in zip(comps.app_a, comps.app_b)}
    all_pairs = {tuple(sorted(p)) for p in itertools.combinations(sorted(ids), 2)}
    tot = (comps.votes_a + comps.votes_b)
    w = comps.winner.astype(str)     # stored as the label "app_a" / "app_b"
    win_ok = ((comps.votes_a > comps.votes_b) & (w == "app_a")) | ((comps.votes_b > comps.votes_a) & (w == "app_b")) | \
             (comps.votes_a == comps.votes_b)
    sizes = {}
    for v in apps.screenshot:
        b = v.get("bytes") if isinstance(v, dict) else v
        with Image.open(io.BytesIO(b)) as im:
            sizes[im.size] = sizes.get(im.size, 0) + 1
    share = (comps[["votes_a", "votes_b"]].max(axis=1) / tot)
    rep = {
        "repo": REPO, "revision": rev, "license_api": lic,
        "license_in_card": "cc-by-4.0" in readme.lower() or "cc by 4.0" in readme.lower(),
        "n_apps": len(apps), "n_comparisons": len(comps), "unique_pairs": len(pairs),
        "covers_all_pairs": pairs == all_pairs,
        "votes_per_pair_counts": {int(k): int(v) for k, v in sorted(tot.value_counts().items())},
        "pairs_with_30_votes": int((tot == 30).sum()), "total_votes": int(tot.sum()), "min_votes": int(tot.min()),
        "winner_consistent": bool(win_ok.all()), "screenshot_sizes": {f"{w}x{h}": n for (w, h), n in sizes.items()},
        "majority_share_quartiles": [round(float(x), 3) for x in share.quantile([0, .25, .5, .75, 1])],
        "pairs_clear_ge_20of30": int((share >= 2 / 3).sum()), "pairs_close_55_80": int(((share >= .55) & (share <= .8)).sum()),
        "columns_apps": [c for c in apps.columns if c not in ("creator", "location", "revenue")],
        "columns_comparisons": list(comps.columns),
    }
    (OUT / "VERIFY.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    ok = (rep["license_api"] == "cc-by-4.0" and rep["license_in_card"] and rep["n_apps"] == 60
          and rep["n_comparisons"] == 1770 and rep["covers_all_pairs"] and rep["winner_consistent"])
    print("VERIFIED" if ok else "VERIFICATION FAILED")
    if rep["pairs_with_30_votes"] != 1770:
        print(f"NOTE: the card says 30 votes per pair; {rep['pairs_with_30_votes']} of 1770 pairs have 30 "
              f"(min {rep['min_votes']}, total {rep['total_votes']}). Metrics use each pair's actual count.")


if __name__ == "__main__":
    main()
