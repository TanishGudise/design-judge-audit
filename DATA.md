# Data

No images or raw datasets are redistributed here. To re-run the analysis, put each dataset under `data/raw/`
(gitignored) exactly as below. `make reproduce` then needs no API key: every model response is in `runs/cache/`.

| Dataset | License | Where it goes |
|---|---|---|
| Vibe Design Arena (datapointai/vibe-design-arena, revision ee85ae467e14b1f454036544eb37eec0e2ab6368) | CC BY 4.0, gated on Hugging Face | Accept the gate on your own Hugging Face account, put a read-only `HF_TOKEN` in `.env`, run `python -m dja.fetch_vda` (downloads to `data/raw/vda/` and writes `VERIFY.json`). Creator, location and revenue columns are dropped at load. |
| GDE graphic-design evaluation (CyberAgentAILab/Graphic-design-evaluation, commit 8666cb6) | Apache-2.0 | Copy its `human_*` and `gpt_*` CSV files to `data/raw/gde/csv/` and extract its image archive so that `data/raw/gde/img/` holds `org/`, `lefttop_{small,medium,large}/` and `size_{small,medium,large}/`. |
| Web Design Prototypicality, banks (Miniukovich & Figl; Harvard Dataverse doi:10.7910/DVN/Z7KLIH, v4) | CC0 1.0 | Put `ratings.raw.banks.txt`, `ratings.avg.banks.txt` and `demographics.banks.txt` in `data/raw/mf_banks/meta/`, copy `data_ids/mf_banks/sample200.csv` there too, and download the 200 sampled screenshots listed in `data_ids/mf_banks/sample200_urls.txt` (Dataverse file id and md5 per file) to `data/raw/mf_banks/img/`. |
| UIClip (biglab/uiclip_jitteredwebsites-2-224-paraphrased_webpairs_humanpairs) + CLIP processor (openai/clip-vit-base-patch32) | MIT | Downloaded by `transformers` at the revisions pinned in `dja/clip_arm.py` (about 605 MB). Only needed to recompute embeddings; `runs/embeddings/` already holds them. |

See NOTICE for full credits.
