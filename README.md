# design-judge-audit

How AI judges of design taste get gamed, and a probe-based judge for web pages that is harder to game.

The audit scores LLM judges (Claude Haiku 5.5 as a rubric scorer, a pairwise judge and a yes/no checklist; Claude
Sonnet 5.5 as a carry-over check) and a small probe (a linear model on frozen UIClip embeddings) against public human
preferences: Vibe Design Arena (60 vibe-coded web apps, 52,292 pairwise votes; 20 apps to build, 40 held out) and GDE
graphic designs. Every bias and exploit is measured against a control with the same geometry, with 95% cluster
bootstrap intervals; tests were pre-registered before each paid run (`PREREGISTRATION.md`), and post-hoc additions
are labelled as such. Total API spend:
$5.71.

## Findings (details and every number in `results/CP1_SUMMARY.md`, `CP2_SUMMARY.md`, `CP3_SUMMARY.md`)
- The probe agrees with human voters as well as or better than the LLM judges on 780 held-out web-app pairs (0.591 vs
  0.545-0.572; level with the pairwise judge within noise). Two random voters agree 0.542; the best possible is 0.641.
- Each judge has its own easy exploit: a designer note in the rubric's words raises the LLM rubric score by +0.44 on a
  1-10 scale with the image unchanged; a flattery caption under the design flips 25% of clear verdicts for the probe; the
  pairwise judge changes its answer on 13.1% of pairs when the two designs swap places.
- Hardened web-page judge: the probe retrained with overlay versions of the build apps. On close held-out pairs it cuts
  the average lift of six exploits (mostly overlay types it was trained on) by 97% [58%, 142%], removes a held-out
  stamp's effect (new wording in a trained position), and keeps agreement (0.597 vs 0.591).
- New exploit types it never saw (CP3): it holds against a new device frame, an award seal and a watermark, but a quality
  claim written into the page itself still gets through (+0.067 [+0.007, +0.157]).
- Limitation: on graphic designs (GDE) the recipe does not hold (the LLM stays in charge and keeps its weakness to
  presentation polish). The pre-registered bar was met on close web-page pairs only.
- No human blind check was done; note exploits leave the design pixel-identical and image exploits are overlays compared
  against geometry-matched controls.

## Reproduce
```
make test        # offline tests (network blocked, no key read)
make reproduce   # recompute results/ from runs/cache + runs/embeddings, no API calls
```
`make reproduce` needs the datasets under `data/raw/` (see `DATA.md`). Re-running paid calls needs `ANTHROPIC_API_KEY`
in a gitignored `.env` and goes through the ledger's per-checkpoint caps.

## Layout
- `dja/` harness: `render.py` (conditions and matched controls), `judges.py` (prompts and schemas), `clip_arm.py`
  (UIClip zero-shot and the probe), `harden.py` (sanitizer, locator, augmented probe, combination rules), `client.py`
  (metered calls, cache, batch), `ledger.py` (worst-case reservations, nested caps), `metrics.py`, `stages*.py`
  (pre-registered manifests), `analyze*.py`.
- `runs/manifests/` frozen request lists; `runs/cache/` one JSON per API call; `runs/ledger.jsonl` spend;
  `runs/embeddings/` UIClip embeddings.
- `results/` derived numbers and checkpoint summaries (no images).
- `data_ids/` the 200-page Miniukovich & Figl sample (IDs only).

## Data and models
Vibe Design Arena (datapointai, CC BY 4.0), GDE (Haraguchi et al., Apache-2.0), Web Design Prototypicality (Miniukovich &
Figl, CC0), UIClip (Wu et al., MIT). Full credits in `NOTICE`; fetch instructions in `DATA.md`.

## License
The code in this repository (`dja/`, `tests/`, `Makefile`) is MIT-licensed (`LICENSE`). Datasets and models keep their
own licenses: Vibe Design Arena CC BY 4.0, GDE Apache-2.0, Web Design Prototypicality CC0, UIClip and CLIP MIT. None of
their images, raw files or model weights are re-hosted here; the repository holds item IDs, vote-derived numbers, UIClip
embeddings of the screenshots and the judges' cached text responses, shared with attribution under the source licenses
(see `NOTICE`).
