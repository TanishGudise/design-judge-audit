# Provenance

This repository is a single-commit export of the author's private working repository. Its commit log (39 commits) is
listed below, oldest first, in local time, 2026-11-09 to 2026-11-10, so the order is visible: each manifest under
runs/manifests was committed before the paid calls it lists, and results were committed after. Hashes refer to the
private repository. Left out of the export: two internal code-review notes and the batch-job log (batch ids).

```
890aa56e 2026-11-09 19:54 CP1 harness: renderers, judges, ledger, metered client, UIClip arms, analysis, tests
fdfb8f67 2026-11-09 19:54 Freeze pilot and main manifests
ce3510d5 2026-11-09 20:28 Fix harness review findings 1-16: single-use reservation ids, own retries, batch resume and atomic chunk reservations, freeze and leak-check hardening, pair-level clustering, stage-scoped analysis, GDE missing ratings, render v2
ad23a373 2026-11-09 20:29 Re-freeze pilot and main manifests after review fixes
3462a7c9 2026-11-09 20:44 Fix harness review r2 findings N1-N13: batch resume caches across model filters and only for unchanged requests, 5xx create = unknown (charged, blocks), idempotent settle/release, run lock, own-reservation waits, stage/mode/bucket rules, stage-scoped MF pairs, per-arm half-panel ceiling, A/A k from pre-registered reps, parse rates over sub-stages
02cb8526 2026-11-09 20:44 Re-freeze manifests at the r2-fixed harness (requests unchanged)
6e6054b1 2026-11-09 20:50 Review r3 fixes: resume only batches this stage can cache, unknown batch creates stay reserved until resolved (dja resolve), payload built once per chunk, pilot live=Haiku only, --only-dataset, ledger docs
af57093b 2026-11-09 20:51 Re-freeze manifests at the r3-fixed harness (requests unchanged)
e66f9cfd 2026-11-09 21:15 Amendment A4: flip tests on close pairs (human majority 55-80%) for GDE, reported beside the pre-registered clear-pair test
7ca1ba83 2026-11-09 21:16 Re-freeze main with A4 close-pair flips (existing requests unchanged)
edca282b 2026-11-09 21:17 VDA fetch + verification (token read by name from .env, never printed)
2b5eb468 2026-11-09 21:17 Re-freeze manifests (requests unchanged)
232edfdb 2026-11-09 21:35 VDA verification: winner is stored as app_a/app_b; report the vote-count distribution (card says 30/pair; 1,631 of 1,770 have 30)
54459787 2026-11-09 21:36 Re-freeze with Vibe Design Arena as the web dataset (verified rev ee85ae4); GDE and existing pilot requests unchanged
26b5d549 2026-11-09 21:37 Batch throughput: --max-in-flight and 200 MB chunks (API limit 256 MB); cap checks unchanged
d2270d15 2026-11-09 21:38 Re-freeze manifests (requests unchanged)
4d17535c 2026-11-09 22:25 Analysis: VDA rank correlation vs Bradley-Terry of test votes (pre-registered, was missing); majority-oracle bound (descriptive)
5ff5684d 2026-11-09 22:26 CP1 results: cached responses, ledger, batch log, embeddings, results/*.json and CP1_SUMMARY.md
5e43eba0 2026-11-09 22:36 CP2 (A5): note sanitizer, design locator, augmented classifier J4H, combination rules, CP2 manifests and stage rules
c419580e 2026-11-09 22:37 Freeze CP2 phase A (sanitizer + locator); re-freeze CP1 manifests (requests unchanged)
c91dd912 2026-11-09 22:37 CP2 phase A: sanitizer results (16 calls) and the sanitized-note table
ee676f72 2026-11-09 22:45 CP2 phase A: locator results (1,458 calls)
01efb0ad 2026-11-09 22:57 CP2 review r1 fixes (A6): alias arm (ii) rows that collapse to pixels-only, same-batch R2/R3 controls, seeded Sonnet subset, sanitized table built from code, A/A bands from control rep 1, stratified aggregate cut with positive-part sensitivity and bar flag, best-CP1-arm comparator per dataset, guards, tests
b1dc9fd9 2026-11-09 22:57 Freeze CP2 phases (cp2: 7,700 Haiku calls; cp2s: 680 Sonnet calls incl. 120 cached CP1 rows)
d2d9bf6d 2026-11-09 23:08 CP2 review r2 fixes: noise bands from all repeat pairs (scaled) and CP1 control reps for 1-rep controls, sanitized-table assert, ridge poisoning test; commit CP2 embeddings
1fef3b77 2026-11-09 23:09 Re-freeze CP2 manifests (requests unchanged)
fc2aad9e 2026-11-10 00:03 CP2 results: cached responses, ledger, results/cp2/cp2.json; make reproduce runs both analyses
135932be 2026-11-10 00:09 CP2 summary
1eb8dd5b 2026-11-10 01:49 CP3 harness (plan A7): new exploit types X1-X4 with matched controls, cp3 stage, analysis
d978fcc6 2026-11-10 01:50 Freeze CP3 manifest (3,240 Haiku pairwise calls, typical $0.51 batch)
3cbb7550 2026-11-10 02:33 CP1 analysis: skip renders only CP2 needs; CP3 analysis records render geometry
676841c1 2026-11-10 02:33 CP3 results: new-exploit lifts, duplicate-screenshot check, cached responses, ledger
10e66e80 2026-11-10 02:33 CP3 summary: reproduction note (CP1 drift found and fixed)
0299795c 2026-11-10 02:45 Neutral wording in comments and summaries (no author name); pre-registration file path
405306c5 2026-11-10 02:54 Post-review: paired probe-vs-LLM agreement (post-hoc), PREREGISTRATION.md, summary caveats
65d69f65 2026-11-10 02:55 Public README, DATA.md fetch layout, MF sample IDs, NOTICE attribution for derived data
719fdfd8 2026-11-10 02:55 README: pre-registration wording
c8bc4fe8 2026-11-10 03:08 Review r2: neutral probe docstring, prereg wording, README 97% scope and stamp caveat
ffffa5fe 2026-11-10 11:30 MIT license for the code; README/NOTICE state that datasets and models keep their own licenses
```
