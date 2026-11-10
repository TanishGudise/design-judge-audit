"""python -m dja.cli <command>

  plan  <pilot|main>                 build the manifest (no API calls), print cost bounds, run the leak check
  run   <pilot|main> --mode live|batch --bucket <pilot|cp1|...> [--yes]
                                     execute a manifest under the ledger (cached calls are free)
  embed                              UIClip embeddings for every image the manifests use ($0, CPU, offline)
  analyze                            regenerate every number in results/ from the cache (offline)
  ledger                             spend summary
  blind                              blind-rating pairs for a human rater (original vs exploit, randomized)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

from . import client, config, experiments, stages
from .ledger import Ledger

MANIFESTS = config.RUNS / "manifests"


def _git_state():
    try:
        # code version = last commit touching the harness (manifests/results commits don't change it)
        sha = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", "dja"], cwd=config.ROOT,
                                      text=True).strip() or "unknown"
        dirty = (subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "dja"], cwd=config.ROOT).returncode != 0
                 or bool(subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "dja"],
                                                 cwd=config.ROOT, text=True).strip()))
    except Exception:
        sha, dirty = "unknown", True
    return sha, dirty


def build(stage: str, drop=()):
    ds = stages.load_all()
    items = {k: v for d in ds.values() for k, v in d[0].items()}
    if stage == "cp2a":
        from . import harden, render, stages_cp2
        san, loc = stages_cp2.cp2a_rows(ds)
        reqs = [harden.sanitize_req(text, f"{kind}:{d}:{n}") for kind, d, n, text in san]
        reqs += [harden.locate_req(iid, d, c, render.render_cached(iid, items[iid].path, d, c)) for d, iid, c in loc]
        rows = [r.meta for r in reqs]
        ids = [r.custom_id for r in reqs]
        assert len(ids) == len(set(ids)), "duplicate custom ids in manifest"
        return ds, rows, reqs
    if stage in ("cp2", "cp2s"):
        from . import stages_cp2
        rows = stages_cp2.cp2_rows(ds) if stage == "cp2" else stages_cp2.cp2s_rows(ds)
    elif stage == "cp3":
        from . import stages_cp3
        rows = stages_cp3.cp3_rows(ds)
    else:
        rows = stages.pilot_rows(ds) if stage == "pilot" else stages.main_rows(ds, drop=tuple(drop))
    reqs = [experiments.to_req(r, items) for r in rows]
    ids = [r.custom_id for r in reqs]
    assert len(ids) == len(set(ids)), "duplicate custom ids in manifest"
    return ds, rows, reqs


DECIMAL = re.compile(r"\d\.\d")


def banned_tokens(ds) -> set:
    banned = set()
    for d in ds.values():
        for iid in d[0]:
            banned.add(iid.split("/")[-1].rsplit(".", 1)[0])
    banned |= {"C1", "C2", "E1", "E2", "E3", "E4a", "E4b", "E5", "R2", "R3", "R4", "N0", "pad10", "jpeg40",
               "webp", "lefttop", "size_large", "exploit", "control", "W.flip", "P.base", "gde", "vda", "mf",
               "N0s", "R2s", "R3s", "E4bs", "F2", "W.build", "W.agree", "SAN", "LOC", "crop",
               "X1", "X2", "X3", "X4", "K1", "K2", "K3", "K4", "F3", "badge", "laptop", "watermark"}
    return banned


def leak_check_text(text: str, banned: set) -> str | None:
    """Return the first banned token found on letter/digit boundaries, or a decimal number (no human score,
    which are all non-integers, may appear; the prompts themselves contain no decimals)."""
    for tok in banned:
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(tok) + r"(?![A-Za-z0-9])", text):
            return tok
    m = DECIMAL.search(text)
    return m.group(0) if m else None


def leak_check(reqs, ds) -> list[str]:
    """No item id, dataset id, condition/test name or human score may appear in anything sent."""
    banned = banned_tokens(ds)
    bad = []
    for r in reqs:
        text = r.system + "\n" + "\n".join(b.get("text", "") for b in r.content if b.get("type") == "text")
        hit = leak_check_text(text, banned)
        if hit:
            bad.append(f"{r.custom_id}: '{hit}'")
    return bad


def cmd_plan(a):
    ds, rows, reqs = build(a.stage, a.drop)
    est_by = {}
    for mk in ("haiku", "sonnet"):
        sub = [r for r in reqs if r.model_key == mk]
        if sub:
            est_by[mk] = experiments.estimate(sub)
    tests = {}
    for r in rows:
        k = f"{r['dataset']}/{r['model_key']}/{r['test']}"
        tests[k] = tests.get(k, 0) + 1
    leaks = leak_check(reqs, ds)
    if leaks:
        print("LEAKS (manifest NOT written):", leaks[:10])
        sys.exit(2)
    sha, dirty = _git_state()
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    man = {"stage": a.stage, "git": sha, "dirty": dirty, "datasets": sorted(ds), "drop": a.drop,
           "n": len(reqs), "estimate": est_by, "rows": rows,
           "custom_ids": [r.custom_id for r in reqs], "request_hashes": [r.request_hash() for r in reqs]}
    (MANIFESTS / f"{a.stage}.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps({"stage": a.stage, "datasets": sorted(ds), "n": len(reqs), "estimate": est_by,
                      "git": sha[:10], "dirty": dirty, "leaks": len(leaks)}, indent=1))
    print("calls per dataset/model/test:")
    for k in sorted(tests):
        print(f"  {k:40s} {tests[k]}")


def _manifest_committed(path) -> bool:
    rel = str(path.relative_to(config.ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=config.ROOT,
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=config.ROOT).returncode == 0
    return tracked and clean


# Pre-registered execution per stage (plan A2/A3): main is batch-only on cp1; the pilot is live on 'pilot',
# except its Sonnet rows, which go through the batch path once to exercise it before main.
STAGE_RULES = {"pilot": {"bucket": "pilot", "modes": {"live", "batch"}},
               "main": {"bucket": "cp1", "modes": {"batch"}},
               # CP2 (A5): sanitizer calls live (16 tiny text calls), locator + judge calls batch
               "cp2a": {"bucket": "cp2", "modes": {"live", "batch"}},
               "cp2": {"bucket": "cp2", "modes": {"batch"}},
               "cp2s": {"bucket": "cp2", "modes": {"batch"}},
               # CP3 (A7): new exploit types, Haiku pairwise baseline, batch only
               "cp3": {"bucket": "cp3", "modes": {"batch"}}}
STAGES = ["pilot", "main", "cp2a", "cp2", "cp2s", "cp3"]


def cmd_run(a):
    import fcntl
    rule = STAGE_RULES[a.stage]
    if a.bucket != rule["bucket"] or a.mode not in rule["modes"]:
        sys.exit(f"refusing to run: stage {a.stage} is pre-registered as bucket {rule['bucket']}, "
                 f"modes {sorted(rule['modes'])}")
    if a.stage == "pilot" and a.mode == "batch" and a.only_model != "sonnet":
        sys.exit("refusing to run: the pilot uses batch mode only for its Sonnet rows (--only-model sonnet)")
    if a.stage == "pilot" and a.mode == "live" and a.only_model != "haiku":
        sys.exit("refusing to run: the pilot runs live only for its Haiku rows (--only-model haiku)")
    if a.stage == "cp2a" and (a.mode == "live") != (a.only_judge == "SAN"):
        sys.exit("refusing to run: cp2a runs the sanitizer (--only-judge SAN) live and the locator (--only-judge LOC) in batch")
    config.RUNS.mkdir(parents=True, exist_ok=True)
    lock_fd = open(config.RUNS / "run.lock", "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("refusing to run: another `dja run` holds runs/run.lock")
    man_p = MANIFESTS / f"{a.stage}.json"
    man = json.loads(man_p.read_text())
    if not _manifest_committed(man_p):
        sys.exit("refusing to run: the manifest is not committed or has uncommitted changes (re-freeze = commit)")
    if man.get("dirty"):
        sys.exit("refusing to run: the manifest was planned from uncommitted code")
    sha, dirty = _git_state()
    if dirty or sha != man["git"]:
        sys.exit(f"refusing to run: code is {'dirty' if dirty else 'at ' + sha[:10]}, manifest was built at "
                 f"{man['git'][:10]}. Commit and re-plan first.")
    ds, rows, reqs = build(a.stage, man.get("drop") or ())
    if [r.custom_id for r in reqs] != man["custom_ids"] or [r.request_hash() for r in reqs] != man["request_hashes"]:
        sys.exit("refusing to run: rebuilt requests differ from the frozen manifest")
    leaks = leak_check(reqs, ds)
    if leaks:
        sys.exit(f"refusing to run: leak check failed: {leaks[:5]}")
    lookup = {r.custom_id: r for r in reqs}        # whole frozen manifest (batch resume caches across filters)
    if a.only_model:
        reqs = [r for r in reqs if r.model_key == a.only_model]
    if a.only_dataset:      # execution order only; the frozen manifest and its analysis are unchanged
        reqs = [r for r in reqs if r.meta["dataset"] == a.only_dataset]
    if a.only_judge:
        reqs = [r for r in reqs if r.meta["judge"] == a.only_judge]
    todo = [r for r in reqs if client.cached(r) is None]
    wc = sum(r.worst_case_usd() for r in todo)
    led = Ledger()
    print(f"{len(reqs)} requests, {len(todo)} uncached; worst case ${wc:.4f} (live price); ledger {led.summary()}")
    if not a.yes:
        sys.exit("dry run (pass --yes to call the API)")
    if a.mode == "live":
        client.run_live(reqs, a.bucket, led, workers=a.workers)
    else:
        client.run_batch(reqs, a.bucket, led, lookup=lookup, max_in_flight=a.max_in_flight)
    print("ledger after:", led.summary())


def cmd_ledger(a):
    print(json.dumps(Ledger().summary(), indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dja")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--drop", nargs="*", default=[])
    p.set_defaults(fn=cmd_plan)
    r = sub.add_parser("run")
    r.add_argument("stage", choices=STAGES)
    r.add_argument("--mode", choices=["live", "batch"], required=True)
    r.add_argument("--bucket", choices=list(config.CAP_PARENTS), required=True)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--max-in-flight", type=int, default=4,
                   help="batch chunks in flight at once (each still needs room under the cap to be submitted)")
    r.add_argument("--only-model", choices=["haiku", "sonnet"])
    r.add_argument("--only-dataset", choices=["gde", "mf", "vda"])
    r.add_argument("--only-judge", choices=["J1", "J2", "J3", "SAN", "LOC"])
    r.add_argument("--yes", action="store_true")
    r.set_defaults(fn=cmd_run)
    sub.add_parser("ledger").set_defaults(fn=cmd_ledger)
    rs = sub.add_parser("resolve", help="resolve a batch create with unknown outcome (after checking batches.list)")
    rs.add_argument("attempt")
    rs.add_argument("--batch-id", default=None, help="the batch exists: re-enter it so the next run settles it")
    rs.set_defaults(fn=lambda a: print(client.resolve_create_unknown(a.attempt, a.batch_id)))
    e = sub.add_parser("embed")
    e.set_defaults(fn=lambda a: __import__("dja.embed_run", fromlist=["main"]).main())
    an = sub.add_parser("analyze")
    an.add_argument("stage", nargs="?", default="main", choices=["pilot", "main"])
    an.set_defaults(fn=lambda a: __import__("dja.analyze", fromlist=["main"]).main(a.stage))
    bl = sub.add_parser("blind")
    bl.set_defaults(fn=lambda a: __import__("dja.blind", fromlist=["main"]).main())
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
