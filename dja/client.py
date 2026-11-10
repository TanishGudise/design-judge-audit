"""Metered Anthropic calls: reserve worst case -> call -> settle real cost -> cache response.

The API key is read from ROOT/.env by name and handed straight to the SDK; it is never printed,
logged or written anywhere else. Every response is cached under runs/cache/<custom_id>.json together
with a hash of the exact request, so re-running a manifest costs nothing and `make reproduce`
regenerates every number offline.
"""
from __future__ import annotations

import base64
import concurrent.futures as cf
import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .ledger import CapExceeded, Ledger, Overrun


# ---------------------------------------------------------------------------------------------
# request objects
# ---------------------------------------------------------------------------------------------
@dataclass
class Req:
    custom_id: str                 # opaque, no labels (hash of the plan row)
    model_key: str                 # "haiku" | "sonnet"
    system: str
    content: list                  # Messages API user content blocks (images as base64)
    schema: dict                   # JSON schema for output_config.format
    image_dims: list = field(default_factory=list)  # [(w, h)] of each image, for the bound
    meta: dict = field(default_factory=dict)        # local only, never sent

    @property
    def model(self) -> str:
        return config.MODELS[self.model_key]

    def _content(self, materialize: bool) -> list:
        out = []
        for b in self.content:
            if b.get("type") == "image_ref":
                if materialize:
                    data = Path(b["path"]).read_bytes()
                    if hashlib.sha256(data).hexdigest() != b["sha256"]:
                        raise RuntimeError(f"image changed on disk since planning: {b['path']}")
                    out.append(image_block(data, b["media_type"]))
                else:
                    out.append({"type": "image", "sha256": b["sha256"], "media_type": b["media_type"]})
            else:
                out.append(b)
        return out

    def params(self, materialize: bool = True) -> dict:
        return {
            "model": self.model,
            "max_tokens": config.MAX_TOKENS,
            "system": self.system,
            "messages": [{"role": "user", "content": self._content(materialize)}],
            "thinking": config.THINKING[self.model_key],
            "output_config": {"effort": config.EFFORT,
                              "format": {"type": "json_schema", "schema": self.schema}},
        }

    def request_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.params(materialize=False), sort_keys=True).encode()).hexdigest()

    def input_bound(self) -> int:
        text_chars = len(self.system) + sum(len(b.get("text", "")) for b in self.content
                                            if b.get("type") == "text")
        text_chars += len(json.dumps(self.schema))
        img = sum(min(config.IMAGE_TOKEN_CAP, math.ceil(w / 28) * math.ceil(h / 28))
                  + config.PER_IMAGE_OVERHEAD for (w, h) in self.image_dims)
        return text_chars + img + config.PER_REQUEST_OVERHEAD

    def worst_case_usd(self) -> float:
        p = config.PRICES[self.model]
        return (self.input_bound() * p["in"] + config.MAX_TOKENS * p["out"]) / 1e6


def image_ref(path: str, media_type: str, sha256: str) -> dict:
    """Placeholder kept in a planned request; the bytes are read (and re-verified) only when sent."""
    return {"type": "image_ref", "path": str(path), "media_type": media_type, "sha256": sha256}


def image_block(png_or_jpeg: bytes, media_type: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                        "data": base64.standard_b64encode(png_or_jpeg).decode()}}


def _u(usage: dict, k: str) -> int:
    v = (usage or {}).get(k)
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def cost_usd(model: str, usage: dict, batch: bool) -> float:
    p = config.PRICES[model]
    c = (_u(usage, "input_tokens") * p["in"]
         + _u(usage, "cache_creation_input_tokens") * p["cache_write"]
         + _u(usage, "cache_read_input_tokens") * p["cache_read"]
         + _u(usage, "output_tokens") * p["out"]) / 1e6
    if (usage or {}).get("inference_geo") == "us":     # US-only routing is priced at 1.1x
        c *= 1.1
    return c * (config.BATCH_DISCOUNT if batch else 1.0)


def price_flags(usage: dict, batch: bool) -> list[str]:
    """Anything that would make the configured prices wrong (we never request these modes)."""
    u = usage or {}
    flags = []
    if u.get("speed") not in (None, "standard"):
        flags.append(f"speed={u.get('speed')}")
    if u.get("service_tier") not in (None, "standard", "batch"):
        flags.append(f"service_tier={u.get('service_tier')}")
    if u.get("inference_geo") not in (None, "global", "not_available"):
        flags.append(f"inference_geo={u.get('inference_geo')}")
    return flags


# ---------------------------------------------------------------------------------------------
# key and SDK client
# ---------------------------------------------------------------------------------------------
def _read_key() -> str:
    env = config.ROOT / ".env"
    if not env.exists():
        raise SystemExit("no .env in the project folder (add ANTHROPIC_API_KEY to a gitignored .env)")
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("ANTHROPIC_API_KEY="):
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            if v:
                return v
    raise SystemExit("ANTHROPIC_API_KEY not set in .env")


def make_sdk():
    import anthropic
    return anthropic.Anthropic(api_key=_read_key(), max_retries=0, timeout=120.0)


# ---------------------------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------------------------
def cache_path(custom_id: str) -> Path:
    return config.CACHE / f"{custom_id}.json"


def cached(req: Req) -> dict | None:
    p = cache_path(req.custom_id)
    if not p.exists():
        return None
    rec = json.loads(p.read_text())
    if rec.get("request_hash") != req.request_hash():
        raise RuntimeError(f"cache for {req.custom_id} was made from a different request; "
                           "custom ids must change when a request changes")
    return rec


def _write_cache(req: Req, rec: dict):
    config.CACHE.mkdir(parents=True, exist_ok=True)
    tmp = cache_path(req.custom_id).with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, sort_keys=True))
    tmp.replace(cache_path(req.custom_id))


def _record(req: Req, msg: dict, mode: str, cost: float, reserved: float) -> dict:
    text = next((b.get("text", "") for b in msg.get("content", []) if b.get("type") == "text"), "")
    rec = {"custom_id": req.custom_id, "request_hash": req.request_hash(), "model": req.model,
           "mode": mode, "cost_usd": cost, "reserved_usd": reserved,
           "usage": msg.get("usage") or {}, "stop_reason": msg.get("stop_reason"),
           "price_flags": price_flags(msg.get("usage") or {}, mode == "batch"),
           "text": text, "meta": req.meta, "input_bound": req.input_bound(), "t": time.time()}
    _write_cache(req, rec)
    return rec


# ---------------------------------------------------------------------------------------------
# live calls
# ---------------------------------------------------------------------------------------------
RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}
MAX_ATTEMPTS = 3


def _attempt_id(req: Req) -> str:
    return f"{req.custom_id}#{uuid.uuid4().hex[:10]}"


def run_live(reqs: list[Req], bucket: str, ledger: Ledger | None = None, workers: int = 4,
             sdk=None, log=print, sleep=time.sleep) -> list[dict]:
    """One call per request, each attempt under its own fresh reservation (SDK retries are off).
    - API status errors: the request was not billed -> release; retry 429/5xx up to MAX_ATTEMPTS.
    - connection errors / timeouts: outcome unknown -> settle the full reservation (conservative), retry.
    - anything failing after a response arrived: settle the full reservation, stop.
    Stops cleanly (no new calls) on CapExceeded, Overrun, a price-assumption flag or a non-retryable error."""
    import anthropic
    ledger = ledger or Ledger()
    out = {}
    todo = []
    for r in reqs:
        c = cached(r)
        if c is None:
            todo.append(r)
        else:
            out[r.custom_id] = c
    log(f"[live] {len(reqs)} requests, {len(out)} cached, {len(todo)} to call, bucket={bucket}")
    if not todo:
        return [out[r.custom_id] for r in reqs]
    sdk = sdk or make_sdk()
    stop = {"why": None}

    def one(req: Req):
        try:
            params = req.params()          # reads + re-verifies images locally; nothing is sent yet
        except Exception as e:
            stop["why"] = stop["why"] or f"could not build {req.custom_id} locally ({type(e).__name__}: {e})"
            return None
        for attempt in range(MAX_ATTEMPTS):
            if stop["why"]:
                return None
            rid = _attempt_id(req)
            reserved = req.worst_case_usd()
            try:
                ledger.reserve(rid, bucket, reserved, meta={"model": req.model, "cid": req.custom_id})
            except CapExceeded as e:
                stop["why"] = stop["why"] or str(e)
                return None
            try:
                msg = sdk.messages.create(**params)
            except anthropic.APIStatusError as e:
                ledger.release(rid, bucket, f"api_status_{e.status_code}")   # rejected requests are not billed
                if e.status_code in RETRYABLE_STATUS and attempt + 1 < MAX_ATTEMPTS:
                    sleep(2 ** attempt * 3)
                    continue
                stop["why"] = stop["why"] or f"API error {e.status_code} on {req.custom_id}"
                return None
            except anthropic.APIConnectionError as e:
                ledger.settle(rid, bucket, reserved, reserved, {"unknown_outcome": type(e).__name__})
                if attempt + 1 < MAX_ATTEMPTS:
                    sleep(2 ** attempt * 3)
                    continue
                stop["why"] = stop["why"] or f"{type(e).__name__} on {req.custom_id}"
                return None
            except Exception as e:
                ledger.settle(rid, bucket, reserved, reserved, {"unknown_outcome": type(e).__name__})
                stop["why"] = stop["why"] or f"{type(e).__name__} on {req.custom_id}"
                return None
            try:
                d = msg.to_dict()
                usage = d.get("usage") or {}
                if not isinstance(usage.get("input_tokens"), int):
                    raise ValueError("response without usage")
                cost = cost_usd(req.model, usage, batch=False)
            except Exception as e:   # billed but unreadable: count the full reservation
                ledger.settle(rid, bucket, reserved, reserved, {"unparsed_response": type(e).__name__})
                stop["why"] = stop["why"] or f"unreadable response on {req.custom_id}"
                return None
            try:
                ledger.settle(rid, bucket, cost, reserved, usage)
            except Overrun as e:
                stop["why"] = stop["why"] or str(e)
            flags = price_flags(usage, False)
            if flags:
                stop["why"] = stop["why"] or f"price assumption broken on {req.custom_id}: {flags}"
            return _record(req, d, "live", cost, reserved)
        return None

    with cf.ThreadPoolExecutor(workers) as ex:
        for rec in ex.map(one, todo):
            if rec:
                out[rec["custom_id"]] = rec
    if stop["why"]:
        log(f"[live] STOPPED: {stop['why']}")
    return [out.get(r.custom_id) for r in reqs]


# ---------------------------------------------------------------------------------------------
# batch calls (50% price; results usually within minutes, up to 24 h)
# ---------------------------------------------------------------------------------------------
MAX_BATCH_BYTES = 200_000_000   # API limit is 256 MB per batch; stay under it
MAX_BATCH_REQUESTS = 1000       # small chunks: reservations and results arrive in pieces
MAX_CHUNK_WORST_USD = 0.60      # a chunk's batch-price worst case; keeps any one reservation small


def _chunks(reqs: list[Req]):
    chunk, size, usd = [], 0, 0.0
    for r in reqs:
        u = r.worst_case_usd() * config.BATCH_DISCOUNT
        b = sum(Path(x["path"]).stat().st_size * 4 // 3 + 200 for x in r.content if x.get("type") == "image_ref")
        b += len(json.dumps(r.params(materialize=False)))
        if chunk and (size + b > MAX_BATCH_BYTES or len(chunk) >= MAX_BATCH_REQUESTS
                      or usd + u > MAX_CHUNK_WORST_USD):
            yield chunk
            chunk, size, usd = [], 0, 0.0
        chunk.append(r)
        size += b
        usd += u
    if chunk:
        yield chunk


def _batch_log_path() -> Path:
    return config.RUNS / "batches.jsonl"


def _batch_log_append(entry: dict):
    p = _batch_log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps({"t": time.time(), **entry}, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _open_batches() -> tuple[dict, list]:
    """Returns (open batches, unresolved create_unknown entries).
    open: batch id -> {"bucket", "rids": {custom_id: (reservation id, reserved, request hash)}}."""
    p = _batch_log_path()
    opened, closed, unknown, resolved = {}, set(), {}, set()
    if p.exists():
        for line in p.read_text().splitlines():
            e = json.loads(line)
            if e["kind"] == "submitted":
                opened[e["batch_id"]] = {"bucket": e["bucket"], "rids": {k: tuple(v) for k, v in e["rids"].items()}}
            elif e["kind"] == "processed":
                closed.add(e["batch_id"])
            elif e["kind"] == "create_unknown":
                unknown[e["attempt"]] = e
            elif e["kind"] == "create_unknown_resolved":
                resolved.add(e["attempt"])
    return ({b: v for b, v in opened.items() if b not in closed},
            [v for k, v in unknown.items() if k not in resolved])


def _orphan(cid: str, d: dict, why: str):
    """A billed result we must not cache as the answer to the current request: keep it for audit."""
    p = config.RUNS / "orphans" / f"{cid}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"why": why, "message": d}, sort_keys=True))


def run_batch(reqs: list[Req], bucket: str, ledger: Ledger | None = None, sdk=None,
              poll_s: float = 20.0, max_wait_s: float = 6 * 3600, max_in_flight: int = 4,
              lookup: dict | None = None, log=print) -> list[dict]:
    """Chunks are reserved all-or-nothing at their batch-price worst case (one ledger lock), submitted
    while they fit (up to max_in_flight), logged to runs/batches.jsonl with each request's hash, and settled
    from real usage when each batch ends. Open batches from an interrupted run are resumed first; their
    results are cached only when the request they answer is unchanged (same hash), otherwise kept as orphans.
    `lookup` maps every custom id of the frozen manifest to its Req (so a resume run filtered to one model
    still caches the other model's results). A create with unknown outcome blocks further submissions until
    resolved by hand. A chunk that doesn't fit waits for in-flight chunks; with nothing in flight the run
    stops cleanly. A finished chunk with zero successes stops the run."""
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    ledger = ledger or Ledger()
    by_cid = dict(lookup or {})
    by_cid.update({r.custom_id: r for r in reqs})
    stop = None
    open_b, unknown = _open_batches()
    if unknown:
        log(f"[batch] REFUSING to submit: {len(unknown)} batch create(s) with unknown outcome in runs/batches.jsonl; "
            "check `batches.list` and append a create_unknown_resolved entry first")
    in_flight = {}
    for bid, info in open_b.items():
        if all(cid in by_cid for cid in info["rids"]):
            in_flight[bid] = {**info, "t0": time.time()}
            log(f"[batch] resuming open batch {bid} ({len(info['rids'])} requests)")
        else:   # belongs to another stage/manifest: leave it open for the run that can cache its results
            log(f"[batch] leaving open batch {bid} for the stage it belongs to")
    if in_flight and sdk is None:
        sdk = make_sdk()
    resumed = {cid for v in in_flight.values() for cid in v["rids"]}
    todo = [r for r in reqs if cached(r) is None and r.custom_id not in resumed]
    log(f"[batch] {len(reqs)} requests, {len(reqs) - len(todo)} cached or in flight, {len(todo)} to submit, bucket={bucket}")
    pending = [] if unknown else [[c, None] for c in _chunks(todo)]   # [chunk, built payload]
    if pending and sdk is None:
        sdk = make_sdk()

    def finish(bid):
        nonlocal stop
        info = in_flight.pop(bid)
        rids = dict(info["rids"])
        ok = 0
        for res in sdk.messages.batches.results(bid):
            if res.custom_id not in rids:
                continue
            rid, reserved, h = rids.pop(res.custom_id)
            r = by_cid.get(res.custom_id)
            if res.result.type == "succeeded":
                try:
                    d = res.result.message.to_dict()
                    usage = d.get("usage") or {}
                    if not isinstance(usage.get("input_tokens"), int):
                        raise ValueError("result without usage")
                    cost = cost_usd(d.get("model") if r is None else r.model, usage, batch=True)
                except Exception as e:
                    ledger.settle(rid, info["bucket"], reserved, reserved, {"unparsed_response": type(e).__name__})
                    stop = stop or f"unreadable batch result in {bid}"
                    continue
                try:
                    ledger.settle(rid, info["bucket"], cost, reserved, usage)
                except Overrun as e:
                    stop = stop or str(e)
                if price_flags(usage, True):
                    stop = stop or f"price assumption broken in {bid}: {price_flags(usage, True)}"
                if r is None:
                    _orphan(res.custom_id, d, "custom id not in the frozen manifest")
                elif r.request_hash() != h:
                    _orphan(res.custom_id, d, "request changed since submission")
                else:
                    _record(r, d, "batch", cost, reserved)
                    ok += 1
            else:   # errored / canceled / expired requests are not billed
                ledger.release(rid, info["bucket"], f"batch_{res.result.type}")
        for cid, (rid, reserved, _) in rids.items():   # never returned: keep the reservation as spent
            ledger.settle(rid, info["bucket"], reserved, reserved, {"missing_from_results": True})
        _batch_log_append({"kind": "processed", "batch_id": bid, "succeeded": ok})
        if ok == 0:
            stop = stop or f"batch {bid} finished with 0 cached successes"
        log(f"[batch] {bid} processed: {ok} succeeded; ledger {ledger.summary()}")

    while (pending and not stop) or in_flight:
        while pending and not stop and len(in_flight) < max_in_flight:
            chunk = pending[0][0]
            if pending[0][1] is None:
                try:      # built (and images re-verified) once, before anything is reserved
                    pending[0][1] = [Request(custom_id=r.custom_id,
                                             params=MessageCreateParamsNonStreaming(**r.params())) for r in chunk]
                except Exception as e:
                    stop = f"could not build chunk locally ({type(e).__name__}: {e})"
                    break
            payload = pending[0][1]
            entries = [(_attempt_id(r), r.worst_case_usd() * config.BATCH_DISCOUNT,
                        {"model": r.model, "batch": True, "cid": r.custom_id}) for r in chunk]
            if not ledger.reserve_many(entries, bucket):
                if not in_flight:
                    stop = f"cap reached: next chunk of {len(chunk)} requests does not fit"
                break
            rids = {r.custom_id: (e[0], e[1], r.request_hash()) for r, e in zip(chunk, entries)}
            attempt = uuid.uuid4().hex[:12]
            try:
                b = sdk.messages.batches.create(requests=payload)
            except anthropic.APIStatusError as e:
                if e.status_code < 500:              # rejected request: nothing created, nothing billed
                    ledger.release_many([v[0] for v in rids.values()], bucket, f"batch_create_status_{e.status_code}")
                    stop = f"batch create rejected ({e.status_code})"
                    break
                err = f"status {e.status_code}"
            except Exception as e:
                err = type(e).__name__
            else:
                err = None
            if err is not None:
                # unknown: a batch may exist. Reservations stay OUTSTANDING (still counted against the cap) and
                # every detail is logged; `dja resolve <attempt>` later re-enters the batch as submitted (if it
                # exists in batches.list) or releases the reservations (if it doesn't).
                _batch_log_append({"kind": "create_unknown", "attempt": attempt, "bucket": bucket, "error": err,
                                   "rids": {k: list(v) for k, v in rids.items()}})
                stop = f"batch create outcome unknown ({err}); run `dja resolve {attempt}` after checking batches.list"
                break
            pending.pop(0)
            _batch_log_append({"kind": "submitted", "batch_id": b.id, "bucket": bucket,
                               "rids": {k: list(v) for k, v in rids.items()}})
            in_flight[b.id] = {"bucket": bucket, "rids": rids, "t0": time.time()}
            log(f"[batch] submitted {b.id}: {len(chunk)} requests, reserved ${sum(v[1] for v in rids.values()):.4f}")
        if in_flight:
            time.sleep(poll_s)
        for bid in list(in_flight):
            st = sdk.messages.batches.retrieve(bid)
            if st.processing_status == "ended":
                finish(bid)
            elif time.time() - in_flight[bid]["t0"] > max_wait_s:
                sdk.messages.batches.cancel(bid)
                log(f"[batch] {bid} cancel requested after {max_wait_s}s")
                in_flight[bid]["t0"] = float("inf")
    if stop:
        log(f"[batch] STOPPED: {stop}")
    return [cached(r) for r in reqs]


def resolve_create_unknown(attempt: str, batch_id: str | None, ledger: Ledger | None = None) -> str:
    """Resolve a create_unknown entry by hand after checking `batches.list`:
    batch_id given  -> the batch exists: log it as submitted so the next run resumes and settles it;
    batch_id None   -> nothing was created: release its reservations (idempotent)."""
    ledger = ledger or Ledger()
    _, unknown = _open_batches()
    e = next((u for u in unknown if u["attempt"] == attempt), None)
    if e is None:
        return f"no unresolved create_unknown entry {attempt}"
    if batch_id:
        _batch_log_append({"kind": "submitted", "batch_id": batch_id, "bucket": e["bucket"], "rids": e["rids"]})
    else:
        ledger.release_many([v[0] for v in e["rids"].values()], e["bucket"], "batch_create_never_happened")
    _batch_log_append({"kind": "create_unknown_resolved", "attempt": attempt, "batch_id": batch_id})
    return f"resolved {attempt}: " + (f"re-entered as {batch_id}" if batch_id else "reservations released")
