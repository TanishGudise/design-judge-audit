import json
import types

import numpy as np
import pytest
from PIL import Image

from dja import client, config, judges, render
from dja.ledger import Ledger


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DERIVED", tmp_path / "derived")
    monkeypatch.setattr(config, "CACHE", tmp_path / "cache")
    render._MEMO.clear()
    src = tmp_path / "src.png"
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (300, 480, 3), dtype=np.uint8)).save(src)
    led = Ledger(tmp_path / "ledger.jsonl")
    return src, led


def _req(src, judge="J1", rep=0, model="haiku", cond="E4a", note=None):
    img = render.render_cached("gde/org/abc123def456", str(src), "gde", cond)
    row = {"judge": judge, "model_key": model, "dataset": "gde", "test": f"P.{cond}", "item": "gde/org/abc123def456",
           "cond": cond, "note": note, "rep": rep, "stage": "t"}
    if judge == "J2":
        row.update(a="x", b="y")
        return judges.build_j2("gde", img, img, note, None, model, row)
    return (judges.build_j1 if judge == "J1" else judges.build_j3)("gde", img, note, model, row)


def test_prompts_have_no_ids_or_condition_names(env):
    src, _ = env
    for j in ("J1", "J2", "J3"):
        for note in (None, "E4b", "R3"):
            r = _req(src, j, note=note)
            text = r.system + " ".join(b.get("text", "") for b in r.content if b.get("type") == "text")
            for tok in ("abc123def456", "gde", "E4a", "E4b", "R3", "exploit", "control", "P."):
                assert tok not in text, (j, note, tok)


def test_custom_ids_unique_per_rep_and_stable(env):
    src, _ = env
    a, b, a2 = _req(src, rep=0), _req(src, rep=1), _req(src, rep=0)
    assert a.custom_id != b.custom_id and a.custom_id == a2.custom_id
    assert a.request_hash() == b.request_hash()          # same request, different repeat


def test_bound_covers_images_and_text(env):
    src, _ = env
    r = _req(src, "J2")
    w, h = r.image_dims[0]
    img_tok = -(-w // 28) * -(-h // 28)
    text = sum(len(b.get("text", "")) for b in r.content if b.get("type") == "text") + len(r.system)
    assert r.input_bound() >= 2 * img_tok + text
    assert r.worst_case_usd() > 0


def test_materialize_and_tamper_detection(env):
    src, _ = env
    r = _req(src)
    p = r.params()
    blocks = p["messages"][0]["content"]
    assert blocks[0]["type"] == "image" and len(blocks[0]["source"]["data"]) > 100
    assert "image_ref" not in json.dumps(r.params(materialize=False))
    path = next(b["path"] for b in r.content if b.get("type") == "image_ref")
    with open(path, "ab") as f:
        f.write(b"x")
    with pytest.raises(RuntimeError):
        r.params()


def test_parse_validates():
    mk = lambda j, text, stop="end_turn": {"meta": {"judge": j}, "text": text, "stop_reason": stop}
    ok1 = json.dumps({"hierarchy": 7, "layout_spacing": 6, "typography": 8, "color_contrast": 7,
                      "consistency": 7, "overall": 7, "reason": "r"})
    assert judges.parse(mk("J1", ok1))["score"] == 7.0
    assert judges.parse(mk("J1", ok1.replace('"overall": 7', '"overall": 11'))) is None
    assert judges.parse(mk("J1", ok1, stop="refusal")) is None
    assert judges.parse(mk("J1", ok1.replace('"overall": 7', '"overall": true'))) is None
    assert judges.parse(mk("J2", json.dumps({"winner": "B", "confidence": 3, "reason": ""})))["winner"] == "B"
    assert judges.parse(mk("J2", json.dumps({"winner": "C", "confidence": 3, "reason": ""}))) is None
    j3 = {k: True for k, _ in judges.J3_CHECKS}
    j3["no_overlap"] = False
    assert judges.parse(mk("J3", json.dumps(j3)))["score"] == pytest.approx(0.9)
    assert judges.parse(mk("J3", "not json")) is None


# ---- fake SDK ------------------------------------------------------------------------------
class _Msg:
    def __init__(self, usage, text='{"winner": "A", "confidence": 3, "reason": "x"}'):
        self._d = {"content": [{"type": "text", "text": text}], "usage": usage, "stop_reason": "end_turn"}

    def to_dict(self):
        return self._d


class FakeSDK:
    def __init__(self, usage):
        self.calls = 0
        self.usage = usage
        sdk = self

        class Messages:
            def create(self_inner, **kw):
                sdk.calls += 1
                assert kw["messages"][0]["content"][0]["type"] in ("image", "text")
                return _Msg(sdk.usage)

        class Batches:
            def __init__(self_inner):
                self_inner.store = {}

            def create(self_inner, requests):
                bid = f"b{len(self_inner.store)}"
                self_inner.store[bid] = [r["custom_id"] if isinstance(r, dict) else r.custom_id for r in requests]
                sdk.calls += len(requests)
                return types.SimpleNamespace(id=bid)

            def retrieve(self_inner, bid):
                return types.SimpleNamespace(processing_status="ended")

            def results(self_inner, bid):
                for cid in self_inner.store[bid]:
                    yield types.SimpleNamespace(custom_id=cid, result=types.SimpleNamespace(
                        type="succeeded", message=_Msg(sdk.usage)))

            def cancel(self_inner, bid):
                pass

        self.messages = Messages()
        self.messages.batches = Batches()


def test_run_live_settles_caches_and_is_free_on_rerun(env):
    src, led = env
    reqs = [_req(src, "J2", rep=i) for i in range(5)]
    sdk = FakeSDK({"input_tokens": 900, "output_tokens": 60})
    out = client.run_live(reqs, "pilot", led, workers=3, sdk=sdk, log=lambda *a: None)
    assert sdk.calls == 5 and all(o and o["mode"] == "live" for o in out)
    s = led.summary()
    assert s["pilot"] == pytest.approx(5 * (900 * 0.10 + 60 * 0.50) / 1e6)
    assert s["outstanding_reservations"] == 0
    client.run_live(reqs, "pilot", led, sdk=sdk, log=lambda *a: None)
    assert sdk.calls == 5


def test_run_live_stops_on_overrun(env):
    src, led = env
    reqs = [_req(src, "J2", rep=i) for i in range(3)]
    sdk = FakeSDK({"input_tokens": 10_000_000, "output_tokens": 60})
    logs = []
    client.run_live(reqs, "pilot", led, workers=1, sdk=sdk, log=logs.append)
    assert sdk.calls == 1 and any("STOPPED" in l for l in logs)


def test_run_batch_reserves_at_batch_price_and_settles(env):
    src, led = env
    reqs = [_req(src, "J1", rep=i) for i in range(4)]
    sdk = FakeSDK({"input_tokens": 800, "output_tokens": 100})
    out = client.run_batch(reqs, "cp1", led, sdk=sdk, poll_s=0, log=lambda *a: None)
    assert all(o and o["mode"] == "batch" for o in out)
    assert led.summary()["cp1"] == pytest.approx(4 * 0.5 * (800 * 0.10 + 100 * 0.50) / 1e6)
    lines = [json.loads(l) for l in (led.path).read_text().splitlines()]
    res = [l for l in lines if l["kind"] == "reserve"]
    assert all(abs(l["amount"] - r.worst_case_usd() * 0.5) < 1e-12 for l, r in zip(res, reqs))


def test_run_batch_stops_cleanly_at_cap(env, tmp_path):
    src, _ = env
    led = Ledger(tmp_path / "small.jsonl", caps={"pilot": 9, "cp1": 0.0001, "cp2": 9, "cp3": 9,
                                                 "reserve": 9, "total": 9}, stop_fraction=1.0)
    reqs = [_req(src, "J1", rep=i) for i in range(4)]
    sdk = FakeSDK({"input_tokens": 800, "output_tokens": 100})
    logs = []
    client.run_batch(reqs, "cp1", led, sdk=sdk, poll_s=0, log=logs.append)
    assert sdk.calls == 0 and any("STOPPED" in l for l in logs)
    assert led.summary()["outstanding_reservations"] == 0


# ---- failure paths -------------------------------------------------------------------------
def _status_error(code):
    import anthropic
    import httpx2
    resp = httpx2.Response(code, request=httpx2.Request("POST", "https://api.invalid/v1/messages"))
    return anthropic.APIStatusError("err", response=resp, body=None)


def test_live_retries_retryable_status_with_fresh_reservations(env):
    src, led = env
    req = _req(src, "J2")
    sdk = FakeSDK({"input_tokens": 900, "output_tokens": 60})
    real = sdk.messages.create
    state = {"n": 0}

    def flaky(**kw):
        state["n"] += 1
        if state["n"] == 1:
            raise _status_error(529)
        return real(**kw)
    sdk.messages.create = flaky
    out = client.run_live([req], "pilot", led, sdk=sdk, log=lambda *a: None, sleep=lambda s: None)
    assert out[0] is not None and state["n"] == 2
    lines = [json.loads(l) for l in led.path.read_text().splitlines()]
    assert [l["kind"] for l in lines] == ["reserve", "release", "reserve", "settle"]
    assert lines[0]["id"] != lines[2]["id"]


def test_live_response_without_usage_is_charged_in_full(env):
    src, led = env
    req = _req(src, "J2")
    sdk = FakeSDK(None)
    logs = []
    out = client.run_live([req], "pilot", led, sdk=sdk, log=logs.append, sleep=lambda s: None)
    assert out[0] is None and any("STOPPED" in l for l in logs)
    assert led.summary()["pilot"] == pytest.approx(req.worst_case_usd(), abs=1e-6)
    assert led.summary()["outstanding_reservations"] == 0


def test_batch_create_rejected_releases_and_stops(env):
    src, led = env
    reqs = [_req(src, "J1", rep=i) for i in range(3)]
    sdk = FakeSDK({"input_tokens": 800, "output_tokens": 100})

    def reject(requests):
        raise _status_error(400)
    sdk.messages.batches.create = reject
    logs = []
    client.run_batch(reqs, "cp1", led, sdk=sdk, poll_s=0, log=logs.append)
    s = led.summary()
    assert s["cp1"] == 0 and s["outstanding_reservations"] == 0 and any("STOPPED" in l for l in logs)


def test_batch_all_errored_chunk_stops(env, monkeypatch):
    src, led = env
    monkeypatch.setattr(client, "MAX_BATCH_REQUESTS", 2)
    reqs = [_req(src, "J1", rep=i) for i in range(6)]
    sdk = FakeSDK({"input_tokens": 800, "output_tokens": 100})
    store = sdk.messages.batches

    def results(bid):
        for cid in store.store[bid]:
            yield types.SimpleNamespace(custom_id=cid, result=types.SimpleNamespace(type="errored"))
    store.results = results
    logs = []
    client.run_batch(reqs, "cp1", led, sdk=sdk, poll_s=0, max_in_flight=1, log=logs.append)
    assert len(store.store) == 1, "no further chunks after a chunk with zero successes"
    assert led.summary()["outstanding_reservations"] == 0 and led.summary()["cp1"] == 0


def test_batch_resumes_open_batches_without_resubmitting(env):
    src, led = env
    reqs = [_req(src, "J1", rep=i) for i in range(3)]
    sdk = FakeSDK({"input_tokens": 800, "output_tokens": 100})
    store = sdk.messages.batches
    # simulate an interrupted run: reservations made, batch submitted and logged, never processed
    entries = [(f"{r.custom_id}#old", r.worst_case_usd() * 0.5, {}) for r in reqs]
    assert led.reserve_many(entries, "cp1")
    store.store["b_old"] = [r.custom_id for r in reqs]
    client._batch_log_append({"kind": "submitted", "batch_id": "b_old", "bucket": "cp1",
                              "rids": {r.custom_id: [e[0], e[1], r.request_hash()] for r, e in zip(reqs, entries)}})
    sdk.calls = 0
    out = client.run_batch(reqs, "cp1", led, sdk=sdk, poll_s=0, log=lambda *a: None)
    assert sdk.calls == 0, "resumed batch must not be resubmitted"
    assert all(o and o["mode"] == "batch" for o in out)
    assert led.summary()["outstanding_reservations"] == 0
    assert client._open_batches() == ({}, [])


def _open_batch(led, store, reqs, hashes=None, bid="b_old"):
    entries = [(f"{r.custom_id}#{bid}", r.worst_case_usd() * 0.5, {}) for r in reqs]
    assert led.reserve_many(entries, "cp1")
    store.store[bid] = [r.custom_id for r in reqs]
    hashes = hashes or [r.request_hash() for r in reqs]
    client._batch_log_append({"kind": "submitted", "batch_id": bid, "bucket": "cp1",
                              "rids": {r.custom_id: [e[0], e[1], h] for r, e, h in zip(reqs, entries, hashes)}})


def test_resume_keeps_other_models_results_via_lookup(env):
    src, led = env
    h = [_req(src, "J1", rep=i, model="haiku") for i in range(2)]
    s = [_req(src, "J1", rep=i, model="sonnet") for i in range(2)]
    sdk = FakeSDK({"input_tokens": 800, "output_tokens": 100})
    _open_batch(led, sdk.messages.batches, s)
    sdk.calls = 0
    lookup = {r.custom_id: r for r in h + s}
    client.run_batch(h, "cp1", led, sdk=sdk, poll_s=0, lookup=lookup, log=lambda *a: None)
    assert all(client.cached(r) for r in s), "the resumed Sonnet results are cached, not discarded"


def test_resume_never_caches_a_result_for_a_changed_request(env):
    src, led = env
    reqs = [_req(src, "J1", rep=i) for i in range(2)]
    sdk = FakeSDK({"input_tokens": 800, "output_tokens": 100})
    _open_batch(led, sdk.messages.batches, reqs, hashes=["stale-hash"] * 2)
    client.run_batch(reqs[:0], "cp1", led, sdk=sdk, poll_s=0, lookup={r.custom_id: r for r in reqs},
                     log=lambda *a: None)
    assert not any(client.cached(r) for r in reqs)
    assert len(list((config.RUNS / "orphans").glob("*.json"))) == 2
    assert led.summary()["outstanding_reservations"] == 0        # still billed and counted


def test_batch_create_5xx_is_charged_and_blocks_further_submissions(env):
    src, led = env
    reqs = [_req(src, "J1", rep=i) for i in range(2)]
    sdk = FakeSDK({"input_tokens": 800, "output_tokens": 100})

    def gateway(requests):
        raise _status_error(502)
    sdk.messages.batches.create = gateway
    client.run_batch(reqs, "cp1", led, sdk=sdk, poll_s=0, log=lambda *a: None)
    s = led.summary()     # still counted against the cap, as outstanding reservations (not settled)
    assert s["cp1"] == pytest.approx(sum(r.worst_case_usd() * 0.5 for r in reqs), abs=1e-6)
    assert s["outstanding_reservations"] == 2
    _, unknown = client._open_batches()
    assert len(unknown) == 1
    sdk2 = FakeSDK({"input_tokens": 800, "output_tokens": 100})
    logs = []
    client.run_batch(reqs, "cp1", led, sdk=sdk2, poll_s=0, log=logs.append)
    assert sdk2.calls == 0 and any("REFUSING" in l for l in logs)
    # resolved by hand as "never created": reservations released, submissions unblocked
    msg = client.resolve_create_unknown(unknown[0]["attempt"], None, led)
    assert "released" in msg and led.summary()["cp1"] == 0 and client._open_batches() == ({}, [])
    client.run_batch(reqs, "cp1", led, sdk=sdk2, poll_s=0, log=lambda *a: None)
    assert sdk2.calls == 2 and all(client.cached(r) for r in reqs)


def test_resolve_existing_batch_reenters_it_for_resume(env):
    src, led = env
    reqs = [_req(src, "J1", rep=i) for i in range(2)]
    sdk = FakeSDK({"input_tokens": 800, "output_tokens": 100})
    real_create = sdk.messages.batches.create

    def created_but_timed_out(requests):
        real_create(requests)                    # the batch exists server-side ("b0")
        raise RuntimeError("read timeout")
    sdk.messages.batches.create = created_but_timed_out
    client.run_batch(reqs, "cp1", led, sdk=sdk, poll_s=0, log=lambda *a: None)
    _, unknown = client._open_batches()
    client.resolve_create_unknown(unknown[0]["attempt"], "b0", led)
    sdk.calls = 0
    client.run_batch(reqs, "cp1", led, sdk=sdk, poll_s=0, log=lambda *a: None)
    assert sdk.calls == 0 and all(client.cached(r) for r in reqs)
    assert led.summary()["outstanding_reservations"] == 0


def test_resume_leaves_other_stages_batches_open(env):
    src, led = env
    pilot = [_req(src, "J1", rep=i, model="sonnet") for i in range(2)]
    main = [_req(src, "J3", rep=i) for i in range(2)]
    sdk = FakeSDK({"input_tokens": 800, "output_tokens": 100})
    _open_batch(led, sdk.messages.batches, pilot, bid="b_pilot")
    client.run_batch(main, "cp1", led, sdk=sdk, poll_s=0, lookup={r.custom_id: r for r in main},
                     log=lambda *a: None)
    assert "b_pilot" in client._open_batches()[0]
    client.run_batch(pilot, "cp1", led, sdk=sdk, poll_s=0, log=lambda *a: None)
    assert all(client.cached(r) for r in pilot) and client._open_batches() == ({}, [])
