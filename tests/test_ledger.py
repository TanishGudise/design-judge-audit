import json
import random
import threading

import pytest

from dja.ledger import CapExceeded, Ledger, Overrun


def test_concurrent_reserve_settle_never_crosses_cap(tmp_path):
    caps = {"pilot": 1.0, "cp1": 1.5, "cp2": 9, "cp3": 9, "reserve": 9, "total": 2.0}
    led = Ledger(tmp_path / "l.jsonl", caps=caps, stop_fraction=1.0)
    stopped = []
    lock = threading.Lock()

    def worker(k):
        rnd = random.Random(k)
        for i in range(40):
            rid = f"{k}-{i}"
            amt = rnd.uniform(0.005, 0.02)
            try:
                led.reserve(rid, "pilot", amt, timeout=30)
            except CapExceeded:
                with lock:
                    stopped.append(rid)
                return
            led.settle(rid, "pilot", amt * rnd.uniform(0.2, 1.0), amt)

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    s = led.summary()
    assert s["pilot"] <= 1.0 + 1e-9
    assert s["outstanding_reservations"] == 0
    assert stopped, "the cap should have been reached"
    # every line is complete JSON (flush + fsync before unlock)
    for line in (tmp_path / "l.jsonl").read_text().splitlines():
        json.loads(line)


def test_nested_caps_and_stop_fraction(tmp_path):
    caps = {"pilot": 2.0, "cp1": 4.5, "cp2": 5.5, "cp3": 3, "reserve": 2, "total": 15}
    led = Ledger(tmp_path / "l.jsonl", caps=caps, stop_fraction=0.9)
    led.reserve("a", "pilot", 1.7)
    led.settle("a", "pilot", 1.7, 1.7)
    with pytest.raises(CapExceeded):           # 1.7 + 0.2 > 0.9 * 2.0
        led.reserve("b", "pilot", 0.2, timeout=0.2)
    led.reserve("c", "cp1", 2.3)               # pilot counts toward cp1: 1.7 + 2.3 = 4.0 <= 0.9 * 4.5
    led.settle("c", "cp1", 2.3, 2.3)
    with pytest.raises(CapExceeded):
        led.reserve("d", "cp1", 0.1, timeout=0.2)
    assert led.summary()["cp1"] == pytest.approx(4.0)
    assert led.summary()["total"] == pytest.approx(4.0)


def test_overrun_is_recorded_then_raised(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    led.reserve("x", "pilot", 0.01)
    with pytest.raises(Overrun):
        led.settle("x", "pilot", 0.02, 0.01)
    assert led.summary()["pilot"] == pytest.approx(0.02)   # real cost is still counted


def test_release_frees_reservation(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", caps={"pilot": 0.1, "cp1": 1, "cp2": 1, "cp3": 1, "reserve": 1, "total": 1},
                 stop_fraction=1.0)
    led.reserve("x", "pilot", 0.09)
    led.release("x", "pilot", "api_status_400")
    led.reserve("y", "pilot", 0.09)
    assert led.summary()["outstanding_reservations"] == 1


def test_duplicate_reservation_ids_are_refused(tmp_path):
    from dja.ledger import DuplicateReservation
    led = Ledger(tmp_path / "l.jsonl")
    led.reserve("x", "pilot", 0.5)
    with pytest.raises(DuplicateReservation):
        led.reserve("x", "pilot", 0.5)
    led.settle("x", "pilot", 0.1, 0.5)
    with pytest.raises(DuplicateReservation):     # ids are single-use even after settling
        led.reserve("x", "pilot", 0.5)
    assert led.summary()["pilot"] == pytest.approx(0.1)


def test_reserve_many_is_all_or_nothing(tmp_path):
    caps = {"pilot": 9, "cp1": 1.0, "cp2": 9, "cp3": 9, "reserve": 9, "total": 9}
    led = Ledger(tmp_path / "l.jsonl", caps=caps, stop_fraction=1.0)
    assert led.reserve_many([("a", 0.4, {}), ("b", 0.4, {})], "cp1")
    assert not led.reserve_many([("c", 0.15, {}), ("d", 0.15, {})], "cp1")   # 0.8 + 0.3 > 1.0
    assert led.summary()["cp1"] == pytest.approx(0.8)
    assert led.summary()["outstanding_reservations"] == 2


def test_second_settle_or_release_is_ignored(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    led.reserve("x", "pilot", 0.5)
    assert led.settle("x", "pilot", 0.2, 0.5)
    assert not led.settle("x", "pilot", 0.2, 0.5)
    assert not led.release("x", "pilot", "dup")
    assert led.summary()["pilot"] == pytest.approx(0.2)


def test_reserve_does_not_wait_on_foreign_stranded_reservations(tmp_path):
    caps = {"pilot": 1.0, "cp1": 9, "cp2": 9, "cp3": 9, "reserve": 9, "total": 9}
    other = Ledger(tmp_path / "l.jsonl", caps=caps, stop_fraction=1.0)
    other.reserve("stranded", "pilot", 0.9)            # a killed process's reservation
    me = Ledger(tmp_path / "l.jsonl", caps=caps, stop_fraction=1.0)
    import time as _t
    t0 = _t.time()
    with pytest.raises(CapExceeded):
        me.reserve("y", "pilot", 0.2, timeout=30)
    assert _t.time() - t0 < 5
