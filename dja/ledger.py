"""Spend ledger: worst-case reservation before every call, settlement from real usage after.

The ledger is an append-only JSONL file guarded by an exclusive flock. Every write is flushed and
fsynced before the lock is released, so a concurrent reader never sees a half-written state.
State is rebuilt by replaying the file under the lock.

Reservation ids are single-use: reserving an id that was ever reserved before raises, so a billed call
can never be forgotten by a later reservation overwriting it. Callers use one fresh id per attempt.

reserve(): one reservation; if it does not fit it waits while THIS PROCESS's own reservations (this Ledger
  object's) are outstanding, since they will settle (often lower); reservations left by other or killed
  processes still count against the cap but are not waited on. FIFO tickets. The CLI uses one Ledger object
  per run, so "this process" and "this object" coincide.
reserve_many(): all-or-nothing for a batch chunk under one lock; returns False if the chunk does not fit.
"""
from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import config


class CapExceeded(RuntimeError):
    pass


class Overrun(RuntimeError):
    """A settled cost exceeded its reservation: the worst-case bound was wrong. Stop the run."""


class DuplicateReservation(RuntimeError):
    pass


@dataclass
class State:
    settled: dict          # bucket -> settled USD
    outstanding: dict      # reservation id -> (bucket, amount)
    seen: set              # every id ever reserved

    def committed(self, cap: str) -> float:
        buckets = [b for b, parents in config.CAP_PARENTS.items() if cap in parents]
        s = sum(self.settled.get(b, 0.0) for b in buckets)
        s += sum(a for (b, a) in self.outstanding.values() if b in buckets)
        return s


class Ledger:
    def __init__(self, path: Path | None = None, caps: dict | None = None,
                 stop_fraction: float = config.STOP_FRACTION):
        self.path = Path(path) if path is not None else config.LEDGER
        self.lock_path = self.path.with_suffix(".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.caps = dict(caps or config.CAPS)
        self.stop_fraction = stop_fraction
        self._tickets = threading.Condition()
        self._next_ticket = 0
        self._serving = 0
        self._mine: set = set()          # ids this process reserved and has not settled/released

    # ---- low level -------------------------------------------------------------------------
    def _locked(self):
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd

    @staticmethod
    def _unlock(fd):
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def _replay(self) -> State:
        settled, outstanding, seen = {}, {}, set()
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    e = json.loads(line)
                    if e["kind"] == "reserve":
                        outstanding[e["id"]] = (e["bucket"], e["amount"])
                        seen.add(e["id"])
                    elif e["kind"] == "settle":
                        outstanding.pop(e["id"], None)
                        settled[e["bucket"]] = settled.get(e["bucket"], 0.0) + e["amount"]
                    elif e["kind"] == "release":
                        outstanding.pop(e["id"], None)
        return State(settled, outstanding, seen)

    def _append(self, entries: list[dict]):
        now = time.time()
        with open(self.path, "a") as f:
            f.write("".join(json.dumps({"t": now, **e}, sort_keys=True) + "\n" for e in entries))
            f.flush()
            os.fsync(f.fileno())

    def _fits(self, st: State, bucket: str, amount: float) -> tuple[bool, str]:
        for cap in config.CAP_PARENTS[bucket]:
            limit = self.caps[cap]
            # the checkpoint's own cap stops at STOP_FRACTION; parent caps at 100%
            if cap == bucket:
                limit *= self.stop_fraction
            if st.committed(cap) + amount > limit + 1e-12:
                return False, cap
        return True, ""

    # ---- public ----------------------------------------------------------------------------
    def state(self) -> State:
        fd = self._locked()
        try:
            return self._replay()
        finally:
            self._unlock(fd)

    def reserve(self, rid: str, bucket: str, amount: float, meta: dict | None = None,
                timeout: float = 600.0) -> None:
        if bucket not in config.CAP_PARENTS:
            raise ValueError(f"unknown bucket {bucket}")
        with self._tickets:
            ticket = self._next_ticket
            self._next_ticket += 1
            while self._serving != ticket:
                self._tickets.wait()
        try:
            deadline = time.time() + timeout
            while True:
                fd = self._locked()
                try:
                    st = self._replay()
                    if rid in st.seen:
                        raise DuplicateReservation(rid)
                    ok, cap = self._fits(st, bucket, amount)
                    if ok:
                        self._append([{"kind": "reserve", "id": rid, "bucket": bucket,
                                       "amount": amount, "meta": meta or {}}])
                        self._mine.add(rid)
                        return
                    # wait only for this process's own reservations: one stranded by a killed process
                    # never settles and must not make live workers hang
                    if not (self._mine & set(st.outstanding)):
                        raise CapExceeded(f"{bucket}: reserving ${amount:.4f} would cross cap '{cap}'")
                finally:
                    self._unlock(fd)
                if time.time() > deadline:
                    raise CapExceeded(f"{bucket}: timed out waiting for room under cap")
                time.sleep(0.05)
        finally:
            with self._tickets:
                self._serving += 1
                self._tickets.notify_all()

    def reserve_many(self, entries: list[tuple], bucket: str) -> bool:
        """entries: [(rid, amount, meta)]. All-or-nothing under one lock; False if the total doesn't fit."""
        fd = self._locked()
        try:
            st = self._replay()
            dup = [rid for rid, _, _ in entries if rid in st.seen]
            if dup or len({rid for rid, _, _ in entries}) != len(entries):
                raise DuplicateReservation(str(dup[:3]))
            ok, _ = self._fits(st, bucket, sum(a for _, a, _ in entries))
            if not ok:
                return False
            self._append([{"kind": "reserve", "id": rid, "bucket": bucket, "amount": a, "meta": m or {}}
                          for rid, a, m in entries])
            self._mine.update(rid for rid, _, _ in entries)
            return True
        finally:
            self._unlock(fd)

    def settle(self, rid: str, bucket: str, amount: float, reserved: float, usage: dict | None = None) -> bool:
        """Settle an OPEN reservation. Settling an id that is not open (already settled/released, e.g. a
        second resume of the same batch) is ignored so nothing is counted twice. Returns True if applied."""
        fd = self._locked()
        try:
            if rid not in self._replay().outstanding:
                return False
            self._append([{"kind": "settle", "id": rid, "bucket": bucket, "amount": amount,
                           "reserved": reserved, "usage": usage or {}}])
            self._mine.discard(rid)
        finally:
            self._unlock(fd)
        if amount > reserved + 1e-12:
            raise Overrun(f"{rid}: cost ${amount:.6f} exceeded reservation ${reserved:.6f}")
        return True

    def release(self, rid: str, bucket: str, reason: str) -> bool:
        fd = self._locked()
        try:
            if rid not in self._replay().outstanding:
                return False
            self._append([{"kind": "release", "id": rid, "bucket": bucket, "reason": reason}])
            self._mine.discard(rid)
            return True
        finally:
            self._unlock(fd)

    def release_many(self, rids: list[str], bucket: str, reason: str):
        fd = self._locked()
        try:
            open_ids = self._replay().outstanding
            self._append([{"kind": "release", "id": r, "bucket": bucket, "reason": reason}
                          for r in rids if r in open_ids])
            self._mine.difference_update(rids)
        finally:
            self._unlock(fd)

    def summary(self) -> dict:
        st = self.state()
        return {cap: round(st.committed(cap), 6) for cap in self.caps} | {
            "outstanding_reservations": len(st.outstanding)}
