"""Tests never reach the network or read the API key."""
import socket

import pytest

_real_connect = socket.socket.connect


def _guarded_connect(self, address):
    host = address[0] if isinstance(address, tuple) else address
    if isinstance(host, str) and (host.startswith("127.") or host in ("localhost", "::1")) or self.family == socket.AF_UNIX:
        return _real_connect(self, address)
    raise RuntimeError(f"network access blocked in tests: {address!r}")


@pytest.fixture(autouse=True)
def _no_network_no_key(monkeypatch, tmp_path):
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    from dja import config
    monkeypatch.setattr(config, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(config, "LEDGER", tmp_path / "runs" / "ledger.jsonl")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from dja import client

    def _no_key():
        raise RuntimeError("tests must not read the API key")
    monkeypatch.setattr(client, "_read_key", _no_key)
    yield
