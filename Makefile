PY := .venv/bin/python
export HF_HUB_DISABLE_TELEMETRY := 1

test:
	$(PY) -m pytest -q tests

reproduce:
	HF_HUB_OFFLINE=1 $(PY) -m dja.cli analyze
	HF_HUB_OFFLINE=1 $(PY) -m dja.analyze_cp2
	HF_HUB_OFFLINE=1 $(PY) -m dja.analyze_cp3

plan:
	$(PY) -m dja.cli plan pilot && $(PY) -m dja.cli plan main

.PHONY: test reproduce plan
