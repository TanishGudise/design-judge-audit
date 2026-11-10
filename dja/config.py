"""Paths, models, prices and spend caps. Everything a run depends on lives here."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
DERIVED = DATA / "derived"          # rendered condition images (gitignored)
RUNS = ROOT / "runs"
CACHE = RUNS / "cache"              # one JSON per API call (request hash + response + cost)
LEDGER = RUNS / "ledger.jsonl"
RESULTS = ROOT / "results"          # committed: derived numbers only, no images

MODELS = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
}
# Thinking off: Haiku 5.5 accepts {"type": "disabled"} at effort <= high;
# Sonnet 5.5 turns thinking off with {"type": "between_tools"} at effort <= high.
THINKING = {
    "haiku": {"type": "disabled"},
    "sonnet": {"type": "between_tools"},
}
EFFORT = "low"
MAX_TOKENS = 600

# USD per million tokens, read from platform.claude.com/docs/en/about-claude/pricing on 2026-11-09.
# Haiku 5.5 rates are for prompts <= 100K tokens (every request here is far below that).
PRICES_READ_ON = "2026-11-09"
PRICES = {
    "claude-haiku-5-5": {"in": 0.10, "out": 0.50, "cache_write": 0.125, "cache_read": 0.01},
    "claude-sonnet-5-5": {"in": 2.00, "out": 10.00, "cache_write": 2.50, "cache_read": 0.10},
}
BATCH_DISCOUNT = 0.5

# Nested caps (2026-11-09): a call billed to a bucket must fit every cap that contains it.
CAPS = {
    "pilot": 2.00,
    "cp1": 4.50,      # includes pilot
    "cp2": 5.50,
    "cp3": 1.50,      # lowered from 3.00 before CP3 (2026-11-10)
    "reserve": 2.00,
    "total": 15.00,
}
CAP_PARENTS = {
    "pilot": ["pilot", "cp1", "total"],
    "cp1": ["cp1", "total"],
    "cp2": ["cp2", "total"],
    "cp3": ["cp3", "total"],
    "reserve": ["reserve", "total"],
}
STOP_FRACTION = 0.90   # stop a checkpoint once 90% of its cap is committed

# Worst-case input bound: 1 token per character of text (conservative), visual tokens per image
# ceil(w/28) * ceil(h/28) capped at 4,784 (images larger than the API limit are downscaled by the API and cost
# at most the cap; canonical renders keep every condition under 2,576 px), plus per-image and per-request margins.
IMAGE_TOKEN_CAP = 4784
PER_IMAGE_OVERHEAD = 64
PER_REQUEST_OVERHEAD = 200

SEED = 20261109
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD_PATH = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
