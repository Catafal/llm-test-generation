"""Single place that reads .env. Nothing else in the repo touches os.environ.

Why: reproducibility (FT3). Every run manifest records the config it ran with,
so all tunables must be visible here, not scattered across modules.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Where raw run outputs and manifests go. Gitignored except manifests.
RUNS_DIR = Path(os.getenv("RUNS_DIR", ROOT / "runs"))

# Every model this project downloads lives inside the repo under models/hf/
# (gitignored) so the whole experiment can be removed with `make models-rm`.
# huggingface_hub reads HF_HUB_CACHE at import time, so it is set here, before
# any mlx_lm / sentence_transformers import. See testgen/models.py.
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", ROOT / "models" / "hf"))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE_DIR))

# Sandbox limits for running generated tests (harness/runner.py).
TEST_TIMEOUT_SECONDS = int(os.getenv("TEST_TIMEOUT_SECONDS", "10"))
# Parallel sandboxed runs. Each run is its own subprocess, so a thread pool
# is enough; half the cores leaves room for the model process during eval.
HARNESS_WORKERS = int(os.getenv("HARNESS_WORKERS", str(max(1, (os.cpu_count() or 2) // 2))))

# Fixed generation budget shared by every condition (FT16). Changing these
# invalidates comparability across runs; the manifest records them.
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2048"))  # D019: 1024 truncated the 9B
MAX_TESTS_PER_SUITE = int(os.getenv("MAX_TESTS_PER_SUITE", "8"))

# D023: an oracle-filled expected value is only written back when its repr is
# this short. Longer values are ones the model has no path to compute by hand.
MAX_ORACLE_REPR = int(os.getenv("MAX_ORACLE_REPR", "40"))

# D024: mlx-lm's Qwen3.5 DeltaNet training path OOMs above ~1024 tokens on 48 GB.
# SFT examples longer than this (prompt + target, chat template) are not kept.
MAX_TRAIN_TOKENS = int(os.getenv("MAX_TRAIN_TOKENS", "1024"))
