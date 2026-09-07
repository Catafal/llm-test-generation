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

# Local model cache for mlx-lm downloads. Gitignored.
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", ROOT / ".cache" / "models"))

# Sandbox limits for running generated tests (harness/runner.py).
TEST_TIMEOUT_SECONDS = int(os.getenv("TEST_TIMEOUT_SECONDS", "10"))

# Fixed generation budget shared by every condition (FT16). Changing these
# invalidates comparability across runs; the manifest records them.
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1024"))
MAX_TESTS_PER_SUITE = int(os.getenv("MAX_TESTS_PER_SUITE", "8"))
