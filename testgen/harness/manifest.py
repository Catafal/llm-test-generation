"""Run manifests (principle 5, FT3): every run records what produced its numbers.

runs/<run_id>/manifest.json is committed; runs/<run_id>/outputs.jsonl is not.
"""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import config
from testgen.mutate.artifact import OPERATORS_VERSION
from testgen.mutate.profiles import EVAL_PROFILE_NAME


def git_sha() -> str:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=config.ROOT, capture_output=True, text=True
    )
    return r.stdout.strip() or "unknown"


def new_run(name: str, **extra) -> Path:
    """Create runs/<name>-<timestamp>/ with a manifest; return the directory."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.RUNS_DIR / f"{name}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run": run_dir.name,
        "created": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "operators_version": OPERATORS_VERSION,
        "profile": EVAL_PROFILE_NAME,
        "budget": {
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "max_tests_per_suite": config.MAX_TESTS_PER_SUITE,
            "test_timeout_seconds": config.TEST_TIMEOUT_SECONDS,
        },
        **extra,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return run_dir


def update(run_dir: Path, **fields) -> None:
    path = run_dir / "manifest.json"
    data = json.loads(path.read_text())
    data.update(fields)
    path.write_text(json.dumps(data, indent=1))
