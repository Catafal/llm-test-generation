"""Fetch MBPP (the training-function pool) into the local cache with a recorded hash.

MBPP is CC-BY-4.0. We never commit it; the manifest records the URL and the
SHA-256 of the file actually used so a reviewer can verify the same bytes.
"""

import hashlib
import json
import urllib.request
from pathlib import Path

from config import ROOT

MBPP_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
)
MBPP_PATH = ROOT / ".cache" / "mbpp" / "mbpp.jsonl"


def fetch(path: Path = MBPP_PATH, url: str = MBPP_URL) -> tuple[Path, str]:
    """Download once; return (path, sha256)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 - fixed https URL
            path.write_bytes(r.read())
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path = MBPP_PATH) -> list[dict]:
    """Rows as {id, source, family}. Family = task_id: MBPP problems are independent."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append(
            {"id": f"mbpp:{r['task_id']}", "source": r["code"], "family": f"mbpp:{r['task_id']}"}
        )
    return rows
