"""Harvest post-cutoff pure functions from GitHub (D017).

    uv run python -m testgen.data.harvest --repos 50 --floor 2026-06-01

For each permissively-licensed Python repository created after the floor:
clone with history but without blobs, walk non-test ``.py`` files, keep files
whose *introducing commit* is dated after the floor (repo creation is not
proof of new code), extract pure candidates, and count live mutants under
the eval profile. Candidates with >= MIN_LIVE_MUTANTS are appended to
``data/heldout/candidates.jsonl`` with full provenance.

Uses ``gh`` for the search API (already authenticated) and ``git`` for
clones. Clones live under ``.cache/harvest/`` and are gitignored.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import warnings
from datetime import date
from pathlib import Path

from config import ROOT
from testgen.data.purity import extract_candidates
from testgen.harness.runner import run_suite
from testgen.mutate.equivalence import split_equivalent
from testgen.mutate.operators import generate_mutants

CACHE = ROOT / ".cache" / "harvest"
OUT = ROOT / "data" / "heldout" / "candidates.jsonl"
LICENCES = ("mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause")
MIN_LIVE_MUTANTS = 8
MAX_LINES = 60  # longer functions dominate eval cost and distort the fixed generation budget
PER_REPO_CAP = 10  # spread the pool across sources so the family split means something
MIN_ASCII_RATIO = 0.95  # docstring must be readable English for the model's spec
SKIP_PATH_PARTS = ("test", "tests", "setup.py", "conftest", "example", "examples", "docs", "bench")


def _gh(query: str, page: int) -> list[dict]:
    cmd = [
        "gh",
        "api",
        "-X",
        "GET",
        "search/repositories",
        "-f",
        f"q={query}",
        "-f",
        "sort=stars",
        "-f",
        "order=desc",
        "-f",
        "per_page=100",
        "-f",
        f"page={page}",
        "--jq",
        ".items[] | {full_name, html_url, license: .license.spdx_id, created_at, stargazers_count}",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def search_repos(floor: date, limit: int) -> list[dict]:
    """Permissively licensed, non-fork, non-archived Python repos created on/after the floor."""
    repos: list[dict] = []
    for lic in LICENCES:
        q = f"language:Python created:>={floor.isoformat()} license:{lic} fork:false archived:false"
        page = 1
        while len(repos) < limit and page <= 10:
            batch = _gh(q, page)
            if not batch:
                break
            repos.extend(batch)
            page += 1
        if len(repos) >= limit:
            break
    return repos[:limit]


def clone(repo: dict) -> Path | None:
    dest = CACHE / repo["full_name"].replace("/", "__")
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    # blob:none keeps full history (for introducing-commit dates) but fetches file
    # contents lazily, which is what we want: most files are never read.
    r = subprocess.run(
        ["git", "clone", "--quiet", "--filter=blob:none", repo["html_url"], str(dest)],
        capture_output=True,
        text=True,
    )
    return dest if r.returncode == 0 else None


def introducing_commits(repo_dir: Path) -> dict[str, tuple[str, str]]:
    """path -> (sha, author date ISO) of the commit that first added it. One git call per repo.

    Renames are not followed: a renamed file looks new, which errs on the side
    of *accepting* possibly-old code. The n-gram novelty check and the
    per-repo cap limit the damage; stated in the datasheet.
    """
    r = subprocess.run(
        ["git", "log", "--diff-filter=A", "--reverse", "--format=%x00%H %aI", "--name-only"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    first: dict[str, tuple[str, str]] = {}
    sha = when = ""
    for line in r.stdout.splitlines():
        if line.startswith("\x00"):
            sha, when = line[1:].split(" ", 1)
        elif line.strip() and line not in first:
            first[line] = (sha, when)
    return first


def _english(docstring: str) -> bool:
    return sum(ch.isascii() for ch in docstring) / max(1, len(docstring)) >= MIN_ASCII_RATIO


def _imports_cleanly(source: str) -> bool:
    """Run `import solution` in the sandbox: catches NameErrors from missed free names."""
    r = run_suite("import solution\n\ndef test_import():\n    assert True\n", source, timeout_s=10)
    return r.status == "ok" and not r.any_failed


def live_mutant_count(source: str) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # harvested code may hold bad escapes
        live, _ = split_equivalent(source, generate_mutants(source))
    return len(live)


def harvest_repo(repo: dict, floor: date, seen: set[str]) -> list[dict]:
    repo_dir = clone(repo)
    if repo_dir is None:
        return []
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True
    ).stdout.strip()
    found: list[dict] = []
    intro_by_path = introducing_commits(repo_dir)
    for path in sorted(repo_dir.rglob("*.py")):
        if len(found) >= PER_REPO_CAP:
            break
        rel = path.relative_to(repo_dir).as_posix()
        if (
            any(part in SKIP_PATH_PARTS for part in Path(rel.lower()).parts)
            or path.stat().st_size > 200_000
        ):
            continue
        intro = intro_by_path.get(rel)
        if intro is None or date.fromisoformat(intro[1][:10]) < floor:
            continue
        try:
            module_src = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        accepted, _ = extract_candidates(module_src)
        for cand in accepted:
            if len(found) >= PER_REPO_CAP:
                break
            if cand.end_lineno - cand.lineno + 1 > MAX_LINES or not _english(cand.docstring):
                continue
            fingerprint = hashlib.sha256(cand.source.encode()).hexdigest()
            if fingerprint in seen:
                continue
            n_live = live_mutant_count(cand.source)
            if n_live < MIN_LIVE_MUTANTS or not _imports_cleanly(cand.source):
                continue
            seen.add(fingerprint)
            found.append(
                {
                    "id": f"{repo['full_name']}:{rel}::{cand.name}",
                    "function": cand.name,
                    "repo": repo["html_url"],
                    "licence": repo["license"],
                    "path": rel,
                    "head_sha": head,
                    "introduced_sha": intro[0],
                    "introduced_at": intro[1],
                    "lines": [cand.lineno, cand.end_lineno],
                    "live_mutants": n_live,
                    "source_sha256": fingerprint,
                    "docstring": cand.docstring,
                    "source": cand.source,
                }
            )
    return found


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", type=int, default=50)
    ap.add_argument("--floor", type=date.fromisoformat, default=date(2026, 6, 1))
    args = ap.parse_args(argv)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if OUT.exists():
        seen = {
            json.loads(ln)["source_sha256"] for ln in OUT.read_text().splitlines() if ln.strip()
        }
    repos = search_repos(args.floor, args.repos)
    print(f"{len(repos)} repos to scan (floor {args.floor}, already have {len(seen)} candidates)")
    total = 0
    with OUT.open("a") as out:
        for i, repo in enumerate(repos, 1):
            found = harvest_repo(repo, args.floor, seen)
            for row in found:
                out.write(json.dumps(row) + "\n")
            total += len(found)
            lic = repo["license"] or "?"
            tag = f"[{i:>3}/{len(repos)}] {repo['full_name']:<45} {lic:<12}"
            print(f"{tag} +{len(found):<3} total {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
