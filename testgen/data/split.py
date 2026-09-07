"""Family-wise test/dev split of the frozen pool (D020).

    uv run python -m testgen.data.split --test 300 --seed 20260907

Whole families move together (a family = repo plus union-find over flagged
near-duplicate pairs, from decontaminate.py), so no near-duplicate can sit on
both sides. Also re-applies the per-repo cap defensively: the harvester's cap
is per run, and two runs appended to the same candidates file.
"""

import argparse
import json
import random
import sys
from collections import defaultdict

from config import ROOT

HELDOUT = ROOT / "data" / "heldout"
PER_REPO_CAP = 10


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260907)
    args = ap.parse_args(argv)

    rows = [
        json.loads(ln) for ln in (HELDOUT / "pool.jsonl").read_text().splitlines() if ln.strip()
    ]
    by_repo: dict[str, int] = defaultdict(int)
    capped = []
    for r in sorted(rows, key=lambda r: r["id"]):
        if by_repo[r["repo"]] < PER_REPO_CAP:
            by_repo[r["repo"]] += 1
            capped.append(r)
    families: dict[str, list[dict]] = defaultdict(list)
    for r in capped:
        families[r["family"]].append(r)
    order = sorted(families)
    random.Random(args.seed).shuffle(order)
    test_ids, n = set(), 0
    for fam in order:
        if n >= args.test:
            break
        test_ids.update(r["id"] for r in families[fam])
        n += len(families[fam])
    for r in capped:
        r["split"] = "test" if r["id"] in test_ids else "dev"
    (HELDOUT / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in capped))
    manifest = json.loads((HELDOUT / "manifest.json").read_text())
    manifest.update(
        {
            "kept": len(capped),
            "per_repo_cap": PER_REPO_CAP,
            "split_seed": args.seed,
            "test": sum(r["split"] == "test" for r in capped),
            "dev": sum(r["split"] == "dev" for r in capped),
            "families": len(families),
        }
    )
    (HELDOUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print({k: manifest[k] for k in ("kept", "test", "dev", "families", "split_seed")})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
