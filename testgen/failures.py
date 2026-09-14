"""Bucket the tests that still fail on the reference after the oracle fill (D038).

    uv run python -m testgen.failures runs/<run>        # grounded run, no GPU

For every grounded-invalid record: run the filled suite against the reference,
take each failing test, and classify it by the pytest message and the assert
shapes inside that test function. The buckets answer "what is left for the
harness": a test whose assert shape the oracle does not fill (``in``, ``is``,
``approx``, non-literal ``==``) is harness headroom; a test where the function
itself raised is the model's input choice, which no fill can repair.
"""

import argparse
import ast
import collections
import json
import re
import sys
from pathlib import Path

from testgen.baselines import load_pool
from testgen.harness.runner import run_suite

# Leading exception name in a pytest failure message ("KeyError: 'x'").
_EXC = re.compile(r"^(\w*(Error|Exception|Warning|Exit|Interrupt))\b")
_FLAKY = "filled literal still failing (nondeterministic)"


def assert_shapes(fn: ast.FunctionDef) -> set[str]:
    """Which assert forms a test function uses; the oracle fills only ``eq_literal``."""
    out: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.With):
            if any("raises" in ast.unparse(i.context_expr) for i in n.items):
                out.add("raises")
        if not isinstance(n, ast.Assert):
            continue
        t = n.test
        if isinstance(t, ast.Compare) and len(t.ops) == 1:
            op, rhs = t.ops[0], t.comparators[0]
            if isinstance(op, ast.Eq):
                try:
                    ast.literal_eval(rhs)
                    out.add("eq_literal")
                except (ValueError, TypeError, SyntaxError):
                    out.add("eq_approx" if "approx" in ast.unparse(rhs) else "eq_nonliteral")
            elif isinstance(op, (ast.Is, ast.IsNot)):
                out.add("is")
            elif isinstance(op, (ast.In, ast.NotIn)):
                out.add("in")
            else:
                out.add("order")
        else:
            out.add("truthy")
    return out


def classify(message: str, shapes: set[str]) -> str:
    """One bucket per failing test: why it failed on the correct function."""
    m = _EXC.match(message or "")
    if "DID NOT RAISE" in (message or ""):
        return "expected exception not raised"
    if m and m.group(1) != "AssertionError":
        return f"function raised ({m.group(1)})"
    for shape, label in (
        ("eq_nonliteral", "assert == non-literal expression"),
        ("eq_approx", "assert == pytest.approx"),
        ("is", "assert is None/True/False"),
        ("in", "assert in / not in"),
        ("order", "assert < > !="),
        ("truthy", "assert truthiness / other"),
    ):
        if shape in shapes:
            return label
    return _FLAKY if shapes == {"eq_literal"} else "unclassified"


def bucket(run_dir: Path) -> list[dict]:
    """One row per failing test across the run's grounded-invalid suites."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    pool = {r["id"]: r for r in load_pool(manifest["pool"])}
    rows = []
    for line in (run_dir / "outputs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["grounded"]["score"]["valid"]:
            continue
        suite = r["grounded"]["suite"]
        res = run_suite(suite, pool[r["id"]]["source"])
        try:
            fns = {f.name: f for f in ast.walk(ast.parse(suite)) if isinstance(f, ast.FunctionDef)}
        except SyntaxError:
            fns = {}
        failing = [t for t in res.tests if t.outcome != "passed"]
        for t in failing:
            shapes = assert_shapes(fns[t.name]) if t.name in fns else set()
            rows.append(
                {
                    "id": r["id"],
                    "test": t.name,
                    "bucket": classify(t.message, shapes),
                    "shapes": sorted(shapes),
                    "message": t.message[:200],
                    "n_tests": len(res.tests),
                    "n_failing": len(failing),
                }
            )
    return rows


def report(rows: list[dict]) -> None:
    suites = {r["id"] for r in rows}
    by_bucket = collections.Counter(r["bucket"] for r in rows)
    raised = sum(n for b, n in by_bucket.items() if b.startswith("function raised"))
    unfilled = sum(n for b, n in by_bucket.items() if b.startswith("assert"))
    all_fail = {r["id"] for r in rows if r["n_failing"] == r["n_tests"]}
    print(f"{len(rows)} failing tests in {len(suites)} grounded-invalid suites")
    for b, n in by_bucket.most_common():
        print(f"  {n:4d}  {b}")
    flaky = by_bucket[_FLAKY]
    print(f"function raised: {raised}; unfilled assert shapes: {unfilled}; flaky: {flaky}")
    print(f"suites where every test fails: {len(all_fail)}; the rest would survive pruning")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="runs/<run> directory with grounded outputs.jsonl")
    ap.add_argument("--out", help="write the per-test rows as JSON here")
    args = ap.parse_args(argv)
    rows = bucket(Path(args.run))
    report(rows)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
