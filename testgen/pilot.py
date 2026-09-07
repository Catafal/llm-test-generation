"""T4 pilot: prove the metric ranks a strong suite above a weak one on every case.

Usage:
    uv run python -m testgen.pilot            # enforce the rule, exit 1 on failure
    uv run python -m testgen.pilot --survivors  # list surviving mutants per case for labelling

Each case under data/pilot/cases/ is a Python module (see data/pilot/README.md).
Mutants come from the eval profile; trivially-equivalent ones are dropped by
bytecode comparison and hand-labelled ones by the case's EQUIVALENT dict.
"""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

from config import ROOT
from testgen.harness.runner import RunResult, run_many, run_suite
from testgen.harness.score import SuiteScore, score_suite
from testgen.mutate.equivalence import split_equivalent
from testgen.mutate.operators import Mutant, generate_mutants

CASES_DIR = ROOT / "data" / "pilot" / "cases"


@dataclass
class Case:
    name: str
    source: str
    weak: str
    strong: str
    equivalent: dict[str, str]


def load_cases(directory: Path = CASES_DIR) -> list[Case]:
    cases = []
    for path in sorted(directory.glob("*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        cases.append(Case(path.stem, mod.SOURCE, mod.WEAK, mod.STRONG, dict(mod.EQUIVALENT)))
    return cases


@dataclass
class CaseResult:
    case: Case
    live: list[Mutant]
    excluded: dict[str, str]  # mutant id -> reason (trivial or hand-labelled)
    weak: SuiteScore
    strong: SuiteScore
    weak_runs: dict[str, RunResult]
    strong_runs: dict[str, RunResult]


def _evaluate(case: Case) -> CaseResult:
    mutants = generate_mutants(case.source)
    live, trivial = split_equivalent(case.source, mutants)
    excluded = {mid: "trivially equivalent (bytecode)" for mid in trivial}
    unknown = set(case.equivalent) - {m.id for m in mutants}
    if unknown:
        raise SystemExit(
            f"{case.name}: EQUIVALENT labels for unknown mutant ids: {sorted(unknown)}"
        )
    excluded.update(case.equivalent)
    live = [m for m in live if m.id not in excluded]

    def score(suite: str) -> tuple[SuiteScore, dict[str, RunResult]]:
        runs = run_many(suite, {m.id: m.source for m in live})
        return score_suite(run_suite(suite, case.source), runs, set(excluded)), runs

    weak, weak_runs = score(case.weak)
    strong, strong_runs = score(case.strong)
    return CaseResult(case, live, excluded, weak, strong, weak_runs, strong_runs)


def _fmt(s: SuiteScore) -> str:
    return (
        f"{s.mutation_score:.2f} ({s.mutants_killed}/{s.mutants_total - s.mutants_equivalent})"
        if s.valid
        else f"INVALID:{s.invalid_reason}"
    )


def _survivors(r: CaseResult, which: str) -> list[Mutant]:
    kills = (r.strong if which == "strong" else r.weak).kills
    return [m for m in r.live if m.id not in kills]


def main(argv: list[str]) -> int:
    show_survivors = "--survivors" in argv
    failures = 0
    print(f"{'case':<22}{'mutants':>8}{'excl':>6}{'weak':>14}{'strong':>14}  verdict")
    for case in load_cases():
        r = _evaluate(case)
        ok = (
            r.weak.valid
            and r.strong.valid
            and (r.strong.mutation_score or 0) > (r.weak.mutation_score or 0)
        )
        failures += not ok
        print(
            f"{case.name:<22}{len(r.live):>8}{len(r.excluded):>6}"
            f"{_fmt(r.weak):>14}{_fmt(r.strong):>14}  {'ok' if ok else 'FAIL'}"
        )
        if show_survivors:
            for m in _survivors(r, "strong"):
                print(f"    survives STRONG  {m.id:<28} {m.description}")
            for mid, why in r.excluded.items():
                print(f"    excluded         {mid:<28} {why}")
    print(f"\n{failures} failing case(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
