"""Run one generated test suite against one implementation, in isolation.

Contract (D014, T2 decisions 1–4):
- One subprocess per (suite, implementation) pair. Slow but nothing leaks
  between runs: a hanging or monkeypatching suite cannot poison the next one.
- The implementation is always written as ``solution.py``; suites must
  ``from solution import ...``. Reference and mutants use the same file name,
  so a suite cannot tell which one it is running against.
- On macOS the subprocess is wrapped in ``sandbox-exec`` (no network, writes
  only inside the temp dir). Generated code is untrusted input. The caller
  can disable the sandbox; the run records whether it was used.
- Per-test outcomes are parsed from pytest's built-in JUnit XML, so later
  analysis can say which test killed which mutant.
"""

import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from config import HARNESS_WORKERS, TEST_TIMEOUT_SECONDS

SOLUTION_FILE = "solution.py"
SUITE_FILE = "test_solution.py"
REPORT_FILE = "report.xml"

# pytest exit codes we care about (https://docs.pytest.org/en/stable/reference/exit-codes.html)
# 5 = no tests collected: a normal run of an empty suite, not a crash; the
# scorer turns it into invalid_reason="no_tests".
_EXIT_OK, _EXIT_TESTS_FAILED, _EXIT_NO_TESTS = 0, 1, 5


@dataclass
class TestResult:
    __test__ = False  # stop pytest trying to collect this dataclass

    name: str
    outcome: str  # "passed" | "failed" | "error"
    message: str = ""


@dataclass
class RunResult:
    status: str  # "ok" | "timeout" | "crash"
    tests: list[TestResult] = field(default_factory=list)
    sandboxed: bool = False
    duration_s: float = 0.0
    stderr_tail: str = ""  # last lines only; enough to diagnose a crash

    @property
    def any_failed(self) -> bool:
        return any(t.outcome != "passed" for t in self.tests)


def sandbox_available() -> bool:
    """True when macOS ``sandbox-exec`` is on PATH."""
    return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None


def _sandbox_profile(workdir: Path) -> str:
    # SBPL: last matching rule wins. Allow everything, then deny network and
    # all writes, then re-allow writes to the work dir only. TMPDIR is pointed
    # at the work dir by the caller so Python's own temp files land there too.
    return (
        "(version 1)"
        "(allow default)"
        "(deny network*)"
        "(deny file-write*)"
        f'(allow file-write* (subpath "{workdir}"))'
        '(allow file-write* (literal "/dev/null"))'
    )


def _pytest_command(workdir: Path, sandbox: bool) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        f"--junitxml={REPORT_FILE}",
        SUITE_FILE,
    ]
    if sandbox:
        cmd = ["sandbox-exec", "-p", _sandbox_profile(workdir), *cmd]
    return cmd


def _parse_junit(report: Path) -> list[TestResult]:
    if not report.exists():
        return []
    results = []
    for case in ET.parse(report).iter("testcase"):
        name = case.get("name", "?")
        failure = case.find("failure")
        error = case.find("error")
        if failure is not None:
            results.append(TestResult(name, "failed", failure.get("message", "")[:500]))
        elif error is not None:
            results.append(TestResult(name, "error", error.get("message", "")[:500]))
        else:
            results.append(TestResult(name, "passed"))
    return results


def run_suite(
    suite_src: str,
    impl_src: str,
    timeout_s: int = TEST_TIMEOUT_SECONDS,
    sandbox: bool | None = None,
) -> RunResult:
    """Execute ``suite_src`` against ``impl_src`` and return per-test outcomes.

    ``sandbox=None`` means "use it if available". A timeout returns
    ``status="timeout"`` with no tests; a collection/syntax failure returns
    ``status="crash"``. The scorer decides what those mean (decision 5).
    """
    use_sandbox = sandbox_available() if sandbox is None else sandbox
    with tempfile.TemporaryDirectory(prefix="testgen-") as tmp:
        workdir = Path(tmp).resolve()
        (workdir / SOLUTION_FILE).write_text(impl_src)
        (workdir / SUITE_FILE).write_text(suite_src)
        env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(workdir),
            "TMPDIR": str(workdir),  # keeps every temp write inside the sandboxed dir
        }
        start = time.monotonic()
        try:
            proc = subprocess.run(
                _pytest_command(workdir, use_sandbox),
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return RunResult("timeout", sandboxed=use_sandbox, duration_s=time.monotonic() - start)
        duration = time.monotonic() - start
        tests = _parse_junit(workdir / REPORT_FILE)
        # Exit code 0/1 with test cases = normal run. Anything else, or no
        # test cases at all, means pytest never got to run the suite.
        if proc.returncode == _EXIT_NO_TESTS:
            return RunResult("ok", [], use_sandbox, duration, "")
        crashed = proc.returncode not in (_EXIT_OK, _EXIT_TESTS_FAILED) or not tests
        status = "crash" if crashed else "ok"
        tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-15:])
        return RunResult(status, tests, use_sandbox, duration, tail)


def run_many(
    suite_src: str,
    impls: dict[str, str],
    timeout_s: int = TEST_TIMEOUT_SECONDS,
    sandbox: bool | None = None,
    workers: int = HARNESS_WORKERS,
) -> dict[str, RunResult]:
    """Run one suite against many implementations in parallel.

    Isolation is unchanged: every run is still its own sandboxed subprocess;
    the pool only overlaps their wall-clock. Result order follows ``impls``.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            key: pool.submit(run_suite, suite_src, src, timeout_s, sandbox)
            for key, src in impls.items()
        }
        return {key: f.result() for key, f in futures.items()}
