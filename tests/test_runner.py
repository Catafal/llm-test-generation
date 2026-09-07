"""Runner tests hit real subprocesses on purpose: the isolation model is the thing under test."""

import pytest

from testgen.harness.runner import run_many, run_suite, sandbox_available

IMPL = "def add(a, b):\n    return a + b\n"


def test_passing_suite_reports_each_test():
    suite = (
        "from solution import add\n\n"
        "def test_one():\n    assert add(1, 2) == 3\n\n"
        "def test_two():\n    assert add(0, 0) == 0\n"
    )
    r = run_suite(suite, IMPL)
    assert r.status == "ok"
    assert sorted(t.name for t in r.tests) == ["test_one", "test_two"]
    assert not r.any_failed


def test_failing_test_is_reported_by_name():
    suite = (
        "from solution import add\n\n"
        "def test_ok():\n    assert add(1, 1) == 2\n\n"
        "def test_bad():\n    assert add(1, 1) == 3\n"
    )
    r = run_suite(suite, IMPL)
    assert r.status == "ok"
    failed = [t.name for t in r.tests if t.outcome == "failed"]
    assert failed == ["test_bad"]


def test_hanging_suite_times_out():
    suite = "import time\n\ndef test_hang():\n    time.sleep(30)\n"
    r = run_suite(suite, IMPL, timeout_s=2)
    assert r.status == "timeout"
    assert r.tests == []


def test_syntax_error_in_suite_is_a_crash():
    r = run_suite("def test_x(:\n    pass\n", IMPL)
    assert r.status == "crash"


def test_wrong_import_is_a_crash():
    # Decision 3: suites must import from `solution`; anything else fails collection.
    r = run_suite("from answer import add\n\ndef test_x():\n    assert add(1, 1) == 2\n", IMPL)
    assert r.status == "crash"


def test_runtime_error_in_implementation_is_reported_as_error():
    impl = "def add(a, b):\n    raise RuntimeError('boom')\n"
    r = run_suite("from solution import add\n\ndef test_x():\n    assert add(1, 1) == 2\n", impl)
    assert r.status == "ok"
    assert r.tests[0].outcome in ("failed", "error")


@pytest.mark.skipif(not sandbox_available(), reason="sandbox-exec not available")
def test_sandbox_denies_network():
    suite = (
        "import urllib.request\n\n"
        "def test_net():\n"
        "    urllib.request.urlopen('http://example.com', timeout=3)\n"
    )
    r = run_suite(suite, IMPL, sandbox=True)
    assert r.sandboxed
    assert r.status == "ok" and r.any_failed, r.stderr_tail


@pytest.mark.skipif(not sandbox_available(), reason="sandbox-exec not available")
def test_sandbox_denies_writes_outside_workdir(tmp_path):
    target = tmp_path / "escaped.txt"
    suite = f"def test_write():\n    open({str(target)!r}, 'w').write('x')\n"
    r = run_suite(suite, IMPL, sandbox=True)
    assert r.status == "ok" and r.any_failed
    assert not target.exists()


def test_run_many_matches_sequential_and_keeps_keys():
    suite = "from solution import add\n\ndef test_x():\n    assert add(1, 1) == 2\n"
    impls = {
        "ok": IMPL,
        "bad": "def add(a, b):\n    return a - b\n",
        "hang": "import time\ntime.sleep(30)\n",
    }
    results = run_many(suite, impls, timeout_s=2, workers=3)
    assert list(results) == ["ok", "bad", "hang"]
    assert not results["ok"].any_failed
    assert results["bad"].any_failed
    assert results["hang"].status == "timeout"
