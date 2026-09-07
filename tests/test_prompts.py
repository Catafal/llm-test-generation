from testgen.generate.prompts import build_messages, enforce_test_budget, extract_suite

SUITE = "from solution import f\n\ndef test_a():\n    assert f(1) == 1\n"


def test_messages_shape_zero_and_few_shot():
    z = build_messages("def f(x):\n    return x\n", max_tests=8)
    assert [m["role"] for m in z] == ["system", "user"]
    assert "at most 8 test functions" in z[0]["content"]
    f = build_messages("def g(x):\n    return x\n", 8, shots=[("def f(x):\n    return x\n", SUITE)])
    assert [m["role"] for m in f] == ["system", "user", "assistant", "user"]
    assert f[2]["content"].startswith("```python\n")


def test_extract_prefers_fenced_block_and_rejects_garbage():
    suite, flags = extract_suite("Here you go:\n```python\n" + SUITE + "```\nHope it helps")
    assert suite == SUITE and not any(flags.values())
    assert extract_suite(SUITE)[0] == SUITE  # bare code
    assert extract_suite("I cannot do that.")[0] is None


def test_extract_salvages_truncated_fence_and_flags_it():
    text = "```python\n" + SUITE + "\ndef test_b():\n    assert f(2) ==\n"  # cut mid-line, no close
    suite, flags = extract_suite(text)
    assert flags["truncated"] and "test_a" in suite and "test_b" not in suite


def test_extract_injects_missing_pytest_import_and_flags_it():
    text = (
        "```python\nfrom solution import f\n\ndef test_e():\n"
        "    with pytest.raises(ValueError):\n        f(-1)\n```"
    )
    suite, flags = extract_suite(text)
    assert flags["pytest_import"] and suite.startswith("import pytest\n")
    assert not extract_suite("```python\nimport pytest\n" + SUITE + "```")[1]["pytest_import"]


def test_budget_truncates_tests_but_keeps_helpers():
    suite = (
        "import pytest\nfrom solution import f\n\nHELPER = 3\n\n"
        "def test_1():\n    assert f(1) == 1\n\n"
        "def helper():\n    return 2\n\n"
        "def test_2():\n    assert f(2) == 2\n\n"
        "def test_3():\n    assert f(3) == 3\n"
    )
    out, n = enforce_test_budget(suite, max_tests=2)
    assert n == 3
    assert "test_3" not in out and "test_2" in out and "def helper" in out and "HELPER = 3" in out
