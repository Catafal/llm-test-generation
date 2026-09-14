"""Prompt construction and output normalisation under a fixed budget (D007, FT16).

Every condition (zero-shot, few-shot, fine-tuned) goes through the same
functions, so the only thing that differs between arms is the exemplars and
the weights. The test-count budget is enforced *after* generation by
truncating to the first ``max_tests`` test functions, identically for all
arms, so a model cannot win by writing more tests.
"""

import ast
import re

SYSTEM = (
    "You are an expert Python engineer writing pytest unit tests.\n"
    "Rules:\n"
    "- The function under test is in a module named `solution`; import it with "
    "`from solution import <name>`.\n"
    "- Use plain pytest: functions named `test_*`, `assert`, and `pytest.raises` for errors. "
    "No fixtures, no mocks, no I/O, no network, no randomness.\n"
    "- Every test must pass on the implementation shown.\n"
    "- Aim to expose bugs: cover boundaries, empty and degenerate inputs, both sides of "
    "every condition, and error paths. Assert exact values, not just truthiness.\n"
    "- Write at most {max_tests} test functions.\n"
    "- Reply with one ```python code block containing the complete test module and nothing else."
)

# D028 oracle-shape variant: identical except the assertion rule. The default
# rule ("assert exact values") demands the one skill the model lacks, predicting
# outputs; this one asks for oracles it can get right and exact values only
# where they can be read off the code.
SYSTEM_SHAPE = SYSTEM.replace(
    "Assert exact values, not just truthiness.",
    "Prefer assertions you can be certain of without computing outputs by hand: "
    "`pytest.raises` for error paths, type/length/shape checks, membership, "
    "relations between calls (round-trip, idempotence, ordering, monotonicity), "
    "and comparisons to the result of a simpler equivalent computation. Assert an "
    "exact value only when it is trivially readable from the code (booleans, None, "
    "empty results, small integers, short literal strings). Never assert a value "
    "you would have to work out.",
)
# D035 control: the teacher's style, prompted. Short suites of exact-value
# asserts, which is what the KodCode adapters learned to write and what the
# grounded harness can repair. If prompting for the style recovers the
# adapters' gain, the fine-tune bought formatting.
SYSTEM_TEACHER = SYSTEM.replace(
    "- Aim to expose bugs: cover boundaries, empty and degenerate inputs, both sides of "
    "every condition, and error paths. Assert exact values, not just truthiness.\n",
    "- One short test per behaviour: call the function once and assert its exact return "
    "value with `==`. Cover the base case, a typical case, and the boundaries. "
    "No comments, no docstrings, no helper code.\n",
)
STYLES = {"default": SYSTEM, "shape": SYSTEM_SHAPE, "teacher": SYSTEM_TEACHER}
TEACHER_MAX_TESTS = (
    5  # the adapters write 4.7-4.9 tests per suite; the budget cap of 8 still applies
)

# Two KodCode training examples (data/train/ext/train.jsonl), the shortest with
# 4-5 tests, used as few-shot exemplars of the teacher's style.
TEACHER_SHOTS = [
    (
        'def factorial(n):\n    """\n    Returns the factorial of n using recursion.\n    """\n'
        "    if n == 0:\n        return 1\n    else:\n        return n * factorial(n - 1)\n",
        "from solution import factorial\n\ndef test_factorial_base_case():\n"
        "    assert factorial(0) == 1\n\ndef test_factorial_of_one():\n"
        "    assert factorial(1) == 1\n\ndef test_factorial_of_positive_number():\n"
        "    assert factorial(5) == 120\n\ndef test_factorial_of_another_positive_number():\n"
        "    assert factorial(3) == 6\n",
    ),
    (
        'def and_gate(input1, input2):\n    """\n    Simulates an AND gate.\n'
        '    Returns 1 if both inputs are 1, otherwise returns 0.\n    """\n'
        "    return 1 if input1 == 1 and input2 == 1 else 0\n",
        "from solution import and_gate\n\ndef test_and_gate_both_inputs_high():\n"
        "    assert and_gate(1, 1) == 1\n\ndef test_and_gate_first_input_low():\n"
        "    assert and_gate(0, 1) == 0\n\ndef test_and_gate_second_input_low():\n"
        "    assert and_gate(1, 0) == 0\n\ndef test_and_gate_both_inputs_low():\n"
        "    assert and_gate(0, 0) == 0\n",
    ),
]

USER = "Write the pytest test module for this function.\n\n```python\n{source}\n```"

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)
_OPEN_FENCE = re.compile(r"```(?:python)?\s*\n")


def build_messages(
    source: str,
    max_tests: int,
    shots: list[tuple[str, str]] | None = None,
    style: str = "default",
) -> list[dict[str, str]]:
    """Chat messages: system, then (user, assistant) per exemplar, then the target."""
    messages = [{"role": "system", "content": STYLES[style].format(max_tests=max_tests)}]
    for shot_source, shot_suite in shots or []:
        messages.append({"role": "user", "content": USER.format(source=shot_source)})
        messages.append({"role": "assistant", "content": f"```python\n{shot_suite}```"})
    messages.append({"role": "user", "content": USER.format(source=source)})
    return messages


def extract_suite(text: str) -> tuple[str | None, dict[str, bool]]:
    """Normalise a reply into a parseable test module, identically for every arm (D018).

    Returns (suite or None, flags). Flags record what normalisation happened so
    each arm's reliance on it can be reported:
      truncated      the fenced block never closed (hit the token budget); the
                     text was trimmed back to the last parseable statement.
      pytest_import  ``pytest.`` was used without ``import pytest``; injected.
    """
    flags = {"truncated": False, "pytest_import": False}
    m = _FENCE.search(text)
    if m:
        candidate = m.group(1)
    elif (open_ := _OPEN_FENCE.search(text)) is not None:
        candidate, flags["truncated"] = text[open_.end() :], True
    else:
        candidate = text
    candidate = _trim_to_parseable(candidate.strip() + "\n")
    if candidate is None:
        return None, flags
    if "pytest." in candidate and not re.search(r"^\s*import pytest\b", candidate, re.M):
        candidate, flags["pytest_import"] = "import pytest\n" + candidate, True
    return candidate, flags


def _trim_to_parseable(src: str) -> str | None:
    """Drop trailing lines until the module parses; None if nothing parses."""
    lines = src.rstrip("\n").split("\n")
    while lines:
        try:
            ast.parse("\n".join(lines) + "\n")
            return "\n".join(lines) + "\n"
        except SyntaxError:
            lines.pop()
    return None


def enforce_test_budget(suite: str, max_tests: int) -> tuple[str, int]:
    """Keep non-test statements and the first ``max_tests`` test functions, in source order.

    Returns (truncated suite, number of test functions before truncation).
    """
    tree = ast.parse(suite)
    kept, n_tests = [], 0
    for node in tree.body:
        is_test = isinstance(node, ast.FunctionDef) and node.name.startswith("test")
        if is_test:
            n_tests += 1
            if n_tests > max_tests:
                continue
        kept.append(node)
    return ast.unparse(ast.Module(body=kept, type_ignores=[])) + "\n", n_tests
