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

USER = "Write the pytest test module for this function.\n\n```python\n{source}\n```"

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def build_messages(
    source: str, max_tests: int, shots: list[tuple[str, str]] | None = None
) -> list[dict[str, str]]:
    """Chat messages: system, then (user, assistant) per exemplar, then the target."""
    messages = [{"role": "system", "content": SYSTEM.format(max_tests=max_tests)}]
    for shot_source, shot_suite in shots or []:
        messages.append({"role": "user", "content": USER.format(source=shot_source)})
        messages.append({"role": "assistant", "content": f"```python\n{shot_suite}```"})
    messages.append({"role": "user", "content": USER.format(source=source)})
    return messages


def extract_suite(text: str) -> str | None:
    """First fenced python block; else the whole reply if it parses; else None."""
    m = _FENCE.search(text)
    candidate = m.group(1) if m else text
    candidate = candidate.strip() + "\n"
    try:
        ast.parse(candidate)
    except SyntaxError:
        return None
    return candidate


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
