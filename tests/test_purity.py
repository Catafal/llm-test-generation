from testgen.data.purity import extract_candidates

MODULE = '''
import math
import os
from collections import Counter
import numpy as np

def pure(xs):
    """Sum of squares."""
    return sum(x * x for x in xs) + math.floor(0.5)

def uses_counter(words):
    """Most common word."""
    return Counter(words).most_common(1)[0][0]

def no_doc(x):
    return x

def does_io(path):
    """Reads a file."""
    return open(path).read()

def uses_os():
    """Impure import."""
    return os.getcwd()

def uses_numpy(a):
    """Third-party."""
    return np.sum(a)

def free_name(x):
    """References something undefined here."""
    return helper(x)

def uses_hash(s):
    """Nondeterministic across processes."""
    return hash(s) % 10

def with_nested(xs):
    """Nested def and comprehension are fine."""
    def sq(v):
        return v * v
    return [sq(x) for x in xs if x > 0]

class K:
    def method(self):
        """Not top-level."""
        return 1
'''


def test_accepts_pure_and_rejects_with_reasons():
    accepted, rejected = extract_candidates(MODULE)
    assert [c.name for c in accepted] == ["pure", "uses_counter", "with_nested"]
    assert rejected == {
        "no_doc": "no docstring",
        "does_io": "forbidden builtin: open",
        "uses_os": "impure import: os",
        "uses_numpy": "impure import: numpy",
        "free_name": "free name: helper",
        "uses_hash": "forbidden builtin: hash",
    }


def test_extracted_source_carries_only_needed_imports_and_runs():
    accepted, _ = extract_candidates(MODULE)
    by_name = {c.name: c for c in accepted}
    assert by_name["pure"].source.startswith("import math\n")
    assert "os" not in by_name["pure"].source
    assert by_name["uses_counter"].source.startswith("from collections import Counter\n")
    assert "import" not in by_name["with_nested"].source
    ns: dict = {}
    exec(by_name["uses_counter"].source, ns)  # noqa: S102 - test-only, trusted literal
    assert ns["uses_counter"](["a", "b", "a"]) == "a"


def test_syntax_error_module_is_rejected_whole():
    assert extract_candidates("def f(:\n") == ([], {"<module>": "syntax error"})


def test_is_test_code_catches_files_and_functions():
    from testgen.data.harvest import is_test_code

    assert is_test_code("evals/deterministic/test_experimental_toggle.py")
    assert is_test_code("pkg/utils_test.py")
    assert is_test_code("pkg/tests/helpers.py")
    assert is_test_code("pkg/core.py", "test_something")
    assert not is_test_code("pkg/core.py", "compute")
    assert not is_test_code("pkg/attest.py", "attest")
