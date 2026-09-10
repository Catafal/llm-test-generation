"""Oracle-filled expected values for training targets (D023, FT6).

Weekend one measured that most invalid suites fail on a wrong hand-computed
expected value, not on test design. The harness already executes the
reference implementation, so it can supply the value instead of the model:

1. ``sites``       find every ``assert <expr> == <literal>`` in a suite.
2. ``instrument``  replace each such assert with a call that records
                   ``repr(<expr>)`` to ``oracle.json`` in the sandbox workdir.
3. ``fill``        run the instrumented suite on the reference, then rewrite
                   the literals whose recorded value differs, if the repr is
                   short enough (``MAX_ORACLE_REPR``): a 15-digit float is a
                   value the model could never compute, so teaching it to
                   emit one would teach confident guessing.

Only exact-equality literal asserts are touched. ``pytest.approx``,
relational asserts, parametrised ``expected`` names and asserts inside
``pytest.raises`` blocks are left as written; they are counted so the
assertion-style breakdown can report them.
"""

import ast
import json
from dataclasses import dataclass, field

from config import MAX_ORACLE_REPR
from testgen.harness.runner import RunResult, run_suite

ORACLE_FILE = "oracle.json"
_RECORDER = "__oracle"

# Prepended to the instrumented suite. Rewrites the file on every call so a
# later crash in the same test still leaves the earlier values on disk.
_HELPER = f'''import json as __json
__ORACLE_VALUES = {{}}


def {_RECORDER}(index, value):
    __ORACLE_VALUES[index] = repr(value)
    with open("{ORACLE_FILE}", "w") as __f:
        __json.dump(__ORACLE_VALUES, __f)
'''


@dataclass
class OracleStats:
    sites: int = 0  # literal-equality asserts found
    recorded: int = 0  # sites whose expression ran and was recorded
    replaced: int = 0  # literals rewritten with the executed value
    already_correct: int = 0  # recorded value equal to the written literal
    too_long: int = 0  # differing value skipped: repr over MAX_ORACLE_REPR
    unliteral: int = 0  # differing value skipped: repr is not a Python literal
    repr_lengths: list[int] = field(default_factory=list)  # of replaced values


def _is_literal(node: ast.AST) -> bool:
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, TypeError, SyntaxError):
        return False


def _is_site(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Assert)
        and isinstance(node.test, ast.Compare)
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and _is_literal(node.test.comparators[0])
    )


def sites(suite: str) -> list[ast.Assert]:
    """All ``assert <expr> == <literal>`` nodes, in ``ast.walk`` order."""
    return [n for n in ast.walk(ast.parse(suite)) if _is_site(n)]


class _Instrument(ast.NodeTransformer):
    def __init__(self) -> None:
        self.index = 0

    def visit_Assert(self, node: ast.Assert) -> ast.AST:
        if not _is_site(node):
            return node
        call = ast.Call(
            func=ast.Name(_RECORDER, ast.Load()),
            args=[ast.Constant(self.index), node.test.left],
            keywords=[],
        )
        self.index += 1
        return ast.copy_location(ast.Expr(call), node)


def instrument(suite: str) -> str:
    """Suite with every literal-equality assert replaced by a recording call."""
    tree = _Instrument().visit(ast.parse(suite))
    ast.fix_missing_locations(tree)
    return _HELPER + ast.unparse(tree) + "\n"


class _Fill(ast.NodeTransformer):
    def __init__(self, values: dict[int, str], stats: OracleStats, max_repr: int | None) -> None:
        self.values, self.stats, self.index, self.max_repr = values, stats, 0, max_repr

    def visit_Assert(self, node: ast.Assert) -> ast.AST:
        if not _is_site(node):
            return node
        i, self.index = self.index, self.index + 1
        self.stats.sites += 1
        if i not in self.values:
            return node
        self.stats.recorded += 1
        text = self.values[i]
        try:
            new = ast.parse(text, mode="eval").body
            new_value = ast.literal_eval(new)
        except (ValueError, TypeError, SyntaxError):
            self.stats.unliteral += 1
            return node
        # Same value (True == 1, 1 == 1.0) is not a wrong literal; leave it.
        if _same(new_value, ast.literal_eval(node.test.comparators[0])):
            self.stats.already_correct += 1
            return node
        if self.max_repr is not None and len(text) > self.max_repr:
            self.stats.too_long += 1
            return node
        node.test.comparators[0] = new
        self.stats.replaced += 1
        self.stats.repr_lengths.append(len(text))
        return node


def _same(a, b) -> bool:
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001  (exotic __eq__ in generated literals)
        return False


def fill(
    suite: str, reference: str, max_repr: int | None = MAX_ORACLE_REPR
) -> tuple[str, OracleStats, RunResult]:
    """Rewrite wrong literal expected values with the reference's executed value.

    ``max_repr`` caps the length of a written-back value. Training data keeps
    the default (D023: never teach a value the model cannot compute);
    evaluation under D030 passes ``None`` because there the harness, not the
    model, owns the values. Returns (filled suite, stats, the instrumented
    run). The instrumented run is *not* a validity check: asserts were
    removed. Callers rerun the filled suite on the reference to decide
    validity, exactly as for any other suite.
    """
    run = run_suite(instrument(suite), reference, collect=(ORACLE_FILE,))
    raw = run.artifacts.get(ORACLE_FILE)
    values = {int(k): v for k, v in json.loads(raw).items()} if raw else {}
    stats = OracleStats()
    tree = _Fill(values, stats, max_repr).visit(ast.parse(suite))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n", stats, run


def assertion_styles(suite: str) -> dict[str, int]:
    """Count assert kinds: what the model leans on to check behaviour.

    literal_eq   ``== <literal>``: needs a hand-computed value (the failure mode)
    approx       ``pytest.approx``
    relational   ``<``, ``<=``, ``>``, ``>=``, ``!=``
    membership   ``in`` / ``not in`` / ``is`` / ``is not``
    expr_eq      ``==`` against a non-literal (round-trip, another call, a name)
    other        truthiness, ``isinstance``, anything else
    """
    counts = dict.fromkeys(
        ("literal_eq", "approx", "relational", "membership", "expr_eq", "other"), 0
    )
    for node in ast.walk(ast.parse(suite)):
        if not isinstance(node, ast.Assert):
            continue
        t = node.test
        if isinstance(t, ast.Compare) and len(t.ops) == 1:
            op, rhs = t.ops[0], t.comparators[0]
            if "approx" in ast.unparse(rhs):
                counts["approx"] += 1
            elif isinstance(op, ast.Eq):
                counts["literal_eq" if _is_literal(rhs) else "expr_eq"] += 1
            elif isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.NotEq)):
                counts["relational"] += 1
            else:
                counts["membership"] += 1
        else:
            counts["other"] += 1
    return counts
