"""Trivial-compiler-equivalence filter (D013, corrected by D015).

Two sources are *trivially equivalent* when CPython compiles them to the same
bytecode: e.g. a constant changed inside an ``if False:`` block that the
compiler drops. This is cheap and sound (identical bytecode cannot behave
differently) but catches almost nothing for our operators, because a changed
constant or operator almost always changes bytecode. The honest control is
hand-labelling on the probe set; this filter only removes the free wins and
reports how many it found.
"""

import types

from testgen.mutate.operators import Mutant


def _fingerprint(code: types.CodeType) -> tuple:
    """Bytecode plus constants, recursing into nested code objects; ignores names/lines."""
    consts = tuple(_fingerprint(c) if isinstance(c, types.CodeType) else c for c in code.co_consts)
    return (code.co_code, consts, code.co_names, code.co_varnames)


def is_trivially_equivalent(original: str, mutant: str) -> bool:
    try:
        return _fingerprint(compile(original, "<m>", "exec")) == _fingerprint(
            compile(mutant, "<m>", "exec")
        )
    except SyntaxError:
        return False


def split_equivalent(source: str, mutants: list[Mutant]) -> tuple[list[Mutant], set[str]]:
    """Return (live mutants, ids of trivially equivalent ones)."""
    equivalent = {m.id for m in mutants if is_trivially_equivalent(source, m.source)}
    return [m for m in mutants if m.id not in equivalent], equivalent
