"""Execution-explanation targets: trace literal asserts, render inline or as a prefix."""

from testgen.train.tracesuite import inline_target, prefix_target, trace_sites

REF = '''
def clamp(x, lo, hi):
    """Clamp x into [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
'''

SUITE = """from solution import clamp


def test_low():
    assert clamp(-5, 0, 10) == 0


def test_via_local():
    result = clamp(50, 0, 10)
    assert result == 10


def test_relational():
    assert clamp(3, 0, 10) <= 10
"""


def test_trace_sites_and_inline():
    sites, stats = trace_sites(SUITE, REF, "clamp")
    assert stats.sites == 2 and stats.traced == 2
    assert [ln for ln, _ in sites] == [5, 10]
    out = inline_target(SUITE, sites)
    assert "# clamp(-5, 0, 10)" in out and "#   returns 0" in out
    assert "# clamp(50, 0, 10)" in out and "#   returns 10" in out
    # comments sit above the asserts, indented like them
    lines = out.splitlines()
    i = lines.index("    assert clamp(-5, 0, 10) == 0")
    assert lines[i - 1].startswith("    #   returns")
    import ast

    ast.parse(out)  # still a valid module


def test_caps_and_prefix():
    sites, stats = trace_sites(SUITE, REF, "clamp", max_asserts=1)
    assert stats.traced == 1 and stats.skipped_cap == 1
    out = prefix_target(SUITE, sites)
    assert out.startswith("<derivation>\n# clamp(-5, 0, 10)") and out.endswith("```")
