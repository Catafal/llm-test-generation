SOURCE = '''def clamp(x: int, lo: int, hi: int) -> int:
    """Return x limited to the inclusive range [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
'''
WEAK = """from solution import clamp

def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
"""
STRONG = """from solution import clamp

def test_inside():
    assert clamp(5, 0, 10) == 5

def test_boundaries():
    assert clamp(0, 0, 10) == 0
    assert clamp(10, 0, 10) == 10

def test_outside():
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10

def test_degenerate_range():
    assert clamp(3, 4, 4) == 4
"""
EQUIVALENT = {
    "compare:3:7:Lt": "x < lo vs x <= lo: at x == lo both branches return lo",
    "compare:5:7:Gt": "x > hi vs x >= hi: at x == hi both branches return hi",
}
NOTES = "Boundary-heavy. Comparison flips at the boundary return the same value either way."
