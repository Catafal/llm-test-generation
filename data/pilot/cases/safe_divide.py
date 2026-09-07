SOURCE = '''def safe_divide(a: float, b: float) -> float | None:
    """Return a / b, or None when b is zero."""
    try:
        return a / b
    except ZeroDivisionError:
        return None
'''
WEAK = """from solution import safe_divide

def test_divides():
    assert safe_divide(6, 3) == 2
"""
STRONG = """from solution import safe_divide

def test_divides():
    assert safe_divide(6, 3) == 2
    assert safe_divide(1, 4) == 0.25

def test_zero_divisor_returns_none():
    assert safe_divide(1, 0) is None
    assert safe_divide(0, 0) is None

def test_negative_and_zero_numerator():
    assert safe_divide(-6, 3) == -2
    assert safe_divide(0, 5) == 0
"""
EQUIVALENT = {}
NOTES = "Exception handler mutant (`except ():`) and return-None mutants; probe swaps `/` -> `*`."
