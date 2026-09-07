SOURCE = '''def median(xs: list[float]) -> float:
    """Median of a non-empty list. Raises ValueError on empty input."""
    if not xs:
        raise ValueError("median of empty list")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2
'''
WEAK = """from solution import median

def test_odd():
    assert median([3, 1, 2]) == 2
"""
STRONG = """import pytest
from solution import median

def test_odd_and_even_length():
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 3, 2]) == 2.5

def test_single_and_pairs():
    assert median([7]) == 7
    assert median([1, 3]) == 2

def test_unsorted_with_duplicates_and_negatives():
    assert median([5, -1, 5, 0]) == 2.5
    assert median([-3, -1, -2]) == -2

def test_empty_raises():
    with pytest.raises(ValueError, match="median of empty list"):
        median([])
"""
EQUIVALENT = {}
NOTES = "Dense arithmetic: probe, boundary, compare and return mutants all present."
