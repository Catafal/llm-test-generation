SOURCE = '''def chunk(xs: list[int], size: int) -> list[list[int]]:
    """Split xs into consecutive pieces of length size; the last may be shorter.

    Raises ValueError if size < 1.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    return [xs[i:i + size] for i in range(0, len(xs), size)]
'''
WEAK = """from solution import chunk

def test_even_split():
    assert chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]
"""
STRONG = """import pytest
from solution import chunk

def test_even_and_ragged_split():
    assert chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]
    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

def test_size_larger_than_input_and_size_one():
    assert chunk([1, 2], 5) == [[1, 2]]
    assert chunk([1, 2, 3], 1) == [[1], [2], [3]]

def test_empty_input():
    assert chunk([], 3) == []

def test_invalid_size_raises():
    with pytest.raises(ValueError, match="size must be >= 1"):
        chunk([1], 0)
    with pytest.raises(ValueError, match="size must be >= 1"):
        chunk([1], -2)
"""
EQUIVALENT = {}
NOTES = "Error path + slicing arithmetic; `size < 1` vs `size <= 1` is a real boundary bug."
