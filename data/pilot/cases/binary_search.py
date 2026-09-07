SOURCE = '''def binary_search(xs: list[int], target: int) -> int:
    """Return the index of target in sorted xs, or -1 if absent."""
    lo, hi = 0, len(xs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if xs[mid] == target:
            return mid
        if xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
'''
WEAK = """from solution import binary_search

def test_found():
    assert binary_search([1, 3, 5, 7], 5) == 2
"""
STRONG = """from solution import binary_search

def test_found_middle_and_ends():
    xs = [1, 3, 5, 7, 9]
    assert binary_search(xs, 5) == 2
    assert binary_search(xs, 1) == 0
    assert binary_search(xs, 9) == 4

def test_every_position_is_found():
    xs = [1, 3, 5, 7, 9]
    assert [binary_search(xs, v) for v in xs] == [0, 1, 2, 3, 4]

def test_absent_inside_and_outside_range():
    xs = [1, 3, 5, 7, 9]
    assert binary_search(xs, 4) == -1
    assert binary_search(xs, 0) == -1
    assert binary_search(xs, 10) == -1

def test_empty_and_single():
    assert binary_search([], 1) == -1
    assert binary_search([1], 1) == 0
    assert binary_search([1], 2) == -1

def test_even_length():
    assert binary_search([2, 4], 4) == 1
    assert binary_search([2, 4], 2) == 0
"""
EQUIVALENT = {
    "compare:8:11:Lt": "xs[mid] < target vs <=: equality already returned on the line above",
}
NOTES = "Off-by-one territory: +1 on `mid + 1`/`mid - 1`; `<=` vs `<` on the loop guard."
