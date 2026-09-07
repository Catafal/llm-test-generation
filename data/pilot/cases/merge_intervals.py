SOURCE = '''def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching closed intervals. Empty input gives []."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            last_start, last_end = merged[-1]
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged
'''
WEAK = """from solution import merge_intervals

def test_merges_overlap():
    assert merge_intervals([(1, 3), (2, 5)]) == [(1, 5)]
"""
STRONG = """from solution import merge_intervals

def test_overlap_and_disjoint():
    assert merge_intervals([(1, 3), (2, 5)]) == [(1, 5)]
    assert merge_intervals([(1, 2), (4, 5)]) == [(1, 2), (4, 5)]

def test_touching_and_nested():
    assert merge_intervals([(1, 3), (3, 5)]) == [(1, 5)]
    assert merge_intervals([(1, 10), (2, 3)]) == [(1, 10)]

def test_unsorted_input_and_chain():
    assert merge_intervals([(5, 6), (1, 2), (2, 5)]) == [(1, 6)]

def test_empty_and_single():
    assert merge_intervals([]) == []
    assert merge_intervals([(7, 7)]) == [(7, 7)]
"""
EQUIVALENT = {}
NOTES = "Touching intervals (<= vs <) and nested intervals (max) are the discriminating cases."
