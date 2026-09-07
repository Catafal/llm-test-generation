SOURCE = '''def running_max(xs: list[int]) -> list[int]:
    """Return a list where element i is the max of xs[:i+1]. Empty input gives []."""
    result: list[int] = []
    current = None
    for x in xs:
        if current is None or x > current:
            current = x
        result.append(current)
    return result
'''
WEAK = """from solution import running_max

def test_increasing():
    assert running_max([1, 2, 3]) == [1, 2, 3]
"""
STRONG = """from solution import running_max

def test_increasing():
    assert running_max([1, 2, 3]) == [1, 2, 3]

def test_plateaus_and_drops():
    assert running_max([3, 1, 4, 1, 5]) == [3, 3, 4, 4, 5]
    assert running_max([2, 2, 1]) == [2, 2, 2]

def test_empty_and_single():
    assert running_max([]) == []
    assert running_max([7]) == [7]

def test_negative_values():
    assert running_max([-3, -1, -2]) == [-3, -1, -1]
"""
EQUIVALENT = {}
NOTES = (
    "Sentinel None + strict comparison; `>` vs `>=` is equivalent here (ties keep the same max)."
)
