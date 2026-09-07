SOURCE = '''def word_count(text: str) -> dict[str, int]:
    """Count case-insensitive whitespace-separated words. Empty input gives {}."""
    counts: dict[str, int] = {}
    for word in text.lower().split():
        counts[word] = counts.get(word, 0) + 1
    return counts
'''
WEAK = """from solution import word_count

def test_counts():
    assert word_count("a b a") == {"a": 2, "b": 1}
"""
STRONG = """from solution import word_count

def test_counts():
    assert word_count("a b a") == {"a": 2, "b": 1}

def test_case_insensitive():
    assert word_count("Go go GO") == {"go": 3}

def test_empty_and_whitespace_only():
    assert word_count("") == {}
    assert word_count("   \\n\\t ") == {}

def test_multiple_spaces_and_newlines():
    assert word_count("x  y\\nx") == {"x": 2, "y": 1}
"""
EQUIVALENT = {}
NOTES = "Loop + dict accumulation; zero-iteration loop and arith probe both apply."
