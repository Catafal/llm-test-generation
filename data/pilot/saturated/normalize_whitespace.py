SOURCE = '''def normalize_whitespace(s: str) -> str:
    """Collapse runs of whitespace to single spaces and strip both ends."""
    return " ".join(s.split())
'''
WEAK = """from solution import normalize_whitespace

def test_collapses():
    assert normalize_whitespace("a  b") == "a b"
"""
STRONG = """from solution import normalize_whitespace

def test_collapses_and_strips():
    assert normalize_whitespace("  a   b  ") == "a b"
    assert normalize_whitespace("a\\t\\nb") == "a b"

def test_empty_and_whitespace_only():
    assert normalize_whitespace("") == ""
    assert normalize_whitespace("   ") == ""

def test_single_word_unchanged():
    assert normalize_whitespace("word") == "word"
"""
EQUIVALENT = {}
NOTES = "Tiny function; tests whether string-constant mutants (' ' -> 'XX XX') are the only signal."
