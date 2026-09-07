SOURCE = '''def is_palindrome(s: str) -> bool:
    """True if s reads the same backwards, ignoring case and non-alphanumerics."""
    cleaned = "".join(ch.lower() for ch in s if ch.isalnum())
    return cleaned == cleaned[::-1]
'''
WEAK = """from solution import is_palindrome

def test_basic():
    assert is_palindrome("racecar")
    assert not is_palindrome("hello")
"""
STRONG = """from solution import is_palindrome

def test_basic():
    assert is_palindrome("racecar")
    assert not is_palindrome("hello")

def test_ignores_case_and_punctuation():
    assert is_palindrome("A man, a plan, a canal: Panama")
    assert is_palindrome("No 'x' in Nixon")

def test_empty_and_single():
    assert is_palindrome("")
    assert is_palindrome("z")

def test_case_matters_only_after_lowering():
    assert is_palindrome("Aa")
    assert not is_palindrome("ab")
"""
EQUIVALENT = {}
NOTES = "String normalisation; exercises string method swaps and the return path."
