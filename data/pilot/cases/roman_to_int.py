SOURCE = '''def roman_to_int(s: str) -> int:
    """Convert a Roman numeral (I..MMMCMXCIX) to an int using subtractive notation."""
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0
    for ch in reversed(s):
        value = values[ch]
        if value < prev:
            total -= value
        else:
            total += value
        prev = value
    return total
'''
WEAK = """from solution import roman_to_int

def test_simple():
    assert roman_to_int("III") == 3
    assert roman_to_int("X") == 10
"""
STRONG = """from solution import roman_to_int

def test_additive():
    assert roman_to_int("III") == 3
    assert roman_to_int("XVI") == 16
    assert roman_to_int("MMXXIV") == 2024

def test_subtractive():
    assert roman_to_int("IV") == 4
    assert roman_to_int("IX") == 9
    assert roman_to_int("XL") == 40
    assert roman_to_int("XC") == 90
    assert roman_to_int("CD") == 400
    assert roman_to_int("CM") == 900

def test_mixed_and_largest():
    assert roman_to_int("MCMXCIV") == 1994
    assert roman_to_int("MMMCMXCIX") == 3999

def test_single_letters():
    assert roman_to_int("I") == 1
    assert roman_to_int("M") == 1000
"""
EQUIVALENT = {
    "boundary:5:11:plus1": "prev 0 -> 1: numeral values are >= 1, so `value < prev` never changes",
}
NOTES = "Reverse loop with strict comparison; weak suite never triggers the subtractive branch."
