SOURCE = '''def is_leap_year(year: int) -> bool:
    """Gregorian leap year: divisible by 4, except centuries unless divisible by 400."""
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    return year % 4 == 0
'''
WEAK = """from solution import is_leap_year

def test_leap():
    assert is_leap_year(2000)
    assert not is_leap_year(2001)
"""
STRONG = """from solution import is_leap_year

def test_common_years():
    assert not is_leap_year(2001)
    assert not is_leap_year(2023)

def test_ordinary_leap_years():
    assert is_leap_year(2004)
    assert is_leap_year(1996)

def test_century_rules():
    assert is_leap_year(1900) is False
    assert is_leap_year(2100) is False
    assert is_leap_year(2000) is True
    assert is_leap_year(1600) is True
"""
EQUIVALENT = {}
NOTES = "Three-rule boolean logic; the 1900 case is the classic miss."
