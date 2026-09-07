SOURCE = '''def letter_grade(score: int) -> str:
    """Map 0-100 to A/B/C/D/F at 90/80/70/60. Raises ValueError outside 0-100."""
    if score < 0 or score > 100:
        raise ValueError("score out of range")
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"
'''
WEAK = """from solution import letter_grade

def test_grades():
    assert letter_grade(95) == "A"
    assert letter_grade(50) == "F"
"""
STRONG = """import pytest
from solution import letter_grade

def test_each_band():
    assert letter_grade(95) == "A"
    assert letter_grade(85) == "B"
    assert letter_grade(75) == "C"
    assert letter_grade(65) == "D"
    assert letter_grade(50) == "F"

def test_exact_thresholds_and_one_below():
    assert letter_grade(90) == "A" and letter_grade(89) == "B"
    assert letter_grade(80) == "B" and letter_grade(79) == "C"
    assert letter_grade(70) == "C" and letter_grade(69) == "D"
    assert letter_grade(60) == "D" and letter_grade(59) == "F"

def test_range_edges_and_errors():
    assert letter_grade(100) == "A"
    assert letter_grade(0) == "F"
    with pytest.raises(ValueError, match="score out of range"):
        letter_grade(101)
    with pytest.raises(ValueError, match="score out of range"):
        letter_grade(-1)
"""
EQUIVALENT = {}
NOTES = "Pure threshold logic; weak suite hits two bands and no boundary."
