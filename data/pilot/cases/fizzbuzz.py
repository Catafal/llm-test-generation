SOURCE = '''def fizzbuzz(n: int) -> str:
    """'FizzBuzz' if n divisible by 15, 'Fizz' by 3, 'Buzz' by 5, else str(n)."""
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)
'''
WEAK = """from solution import fizzbuzz

def test_fizz_and_number():
    assert fizzbuzz(3) == "Fizz"
    assert fizzbuzz(1) == "1"
"""
STRONG = """from solution import fizzbuzz

def test_each_branch():
    assert fizzbuzz(15) == "FizzBuzz"
    assert fizzbuzz(3) == "Fizz"
    assert fizzbuzz(5) == "Buzz"
    assert fizzbuzz(7) == "7"

def test_multiples():
    assert fizzbuzz(30) == "FizzBuzz"
    assert fizzbuzz(9) == "Fizz"
    assert fizzbuzz(10) == "Buzz"

def test_zero_and_negative():
    assert fizzbuzz(0) == "FizzBuzz"
    assert fizzbuzz(-3) == "Fizz"
"""
EQUIVALENT = {}
NOTES = "Arithmetic (probe) and string constants; string mutants die to exact-match asserts."
