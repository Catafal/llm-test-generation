from testgen.data.similarity import (
    ast_hash,
    ast_shingles,
    cosine_matrix,
    jaccard,
    ngram_overlap,
    ngrams,
)

A = 'def add(a, b):\n    """Add."""\n    return a + b\n'
A_RENAMED = 'def plus(x, y):\n    """Sum two numbers."""\n    return x + y\n'
B = (
    'def clamp(x, lo, hi):\n    """Clamp."""\n    if x < lo:\n        return lo\n'
    "    if x > hi:\n        return hi\n    return x\n"
)


def test_ngram_overlap_needs_ten_shared_tokens():
    assert ngram_overlap(ngrams(A), ngrams(A_RENAMED)) == 0  # renamed, too short to share 10-grams
    long = "x = " + " + ".join(f"a{i}" for i in range(12))
    assert ngram_overlap(ngrams(long), ngrams(long + "\ny = 1")) > 0


def test_ast_hash_ignores_names_and_docstrings():
    assert ast_hash(A) == ast_hash(A_RENAMED)
    assert ast_hash(A) != ast_hash(B)


def test_ast_jaccard_orders_similarity():
    same = jaccard(ast_shingles(A), ast_shingles(A_RENAMED))
    diff = jaccard(ast_shingles(A), ast_shingles(B))
    assert same == 1.0 and diff < 0.85


def test_cosine_matrix_basic():
    m = cosine_matrix([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]])
    assert round(m[0][0], 3) == 1.0 and round(m[1][0], 3) == 0.0 and round(m[0][1], 3) == 0.707
