"""The three decontamination similarity layers (D012, D017).

Each layer exposes one pairwise function returning a score in [0, 1] plus a
threshold from the research synthesis. Layers 1 and 3 are pure Python;
layer 2 takes an embedding callable so tests can inject a fake.

    n-gram    word-level 10-gram overlap        flag if any shared 10-gram
    AST       identifier-normalised structure   exact hash, or k=5 shingle Jaccard >= 0.85
    embedding cosine on a code-embedding model  flag >= 0.90, review 0.80-0.90
"""

import ast
import hashlib
import re
from collections.abc import Callable, Sequence

NGRAM_N = 10
AST_SHINGLE_K = 5
AST_JACCARD_FLAG = 0.85
COSINE_FLAG = 0.90
COSINE_REVIEW = 0.80

_WORD = re.compile(r"\w+")


# ---- layer 1: n-gram ---------------------------------------------------------


def ngrams(source: str, n: int = NGRAM_N) -> set[tuple[str, ...]]:
    toks = _WORD.findall(source)
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def ngram_overlap(a: set, b: set) -> int:
    """Number of shared n-grams. Any overlap flags (Qwen2.5-Coder convention)."""
    return len(a & b)


# ---- layer 3: AST -------------------------------------------------------------


class _Normalise(ast.NodeTransformer):
    """Identifiers -> positional slots; docstrings dropped; constants -> type buckets."""

    def __init__(self) -> None:
        self.slots: dict[str, str] = {}

    def _slot(self, name: str) -> str:
        return self.slots.setdefault(name, f"v{len(self.slots)}")

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id=self._slot(node.id), ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        return ast.copy_location(ast.arg(arg=self._slot(node.arg), annotation=None), node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = self._slot(node.name)
        node.returns = None
        node.decorator_list = []
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            node.body = node.body[1:] or [ast.Pass()]
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node.attr = self._slot(node.attr)
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        return ast.copy_location(ast.Constant(value=type(node.value).__name__), node)


def normalised_dump(source: str) -> str:
    tree = ast.parse(source)
    return ast.dump(_Normalise().visit(tree), annotate_fields=False)


def ast_hash(source: str) -> str:
    return hashlib.sha256(normalised_dump(source).encode()).hexdigest()


def ast_shingles(source: str, k: int = AST_SHINGLE_K) -> set[tuple[str, ...]]:
    seq = [type(n).__name__ for n in ast.walk(_Normalise().visit(ast.parse(source)))]
    return {tuple(seq[i : i + k]) for i in range(len(seq) - k + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


# ---- layer 2: embeddings -------------------------------------------------------

Embedder = Callable[[Sequence[str]], list[list[float]]]


def cosine_matrix(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Plain-Python cosine; fine for ~1,500 x 300 x 768."""

    def norm(v: list[float]) -> float:
        return sum(x * x for x in v) ** 0.5 or 1.0

    na, nb = [norm(v) for v in a], [norm(v) for v in b]
    return [
        [
            sum(x * y for x, y in zip(va, vb, strict=True)) / (na[i] * nb[j])
            for j, vb in enumerate(b)
        ]
        for i, va in enumerate(a)
    ]


def jina_embedder(model_name: str = "jinaai/jina-embeddings-v2-base-code") -> Embedder:
    """Real embedder; imports torch lazily so the rest of the package stays light."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed(texts: Sequence[str]) -> list[list[float]]:
        return model.encode(list(texts), normalize_embeddings=True, batch_size=32).tolist()

    return embed
