"""D030 pairs: chosen = most kills after fill; rejected = invalid, then zero-kill, then weak."""

from testgen.train.pairs import GROUNDED_KILL_GAP, build_pairs_grounded


class _Tok:  # every suite fits; the token check is exercised in test_filter
    def apply_chat_template(self, msgs, tokenize=False):
        return "x"

    def __call__(self, text):
        class R:
            input_ids = [0]

        return R()


def _c(name: str, valid: bool, kills: int, n_tests: int = 4) -> dict:
    return {
        "parsed": True,
        "valid": valid,
        "kills": kills,
        "n_tests": n_tests,
        "suite_unaided": name,
    }


def test_grounded_pairs_rank_rejected_by_type_then_kills():
    cands = [
        _c("best", True, 6),
        _c("weak", True, 6 - GROUNDED_KILL_GAP),
        _c("near", True, 5),  # too close to the best: never a rejected
        _c("zero", True, 0),
        _c("bad", False, 0),
    ]
    pairs = build_pairs_grounded(cands, _Tok(), "s", "p")
    assert [(a["suite_unaided"], b["suite_unaided"]) for a, b in pairs] == [
        ("best", "bad"),
        (
            "near",
            "zero",
        ),  # second-best chosen; "weak" trails the best by the gap and is a rejected candidate
    ]


def test_grounded_pairs_need_a_killing_valid_chosen():
    assert build_pairs_grounded([_c("bad", False, 0), _c("zero", True, 0)], _Tok(), "s", "p") == []
