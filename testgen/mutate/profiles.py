"""Which mutation categories are active, and why (D016).

The engine knows every category; an experiment activates a named profile and
the frozen artifact records it, so two runs are comparable only if their
profile names match. Reasons live next to the switches so the write-up can
quote them.
"""

PROBE_CATEGORY = "arith"  # held out of all training curation (D015)

# category -> (active in eval, reason)
EVAL_PROFILE: dict[str, tuple[bool, str]] = {
    "compare": (True, "comparison, membership and identity flips: classic logic bugs"),
    "boundary": (True, "numeric constant +1: off-by-one"),
    "boolean": (True, "and/or, not, True/False, forced ternary arms"),
    "return": (True, "return value dropped"),
    "string": (True, "string constants and string-method swaps"),
    "loop": (True, "zero-iteration loops, break/continue"),
    "unary": (True, "unary minus / invert removed"),
    "assign": (True, "augmented assign flattened, assigned value dropped"),
    "bitwise": (True, "bitwise and shift swaps"),
    "exception": (True, "handler made to never match"),
    "decorator": (True, "decorator removed"),
    "match": (True, "one case dropped from a match"),
    PROBE_CATEGORY: (True, "arithmetic swaps; HELD-OUT PROBE, scored separately"),
    "call": (
        False,
        "argument -> None / kwarg renamed: kills by crash, not logic; inflates every suite",
    ),
    "lambda": (False, "lambda body -> None: kills by crash in practice"),
}

EVAL_PROFILE_NAME = "eval-v1"


def active_categories(profile: dict[str, tuple[bool, str]] = EVAL_PROFILE) -> tuple[str, ...]:
    return tuple(c for c, (on, _) in profile.items() if on)


def training_categories(profile: dict[str, tuple[bool, str]] = EVAL_PROFILE) -> tuple[str, ...]:
    """Eval set minus the probe. The probe must never feed a curation signal."""
    return tuple(c for c in active_categories(profile) if c != PROBE_CATEGORY)


ALL_CATEGORIES = tuple(EVAL_PROFILE)
