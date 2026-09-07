# Pilot cases (T4)

Ten hand-written pure functions. Each case module defines:

- `SOURCE`: the reference implementation with a docstring (this is what the
  model will see).
- `WEAK`: a plausible but shallow pytest suite (happy path only).
- `STRONG`: a suite a careful engineer would write (boundaries, both sides,
  degenerate inputs, error paths).
- `EQUIVALENT`: hand-labelled `{mutant_id: reason}` for mutants no test can
  kill. Labelled by reading the mutant, not by whether a suite happened to
  miss it. Every entry carries its reason so a reviewer can disagree.
- `NOTES`: anything non-obvious about the case.

Acceptance rule (`make pilot`): after excluding trivially- and hand-labelled
equivalent mutants, the harness must score STRONG strictly above WEAK on every
case. Both suites must be valid (pass on the reference). If the rule fails,
the metric or the operators are wrong, not the suites.

Mutant ids are stable for a given `SOURCE` and `OPERATORS_VERSION`; editing a
source invalidates its equivalence labels.
