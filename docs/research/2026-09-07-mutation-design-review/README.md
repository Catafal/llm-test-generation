# Mutation Design Review: mutant cap and held-out operator

**Date:** 2026-09-07
**Method:** Two proposals sent to three independent Claude Opus reviewers with
an identical brief; the third was instructed to be deliberately skeptical.
No reviewer saw another's answer. Verdicts summarised below; the proposals
were revised as a result.

## Proposals under review

- **Q1.** Cap mutants at 5 per function, sampled with a fixed seed, for
  predictable run time and comparable counts across functions.
- **Q2.** Hold out "boundary constant ±1" from all training-curation signals
  as the operator-overfitting probe, because it is the class most tied to
  the boundary tests the model should learn.

## Verdicts

| | Reviewer A | Reviewer B | Reviewer C (skeptical) |
|---|---|---|---|
| Q1 cap at 5 | agree-with-change | agree-with-change | disagree |
| Q2 boundary held out | disagree | disagree | disagree |

## Convergent arguments on Q1

- A 5-mutant denominator quantises per-function score to steps of 0.2. That
  sampling noise plausibly exceeds the fine-tune-vs-few-shot gap being
  measured, and it is noise chosen voluntarily; mutation runs on single pure
  functions are cheap.
- Uniform site sampling is dominated by whichever operator has the most
  sites, so the probe category gets zero mutants on many functions and the
  Goodhart check silently voids itself.
- Capping flattens difficulty: a 40-site function and a 5-site function
  weigh the same, which rewards shallow suites and hides exactly where a
  fine-tune would differ.
- "Same seed" is not enough if enumeration order depends on AST traversal
  that changes with any operator edit; the mutant set must be a frozen,
  versioned artifact shared byte-identically across all arms.
- All three: keep a small cap for **training-data curation** where cost
  binds; run the **held-out eval uncapped**, one mutant per site, or with a
  large cap stratified by operator. Report micro and macro averages with
  bootstrap CIs.

## Convergent arguments on Q2

- The rationale inverts the requirement. A held-out probe measures
  generalisation only if the held-out operator is not implied by the
  retained ones. Boundary ±1 is the most entangled: the on-boundary input
  that kills `<` → `<=` kills `n` → `n+1` at the same site. Training on
  comparison flips transfers boundary competence mechanically, so the probe
  cannot fail in the way it needs to.
- Boundary constants also have the highest true equivalent-mutant rate
  (dead branches, unbounded loop guards), and bytecode comparison catches
  none of it because differing constants always differ in bytecode. The
  probe denominator would be inflated for reasons unrelated to the model.
- Alternatives offered: return-value change (A, C), boolean swap (B), or a
  **fifth operator never used in training** such as arithmetic swap (C).

## Revised decisions (proposed to Jordi)

1. **Eval mutants uncapped**, one mutant per site, frozen to a versioned
   artifact shared across every arm. Cap only for training curation, with a
   different seed. Report micro and macro scores with bootstrap CIs.
2. **Held-out probe is a fifth operator, arithmetic swap** (`+`↔`-`,
   `*`↔`//`), never used in any curation signal. All four original operators
   stay in training. This is a true out-of-distribution probe and costs no
   training signal.
3. **Equivalence handling corrected.** Bytecode comparison is kept as a
   cheap filter but stated to catch almost nothing for these operators. The
   real control is hand-labelling equivalents on the full probe set and on a
   stratified sample of the main set, reported as a limitation. Amends D013.
