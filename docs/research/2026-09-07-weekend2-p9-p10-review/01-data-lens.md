# Reviewer 1 — data / distribution lens (Opus, independent, no file access)

Question: P9, where do the training functions come from?

## 1. The full option space

- (a) MBPP only. Optimises cleanliness and zero harvest cost. Risks: toy difficulty, synthesised docstrings, and the failure mode you are training against barely occurs there.
- (b) Pre-floor GitHub harvest (2025-06 to 2026-05). Optimises distribution match. Risk: the teacher has seen these functions.
- (b') Post-floor overflow harvest: repos created after 2026-06-01 that were not drawn for the held-out pool. Not on the list, and it strictly dominates (b) on the memorisation axis while matching distribution exactly. Risk: thinner supply, since the floor is only three months back.
- (c) Mixed, MBPP as minority. Optimises volume insurance. Risk: you cannot attribute a gain or a regression to either source.
- (d) PyPI sdists. Better function density and docstring coverage. Risk: corpus skews old and memorised; release dates do not bound authorship.
- (e) Teacher-synthesised functions. Total control over difficulty and operator coverage. Risk: synthetic style drifts from real code, and the teacher writes functions it finds easy, the opposite of what you need.
- (f) Curriculum or stratification by mutant count and expected-value difficulty. Not a source, an orthogonal selection policy, and the one that targets the bottleneck. Applies on top of any source.
- (g) The dev split itself. Free and perfectly matched, and it destroys early stopping. Reject.
- (h) Mutating existing functions into near-duplicates. Cheap volume, near-zero new signal, strains Jaccard decontamination.

## 2. What property actually moves validity

Invalid suites are wrong hand-computed expected values, not malformed code. That is a numeric-reasoning failure, and the fix a fine-tune can plausibly install is behavioural: prefer assertions whose expected value the model can produce reliably (approximate comparisons, round-trip and inverse checks, invariants, relational assertions, exact values only where it can evaluate them).

That behaviour only appears in the SFT set if the training functions are hard enough that naive computation fails. On MBPP-style functions the teacher gets the literal right by direct arithmetic, so every kept suite is full of hard-coded literals. You would be fine-tuning the exact habit that is breaking you. This is the decisive argument, and it is about difficulty, not provenance.

Ranked by how much each property moves the bottleneck: difficulty of expected-value computation dominates; operator coverage second (the arith holdout only reads as a probe if the other categories are represented); surface distribution match third, mostly for prompt-format robustness; docstring presence matters because it is in the inference-time input, and synthesising docstrings from MBPP prompts creates a train/inference template mismatch already forbidden; volume matters least past a few hundred.

## 3. Confounds an evals engineer will raise

(b): the teacher may have memorised these functions, so its suites carry correct literals without correct reasoning. Rejection sampling then keeps preferentially the functions where memorisation rescued the teacher, biasing the SFT set toward literal-heavy suites on well-known code. Undetectable from the pass rate. Also: content-hash families will not catch the same author or organisation publishing related code either side of the floor; extend union-find to owner and package name. Pre-floor GitHub is dense with forks and vendored copies, so effective diversity is lower than the count.

(a): distribution shift, synthesised docstrings breaking template parity, and MBPP's presence in every base model's pretraining, so teacher success is evidence of recall, not of writing a suite from the reference.

(c): nothing is attributable.

(b'): the reviewer asks whether the held-out pool skimmed the best functions and left training the residue; show the split was family-level and blind to quality.

Across every option, the teacher's identity and whether it saw the training functions determines yield and bias far more than pre-floor versus post-floor does.

## 4. Ranked recommendation

Criteria in priority order: does the source contain the failure mode you are training away; is teacher success evidence of reasoning rather than recall; does it match the inference-time input format including real docstrings; can you defend the split in one paragraph.

1. (b') post-floor overflow, stratified by expected-value difficulty. Same distribution as test, no memorisation ambiguity, cleanest story.
2. (b) pre-floor, as top-up once post-floor supply runs dry.
3. (c) with MBPP capped near 15 percent, volume insurance only if under 500.
4. (a) alone, only if the harvest fails, stated plainly as toy.

Switch to (b) as primary if post-floor overflow yields under roughly 350 functions clearing the >=8-mutant filter after family-disjoint exclusion. Move toward (c) if the teacher's keep rate on real harvested functions falls below roughly 40 percent.

## 5. One-hour de-risking experiment

Sample 25 MBPP functions and 25 real harvested functions, matched on mutant count. Run the teacher at planned K, execute every candidate, record per source: yield (fraction with at least one valid killing suite) and assertion style (fraction of assertions in kept suites that are hard-coded literals versus approximate, relational, invariant or round-trip). If MBPP's kept suites are overwhelmingly literal while the real ones are not, option (a) is dead on the merits. If the two look the same, difficulty is not the lever and the harvest is harder to justify.

**Verdict.** P9 -> (b') because post-floor overflow buys the distribution match without conceding that the teacher may have memorised the training functions, and difficulty stratification, not provenance, is what attacks the validity bottleneck.
