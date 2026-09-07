# Source report 03 — Experimental rigor for a LoRA fine-tune scored by mutation score

Research agent report (Claude Sonnet), 2026-09-07. Unedited except for this header.
Claims marked unverified by the agent remain unverified.

---

## 1. Held-out design and decontamination

- **Split by function family, not by function.** Cluster by repo, module, or semantic family first, then split at the cluster level — train-test overlap silently inflates scores ([Deduplicating Training Data Makes Language Models Better](https://aclanthology.org/2022.acl-long.577.pdf)).
- **Layer three duplicate detectors:** n-gram/token overlap (copy-paste, renaming), embedding cosine similarity (semantic near-duplicates; Open-Platypus removed items with >80% SentenceTransformer cosine), and AST/structural similarity (logic-identical code with different tokens). "Embeddings-based retrieval is effective, whereas n-gram overlap fails" for code ([contamination detection survey](https://arxiv.org/html/2502.14425v2), [comprehensive survey](https://arxiv.org/pdf/2404.00699)); the AST/CodeBLEU layer (Dolos, tree-sitter k-gram) is not optional for code.
- **Date-cutoff pool as primary defense.** LiveCodeBench: evaluate only on problems dated after the model's training cutoff, so contamination is "impossible by construction" ([LiveCodeBench](https://arxiv.org/html/2403.07974v2)). SWE-bench's problems — 32.67% of "solved" patches involving solution leakage, models recalling file paths up to 76% of the time — are the cautionary tale ([SWE-rebench](https://www.alphaxiv.org/abs/2505.20411)).
- **Document as a decontamination report:** source pool and dates, base-model cutoff and how verified, dedup thresholds per layer, count removed per layer, final provenance.

## 2. Variance and significance on small eval sets (100–300 functions)

- **Greedy decoding for the headline number; pass@k sampling as secondary.** Codex used n=200 samples at temperature 0.6 for the unbiased pass@k estimator ([pass@k background](https://leehanchung.github.io/blogs/2025/09/08/pass-at-k/)). At 100–300 functions × 3 conditions, use greedy as primary and a small number of seeds (~5) at low temperature (0.2–0.6) as a variance check.
- **Paired bootstrap CIs on the difference, not per-arm CIs.** Same held-out functions across conditions → within-subject design; resample functions with replacement (≥1000 replicates), report the 2.5th/97.5th percentile of the *difference*. "If the CI on the difference contains zero, one cannot claim one system is better" ([paired bootstrap](https://medium.com/ai-enthusiast/comparing-nlp-models-with-confidence-the-paired-bootstrap-test-explained-c9a88532ea3d)).
- **McNemar's test for paired binary outcomes** (fine-tuned suite kills mutant M vs baseline kills M): χ² = (b−c)²/(b+c) on the disagreement cells ([McNemar](https://machinelearningmastery.com/mcnemars-test-for-machine-learning/)). Correct for multiple comparisons across many mutants/conditions ([Model Evaluation survey](https://arxiv.org/pdf/1811.12808)).
- **Sample-size sanity:** ~200 examples gives power ≈0.96 at α=0.05 for a medium effect; ~246 for a 95% CI with 5% margin at 80% base rate ([sample-size analysis](https://latitude.so/blog/sample-size-affects-llm-prompt-testing/)). A small true effect (a few points) will not be resolvable at this N; say so.

## 3. Equal-budget comparison

- Hold constant: max output tokens, temperature/top-p, few-shot exemplars counted toward budget, and **number of test cases per function** (mutation score is sensitive to raw test count). State whether "equal budget" means wall-clock, generated tokens, or dollars.
- Report cost/latency **alongside** quality: tokens per function, wall-clock per function, $-per-1000-functions. Compare at matched *output* budget and at matched *total* (prompt+output) cost as two rows; few-shot inflates prompt tokens.

## 4. Avoiding metric gaming

- **Equivalent mutants are the biggest bias:** rates 4–39% in real code; they disproportionately survive, so "if 5% of mutants are equivalent and a suite kills 80%, 25% of the remaining unkilled are equivalent" ([ISSTA 2024](https://homes.cs.washington.edu/~rjust/publ/equi_mutants_ems_issta_2024.pdf)). Mitigate with Trivial Compiler Equivalence (identical bytecode ⇒ equivalent, filter) and report filtered counts.
- **Trivial mutants inflate scores** ([mutation testing guide](https://testrigor.com/blog/understanding-mutation-testing-a-comprehensive-guide/)); an established operator set (mutmut/cosmic-ray defaults) makes trivial-mutant rate a known, comparable quantity.
- **Tests that hard-code the oracle:** "a test that invokes a method without asserting anything meaningful will still increase coverage"; weak assertions can accidentally kill mutants ([replicability study](https://arxiv.org/html/2607.22880v1), [test-oracle study](https://arxiv.org/pdf/2410.21136)). Guardrails: **validity rate** (suite runs and passes on the original) and **false-failure rate** (fails on correct code). SWE-bench Lite flakiness on unmodified code is 11.3% ([flaky-test dataset](https://arxiv.org/pdf/2605.21677)).
- **Manual inspection:** stratified sample of 20–30 suites (oversample extremes); check for re-executed literal outputs, test count vs assertion strength, whether kills are real behavioural checks. EvalPlus found "18 defects (11%) even in HumanEval's ground truth" by manual audit ([EvalPlus](https://arxiv.org/pdf/2305.01210)).

## 5. Regressions

- **HumanEval+ before and after**, not plain HumanEval: EvalPlus catches 19.3–28.9% pass@k drops invisible to the base benchmark.
- **Report the general-vs-task trade-off explicitly.** LoRA reduces but does not eliminate forgetting ([forgetting-aware pruning](https://arxiv.org/html/2509.08255v1), [O-LoRA](https://ideas.repec.org/a/axf/aidtaa/v3y2026i1p52-61.html)). Same paired-bootstrap CI on the HumanEval+ delta.
- **Operator-set overfitting:** hold out one mutation operator category from any curation signal and check whether held-out-category score tracks trained-on categories. If training data was filtered by mutation score against a fixed operator set, this is where Goodhart hits; score the held-out set on a different tool/operator set if possible.

## 6. Reporting standards

- **Model card** (intended use, training data, results, limitations) and **datasheet** for the held-out set ([Datasheets](https://www.emergentmind.com/topics/datasheets-for-datasets), [Model Cards](https://www.emergentmind.com/topics/model-cards-and-datasheets)).
- **Run manifest:** base model + revision hash, LoRA rank/alpha/targets, LR/schedule, data size and source, epochs, seeds, decoding config per condition, harness commit, mutation tool + operator set + version.
- **Negative/null results and limitations as a first-class section.** OpenAI walked back SWE-bench Verified as "no longer a meaningful signal"; the Anthropic–OpenAI joint evaluation is cited for publishing cross-checks and limitations ([joint eval](https://openai.com/index/openai-anthropic-safety-evaluation/), [Alignment Science writeup](https://alignment.anthropic.com/2025/openai-findings)). State: sample size and power, residual contamination risk, equivalent-mutant bias, any condition where the fine-tune did not win.
- Anthropic's agent-eval guidance: "combine deterministic checks, model-based judges, and human calibration" ([Demystifying evals](https://ai-eval.org/post/anthropic-demystifying-evals-for-ai-agents)).

## 7. Ablations for a two-weekend budget

**Run:** LoRA rank sweep (8/16/32) at fixed compute ([How Small Can You Go?](https://arxiv.org/html/2607.25583)); few-shot exemplar count (0/2/5) at matched budget; greedy vs low-temperature sampling; held-out operator category.

**Skip:** full-parameter fine-tune comparison; base-model size sweep; n=200 pass@k across all conditions; cross-language transfer.

---

## (a) Rigor principles

1. Split held-out functions by family/cluster, not by function ([ACL 2022](https://aclanthology.org/2022.acl-long.577.pdf)).
2. Layer n-gram, embedding, and AST similarity for decontamination ([survey](https://arxiv.org/html/2502.14425v2)).
3. Date-cutoff pool strictly after the base model's cutoff ([LiveCodeBench](https://arxiv.org/html/2403.07974v2)).
4. Greedy for primary results; small-seed low-temperature pass@1 for variance ([pass@k](https://leehanchung.github.io/blogs/2025/09/08/pass-at-k/)).
5. Paired bootstrap CIs on the score *difference* ([paired bootstrap](https://medium.com/ai-enthusiast/comparing-nlp-models-with-confidence-the-paired-bootstrap-test-explained-c9a88532ea3d)).
6. McNemar for paired binary mutant kills, with multiple-comparison correction ([McNemar](https://machinelearningmastery.com/mcnemars-test-for-machine-learning/)).
7. Hold decoding, max tokens, and test count constant; report cost/latency as separate rows.
8. Filter equivalent mutants (trivial compiler equivalence) before scoring ([ISSTA 2024](https://homes.cs.washington.edu/~rjust/publ/equi_mutants_ems_issta_2024.pdf)).
9. Track validity rate and false-failure rate as guardrails ([replicability study](https://arxiv.org/html/2607.22880v1)).
10. Manually inspect a stratified sample of suites ([EvalPlus](https://arxiv.org/pdf/2305.01210)).
11. HumanEval+ before/after for regressions ([EvalPlus](https://arxiv.org/pdf/2305.01210)).
12. Held-out mutation-operator category for overfitting.
13. Model card, datasheet, full run manifest ([Model Cards](https://www.emergentmind.com/topics/model-cards-and-datasheets)).
14. Limitations and null results as a first-class section ([OpenAI/Anthropic](https://openai.com/index/openai-anthropic-safety-evaluation/)).
15. Deterministic scoring plus human calibration ([Demystifying evals](https://ai-eval.org/post/anthropic-demystifying-evals-for-ai-agents)).

## (b) Minimal evidence packet

- Decontamination report (source/date range, cutoff, per-layer thresholds and counts)
- Held-out set datasheet
- Run manifest
- Primary results: mutation score per condition, paired bootstrap CI on deltas, McNemar p-value
- Cost/latency table at matched budget
- Validity and false-failure rates per condition
- Equivalent/trivial mutant filtering method and residual counts
- Manual-inspection sample (20–30) with notes
- HumanEval+ before/after with CI
- Held-out-operator-category score
- Limitations section

## (c) Could not verify

- No canonical "minimum seeds/samples" for a 100–300-function mutation-score comparison; the power numbers are from general LLM-eval discussions.
- No field-agreed acceptable false-failure/flakiness threshold for LLM test generation.
- No frontier-lab post specifically about mutation-score test-generation evals; the reporting analogy is drawn from adjacent domains.
