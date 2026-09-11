# Existing (function, test) datasets for stage 1 of D032 (2026-09-11)

Two Sonnet passes (`01-huggingface-survey.md`, `02-papers-github-kaggle-survey.md`;
both hit the session's web-search quota and worked from dataset cards, arXiv
and GitHub directly) plus my own verification of the three finalists' cards.

## Verified finalists

| dataset | rows | licence | tests | solutions | origin | fit |
|---|---|---|---|---|---|---|
| **KodCode-V1** (HF `KodCode/KodCode-V1`, 2025-03) | 487K, 12 subsets | **CC BY-NC 4.0** | pytest modules, `from solution import <name>`, execution-verified | GPT-4o-0513, pass rate over 10 trials | synthetic | 5: exact format match |
| **OpenCodeInstruct** (HF `nvidia/OpenCodeInstruct`, 2025-04) | ~4.97M | CC BY 4.0 | list of `assert` statements, execution status + score per row | in `output` | synthetic (self/evol-instruct) | 4: wrap asserts into pytest |
| **AceCode-87K** (HF `TIGER-Lab/AceCode-87K`, 2025-02) | 87K | MIT | `assert` lists, LLM-imagined then filtered by pass rate | in `inferences` with pass rates | synthetic | 4: wrap asserts, pick a passing completion |

Ruled out (see reports): methods2test (Java), TestGenEval (CC BY-NC,
file-level, 11 repos), UniTSyn (Apache, but a mining pipeline to re-run,
not a file), CAT-LM (file-level, 260 GB), SWE-bench family (repo-level),
APPS/TACO/CodeContests/LiveCodeBench (I/O judges), code_exercises and
Tested-143k (no tests), Kaggle (nothing), The Stack mining (a project of
its own). All candidates predate 2026, so none can contain the post-June-
2026 held-out pool; the three-layer decontamination still runs.

## What this changes about the ladder

Every usable dataset is **model-generated**: KodCode by GPT-4o, the other
two by NVIDIA/TIGER-Lab pipelines. Stage 1 is therefore teacher
distillation with a frontier teacher at scale, which is the recipe the
field's positive results use. If it works, stage 2 (a local 30B teacher)
is redundant; if it fails at 4,000 execution-verified frontier-quality
examples, a weaker local teacher will not do better, and stage 3 (RL) is
the remaining angle.

## Recommendation

1. **KodCode-V1** first: exact format, execution-verified pytest with edge
   cases, difficulty labels. Subsets closest to our real-world functions:
   `Docs` (43K), `Package` (7K), `Filter` (77K), `Prefill` (43K, seeded
   from benchmark-style problems; check MBPP overlap and drop matches).
   Curation = our existing filter (purity, sandbox pass, ≥1 kill, budget,
   decontamination) plus a purity pass on the solution. **Licence:** CC
   BY-NC 4.0. Fine for a portfolio and a blog; the resulting adapter is
   non-commercial and NOTICE.md says so. Jordi's call.
2. **OpenCodeInstruct** if NC is unacceptable: CC BY 4.0, asserts wrapped
   one per `test_*` function, rows filtered by `tests_execution_status`.
3. **AceCode-87K** as the permissive small fallback.

Load:
```python
from datasets import load_dataset
kod = load_dataset("KodCode/KodCode-V1", split="train")           # fields: question, solution, test, subset, gpt_pass_percentage
oci = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)  # input, output, unit_tests, tests_execution_status
```
