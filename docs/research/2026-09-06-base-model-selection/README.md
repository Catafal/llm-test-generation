# Base Model Selection for Local Test-Generation Fine-Tuning

**Date:** 2026-09-06
**Question:** Which open-weight coder model should we LoRA fine-tune to write
bug-exposing pytest suites, given training and inference run on an Apple M4 Pro
with 48 GB unified memory?
**Method:** Three parallel research passes (Claude Sonnet agents with web
search), raw reports preserved unedited in `sources/`. This document is the
synthesis and carries the recommendation. Where the sources disagree, the
disagreement is stated, not resolved by preference.

| Source | Covers |
|---|---|
| [01-candidate-coder-models.md](sources/01-candidate-coder-models.md) | Model families 0.5B–9B: ids, licences, scores, MLX availability |
| [02-m4-pro-finetuning-feasibility.md](sources/02-m4-pro-finetuning-feasibility.md) | mlx-lm LoRA/QLoRA, memory and throughput, alternatives, cloud fallback |
| [03-test-generation-prior-art.md](sources/03-test-generation-prior-art.md) | Papers on test-generation fine-tunes, small-model baselines, contamination, mutation tooling |

---

## 1. What the evidence says

### Hardware is not the constraint below 8B

- LoRA and QLoRA on 0.5B–3B are comfortable on this machine. 7B–8B QLoRA fits
  in memory with large headroom (an M2 Max 32 GB peaked at ~7 GB on Mistral-7B
  LoRA). Training throughput on M4 Pro specifically is unbenchmarked; the
  memory-bandwidth extrapolation (273 GB/s, below an M2 Max) puts 7B LoRA in the
  low hundreds of tokens per second, so a run over ~1,000 examples × 3 epochs is
  tens of minutes to a couple of hours.
- Cloud fallback for that same workload is roughly $0.50–$4 on an A10G/A100
  (agent estimate, not a benchmark). Cost is not a reason to avoid 7B.
- mlx-lm is the right tool: native QLoRA when the base is quantised, chat-format
  JSONL, adapter fusion. PyTorch MPS has a documented silent-NaN bug; Unsloth
  has no official Apple Silicon support.
- Batched generation for hundreds of eval suites needs `mlx_parallm` or custom
  batching; the stock server does not batch.

### The candidate field is narrower than it looks

- **Qwen2.5-Coder** is the only family that combines a permissive licence, an
  instruct variant at every size, confirmed MLX conversions, and published
  scores. The **3B is non-commercial** (Qwen Research licence); 0.5B, 1.5B and
  7B are Apache 2.0.
- **Qwen3-Coder has no small dense model** (only MoE at 30B-A3B and above).
- Phi-4-mini (3.8B, MIT) and Ministral 3 (3B/8B, Apache 2.0) are plausible but
  lack verified test-relevant scores and confirmed MLX conversions.
- DeepSeek-Coder-V2-Lite is a 16B-total MoE; unproven on MLX for training.
- Everything else (CodeGemma, StarCoder2, Granite, SmolLM, Llama 3.2) loses on
  licence, on scores, or on being non-code-specialised.

### Prior art says small models may be too weak to teach judgment

This is the finding that changes the recommendation.

- On TestGenEval (real-repo test generation, prompting only) CodeLlama-7B
  scores a 0.5% mutation score, Llama-3.1-8B 6.8%, Gemma-9B 9.0%, GPT-4o 18.8%.
  A 7B-class model can be nearly non-functional at this task under prompting.
- **7B is the smallest size with an execution-verified fine-tune gain in the
  literature** (UTGen, Qwen2.5-7B SFT). Evidence at 1.3B–3B exists but is
  judged by another LLM, not by execution, and the authors say so.
- No published mutation-score baseline exists for Qwen2.5-Coder-1.5B or 3B at
  all. We would be the first measurement.
- The risk with a 1.5B base is specific: if its prompting baseline is near
  zero, the fine-tune "gain" mostly measures learning pytest syntax and file
  shape, not learning which cases expose bugs. That is a weaker demo claim.

### Other lessons that confirm or adjust existing decisions

- Execution-filtered training data (our D003) is what Meta's ACH pipeline does;
  only ~29% of generated mutants built and ~51% of those were non-equivalent.
- Mutation score over coverage (our metric) is the right call; the two diverge.
- UTGen's strongest lesson: small models fail on **assertion values** more than
  on test shape. Training examples must carry executed, correct expected
  outputs, not just plausible-looking tests. Our execution filter guarantees
  this only if we keep suites that pass on the reference.
- **Contamination is worse than assumed.** Qwen2.5-Coder's self-reported
  MBPP/HumanEval numbers have not reproduced independently, and no tool
  detects equivalent mutants. The held-out pool should follow the LiveCodeBench
  pattern: functions written after the base model's training cutoff. This
  affects decision D004 and is flagged for revisit.

---

## 2. Recommendation

**Do not pick the size from the literature. Measure it in the pilot.** Both
candidates are cheap to run and both have MLX 4-bit conversions.

| Candidate | Why it is in | Why it might lose |
|---|---|---|
| `Qwen2.5-Coder-1.5B-Instruct` | Fastest iteration; largest potential visible delta; runs anywhere in the demo | Prompting baseline may be near floor; fine-tune could learn format, not judgment |
| `Qwen2.5-Coder-7B-Instruct` | Smallest size with published execution-verified fine-tune gains; functional baseline; Apache 2.0 | Slower local training (unbenchmarked on M4 Pro); smaller visible delta; heavier demo |

**Decision rule, fixed before seeing numbers.** Run the 10-case pilot baseline
(zero-shot and few-shot) on both. Choose the **smallest** model whose prompting
baseline is *functional*: at least ~70% of generated suites valid and passing
on the reference, and a mutation score clearly above zero but clearly below the
teacher. If 1.5B meets that bar, take it. If it sits near the floor, take 7B.
Either way, the pilot numbers are recorded and become part of the write-up.

**Expected outcome given the evidence:** 7B. The literature gives no example of
a 1.5B model being functional at this task under prompting. Treat 1.5B passing
the bar as a pleasant surprise, not the plan.

**Excluded by rule:** Qwen2.5-Coder-3B (non-commercial licence), anything
without a confirmed MLX conversion (Phi-4-mini, Ministral 3, DeepSeek-V2-Lite)
unless a ten-minute check confirms one and there is slack.

---

## 3. What this research did not establish

- No M4 Pro training benchmark exists; the 7B timing is an extrapolation.
- No mutation-score baseline exists for any Qwen2.5-Coder size; the pilot fills
  this gap.
- The agents searched the web as of 2026-09-06 but a 2026 small coder release
  could have been missed. Before committing, spend ten minutes on Hugging Face
  filtering for code models released in 2026 under 9B with an MLX conversion.
- "Luna" (named as a possible teacher in D008) was not part of this research
  and remains unverified as a model id.
- Whether mlx-lm supports architectures newer than its documented list was not
  confirmed; Qwen2 architecture (which Qwen2.5-Coder uses) is documented as
  supported.

## 4. Consequences for the decision log

- **P1 → D010** once Jordi confirms the decision rule above.
- **D004 flagged for revisit**: HumanEval as held-out is contaminated for these
  models; a post-cutoff function pool is the stronger choice.
- **D003 strengthened**: keep only suites that pass on the reference *and* kill
  at least one mutant, per ACH and UTGen.
- **D005 confirmed**: no off-the-shelf tool handles equivalent mutants; a small
  custom operator set with manual spot-checks is what the papers do too.
