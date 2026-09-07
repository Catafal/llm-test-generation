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

### Correction and 2026 follow-up check (same day)

The source-01 figure of ~43% HumanEval for Qwen2.5-Coder-1.5B-Instruct is
wrong. The technical report's Table 16 ([arXiv 2409.12186](https://arxiv.org/html/2409.12186v3))
gives:

| Model (Instruct) | HumanEval | HumanEval+ | MBPP | MBPP+ | BigCodeBench full / hard | LiveCodeBench (2407–2409) |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | 70.7 | 66.5 | 69.2 | 59.4 | 32.5 / 6.8 | 6.1 |
| Qwen2.5-Coder-7B | 88.4 | 84.1 | 83.5 | 71.7 | 41.0 / 18.2 | 37.6 |

So the 1.5B is functional on function-level tasks, but the LiveCodeBench and
BigCodeBench-hard gaps show it reasons far less. The pilot decides.

The agents' "as of search date" coverage missed several 2026 releases. Checked
against primary model cards on 2026-09-06:

| Model | Released | Arch | Licence | Thinking mode | Code scores | MLX |
|---|---|---|---|---|---|---|
| [IBM Granite-4.1-8B](https://huggingface.co/ibm-granite/granite-4.1-8b) | 2026-04-29 | dense transformer, 40 layers | Apache 2.0 | none | HumanEval+ 79.9, MBPP+ 73.8, EvalPlus avg 80.2, BigCodeBench 35.0 | `mlx-community/granite-4.1-8b-4bit` exists; `granite.py` in mlx-lm |
| [IBM Granite-4.2-8B](https://huggingface.co/ibm-granite/granite-4.2-8b) | 2026-08-25 | dense transformer, reasoning | Apache 2.0 | yes (reasoning model) | LiveCodeBench v6 73.2, SWE-bench Verified 47.7; no EvalPlus published | not checked |
| [Google Gemma 4 E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | 2026-07-02 | 4.5B effective / 8B total | Apache 2.0 | yes, toggled by token | LiveCodeBench v6 52.0; no HumanEval/MBPP | `gemma4.py` in mlx-lm |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 2026-02 | hybrid Gated-DeltaNet + sparse MoE, multimodal | Apache 2.0 | yes | LiveCodeBench v6 65.6; no EvalPlus | `qwen3_5.py` in mlx-lm; training on hybrid arch unproven |
| Qwen3-Coder small dense | — | none exists (only 30B-A3B MoE and up) | — | — | — | — |

mlx-lm's LoRA tuner attaches adapters by layer type (`nn.Linear`,
`QuantizedLinear`), not by model name, so any of these loads; training
stability on hybrid or multimodal stacks is the unverified part.

**How this changes the picture.** Granite-4.1-8B is the one serious new
alternative: Apache 2.0, dense, no thinking mode, EvalPlus within a few points
of Qwen2.5-Coder-7B (79.9 vs 84.1 HumanEval+; 73.8 vs 71.7 MBPP+), MLX
conversion ready. It is a general model, not code-specialised, and its
2026 cutoff makes a post-cutoff held-out pool harder to assemble. The thinking
models (Granite 4.2, Gemma 4, Qwen3.5) are excluded for this project: reasoning
tokens consume the fixed generation budget and complicate SFT formatting,
which adds confounds to a comparison that must stay clean.

### Qwen3.5 as a base to specialise into a test writer (2026-09-07)

Jordi asked whether a general Qwen3.5 model could be turned into a
test-writing coder instead of starting from Qwen2.5-Coder. Checked all four
cards plus the [collection](https://huggingface.co/collections/Qwen/qwen35).

**What the cards publish.** Only LiveCodeBench v6 and OJBench, and only for
the instruct models. Neither Base card publishes any coding benchmark. The
instruct numbers are reported with thinking enabled; no non-thinking numbers
are published.

| Model | Params | Arch | Thinking | LiveCodeBench v6 | OJBench | HumanEval+ / MBPP+ | MLX conversion |
|---|---|---|---|---|---|---|---|
| [Qwen2.5-Coder-7B-Instruct](https://arxiv.org/html/2409.12186v3) | 7B | dense transformer | none | 18.9 (secondary source, [IQuest-Coder report](https://arxiv.org/pdf/2603.16733)) | — | 84.1 / 71.7 | yes |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9B | hybrid: 3× Gated DeltaNet + 1× gated attention per block, vision encoder | on by default, `enable_thinking=False` supported | 65.6 | 29.2 | not published | `mlx-community/Qwen3.5-9B-OptiQ-4bit` |
| [Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base) | 9B | same | n/a | not published | — | not published | not checked |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 4B | same, hidden 2560 | on by default, disable supported | 55.8 | 24.1 | not published | `mlx-community/Qwen3.5-4B-4bit`, `-MLX-4bit`, `-bf16` and more |
| [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | 4B | same | n/a | not published | — | not published | not checked |

All Qwen3.5 models are Apache 2.0 and multimodal (image-text-to-text). No
coder-specific Qwen3.5 variant exists.

**Reading the numbers.** LiveCodeBench is competitive programming and rewards
long reasoning; the Qwen3.5 scores are with thinking on. With thinking off,
which our fixed-budget comparison requires, the gap to Qwen2.5-Coder-7B is
unknown and certainly smaller. HumanEval+/MBPP+ is the benchmark closest to
"write correct code for one function", and Qwen3.5 publishes nothing there.
So the honest statement is: Qwen3.5-9B is probably the stronger coder, but the
evidence for the regime we would run it in does not exist yet.

**Feasibility on a Mac, verified precedent.** [sciences44/mlx-lora-finetune](https://github.com/sciences44/mlx-lora-finetune)
LoRA-tuned Qwen3.5-0.8B, 2B and 4B on an M1 64 GB with mlx-lm for text-to-SQL:
8–15 min per run, peak memory 3.9–11.1 GB, 115–475 tok/s. The 4B needed the
vision weights stripped manually. Their 2B beat their 4B after tuning, and
their eval was string-matching, not execution. Nobody has published a 9B run.
mlx-lm's LoRA attaches to `q/k/v/o_proj` and MLP projections; in Qwen3.5 the
attention projections exist in only one of every four sublayers, so default
adapters cover less of the network than on a dense transformer. Workable, but
it is a config decision that must be recorded, not a default.

**Base or Instruct.** Instruct, with thinking disabled for every condition.
A Base model has no usable prompting baseline, so "fine-tune beats prompting"
becomes trivially true and proves nothing. Prior work (UTGen) fine-tunes from
instruct models. Base is the fallback only if the instruct's non-thinking
behaviour turns out broken.

**Does "general model → test specialist" make a better story?** Somewhat. It
shows specialising a general 2026 model rather than nudging a coder model.
But 500–1,000 SFT examples do not teach coding; they teach test-writing
behaviour on top of coding ability the base already has. The story holds
only if Qwen3.5's non-thinking coding ability is at least on par with
Qwen2.5-Coder-7B, which is exactly the unmeasured thing.

**Trade-offs specific to Qwen3.5.**
- Contamination: February 2026 cutoff leaves a small post-cutoff function
  pool for held-out evaluation. Qwen2.5-Coder's 2024 cutoff leaves a large one.
- Multimodal weights: dead weight for this task, and the 4B needed a manual
  strip to train. Memory and load time cost, no benefit.
- Architecture risk: hybrid Gated DeltaNet is newer in mlx-lm; a training
  bug costs a weekend. Dense Qwen2.5-Coder has none of this risk.
- Narrative gain: a 2026 general model specialised into a test writer reads
  better in an application than a 2024 coder model nudged.

**Revised pilot set.** Replace the 1.5B slot with Qwen3.5-4B, which fills the
"small and fast" role with a working Mac precedent and a 2026 story:

1. `Qwen2.5-Coder-7B-Instruct` — control; known-good, dense, non-thinking.
2. `Qwen3.5-4B` (thinking off) — small, fast, 2026, precedent exists.
3. `Qwen3.5-9B` (thinking off) — strongest candidate on paper, untested on Mac.

Same decision rule: the smallest model whose prompting baseline is functional
on the pilot's mutation score. If Qwen3.5-9B's non-thinking baseline is as
good as Qwen2.5-Coder-7B's, take it for the narrative; if it is clearly worse
or training misbehaves, the control wins. Either way the pilot numbers are the
first published non-thinking test-generation baselines for Qwen3.5 and belong
in the write-up.

---

## 2. Recommendation

**Do not pick the size from the literature. Measure it in the pilot.** Both
candidates are cheap to run and both have MLX 4-bit conversions.

| Candidate | Why it is in | Why it might lose |
|---|---|---|
| `Qwen2.5-Coder-1.5B-Instruct` | Fastest iteration; largest potential visible delta; runs anywhere in the demo; HumanEval+ 66.5 says it is functional | LiveCodeBench 6.1 says it reasons little; fine-tune could learn format, not judgment |
| `Qwen2.5-Coder-7B-Instruct` | Best EvalPlus at size (84.1 / 71.7); code-specialised; non-thinking; smallest size with published execution-verified fine-tune gains; 2024 cutoff makes a post-cutoff held-out pool easy | Slower local training (unbenchmarked on M4 Pro); 2024 model, an interviewer may ask why not a 2026 one |
| `Granite-4.1-8B` (alternative) | 2026 release; Apache 2.0; dense, non-thinking; EvalPlus 80.2; MLX 4-bit ready | General model, not code-specialised; 2026 cutoff shrinks the post-cutoff pool; no published test-generation fine-tune evidence |

**Decision rule, fixed before seeing numbers.** Run the 10-case pilot baseline
(zero-shot and few-shot) on both. Choose the **smallest** model whose prompting
baseline is *functional*: at least ~70% of generated suites valid and passing
on the reference, and a mutation score clearly above zero but clearly below the
teacher. If 1.5B meets that bar, take it. If it sits near the floor, take 7B.
Either way, the pilot numbers are recorded and become part of the write-up.

**Expected outcome given the evidence:** 7B. The literature gives no example of
a 1.5B model being functional at this task under prompting. Treat 1.5B passing
the bar as a pleasant surprise, not the plan.

**Primary candidates for the pilot: superseded by the revised pilot set in the Qwen3.5 section above (Qwen2.5-Coder-7B control, Qwen3.5-4B, Qwen3.5-9B).** Original note: Add Granite-4.1-8B
to the pilot only if the 7B Qwen baseline is unexpectedly poor or if a 2026
base matters for the application narrative; baseline generation is cheap.

**Excluded by rule:** Qwen2.5-Coder-3B (non-commercial licence); thinking
models (Granite 4.2, Gemma 4, Qwen3.5) for budget and formatting confounds;
anything without a confirmed MLX conversion (Phi-4-mini, Ministral 3,
DeepSeek-V2-Lite).

---

## 3. What this research did not establish

- No M4 Pro training benchmark exists; the 7B timing is an extrapolation.
- No mutation-score baseline exists for any Qwen2.5-Coder size; the pilot fills
  this gap.
- The 2026 follow-up check above covers Granite 4.1/4.2, Gemma 4 and Qwen3.5.
  Training stability of mlx-lm LoRA on the hybrid/multimodal stacks was not
  tested; Granite 4.1 is a standard dense transformer and carries no such risk.
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
