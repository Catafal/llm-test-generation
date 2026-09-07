# Fine-Tuning Principles, Stack and Rigor for llm-test-generation

**Date:** 2026-09-07
**Question:** Before writing a plan, what first principles of ML and fine-tuning
should govern this project, which stack should run it, and what rigor makes the
result survive review by an applied-AI or evals engineer at a frontier lab?
**Method:** Three parallel research passes (Claude Sonnet agents with web
search); raw reports unedited in `sources/`. This document is the synthesis.

| Source | Covers |
|---|---|
| [01-sft-lora-methodology.md](sources/01-sft-lora-methodology.md) | LoRA/QLoRA mechanics, data quality, process discipline, starting hyperparameters |
| [02-tooling-stack-2026.md](sources/02-tooling-stack-2026.md) | Unsloth, mlx-lm, TRL/PEFT, others; Qwen3.5-specific risks; cloud cost |
| [03-experimental-rigor.md](sources/03-experimental-rigor.md) | Decontamination, significance, equal budgets, metric gaming, reporting |

---

## 1. Unsloth or the typical approach

Not a real choice on this machine. Unsloth has no first-party Apple Silicon
support; the Mac path is an unofficial wrapper. The local tool is mlx-lm (or
its community superset mlx-lm-lora). Unsloth matters only on a rented NVIDIA
GPU, where it is roughly 1.5x faster than TRL/PEFT at half the VRAM.

Two Qwen3.5-specific findings change the shape of the plan:

1. **Unsloth's own docs say do not QLoRA Qwen3.5**, dense or MoE, because of
   "higher than normal quantization differences." So the 4-bit training route
   is out. bf16 LoRA on the 9B needs about 22 GB, which fits the 48 GB Mac.
2. **mlx-lm's Qwen3.5 support is months old and has had real correctness
   bugs**: a Gated DeltaNet state-dtype kernel bug (bf16 written where fp32 is
   needed) and a documented LoRA-fusion divergence on Qwen3.5-4B (issue #1058).
   The tooling report recommends cloud-first for the real training run, with
   the Mac used for pipeline iteration.

The honest reading: the Mac can still be the primary path, but only if the
first local run passes a fidelity check (adapter output on the Mac matches a
reference forward pass, and the tiny-batch overfit reaches near-zero loss).
If it does not, a bf16 LoRA run on a rented 24 GB-class GPU with TRL/PEFT
costs roughly $5–20 and an afternoon. That spend needs Jordi's approval and is
recorded as a pending decision, not assumed.

## 2. The first principles, merged from the three reports

Grouped by what they govern. Each carries its source.

**Process discipline**
1. Simplest working baseline first; then change exactly one thing per run
   (Google Tuning Playbook).
2. Read the data by hand before training on it; then overfit a tiny batch to
   near-zero loss to prove the pipeline (Karpathy).
3. Config-as-code with fixed seeds, pinned versions, a run manifest per run,
   and adapters tagged with dataset hash and base revision (tooling report).

**Data**
4. Quality and diversity over volume past a few hundred examples; SFT teaches
   behaviour, pretraining supplies capability (LIMA).
5. Filter generated training suites by execution: run, pass on the reference,
   kill at least one mutant. Llama 3 and Qwen2.5-Coder do the same with
   rejection sampling at K=10–30 candidates per prompt.
6. Train on the (input, expected-output) pair; small models fail on assertion
   values more than on test shape (UTGen, from the earlier research).

**Training**
7. LoRA on all linear layers, not attention only; above rank 8 the coverage
   decision matters more than the rank (QLoRA paper, Databricks).
8. Alpha = 2× rank and dropout 0.05 as starting points to perturb, not laws
   (Raschka).
9. One to two epochs on a static ~1k set; more risks overfitting (Raschka).
10. Loss on the completion only; same chat template at train and inference;
    thinking off in both (TRL docs; D010).
11. bf16 LoRA, not QLoRA, for Qwen3.5 (Unsloth docs).
12. LoRA learns less and forgets less than full fine-tuning; expect a bounded
    gain and measure the forgetting (Biderman et al.).
13. Validation loss is a weak signal for a generation task; the harness score
    on a held-out slice is the early-stopping signal (methodology report).

**Evaluation and rigor**
14. Held-out pool by date cutoff after the base model's training cutoff, split
    by function family, decontaminated with n-gram plus embedding plus AST
    checks, and documented as a decontamination report (LiveCodeBench, ACL
    2022 dedup, contamination surveys).
15. Greedy decoding for the headline number; a few low-temperature seeds for
    variance (Codex pass@k methodology).
16. Paired bootstrap confidence intervals on the difference between
    conditions; McNemar on paired mutant kills; say plainly that a few-point
    effect is not resolvable at 100–300 functions.
17. Equal budget means equal max tokens, decoding settings, and test count per
    function; report cost and latency as separate rows.
18. Filter equivalent mutants by bytecode comparison; track validity rate and
    false-failure rate as guardrails; hand-inspect a stratified sample of
    20–30 suites for hard-coded or vacuous assertions (ISSTA 2024, EvalPlus).
19. HumanEval+ before and after, with the same CI machinery, to make the
    forgetting cost visible.
20. Hold out one mutation-operator category from any curation signal to check
    for operator-set overfitting.
21. Model card, held-out datasheet, run manifest, and a first-class
    limitations section including negative results.

## 3. Starting hyperparameters (to be perturbed one at a time)

| Parameter | Start | Why |
|---|---|---|
| Precision | bf16 LoRA | Qwen3.5 quantisation sensitivity; QLoRA excluded |
| Rank / alpha | 16 / 32 | Mid-range; coverage matters more than rank |
| Targets | all linear layers, including the Gated DeltaNet projections | Attention-only underperforms; DeltaNet layers are 3 of every 4 sublayers |
| Dropout | 0.05 | Raschka default |
| LR | 1e-4, cosine, warmup | Practitioner range, not a cited constant |
| Epochs | 2, early stop on harness score | Static small set |
| Effective batch | 8–16 via accumulation | Stability at this scale |
| Max seq len | 2,048 | Covers function + suite without truncation |
| Loss | completion only | TRL default; `--mask-prompt` in mlx-lm |
| Packing | off | Interaction with masking undocumented |
| Seeds | 1 fixed, rerun once | Raschka found low variance; confirm |

The Gated DeltaNet module names (`linear_attn.in_proj_qkv`, `in_proj_z`,
`out_proj`) come from third-party writeups and must be checked against the
loaded model's `named_modules()` before any config is committed.

## 4. Consequences for existing decisions

- **D010 addendum:** bf16 LoRA, not QLoRA. Local fidelity check before
  trusting a Mac run. Cloud fallback is a pending spend decision (P5).
- **D004 resolution proposed:** replace HumanEval as held-out with a
  post-cutoff pool (functions authored after Qwen3.5's training cutoff),
  split by family, three-layer decontamination. MBPP stays as the training
  function pool only if it survives the same decontamination against the
  held-out set.
- **D005 amendment:** keep the small custom AST operator set for
  auditability, but add bytecode-equivalence filtering, report residual
  counts, and hold one operator category out of the curation signal. The
  rigor report prefers an established tool's operator set for comparability;
  we accept lower comparability for higher auditability and say so.
- **D003 strengthened:** the execution filter keeps suites that pass on the
  reference and kill at least one mutant, sampled at K candidates per
  function.
- **Ablations in budget:** rank 8/16/32, few-shot 0/2/5 at matched budget,
  greedy vs sampled, held-out operator category. Skipped: full fine-tune,
  size sweep, n=200 pass@k, cross-language.

## 5. What this research did not settle

- Exact mlx-lm LoRA key names for Qwen3.5's DeltaNet layers.
- Whether the Gated DeltaNet dtype bug affects mlx-lm's training path or only
  the Ollama runner.
- Whether mlx-lm-lora handles the vision weights or needs them stripped, as
  the 4B precedent did.
- Any field-agreed acceptable false-failure rate; we will set one and justify it.
- The 1e-4 learning rate is a prior, not a citation.
