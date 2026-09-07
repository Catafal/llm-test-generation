# Source report 01 — SFT/LoRA methodology first principles

Research agent report (Claude Sonnet), 2026-09-07. Unedited except for this header.
Claims marked unverified by the agent remain unverified.

---

## First principles for SFT/LoRA of a 4B-9B instruct model on a narrow task (pytest generation, 500-2,000 examples)

### 1. Numbered first principles (each cited)

1. **Target all linear layers, not just attention.** LoRA on q/v attention alone underperforms; applying LoRA to all linear transformer-block layers (attention + MLP + projections) is required to approach full-finetuning quality, and above r=8 the rank itself matters less than this coverage decision. [QLoRA paper](https://arxiv.org/abs/2305.14314), [Databricks LoRA guide](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms)
2. **Alpha = 2x rank is the default starting heuristic, not a law.** Raschka's own sweeps found r=256/alpha=128 (0.5x) beat the 2x rule on one dataset, so treat 2x as a starting point to perturb, not a fixed setting. [Raschka, "Practical Tips for Finetuning LLMs Using LoRA"](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
3. **LoRA dropout of 0.05 is a reasonable fixed default**; Raschka used it throughout his experiments without tuning it further, flagging it as still an open question. [Raschka, same article](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
4. **More than 1-2 epochs on a static dataset risks quality decline from overfitting.** Raschka observed doubling iterations over a fixed 50k-example set hurt performance; for instruction tuning, multi-epoch training is not reliably beneficial. [Raschka, same article](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
5. **QLoRA's quality cost is near zero, but it isn't free** — expect roughly 33% memory savings paired with roughly 39% longer wall-clock training in Raschka's controlled comparison. [Raschka, same article](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
6. **Seed-to-seed variance in LoRA runs is small enough to trust single-run comparisons** in Raschka's tests, though this should still be sanity-checked for a new task/dataset rather than assumed. [Raschka, same article](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
7. **LoRA reliably underperforms full fine-tuning on the target skill, especially in low-rank/low-lr regimes, in exchange for better retention of general ability.** This is the central learning-vs-forgetting tradeoff to expect for a code-generation LoRA. [Biderman et al., "LoRA Learns Less and Forgets Less" (TMLR 2024)](https://arxiv.org/abs/2405.09673)
8. **DoRA (weight-decomposed LoRA) can match or beat LoRA at the same or even half the rank**, and is less sensitive to rank choice — worth trying if the plain-LoRA run underperforms. [Raschka, "Improving LoRA... DoRA from Scratch"](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch); [NVlabs/DoRA](https://github.com/NVlabs/DoRA)
9. **rsLoRA's rank-stabilized scaling is needed to actually benefit from higher ranks** — vanilla LoRA scaling causes runs at different ranks to converge to similar loss, masking rank's effect; rsLoRA removes that ceiling. [Kalajdzievski, "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA"](https://arxiv.org/abs/2312.03732)
10. **Data quality and diversity beat raw volume once you're past a few hundred examples** — LIMA showed 1,000 carefully curated, diverse prompt/response pairs (no RL) produced outputs judged equivalent-or-better than GPT-4 in 43% of cases, evidence that most capability comes from pretraining and SFT mainly teaches format/behavior. [Zhou et al., "LIMA: Less Is More for Alignment"](https://arxiv.org/abs/2305.11206)
11. **Filter synthetic/generated training examples by execution correctness before adding them to the SFT set** — Qwen2.5-Coder and Llama 3 both use sandboxed execution or reward-model rejection sampling (Llama 3 samples K=10-30 candidates per prompt and keeps the best) to raise SFT data quality rather than relying on unfiltered generation. [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186); [Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
12. **Establish a simple, fast, low-resource baseline first, then change exactly one thing per experiment** — otherwise you cannot attribute a result to its cause. [Google, Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)
13. **Before scaling up, look at the raw data by hand and overfit a single tiny batch to near-zero loss** to confirm the data pipeline, tokenization, and loss are wired correctly — a training-loss failure here is a pipeline bug, not a modeling problem. [Karpathy, "A Recipe for Training Neural Networks"](http://karpathy.github.io/2019/04/25/recipe/)
14. **Mask the prompt out of the loss (train on completions/assistant turns only)** — TRL's SFTTrainer defaults to completion-only loss for prompt-completion data, and offers `assistant_only_loss=True` for chat-template data so system/user tokens don't get gradient signal; getting this wrong (training on the whole sequence) silently degrades instruction-following. [HF TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
15. **Validation loss on a generation task is a weak proxy for what you actually care about** — for a narrow code-gen task, track an execution-based metric (do the generated tests actually run and pass against the target function) alongside loss, since perplexity/loss can improve while functional correctness does not. (Synthesized from the rejection-sampling literature above and general SFT practice; no single canonical citation — flagged as principle, not a direct quote.)

### 2. Concrete starting hyperparameter table — 9B QLoRA, ~1,000 examples, ~1,500 tokens each

| Parameter | Starting value | Rationale |
|---|---|---|
| Quantization | 4-bit NF4 (QLoRA) | ~33% memory savings, "barely affected" quality per Raschka's controlled test |
| LoRA rank (r) | 16 | Mid-range; above r=8 rank matters less than module coverage once all-linear is used |
| LoRA alpha | 32 (2x rank) | Standard heuristic; treat as first thing to perturb if results are weak |
| Target modules | All linear layers (attention q/k/v/o + MLP gate/up/down), not attention-only | Required to approach full-finetune quality |
| LoRA dropout | 0.05 | Raschka's fixed default |
| Learning rate | 1e-4 to 2e-4 (AdamW) | Typical QLoRA range for this rank/dataset size; cosine schedule with warmup |
| Optimizer | AdamW (or paged AdamW for QLoRA) | Recommended over SGD unless using cosine annealing with SGD |
| Epochs | 2-3, with early stopping on held-out execution-pass-rate | More risks overfitting on a static ~1k-example set |
| Batch size (effective) | 8-16 via gradient accumulation | Balance noise/regularization vs. stability at this data scale |
| Max sequence length | ~2,048 | Avoid truncating function+test pairs |
| Loss masking | Completion/assistant-only | TRL SFTTrainer default / `assistant_only_loss=True` |
| Packing | Off, or on only if sequences are short and homogeneous | Packing can interact badly with completion-only masking |
| Eval metric | Execution pass rate of generated tests + held-out loss | Loss alone is a weak proxy |
| Seed | Fix one, but rerun once to confirm result isn't seed noise | Raschka found low variance, but confirm on your own setup |

### 3. Process principles to run alongside the hyperparameters
- Baseline first (smallest working config), change one variable per iteration, don't stack changes ([Tuning Playbook](https://github.com/google-research/tuning_playbook)).
- Before any real run: manually read 20-30 training examples, then overfit a batch of 8-16 examples to near-zero loss to confirm the pipeline ([Karpathy](http://karpathy.github.io/2019/04/25/recipe/)).
- Watch for catastrophic forgetting on general instruction-following/coding ability outside the narrow task — LoRA forgets less than full fine-tuning but is not immune ([Biderman et al.](https://arxiv.org/abs/2405.09673)).
- If plain LoRA underperforms at low rank, try DoRA or rsLoRA rather than just raising rank ([rsLoRA](https://arxiv.org/abs/2312.03732); [DoRA](https://github.com/NVlabs/DoRA)).

### 4. Claims the agent could not verify or fully cite
- **Qwen3.5 (Gated DeltaNet hybrid) LoRA target modules**: third-party guidance names `linear_attn.in_proj_qkv`, `linear_attn.in_proj_z`, `linear_attn.out_proj` for the linear-attention blocks and `mlp.experts.gate_up_proj`/`down_proj` for MoE experts, but not confirmed in an official Qwen report or HF Transformers source — check the actual `named_modules()` output before committing to a config. Sources: [Labonne, "Qwen3.5: Nobody Agrees on Attention Anymore"](https://huggingface.co/blog/mlabonne/qwen35); [Axolotl Qwen 3.5 docs](https://docs.axolotl.ai/docs/models/qwen3.5.html)
- **LoRA+**: not independently verified beyond the paper's existence.
- **Exact learning-rate values for a 9B QLoRA run**: 1e-4–2e-4 is a practitioner range, not a citation-backed constant.
- **Packing + completion-only-loss interaction**: no explicit statement found on how they interact; test empirically.
