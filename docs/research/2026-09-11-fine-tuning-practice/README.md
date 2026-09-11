# Did we hit the model's ceiling, or miss a fundamental? (2026-09-11)

Three Sonnet research passes after D031, prompted by Jordi's question:
"do you really think we got maximum model capacity, or is some LLM
fundamental we're missing?" Reports: `01-how-labs-fine-tune.md` (NVIDIA,
Unsloth, HF, Meta, Qwen, AI2 practice), `02-data-scale-and-small-models.md`
(evidence on data scale, self-improvement, distillation, statistical
power), `03-recipe-audit.md` (line-by-line audit of our recipe against
mlx-lm source and current LoRA/DPO guidance).

## Verdict

**No, we did not hit the model's ceiling. We hit the ceiling of
self-generated data at a few hundred examples, which is a regime no
published recipe uses.** The recipe itself is not broken: the audit
verified LoRA scale semantics, all-linear targets, the learning-rate
schedule and optimizer-step accounting, completion-only loss masking,
chat-template consistency and checkpoint selection against mlx-lm's source
and found nothing that would explain a null. Two things we did are the
fundamental omissions, and both are choices we made on purpose in D023.

| What every working recipe has | What we did | Evidence |
|---|---|---|
| Data from a stronger model or a human-trained reward model, at 10^4–10^6 examples | 388–645 examples, all from the student itself | Qwen2.5-Coder 200K synthetic pairs; Llama 3 2.7M synthetic code examples filtered by a reward model; Tülu 3 939K prompts; Nemotron Nano 2 ~11.5M SFT samples; Magicoder 75K (the smallest teacher set with a large gain at 7B) |
| A self-improvement floor the model must already clear | 4B, on a task whose bottleneck is output prediction | "Mind the Gap" (ICLR 2025): the generation-verification gap that bounds self-improvement scales with pre-training compute and is near zero for small models; ReST-EM gains "scale favourably with model size"; STaR could not bootstrap GPT-2 |
| Online RL with a verifiable reward when a checker exists | Offline DPO on fixed pairs | Skopin & Kotelnikov 2026 (arXiv 2605.30478): Qwen3-0.6B and Llama3.2-1B, LoRA GRPO on 374 MBPP tasks, unit-test reward, up to +13 pp pass@1; Qwen3 credits large rollout counts; `mlx-lm-lora` ships GRPO |

Smaller findings from the audit, none of which changes the verdict:
DPO had no SFT warm start on the chosen responses and only 108 of 645
grounded pairs were near-miss contrasts (the rest oppose good suites to
broken ones), a known way to get "training fits, held-out flat"; the
`mlx-lm-lora` DPO loss does not mask the prompt, which cancels exactly
under the sigmoid loss we used but would bite IPO; DPO rank 8 is below
Unsloth's default band of 16–32; the D024 last-16-layer restriction on
the hybrid base was never tested in isolation.

## Should we scale data?

Scaling the *same* self-generated recipe: **not worth it** (agent estimate
10–15% for a resolved ≥ +0.05). Our own four iterations are a flat
dose-response from 388 to 645 examples, and the self-improvement
literature says the verification floor, not volume, is what binds at 4B.

Scaling with a **teacher**: the best-evidenced path (45–55%). A 30B-class
coder in 4-bit fits the 48 GB Mac. The cost is sampling ten thousand
suites at 30B speed, roughly a weekend of machine time, plus the D023
story change ("no external teacher") which the write-up would state.

**RL with the execution reward** (20–30%): the reward is computed by the
harness, not by the model, so it sidesteps the self-verification bound;
one directly analogous small-model result exists; runnable via
`mlx-lm-lora` GRPO but unmeasured on Apple Silicon at our sequence
lengths, and the hybrid DeltaNet training path (12–40 tok/s) makes
rollouts expensive. Would run on the dense base.

**Base model**: a code-pretrained base (Qwen2.5-Coder-3B/7B) or any base
with strong output prediction (CRUXEval-O ≥ 70) is the lever that touches
the actual bottleneck; untested here, and the earlier 9B-vs-4B null says
size alone within one family is not it.

## What this means for the write-up and the blog

The honest sentence is: "Four pre-registered iterations show that a 4B
model cannot be self-improved on this task with a few hundred of its own
examples; the harness that grounds expected values in execution is worth
28 validity points; the two levers the field actually uses, a stronger
teacher at 10^4 examples and online RL with the execution reward, were out
of scope by design and are priced here." That is a stronger story than a
+2 point fine-tune, and it is true.

## Caveats on the research itself

Agent 2 ran out of search budget before verifying CodeDPO, PLUM, V-STaR
and Llama 3's rejection-sampling scale first-hand; the RLVR paper's
hardware claim (single RTX 3090) comes from its body, which I did not
re-verify; "Mind the Gap" per-model numbers are the agent's reading of the
full text, the abstract states only the scaling law. Probabilities are
analogies, no paper runs this exact task at this scale.
