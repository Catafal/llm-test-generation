# llm-test-generation

Can a 4B model learn to write pytest suites that expose bugs, measured against
the same model prompted, on 315 real functions written after its training
cutoff? Nine days, one Mac, six fine-tuning runs and a control, every decision logged.

**Result, in order of what moved the metric.**

1. **The harness.** Filling expected values by executing the function: validity 0.44 to 0.72, no training, every arm.
2. **A paragraph of system prompt.** The base asked for short suites of exact-value asserts: +0.076 grounded score over zero-shot [+0.034, +0.118], the fastest arm in the series.
3. **12,000 teacher examples.** Fine-tuning on execution-verified GPT-4o data: +0.062 [+0.015, +0.106], and −0.014 against the prompt with an interval including zero. Same style, slower, nothing on top.
4. **The model's own outputs**, four ways: nothing that resolved, each measured down to its mechanism.

The fine-tune is not wrong, it is redundant. The production answer for this
task is the harness and the prompt, with no training. The rule the project leaves
behind: when the model lacks a capability the task needs, give the job to a
tool; when the gap is style, prompt first, and fine-tune only if the prompt
cannot reach it. Every number below is paired over the same 315 post-cutoff
functions, and the fine-tune rows were pre-registered before they ran. One base arm is
compared eight times across the series with no multiplicity correction;
corrected, the adapters' lower bounds would not clear zero and the control's
still would.

| arm (Qwen3.5-4B, test n = 315, grounded harness) | grounded validity | mutation score on grounded-valid | grounded score | Δ vs base zero-shot, 95% CI |
|---|---|---|---|---|
| base zero-shot | 0.717 | 0.842 | 0.604 | — |
| base few-shot | 0.702 | 0.854 | 0.599 | −0.005 [−0.051, +0.041] |
| SFT, 388 self-generated | 0.743 | 0.846 | 0.629 | +0.024 [−0.021, +0.070] |
| DPO, 645 own grounded pairs | 0.740 | 0.850 | 0.629 | +0.024 [−0.013, +0.064] |
| SFT, 4,000 KodCode | 0.803 | 0.826 | 0.664 | +0.059 [+0.013, +0.105] |
| SFT, 12,000 KodCode | 0.822 | 0.811 | 0.667 | +0.062 [+0.015, +0.106] |
| SFT, 12,000 KodCode, under the style prompt | 0.867 | 0.769 | 0.667 | +0.062 [+0.016, +0.109] |
| **base, prompted for the style (control)** | 0.860 | 0.791 | **0.680** | **+0.076 [+0.034, +0.118]** |

Adapters against the control: 12k −0.014 [−0.051, +0.022], 4k −0.017
[−0.057, +0.022]. The 12k adapter *with* the style prompt: −0.014 [−0.044,
+0.019] against the control and 0.000 against itself under the plain
prompt; the adapter and the prompt teach the same thing and do not stack.
The control also lifts unaided validity, 0.571 vs 0.438, which no fine-tune
did.

![The prompt for the teacher's style beat the fine-tune](docs/charts/results.png)

## How it works

```
function (post-cutoff GitHub)  →  model writes a pytest module (2,048 tokens, ≤ 8 tests, greedy)
                               →  harness fills `assert f(x) == <literal>` from execution
                               →  run on the correct function (grounded validity)
                               →  run on every live mutant (mutation score)
grounded score = mutation score, 0 if the suite still fails · paired bootstrap over all 315 functions
```

Mutants are AST edits (comparisons, constants, boundaries, arithmetic);
bytecode-identical ones are dropped; one category is never used in training
curation so operator overfitting has a probe. Held-out functions are split by
family and decontaminated in three layers.

## Read more

- **The series:** [`docs/the-harness/`](docs/the-harness/index.md), five
  experiments and a closing note, in order, with the reasoning at each step.
- **Model card and datasheets:** [`docs/model-card.md`](docs/model-card.md).
  Adapters on the Hub: [`jorcagra/qwen3.5-4b-testgen-lora-ext12k`](https://huggingface.co/jorcagra/qwen3.5-4b-testgen-lora-ext12k)
  and [`-ext4k`](https://huggingface.co/jorcagra/qwen3.5-4b-testgen-lora-ext4k) (CC BY-NC 4.0, inherited from KodCode).
- **Demo:** [`docs/demo.md`](docs/demo.md), `make demo`, base vs fine-tune on one held-out function, terminal UI.
- **Results document with every number:** [`docs/results/weekend-2.md`](docs/results/weekend-2.md).
- **Charts:** [`docs/charts/`](docs/charts/README.md), generated from the run artifacts by `make charts`.
- **Research trail:** [`docs/research/`](docs/research/), one folder per decision.

## Run it

```
make setup && make test                         # harness unit tests (82)
make sync-models && make models-pull KEYS="4b-bf16"
make adapters-pull                              # both adapters from the Hub
make demo ID="NanmiCoder/open-image-prompts:retrieval/engine.py::weighted_tag_similarity" FROM_RUNS=1
make eval ADAPTER=models/adapters/lora-4b-ext12k/ckpt-0002400 GROUNDED=1
uv run python -m testgen.stats --grounded runs/<run>/outputs.jsonl 4b/zero 4b/few
```

Generation needs Apple Silicon (mlx). The harness, statistics, charts and the
replay demo run anywhere. Every run writes a manifest (git sha, model,
budget, results) and the manifests are committed; raw generations are not,
except the eight test-split runs behind the table above.

## Layout

```
testgen/harness/   sandboxed pytest runs; scoring; run manifests
testgen/mutate/    AST mutation operators; equivalence filter; profiles
testgen/data/      GitHub harvest, purity filter, decontamination, splits
testgen/generate/  prompts and mlx-lm generation under a fixed budget
testgen/train/     oracle fill, self-sampling, filters, pairs, curation, training wrappers
testgen/stats.py   paired bootstrap and exact McNemar
tests/             pytest for the harness, oracle, filters and statistics
data/              held-out pool, training pools, curated sets (manifests, notices, decontamination reports)
runs/              run manifests (committed), raw outputs (gitignored)
models/            weights and adapters, inside the repo, gitignored
docs/              series, model card, demo, charts, results, research
```

Everything large lives inside the repo and is gitignored: `make models-rm
ALL=1` and `make clean-harvest` remove it.
