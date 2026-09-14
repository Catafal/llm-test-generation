# 90-second demo

One held-out function. The base model and the fine-tuned model each write a
pytest suite. The harness runs both against the correct function, fills the
expected values by execution, then runs them against every mutant. Kill
counts side by side, and one mutant only the fine-tuned suite catches.

```bash
make demo PICK=1                                   # functions where the fine-tune catches more
make demo ID="NanmiCoder/open-image-prompts:retrieval/engine.py::weighted_tag_similarity"
make demo ID="…" FROM_RUNS=1                       # replay the scored generations (no model load)
```

Live generation loads the base (~30 s), generates greedily under the
evaluation budget, then loads the adapter and generates again. Total about
90 s on the M4 Pro. The `FROM_RUNS=1` replay takes ten seconds and prints
the exact generations the reported numbers came from. **Record the replay**:
live greedy decoding at batch size 1 is not bit-identical to the scored
batch-of-8 runs, so a live suite can differ (in one live run the base wrote
`== 1.0 / 3.0`, an expression the harness does not fill, and stayed invalid;
the fine-tuned suite was identical to the scored one). Live is the "it
really runs" proof; the replay is the evidence.

## Narration (for the recording)

1. **The function.** Nine lines from a repository created after the base
   model's training cutoff. 35 mutants, all live.
2. **Base model, prompted.** Eight tests. Six pass as written; two assert
   values the model computed wrong. The harness executes the function and
   rewrites those two. Valid. Kills 23 of 35.
3. **Same model, fine-tuned on 12,000 execution-verified examples.** Six
   tests, longer inputs that hit every weighted tag. Two wrong values,
   rewritten the same way. Valid. Kills 33 of 35.
4. **A mutant only the fine-tuned suite catches:** one tag name in the
   `important` set is corrupted. The base's inputs never used that tag; the
   fine-tuned suite's inputs did.
5. **Close.** Across 315 held-out functions this is +6 points of mutants
   caught per function, replicated twice, interval excluding zero. The
   model did not learn to predict outputs; it learned to write suites the
   harness can complete, and to choose inputs that exercise the code.

## What the demo does not claim

The fine-tuned suite as written also fails on the correct function (two
wrong values). Without the harness neither arm is reliably valid. The
demo shows the product setting the adapters were selected for, and says so.
