# Charts

Six editorial charts, one conclusion each, every number computed from the
committed run artifacts by `make charts` (`testgen/charts.py`). The `.html`
files are the interactive originals (hover for the exact values, click to
replay the reveal); the `.png` files are headless-browser snapshots for the
markdown.

Visual grammar: [Lieflat Charts](https://github.com/larashero3-dotcom/lieflat-charts)
(MIT), templates F5 Tick Rows, F12 Dumbbell Queue, F6 Paired Rungs, F7 Stacked
Rungs, F2 Hairline Line and L13 Hourglass Stream, skinned with its "wire"
preset: greys carry the data, one orange element per chart is the point.

| chart | says |
|---|---|
| [results](results.html) | grounded score per arm on the 315 test functions, paired 95% CI vs base zero-shot; the base under the style prompt is the top arm and both adapters sit below it |
| [harness](harness.html) | filling expected values by execution lifts every arm by ~28 validity points before any fine-tune |
| [discordant](discordant.html) | how many functions become valid only under each arm vs only under the base; McNemar's two counts |
| [styles](styles.html) | assertion mix per arm: the adapters and the prompted base write the teacher's style, 82–89% exact-value literals |
| [devcurves](devcurves.html) | dev-171 grounded score per checkpoint for both adapters, base as the dashed floor |
| [funnel](funnel.html) | KodCode curation, 169,000 rows poured down to 12,000 training examples |
