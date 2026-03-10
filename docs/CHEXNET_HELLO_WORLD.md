# CheXNet Hello World

CheXNet is the repo's canonical hello-world example.

It is not the product center and it is not the only intended use case. It is
the end-to-end demo that proves the repo can:

- prepare a real public medical imaging dataset
- run a fixed-budget autoresearch-style experiment
- log results to `results.tsv`
- save a checkpoint
- separate inner-loop model selection from final held-out reporting
- generate a draft paper from run results

That makes it the right example to reference when promoting the repo publicly.

## Why CheXNet

CheXNet is a good hello-world because it is:

- well known in medical imaging
- multi-label rather than toy single-label classification
- easy to explain as a complete workflow
- representative of the repo's current 2D scope

## Commands

Prepare the dataset:

```bash
uv run prepare.py --dataset chestxray14
```

If you already have the Kaggle mirror downloaded locally and want to rebuild the
current paper-grade split protocol, use:

```bash
uv run prepare.py --dataset chestxray14 --refresh-splits
```

Run the baseline:

```bash
uv run train.py --recipe chexnet --save-checkpoint
```

Regenerate the end-of-run artifacts, including the held-out `test` evaluation:

```bash
uv run python tools/finalize_run.py --dataset chestxray14
```

Generate the paper from the run results and checkpoint data:

```bash
uv run paper/generate_paper.py --dataset chestxray14 --output paper/generated/chexnet_hello_world.tex
```

Generate manuscript figures from the run artifacts:

```bash
uv run python paper/generate_figures.py --dataset chestxray14 --output-dir paper/generated/chestxray14_figures
```

Generate the progress plot for the overnight or multi-experiment run:

```bash
uv run python tools/progress_plot.py --dataset chestxray14 --output progress.png
```

## What To Show Publicly

When promoting the repo, the strongest story is:

1. start from a real benchmark dataset
2. run a reproducible 300-second baseline selected on `val`
3. finalize the run and report the untouched `test` result
4. show the experiment ledger entry, checkpoint, and generated figures
5. explain that users can then swap in their own dataset and objective

