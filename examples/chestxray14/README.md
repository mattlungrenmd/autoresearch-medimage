# ChestXray14 / CheXNet Hello World

This is a dedicated hello-world walkthrough for ChestXray14 using the
`chestxray14_baseline` recipe. The legacy `chexnet` recipe name still works as
an alias.

It is an end-to-end demo that shows the repo can:

- prepare a real medical imaging dataset
- run a fixed-budget autoresearch-style experiment
- log results to `results.tsv`
- save a checkpoint
- separate inner-loop model selection from final held-out reporting
- generate a draft paper from run results

## Benchmark Context

If you want the original paper comparison table, see
[benchmark.md](benchmark.md).

That file includes:

- the published mean-AUC table
- the per-pathology CheXNet-vs-Wang breakdown
- a note separating the 14-label mean-AUC benchmark from the paper's separate
  420-image pneumonia F1 comparison against radiologists
- the place where you can record your own repo results

If you generate figures from your run artifacts, the most relevant outputs are:

- `paper/generated/chestxray14_figures/benchmark_comparison.png`
- `paper/generated/chestxray14_figures/per_class_comparison.png`

For Colab specifically:

- use `examples/chestxray14/colab_demo.ipynb` for this single-baseline hello-world
- use `notebooks/colab_autoresearch.ipynb` for a real multi-experiment Colab
  search session on the same dataset

## Why This Example

This benchmark is a good hello-world because it is:

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
uv run train.py --recipe chestxray14_baseline --save-checkpoint
```

For a real search session instead of one baseline, use a named session:

```bash
uv run train.py --recipe chestxray14_baseline --session chestxray14_search_20260311 --save-checkpoint
```

Regenerate the end-of-run artifacts, including the held-out `test` evaluation:

```bash
uv run python tools/finalize_run.py --dataset chestxray14
```

Generate the paper from the run results and checkpoint data:

```bash
uv run paper/generate_paper.py --dataset chestxray14 --output paper/generated/chestxray14_hello_world.tex
```

Generate manuscript figures from the run artifacts:

```bash
uv run python paper/generate_figures.py --dataset chestxray14 --output-dir paper/generated/chestxray14_figures
```

Generate the progress plot for the overnight or multi-experiment run:

```bash
uv run python tools/progress_plot.py --dataset chestxray14 --output progress.png
```
