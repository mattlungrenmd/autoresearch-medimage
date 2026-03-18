# CheXNet Hello World

CheXNet is the repo's canonical hello-world example.

It is not the product center and it is not the only intended use case. It is
the end-to-end demo that proves the repo can:

- prepare a real public medical imaging dataset
- run a fixed-budget autoresearch-style experiment
- log results to `results.tsv`
- save a checkpoint
- separate inner-loop model selection from final held-out reporting
- generate a draft paper scaffold
- generate regulatory planning scaffolds

That makes it the right example to reference when promoting the repo publicly.

## Why CheXNet

CheXNet is a good hello-world because it is:

- well known in medical imaging
- multi-label rather than toy single-label classification
- easy to explain as a complete workflow
- representative of the repo's current 2D scope

## Benchmark Context

If you want the original paper comparison tables, use
[`benchmarks/chestxray14.md`](../benchmarks/chestxray14.md).

That benchmark note includes:

- the published mean AUC table with Wang et al. and the original CheXNet paper
- the per-pathology CheXNet-vs-Wang breakdown from the paper
- a placeholder section for repo-generated results

After you finalize a run and generate figures, the repo can also produce:

- `paper/generated/chestxray14_figures/benchmark_comparison.png`
- `paper/generated/chestxray14_figures/per_class_comparison.png`

Those figures are the easiest way to show your run against the original paper
tables without manually rebuilding the plots.

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

Generate the experimental paper scaffold from the resulting run ledger:

```bash
uv run paper/generate_paper.py --dataset chestxray14 --output paper/generated/chexnet_hello_world.tex
```

Generate manuscript figures from the run artifacts:

```bash
uv run python paper/generate_figures.py --dataset chestxray14 --output-dir paper/generated/chestxray14_figures
```

That figure pass reads the benchmark tables from
[`benchmarks/chestxray14.md`](../benchmarks/chestxray14.md) and writes the
paper-comparison plots into `paper/generated/chestxray14_figures/`.

Generate the progress plot for the overnight or multi-experiment run:

```bash
uv run python tools/progress_plot.py --dataset chestxray14 --output progress.png
```

Generate the experimental regulatory scaffolds:

```bash
uv run regulatory/generate_regulatory.py --dataset chestxray14 --output-dir regulatory/generated/chexnet_hello_world
```

## What To Show Publicly

When promoting the repo, the strongest story is:

1. start from a real benchmark dataset
2. show the original paper comparison table and per-pathology breakdown
3. run a reproducible 300-second baseline selected on `val`
4. finalize the run and report the untouched `test` result
5. show the experiment ledger entry, checkpoint, and generated figures
6. explain that users can then swap in their own dataset and objective

## Important Positioning

- The paper generator output is a draft scaffold, not a submission-ready paper.
- The regulatory outputs are planning scaffolds, not FDA, CE, or UKCA submission packs.
- Be explicit that public v1 is optimized for 2D workflows.
- Pair the CheXNet hello-world with a bring-your-own-data example in the README.
- The stricter held-out `test` evaluation path is a benchmark/demo refinement, not a ChestXNet-only hardcode in the core training loop.
- Treat the benchmark comparison as a worked throwback and case study, not as a claim that one repo run replaces the original paper.

## Promotion Guidance

For launch materials, treat the eventual CheXNet paper as:

- the hello-world case study
- proof that the repo can connect data, training, evaluation, and writing
- a revisit of a classic benchmark using a Karpathy-style autoresearch workflow
- a concrete artifact generated from the same run stack users can inspect locally

The broader repo story should still be: use Claude or Codex on your own medical
imaging dataset, with CheXNet as the first complete example.
