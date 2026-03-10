# autoresearch-medimage program

This repo uses fast, fixed-budget experiments to search for the best model
backbone, hyperparameters, and augmentation strategy for a medical imaging
dataset. The winning configuration then informs a full-scale training run.

## Setup

Before starting experiments, work with the user to:

1. validate or prepare the dataset with `uv run prepare.py --dataset <name>`
2. use a dedicated run branch if git workflows are available
3. choose a fresh session tag for the run
4. establish the session baseline first
5. keep experiment output in `run.log`
6. leave `results.tsv` untracked

## Search Phase

Each experiment uses the fixed 300-second training budget from `prepare.py`.
The goal is to find the winning configuration, not to train a production model.

The canonical loop is:

1. inspect the current branch state and the most recent results
2. modify `train.py`
3. run `uv run train.py --session <tag> > run.log 2>&1`
4. read the metric summary from `run.log`
5. check the appended row in `results.tsv`
6. mark the row with `uv run python tools/mark_result.py --session <tag> --status keep|discard`
7. keep or discard the idea externally, then continue

For multi-hour sessions:
- use a fresh session tag so the run starts from a clean inner-loop baseline
- treat experiment 1 in that session as the only starting baseline
- do not use literature numbers for keep/discard decisions
- do not rely on vague instructions like "keep going" or "continue until interrupted" alone
- prefer a concrete stop condition such as "50 completed experiments after the session baseline"
- tell the agent not to send a final summary before that stop condition unless it hits a real blocker

## Boundaries

- `prepare.py` is fixed during research
- `train.py` is the only experiment surface
- built-in recipes are baselines and examples
- custom datasets are first-class through the dataset contract under `data/<dataset>/`
- paper and regulatory generators produce real first drafts from experiment data, run after the loop
- `tools/finalize_run.py` can regenerate the end-of-run artifacts after an overnight session

## Full Training

Once the search converges on a winning configuration, train to convergence
with `--time-budget` set to the full compute budget. Generate paper and
regulatory artifacts from the final checkpoint with `tools/finalize_run.py`.

## Goal

Find the best model configuration for the dataset's primary metric, then
train it to convergence.

## Public v1 Scope

- 2D classification
- 2D multilabel classification
- binary segmentation

## Not In Scope Yet

- MONAI backend
- multislice / 2.5D / 3D support
- submission-ready regulatory documents
- automatic publication pipelines
