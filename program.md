# autoresearch-medimage program

This repo uses fast, fixed-budget experiments to search for the best model
backbone, hyperparameters, and augmentation strategy for a medical imaging
dataset. The winning configuration then informs a full-scale training run.

## Setup

Before starting the search phase, work with the user to:

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

## Full Training

Once the search converges on a winning configuration, train to convergence
with `--time-budget` set to the full compute budget. Generate paper and
paper artifacts from the final checkpoint with `tools/finalize_run.py`.

## Boundaries

- `prepare.py` is fixed during the search phase
- `train.py` is the only experiment surface
- built-in recipes are baselines and examples
- custom datasets are first-class through the dataset contract under `data/<dataset>/`
- paper generator produces a real first draft from experiment data, run after the search phase
- `tools/finalize_run.py` regenerates end-of-search artifacts including the winning configuration

## Using With Claude or Codex

### Commands

Built-in ChestXray14 example:

```bash
uv run prepare.py --dataset chestxray14
uv run train.py --recipe chestxray14_baseline --save-checkpoint
```

Custom dataset:

```bash
uv run prepare.py --dataset your_dataset
uv run train.py --dataset your_dataset
```

Smoke test:

```bash
uv run prepare.py --dataset toy_binary
uv run train.py --recipe toy_binary_smoke --time-budget 5
```

### Good Prompts

- "Read the repo, validate my dataset contract, then run a baseline."
- "Use the current best run as the baseline and improve mean AUC without changing `prepare.py`."
- "Generate a draft paper from the best checkpoint."

### Overnight Runbook

See [examples/chestxray14/overnight_run.md](examples/chestxray14/overnight_run.md) for a
ChestXray14-specific overnight search example with kickoff prompt.

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
- submission-ready publications
- automatic publication pipelines
