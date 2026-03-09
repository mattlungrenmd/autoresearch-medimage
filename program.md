# autoresearch-medimage program

This repo follows a Karpathy-style autonomous research loop for medical
imaging.

## Setup

Before starting experiments, work with the user to:

1. validate or prepare the dataset with `uv run prepare.py --dataset <name>`
2. use a dedicated run branch if git workflows are available
3. establish the baseline first
4. keep experiment output in `run.log`
5. leave `results.tsv` untracked

## Experiment Loop

Each experiment uses the fixed 300-second training budget from `prepare.py`.

The canonical loop is:

1. inspect the current branch state and the most recent results
2. modify `train.py`
3. run `uv run train.py > run.log 2>&1`
4. read the metric summary from `run.log`
5. check the appended row in `results.tsv`
6. mark the row with `uv run python tools/mark_result.py --status keep|discard`
7. keep or discard the idea externally, then continue

For multi-hour sessions, do not rely on vague instructions like "keep going"
or "continue until interrupted" alone. Use an explicit absolute local deadline
and tell the agent not to send a final summary before that deadline unless it
hits a real blocker.

## Boundaries

- `prepare.py` is fixed during research
- `train.py` is the only experiment surface
- built-in recipes are baselines and examples
- custom datasets are first-class through the dataset contract under `data/<dataset>/`
- paper and regulatory generators are downstream experimental artifacts, not the loop itself
- `tools/finalize_run.py` can regenerate the end-of-run artifacts after an overnight session

## Goal

Improve the dataset's primary metric while preserving a simple and auditable
workflow.

## Public v1 Scope

- 2D classification
- 2D multilabel classification
- binary segmentation

## Not In Scope Yet

- MONAI backend
- multislice / 2.5D / 3D support
- submission-ready regulatory documents
- automatic publication pipelines
