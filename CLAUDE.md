# CLAUDE.md

## Purpose

This repo uses fast, fixed-budget experiments to search for the best model
backbone, hyperparameters, and augmentation strategy for a medical imaging
dataset. The winning configuration then informs a full-scale training run.

The core contract:

- `prepare.py` is fixed during the search phase
- `train.py` is the experiment surface
- `program.md` describes the search loop
- `results.tsv` is the untracked experiment ledger

Workflow: **search** (50-100 quick experiments) → **train** (winning config,
full compute budget) → **report** (paper and figures from the final model).

Public v1 scope: 2D classification, 2D multilabel classification, binary
segmentation.

## Commands

```bash
uv sync
uv run prepare.py --list
uv run prepare.py --dataset <name>
uv run train.py --recipe <name> --save-checkpoint
uv run train.py --recipe <name> --session <tag> --save-checkpoint
uv run train.py --list-recipes
uv run python tools/mark_result.py --dataset <name> --session <tag> --status keep|discard
uv run python tools/finalize_run.py --dataset <name> --session <tag>
```

## Overnight Search Pattern

For long agent-driven search sessions, use an isolated workspace or clone:

- freeze one workspace for the agent session
- keep working in a separate workspace
- do not edit the run workspace while the agent is iterating

During the search phase:

- do not modify `prepare.py`
- only modify `train.py`
- create a fresh session tag; treat experiment 1 as the session baseline
- write run output to `run.log`
- compare experiments on `val` only
- do not use literature numbers for keep/discard decisions
- mark rows as `keep` or `discard` in `results.tsv`
- call `tools/finalize_run.py` once at the end for held-out `test` report

## Guardrails

- do not claim MONAI, 2.5D, 3D, DICOM-series, or volumetric support as current
- do not treat the generated paper as submission-ready without human revision
- keep CheXNet as the hello-world demo, paired with the bring-your-own-data story
