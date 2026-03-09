# CLAUDE.md

## Purpose

This repo adapts the spirit of `karpathy/autoresearch` to medical imaging.

The core contract is still intentionally small:

- `prepare.py` is fixed during experiment loops
- `train.py` is the experiment surface
- `program.md` describes the autonomous research loop
- `results.tsv` is the untracked experiment ledger

Public v1 is scoped to 2D classification, 2D multilabel classification, and
binary segmentation. Paper and regulatory generators produce real first drafts
from experiment data, but are not the main product claim.

## Current State

The repo now supports:

- built-in benchmark datasets plus bring-your-own local datasets
- automatic result logging to `results.tsv`
- session-scoped overnight runs via `--session`
- checkpoint saving with run metadata and evaluation artifacts
- end-of-run artifact regeneration with `tools/finalize_run.py`
- manuscript figures and a paper generated from actual run outputs
- seeded dataloaders and richer run metadata for more stable overnight comparisons

CheXNet is the canonical hello-world demo, not the whole product story.

## Current CheXNet Baseline

ChestX-ray14 has been tightened into a more paper-grade protocol:

- `data/chestxray14/splits.csv` now contains `train`, `val`, and `test`
- the Kaggle mirror fallback reads `splits_meta/train_split.csv`,
  `val_split.csv`, and `test_split.csv` when the original NIH CSV is absent
- model selection happens on `val`
- held-out reporting happens on `test` only during finalization

Current baseline ledger state:

- legacy validation-only baseline: `73.41` mean AUC, marked `discard`
- active paper-grade split baseline: `64.41` validation mean AUC, marked `keep`
- held-out final test result for the active baseline: `65.31` mean AUC

## Generic vs Dataset-Specific Logic

Generic framework pieces:

- `train.py` experiment loop and checkpoint writing
- `tools/mark_result.py`
- `tools/evaluate_checkpoint.py`
- `tools/finalize_run.py`
- `paper/generate_figures.py`
- `paper/generate_paper.py`

ChestX-ray14-specific demo pieces:

- benchmark parsing from `benchmarks/chestxray14.md`
- per-pathology comparison figures/tables
- `prepare.py` fallback for the `ksw2000/chestxray14` Kaggle mirror layout

Keep the demo-rich ChestX-ray14 path, but do not let it hardcode the generic
training/evaluation framework into a chest-only repo.

## Commands

```bash
uv sync
uv run prepare.py --list
uv run prepare.py --dataset chestxray14
uv run prepare.py --dataset chestxray14 --refresh-splits
uv run train.py --recipe chexnet --save-checkpoint
uv run train.py --recipe chexnet --session chexnet_overnight_20260310 --save-checkpoint
uv run train.py --list-recipes
uv run python tools/mark_result.py --dataset chestxray14 --session chexnet_overnight_20260310 --status keep
uv run python tools/finalize_run.py --dataset chestxray14 --session chexnet_overnight_20260310
uv run paper/generate_paper.py
uv run regulatory/generate_regulatory.py
```

## Overnight Run Pattern

The repo is designed for agent-driven fixed-budget experiments, but it does not
ship a detached scheduler that mutates `train.py` without an attached agent
session.

For long runs, use an isolated workspace or clone:

- freeze one workspace for the overnight agent session
- let the user keep working in a separate workspace
- share only read-only heavy assets if helpful (`data`, `.venv`)
- do not edit the run workspace while the agent is iterating

During the loop:

- do not modify `prepare.py`
- only modify `train.py`
- create a fresh session tag and treat experiment 1 in that session as the only inner-loop baseline
- write run output to `run.log`
- compare experiments on `val` only
- do not use literature numbers for keep/discard decisions
- mark rows as `keep` or `discard` in `results.tsv`
- call `tools/finalize_run.py` once at the end to get the held-out `test` report

## Key Artifacts

Current important outputs include:

- `results.tsv`
- `run.log`
- `progress.png`
- `paper/generated/chestxray14_final_eval.pt`
- `paper/generated/chestxray14_figures/`
- `paper/generated/chestxray14_overnight.tex`
- `regulatory/generated/chestxray14_overnight/`

## Guardrails

- do not claim MONAI, 2.5D, 3D, DICOM-series, or volumetric support as current
- do not treat the generated paper as submission-ready without human revision
- do not treat the regulatory outputs as real FDA/CE/UKCA submission packs
- keep CheXNet as the public hello-world, paired with the bring-your-own-data story
