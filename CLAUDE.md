# CLAUDE.md

## Purpose

This repo adapts the spirit of `karpathy/autoresearch` to medical imaging.

The core contract is intentionally small:

- `prepare.py` is fixed during experiment loops
- `train.py` is the experiment surface
- `program.md` describes the autonomous research loop
- `results.tsv` is the untracked experiment ledger

Public v1 is scoped to 2D classification, 2D multilabel classification, and
binary segmentation. Paper and regulatory generators produce real first drafts
from experiment data, but are not the main product claim.

## Commands

```bash
uv sync
uv run prepare.py --list
uv run prepare.py --dataset <name>
uv run prepare.py --data /path/to/local/dataset
uv run train.py --recipe <name> --save-checkpoint
uv run train.py --list-recipes
uv run python tools/mark_result.py --dataset <name> --status keep
uv run python tools/finalize_run.py --dataset <name>
uv run paper/generate_paper.py
uv run regulatory/generate_regulatory.py
```

## Architecture

Generic framework pieces:

- `train.py` experiment loop and checkpoint writing
- `tools/mark_result.py`
- `tools/evaluate_checkpoint.py`
- `tools/finalize_run.py`
- `paper/generate_figures.py`
- `paper/generate_paper.py`

Dataset-specific demo pieces:

- benchmark parsing from `benchmarks/*.md`
- per-pathology comparison figures/tables in paper generator
- `prepare.py` fallback handlers for specific Kaggle mirror layouts

Keep the demo-rich CheXNet path, but do not hardcode the generic
training/evaluation framework into a chest-only repo.

## Experiment Loop

During agent-driven experiments:

- do not modify `prepare.py`
- only modify `train.py`
- write run output to `run.log`
- compare experiments on `val` only
- mark rows as `keep` or `discard` in `results.tsv`
- call `tools/finalize_run.py` once at the end for held-out `test` reporting

## Overnight Run Pattern

For long runs, use an isolated workspace or clone:

- freeze one workspace for the overnight agent session
- let the user keep working in a separate workspace
- share only read-only heavy assets if helpful (`data`, `.venv`)
- do not edit the run workspace while the agent is iterating

## Guardrails

- do not claim MONAI, 2.5D, 3D, DICOM-series, or volumetric support as current
- do not treat the generated paper as submission-ready without human revision
- do not treat the regulatory outputs as real FDA/CE/UKCA submission packs
- CheXNet is the public hello-world, paired with the bring-your-own-data story
