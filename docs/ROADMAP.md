# Roadmap

This roadmap is scoped to the public release of `autoresearch-medimage` as an
agent-first medical imaging research framework.

## v1.0 — Public Preview

Status: current release

- agent-first workflow centered on Claude/Codex editing `train.py`
- built-in datasets plus bring-your-own-data via `data/<dataset>/dataset.yaml`
- auto-detect task type from local dataset directories (`prepare.py --data /path`)
- fixed 300-second experiment budget and `results.tsv` experiment ledger
- 2D classification, multilabel classification, and binary segmentation
- advanced training features: EMA, CutMix, MixUp, LLRD, SWA, knowledge distillation, ordinal regression
- Kaggle dataset auto-detection (`tools/kaggle_autodetect.py`)
- paper tracker for monitoring new publications (`tools/paper_tracker.py`)
- paper and regulatory document generators (real first drafts from experiment data)
- CheXNet as the documented hello-world, not the only supported workflow

## v1.1 — Extended Benchmarks and Export

Status: next milestone

| Capability | Notes |
|---|---|
| MedMNIST integration | Register 18 standardized benchmarks as built-in datasets |
| ONNX export tool | Export trained models for deployment |
| Richer dataset validation | Template generation, objective templates |
| Stronger run metadata | Experiment descriptions and run metadata capture |
| Lightweight CI | Smoke-test fixtures around the data contract |

## v2.0 — Ecosystem Integration

Status: planned

| Capability | Notes |
|---|---|
| HuggingFace model upload | Model card generation + checkpoint push to Hub |
| Automated arXiv pipeline | End-to-end paper generation from run artifacts |
| Multi-GPU / distributed | DDP wrapper for parallel training |
| Richer paper generation | Multi-experiment comparison tables, auto-figure selection |
| Regulatory evidence tracking | Automated evidence-gap closure from run artifacts |

## v3.0 — MONAI Backend

Status: planned

- optional MONAI backend for DICOM/NIfTI workflows
- 2.5D and 3D volume training
- study/series-level data handling
- multislice and patch-based inference flows

## Non-Goals for Public v1

- general-purpose all-modality medical imaging support
- submission-ready FDA, CE, or UKCA documentation
- autonomous publication pipelines
- multi-GPU orchestration and distributed training
