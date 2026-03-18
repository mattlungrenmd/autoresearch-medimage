# Roadmap

This roadmap is scoped to the public release of `autoresearch-medimage` as an
agent-first medical imaging research framework.

## Phase 1: Public Preview

Status: in repo now

- agent-first workflow centered on Claude/Codex editing `train.py`
- built-in datasets plus bring-your-own-data via `data/<dataset>/dataset.yaml`
- fixed 300-second experiment budget and `results.tsv` experiment ledger
- 2D classification, multilabel classification, and binary segmentation
- CheXNet as the documented hello-world, not the only supported workflow
- experimental paper and regulatory scaffold generators

## Phase 2: Custom Research Scaffolding

Status: next milestone

- richer dataset validation and template generation
- objective templates for common medical-imaging research goals
- generic recipe templates for new datasets and task types
- stronger experiment descriptions and run metadata capture
- better smoke-test fixtures and lightweight CI around the data contract

## Phase 3: MONAI Backend

Status: planned

- optional MONAI backend for DICOM/NIfTI workflows
- 2.5D and 3D volume training
- study/series-level data handling
- multislice and patch-based inference flows

## Phase 4: Reporting and Packaging

Status: planned

- stronger paper/report generation from run artifacts
- safer regulatory planning outputs with clearer evidence-gap tracking
- deployment/export tracks for research packaging

## Non-Goals for Public v1

- general-purpose all-modality medical imaging support
- submission-ready FDA, CE, or UKCA documentation
- autonomous publication pipelines
- multi-GPU orchestration and distributed training
