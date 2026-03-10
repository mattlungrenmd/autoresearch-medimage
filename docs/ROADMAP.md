# Roadmap

This repo is the autonomous experiment runner that sits on top of training
frameworks — not a framework competing with MONAI, nnU-Net, or HuggingFace.

The roadmap reflects that: improvements focus on making the search loop
smarter, the handoff to production cleaner, and the integration with the
existing ecosystem tighter.

## v1.0 — Public Preview (current)

The search → train → report loop works end-to-end for 2D tasks.

- agent-first workflow: Claude/Codex edits `train.py`, follows `program.md`
- bring-your-own-data via `data/<dataset>/dataset.yaml` or auto-detect
- fixed 300-second experiment budget, `results.tsv` ledger, session isolation
- 2D classification, multilabel classification, binary segmentation
- 13 recipes, 7 built-in datasets, Colab notebook
- winning configuration output at end of search
- paper generator from experiment data

## v1.1 — Smarter Search

Make the agent find better models faster.

| Capability | Why it matters |
|---|---|
| Search history awareness | Agent reads past experiments to avoid repeating failed ideas |
| Bayesian-style next-experiment suggestion | Reduce wasted experiments on unpromising regions |
| External validation split | Auto-hold a site/scanner cohort for generalization testing |
| Configurable time budgets per phase | Quick sweeps early, longer runs for promising configs |
| Lightweight CI | Smoke-test fixtures to catch data contract regressions |

## v1.2 — Production Handoff

Make the winning model deployable.

| Capability | Why it matters |
|---|---|
| ONNX export of winning checkpoint | Standard interchange format for inference runtimes |
| Model card generation | Document what the model does, how it was validated, and its limits |
| HuggingFace Hub push | One command to publish the winning model with card and weights |
| Reproducibility manifest | Lock exact versions, seeds, and config for the winning run |

## v2.0 — Ecosystem Backends

Use the best tools for each task, not just the built-in training loop.

| Capability | Why it matters |
|---|---|
| MONAI transforms and data loading | Leverage MONAI's medical-grade preprocessing (resampling, windowing, bias correction) while keeping the search loop |
| nnU-Net as a search backend | Self-configuring segmentation pipeline as one of the experiment options |
| Foundation model fine-tuning recipes | Systematic search over adaptation strategies for BiomedCLIP, RETFound, etc. |
| Vision-language model integration | Search over VLM fine-tuning for medical tasks — Phi-4V, MedImageInsight, MedGemma, and other open-source medical VLMs |
| 2.5D and 3D volume support | Extend the search loop to CT/MR volumes via backend integration |

## v3.0 — Clinical Readiness

Bridge the gap between research search and clinical deployment.

| Capability | Why it matters |
|---|---|
| DICOM-native data loading | Read directly from DICOM studies, not just PNGs |
| DICOMweb integration | Pull studies from PACS/VNA for training and push results back |
| Structured result output | Write model predictions as DICOM SR/SEG for clinical workflow integration |
| Drift monitoring scaffold | Track post-deployment performance against the validation baseline |

## Non-Goals

These are outside the repo's purpose:

- becoming a training framework (that's MONAI, nnU-Net, HuggingFace)
- replacing clinical deployment platforms (that's MONAI Deploy, Clara)
- autonomous publication without human review
- general-purpose all-modality support without backend integration
