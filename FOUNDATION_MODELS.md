# Foundation Models

This repo can use some HuggingFace-hosted vision encoders for 2D medical
imaging by prefixing the backbone with `hf:`.

## Current Scope

- classification and multilabel classification only
- encoder extraction through `MedicalClassifier`
- no claim of automatic freeze/unfreeze schedules
- external checkpoints remain experimental until you validate them on your task

## Examples

```python
BACKBONE = "hf:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BACKBONE = "hf:microsoft/rad-dino"
BACKBONE = "hf:rmaphoh/RETFound_MAE"
```

## Guidance

- use lower learning rates than standard ImageNet baselines
- expect larger memory usage and slower startup due to checkpoint download
- validate preprocessing and channel assumptions on your dataset
- treat these backbones as optional experiments, not guaranteed improvements
