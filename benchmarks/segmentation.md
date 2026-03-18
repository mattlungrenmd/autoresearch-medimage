# Medical Image Segmentation Benchmark

## Brain Tumor Segmentation

| Method | Dice | Year | Reference |
|--------|------|------|-----------|
| U-Net baseline | 82-85% | 2015 | [arxiv 1505.04597](https://arxiv.org/abs/1505.04597) |
| nnU-Net | 88-92% | 2021 | [Nature Methods](https://doi.org/10.1038/s41592-020-01008-z) |
| Attention U-Net | 86-89% | 2018 | [arxiv 1804.03999](https://arxiv.org/abs/1804.03999) |
| MedSAM | 85-90% | 2024 | [Nature Comms](https://doi.org/10.1038/s41467-024-44824-z) |

## Retinal Vessel Segmentation (DRIVE / CHASE_DB1)

| Method | Dice/F1 | AUC | Year | Reference |
|--------|---------|-----|------|-----------|
| U-Net | 79-82% | 97.5% | 2015 | Standard baseline |
| CS-Net | 83-85% | 98.1% | 2019 | MICCAI |
| SA-UNet | 82-84% | 98.0% | 2020 | Expert Systems |

## Pneumothorax Segmentation (SIIM-ACR)

| Method | Dice | Year | Reference |
|--------|------|------|-----------|
| SIIM-ACR 1st Place | 86.7% | 2019 | Kaggle |
| U-Net baseline | 72-78% | — | Standard baseline |
| EfficientNet + U-Net | 83-85% | — | Top Kaggle solutions |

## FDA-Cleared Segmentation Tools

| Device | Anatomy | Dice (reported) | Clearance |
|--------|---------|-----------------|-----------|
| Viz.ai (LVO) | Brain vessels | 94%+ | FDA 510(k) |
| Aidoc (ICH) | Brain hemorrhage | 95%+ | FDA 510(k) |
| Arterys (Cardiac) | Heart chambers | 92%+ | FDA 510(k) |
| Brainomix (Stroke) | Brain ischemia | 90%+ | CE Mark, FDA |

## autoresearch-medimage Results

| Dataset | Commit | Dice | IoU | Backbone | Notes |
|---------|--------|------|-----|----------|-------|
| brain_mri | — | — | — | — | *Run your first experiment!* |
| retina_vessel | — | — | — | — | — |
| pneumothorax | — | — | — | — | — |
