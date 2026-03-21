# ChestXray14 Benchmark - 14-Pathology Classification

Important: the mean AUC numbers below refer to the 14-pathology ChestXray14
classification setting reported in Table 2 of the original CheXNet paper.
They are not the same evaluation as the paper's separate 420-image pneumonia
vs. radiologist comparison, which reported F1 on a curated test set.

## Published Results (Mean AUC %)

| Method | Mean AUC | Year | Reference |
|--------|----------|------|-----------|
| Wang et al. (baseline) | 74.5 | 2017 | [arxiv 1705.02315](https://arxiv.org/abs/1705.02315) |
| **CheXNet** (DenseNet-121) | **84.1** | 2017 | [arxiv 1711.05225](https://arxiv.org/abs/1711.05225) |
| CheXNet reproduction | 85.0 | 2025 | [arxiv 2505.06646](https://arxiv.org/abs/2505.06646) |
| TorchXRayVision (DenseNet) | 85.1 | 2022 | [GitHub](https://github.com/mlmed/torchxrayvision) |

## FDA-Cleared / Regulatory Benchmarks

| Device | Cleared For | Sensitivity | Specificity | Clearance |
|--------|------------|-------------|-------------|-----------|
| Qure.ai qXR | Pneumothorax, TB | 95% | 90% | FDA 510(k), CE |
| Lunit INSIGHT CXR | Nodule, Consolidation | 97% | 87% | FDA, CE |
| Annalise CXR | 124 findings | 90%+ | 90%+ | FDA, CE, TGA |
| Zebra Medical | Pneumothorax | 93% | 94% | FDA 510(k) |

*Note: FDA-cleared devices are evaluated on proprietary datasets with different distributions. Direct comparison with academic benchmarks is approximate.*

## Per-Pathology AUC (CheXNet Paper)

| Pathology | CheXNet AUC | Wang et al. AUC |
|-----------|-------------|-----------------|
| Atelectasis | 80.94 | 71.6 |
| Cardiomegaly | 92.48 | 80.7 |
| Effusion | 86.38 | 78.4 |
| Infiltration | 73.45 | 60.9 |
| Mass | 86.81 | 70.6 |
| Nodule | 78.02 | 67.1 |
| Pneumonia | 76.80 | 63.3 |
| Pneumothorax | 93.71 | 80.6 |
| Consolidation | 80.07 | 70.8 |
| Edema | 88.78 | 83.5 |
| Emphysema | 93.71 | 81.5 |
| Fibrosis | 80.47 | 76.9 |
| Pleural Thickening | 80.62 | 68.4 |
| Hernia | 91.64 | 87.2 |

## Separate CheXNet Radiologist Comparison

The original CheXNet paper also reported a different evaluation:

- task: binary pneumonia detection
- test set: 420 frontal chest X-rays with four radiologist annotations
- metric: F1 score against radiologist labels
- result: CheXNet 0.435 vs radiologist average 0.387

Do not compare that F1 result directly to the mean AUC values above.

## autoresearch-medimage Results

*Run `uv run train.py --recipe chestxray14_baseline --save-checkpoint` to add your results here.*

| Commit | Mean AUC | Backbone | Recipe | Memory (GB) | Notes |
|--------|----------|----------|--------|-------------|-------|
| - | - | - | - | - | *Run your first experiment!* |
