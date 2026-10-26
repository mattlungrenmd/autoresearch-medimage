# ChestXray14 Literature References

## Scope

Mean AUROC across 14 findings. Each paper uses its own evaluation protocol. These are literature reference values, not a matched-protocol ranking or a claim of current state of the art.

## Published Results (Mean AUC %)

| Method | Mean AUC | Year | Evaluation protocol | Reference |
| --- | --- | --- | --- | --- |
| Wang et al. (ResNet-50) | 74.51 | 2017 | Published patient-wise test split | [1705.02315v5](https://arxiv.org/pdf/1705.02315v5#page=17) |
| CheXNet (DenseNet-121) | 84.14 | 2017 | Patient-disjoint 70/10/20 split | [1711.05225v3](https://arxiv.org/html/1711.05225v3) |
| DACNet (Strick et al.) (DenseNet-121) | 85.27 | 2025 / 2026 revision | Stratified patient-level split | [2505.06646v3](https://arxiv.org/html/2505.06646v3) |

Wang and CheXNet means are calculated as the unweighted arithmetic mean of 14 per-finding values in their original tables. DACNet's mean is taken directly from Table 3; its per-finding values below were rounded in the source.

## Source Details

- **Wang et al.:** arXiv 1705.02315v5, Table 17, ChestX-ray14 column, and Appendix B.2. This uses the updated values in the original paper, which differ from the older Wang results reproduced in CheXNet's Table 2.
- **CheXNet:** arXiv 1711.05225v3, Table 2 and Section 5. The 14-label task uses patient-disjoint train/validation/test partitions of 70%/10%/20%.
- **DACNet:** arXiv 2505.06646v3 (February 24, 2026), Tables 2 and 3. The improved DenseNet-121 uses a stratified patient-level split. The faithful CheXNet reproduction in that study scored **79.28%** mean test AUROC; **85.27%** belongs to DACNet, the improved model.

## Per-Pathology AUC (CheXNet Paper)

CheXNet values come from its Table 2; Wang values come from the original Wang v5 Table 17, and DACNet values from its v3 Table 2. All values below are percentages.

| Pathology | CheXNet AUC | Wang et al. AUC | DACNet AUC |
| --- | --- | --- | --- |
| Atelectasis | 80.94 | 70.03 | 81.70 |
| Cardiomegaly | 92.48 | 81.00 | 93.20 |
| Effusion | 86.38 | 75.85 | 90.50 |
| Infiltration | 73.45 | 66.14 | 70.80 |
| Mass | 86.76 | 69.33 | 91.90 |
| Nodule | 78.02 | 66.87 | 78.90 |
| Pneumonia | 76.80 | 65.80 | 74.00 |
| Pneumothorax | 88.87 | 79.93 | 87.50 |
| Consolidation | 79.01 | 70.32 | 78.30 |
| Edema | 88.78 | 80.52 | 89.60 |
| Emphysema | 93.71 | 83.30 | 96.30 |
| Fibrosis | 80.47 | 78.59 | 81.40 |
| Pleural Thickening | 80.62 | 68.35 | 80.10 |
| Hernia | 91.64 | 87.17 | 99.70 |

## Separate CheXNet Radiologist Comparison

CheXNet's Table 1 evaluates pneumonia F1 on a separate 420-image, radiologist-annotated set. It is not the 14-label AUROC evaluation above.

## Local Candidate and Final Evaluation

The [case study](case_study.md) records an alternate EfficientNet-B0 recipe at **84.802390% mean AUC after 10 epochs**. The repository does not archive that run's checkpoint, predictions, split manifest, or held-out test report. It cannot yet be entered as a verified final test result alongside literature results.

Before adding a final model comparison, record the checkpoint and recipe identifiers, test split definition, image/patient counts, all 14 per-finding AUROCs, and their unweighted mean. Preserve patient-level separation and use validation data for model selection.

A machine-readable transcription with versioned sources is available at [`docs/site/data/chestxray14-literature.json`](../../docs/site/data/chestxray14-literature.json).
