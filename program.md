# autoresearch-medimage

Autonomous AI research for medical image classification and segmentation, focused on ophthalmology and radiology. The default task reproduces CheXNet (Rajpurkar et al., 2017).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date and task (e.g. `mar9-chexnet`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, Kaggle download, dataloaders, evaluation metrics. **Do not modify.**
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `./data/<dataset>/splits.csv` exists. If not, tell the human to run `uv run prepare.py --dataset <name>`.
5. **Initialize results.tsv**: Create with header row. Baseline recorded after first run.
6. **Confirm and go**.

## Available Datasets

Run `uv run prepare.py --list` to see all datasets. Summary:

### Ophthalmology
| Dataset | Task | Classes | Metric | Kaggle Source |
|---------|------|---------|--------|---------------|
| `aptos` | Classification | 5 (DR severity) | Quadratic Kappa | APTOS 2019 Blindness Detection |
| `oct` | Classification | 4 (CNV/DME/Drusen/Normal) | Accuracy | Kermany 2018 Retinal OCT |
| `retina_vessel` | Segmentation | 2 (vessel/background) | Dice | Retina Blood Vessel |

### Radiology
| Dataset | Task | Classes | Metric | Kaggle Source |
|---------|------|---------|--------|---------------|
| `chestxray14` | Multi-label | 14 pathologies | Mean AUC | NIH ChestX-ray14 (CheXNet) |
| `chest_xray` | Classification | 2 (Normal/Pneumonia) | AUC | Chest X-ray Pneumonia |
| `brain_mri` | Segmentation | 2 (tumor/background) | Dice | LGG MRI Segmentation |
| `pneumothorax` | Segmentation | 2 (pneumothorax/bg) | Dice | SIIM-ACR Pneumothorax |

## Experimentation

Each experiment runs on a single GPU with a **fixed 5-minute time budget**. Launch:

```bash
uv run train.py                           # CheXNet on ChestX-ray14 (default)
uv run train.py --dataset aptos           # DR grading
uv run train.py --dataset brain_mri       # Brain tumor segmentation
```

**What you CAN do:**
- Modify `train.py` — the only file you edit. Everything is fair game:
  - **Backbone**: DenseNet-121/169/201, ResNet-50/101, EfficientNet-B0 through B7, ConvNeXt, Vision Transformers (via `timm`)
  - **Optimizer**: Adam, AdamW, SGD, LARS, lookahead
  - **Loss**: BCE, focal loss, weighted CE, Dice+BCE, ordinal regression (for DR grading)
  - **Augmentation**: CLAHE, CutOut, MixUp, RandAugment, elastic deformations, stain normalization
  - **Training tricks**: Freeze/unfreeze schedule, discriminative LR, EMA, test-time augmentation
  - **Architecture**: Attention mechanisms, feature pyramid networks, multi-scale fusion

**What you CANNOT do:**
- Modify `prepare.py`. It contains fixed evaluation metrics and data loading.
- Install new packages beyond what's in `pyproject.toml`.
- Modify the evaluation functions (`evaluate_classification`, `evaluate_multilabel`, `evaluate_segmentation`).

### Goals by Task

- **ChestX-ray14 (CheXNet)**: Highest **mean_auc** across 14 pathologies. Paper reports ~84.1% mean AUC.
- **APTOS (DR grading)**: Highest **quadratic_kappa**. Competition winners achieved ~0.93.
- **OCT classification**: Highest **accuracy**. State-of-art ~99%+.
- **Chest X-ray (pneumonia)**: Highest **auc**.
- **Segmentation tasks**: Highest **dice** coefficient.

### Medical Imaging Best Practices

These are known to work well in medical imaging — try them:

1. **Transfer learning is essential**: Always start from ImageNet pretrained. Fine-tuning beats training from scratch by large margins on small medical datasets.
2. **Gradual unfreezing**: Freeze backbone initially, train head for a few steps, then unfreeze with discriminative learning rates (lower LR for earlier layers).
3. **CLAHE preprocessing**: `prepare.py` exports `apply_clahe()` — improves contrast in fundus images and X-rays.
4. **Class imbalance**: Medical datasets are almost always imbalanced. Use weighted loss, oversampling, or focal loss.
5. **Resolution matters**: Try larger image sizes (384, 512) for pathology that depends on fine details. Trade off batch size.
6. **Test-time augmentation (TTA)**: Average predictions over flipped/rotated versions at eval time.
7. **Label smoothing + mixup**: Regularization that helps with small, noisy medical datasets.
8. **Ordinal regression**: For graded diseases (DR severity 0-4), ordinal approaches > standard classification.

## Output format

Classification (CheXNet):
```
---
mean_auc:         84.10%
per_class_auc:
  Atelectasis            80.94%
  Cardiomegaly           92.48%
  ...
training_seconds: 300.1
total_seconds:    340.2
peak_vram_mb:     6230.5
num_steps:        5340
num_params_M:     7.0
num_epochs:       2
dataset:          chestxray14
task:             multilabel
backbone:         densenet121
```

Segmentation:
```
---
val_dice:         78.50%
val_iou:          64.70%
val_pixel_acc:    96.20%
val_sensitivity:  82.30%
val_specificity:  97.10%
training_seconds: 300.1
...
```

Extract key metric: `grep "^mean_auc:\|^val_accuracy:\|^val_dice:\|^val_kappa:" run.log`

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	metric	memory_gb	status	description
```

Example:
```
commit	metric	memory_gb	status	description
a1b2c3d	84.10	6.1	keep	baseline CheXNet DenseNet-121
b2c3d4e	85.30	6.2	keep	switch to AdamW + focal loss
c3d4e5f	84.00	6.1	discard	remove pretrained weights
d4e5f6g	86.10	8.3	keep	EfficientNet-B3 backbone
e5f6g7h	0.00	0.0	crash	ViT-Large (OOM)
```

## The experiment loop

LOOP FOREVER:

1. Check git state
2. Modify `train.py` with an experimental idea
3. `git commit`
4. `uv run train.py > run.log 2>&1`
5. `grep "^mean_auc:\|^val_accuracy:\|^val_dice:" run.log`
6. If empty → crashed. `tail -n 50 run.log` for traceback.
7. Record in `results.tsv` (do not commit this file)
8. If metric improved → keep commit
9. If metric equal or worse → `git reset` to previous

### Experiment strategy: Architecture search + hyperparameter tuning

The agent should try **both** architecture changes AND hyperparameter tuning. Medical imaging is especially sensitive to backbone choice — a ResNet-50 vs EfficientNet-B3 vs ConvNeXt can differ by 3-5% AUC. Suggested experiment order:

**Phase 1 — Architecture sweep (experiments 1-20)**:
Try different backbones while keeping other settings constant. The `BACKBONE` variable in `train.py` accepts any `timm` model name. Priority order:

```python
# Classification backbones to try (via timm):
BACKBONE = "densenet121"          # CheXNet baseline
BACKBONE = "densenet169"          # Deeper DenseNet
BACKBONE = "resnet50"             # Classic ResNet
BACKBONE = "efficientnet_b0"      # EfficientNet (small)
BACKBONE = "efficientnet_b3"      # EfficientNet (medium) ← usually best value
BACKBONE = "efficientnet_b4"      # EfficientNet (large)
BACKBONE = "convnext_small"       # ConvNeXt
BACKBONE = "convnext_base"        # ConvNeXt (larger)
BACKBONE = "swin_tiny_patch4_window7_224"  # Swin Transformer
BACKBONE = "vit_small_patch16_224"  # Vision Transformer
BACKBONE = "tf_efficientnetv2_s"  # EfficientNetV2
```

**Phase 2 — Optimize best architecture (experiments 20-50)**:
Take the winning backbone and tune:
- Optimizer: Adam → AdamW → SGD with momentum
- Learning rate: 1e-4, 3e-4, 1e-3, 3e-3
- Loss: BCE → focal loss (gamma=1,2,3)
- Weight decay: 1e-5, 1e-4, 1e-3, 1e-2
- Dropout: 0.1, 0.2, 0.3, 0.5

**Phase 3 — Augmentation + tricks (experiments 50-100)**:
- Add CLAHE preprocessing
- Add CutOut / CutMix / MixUp
- Increase image resolution (224 → 384 → 512)
- Label smoothing (0.05, 0.1)
- Test-time augmentation
- Gradual unfreezing + discriminative LR
- EMA (exponential moving average)

### Ideas to explore (CheXNet / ChestX-ray14)

- **Backbone upgrades**: DenseNet-169/201, EfficientNet-B3/B4, ConvNeXt-Small, DeiT/ViT-Small
- **Optimizer**: AdamW (shown to improve over Adam in reproduction studies)
- **Loss**: Focal loss (handles class imbalance in multi-label setting) — set `LOSS_TYPE = "focal"`
- **Augmentation**: Color jitter, CLAHE, random rotation ±15°, CutOut
- **Architecture**: Multi-scale feature aggregation, squeeze-and-excitation in classifier head
- **Training**: Gradual unfreezing, cosine annealing with warm restarts, EMA

### Ideas to explore (Ophthalmology)

- **DR grading**: Ordinal regression loss, CLAHE preprocessing, larger input resolution (384/512), attention-based pooling
- **OCT**: Fine-grained classification with mixup, self-supervised pretraining

### Ideas to explore (Segmentation)

- **Architecture**: Attention U-Net, U-Net++, DeepLabV3+ with pretrained encoder
- **Loss**: Dice + focal, boundary-aware loss, Tversky loss — set `LOSS_TYPE = "dice_bce"`
- **Post-processing**: CRF, test-time augmentation with averaging

**NEVER STOP**: Once experiments begin, do NOT pause. Run until manually stopped. If stuck, try radical changes — different backbone, different loss, different augmentation strategy.
