# CLAUDE.md — Agent Instructions for autoresearch-medvision

## Project Overview
Autonomous medical imaging AI research framework. Iteratively experiments with architectures, optimizers, losses, and augmentations — running fixed 5-minute experiments and keeping only improvements.

## Architecture (3-file design from karpathy/autoresearch)
- `prepare.py` — **READ-ONLY**. Dataset download, dataloaders, evaluation metrics. Never modify.
- `train.py` — **Agent-editable**. The ONLY file agents should modify for experiments.
- `program.md` — Agent instructions. Reference for experiment strategy.
- `recipes/` — YAML configs for reproducing published papers. Add new ones freely.

## Key Commands
```bash
# Install dependencies
uv sync

# Download a dataset
uv run prepare.py --dataset chestxray14
uv run prepare.py --list                    # See all datasets

# Run training with a recipe
uv run train.py --recipe chexnet --save-checkpoint
uv run train.py --list-recipes              # See all recipes

# Run default (CheXNet)
uv run train.py

# Tools
python tools/paper_tracker.py --days 7 --with-data    # Find new papers
python tools/kaggle_autodetect.py <kaggle-slug>        # Auto-detect new dataset
python paper/generate_paper.py --compile               # Generate arXiv paper
python regulatory/generate_regulatory.py               # Generate regulatory docs
```

## Experiment Workflow
1. Modify globals in `train.py` (BACKBONE, OPTIMIZER, LR, LOSS_TYPE, etc.)
2. Run: `uv run train.py`
3. Check output: metric value, memory usage
4. If improved: `git add train.py && git commit -m "description of change"`
5. If worse: `git checkout train.py` and try something else
6. Results auto-append to `results.tsv`

## What to Modify in train.py
- `BACKBONE` — any timm model name (densenet121, efficientnet_b3, convnext_small, swin_tiny_patch4_window7_224, vit_small_patch16_224)
- `OPTIMIZER` — adam, adamw, sgd
- `LR` — learning rate (try 1e-4 to 3e-3)
- `WEIGHT_DECAY` — (0 to 0.05)
- `BATCH_SIZE` — (8, 16, 32, 64)
- `LOSS_TYPE` — auto, bce, focal_bce, cross_entropy, focal, dice_bce
- `IMAGE_SIZE` — (224, 256, 384, 512)
- `DROPOUT` — (0.0 to 0.5)
- Augmentation settings in the training section

## Datasets Available
| Dataset | Task | Metric | Kaggle Slug |
|---------|------|--------|-------------|
| chestxray14 | Multi-label (14) | Mean AUC | nih-clinical-center/nih-chest-xrays |
| aptos | Classification (5) | Quad Kappa | mariaherrerot/aptos2019 |
| oct | Classification (4) | Accuracy | paultimothymooney/kermany2018 |
| chest_xray | Binary classification | AUC | paultimothymooney/chest-xray-pneumonia |
| brain_mri | Segmentation | Dice | mateuszbuda/lgg-mri-segmentation |
| retina_vessel | Segmentation | Dice | abdallahwagih/retina-blood-vessel |
| pneumothorax | Segmentation | Dice | vbookshelf/pneumothorax-chest-xray-images-and-masks |

## 3-Phase Strategy (see program.md for details)
- **Phase 1 (Experiments 1-20):** Architecture sweep — try DenseNet, EfficientNet, ConvNeXt, ViT, Swin
- **Phase 2 (Experiments 20-50):** Hyperparameter tuning on winning architecture
- **Phase 3 (Experiments 50-100):** Augmentation optimization + advanced tricks

## Rules
- Each experiment has a 5-minute wall-clock budget (TIME_BUDGET=300 in prepare.py)
- Only modify `train.py` — never touch `prepare.py`
- Always commit improvements with descriptive messages
- Always revert regressions: `git checkout train.py`
- Log results to results.tsv (auto-handled by train.py)
- Use mixed precision (FP16) — already enabled
- Start from pretrained ImageNet weights — already the default
