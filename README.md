# autoresearch-medimage

**Autonomous AI research for medical image classification and segmentation.** An AI agent iteratively experiments with architectures, optimizers, losses, and augmentations — running fixed 5-minute experiments overnight and keeping only improvements.

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for ophthalmology and radiology. Automates Kaggle dataset download and includes reproducible "recipes" for famous medical imaging papers.

## What This Does

1. **Downloads medical imaging datasets from Kaggle** with one command
2. **Reproduces published papers** via recipe configs (CheXNet, APTOS winner, etc.)
3. **Autonomously searches** architectures + hyperparameters overnight
4. **Benchmarks against FDA-cleared devices** with comparison tables
5. **Saves best models** with checkpoints

```bash
# Reproduce CheXNet in one command
uv run prepare.py --dataset chestxray14
uv run train.py --recipe chexnet --save-checkpoint
```

## Supported Datasets

### Ophthalmology
| Dataset | Task | Classes | Metric | Recipe |
|---------|------|---------|--------|--------|
| APTOS 2019 (DR grading) | Classification | 5 | Quadratic Kappa | `aptos_winner` |
| Retinal OCT | Classification | 4 | Accuracy | `retinal_oct_kermany` |
| Retinal Vessel | Segmentation | 2 | Dice | — |

### Radiology
| Dataset | Task | Classes | Metric | Recipe |
|---------|------|---------|--------|--------|
| **ChestX-ray14** (CheXNet) | Multi-label | 14 | Mean AUC | `chexnet` |
| Chest X-ray Pneumonia | Classification | 2 | AUC | `chest_xray_pneumonia` |
| Brain MRI (LGG) | Segmentation | 2 | Dice | `brain_mri_unet` |
| Pneumothorax | Segmentation | 2 | Dice | — |

## Quick Start

```bash
# Install
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Kaggle credentials (one-time)
# 1. https://www.kaggle.com/settings → API → Create New Token
# 2. Place at ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json

# Download + train
uv run prepare.py --dataset chestxray14
uv run train.py --recipe chexnet --save-checkpoint

# List what's available
uv run prepare.py --list              # datasets
uv run train.py --list-recipes        # paper recipes
```

## Recipes — One-Command Paper Reproduction

Recipes are YAML configs in `recipes/` that reproduce published results:

```bash
uv run train.py --recipe chexnet              # CheXNet (Rajpurkar 2017) — DenseNet-121
uv run train.py --recipe chexnet_improved     # CheXNet + AdamW + focal loss
uv run train.py --recipe aptos_winner         # Kaggle DR 1st place approach
uv run train.py --recipe retinal_oct_kermany  # Kermany 2018 OCT classification
uv run train.py --recipe chest_xray_pneumonia # Pneumonia detection
uv run train.py --recipe brain_mri_unet       # Brain tumor U-Net segmentation
```

Each recipe specifies backbone, optimizer, loss, augmentation, and links to the original paper with published metrics to beat.

## Architecture Search

The agent tries **both** different model architectures AND hyperparameter tuning. Any [timm](https://github.com/huggingface/pytorch-image-models) model works as a backbone:

```python
# In train.py, just change BACKBONE:
BACKBONE = "densenet121"          # CheXNet default
BACKBONE = "efficientnet_b3"      # Usually best value for medical imaging
BACKBONE = "convnext_small"       # Strong modern CNN
BACKBONE = "swin_tiny_patch4_window7_224"  # Vision Transformer
BACKBONE = "vit_small_patch16_224"         # ViT
```

The experiment loop in `program.md` guides the agent through: architecture sweep → hyperparameter tuning → augmentation optimization → advanced tricks.

## Benchmarks vs Published & FDA-Cleared

See `benchmarks/` for comparison tables:

- [`benchmarks/chestxray14.md`](benchmarks/chestxray14.md) — CheXNet vs Qure.ai, Lunit, Annalise
- [`benchmarks/diabetic_retinopathy.md`](benchmarks/diabetic_retinopathy.md) — vs IDx-DR, EyeArt, AEYE Health
- [`benchmarks/segmentation.md`](benchmarks/segmentation.md) — vs nnU-Net, MedSAM, Viz.ai, Aidoc

## GPU Options

See [`GPU_SETUP.md`](GPU_SETUP.md) for detailed setup. Quick summary:

| Platform | GPU | Cost | Best For |
|----------|-----|------|----------|
| **Kaggle Notebooks** | T4 x2 | Free (30h/wk) | Kaggle competitions |
| **Google Colab** | T4 | Free (limited) | Quick experiments |
| **Lightning AI** | T4/A10G | Free credits | Persistent dev |
| Lambda/RunPod/Vast | A100/H100 | $0.80-3/hr | Overnight runs |

## Architecture

Three-file design (same as [karpathy/autoresearch](https://github.com/karpathy/autoresearch)):

| File | Role | Agent Editable? |
|------|------|-----------------|
| `prepare.py` | Kaggle download, dataloaders, evaluation metrics | **Read-only** |
| `train.py` | Model architecture, optimizer, training loop | **Yes — the only file agents modify** |
| `program.md` | Autonomous agent instructions | Reference |
| `recipes/` | YAML configs for paper reproduction | Add new ones |
| `benchmarks/` | Comparison tables vs published results | Update with your results |

## Dependencies

- `torch` + `torchvision` — PyTorch with CUDA
- `timm` — 700+ pretrained vision models
- `kaggle` — automated dataset download
- `albumentations` — medical imaging augmentations
- `opencv-python-headless` — image processing (CLAHE)
- `scikit-learn` — evaluation metrics (AUC, kappa)

## Autonomous Research Loop

The agent (see `program.md`):

1. Establishes baseline with a recipe
2. **Phase 1**: Sweeps architectures (DenseNet → EfficientNet → ConvNeXt → ViT)
3. **Phase 2**: Tunes hyperparameters on winning architecture
4. **Phase 3**: Optimizes augmentation + training tricks
5. Logs all results to `results.tsv`, keeps improvements, discards regressions
6. Runs indefinitely until stopped — ~100 experiments per night

## Roadmap

- [x] Auto-detect task from any Kaggle competition URL (`tools/kaggle_autodetect.py`)
- [x] Paper tracker — auto-pull new medical imaging papers + datasets (`tools/paper_tracker.py`)
- [x] FDA/CE/UKCA regulatory documentation generator (`regulatory/generate_regulatory.py`)
- [x] Automated arXiv paper generation (`paper/generate_paper.py`)
- [ ] HuggingFace model zoo upload
- [ ] Multi-GPU / distributed training support
- [ ] ONNX export for edge deployment

## License

MIT
