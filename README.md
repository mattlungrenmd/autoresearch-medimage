# autoresearch-medvision

Autonomous AI research for medical image classification and segmentation. Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for ophthalmology and radiology tasks.

An AI agent autonomously experiments with model architectures, optimizers, and training strategies on medical imaging datasets from Kaggle — running fixed 5-minute experiments and keeping only improvements.

**Default baseline**: Reproduces [CheXNet](https://arxiv.org/abs/1711.05225) (Rajpurkar et al., 2017) — DenseNet-121 on NIH ChestX-ray14 for 14-pathology chest X-ray classification.

## Supported Tasks

### Ophthalmology
| Dataset | Task | Classes | Metric | Command |
|---------|------|---------|--------|---------|
| APTOS 2019 (DR grading) | Classification | 5 | Quadratic Kappa | `uv run train.py --dataset aptos` |
| Retinal OCT | Classification | 4 | Accuracy | `uv run train.py --dataset oct` |
| Retinal Vessel | Segmentation | 2 | Dice | `uv run train.py --dataset retina_vessel` |

### Radiology
| Dataset | Task | Classes | Metric | Command |
|---------|------|---------|--------|---------|
| ChestX-ray14 (CheXNet) | Multi-label | 14 | Mean AUC | `uv run train.py` |
| Chest X-ray Pneumonia | Classification | 2 | AUC | `uv run train.py --dataset chest_xray` |
| Brain MRI (LGG) | Segmentation | 2 | Dice | `uv run train.py --dataset brain_mri` |
| Pneumothorax | Segmentation | 2 | Dice | `uv run train.py --dataset pneumothorax` |

## Architecture

Three-file design (same as autoresearch):

- **`prepare.py`** — Kaggle dataset automation, dataloaders, fixed evaluation metrics. **Read-only** for agents.
- **`train.py`** — The file agents modify. CheXNet baseline (DenseNet-121), optimizer, training loop. Supports any `timm` backbone.
- **`program.md`** — Autonomous agent instructions.

## Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set up Kaggle API credentials
# 1. Go to https://www.kaggle.com/settings → API → Create New Token
# 2. Place kaggle.json at ~/.kaggle/kaggle.json
# 3. chmod 600 ~/.kaggle/kaggle.json

# List available datasets
uv run prepare.py --list

# Download dataset(s) from Kaggle
uv run prepare.py --dataset chestxray14    # CheXNet (NIH, ~45GB)
uv run prepare.py --dataset aptos          # DR grading (~5GB)
uv run prepare.py --dataset oct            # Retinal OCT (~6GB)
uv run prepare.py --dataset chest_xray     # Pneumonia (~1.3GB)
uv run prepare.py --dataset brain_mri      # Brain MRI (~200MB)
uv run prepare.py --dataset all            # Everything

# Run a single 5-minute experiment
uv run train.py                            # CheXNet on ChestX-ray14
```

## CheXNet Reproduction (Default Baseline)

Reproduces the CheXNet paper ([arxiv 1711.05225](https://arxiv.org/abs/1711.05225)):

- **Model**: DenseNet-121 pretrained on ImageNet
- **Task**: 14-pathology multi-label classification on NIH ChestX-ray14
- **Loss**: BCEWithLogitsLoss with class weights
- **Optimizer**: Adam (lr=0.001)
- **Metric**: Mean AUC across 14 pathologies (paper: ~84.1%)
- **Input**: 224x224 grayscale chest X-rays (expanded to 3-channel for pretrained model)

The agent can then iterate on this baseline: try EfficientNet backbones, focal loss, better augmentation, etc.

## Dependencies

- `torch` + `torchvision` — PyTorch with CUDA
- `timm` — 700+ pretrained vision models (EfficientNet, ConvNeXt, ViT, etc.)
- `kaggle` — automated dataset download
- `albumentations` — medical imaging augmentations
- `opencv-python-headless` — image processing (CLAHE, etc.)
- `scikit-learn` — evaluation metrics (AUC, kappa)
- `pandas`, `numpy`, `Pillow`, `matplotlib`

## Autonomous Research Loop

See `program.md` for full instructions. The agent:

1. Establishes CheXNet baseline on ChestX-ray14
2. Experiments with architecture changes (backbone, attention, loss)
3. Keeps improvements, discards regressions
4. Logs all results to `results.tsv`
5. Repeats indefinitely until stopped

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- `uv` package manager
- Kaggle API credentials (`~/.kaggle/kaggle.json`)

## License

MIT
