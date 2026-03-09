# autoresearch-vision

Autonomous AI research for image classification and segmentation. Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

An AI agent modifies model architectures, optimizers, and training strategies, then runs fixed 5-minute experiments to find the best configuration — all without human intervention.

## Supported Tasks

| Task | Dataset | Metric | Command |
|------|---------|--------|---------|
| Classification | CIFAR-10 | Top-1 Accuracy (%) | `uv run train.py` |
| Classification | CIFAR-100 | Top-1 Accuracy (%) | `uv run train.py --dataset cifar100` |
| Segmentation | Pascal VOC 2012 | Mean IoU (%) | `uv run train.py --task segmentation --dataset voc` |

## Architecture

The repo follows the same three-file design as autoresearch:

- **`prepare.py`** — Fixed constants, dataset downloads, dataloaders, evaluation functions. **Read-only** for agents.
- **`train.py`** — The file agents modify. Model architecture (ResNet classifier, U-Net segmenter), optimizer, training loop, hyperparameters.
- **`program.md`** — Instructions for the autonomous agent.

## Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download dataset(s)
uv run prepare.py                    # CIFAR-10 (default)
uv run prepare.py --dataset cifar100 # CIFAR-100
uv run prepare.py --dataset voc      # Pascal VOC 2012
uv run prepare.py --dataset all      # All datasets

# Run a single 5-minute experiment
uv run train.py
```

## Baseline Models

### Classification (CIFAR-10)
- **Architecture**: ResNet-style CNN with residual blocks
- **Optimizer**: SGD with momentum + weight decay
- **LR Schedule**: Warmup → constant → cosine decay
- **Augmentation**: Random crop + horizontal flip

### Segmentation (Pascal VOC)
- **Architecture**: U-Net with encoder-decoder skip connections
- **Optimizer**: SGD with momentum + weight decay
- **LR Schedule**: Warmup → constant → cosine decay
- **Augmentation**: Random horizontal flip + resize

## Autonomous Research

Read `program.md` for full instructions. The agent:

1. Establishes a baseline by running `train.py` unmodified
2. Modifies `train.py` with an experimental idea
3. Commits and runs a 5-minute experiment
4. Keeps improvements, discards regressions
5. Repeats indefinitely

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- `uv` package manager

## License

MIT
