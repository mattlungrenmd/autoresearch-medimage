# autoresearch-medimage

Autonomous medical imaging AI research — bring any dataset, the agent runs
experiments automatically. Dataset-agnostic framework with auto-detection,
patient-level splits, DICOM support, and 7 built-in benchmarks. Built on
[karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## How It Works

Point an AI agent (Claude, Codex) at this repo. The agent edits `train.py`,
runs fixed-budget experiments, logs results to `results.tsv`, keeps
improvements, and discards regressions — automatically.

```
prepare.py  — dataset download, data loading, evaluation (READ-ONLY for agents)
train.py    — the experiment surface (agents modify this)
program.md  — agent instructions for the 3-phase research loop
results.tsv — experiment ledger (auto-generated, not tracked)
```

## Quick Start

### Bring your own dataset

```bash
# Auto-detect task type, classes, and splits from any image directory
uv run prepare.py --data /path/to/your/images

# Train on it
uv run train.py --dataset your_dataset
```

Or use the manual contract:

1. Create `data/<dataset>/dataset.yaml` and `data/<dataset>/splits.csv`
2. Run `uv run prepare.py --dataset your_dataset`
3. Run `uv run train.py --dataset your_dataset`

See [docs/BRING_YOUR_OWN_DATA.md](docs/BRING_YOUR_OWN_DATA.md) for details.

### Built-in demo: CheXNet

```bash
uv run prepare.py --dataset chestxray14
uv run train.py --recipe chexnet --save-checkpoint
uv run python tools/finalize_run.py --dataset chestxray14
```

See [docs/CHEXNET_HELLO_WORLD.md](docs/CHEXNET_HELLO_WORLD.md) for the full
walkthrough, including paper and regulatory document generation.

### Smoke test (no GPU, no download)

```bash
uv run prepare.py --dataset toy_binary
uv run train.py --recipe toy_binary_smoke --time-budget 5
```

## Built-In Datasets

| Dataset | Task | Metric | Source |
|---------|------|--------|--------|
| `chestxray14` | Multi-label (14 pathologies) | Mean AUC | NIH ChestX-ray14 |
| `aptos` | Classification (5 grades) | Quadratic Kappa | APTOS 2019 |
| `oct` | Classification (4 classes) | Accuracy | Kermany OCT |
| `chest_xray` | Binary classification | AUC | Chest X-ray Pneumonia |
| `brain_mri` | Segmentation | Dice | LGG MRI |
| `retina_vessel` | Segmentation | Dice | Retinal Blood Vessel |
| `pneumothorax` | Segmentation | Dice | SIIM-ACR |

```bash
uv run prepare.py --list          # list all datasets
uv run train.py --list-recipes    # list all recipes
```

## Features

- **Auto-detection**: point `--data` at any image directory and the framework
  detects task type, class labels, channels, and creates train/val splits
- **Patient-level splitting**: prevents data leakage with `--patient-column`
- **Advanced training**: EMA, CutMix, MixUp, LLRD, TTA, SWA, knowledge
  distillation, gradient clipping, ordinal regression
- **Pretrained U-Net**: segmentation with any timm encoder backbone + attention gates
- **18 recipes**: from simple baselines to advanced pipelines (distillation,
  foundation models, ordinal regression)
- **Session-scoped runs**: `--session` isolates overnight experiments with seeded dataloaders
- **Paper generation**: generate LaTeX manuscripts from checkpoint data and results
- **Regulatory documents**: generate FDA 510(k), CE/EU MDR, and UKCA planning documents from experiment results
- **Kaggle auto-detect**: `tools/kaggle_autodetect.py` for new Kaggle datasets
- **Paper tracker**: `tools/paper_tracker.py` to monitor new publications

## Agent Workflow

The intended 3-phase loop:

1. **Architecture sweep** (experiments 1-20): try DenseNet, EfficientNet, ConvNeXt, ViT, Swin
2. **Hyperparameter tuning** (experiments 20-50): optimize LR, batch size, augmentation on the winning architecture
3. **Advanced tricks** (experiments 50-100): EMA, CutMix, label smoothing, distillation

See [program.md](program.md) and
[docs/USING_WITH_CLAUDE_CODEX.md](docs/USING_WITH_CLAUDE_CODEX.md).

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full plan. Key upcoming milestones:

- **v1.1**: ONNX export, extended benchmarks
- **v2.0**: HuggingFace model upload, multi-GPU, automated arXiv pipeline
- **v3.0**: MONAI backend for DICOM/NIfTI and 3D volume training

## License

MIT. See [LICENSE](LICENSE).
