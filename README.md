# autoresearch-medimage

**Autonomous AI research for medical imaging — bring any dataset, the agent runs experiments automatically.**

Point this framework at your medical imaging data (any folder of images, Kaggle dataset, DICOM directory, or CSV+images). It auto-detects everything — task type, classes, channels, splits — then the AI agent loop takes over: sweeping architectures, tuning hyperparameters, and optimizing augmentations in automated 5-minute experiments.

Built on [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Pulls best practices from nnU-Net, MONAI, and Kaggle winning solutions into the autoresearch 3-file pattern.

## The Idea

```bash
# You have medical imaging data. You want results.
uv run prepare.py --data /path/to/your/images
uv run train.py --dataset your_images

# The agent loop runs automatically:
# Architecture sweep → Hyperparameter tuning → Augmentation optimization
# Keeps improvements, reverts regressions, logs everything to results.tsv
```

No hardcoding. No dataset-specific scripts. Any medical imaging data works.

## Quick Start

```bash
# Install
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/mlungrengmail/autoresearch-medimage.git
cd autoresearch-medimage
uv sync

# Option A: Bring your own data
uv run prepare.py --data /path/to/your/images
uv run train.py --dataset your_images

# Option B: Use a built-in benchmark
uv run prepare.py --dataset chestxray14    # requires Kaggle credentials
uv run train.py --recipe chexnet --save-checkpoint

# Option C: Download from any Kaggle dataset
uv run prepare.py --kaggle username/dataset-slug
uv run train.py --dataset username_dataset_slug
```

## Bring Your Own Data

The framework auto-detects your dataset structure:

| Your Data Looks Like | What It Detects | Example |
|----------------------|-----------------|---------|
| `class_a/*.png`, `class_b/*.png` | Classification | Skin lesion folders |
| `images/` + `masks/` | Segmentation | Tumor boundaries |
| `images/` + `labels.csv` | Classification from CSV | DR grading |
| `train/class_a/`, `val/class_b/` | Pre-split classification | ImageNet-style |
| `*.dcm` files | DICOM → auto-convert PNG | Hospital PACS export |
| CSV with pipe-separated labels | Multi-label | ChestX-ray findings |

```bash
# Auto-detect everything
uv run prepare.py --data data/my_skin_lesions

# Override if needed
uv run prepare.py --data data/my_data --task segmentation
uv run prepare.py --data data/my_data --patient-column PatientID
uv run prepare.py --data data/my_data --compute-stats

# Creates dataset.json + splits.csv, then:
uv run train.py --dataset my_skin_lesions
```

### Patient-Level Splitting

Medical imaging requires patient-level splits to prevent data leakage (same patient's scans in both train and val). The framework handles this automatically:

- **From folder structure**: `patient_001/scan_a.png`, `patient_001/scan_b.png`
- **From CSV columns**: auto-detects `patient_id`, `PatientID`, `subject_id`
- **From DICOM tags**: extracts `PatientID` during conversion
- **Manual override**: `--patient-column MyColumn`

## Built-In Benchmarks (7 Datasets)

For reproducing published papers and benchmarking:

### Ophthalmology
| Dataset | Task | Classes | Metric | Recipe |
|---------|------|---------|--------|--------|
| APTOS 2019 (DR grading) | Classification | 5 | Quadratic Kappa | `aptos_winner`, `aptos_ordinal` |
| Retinal OCT | Classification | 4 | Accuracy | `retinal_oct_kermany` |
| Retinal Vessel | Segmentation | 2 | Dice | — |

### Radiology
| Dataset | Task | Classes | Metric | Recipe |
|---------|------|---------|--------|--------|
| **ChestX-ray14** (CheXNet) | Multi-label | 14 | Mean AUC | `chexnet`, `chexnet_advanced` |
| Chest X-ray Pneumonia | Classification | 2 | AUC | `chest_xray_pneumonia` |
| Brain MRI (LGG) | Segmentation | 2 | Dice | `brain_mri_unet`, `segmentation_advanced` |
| Pneumothorax | Segmentation | 2 | Dice | — |

```bash
uv run prepare.py --list    # See all datasets (built-in + your custom ones)
```

## Recipes — One-Command Paper Reproduction

Recipes are YAML configs in `recipes/` that reproduce published results:

```bash
uv run train.py --recipe chexnet                # CheXNet (Rajpurkar 2017)
uv run train.py --recipe chexnet_advanced        # + EMA, focal loss, grad clip
uv run train.py --recipe aptos_winner            # Kaggle DR 1st place
uv run train.py --recipe aptos_ordinal           # Ordinal regression for DR
uv run train.py --recipe retinal_oct_kermany     # Kermany 2018 OCT
uv run train.py --recipe brain_mri_unet          # Brain tumor segmentation
uv run train.py --recipe segmentation_advanced   # Pretrained U-Net + attention + Tversky
uv run train.py --recipe distillation_example    # Knowledge distillation
uv run train.py --recipe chexnet_biomedclip      # BiomedCLIP backbone
uv run train.py --recipe chexnet_raddino         # RAD-DINO backbone
uv run train.py --recipe retfound_aptos          # RETFound backbone

uv run train.py --list-recipes                   # See all recipes
```

## What the Agent Can Tune

```python
# Architecture — any timm model (700+ options)
BACKBONE = "efficientnet_b3"

# Training
OPTIMIZER = "adamw"           # adam, adamw, sgd
LR = 0.001                    # 1e-4 to 3e-3
WEIGHT_DECAY = 0.01
BATCH_SIZE = 32               # 8, 16, 32, 64
DROPOUT = 0.3

# Loss functions
LOSS_TYPE = "focal_bce"       # bce, focal_bce, cross_entropy, focal,
                              # dice_bce, tversky_bce, ordinal

# Learning rate schedule
LR_SCHEDULE = "cosine"        # cosine, cosine_restart, one_cycle

# Advanced
GRAD_CLIP_NORM = 1.0          # Gradient clipping
EMA = True                    # Exponential moving average
SWA = True                    # Stochastic weight averaging
CUTMIX = True                 # CutMix augmentation
LLRD = True                   # Layer-wise learning rate decay
DISTILLATION = True           # Knowledge distillation (teacher→student)

# Augmentation (albumentations)
ELASTIC_TRANSFORM = True
GRID_DISTORTION = True
COARSE_DROPOUT = True
```

## Tools

```bash
# Analysis
python tools/subgroup_analysis.py checkpoint.pth     # Per-class fairness analysis
python tools/calibration_plot.py checkpoint.pth       # Calibration + ECE
python tools/ensemble.py ckpt1.pth ckpt2.pth          # Multi-model ensemble
python tools/results_dashboard.py                      # HTML dashboard from results.tsv
python tools/gradcam_viewer.py checkpoint.pth          # Grad-CAM visualizations

# Export & conversion
python tools/onnx_export.py checkpoint.pth             # ONNX export + benchmarking
python tools/dicom_convert.py input.dcm output.png     # DICOM to PNG

# Research
python tools/paper_tracker.py --days 7 --with-data     # Find new papers
python tools/kaggle_autodetect.py <kaggle-slug>         # Auto-detect Kaggle dataset

# Documentation
python paper/generate_paper.py --compile                # Generate arXiv paper
python regulatory/generate_regulatory.py                # FDA 510(k) / CE MDR docs
```

## Architecture

Three-file design from [karpathy/autoresearch](https://github.com/karpathy/autoresearch):

| File | Role | Agent Editable? |
|------|------|-----------------|
| `prepare.py` | Dataset auto-detection, download, dataloaders, evaluation | **Read-only** |
| `train.py` | Model, optimizer, training loop, all experiments | **Yes — the only file agents modify** |
| `program.md` | Autonomous agent instructions | Reference |
| `recipes/` | YAML configs for paper reproduction | Add new ones |
| `tools/` | Analysis, export, visualization tools | Standalone |

### How Auto-Detection Works

```
uv run prepare.py --data /path/to/images
    |
    |-- Scan directory structure
    |     |-- DICOM files? -> Convert to PNG, extract PatientID
    |     |-- Class folders? -> Classification
    |     |-- images/ + masks/? -> Segmentation
    |     +-- CSV + images? -> Read labels from CSV
    |
    |-- Detect metadata
    |     |-- Task type, num_classes, class_names
    |     |-- Image channels (grayscale vs RGB)
    |     +-- Patient IDs (for leakage-free splits)
    |
    |-- Create splits
    |     |-- Patient-level GroupShuffleSplit (if patient IDs found)
    |     +-- Stratified split (fallback)
    |
    +-- Write manifest
          |-- dataset.json (registered in dynamic DATASETS dict)
          +-- splits.csv (train/val assignments)
```

## Autonomous Research Loop

The agent (guided by `program.md`):

1. Establishes baseline with a recipe or default config
2. **Phase 1 (1-20)**: Sweeps architectures (DenseNet, EfficientNet, ConvNeXt, ViT, Swin)
3. **Phase 2 (20-50)**: Tunes hyperparameters on winning architecture
4. **Phase 3 (50-100)**: Optimizes augmentation + advanced tricks (CutMix, SWA, EMA, LLRD)
5. Logs all results to `results.tsv`, commits improvements, reverts regressions
6. Each experiment has a 5-minute wall-clock budget — runs indefinitely until stopped

## GPU Setup

| Platform | GPU | Cost | Best For |
|----------|-----|------|----------|
| **Kaggle Notebooks** | T4 x2 | Free (30h/wk) | Kaggle competitions |
| **Google Colab** | T4 | Free (limited) | Quick experiments |
| **Lightning AI** | T4/A10G | Free credits | Persistent dev |
| Lambda/RunPod/Vast | A100/H100 | $0.80-3/hr | Overnight runs |

See [`GPU_SETUP.md`](GPU_SETUP.md) for detailed setup instructions.

## Dependencies

Core: `torch`, `torchvision`, `timm`, `albumentations`, `scikit-learn`, `opencv-python-headless`, `kaggle`

Optional: `onnx` + `onnxruntime` (export), `pydicom` (DICOM), `pyyaml` (config overrides)

```bash
uv sync                           # Core dependencies
uv sync --extra deploy             # + ONNX export
uv sync --extra dicom              # + DICOM support
uv sync --extra all                # Everything
```

## License

MIT
