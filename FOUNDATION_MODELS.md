# Medical Foundation Models — Backbone Guide

## Overview

Beyond standard ImageNet-pretrained backbones (via `timm`), this framework supports **medical foundation models** from HuggingFace that are pretrained on large-scale medical imaging datasets. These often outperform ImageNet-pretrained models on medical tasks.

## Supported Foundation Models

### Classification / Feature Extraction

| Model | Source | Pretrained On | Best For | `BACKBONE` value |
|-------|--------|---------------|----------|-----------------|
| **BiomedCLIP** | Microsoft | PMC-15M (PubMed) | General medical | `hf:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` |
| **MedCLIP** | Community | MIMIC-CXR + CheXpert | Chest X-ray | `hf:flaviagiammarino/pubmed-clip-vit-base-patch32` |
| **RETFound** | UCL | 1.6M retinal images | Ophthalmology | `hf:rmaphoh/RETFound_MAE` |
| **CheXzero** | Stanford | CheXpert (224K CXR) | Chest X-ray | `hf:stanford-crfm/CheXzero` |
| **UNI** | Harvard | 100K+ WSI patches | Pathology | `hf:MahmoodLab/UNI` |
| **PLIP** | OpenAI-style | PathLaion-2M | Pathology | `hf:vinid/plip` |
| **RAD-DINO** | Microsoft | Radiology images | Radiology | `hf:microsoft/rad-dino` |

### Google MedGemma (Multimodal)
| Model | Pretrained On | Best For | `BACKBONE` value |
|-------|---------------|----------|-----------------|
| **MedGemma** | Medical images + text | General medical | `hf:google/medgemma-4b-it` |

> **Note**: MedGemma is a multimodal LLM — using it as a backbone requires extracting the vision encoder. See usage below.

### Microsoft MedImageInsight
| Model | Pretrained On | Best For | `BACKBONE` value |
|-------|---------------|----------|-----------------|
| **MedImageInsight** | Large-scale medical | General medical | `hf:microsoft/medimageinsight` |

## Usage

### Standard timm Backbones (Default)
```python
# In train.py, set:
BACKBONE = "densenet121"            # CheXNet default
BACKBONE = "efficientnet_b3"        # Best value for medical imaging
BACKBONE = "convnext_small"         # Modern CNN
BACKBONE = "swin_tiny_patch4_window7_224"  # Vision Transformer
```

### HuggingFace Foundation Models
```python
# Prefix with "hf:" to use HuggingFace models
BACKBONE = "hf:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BACKBONE = "hf:rmaphoh/RETFound_MAE"
BACKBONE = "hf:microsoft/rad-dino"
```

The `MedicalClassifier` in `train.py` automatically:
1. Detects the `hf:` prefix
2. Downloads the model from HuggingFace Hub
3. Extracts the vision encoder
4. Attaches a task-specific classification head
5. Freezes/unfreezes layers as appropriate

### Recipe Example
```yaml
# recipes/chexnet_biomedclip.yaml
paper:
  title: "CheXNet with BiomedCLIP backbone"
  dataset: "ChestX-ray14"

dataset: chestxray14
model:
  backbone: "hf:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
  pretrained: true
  dropout: 0.2

training:
  batch_size: 16
  optimizer: adamw
  learning_rate: 0.0001
  weight_decay: 0.01
```

## Installation

Foundation models require additional dependencies:
```bash
# For HuggingFace models
pip install transformers huggingface_hub

# For some models that use safetensors
pip install safetensors
```

Or add to your `uv sync`:
```bash
uv add transformers huggingface_hub
```

## Recommended Combinations

### Chest X-ray Tasks
1. `hf:microsoft/BiomedCLIP-*` — strong general medical baseline
2. `hf:microsoft/rad-dino` — radiology-specific DINO
3. `densenet121` — CheXNet classic (fast, well-validated)
4. `efficientnet_b3` — best efficiency/accuracy tradeoff

### Ophthalmology Tasks
1. `hf:rmaphoh/RETFound_MAE` — **best for retinal imaging** (pretrained on 1.6M fundus/OCT)
2. `efficientnet_b3` — Kaggle winner backbone for DR grading
3. `hf:microsoft/BiomedCLIP-*` — general medical

### Pathology Tasks
1. `hf:MahmoodLab/UNI` — pretrained on 100K+ WSI patches
2. `hf:vinid/plip` — pathology-specific CLIP
3. `hf:microsoft/BiomedCLIP-*` — general medical

### Brain MRI / CT Tasks
1. `hf:microsoft/rad-dino` — radiology-specific
2. `hf:microsoft/BiomedCLIP-*` — general medical
3. `convnext_small` — strong modern CNN baseline

## Performance Tips

- **Learning rate**: Use lower LR (1e-5 to 5e-5) for foundation models vs. ImageNet-pretrained (1e-4 to 1e-3)
- **Freezing**: Foundation models often work best with frozen encoder + trained head for small datasets
- **Image size**: Match the model's pretrained resolution (usually 224px for ViT-based)
- **Batch size**: Foundation models are larger — reduce batch size if OOM
