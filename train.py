"""
Autoresearch-MedImage training script.

This is the experiment surface for the Karpathy-style inner loop:
- fixed time budget
- mutate train.py, not prepare.py
- log every run to results.tsv
- keep or discard externally after reading the run metric

Usage:
    uv run train.py                                    # CheXNet on ChestX-ray14 (default)
    uv run train.py --recipe chexnet                   # Load recipe config
    uv run train.py --recipe-file logs/run/cand.yaml   # Load a session-local overlay recipe
    uv run train.py --recipe aptos_winner              # Kaggle-winning DR approach
    uv run train.py --dataset aptos                    # DR grading (manual config)
    uv run train.py --dataset brain_mri                # Brain tumor segmentation
    uv run train.py --recipe chexnet --epochs 10       # Optional full-train mode
    uv run train.py --save-checkpoint                  # Save best model
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import random
import csv
import gc
import json
import math
import time
import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torchvision.models as models
import timm

from prepare import (
    DATA_DIR,
    NUM_WORKERS,
    TIME_BUDGET,
    MedicalClassificationDataset,
    MedicalMultilabelDataset,
    MedicalSegmentationDataset,
    apply_clahe,
    get_dataset_config,
)

# ---------------------------------------------------------------------------
# Recipe loader — load configs from recipes/*.yaml
# ---------------------------------------------------------------------------

def load_recipe_file(recipe_path):
    """Load a recipe YAML file and return config dict."""
    try:
        import yaml
    except ImportError:
        yaml = None

    if not recipe_path.exists():
        raise FileNotFoundError(f"Recipe file not found: {recipe_path}")

    if yaml is not None:
        with open(recipe_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    else:
        raise RuntimeError("PyYAML is required for recipe loading. Run: uv sync")


def load_recipe(recipe_name):
    """Load a recipe YAML file from recipes/ and return config dict."""
    recipe_dir = Path(__file__).parent / "recipes"
    recipe_path = recipe_dir / f"{recipe_name}.yaml"

    if not recipe_path.exists():
        available = [f.stem for f in recipe_dir.glob("*.yaml")]
        print(f"Recipe '{recipe_name}' not found.")
        print(f"Available recipes: {', '.join(sorted(available))}")
        exit(1)

    return load_recipe_file(recipe_path)


def apply_recipe(recipe):
    """Apply recipe config to global hyperparameters. Returns overridden values."""
    global DATASET, BACKBONE, PRETRAINED, DROPOUT, BATCH_SIZE, LEARNING_RATE
    global OPTIMIZER, WEIGHT_DECAY, WARMUP_RATIO, WARMDOWN_RATIO
    global USE_CLASS_WEIGHTS, LABEL_SMOOTHING, LOSS_TYPE, FOCAL_GAMMA
    global SEG_BATCH_SIZE, SEG_BASE_CHANNELS, IMAGE_SIZE_OVERRIDE
    global AUG_HORIZONTAL_FLIP, AUG_VERTICAL_FLIP, AUG_ROTATION_DEGREES
    global AUG_COLOR_JITTER, AUG_COLOR_JITTER_BRIGHTNESS, AUG_COLOR_JITTER_CONTRAST
    global AUG_USE_CLAHE
    global GRAD_ACCUM_STEPS, USE_EMA, EMA_DECAY, USE_TTA, USE_COMPILE
    global USE_MIXUP, MIXUP_ALPHA, USE_GRADUAL_UNFREEZE, UNFREEZE_AFTER_RATIO
    global DISCRIMINATIVE_LR, BACKBONE_LR_FACTOR, GRAD_CLIP_NORM
    global COMPUTE_CI, CI_BOOTSTRAPS
    global USE_CUTMIX, CUTMIX_ALPHA, LR_SCHEDULE, USE_SWA, SWA_START_RATIO
    global OPTIMIZE_THRESHOLDS
    global TVERSKY_ALPHA, TVERSKY_BETA
    global SEG_BACKBONE, SEG_USE_ATTENTION
    global USE_LLRD, LLRD_DECAY
    global USE_DISTILLATION, TEACHER_CHECKPOINT, DISTILL_ALPHA, DISTILL_TEMPERATURE
    global ELASTIC_TRANSFORM, GRID_DISTORTION, COARSE_DROPOUT, COARSE_DROPOUT_RATIO

    if "dataset" in recipe:
        DATASET = recipe["dataset"]

    model = recipe.get("model", {})
    if isinstance(model, dict):
        BACKBONE = model.get("backbone", BACKBONE)
        PRETRAINED = model.get("pretrained", PRETRAINED)
        DROPOUT = model.get("dropout", DROPOUT)
        if "base_channels" in model:
            SEG_BASE_CHANNELS = model["base_channels"]

    training = recipe.get("training", {})
    if isinstance(training, dict):
        BATCH_SIZE = training.get("batch_size", BATCH_SIZE)
        LEARNING_RATE = training.get("learning_rate", LEARNING_RATE)
        OPTIMIZER = training.get("optimizer", OPTIMIZER)
        WEIGHT_DECAY = training.get("weight_decay", WEIGHT_DECAY)
        WARMUP_RATIO = training.get("warmup_ratio", WARMUP_RATIO)
        WARMDOWN_RATIO = training.get("warmdown_ratio", WARMDOWN_RATIO)
        LABEL_SMOOTHING = training.get("label_smoothing", LABEL_SMOOTHING)

    loss = recipe.get("loss", {})
    if isinstance(loss, dict):
        LOSS_TYPE = loss.get("type", LOSS_TYPE)
        FOCAL_GAMMA = loss.get("focal_gamma", FOCAL_GAMMA)
        USE_CLASS_WEIGHTS = loss.get("class_weights", USE_CLASS_WEIGHTS)

    augmentation = recipe.get("augmentation", {})
    if isinstance(augmentation, dict):
        AUG_HORIZONTAL_FLIP = augmentation.get("horizontal_flip", AUG_HORIZONTAL_FLIP)
        AUG_VERTICAL_FLIP = augmentation.get("vertical_flip", AUG_VERTICAL_FLIP)
        AUG_ROTATION_DEGREES = augmentation.get("rotation", AUG_ROTATION_DEGREES)
        AUG_COLOR_JITTER = augmentation.get("color_jitter", AUG_COLOR_JITTER)
        AUG_COLOR_JITTER_BRIGHTNESS = augmentation.get(
            "color_jitter_brightness", AUG_COLOR_JITTER_BRIGHTNESS
        )
        AUG_COLOR_JITTER_CONTRAST = augmentation.get(
            "color_jitter_contrast", AUG_COLOR_JITTER_CONTRAST
        )
        AUG_USE_CLAHE = augmentation.get("clahe", AUG_USE_CLAHE)

    IMAGE_SIZE_OVERRIDE = recipe.get("image_size", None)

    # Advanced training settings
    advanced = recipe.get("advanced", {})
    if isinstance(advanced, dict):
        GRAD_ACCUM_STEPS = advanced.get("grad_accum_steps", GRAD_ACCUM_STEPS)
        USE_EMA = advanced.get("ema", USE_EMA)
        EMA_DECAY = advanced.get("ema_decay", EMA_DECAY)
        USE_TTA = advanced.get("tta", USE_TTA)
        USE_COMPILE = advanced.get("compile", USE_COMPILE)
        USE_MIXUP = advanced.get("mixup", USE_MIXUP)
        MIXUP_ALPHA = advanced.get("mixup_alpha", MIXUP_ALPHA)
        USE_GRADUAL_UNFREEZE = advanced.get("gradual_unfreeze", USE_GRADUAL_UNFREEZE)
        UNFREEZE_AFTER_RATIO = advanced.get("unfreeze_after_ratio", UNFREEZE_AFTER_RATIO)
        DISCRIMINATIVE_LR = advanced.get("discriminative_lr", DISCRIMINATIVE_LR)
        BACKBONE_LR_FACTOR = advanced.get("backbone_lr_factor", BACKBONE_LR_FACTOR)
        GRAD_CLIP_NORM = advanced.get("grad_clip_norm", GRAD_CLIP_NORM)
        COMPUTE_CI = advanced.get("compute_ci", COMPUTE_CI)
        CI_BOOTSTRAPS = advanced.get("ci_bootstraps", CI_BOOTSTRAPS)
        USE_CUTMIX = advanced.get("cutmix", USE_CUTMIX)
        CUTMIX_ALPHA = advanced.get("cutmix_alpha", CUTMIX_ALPHA)
        LR_SCHEDULE = advanced.get("lr_schedule", LR_SCHEDULE)
        USE_SWA = advanced.get("swa", USE_SWA)
        SWA_START_RATIO = advanced.get("swa_start_ratio", SWA_START_RATIO)
        OPTIMIZE_THRESHOLDS = advanced.get("optimize_thresholds", OPTIMIZE_THRESHOLDS)
        USE_LLRD = advanced.get("llrd", USE_LLRD)
        LLRD_DECAY = advanced.get("llrd_decay", LLRD_DECAY)
        USE_DISTILLATION = advanced.get("distillation", USE_DISTILLATION)
        TEACHER_CHECKPOINT = advanced.get("teacher_checkpoint", TEACHER_CHECKPOINT)
        DISTILL_ALPHA = advanced.get("distill_alpha", DISTILL_ALPHA)
        DISTILL_TEMPERATURE = advanced.get("distill_temperature", DISTILL_TEMPERATURE)

    # Tversky loss parameters from loss section
    if isinstance(loss, dict):
        TVERSKY_ALPHA = loss.get("tversky_alpha", TVERSKY_ALPHA)
        TVERSKY_BETA = loss.get("tversky_beta", TVERSKY_BETA)

    # Model section: segmentation backbone and attention
    if isinstance(model, dict):
        SEG_BACKBONE = model.get("seg_backbone", SEG_BACKBONE)
        SEG_USE_ATTENTION = model.get("seg_attention", SEG_USE_ATTENTION)

    # Augmentation: albumentations extras
    if isinstance(augmentation, dict):
        ELASTIC_TRANSFORM = augmentation.get("elastic_transform", ELASTIC_TRANSFORM)
        GRID_DISTORTION = augmentation.get("grid_distortion", GRID_DISTORTION)
        COARSE_DROPOUT = augmentation.get("coarse_dropout", COARSE_DROPOUT)
        COARSE_DROPOUT_RATIO = augmentation.get("coarse_dropout_ratio", COARSE_DROPOUT_RATIO)

    paper = recipe.get("paper", {})
    if isinstance(paper, dict) and paper:
        print(f"Recipe: {paper.get('title', recipe.get('dataset', 'unknown'))}")
        if "reported_metric" in paper:
            print(f"Published result: {paper['reported_metric']}")
        if "arxiv" in paper:
            print(f"Paper: https://arxiv.org/abs/{paper['arxiv']}")
        print()


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Task selection — default: CheXNet reproduction
DATASET = "chestxray14"       # "chestxray14", "aptos", "oct", "chest_xray", "brain_mri", etc.

# Model architecture
BACKBONE = "densenet121"      # CheXNet uses DenseNet-121
PRETRAINED = True             # ImageNet pretrained (critical for medical imaging)
DROPOUT = 0.2                 # dropout before final classifier

# Optimization — CheXNet paper settings
BATCH_SIZE = 16               # CheXNet used 16
LEARNING_RATE = 0.001         # Adam LR
OPTIMIZER = "adam"            # "adam", "adamw", "sgd"
WEIGHT_DECAY = 1e-4           # L2 regularization
WARMUP_RATIO = 0.02           # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.3          # fraction of time budget for cosine decay

# Loss
USE_CLASS_WEIGHTS = True      # handle class imbalance
LABEL_SMOOTHING = 0.0         # label smoothing for classification
LOSS_TYPE = "auto"            # "auto", "bce", "cross_entropy", "focal", "dice_bce"
FOCAL_GAMMA = 2.0             # gamma for focal loss

# Segmentation-specific
SEG_BATCH_SIZE = 8            # batch size for segmentation (larger images)
SEG_BASE_CHANNELS = 64        # U-Net base channels

# Image size override (None = use dataset default)
IMAGE_SIZE_OVERRIDE = None

# Augmentation controls
AUG_HORIZONTAL_FLIP = True
AUG_VERTICAL_FLIP = True
AUG_ROTATION_DEGREES = 15
AUG_COLOR_JITTER = True
AUG_COLOR_JITTER_BRIGHTNESS = 0.1
AUG_COLOR_JITTER_CONTRAST = 0.1
AUG_USE_CLAHE = False

# Advanced training
GRAD_ACCUM_STEPS = 1          # gradient accumulation steps
USE_EMA = False               # exponential moving average of model weights
EMA_DECAY = 0.999             # EMA decay rate
USE_TTA = False               # test-time augmentation (horizontal flip)
USE_COMPILE = True            # torch.compile on supported CUDA platforms
USE_MIXUP = False             # MixUp regularization
MIXUP_ALPHA = 0.2             # MixUp interpolation strength
USE_GRADUAL_UNFREEZE = False  # freeze backbone initially, then unfreeze
UNFREEZE_AFTER_RATIO = 0.1   # fraction of time budget before unfreezing backbone
DISCRIMINATIVE_LR = False     # lower LR for backbone, higher for head
BACKBONE_LR_FACTOR = 0.1      # backbone LR = LEARNING_RATE * this factor
GRAD_CLIP_NORM = 1.0          # gradient clipping max norm (0 = disabled)
COMPUTE_CI = True             # compute 95% bootstrap confidence intervals
CI_BOOTSTRAPS = 1000          # number of bootstrap iterations for CI

# Albumentations augmentation
ELASTIC_TRANSFORM = False     # elastic deformations
GRID_DISTORTION = False       # grid-based distortion
COARSE_DROPOUT = False        # CoarseDropout (CutOut-style)
COARSE_DROPOUT_RATIO = 0.1   # fraction of image to drop

# CutMix and scheduling
USE_CUTMIX = False            # CutMix augmentation
CUTMIX_ALPHA = 1.0            # CutMix Beta distribution parameter
LR_SCHEDULE = "cosine"        # "cosine", "cosine_restart", "one_cycle"
USE_SWA = False               # stochastic weight averaging
SWA_START_RATIO = 0.7         # start SWA after this fraction of training
TVERSKY_ALPHA = 0.7           # Tversky loss: FP weight
TVERSKY_BETA = 0.3            # Tversky loss: FN weight
OPTIMIZE_THRESHOLDS = False   # per-class sigmoid threshold optimization

# Architecture upgrades
SEG_BACKBONE = ""             # pretrained encoder for U-Net (any timm model, "" = scratch)
SEG_USE_ATTENTION = False     # attention gates in U-Net decoder
USE_LLRD = False              # layer-wise learning rate decay
LLRD_DECAY = 0.9              # decay factor per layer

# Knowledge distillation
USE_DISTILLATION = False      # knowledge distillation from teacher
TEACHER_CHECKPOINT = ""       # path to teacher model checkpoint
DISTILL_ALPHA = 0.5           # weight for distillation loss vs task loss
DISTILL_TEMPERATURE = 3.0     # softmax temperature for distillation

# Checkpointing
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.tsv")
RESULTS_HEADER = [
    "timestamp",
    "commit",
    "session",
    "dataset",
    "recipe",
    "backbone",
    "optimizer",
    "loss",
    "image_size",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "dropout",
    "augmentations",
    "seed",
    "time_budget_seconds",
    "budget_mode",
    "epoch_budget",
    "metric_name",
    "metric",
    "memory_gb",
    "status",
    "checkpoint_path",
    "description",
    "candidate_id",
    "parent_id",
    "search_algo",
    "mutation_op",
    "base_recipe",
    "replicate_group",
    "failure_reason",
]
RUN_TIME_BUDGET = TIME_BUDGET
RUN_BUDGET_MODE = "time"
RUN_EPOCH_BUDGET = None
CURRENT_RUN_DATASET = DATASET
CURRENT_RUN_RECIPE = ""
CURRENT_RUN_BACKBONE = BACKBONE
CURRENT_RUN_SESSION = ""
CURRENT_RUN_METADATA = {}
CURRENT_RUN_RECIPE_SOURCE = ""
RUN_SEED = 42


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def summarize_augmentations():
    flags = []
    if AUG_HORIZONTAL_FLIP:
        flags.append("hflip")
    if AUG_VERTICAL_FLIP:
        flags.append("vflip")
    if AUG_ROTATION_DEGREES:
        flags.append(f"rot={AUG_ROTATION_DEGREES}")
    if AUG_COLOR_JITTER:
        flags.append("color")
    if AUG_USE_CLAHE:
        flags.append("clahe")
    if ELASTIC_TRANSFORM:
        flags.append("elastic")
    if GRID_DISTORTION:
        flags.append("grid")
    if COARSE_DROPOUT:
        flags.append(f"coarse={COARSE_DROPOUT_RATIO:g}")
    return ",".join(flags) if flags else "none"


def stringify_results_value(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def current_results_metadata(cfg, loss_type, batch_size=None):
    search_metadata = {
        "candidate_id": CURRENT_RUN_METADATA.get("candidate_id", ""),
        "parent_id": CURRENT_RUN_METADATA.get("parent_id", ""),
        "search_algo": CURRENT_RUN_METADATA.get("search_algo", ""),
        "mutation_op": CURRENT_RUN_METADATA.get("mutation_op", ""),
        "base_recipe": CURRENT_RUN_METADATA.get("base_recipe", ""),
        "replicate_group": CURRENT_RUN_METADATA.get("replicate_group", ""),
        "failure_reason": CURRENT_RUN_METADATA.get("failure_reason", ""),
    }
    return {
        "session": CURRENT_RUN_SESSION or "",
        "optimizer": OPTIMIZER,
        "loss": loss_type,
        "image_size": str(get_image_size(cfg)),
        "batch_size": str(batch_size if batch_size is not None else BATCH_SIZE),
        "learning_rate": f"{LEARNING_RATE:.8g}",
        "weight_decay": f"{WEIGHT_DECAY:.8g}",
        "dropout": f"{DROPOUT:.4f}",
        "augmentations": summarize_augmentations(),
        "seed": str(RUN_SEED),
        "time_budget_seconds": f"{RUN_TIME_BUDGET:.1f}" if RUN_BUDGET_MODE == "time" else "",
        "budget_mode": RUN_BUDGET_MODE,
        "epoch_budget": str(RUN_EPOCH_BUDGET or ""),
        **{key: stringify_results_value(value) for key, value in search_metadata.items()},
    }

# ---------------------------------------------------------------------------
# Classification Model: CheXNet-style (DenseNet-121 + pretrained)
# ---------------------------------------------------------------------------

class HuggingFaceVisionEncoder(nn.Module):
    """Wrapper for HuggingFace vision models used as feature extractors.

    Handles multiple model architectures:
    - CLIP-style models (BiomedCLIP, MedCLIP, PLIP): extracts vision encoder
    - DINOv2-style models (RAD-DINO): uses CLS token from ViT
    - MAE-style models (RETFound): uses CLS token from ViT encoder
    - Standard ViT models: uses pooler output or CLS token
    """

    def __init__(self, model_name, pretrained=True):
        super().__init__()
        from transformers import AutoModel, AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")

        if pretrained:
            full_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        else:
            full_model = AutoModel.from_config(config, trust_remote_code=True)

        # Extract the vision encoder for CLIP-style models
        if model_type in ("clip", "clap") or hasattr(full_model, "vision_model"):
            self.model = full_model.vision_model
            vision_config = getattr(config, "vision_config", config)
            self.num_features = getattr(vision_config, "hidden_size",
                                getattr(vision_config, "projection_dim", 768))
        else:
            self.model = full_model
            self.num_features = getattr(config, "hidden_size",
                                getattr(config, "projection_dim",
                                getattr(config, "embed_dim", 768)))

        self._model_type = model_type

    def forward(self, x):
        if self._model_type in ("clip", "clap") or hasattr(self.model, "embeddings"):
            # CLIP vision model takes pixel_values directly
            outputs = self.model(pixel_values=x)
        else:
            outputs = self.model(pixel_values=x)

        # Handle different output formats
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, 0]  # CLS token
        else:
            out = outputs[0]
            return out[:, 0] if len(out.shape) == 3 else out


class MedicalClassifier(nn.Module):
    """Pretrained backbone + classification head.
    Default: DenseNet-121 (CheXNet architecture).
    Supports any torchvision or timm model: densenet*, resnet*, efficientnet*, convnext*, vit_*, etc.
    Also supports HuggingFace models with "hf:" prefix (e.g., "hf:microsoft/BiomedCLIP-*").
    See FOUNDATION_MODELS.md for medical-specific backbone recommendations.
    """

    def __init__(self, num_classes, backbone="densenet121", pretrained=True,
                 dropout=0.2, multilabel=False):
        super().__init__()
        self.multilabel = multilabel

        if backbone.startswith("hf:"):
            # HuggingFace model (BiomedCLIP, RETFound, RAD-DINO, etc.)
            hf_name = backbone[3:]  # strip "hf:" prefix
            print(f"Loading HuggingFace model: {hf_name}")
            self.encoder = HuggingFaceVisionEncoder(hf_name, pretrained=pretrained)
            num_features = self.encoder.num_features
        elif backbone.startswith("densenet"):
            self.encoder = getattr(models, backbone)(
                weights="IMAGENET1K_V1" if pretrained else None,
            )
            num_features = self.encoder.classifier.in_features
            self.encoder.classifier = nn.Identity()
        elif backbone.startswith("resnet"):
            self.encoder = getattr(models, backbone)(
                weights="IMAGENET1K_V1" if pretrained else None,
            )
            num_features = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            # timm: covers efficientnet, convnext, vit, swin, etc.
            self.encoder = timm.create_model(
                backbone, pretrained=pretrained, num_classes=0,
            )
            num_features = self.encoder.num_features

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(num_features, num_classes)

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        features = self.encoder(x)
        features = self.dropout(features)
        return self.head(features)


# ---------------------------------------------------------------------------
# Segmentation Model: U-Net
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetMedical(nn.Module):
    """U-Net for medical segmentation."""

    def __init__(self, num_classes, in_channels=3, base_channels=64, pretrained=True):
        super().__init__()
        ch = base_channels

        self.enc1 = ConvBlock(in_channels, ch)
        self.enc2 = ConvBlock(ch, ch * 2)
        self.enc3 = ConvBlock(ch * 2, ch * 4)
        self.enc4 = ConvBlock(ch * 4, ch * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(ch * 8, ch * 16)

        self.up4 = nn.ConvTranspose2d(ch * 16, ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(ch * 16, ch * 8)
        self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(ch * 8, ch * 4)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(ch * 4, ch * 2)
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.dec1 = ConvBlock(ch * 2, ch)

        self.head = nn.Conv2d(ch, num_classes, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)


class AttentionGate(nn.Module):
    """Attention gate for U-Net skip connections."""
    def __init__(self, gate_ch, skip_ch, inter_ch=None):
        super().__init__()
        inter_ch = inter_ch or skip_ch // 2
        self.W_gate = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.W_skip = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        if g.shape[-2:] != s.shape[-2:]:
            g = F.interpolate(g, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return skip * self.psi(self.relu(g + s))


class DecoderBlock(nn.Module):
    """Decoder block with upsampling + skip connection + conv."""
    def __init__(self, in_ch, skip_ch, out_ch, use_attention=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.attention = AttentionGate(out_ch, skip_ch) if use_attention else None
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.attention is not None:
            skip = self.attention(x, skip)
        return self.conv(torch.cat([x, skip], dim=1))


class PretrainedUNet(nn.Module):
    """U-Net with a pretrained timm encoder for improved segmentation."""
    def __init__(self, num_classes, backbone="resnet34", pretrained=True,
                 in_channels=3, use_attention=False):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained,
                                          features_only=True, in_chans=in_channels)
        encoder_channels = self.encoder.feature_info.channels()
        decoder_channels = list(reversed(encoder_channels))
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_channels) - 1):
            self.decoder_blocks.append(
                DecoderBlock(decoder_channels[i], decoder_channels[i + 1],
                            decoder_channels[i + 1], use_attention=use_attention))
        self.head = nn.Conv2d(decoder_channels[-1], num_classes, 1)
        self._init_decoder()

    def _init_decoder(self):
        for m in self.decoder_blocks.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.encoder(x)
        x = features[-1]
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, features[-(i + 2)])
        return self.head(x)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for classification (handles class imbalance)."""

    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma * ce).mean()
        return focal


class FocalBCELoss(nn.Module):
    """Focal loss for multi-label (sigmoid) classification."""

    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal = ((1 - pt) ** self.gamma * bce).mean()
        return focal


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for segmentation."""

    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce = F.cross_entropy(logits, targets, ignore_index=-1)

        probs = F.softmax(logits, dim=1)[:, 1]
        targets_float = (targets == 1).float()
        intersection = (probs * targets_float).sum()
        dice = 1 - (2 * intersection + 1) / (probs.sum() + targets_float.sum() + 1)

        return (1 - self.dice_weight) * bce + self.dice_weight * dice


class TverskyLoss(nn.Module):
    """Tversky loss for segmentation — generalizes Dice with separate FP/FN weights."""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)[:, 1]
        targets_float = (targets == 1).float()
        tp = (probs * targets_float).sum()
        fp = (probs * (1 - targets_float)).sum()
        fn = ((1 - probs) * targets_float).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class TverskyBCELoss(nn.Module):
    """Combined Tversky + BCE loss for segmentation."""
    def __init__(self, alpha=0.7, beta=0.3, tversky_weight=0.5):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.tversky_weight = tversky_weight

    def forward(self, logits, targets):
        bce = F.cross_entropy(logits, targets, ignore_index=-1)
        tversky = self.tversky(logits, targets)
        return (1 - self.tversky_weight) * bce + self.tversky_weight * tversky


class OrdinalRegressionLoss(nn.Module):
    """Ordinal regression loss for graded diseases (e.g., DR severity 0-4)."""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        batch_size = targets.size(0)
        ordinal_targets = torch.zeros(batch_size, self.num_classes - 1, device=targets.device)
        for i in range(self.num_classes - 1):
            ordinal_targets[:, i] = (targets > i).float()
        ordinal_logits = logits[:, : self.num_classes - 1]
        return F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_targets)


# ---------------------------------------------------------------------------
# Runtime and data utilities
# ---------------------------------------------------------------------------

class RepeatChannels:
    """Expand a single-channel tensor to 3 channels without a lambda transform."""

    def __call__(self, tensor):
        return tensor.repeat(3, 1, 1)


class CLAHETransform:
    """Apply CLAHE to grayscale or RGB PIL images."""

    def __call__(self, image):
        return Image.fromarray(apply_clahe(np.array(image)))


def autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("CUDA not available. Using Apple MPS backend.")
        return torch.device("mps")
    print("WARNING: No CUDA device found. Falling back to CPU.")
    return torch.device("cpu")


def maybe_compile_model(model, device):
    if not USE_COMPILE:
        print("Compile: disabled by config.")
        return model
    if not hasattr(torch, "compile"):
        print("Compile: unavailable on this torch build; running eager.")
        return model
    if device.type != "cuda":
        print("Compile: skipped outside CUDA.")
        return model
    if os.name == "nt":
        print("Compile: skipped on Windows for stability.")
        return model
    try:
        model = torch.compile(model, dynamic=False)
        print("Compile: enabled.")
        return model
    except Exception as exc:
        print(f"Compile: disabled ({exc}).")
        return model


def get_image_size(cfg):
    return int(IMAGE_SIZE_OVERRIDE or cfg["image_size"])


def recommended_windows_worker_count(base_workers, batch_size, image_size):
    worker_count = max(0, min(int(base_workers), 2))
    pixel_budget = max(1, int(batch_size)) * max(1, int(image_size)) * max(1, int(image_size))
    if pixel_budget >= 2_000_000:
        return 0
    if pixel_budget >= 1_000_000:
        return min(worker_count, 1)
    return worker_count


def resolve_worker_count(cfg, batch_size):
    if os.name == "nt":
        return recommended_windows_worker_count(NUM_WORKERS, batch_size, get_image_size(cfg))
    return NUM_WORKERS


def build_classification_transforms(cfg, train=True):
    transform_list = [T.Resize((get_image_size(cfg), get_image_size(cfg)))]

    if AUG_USE_CLAHE:
        transform_list.append(CLAHETransform())

    if train:
        if AUG_HORIZONTAL_FLIP:
            transform_list.append(T.RandomHorizontalFlip())
        if AUG_VERTICAL_FLIP:
            transform_list.append(T.RandomVerticalFlip())
        if AUG_ROTATION_DEGREES:
            transform_list.append(T.RandomRotation(float(AUG_ROTATION_DEGREES)))
        if AUG_COLOR_JITTER and cfg["in_channels"] == 3:
            transform_list.append(
                T.ColorJitter(
                    brightness=AUG_COLOR_JITTER_BRIGHTNESS,
                    contrast=AUG_COLOR_JITTER_CONTRAST,
                )
            )

    transform_list.extend([T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])

    if cfg["in_channels"] == 1:
        transform_list.append(RepeatChannels())

    return T.Compose(transform_list)


def build_loader(dataset_name, cfg, batch_size, split="train"):
    is_train = split == "train"
    generator = torch.Generator()
    split_offset = {"train": 0, "val": 1, "test": 2}.get(split, 0)
    generator.manual_seed(RUN_SEED + split_offset)

    if cfg["task"] == "classification":
        dataset = MedicalClassificationDataset(
            dataset_name,
            split=split,
            transform=build_classification_transforms(cfg, train=is_train),
        )
    elif cfg["task"] == "multilabel":
        dataset = MedicalMultilabelDataset(
            dataset_name,
            split=split,
            transform=build_classification_transforms(cfg, train=is_train),
        )
    else:
        dataset = MedicalSegmentationDataset(
            dataset_name,
            split=split,
            image_size=get_image_size(cfg),
            augment=is_train,
        )

    effective_batch_size = max(1, min(batch_size, len(dataset)))
    worker_count = resolve_worker_count(cfg, effective_batch_size)
    loader_kwargs = {
        "pin_memory": torch.cuda.is_available(),
        "drop_last": is_train and len(dataset) > effective_batch_size,
        "persistent_workers": worker_count > 0 and os.name != "nt",
        "worker_init_fn": seed_worker if worker_count > 0 else None,
        "generator": generator,
    }
    if worker_count > 0 and os.name == "nt":
        # Windows file-mapping pressure is much more likely to bite large-image overnight runs.
        loader_kwargs["prefetch_factor"] = 1

    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=is_train,
        num_workers=worker_count,
        **loader_kwargs,
    )


def build_infinite_loader(dataset_name, cfg, batch_size, split="train", loader=None):
    loader = loader or build_loader(dataset_name, cfg, batch_size, split=split)
    epoch = 1
    while True:
        for batch in loader:
            images, targets = batch
            yield images, targets, epoch
        epoch += 1


def resolve_split_file(dataset_name):
    return os.path.join(DATA_DIR, dataset_name, "splits.csv")


def compute_class_weights(dataset_name, cfg, device):
    df = pd.read_csv(resolve_split_file(dataset_name))
    train_df = df[df["split"] == "train"]
    num_classes = cfg["num_classes"]

    if cfg["task"] == "multilabel":
        labels = np.stack(
            train_df["labels"].astype(str).map(lambda raw: [int(x) for x in raw.split("|")]).to_list()
        )
        positives = labels.sum(axis=0)
        negatives = labels.shape[0] - positives
        weights = negatives / np.maximum(positives, 1)
    else:
        counts = train_df["label"].astype(int).value_counts().sort_index()
        all_counts = np.zeros(num_classes)
        for cls_id, count in counts.items():
            if 0 <= cls_id < num_classes:
                all_counts[cls_id] = count
        total = all_counts.sum()
        weights = total / (num_classes * np.maximum(all_counts, 1))

    weights = np.clip(weights, 0.5, 10.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    """Resolve the normalized LR multiplier for the active schedule."""
    schedule = str(LR_SCHEDULE).lower().strip() or "cosine"
    progress = float(min(max(progress, 0.0), 1.0))

    if schedule == "one_cycle":
        warmup_end = min(max(WARMUP_RATIO if WARMUP_RATIO > 0 else 0.3, 0.05), 0.5)
        if progress < warmup_end:
            return 0.1 + 0.9 * (progress / warmup_end)
        decay_progress = (progress - warmup_end) / max(1.0 - warmup_end, 1e-8)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * decay_progress))

    if schedule == "constant":
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        return 1.0

    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown_progress = (1.0 - progress) / max(WARMDOWN_RATIO, 1e-8)
    return 0.5 * (1.0 + math.cos(math.pi * (1 - cooldown_progress)))


# ---------------------------------------------------------------------------
# Advanced training utilities
# ---------------------------------------------------------------------------

class ModelEMA:
    """Maintains an exponential moving average of model parameters."""
    def __init__(self, model, decay=0.999):
        import copy
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        self.ema_model = copy.deepcopy(raw)
        self.ema_model.eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        for ema_p, model_p in zip(self.ema_model.parameters(), raw.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)


def bootstrap_ci(metric_fn, preds, targets, n_bootstraps=1000, ci=0.95):
    """Compute bootstrap confidence interval for a metric function."""
    rng = np.random.RandomState(42)
    n = len(preds)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        try:
            score = metric_fn(preds[idx], targets[idx])
            if not np.isnan(score):
                scores.append(score)
        except (ValueError, IndexError):
            continue
    if not scores:
        return 0.0, 0.0, 0.0
    scores = np.array(scores)
    alpha = 1 - ci
    return np.mean(scores), np.percentile(scores, 100 * alpha / 2), np.percentile(scores, 100 * (1 - alpha / 2))


def mixup_data(images, targets, alpha=0.2):
    """Apply MixUp augmentation."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[index]
    return mixed, targets, targets[index], lam


def mixup_criterion(criterion, logits, targets_a, targets_b, lam):
    """Compute MixUp loss as weighted combination."""
    return lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)


def cutmix_data(images, targets, alpha=1.0):
    """Apply CutMix augmentation."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    _, _, H, W = images.shape
    cut_ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cy = torch.randint(0, H, (1,)).item()
    cx = torch.randint(0, W, (1,)).item()
    y1, y2 = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)
    x1, x2 = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed, targets, targets[index], lam


def get_llrd_param_groups(model, base_lr, decay=0.9):
    """Create per-layer parameter groups with exponentially decaying LR."""
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    if not hasattr(raw, "encoder"):
        return [{"params": model.parameters(), "lr": base_lr}]
    encoder_params = list(raw.encoder.named_parameters())
    head_params = [p for n, p in raw.named_parameters() if not n.startswith("encoder")]
    layer_groups = {}
    for name, param in encoder_params:
        parts = name.split(".")
        depth_key = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
        layer_groups.setdefault(depth_key, []).append(param)
    sorted_groups = list(layer_groups.values())
    num_layers = len(sorted_groups)
    param_groups = [{"params": g, "lr": base_lr * (decay ** (num_layers - i))}
                    for i, g in enumerate(sorted_groups)]
    param_groups.append({"params": head_params, "lr": base_lr})
    return param_groups


@torch.no_grad()
def tta_predict(model, images, device):
    """Average predictions over original + horizontally flipped images."""
    with autocast_context(device):
        logits1 = model(images.to(device))
        logits2 = model(torch.flip(images.to(device), dims=[-1]))
    return (logits1 + logits2) / 2.0


def optimize_thresholds(probs, targets, num_classes):
    """Find optimal sigmoid threshold per class for multilabel tasks."""
    thresholds = np.full(num_classes, 0.5)
    for i in range(num_classes):
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.1, 0.9, 0.02):
            preds = (probs[:, i] >= t).astype(int)
            try:
                f1 = f1_score(targets[:, i], preds, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            except ValueError:
                continue
        thresholds[i] = best_t
    return thresholds


def distillation_loss(student_logits, teacher_logits, targets, criterion,
                      alpha=0.5, temperature=3.0):
    """Combined task loss + KL divergence from teacher soft targets."""
    task_loss = criterion(student_logits, targets)
    with torch.no_grad():
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
    kd_loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)
    return (1 - alpha) * task_loss + alpha * kd_loss


def model_soup(checkpoint_paths, model):
    """Average weights from multiple checkpoints (model soup)."""
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    state_dicts = []
    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dicts.append(ckpt["model_state_dict"])
    if not state_dicts:
        print("Model soup: no checkpoints found")
        return model
    avg_state = {}
    for key in state_dicts[0]:
        avg_state[key] = sum(sd[key].float() for sd in state_dicts) / len(state_dicts)
    raw.load_state_dict(avg_state)
    print(f"Model soup: averaged {len(state_dicts)} checkpoints")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_classification_loader(model, loader, cfg, device):
    num_classes = cfg["num_classes"]
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    total_loss = 0.0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        if USE_TTA:
            logits = tta_predict(model, images, device)
        else:
            with autocast_context(device):
                logits = model(images)

        loss = F.cross_entropy(logits.float(), targets, reduction="sum")
        total_loss += loss.item()
        total_samples += targets.size(0)

        probs = F.softmax(logits.float(), dim=1)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()

    accuracy = 100.0 * (all_preds == all_targets).mean()
    avg_loss = total_loss / max(total_samples, 1)
    kappa = cohen_kappa_score(all_targets, all_preds, weights="quadratic")

    try:
        if num_classes == 2:
            auc = 100.0 * roc_auc_score(all_targets, all_probs[:, 1])
        else:
            auc = 100.0 * roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0

    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "auc": auc,
        "kappa": float(kappa),
        "artifacts": {
            "val_probs": all_probs.astype(np.float16),
            "val_targets": all_targets.astype(np.int64),
            "val_preds": all_preds.astype(np.int64),
        },
    }


@torch.no_grad()
def evaluate_multilabel_loader(model, loader, cfg, device):
    num_classes = cfg["num_classes"]
    model.eval()
    all_probs = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        if USE_TTA:
            logits = tta_predict(model, images, device)
        else:
            with autocast_context(device):
                logits = model(images)
        probs = torch.sigmoid(logits.float())
        all_probs.append(probs.cpu())
        all_targets.append(targets.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    per_class_auc = {}
    valid_aucs = []
    for idx in range(num_classes):
        try:
            auc = roc_auc_score(all_targets[:, idx], all_probs[:, idx])
            per_class_auc[cfg["class_names"][idx]] = round(100.0 * auc, 2)
            valid_aucs.append(auc)
        except ValueError:
            per_class_auc[cfg["class_names"][idx]] = 0.0

    mean_auc = 100.0 * np.mean(valid_aucs) if valid_aucs else 0.0
    return {
        "mean_auc": mean_auc,
        "per_class_auc": per_class_auc,
        "artifacts": {
            "val_probs": all_probs.astype(np.float16),
            "val_targets": all_targets.astype(np.uint8),
            "class_names": list(cfg["class_names"]),
        },
    }


@torch.no_grad()
def evaluate_segmentation_loader(model, loader, device):
    model.eval()
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_correct = 0
    total_pixels = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        with autocast_context(device):
            logits = model(images)

        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        preds = logits.argmax(dim=1)
        pred_pos = preds == 1
        target_pos = targets == 1
        total_tp += (pred_pos & target_pos).sum().item()
        total_fp += (pred_pos & ~target_pos).sum().item()
        total_fn += (~pred_pos & target_pos).sum().item()
        total_tn += (~pred_pos & ~target_pos).sum().item()
        total_correct += (preds == targets).sum().item()
        total_pixels += targets.numel()

    dice = 100.0 * (2 * total_tp) / max(2 * total_tp + total_fp + total_fn, 1)
    iou = 100.0 * total_tp / max(total_tp + total_fp + total_fn, 1)
    pixel_acc = 100.0 * total_correct / max(total_pixels, 1)
    sensitivity = 100.0 * total_tp / max(total_tp + total_fn, 1)
    specificity = 100.0 * total_tn / max(total_tn + total_fp, 1)

    return {
        "dice": dice,
        "iou": iou,
        "pixel_acc": pixel_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def ensure_results_file():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULTS_HEADER, delimiter="\t")
            writer.writeheader()
        return

    with open(RESULTS_FILE, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        existing_header = reader.fieldnames or []
        rows = list(reader)

    if not existing_header:
        with open(RESULTS_FILE, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULTS_HEADER, delimiter="\t")
            writer.writeheader()
        return

    if existing_header == RESULTS_HEADER:
        return

    migrated_rows = []
    for row in rows:
        normalized = {}
        for field in RESULTS_HEADER:
            value = row.get(field, "")
            normalized[field] = "" if value is None else value
        migrated_rows.append(normalized)

    with open(RESULTS_FILE, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_HEADER, delimiter="\t")
        writer.writeheader()
        writer.writerows(migrated_rows)


def resolve_git_dir(repo_root):
    git_path = Path(repo_root) / ".git"
    if git_path.is_dir():
        return git_path
    if git_path.is_file():
        content = git_path.read_text(encoding="utf-8").strip()
        if content.startswith("gitdir:"):
            return (git_path.parent / content.split(":", 1)[1].strip()).resolve()
    return None


def read_commit_hash(repo_root):
    git_dir = resolve_git_dir(repo_root)
    if git_dir is None:
        return "unknown"

    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return "unknown"

    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head[:12]

    ref = head[5:]
    ref_path = git_dir / ref
    if ref_path.exists():
        return ref_path.read_text(encoding="utf-8").strip()[:12]

    packed_refs = git_dir / "packed-refs"
    if packed_refs.exists():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or line.startswith("^") or not line.strip():
                continue
            commit, name = line.split(" ", 1)
            if name.strip() == ref:
                return commit[:12]
    return "unknown"


def append_results_row(
    dataset,
    recipe,
    backbone,
    metric_name,
    metric,
    memory_gb,
    status,
    checkpoint_path,
    description,
    metadata=None,
):
    ensure_results_file()
    row = {field: "" for field in RESULTS_HEADER}
    row.update({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit": read_commit_hash(os.path.dirname(os.path.abspath(__file__))),
        "dataset": dataset,
        "recipe": recipe or "",
        "backbone": backbone,
        "metric_name": metric_name,
        "metric": f"{metric:.6f}",
        "memory_gb": f"{memory_gb:.3f}",
        "status": status,
        "checkpoint_path": checkpoint_path or "",
        "description": description,
    })
    if metadata:
        for key, value in metadata.items():
            if key in row:
                row[key] = value
    with open(RESULTS_FILE, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_HEADER, delimiter="\t")
        writer.writerow(row)


def build_run_description(dataset_name, cfg, recipe_name, backbone_name, loss_type):
    parts = []
    if recipe_name:
        parts.append(f"recipe={recipe_name}")
    else:
        parts.append(f"dataset={dataset_name}")
    if CURRENT_RUN_SESSION:
        parts.append(f"session={CURRENT_RUN_SESSION}")
    parts.extend(
        [
            backbone_name,
            OPTIMIZER,
            loss_type,
            f"img={get_image_size(cfg)}",
            f"bs={BATCH_SIZE}",
            f"lr={LEARNING_RATE:.2e}",
            f"wd={WEIGHT_DECAY:.2e}",
            f"drop={DROPOUT:g}",
            f"aug={summarize_augmentations()}",
            f"budget={RUN_EPOCH_BUDGET}ep" if RUN_BUDGET_MODE == "epochs" else f"budget={RUN_TIME_BUDGET:.0f}s",
        ]
    )
    if CURRENT_RUN_METADATA.get("mutation_op"):
        parts.append(f"mut={CURRENT_RUN_METADATA['mutation_op']}")
    if CURRENT_RUN_METADATA.get("base_recipe"):
        parts.append(f"base={CURRENT_RUN_METADATA['base_recipe']}")
    return " | ".join(parts)


def build_effective_config(
    dataset_name,
    cfg,
    task,
    num_classes,
    backbone_name,
    batch_size,
    loss_type,
    primary_metric_name=None,
):
    return {
        "dataset": dataset_name,
        "dataset_source": cfg.get("source", "builtin"),
        "task": task,
        "metric_name": primary_metric_name or cfg.get("metric", ""),
        "backbone": backbone_name,
        "num_classes": num_classes,
        "class_names": cfg.get("class_names", []),
        "batch_size": batch_size,
        "dropout": DROPOUT,
        "optimizer": OPTIMIZER,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "loss": loss_type,
        "recipe": CURRENT_RUN_RECIPE,
        "recipe_source": CURRENT_RUN_RECIPE_SOURCE,
        "session": CURRENT_RUN_SESSION,
        "seed": RUN_SEED,
        "time_budget_seconds": RUN_TIME_BUDGET if RUN_BUDGET_MODE == "time" else None,
        "budget_mode": RUN_BUDGET_MODE,
        "epoch_budget": RUN_EPOCH_BUDGET,
        "selection_split": "val",
        "image_size": get_image_size(cfg),
        "model": {
            "backbone": BACKBONE,
            "pretrained": PRETRAINED,
            "dropout": DROPOUT,
        },
        "training": {
            "batch_size": batch_size,
            "learning_rate": LEARNING_RATE,
            "optimizer": OPTIMIZER,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "warmdown_ratio": WARMDOWN_RATIO,
            "label_smoothing": LABEL_SMOOTHING,
        },
        "loss_config": {
            "type": loss_type,
            "class_weights": USE_CLASS_WEIGHTS,
            "focal_gamma": FOCAL_GAMMA,
            "tversky_alpha": TVERSKY_ALPHA,
            "tversky_beta": TVERSKY_BETA,
        },
        "augmentations": {
            "horizontal_flip": AUG_HORIZONTAL_FLIP,
            "vertical_flip": AUG_VERTICAL_FLIP,
            "rotation": AUG_ROTATION_DEGREES,
            "color_jitter": AUG_COLOR_JITTER,
            "color_jitter_brightness": AUG_COLOR_JITTER_BRIGHTNESS,
            "color_jitter_contrast": AUG_COLOR_JITTER_CONTRAST,
            "clahe": AUG_USE_CLAHE,
            "elastic_transform": ELASTIC_TRANSFORM,
            "grid_distortion": GRID_DISTORTION,
            "coarse_dropout": COARSE_DROPOUT,
            "coarse_dropout_ratio": COARSE_DROPOUT_RATIO,
        },
        "advanced": {
            "ema": USE_EMA,
            "ema_decay": EMA_DECAY,
            "tta": USE_TTA,
            "compile": USE_COMPILE,
            "lr_schedule": LR_SCHEDULE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "mixup": USE_MIXUP,
            "mixup_alpha": MIXUP_ALPHA,
            "cutmix": USE_CUTMIX,
            "cutmix_alpha": CUTMIX_ALPHA,
            "swa": USE_SWA,
            "swa_start_ratio": SWA_START_RATIO,
        },
        "search_metadata": dict(CURRENT_RUN_METADATA),
    }


def dump_effective_config(path, config):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".json":
        target.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        return
    try:
        import yaml
    except ImportError:
        target.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        return
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

def save_checkpoint(model, results, config, path):
    """Save model checkpoint with metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Unwrap compiled model
    raw_model = model
    if hasattr(model, "_orig_mod"):
        raw_model = model._orig_mod

    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "results": results,
        "config": config,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def list_recipes():
    recipe_dir = Path(__file__).parent / "recipes"
    print("\nAvailable recipes:\n")
    for recipe_file in sorted(recipe_dir.glob("*.yaml")):
        desc = ""
        with open(recipe_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("#") and not desc:
                    desc = line.lstrip("# ").strip()
                    break
        print(f"  {recipe_file.stem:<28s} {desc}")
    print("\nUsage: uv run train.py --recipe <name>")


def primary_metric_for_results(results, cfg):
    if cfg["task"] == "multilabel":
        return "mean_auc", float(results["mean_auc"])
    if cfg["task"] == "segmentation":
        metric_name = cfg["metric"]
        return metric_name, float(results.get(metric_name, results["dice"]))
    if cfg["metric"] in {"quadratic_kappa", "kappa"}:
        return cfg["metric"], float(results["kappa"])
    return cfg["metric"], float(results.get(cfg["metric"], results["accuracy"]))


def main():
    global CURRENT_RUN_DATASET, CURRENT_RUN_RECIPE, CURRENT_RUN_BACKBONE, CURRENT_RUN_SESSION
    global CURRENT_RUN_METADATA, CURRENT_RUN_RECIPE_SOURCE, RUN_TIME_BUDGET
    global RUN_BUDGET_MODE, RUN_EPOCH_BUDGET
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="Load a recipe from recipes/ (for example: chexnet, aptos_winner)",
    )
    parser.add_argument(
        "--recipe-file",
        type=str,
        default=None,
        help="Load a recipe from an explicit YAML path (for adaptive search overlays).",
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Save the run checkpoint under checkpoints/",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="Optional override for smoke tests. Defaults to the 300s autoresearch budget.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch budget for a full-train refinement run. Keeps search-time 300s mode unchanged unless explicitly set.",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Optional run-session tag used to isolate a fresh overnight search in results.tsv and downstream artifacts.",
    )
    parser.add_argument(
        "--dump-effective-config",
        type=str,
        default=None,
        help="Optional path to write the fully resolved training config used for this run.",
    )
    parser.add_argument("--list-recipes", action="store_true", help="List available recipes and exit")
    args = parser.parse_args()

    if args.list_recipes:
        list_recipes()
        return

    if args.recipe and args.recipe_file:
        raise ValueError("Use either --recipe or --recipe-file, not both")
    if args.time_budget is not None and args.epochs is not None:
        raise ValueError("Use either --time-budget or --epochs, not both")

    recipe = None
    recipe_source = ""
    run_recipe_name = ""
    if args.recipe:
        recipe = load_recipe(args.recipe)
        recipe_source = str((Path(__file__).parent / "recipes" / f"{args.recipe}.yaml").resolve())
        run_recipe_name = args.recipe
    elif args.recipe_file:
        recipe_path = Path(args.recipe_file).expanduser().resolve()
        recipe = load_recipe_file(recipe_path)
        recipe_source = str(recipe_path)
        run_recipe_name = recipe_path.stem

    CURRENT_RUN_METADATA = {}
    CURRENT_RUN_RECIPE_SOURCE = recipe_source
    if recipe:
        if isinstance(recipe.get("search_metadata"), dict):
            CURRENT_RUN_METADATA = dict(recipe["search_metadata"])
        apply_recipe(recipe)

    dataset = args.dataset or DATASET
    CURRENT_RUN_DATASET = dataset
    CURRENT_RUN_RECIPE = run_recipe_name
    CURRENT_RUN_SESSION = (args.session or "").strip()
    RUN_EPOCH_BUDGET = int(args.epochs) if args.epochs is not None else None
    if RUN_EPOCH_BUDGET is not None and RUN_EPOCH_BUDGET <= 0:
        raise ValueError("--epochs must be positive")
    RUN_BUDGET_MODE = "epochs" if RUN_EPOCH_BUDGET is not None else "time"
    RUN_TIME_BUDGET = 0.0 if RUN_BUDGET_MODE == "epochs" else (
        float(args.time_budget) if args.time_budget is not None else float(TIME_BUDGET)
    )
    if RUN_BUDGET_MODE == "time" and RUN_TIME_BUDGET <= 0:
        raise ValueError("--time-budget must be positive")
    cfg = get_dataset_config(dataset)
    task = cfg["task"]
    num_classes = cfg["num_classes"]
    is_multilabel = task == "multilabel"

    device = select_device()
    t_start = time.time()
    seed_everything(RUN_SEED)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    print(f"Task: {task}")
    print(f"Dataset: {dataset}")
    print(f"Dataset source: {cfg.get('source', 'builtin')}")
    print(f"Num classes: {num_classes}")
    print(f"Primary metric: {cfg['metric']}")
    if "description" in cfg:
        print(f"Description: {cfg['description']}")
    if is_multilabel:
        print(f"Class names: {', '.join(cfg['class_names'])}")

    if task in ("classification", "multilabel"):
        model = MedicalClassifier(
            num_classes=num_classes,
            backbone=BACKBONE,
            pretrained=PRETRAINED,
            dropout=DROPOUT,
        )
        batch_size = BATCH_SIZE
    elif task == "segmentation":
        if SEG_BACKBONE:
            model = PretrainedUNet(num_classes=num_classes, backbone=SEG_BACKBONE,
                                   pretrained=PRETRAINED, in_channels=cfg["in_channels"],
                                   use_attention=SEG_USE_ATTENTION)
        else:
            model = UNetMedical(
                num_classes=num_classes,
                in_channels=cfg["in_channels"],
                base_channels=SEG_BASE_CHANNELS,
                pretrained=PRETRAINED,
            )
        batch_size = SEG_BATCH_SIZE
    else:
        raise ValueError(f"Unknown task: {task}")

    model = model.to(device)
    model = maybe_compile_model(model, device)
    num_params = sum(param.numel() for param in model.parameters())
    backbone_name = BACKBONE if task != "segmentation" else "unet"
    CURRENT_RUN_BACKBONE = backbone_name

    print(f"Model: {backbone_name}")
    print(f"Pretrained: {PRETRAINED}")
    print(f"Parameters: {num_params:,}")
    if RUN_BUDGET_MODE == "epochs":
        print(f"Epoch budget: {RUN_EPOCH_BUDGET}")
    else:
        print(f"Time budget: {RUN_TIME_BUDGET}s")
    print(f"Batch size: {batch_size}")
    print(f"Optimizer: {OPTIMIZER}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Image size: {get_image_size(cfg)}")
    if LR_SCHEDULE != "cosine":
        print(f"LR schedule: {LR_SCHEDULE}")

    if OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

    loss_type = LOSS_TYPE
    if loss_type == "auto":
        if is_multilabel:
            loss_type = "bce"
        elif task == "segmentation":
            loss_type = "dice_bce"
        else:
            loss_type = "cross_entropy"

    class_weights = None
    if USE_CLASS_WEIGHTS and task in ("classification", "multilabel"):
        class_weights = compute_class_weights(dataset, cfg, device)

    if loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights) if class_weights is not None else nn.BCEWithLogitsLoss()
    elif loss_type == "focal" and is_multilabel:
        criterion = FocalBCELoss(gamma=FOCAL_GAMMA, pos_weight=class_weights)
    elif loss_type == "focal":
        criterion = FocalLoss(
            gamma=FOCAL_GAMMA,
            weight=class_weights,
            label_smoothing=LABEL_SMOOTHING,
        )
    elif loss_type == "dice_bce":
        criterion = DiceBCELoss(dice_weight=0.5)
    elif loss_type == "tversky_bce":
        criterion = TverskyBCELoss(alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA)
    elif loss_type == "ordinal":
        criterion = OrdinalRegressionLoss(num_classes=num_classes)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    print(f"Loss: {loss_type}")
    if class_weights is not None:
        print("Class weighting: enabled")
    print()

    effective_config = build_effective_config(
        dataset_name=dataset,
        cfg=cfg,
        task=task,
        num_classes=num_classes,
        backbone_name=backbone_name,
        batch_size=batch_size,
        loss_type=loss_type,
    )
    if args.dump_effective_config:
        dump_effective_config(args.dump_effective_config, effective_config)

    train_epoch_loader = build_loader(dataset, cfg, batch_size, split="train")
    steps_per_epoch = max(len(train_epoch_loader), 1)
    total_step_budget = steps_per_epoch * RUN_EPOCH_BUDGET if RUN_EPOCH_BUDGET is not None else None
    train_loader = build_infinite_loader(dataset, cfg, batch_size, split="train", loader=train_epoch_loader)
    val_loader = build_loader(dataset, cfg, batch_size, split="val")
    images, targets, epoch = next(train_loader)

    smooth_train_loss = 0.0
    total_training_time = 0.0
    step = 0
    model.train()
    ema_model = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None

    while True:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        images_d = images.to(device, non_blocking=device.type == "cuda")
        targets_d = targets.to(device, non_blocking=device.type == "cuda")

        with autocast_context(device):
            logits = model(images_d)
            loss = criterion(logits, targets_d)

        scaler.scale(loss).backward()

        if RUN_BUDGET_MODE == "epochs":
            progress = min(step / max(total_step_budget, 1), 1.0)
        else:
            progress = min(total_training_time / RUN_TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        for param_group in optimizer.param_groups:
            param_group["lr"] = LEARNING_RATE * lrm

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if ema_model is not None:
            ema_model.update(model)

        train_loss_f = loss.item()
        if train_loss_f > 100 or train_loss_f != train_loss_f:
            raise RuntimeError("loss exploded or became NaN")

        images, targets, epoch = next(train_loader)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        if RUN_BUDGET_MODE == "epochs":
            total_training_time += dt
        elif step > 5:
            total_training_time += dt

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * progress
        display_epoch = epoch if RUN_BUDGET_MODE == "time" else min((step // steps_per_epoch) + 1, RUN_EPOCH_BUDGET)
        if RUN_BUDGET_MODE == "epochs":
            remaining_steps = max(0, total_step_budget - step)
            remaining_display = f"{remaining_steps / steps_per_epoch:.1f}ep"
        else:
            remaining_display = f"{max(0.0, RUN_TIME_BUDGET - total_training_time):.0f}s"

        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | "
            f"loss: {debiased_smooth_loss:.4f} | "
            f"lr: {LEARNING_RATE * lrm:.6f} | "
            f"dt: {dt * 1000:.0f}ms | "
            f"epoch: {display_epoch} | "
            f"remaining: {remaining_display}    ",
            end="",
            flush=True,
        )

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1
        if RUN_BUDGET_MODE == "epochs":
            if step >= total_step_budget:
                break
        elif step > 5 and total_training_time >= RUN_TIME_BUDGET:
            break

    print()
    print("---")

    eval_model = ema_model.ema_model if ema_model is not None else model
    if task == "multilabel":
        results = evaluate_multilabel_loader(eval_model, val_loader, cfg, device)
        print(f"mean_auc:         {results['mean_auc']:.2f}%")
        print("per_class_auc:")
        for pathology, auc in results["per_class_auc"].items():
            print(f"  {pathology:<22s} {auc:.2f}%")
    elif task == "classification":
        results = evaluate_classification_loader(eval_model, val_loader, cfg, device)
        print(f"val_accuracy:     {results['accuracy']:.2f}%")
        print(f"val_auc:          {results['auc']:.2f}%")
        print(f"val_kappa:        {results['kappa']:.4f}")
        print(f"val_loss:         {results['loss']:.6f}")
    else:
        results = evaluate_segmentation_loader(eval_model, val_loader, device)
        print(f"val_dice:         {results['dice']:.2f}%")
        print(f"val_iou:          {results['iou']:.2f}%")
        print(f"val_pixel_acc:    {results['pixel_acc']:.2f}%")
        print(f"val_sensitivity:  {results['sensitivity']:.2f}%")
        print(f"val_specificity:  {results['specificity']:.2f}%")

    metric_name, primary_metric = primary_metric_for_results(results, cfg)
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0
    memory_gb = peak_vram_mb / 1024.0

    completed_epochs = epoch if RUN_BUDGET_MODE == "time" else math.ceil(step / max(steps_per_epoch, 1))
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"num_epochs:       {completed_epochs}")
    if RUN_BUDGET_MODE == "epochs":
        print(f"epoch_budget:     {RUN_EPOCH_BUDGET}")
    print(f"dataset:          {dataset}")
    if CURRENT_RUN_SESSION:
        print(f"session:          {CURRENT_RUN_SESSION}")
    print(f"task:             {task}")
    print(f"backbone:         {backbone_name}")
    if CURRENT_RUN_RECIPE:
        print(f"recipe:           {CURRENT_RUN_RECIPE}")

    checkpoint_path = ""
    if args.save_checkpoint:
        ckpt_config = dict(effective_config)
        ckpt_config["metric_name"] = metric_name
        ckpt_name = f"{dataset}_{backbone_name}_{primary_metric:.2f}.pth"
        checkpoint_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
        save_checkpoint(eval_model, results, ckpt_config, checkpoint_path)

    append_results_row(
        dataset=dataset,
        recipe=CURRENT_RUN_RECIPE,
        backbone=backbone_name,
        metric_name=metric_name,
        metric=primary_metric,
        memory_gb=memory_gb,
        status="candidate",
        checkpoint_path=checkpoint_path,
        description=build_run_description(dataset, cfg, CURRENT_RUN_RECIPE, backbone_name, loss_type),
        metadata=current_results_metadata(cfg, loss_type, batch_size=batch_size),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            crash_metadata = {
                "session": CURRENT_RUN_SESSION or "",
                "seed": str(RUN_SEED),
                "budget_mode": RUN_BUDGET_MODE,
                "time_budget_seconds": f"{RUN_TIME_BUDGET:.1f}" if RUN_BUDGET_MODE == "time" else "",
                "epoch_budget": str(RUN_EPOCH_BUDGET or ""),
            }
            for key, value in CURRENT_RUN_METADATA.items():
                if key in RESULTS_HEADER:
                    crash_metadata[key] = stringify_results_value(value)
            crash_metadata["failure_reason"] = f"{type(exc).__name__}: {exc}"
            append_results_row(
                dataset=CURRENT_RUN_DATASET,
                recipe=CURRENT_RUN_RECIPE,
                backbone=CURRENT_RUN_BACKBONE,
                metric_name="error",
                metric=0.0,
                memory_gb=0.0,
                status="crash",
                checkpoint_path="",
                description=f"crash: {type(exc).__name__}: {exc}",
                metadata=crash_metadata,
            )
        except Exception:
            pass
        raise
