"""
Autoresearch-MedVision training script.

This is the experiment surface for the Karpathy-style inner loop:
- fixed time budget
- mutate train.py, not prepare.py
- log every run to results.tsv
- keep or discard externally after reading the run metric

Usage:
    uv run train.py                                    # CheXNet on ChestX-ray14 (default)
    uv run train.py --recipe chexnet                   # Load recipe config
    uv run train.py --recipe aptos_winner              # Kaggle-winning DR approach
    uv run train.py --dataset aptos                    # DR grading (manual config)
    uv run train.py --dataset brain_mri                # Brain tumor segmentation
    uv run train.py --save-checkpoint                  # Save best model
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import copy
import csv
import gc
import math
import random
import sys
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

def load_recipe(recipe_name):
    """Load a recipe YAML file and return config dict."""
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required for recipe support. Run `uv sync` to install dependencies."
        ) from exc

    recipe_dir = Path(__file__).parent / "recipes"
    recipe_path = recipe_dir / f"{recipe_name}.yaml"

    if not recipe_path.exists():
        available = [f.stem for f in recipe_dir.glob("*.yaml")]
        print(f"Recipe '{recipe_name}' not found.")
        print(f"Available recipes: {', '.join(sorted(available))}")
        exit(1)

    with open(recipe_path, encoding="utf-8") as f:
        recipe = yaml.safe_load(f)

    if not isinstance(recipe, dict):
        raise SystemExit(f"Recipe '{recipe_name}' is invalid: expected a YAML mapping.")

    return recipe


def apply_recipe(recipe):
    """Apply recipe config to global hyperparameters. Returns overridden values."""
    global DATASET, BACKBONE, PRETRAINED, DROPOUT, BATCH_SIZE, LEARNING_RATE
    global OPTIMIZER, WEIGHT_DECAY, WARMUP_RATIO, WARMDOWN_RATIO
    global USE_CLASS_WEIGHTS, LABEL_SMOOTHING, LOSS_TYPE, FOCAL_GAMMA
    global SEG_BATCH_SIZE, SEG_BASE_CHANNELS, IMAGE_SIZE_OVERRIDE
    global AUG_HORIZONTAL_FLIP, AUG_VERTICAL_FLIP, AUG_ROTATION_DEGREES
    global AUG_COLOR_JITTER, AUG_COLOR_JITTER_BRIGHTNESS, AUG_COLOR_JITTER_CONTRAST
    global AUG_USE_CLAHE
    global GRAD_ACCUM_STEPS, USE_EMA, EMA_DECAY, USE_TTA, USE_MIXUP, MIXUP_ALPHA
    global USE_GRADUAL_UNFREEZE, UNFREEZE_AFTER_RATIO, DISCRIMINATIVE_LR, BACKBONE_LR_FACTOR
    global GRAD_CLIP_NORM, COMPUTE_CI, CI_BOOTSTRAPS
    global ELASTIC_TRANSFORM, GRID_DISTORTION, COARSE_DROPOUT, COARSE_DROPOUT_RATIO
    global USE_CUTMIX, CUTMIX_ALPHA, LR_SCHEDULE, USE_SWA, SWA_START_RATIO
    global TVERSKY_ALPHA, TVERSKY_BETA, OPTIMIZE_THRESHOLDS
    global SEG_BACKBONE, SEG_USE_ATTENTION, USE_LLRD, LLRD_DECAY
    global GENERATE_GRADCAM, GRADCAM_NUM_SAMPLES, EXPORT_ONNX
    global USE_DISTILLATION, TEACHER_CHECKPOINT, DISTILL_ALPHA, DISTILL_TEMPERATURE
    global USE_MODEL_SOUP, SOUP_CHECKPOINTS

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
        if "batch_size" in training:
            BATCH_SIZE = training["batch_size"]
            SEG_BATCH_SIZE = training["batch_size"]
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

    # Advanced training options
    advanced = recipe.get("advanced", {})
    if isinstance(advanced, dict):
        GRAD_ACCUM_STEPS = advanced.get("grad_accum_steps", GRAD_ACCUM_STEPS)
        USE_EMA = advanced.get("ema", USE_EMA)
        EMA_DECAY = advanced.get("ema_decay", EMA_DECAY)
        USE_TTA = advanced.get("tta", USE_TTA)
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
        GENERATE_GRADCAM = advanced.get("gradcam", GENERATE_GRADCAM)
        GRADCAM_NUM_SAMPLES = advanced.get("gradcam_samples", GRADCAM_NUM_SAMPLES)
        EXPORT_ONNX = advanced.get("export_onnx", EXPORT_ONNX)
        USE_DISTILLATION = advanced.get("distillation", USE_DISTILLATION)
        TEACHER_CHECKPOINT = advanced.get("teacher_checkpoint", TEACHER_CHECKPOINT)
        DISTILL_ALPHA = advanced.get("distill_alpha", DISTILL_ALPHA)
        DISTILL_TEMPERATURE = advanced.get("distill_temperature", DISTILL_TEMPERATURE)
        USE_MODEL_SOUP = advanced.get("model_soup", USE_MODEL_SOUP)
        SOUP_CHECKPOINTS = advanced.get("soup_checkpoints", SOUP_CHECKPOINTS)

    IMAGE_SIZE_OVERRIDE = recipe.get("image_size", None)

    # Loss extras (Tversky)
    if isinstance(loss, dict):
        TVERSKY_ALPHA = loss.get("tversky_alpha", TVERSKY_ALPHA)
        TVERSKY_BETA = loss.get("tversky_beta", TVERSKY_BETA)

    # Model extras (segmentation backbone, attention)
    if isinstance(model, dict):
        SEG_BACKBONE = model.get("seg_backbone", SEG_BACKBONE)
        SEG_USE_ATTENTION = model.get("seg_attention", SEG_USE_ATTENTION)

    # Albumentations augmentation
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

# Grad-CAM and ONNX export
GENERATE_GRADCAM = False      # generate Grad-CAM heatmaps after eval
GRADCAM_NUM_SAMPLES = 10      # number of samples for Grad-CAM
EXPORT_ONNX = False           # export model to ONNX after eval

# Model soup
USE_MODEL_SOUP = False        # model soups (weight averaging of checkpoints)
SOUP_CHECKPOINTS = []         # list of checkpoint paths for model soup

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
    "metric_name",
    "metric",
    "memory_gb",
    "status",
    "checkpoint_path",
    "description",
]
RUN_TIME_BUDGET = TIME_BUDGET
CURRENT_RUN_DATASET = DATASET
CURRENT_RUN_RECIPE = ""
CURRENT_RUN_BACKBONE = BACKBONE
CURRENT_RUN_SESSION = ""
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
    return ",".join(flags) if flags else "none"


def current_results_metadata(cfg, loss_type):
    return {
        "session": CURRENT_RUN_SESSION or "",
        "optimizer": OPTIMIZER,
        "loss": loss_type,
        "image_size": str(get_image_size(cfg)),
        "batch_size": str(BATCH_SIZE),
        "learning_rate": f"{LEARNING_RATE:.8g}",
        "weight_decay": f"{WEIGHT_DECAY:.8g}",
        "dropout": f"{DROPOUT:.4f}",
        "augmentations": summarize_augmentations(),
        "seed": str(RUN_SEED),
        "time_budget_seconds": f"{RUN_TIME_BUDGET:.1f}",
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
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class ModelEMA:
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model, decay=0.999):
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


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(metric_fn, preds, targets, n_bootstraps=1000, ci=0.95):
    """Compute bootstrap confidence interval for a metric function.

    Args:
        metric_fn: callable(preds, targets) -> float
        preds: numpy array of predictions
        targets: numpy array of targets
        n_bootstraps: number of bootstrap iterations
        ci: confidence level (default 0.95)

    Returns:
        (mean, lower, upper) tuple
    """
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
    mean = np.mean(scores)
    alpha = 1 - ci
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return mean, lower, upper


# ---------------------------------------------------------------------------
# MixUp
# ---------------------------------------------------------------------------

def mixup_data(images, targets, alpha=0.2):
    """Apply MixUp augmentation. Returns mixed inputs and pair of targets with lambda."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    return mixed_images, targets, targets[index], lam


def mixup_criterion(criterion, logits, targets_a, targets_b, lam):
    """Compute MixUp loss as weighted combination."""
    return lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)


# ---------------------------------------------------------------------------
# CutMix
# ---------------------------------------------------------------------------

def cutmix_data(images, targets, alpha=1.0):
    """Apply CutMix augmentation. Cuts a random patch from one image and pastes onto another."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    # Generate random bounding box
    _, _, H, W = images.shape
    cut_ratio = math.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cy = torch.randint(0, H, (1,)).item()
    cx = torch.randint(0, W, (1,)).item()

    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    # Adjust lambda to exactly match the pixel ratio
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)

    return mixed_images, targets, targets[index], lam


# ---------------------------------------------------------------------------
# Per-class threshold optimization
# ---------------------------------------------------------------------------

def optimize_thresholds(probs, targets, num_classes):
    """Find optimal sigmoid threshold per class for multilabel tasks.
    Maximizes F1 score per class.
    """
    thresholds = np.full(num_classes, 0.5)
    for i in range(num_classes):
        best_f1 = 0.0
        best_t = 0.5
        for t in np.arange(0.1, 0.9, 0.02):
            preds = (probs[:, i] >= t).astype(int)
            try:
                f1 = f1_score(targets[:, i], preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            except ValueError:
                continue
        thresholds[i] = best_t
    return thresholds


# ---------------------------------------------------------------------------
# Test-Time Augmentation
# ---------------------------------------------------------------------------

@torch.no_grad()
def tta_predict(model, images, device):
    """Average predictions over original + horizontally flipped images."""
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    with autocast_ctx:
        logits1 = model(images.to(device))
        logits2 = model(torch.flip(images.to(device), dims=[-1]))
    return (logits1 + logits2) / 2.0


# ---------------------------------------------------------------------------
# LLRD -- Layer-wise Learning Rate Decay
# ---------------------------------------------------------------------------

def get_llrd_param_groups(model, base_lr, decay=0.9):
    """Create per-layer parameter groups with exponentially decaying LR.
    Deeper (earlier) layers get lower LR. Head gets base_lr.
    """
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    if not hasattr(raw, "encoder"):
        return [{"params": model.parameters(), "lr": base_lr}]

    # Collect named parameters into layers
    encoder_params = list(raw.encoder.named_parameters())
    head_params = [p for n, p in raw.named_parameters()
                   if not n.startswith("encoder")]

    # Group encoder params by depth (heuristic: split by '.' depth)
    layer_groups = {}
    for name, param in encoder_params:
        parts = name.split(".")
        # Use first 2 parts as group key for depth
        depth_key = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
        if depth_key not in layer_groups:
            layer_groups[depth_key] = []
        layer_groups[depth_key].append(param)

    sorted_groups = list(layer_groups.values())
    num_layers = len(sorted_groups)

    param_groups = []
    for i, group in enumerate(sorted_groups):
        layer_lr = base_lr * (decay ** (num_layers - i))
        param_groups.append({"params": group, "lr": layer_lr})

    # Head at full LR
    param_groups.append({"params": head_params, "lr": base_lr})

    return param_groups


# ---------------------------------------------------------------------------
# Knowledge Distillation
# ---------------------------------------------------------------------------

def distillation_loss(student_logits, teacher_logits, targets, criterion,
                      alpha=0.5, temperature=3.0):
    """Combined task loss + KL divergence from teacher soft targets."""
    task_loss = criterion(student_logits, targets)

    # Soft targets from teacher
    with torch.no_grad():
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)

    kd_loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)

    return (1 - alpha) * task_loss + alpha * kd_loss


# ---------------------------------------------------------------------------
# Model Soup
# ---------------------------------------------------------------------------

def model_soup(checkpoint_paths, model):
    """Average weights from multiple checkpoints (model soup).
    Returns the model with averaged weights.
    """
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    state_dicts = []

    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dicts.append(ckpt["model_state_dict"])

    if not state_dicts:
        print("Model soup: no checkpoints found")
        return model

    # Average all state dicts
    avg_state = {}
    for key in state_dicts[0]:
        avg_state[key] = sum(sd[key].float() for sd in state_dicts) / len(state_dicts)

    raw.load_state_dict(avg_state)
    print(f"Model soup: averaged {len(state_dicts)} checkpoints")
    return model


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def generate_gradcam(model, dataloader, device, num_samples=10, save_dir=None):
    """Generate Grad-CAM heatmaps for interpretability.
    Works with classification models that have an encoder attribute.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save_dir is None:
        save_dir = os.path.join(CHECKPOINT_DIR, "gradcam")
    os.makedirs(save_dir, exist_ok=True)

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    if not hasattr(raw_model, "encoder"):
        print("Grad-CAM: model has no encoder attribute, skipping")
        return

    raw_model.eval()
    activations = {}
    gradients = {}

    # Hook into last conv layer
    target_layer = None
    for name, module in raw_model.encoder.named_modules():
        if isinstance(module, (nn.Conv2d,)):
            target_layer = module
            target_name = name

    if target_layer is None:
        print("Grad-CAM: no conv layer found, skipping")
        return

    def fwd_hook(m, inp, out):
        activations["value"] = out.detach()

    def bwd_hook(m, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    count = 0
    for images, targets in dataloader:
        if count >= num_samples:
            break
        for i in range(min(images.size(0), num_samples - count)):
            img = images[i:i+1].to(device).requires_grad_(True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                logits = raw_model(img.float())

            pred_class = logits.argmax(dim=1).item()
            logits[0, pred_class].backward()

            act = activations["value"][0]
            grad = gradients["value"][0]
            weights = grad.mean(dim=(1, 2), keepdim=True)
            cam = (weights * act).sum(dim=0).clamp(min=0)
            cam = cam / (cam.max() + 1e-8)
            cam = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=images.shape[-2:],
                mode="bilinear", align_corners=False,
            )[0, 0].cpu().numpy()

            # Save heatmap overlay
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            img_np = images[i].permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            axes[0].imshow(img_np)
            axes[0].set_title(f"Input (true={targets[i].item() if targets.dim() == 1 else 'multi'})")
            axes[0].axis("off")
            axes[1].imshow(img_np)
            axes[1].imshow(cam, alpha=0.5, cmap="jet")
            axes[1].set_title(f"Grad-CAM (pred={pred_class})")
            axes[1].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"gradcam_{count:03d}.png"), dpi=100)
            plt.close()

            raw_model.zero_grad()
            count += 1

    h1.remove()
    h2.remove()
    print(f"Grad-CAM: saved {count} heatmaps to {save_dir}")


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_onnx(model, input_shape, save_path=None, dataset_name="model"):
    """Export model to ONNX format with dynamic batch axis."""
    if save_path is None:
        save_path = os.path.join(CHECKPOINT_DIR, f"{dataset_name}.onnx")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.eval()
    raw_model = raw_model.cpu()

    dummy = torch.randn(1, *input_shape)
    try:
        torch.onnx.export(
            raw_model, dummy, save_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
        print(f"ONNX export: {save_path} ({os.path.getsize(save_path) / 1024 / 1024:.1f} MB)")
    except Exception as e:
        print(f"ONNX export failed: {e}")
    finally:
        raw_model.to("cuda")


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
    print("WARNING: CUDA not available. Falling back to CPU for smoke-test style runs.")
    return torch.device("cpu")


def maybe_compile_model(model, device):
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


def resolve_worker_count():
    if os.name == "nt":
        return max(0, min(NUM_WORKERS, 2))
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
    worker_count = resolve_worker_count()
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

    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=is_train,
        num_workers=worker_count,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_train and len(dataset) > effective_batch_size,
        persistent_workers=worker_count > 0,
        generator=generator,
        worker_init_fn=seed_worker if worker_count > 0 else None,
    )


def build_infinite_loader(dataset_name, cfg, batch_size, split="train"):
    loader = build_loader(dataset_name, cfg, batch_size, split=split)
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
    """LR schedule with multiple strategies.

    Strategies:
        "cosine": warmup -> constant -> cosine decay (default)
        "cosine_restart": cosine annealing with warm restarts (3 cycles)
        "one_cycle": triangular one-cycle policy (ideal for short training)
    """
    if LR_SCHEDULE == "cosine_restart":
        # Cosine annealing with warm restarts (3 cycles)
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        adj = (progress - WARMUP_RATIO) / (1.0 - WARMUP_RATIO)
        n_restarts = 3
        cycle_progress = (adj * n_restarts) % 1.0
        return 0.5 * (1.0 + math.cos(math.pi * cycle_progress))

    elif LR_SCHEDULE == "one_cycle":
        # One-cycle policy: ramp up to peak then decay
        if progress < 0.3:
            # Warmup phase: ramp from 0.1x to 1x
            return 0.1 + 0.9 * (progress / 0.3)
        elif progress < 0.7:
            # Sustain at peak with gentle cosine
            p = (progress - 0.3) / 0.4
            return 1.0 - 0.1 * (1 - math.cos(math.pi * p)) / 2
        else:
            # Annihilation phase: decay to near zero
            p = (progress - 0.7) / 0.3
            return max(0.01, 0.9 * 0.5 * (1.0 + math.cos(math.pi * p)))

    else:
        # Default: warmup -> constant -> cosine decay
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        elif progress < 1.0 - WARMDOWN_RATIO:
            return 1.0
        else:
            cooldown_progress = (1.0 - progress) / WARMDOWN_RATIO
            return 0.5 * (1.0 + math.cos(math.pi * (1 - cooldown_progress)))


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
    parts.extend([backbone_name, OPTIMIZER, loss_type, f"{get_image_size(cfg)}px"])
    if AUG_USE_CLAHE:
        parts.append("clahe")
    return " | ".join(parts)

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
    global CURRENT_RUN_DATASET, CURRENT_RUN_RECIPE, CURRENT_RUN_BACKBONE, CURRENT_RUN_SESSION, RUN_TIME_BUDGET
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="Load a recipe from recipes/ (for example: chexnet, aptos_winner)",
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
        "--session",
        type=str,
        default=None,
        help="Optional run-session tag used to isolate a fresh overnight search in results.tsv and downstream artifacts.",
    )
    parser.add_argument("--list-recipes", action="store_true", help="List available recipes and exit")
    args = parser.parse_args()

    if args.list_recipes:
        list_recipes()
        return

    if args.recipe:
        recipe = load_recipe(args.recipe)
        apply_recipe(recipe)

    dataset = args.dataset or DATASET
    CURRENT_RUN_DATASET = dataset
    CURRENT_RUN_RECIPE = args.recipe or ""
    CURRENT_RUN_SESSION = (args.session or "").strip()
    RUN_TIME_BUDGET = float(args.time_budget) if args.time_budget is not None else float(TIME_BUDGET)
    if RUN_TIME_BUDGET <= 0:
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
            multilabel=is_multilabel,
        )
        batch_size = BATCH_SIZE
    elif task == "segmentation":
        if SEG_BACKBONE:
            model = PretrainedUNet(
                num_classes=num_classes,
                backbone=SEG_BACKBONE,
                pretrained=PRETRAINED,
                in_channels=cfg["in_channels"],
                use_attention=SEG_USE_ATTENTION,
            )
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

    # Gradual unfreezing: freeze backbone initially
    if USE_GRADUAL_UNFREEZE and task in ("classification", "multilabel"):
        raw = model
        if hasattr(model, "_orig_mod"):
            raw = model._orig_mod
        for param in raw.encoder.parameters():
            param.requires_grad = False
        print("Gradual unfreeze: backbone frozen initially")

    model = maybe_compile_model(model, device)
    num_params = sum(param.numel() for param in model.parameters())
    backbone_name = BACKBONE if task != "segmentation" else (SEG_BACKBONE or "unet")
    CURRENT_RUN_BACKBONE = backbone_name

    # EMA setup
    ema = None
    if USE_EMA:
        ema = ModelEMA(model, decay=EMA_DECAY)
        print(f"EMA enabled (decay={EMA_DECAY})")

    print(f"Model: {backbone_name}")
    print(f"Pretrained: {PRETRAINED}")
    print(f"Parameters: {num_params:,}")
    print(f"Time budget: {RUN_TIME_BUDGET}s")
    print(f"Batch size: {batch_size}")
    print(f"Optimizer: {OPTIMIZER}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Image size: {get_image_size(cfg)}")

    # Optimizer -- with LLRD, discriminative LR, or standard
    if USE_LLRD and task in ("classification", "multilabel"):
        param_groups = get_llrd_param_groups(model, LEARNING_RATE, decay=LLRD_DECAY)
        print(f"LLRD enabled (decay={LLRD_DECAY}, {len(param_groups)} groups)")
    elif DISCRIMINATIVE_LR and task in ("classification", "multilabel"):
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        param_groups = [
            {"params": raw.encoder.parameters(), "lr": LEARNING_RATE * BACKBONE_LR_FACTOR},
            {"params": raw.head.parameters(), "lr": LEARNING_RATE},
            {"params": raw.dropout.parameters(), "lr": LEARNING_RATE},
        ]
        print(f"Discriminative LR: backbone={LEARNING_RATE * BACKBONE_LR_FACTOR:.6f}, head={LEARNING_RATE:.6f}")
    else:
        param_groups = [{"params": model.parameters(), "lr": LEARNING_RATE}]

    if OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(param_groups, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(
            param_groups, momentum=0.9, weight_decay=WEIGHT_DECAY
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
    elif loss_type == "tversky":
        criterion = TverskyLoss(alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA)
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

    train_loader = build_infinite_loader(dataset, cfg, batch_size, split="train")
    val_loader = build_loader(dataset, cfg, batch_size, split="val")
    images, targets, epoch = next(train_loader)

    # -----------------------------------------------------------------------
    # Training loop setup
    # -----------------------------------------------------------------------

    smooth_train_loss = 0.0
    total_training_time = 0.0
    step = 0
    accum_step = 0
    lrm = 0.0  # will be set on first optimizer step
    backbone_unfrozen = not USE_GRADUAL_UNFREEZE
    loss_history = []  # for richer checkpoints

    # SWA setup
    swa_model = None
    swa_started = False
    if USE_SWA:
        print(f"SWA enabled (starts at {SWA_START_RATIO*100:.0f}% progress)")

    # Teacher model for distillation
    teacher_model = None
    if USE_DISTILLATION and TEACHER_CHECKPOINT and os.path.exists(TEACHER_CHECKPOINT):
        teacher_ckpt = torch.load(TEACHER_CHECKPOINT, map_location=device, weights_only=False)
        teacher_cfg = teacher_ckpt.get("config", {})
        teacher_backbone = teacher_cfg.get("backbone", BACKBONE)
        teacher_model = MedicalClassifier(
            num_classes=num_classes, backbone=teacher_backbone,
            pretrained=False, dropout=0.0, multilabel=is_multilabel,
        ).to(device)
        teacher_model.load_state_dict(teacher_ckpt["model_state_dict"])
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        print(f"Distillation: teacher loaded from {TEACHER_CHECKPOINT}")

    if GRAD_ACCUM_STEPS > 1:
        print(f"Gradient accumulation: {GRAD_ACCUM_STEPS} steps (effective BS={batch_size * GRAD_ACCUM_STEPS})")
    if USE_MIXUP:
        print(f"MixUp enabled (alpha={MIXUP_ALPHA})")
    if USE_CUTMIX:
        print(f"CutMix enabled (alpha={CUTMIX_ALPHA})")
    if USE_TTA:
        print("TTA enabled (horizontal flip)")
    if LR_SCHEDULE != "cosine":
        print(f"LR schedule: {LR_SCHEDULE}")

    model.train()

    while True:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        images_d = images.to(device, non_blocking=device.type == "cuda")
        targets_d = targets.to(device, non_blocking=device.type == "cuda")

        # MixUp or CutMix (randomly choose one per batch)
        use_mix = False
        if USE_MIXUP and USE_CUTMIX:
            if torch.rand(1).item() < 0.5:
                images_d, targets_a, targets_b, lam = mixup_data(images_d, targets_d, MIXUP_ALPHA)
            else:
                images_d, targets_a, targets_b, lam = cutmix_data(images_d, targets_d, CUTMIX_ALPHA)
            use_mix = True
        elif USE_MIXUP:
            images_d, targets_a, targets_b, lam = mixup_data(images_d, targets_d, MIXUP_ALPHA)
            use_mix = True
        elif USE_CUTMIX:
            images_d, targets_a, targets_b, lam = cutmix_data(images_d, targets_d, CUTMIX_ALPHA)
            use_mix = True

        with autocast_context(device):
            logits = model(images_d)
            if use_mix:
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            elif teacher_model is not None:
                with torch.no_grad():
                    teacher_logits = teacher_model(images_d)
                loss = distillation_loss(
                    logits, teacher_logits, targets_d, criterion,
                    alpha=DISTILL_ALPHA, temperature=DISTILL_TEMPERATURE,
                )
            else:
                loss = criterion(logits, targets_d)
            if GRAD_ACCUM_STEPS > 1:
                loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        accum_step += 1

        # Only step optimizer after accumulation
        if accum_step % GRAD_ACCUM_STEPS == 0:
            progress = min(total_training_time / RUN_TIME_BUDGET, 1.0)
            lrm = get_lr_multiplier(progress)

            # Gradual unfreeze check
            if not backbone_unfrozen and progress >= UNFREEZE_AFTER_RATIO:
                raw = model._orig_mod if hasattr(model, "_orig_mod") else model
                for param in raw.encoder.parameters():
                    param.requires_grad = True
                backbone_unfrozen = True
                print(f"\nBackbone unfrozen at {progress*100:.1f}% progress")

            if DISCRIMINATIVE_LR:
                for i, param_group in enumerate(optimizer.param_groups):
                    if i == 0:  # backbone
                        param_group["lr"] = LEARNING_RATE * BACKBONE_LR_FACTOR * lrm
                    else:  # head / dropout
                        param_group["lr"] = LEARNING_RATE * lrm
            else:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = LEARNING_RATE * lrm

            # Gradient clipping
            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # EMA update
            if ema is not None:
                ema.update(model)

            # SWA update
            if USE_SWA and progress >= SWA_START_RATIO:
                if not swa_started:
                    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
                    swa_model = torch.optim.swa_utils.AveragedModel(raw)
                    swa_started = True
                    print(f"\nSWA started at {progress*100:.1f}% progress")
                swa_model.update_parameters(
                    model._orig_mod if hasattr(model, "_orig_mod") else model
                )

        train_loss_f = loss.item() * (GRAD_ACCUM_STEPS if GRAD_ACCUM_STEPS > 1 else 1)

        if train_loss_f > 100 or train_loss_f != train_loss_f:
            raise RuntimeError("loss exploded or became NaN")

        # Track loss history for checkpoints (every 100 steps)
        if step % 100 == 0:
            loss_history.append({"step": step, "loss": round(train_loss_f, 6)})

        images, targets, epoch = next(train_loader)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        if step > 5:
            total_training_time += dt

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        progress = min(total_training_time / RUN_TIME_BUDGET, 1.0)
        pct_done = 100 * progress
        remaining = max(0.0, RUN_TIME_BUDGET - total_training_time)
        current_lr = LEARNING_RATE * lrm

        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | "
            f"loss: {debiased_smooth_loss:.4f} | "
            f"lr: {current_lr:.6f} | "
            f"dt: {dt * 1000:.0f}ms | "
            f"epoch: {epoch} | "
            f"remaining: {remaining:.0f}s    ",
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
        if step > 5 and total_training_time >= RUN_TIME_BUDGET:
            break

    print()

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    # Choose eval model: SWA > EMA > base
    eval_model = model
    if swa_model is not None:
        # Update BN statistics for SWA model
        swa_loader = build_loader(dataset, cfg, batch_size, split="train")
        torch.optim.swa_utils.update_bn(swa_loader, swa_model, device=device)
        eval_model = swa_model
        print("Using SWA model for evaluation")
    elif ema is not None:
        eval_model = ema.ema_model.to(device)
        print("Using EMA model for evaluation")

    # Model soup mode
    if USE_MODEL_SOUP and SOUP_CHECKPOINTS:
        eval_model = model_soup(SOUP_CHECKPOINTS, eval_model)

    eval_model.eval()
    print("---")

    per_class_thresholds = None
    confidence_intervals = None

    if task == "multilabel":
        results = evaluate_multilabel_loader(eval_model, val_loader, cfg, device)
        print(f"mean_auc:         {results['mean_auc']:.2f}%")

        # Bootstrap confidence intervals
        if COMPUTE_CI:
            all_probs = results["artifacts"]["val_probs"].astype(np.float32)
            all_targets = results["artifacts"]["val_targets"].astype(np.float32)
            def _mean_auc_fn(p, t):
                aucs = []
                for i in range(p.shape[1]):
                    try:
                        aucs.append(roc_auc_score(t[:, i], p[:, i]))
                    except ValueError:
                        pass
                return 100.0 * np.mean(aucs) if aucs else 0.0
            _, ci_lower, ci_upper = bootstrap_ci(
                _mean_auc_fn, all_probs, all_targets, n_bootstraps=CI_BOOTSTRAPS,
            )
            confidence_intervals = {"mean_auc": {"lower": ci_lower, "upper": ci_upper}}
            print(f"mean_auc_ci:      {results['mean_auc']:.2f}% (95% CI: {ci_lower:.2f}-{ci_upper:.2f})")

        print("per_class_auc:")
        for pathology, auc in results["per_class_auc"].items():
            print(f"  {pathology:<22s} {auc:.2f}%")

        # Per-class threshold optimization
        if OPTIMIZE_THRESHOLDS and "artifacts" in results:
            per_class_thresholds = optimize_thresholds(
                results["artifacts"]["val_probs"].astype(np.float32),
                results["artifacts"]["val_targets"].astype(np.float32),
                num_classes,
            )
            class_names = cfg.get("class_names", [f"class_{i}" for i in range(num_classes)])
            print("optimized_thresholds:")
            for i, name in enumerate(class_names):
                print(f"  {name:<22s} {per_class_thresholds[i]:.2f}")

    elif task == "classification":
        results = evaluate_classification_loader(eval_model, val_loader, cfg, device)
        print(f"val_accuracy:     {results['accuracy']:.2f}%")
        print(f"val_auc:          {results['auc']:.2f}%")
        print(f"val_kappa:        {results['kappa']:.4f}")
        print(f"val_loss:         {results['loss']:.6f}")

        # Bootstrap confidence intervals
        if COMPUTE_CI:
            all_probs = results["artifacts"]["val_probs"].astype(np.float32)
            all_targets = results["artifacts"]["val_targets"].astype(np.int64)
            metric_name_for_ci = cfg.get("metric", "accuracy")
            if metric_name_for_ci == "quadratic_kappa":
                def _kappa_fn(p, t):
                    return cohen_kappa_score(t, p.argmax(axis=1), weights="quadratic")
                _, ci_lower, ci_upper = bootstrap_ci(
                    _kappa_fn, all_probs, all_targets, n_bootstraps=CI_BOOTSTRAPS,
                )
                confidence_intervals = {"primary": {"lower": ci_lower, "upper": ci_upper}}
                print(f"confidence_int:   {results['kappa']:.4f} (95% CI: {ci_lower:.4f}-{ci_upper:.4f})")
            else:
                def _acc_fn(p, t):
                    return 100.0 * (p.argmax(axis=1) == t).mean()
                _, ci_lower, ci_upper = bootstrap_ci(
                    _acc_fn, all_probs, all_targets, n_bootstraps=CI_BOOTSTRAPS,
                )
                confidence_intervals = {"primary": {"lower": ci_lower, "upper": ci_upper}}
                print(f"confidence_int:   {results['accuracy']:.2f}% (95% CI: {ci_lower:.2f}-{ci_upper:.2f})")

    else:
        results = evaluate_segmentation_loader(eval_model, val_loader, device)
        print(f"val_dice:         {results['dice']:.2f}%")
        print(f"val_iou:          {results['iou']:.2f}%")
        print(f"val_pixel_acc:    {results['pixel_acc']:.2f}%")
        print(f"val_sensitivity:  {results['sensitivity']:.2f}%")
        print(f"val_specificity:  {results['specificity']:.2f}%")

    # Grad-CAM generation
    if GENERATE_GRADCAM and task in ("classification", "multilabel"):
        print("Generating Grad-CAM heatmaps...")
        generate_gradcam(eval_model, val_loader, device,
                         num_samples=GRADCAM_NUM_SAMPLES)

    # ONNX export
    if EXPORT_ONNX:
        img_size = get_image_size(cfg)
        in_ch = 3  # pretrained backbones always expect 3ch
        export_onnx(eval_model, (in_ch, img_size, img_size),
                     dataset_name=dataset)

    metric_name, primary_metric = primary_metric_for_results(results, cfg)
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0
    memory_gb = peak_vram_mb / 1024.0

    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"num_epochs:       {epoch}")
    print(f"dataset:          {dataset}")
    if CURRENT_RUN_SESSION:
        print(f"session:          {CURRENT_RUN_SESSION}")
    print(f"task:             {task}")
    print(f"backbone:         {backbone_name}")
    if args.recipe:
        print(f"recipe:           {args.recipe}")

    checkpoint_path = ""
    if args.save_checkpoint:
        ckpt_config = {
            "dataset": dataset,
            "dataset_source": cfg.get("source", "builtin"),
            "task": task,
            "metric_name": metric_name,
            "backbone": backbone_name,
            "num_classes": num_classes,
            "class_names": cfg.get("class_names", []),
            "batch_size": batch_size,
            "dropout": DROPOUT,
            "optimizer": OPTIMIZER,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": loss_type,
            "recipe": args.recipe,
            "session": CURRENT_RUN_SESSION,
            "seed": RUN_SEED,
            "time_budget_seconds": RUN_TIME_BUDGET,
            "selection_split": "val",
            "image_size": get_image_size(cfg),
            "ema": USE_EMA, "tta": USE_TTA, "mixup": USE_MIXUP,
            "cutmix": USE_CUTMIX, "swa": USE_SWA,
            "llrd": USE_LLRD, "llrd_decay": LLRD_DECAY,
            "lr_schedule": LR_SCHEDULE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "gradual_unfreeze": USE_GRADUAL_UNFREEZE,
            "discriminative_lr": DISCRIMINATIVE_LR,
            "distillation": USE_DISTILLATION,
            "seg_backbone": SEG_BACKBONE,
            "seg_attention": SEG_USE_ATTENTION,
            "augmentations": {
                "horizontal_flip": AUG_HORIZONTAL_FLIP,
                "vertical_flip": AUG_VERTICAL_FLIP,
                "rotation": AUG_ROTATION_DEGREES,
                "color_jitter": AUG_COLOR_JITTER,
                "clahe": AUG_USE_CLAHE,
                "elastic_transform": ELASTIC_TRANSFORM,
                "grid_distortion": GRID_DISTORTION,
                "coarse_dropout": COARSE_DROPOUT,
            },
        }
        ckpt_name = f"{dataset}_{backbone_name}_{primary_metric:.2f}.pth"
        checkpoint_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
        save_model = swa_model if swa_model is not None else (
            ema.ema_model if ema is not None else model)
        save_checkpoint(save_model, results, ckpt_config, checkpoint_path)

    append_results_row(
        dataset=dataset,
        recipe=args.recipe,
        backbone=backbone_name,
        metric_name=metric_name,
        metric=primary_metric,
        memory_gb=memory_gb,
        status="candidate",
        checkpoint_path=checkpoint_path,
        description=build_run_description(dataset, cfg, args.recipe, backbone_name, loss_type),
        metadata=current_results_metadata(cfg, loss_type),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
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
                metadata={"session": CURRENT_RUN_SESSION or "", "seed": str(RUN_SEED)},
            )
        except Exception:
            pass
        raise
