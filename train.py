"""
Autoresearch-MedVision training script. Single-GPU, single-file.
Default configuration reproduces CheXNet (Rajpurkar et al., 2017):
  DenseNet-121 pretrained on ImageNet, fine-tuned on ChestX-ray14
  for 14-pathology multi-label classification.

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

import gc
import time
import argparse
import csv
import json
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

from prepare import (
    TIME_BUDGET, DATASETS,
    make_infinite_dataloader, make_dataloader,
    evaluate_classification, evaluate_multilabel, evaluate_segmentation,
    get_class_weights,
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
    global SEG_BATCH_SIZE, SEG_BASE_CHANNELS, IMAGE_SIZE_OVERRIDE, AUGMENTATION_CONFIG
    global GRAD_ACCUM_STEPS, USE_EMA, EMA_DECAY, USE_TTA, USE_MIXUP, MIXUP_ALPHA
    global USE_GRADUAL_UNFREEZE, UNFREEZE_AFTER_RATIO, DISCRIMINATIVE_LR, BACKBONE_LR_FACTOR

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

    IMAGE_SIZE_OVERRIDE = recipe.get("image_size", None)
    augmentation = recipe.get("augmentation", {})
    AUGMENTATION_CONFIG = augmentation if isinstance(augmentation, dict) else {}

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

# Advanced training
GRAD_ACCUM_STEPS = 1          # gradient accumulation steps (effective BS = BATCH_SIZE * GRAD_ACCUM_STEPS)
USE_EMA = False               # exponential moving average of model weights
EMA_DECAY = 0.999             # EMA decay rate
USE_TTA = False               # test-time augmentation (horizontal flip)
USE_MIXUP = False             # MixUp regularization
MIXUP_ALPHA = 0.2             # MixUp interpolation strength
USE_GRADUAL_UNFREEZE = False  # freeze backbone initially, then unfreeze
UNFREEZE_AFTER_RATIO = 0.1    # fraction of time budget before unfreezing backbone
DISCRIMINATIVE_LR = False     # lower LR for backbone, higher for head
BACKBONE_LR_FACTOR = 0.1      # backbone LR = LEARNING_RATE * this factor

# Segmentation-specific
SEG_BATCH_SIZE = 8            # batch size for segmentation (larger images)
SEG_BASE_CHANNELS = 64        # U-Net base channels

# Image size override (None = use dataset default)
IMAGE_SIZE_OVERRIDE = None
AUGMENTATION_CONFIG = {}

# Checkpointing
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.tsv")


def metric_key_for_dataset(cfg, task):
    """Map dataset config metrics to the evaluation output keys."""
    if task == "multilabel":
        return "mean_auc"
    if task == "segmentation":
        return "dice"

    if cfg["metric"] == "quadratic_kappa":
        return "kappa"
    return cfg["metric"]


def format_metric_for_filename(metric_value):
    """Keep filenames readable for both 0-1 and percentage-style metrics."""
    if metric_value < 10:
        return f"{metric_value:.4f}"
    return f"{metric_value:.1f}"


def build_results_description(dataset, task, backbone_name, loss_type, recipe_name):
    """Encode enough context to compare runs from the same task in results.tsv."""
    recipe_value = recipe_name or "-"
    extras = []
    if USE_EMA:
        extras.append("ema")
    if USE_TTA:
        extras.append("tta")
    if USE_MIXUP:
        extras.append("mixup")
    if USE_GRADUAL_UNFREEZE:
        extras.append("unfreeze")
    if DISCRIMINATIVE_LR:
        extras.append("disc_lr")
    if GRAD_ACCUM_STEPS > 1:
        extras.append(f"accum{GRAD_ACCUM_STEPS}")
    extras_str = "+".join(extras) if extras else "-"
    return (
        f"dataset={dataset};task={task};backbone={backbone_name};"
        f"recipe={recipe_value};loss={loss_type};extras={extras_str}"
    )


def get_git_commit():
    """Return the current short git commit hash, if available."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def ensure_results_file():
    """Create the default results log if it does not exist."""
    if os.path.exists(RESULTS_FILE):
        return

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["commit", "metric", "memory_gb", "status", "description"])


def read_best_metric(dataset, task):
    """Read the best prior metric for the same dataset/task from results.tsv."""
    if not os.path.exists(RESULTS_FILE):
        return None

    best = None
    with open(RESULTS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            description = row.get("description", "")
            if f"dataset={dataset};" not in description or f"task={task};" not in description:
                continue
            try:
                metric = float(row["metric"])
            except (KeyError, TypeError, ValueError):
                continue
            if best is None or metric > best:
                best = metric

    return best


def append_results_row(dataset, task, backbone_name, loss_type, recipe_name, metric_value, peak_vram_mb):
    """Append a run summary to results.tsv and classify it against prior runs."""
    ensure_results_file()

    prior_best = read_best_metric(dataset, task)
    status = "keep" if prior_best is None or metric_value >= prior_best else "discard"
    description = build_results_description(dataset, task, backbone_name, loss_type, recipe_name)

    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            get_git_commit(),
            f"{metric_value:.6f}",
            f"{peak_vram_mb / 1024:.2f}",
            status,
            description,
        ])

    return status

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


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    """LR schedule: warmup -> constant -> cosine decay."""
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown_progress = (1.0 - progress) / WARMDOWN_RATIO
        return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * (1 - cooldown_progress))).item())


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
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
# Checkpointing
# ---------------------------------------------------------------------------

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--recipe", type=str, default=None,
                        help="Load a recipe from recipes/ (e.g., chexnet, aptos_winner)")
    parser.add_argument("--save-checkpoint", action="store_true",
                        help="Save best model checkpoint to checkpoints/")
    parser.add_argument("--list-recipes", action="store_true",
                        help="List available recipes and exit")
    args = parser.parse_args()

    # List recipes
    if args.list_recipes:
        recipe_dir = Path(__file__).parent / "recipes"
        print("\nAvailable recipes:\n")
        for recipe_file in sorted(recipe_dir.glob("*.yaml")):
            name = recipe_file.stem
            # Read first non-comment line for description
            with open(recipe_file) as f:
                desc = ""
                for line in f:
                    line = line.strip()
                    if line.startswith("#") and not desc:
                        desc = line.lstrip("# ").strip()
                        break
            print(f"  {name:<24s} {desc}")
        print(f"\nUsage: uv run train.py --recipe <name>")
        return

    # Load recipe if specified
    if args.recipe:
        recipe = load_recipe(args.recipe)
        apply_recipe(recipe)

    # CLI overrides recipe
    dataset = args.dataset or DATASET
    cfg = DATASETS[dataset]
    task = cfg["task"]
    num_classes = cfg["num_classes"]
    is_multilabel = (task == "multilabel")

    print(f"Task: {task}")
    print(f"Dataset: {dataset}")
    print(f"Num classes: {num_classes}")
    if is_multilabel:
        print(f"Pathologies: {', '.join(cfg['class_names'])}")

    # Setup
    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    scaler = torch.amp.GradScaler()

    # Build model
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
    num_params = sum(p.numel() for p in model.parameters())
    backbone_name = BACKBONE if task != "segmentation" else "unet"
    print(f"Model: {backbone_name}")
    print(f"Pretrained: {PRETRAINED}")
    print(f"Parameters: {num_params:,}")

    # Gradual unfreezing: freeze backbone initially
    if USE_GRADUAL_UNFREEZE and task in ("classification", "multilabel"):
        raw = model
        if hasattr(model, "_orig_mod"):
            raw = model._orig_mod
        for param in raw.encoder.parameters():
            param.requires_grad = False
        print("Gradual unfreeze: backbone frozen initially")

    # Compile model
    model = torch.compile(model, dynamic=False)

    # EMA setup
    ema = None
    if USE_EMA:
        ema = ModelEMA(model, decay=EMA_DECAY)
        print(f"EMA enabled (decay={EMA_DECAY})")

    # Optimizer — with discriminative LR if enabled
    if DISCRIMINATIVE_LR and task in ("classification", "multilabel"):
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
        optimizer = torch.optim.Adam(
            param_groups, weight_decay=WEIGHT_DECAY,
        )
    elif OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=WEIGHT_DECAY,
        )
    elif OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(
            param_groups, momentum=0.9, weight_decay=WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

    # Loss function
    loss_type = LOSS_TYPE
    if loss_type == "auto":
        if is_multilabel:
            loss_type = "bce"
        elif task == "segmentation":
            loss_type = "dice_bce"
        else:
            loss_type = "cross_entropy"

    if loss_type == "bce":
        if USE_CLASS_WEIGHTS:
            try:
                w = get_class_weights(dataset, device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=w)
            except Exception:
                criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
    elif loss_type == "focal" and is_multilabel:
        if USE_CLASS_WEIGHTS:
            try:
                w = get_class_weights(dataset, device)
                criterion = FocalBCELoss(gamma=FOCAL_GAMMA, pos_weight=w)
            except Exception:
                criterion = FocalBCELoss(gamma=FOCAL_GAMMA)
        else:
            criterion = FocalBCELoss(gamma=FOCAL_GAMMA)
    elif loss_type == "focal":
        if USE_CLASS_WEIGHTS:
            try:
                w = get_class_weights(dataset, device)
                criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=w,
                                      label_smoothing=LABEL_SMOOTHING)
            except Exception:
                criterion = FocalLoss(gamma=FOCAL_GAMMA,
                                      label_smoothing=LABEL_SMOOTHING)
        else:
            criterion = FocalLoss(gamma=FOCAL_GAMMA,
                                  label_smoothing=LABEL_SMOOTHING)
    elif loss_type == "dice_bce":
        criterion = DiceBCELoss(dice_weight=0.5)
    else:
        # cross_entropy
        if USE_CLASS_WEIGHTS:
            try:
                w = get_class_weights(dataset, device)
                criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTHING)
            except Exception:
                criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    print(f"Loss: {loss_type}")

    # Dataloader
    train_loader = make_infinite_dataloader(
        dataset,
        batch_size,
        split="train",
        image_size=IMAGE_SIZE_OVERRIDE,
        augmentation_config=AUGMENTATION_CONFIG,
    )
    images, targets, epoch = next(train_loader)

    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Batch size: {batch_size}")
    print(f"Optimizer: {OPTIMIZER}")
    print(f"Learning rate: {LEARNING_RATE}")
    if IMAGE_SIZE_OVERRIDE is not None:
        print(f"Image size: {IMAGE_SIZE_OVERRIDE}")
    if AUGMENTATION_CONFIG:
        enabled_augments = ", ".join(sorted(k for k, v in AUGMENTATION_CONFIG.items() if v))
        if enabled_augments:
            print(f"Augmentation: {enabled_augments}")
    print()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    t_start_training = time.time()
    smooth_train_loss = 0
    total_training_time = 0
    step = 0
    accum_step = 0
    lrm = 0.0  # will be set on first optimizer step
    backbone_unfrozen = not USE_GRADUAL_UNFREEZE

    if GRAD_ACCUM_STEPS > 1:
        print(f"Gradient accumulation: {GRAD_ACCUM_STEPS} steps (effective BS={batch_size * GRAD_ACCUM_STEPS})")
    if USE_MIXUP:
        print(f"MixUp enabled (alpha={MIXUP_ALPHA})")
    if USE_TTA:
        print("TTA enabled (horizontal flip)")

    model.train()

    while True:
        torch.cuda.synchronize()
        t0 = time.time()

        images_d = images.to(device, non_blocking=True)
        targets_d = targets.to(device, non_blocking=True)

        # MixUp
        if USE_MIXUP:
            images_d, targets_a, targets_b, lam = mixup_data(images_d, targets_d, MIXUP_ALPHA)

        with autocast_ctx:
            logits = model(images_d)
            if USE_MIXUP:
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                loss = criterion(logits, targets_d)
            if GRAD_ACCUM_STEPS > 1:
                loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        accum_step += 1

        # Only step optimizer after accumulation
        if accum_step % GRAD_ACCUM_STEPS == 0:
            progress = min(total_training_time / TIME_BUDGET, 1.0)
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

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # EMA update
            if ema is not None:
                ema.update(model)

        train_loss_f = loss.item() * (GRAD_ACCUM_STEPS if GRAD_ACCUM_STEPS > 1 else 1)

        if train_loss_f > 100 or train_loss_f != train_loss_f:
            print("FAIL: loss exploded or NaN")
            exit(1)

        images, targets, epoch = next(train_loader)

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        if step > 5:
            total_training_time += dt

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        pct_done = 100 * progress
        remaining = max(0, TIME_BUDGET - total_training_time)
        current_lr = LEARNING_RATE * lrm

        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | "
            f"loss: {debiased_smooth_loss:.4f} | "
            f"lr: {current_lr:.6f} | "
            f"dt: {dt * 1000:.0f}ms | "
            f"epoch: {epoch} | "
            f"remaining: {remaining:.0f}s    ",
            end="", flush=True,
        )

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        if step > 5 and total_training_time >= TIME_BUDGET:
            break

    print()

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    # Use EMA model for evaluation if available
    eval_model = model
    if ema is not None:
        eval_model = ema.ema_model.to(device)
        print("Using EMA model for evaluation")

    eval_model.eval()
    print("---")
    primary_metric_key = metric_key_for_dataset(cfg, task)

    if is_multilabel:
        results = evaluate_multilabel(
            eval_model,
            dataset,
            batch_size,
            device,
            image_size=IMAGE_SIZE_OVERRIDE,
            augmentation_config=AUGMENTATION_CONFIG,
        )
        primary_metric = results[primary_metric_key]
        print(f"mean_auc:         {results['mean_auc']:.2f}%")
        print("per_class_auc:")
        for pathology, auc in results["per_class_auc"].items():
            print(f"  {pathology:<22s} {auc:.2f}%")

    elif task == "classification":
        results = evaluate_classification(
            eval_model,
            dataset,
            batch_size,
            device,
            image_size=IMAGE_SIZE_OVERRIDE,
            augmentation_config=AUGMENTATION_CONFIG,
        )
        primary_metric = results[primary_metric_key]
        print(f"val_accuracy:     {results['accuracy']:.2f}%")
        print(f"val_auc:          {results['auc']:.2f}%")
        print(f"val_kappa:        {results['kappa']:.4f}")
        print(f"val_loss:         {results['loss']:.6f}")

    elif task == "segmentation":
        results = evaluate_segmentation(
            eval_model,
            dataset,
            batch_size,
            device,
            image_size=IMAGE_SIZE_OVERRIDE,
        )
        primary_metric = results[primary_metric_key]
        print(f"val_dice:         {results['dice']:.2f}%")
        print(f"val_iou:          {results['iou']:.2f}%")
        print(f"val_pixel_acc:    {results['pixel_acc']:.2f}%")
        print(f"val_sensitivity:  {results['sensitivity']:.2f}%")
        print(f"val_specificity:  {results['specificity']:.2f}%")

    # Summary
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"num_epochs:       {epoch}")
    print(f"dataset:          {dataset}")
    print(f"task:             {task}")
    print(f"backbone:         {backbone_name}")
    if args.recipe:
        print(f"recipe:           {args.recipe}")

    results_status = append_results_row(
        dataset,
        task,
        backbone_name,
        loss_type,
        args.recipe,
        primary_metric,
        peak_vram_mb,
    )
    print(f"results_tsv:      {RESULTS_FILE}")
    print(f"results_status:   {results_status}")

    # Save checkpoint if requested
    if args.save_checkpoint:
        ckpt_config = {
            "dataset": dataset, "task": task, "backbone": backbone_name,
            "num_classes": num_classes, "batch_size": batch_size,
            "optimizer": OPTIMIZER, "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "loss": loss_type,
            "recipe": args.recipe, "dropout": DROPOUT,
            "ema": USE_EMA, "tta": USE_TTA, "mixup": USE_MIXUP,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "gradual_unfreeze": USE_GRADUAL_UNFREEZE,
            "discriminative_lr": DISCRIMINATIVE_LR,
            "image_size": IMAGE_SIZE_OVERRIDE,
            "augmentation_config": AUGMENTATION_CONFIG,
        }
        ckpt_name = f"{dataset}_{backbone_name}_{format_metric_for_filename(primary_metric)}.pth"
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
        save_model = ema.ema_model if ema is not None else model
        save_checkpoint(save_model, results, ckpt_config, ckpt_path)


if __name__ == "__main__":
    main()
