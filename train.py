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
import json
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
    except ImportError:
        yaml = None

    recipe_dir = Path(__file__).parent / "recipes"
    recipe_path = recipe_dir / f"{recipe_name}.yaml"

    if not recipe_path.exists():
        available = [f.stem for f in recipe_dir.glob("*.yaml")]
        print(f"Recipe '{recipe_name}' not found.")
        print(f"Available recipes: {', '.join(sorted(available))}")
        exit(1)

    if yaml is not None:
        with open(recipe_path) as f:
            return yaml.safe_load(f)
    else:
        # Minimal YAML parser for simple key-value configs (no pyyaml dependency)
        config = {}
        with open(recipe_path) as f:
            content = f.read()
        # Simple parser: extract key fields we care about
        import re
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("#") or not line or line.startswith("-") or line.startswith("|"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
                elif val.replace(".", "").replace("-", "").isdigit():
                    val = float(val) if "." in val else int(val)
                if key and val != "":
                    config[key] = val
        return config


def apply_recipe(recipe):
    """Apply recipe config to global hyperparameters. Returns overridden values."""
    global DATASET, BACKBONE, PRETRAINED, DROPOUT, BATCH_SIZE, LEARNING_RATE
    global OPTIMIZER, WEIGHT_DECAY, WARMUP_RATIO, WARMDOWN_RATIO
    global USE_CLASS_WEIGHTS, LABEL_SMOOTHING, LOSS_TYPE, FOCAL_GAMMA
    global SEG_BATCH_SIZE, SEG_BASE_CHANNELS, IMAGE_SIZE_OVERRIDE

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

    IMAGE_SIZE_OVERRIDE = recipe.get("image_size", None)

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

# Checkpointing
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

# ---------------------------------------------------------------------------
# Classification Model: CheXNet-style (DenseNet-121 + pretrained)
# ---------------------------------------------------------------------------

class HuggingFaceVisionEncoder(nn.Module):
    """Wrapper for HuggingFace vision models used as feature extractors."""

    def __init__(self, model_name, pretrained=True):
        super().__init__()
        from transformers import AutoModel, AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        else:
            self.model = AutoModel.from_config(config, trust_remote_code=True)

        # Determine feature dimension from config
        self.num_features = getattr(config, "hidden_size",
                            getattr(config, "projection_dim",
                            getattr(config, "embed_dim", 768)))

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        # Handle different model output formats
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, 0]  # CLS token
        else:
            return outputs[0][:, 0] if len(outputs[0].shape) == 3 else outputs[0]


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

    # Compile model
    model = torch.compile(model, dynamic=False)

    # Optimizer
    if OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )
    elif OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )
    elif OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=LEARNING_RATE,
            momentum=0.9, weight_decay=WEIGHT_DECAY,
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
    train_loader = make_infinite_dataloader(dataset, batch_size, split="train")
    images, targets, epoch = next(train_loader)

    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Batch size: {batch_size}")
    print(f"Optimizer: {OPTIMIZER}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    t_start_training = time.time()
    smooth_train_loss = 0
    total_training_time = 0
    step = 0

    model.train()

    while True:
        torch.cuda.synchronize()
        t0 = time.time()

        images_d = images.to(device, non_blocking=True)
        targets_d = targets.to(device, non_blocking=True)

        with autocast_ctx:
            logits = model(images_d)
            loss = criterion(logits, targets_d)

        scaler.scale(loss).backward()

        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        for param_group in optimizer.param_groups:
            param_group["lr"] = LEARNING_RATE * lrm

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        train_loss_f = loss.item()

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
        pct_done = 100 * progress
        remaining = max(0, TIME_BUDGET - total_training_time)

        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | "
            f"loss: {debiased_smooth_loss:.4f} | "
            f"lr: {LEARNING_RATE * lrm:.6f} | "
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

    model.eval()
    print("---")

    if is_multilabel:
        results = evaluate_multilabel(model, dataset, batch_size, device)
        primary_metric = results["mean_auc"]
        print(f"mean_auc:         {results['mean_auc']:.2f}%")
        print("per_class_auc:")
        for pathology, auc in results["per_class_auc"].items():
            print(f"  {pathology:<22s} {auc:.2f}%")

    elif task == "classification":
        results = evaluate_classification(model, dataset, batch_size, device)
        primary_metric = results.get("auc", results["accuracy"])
        print(f"val_accuracy:     {results['accuracy']:.2f}%")
        print(f"val_auc:          {results['auc']:.2f}%")
        print(f"val_kappa:        {results['kappa']:.4f}")
        print(f"val_loss:         {results['loss']:.6f}")

    elif task == "segmentation":
        results = evaluate_segmentation(model, dataset, batch_size, device)
        primary_metric = results["dice"]
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

    # Save checkpoint if requested
    if args.save_checkpoint:
        ckpt_config = {
            "dataset": dataset, "task": task, "backbone": backbone_name,
            "num_classes": num_classes, "batch_size": batch_size,
            "optimizer": OPTIMIZER, "learning_rate": LEARNING_RATE,
            "loss": loss_type, "recipe": args.recipe,
        }
        ckpt_name = f"{dataset}_{backbone_name}_{primary_metric:.1f}.pth"
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
        save_checkpoint(model, results, ckpt_config, ckpt_path)


if __name__ == "__main__":
    main()
