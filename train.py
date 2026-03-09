"""
Autoresearch-MedVision training script. Single-GPU, single-file.
Default configuration reproduces CheXNet (Rajpurkar et al., 2017):
  DenseNet-121 pretrained on ImageNet, fine-tuned on ChestX-ray14
  for 14-pathology multi-label classification.

Usage:
    uv run train.py                                    # CheXNet on ChestX-ray14 (default)
    uv run train.py --dataset aptos                    # DR grading
    uv run train.py --dataset chest_xray               # Pneumonia binary
    uv run train.py --dataset brain_mri                # Brain tumor segmentation
    uv run train.py --dataset oct                      # Retinal OCT classification
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
import argparse

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

# Segmentation-specific
SEG_BATCH_SIZE = 8            # batch size for segmentation (larger images)
SEG_BASE_CHANNELS = 64        # U-Net base channels

# ---------------------------------------------------------------------------
# Classification Model: CheXNet-style (DenseNet-121 + pretrained)
# ---------------------------------------------------------------------------

class MedicalClassifier(nn.Module):
    """Pretrained backbone + classification head.
    Default: DenseNet-121 (CheXNet architecture).
    Supports: densenet121, densenet169, densenet201, resnet50, efficientnet_b0-b4.
    """

    def __init__(self, num_classes, backbone="densenet121", pretrained=True,
                 dropout=0.2, multilabel=False):
        super().__init__()
        self.multilabel = multilabel

        if backbone.startswith("densenet"):
            # DenseNet family (CheXNet default)
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
        elif backbone.startswith("efficientnet"):
            # Use timm for EfficientNet variants
            self.encoder = timm.create_model(
                backbone, pretrained=pretrained, num_classes=0,
            )
            num_features = self.encoder.num_features
        else:
            # Fallback to timm for any backbone
            self.encoder = timm.create_model(
                backbone, pretrained=pretrained, num_classes=0,
            )
            num_features = self.encoder.num_features

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(num_features, num_classes)

        # Initialize head
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        features = self.encoder(x)
        features = self.dropout(features)
        logits = self.head(features)
        return logits


# ---------------------------------------------------------------------------
# Segmentation Model: U-Net with pretrained encoder
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
    """U-Net with optional pretrained ResNet encoder for medical segmentation."""

    def __init__(self, num_classes, in_channels=3, base_channels=64, pretrained=True):
        super().__init__()
        ch = base_channels

        # Encoder (from scratch for medical segmentation)
        self.enc1 = ConvBlock(in_channels, ch)
        self.enc2 = ConvBlock(ch, ch * 2)
        self.enc3 = ConvBlock(ch * 2, ch * 4)
        self.enc4 = ConvBlock(ch * 4, ch * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(ch * 8, ch * 16)

        # Decoder
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

class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for segmentation."""

    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce = F.cross_entropy(logits, targets, ignore_index=-1)

        # Dice on foreground class
        probs = F.softmax(logits, dim=1)[:, 1]  # foreground probability
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
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

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
        in_ch = 3  # grayscale images are expanded to 3ch in transforms
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
    print(f"Model: {BACKBONE if task != 'segmentation' else 'U-Net'}")
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
    if is_multilabel:
        # CheXNet: BCEWithLogitsLoss for multi-label
        if USE_CLASS_WEIGHTS:
            # For multi-label, compute pos_weight from class frequencies
            try:
                w = get_class_weights(dataset, device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=w)
            except Exception:
                criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
    elif task == "segmentation":
        criterion = DiceBCELoss(dice_weight=0.5)
    else:
        # Standard classification
        if USE_CLASS_WEIGHTS:
            try:
                w = get_class_weights(dataset, device)
                criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTHING)
            except Exception:
                criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

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

        # Forward
        with autocast_ctx:
            logits = model(images_d)
            loss = criterion(logits, targets_d)

        # Backward
        scaler.scale(loss).backward()

        # LR schedule
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        for param_group in optimizer.param_groups:
            param_group["lr"] = LEARNING_RATE * lrm

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        train_loss_f = loss.item()

        # Fast fail
        if train_loss_f > 100 or train_loss_f != train_loss_f:  # NaN check
            print("FAIL: loss exploded or NaN")
            exit(1)

        # Prefetch next batch
        images, targets, epoch = next(train_loader)

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        if step > 5:
            total_training_time += dt

        # Logging
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

        # GC management
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        if step > 5 and total_training_time >= TIME_BUDGET:
            break

    print()  # newline after \r training log

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
    print(f"backbone:         {BACKBONE if task != 'segmentation' else 'unet'}")


if __name__ == "__main__":
    main()
