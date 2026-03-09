"""
Autoresearch-Vision training script. Single-GPU, single-file.
Adapted from karpathy/autoresearch for image classification and segmentation.

Usage:
    uv run train.py                          # CIFAR-10 classification (default)
    uv run train.py --task segmentation      # VOC segmentation
    uv run train.py --dataset cifar100       # CIFAR-100 classification
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    TIME_BUDGET, DATASETS,
    make_infinite_dataloader, make_dataloader,
    evaluate_classification, evaluate_segmentation,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Task selection
DATASET = "cifar10"           # "cifar10", "cifar100", or "voc"
TASK = "classification"       # "classification" or "segmentation"

# Model architecture
BASE_CHANNELS = 64            # base channel width
DEPTH = 4                     # number of residual blocks per stage
NUM_STAGES = 3                # number of downsampling stages

# Optimization
BATCH_SIZE = 128              # training batch size
LEARNING_RATE = 0.1           # initial learning rate
MOMENTUM = 0.9                # SGD momentum
WEIGHT_DECAY = 5e-4           # L2 regularization
WARMUP_RATIO = 0.05           # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.3          # fraction of time budget for cosine decay

# Segmentation-specific
SEG_BATCH_SIZE = 16           # batch size for segmentation (larger images)

# ---------------------------------------------------------------------------
# Classification Model: ResNet-style
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ResBlock(nn.Module):
    def __init__(self, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out, inplace=True)


class ImageClassifier(nn.Module):
    """Simple ResNet-style classifier."""

    def __init__(self, num_classes, in_channels=3, base_channels=64,
                 depth=4, num_stages=3):
        super().__init__()
        self.stem = ConvBNReLU(in_channels, base_channels)

        stages = []
        ch = base_channels
        for stage_idx in range(num_stages):
            if stage_idx > 0:
                # Downsample: double channels, halve spatial
                new_ch = ch * 2
                stages.append(ConvBNReLU(ch, new_ch, stride=2))
                ch = new_ch
            for _ in range(depth):
                stages.append(ResBlock(ch))
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(ch, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Segmentation Model: U-Net style
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
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


class UNetSegmenter(nn.Module):
    """Compact U-Net for segmentation."""

    def __init__(self, num_classes, in_channels=3, base_channels=64, depth=4):
        super().__init__()
        ch = base_channels

        # Encoder
        self.enc1 = DoubleConv(in_channels, ch)
        self.enc2 = DoubleConv(ch, ch * 2)
        self.enc3 = DoubleConv(ch * 2, ch * 4)
        self.enc4 = DoubleConv(ch * 4, ch * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(ch * 8, ch * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(ch * 16, ch * 8, 2, stride=2)
        self.dec4 = DoubleConv(ch * 16, ch * 8)
        self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2)
        self.dec3 = DoubleConv(ch * 8, ch * 4)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
        self.dec2 = DoubleConv(ch * 4, ch * 2)
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.dec1 = DoubleConv(ch * 2, ch)

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
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)


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
        # Cosine decay to 0
        return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * (1 - cooldown_progress))).item())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    args = parser.parse_args()

    dataset = args.dataset or DATASET
    task = args.task or TASK

    # Auto-detect task from dataset
    if dataset == "voc":
        task = "segmentation"

    cfg = DATASETS[dataset]
    num_classes = cfg["num_classes"]

    print(f"Task: {task}")
    print(f"Dataset: {dataset}")
    print(f"Num classes: {num_classes}")

    # Setup
    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    scaler = torch.amp.GradScaler()

    # Build model
    if task == "classification":
        model = ImageClassifier(
            num_classes=num_classes,
            base_channels=BASE_CHANNELS,
            depth=DEPTH,
            num_stages=NUM_STAGES,
        )
        batch_size = BATCH_SIZE
    elif task == "segmentation":
        model = UNetSegmenter(
            num_classes=num_classes,
            base_channels=BASE_CHANNELS,
        )
        batch_size = SEG_BATCH_SIZE
    else:
        raise ValueError(f"Unknown task: {task}")

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Compile model
    model = torch.compile(model, dynamic=False)

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    # Dataloader
    train_loader = make_infinite_dataloader(dataset, batch_size, split="train")
    images, targets, epoch = next(train_loader)

    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Batch size: {batch_size}")
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

        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        # Forward
        with autocast_ctx:
            logits = model(images)
            if task == "classification":
                loss = F.cross_entropy(logits, targets)
            else:
                # Segmentation: ignore index -1
                loss = F.cross_entropy(logits, targets, ignore_index=-1)

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
        if train_loss_f > 100:
            print("FAIL: loss exploded")
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
            f"lr: {LEARNING_RATE * lrm:.5f} | "
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

    if task == "classification":
        accuracy, val_loss = evaluate_classification(model, dataset, batch_size, device)
        metric_name = "val_accuracy"
        metric_value = accuracy
        print("---")
        print(f"val_accuracy:     {accuracy:.2f}%")
        print(f"val_loss:         {val_loss:.6f}")
    else:
        miou, pixel_acc = evaluate_segmentation(model, batch_size, device)
        metric_name = "val_miou"
        metric_value = miou
        print("---")
        print(f"val_miou:         {miou:.2f}%")
        print(f"val_pixel_acc:    {pixel_acc:.2f}%")

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


if __name__ == "__main__":
    main()
