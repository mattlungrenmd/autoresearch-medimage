#!/usr/bin/env python3
"""Calibration analysis tool — reliability diagrams + Expected Calibration Error (ECE).

Loads a checkpoint with saved val_predictions/val_targets and generates:
1. Reliability diagram (calibration curve)
2. ECE (Expected Calibration Error) metric
3. Temperature scaling post-hoc calibration

Usage:
    python tools/calibration_plot.py checkpoints/chestxray14_densenet121_84.1.pth
    python tools/calibration_plot.py checkpoints/*.pth --output calibration_report.png
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_ece(probs, targets, n_bins=15):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs)

    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_confidence = probs[mask].mean()
        bin_accuracy = targets[mask].mean()
        bin_weight = mask.sum() / total
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece


def temperature_scale(logits, targets, lr=0.01, max_iter=100):
    """Learn optimal temperature for calibration via NLL minimization."""
    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    logits_t = torch.tensor(logits, dtype=torch.float32)
    targets_t = torch.tensor(targets, dtype=torch.long)

    def closure():
        optimizer.zero_grad()
        scaled = logits_t / temperature
        loss = torch.nn.functional.cross_entropy(scaled, targets_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.item()


def plot_reliability_diagram(probs, targets, title="Reliability Diagram", ax=None):
    """Plot reliability diagram (calibration curve)."""
    n_bins = 15
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() == 0:
            bin_accs.append(0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_counts.append(0)
            continue
        bin_accs.append(targets[mask].mean())
        bin_confs.append(probs[mask].mean())
        bin_counts.append(mask.sum())

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    # Bar chart
    bin_width = 1.0 / n_bins
    bin_centers = [(bin_boundaries[i] + bin_boundaries[i + 1]) / 2 for i in range(n_bins)]
    ax.bar(bin_centers, bin_accs, width=bin_width * 0.8, alpha=0.7,
           color="steelblue", edgecolor="black", label="Model")

    ece = compute_ece(probs, targets, n_bins)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return ece


def main():
    parser = argparse.ArgumentParser(description="Calibration analysis from checkpoints")
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint file(s)")
    parser.add_argument("--output", "-o", default="calibration_report.png",
                        help="Output image path")
    args = parser.parse_args()

    n = len(args.checkpoints)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)

    for idx, ckpt_path in enumerate(args.checkpoints):
        print(f"Loading {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "val_predictions" not in ckpt or "val_targets" not in ckpt:
            print(f"  Skipping — no val_predictions/val_targets in checkpoint")
            continue

        preds = np.array(ckpt["val_predictions"])
        targets = np.array(ckpt["val_targets"])
        config = ckpt.get("config", {})
        task = config.get("task", "unknown")

        if task == "multilabel" or (len(preds.shape) == 2 and len(targets.shape) == 2):
            # Multilabel: flatten all classes
            probs = preds.flatten()
            tgts = targets.flatten()
        elif len(preds.shape) == 2:
            # Classification: use max prob
            probs = preds.max(axis=1)
            tgts = (preds.argmax(axis=1) == targets).astype(float)
        else:
            probs = preds
            tgts = targets

        name = Path(ckpt_path).stem
        ece = plot_reliability_diagram(probs, tgts, title=name, ax=axes[0, idx])
        print(f"  ECE: {ece:.4f}")

        # Confidence intervals if available
        ci = ckpt.get("confidence_intervals", {})
        if ci:
            print(f"  CIs: {ci}")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
