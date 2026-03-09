#!/usr/bin/env python3
"""Ensemble evaluation tool — combine predictions from multiple checkpoints.

Supports soft averaging and evaluates the ensemble against individual models.

Usage:
    python tools/ensemble.py checkpoints/model_a.pth checkpoints/model_b.pth
    python tools/ensemble.py checkpoints/*.pth --method soft_average
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def ensemble_predictions(checkpoint_paths, method="soft_average"):
    """Load predictions from multiple checkpoints and ensemble them."""
    all_preds = []
    all_targets = None
    configs = []

    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "val_predictions" not in ckpt:
            print(f"  Skipping {path} — no val_predictions")
            continue

        preds = np.array(ckpt["val_predictions"])
        targets = np.array(ckpt["val_targets"])
        all_preds.append(preds)

        if all_targets is None:
            all_targets = targets
        configs.append(ckpt.get("config", {}))

    if not all_preds:
        print("No valid checkpoints with predictions found.")
        return None, None, None

    if method == "soft_average":
        ensemble_pred = np.mean(all_preds, axis=0)
    elif method == "max":
        ensemble_pred = np.max(all_preds, axis=0)
    else:
        ensemble_pred = np.mean(all_preds, axis=0)

    return ensemble_pred, all_targets, configs


def evaluate_ensemble(preds, targets, task):
    """Evaluate ensemble predictions."""
    from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score

    if task == "multilabel":
        aucs = []
        for i in range(preds.shape[1]):
            try:
                auc = roc_auc_score(targets[:, i], preds[:, i])
                aucs.append(auc)
            except ValueError:
                pass
        mean_auc = 100.0 * np.mean(aucs) if aucs else 0.0
        return {"mean_auc": mean_auc, "num_valid_classes": len(aucs)}
    else:
        pred_classes = preds.argmax(axis=1)
        acc = 100.0 * accuracy_score(targets, pred_classes)
        kappa = cohen_kappa_score(targets, pred_classes, weights="quadratic")
        return {"accuracy": acc, "kappa": kappa}


def main():
    parser = argparse.ArgumentParser(description="Ensemble evaluation from checkpoints")
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint files")
    parser.add_argument("--method", default="soft_average",
                        choices=["soft_average", "max"],
                        help="Ensemble method")
    args = parser.parse_args()

    print(f"Ensemble method: {args.method}")
    print(f"Checkpoints: {len(args.checkpoints)}")
    print()

    # Individual model metrics
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        results = ckpt.get("results", {})
        config = ckpt.get("config", {})
        task = config.get("task", "unknown")

        if "mean_auc" in results:
            print(f"  {Path(path).stem}: mean_auc={results['mean_auc']:.2f}%")
        elif "accuracy" in results:
            print(f"  {Path(path).stem}: accuracy={results['accuracy']:.2f}%")

    # Ensemble evaluation
    ensemble_pred, targets, configs = ensemble_predictions(
        args.checkpoints, method=args.method)

    if ensemble_pred is None:
        return

    task = configs[0].get("task", "classification")
    metrics = evaluate_ensemble(ensemble_pred, targets, task)

    print(f"\nEnsemble results ({args.method}):")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
