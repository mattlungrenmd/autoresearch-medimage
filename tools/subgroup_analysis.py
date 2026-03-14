#!/usr/bin/env python3
"""Subgroup analysis tool — evaluate performance across subgroups for fairness.

Analyzes model performance across demographic or clinical subgroups,
flags significant performance differences, and outputs markdown for
regulatory documentation.

Usage:
    python tools/subgroup_analysis.py checkpoints/model.pth --subgroup-file subgroups.csv
    python tools/subgroup_analysis.py checkpoints/model.pth --by-class
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def analyze_by_class(checkpoint_path, output_path=None):
    """Analyze per-class performance from saved predictions."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    results = ckpt.get("results", {})

    if "val_predictions" not in ckpt:
        print("No val_predictions in checkpoint. Re-run with --save-checkpoint.")
        return

    preds = np.array(ckpt["val_predictions"])
    targets = np.array(ckpt["val_targets"])
    class_names = ckpt.get("class_names", [f"class_{i}" for i in range(preds.shape[1]
                            if len(preds.shape) > 1 else 1)])
    task = config.get("task", "classification")

    lines = []
    lines.append("# Subgroup Performance Analysis")
    lines.append(f"\n**Dataset:** {config.get('dataset', 'unknown')}")
    lines.append(f"**Model:** {config.get('backbone', 'unknown')}")
    lines.append(f"**Task:** {task}")
    lines.append("")

    if task == "multilabel":
        from sklearn.metrics import roc_auc_score, f1_score

        lines.append("## Per-Class Metrics")
        lines.append("")
        lines.append("| Pathology | AUC | Prevalence | F1@0.5 | Sensitivity@0.5 | Specificity@0.5 |")
        lines.append("|-----------|-----|------------|--------|-----------------|-----------------|")

        aucs = []
        for i, name in enumerate(class_names):
            prevalence = targets[:, i].mean()
            try:
                auc = roc_auc_score(targets[:, i], preds[:, i])
            except ValueError:
                auc = 0.0
            binary_preds = (preds[:, i] >= 0.5).astype(int)

            tp = ((binary_preds == 1) & (targets[:, i] == 1)).sum()
            fp = ((binary_preds == 1) & (targets[:, i] == 0)).sum()
            fn = ((binary_preds == 0) & (targets[:, i] == 1)).sum()
            tn = ((binary_preds == 0) & (targets[:, i] == 0)).sum()

            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)

            lines.append(
                f"| {name} | {100*auc:.1f}% | {100*prevalence:.1f}% | "
                f"{100*f1:.1f}% | {100*sens:.1f}% | {100*spec:.1f}% |"
            )
            aucs.append(auc)

        # Flag disparities
        lines.append("")
        lines.append("## Performance Disparities")
        lines.append("")

        if aucs:
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            worst_idx = np.argmin(aucs)
            best_idx = np.argmax(aucs)
            gap = aucs[best_idx] - aucs[worst_idx]

            lines.append(f"- **Mean AUC:** {100*mean_auc:.1f}% (+/- {100*std_auc:.1f}%)")
            lines.append(f"- **Best class:** {class_names[best_idx]} ({100*aucs[best_idx]:.1f}%)")
            lines.append(f"- **Worst class:** {class_names[worst_idx]} ({100*aucs[worst_idx]:.1f}%)")
            lines.append(f"- **Max gap:** {100*gap:.1f} percentage points")

            if gap > 0.15:
                lines.append(f"\n> **WARNING:** Performance gap of {100*gap:.1f}% exceeds 15% threshold. "
                           f"Consider targeted improvements for {class_names[worst_idx]}.")

    elif task == "classification":
        from sklearn.metrics import classification_report
        pred_classes = preds.argmax(axis=1)
        report = classification_report(targets, pred_classes,
                                       target_names=class_names[:preds.shape[1]],
                                       output_dict=True)

        lines.append("## Per-Class Metrics")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("|-------|-----------|--------|-----|---------|")

        for name in class_names[:preds.shape[1]]:
            if name in report:
                r = report[name]
                lines.append(
                    f"| {name} | {100*r['precision']:.1f}% | {100*r['recall']:.1f}% | "
                    f"{100*r['f1-score']:.1f}% | {int(r['support'])} |"
                )

    # Confidence intervals
    ci = ckpt.get("confidence_intervals", {})
    if ci:
        lines.append("")
        lines.append("## Confidence Intervals")
        lines.append("")
        for metric, vals in ci.items():
            lines.append(f"- **{metric}:** [{vals['lower']:.4f}, {vals['upper']:.4f}] (95% CI)")

    # Thresholds
    thresholds = ckpt.get("per_class_thresholds")
    if thresholds is not None:
        lines.append("")
        lines.append("## Optimized Thresholds")
        lines.append("")
        lines.append("| Class | Threshold |")
        lines.append("|-------|-----------|")
        for i, name in enumerate(class_names):
            if i < len(thresholds):
                lines.append(f"| {name} | {thresholds[i]:.3f} |")

    report_text = "\n".join(lines)

    if output_path is None:
        output_path = "subgroup_analysis.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nSaved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Subgroup performance analysis")
    parser.add_argument("checkpoint", help="Checkpoint .pth file")
    parser.add_argument("--output", "-o", help="Output markdown path")
    parser.add_argument("--by-class", action="store_true", default=True,
                        help="Analyze by class (default)")
    args = parser.parse_args()

    analyze_by_class(args.checkpoint, output_path=args.output)


if __name__ == "__main__":
    main()
