"""
Evaluate a saved checkpoint on a chosen split (typically the held-out test set).
"""

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import train as train_mod
from prepare import get_dataset_config


def instantiate_model(task, cfg, checkpoint_config):
    if task in ("classification", "multilabel"):
        backbone = checkpoint_config.get("backbone", "densenet121")
        dropout = checkpoint_config.get("dropout", 0.2)
        model = train_mod.MedicalClassifier(
            num_classes=cfg["num_classes"],
            backbone=backbone,
            pretrained=False,
            dropout=dropout,
            multilabel=task == "multilabel",
        )
    elif task == "segmentation":
        model = train_mod.UNetMedical(
            num_classes=cfg["num_classes"],
            in_channels=cfg["in_channels"],
            base_channels=int(checkpoint_config.get("base_channels", train_mod.SEG_BASE_CHANNELS)),
            pretrained=False,
        )
    else:
        raise ValueError(f"Unknown task: {task}")
    return model


def split_exists(dataset_name, split_name):
    split_file = Path(train_mod.DATA_DIR) / dataset_name / "splits.csv"
    if not split_file.exists():
        return False
    import pandas as pd

    df = pd.read_csv(split_file)
    return split_name in set(df["split"].astype(str))


def metric_summary(results, cfg):
    metric_name, primary_metric = train_mod.primary_metric_for_results(results, cfg)
    lines = [f"{metric_name}: {primary_metric:.6f}"]
    if "per_class_auc" in results:
        lines.append("per_class_auc:")
        for name, value in results["per_class_auc"].items():
            lines.append(f"  {name}: {float(value):.2f}")
    if "per_class_f1" in results:
        lines.append("per_class_f1:")
        for name, value in results["per_class_f1"].items():
            lines.append(f"  {name}: {float(value):.2f}")
    return metric_name, primary_metric, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on a requested split")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate (default: test)")
    parser.add_argument("--output", type=str, default=None, help="Optional output .pt path for evaluation payload")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = payload.get("config", {})
    dataset_name = checkpoint_config.get("dataset")
    if not dataset_name:
        raise ValueError("Checkpoint config is missing dataset metadata")

    if not split_exists(dataset_name, args.split):
        raise ValueError(f"Split '{args.split}' not found for dataset '{dataset_name}'")

    cfg = get_dataset_config(dataset_name)
    task = checkpoint_config.get("task", cfg["task"])

    if "image_size" in checkpoint_config:
        train_mod.IMAGE_SIZE_OVERRIDE = checkpoint_config["image_size"]

    device = train_mod.select_device()
    model = instantiate_model(task, cfg, checkpoint_config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    batch_size = int(checkpoint_config.get("batch_size", train_mod.BATCH_SIZE if task != "segmentation" else train_mod.SEG_BATCH_SIZE))
    loader = train_mod.build_loader(dataset_name, cfg, batch_size, split=args.split)

    if task == "multilabel":
        stored_thresholds = None
        result_thresholds = payload.get("results", {}).get("f1_thresholds")
        if isinstance(result_thresholds, dict):
            stored_thresholds = [
                float(result_thresholds.get(class_name, 0.5))
                for class_name in cfg["class_names"]
            ]
        elif "artifacts" in payload.get("results", {}):
            artifacts = payload["results"]["artifacts"]
            stored_thresholds = artifacts.get("f1_thresholds")
        results = train_mod.evaluate_multilabel_loader(model, loader, cfg, device, thresholds=stored_thresholds)
    elif task == "classification":
        results = train_mod.evaluate_classification_loader(model, loader, cfg, device)
    else:
        results = train_mod.evaluate_segmentation_loader(model, loader, device)

    metric_name, primary_metric, summary = metric_summary(results, cfg)
    print(f"Dataset: {dataset_name}")
    print(f"Split: {args.split}")
    print(summary)

    output_path = Path(args.output) if args.output else checkpoint_path.with_suffix(f".{args.split}_eval.pt")
    eval_payload = {
        "dataset": dataset_name,
        "split": args.split,
        "metric_name": metric_name,
        "primary_metric": primary_metric,
        "results": results,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_config": checkpoint_config,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(eval_payload, output_path)
    print(f"Saved evaluation payload to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
