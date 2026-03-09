#!/usr/bin/env python3
"""Grad-CAM viewer — generate heatmap overlays for all classes from a checkpoint.

Creates an HTML report with Grad-CAM visualizations for interpretability
and regulatory documentation.

Usage:
    python tools/gradcam_viewer.py checkpoints/model.pth --dataset chestxray14
    python tools/gradcam_viewer.py checkpoints/model.pth --num-samples 20
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_gradcam_report(checkpoint_path, dataset_name=None, num_samples=10,
                            output_dir=None):
    """Generate comprehensive Grad-CAM report from a checkpoint."""
    from train import (MedicalClassifier, make_dataloader, DATASETS)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    dataset = dataset_name or config.get("dataset", "chestxray14")
    task = config.get("task", "classification")
    num_classes = config.get("num_classes", 14)
    backbone = config.get("backbone", "densenet121")
    image_size = config.get("image_size", 224)

    if output_dir is None:
        output_dir = os.path.join("checkpoints", "gradcam_report")
    os.makedirs(output_dir, exist_ok=True)

    # Build model
    model = MedicalClassifier(
        num_classes=num_classes, backbone=backbone,
        pretrained=False, dropout=0.0,
        multilabel=(task == "multilabel"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Find target conv layer
    target_layer = None
    for name, module in model.encoder.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        print("No Conv2d layer found for Grad-CAM")
        return

    # Hooks
    activations = {}
    gradients = {}

    def fwd_hook(m, inp, out):
        activations["value"] = out.detach()

    def bwd_hook(m, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    # Get validation data
    cfg = DATASETS[dataset]
    loader, _ = make_dataloader(dataset, 1, split="val", image_size=image_size)
    class_names = cfg.get("class_names", [f"class_{i}" for i in range(num_classes)])

    image_paths = []
    count = 0

    for images, targets in loader:
        if count >= num_samples:
            break

        img = images.to(device).requires_grad_(True)
        logits = model(img.float())

        # Generate CAM for each class
        if task == "multilabel":
            top_classes = torch.where(torch.sigmoid(logits[0]) > 0.5)[0].tolist()
            if not top_classes:
                top_classes = [logits[0].argmax().item()]
        else:
            top_classes = [logits[0].argmax().item()]

        fig, axes = plt.subplots(1, 1 + len(top_classes), figsize=(4 * (1 + len(top_classes)), 4))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        # Original image
        img_np = images[0].permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        axes[0].imshow(img_np)
        axes[0].set_title("Input")
        axes[0].axis("off")

        for j, cls_idx in enumerate(top_classes):
            model.zero_grad()
            logits[0, cls_idx].backward(retain_graph=True)

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

            axes[j + 1].imshow(img_np)
            axes[j + 1].imshow(cam, alpha=0.5, cmap="jet")
            name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
            conf = torch.sigmoid(logits[0, cls_idx]).item() if task == "multilabel" else \
                   torch.softmax(logits[0], dim=0)[cls_idx].item()
            axes[j + 1].set_title(f"{name}\n({conf:.2f})")
            axes[j + 1].axis("off")

        plt.tight_layout()
        fname = f"sample_{count:03d}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=100)
        plt.close()
        image_paths.append(fname)
        count += 1

    h1.remove()
    h2.remove()

    # Generate HTML report
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Grad-CAM Report — {dataset}</title>
<style>
body {{ font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #2c3e50; }}
img {{ max-width: 100%; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }}
.info {{ background: #ecf0f1; padding: 15px; border-radius: 8px; margin: 15px 0; }}
</style></head><body>
<h1>Grad-CAM Interpretability Report</h1>
<div class="info">
<p><strong>Dataset:</strong> {dataset} | <strong>Model:</strong> {backbone} |
<strong>Classes:</strong> {num_classes} | <strong>Samples:</strong> {count}</p>
</div>
"""
    for path in image_paths:
        html += f'<img src="{path}">\n'
    html += "</body></html>"

    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w") as f:
        f.write(html)

    print(f"Grad-CAM report: {report_path} ({count} samples)")


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization report")
    parser.add_argument("checkpoint", help="Checkpoint .pth file")
    parser.add_argument("--dataset", help="Dataset name (default: from checkpoint)")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output-dir", help="Output directory")
    args = parser.parse_args()

    generate_gradcam_report(
        args.checkpoint,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
