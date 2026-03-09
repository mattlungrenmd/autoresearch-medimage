#!/usr/bin/env python3
"""Standalone ONNX export tool with validation and benchmarking.

Exports a checkpoint to ONNX, validates output match, and benchmarks inference.

Usage:
    python tools/onnx_export.py checkpoints/chestxray14_densenet121_84.1.pth
    python tools/onnx_export.py checkpoints/model.pth --output model.onnx --validate
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def export_and_validate(checkpoint_path, output_path=None, validate=True, benchmark=True):
    """Export checkpoint to ONNX with optional validation and benchmarking."""
    from train import MedicalClassifier, UNetMedical, PretrainedUNet

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    task = config.get("task", "classification")
    num_classes = config.get("num_classes", 14)
    backbone = config.get("backbone", "densenet121")
    dropout = config.get("dropout", 0.0)
    image_size = config.get("image_size", 224)
    seg_backbone = config.get("seg_backbone", "")
    seg_attention = config.get("seg_attention", False)

    # Build model
    if task == "segmentation":
        if seg_backbone:
            model = PretrainedUNet(num_classes=num_classes, backbone=seg_backbone,
                                   pretrained=False, use_attention=seg_attention)
        else:
            model = UNetMedical(num_classes=num_classes, pretrained=False)
    else:
        model = MedicalClassifier(
            num_classes=num_classes, backbone=backbone,
            pretrained=False, dropout=dropout,
            multilabel=(task == "multilabel"),
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if output_path is None:
        output_path = str(Path(checkpoint_path).with_suffix(".onnx"))

    dummy = torch.randn(1, 3, image_size, image_size)

    # Export
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Size: {size_mb:.1f} MB")

    # Validate
    if validate:
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(output_path)
            onnx_out = session.run(None, {"input": dummy.numpy()})[0]
            torch_out = model(dummy).detach().numpy()
            max_diff = np.abs(onnx_out - torch_out).max()
            print(f"  Validation: max diff = {max_diff:.6f} {'PASS' if max_diff < 1e-3 else 'FAIL'}")
        except ImportError:
            print("  Validation: onnxruntime not installed, skipping")

    # Benchmark
    if benchmark:
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(output_path)
            # Warmup
            for _ in range(5):
                session.run(None, {"input": dummy.numpy()})
            # Benchmark
            times = []
            for _ in range(50):
                t0 = time.perf_counter()
                session.run(None, {"input": dummy.numpy()})
                times.append(time.perf_counter() - t0)
            mean_ms = 1000 * np.mean(times)
            std_ms = 1000 * np.std(times)
            print(f"  ONNX inference: {mean_ms:.1f} +/- {std_ms:.1f} ms")
        except ImportError:
            print("  Benchmark: onnxruntime not installed, skipping")

    # Deployment manifest
    manifest = {
        "model": Path(output_path).name,
        "task": task,
        "num_classes": num_classes,
        "backbone": backbone,
        "image_size": image_size,
        "input_shape": [1, 3, image_size, image_size],
        "opset_version": 17,
        "size_mb": round(size_mb, 1),
    }
    import json
    manifest_path = str(Path(output_path).with_suffix(".json"))
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="ONNX export with validation")
    parser.add_argument("checkpoint", help="Checkpoint .pth file")
    parser.add_argument("--output", "-o", help="Output ONNX path")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip benchmark")
    args = parser.parse_args()

    export_and_validate(
        args.checkpoint,
        output_path=args.output,
        validate=not args.no_validate,
        benchmark=not args.no_benchmark,
    )


if __name__ == "__main__":
    main()
