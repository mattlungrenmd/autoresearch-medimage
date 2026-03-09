"""
Kaggle Auto-Detect — Point at any Kaggle competition or dataset, auto-configure everything.

Analyzes the dataset structure, detects the task (classification, multilabel, segmentation),
identifies image columns, label columns, and generates a dataset config for prepare.py.

Usage:
    python tools/kaggle_autodetect.py rsna-2024-lumbar-spine
    python tools/kaggle_autodetect.py paultimothymooney/chest-xray-pneumonia
    python tools/kaggle_autodetect.py --url https://www.kaggle.com/competitions/rsna-2025-lumbar
    python tools/kaggle_autodetect.py --scan            # Scan already-downloaded data/
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ---------------------------------------------------------------------------
# Kaggle download
# ---------------------------------------------------------------------------

def download_kaggle(slug):
    """Download a Kaggle dataset or competition."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    # Determine if it's a competition or dataset
    is_competition = "/" not in slug

    dest_dir = os.path.join(DATA_DIR, slug.replace("/", "_"))
    os.makedirs(dest_dir, exist_ok=True)

    marker = os.path.join(dest_dir, ".download_complete")
    if os.path.exists(marker):
        print(f"Already downloaded: {slug}")
        return dest_dir

    api = KaggleApi()
    api.authenticate()

    if is_competition:
        print(f"Downloading competition: {slug}")
        api.competition_download_files(slug, path=dest_dir, quiet=False)
    else:
        print(f"Downloading dataset: {slug}")
        api.dataset_download_files(slug, path=dest_dir, unzip=True, quiet=False)

    # Unzip any remaining zip files
    import zipfile
    for zf in Path(dest_dir).rglob("*.zip"):
        print(f"  Extracting {zf.name}...")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(zf.parent)
        zf.unlink()

    Path(marker).touch()
    return dest_dir


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm", ".nii", ".gif"}


def scan_directory(data_dir):
    """Scan a data directory and detect structure."""
    result = {
        "csv_files": [],
        "image_dirs": [],
        "image_count": 0,
        "mask_dirs": [],
        "class_dirs": [],
        "total_files": 0,
    }

    for root, dirs, files in os.walk(data_dir):
        rel_root = os.path.relpath(root, data_dir)

        # Count images
        images = [f for f in files if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
        if images:
            result["image_dirs"].append({"path": root, "count": len(images)})
            result["image_count"] += len(images)

        # Find CSVs
        csvs = [f for f in files if f.endswith(".csv")]
        for csv_name in csvs:
            csv_path = os.path.join(root, csv_name)
            try:
                df = pd.read_csv(csv_path, nrows=5)
                result["csv_files"].append({
                    "path": csv_path,
                    "name": csv_name,
                    "columns": list(df.columns),
                    "rows": len(pd.read_csv(csv_path)),
                })
            except Exception:
                pass

        # Detect class-structured directories (train/class_a, train/class_b)
        subdirs = [d for d in dirs if os.path.isdir(os.path.join(root, d))]
        has_images_in_subdirs = False
        for d in subdirs:
            subdir_path = os.path.join(root, d)
            sub_images = [f for f in os.listdir(subdir_path)
                         if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
            if sub_images:
                has_images_in_subdirs = True

        if has_images_in_subdirs and len(subdirs) >= 2:
            # Check if this looks like class directories
            if not any(d.lower() in ("train", "test", "val", "images", "masks") for d in subdirs):
                result["class_dirs"].append({
                    "path": root,
                    "classes": subdirs,
                    "num_classes": len(subdirs),
                })

        # Detect mask directories
        for d in subdirs:
            if any(kw in d.lower() for kw in ["mask", "label", "annotation", "segmentation", "gt"]):
                mask_path = os.path.join(root, d)
                mask_count = len([f for f in os.listdir(mask_path)
                                 if Path(f).suffix.lower() in IMAGE_EXTENSIONS])
                if mask_count > 0:
                    result["mask_dirs"].append({"path": mask_path, "count": mask_count})

        result["total_files"] += len(files)

    return result


def detect_task(scan_result):
    """Detect the task type from directory scan."""
    has_masks = len(scan_result["mask_dirs"]) > 0
    has_class_dirs = len(scan_result["class_dirs"]) > 0
    has_csvs = len(scan_result["csv_files"]) > 0

    # Segmentation: has mask directories
    if has_masks:
        return "segmentation"

    # Classification with folder structure
    if has_class_dirs:
        num_classes = scan_result["class_dirs"][0]["num_classes"]
        return "classification"

    # Check CSVs for multi-label indicators
    if has_csvs:
        for csv_info in scan_result["csv_files"]:
            cols = [c.lower() for c in csv_info["columns"]]
            # Multi-label: multiple label columns or pipe-separated labels
            if any("|" in str(c) for c in cols):
                return "multilabel"
            label_cols = [c for c in cols if any(kw in c for kw in
                         ["label", "target", "class", "diagnosis", "finding", "category"])]
            if label_cols:
                csv_path = csv_info["path"]
                try:
                    df = pd.read_csv(csv_path, nrows=100)
                    for col in label_cols:
                        orig_col = [c for c in df.columns if c.lower() == col][0]
                        vals = df[orig_col].astype(str)
                        if vals.str.contains(r"\|").any():
                            return "multilabel"
                except Exception:
                    pass
                return "classification"

    return "classification"  # default


def detect_label_info(scan_result, data_dir):
    """Detect label columns and class names from CSVs or folder structure."""
    info = {
        "num_classes": 2,
        "class_names": [],
        "label_column": None,
        "image_column": None,
    }

    # From folder structure
    if scan_result["class_dirs"]:
        class_info = scan_result["class_dirs"][0]
        info["num_classes"] = class_info["num_classes"]
        info["class_names"] = sorted(class_info["classes"])
        return info

    # From CSV
    if scan_result["csv_files"]:
        csv_info = scan_result["csv_files"][0]  # Use first/largest CSV
        csv_path = csv_info["path"]
        try:
            df = pd.read_csv(csv_path, nrows=1000)
        except Exception:
            return info

        cols = list(df.columns)
        cols_lower = [c.lower() for c in cols]

        # Detect image column
        for pattern in ["image", "file", "path", "id", "filename"]:
            matches = [c for c, cl in zip(cols, cols_lower) if pattern in cl]
            if matches:
                info["image_column"] = matches[0]
                break

        # Detect label column
        for pattern in ["label", "target", "class", "diagnosis", "finding", "category", "grade"]:
            matches = [c for c, cl in zip(cols, cols_lower) if pattern in cl]
            if matches:
                info["label_column"] = matches[0]
                break

        if info["label_column"]:
            labels = df[info["label_column"]].dropna()
            unique_labels = labels.unique()
            info["num_classes"] = len(unique_labels)
            info["class_names"] = [str(l) for l in sorted(unique_labels)][:20]

    return info


def generate_config(slug, scan_result, task, label_info, data_dir):
    """Generate a dataset config dict for prepare.py."""
    # Detect channels
    sample_image = None
    for img_dir in scan_result["image_dirs"]:
        for f in os.listdir(img_dir["path"])[:1]:
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                sample_image = os.path.join(img_dir["path"], f)
                break
        if sample_image:
            break

    in_channels = 3
    if sample_image:
        try:
            from PIL import Image
            img = Image.open(sample_image)
            if img.mode in ("L", "1"):
                in_channels = 1
        except Exception:
            pass

    config_name = slug.replace("/", "_").replace("-", "_").lower()

    config = {
        "name": config_name,
        "kaggle_slug": slug,
        "task": task,
        "num_classes": label_info["num_classes"],
        "class_names": label_info["class_names"],
        "image_size": 256 if task == "segmentation" else 224,
        "in_channels": in_channels,
        "mean": (0.485,) if in_channels == 1 else (0.485, 0.456, 0.406),
        "std": (0.229,) if in_channels == 1 else (0.229, 0.224, 0.225),
        "metric": "dice" if task == "segmentation" else "auc" if label_info["num_classes"] == 2 else "accuracy",
        "description": f"Auto-detected from Kaggle: {slug}",
        "data_dir": data_dir,
    }

    return config


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_detection_report(slug, scan_result, task, label_info, config):
    """Print a human-readable detection report."""
    print(f"\n{'='*60}")
    print(f"Auto-Detection Report: {slug}")
    print(f"{'='*60}\n")

    print(f"Images found:     {scan_result['image_count']:,}")
    print(f"Image dirs:       {len(scan_result['image_dirs'])}")
    print(f"CSV files:        {len(scan_result['csv_files'])}")
    print(f"Mask dirs:        {len(scan_result['mask_dirs'])}")
    print(f"Class dirs:       {len(scan_result['class_dirs'])}")
    print()

    print(f"Detected task:    {task}")
    print(f"Num classes:      {label_info['num_classes']}")
    if label_info["class_names"]:
        print(f"Classes:          {', '.join(label_info['class_names'][:10])}")
    if label_info["label_column"]:
        print(f"Label column:     {label_info['label_column']}")
    if label_info["image_column"]:
        print(f"Image column:     {label_info['image_column']}")
    print(f"Channels:         {config['in_channels']}")
    print(f"Suggested metric: {config['metric']}")

    print(f"\n--- Suggested dataset.yaml ---\n")
    print(f"name: {config['name']}")
    print(f"task: {config['task']}")
    print(f"metric: {config['metric']}")
    print(f"num_classes: {config['num_classes']}")
    print(f"class_names: {json.dumps(config['class_names'][:20])}")
    print(f"in_channels: {config['in_channels']}")
    print(f"image_size: {config['image_size']}")
    print(f"mean: {json.dumps(list(config['mean']))}")
    print(f"std: {json.dumps(list(config['std']))}")
    print(f'description: "{config["description"]}"')

    print(f"\nTo use this dataset:")
    print(f"  1. Create data\\{config['name']}\\dataset.yaml from the block above")
    print(f"  2. Create data\\{config['name']}\\splits.csv that matches the repo contract")
    print(f"  3. Validate it with: uv run prepare.py --dataset {config['name']}")
    print(f"  4. Train it with: uv run train.py --dataset {config['name']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Auto-detect Kaggle dataset for medical imaging")
    parser.add_argument("slug", nargs="?", default=None,
                        help="Kaggle dataset or competition slug")
    parser.add_argument("--url", type=str, default=None,
                        help="Full Kaggle URL")
    parser.add_argument("--scan", type=str, default=None,
                        help="Scan an existing local directory")
    parser.add_argument("--download", action="store_true",
                        help="Download the dataset first")
    args = parser.parse_args()

    # Parse Kaggle URL
    if args.url:
        match = re.search(r"kaggle\.com/(?:competitions|datasets)/([^/\s?]+(?:/[^/\s?]+)?)", args.url)
        if match:
            slug = match.group(1)
        else:
            print(f"Could not parse Kaggle URL: {args.url}")
            sys.exit(1)
    elif args.slug:
        slug = args.slug
    elif args.scan:
        slug = Path(args.scan).name
    else:
        print("Usage: python tools/kaggle_autodetect.py <kaggle-slug>")
        print("       python tools/kaggle_autodetect.py --url <kaggle-url>")
        print("       python tools/kaggle_autodetect.py --scan <local-dir>")
        sys.exit(1)

    # Determine data directory
    if args.scan:
        data_dir = args.scan
    else:
        data_dir = os.path.join(DATA_DIR, slug.replace("/", "_"))

    # Download if needed
    if args.download and not os.path.exists(os.path.join(data_dir, ".download_complete")):
        data_dir = download_kaggle(slug)

    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print(f"Run with --download to download first, or use --scan for existing data")
        sys.exit(1)

    print(f"Scanning {data_dir}...")
    scan_result = scan_directory(data_dir)

    task = detect_task(scan_result)
    label_info = detect_label_info(scan_result, data_dir)
    config = generate_config(slug, scan_result, task, label_info, data_dir)

    print_detection_report(slug, scan_result, task, label_info, config)


if __name__ == "__main__":
    main()
