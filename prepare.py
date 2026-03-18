"""
Data preparation and evaluation for autoresearch-medimage experiments.

This module has two public roles:
1. download and prepare built-in benchmark datasets from Kaggle
2. validate bring-your-own datasets stored under ./data/<dataset_name>/

During autoresearch runs this file remains fixed. Agents should mutate train.py,
not prepare.py, when exploring experiments.

Usage:
    python prepare.py --dataset aptos            # Download built-in APTOS
    python prepare.py --dataset chestxray14      # Download built-in ChestX-ray14
    uv run prepare.py --dataset toy_binary       # Validate a local custom dataset
    python prepare.py --dataset all              # Download all built-in datasets
    python prepare.py --list                     # List built-in + custom datasets
"""

import os
import sys
import json
import time
import math
import re
import shutil
import zipfile
import argparse
import hashlib
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import cohen_kappa_score, roc_auc_score

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300           # training time budget in seconds (5 minutes)
NUM_WORKERS = 4             # dataloader workers
VAL_RATIO = 0.15            # validation split ratio
RANDOM_SEED = 42            # reproducible splits

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    # === OPHTHALMOLOGY ===
    "aptos": {
        "kaggle_slug": "mariaherrerot/aptos2019",
        "task": "classification",
        "num_classes": 5,
        "class_names": ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
        "image_size": 224,
        "in_channels": 3,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "metric": "quadratic_kappa",
        "description": "APTOS 2019 Blindness Detection - diabetic retinopathy severity grading (fundus photos)",
    },
    "oct": {
        "kaggle_slug": "paultimothymooney/kermany2018",
        "task": "classification",
        "num_classes": 4,
        "class_names": ["CNV", "DME", "DRUSEN", "NORMAL"],
        "image_size": 224,
        "in_channels": 1,  # grayscale OCT
        "mean": (0.485,),
        "std": (0.229,),
        "metric": "accuracy",
        "description": "Retinal OCT images - 4-class classification (CNV, DME, Drusen, Normal)",
    },
    "retina_vessel": {
        "kaggle_slug": "abdallahwagih/retina-blood-vessel",
        "task": "segmentation",
        "num_classes": 2,
        "class_names": ["background", "vessel"],
        "image_size": 256,
        "in_channels": 3,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "metric": "dice",
        "description": "Retinal blood vessel segmentation from fundus images",
    },

    # === RADIOLOGY ===
    "chestxray14": {
        "kaggle_slug": "ksw2000/chestxray14",
        "task": "multilabel",
        "num_classes": 14,
        "class_names": [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax",
            "Consolidation", "Edema", "Emphysema", "Fibrosis",
            "Pleural_Thickening", "Hernia",
        ],
        "image_size": 224,
        "in_channels": 1,
        "mean": (0.485,),
        "std": (0.229,),
        "metric": "mean_auc",
        "description": "NIH ChestX-ray14 - 14 thoracic pathology multi-label classification (CheXNet baseline)",
    },
    "chest_xray": {
        "kaggle_slug": "paultimothymooney/chest-xray-pneumonia",
        "task": "classification",
        "num_classes": 2,
        "class_names": ["NORMAL", "PNEUMONIA"],
        "image_size": 224,
        "in_channels": 1,  # grayscale X-ray
        "mean": (0.485,),
        "std": (0.229,),
        "metric": "auc",
        "description": "Chest X-ray pneumonia detection (binary classification)",
    },
    "brain_mri": {
        "kaggle_slug": "mateuszbuda/lgg-mri-segmentation",
        "task": "segmentation",
        "num_classes": 2,
        "class_names": ["background", "tumor"],
        "image_size": 256,
        "in_channels": 3,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "metric": "dice",
        "description": "Lower-grade glioma MRI segmentation (brain tumor boundaries)",
    },
    "pneumothorax": {
        "kaggle_slug": "vbookshelf/pneumothorax-chest-xray-images-and-masks",
        "task": "segmentation",
        "num_classes": 2,
        "class_names": ["background", "pneumothorax"],
        "image_size": 256,
        "in_channels": 1,
        "mean": (0.485,),
        "std": (0.229,),
        "metric": "dice",
        "description": "SIIM-ACR pneumothorax segmentation from chest X-rays",
    },
}

SUPPORTED_TASKS = {"classification", "multilabel", "segmentation"}
SUPPORTED_METRICS = {
    "classification": {"accuracy", "auc", "quadratic_kappa", "kappa"},
    "multilabel": {"auc", "mean_auc"},
    "segmentation": {"dice", "iou", "pixel_acc"},
}


def _default_channel_stats(in_channels):
    if in_channels == 1:
        return (0.485,), (0.229,)
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _load_yaml_file(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read dataset.yaml files. Install it with your environment manager."
        ) from exc

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a top-level mapping")
    return data


def _normalize_dataset_config(dataset_name, cfg, source="builtin"):
    normalized = dict(cfg)
    normalized["id"] = dataset_name
    normalized["source"] = source
    normalized["name"] = str(normalized.get("name", dataset_name))
    normalized["description"] = str(normalized.get("description", "Custom medical imaging dataset"))

    required = ["task", "metric", "num_classes", "class_names", "in_channels", "image_size"]
    missing = [field for field in required if field not in normalized]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")

    task = str(normalized["task"]).strip().lower()
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"task must be one of {sorted(SUPPORTED_TASKS)}")
    normalized["task"] = task

    metric = str(normalized["metric"]).strip().lower()
    if metric not in SUPPORTED_METRICS[task]:
        raise ValueError(f"metric '{metric}' is not supported for task '{task}'")
    normalized["metric"] = metric

    try:
        num_classes = int(normalized["num_classes"])
    except Exception as exc:
        raise ValueError("num_classes must be an integer") from exc
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    normalized["num_classes"] = num_classes
    if task == "segmentation" and num_classes != 2:
        raise ValueError("v1 segmentation support is binary only (num_classes must be 2)")

    class_names = normalized["class_names"]
    if not isinstance(class_names, (list, tuple)) or len(class_names) != num_classes:
        raise ValueError("class_names must be a list with length equal to num_classes")
    normalized["class_names"] = [str(name) for name in class_names]

    try:
        in_channels = int(normalized["in_channels"])
    except Exception as exc:
        raise ValueError("in_channels must be an integer") from exc
    if in_channels not in (1, 3):
        raise ValueError("in_channels must be 1 or 3")
    normalized["in_channels"] = in_channels

    try:
        image_size = int(normalized["image_size"])
    except Exception as exc:
        raise ValueError("image_size must be an integer") from exc
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    normalized["image_size"] = image_size

    default_mean, default_std = _default_channel_stats(in_channels)
    mean = normalized.get("mean", default_mean)
    std = normalized.get("std", default_std)
    if not isinstance(mean, (list, tuple)) or len(mean) != in_channels:
        raise ValueError("mean must be a list with length matching in_channels")
    if not isinstance(std, (list, tuple)) or len(std) != in_channels:
        raise ValueError("std must be a list with length matching in_channels")
    normalized["mean"] = tuple(float(value) for value in mean)
    normalized["std"] = tuple(float(value) for value in std)
    return normalized


def _find_manifest_path(dataset_name):
    dataset_dir = Path(DATA_DIR) / dataset_name
    for filename in ("dataset.yaml", "dataset.yml"):
        candidate = dataset_dir / filename
        if candidate.exists():
            return candidate
    return None


def _resolve_split_path(dataset_dir, path_value):
    path_str = str(path_value).strip()
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((Path(dataset_dir) / path).resolve())


def _read_custom_dataset_config(dataset_name):
    manifest_path = _find_manifest_path(dataset_name)
    if manifest_path is None:
        raise FileNotFoundError(f"No dataset.yaml found in data/{dataset_name}")
    raw_cfg = _load_yaml_file(manifest_path)
    return _normalize_dataset_config(dataset_name, raw_cfg, source="custom"), manifest_path


def discover_custom_datasets(include_errors=False):
    datasets = {}
    errors = {}
    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        return (datasets, errors) if include_errors else datasets

    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        dataset_name = child.name
        if dataset_name in DATASETS:
            continue
        manifest_path = _find_manifest_path(dataset_name)
        if manifest_path is None:
            continue
        try:
            cfg, _ = _read_custom_dataset_config(dataset_name)
            datasets[dataset_name] = cfg
        except Exception as exc:
            errors[dataset_name] = str(exc)

    if include_errors:
        return datasets, errors
    return datasets


def get_all_datasets(include_errors=False):
    builtins = {
        name: _normalize_dataset_config(name, cfg, source="builtin")
        for name, cfg in DATASETS.items()
    }
    if include_errors:
        custom, errors = discover_custom_datasets(include_errors=True)
        merged = dict(builtins)
        merged.update(custom)
        return merged, errors

    merged = dict(builtins)
    merged.update(discover_custom_datasets())
    return merged


def validate_custom_dataset(dataset_name):
    dataset_dir = Path(DATA_DIR) / dataset_name
    errors = []
    warnings = []

    if not dataset_dir.exists():
        errors.append(f"Dataset directory does not exist: {dataset_dir}")
        return {"errors": errors, "warnings": warnings}

    manifest_path = _find_manifest_path(dataset_name)
    if manifest_path is None:
        errors.append(f"Missing dataset.yaml in {dataset_dir}")
        return {"errors": errors, "warnings": warnings}

    try:
        cfg, manifest_path = _read_custom_dataset_config(dataset_name)
    except Exception as exc:
        errors.append(f"Invalid dataset.yaml: {exc}")
        return {"errors": errors, "warnings": warnings}

    split_file = dataset_dir / "splits.csv"
    if not split_file.exists():
        errors.append(f"Missing splits.csv in {dataset_dir}")
        return {"errors": errors, "warnings": warnings, "config": cfg}

    try:
        df = pd.read_csv(split_file)
    except Exception as exc:
        errors.append(f"Could not read splits.csv: {exc}")
        return {"errors": errors, "warnings": warnings, "config": cfg}

    task = cfg["task"]
    required_columns = {
        "classification": {"split", "image_path", "label"},
        "multilabel": {"split", "image_path", "labels"},
        "segmentation": {"split", "image_path", "mask_path"},
    }[task]

    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        errors.append(f"splits.csv is missing columns: {', '.join(missing_columns)}")
        return {"errors": errors, "warnings": warnings, "config": cfg}

    allowed_splits = {"train", "val", "test"}
    invalid_splits = sorted(set(df["split"].astype(str)) - allowed_splits)
    if invalid_splits:
        errors.append(f"split column contains invalid values: {', '.join(invalid_splits)}")

    split_counts = df["split"].astype(str).value_counts().to_dict()
    if split_counts.get("train", 0) == 0:
        errors.append("sufficient training data not found: splits.csv needs at least one train row")
    if split_counts.get("val", 0) == 0:
        errors.append("sufficient validation data not found: splits.csv needs at least one val row")

    missing_image_paths = []
    missing_mask_paths = []

    for idx, row in df.iterrows():
        image_path = _resolve_split_path(dataset_dir, row["image_path"])
        if not os.path.exists(image_path):
            missing_image_paths.append((idx + 2, row["image_path"]))
            if len(missing_image_paths) >= 5:
                break

    if task == "segmentation":
        for idx, row in df.iterrows():
            mask_path = _resolve_split_path(dataset_dir, row["mask_path"])
            if not os.path.exists(mask_path):
                missing_mask_paths.append((idx + 2, row["mask_path"]))
                if len(missing_mask_paths) >= 5:
                    break

    if missing_image_paths:
        pretty = ", ".join(f"row {row}: {path}" for row, path in missing_image_paths)
        errors.append(f"image_path entries do not exist ({pretty})")
    if missing_mask_paths:
        pretty = ", ".join(f"row {row}: {path}" for row, path in missing_mask_paths)
        errors.append(f"mask_path entries do not exist ({pretty})")

    if task == "classification":
        try:
            labels = df["label"].astype(int)
        except Exception:
            errors.append("label column must contain integer class ids")
        else:
            invalid = labels[(labels < 0) | (labels >= cfg["num_classes"])]
            if not invalid.empty:
                errors.append("label values must be between 0 and num_classes - 1")
    elif task == "multilabel":
        for idx, raw_labels in enumerate(df["labels"].astype(str), start=2):
            parts = raw_labels.split("|")
            if len(parts) != cfg["num_classes"]:
                errors.append(
                    f"labels column row {idx} must contain {cfg['num_classes']} pipe-separated values"
                )
                break
            if any(part not in {"0", "1"} for part in parts):
                errors.append(f"labels column row {idx} must contain only 0/1 values")
                break

    summary = {
        "dataset": dataset_name,
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(manifest_path),
        "split_file": str(split_file),
        "split_counts": split_counts,
        "num_rows": len(df),
    }
    return {"errors": errors, "warnings": warnings, "config": cfg, "summary": summary}

# ---------------------------------------------------------------------------
# Kaggle download automation
# ---------------------------------------------------------------------------

def check_kaggle_credentials():
    """Verify Kaggle API credentials exist."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return

    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print("ERROR: Kaggle API credentials not found.")
        print("To set up:")
        print("  Option A:")
        print("    1. Go to https://www.kaggle.com/settings")
        print("    2. Click 'Create New Token' under API section")
        print("    3. Place the downloaded kaggle.json at ~/.kaggle/kaggle.json")
        print("    4. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("  Option B:")
        print("    Set the KAGGLE_API_TOKEN environment variable")
        sys.exit(1)
    try:
        with open(kaggle_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        print("ERROR: ~/.kaggle/kaggle.json is not valid JSON.")
        print("Download a fresh token from https://www.kaggle.com/settings and replace the file.")
        sys.exit(1)
    if not payload.get("username") or not payload.get("key"):
        print("ERROR: ~/.kaggle/kaggle.json is missing a Kaggle username or key.")
        print("Download a fresh token from https://www.kaggle.com/settings and replace the file.")
        sys.exit(1)
    if os.name != "nt":  # Unix-style permissions not supported on Windows
        os.chmod(kaggle_json, 0o600)


def download_kaggle_dataset(slug, dest_dir):
    """Download and extract a Kaggle dataset."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    os.makedirs(dest_dir, exist_ok=True)

    marker = os.path.join(dest_dir, ".download_complete")
    if os.path.exists(marker):
        print(f"  Already downloaded: {slug}")
        return

    print(f"  Downloading {slug}...")
    api = KaggleApi()
    api.authenticate()

    # Try as dataset first, fall back to competition
    try:
        api.dataset_download_files(slug, path=dest_dir, unzip=True)
    except Exception:
        try:
            comp_name = slug.split("/")[-1]
            api.competition_download_files(comp_name, path=dest_dir)
            # Unzip competition files
            for zf in Path(dest_dir).glob("*.zip"):
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(dest_dir)
                zf.unlink()
        except Exception as e:
            print(f"  ERROR downloading {slug}: {e}")
            return

    Path(marker).touch()
    print(f"  Done: {slug}")


def download_dataset(dataset_name, refresh_splits=False):
    """Download a specific medical imaging dataset from Kaggle."""
    if dataset_name not in DATASETS:
        validation = validate_custom_dataset(dataset_name)
        if validation.get("errors"):
            print(f"Custom dataset '{dataset_name}' failed validation:")
            for error in validation["errors"]:
                print(f"  - {error}")
            sys.exit(1)

        cfg = validation["config"]
        summary = validation["summary"]
        print(f"Custom dataset '{dataset_name}' is valid.")
        print(f"  Task: {cfg['task']}")
        print(f"  Metric: {cfg['metric']}")
        print(f"  Rows: {summary['num_rows']}")
        print(
            "  Splits: "
            + ", ".join(f"{split}={count}" for split, count in sorted(summary["split_counts"].items()))
        )
        return

    cfg = DATASETS[dataset_name]
    dest_dir = os.path.join(DATA_DIR, dataset_name)
    marker = os.path.join(dest_dir, ".download_complete")
    dataset_present = os.path.exists(marker)

    print(f"Downloading {dataset_name}: {cfg['description']}")
    if dataset_present:
        print(f"  Found existing local dataset at {dest_dir}; skipping Kaggle download.")
    else:
        check_kaggle_credentials()
        download_kaggle_dataset(cfg["kaggle_slug"], dest_dir)

    # Post-download: create train/val splits if needed
    _prepare_splits(dataset_name, dest_dir, force=refresh_splits)


def _prepare_splits(dataset_name, dest_dir, force=False):
    """Create reproducible train/val CSV splits for each dataset."""
    split_file = os.path.join(dest_dir, "splits.csv")
    if os.path.exists(split_file) and not force:
        return

    print(f"  Creating train/val splits for {dataset_name}...")

    if dataset_name == "aptos":
        _prepare_aptos(dest_dir, split_file)
    elif dataset_name == "oct":
        _prepare_oct(dest_dir, split_file)
    elif dataset_name == "chestxray14":
        _prepare_chestxray14(dest_dir, split_file)
    elif dataset_name == "chest_xray":
        _prepare_chest_xray(dest_dir, split_file)
    elif dataset_name == "brain_mri":
        _prepare_brain_mri(dest_dir, split_file)
    elif dataset_name == "retina_vessel":
        _prepare_retina_vessel(dest_dir, split_file)
    elif dataset_name == "pneumothorax":
        _prepare_pneumothorax(dest_dir, split_file)


def _find_file(base_dir, filename):
    """Recursively find a file by name under base_dir."""
    for root, dirs, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def _find_dir(base_dir, dirname):
    """Recursively find a directory by name under base_dir."""
    for root, dirs, files in os.walk(base_dir):
        if dirname in dirs:
            return os.path.join(root, dirname)
    return None


def _prepare_aptos(dest_dir, split_file):
    """APTOS 2019: CSV with id_code and diagnosis (0-4 severity)."""
    csv_path = _find_file(dest_dir, "train.csv")
    if csv_path is None:
        # Try alternative names
        for name in ["train_images.csv", "labels.csv"]:
            csv_path = _find_file(dest_dir, name)
            if csv_path:
                break
    if csv_path is None:
        print("  WARNING: Could not find APTOS labels CSV. Listing directory:")
        for root, dirs, files in os.walk(dest_dir):
            for f in files[:20]:
                print(f"    {os.path.join(root, f)}")
        return

    df = pd.read_csv(csv_path)
    # Find the image column and label column
    img_col = [c for c in df.columns if "id" in c.lower() or "image" in c.lower()][0]
    label_col = [c for c in df.columns if "diagnosis" in c.lower() or "label" in c.lower() or "level" in c.lower()][0]

    # Find the image directory
    img_dir = None
    for candidate in ["train_images", "train", "images"]:
        img_dir = _find_dir(dest_dir, candidate)
        if img_dir:
            break
    if img_dir is None:
        # Use directory containing the most images
        img_dir = dest_dir

    # Build split
    rows = []
    for _, row in df.iterrows():
        img_name = str(row[img_col])
        label = int(row[label_col])
        # Find actual image file
        for ext in [".png", ".jpg", ".jpeg", ""]:
            candidate = os.path.join(img_dir, img_name + ext)
            if os.path.exists(candidate):
                rows.append({"image_path": candidate, "label": label})
                break
        else:
            # Try without extension matching
            matches = list(Path(img_dir).glob(f"{img_name}.*"))
            if matches:
                rows.append({"image_path": str(matches[0]), "label": label})

    split_df = pd.DataFrame(rows)
    skf = StratifiedKFold(n_splits=int(1 / VAL_RATIO), shuffle=True, random_state=RANDOM_SEED)
    split_df["split"] = "train"
    train_idx, val_idx = next(skf.split(split_df, split_df["label"]))
    split_df.loc[val_idx, "split"] = "val"
    split_df.to_csv(split_file, index=False)
    print(f"  APTOS: {len(train_idx)} train, {len(val_idx)} val")


def _prepare_oct(dest_dir, split_file):
    """OCT: folder structure with class subdirectories."""
    # Look for train/test/val directory structure
    rows = []
    for split_name in ["train", "test", "val"]:
        split_dir = _find_dir(dest_dir, split_name)
        if split_dir is None:
            continue
        class_names = sorted([d for d in os.listdir(split_dir)
                             if os.path.isdir(os.path.join(split_dir, d))])
        for cls_idx, cls_name in enumerate(class_names):
            cls_dir = os.path.join(split_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    rows.append({
                        "image_path": os.path.join(cls_dir, img_name),
                        "label": cls_idx,
                        "split": "val" if split_name in ("test", "val") else "train",
                    })

    if not rows:
        print("  WARNING: No OCT images found")
        return

    split_df = pd.DataFrame(rows)
    # If no val split found, create one from train
    if (split_df["split"] == "val").sum() == 0:
        train_mask = split_df["split"] == "train"
        train_df = split_df[train_mask]
        skf = StratifiedKFold(n_splits=int(1 / VAL_RATIO), shuffle=True, random_state=RANDOM_SEED)
        _, val_idx = next(skf.split(train_df, train_df["label"]))
        split_df.loc[train_df.index[val_idx], "split"] = "val"

    split_df.to_csv(split_file, index=False)
    n_train = (split_df["split"] == "train").sum()
    n_val = (split_df["split"] == "val").sum()
    print(f"  OCT: {n_train} train, {n_val} val")


def _prepare_chestxray14(dest_dir, split_file):
    """NIH ChestX-ray14: 112,120 images with 14 multi-label pathologies."""
    # Find the CSV with labels
    csv_path = None
    for candidate in ["Data_Entry_2017.csv", "Data_Entry_2017_v2020.csv"]:
        csv_path = _find_file(dest_dir, candidate)
        if csv_path:
            break

    class_names = DATASETS["chestxray14"]["class_names"]

    # Find image column (usually "Image Index")
    # Find image directories (images are split across images_001/ through images_012/)
    img_dirs = []
    for root, dirs, files in os.walk(dest_dir):
        if any(f.lower().endswith((".png", ".jpg")) for f in files[:5]):
            img_dirs.append(root)

    # Build index: filename -> full path
    img_index = {}
    for d in img_dirs:
        for f in os.listdir(d):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                img_index[f] = os.path.join(d, f)

    if csv_path is None:
        split_meta_dir = _find_dir(dest_dir, "splits_meta")
        if split_meta_dir:
            split_rows = []
            for split_name, filename in [
                ("train", "train_split.csv"),
                ("val", "val_split.csv"),
                ("test", "test_split.csv"),
            ]:
                csv_candidate = os.path.join(split_meta_dir, filename)
                if not os.path.exists(csv_candidate):
                    continue
                split_df = pd.read_csv(csv_candidate)
                if "Image Index" not in split_df.columns:
                    continue
                for _, row in split_df.iterrows():
                    img_name = str(row["Image Index"])
                    image_path = img_index.get(img_name)
                    if image_path is None:
                        continue
                    label_vec = [int(row.get(class_name, 0)) for class_name in class_names]
                    split_rows.append(
                        {
                            "image_path": image_path,
                            "labels": "|".join(str(x) for x in label_vec),
                            "label": sum(label_vec),
                            "split": split_name,
                        }
                    )
            if split_rows:
                split_df = pd.DataFrame(split_rows)
                split_df.to_csv(split_file, index=False)
                n_train = (split_df["split"] == "train").sum()
                n_val = (split_df["split"] == "val").sum()
                n_test = (split_df["split"] == "test").sum()
                print(
                    f"  ChestX-ray14: {n_train} train, {n_val} val, {n_test} test, "
                    f"{len(class_names)} pathologies (from splits_meta)"
                )
                return

        print("  WARNING: Could not find ChestX-ray14 labels CSV")
        for root, dirs, files in os.walk(dest_dir):
            for f in files[:20]:
                print(f"    {os.path.join(root, f)}")
        return

    df = pd.read_csv(csv_path)

    # Find image column (usually "Image Index")
    img_col = [c for c in df.columns if "image" in c.lower() and "index" in c.lower()]
    if not img_col:
        img_col = [c for c in df.columns if "image" in c.lower()]
    img_col = img_col[0] if img_col else df.columns[0]

    # Find label column (usually "Finding Labels")
    label_col = [c for c in df.columns if "finding" in c.lower() or "label" in c.lower()]
    label_col = label_col[0] if label_col else df.columns[1]

    # Use official train/val/test split if available
    train_list_path = _find_file(dest_dir, "train_val_list.txt")
    test_list_path = _find_file(dest_dir, "test_list.txt")

    def patient_id_from_image_name(image_name):
        match = re.match(r"(\d+)_\d+", str(image_name))
        return match.group(1) if match else str(image_name)

    if train_list_path and test_list_path:
        with open(train_list_path) as f:
            train_files = set(line.strip() for line in f)
        with open(test_list_path) as f:
            test_files = set(line.strip() for line in f)

        train_patients = sorted({patient_id_from_image_name(name) for name in train_files})
        rng = np.random.RandomState(RANDOM_SEED)
        rng.shuffle(train_patients)
        val_patient_count = max(1, int(len(train_patients) * VAL_RATIO))
        val_patients = set(train_patients[:val_patient_count])
    else:
        train_files = None
        test_files = None
        val_patients = None

    rows = []
    for _, row in df.iterrows():
        img_name = str(row[img_col])
        if img_name not in img_index:
            continue

        labels_str = str(row[label_col])
        # Multi-label: encode as pipe-separated binary vector
        label_vec = [0] * len(class_names)
        for finding in labels_str.split("|"):
            finding = finding.strip()
            if finding in class_names:
                idx = class_names.index(finding)
                label_vec[idx] = 1

        # Determine split
        if train_files is not None:
            if img_name in test_files:
                split = "test"
            else:
                split = "val" if patient_id_from_image_name(img_name) in val_patients else "train"
        else:
            split = "train"

        rows.append({
            "image_path": img_index[img_name],
            "labels": "|".join(str(x) for x in label_vec),
            "label": sum(label_vec),  # for stratification
            "split": split,
        })

    split_df = pd.DataFrame(rows)

    # If no official split, create one
    if train_files is None:
        n = len(split_df)
        indices = np.random.RandomState(RANDOM_SEED).permutation(n)
        val_count = max(1, int(n * VAL_RATIO))
        split_df["split"] = "train"
        split_df.loc[split_df.index[indices[:val_count]], "split"] = "val"

    split_df.to_csv(split_file, index=False)
    n_train = (split_df["split"] == "train").sum()
    n_val = (split_df["split"] == "val").sum()
    n_test = (split_df["split"] == "test").sum()
    print(f"  ChestX-ray14: {n_train} train, {n_val} val, {n_test} test, {len(class_names)} pathologies")


def _prepare_chest_xray(dest_dir, split_file):
    """Chest X-ray: folder structure NORMAL/PNEUMONIA."""
    rows = []
    for split_name in ["train", "test", "val"]:
        split_dir = _find_dir(dest_dir, split_name)
        if split_dir is None:
            continue
        for cls_idx, cls_name in enumerate(["NORMAL", "PNEUMONIA"]):
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    rows.append({
                        "image_path": os.path.join(cls_dir, img_name),
                        "label": cls_idx,
                        "split": "val" if split_name in ("test", "val") else "train",
                    })

    if not rows:
        print("  WARNING: No chest X-ray images found")
        return

    split_df = pd.DataFrame(rows)
    split_df.to_csv(split_file, index=False)
    n_train = (split_df["split"] == "train").sum()
    n_val = (split_df["split"] == "val").sum()
    print(f"  Chest X-ray: {n_train} train, {n_val} val")


def _prepare_brain_mri(dest_dir, split_file):
    """Brain MRI: TCIA LGG dataset with FLAIR images and masks."""
    rows = []
    # Walk through patient directories looking for image/mask pairs
    for root, dirs, files in os.walk(dest_dir):
        images = sorted([f for f in files if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))
                        and "mask" not in f.lower()])
        masks = sorted([f for f in files if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))
                       and "mask" in f.lower()])

        if not images or not masks:
            continue

        for img_name, mask_name in zip(images, masks):
            rows.append({
                "image_path": os.path.join(root, img_name),
                "mask_path": os.path.join(root, mask_name),
                "label": 0,  # placeholder for stratification
            })

    if not rows:
        print("  WARNING: No brain MRI image/mask pairs found")
        return

    split_df = pd.DataFrame(rows)
    # Stratify by whether mask has positive pixels (tumor present)
    has_tumor = []
    for _, row in split_df.iterrows():
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        has_tumor.append(1 if mask is not None and mask.max() > 0 else 0)
    split_df["label"] = has_tumor

    skf = StratifiedKFold(n_splits=int(1 / VAL_RATIO), shuffle=True, random_state=RANDOM_SEED)
    split_df["split"] = "train"
    _, val_idx = next(skf.split(split_df, split_df["label"]))
    split_df.loc[split_df.index[val_idx], "split"] = "val"
    split_df.to_csv(split_file, index=False)
    n_train = (split_df["split"] == "train").sum()
    n_val = (split_df["split"] == "val").sum()
    print(f"  Brain MRI: {n_train} train, {n_val} val")


def _prepare_retina_vessel(dest_dir, split_file):
    """Retinal vessel segmentation: image + mask pairs."""
    rows = []
    # Look for common directory structures
    for img_dirname in ["images", "training/images", "train/images"]:
        img_dir = os.path.join(dest_dir, img_dirname)
        if not os.path.isdir(img_dir):
            img_dir = _find_dir(dest_dir, "images")
        if img_dir and os.path.isdir(img_dir):
            break
    else:
        img_dir = None

    mask_dir = None
    for mask_dirname in ["masks", "training/1st_manual", "train/masks", "labels", "annotations"]:
        candidate = os.path.join(dest_dir, mask_dirname)
        if os.path.isdir(candidate):
            mask_dir = candidate
            break
    if mask_dir is None:
        mask_dir = _find_dir(dest_dir, "masks")
        if mask_dir is None:
            mask_dir = _find_dir(dest_dir, "labels")

    if img_dir and mask_dir:
        img_files = sorted([f for f in os.listdir(img_dir)
                           if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"))])
        mask_files = sorted([f for f in os.listdir(mask_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"))])

        for img_name in img_files:
            # Try to find matching mask
            stem = Path(img_name).stem
            mask_match = None
            for mf in mask_files:
                if stem in mf or Path(mf).stem in stem:
                    mask_match = mf
                    break
            if mask_match is None and len(img_files) == len(mask_files):
                idx = img_files.index(img_name)
                mask_match = mask_files[idx]

            if mask_match:
                rows.append({
                    "image_path": os.path.join(img_dir, img_name),
                    "mask_path": os.path.join(mask_dir, mask_match),
                    "label": 0,
                })

    if not rows:
        # Fallback: walk entire directory for paired files
        print("  Scanning for image/mask pairs...")
        for root, dirs, files in os.walk(dest_dir):
            images = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
                      and "mask" not in f.lower() and "label" not in f.lower()]
            masks = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
                     and ("mask" in f.lower() or "label" in f.lower())]
            for img, msk in zip(sorted(images), sorted(masks)):
                rows.append({
                    "image_path": os.path.join(root, img),
                    "mask_path": os.path.join(root, msk),
                    "label": 0,
                })

    if not rows:
        print("  WARNING: No retinal vessel image/mask pairs found")
        return

    split_df = pd.DataFrame(rows)
    n = len(split_df)
    indices = np.random.RandomState(RANDOM_SEED).permutation(n)
    val_count = max(1, int(n * VAL_RATIO))
    split_df["split"] = "train"
    split_df.loc[split_df.index[indices[:val_count]], "split"] = "val"
    split_df.to_csv(split_file, index=False)
    n_train = (split_df["split"] == "train").sum()
    n_val = (split_df["split"] == "val").sum()
    print(f"  Retina vessel: {n_train} train, {n_val} val")


def _prepare_pneumothorax(dest_dir, split_file):
    """Pneumothorax segmentation: chest X-rays with masks."""
    rows = []

    # Look for structured directories
    for root, dirs, files in os.walk(dest_dir):
        images = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".dcm"))
                  and "mask" not in f.lower()]
        masks = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))
                 and "mask" in f.lower()]

        if images and masks:
            for img_name in sorted(images):
                stem = Path(img_name).stem
                mask_match = None
                for mf in sorted(masks):
                    if stem in mf:
                        mask_match = mf
                        break
                if mask_match:
                    rows.append({
                        "image_path": os.path.join(root, img_name),
                        "mask_path": os.path.join(root, mask_match),
                        "label": 0,
                    })

    # Also check for separate image/mask directories
    if not rows:
        img_dir = _find_dir(dest_dir, "images")
        mask_dir = _find_dir(dest_dir, "masks")
        if img_dir and mask_dir:
            img_files = sorted(os.listdir(img_dir))
            mask_files = sorted(os.listdir(mask_dir))
            for img_f, mask_f in zip(img_files, mask_files):
                if img_f.lower().endswith((".png", ".jpg", ".jpeg")):
                    rows.append({
                        "image_path": os.path.join(img_dir, img_f),
                        "mask_path": os.path.join(mask_dir, mask_f),
                        "label": 0,
                    })

    if not rows:
        print("  WARNING: No pneumothorax image/mask pairs found")
        return

    split_df = pd.DataFrame(rows)
    # Stratify by whether mask has positive pixels
    has_positive = []
    for _, row in split_df.head(min(500, len(split_df))).iterrows():
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        has_positive.append(1 if mask is not None and mask.max() > 0 else 0)
    # Extend for remaining
    if len(has_positive) < len(split_df):
        has_positive.extend([0] * (len(split_df) - len(has_positive)))
    split_df["label"] = has_positive

    n = len(split_df)
    indices = np.random.RandomState(RANDOM_SEED).permutation(n)
    val_count = max(1, int(n * VAL_RATIO))
    split_df["split"] = "train"
    split_df.loc[split_df.index[indices[:val_count]], "split"] = "val"
    split_df.to_csv(split_file, index=False)
    n_train = (split_df["split"] == "train").sum()
    n_val = (split_df["split"] == "val").sum()
    print(f"  Pneumothorax: {n_train} train, {n_val} val")


# ---------------------------------------------------------------------------
# Dataset classes (imported by train.py)
# ---------------------------------------------------------------------------

def get_dataset_config(dataset_name):
    """Return dataset configuration dict."""
    datasets = get_all_datasets()
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {sorted(datasets.keys())}")
    return datasets[dataset_name]


class MedicalClassificationDataset(Dataset):
    """Generic medical image classification dataset loaded from splits.csv."""

    def __init__(self, dataset_name, split="train", transform=None):
        cfg = get_dataset_config(dataset_name)
        data_dir = os.path.join(DATA_DIR, dataset_name)
        split_file = os.path.join(data_dir, "splits.csv")

        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"splits.csv not found for {dataset_name}. Run: python prepare.py --dataset {dataset_name}"
            )

        df = pd.read_csv(split_file)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.dataset_name = dataset_name
        self.transform = transform
        self.in_channels = cfg["in_channels"]
        self.num_classes = cfg["num_classes"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = _resolve_split_path(os.path.join(DATA_DIR, self.dataset_name), row["image_path"])
        label = int(row["label"])

        if self.in_channels == 1:
            image = Image.open(img_path).convert("L")
        else:
            image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class MedicalMultilabelDataset(Dataset):
    """Medical multi-label classification dataset (e.g., ChestX-ray14).
    Each image can have multiple pathology labels simultaneously.
    """

    def __init__(self, dataset_name, split="train", transform=None):
        cfg = get_dataset_config(dataset_name)
        data_dir = os.path.join(DATA_DIR, dataset_name)
        split_file = os.path.join(data_dir, "splits.csv")

        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"splits.csv not found for {dataset_name}. Run: python prepare.py --dataset {dataset_name}"
            )

        df = pd.read_csv(split_file)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.dataset_name = dataset_name
        self.transform = transform
        self.in_channels = cfg["in_channels"]
        self.num_classes = cfg["num_classes"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = _resolve_split_path(os.path.join(DATA_DIR, self.dataset_name), row["image_path"])

        # Parse multi-label vector from pipe-separated string
        labels = torch.tensor(
            [int(x) for x in str(row["labels"]).split("|")],
            dtype=torch.float32,
        )

        if self.in_channels == 1:
            image = Image.open(img_path).convert("L")
        else:
            image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, labels


class MedicalSegmentationDataset(Dataset):
    """Generic medical image segmentation dataset loaded from splits.csv."""

    def __init__(self, dataset_name, split="train", image_size=256, augment=False):
        cfg = get_dataset_config(dataset_name)
        data_dir = os.path.join(DATA_DIR, dataset_name)
        split_file = os.path.join(data_dir, "splits.csv")

        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"splits.csv not found for {dataset_name}. Run: python prepare.py --dataset {dataset_name}"
            )

        df = pd.read_csv(split_file)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.augment = augment
        self.in_channels = cfg["in_channels"]
        self.num_classes = cfg["num_classes"]
        self.mean = cfg["mean"]
        self.std = cfg["std"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = _resolve_split_path(os.path.join(DATA_DIR, self.dataset_name), row["image_path"])
        mask_path = _resolve_split_path(os.path.join(DATA_DIR, self.dataset_name), row["mask_path"])

        # Load image
        if self.in_channels == 1:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = image.astype(np.float32) / 255.0
            image = (image - self.mean[0]) / self.std[0]
            image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        else:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = image.astype(np.float32) / 255.0
            for c in range(3):
                image[:, :, c] = (image[:, :, c] - self.mean[c]) / self.std[c]
            image = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        # Binarize mask (threshold at 128 for binary segmentation)
        if self.num_classes == 2:
            mask = (mask > 128).astype(np.int64)
        mask = torch.from_numpy(mask).long()

        # Simple augmentation
        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, [-1])
                mask = torch.flip(mask, [-1])
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, [-2])
                mask = torch.flip(mask, [-2])

        return image, mask


# ---------------------------------------------------------------------------
# Dataloaders (imported by train.py)
# ---------------------------------------------------------------------------

class RepeatChannels:
    """Expand a single-channel tensor to 3 channels without a lambda transform."""

    def __call__(self, tensor):
        return tensor.repeat(3, 1, 1)

def make_classification_transforms(dataset_name, train=True):
    """Build transforms for medical classification."""
    cfg = get_dataset_config(dataset_name)
    img_size = cfg["image_size"]

    if cfg["in_channels"] == 1:
        mean, std = cfg["mean"], cfg["std"]
    else:
        mean, std = cfg["mean"], cfg["std"]

    if train:
        transform_list = [
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.1, contrast=0.1) if cfg["in_channels"] == 3 else T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    else:
        transform_list = [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]

    # For grayscale: expand to 3 channels for pretrained models
    if cfg["in_channels"] == 1:
        transform_list.append(RepeatChannels())

    return T.Compose(transform_list)


def make_dataloader(dataset_name, batch_size, split="train"):
    """Create a dataloader for the specified dataset and split.

    Returns:
        dataloader, num_classes
    """
    cfg = get_dataset_config(dataset_name)
    is_train = (split == "train")

    if cfg["task"] in ("classification", "multilabel"):
        transforms = make_classification_transforms(dataset_name, train=is_train)
        if cfg["task"] == "multilabel":
            dataset = MedicalMultilabelDataset(
                dataset_name, split=split, transform=transforms
            )
        else:
            dataset = MedicalClassificationDataset(
                dataset_name, split=split, transform=transforms
            )
    else:
        dataset = MedicalSegmentationDataset(
            dataset_name, split=split,
            image_size=cfg["image_size"],
            augment=is_train,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    return loader, cfg["num_classes"]


def make_infinite_dataloader(dataset_name, batch_size, split="train"):
    """Create an infinite dataloader that loops over the dataset.

    Yields:
        (images, targets, epoch)
    """
    loader, num_classes = make_dataloader(dataset_name, batch_size, split)
    epoch = 1
    while True:
        for batch in loader:
            images, targets = batch
            yield images, targets, epoch
        epoch += 1


def get_class_weights(dataset_name, device=None):
    """Compute inverse-frequency class weights for imbalanced datasets."""
    device = resolve_eval_device(device)
    data_dir = os.path.join(DATA_DIR, dataset_name)
    split_file = os.path.join(data_dir, "splits.csv")
    df = pd.read_csv(split_file)
    train_df = df[df["split"] == "train"]

    num_classes = get_dataset_config(dataset_name)["num_classes"]
    counts = train_df["label"].value_counts().sort_index()

    # Ensure all classes represented
    all_counts = np.zeros(num_classes)
    for cls_id, count in counts.items():
        if cls_id < num_classes:
            all_counts[cls_id] = count

    # Inverse frequency weighting, clamped
    total = all_counts.sum()
    weights = total / (num_classes * np.maximum(all_counts, 1))
    weights = np.clip(weights, 0.5, 10.0)

    return torch.tensor(weights, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — these are the fixed metrics)
# ---------------------------------------------------------------------------

def resolve_eval_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluation_autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def evaluate_classification(model, dataset_name, batch_size, device=None):
    """
    Evaluate classification on the validation set.

    Returns dict with:
        accuracy (float): Top-1 accuracy (0-100)
        loss (float): Average cross-entropy loss
        auc (float): ROC AUC (macro, for binary/multiclass) (0-100)
        kappa (float): Quadratic weighted Cohen's kappa (0-1)
    """
    cfg = get_dataset_config(dataset_name)
    loader, num_classes = make_dataloader(dataset_name, batch_size, split="val")
    device = resolve_eval_device(device)
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []
    total_loss = 0.0
    total_samples = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        with evaluation_autocast_context(device):
            logits = model(images)

        loss = F.cross_entropy(logits, targets, reduction="sum")
        total_loss += loss.item()
        total_samples += targets.size(0)

        probs = F.softmax(logits.float(), dim=1)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()

    accuracy = 100.0 * (all_preds == all_targets).mean()
    avg_loss = total_loss / total_samples

    # Quadratic weighted kappa (important for DR grading)
    kappa = cohen_kappa_score(all_targets, all_preds, weights="quadratic")

    # AUC
    try:
        if num_classes == 2:
            auc = 100.0 * roc_auc_score(all_targets, all_probs[:, 1])
        else:
            auc = 100.0 * roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0

    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "auc": auc,
        "kappa": kappa,
    }


@torch.no_grad()
def evaluate_multilabel(model, dataset_name, batch_size, device=None):
    """
    Evaluate multi-label classification (e.g., CheXNet/ChestX-ray14).

    Returns dict with:
        mean_auc (float): Mean AUC across all pathologies (0-100)
        per_class_auc (dict): AUC for each pathology
    """
    cfg = get_dataset_config(dataset_name)
    loader, num_classes = make_dataloader(dataset_name, batch_size, split="val")
    device = resolve_eval_device(device)
    model = model.to(device)
    model.eval()

    all_probs = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)

        with evaluation_autocast_context(device):
            logits = model(images)

        probs = torch.sigmoid(logits.float())
        all_probs.append(probs.cpu())
        all_targets.append(targets.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    per_class_auc = {}
    valid_aucs = []
    class_names = cfg["class_names"]

    for i in range(num_classes):
        try:
            auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
            per_class_auc[class_names[i]] = round(100.0 * auc, 2)
            valid_aucs.append(auc)
        except ValueError:
            per_class_auc[class_names[i]] = 0.0

    mean_auc = 100.0 * np.mean(valid_aucs) if valid_aucs else 0.0

    return {
        "mean_auc": mean_auc,
        "per_class_auc": per_class_auc,
    }


@torch.no_grad()
def evaluate_segmentation(model, dataset_name, batch_size, device=None):
    """
    Evaluate segmentation on the validation set.

    Returns dict with:
        dice (float): Dice coefficient (0-100)
        iou (float): Intersection over Union (0-100)
        pixel_acc (float): Pixel accuracy (0-100)
        sensitivity (float): True positive rate (0-100)
        specificity (float): True negative rate (0-100)
    """
    cfg = get_dataset_config(dataset_name)
    loader, num_classes = make_dataloader(dataset_name, batch_size, split="val")
    device = resolve_eval_device(device)
    model = model.to(device)
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_correct = 0
    total_pixels = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        with evaluation_autocast_context(device):
            logits = model(images)

        # Upsample if needed
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        preds = logits.argmax(dim=1)

        # For binary segmentation, compute on foreground class (class 1)
        if num_classes == 2:
            pred_pos = preds == 1
            target_pos = targets == 1
            total_tp += (pred_pos & target_pos).sum().item()
            total_fp += (pred_pos & ~target_pos).sum().item()
            total_fn += (~pred_pos & target_pos).sum().item()
            total_tn += (~pred_pos & ~target_pos).sum().item()

        total_correct += (preds == targets).sum().item()
        total_pixels += targets.numel()

    # Dice = 2*TP / (2*TP + FP + FN)
    dice = 100.0 * (2 * total_tp) / max(2 * total_tp + total_fp + total_fn, 1)
    # IoU = TP / (TP + FP + FN)
    iou = 100.0 * total_tp / max(total_tp + total_fp + total_fn, 1)
    pixel_acc = 100.0 * total_correct / max(total_pixels, 1)
    sensitivity = 100.0 * total_tp / max(total_tp + total_fn, 1)
    specificity = 100.0 * total_tn / max(total_tn + total_fp, 1)

    return {
        "dice": dice,
        "iou": iou,
        "pixel_acc": pixel_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


# ---------------------------------------------------------------------------
# Utility: CLAHE preprocessing for medical images
# ---------------------------------------------------------------------------

def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Useful for fundus images and X-rays. Call from train.py augmentations.
    """
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(image)


# ---------------------------------------------------------------------------
# BYOD (bring-your-own-data) helpers
# ---------------------------------------------------------------------------


def _auto_metric(task, num_classes):
    """Auto-select the primary metric for a given task type."""
    if task == "segmentation":
        return "dice"
    elif task == "multilabel":
        return "mean_auc"
    elif num_classes == 2:
        return "auc"
    elif num_classes <= 5:
        return "kappa"
    return "accuracy"


def scan_directory(data_dir):
    """Scan a data directory and detect its structure."""
    result = {
        "csv_files": [], "image_dirs": [], "image_count": 0,
        "mask_dirs": [], "class_dirs": [], "dicom_files": [],
        "has_splits": False, "total_files": 0,
    }
    for root, dirs, files in os.walk(data_dir):
        dir_names_lower = [d.lower() for d in dirs]
        if "train" in dir_names_lower or "val" in dir_names_lower:
            result["has_splits"] = True
        images = [f for f in files if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
        if images:
            result["image_dirs"].append({"path": root, "count": len(images)})
            result["image_count"] += len(images)
        csvs = [f for f in files if f.lower().endswith(".csv")]
        for csv_name in csvs:
            csv_path = os.path.join(root, csv_name)
            try:
                df = pd.read_csv(csv_path, nrows=5)
                result["csv_files"].append({
                    "path": csv_path, "name": csv_name,
                    "columns": list(df.columns),
                    "rows": len(pd.read_csv(csv_path)),
                })
            except Exception:
                pass
        subdirs = [d for d in dirs if os.path.isdir(os.path.join(root, d))]
        has_images_in_subdirs = False
        for d in subdirs:
            subdir_path = os.path.join(root, d)
            sub_images = [f for f in os.listdir(subdir_path)
                         if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
            if sub_images:
                has_images_in_subdirs = True
        if has_images_in_subdirs and len(subdirs) >= 2:
            if not any(d.lower() in ("train", "test", "val", "validation",
                                      "images", "masks", "labels", "annotations")
                       for d in subdirs):
                result["class_dirs"].append({
                    "path": root, "classes": sorted(subdirs),
                    "num_classes": len(subdirs),
                })
        for d in subdirs:
            if any(kw in d.lower() for kw in ["mask", "label", "annotation",
                                                "segmentation", "gt", "ground_truth"]):
                mask_path = os.path.join(root, d)
                mask_count = len([f for f in os.listdir(mask_path)
                                 if Path(f).suffix.lower() in IMAGE_EXTENSIONS])
                if mask_count > 0:
                    result["mask_dirs"].append({"path": mask_path, "count": mask_count})
        result["total_files"] += len(files)
    return result


def detect_task(scan_result):
    """Detect the task type from a directory scan result."""
    if scan_result["mask_dirs"]:
        return "segmentation"
    if scan_result["class_dirs"]:
        return "classification"
    if scan_result["csv_files"]:
        for csv_info in scan_result["csv_files"]:
            cols = [c.lower() for c in csv_info["columns"]]
            if any("|" in str(c) for c in cols):
                return "multilabel"
            label_cols = [c for c in cols if any(kw in c for kw in
                         ["label", "target", "class", "diagnosis", "finding", "category"])]
            if label_cols:
                try:
                    df = pd.read_csv(csv_info["path"], nrows=100)
                    for col in label_cols:
                        orig_col = [c for c in df.columns if c.lower() == col][0]
                        if df[orig_col].astype(str).str.contains(r"\|").any():
                            return "multilabel"
                except Exception:
                    pass
                return "classification"
    return "classification"


def detect_label_info(scan_result, data_dir):
    """Detect label columns, class names, and image column from the scan."""
    info = {"num_classes": 2, "class_names": [], "label_column": None,
            "image_column": None, "patient_column": None}
    if scan_result["class_dirs"]:
        class_info = scan_result["class_dirs"][0]
        info["num_classes"] = class_info["num_classes"]
        info["class_names"] = class_info["classes"]
        return info
    if scan_result["csv_files"]:
        csv_info = scan_result["csv_files"][0]
        try:
            df = pd.read_csv(csv_info["path"], nrows=1000)
        except Exception:
            return info
        cols = list(df.columns)
        cols_lower = [c.lower() for c in cols]
        for pattern in ["image", "file", "path", "filename", "id_code"]:
            matches = [c for c, cl in zip(cols, cols_lower) if pattern in cl]
            if matches:
                info["image_column"] = matches[0]
                break
        for pattern in ["label", "target", "class", "diagnosis",
                        "finding", "category", "grade", "level"]:
            matches = [c for c, cl in zip(cols, cols_lower)
                       if pattern in cl and "image" not in cl]
            if matches:
                info["label_column"] = matches[0]
                break
        for pattern in ["patient_id", "patientid", "patient", "subject_id",
                        "subject", "case_id", "case"]:
            matches = [c for c, cl in zip(cols, cols_lower) if cl == pattern]
            if matches:
                info["patient_column"] = matches[0]
                break
        if info["label_column"]:
            labels = df[info["label_column"]].dropna()
            unique_labels = sorted(labels.unique(), key=lambda x: str(x))
            info["num_classes"] = len(unique_labels)
            info["class_names"] = [str(l) for l in unique_labels][:50]
    return info


def detect_channels(data_dir, scan_result):
    """Detect number of image channels from a sample image."""
    for img_dir_info in scan_result["image_dirs"]:
        try:
            for f in os.listdir(img_dir_info["path"]):
                if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                    img = Image.open(os.path.join(img_dir_info["path"], f))
                    if img.mode in ("L", "1", "I", "F"):
                        return 1
                    return 3
        except Exception:
            continue
    return 3


def compute_dataset_statistics(data_dir, scan_result, in_channels=3, max_samples=1000):
    """Compute per-channel mean and std from sampled images."""
    all_images = []
    for img_dir_info in scan_result["image_dirs"]:
        for f in os.listdir(img_dir_info["path"]):
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                all_images.append(os.path.join(img_dir_info["path"], f))
    if not all_images:
        if in_channels == 1:
            return (0.485,), (0.229,)
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    rng = np.random.RandomState(RANDOM_SEED)
    sample = rng.choice(all_images, size=min(max_samples, len(all_images)), replace=False)
    n = 0
    channel_sum = np.zeros(in_channels, dtype=np.float64)
    channel_sq_sum = np.zeros(in_channels, dtype=np.float64)
    for img_path in sample:
        try:
            img = Image.open(img_path)
            if in_channels == 1:
                img = img.convert("L")
                arr = np.array(img, dtype=np.float64).ravel() / 255.0
                channel_sum[0] += arr.mean()
                channel_sq_sum[0] += (arr ** 2).mean()
            else:
                img = img.convert("RGB")
                arr = np.array(img, dtype=np.float64) / 255.0
                for c in range(3):
                    channel_sum[c] += arr[:, :, c].mean()
                    channel_sq_sum[c] += (arr[:, :, c] ** 2).mean()
            n += 1
        except Exception:
            continue
    if n == 0:
        if in_channels == 1:
            return (0.485,), (0.229,)
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    mean = channel_sum / n
    std = np.sqrt(channel_sq_sum / n - mean ** 2)
    std = np.maximum(std, 1e-6)
    return tuple(float(m) for m in mean), tuple(float(s) for s in std)


def _detect_patient_ids_from_paths(image_paths):
    """Try to extract patient IDs from image file paths."""
    parents = [os.path.basename(os.path.dirname(p)) for p in image_paths]
    unique_parents = set(parents)
    if 1 < len(unique_parents) < len(image_paths) * 0.5:
        return parents
    stems = [Path(p).stem for p in image_paths]
    prefixes = []
    for s in stems:
        for sep in ["_", "-"]:
            parts = s.rsplit(sep, 1)
            if len(parts) == 2:
                prefixes.append(parts[0])
                break
        else:
            prefixes.append(s)
    unique_prefixes = set(prefixes)
    if 1 < len(unique_prefixes) < len(image_paths) * 0.5:
        return prefixes
    return None


def _stratified_split(df, val_ratio=VAL_RATIO, seed=RANDOM_SEED):
    """Standard stratified train/val split."""
    df = df.copy()
    if "label" in df.columns and df["label"].nunique() >= 2:
        skf = StratifiedKFold(n_splits=max(2, int(1 / val_ratio)), shuffle=True, random_state=seed)
        df["split"] = "train"
        _, val_idx = next(skf.split(df, df["label"]))
        df.loc[df.index[val_idx], "split"] = "val"
    else:
        n = len(df)
        indices = np.random.RandomState(seed).permutation(n)
        val_count = max(1, int(n * val_ratio))
        df["split"] = "train"
        df.loc[df.index[indices[:val_count]], "split"] = "val"
    return df


def _patient_level_split(df, patient_col=None, val_ratio=VAL_RATIO, seed=RANDOM_SEED):
    """Split using patient-level grouping to prevent data leakage."""
    if patient_col and patient_col in df.columns:
        groups = df[patient_col].values
    elif "image_path" in df.columns:
        groups = _detect_patient_ids_from_paths(df["image_path"].tolist())
        if groups is None:
            return _stratified_split(df, val_ratio, seed)
        groups = np.array(groups)
    else:
        return _stratified_split(df, val_ratio, seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    try:
        train_idx, val_idx = next(gss.split(df, groups=groups))
        df = df.copy()
        df["split"] = "train"
        df.loc[df.index[val_idx], "split"] = "val"
        n_patients_train = len(set(groups[train_idx]))
        n_patients_val = len(set(groups[val_idx]))
        print(f"  Patient-level split: {n_patients_train} patients train, "
              f"{n_patients_val} patients val")
        return df
    except Exception:
        return _stratified_split(df, val_ratio, seed)


def _prepare_generic_classification(data_dir, scan_result, label_info, split_file,
                                     patient_column=None):
    """Generic classification split from folder-of-folders structure."""
    rows = []
    if scan_result["has_splits"]:
        for split_name in ["train", "val", "validation", "test"]:
            split_dir = _find_dir(data_dir, split_name)
            if split_dir is None:
                continue
            mapped_split = "val" if split_name in ("test", "val", "validation") else "train"
            class_dirs = sorted([d for d in os.listdir(split_dir)
                                if os.path.isdir(os.path.join(split_dir, d))])
            for cls_idx, cls_name in enumerate(class_dirs):
                cls_dir = os.path.join(split_dir, cls_name)
                for img in os.listdir(cls_dir):
                    if Path(img).suffix.lower() in IMAGE_EXTENSIONS:
                        rows.append({"image_path": os.path.join(cls_dir, img),
                                    "label": cls_idx, "split": mapped_split})
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(split_file, index=False)
            n_train = (df["split"] == "train").sum()
            n_val = (df["split"] == "val").sum()
            print(f"  Classification (pre-split): {n_train} train, {n_val} val, "
                  f"{len(set(df['label']))} classes")
            return
    if scan_result["class_dirs"]:
        class_info = scan_result["class_dirs"][0]
        for cls_idx, cls_name in enumerate(class_info["classes"]):
            cls_dir = os.path.join(class_info["path"], cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img in os.listdir(cls_dir):
                if Path(img).suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append({"image_path": os.path.join(cls_dir, img), "label": cls_idx})
    if not rows:
        print("  WARNING: No images found for classification")
        return
    df = pd.DataFrame(rows)
    df = _patient_level_split(df, patient_column)
    df.to_csv(split_file, index=False)
    n_train = (df["split"] == "train").sum()
    n_val = (df["split"] == "val").sum()
    print(f"  Classification: {n_train} train, {n_val} val, {len(set(df['label']))} classes")


def _prepare_generic_csv(data_dir, scan_result, label_info, task, split_file,
                          patient_column=None):
    """Generic split from CSV + images directory."""
    csv_info = scan_result["csv_files"][0]
    df = pd.read_csv(csv_info["path"])
    img_col = label_info.get("image_column")
    label_col = label_info.get("label_column")
    if not img_col or not label_col:
        print(f"  WARNING: Could not detect image/label columns in {csv_info['name']}")
        return
    img_dir = None
    for candidate in ["images", "train_images", "train", "imgs"]:
        img_dir = _find_dir(data_dir, candidate)
        if img_dir:
            break
    if img_dir is None:
        if scan_result["image_dirs"]:
            img_dir = max(scan_result["image_dirs"], key=lambda x: x["count"])["path"]
        else:
            img_dir = data_dir
    rows = []
    for _, row in df.iterrows():
        img_name = str(row[img_col])
        label = row[label_col]
        if task == "multilabel":
            rows.append({"image_path": "", "labels": str(label), "label": 0})
        else:
            if isinstance(label, str):
                all_labels = sorted(df[label_col].unique(), key=str)
                label = all_labels.index(label)
            rows.append({"image_path": "", "label": int(label)})
        resolved = None
        for ext in ["", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            candidate_path = os.path.join(img_dir, img_name + ext)
            if os.path.exists(candidate_path):
                resolved = candidate_path
                break
        if resolved is None:
            matches = list(Path(img_dir).glob(f"{img_name}.*"))
            if matches:
                resolved = str(matches[0])
        if resolved:
            rows[-1]["image_path"] = resolved
        else:
            rows.pop()
    if not rows:
        print("  WARNING: No images matched from CSV")
        return
    result_df = pd.DataFrame(rows)
    patient_col = patient_column or label_info.get("patient_column")
    result_df = _patient_level_split(result_df, patient_col)
    result_df.to_csv(split_file, index=False)
    n_train = (result_df["split"] == "train").sum()
    n_val = (result_df["split"] == "val").sum()
    print(f"  CSV-based ({task}): {n_train} train, {n_val} val")


def _prepare_generic_segmentation(data_dir, scan_result, split_file, patient_column=None):
    """Generic segmentation split from images + masks directories."""
    rows = []
    img_dir = mask_dir = None
    for img_name in ["images", "imgs", "image", "train"]:
        candidate = _find_dir(data_dir, img_name)
        if candidate:
            img_dir = candidate
            break
    for mask_name in ["masks", "mask", "labels", "annotations", "gt",
                      "ground_truth", "segmentation"]:
        candidate = _find_dir(data_dir, mask_name)
        if candidate:
            mask_dir = candidate
            break
    if img_dir and mask_dir:
        img_files = sorted([f for f in os.listdir(img_dir)
                           if Path(f).suffix.lower() in IMAGE_EXTENSIONS])
        mask_files = sorted([f for f in os.listdir(mask_dir)
                            if Path(f).suffix.lower() in IMAGE_EXTENSIONS])
        for img_name in img_files:
            stem = Path(img_name).stem
            mask_match = None
            for mf in mask_files:
                if stem in mf or Path(mf).stem in stem:
                    mask_match = mf
                    break
            if mask_match is None and len(img_files) == len(mask_files):
                idx = img_files.index(img_name)
                mask_match = mask_files[idx]
            if mask_match:
                rows.append({"image_path": os.path.join(img_dir, img_name),
                            "mask_path": os.path.join(mask_dir, mask_match), "label": 0})
    if not rows:
        for root, dirs, files in os.walk(data_dir):
            images = sorted([f for f in files if Path(f).suffix.lower() in IMAGE_EXTENSIONS
                            and "mask" not in f.lower() and "label" not in f.lower()])
            masks = sorted([f for f in files if Path(f).suffix.lower() in IMAGE_EXTENSIONS
                           and ("mask" in f.lower() or "label" in f.lower())])
            for img, msk in zip(images, masks):
                rows.append({"image_path": os.path.join(root, img),
                            "mask_path": os.path.join(root, msk), "label": 0})
    if not rows:
        print("  WARNING: No image/mask pairs found for segmentation")
        return
    df = pd.DataFrame(rows)
    df = _patient_level_split(df, patient_column)
    df.to_csv(split_file, index=False)
    n_train = (df["split"] == "train").sum()
    n_val = (df["split"] == "val").sum()
    print(f"  Segmentation: {n_train} train, {n_val} val")


def prepare_custom_dataset(data_path, task_override=None, config_overrides=None,
                            patient_column=None, compute_stats=False):
    """Auto-detect and prepare any medical imaging dataset.

    Scans directory structure, infers task type and metadata, creates
    train/val splits (patient-level if possible), and writes a dataset.json
    manifest for the dynamic registry.
    """
    import json as _json

    data_path = os.path.abspath(data_path)
    ds_name = os.path.basename(data_path).replace("-", "_").replace(" ", "_").lower()
    if not ds_name:
        ds_name = "custom_dataset"

    target_dir = os.path.join(DATA_DIR, ds_name)
    if os.path.abspath(data_path) != os.path.abspath(target_dir):
        if not os.path.isdir(target_dir):
            os.makedirs(DATA_DIR, exist_ok=True)
            try:
                os.symlink(os.path.abspath(data_path), target_dir)
                print(f"  Linked {data_path} -> {target_dir}")
            except (OSError, NotImplementedError):
                target_dir = data_path

    data_dir = target_dir

    print(f"\n{'='*60}")
    print(f"Auto-detecting dataset: {ds_name}")
    print(f"{'='*60}")
    print(f"Path: {data_path}\n")

    print("Scanning directory structure...")
    scan_result = scan_directory(data_dir)
    print(f"  Images found:  {scan_result['image_count']:,}")
    print(f"  CSV files:     {len(scan_result['csv_files'])}")
    print(f"  Class dirs:    {len(scan_result['class_dirs'])}")
    print(f"  Mask dirs:     {len(scan_result['mask_dirs'])}")
    print(f"  Pre-split:     {'yes' if scan_result['has_splits'] else 'no'}")

    task = task_override or detect_task(scan_result)
    print(f"\nDetected task: {task}")

    label_info = detect_label_info(scan_result, data_dir)
    print(f"Num classes:   {label_info['num_classes']}")
    if label_info["class_names"]:
        names_str = ", ".join(label_info["class_names"][:10])
        if len(label_info["class_names"]) > 10:
            names_str += f"... (+{len(label_info['class_names'])-10} more)"
        print(f"Classes:       {names_str}")

    in_channels = detect_channels(data_dir, scan_result)
    print(f"Channels:      {in_channels}")

    if compute_stats:
        print("\nComputing dataset statistics...")
        mean, std = compute_dataset_statistics(data_dir, scan_result, in_channels)
        print(f"  Mean: {mean}")
        print(f"  Std:  {std}")
    else:
        mean = (0.485,) if in_channels == 1 else (0.485, 0.456, 0.406)
        std = (0.229,) if in_channels == 1 else (0.229, 0.224, 0.225)

    if config_overrides:
        if isinstance(config_overrides, str):
            try:
                import yaml
                with open(config_overrides) as f:
                    config_overrides = yaml.safe_load(f)
            except ImportError:
                config_overrides = {}
                print("  WARNING: PyYAML not installed, ignoring config file")
        if isinstance(config_overrides, dict):
            task = config_overrides.get("task", task)
            label_info["num_classes"] = config_overrides.get("num_classes", label_info["num_classes"])
            if "class_names" in config_overrides:
                label_info["class_names"] = config_overrides["class_names"]
            if "image_column" in config_overrides:
                label_info["image_column"] = config_overrides["image_column"]
            if "label_column" in config_overrides:
                label_info["label_column"] = config_overrides["label_column"]
            if "patient_column" in config_overrides:
                patient_column = config_overrides["patient_column"]
            in_channels = config_overrides.get("in_channels", in_channels)

    split_file = os.path.join(data_dir, "splits.csv")
    if os.path.exists(split_file):
        print(f"\n  splits.csv already exists -- skipping split creation")
        print(f"  Delete {split_file} to regenerate")
    else:
        print(f"\nCreating train/val splits...")
        patient_col = patient_column or label_info.get("patient_column")
        if task == "segmentation":
            _prepare_generic_segmentation(data_dir, scan_result, split_file,
                                          patient_column=patient_col)
        elif scan_result["class_dirs"] or scan_result["has_splits"]:
            _prepare_generic_classification(data_dir, scan_result, label_info,
                                             split_file, patient_column=patient_col)
        elif scan_result["csv_files"]:
            _prepare_generic_csv(data_dir, scan_result, label_info, task,
                                  split_file, patient_column=patient_col)
        else:
            print("  WARNING: Could not determine how to create splits.")
            return None

    metric = _auto_metric(task, label_info["num_classes"])
    if config_overrides and isinstance(config_overrides, dict):
        metric = config_overrides.get("metric", metric)

    manifest = {
        "name": ds_name, "task": task,
        "num_classes": label_info["num_classes"],
        "class_names": label_info["class_names"],
        "image_size": 256 if task == "segmentation" else 224,
        "in_channels": in_channels,
        "mean": list(mean), "std": list(std),
        "metric": metric,
        "description": f"Auto-detected from {data_path}",
    }
    manifest_path = os.path.join(data_dir, "dataset.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        _json.dump(manifest, f, indent=2)

    print(f"\nWrote {manifest_path}")
    print(f"\n{'='*60}")
    print(f"Ready! Run: uv run train.py --dataset {ds_name}")
    print(f"{'='*60}\n")
    return ds_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def list_datasets():
    """Print all available datasets."""
    datasets, errors = get_all_datasets(include_errors=True)
    builtins = {name: cfg for name, cfg in datasets.items() if cfg["source"] == "builtin"}
    custom = {name: cfg for name, cfg in datasets.items() if cfg["source"] == "custom"}

    print("\nAvailable medical imaging datasets:\n")
    print(f"{'Name':<18} {'Source':<8} {'Task':<16} {'Classes':<8} {'Metric':<18} Description")
    print("-" * 120)
    for name, cfg in builtins.items():
        print(
            f"{name:<18} {'built-in':<8} {cfg['task']:<16} {cfg['num_classes']:<8} "
            f"{cfg['metric']:<18} {cfg['description']}"
        )
    for name, cfg in custom.items():
        print(
            f"{name:<18} {'custom':<8} {cfg['task']:<16} {cfg['num_classes']:<8} "
            f"{cfg['metric']:<18} {cfg['description']}"
        )
    if errors:
        print("\nCustom dataset manifests with errors:")
        for name, message in sorted(errors.items()):
            print(f"  {name}: {message}")
    print()
    print("Built-in examples: aptos, oct, chestxray14, chest_xray, brain_mri, retina_vessel, pneumothorax")
    if custom:
        print("Custom datasets are discovered from data/<dataset>/dataset.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare built-in Kaggle datasets or validate local custom datasets"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Built-in dataset to download or custom dataset to validate",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument(
        "--refresh-splits",
        action="store_true",
        help="Regenerate splits.csv for a built-in dataset using the current preparation logic",
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Path to your own dataset directory (auto-detect)")
    parser.add_argument("--task", type=str, default=None,
                        choices=["classification", "multilabel", "segmentation"],
                        help="Override auto-detected task type")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file with dataset overrides")
    parser.add_argument("--patient-column", type=str, default=None,
                        help="CSV column name for patient-level splitting")
    parser.add_argument("--compute-stats", action="store_true",
                        help="Compute dataset-specific mean/std (default: ImageNet stats)")
    args = parser.parse_args()

    if args.list or (args.dataset is None and args.data is None):
        list_datasets()
        if args.dataset is None:
            print("\nUsage: uv run prepare.py --dataset <name>")
        sys.exit(0)

    if args.data:
        if not os.path.isdir(args.data):
            print(f"ERROR: directory not found: {args.data}")
            sys.exit(1)
        ds_name = prepare_custom_dataset(
            args.data, task_override=args.task,
            config_overrides=args.config,
            patient_column=getattr(args, 'patient_column', None),
            compute_stats=args.compute_stats,
        )
        if ds_name:
            print("Done! Ready to train.")
        sys.exit(0)

    print(f"Data directory: {DATA_DIR}")
    print()

    if args.dataset == "all":
        for ds_name in DATASETS:
            download_dataset(ds_name, refresh_splits=args.refresh_splits)
            print()
    else:
        download_dataset(args.dataset, refresh_splits=args.refresh_splits)

    print()
    print("Done! Ready to train.")
