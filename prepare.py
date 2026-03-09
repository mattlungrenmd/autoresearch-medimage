"""
Data preparation and evaluation for autoresearch-medvision experiments.

Supports two modes:
  1. Built-in presets — download 7 curated medical imaging datasets from Kaggle
  2. Bring-your-own-data — auto-detect ANY medical imaging dataset structure

This file is READ-ONLY for agents during experiments. Do not modify.

Usage:
    # Built-in datasets (Kaggle download)
    python prepare.py --dataset aptos            # Diabetic Retinopathy grading
    python prepare.py --dataset chestxray14      # CheXNet multi-label (14 pathologies)
    python prepare.py --dataset brain_mri        # Brain MRI segmentation
    python prepare.py --dataset all              # Download all 7 datasets
    python prepare.py --list                     # List all available datasets

    # Bring your own data (auto-detect)
    python prepare.py --data /path/to/images                  # Auto-detect everything
    python prepare.py --data /path --task segmentation        # Override task type
    python prepare.py --data /path --patient-column PatientID # Patient-level splits
    python prepare.py --data /path --compute-stats            # Compute mean/std

    # Download from Kaggle + auto-detect
    python prepare.py --kaggle username/dataset-slug

Requires: ~/.kaggle/kaggle.json with valid API credentials (for Kaggle downloads).
Datasets are stored in ./data/<dataset_name>/
Each custom dataset gets a dataset.json manifest for the dynamic registry.
"""

import os
import sys
import json
import time
import math
import shutil
import zipfile
import argparse
import hashlib
import glob as glob_module
import re
from pathlib import Path
from collections import Counter

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

# ---------------------------------------------------------------------------
# Dataset registry — built-in presets (7 demo datasets)
# ---------------------------------------------------------------------------

BUILTIN_DATASETS = {
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
        "kaggle_slug": "nih-chest-xrays/data",
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
        "metric": "auc",
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

# ---------------------------------------------------------------------------
# Image file extensions recognized by auto-detection
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
DICOM_EXTENSIONS = {".dcm", ".dicom"}

# ---------------------------------------------------------------------------
# Dynamic dataset registry — merges built-ins + custom dataset.json files
# ---------------------------------------------------------------------------

def build_registry():
    """Build DATASETS by merging BUILTIN_DATASETS with any data/<name>/dataset.json found on disk.

    Custom datasets discovered via dataset.json are added transparently so
    train.py can access them via ``DATASETS[name]`` without code changes.
    """
    registry = dict(BUILTIN_DATASETS)

    if not os.path.isdir(DATA_DIR):
        return registry

    for entry in sorted(os.listdir(DATA_DIR)):
        entry_dir = os.path.join(DATA_DIR, entry)
        manifest = os.path.join(entry_dir, "dataset.json")
        if not os.path.isfile(manifest):
            continue
        try:
            with open(manifest, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # Validate required keys
            name = cfg.get("name", entry)
            required = ("task", "num_classes")
            if not all(k in cfg for k in required):
                print(f"  WARNING: skipping {manifest} — missing keys: "
                      f"{[k for k in required if k not in cfg]}")
                continue
            # Normalize tuples stored as lists
            for key in ("mean", "std"):
                if key in cfg and isinstance(cfg[key], list):
                    cfg[key] = tuple(cfg[key])
            # Defaults
            cfg.setdefault("class_names", [f"class_{i}" for i in range(cfg["num_classes"])])
            cfg.setdefault("image_size", 256 if cfg["task"] == "segmentation" else 224)
            cfg.setdefault("in_channels", 3)
            cfg.setdefault("mean", (0.485, 0.456, 0.406) if cfg.get("in_channels", 3) == 3 else (0.485,))
            cfg.setdefault("std", (0.229, 0.224, 0.225) if cfg.get("in_channels", 3) == 3 else (0.229,))
            cfg.setdefault("metric", _auto_metric(cfg["task"], cfg["num_classes"]))
            cfg.setdefault("description", f"Custom dataset: {name}")
            registry[name] = cfg
        except Exception as e:
            print(f"  WARNING: failed to load {manifest}: {e}")

    return registry


def _auto_metric(task, num_classes):
    """Choose a default metric based on task type and class count."""
    if task == "segmentation":
        return "dice"
    elif task == "multilabel":
        return "auc"
    elif task == "classification" and num_classes == 2:
        return "auc"
    elif task == "ordinal":
        return "quadratic_kappa"
    return "accuracy"


DATASETS = build_registry()


# ---------------------------------------------------------------------------
# Auto-detection pipeline: bring-your-own-data
# ---------------------------------------------------------------------------

def scan_directory(data_dir):
    """Scan a data directory and detect its structure.

    Returns a dict describing what was found: CSV files, image directories,
    class-folder structures, mask directories, and DICOM files.
    """
    result = {
        "csv_files": [],
        "image_dirs": [],
        "image_count": 0,
        "mask_dirs": [],
        "class_dirs": [],
        "dicom_files": [],
        "has_splits": False,
        "total_files": 0,
    }

    for root, dirs, files in os.walk(data_dir):
        # Detect pre-existing train/val/test splits
        dir_names_lower = [d.lower() for d in dirs]
        if "train" in dir_names_lower or "val" in dir_names_lower:
            result["has_splits"] = True

        # Count images
        images = [f for f in files if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
        if images:
            result["image_dirs"].append({"path": root, "count": len(images)})
            result["image_count"] += len(images)

        # Detect DICOM
        dicoms = [f for f in files if Path(f).suffix.lower() in DICOM_EXTENSIONS]
        if dicoms:
            result["dicom_files"].extend(
                os.path.join(root, f) for f in dicoms
            )

        # Find CSVs
        csvs = [f for f in files if f.lower().endswith(".csv")]
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
            # Exclude split-level directories (train/val/test/images/masks)
            if not any(d.lower() in ("train", "test", "val", "validation",
                                      "images", "masks", "labels", "annotations")
                       for d in subdirs):
                result["class_dirs"].append({
                    "path": root,
                    "classes": sorted(subdirs),
                    "num_classes": len(subdirs),
                })

        # Detect mask directories
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
    has_masks = len(scan_result["mask_dirs"]) > 0
    has_class_dirs = len(scan_result["class_dirs"]) > 0
    has_csvs = len(scan_result["csv_files"]) > 0

    # Segmentation: has mask directories
    if has_masks:
        return "segmentation"

    # Classification with folder structure
    if has_class_dirs:
        return "classification"

    # Check CSVs for multi-label indicators
    if has_csvs:
        for csv_info in scan_result["csv_files"]:
            cols = [c.lower() for c in csv_info["columns"]]
            # Pipe-separated column names → multilabel
            if any("|" in str(c) for c in cols):
                return "multilabel"
            # Read a sample to check for pipe-separated label values
            label_cols = [c for c in cols if any(kw in c for kw in
                         ["label", "target", "class", "diagnosis",
                          "finding", "category"])]
            if label_cols:
                try:
                    df = pd.read_csv(csv_info["path"], nrows=100)
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
    """Detect label columns, class names, and image column from the scan."""
    info = {
        "num_classes": 2,
        "class_names": [],
        "label_column": None,
        "image_column": None,
        "patient_column": None,
    }

    # From folder structure
    if scan_result["class_dirs"]:
        class_info = scan_result["class_dirs"][0]
        info["num_classes"] = class_info["num_classes"]
        info["class_names"] = class_info["classes"]
        return info

    # From CSV
    if scan_result["csv_files"]:
        csv_info = scan_result["csv_files"][0]
        csv_path = csv_info["path"]
        try:
            df = pd.read_csv(csv_path, nrows=1000)
        except Exception:
            return info

        cols = list(df.columns)
        cols_lower = [c.lower() for c in cols]

        # Detect image column
        for pattern in ["image", "file", "path", "filename", "id_code"]:
            matches = [c for c, cl in zip(cols, cols_lower) if pattern in cl]
            if matches:
                info["image_column"] = matches[0]
                break

        # Detect label column
        for pattern in ["label", "target", "class", "diagnosis",
                        "finding", "category", "grade", "level"]:
            matches = [c for c, cl in zip(cols, cols_lower)
                       if pattern in cl and "image" not in cl]
            if matches:
                info["label_column"] = matches[0]
                break

        # Detect patient column (for patient-level splitting)
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
    return 3  # default


def compute_dataset_statistics(data_dir, scan_result, in_channels=3, max_samples=1000):
    """Compute per-channel mean and std using Welford's online algorithm.

    Samples up to max_samples images for efficiency.
    Returns (mean_tuple, std_tuple).
    """
    # Collect image paths
    all_images = []
    for img_dir_info in scan_result["image_dirs"]:
        for f in os.listdir(img_dir_info["path"]):
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                all_images.append(os.path.join(img_dir_info["path"], f))

    if not all_images:
        if in_channels == 1:
            return (0.485,), (0.229,)
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    # Sample
    rng = np.random.RandomState(RANDOM_SEED)
    sample = rng.choice(all_images, size=min(max_samples, len(all_images)), replace=False)

    # Welford's online algorithm per channel
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
    std = np.maximum(std, 1e-6)  # avoid zero std

    return tuple(float(m) for m in mean), tuple(float(s) for s in std)


def _detect_patient_ids_from_paths(image_paths):
    """Try to extract patient IDs from image file paths.

    Heuristic: look for common patterns like 'patient_001/scan.png' or
    'TCGA_AB_1234_slice01.png' where the first few path components or
    filename prefix encodes the patient.
    """
    # Strategy 1: directory-based (images grouped by patient folders)
    parents = [os.path.basename(os.path.dirname(p)) for p in image_paths]
    unique_parents = set(parents)
    # If there are many fewer parents than images, they're probably patients
    if 1 < len(unique_parents) < len(image_paths) * 0.5:
        return parents

    # Strategy 2: filename prefix before last underscore or hyphen
    # e.g., "patient001_slice01.png" -> "patient001"
    stems = [Path(p).stem for p in image_paths]
    prefixes = []
    for s in stems:
        # Try to split on last underscore/hyphen
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

    return None  # Cannot determine patient IDs


def _patient_level_split(df, patient_col=None, val_ratio=VAL_RATIO, seed=RANDOM_SEED):
    """Split a DataFrame using patient-level grouping to prevent leakage.

    If patient_col is provided, use that column for grouping.
    Otherwise, try to auto-detect from image paths.
    """
    if patient_col and patient_col in df.columns:
        groups = df[patient_col].values
    elif "image_path" in df.columns:
        groups = _detect_patient_ids_from_paths(df["image_path"].tolist())
        if groups is None:
            # Fall back to stratified split
            return _stratified_split(df, val_ratio, seed)
        groups = np.array(groups)
    else:
        return _stratified_split(df, val_ratio, seed)

    # Use GroupShuffleSplit for patient-level splitting
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    stratify_col = "label" if "label" in df.columns else None

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


def _stratified_split(df, val_ratio=VAL_RATIO, seed=RANDOM_SEED):
    """Standard stratified train/val split."""
    df = df.copy()
    if "label" in df.columns and df["label"].nunique() >= 2:
        skf = StratifiedKFold(
            n_splits=max(2, int(1 / val_ratio)),
            shuffle=True,
            random_state=seed,
        )
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


def _prepare_generic_classification(data_dir, scan_result, label_info, split_file,
                                     patient_column=None):
    """Generic classification split from folder-of-folders structure."""
    rows = []

    if scan_result["has_splits"]:
        # Honor existing train/val/test splits
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
                        rows.append({
                            "image_path": os.path.join(cls_dir, img),
                            "label": cls_idx,
                            "split": mapped_split,
                        })

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(split_file, index=False)
            n_train = (df["split"] == "train").sum()
            n_val = (df["split"] == "val").sum()
            print(f"  Classification (pre-split): {n_train} train, {n_val} val, "
                  f"{len(set(df['label']))} classes")
            return

    # No pre-existing splits: single class-folder structure
    if scan_result["class_dirs"]:
        class_info = scan_result["class_dirs"][0]
        for cls_idx, cls_name in enumerate(class_info["classes"]):
            cls_dir = os.path.join(class_info["path"], cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img in os.listdir(cls_dir):
                if Path(img).suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append({
                        "image_path": os.path.join(cls_dir, img),
                        "label": cls_idx,
                    })

    if not rows:
        print("  WARNING: No images found for classification")
        return

    df = pd.DataFrame(rows)
    df = _patient_level_split(df, patient_column)
    df.to_csv(split_file, index=False)
    n_train = (df["split"] == "train").sum()
    n_val = (df["split"] == "val").sum()
    print(f"  Classification: {n_train} train, {n_val} val, "
          f"{len(set(df['label']))} classes")


def _prepare_generic_csv(data_dir, scan_result, label_info, task, split_file,
                          patient_column=None):
    """Generic split from CSV + images directory."""
    csv_info = scan_result["csv_files"][0]
    df = pd.read_csv(csv_info["path"])

    img_col = label_info.get("image_column")
    label_col = label_info.get("label_column")

    if not img_col or not label_col:
        print(f"  WARNING: Could not detect image/label columns in {csv_info['name']}")
        print(f"  Columns found: {csv_info['columns']}")
        print("  Hint: use --config with image_column and label_column overrides")
        return

    # Find image directory
    img_dir = None
    for candidate in ["images", "train_images", "train", "imgs"]:
        img_dir = _find_dir(data_dir, candidate)
        if img_dir:
            break
    if img_dir is None:
        # Use directory with most images
        if scan_result["image_dirs"]:
            img_dir = max(scan_result["image_dirs"], key=lambda x: x["count"])["path"]
        else:
            img_dir = data_dir

    rows = []
    for _, row in df.iterrows():
        img_name = str(row[img_col])
        label = row[label_col]

        if task == "multilabel":
            # Multi-label: pipe-separated values
            label_str = str(label)
            rows.append({"image_path": "", "labels": label_str, "label": 0})
        else:
            # Classification: integer or string label
            if isinstance(label, str):
                # Map string labels to indices
                all_labels = sorted(df[label_col].unique(), key=str)
                label = all_labels.index(label)
            rows.append({"image_path": "", "label": int(label)})

        # Resolve image path
        resolved = None
        for ext in ["", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            candidate = os.path.join(img_dir, img_name + ext)
            if os.path.exists(candidate):
                resolved = candidate
                break
        if resolved is None:
            matches = list(Path(img_dir).glob(f"{img_name}.*"))
            if matches:
                resolved = str(matches[0])
        if resolved:
            rows[-1]["image_path"] = resolved
        else:
            rows.pop()  # Remove row if image not found

    if not rows:
        print("  WARNING: No images matched from CSV")
        return

    result_df = pd.DataFrame(rows)

    # Add patient column if present in source CSV
    patient_col = patient_column or label_info.get("patient_column")
    if patient_col and patient_col in df.columns:
        # Re-read to get patient column, matching indices
        matched = df[df[img_col].apply(
            lambda x: any(os.path.exists(os.path.join(img_dir, str(x) + ext))
                         for ext in ["", ".png", ".jpg", ".jpeg"])
        )].reset_index(drop=True)
        if len(matched) == len(result_df) and patient_col in matched.columns:
            result_df[patient_col] = matched[patient_col].values

    result_df = _patient_level_split(result_df, patient_col)
    result_df.to_csv(split_file, index=False)
    n_train = (result_df["split"] == "train").sum()
    n_val = (result_df["split"] == "val").sum()
    print(f"  CSV-based ({task}): {n_train} train, {n_val} val")


def _prepare_generic_segmentation(data_dir, scan_result, split_file,
                                    patient_column=None):
    """Generic segmentation split from images + masks directories."""
    rows = []

    # Find image and mask directories
    img_dir = None
    mask_dir = None

    # Look for paired image/mask directories
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
                rows.append({
                    "image_path": os.path.join(img_dir, img_name),
                    "mask_path": os.path.join(mask_dir, mask_match),
                    "label": 0,
                })

    # Fallback: walk directory for co-located image/mask pairs
    if not rows:
        for root, dirs, files in os.walk(data_dir):
            images = sorted([f for f in files
                            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
                            and "mask" not in f.lower() and "label" not in f.lower()])
            masks = sorted([f for f in files
                           if Path(f).suffix.lower() in IMAGE_EXTENSIONS
                           and ("mask" in f.lower() or "label" in f.lower())])
            for img, msk in zip(images, masks):
                rows.append({
                    "image_path": os.path.join(root, img),
                    "mask_path": os.path.join(root, msk),
                    "label": 0,
                })

    if not rows:
        print("  WARNING: No image/mask pairs found for segmentation")
        return

    df = pd.DataFrame(rows)
    df = _patient_level_split(df, patient_column)
    df.to_csv(split_file, index=False)
    n_train = (df["split"] == "train").sum()
    n_val = (df["split"] == "val").sum()
    print(f"  Segmentation: {n_train} train, {n_val} val")


def _convert_dicoms(data_dir):
    """Auto-convert any DICOM files in data_dir to PNG.

    Returns path to the converted PNG directory (or original data_dir if
    no DICOM files found).
    """
    dcm_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if Path(f).suffix.lower() in DICOM_EXTENSIONS:
                dcm_files.append(os.path.join(root, f))

    if not dcm_files:
        return data_dir

    print(f"  Found {len(dcm_files)} DICOM files — converting to PNG...")
    png_dir = os.path.join(data_dir, "converted_png")
    os.makedirs(png_dir, exist_ok=True)

    try:
        import pydicom
    except ImportError:
        print("  WARNING: pydicom not installed. Install with: uv pip install pydicom")
        print("  Skipping DICOM conversion.")
        return data_dir

    patient_ids = {}
    for i, dcm_path in enumerate(dcm_files):
        try:
            ds = pydicom.dcmread(dcm_path)
            pixel_array = ds.pixel_array.astype(np.float64)

            slope = getattr(ds, "RescaleSlope", 1)
            intercept = getattr(ds, "RescaleIntercept", 0)
            pixel_array = pixel_array * slope + intercept

            # Windowing
            if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
                wc = ds.WindowCenter
                ww = ds.WindowWidth
                if isinstance(wc, (list, pydicom.multival.MultiValue)):
                    wc = wc[0]
                if isinstance(ww, (list, pydicom.multival.MultiValue)):
                    ww = ww[0]
                wc, ww = float(wc), float(ww)
            else:
                wc = (pixel_array.max() + pixel_array.min()) / 2
                ww = max(pixel_array.max() - pixel_array.min(), 1)

            lower = wc - ww / 2
            upper = wc + ww / 2
            img_array = np.clip(pixel_array, lower, upper)
            img_array = ((img_array - lower) / max(upper - lower, 1) * 255).astype(np.uint8)

            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                img_array = 255 - img_array

            png_name = Path(dcm_path).stem + ".png"
            png_path = os.path.join(png_dir, png_name)
            Image.fromarray(img_array).save(png_path)

            pid = getattr(ds, "PatientID", "unknown")
            patient_ids[png_name] = pid

            if (i + 1) % 100 == 0:
                print(f"    Converted {i+1}/{len(dcm_files)} DICOM files...")
        except Exception as e:
            if i < 3:
                print(f"    WARNING: skipping {Path(dcm_path).name}: {e}")

    # Save patient mapping if detected
    if patient_ids:
        mapping_path = os.path.join(data_dir, "dicom_patient_map.json")
        with open(mapping_path, "w") as f:
            json.dump(patient_ids, f, indent=2)

    print(f"  Converted {len(os.listdir(png_dir))} DICOM → PNG in {png_dir}")
    return data_dir


def prepare_custom_dataset(data_path, task_override=None, config_overrides=None,
                            patient_column=None, compute_stats=False):
    """Auto-detect and prepare ANY medical imaging dataset.

    This is the main entry point for bring-your-own-data. It:
    1. Detects DICOM files and converts them to PNG
    2. Scans directory structure to detect task type
    3. Infers metadata (num_classes, class_names, channels)
    4. Creates train/val splits (patient-level if possible)
    5. Writes dataset.json manifest for the dynamic registry

    Args:
        data_path: Path to the dataset directory
        task_override: Optional task type override ('classification', 'multilabel', 'segmentation')
        config_overrides: Optional dict or YAML path with overrides
        patient_column: CSV column name for patient-level splitting
        compute_stats: Whether to compute per-channel mean/std (default: use ImageNet stats)

    Returns:
        Dataset name (str) for use with train.py
    """
    data_path = os.path.abspath(data_path)

    # Determine dataset name from path
    ds_name = os.path.basename(data_path).replace("-", "_").replace(" ", "_").lower()
    if not ds_name:
        ds_name = "custom_dataset"

    # Ensure data lives under DATA_DIR for consistent access
    target_dir = os.path.join(DATA_DIR, ds_name)
    if os.path.abspath(data_path) != os.path.abspath(target_dir):
        if not os.path.isdir(target_dir):
            # Symlink or copy into data/
            os.makedirs(DATA_DIR, exist_ok=True)
            try:
                os.symlink(os.path.abspath(data_path), target_dir)
                print(f"  Linked {data_path} → {target_dir}")
            except (OSError, NotImplementedError):
                # Windows or no symlink support: just use original path
                target_dir = data_path
    else:
        pass  # Already under data/

    data_dir = target_dir

    print(f"\n{'='*60}")
    print(f"Auto-detecting dataset: {ds_name}")
    print(f"{'='*60}")
    print(f"Path: {data_path}\n")

    # Step 1: DICOM conversion
    _convert_dicoms(data_dir)

    # Step 2: Scan directory structure
    print("Scanning directory structure...")
    scan_result = scan_directory(data_dir)
    print(f"  Images found:  {scan_result['image_count']:,}")
    print(f"  CSV files:     {len(scan_result['csv_files'])}")
    print(f"  Class dirs:    {len(scan_result['class_dirs'])}")
    print(f"  Mask dirs:     {len(scan_result['mask_dirs'])}")
    print(f"  DICOM files:   {len(scan_result['dicom_files'])}")
    print(f"  Pre-split:     {'yes' if scan_result['has_splits'] else 'no'}")

    # Step 3: Detect task
    task = task_override or detect_task(scan_result)
    print(f"\nDetected task: {task}")

    # Step 4: Detect label info
    label_info = detect_label_info(scan_result, data_dir)
    print(f"Num classes:   {label_info['num_classes']}")
    if label_info["class_names"]:
        names_str = ", ".join(label_info["class_names"][:10])
        if len(label_info["class_names"]) > 10:
            names_str += f"... (+{len(label_info['class_names'])-10} more)"
        print(f"Classes:       {names_str}")
    if label_info.get("label_column"):
        print(f"Label column:  {label_info['label_column']}")
    if label_info.get("image_column"):
        print(f"Image column:  {label_info['image_column']}")
    if label_info.get("patient_column"):
        print(f"Patient col:   {label_info['patient_column']}")

    # Step 5: Detect channels
    in_channels = detect_channels(data_dir, scan_result)
    print(f"Channels:      {in_channels}")

    # Step 6: Compute or use default statistics
    if compute_stats:
        print("\nComputing dataset statistics...")
        mean, std = compute_dataset_statistics(data_dir, scan_result, in_channels)
        print(f"  Mean: {mean}")
        print(f"  Std:  {std}")
    else:
        mean = (0.485,) if in_channels == 1 else (0.485, 0.456, 0.406)
        std = (0.229,) if in_channels == 1 else (0.229, 0.224, 0.225)

    # Apply config overrides
    if config_overrides:
        if isinstance(config_overrides, str):
            # Load from YAML file
            try:
                import yaml
                with open(config_overrides) as f:
                    config_overrides = yaml.safe_load(f)
            except ImportError:
                # Parse simple key=value from file
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
            if "metric" in config_overrides:
                pass  # handled below
            in_channels = config_overrides.get("in_channels", in_channels)

    # Step 7: Create splits
    split_file = os.path.join(data_dir, "splits.csv")
    if os.path.exists(split_file):
        print(f"\n  splits.csv already exists — skipping split creation")
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
            print("  Provide a CSV with image paths and labels, or organize images into class folders.")
            return None

    # Step 8: Write dataset.json manifest
    metric = _auto_metric(task, label_info["num_classes"])
    if config_overrides and isinstance(config_overrides, dict):
        metric = config_overrides.get("metric", metric)

    manifest = {
        "name": ds_name,
        "task": task,
        "num_classes": label_info["num_classes"],
        "class_names": label_info["class_names"],
        "image_size": 256 if task == "segmentation" else 224,
        "in_channels": in_channels,
        "mean": list(mean),
        "std": list(std),
        "metric": metric,
        "description": f"Auto-detected from {data_path}",
    }

    manifest_path = os.path.join(data_dir, "dataset.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {manifest_path}")
    print(f"\n{'='*60}")
    print(f"Ready! Run: uv run train.py --dataset {ds_name}")
    print(f"{'='*60}\n")

    # Refresh the global registry
    global DATASETS
    DATASETS = build_registry()

    return ds_name


def download_kaggle_auto(slug):
    """Download a Kaggle dataset and auto-detect its structure.

    Combines Kaggle download with the auto-detection pipeline.
    """
    check_kaggle_credentials()

    # Determine destination
    ds_name = slug.replace("/", "_").replace("-", "_").lower()
    dest_dir = os.path.join(DATA_DIR, ds_name)

    if not os.path.exists(os.path.join(dest_dir, ".download_complete")):
        print(f"Downloading from Kaggle: {slug}")
        download_kaggle_dataset(slug, dest_dir)
    else:
        print(f"Already downloaded: {slug}")

    # Now auto-detect
    return prepare_custom_dataset(dest_dir)


# ---------------------------------------------------------------------------
# Kaggle download automation
# ---------------------------------------------------------------------------

def check_kaggle_credentials():
    """Verify Kaggle API credentials exist."""
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print("ERROR: Kaggle API credentials not found.")
        print("To set up:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New Token' under API section")
        print("  3. Place the downloaded kaggle.json at ~/.kaggle/kaggle.json")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)
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


def download_dataset(dataset_name):
    """Download a specific medical imaging dataset from Kaggle."""
    check_kaggle_credentials()

    if dataset_name not in DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {list(DATASETS.keys())}")
        sys.exit(1)

    cfg = DATASETS[dataset_name]
    dest_dir = os.path.join(DATA_DIR, dataset_name)
    print(f"Downloading {dataset_name}: {cfg['description']}")
    download_kaggle_dataset(cfg["kaggle_slug"], dest_dir)

    # Post-download: create train/val splits if needed
    _prepare_splits(dataset_name, dest_dir)


def _prepare_splits(dataset_name, dest_dir):
    """Create reproducible train/val CSV splits for each dataset."""
    split_file = os.path.join(dest_dir, "splits.csv")
    if os.path.exists(split_file):
        return

    print(f"  Creating train/val splits for {dataset_name}...")

    # Built-in datasets have specialized preparation functions
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
    else:
        # Custom dataset: use generic auto-detection
        prepare_custom_dataset(dest_dir)


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

    if csv_path is None:
        print("  WARNING: Could not find ChestX-ray14 labels CSV")
        for root, dirs, files in os.walk(dest_dir):
            for f in files[:20]:
                print(f"    {os.path.join(root, f)}")
        return

    class_names = DATASETS["chestxray14"]["class_names"]
    df = pd.read_csv(csv_path)

    # Find image column (usually "Image Index")
    img_col = [c for c in df.columns if "image" in c.lower() and "index" in c.lower()]
    if not img_col:
        img_col = [c for c in df.columns if "image" in c.lower()]
    img_col = img_col[0] if img_col else df.columns[0]

    # Find label column (usually "Finding Labels")
    label_col = [c for c in df.columns if "finding" in c.lower() or "label" in c.lower()]
    label_col = label_col[0] if label_col else df.columns[1]

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

    # Use official train/val/test split if available
    train_list_path = _find_file(dest_dir, "train_val_list.txt")
    test_list_path = _find_file(dest_dir, "test_list.txt")

    if train_list_path and test_list_path:
        with open(train_list_path) as f:
            train_files = set(line.strip() for line in f)
        with open(test_list_path) as f:
            test_files = set(line.strip() for line in f)
    else:
        train_files = None
        test_files = None

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
            split = "val" if img_name in test_files else "train"
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
    print(f"  ChestX-ray14: {n_train} train, {n_val} val, {len(class_names)} pathologies")


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
    if dataset_name not in DATASETS:
        available = sorted(DATASETS.keys())
        raise ValueError(
            f"Unknown dataset: {dataset_name}\n"
            f"Available: {available}\n"
            f"To add your own: uv run prepare.py --data /path/to/your/images"
        )
    return DATASETS[dataset_name]


class MedicalClassificationDataset(Dataset):
    """Generic medical image classification dataset loaded from splits.csv."""

    def __init__(self, dataset_name, split="train", transform=None):
        cfg = DATASETS[dataset_name]
        data_dir = os.path.join(DATA_DIR, dataset_name)
        split_file = os.path.join(data_dir, "splits.csv")

        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"splits.csv not found for {dataset_name}. Run: python prepare.py --dataset {dataset_name}"
            )

        df = pd.read_csv(split_file)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.in_channels = cfg["in_channels"]
        self.num_classes = cfg["num_classes"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
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
        cfg = DATASETS[dataset_name]
        data_dir = os.path.join(DATA_DIR, dataset_name)
        split_file = os.path.join(data_dir, "splits.csv")

        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"splits.csv not found for {dataset_name}. Run: python prepare.py --dataset {dataset_name}"
            )

        df = pd.read_csv(split_file)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.in_channels = cfg["in_channels"]
        self.num_classes = cfg["num_classes"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

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
        cfg = DATASETS[dataset_name]
        data_dir = os.path.join(DATA_DIR, dataset_name)
        split_file = os.path.join(data_dir, "splits.csv")

        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"splits.csv not found for {dataset_name}. Run: python prepare.py --dataset {dataset_name}"
            )

        df = pd.read_csv(split_file)
        self.df = df[df["split"] == split].reset_index(drop=True)
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
        img_path = row["image_path"]
        mask_path = row["mask_path"]

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

def make_classification_transforms(dataset_name, train=True, image_size=None, augmentation_config=None):
    """Build transforms for medical classification."""
    cfg = DATASETS[dataset_name]
    img_size = image_size or cfg["image_size"]
    augmentation_config = augmentation_config or {}

    if cfg["in_channels"] == 1:
        mean, std = cfg["mean"], cfg["std"]
    else:
        mean, std = cfg["mean"], cfg["std"]

    horizontal_flip = augmentation_config.get("horizontal_flip", True)
    vertical_flip = augmentation_config.get("vertical_flip", True)
    rotation = augmentation_config.get("rotation", 15)
    use_clahe = augmentation_config.get("clahe", False)
    use_color_jitter = augmentation_config.get("color_jitter", cfg["in_channels"] == 3)
    brightness = augmentation_config.get("color_jitter_brightness", 0.1)
    contrast = augmentation_config.get("color_jitter_contrast", 0.1)
    use_translation = augmentation_config.get("translate", not augmentation_config)

    transform_list = [T.Resize((img_size, img_size))]

    if train:
        if horizontal_flip:
            transform_list.append(T.RandomHorizontalFlip())
        if vertical_flip:
            transform_list.append(T.RandomVerticalFlip())
        if rotation:
            transform_list.append(T.RandomRotation(rotation))
        if cfg["in_channels"] == 3 and use_color_jitter:
            transform_list.append(T.ColorJitter(brightness=brightness, contrast=contrast))
        elif cfg["in_channels"] == 1 and use_translation:
            transform_list.append(T.RandomAffine(degrees=0, translate=(0.05, 0.05)))

    if use_clahe:
        transform_list.append(T.Lambda(_apply_clahe_pil))

    transform_list.extend([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # For grayscale: expand to 3 channels for pretrained models
    if cfg["in_channels"] == 1:
        transform_list.append(T.Lambda(lambda x: x.repeat(3, 1, 1)))

    return T.Compose(transform_list)


def make_dataloader(dataset_name, batch_size, split="train", image_size=None, augmentation_config=None):
    """Create a dataloader for the specified dataset and split.

    Returns:
        dataloader, num_classes
    """
    cfg = DATASETS[dataset_name]
    is_train = (split == "train")
    image_size = image_size or cfg["image_size"]

    if cfg["task"] in ("classification", "multilabel"):
        transforms = make_classification_transforms(
            dataset_name,
            train=is_train,
            image_size=image_size,
            augmentation_config=augmentation_config,
        )
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
            image_size=image_size,
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


def make_infinite_dataloader(dataset_name, batch_size, split="train", image_size=None, augmentation_config=None):
    """Create an infinite dataloader that loops over the dataset.

    Yields:
        (images, targets, epoch)
    """
    loader, num_classes = make_dataloader(
        dataset_name,
        batch_size,
        split,
        image_size=image_size,
        augmentation_config=augmentation_config,
    )
    epoch = 1
    while True:
        for batch in loader:
            images, targets = batch
            yield images, targets, epoch
        epoch += 1


def get_class_weights(dataset_name, device="cuda"):
    """Compute inverse-frequency class weights for imbalanced datasets."""
    data_dir = os.path.join(DATA_DIR, dataset_name)
    split_file = os.path.join(data_dir, "splits.csv")
    df = pd.read_csv(split_file)
    train_df = df[df["split"] == "train"]
    task = DATASETS[dataset_name]["task"]

    num_classes = DATASETS[dataset_name]["num_classes"]

    if task == "multilabel":
        label_matrix = np.stack([
            np.array([int(x) for x in str(label_str).split("|")], dtype=np.float32)
            for label_str in train_df["labels"]
        ])
        positives = label_matrix.sum(axis=0)
        negatives = len(label_matrix) - positives
        weights = negatives / np.maximum(positives, 1.0)
        weights = np.clip(weights, 0.5, 20.0)
        return torch.tensor(weights, dtype=torch.float32, device=device)

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

@torch.no_grad()
def evaluate_classification(model, dataset_name, batch_size, device="cuda", image_size=None, augmentation_config=None):
    """
    Evaluate classification on the validation set.

    Returns dict with:
        accuracy (float): Top-1 accuracy (0-100)
        loss (float): Average cross-entropy loss
        auc (float): ROC AUC (macro, for binary/multiclass) (0-100)
        kappa (float): Quadratic weighted Cohen's kappa (0-1)
    """
    cfg = DATASETS[dataset_name]
    loader, num_classes = make_dataloader(
        dataset_name,
        batch_size,
        split="val",
        image_size=image_size,
        augmentation_config=augmentation_config,
    )
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []
    total_loss = 0.0
    total_samples = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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
def evaluate_multilabel(model, dataset_name, batch_size, device="cuda", image_size=None, augmentation_config=None):
    """
    Evaluate multi-label classification (e.g., CheXNet/ChestX-ray14).

    Returns dict with:
        mean_auc (float): Mean AUC across all pathologies (0-100)
        per_class_auc (dict): AUC for each pathology
    """
    cfg = DATASETS[dataset_name]
    loader, num_classes = make_dataloader(
        dataset_name,
        batch_size,
        split="val",
        image_size=image_size,
        augmentation_config=augmentation_config,
    )
    model.eval()

    all_probs = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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
def evaluate_segmentation(model, dataset_name, batch_size, device="cuda", image_size=None):
    """
    Evaluate segmentation on the validation set.

    Returns dict with:
        dice (float): Dice coefficient (0-100)
        iou (float): Intersection over Union (0-100)
        pixel_acc (float): Pixel accuracy (0-100)
        sensitivity (float): True positive rate (0-100)
        specificity (float): True negative rate (0-100)
    """
    cfg = DATASETS[dataset_name]
    loader, num_classes = make_dataloader(
        dataset_name,
        batch_size,
        split="val",
        image_size=image_size,
    )
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_correct = 0
    total_pixels = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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


def _apply_clahe_pil(image):
    """Torchvision-friendly wrapper around CLAHE preprocessing."""
    image_np = np.array(image)
    image_np = apply_clahe(image_np)
    return Image.fromarray(image_np)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def list_datasets():
    """Print all available datasets (built-in + custom)."""
    # Refresh registry to pick up any new dataset.json files
    global DATASETS
    DATASETS = build_registry()

    builtin_names = set(BUILTIN_DATASETS.keys())
    custom_names = set(DATASETS.keys()) - builtin_names

    print("\nAvailable medical imaging datasets:\n")
    print(f"{'Name':<20} {'Task':<16} {'Classes':<8} {'Metric':<16} Description")
    print("-" * 110)

    # Built-in datasets
    if builtin_names:
        print("\n  Built-in presets:")
        for name in sorted(builtin_names):
            cfg = DATASETS[name]
            print(f"  {name:<18} {cfg['task']:<16} {cfg['num_classes']:<8} "
                  f"{cfg['metric']:<16} {cfg['description']}")

    # Custom datasets
    if custom_names:
        print("\n  Custom datasets:")
        for name in sorted(custom_names):
            cfg = DATASETS[name]
            print(f"  {name:<18} {cfg['task']:<16} {cfg['num_classes']:<8} "
                  f"{cfg['metric']:<16} {cfg['description']}")

    print(f"\nTotal: {len(DATASETS)} datasets ({len(builtin_names)} built-in, "
          f"{len(custom_names)} custom)")
    print()
    print("To add a custom dataset:")
    print("  uv run prepare.py --data /path/to/your/images")
    print("  uv run prepare.py --kaggle username/dataset-slug")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare medical imaging datasets — built-in presets or bring-your-own-data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Built-in datasets (download from Kaggle)
  uv run prepare.py --dataset chestxray14
  uv run prepare.py --dataset all

  # Bring your own data (auto-detect everything)
  uv run prepare.py --data /path/to/my/images
  uv run prepare.py --data /path --task segmentation
  uv run prepare.py --data /path --patient-column PatientID
  uv run prepare.py --data /path --compute-stats

  # Download from Kaggle + auto-detect
  uv run prepare.py --kaggle username/dataset-slug

  # List all available datasets
  uv run prepare.py --list
""",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Built-in dataset to download (use --list to see options)",
    )
    parser.add_argument("--list", action="store_true",
                        help="List all available datasets (built-in + custom)")

    # Bring-your-own-data
    parser.add_argument("--data", type=str, default=None,
                        help="Path to your own dataset directory (auto-detect)")
    parser.add_argument("--kaggle", type=str, default=None,
                        help="Kaggle dataset slug to download and auto-detect")
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

    # --- List mode ---
    if args.list:
        list_datasets()
        sys.exit(0)

    # --- Bring-your-own-data mode ---
    if args.data:
        if not os.path.isdir(args.data):
            print(f"ERROR: directory not found: {args.data}")
            sys.exit(1)
        ds_name = prepare_custom_dataset(
            args.data,
            task_override=args.task,
            config_overrides=args.config,
            patient_column=args.patient_column,
            compute_stats=args.compute_stats,
        )
        if ds_name:
            print("Done! Ready to train.")
        sys.exit(0)

    # --- Kaggle auto-detect mode ---
    if args.kaggle:
        ds_name = download_kaggle_auto(args.kaggle)
        if ds_name:
            print("Done! Ready to train.")
        sys.exit(0)

    # --- Built-in dataset mode ---
    if args.dataset:
        # Validate against known datasets
        valid_choices = list(DATASETS.keys()) + ["all"]
        if args.dataset not in valid_choices:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {sorted(DATASETS.keys())}")
            print("\nOr bring your own data: uv run prepare.py --data /path/to/images")
            sys.exit(1)

        print(f"Data directory: {DATA_DIR}")
        print()

        if args.dataset == "all":
            for ds_name in BUILTIN_DATASETS:
                download_dataset(ds_name)
                print()
        else:
            download_dataset(args.dataset)

        print()
        print("Done! Ready to train.")
        sys.exit(0)

    # --- No arguments: show help ---
    list_datasets()
    print("\nUsage:")
    print("  uv run prepare.py --dataset <name>     # Download built-in dataset")
    print("  uv run prepare.py --data /path          # Bring your own data")
    print("  uv run prepare.py --kaggle slug          # Download from Kaggle + auto-detect")
    print("  uv run prepare.py --list                 # List all datasets")
