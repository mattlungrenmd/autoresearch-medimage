"""
Data preparation and evaluation for autoresearch-medimage experiments.
Automates Kaggle dataset download for ophthalmology and radiology tasks.

This file is READ-ONLY for agents. Do not modify.

Usage:
    python prepare.py --dataset aptos            # Diabetic Retinopathy grading
    python prepare.py --dataset oct              # Retinal OCT classification
    python prepare.py --dataset chest_xray       # Chest X-ray pneumonia
    python prepare.py --dataset brain_mri        # Brain MRI segmentation
    python prepare.py --dataset pneumothorax     # Chest pneumothorax segmentation
    python prepare.py --dataset retina_vessel    # Retinal vessel segmentation
    python prepare.py --dataset all              # Download all datasets
    python prepare.py --list                     # List available datasets

Requires: ~/.kaggle/kaggle.json with valid API credentials.
Datasets are stored in ./data/<dataset_name>/
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
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedKFold
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
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASETS.keys())}")
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

def make_classification_transforms(dataset_name, train=True):
    """Build transforms for medical classification."""
    cfg = DATASETS[dataset_name]
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
        transform_list.append(T.Lambda(lambda x: x.repeat(3, 1, 1)))

    return T.Compose(transform_list)


def make_dataloader(dataset_name, batch_size, split="train"):
    """Create a dataloader for the specified dataset and split.

    Returns:
        dataloader, num_classes
    """
    cfg = DATASETS[dataset_name]
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


def get_class_weights(dataset_name, device="cuda"):
    """Compute inverse-frequency class weights for imbalanced datasets."""
    data_dir = os.path.join(DATA_DIR, dataset_name)
    split_file = os.path.join(data_dir, "splits.csv")
    df = pd.read_csv(split_file)
    train_df = df[df["split"] == "train"]

    num_classes = DATASETS[dataset_name]["num_classes"]
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
def evaluate_classification(model, dataset_name, batch_size, device="cuda"):
    """
    Evaluate classification on the validation set.

    Returns dict with:
        accuracy (float): Top-1 accuracy (0-100)
        loss (float): Average cross-entropy loss
        auc (float): ROC AUC (macro, for binary/multiclass) (0-100)
        kappa (float): Quadratic weighted Cohen's kappa (0-1)
    """
    cfg = DATASETS[dataset_name]
    loader, num_classes = make_dataloader(dataset_name, batch_size, split="val")
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
def evaluate_multilabel(model, dataset_name, batch_size, device="cuda"):
    """
    Evaluate multi-label classification (e.g., CheXNet/ChestX-ray14).

    Returns dict with:
        mean_auc (float): Mean AUC across all pathologies (0-100)
        per_class_auc (dict): AUC for each pathology
    """
    cfg = DATASETS[dataset_name]
    loader, num_classes = make_dataloader(dataset_name, batch_size, split="val")
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
def evaluate_segmentation(model, dataset_name, batch_size, device="cuda"):
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
    loader, num_classes = make_dataloader(dataset_name, batch_size, split="val")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def list_datasets():
    """Print all available datasets."""
    print("\nAvailable medical imaging datasets:\n")
    print(f"{'Name':<16} {'Task':<16} {'Classes':<8} {'Metric':<16} Description")
    print("-" * 100)
    for name, cfg in DATASETS.items():
        print(f"{name:<16} {cfg['task']:<16} {cfg['num_classes']:<8} {cfg['metric']:<16} {cfg['description']}")
    print()
    print("Ophthalmology: aptos, oct, retina_vessel")
    print("Radiology:     chestxray14 (CheXNet), chest_xray, brain_mri, pneumothorax")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare medical imaging datasets from Kaggle")
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=list(DATASETS.keys()) + ["all"],
        help="Dataset to download",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    if args.list or args.dataset is None:
        list_datasets()
        if args.dataset is None:
            print("\nUsage: python prepare.py --dataset <name>")
        sys.exit(0)

    print(f"Data directory: {DATA_DIR}")
    print()

    if args.dataset == "all":
        for ds_name in DATASETS:
            download_dataset(ds_name)
            print()
    else:
        download_dataset(args.dataset)

    print()
    print("Done! Ready to train.")
