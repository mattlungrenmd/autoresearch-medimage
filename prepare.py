"""
Data preparation and evaluation for autoresearch-vision experiments.
Downloads datasets and provides dataloaders and evaluation functions.

This file is READ-ONLY for agents. Do not modify.

Usage:
    python prepare.py                          # prepare CIFAR-10 (default)
    python prepare.py --task segmentation      # prepare VOC segmentation
    python prepare.py --dataset cifar100       # prepare CIFAR-100

Datasets are stored in ./data/
"""

import os
import sys
import time
import math
import argparse

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300          # training time budget in seconds (5 minutes)
IMAGE_SIZE = 32            # input image size (CIFAR native)
IMAGE_SIZE_SEG = 128       # input image size for segmentation
NUM_WORKERS = 4            # dataloader workers

# Dataset configs
DATASETS = {
    "cifar10": {
        "num_classes": 10,
        "task": "classification",
        "image_size": 32,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "num_classes": 100,
        "task": "classification",
        "image_size": 32,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "voc": {
        "num_classes": 21,  # 20 classes + background
        "task": "segmentation",
        "image_size": 128,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------------------------------------------------------------------
# Dataset download and preparation
# ---------------------------------------------------------------------------

def download_dataset(dataset_name="cifar10"):
    """Download and verify dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if dataset_name == "cifar10":
        print("Downloading CIFAR-10...")
        torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
        torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True)
        print("CIFAR-10 ready.")

    elif dataset_name == "cifar100":
        print("Downloading CIFAR-100...")
        torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=True)
        torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=True)
        print("CIFAR-100 ready.")

    elif dataset_name == "voc":
        print("Downloading Pascal VOC 2012...")
        torchvision.datasets.VOCSegmentation(
            root=DATA_DIR, year="2012", image_set="train", download=True
        )
        torchvision.datasets.VOCSegmentation(
            root=DATA_DIR, year="2012", image_set="val", download=True
        )
        print("Pascal VOC 2012 ready.")

    else:
        print(f"Unknown dataset: {dataset_name}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Dataloaders (imported by train.py)
# ---------------------------------------------------------------------------

def get_dataset_config(dataset_name="cifar10"):
    """Return dataset configuration dict."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASETS.keys())}")
    return DATASETS[dataset_name]


def make_classification_transforms(dataset_name, train=True):
    """Build transforms for classification datasets."""
    cfg = DATASETS[dataset_name]
    img_size = cfg["image_size"]
    mean, std = cfg["mean"], cfg["std"]

    if train:
        return T.Compose([
            T.RandomCrop(img_size, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


def make_segmentation_transforms(train=True):
    """Build transforms for segmentation datasets.
    Returns a callable that takes (image, target) and returns (tensor, tensor).
    """
    img_size = IMAGE_SIZE_SEG

    def transform(image, target):
        # Resize
        image = T.functional.resize(image, (img_size, img_size), antialias=True)
        target = T.functional.resize(
            target, (img_size, img_size),
            interpolation=T.InterpolationMode.NEAREST,
        )

        if train:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                image = T.functional.hflip(image)
                target = T.functional.hflip(target)

        # To tensor and normalize
        image = T.functional.to_tensor(image)
        image = T.functional.normalize(image, DATASETS["voc"]["mean"], DATASETS["voc"]["std"])

        # Target: convert PIL to long tensor, map 255 (border) to ignore
        target = torch.as_tensor(np.array(target), dtype=torch.long)
        target[target == 255] = -1  # ignore index

        return image, target

    return transform


class VOCSegDataset(torch.utils.data.Dataset):
    """Wrapper for VOC that applies joint image+target transforms."""

    def __init__(self, root, image_set="train", transform_fn=None):
        self.voc = torchvision.datasets.VOCSegmentation(
            root=root, year="2012", image_set=image_set, download=False
        )
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        image, target = self.voc[idx]
        if self.transform_fn is not None:
            image, target = self.transform_fn(image, target)
        return image, target


def make_dataloader(dataset_name, batch_size, split="train"):
    """Create a dataloader for the specified dataset and split.

    Args:
        dataset_name: "cifar10", "cifar100", or "voc"
        batch_size: batch size
        split: "train" or "val"

    Returns:
        dataloader, num_classes
    """
    cfg = DATASETS[dataset_name]
    is_train = (split == "train")

    if dataset_name in ("cifar10", "cifar100"):
        transforms = make_classification_transforms(dataset_name, train=is_train)
        cls = torchvision.datasets.CIFAR10 if dataset_name == "cifar10" else torchvision.datasets.CIFAR100
        dataset = cls(root=DATA_DIR, train=is_train, download=False, transform=transforms)

    elif dataset_name == "voc":
        transform_fn = make_segmentation_transforms(train=is_train)
        image_set = "train" if is_train else "val"
        dataset = VOCSegDataset(root=DATA_DIR, image_set=image_set, transform_fn=transform_fn)

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


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — these are the fixed metrics)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_classification(model, dataset_name, batch_size, device="cuda"):
    """
    Evaluate classification accuracy on the validation set.

    Returns:
        accuracy (float): Top-1 accuracy as a percentage (0-100)
        loss (float): Average cross-entropy loss
    """
    loader, num_classes = make_dataloader(dataset_name, batch_size, split="val")
    model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    num_batches = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)

        loss = F.cross_entropy(logits, targets, reduction="sum")
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)
        num_batches += 1

    accuracy = 100.0 * total_correct / total_samples
    avg_loss = total_loss / total_samples
    return accuracy, avg_loss


@torch.no_grad()
def evaluate_segmentation(model, batch_size, device="cuda"):
    """
    Evaluate segmentation using mean IoU on the VOC validation set.

    Returns:
        miou (float): Mean Intersection over Union (0-100)
        pixel_acc (float): Pixel accuracy as a percentage (0-100)
    """
    loader, num_classes = make_dataloader("voc", batch_size, split="val")
    model.eval()

    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    total_correct = 0
    total_pixels = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)

        # Upsample predictions to target size if needed
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(
                logits, size=targets.shape[-2:], mode="bilinear", align_corners=False
            )

        preds = logits.argmax(dim=1)

        # Mask out ignore regions
        valid = targets >= 0
        preds_valid = preds[valid]
        targets_valid = targets[valid]

        total_correct += (preds_valid == targets_valid).sum().item()
        total_pixels += valid.sum().item()

        for cls_id in range(num_classes):
            pred_mask = preds_valid == cls_id
            target_mask = targets_valid == cls_id
            intersection[cls_id] += (pred_mask & target_mask).sum()
            union[cls_id] += (pred_mask | target_mask).sum()

    # Compute mIoU (only over classes that appear in GT)
    valid_classes = union > 0
    iou_per_class = intersection[valid_classes] / (union[valid_classes] + 1e-6)
    miou = 100.0 * iou_per_class.mean().item()
    pixel_acc = 100.0 * total_correct / (total_pixels + 1e-6)

    return miou, pixel_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets for autoresearch-vision")
    parser.add_argument(
        "--dataset", type=str, default="cifar10",
        choices=["cifar10", "cifar100", "voc", "all"],
        help="Dataset to download",
    )
    args = parser.parse_args()

    print(f"Data directory: {DATA_DIR}")
    print()

    if args.dataset == "all":
        for ds in ["cifar10", "cifar100", "voc"]:
            download_dataset(ds)
            print()
    else:
        download_dataset(args.dataset)

    print()
    print("Done! Ready to train.")
