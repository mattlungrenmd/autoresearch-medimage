import copy
import hashlib
import json
import re
from pathlib import Path

import torch
import yaml


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def slugify(value):
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return text.strip("_") or "candidate"


def load_checkpoint_config(checkpoint_path):
    path = Path(str(checkpoint_path)).expanduser()
    if not path.exists():
        return {}
    payload = torch.load(path, map_location="cpu")
    config = payload.get("config", {})
    return config if isinstance(config, dict) else {}


def recipe_signature(recipe_spec):
    training = recipe_spec.get("training", {})
    model = recipe_spec.get("model", {})
    loss = recipe_spec.get("loss", {})
    augmentation = recipe_spec.get("augmentation", {})
    advanced = recipe_spec.get("advanced", {})
    key = {
        "dataset": recipe_spec.get("dataset", ""),
        "task": recipe_spec.get("task", ""),
        "image_size": int(recipe_spec.get("image_size", 224)),
        "model": {
            "backbone": model.get("backbone", ""),
            "dropout": round(_safe_float(model.get("dropout", 0.0), 0.0), 4),
        },
        "training": {
            "batch_size": int(training.get("batch_size", 0)),
            "optimizer": training.get("optimizer", ""),
            "learning_rate": round(_safe_float(training.get("learning_rate", 0.0), 0.0), 8),
            "weight_decay": round(_safe_float(training.get("weight_decay", 0.0), 0.0), 8),
            "warmup_ratio": round(_safe_float(training.get("warmup_ratio", 0.0), 0.0), 4),
            "warmdown_ratio": round(_safe_float(training.get("warmdown_ratio", 0.0), 0.0), 4),
        },
        "loss": {
            "type": loss.get("type", ""),
            "focal_gamma": round(_safe_float(loss.get("focal_gamma", 2.0), 2.0), 4),
        },
        "augmentation": {
            "horizontal_flip": bool(augmentation.get("horizontal_flip", False)),
            "vertical_flip": bool(augmentation.get("vertical_flip", False)),
            "rotation": int(_safe_float(augmentation.get("rotation", 0), 0)),
            "color_jitter": bool(augmentation.get("color_jitter", False)),
            "color_jitter_brightness": round(
                _safe_float(augmentation.get("color_jitter_brightness", 0.0), 0.0), 4
            ),
            "color_jitter_contrast": round(
                _safe_float(augmentation.get("color_jitter_contrast", 0.0), 0.0), 4
            ),
        },
        "advanced": {
            "ema": bool(advanced.get("ema", False)),
            "gradual_unfreeze": bool(advanced.get("gradual_unfreeze", False)),
            "discriminative_lr": bool(advanced.get("discriminative_lr", False)),
            "lr_schedule": advanced.get("lr_schedule", ""),
        },
    }
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()


def recipe_spec_from_run(run_row, checkpoint_config=None, dataset_name="chestxray14"):
    config = dict(checkpoint_config or {})

    backbone = str(config.get("backbone") or run_row.get("backbone") or "densenet121")
    optimizer = str(config.get("optimizer") or run_row.get("optimizer") or "adam")
    learning_rate = _safe_float(
        config.get("learning_rate", run_row.get("learning_rate", 1e-3)), 1e-3
    )
    weight_decay = _safe_float(
        config.get("weight_decay", run_row.get("weight_decay", 1e-4)), 1e-4
    )
    batch_size = _safe_int(config.get("batch_size", run_row.get("batch_size", 16)), 16)
    dropout = _safe_float(config.get("dropout", run_row.get("dropout", 0.2)), 0.2)
    image_size = _safe_int(config.get("image_size", run_row.get("image_size", 224)), 224)
    loss_type = str(config.get("loss") or run_row.get("loss") or "bce")
    augmentations = config.get("augmentations", {})
    if not isinstance(augmentations, dict):
        augmentations = {}

    spec = {
        "paper": {
            "title": f"Autoresearch candidate: {backbone}",
            "dataset": "NIH ChestX-ray14",
            "reported_metric": "Autoresearch mutation candidate",
        },
        "dataset": dataset_name,
        "task": "multilabel",
        "model": {
            "backbone": backbone,
            "pretrained": True,
            "dropout": round(dropout, 4),
        },
        "training": {
            "batch_size": batch_size,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": _safe_float(config.get("warmup_ratio", 0.05), 0.05),
            "warmdown_ratio": _safe_float(config.get("warmdown_ratio", 0.3), 0.3),
        },
        "loss": {
            "type": loss_type,
            "focal_gamma": _safe_float(config.get("focal_gamma", 2.0), 2.0),
            "class_weights": True,
        },
        "augmentation": {
            "horizontal_flip": bool(augmentations.get("horizontal_flip", True)),
            "vertical_flip": bool(augmentations.get("vertical_flip", False)),
            "rotation": int(_safe_float(augmentations.get("rotation", 10), 10)),
            "color_jitter": bool(augmentations.get("color_jitter", True)),
            "color_jitter_brightness": _safe_float(
                augmentations.get("color_jitter_brightness", 0.15), 0.15
            ),
            "color_jitter_contrast": _safe_float(
                augmentations.get("color_jitter_contrast", 0.15), 0.15
            ),
            "clahe": bool(augmentations.get("clahe", False)),
            "elastic_transform": bool(augmentations.get("elastic_transform", False)),
            "grid_distortion": bool(augmentations.get("grid_distortion", False)),
            "coarse_dropout": bool(augmentations.get("coarse_dropout", False)),
        },
        "image_size": image_size,
        "notes": (
            "Generated by the Colab autoresearch loop. This recipe is a mutation of "
            f"the current best session run for backbone {backbone}."
        ),
    }

    advanced = {}
    for key in [
        "ema",
        "gradual_unfreeze",
        "discriminative_lr",
        "mixup",
        "cutmix",
        "swa",
        "llrd",
        "distillation",
        "model_soup",
    ]:
        if config.get(key):
            advanced[key] = True
    if config.get("ema_decay") is not None:
        advanced["ema_decay"] = _safe_float(config.get("ema_decay", 0.999), 0.999)
    if config.get("unfreeze_after_ratio") is not None:
        advanced["unfreeze_after_ratio"] = _safe_float(
            config.get("unfreeze_after_ratio", 0.1), 0.1
        )
    if config.get("backbone_lr_factor") is not None:
        advanced["backbone_lr_factor"] = _safe_float(
            config.get("backbone_lr_factor", 0.1), 0.1
        )
    if config.get("llrd_decay") is not None:
        advanced["llrd_decay"] = _safe_float(config.get("llrd_decay", 0.8), 0.8)
    if config.get("lr_schedule") not in (None, "", "cosine"):
        advanced["lr_schedule"] = str(config.get("lr_schedule"))
    if config.get("grad_accum_steps") is not None:
        advanced["grad_accum_steps"] = _safe_int(config.get("grad_accum_steps", 1), 1)
    if advanced:
        spec["advanced"] = advanced

    return spec


def propose_mutation_specs(base_recipe, wave_index=0):
    base = copy.deepcopy(base_recipe)
    training = base.setdefault("training", {})
    model = base.setdefault("model", {})
    loss = base.setdefault("loss", {})
    aug = base.setdefault("augmentation", {})
    advanced = copy.deepcopy(base.get("advanced", {}))

    lr = _safe_float(training.get("learning_rate", 1e-3), 1e-3)
    wd = _safe_float(training.get("weight_decay", 1e-4), 1e-4)
    batch_size = _safe_int(training.get("batch_size", 16), 16)
    dropout = _safe_float(model.get("dropout", 0.2), 0.2)
    image_size = _safe_int(base.get("image_size", 224), 224)
    rotation = _safe_int(aug.get("rotation", 10), 10)
    brightness = _safe_float(aug.get("color_jitter_brightness", 0.15), 0.15)
    contrast = _safe_float(aug.get("color_jitter_contrast", 0.15), 0.15)

    proposals = []

    def add(label, mutator, note):
        spec = copy.deepcopy(base)
        mutator(spec)
        spec["paper"] = dict(spec.get("paper", {}))
        spec["paper"]["title"] = f"Autoresearch mutation: {label}"
        spec["paper"]["reported_metric"] = "Autoresearch mutation candidate"
        spec["notes"] = f"{spec.get('notes', '').strip()}\n\nMutation note: {note}".strip()
        proposals.append((label, spec))

    wave_group = wave_index % 4

    if wave_group == 0:
        add("lr_down", lambda s: s["training"].update({"learning_rate": round(lr * 0.7, 8)}), "Lower learning rate to stabilize short-budget training.")
        add("lr_up", lambda s: s["training"].update({"learning_rate": round(lr * 1.3, 8)}), "Higher learning rate to test faster early progress.")
        add("dropout_down", lambda s: s["model"].update({"dropout": round(_clamp(dropout - 0.1, 0.0, 0.6), 4)}), "Reduce dropout around the current best.")
        add(
            "img256",
            lambda s: (s.update({"image_size": max(image_size, 256)}), s["training"].update({"batch_size": max(16, min(batch_size, 24))})),
            "Increase image size to 256 while keeping the batch size T4-friendly.",
        )
    elif wave_group == 1:
        add("wd_down", lambda s: s["training"].update({"weight_decay": round(max(wd * 0.5, 1e-5), 8)}), "Lower weight decay to reduce regularization."),
        add("wd_up", lambda s: s["training"].update({"weight_decay": round(min(wd * 2.0, 0.1), 8)}), "Increase weight decay to test stronger regularization."),
        add(
            "one_cycle",
            lambda s: s.setdefault("advanced", {}).update({"lr_schedule": "one_cycle"}),
            "Switch to a one-cycle learning rate schedule for the 300-second budget.",
        )
        add(
            "stronger_color",
            lambda s: s["augmentation"].update({
                "color_jitter": True,
                "color_jitter_brightness": round(min(brightness + 0.05, 0.3), 4),
                "color_jitter_contrast": round(min(contrast + 0.05, 0.3), 4),
            }),
            "Make color jitter slightly stronger.",
        )
    elif wave_group == 2:
        add(
            "ema",
            lambda s: s.setdefault("advanced", {}).update({"ema": True, "ema_decay": 0.999}),
            "Enable EMA on the current best recipe.",
        )
        add(
            "img288",
            lambda s: (s.update({"image_size": max(image_size, 288)}), s["training"].update({"batch_size": max(12, min(batch_size, 16))})),
            "Try a higher-resolution 288px variant with a smaller batch size.",
        )
        add(
            "rotation15",
            lambda s: s["augmentation"].update({"rotation": min(rotation + 5, 20)}),
            "Increase rotation augmentation slightly.",
        )
        add(
            "dropout_up",
            lambda s: s["model"].update({"dropout": round(_clamp(dropout + 0.1, 0.0, 0.6), 4)}),
            "Increase dropout to test stronger regularization.",
        )
    else:
        add(
            "lr_down_img256",
            lambda s: (
                s["training"].update({"learning_rate": round(lr * 0.8, 8), "batch_size": max(16, min(batch_size, 24))}),
                s.update({"image_size": max(image_size, 256)}),
            ),
            "Combine a slightly lower learning rate with 256px inputs.",
        )
        add(
            "warmdown_longer",
            lambda s: s["training"].update({"warmdown_ratio": round(min(_safe_float(training.get("warmdown_ratio", 0.3), 0.3) + 0.1, 0.5), 4)}),
            "Lengthen the cosine cooldown phase.",
        )
        add(
            "batch24",
            lambda s: s["training"].update({"batch_size": max(16, min(24, batch_size))}),
            "Normalize batch size to 24 for a different throughput/gradient noise balance.",
        )
        add(
            "focal_gamma_1_5",
            lambda s: s["loss"].update({"type": "focal", "focal_gamma": 1.5}),
            "Reduce focal gamma to soften the class-imbalance emphasis.",
        )

    for _, spec in proposals:
        if "advanced" in spec and not spec["advanced"]:
            spec.pop("advanced", None)
        if loss.get("type") != "focal":
            spec["loss"].pop("focal_gamma", None)
    return proposals


def write_recipe_file(recipe_dir, recipe_name, recipe_spec):
    path = Path(recipe_dir) / f"{recipe_name}.yaml"
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(recipe_spec, handle, sort_keys=False)
    return path
