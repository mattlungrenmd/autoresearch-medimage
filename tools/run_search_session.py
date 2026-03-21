"""
Run an optional adaptive search loop on top of train.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from tools.search_report import generate_search_report

RESULTS_FILE = REPO_DIR / "results.tsv"
RECIPES_DIR = REPO_DIR / "recipes"
SEARCH_SPACES_DIR = REPO_DIR / "search_spaces"
DEFAULT_ALGO = "adaptive_search"
MAX_RECIPE_STEM = 96
SEARCH_METADATA_FIELDS = [
    "candidate_id",
    "parent_id",
    "search_algo",
    "mutation_op",
    "base_recipe",
    "replicate_group",
]


def normalize_status(value: object) -> str:
    return str(value or "").strip().lower()


def safe_float(value, default=float("nan")) -> float:
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def slugify(text: str) -> str:
    keep = []
    for ch in str(text or ""):
        if ch.isalnum():
            keep.append(ch.lower())
        elif ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "candidate"


def compact_slug(text: str, limit: int) -> str:
    return slugify(text)[:limit].strip("_") or "candidate"


def build_recipe_stem(candidate_id: str, *parts: str) -> str:
    tokens = [candidate_id]
    for part in parts:
        part = str(part or "").strip()
        if not part:
            continue
        tokens.append(compact_slug(part, 24))
    stem = "_".join(tokens).strip("_")
    return stem[:MAX_RECIPE_STEM].rstrip("_") or candidate_id


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_results(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(results_path, sep="\t")
    if "metric" in df.columns:
        df["metric"] = pd.to_numeric(df["metric"], errors="coerce")
    if "status" in df.columns:
        df["status_norm"] = df["status"].map(normalize_status)
    else:
        df["status_norm"] = "candidate"
    return df


def tracked_recipe_path(recipe_name: str) -> Path:
    return RECIPES_DIR / f"{recipe_name}.yaml"


def tracked_recipe_exists(recipe_name: str) -> bool:
    return tracked_recipe_path(recipe_name).exists()


def resolve_seed_recipe_file(space: dict, recipe_path: str) -> Path:
    path = Path(str(recipe_path or "")).expanduser()
    if path.is_absolute():
        return path
    space_path = Path(str(space.get("_path", "") or "")).expanduser()
    if space_path:
        return (space_path.parent / path).resolve()
    return path.resolve()


def branch_key_for_backbone(backbone: str) -> str:
    text = str(backbone or "").lower()
    for prefix in ("densenet", "efficientnet", "convnext", "resnet"):
        if text.startswith(prefix):
            return prefix
    return text or "unknown"


def candidate_score(candidate: dict) -> float:
    confirmed = safe_float(candidate.get("confirmed_metric"))
    if not math.isnan(confirmed):
        return confirmed
    return safe_float(candidate.get("metric"))


def choose_from_ladder(current, values, rng: random.Random, prefer_far: bool = False):
    ladder = list(values or [])
    if not ladder:
        return current
    choices = [value for value in ladder if value != current]
    if not choices:
        return current
    if prefer_far:
        if current in ladder:
            current_index = ladder.index(current)
            ranked = sorted(
                choices,
                key=lambda item: (abs(ladder.index(item) - current_index), str(item)),
                reverse=True,
            )
            return ranked[0]
        try:
            ranked = sorted(choices, key=lambda item: abs(float(item) - float(current)), reverse=True)
            return ranked[0]
        except (TypeError, ValueError):
            return sorted(choices, key=lambda item: str(item), reverse=True)[0]
    return rng.choice(choices)


def ensure_dict(recipe: dict, key: str) -> dict:
    value = recipe.get(key)
    if not isinstance(value, dict):
        value = {}
        recipe[key] = value
    return value


def summary_to_augmentations(text: str) -> dict:
    aug = {}
    for token in str(text or "").split(","):
        token = token.strip()
        if not token or token == "none":
            continue
        if token == "hflip":
            aug["horizontal_flip"] = True
        elif token == "vflip":
            aug["vertical_flip"] = True
        elif token == "color":
            aug["color_jitter"] = True
        elif token == "clahe":
            aug["clahe"] = True
        elif token == "elastic":
            aug["elastic_transform"] = True
        elif token == "grid":
            aug["grid_distortion"] = True
        elif token.startswith("rot="):
            aug["rotation"] = int(float(token.split("=", 1)[1]))
        elif token.startswith("coarse="):
            aug["coarse_dropout"] = True
            aug["coarse_dropout_ratio"] = float(token.split("=", 1)[1])
    return aug


def config_to_recipe(config: dict) -> dict:
    recipe = {
        "dataset": config.get("dataset", ""),
        "task": config.get("task"),
        "model": {},
        "training": {},
        "loss": {},
        "augmentation": {},
        "advanced": {},
    }
    if isinstance(config.get("model"), dict):
        recipe["model"].update(copy.deepcopy(config["model"]))
    else:
        recipe["model"].update(
            {
                "backbone": config.get("backbone"),
                "dropout": config.get("dropout"),
                "pretrained": config.get("pretrained", True),
            }
        )
    if isinstance(config.get("training"), dict):
        recipe["training"].update(copy.deepcopy(config["training"]))
    else:
        for key, source in [
            ("batch_size", "batch_size"),
            ("learning_rate", "learning_rate"),
            ("optimizer", "optimizer"),
            ("weight_decay", "weight_decay"),
            ("label_smoothing", "label_smoothing"),
        ]:
            value = config.get(source)
            if value is not None:
                recipe["training"][key] = value
    if isinstance(config.get("loss"), dict):
        recipe["loss"].update(copy.deepcopy(config["loss"]))
    elif isinstance(config.get("loss_config"), dict):
        loss_cfg = copy.deepcopy(config["loss_config"])
        if "type" in loss_cfg:
            recipe["loss"]["type"] = loss_cfg.pop("type")
        recipe["loss"].update(loss_cfg)
    elif config.get("loss") is not None:
        recipe["loss"]["type"] = config.get("loss")
    if isinstance(config.get("augmentations"), dict):
        recipe["augmentation"].update(copy.deepcopy(config["augmentations"]))
    if isinstance(config.get("advanced"), dict):
        recipe["advanced"].update(copy.deepcopy(config["advanced"]))
    if config.get("image_size") is not None:
        recipe["image_size"] = int(config["image_size"])
    if not recipe["advanced"]:
        for key, source in [
            ("ema", "ema"),
            ("ema_decay", "ema_decay"),
            ("tta", "tta"),
            ("compile", "compile"),
            ("lr_schedule", "lr_schedule"),
        ]:
            value = config.get(source)
            if value is not None:
                recipe["advanced"][key] = value
    return recipe


def checkpoint_to_recipe(checkpoint_path: Path) -> dict | None:
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if not isinstance(config, dict) or not config:
        return None
    return config_to_recipe(config)


def row_to_recipe(row: pd.Series) -> dict | None:
    recipe_name = str(row.get("recipe", "")).strip()
    if recipe_name and tracked_recipe_exists(recipe_name):
        return load_yaml(tracked_recipe_path(recipe_name))

    checkpoint_value = str(row.get("checkpoint_path", "")).strip()
    if checkpoint_value:
        recipe = checkpoint_to_recipe(Path(checkpoint_value))
        if recipe:
            return recipe

    recipe = {
        "dataset": str(row.get("dataset", "")).strip(),
        "model": {
            "backbone": str(row.get("backbone", "")).strip(),
        },
        "training": {},
        "loss": {},
        "augmentation": summary_to_augmentations(str(row.get("augmentations", ""))),
        "advanced": {},
    }
    if not recipe["dataset"] or not recipe["model"]["backbone"]:
        return None

    numeric_fields = {
        ("training", "batch_size"): row.get("batch_size"),
        ("training", "learning_rate"): row.get("learning_rate"),
        ("training", "weight_decay"): row.get("weight_decay"),
        ("model", "dropout"): row.get("dropout"),
        ("image_size", None): row.get("image_size"),
    }
    for (section, key), raw in numeric_fields.items():
        value = safe_float(raw)
        if math.isnan(value):
            continue
        if key is None:
            recipe["image_size"] = int(value)
        elif section == "training":
            recipe["training"][key] = int(value) if key == "batch_size" else float(value)
        else:
            recipe[section][key] = float(value)

    optimizer = str(row.get("optimizer", "")).strip()
    if optimizer:
        recipe["training"]["optimizer"] = optimizer
    loss_name = str(row.get("loss", "")).strip()
    if loss_name:
        recipe["loss"]["type"] = loss_name
    return recipe


def resolve_space_path(space_name_or_path: str) -> Path:
    text = str(space_name_or_path or "").strip().strip('"').strip("'")
    raw = Path(text).expanduser()
    if raw.exists():
        return raw.resolve()
    if raw.is_absolute() or raw.suffix or any(sep in text for sep in ("\\", "/")):
        return raw.resolve()
    return (SEARCH_SPACES_DIR / f"{text}.yaml").resolve()


def load_search_space(space_name_or_path: str) -> dict:
    path = resolve_space_path(space_name_or_path)
    if not path.exists():
        raise FileNotFoundError(f"Search space not found: {space_name_or_path}")
    space = load_yaml(path)
    space["_path"] = str(path)
    return space


def ucb_score(stat: dict, total_trials: int, exploration: float = 0.15) -> float:
    trials = int(stat.get("trials", 0))
    reward_sum = float(stat.get("reward_sum", 0.0))
    mean_reward = reward_sum / trials if trials else 0.0
    return mean_reward + exploration * math.sqrt(math.log(total_trials + 1.0) / (trials + 1.0))


def candidate_timeout_seconds(time_budget: float | None) -> float | None:
    if time_budget is None:
        return None
    budget = max(1.0, float(time_budget))
    # Allow time for post-train evaluation/TTA while still recovering from hung workers.
    return max(900.0, budget * 2.0 + 240.0)


def family_pool_for_phase(space: dict, force_explore: bool = False, allow_large_jump: bool = False) -> list[str]:
    phase_families = space.get("phase_families", {})
    if not isinstance(phase_families, dict):
        return list(space.get("mutation_families", []))
    if allow_large_jump and phase_families.get("refresh"):
        return list(phase_families["refresh"])
    if force_explore and phase_families.get("explore"):
        return list(phase_families["explore"])
    if phase_families.get("default"):
        return list(phase_families["default"])
    return list(space.get("mutation_families", []))


def select_family(
    applicable_families: list[str],
    operator_stats: dict,
    rng: random.Random,
    force_explore: bool = False,
) -> str:
    if force_explore:
        ranked = sorted(
            applicable_families,
            key=lambda family: (operator_stats.get(family, {}).get("trials", 0), family),
        )
        return ranked[0]
    total_trials = sum(int(stat.get("trials", 0)) for stat in operator_stats.values())
    scores = []
    for family in applicable_families:
        stat = operator_stats.get(family, {"trials": 0, "reward_sum": 0.0})
        scores.append((ucb_score(stat, total_trials), rng.random(), family))
    scores.sort(reverse=True)
    return scores[0][2]


def apply_augmentation_preset(recipe: dict, preset: dict) -> dict:
    augmentation = ensure_dict(recipe, "augmentation")
    for key, value in preset.items():
        augmentation[key] = value
    return recipe


def mutate_recipe(base_recipe: dict, space: dict, family: str, rng: random.Random, allow_large_jump: bool = False) -> tuple[dict, dict]:
    recipe = copy.deepcopy(base_recipe)
    model = ensure_dict(recipe, "model")
    training = ensure_dict(recipe, "training")
    loss = ensure_dict(recipe, "loss")
    advanced = ensure_dict(recipe, "advanced")
    ranges = space.get("ranges", {})
    categorical = space.get("categorical_choices", {})
    delta = {"family": family}

    if family == "backbone_swap":
        current = str(model.get("backbone", ""))
        new_backbone = choose_from_ladder(current, space.get("backbone_pool", []), rng, prefer_far=allow_large_jump)
        model["backbone"] = new_backbone
        delta["backbone"] = {"from": current, "to": new_backbone}
        return recipe, delta

    if family == "optimizer_scheduler":
        if rng.random() < 0.5:
            current = str(training.get("optimizer", "adamw"))
            new_optimizer = choose_from_ladder(current, categorical.get("optimizer", []), rng, prefer_far=allow_large_jump)
            training["optimizer"] = new_optimizer
            delta["optimizer"] = {"from": current, "to": new_optimizer}
        else:
            current = str(advanced.get("lr_schedule", "cosine"))
            new_schedule = choose_from_ladder(current, categorical.get("lr_schedule", []), rng, prefer_far=allow_large_jump)
            advanced["lr_schedule"] = new_schedule
            delta["lr_schedule"] = {"from": current, "to": new_schedule}
        return recipe, delta

    if family == "lr_wd":
        if rng.random() < 0.5 or allow_large_jump:
            current = safe_float(training.get("learning_rate", 0.001), default=0.001)
            new_value = choose_from_ladder(current, ranges.get("learning_rate", []), rng, prefer_far=allow_large_jump)
            training["learning_rate"] = float(new_value)
            delta["learning_rate"] = {"from": current, "to": float(new_value)}
        if rng.random() >= 0.5 or allow_large_jump:
            current = safe_float(training.get("weight_decay", 0.0), default=0.0)
            new_value = choose_from_ladder(current, ranges.get("weight_decay", []), rng, prefer_far=allow_large_jump)
            training["weight_decay"] = float(new_value)
            delta["weight_decay"] = {"from": current, "to": float(new_value)}
        return recipe, delta

    if family == "batch_image":
        if rng.random() < 0.5:
            current = int(safe_float(training.get("batch_size", 16), default=16))
            new_value = int(choose_from_ladder(current, ranges.get("batch_size", []), rng, prefer_far=allow_large_jump))
            training["batch_size"] = new_value
            delta["batch_size"] = {"from": current, "to": new_value}
        else:
            current = int(safe_float(recipe.get("image_size", 224), default=224))
            new_value = int(choose_from_ladder(current, ranges.get("image_size", []), rng, prefer_far=allow_large_jump))
            recipe["image_size"] = new_value
            delta["image_size"] = {"from": current, "to": new_value}
        return recipe, delta

    if family == "regularization":
        knob = rng.choice(["dropout", "ema", "tta", "compile"])
        if knob == "dropout":
            current = safe_float(model.get("dropout", 0.2), default=0.2)
            new_value = choose_from_ladder(current, ranges.get("dropout", []), rng, prefer_far=allow_large_jump)
            model["dropout"] = float(new_value)
            delta["dropout"] = {"from": current, "to": float(new_value)}
        else:
            current = bool(advanced.get(knob, False))
            advanced[knob] = not current
            delta[knob] = {"from": current, "to": advanced[knob]}
        return recipe, delta

    if family == "augmentation":
        presets = categorical.get("augmentation_presets", {})
        preset_names = list(presets.keys())
        current = str(recipe.get("search_metadata", {}).get("augmentation_preset", "custom"))
        new_preset = choose_from_ladder(current, preset_names, rng, prefer_far=allow_large_jump)
        apply_augmentation_preset(recipe, presets[new_preset])
        delta["preset"] = {"from": current, "to": new_preset}
        recipe.setdefault("search_metadata", {})
        recipe["search_metadata"]["augmentation_preset"] = new_preset
        return recipe, delta

    if family == "loss":
        task = str(recipe.get("task", space.get("task", ""))).lower()
        if task == "classification" and rng.random() < 0.33:
            current = safe_float(training.get("label_smoothing", 0.0), default=0.0)
            new_value = choose_from_ladder(current, ranges.get("label_smoothing", []), rng, prefer_far=allow_large_jump)
            training["label_smoothing"] = float(new_value)
            delta["label_smoothing"] = {"from": current, "to": float(new_value)}
        elif rng.random() < 0.5:
            current = str(loss.get("type", "auto"))
            new_value = choose_from_ladder(current, categorical.get("loss_type", []), rng, prefer_far=allow_large_jump)
            loss["type"] = new_value
            delta["loss_type"] = {"from": current, "to": new_value}
        else:
            current = bool(loss.get("class_weights", False))
            loss["class_weights"] = not current
            delta["class_weights"] = {"from": current, "to": loss["class_weights"]}
        return recipe, delta

    raise ValueError(f"Unsupported mutation family: {family}")


def load_state(search_dir: Path) -> tuple[dict, list[dict]]:
    state_path = search_dir / "search_state.json"
    candidates_path = search_dir / "candidates.jsonl"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {}
    candidates = []
    if candidates_path.exists():
        for line in candidates_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return state, candidates


def save_state(search_dir: Path, state: dict) -> None:
    write_json(search_dir / "search_state.json", state)


def update_candidate_record(search_dir: Path, candidates: list[dict]) -> None:
    path = search_dir / "candidates.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, sort_keys=True) + "\n")


def update_results_status(results_path: Path, candidate_ids: list[str], status: str, note: str | None = None) -> None:
    if not results_path.exists() or not candidate_ids:
        return
    df = pd.read_csv(results_path, sep="\t")
    if "candidate_id" not in df.columns or "status" not in df.columns:
        return
    mask = df["candidate_id"].fillna("").astype(str).isin([str(cid) for cid in candidate_ids])
    if not mask.any():
        return
    df.loc[mask, "status"] = status
    if note and "description" in df.columns:
        descriptions = df.loc[mask, "description"].fillna("").astype(str)
        df.loc[mask, "description"] = descriptions.apply(lambda text: f"{text} | {note}" if text else note)
    df.to_csv(results_path, sep="\t", index=False)


def latest_result_for_candidate(results_path: Path, session: str, candidate_id: str) -> dict | None:
    df = read_results(results_path)
    if df.empty or "candidate_id" not in df.columns:
        return None
    view = df[df["candidate_id"].fillna("").astype(str) == str(candidate_id)].copy()
    if "session" in view.columns:
        view = view[view["session"].fillna("").astype(str) == str(session)]
    if view.empty:
        return None
    row = view.iloc[-1]
    return {column: row[column] for column in view.columns}


def apply_result_row_to_candidate(candidate: dict, row: dict) -> dict:
    candidate["status"] = normalize_status(row.get("status", "")) or "candidate"
    candidate["metric"] = safe_float(row.get("metric"))
    candidate["checkpoint_path"] = str(row.get("checkpoint_path", "") or "")
    candidate["backbone"] = str(row.get("backbone", "") or candidate.get("backbone", ""))
    candidate["recipe_name"] = str(row.get("recipe", "") or candidate.get("recipe_name", ""))
    candidate["metric_name"] = str(row.get("metric_name", "") or "")
    candidate["description"] = str(row.get("description", "") or "")
    return candidate


def sync_candidate_from_results(results_path: Path, session: str, candidate: dict) -> bool:
    row = latest_result_for_candidate(results_path, session, str(candidate.get("candidate_id", "")))
    if row is None:
        return False
    apply_result_row_to_candidate(candidate, row)
    return True


def best_known_seed(space: dict, dataset: str, results_path: Path) -> tuple[str, dict] | None:
    df = read_results(results_path)
    if df.empty:
        return None
    view = df[df["dataset"].astype(str) == str(dataset)].copy()
    if view.empty or "metric" not in view.columns:
        return None
    view = view[view["status_norm"] != "crash"].sort_values("metric", ascending=False)
    for _, row in view.iterrows():
        recipe = row_to_recipe(row)
        if recipe:
            label = str(row.get("recipe", "")).strip() or f"best_known_{slugify(row.get('backbone', 'candidate'))}"
            return label, recipe
    return None


def initialize_seed_candidates(
    search_dir: Path,
    dataset: str,
    space: dict,
    state: dict,
    candidates: list[dict],
    results_path: Path,
) -> None:
    if state.get("seed_initialized"):
        return
    sequence = int(state.get("sequence_counter", 0))
    candidate_counter = int(state.get("candidate_counter", 0))
    seen_labels = set()
    seed_specs: list[tuple[str, dict, str]] = []

    if bool(space.get("include_best_known_seed", True)):
        best_seed = best_known_seed(space, dataset, results_path)
        if best_seed is not None:
            label, recipe = best_seed
            seed_specs.append((label, recipe, "best_known"))
            seen_labels.add(label)

    for recipe_name in space.get("seed_recipes", []):
        if recipe_name in seen_labels:
            continue
        if tracked_recipe_exists(recipe_name):
            seed_specs.append((recipe_name, load_yaml(tracked_recipe_path(recipe_name)), "tracked_recipe"))
            seen_labels.add(recipe_name)

    for recipe_path in space.get("seed_recipe_files", []):
        resolved_path = resolve_seed_recipe_file(space, str(recipe_path))
        if not resolved_path.exists():
            continue
        label = resolved_path.stem
        if label in seen_labels:
            continue
        seed_specs.append((label, load_yaml(resolved_path), "seed_recipe_file"))
        seen_labels.add(label)

    for label, recipe, source in seed_specs:
        candidate_counter += 1
        sequence += 1
        candidate_id = f"cand{candidate_counter:04d}"
        branch_key = branch_key_for_backbone(str(recipe.get("model", {}).get("backbone", "")))
        recipe_name = build_recipe_stem(candidate_id, label, "seed")
        candidate = {
            "sequence_id": sequence,
            "candidate_id": candidate_id,
            "replicate_group": candidate_id,
            "parent_id": "",
            "branch_key": branch_key,
            "base_recipe": label,
            "recipe_name": recipe_name,
            "mutation_op": "seed",
            "mutation_delta": {"source": source},
            "search_algo": DEFAULT_ALGO,
            "status": "pending",
            "metric": None,
            "confirmed_metric": None,
            "improvement": None,
            "checkpoint_path": "",
            "backbone": str(recipe.get("model", {}).get("backbone", "")),
            "attempt_index": 0,
            "source": source,
        }
        recipe_payload = copy.deepcopy(recipe)
        recipe_payload["dataset"] = dataset
        recipe_payload["search_metadata"] = {
            "candidate_id": candidate_id,
            "parent_id": "",
            "search_algo": DEFAULT_ALGO,
            "mutation_op": "seed",
            "base_recipe": label,
            "replicate_group": candidate_id,
        }
        recipe_file = search_dir / "recipes" / f"{recipe_name}.yaml"
        dump_yaml(recipe_file, recipe_payload)
        candidate["recipe_file"] = str(recipe_file)
        candidate["effective_config_file"] = str(search_dir / "configs" / f"{candidate_id}.yaml")
        candidate["log_file"] = str(search_dir / "logs" / f"{candidate_id}.log")
        candidates.append(candidate)

    state["candidate_counter"] = candidate_counter
    state["sequence_counter"] = sequence
    state["seed_initialized"] = True


def confirmed_candidates(candidates: list[dict]) -> list[dict]:
    confirmed = []
    for candidate in candidates:
        if normalize_status(candidate.get("status")) != "keep":
            continue
        score = candidate_score(candidate)
        if math.isnan(score):
            continue
        confirmed.append(candidate)
    return confirmed


def refresh_hot_set(candidates: list[dict], state: dict, space: dict) -> None:
    per_branch = {}
    for candidate in confirmed_candidates(candidates):
        branch = candidate.get("branch_key", "")
        score = candidate_score(candidate)
        current = per_branch.get(branch)
        if current is None or score > current:
            per_branch[branch] = score
    hot_set_size = int(space.get("stagnation_rules", {}).get("hot_set_size", 4))
    ranked = sorted(per_branch.items(), key=lambda item: item[1], reverse=True)
    hot = [branch for branch, _ in ranked[:hot_set_size]]
    if len(hot) < hot_set_size:
        for backbone in space.get("backbone_pool", []):
            branch = branch_key_for_backbone(backbone)
            if branch not in hot:
                hot.append(branch)
            if len(hot) >= hot_set_size:
                break
    state["hot_set"] = hot


def select_parent(candidates: list[dict], state: dict, space: dict, rng: random.Random) -> dict:
    confirmed = confirmed_candidates(candidates)
    if not confirmed:
        pending = [candidate for candidate in candidates if normalize_status(candidate.get("status")) == "pending"]
        if pending:
            return pending[0]
        raise RuntimeError("No confirmed or pending candidates available for parent selection")

    exploit_parent = str(state.get("exploit_parent_id", "")).strip()
    if exploit_parent and int(state.get("exploit_remaining", 0)) > 0:
        for candidate in confirmed:
            if candidate.get("candidate_id") == exploit_parent:
                state["exploit_remaining"] = int(state.get("exploit_remaining", 0)) - 1
                if state["exploit_remaining"] <= 0:
                    state["exploit_parent_id"] = ""
                return candidate

    global_elite_size = int(space.get("stagnation_rules", {}).get("global_elite_size", 6))
    per_branch_limit = int(space.get("stagnation_rules", {}).get("per_branch_elite", 2))
    refresh_hot_set(candidates, state, space)

    ranked = sorted(confirmed, key=candidate_score, reverse=True)
    global_elite = ranked[:global_elite_size]

    branch_archives: dict[str, list[dict]] = {}
    for candidate in ranked:
        branch_archives.setdefault(candidate.get("branch_key", ""), [])
        if len(branch_archives[candidate.get("branch_key", "")]) < per_branch_limit:
            branch_archives[candidate.get("branch_key", "")].append(candidate)

    branch_runs = {branch: 0 for branch in state.get("hot_set", [])}
    for candidate in candidates:
        branch = candidate.get("branch_key", "")
        branch_runs[branch] = branch_runs.get(branch, 0) + 1
    underexplored_branch = None
    if branch_runs:
        underexplored_branch = min(branch_runs.items(), key=lambda item: (item[1], item[0]))[0]

    draw = rng.random()
    if draw < 0.6 and global_elite:
        return rng.choice(global_elite)
    if draw < 0.9 and underexplored_branch and branch_archives.get(underexplored_branch):
        return rng.choice(branch_archives[underexplored_branch])
    return rng.choice(ranked)


def make_child_candidate(
    search_dir: Path,
    state: dict,
    space: dict,
    dataset: str,
    parent: dict,
    recipe: dict,
    mutation_op: str,
    mutation_delta: dict,
    allow_large_jump: bool = False,
) -> dict:
    state["candidate_counter"] = int(state.get("candidate_counter", 0)) + 1
    state["sequence_counter"] = int(state.get("sequence_counter", 0)) + 1
    candidate_id = f"cand{int(state['candidate_counter']):04d}"
    recipe_name = build_recipe_stem(
        candidate_id,
        parent.get("candidate_id", ""),
        mutation_op,
    )
    branch_key = branch_key_for_backbone(str(recipe.get("model", {}).get("backbone", "")))
    candidate = {
        "sequence_id": int(state["sequence_counter"]),
        "candidate_id": candidate_id,
        "replicate_group": candidate_id,
        "parent_id": parent.get("candidate_id", ""),
        "branch_key": branch_key,
        "base_recipe": parent.get("recipe_name", parent.get("base_recipe", "")),
        "recipe_name": recipe_name,
        "mutation_op": mutation_op,
        "mutation_delta": mutation_delta,
        "search_algo": DEFAULT_ALGO,
        "status": "pending",
        "metric": None,
        "confirmed_metric": None,
        "improvement": None,
        "checkpoint_path": "",
        "backbone": str(recipe.get("model", {}).get("backbone", "")),
        "attempt_index": 0,
        "allow_large_jump": allow_large_jump,
    }
    recipe_payload = copy.deepcopy(recipe)
    recipe_payload["dataset"] = dataset
    recipe_payload["search_metadata"] = {
        "candidate_id": candidate_id,
        "parent_id": parent.get("candidate_id", ""),
        "search_algo": DEFAULT_ALGO,
        "mutation_op": mutation_op,
        "base_recipe": candidate["base_recipe"],
        "replicate_group": candidate_id,
    }
    recipe_file = search_dir / "recipes" / f"{recipe_name}.yaml"
    dump_yaml(recipe_file, recipe_payload)
    candidate["recipe_file"] = str(recipe_file)
    candidate["effective_config_file"] = str(search_dir / "configs" / f"{candidate_id}.yaml")
    candidate["log_file"] = str(search_dir / "logs" / f"{candidate_id}.log")
    return candidate


def run_candidate(
    candidate: dict,
    dataset: str,
    session: str,
    results_path: Path,
    time_budget: float | None = None,
    dry_run: bool = False,
) -> dict:
    if dry_run:
        candidate["status"] = "dry_run"
        return candidate

    log_path = Path(candidate["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPO_DIR / "train.py"),
        "--dataset",
        dataset,
        "--recipe-file",
        candidate["recipe_file"],
        "--session",
        session,
        "--save-checkpoint",
        "--dump-effective-config",
        candidate["effective_config_file"],
    ]
    if time_budget is not None:
        command.extend(["--time-budget", str(time_budget)])
    candidate["timestamp_started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    timeout_seconds = candidate_timeout_seconds(time_budget)
    with open(log_path, "w", encoding="utf-8") as handle:
        try:
            result = subprocess.run(
                command,
                cwd=REPO_DIR,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            handle.write(f"\n[search] candidate timed out after {int(timeout_seconds or 0)}s\n")
            candidate["timestamp_finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            candidate["returncode"] = 124
            candidate["status"] = "crash"
            candidate["metric"] = float("nan")
            candidate["failure_reason"] = f"subprocess_timeout_{int(timeout_seconds or 0)}s"
            return candidate
    candidate["timestamp_finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    candidate["returncode"] = result.returncode

    row = latest_result_for_candidate(results_path, session, candidate["candidate_id"])
    if row is None:
        candidate["status"] = "crash"
        candidate["metric"] = float("nan")
        candidate["failure_reason"] = f"returncode_{result.returncode}" if result.returncode != 0 else "missing_result_row"
        return candidate
    return apply_result_row_to_candidate(candidate, row)


def create_rerun_candidate(search_dir: Path, state: dict, parent_candidate: dict) -> dict:
    state["candidate_counter"] = int(state.get("candidate_counter", 0)) + 1
    state["sequence_counter"] = int(state.get("sequence_counter", 0)) + 1
    rerun_id = f"cand{int(state['candidate_counter']):04d}"
    recipe_name = build_recipe_stem(
        rerun_id,
        parent_candidate.get("candidate_id", ""),
        "confirm",
    )
    recipe_file = search_dir / "recipes" / f"{recipe_name}.yaml"
    recipe_payload = load_yaml(Path(parent_candidate["recipe_file"]))
    recipe_payload["search_metadata"] = {
        "candidate_id": rerun_id,
        "parent_id": parent_candidate.get("parent_id", ""),
        "search_algo": DEFAULT_ALGO,
        "mutation_op": parent_candidate.get("mutation_op", ""),
        "base_recipe": parent_candidate.get("base_recipe", ""),
        "replicate_group": parent_candidate.get("replicate_group", parent_candidate["candidate_id"]),
    }
    dump_yaml(recipe_file, recipe_payload)
    return {
        "sequence_id": int(state["sequence_counter"]),
        "candidate_id": rerun_id,
        "replicate_group": parent_candidate.get("replicate_group", parent_candidate["candidate_id"]),
        "parent_id": parent_candidate.get("parent_id", ""),
        "branch_key": parent_candidate.get("branch_key", ""),
        "base_recipe": parent_candidate.get("base_recipe", ""),
        "recipe_name": recipe_name,
        "mutation_op": parent_candidate.get("mutation_op", ""),
        "mutation_delta": parent_candidate.get("mutation_delta", {}),
        "search_algo": DEFAULT_ALGO,
        "status": "pending",
        "metric": None,
        "confirmed_metric": None,
        "improvement": None,
        "checkpoint_path": "",
        "backbone": parent_candidate.get("backbone", ""),
        "attempt_index": int(parent_candidate.get("attempt_index", 0)) + 1,
        "recipe_file": str(recipe_file),
        "effective_config_file": str(search_dir / "configs" / f"{rerun_id}.yaml"),
        "log_file": str(search_dir / "logs" / f"{rerun_id}.log"),
    }


def finalize_candidate(
    search_dir: Path,
    candidate: dict,
    candidates: list[dict],
    state: dict,
    space: dict,
    results_path: Path,
    dataset: str,
    session: str,
    time_budget: float | None,
    dry_run: bool,
) -> dict:
    thresholds = space.get("gating_thresholds", {})
    immediate_keep_delta = float(thresholds.get("immediate_keep_delta", 0.25))
    confirm_delta = float(thresholds.get("confirm_delta", 0.10))
    confirmed_mean_delta = float(thresholds.get("confirmed_mean_delta", 0.15))

    prior_best = state.get("best_confirmed_metric")
    prior_best = float(prior_best) if prior_best is not None else None
    metric = safe_float(candidate.get("metric"))
    if prior_best is None or math.isnan(prior_best):
        delta = float("inf")
    else:
        delta = metric - prior_best
    candidate["improvement"] = None if math.isinf(delta) else delta

    if dry_run:
        return candidate

    if normalize_status(candidate.get("status")) == "crash" or math.isnan(metric):
        candidate["status"] = "crash"
        state["non_improving_runs"] = int(state.get("non_improving_runs", 0)) + 1
        return candidate

    final_status = "discard"
    confirmed_metric = metric
    updated_candidate_ids = [candidate["candidate_id"]]
    note = None
    branch_has_keeper = any(
        normalize_status(other.get("status")) == "keep"
        and other.get("branch_key") == candidate.get("branch_key")
        and other.get("candidate_id") != candidate.get("candidate_id")
        for other in candidates
    )

    if candidate.get("mutation_op") == "seed" and not branch_has_keeper:
        final_status = "keep"
    elif prior_best is None or math.isinf(delta) or delta >= immediate_keep_delta:
        final_status = "keep"
    elif delta >= confirm_delta:
        rerun = create_rerun_candidate(search_dir, state, candidate)
        candidates.append(rerun)
        update_candidate_record(search_dir, candidates)
        run_candidate(rerun, dataset, session, results_path, time_budget=time_budget, dry_run=False)
        rerun = candidates[-1]
        rerun_metric = safe_float(rerun.get("metric"))
        updated_candidate_ids.append(rerun["candidate_id"])
        confirmed_metric = (metric + rerun_metric) / 2.0 if not math.isnan(rerun_metric) else metric
        note = f"confirm_mean={confirmed_metric:.6f}"
        if prior_best is None or confirmed_metric - prior_best >= confirmed_mean_delta:
            final_status = "keep"
        else:
            final_status = "discard"
        rerun["status"] = final_status
        rerun["confirmed_metric"] = confirmed_metric
    else:
        final_status = "discard"

    candidate["status"] = final_status
    candidate["confirmed_metric"] = confirmed_metric
    update_results_status(results_path, updated_candidate_ids, final_status, note=note)

    if final_status == "keep":
        best_metric = state.get("best_confirmed_metric")
        improved_global = best_metric is None or confirmed_metric > float(best_metric)
        if improved_global:
            state["best_confirmed_metric"] = confirmed_metric
            state["best_confirmed_candidate_id"] = candidate["candidate_id"]
            state["non_improving_runs"] = 0
        else:
            state["non_improving_runs"] = int(state.get("non_improving_runs", 0)) + 1
        parent = next((item for item in candidates if item.get("candidate_id") == candidate.get("parent_id")), None)
        parent_metric = candidate_score(parent) if parent is not None else float("nan")
        if improved_global or (
            candidate.get("mutation_op") != "seed"
            and not math.isnan(parent_metric)
            and confirmed_metric > parent_metric
        ):
            state["exploit_parent_id"] = candidate["candidate_id"]
            state["exploit_remaining"] = int(space.get("stagnation_rules", {}).get("exploit_children", 3))
    else:
        state["non_improving_runs"] = int(state.get("non_improving_runs", 0)) + 1

    return candidate


def pick_pending_candidate(candidates: list[dict]) -> dict | None:
    pending = [candidate for candidate in candidates if normalize_status(candidate.get("status")) == "pending"]
    if not pending:
        return None
    pending.sort(key=lambda item: int(item.get("sequence_id", 0)))
    return pending[0]


def applicable_families(space: dict, recipe: dict) -> list[str]:
    return filtered_families_for_phase(space, recipe)


def filtered_families_for_phase(space: dict, recipe: dict, force_explore: bool = False, allow_large_jump: bool = False) -> list[str]:
    families = []
    task = str(recipe.get("task", space.get("task", ""))).lower()
    allowed = set(family_pool_for_phase(space, force_explore=force_explore, allow_large_jump=allow_large_jump))
    for family in space.get("mutation_families", []):
        if family not in allowed:
            continue
        if family == "loss" and task == "segmentation":
            continue
        families.append(family)
    return families


def operator_stats_template(space: dict) -> dict:
    return {family: {"trials": 0, "reward_sum": 0.0} for family in space.get("mutation_families", [])}


def bounded_operator_reward(parent_metric: float, child_metric: float, candidate: dict, space: dict) -> float:
    reward_cfg = space.get("operator_reward", {})
    clip = abs(float(reward_cfg.get("clip", 2.0)))
    crash_penalty = float(reward_cfg.get("crash_penalty", -1.5))
    status = normalize_status(candidate.get("status"))
    if status == "crash" or math.isnan(child_metric):
        reward = crash_penalty
    elif math.isnan(parent_metric):
        reward = 0.0
    else:
        reward = child_metric - parent_metric
    if clip > 0:
        reward = max(-clip, min(clip, reward))
    return reward


def main():
    parser = argparse.ArgumentParser(description="Run an adaptive search session")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset override; defaults to the search-space dataset")
    parser.add_argument("--session", type=str, required=True, help="Session tag for logs/results")
    parser.add_argument("--space", type=str, default="chestxray14", help="Search space name or explicit YAML path")
    parser.add_argument("--results", type=Path, default=RESULTS_FILE, help="Path to results.tsv")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Maximum finalized search candidates for this session; confirmation reruns may add extra train.py executions",
    )
    parser.add_argument("--time-budget", type=float, default=None, help="Optional time budget override passed through to train.py")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for the outer loop")
    parser.add_argument("--dry-run", action="store_true", help="Initialize/search-plan only without launching train.py")
    args = parser.parse_args()

    space = load_search_space(args.space)
    dataset = args.dataset or space.get("dataset")
    if not dataset:
        raise ValueError("Search space must define a dataset or pass --dataset")
    max_runs = args.max_runs if args.max_runs is not None else int(space.get("default_max_runs", 16))
    session_dir = REPO_DIR / "logs" / args.session / "search"
    session_dir.mkdir(parents=True, exist_ok=True)

    state, candidates = load_state(session_dir)
    if not state:
        state = {
            "search_algo": DEFAULT_ALGO,
            "session": args.session,
            "dataset": dataset,
            "space_path": space.get("_path", ""),
            "candidate_counter": 0,
            "sequence_counter": 0,
            "completed_runs": 0,
            "non_improving_runs": 0,
            "best_confirmed_metric": None,
            "best_confirmed_candidate_id": "",
            "seed_initialized": False,
            "exploit_parent_id": "",
            "exploit_remaining": 0,
            "operator_stats": operator_stats_template(space),
            "hot_set": [],
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    else:
        state.setdefault("operator_stats", operator_stats_template(space))
        for family in space.get("mutation_families", []):
            state["operator_stats"].setdefault(family, {"trials": 0, "reward_sum": 0.0})
        state["dataset"] = dataset

    rng = random.Random(args.seed)
    initialize_seed_candidates(session_dir, dataset, space, state, candidates, args.results)
    update_candidate_record(session_dir, candidates)
    save_state(session_dir, state)
    generate_search_report(args.session, results_path=args.results, search_dir=session_dir)

    if args.dry_run:
        pending = [candidate["recipe_name"] for candidate in candidates if normalize_status(candidate.get("status")) == "pending"]
        print(f"Initialized search session: {args.session}")
        print(f"Dataset: {dataset}")
        print(f"Pending candidates: {len(pending)}")
        for name in pending:
            print(f"  - {name}")
        return

    while int(state.get("completed_runs", 0)) < max_runs:
        pending = pick_pending_candidate(candidates)
        resumed_from_results = False
        if pending is not None:
            resumed_from_results = sync_candidate_from_results(args.results, args.session, pending)
        if pending is None:
            force_explore = int(state.get("non_improving_runs", 0)) >= int(space.get("stagnation_rules", {}).get("explore_after", 8))
            allow_large_jump = int(state.get("non_improving_runs", 0)) >= int(space.get("stagnation_rules", {}).get("refresh_after", 16))
            if allow_large_jump:
                refresh_hot_set(candidates, state, space)
            parent = select_parent(candidates, state, space, rng)
            base_recipe = load_yaml(Path(parent["recipe_file"]))
            base_recipe.pop("search_metadata", None)
            families = filtered_families_for_phase(
                space,
                base_recipe,
                force_explore=force_explore,
                allow_large_jump=allow_large_jump,
            )
            family = select_family(families, state["operator_stats"], rng, force_explore=force_explore)
            child_recipe, mutation_delta = mutate_recipe(base_recipe, space, family, rng, allow_large_jump=allow_large_jump)
            child = make_child_candidate(
                search_dir=session_dir,
                state=state,
                space=space,
                dataset=dataset,
                parent=parent,
                recipe=child_recipe,
                mutation_op=family,
                mutation_delta=mutation_delta,
                allow_large_jump=allow_large_jump,
            )
            candidates.append(child)
            update_candidate_record(session_dir, candidates)
            save_state(session_dir, state)
            pending = child

        if resumed_from_results:
            print(f"[search] resuming finalized run {pending['candidate_id']} {pending['recipe_name']}")
        else:
            print(f"[search] running {pending['candidate_id']} {pending['recipe_name']}")
            run_candidate(pending, dataset, args.session, args.results, time_budget=args.time_budget, dry_run=False)
        if pending.get("parent_id"):
            parent = next((candidate for candidate in candidates if candidate.get("candidate_id") == pending["parent_id"]), None)
            parent_metric = candidate_score(parent) if parent is not None else float("nan")
            child_metric = safe_float(pending.get("metric"))
            reward = bounded_operator_reward(parent_metric, child_metric, pending, space)
            stat = state["operator_stats"].setdefault(pending["mutation_op"], {"trials": 0, "reward_sum": 0.0})
            stat["trials"] = int(stat.get("trials", 0)) + 1
            stat["reward_sum"] = float(stat.get("reward_sum", 0.0)) + reward
        pending = finalize_candidate(
            search_dir=session_dir,
            candidate=pending,
            candidates=candidates,
            state=state,
            space=space,
            results_path=args.results,
            dataset=dataset,
            session=args.session,
            time_budget=args.time_budget,
            dry_run=False,
        )
        state["completed_runs"] = int(state.get("completed_runs", 0)) + 1
        update_candidate_record(session_dir, candidates)
        save_state(session_dir, state)
        generate_search_report(args.session, results_path=args.results, search_dir=session_dir)

    print("Adaptive search session complete:")
    print(f"  session: {args.session}")
    print(f"  dataset: {dataset}")
    print(f"  completed runs: {state.get('completed_runs', 0)}")
    if state.get("best_confirmed_metric") is not None:
        print(f"  best confirmed metric: {state['best_confirmed_metric']:.6f}")
        print(f"  best confirmed candidate: {state.get('best_confirmed_candidate_id', '')}")
    print(f"  search dir: {session_dir}")


if __name__ == "__main__":
    main()
