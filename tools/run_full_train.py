"""
Promote a winning search recipe into a longer epoch-budgeted full-train run.

This keeps the default autoresearch search loop time-budgeted while offering a
clean second stage for longer showcase or refinement training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_DIR = Path(__file__).resolve().parent.parent
RESULTS_FILE = REPO_DIR / "results.tsv"
RECIPES_DIR = REPO_DIR / "recipes"
TRAIN_SCRIPT = REPO_DIR / "train.py"


def normalize_status(value: object) -> str:
    return str(value or "").strip().lower()


def read_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"results.tsv not found at: {path}")
    df = pd.read_csv(path, sep="\t")
    if "metric" in df.columns:
        df["metric"] = pd.to_numeric(df["metric"], errors="coerce")
    if "status" in df.columns:
        df["status_norm"] = df["status"].map(normalize_status)
    else:
        df["status_norm"] = ""
    return df


def filter_results(
    df: pd.DataFrame,
    dataset: str | None = None,
    session: str | None = None,
    candidate_id: str | None = None,
) -> pd.DataFrame:
    view = df.copy()
    if dataset:
        view = view[view["dataset"].astype(str) == str(dataset)]
    if session:
        if "session" not in view.columns:
            raise ValueError("results.tsv does not contain a 'session' column")
        view = view[view["session"].fillna("").astype(str) == str(session)]
    if candidate_id:
        if "candidate_id" not in view.columns:
            raise ValueError("results.tsv does not contain a 'candidate_id' column")
        view = view[view["candidate_id"].fillna("").astype(str) == str(candidate_id)]
    if view.empty:
        raise ValueError("No matching rows found in results.tsv")
    return view


def select_winner_row(df: pd.DataFrame) -> pd.Series:
    metric_df = df.copy()
    if "metric" in metric_df.columns:
        metric_df = metric_df[metric_df["metric"].notna()].copy()
    if "status_norm" in metric_df.columns:
        keep_df = metric_df[metric_df["status_norm"] == "keep"].copy()
        if not keep_df.empty:
            metric_df = keep_df
        else:
            non_crash = metric_df[metric_df["status_norm"] != "crash"].copy()
            if not non_crash.empty:
                metric_df = non_crash
    if metric_df.empty:
        raise ValueError("No non-crash winner candidates found")
    if "metric" not in metric_df.columns:
        return metric_df.iloc[-1]
    return metric_df.sort_values("metric", ascending=False).iloc[0]


def read_candidates(search_dir: Path) -> list[dict]:
    candidates_file = search_dir / "candidates.jsonl"
    if not candidates_file.exists():
        return []
    candidates: list[dict] = []
    with open(candidates_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    return candidates


def load_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_session_candidate(search_dir: Path, winner: pd.Series) -> dict | None:
    candidates = read_candidates(search_dir)
    candidate_id = str(winner.get("candidate_id", "") or "").strip()
    recipe_name = str(winner.get("recipe", "") or "").strip()
    for candidate in candidates:
        if candidate_id and str(candidate.get("candidate_id", "")).strip() == candidate_id:
            return candidate
    for candidate in candidates:
        if recipe_name and str(candidate.get("recipe_name", "")).strip() == recipe_name:
            return candidate
    return None


def sanitize_recipe_file(source_path: Path, output_path: Path) -> Path:
    with open(source_path, "r", encoding="utf-8") as handle:
        recipe = yaml.safe_load(handle) or {}
    recipe.pop("search_metadata", None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(recipe, handle, sort_keys=False)
    return output_path


def resolve_full_train_source(
    full_train_dir: Path,
    output_dir: Path,
    workspace_root: Path,
) -> tuple[str, str] | None:
    selection_path = full_train_dir / "selection.json"
    selection = load_json_file(selection_path)
    source_kind = str(selection.get("source_kind", "") or "").strip()
    source_value = str(selection.get("source_value", "") or "").strip()

    if source_kind == "recipe-file" and source_value:
        recipe_path = Path(source_value).expanduser()
        if not recipe_path.is_absolute():
            recipe_path = (full_train_dir / recipe_path).resolve()
        if recipe_path.exists():
            sanitized = sanitize_recipe_file(recipe_path, output_dir / recipe_path.name)
            return "recipe-file", str(sanitized)

    if source_kind == "recipe" and source_value:
        tracked_recipe = workspace_root / "recipes" / f"{source_value}.yaml"
        if tracked_recipe.exists():
            return "recipe", source_value

    yaml_files = sorted(full_train_dir.glob("*.yaml"))
    if len(yaml_files) == 1:
        sanitized = sanitize_recipe_file(yaml_files[0], output_dir / yaml_files[0].name)
        return "recipe-file", str(sanitized)

    return None


def resolve_recipe_source(
    winner: pd.Series | None,
    session: str | None,
    output_dir: Path,
    workspace_root: Path,
) -> tuple[str, str]:
    if winner is None:
        raise ValueError("winner row is required to resolve a session recipe")
    recipe_name = str(winner.get("recipe", "") or "").strip()
    if not recipe_name:
        raise ValueError("Winner row does not contain a recipe name")

    if session:
        search_dir = workspace_root / "logs" / session / "search"
        candidate = resolve_session_candidate(search_dir, winner)
        if candidate:
            recipe_file = Path(str(candidate.get("recipe_file", "") or ""))
            if recipe_file.exists():
                sanitized = sanitize_recipe_file(recipe_file, output_dir / f"{recipe_name}.yaml")
                return "recipe-file", str(sanitized)
        generated_recipe = search_dir / "recipes" / f"{recipe_name}.yaml"
        if generated_recipe.exists():
            sanitized = sanitize_recipe_file(generated_recipe, output_dir / f"{recipe_name}.yaml")
            return "recipe-file", str(sanitized)
        full_train_dir = workspace_root / "logs" / session / "full_train"
        resolved_full_train = resolve_full_train_source(full_train_dir, output_dir, workspace_root)
        if resolved_full_train is not None:
            return resolved_full_train

    tracked_recipe = workspace_root / "recipes" / f"{recipe_name}.yaml"
    if tracked_recipe.exists():
        return "recipe", recipe_name
    raise FileNotFoundError(f"Could not resolve a runnable recipe for '{recipe_name}'")


def default_train_session(base_name: str, epochs: int) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(base_name or "fulltrain"))
    safe = safe.strip("_") or "fulltrain"
    return f"{safe}_fulltrain_ep{epochs}"


def resolve_workspace_python(workspace_root: Path) -> Path:
    candidates = [
        workspace_root / ".venv-gpu" / "Scripts" / "python.exe",
        workspace_root / ".venv" / "Scripts" / "python.exe",
        workspace_root / ".venv-gpu" / "bin" / "python",
        workspace_root / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def resolve_python_executable(workspace_root: Path, explicit_python: str | None = None) -> Path:
    if explicit_python:
        python_path = Path(explicit_python).expanduser().resolve()
        if not python_path.exists():
            raise FileNotFoundError(f"Python executable not found: {python_path}")
        return python_path
    return resolve_workspace_python(workspace_root)


def build_command(
    python_executable: Path,
    train_script: Path,
    source_kind: str,
    source_value: str,
    epochs: int,
    train_session: str,
    dataset: str | None,
    dump_effective_config: Path | None,
    save_checkpoint: bool = True,
) -> list[str]:
    command = [str(python_executable), str(train_script)]
    if dataset:
        command.extend(["--dataset", str(dataset)])
    command.extend(["--session", train_session, "--epochs", str(epochs)])
    if source_kind == "recipe":
        command.extend(["--recipe", source_value])
    elif source_kind == "recipe-file":
        command.extend(["--recipe-file", source_value])
    else:
        raise ValueError(f"Unknown source kind: {source_kind}")
    if save_checkpoint:
        command.append("--save-checkpoint")
    if dump_effective_config is not None:
        command.extend(["--dump-effective-config", str(dump_effective_config)])
    return command


def launcher_paths(logs_dir: Path) -> tuple[Path, Path]:
    return logs_dir / "launcher.log", logs_dir / "launcher.err.log"


def main():
    parser = argparse.ArgumentParser(description="Run a longer epoch-budgeted train for the best discovered recipe")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--session", type=str, default=None, help="Reuse the best row from a prior search or queue session")
    source_group.add_argument("--recipe", type=str, default=None, help="Tracked recipe name under recipes/")
    source_group.add_argument("--recipe-file", type=str, default=None, help="Explicit YAML recipe path")
    parser.add_argument("--results", type=str, default=str(RESULTS_FILE), help="Path to results.tsv")
    parser.add_argument("--workspace-root", type=str, default=None, help="Optional workspace root containing train.py, recipes/, logs/, and results.tsv")
    parser.add_argument("--python-executable", type=str, default=None, help="Optional Python interpreter to use for the underlying train.py run")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset filter/inference override")
    parser.add_argument("--candidate-id", type=str, default=None, help="Optional specific candidate_id from the supplied session instead of the session winner")
    parser.add_argument("--epochs", type=int, required=True, help="Epoch budget for the full-train stage")
    parser.add_argument("--train-session", type=str, default=None, help="Session tag for the promoted full-train run")
    parser.add_argument("--dump-effective-config", type=str, default=None, help="Optional explicit effective-config output path")
    parser.add_argument("--no-save-checkpoint", action="store_true", help="Do not save a checkpoint for the promoted run")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without launching train.py")
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.candidate_id and not args.session:
        raise ValueError("--candidate-id requires --session")

    results_path = Path(args.results).expanduser().resolve()
    workspace_root = (
        Path(args.workspace_root).expanduser().resolve()
        if args.workspace_root
        else results_path.parent
    )
    train_script = workspace_root / "train.py"
    python_executable = resolve_python_executable(workspace_root, explicit_python=args.python_executable)
    if not train_script.exists():
        raise FileNotFoundError(f"train.py not found under workspace root: {workspace_root}")

    logs_dir: Path | None = None
    source_kind = ""
    source_value = ""
    dataset = args.dataset
    winner: pd.Series | None = None

    if args.session:
        df = read_results(results_path)
        filtered = filter_results(df, dataset=args.dataset, session=args.session, candidate_id=args.candidate_id)
        winner = filtered.iloc[-1] if args.candidate_id else select_winner_row(filtered)
        dataset = dataset or str(winner.get("dataset", "") or "").strip() or None
        base_session_name = args.candidate_id or args.session
        train_session = args.train_session or default_train_session(base_session_name, args.epochs)
        logs_dir = workspace_root / "logs" / train_session / "full_train"
        source_kind, source_value = resolve_recipe_source(winner, args.session, logs_dir, workspace_root)
    elif args.recipe:
        source_kind = "recipe"
        source_value = args.recipe
        train_session = args.train_session or default_train_session(args.recipe, args.epochs)
        logs_dir = workspace_root / "logs" / train_session / "full_train"
    else:
        recipe_path = Path(args.recipe_file).expanduser().resolve()
        if not recipe_path.exists():
            raise FileNotFoundError(f"Recipe file not found: {recipe_path}")
        train_session = args.train_session or default_train_session(recipe_path.stem, args.epochs)
        logs_dir = workspace_root / "logs" / train_session / "full_train"
        sanitized = sanitize_recipe_file(recipe_path, logs_dir / f"{recipe_path.stem}.yaml")
        source_kind = "recipe-file"
        source_value = str(sanitized)

    logs_dir.mkdir(parents=True, exist_ok=True)
    dump_effective_config = (
        Path(args.dump_effective_config).expanduser().resolve()
        if args.dump_effective_config
        else logs_dir / "effective_config.json"
    )
    plan_path = logs_dir / "selection.json"
    plan_payload = {
        "source_kind": source_kind,
        "source_value": source_value,
        "dataset": dataset,
        "epochs": args.epochs,
        "train_session": train_session,
        "python_executable": str(python_executable),
        "winner": winner.to_dict() if winner is not None else None,
    }
    plan_path.write_text(json.dumps(plan_payload, indent=2, sort_keys=True), encoding="utf-8")

    command = build_command(
        python_executable=python_executable,
        train_script=train_script,
        source_kind=source_kind,
        source_value=source_value,
        epochs=args.epochs,
        train_session=train_session,
        dataset=dataset,
        dump_effective_config=dump_effective_config,
        save_checkpoint=not args.no_save_checkpoint,
    )

    print("Promoted full-train run:")
    print(f"  session: {train_session}")
    if dataset:
        print(f"  dataset: {dataset}")
    print(f"  source: {source_kind}={source_value}")
    print(f"  epochs: {args.epochs}")
    print(f"  plan: {plan_path}")
    print(f"  command: {' '.join(command)}")

    if args.dry_run:
        return

    launcher_log, launcher_err = launcher_paths(logs_dir)
    preamble = [
        "Promoted full-train run:",
        f"  session: {train_session}",
        f"  dataset: {dataset}" if dataset else None,
        f"  source: {source_kind}={source_value}",
        f"  epochs: {args.epochs}",
        f"  plan: {plan_path}",
        f"  command: {' '.join(command)}",
        "",
    ]
    with open(launcher_log, "w", encoding="utf-8") as log_handle, open(launcher_err, "w", encoding="utf-8") as err_handle:
        log_handle.write("\n".join(line for line in preamble if line is not None))
        result = subprocess.run(command, cwd=str(workspace_root), stdout=log_handle, stderr=err_handle)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
