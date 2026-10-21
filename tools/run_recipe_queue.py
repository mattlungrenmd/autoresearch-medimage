"""
Run a fixed recipe queue inside a single session and mark keep/discard automatically.

This is an outer-loop helper for local or Colab sessions when you want a
reproducible architecture-first sweep instead of a purely manual one-run-at-a-time
agent workflow.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import train as train_mod
from tools.result_utils import normalize_status

DEFAULT_RESULTS = REPO_DIR / "results.tsv"
DEFAULT_LOG_ROOT = REPO_DIR / "logs"

PRESETS = {
    "chestxray14_backbone_frontier": [
        "chestxray14_frontier_densenet121",
        "chestxray14_frontier_densenet169",
        "chestxray14_frontier_resnet50",
        "chestxray14_frontier_convnext_tiny",
        "chestxray14_frontier_efficientnet_b0",
        "chestxray14_frontier_efficientnet_b1",
    ],
    "chestxray14_local_overnight": [
        "chestxray14_baseline",
        "chestxray14_advanced",
        "chestxray14_frontier_densenet121",
        "chestxray14_frontier_densenet169",
        "chestxray14_frontier_resnet50",
        "chestxray14_frontier_convnext_tiny",
        "chestxray14_frontier_efficientnet_b0",
        "chestxray14_frontier_efficientnet_b1",
    ],
    "chestxray14_3060_overnight": [
        "chestxray14_baseline",
        "chestxray14_advanced",
        "chestxray14_frontier_densenet121",
        "chestxray14_frontier_densenet169",
        "chestxray14_frontier_resnet50",
        "chestxray14_frontier_convnext_tiny",
        "chestxray14_frontier_efficientnet_b0",
        "chestxray14_frontier_efficientnet_b1",
    ],
}

TERMINAL_STATUSES = {"keep", "discard", "candidate"}


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def recipe_exists(recipe_name: str) -> bool:
    try:
        resolved = train_mod.resolve_recipe_name(recipe_name)
    except Exception:
        resolved = str(recipe_name).strip()
    recipe_path = REPO_DIR / "recipes" / f"{resolved}.yaml"
    return recipe_path.exists()


def validate_recipes_exist(recipes: list[str]) -> None:
    missing = [recipe for recipe in recipes if not recipe_exists(recipe)]
    if missing:
        raise FileNotFoundError(
            "Queue references recipe files that are not present: " + ", ".join(sorted(missing))
        )


def read_results(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(results_path, sep="\t")
    if "metric" in df.columns:
        df["metric"] = pd.to_numeric(df["metric"], errors="coerce")
    return df


def filter_session_rows(df: pd.DataFrame, dataset: str, session: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    view = df.copy()
    if "dataset" in view.columns:
        view = view[view["dataset"].astype(str) == str(dataset)]
    if "session" in view.columns:
        view = view[view["session"].fillna("").astype(str) == str(session)]
    else:
        return pd.DataFrame()
    return view


def latest_rows_by_recipe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "recipe" not in df.columns:
        return pd.DataFrame(columns=df.columns)
    view = df.reset_index().rename(columns={"index": "_row_index"})
    latest = view.groupby("recipe", as_index=False, sort=False).tail(1)
    return latest.sort_values("_row_index")


def completed_recipes(df: pd.DataFrame, dataset: str, session: str) -> set[str]:
    session_rows = latest_rows_by_recipe(filter_session_rows(df, dataset, session))
    if session_rows.empty:
        return set()
    metric_names = session_rows.get("metric_name", pd.Series(dtype=str)).astype(str)
    statuses = session_rows.get("status", pd.Series(dtype=str)).map(normalize_status)
    keepable = statuses.isin(TERMINAL_STATUSES)
    non_error = metric_names.str.lower() != "error"
    return set(session_rows.loc[keepable & non_error, "recipe"].astype(str))


def current_best_metric(df: pd.DataFrame, dataset: str, session: str) -> float | None:
    session_rows = filter_session_rows(df, dataset, session)
    if session_rows.empty or "metric" not in session_rows.columns:
        return None

    statuses = session_rows.get("status", pd.Series(dtype=str)).map(normalize_status)
    keep_rows = session_rows[statuses == "keep"]
    if not keep_rows.empty:
        value = keep_rows["metric"].max()
        return None if pd.isna(value) else float(value)

    candidate_rows = session_rows[statuses.isin({"candidate", "keep"})]
    if not candidate_rows.empty:
        value = candidate_rows["metric"].max()
        return None if pd.isna(value) else float(value)
    return None


def latest_row_for_recipe(df: pd.DataFrame, dataset: str, session: str, recipe: str) -> pd.Series | None:
    session_rows = filter_session_rows(df, dataset, session)
    if session_rows.empty:
        return None
    recipe_rows = session_rows[session_rows["recipe"].astype(str) == str(recipe)]
    if recipe_rows.empty:
        return None
    return recipe_rows.iloc[-1]


def update_row_status(
    results_path: Path,
    dataset: str,
    session: str,
    recipe: str,
    status: str,
    note: str | None = None,
) -> pd.Series:
    df = pd.read_csv(results_path, sep="\t")
    session_rows = filter_session_rows(df, dataset, session)
    recipe_rows = session_rows[session_rows["recipe"].astype(str) == str(recipe)]
    if recipe_rows.empty:
        raise ValueError(f"No matching row found for recipe '{recipe}' in session '{session}'")
    idx = recipe_rows.index[-1]
    df.at[idx, "status"] = status

    if note:
        existing = str(df.at[idx, "description"]) if "description" in df.columns else ""
        if existing and existing.lower() != "nan":
            df.at[idx, "description"] = f"{existing} | {note}"
        elif "description" in df.columns:
            df.at[idx, "description"] = note

    df.to_csv(results_path, sep="\t", index=False)
    return df.loc[idx]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_hardware_snapshot(log_dir: Path) -> None:
    lines = [
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"python: {sys.executable}",
    ]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines.append(f"gpu: {result.stdout.strip()}")
    except FileNotFoundError:
        lines.append("gpu: nvidia-smi not found")

    write_text(log_dir / "hardware.txt", "\n".join(lines) + "\n")


def write_queue_manifest(log_dir: Path, preset_name: str | None, recipes: list[str], args: argparse.Namespace) -> None:
    lines = [
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"session: {args.session}",
        f"dataset: {args.dataset}",
        f"preset: {preset_name or ''}",
        f"resume: {str(not args.no_resume).lower()}",
        f"min_improvement: {args.min_improvement}",
        "",
        "recipes:",
    ]
    lines.extend(f"- {recipe}" for recipe in recipes)
    write_text(log_dir / "queue.txt", "\n".join(lines) + "\n")


def build_summary_markdown(
    df: pd.DataFrame,
    dataset: str,
    session: str,
    preset_name: str | None,
    completed: int,
    total: int,
    log_dir: Path,
) -> str:
    session_rows = filter_session_rows(df, dataset, session)
    latest = latest_rows_by_recipe(session_rows)
    latest = latest.copy()
    if not latest.empty:
        latest["status_norm"] = latest.get("status", pd.Series(dtype=str)).map(normalize_status)
        latest = latest.sort_values("metric", ascending=False, na_position="last")

    lines = [
        f"# Session Summary: {session}",
        "",
        f"- dataset: `{dataset}`",
        f"- preset: `{preset_name or 'custom'}`",
        f"- completed queue items: `{completed}/{total}`",
        f"- log directory: `{log_dir}`",
    ]

    best_metric = current_best_metric(df, dataset, session)
    if best_metric is not None:
        lines.append(f"- current best metric: `{best_metric:.6f}`")

    lines.extend(["", "## Latest Per Recipe", "", "| recipe | backbone | metric | status | checkpoint |", "|---|---|---:|---|---|"])

    if latest.empty:
        lines.append("| (none) |  |  |  |  |")
    else:
        for _, row in latest.head(15).iterrows():
            lines.append(
                "| {recipe} | {backbone} | {metric} | {status} | {checkpoint} |".format(
                    recipe=str(row.get("recipe", "")),
                    backbone=str(row.get("backbone", "")),
                    metric="" if pd.isna(row.get("metric")) else f"{float(row.get('metric')):.6f}",
                    status=str(row.get("status", "")),
                    checkpoint=str(row.get("checkpoint_path", "")),
                )
            )
    lines.append("")
    return "\n".join(lines)


def run_and_log(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"cwd: {REPO_DIR}\n")
        handle.write(f"command: {' '.join(command)}\n")
        handle.write(f"started_at: {datetime.now().isoformat(timespec='seconds')}\n\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=REPO_DIR,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
        )
        handle.write(f"\nfinished_at: {datetime.now().isoformat(timespec='seconds')}\n")
        handle.write(f"return_code: {result.returncode}\n")
    return result.returncode


def resolve_recipe_queue(args: argparse.Namespace) -> tuple[list[str], str | None]:
    if args.list_presets:
        return [], None

    if bool(args.preset) == bool(args.recipes):
        raise ValueError("Choose exactly one of --preset or --recipes")

    if args.preset:
        recipes = list(PRESETS[args.preset])
        validate_recipes_exist(recipes)
        return recipes, args.preset
    recipes = list(args.recipes)
    validate_recipes_exist(recipes)
    return recipes, None


def finalize_session(args: argparse.Namespace, log_dir: Path) -> None:
    command = [
        sys.executable,
        "-u",
        str(REPO_DIR / "tools" / "finalize_run.py"),
        "--results",
        str(args.results),
        "--dataset",
        args.dataset,
        "--session",
        args.session,
    ]
    if args.compile_paper:
        command.append("--compile-paper")
    run_and_log(command, log_dir / "finalize.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fixed recipe queue inside one session")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Named queue preset")
    parser.add_argument("--recipes", nargs="+", help="Explicit recipe queue")
    parser.add_argument("--dataset", default="chestxray14", help="Dataset name")
    parser.add_argument("--session", help="Session tag")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Path to results.tsv")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT, help="Root directory for per-session logs")
    parser.add_argument("--min-improvement", type=float, default=0.0, help="Minimum metric gain required to mark a new run as keep")
    parser.add_argument("--finalize", action="store_true", help="Run tools/finalize_run.py at the end")
    parser.add_argument("--compile-paper", action="store_true", help="Pass --compile-paper through during finalization")
    parser.add_argument("--dry-run", action="store_true", help="Print the queue without running training")
    parser.add_argument("--list-presets", action="store_true", help="List available presets")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip recipes already completed in the same session")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately if a recipe crashes or fails")
    args = parser.parse_args()

    if args.list_presets:
        print("Available presets:")
        for name, recipes in PRESETS.items():
            print(f"- {name} ({len(recipes)} recipes)")
        return

    if not args.session:
        raise ValueError("--session is required unless --list-presets is used")

    recipes, preset_name = resolve_recipe_queue(args)
    log_dir = args.log_root / sanitize_name(args.session)
    log_dir.mkdir(parents=True, exist_ok=True)
    write_hardware_snapshot(log_dir)
    write_queue_manifest(log_dir, preset_name, recipes, args)

    print(f"Session: {args.session}")
    print(f"Dataset: {args.dataset}")
    print(f"Preset: {preset_name or 'custom'}")
    print(f"Queue length: {len(recipes)}")
    print(f"Logs: {log_dir}")

    if args.dry_run:
        for index, recipe in enumerate(recipes, start=1):
            print(f"{index:02d}. {recipe}")
        return

    results_before = read_results(args.results)
    already_done = completed_recipes(results_before, args.dataset, args.session) if not args.no_resume else set()
    completed = 0

    for index, recipe in enumerate(recipes, start=1):
        if recipe in already_done:
            print(f"[{index:02d}/{len(recipes)}] skipping completed recipe: {recipe}")
            completed += 1
            continue

        best_before = current_best_metric(read_results(args.results), args.dataset, args.session)
        log_name = f"{index:02d}_{sanitize_name(recipe)}.log"
        command = [
            sys.executable,
            "-u",
            str(REPO_DIR / "train.py"),
            "--dataset",
            args.dataset,
            "--recipe",
            recipe,
            "--session",
            args.session,
            "--save-checkpoint",
        ]

        print(f"[{index:02d}/{len(recipes)}] running {recipe}")
        if best_before is None:
            print("  current best: none yet")
        else:
            print(f"  current best: {best_before:.6f}")

        return_code = run_and_log(command, log_dir / log_name)
        results_after = read_results(args.results)
        latest_row = latest_row_for_recipe(results_after, args.dataset, args.session, recipe)

        if latest_row is None:
            print(f"  no row was recorded for {recipe}")
            if args.stop_on_error:
                break
            continue

        metric_name = str(latest_row.get("metric_name", "metric"))
        metric_value = latest_row.get("metric")
        if pd.isna(metric_value):
            metric_text = "nan"
        else:
            metric_text = f"{float(metric_value):.6f}"

        print(f"  latest metric: {metric_name}={metric_text}")
        print(f"  log: {log_dir / log_name}")

        if str(metric_name).lower() == "error" or return_code != 0:
            print("  status: crash")
            if args.stop_on_error:
                break
        else:
            keep_note = "session baseline"
            status = "keep"
            if best_before is not None and float(metric_value) <= float(best_before) + args.min_improvement:
                status = "discard"
                keep_note = f"did not beat session best ({best_before:.6f})"
            elif best_before is not None:
                delta = float(metric_value) - float(best_before)
                keep_note = f"new session best (+{delta:.6f})"

            updated_row = update_row_status(
                args.results,
                dataset=args.dataset,
                session=args.session,
                recipe=recipe,
                status=status,
                note=keep_note,
            )
            print(f"  status: {updated_row.get('status', '')}")

        completed += 1
        summary = build_summary_markdown(
            read_results(args.results),
            dataset=args.dataset,
            session=args.session,
            preset_name=preset_name,
            completed=completed,
            total=len(recipes),
            log_dir=log_dir,
        )
        write_text(log_dir / "session_summary.md", summary + "\n")

    if args.finalize:
        print("Finalizing session artifacts...")
        finalize_session(args, log_dir)


if __name__ == "__main__":
    main()
