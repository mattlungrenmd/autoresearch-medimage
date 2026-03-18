"""
Prune local experiment artifacts without touching source files.

This helper is intentionally conservative:
- defaults to dry-run
- only touches repo-local generated artifacts
- can keep top checkpoints by metric and specific sessions/checkpoints
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_DIR / "results.tsv"


@dataclass(frozen=True)
class PlannedDelete:
    path: Path
    kind: str
    size_bytes: int


def safe_file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def safe_tree_size(path: Path) -> int:
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += safe_file_size(file_path)
    return total


def normalize_name(value: str) -> str:
    return Path(str(value).strip()).name


def load_best_checkpoint_names(results_path: Path, keep_best: int) -> set[str]:
    if keep_best <= 0 or not results_path.exists():
        return set()
    rows: list[tuple[float, str]] = []
    with results_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            checkpoint_path = str(row.get("checkpoint_path", "")).strip()
            metric_text = str(row.get("metric", "")).strip()
            if not checkpoint_path or not metric_text:
                continue
            try:
                metric_value = float(metric_text)
            except ValueError:
                continue
            rows.append((metric_value, normalize_name(checkpoint_path)))
    rows.sort(key=lambda item: item[0], reverse=True)
    keepers: set[str] = set()
    for _, checkpoint_name in rows:
        keepers.add(checkpoint_name)
        if len(keepers) >= keep_best:
            break
    return keepers


def collect_checkpoint_deletes(
    checkpoints_dir: Path,
    keep_names: set[str],
) -> list[PlannedDelete]:
    if not checkpoints_dir.exists():
        return []
    planned: list[PlannedDelete] = []
    for checkpoint_path in checkpoints_dir.iterdir():
        if not checkpoint_path.is_file():
            continue
        if checkpoint_path.name in keep_names:
            continue
        planned.append(
            PlannedDelete(
                path=checkpoint_path,
                kind="checkpoint",
                size_bytes=safe_file_size(checkpoint_path),
            )
        )
    return planned


def collect_log_deletes(
    logs_dir: Path,
    keep_sessions: set[str],
    remove_smoke_logs: bool,
) -> list[PlannedDelete]:
    if not logs_dir.exists():
        return []
    planned: list[PlannedDelete] = []
    for session_dir in logs_dir.iterdir():
        if not session_dir.is_dir():
            continue
        session_name = session_dir.name
        is_smoke = any(token in session_name for token in ("smoke", "verify", "finalize_smoke"))
        if session_name in keep_sessions:
            continue
        if not remove_smoke_logs or not is_smoke:
            continue
        planned.append(
            PlannedDelete(
                path=session_dir,
                kind="log_session",
                size_bytes=safe_tree_size(session_dir),
            )
        )
    return planned


def collect_misc_deletes(repo_dir: Path, remove_tmp: bool, remove_progress_png: bool) -> list[PlannedDelete]:
    planned: list[PlannedDelete] = []
    if remove_tmp:
        tmp_dir = repo_dir / "tmp"
        if tmp_dir.exists():
            planned.append(
                PlannedDelete(path=tmp_dir, kind="tmp_dir", size_bytes=safe_tree_size(tmp_dir))
            )
    if remove_progress_png:
        progress_png = repo_dir / "progress.png"
        if progress_png.exists():
            planned.append(
                PlannedDelete(path=progress_png, kind="progress_png", size_bytes=safe_file_size(progress_png))
            )
    return planned


def delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def format_gb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 ** 3):.2f} GB"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prune repo-local generated artifacts.")
    parser.add_argument("--repo-root", type=Path, default=REPO_DIR, help="Repository root")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Path to results.tsv")
    parser.add_argument(
        "--keep-best",
        type=int,
        default=3,
        help="Keep this many top checkpoint files by metric from results.tsv",
    )
    parser.add_argument(
        "--keep-checkpoint",
        action="append",
        default=[],
        help="Checkpoint basename or path to keep (repeatable)",
    )
    parser.add_argument(
        "--keep-log-session",
        action="append",
        default=[],
        help="Log session directory name to keep (repeatable)",
    )
    parser.add_argument(
        "--remove-smoke-logs",
        action="store_true",
        help="Delete smoke/verify/finalize_smoke log sessions that are not explicitly kept",
    )
    parser.add_argument("--remove-tmp", action="store_true", help="Delete repo-local tmp/ if present")
    parser.add_argument(
        "--remove-progress-png",
        action="store_true",
        help="Delete repo-local progress.png if present",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag the tool is dry-run only.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    checkpoints_dir = repo_root / "checkpoints"
    logs_dir = repo_root / "logs"

    keep_names = load_best_checkpoint_names(args.results, args.keep_best)
    keep_names.update(normalize_name(value) for value in args.keep_checkpoint)
    keep_sessions = {str(value).strip() for value in args.keep_log_session if str(value).strip()}

    planned = []
    planned.extend(collect_checkpoint_deletes(checkpoints_dir, keep_names))
    planned.extend(collect_log_deletes(logs_dir, keep_sessions, args.remove_smoke_logs))
    planned.extend(collect_misc_deletes(repo_root, args.remove_tmp, args.remove_progress_png))
    planned.sort(key=lambda item: (item.kind, item.path.as_posix()))

    total_bytes = sum(item.size_bytes for item in planned)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{mode} cleanup plan for: {repo_root}")
    if keep_names:
        print("Keeping checkpoint files:")
        for checkpoint_name in sorted(keep_names):
            print(f"  - {checkpoint_name}")
    if keep_sessions:
        print("Keeping log sessions:")
        for session_name in sorted(keep_sessions):
            print(f"  - {session_name}")
    if not planned:
        print("Nothing to delete.")
        return

    print("Planned deletes:")
    for item in planned:
        print(f"  - [{item.kind}] {item.path} ({format_gb(item.size_bytes)})")
    print(f"Total reclaimable: {format_gb(total_bytes)}")

    if not args.apply:
        return

    for item in planned:
        delete_path(item.path)
    print("Cleanup completed.")


if __name__ == "__main__":
    main()
