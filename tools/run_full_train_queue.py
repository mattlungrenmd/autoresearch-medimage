"""
Queue multiple epoch-budgeted full-train refinement runs sequentially.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from tools import run_full_train as rft

RUN_FULL_TRAIN = REPO_DIR / "tools" / "run_full_train.py"
DEFAULT_RESULTS = REPO_DIR / "results.tsv"


def default_train_session(base_name: str, epochs: int) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(base_name or "fulltrain"))
    safe = safe.strip("_") or "fulltrain"
    return f"{safe}_fulltrain_ep{epochs}"


def branch_key_for_backbone(backbone: str) -> str:
    text = str(backbone or "").lower()
    for prefix in ("densenet", "efficientnet", "convnext", "resnet"):
        if text.startswith(prefix):
            return prefix
    return text or "unknown"


def unique_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def ranked_session_candidates(results_path: Path, session: str, dataset: str | None = None) -> list[dict]:
    df = rft.read_results(results_path)
    filtered = rft.filter_results(df, dataset=dataset, session=session)
    filtered = filtered.copy()
    if "candidate_id" not in filtered.columns:
        raise ValueError("results.tsv does not contain a 'candidate_id' column for queue candidate selection")
    filtered["candidate_id"] = filtered["candidate_id"].fillna("").astype(str)
    filtered = filtered[filtered["candidate_id"] != ""].copy()
    if "status_norm" in filtered.columns:
        filtered = filtered[filtered["status_norm"] != "crash"].copy()
    if "metric" in filtered.columns:
        filtered = filtered[filtered["metric"].notna()].copy()
        filtered = filtered.sort_values("metric", ascending=False)
    return filtered.to_dict("records")


def diversity_key(row: dict, mode: str) -> str:
    candidate_id = str(row.get("candidate_id", "") or "").strip()
    if mode == "none":
        return candidate_id
    backbone = str(row.get("backbone", "") or "").strip()
    if mode == "backbone":
        return backbone or candidate_id
    if mode == "branch":
        return branch_key_for_backbone(backbone) or candidate_id
    raise ValueError(f"Unknown diversity mode: {mode}")


def select_candidate_ids(
    results_path: Path,
    session: str,
    dataset: str | None = None,
    candidate_ids: list[str] | None = None,
    top_k: int | None = None,
    diverse_by: str = "branch",
) -> list[str]:
    selected = unique_ordered(list(candidate_ids or []))
    if top_k is None or top_k <= 0:
        return selected

    rows = ranked_session_candidates(results_path, session=session, dataset=dataset)
    seen_candidates = set(selected)
    diverse_keys = {
        diversity_key(row, diverse_by)
        for row in rows
        if str(row.get("candidate_id", "") or "").strip() in seen_candidates
    }
    if diverse_by != "none":
        for row in rows:
            candidate_id = str(row.get("candidate_id", "") or "").strip()
            if not candidate_id or candidate_id in seen_candidates:
                continue
            key = diversity_key(row, diverse_by)
            if key in diverse_keys:
                continue
            selected.append(candidate_id)
            seen_candidates.add(candidate_id)
            diverse_keys.add(key)
            if len(selected) >= top_k:
                return selected
    for row in rows:
        candidate_id = str(row.get("candidate_id", "") or "").strip()
        if not candidate_id or candidate_id in seen_candidates:
            continue
        selected.append(candidate_id)
        seen_candidates.add(candidate_id)
        if len(selected) >= top_k:
            break
    return selected


def main():
    parser = argparse.ArgumentParser(description="Run multiple full-train refinement jobs sequentially")
    parser.add_argument("--session", type=str, required=True, help="Source search session containing the candidate recipes")
    parser.add_argument("--candidate-id", action="append", default=None, help="Candidate id to full-train; pass multiple times")
    parser.add_argument("--top-k", type=int, default=None, help="Automatically select the top K session candidates for full-train")
    parser.add_argument(
        "--diverse-by",
        type=str,
        choices=["none", "backbone", "branch"],
        default="branch",
        help="When used with --top-k, select one strong candidate per diversity bucket first, then backfill by metric",
    )
    parser.add_argument("--epochs", type=int, required=True, help="Epoch budget for each full-train job")
    parser.add_argument("--results", type=str, default=str(DEFAULT_RESULTS), help="Path to results.tsv")
    parser.add_argument("--workspace-root", type=str, default=None, help="Optional workspace root containing train.py, recipes/, logs/, and results.tsv")
    parser.add_argument("--python-executable", type=str, default=None, help="Optional Python interpreter to use for the underlying train.py runs")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset filter")
    parser.add_argument("--prefix", type=str, default=None, help="Optional prefix for generated full-train session names")
    parser.add_argument("--dry-run", action="store_true", help="Print the queue plan without launching runs")
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not args.candidate_id and (args.top_k is None or args.top_k <= 0):
        raise ValueError("Pass at least one --candidate-id or a positive --top-k")

    workspace_root = (
        Path(args.workspace_root).expanduser().resolve()
        if args.workspace_root
        else Path(args.results).expanduser().resolve().parent
    )
    results_path = Path(args.results).expanduser().resolve()
    queue_dir = workspace_root / "logs" / f"{args.session}_fulltrain_queue"
    queue_dir.mkdir(parents=True, exist_ok=True)

    selected_candidate_ids = select_candidate_ids(
        results_path=results_path,
        session=args.session,
        dataset=args.dataset,
        candidate_ids=args.candidate_id,
        top_k=args.top_k,
        diverse_by=args.diverse_by,
    )
    if not selected_candidate_ids:
        raise ValueError("No candidates selected for the full-train queue")

    ranked_lookup = {
        str(row.get("candidate_id", "") or "").strip(): row
        for row in ranked_session_candidates(results_path, session=args.session, dataset=args.dataset)
    }
    explicit_set = set(unique_ordered(list(args.candidate_id or [])))

    queue_plan = []
    commands = []
    for candidate_id in selected_candidate_ids:
        train_session = (
            f"{args.prefix}_{candidate_id}"
            if args.prefix
            else default_train_session(candidate_id, args.epochs)
        )
        command = [
            sys.executable,
            str(RUN_FULL_TRAIN),
            "--workspace-root",
            str(workspace_root),
            "--session",
            args.session,
            "--candidate-id",
            candidate_id,
            "--results",
            str(results_path),
            "--epochs",
            str(args.epochs),
            "--train-session",
            train_session,
        ]
        if args.dataset:
            command.extend(["--dataset", args.dataset])
        if args.python_executable:
            command.extend(["--python-executable", args.python_executable])
        commands.append(command)
        row = ranked_lookup.get(candidate_id, {})
        queue_plan.append(
            {
                "candidate_id": candidate_id,
                "metric": row.get("metric"),
                "backbone": row.get("backbone"),
                "train_session": train_session,
                "python_executable": args.python_executable,
                "selection_mode": "explicit" if candidate_id in explicit_set else "auto",
                "command": command,
            }
        )

    plan_path = queue_dir / "queue_plan.json"
    plan_path.write_text(json.dumps(queue_plan, indent=2), encoding="utf-8")
    print(f"Queue plan: {plan_path}")
    for item in queue_plan:
        metric = item.get("metric")
        backbone = item.get("backbone")
        score = f"{float(metric):.6f}" if metric is not None else "n/a"
        print(f"  {item['candidate_id']} -> {item['train_session']} | backbone={backbone or 'n/a'} | metric={score}")

    if args.dry_run:
        return

    for item, command in zip(queue_plan, commands):
        print(f"Running {item['candidate_id']} -> {item['train_session']}")
        result = subprocess.run(command, cwd=str(REPO_DIR))
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
