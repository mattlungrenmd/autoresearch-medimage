"""
Build a stage-2 refinement search space from the top candidates of a prior search session.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from tools import run_full_train as rft
from tools import run_full_train_queue as rftq
from tools import run_search_session as rss


def relative_seed_path(space_path: Path, seed_path: Path) -> str:
    try:
        return str(seed_path.resolve().relative_to(space_path.parent.resolve()))
    except ValueError:
        return str(seed_path.resolve())


def materialize_seed_recipe(
    winner,
    source_session: str,
    seed_dir: Path,
    workspace_root: Path,
) -> Path:
    source_kind, source_value = rft.resolve_recipe_source(winner, source_session, seed_dir, workspace_root)
    if source_kind == "recipe-file":
        return Path(source_value).resolve()
    tracked_path = workspace_root / "recipes" / f"{source_value}.yaml"
    if not tracked_path.exists():
        raise FileNotFoundError(f"Tracked recipe not found for refinement seed: {tracked_path}")
    target_path = seed_dir / tracked_path.name
    recipe = yaml.safe_load(tracked_path.read_text(encoding="utf-8")) or {}
    recipe.pop("search_metadata", None)
    with open(target_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(recipe, handle, sort_keys=False)
    return target_path.resolve()


def selected_label(row) -> str:
    candidate_raw = row.get("candidate_id", "")
    candidate_id = "" if candidate_raw is None or str(candidate_raw).strip().lower() == "nan" else str(candidate_raw).strip()
    if candidate_id:
        return candidate_id
    recipe_raw = row.get("recipe", "")
    recipe_name = "" if recipe_raw is None or str(recipe_raw).strip().lower() == "nan" else str(recipe_raw).strip()
    if recipe_name:
        return recipe_name
    return "seed"


def selected_rows_from_session(
    results_path: Path,
    dataset: str,
    session: str,
    top_k: int,
    diverse_by: str,
) -> list[tuple[str, object]]:
    selected_candidate_ids = rftq.select_candidate_ids(
        results_path=results_path,
        session=session,
        dataset=dataset,
        top_k=top_k,
        diverse_by=diverse_by,
    )
    if not selected_candidate_ids:
        raise ValueError("No refinement seeds were selected")

    session_results = rft.filter_results(rft.read_results(results_path), dataset=dataset, session=session)
    selected_rows: list[tuple[str, object]] = []
    for candidate_id in selected_candidate_ids:
        row = session_results[session_results["candidate_id"].fillna("").astype(str) == candidate_id].iloc[0]
        selected_rows.append((session, row))
    return selected_rows


def selected_rows_from_seed_sessions(
    results_path: Path,
    dataset: str,
    seed_sessions: list[str],
) -> list[tuple[str, object]]:
    results = rft.read_results(results_path)
    selected_rows: list[tuple[str, object]] = []
    for session in seed_sessions:
        filtered = rft.filter_results(results, dataset=dataset, session=session)
        selected_rows.append((session, rft.select_winner_row(filtered)))
    return selected_rows


def default_output_path(workspace_root: Path, dataset: str, session: str | None, seed_sessions: list[str]) -> Path:
    if session:
        stem = f"{session}_refine"
    elif seed_sessions:
        stem = f"{dataset}_refine_from_{seed_sessions[0]}"
    else:
        stem = f"{dataset}_refine"
    return workspace_root / "logs" / stem / "refinement_space.yaml"


def apply_mutation_family_override(refinement_space: dict, families: list[str] | None) -> None:
    if not families:
        return
    normalized = []
    seen = set()
    for family in families:
        text = str(family or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    refinement_space["mutation_families"] = normalized
    phase_families = refinement_space.get("phase_families")
    if isinstance(phase_families, dict):
        filtered = {}
        for phase, entries in phase_families.items():
            filtered[phase] = [entry for entry in entries if entry in seen]
        refinement_space["phase_families"] = filtered


def main():
    parser = argparse.ArgumentParser(description="Build a refinement search space from top diverse candidates")
    parser.add_argument("--session", type=str, default=None, help="Source search session")
    parser.add_argument("--seed-session", action="append", default=None, help="Repeatable promoted session whose winner should become a refinement seed")
    parser.add_argument("--space", type=str, default="chestxray14", help="Base search space name or path")
    parser.add_argument("--results", type=str, default=str(rft.RESULTS_FILE), help="Path to results.tsv")
    parser.add_argument("--workspace-root", type=str, default=None, help="Workspace containing logs/, recipes/, and results.tsv")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset filter override")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidates to seed into the refinement stage")
    parser.add_argument(
        "--diverse-by",
        type=str,
        choices=["none", "backbone", "branch"],
        default="branch",
        help="Pick one strong candidate per diversity bucket first, then backfill by metric",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Optional default_max_runs override for the refinement space")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output YAML path; defaults to logs/<session>_refine/refinement_space.yaml under the workspace",
    )
    parser.add_argument("--mutation-family", action="append", default=None, help="Optional explicit mutation-family allowlist for the generated refinement space")
    parser.add_argument("--keep-tracked-seeds", action="store_true", help="Retain the base space's tracked seed_recipes as additional refinement seeds")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected candidates and output path without writing files")
    args = parser.parse_args()

    if args.session and args.seed_session:
        raise ValueError("Use either --session or --seed-session, not both")
    if not args.session and not args.seed_session:
        raise ValueError("Pass either --session or at least one --seed-session")
    if args.session and args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    results_path = Path(args.results).expanduser().resolve()
    workspace_root = (
        Path(args.workspace_root).expanduser().resolve()
        if args.workspace_root
        else results_path.parent
    )
    base_space = rss.load_search_space(args.space)
    dataset = args.dataset or base_space.get("dataset")
    if not dataset:
        raise ValueError("Base search space must define a dataset or pass --dataset")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output_path(workspace_root, dataset, args.session, list(args.seed_session or []))
    )
    seed_dir = output_path.parent / "refine_seeds"
    if args.session:
        selected_rows = selected_rows_from_session(
            results_path=results_path,
            dataset=dataset,
            session=args.session,
            top_k=args.top_k,
            diverse_by=args.diverse_by,
        )
        refinement_parent_sessions = [args.session]
    else:
        selected_rows = selected_rows_from_seed_sessions(
            results_path=results_path,
            dataset=dataset,
            seed_sessions=list(args.seed_session or []),
        )
        refinement_parent_sessions = [str(session) for session, _ in selected_rows]

    if args.dry_run:
        print(f"Refinement space output: {output_path}")
        for source_session, row in selected_rows:
            print(
                f"  {source_session} | {selected_label(row)} | backbone={row.get('backbone', '')} | metric={float(row.get('metric', float('nan'))):.6f}"
            )
        return

    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_paths = [
        materialize_seed_recipe(row, source_session, seed_dir, workspace_root)
        for source_session, row in selected_rows
    ]

    refinement_space = copy.deepcopy(base_space)
    refinement_space.pop("_path", None)
    refinement_space["include_best_known_seed"] = False
    if not args.keep_tracked_seeds:
        refinement_space["seed_recipes"] = []
    refinement_space["seed_recipe_files"] = [
        relative_seed_path(output_path, seed_path)
        for seed_path in seed_paths
    ]
    apply_mutation_family_override(refinement_space, args.mutation_family)
    if args.max_runs is not None:
        refinement_space["default_max_runs"] = int(args.max_runs)
    if len(refinement_parent_sessions) == 1:
        refinement_space["refinement_parent_session"] = refinement_parent_sessions[0]
    refinement_space["refinement_parent_sessions"] = refinement_parent_sessions
    refinement_space["refinement_candidates"] = [selected_label(row) for _, row in selected_rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(refinement_space, handle, sort_keys=False)

    print(f"Refinement space written: {output_path}")
    for (source_session, row), seed_path in zip(selected_rows, seed_paths):
        print(
            f"  {source_session} | {selected_label(row)} -> {seed_path.name} | backbone={row.get('backbone', '')} | metric={float(row.get('metric', float('nan'))):.6f}"
        )


if __name__ == "__main__":
    main()
