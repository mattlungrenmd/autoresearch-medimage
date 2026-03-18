"""
Generate an AdaEvolve-lite search summary from session-sidecar artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_DIR / "results.tsv"


def normalize_status(value: object) -> str:
    return str(value or "").strip().lower()


def load_candidates(candidates_path: Path) -> list[dict]:
    if not candidates_path.exists():
        return []
    candidates = []
    for line in candidates_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        candidates.append(json.loads(line))
    return candidates


def build_lineage_mermaid(candidates: list[dict]) -> str:
    lines = ["graph TD"]
    seen_nodes = set()
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        if not candidate_id:
            continue
        label = f"{candidate_id}[\"{candidate_id}<br/>{candidate.get('recipe_name', '')}<br/>{candidate.get('metric', '')}\"]"
        if candidate_id not in seen_nodes:
            lines.append(f"    {label}")
            seen_nodes.add(candidate_id)
        parent_id = str(candidate.get("parent_id", "")).strip()
        if parent_id:
            lines.append(f"    {parent_id} --> {candidate_id}")
    if len(lines) == 1:
        lines.append('    empty["No candidates recorded yet"]')
    return "\n".join(lines) + "\n"


def generate_search_report(
    session: str,
    results_path: Path = DEFAULT_RESULTS,
    search_dir: Path | None = None,
    output_path: Path | None = None,
) -> tuple[Path, Path]:
    if search_dir is None:
        search_dir = REPO_DIR / "logs" / session / "search"
    candidates_path = search_dir / "candidates.jsonl"
    state_path = search_dir / "search_state.json"
    output_path = output_path or (search_dir / "search_summary.md")
    lineage_path = search_dir / "lineage.mmd"

    if results_path.exists():
        df = pd.read_csv(results_path, sep="\t")
    else:
        df = pd.DataFrame(columns=["session", "dataset", "recipe", "backbone", "metric", "metric_name", "status"])
    if "session" not in df.columns:
        df["session"] = ""
    session_df = df[df["session"].fillna("").astype(str) == str(session)].copy()
    if not session_df.empty and "metric" in session_df.columns:
        session_df["metric"] = pd.to_numeric(session_df["metric"], errors="coerce")
    if "status" in session_df.columns:
        session_df["status_norm"] = session_df["status"].map(normalize_status)
    else:
        session_df["status_norm"] = "candidate"

    candidates = load_candidates(candidates_path)
    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    lines = [f"# Search Summary: {session}", ""]
    if state:
        lines.extend(
            [
                f"- algorithm: {state.get('search_algo', '')}",
                f"- dataset: {state.get('dataset', '')}",
                f"- completed runs: {state.get('completed_runs', 0)}",
                f"- non-improving streak: {state.get('non_improving_runs', 0)}",
            ]
        )
        best_metric = state.get("best_confirmed_metric")
        if best_metric is not None:
            lines.append(f"- best confirmed metric: {best_metric:.6f}")
        lines.append("")

    if not session_df.empty:
        valid = session_df[session_df["status_norm"] != "crash"].copy()
        if not valid.empty:
            best_row = valid.sort_values("metric", ascending=False).iloc[0]
            lines.extend(
                [
                    "## Session Winner",
                    "",
                    f"- recipe: {best_row.get('recipe', '')}",
                    f"- backbone: {best_row.get('backbone', '')}",
                    f"- metric: {best_row.get('metric_name', 'metric')}={best_row.get('metric', '')}",
                    f"- status: {best_row.get('status', '')}",
                    "",
                ]
            )

            best_by_backbone = (
                valid.sort_values("metric", ascending=False)
                .groupby("backbone", dropna=False)
                .head(1)
                .sort_values("metric", ascending=False)
            )
            lines.extend(
                [
                    "## Best By Backbone",
                    "",
                    "| backbone | recipe | metric | status |",
                    "|---|---|---:|---|",
                ]
            )
            for _, row in best_by_backbone.iterrows():
                lines.append(
                    "| {backbone} | {recipe} | {metric:.6f} | {status} |".format(
                        backbone=str(row.get("backbone", "")),
                        recipe=str(row.get("recipe", "")),
                        metric=float(row.get("metric", 0.0)),
                        status=str(row.get("status", "")),
                    )
                )
            lines.append("")

    if candidates:
        cand_df = pd.DataFrame(candidates)
        if "metric" in cand_df.columns:
            cand_df["metric"] = pd.to_numeric(cand_df["metric"], errors="coerce")
        if "improvement" in cand_df.columns:
            cand_df["improvement"] = pd.to_numeric(cand_df["improvement"], errors="coerce")
        else:
            cand_df["improvement"] = pd.NA
        if "status" in cand_df.columns:
            cand_df["status_norm"] = cand_df["status"].map(normalize_status)
        else:
            cand_df["status_norm"] = ""

        if "mutation_op" in cand_df.columns:
            ops = (
                cand_df[cand_df["mutation_op"].fillna("").astype(str) != "seed"]
                .groupby("mutation_op", dropna=False)
                .agg(
                    trials=("candidate_id", "count"),
                    keeps=("status_norm", lambda s: int((s == "keep").sum())),
                    mean_metric=("metric", "mean"),
                    mean_delta=("improvement", "mean"),
                )
                .reset_index()
                .sort_values(["keeps", "mean_delta", "trials"], ascending=[False, False, False])
            )
            if not ops.empty:
                lines.extend(
                    [
                        "## Mutation Family Hit Rates",
                        "",
                        "| mutation_op | trials | keeps | mean_metric | mean_delta |",
                        "|---|---:|---:|---:|---:|",
                    ]
                )
                for _, row in ops.iterrows():
                    lines.append(
                        "| {mutation_op} | {trials} | {keeps} | {mean_metric:.6f} | {mean_delta:.6f} |".format(
                            mutation_op=str(row.get("mutation_op", "")),
                            trials=int(row.get("trials", 0)),
                            keeps=int(row.get("keeps", 0)),
                            mean_metric=float(row.get("mean_metric", 0.0) or 0.0),
                            mean_delta=float(row.get("mean_delta", 0.0) or 0.0),
                        )
                    )
                lines.append("")

        timeline = cand_df.copy()
        if "sequence_id" in timeline.columns:
            timeline = timeline.sort_values("sequence_id")
        timeline["running_best"] = timeline["metric"].cummax()
        lines.extend(
            [
                "## Branch Timeline",
                "",
                "| step | candidate_id | branch | mutation_op | metric | running_best | status |",
                "|---|---|---|---|---:|---:|---|",
            ]
        )
        for _, row in timeline.iterrows():
            lines.append(
                "| {step} | {candidate_id} | {branch} | {mutation_op} | {metric:.6f} | {running_best:.6f} | {status} |".format(
                    step=int(row.get("sequence_id", 0)),
                    candidate_id=str(row.get("candidate_id", "")),
                    branch=str(row.get("branch_key", "")),
                    mutation_op=str(row.get("mutation_op", "")),
                    metric=float(row.get("metric", 0.0) or 0.0),
                    running_best=float(row.get("running_best", 0.0) or 0.0),
                    status=str(row.get("status", "")),
                )
            )
        lines.append("")

        if "replicate_group" in cand_df.columns:
            repl = (
                cand_df[cand_df["replicate_group"].fillna("").astype(str) != ""]
                .groupby("replicate_group", dropna=False)
                .agg(
                    attempts=("candidate_id", "count"),
                    mean_metric=("metric", "mean"),
                    final_status=("status", "last"),
                )
                .reset_index()
            )
            repl = repl[repl["attempts"] > 1]
            lines.extend(
                [
                    "## Confirmation Reruns",
                    "",
                    "| replicate_group | attempts | mean_metric | final_status |",
                    "|---|---:|---:|---|",
                ]
            )
            if repl.empty:
                lines.append("| none | 0 | 0.000000 | n/a |")
            else:
                for _, row in repl.iterrows():
                    lines.append(
                        "| {replicate_group} | {attempts} | {mean_metric:.6f} | {final_status} |".format(
                            replicate_group=str(row.get("replicate_group", "")),
                            attempts=int(row.get("attempts", 0)),
                            mean_metric=float(row.get("mean_metric", 0.0) or 0.0),
                            final_status=str(row.get("final_status", "")),
                        )
                    )
            lines.append("")

    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_path.write_text(build_lineage_mermaid(candidates), encoding="utf-8")
    lines.extend(
        [
            "## Lineage",
            "",
            "```mermaid",
            lineage_path.read_text(encoding="utf-8").strip(),
            "```",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path, lineage_path


def main():
    parser = argparse.ArgumentParser(description="Generate an AdaEvolve-lite search report for a session")
    parser.add_argument("--session", type=str, required=True, help="Search session tag")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Path to results.tsv")
    parser.add_argument("--search-dir", type=Path, default=None, help="Optional explicit search directory")
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown output path")
    args = parser.parse_args()

    output_path, lineage_path = generate_search_report(
        session=args.session,
        results_path=args.results,
        search_dir=args.search_dir,
        output_path=args.output,
    )
    print(f"Search summary: {output_path}")
    print(f"Lineage graph:  {lineage_path}")


if __name__ == "__main__":
    main()
