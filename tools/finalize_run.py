"""
Finalize an autoresearch run by regenerating the main artifacts.

This is intended for the end of an overnight session:
- progress chart
- paper scaffold
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS = os.path.join(REPO_DIR, "results.tsv")


def select_rows(df, dataset=None, session=None):
    if dataset:
        df = df[df["dataset"].astype(str) == str(dataset)].copy()
    if session:
        if "session" not in df.columns:
            raise ValueError("results.tsv does not contain a 'session' column")
        df = df[df["session"].fillna("").astype(str) == str(session)].copy()
    if df.empty:
        raise ValueError("No matching rows found in results.tsv")

    if "metric" in df.columns:
        df["metric"] = pd.to_numeric(df["metric"], errors="coerce")
    if "status" in df.columns:
        keep_rows = df[df["status"].astype(str).isin(["keep"])]
        if not keep_rows.empty:
            df = keep_rows
        else:
            candidate_rows = df[df["status"].astype(str).isin(["candidate", "keep"])]
            if not candidate_rows.empty:
                df = candidate_rows
    return df


def infer_dataset(df, explicit_dataset=None, explicit_session=None):
    if explicit_session:
        if "session" not in df.columns:
            raise ValueError("results.tsv does not contain a 'session' column")
        session_df = df[df["session"].fillna("").astype(str) == str(explicit_session)].copy()
        if session_df.empty:
            raise ValueError(f"No rows found for session '{explicit_session}'")
        df = session_df
    if explicit_dataset:
        return explicit_dataset
    if "dataset" not in df.columns:
        return "run"
    return str(df.iloc[-1]["dataset"])


def infer_session(df, explicit_session=None, dataset=None):
    if explicit_session:
        return explicit_session
    if dataset and "dataset" in df.columns:
        df = df[df["dataset"].astype(str) == str(dataset)].copy()
    if "session" not in df.columns:
        return None
    session_series = df["session"].fillna("").astype(str).str.strip()
    non_empty = session_series[session_series != ""]
    if non_empty.empty:
        return None
    return str(non_empty.iloc[-1])


def dataset_has_split(dataset_name, split_name):
    split_file = Path(REPO_DIR) / "data" / dataset_name / "splits.csv"
    if not split_file.exists():
        return False
    df = pd.read_csv(split_file)
    return split_name in set(df["split"].astype(str))


def best_row(df):
    if "metric" not in df.columns:
        return df.iloc[-1]
    return df.sort_values("metric", ascending=False).iloc[0]


def run_command(command, workdir):
    result = subprocess.run(command, cwd=workdir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="Regenerate end-of-run artifacts from results.tsv")
    parser.add_argument("--results", type=str, default=DEFAULT_RESULTS, help="Path to results.tsv")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset filter")
    parser.add_argument("--session", type=str, default=None, help="Optional session filter")
    parser.add_argument("--progress-output", type=str, default=None, help="Optional progress PNG path")
    parser.add_argument("--paper-output", type=str, default=None, help="Optional paper .tex output path")
    parser.add_argument("--final-eval-output", type=str, default=None, help="Optional held-out evaluation payload path")
    parser.add_argument("--figures-output-dir", type=str, default=None, help="Optional figures output directory")
    parser.add_argument("--compile-paper", action="store_true", help="Attempt LaTeX compilation if pdflatex is available")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        raise FileNotFoundError(f"results.tsv not found at: {args.results}")

    df = pd.read_csv(args.results, sep="\t")
    session_name = infer_session(df, explicit_session=args.session, dataset=args.dataset)
    dataset_name = infer_dataset(df, explicit_dataset=args.dataset, explicit_session=session_name)
    selected = select_rows(df, dataset=dataset_name, session=session_name)
    winner = best_row(selected)

    progress_output = os.path.abspath(args.progress_output or os.path.join(REPO_DIR, "progress.png"))
    paper_output = os.path.abspath(
        args.paper_output or os.path.join(REPO_DIR, "paper", "generated", f"{dataset_name}_overnight.tex")
    )
    final_eval_output = os.path.abspath(
        args.final_eval_output or os.path.join(REPO_DIR, "paper", "generated", f"{dataset_name}_final_eval.pt")
    )
    figures_dir = os.path.abspath(
        args.figures_output_dir or os.path.join(REPO_DIR, "paper", "generated", f"{dataset_name}_figures")
    )
    python_exe = sys.executable

    run_command(
        [
            python_exe,
            os.path.join(REPO_DIR, "tools", "progress_plot.py"),
            "--results",
            os.path.abspath(args.results),
            "--dataset",
            dataset_name,
            "--output",
            progress_output,
            *(["--session", session_name] if session_name else []),
        ],
        REPO_DIR,
    )

    if dataset_has_split(dataset_name, "test") and str(winner.get("checkpoint_path", "")).strip():
        run_command(
            [
                python_exe,
                os.path.join(REPO_DIR, "tools", "evaluate_checkpoint.py"),
                "--checkpoint",
                str(winner.get("checkpoint_path")),
                "--split",
                "test",
                "--output",
                final_eval_output,
            ],
            REPO_DIR,
        )
    else:
        final_eval_output = ""

    run_command(
        [
                python_exe,
                os.path.join(REPO_DIR, "paper", "generate_figures.py"),
                "--results",
                os.path.abspath(args.results),
                "--dataset",
                dataset_name,
                "--output-dir",
                figures_dir,
                *(["--session", session_name] if session_name else []),
                *([] if not final_eval_output else ["--final-eval", final_eval_output]),
        ],
        REPO_DIR,
    )

    paper_cmd = [
        python_exe,
        os.path.join(REPO_DIR, "paper", "generate_paper.py"),
        "--results",
        os.path.abspath(args.results),
        "--dataset",
        dataset_name,
        "--output",
        paper_output,
        "--figures-dir",
        figures_dir,
        *(["--session", session_name] if session_name else []),
        *([] if not final_eval_output else ["--final-eval", final_eval_output]),
    ]
    if args.compile_paper:
        paper_cmd.append("--compile")
    run_command(paper_cmd, REPO_DIR)

    search_summary = ""
    if session_name:
        search_dir = os.path.join(REPO_DIR, "logs", session_name, "search")
        state_file = os.path.join(search_dir, "search_state.json")
        if os.path.exists(state_file):
            search_summary = os.path.join(search_dir, "search_summary.md")
            run_command(
                [
                    python_exe,
                    os.path.join(REPO_DIR, "tools", "search_report.py"),
                    "--session",
                    session_name,
                    "--results",
                    os.path.abspath(args.results),
                    "--search-dir",
                    search_dir,
                    "--output",
                    search_summary,
                ],
                REPO_DIR,
            )

    print("Finalized run artifacts:")
    print(f"  dataset: {dataset_name}")
    if session_name:
        print(f"  session: {session_name}")
    print(f"  best metric: {winner.get('metric_name', 'metric')}={winner.get('metric', '')}")
    print(f"  status basis: {'keep' if 'status' in selected.columns and (selected['status'] == 'keep').any() else 'candidate'}")
    print(f"  checkpoint: {winner.get('checkpoint_path', '')}")
    print(f"  progress: {progress_output}")
    if final_eval_output:
        print(f"  final eval: {final_eval_output}")
    print(f"  figures: {figures_dir}")
    print(f"  paper: {paper_output}")
    if search_summary:
        print(f"  search summary: {search_summary}")


if __name__ == "__main__":
    main()
