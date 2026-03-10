"""
Finalize an autoresearch search session.

Prints the winning configuration (backbone, optimizer, LR, augmentations,
etc.) so it can be used for a full-scale training run, then regenerates:
- progress chart
- held-out test evaluation (if test split exists)
- manuscript figures and paper
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
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(command, cwd=workdir, capture_output=True, text=True, env=env)
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
    final_eval_output = os.path.abspath(os.path.join(REPO_DIR, "paper", "generated", f"{dataset_name}_final_eval.pt"))
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

    figures_dir = os.path.abspath(os.path.join(REPO_DIR, "paper", "generated", f"{dataset_name}_figures"))
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

    # --- Winning configuration summary ---
    # Try checkpoint config first, fall back to results.tsv columns
    ckpt_config = {}
    ckpt_path = str(winner.get("checkpoint_path", "")).strip()
    if ckpt_path and ckpt_path.lower() != "nan" and os.path.exists(ckpt_path):
        try:
            import torch
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ckpt_config = payload.get("config", {})
        except Exception:
            pass

    def _cfg(key, fallback_key=None):
        val = ckpt_config.get(key, "")
        if not val and fallback_key:
            val = str(winner.get(fallback_key, "")).strip()
        if not val:
            val = str(winner.get(key, "")).strip()
        if val and val.lower() != "nan":
            return val
        return ""

    print("=" * 60)
    print("WINNING CONFIGURATION")
    print("=" * 60)
    print(f"  dataset:       {dataset_name}")
    if session_name:
        print(f"  session:       {session_name}")
    print(f"  best metric:   {winner.get('metric_name', 'metric')} = {winner.get('metric', '')}")
    if final_eval_output:
        print(f"  held-out test: (see {os.path.basename(final_eval_output)})")
    print()
    backbone = _cfg("backbone")
    if backbone:
        print(f"  backbone:      {backbone}")
    optimizer = _cfg("optimizer")
    if optimizer:
        print(f"  optimizer:     {optimizer}")
    lr = _cfg("learning_rate")
    if lr:
        print(f"  learning_rate: {lr}")
    wd = _cfg("weight_decay")
    if wd:
        print(f"  weight_decay:  {wd}")
    bs = _cfg("batch_size")
    if bs:
        print(f"  batch_size:    {bs}")
    loss = _cfg("loss")
    if loss:
        print(f"  loss:          {loss}")
    dropout = _cfg("dropout")
    if dropout:
        print(f"  dropout:       {dropout}")
    img_size = _cfg("image_size")
    if img_size:
        print(f"  image_size:    {img_size}")
    augmentations = _cfg("augmentations")
    if augmentations:
        print(f"  augmentations: {augmentations}")

    # Advanced features
    advanced = []
    for feat in ["ema", "tta", "mixup", "cutmix", "swa", "llrd"]:
        val = ckpt_config.get(feat)
        if val:
            advanced.append(feat.upper())
    if advanced:
        print(f"  advanced:      {', '.join(advanced)}")

    print()
    print(f"  checkpoint:    {ckpt_path}")
    print()
    print("To train this configuration to convergence:")
    print(f"  uv run train.py --dataset {dataset_name} --time-budget <seconds> --save-checkpoint")
    print()
    print("-" * 60)
    print("Generated artifacts:")
    print(f"  progress:   {progress_output}")
    if final_eval_output:
        print(f"  final eval: {final_eval_output}")
    print(f"  figures:    {figures_dir}")
    print(f"  paper:      {paper_output}")


if __name__ == "__main__":
    main()
