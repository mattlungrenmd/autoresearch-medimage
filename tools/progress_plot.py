"""
Generate an autoresearch progress plot from results.tsv.

This generalizes the progress chart idea from karpathy/autoresearch's
analysis.ipynb to the medical-imaging results ledger used in this repo.
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS = os.path.join(REPO_DIR, "results.tsv")
DEFAULT_OUTPUT = os.path.join(REPO_DIR, "progress.png")

HIGHER_IS_BETTER = {
    "accuracy",
    "auc",
    "mean_auc",
    "dice",
    "iou",
    "pixel_acc",
    "sensitivity",
    "specificity",
    "kappa",
    "quadratic_kappa",
}
LOWER_IS_BETTER = {
    "loss",
    "val_bpb",
    "error",
}


def normalize_status(value):
    value = str(value).strip().lower()
    if value in {"keep", "kept"}:
        return "keep"
    if value in {"discard", "discarded"}:
        return "discard"
    if value in {"crash", "failed", "error"}:
        return "crash"
    if value in {"candidate", "pending"}:
        return "candidate"
    return value or "unknown"


def infer_direction(metric_name):
    metric_name = str(metric_name).strip().lower()
    if metric_name in HIGHER_IS_BETTER:
        return "higher"
    if metric_name in LOWER_IS_BETTER:
        return "lower"
    if "loss" in metric_name or "error" in metric_name or "bpb" in metric_name:
        return "lower"
    return "higher"


def truncate_description(text, limit=42):
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def choose_dataset(df, explicit_dataset=None):
    if explicit_dataset:
        filtered = df[df["dataset"].astype(str) == str(explicit_dataset)].copy()
        if filtered.empty:
            raise ValueError(f"No rows found for dataset '{explicit_dataset}'")
        return filtered, explicit_dataset

    if "dataset" not in df.columns or df["dataset"].dropna().nunique() <= 1:
        dataset_name = str(df["dataset"].iloc[-1]) if "dataset" in df.columns else "results"
        return df.copy(), dataset_name

    latest_dataset = str(df["dataset"].iloc[-1])
    filtered = df[df["dataset"].astype(str) == latest_dataset].copy()
    print(f"Multiple datasets found in results.tsv. Defaulting to latest dataset: {latest_dataset}")
    return filtered, latest_dataset


def choose_metric_name(df, explicit_metric=None):
    if explicit_metric:
        filtered = df[df["metric_name"].astype(str) == str(explicit_metric)].copy()
        if filtered.empty:
            raise ValueError(f"No rows found for metric '{explicit_metric}'")
        return filtered, explicit_metric

    if "metric_name" not in df.columns or df["metric_name"].dropna().nunique() <= 1:
        metric_name = str(df["metric_name"].iloc[-1]) if "metric_name" in df.columns else "metric"
        return df.copy(), metric_name

    latest_metric = str(df["metric_name"].iloc[-1])
    filtered = df[df["metric_name"].astype(str) == latest_metric].copy()
    print(f"Multiple metrics found for this dataset. Defaulting to latest metric: {latest_metric}")
    return filtered, latest_metric


def frontier_subset(valid_df):
    keep_rows = valid_df[valid_df["status_norm"] == "keep"].copy()
    if not keep_rows.empty:
        return keep_rows, True

    candidate_rows = valid_df[valid_df["status_norm"].isin(["candidate", "keep"])].copy()
    if not candidate_rows.empty:
        return candidate_rows, False

    return valid_df.copy(), False


def compute_running_best(frontier_df, direction):
    if direction == "lower":
        return frontier_df["metric"].cummin()
    return frontier_df["metric"].cummax()


def make_plot(df, dataset_name, metric_name, direction, output_path):
    if df.empty:
        raise ValueError("No rows available to plot")

    df = df.copy().reset_index(drop=True)
    df["status_norm"] = df["status"].map(normalize_status) if "status" in df.columns else "candidate"
    df["metric"] = pd.to_numeric(df["metric"], errors="coerce")

    valid = df[df["metric"].notna() & (df["status_norm"] != "crash")].copy()
    if valid.empty:
        raise ValueError("No valid metric rows available to plot")

    frontier, explicit_keep = frontier_subset(valid)
    frontier = frontier.copy()
    frontier["running_best"] = compute_running_best(frontier, direction)

    baseline = float(valid.iloc[0]["metric"])
    best_value = float(frontier["running_best"].iloc[-1])

    fig, ax = plt.subplots(figsize=(16, 8))

    discarded = valid[valid["status_norm"] == "discard"]
    if not discarded.empty:
        ax.scatter(
            discarded.index,
            discarded["metric"],
            c="#cccccc",
            s=18,
            alpha=0.55,
            zorder=2,
            label="Discarded",
        )

    crashes = df[df["status_norm"] == "crash"].copy()
    if not crashes.empty:
        ax.scatter(
            crashes.index,
            [baseline] * len(crashes),
            marker="x",
            c="#e74c3c",
            s=32,
            alpha=0.7,
            zorder=2,
            label="Crash",
        )

    candidate_color = "#2ecc71" if not explicit_keep else "#3498db"
    candidate_label = "Kept" if not explicit_keep else "Candidate"
    keep_rows = valid[valid["status_norm"] == "keep"]
    if not keep_rows.empty:
        ax.scatter(
            keep_rows.index,
            keep_rows["metric"],
            c="#2ecc71",
            s=60,
            zorder=4,
            label="Kept",
            edgecolors="black",
            linewidths=0.5,
        )

    candidate_rows = valid[valid["status_norm"] == "candidate"]
    if not candidate_rows.empty:
        ax.scatter(
            candidate_rows.index,
            candidate_rows["metric"],
            c=candidate_color,
            s=56 if not explicit_keep else 40,
            zorder=4 if not explicit_keep else 3,
            label=candidate_label,
            edgecolors="black" if not explicit_keep else "none",
            linewidths=0.5 if not explicit_keep else 0.0,
            alpha=0.95 if not explicit_keep else 0.85,
        )

    ax.step(
        frontier.index,
        frontier["running_best"],
        where="post",
        color="#27ae60",
        linewidth=2.2,
        alpha=0.75,
        zorder=3,
        label="Running best",
    )

    label_rows = frontier
    for idx, row in label_rows.iterrows():
        desc = truncate_description(row.get("description", ""))
        if not desc:
            continue
        ax.annotate(
            desc,
            (idx, row["metric"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8.5,
            color="#1a7a3a",
            alpha=0.9,
            rotation=30,
            ha="left",
            va="bottom",
        )

    total_experiments = len(df)
    kept_count = len(valid[valid["status_norm"] == "keep"])
    if kept_count == 0:
        kept_count = len(frontier)
        kept_label = "frontier points"
    else:
        kept_label = "kept improvements"

    better_text = "lower is better" if direction == "lower" else "higher is better"
    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel(f"{metric_name} ({better_text})", fontsize=12)
    ax.set_title(
        f"Autoresearch Progress: {total_experiments} Experiments, {kept_count} {kept_label}\n{dataset_name}",
        fontsize=16,
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.2)

    min_y = min(float(valid["metric"].min()), best_value)
    max_y = max(float(valid["metric"].max()), baseline)
    span = max(max_y - min_y, abs(baseline) * 0.02, 1e-6)
    margin = span * 0.12
    ax.set_ylim(min_y - margin, max_y + margin)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate a progress chart from results.tsv")
    parser.add_argument("--results", type=str, default=DEFAULT_RESULTS, help="Path to results.tsv")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output PNG path")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset filter")
    parser.add_argument("--metric-name", type=str, default=None, help="Optional metric filter")
    parser.add_argument(
        "--direction",
        type=str,
        default="auto",
        choices=["auto", "higher", "lower"],
        help="Whether larger or smaller metric values are better",
    )
    args = parser.parse_args()

    if not os.path.exists(args.results):
        raise FileNotFoundError(f"results.tsv not found at: {args.results}")

    df = pd.read_csv(args.results, sep="\t")
    if df.empty:
        raise ValueError("results.tsv is empty")

    if "dataset" not in df.columns or "metric" not in df.columns:
        raise ValueError("results.tsv must include at least 'dataset' and 'metric' columns")

    df, dataset_name = choose_dataset(df, args.dataset)
    df, metric_name = choose_metric_name(df, args.metric_name)

    direction = args.direction
    if direction == "auto":
        direction = infer_direction(metric_name)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    make_plot(df, dataset_name, metric_name, direction, output_path)

    print(f"Saved progress chart to: {output_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Metric: {metric_name}")
    print(f"Direction: {direction}")


if __name__ == "__main__":
    main()
