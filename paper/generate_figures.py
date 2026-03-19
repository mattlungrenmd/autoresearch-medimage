"""
Generate manuscript figures from results.tsv, checkpoint metadata, and benchmark notes.
"""

import argparse
import os
import re
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import roc_curve

PAPER_DIR = Path(__file__).resolve().parent
REPO_DIR = PAPER_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from tools.progress_plot import choose_dataset, choose_metric_name, choose_session, infer_direction, make_plot

RESULTS_FILE = REPO_DIR / "results.tsv"
BENCHMARK_FILES = {
    "chestxray14": REPO_DIR / "examples" / "chestxray14" / "benchmark.md",
    "aptos": REPO_DIR / "benchmarks" / "diabetic_retinopathy.md",
    "brain_mri": REPO_DIR / "benchmarks" / "segmentation.md",
    "retina_vessel": REPO_DIR / "benchmarks" / "segmentation.md",
    "pneumothorax": REPO_DIR / "benchmarks" / "segmentation.md",
}
HEADLINE_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Pneumonia",
    "Pneumothorax",
    "Edema",
]


def load_results(results_path, dataset_name, session_name=None):
    df = pd.read_csv(results_path, sep="\t")
    df, _ = choose_dataset(df, explicit_dataset=dataset_name)
    df, active_session = choose_session(df, explicit_session=session_name)
    df, metric_name = choose_metric_name(df)
    return df, metric_name, active_session


def load_best_checkpoint(results_df):
    if "metric" in results_df.columns:
        results_df = results_df.copy()
        results_df["metric"] = pd.to_numeric(results_df["metric"], errors="coerce")
        if "status" in results_df.columns:
            keep = results_df[results_df["status"].astype(str) == "keep"]
            if not keep.empty:
                results_df = keep
        best_row = results_df.sort_values("metric", ascending=False).iloc[0]
    else:
        best_row = results_df.iloc[0]

    checkpoint_path = str(best_row.get("checkpoint_path", "")).strip()
    if not checkpoint_path or checkpoint_path.lower() == "nan":
        return best_row.to_dict(), {}, {}
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return best_row.to_dict(), payload.get("results", {}), payload.get("config", {})


def load_final_eval(final_eval_path):
    if not final_eval_path:
        return None
    path = Path(final_eval_path)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def extract_section(text, heading):
    pattern = rf"## {re.escape(heading)}\n\n(.*?)(?=\n## |\Z)"
    match = re.search(pattern, text, flags=re.S)
    return match.group(1) if match else ""


def parse_markdown_table(section_text):
    lines = [line.strip() for line in section_text.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return []
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def parse_float(text):
    text = str(text)
    text = text.replace("**", "").replace("%", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def benchmark_tables(dataset_name):
    benchmark_path = BENCHMARK_FILES.get(dataset_name)
    if not benchmark_path or not benchmark_path.exists():
        return [], []
    text = benchmark_path.read_text(encoding="utf-8")
    published = parse_markdown_table(extract_section(text, "Published Results (Mean AUC %)"))
    per_pathology = parse_markdown_table(extract_section(text, "Per-Pathology AUC (CheXNet Paper)"))
    return published, per_pathology


def plot_benchmark_comparison(best_row, published_rows, output_path, final_eval=None):
    if not published_rows:
        return None

    methods = []
    values = []
    for row in published_rows:
        method = row.get("Method", "")
        value = parse_float(row.get("Mean AUC", row.get("Score", "")))
        if value is None:
            continue
        methods.append(method)
        values.append(value)

    if final_eval:
        methods.append("autoresearch-medimage (held-out test)")
        values.append(float(final_eval.get("primary_metric", best_row.get("metric", 0.0))))
    else:
        methods.append("autoresearch-medimage")
        values.append(float(best_row.get("metric", 0.0)))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#c7d5e0"] * (len(methods) - 1) + ["#2ecc71"]
    ax.barh(methods, values, color=colors)
    ax.set_xlabel("Mean AUC (%)")
    ax.set_title("ChestX-ray14 Mean AUC Comparison")
    ax.grid(axis="x", alpha=0.2)
    for i, value in enumerate(values):
        ax.text(value + 0.2, i, f"{value:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_per_pathology_comparison(checkpoint_results, per_pathology_rows, output_path, final_eval=None):
    source_results = final_eval.get("results", {}) if final_eval else checkpoint_results
    per_class_auc = source_results.get("per_class_auc", {})
    if not per_class_auc or not per_pathology_rows:
        return None

    classes = []
    ours = []
    chexnet = []
    wang = []
    for row in per_pathology_rows:
        pathology = row.get("Pathology", "")
        ours_value = per_class_auc.get(pathology, per_class_auc.get(pathology.replace(" ", "_")))
        if ours_value is None:
            continue
        classes.append(pathology)
        ours.append(float(ours_value))
        chexnet.append(parse_float(row.get("CheXNet AUC", 0.0)) or 0.0)
        wang.append(parse_float(row.get("Wang et al. AUC", 0.0)) or 0.0)

    if not classes:
        return None

    y = np.arange(len(classes))
    h = 0.24
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.barh(y - h, wang, height=h, color="#d0d0d0", label="Wang et al.")
    ax.barh(y, chexnet, height=h, color="#7fb3d5", label="CheXNet")
    ax.barh(y + h, ours, height=h, color="#2ecc71", label="autoresearch-medimage")
    ax.set_yticks(y)
    ax.set_yticklabels(classes)
    ax.set_xlabel("AUC (%)")
    ax.set_title("Per-Pathology AUC Comparison")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def choose_roc_classes(class_names):
    chosen = [name for name in HEADLINE_CLASSES if name in class_names]
    if len(chosen) >= 4:
        return chosen[:6]
    return class_names[: min(6, len(class_names))]


def plot_roc_curves(checkpoint_results, checkpoint_config, output_path, final_eval=None):
    source_results = final_eval.get("results", {}) if final_eval else checkpoint_results
    source_config = final_eval.get("checkpoint_config", {}) if final_eval else checkpoint_config
    split_name = str(final_eval.get("split", "validation")) if final_eval else "validation"
    artifacts = source_results.get("artifacts", {})
    probs = artifacts.get("val_probs")
    targets = artifacts.get("val_targets")
    class_names = artifacts.get("class_names") or source_config.get("class_names") or []

    if probs is None or targets is None or not class_names:
        return None

    probs = np.asarray(probs, dtype=np.float32)
    targets = np.asarray(targets)
    if probs.ndim == 1:
        probs = probs[:, None]
    if targets.ndim == 1 and probs.shape[1] > 1:
        return None

    if probs.shape[1] == 1:
        fig, ax = plt.subplots(figsize=(6, 6))
        fpr, tpr, _ = roc_curve(targets, probs[:, 0])
        ax.plot(fpr, tpr, label=f"AUC={sklearn_auc(fpr, tpr):.3f}", color="#2ecc71")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#999999")
        ax.set_title(f"{split_name.title()} ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    selected_classes = choose_roc_classes(list(class_names))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")

    for ax, class_name in zip(axes, selected_classes):
        idx = list(class_names).index(class_name)
        y_true = targets[:, idx]
        y_score = probs[:, idx]
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = sklearn_auc(fpr, tpr)
        ax.axis("on")
        ax.plot(fpr, tpr, color="#2ecc71", linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
        ax.set_title(f"{class_name}\nAUC={roc_auc:.3f}", fontsize=10)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.grid(alpha=0.2)

    fig.suptitle(f"Representative {split_name.title()} ROC Curves", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate manuscript figures from experiment artifacts")
    parser.add_argument("--results", type=str, default=str(RESULTS_FILE), help="Path to results.tsv")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to visualize")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for generated figure files")
    parser.add_argument("--session", type=str, default=None, help="Optional session filter")
    parser.add_argument("--final-eval", type=str, default=None, help="Optional held-out final evaluation payload (.pt)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or (PAPER_DIR / "generated" / f"{args.dataset}_figures"))
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df, metric_name, active_session = load_results(args.results, args.dataset, session_name=args.session)
    best_row, checkpoint_results, checkpoint_config = load_best_checkpoint(results_df)
    final_eval = load_final_eval(args.final_eval)
    published_rows, per_pathology_rows = benchmark_tables(args.dataset)

    direction = infer_direction(metric_name)
    progress_output = output_dir / "progress.png"
    make_plot(results_df, args.dataset, metric_name, direction, str(progress_output), session_name=active_session)

    generated = [str(progress_output)]

    benchmark_output = output_dir / "benchmark_comparison.png"
    if plot_benchmark_comparison(best_row, published_rows, str(benchmark_output), final_eval=final_eval):
        generated.append(str(benchmark_output))

    per_class_output = output_dir / "per_class_comparison.png"
    if plot_per_pathology_comparison(checkpoint_results, per_pathology_rows, str(per_class_output), final_eval=final_eval):
        generated.append(str(per_class_output))

    roc_output = output_dir / "roc_curves.png"
    if plot_roc_curves(checkpoint_results, checkpoint_config, str(roc_output), final_eval=final_eval):
        generated.append(str(roc_output))

    print("Generated figures:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
