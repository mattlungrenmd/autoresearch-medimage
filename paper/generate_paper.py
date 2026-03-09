"""
Paper generator for autoresearch-medimage.

Generates a draft LaTeX manuscript from results.tsv, checkpoint metadata,
and final evaluation artifacts. The output is a real first draft — not a
scaffold or placeholder — built from actual experiment data.

Usage:
    uv run paper/generate_paper.py --dataset chestxray14
    uv run paper/generate_paper.py --dataset chestxray14 --final-eval paper/generated/chestxray14_final_eval.pt
    uv run paper/generate_paper.py --dataset chestxray14 --compile
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import re

import pandas as pd
import torch

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(os.path.dirname(PAPER_DIR), "results.tsv")
BENCHMARK_DIR = os.path.join(os.path.dirname(PAPER_DIR), "benchmarks")


def escape_latex(text):
    text = str(text)
    for old, new in {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }.items():
        text = text.replace(old, new)
    return text


def load_results(results_path, dataset_filter=None, session_filter=None):
    if not os.path.exists(results_path):
        return pd.DataFrame()
    df = pd.read_csv(results_path, sep="\t")
    if dataset_filter and "dataset" in df.columns:
        df = df[df["dataset"] == dataset_filter].copy()
    if "session" in df.columns:
        session_series = df["session"].fillna("").astype(str).str.strip()
        if session_filter:
            df = df[session_series == str(session_filter)].copy()
        else:
            non_empty = session_series[session_series != ""]
            if not non_empty.empty:
                df = df[session_series == str(non_empty.iloc[-1])].copy()
    elif session_filter:
        raise ValueError("results.tsv does not contain a 'session' column")
    if "status" in df.columns:
        keep_like = df[df["status"].isin(["keep", "candidate"])]
        if not keep_like.empty:
            df = keep_like
    if "metric" in df.columns:
        df["metric"] = pd.to_numeric(df["metric"], errors="coerce")
        df = df.sort_values("metric", ascending=False)
    return df


def load_checkpoint_metadata(best_row):
    checkpoint_path = str(best_row.get("checkpoint_path", "")).strip()
    if not checkpoint_path or checkpoint_path.lower() == "nan":
        return {}, {}
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(os.path.dirname(PAPER_DIR), checkpoint_path)
    if not os.path.exists(checkpoint_path):
        return {}, {}
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    return payload.get("results", {}), payload.get("config", {})


def load_final_eval(final_eval_path):
    if not final_eval_path:
        return None
    if not os.path.exists(str(final_eval_path)):
        return None
    try:
        return torch.load(str(final_eval_path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(final_eval_path), map_location="cpu")


def format_metric(metric_name, metric_value):
    metric_name = str(metric_name)
    if metric_name in {"kappa", "quadratic_kappa"}:
        return f"{metric_value:.4f}", metric_name
    return f"{metric_value:.2f}\\%", metric_name


def parse_published_benchmarks(dataset_name):
    """Parse published results from benchmarks/*.md files."""
    benchmark_path = os.path.join(BENCHMARK_DIR, f"{dataset_name}.md")
    if not os.path.exists(benchmark_path):
        return []
    text = Path(benchmark_path).read_text(encoding="utf-8")
    results = []
    for match in re.finditer(r"\|\s*\*?\*?([^|]+?)\*?\*?\s*\|\s*\*?\*?(\d+(?:\.\d+)?)\*?\*?\s*\|", text):
        name = match.group(1).strip().strip("*")
        value = float(match.group(2))
        if name and value > 0:
            results.append((name, value))
    return results


# ---------------------------------------------------------------------------
# LaTeX section generators
# ---------------------------------------------------------------------------

def generate_abstract(results_df, checkpoint_config, final_eval, dataset_name):
    if results_df.empty:
        return "No experiment data available."

    best_row = results_df.iloc[0]
    metric_value = float(best_row.get("metric", 0.0))
    metric_name = str(best_row.get("metric_name", "metric"))
    metric_display, _ = format_metric(metric_name, metric_value)
    backbone = checkpoint_config.get("backbone", best_row.get("backbone", "unknown"))
    task = checkpoint_config.get("task", "classification")
    num_experiments = len(results_df)

    lines = [
        f"We present results from an autonomous agent-driven experiment campaign on the "
        f"{escape_latex(dataset_name)} dataset. Over {num_experiments} experiments using fixed "
        f"5-minute training budgets, an AI agent iteratively explored architectures, optimizers, "
        f"and augmentation strategies by modifying a single training script."
    ]

    if final_eval:
        final_metric = float(final_eval.get("primary_metric", 0.0))
        final_display, _ = format_metric(metric_name, final_metric)
        split_name = final_eval.get("split", "test")
        lines.append(
            f"The best model ({escape_latex(backbone)}) achieved {final_display} "
            f"{escape_latex(metric_name)} on a held-out {escape_latex(split_name)} split, "
            f"with model selection performed exclusively on the validation set."
        )
    else:
        lines.append(
            f"The best model ({escape_latex(backbone)}) achieved {metric_display} "
            f"{escape_latex(metric_name)} on the validation set."
        )

    return " ".join(lines)


def generate_methods_section(checkpoint_config, results_df):
    """Generate a real Methods section from checkpoint config."""
    if not checkpoint_config:
        return r"\section{Methods}" + "\nNo checkpoint metadata available for methods description.\n"

    backbone = escape_latex(checkpoint_config.get("backbone", "unknown"))
    optimizer = escape_latex(checkpoint_config.get("optimizer", "adam").upper())
    lr = checkpoint_config.get("learning_rate", 0.001)
    batch_size = checkpoint_config.get("batch_size", 16)
    weight_decay = checkpoint_config.get("weight_decay", 0.0)
    dropout = checkpoint_config.get("dropout", 0.0)
    loss_type = escape_latex(checkpoint_config.get("loss", "cross_entropy"))
    image_size = checkpoint_config.get("image_size", 224)
    task = checkpoint_config.get("task", "classification")
    time_budget = checkpoint_config.get("time_budget_seconds", 300)

    lines = [
        r"\section{Methods}",
        "",
        r"\subsection{Experiment Framework}",
        f"All experiments were conducted using the autoresearch-medimage framework, "
        f"which implements a fixed-budget autonomous research loop. Each experiment was "
        f"limited to {int(time_budget)} seconds of wall-clock training time. An AI agent "
        f"iteratively modified the training configuration, keeping improvements and discarding "
        f"regressions based on validation performance. Model selection was performed exclusively "
        f"on the validation split to preserve the integrity of held-out evaluation.",
        "",
        r"\subsection{Architecture}",
    ]

    if task == "segmentation":
        seg_backbone = checkpoint_config.get("seg_backbone", "")
        if seg_backbone:
            lines.append(
                f"We used a U-Net architecture with a pretrained {escape_latex(seg_backbone)} encoder "
                f"backbone from the timm library. "
            )
            if checkpoint_config.get("seg_attention"):
                lines.append("Attention gates were applied at skip connections. ")
        else:
            lines.append("We used a standard U-Net architecture with four encoder-decoder stages. ")
    else:
        lines.append(
            f"The model used a {backbone} backbone pretrained on ImageNet, with a task-specific "
            f"classification head. Dropout of {dropout:.2f} was applied before the final linear layer. "
        )

    lines.extend([
        "",
        r"\subsection{Training}",
        f"Training used the {optimizer} optimizer with an initial learning rate of {lr:.1e}, "
        f"weight decay of {weight_decay:.1e}, and batch size of {batch_size}. "
        f"Input images were resized to {image_size}$\\times${image_size} pixels. "
        f"The loss function was {loss_type}. "
        f"Mixed precision (FP16) was used throughout. ",
    ])

    # Advanced features
    advanced = []
    if checkpoint_config.get("ema"):
        advanced.append("exponential moving average (EMA) of model weights")
    if checkpoint_config.get("cutmix"):
        advanced.append("CutMix augmentation")
    if checkpoint_config.get("mixup"):
        advanced.append("MixUp augmentation")
    if checkpoint_config.get("swa"):
        advanced.append("stochastic weight averaging (SWA)")
    if checkpoint_config.get("llrd"):
        decay = checkpoint_config.get("llrd_decay", 0.95)
        advanced.append(f"layer-wise learning rate decay (factor {decay})")
    if checkpoint_config.get("tta"):
        advanced.append("test-time augmentation (horizontal flip)")
    if checkpoint_config.get("gradual_unfreeze"):
        advanced.append("gradual backbone unfreezing")
    if checkpoint_config.get("discriminative_lr"):
        advanced.append("discriminative learning rates for backbone and head")
    if checkpoint_config.get("distillation"):
        advanced.append("knowledge distillation from a teacher model")

    if advanced:
        lines.append(f"Additional techniques included: {', '.join(advanced)}. ")

    # Augmentation
    aug = checkpoint_config.get("augmentations", {})
    aug_list = []
    if aug.get("horizontal_flip"):
        aug_list.append("random horizontal flip")
    if aug.get("vertical_flip"):
        aug_list.append("random vertical flip")
    if aug.get("rotation") and aug["rotation"] > 0:
        aug_list.append(f"random rotation ($\\pm${aug['rotation']}$^\\circ$)")
    if aug.get("color_jitter"):
        aug_list.append("color jitter")
    if aug.get("clahe"):
        aug_list.append("CLAHE preprocessing")
    if aug.get("elastic_transform"):
        aug_list.append("elastic transform")
    if aug.get("grid_distortion"):
        aug_list.append("grid distortion")
    if aug.get("coarse_dropout"):
        aug_list.append("coarse dropout")

    if aug_list:
        lines.extend([
            "",
            r"\subsection{Data Augmentation}",
            f"Training augmentations included: {', '.join(aug_list)}. "
            "Standard ImageNet normalization was applied to all inputs.",
        ])

    return "\n".join(lines)


def generate_dataset_section(checkpoint_config, results_df, dataset_name):
    """Generate a Dataset section from available metadata."""
    task = checkpoint_config.get("task", "classification") if checkpoint_config else "classification"
    num_classes = checkpoint_config.get("num_classes", 0) if checkpoint_config else 0
    class_names = checkpoint_config.get("class_names", []) if checkpoint_config else []

    lines = [
        r"\section{Dataset}",
        f"Experiments were conducted on the {escape_latex(dataset_name)} dataset. "
        f"The task was {escape_latex(task)}",
    ]

    if num_classes > 0:
        lines[-1] += f" with {num_classes} classes"
    lines[-1] += ". "

    if class_names and len(class_names) <= 20:
        formatted = ", ".join(escape_latex(c.replace("_", " ")) for c in class_names)
        lines.append(f"The target classes were: {formatted}. ")

    lines.append(
        "Model selection was performed on the validation split. "
        "A separate held-out test split was reserved for final reporting and was not "
        "used to rank intermediate experiments."
    )

    return "\n".join(lines)


def generate_results_table(results_df):
    if results_df.empty:
        return ""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Top experiments from the autonomous search campaign.}",
        r"\label{tab:experiments}",
        r"\begin{tabular}{llllr}",
        r"\toprule",
        r"Recipe & Backbone & Loss & Metric & Value \\",
        r"\midrule",
    ]
    for _, row in results_df.head(10).iterrows():
        recipe = escape_latex(row.get("recipe", "")) or "manual"
        backbone = escape_latex(row.get("backbone", ""))
        try:
            metric_value = float(row.get("metric", 0.0))
        except Exception:
            metric_value = 0.0
        metric_display, _ = format_metric(str(row.get("metric_name", "metric")), metric_value)
        desc = str(row.get("description", ""))
        loss = ""
        for part in desc.split("|"):
            part = part.strip()
            if part in ("bce", "focal_bce", "cross_entropy", "focal", "dice_bce",
                        "tversky_bce", "ordinal"):
                loss = part
                break
        lines.append(f"{recipe} & {backbone} & {escape_latex(loss)} & "
                      f"{escape_latex(str(row.get('metric_name', 'metric')))} & {metric_display} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_per_class_table(checkpoint_results, final_eval=None, benchmarks=None):
    source = final_eval.get("results", {}) if final_eval else checkpoint_results
    split_name = str(final_eval.get("split", "validation")) if final_eval else "validation"
    per_class = source.get("per_class_auc")
    if not per_class:
        return ""

    has_benchmarks = benchmarks and len(benchmarks) > 0
    benchmark_dict = {}
    if has_benchmarks:
        for name, value in benchmarks:
            benchmark_dict[name.lower().replace(" ", "_")] = value

    if has_benchmarks:
        header = r"Class & AUC (\%) & Published (\%) \\"
        cols = "lcc"
    else:
        header = r"Class & AUC (\%) \\"
        cols = "lc"

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{Per-class {escape_latex(split_name)} AUC.}}",
        r"\label{tab:perclass}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for class_name, auc in per_class.items():
        class_display = escape_latex(class_name.replace("_", " "))
        auc_str = f"{float(auc):.2f}"
        if has_benchmarks:
            bench_val = benchmark_dict.get(class_name.lower().replace(" ", "_"), None)
            bench_str = f"{bench_val:.2f}" if bench_val else "---"
            lines.append(f"{class_display} & {auc_str} & {bench_str} \\\\")
        else:
            lines.append(f"{class_display} & {auc_str} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_results_section(results_df, checkpoint_results, checkpoint_config,
                              final_eval, dataset_name):
    """Generate a real Results section with narrative text."""
    if results_df.empty:
        return r"\section{Results}" + "\nNo experiment data available.\n"

    best_row = results_df.iloc[0]
    metric_name = str(best_row.get("metric_name", "metric"))
    metric_value = float(best_row.get("metric", 0.0))
    metric_display, _ = format_metric(metric_name, metric_value)
    backbone = checkpoint_config.get("backbone", best_row.get("backbone", "unknown"))
    num_experiments = len(results_df)

    lines = [
        r"\section{Results}",
        "",
        f"A total of {num_experiments} experiments were logged during the autonomous search campaign. "
    ]

    # Architecture diversity
    if "backbone" in results_df.columns:
        unique_backbones = results_df["backbone"].nunique()
        if unique_backbones > 1:
            lines.append(
                f"The agent explored {unique_backbones} different backbone architectures. "
            )

    lines.append(
        f"The best validation performance was {metric_display} {escape_latex(metric_name)}, "
        f"achieved with {escape_latex(backbone)}. "
    )

    # Final eval
    if final_eval:
        final_metric = float(final_eval.get("primary_metric", 0.0))
        final_display, _ = format_metric(metric_name, final_metric)
        split_name = final_eval.get("split", "test")
        lines.append(
            f"When evaluated on the held-out {escape_latex(split_name)} split, "
            f"this model achieved {final_display} {escape_latex(metric_name)}. "
        )

    # Benchmark comparison
    benchmarks = parse_published_benchmarks(dataset_name)
    if benchmarks and final_eval:
        final_metric = float(final_eval.get("primary_metric", metric_value))
        for pub_name, pub_value in benchmarks[:3]:
            delta = final_metric - pub_value
            direction = "above" if delta >= 0 else "below"
            lines.append(
                f"For reference, {escape_latex(pub_name)} reported {pub_value:.2f}\\%, "
                f"placing our result {abs(delta):.2f} points {direction}. "
            )

    # Confidence intervals
    source = final_eval.get("results", {}) if final_eval else checkpoint_results
    ci = source.get("confidence_interval") or source.get("ci")
    if ci:
        ci_low = ci.get("lower", ci.get("low", 0))
        ci_high = ci.get("upper", ci.get("high", 0))
        lines.append(
            f"The 95\\% bootstrap confidence interval was [{ci_low:.2f}\\%, {ci_high:.2f}\\%]. "
        )

    # Per-class range
    per_class = source.get("per_class_auc")
    if per_class:
        values = [float(v) for v in per_class.values()]
        worst_class = min(per_class, key=lambda k: float(per_class[k]))
        best_class = max(per_class, key=lambda k: float(per_class[k]))
        lines.append(
            f"Per-class performance ranged from {min(values):.2f}\\% "
            f"({escape_latex(worst_class.replace('_', ' '))}) to {max(values):.2f}\\% "
            f"({escape_latex(best_class.replace('_', ' '))}). "
        )

    return "\n".join(lines)


def generate_figure_section(figures_dir, output_path, final_eval=None):
    if not figures_dir:
        return ""
    figures_path = Path(figures_dir)
    if not figures_path.exists():
        return ""

    output_parent = output_path.parent
    split_name = str(final_eval.get("split", "validation")) if final_eval else "validation"
    figure_specs = [
        ("progress.png", "Search trajectory across recorded experiments.", "progress", True),
        ("benchmark_comparison.png", "Benchmark comparison against published baselines.", "benchmark", False),
        ("per_class_comparison.png", "Per-class AUC comparison with published baselines.", "perclass", True),
        ("roc_curves.png", f"ROC curves on the {split_name} set.", "roc", True),
    ]

    blocks = [r"\section{Figures}"]
    found = False
    for filename, caption, label, wide in figure_specs:
        fpath = figures_path / filename
        if not fpath.exists():
            continue
        rel = escape_latex(os.path.relpath(fpath, output_parent).replace("\\", "/"))
        env = "figure*" if wide else "figure"
        w = r"\textwidth" if wide else r"0.98\columnwidth"
        blocks.append("\n".join([
            rf"\begin{{{env}}}[t]", r"\centering",
            rf"\includegraphics[width={w}]{{{rel}}}",
            rf"\caption{{{escape_latex(caption)}}}", rf"\label{{fig:{label}}}",
            rf"\end{{{env}}}",
        ]))
        found = True
    return "\n\n".join(blocks) if found else ""


def generate_reproducibility_section(checkpoint_config, dataset_name):
    """Generate reproducibility section with exact commands and software versions."""
    lines = [
        r"\section{Reproducibility}",
        "All code is available in the autoresearch-medimage repository. "
        "To reproduce the winning configuration:",
        r"\begin{verbatim}",
        f"uv run prepare.py --dataset {dataset_name}",
    ]
    recipe = checkpoint_config.get("recipe", "") if checkpoint_config else ""
    if recipe:
        lines.append(f"uv run train.py --recipe {recipe} --save-checkpoint")
    else:
        lines.append(f"uv run train.py --dataset {dataset_name} --save-checkpoint")
    lines.extend([
        r"\end{verbatim}",
        "",
        f"Experiments were conducted on Python {platform.python_version()} "
        f"with PyTorch {torch.__version__} on {platform.system()} ({platform.machine()}). ",
    ])
    return "\n".join(lines)


def generate_paper(results_df, checkpoint_results, checkpoint_config,
                    dataset_name, figures_dir, output_path, final_eval):
    date_str = datetime.now().strftime("%B %Y")
    benchmarks = parse_published_benchmarks(dataset_name)

    abstract = generate_abstract(results_df, checkpoint_config, final_eval, dataset_name)
    dataset_section = generate_dataset_section(checkpoint_config, results_df, dataset_name)
    methods_section = generate_methods_section(checkpoint_config, results_df)
    results_section = generate_results_section(
        results_df, checkpoint_results, checkpoint_config, final_eval, dataset_name)
    results_table = generate_results_table(results_df)
    per_class_table = generate_per_class_table(checkpoint_results, final_eval, benchmarks)
    figure_section = generate_figure_section(figures_dir, output_path, final_eval)
    repro_section = generate_reproducibility_section(checkpoint_config, dataset_name)

    return rf"""\documentclass[10pt,twocolumn]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{booktabs}}
\usepackage{{geometry}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{amsmath}}
\geometry{{margin=0.85in}}

\title{{Agent-Driven Medical Imaging Research on {escape_latex(dataset_name)}}}
\author{{autoresearch-medimage}}
\date{{{date_str}}}

\begin{{document}}
\maketitle

\begin{{abstract}}
{abstract}
\end{{abstract}}

\section{{Introduction}}
Deep learning has achieved strong results across a range of medical imaging
tasks, but the experiment design process remains largely manual. Researchers
typically select architectures, tune hyperparameters, and evaluate results
through iterative trial and error. This work demonstrates an automated
approach using the autoresearch-medimage framework, in which an AI agent
autonomously explores the model design space under a fixed time budget.

The agent modifies a single training script (\texttt{{train.py}}) while the
data loading and evaluation infrastructure (\texttt{{prepare.py}}) remains
fixed. Each experiment is limited to a 5-minute wall-clock budget. The agent
retains improvements and discards regressions, building on a monotonically
improving baseline.

{dataset_section}

{methods_section}

{results_section}

{results_table}

{per_class_table}

{figure_section}

\section{{Discussion}}
The autonomous experiment loop explored multiple architectures, optimizers,
and augmentation strategies within the fixed time budget constraint. This
approach trades the depth of individual training runs for breadth of
exploration, which is well-suited to the early stages of model development
where the space of viable configurations is large.

\subsection{{Limitations}}
\begin{{itemize}}
\item The fixed 5-minute budget limits each experiment to a fraction of a
full training run, which may underestimate the performance of
slower-converging architectures.
\item The agent's search strategy is greedy and may miss configurations
that require multiple coordinated changes to show improvement.
\item Results should be validated on external datasets before drawing
clinical conclusions.
\end{{itemize}}

{repro_section}

\end{{document}}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate a paper draft from experiment results and checkpoint metadata"
    )
    parser.add_argument("--results", type=str, default=RESULTS_FILE)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--session", type=str, default=None, help="Optional session filter")
    parser.add_argument("--figures-dir", type=str, default=None)
    parser.add_argument("--final-eval", type=str, default=None)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    results_df = load_results(args.results, dataset_filter=args.dataset, session_filter=args.session)
    checkpoint_results, checkpoint_config = {}, {}
    dataset_name = args.dataset or "medical imaging dataset"
    final_eval = load_final_eval(args.final_eval)

    if not results_df.empty:
        checkpoint_results, checkpoint_config = load_checkpoint_metadata(results_df.iloc[0])
        dataset_name = str(results_df.iloc[0].get("dataset", dataset_name))

    output_path = Path(args.output or os.path.join(PAPER_DIR, "paper.tex"))
    figures_dir = args.figures_dir or str(output_path.parent / f"{dataset_name}_figures")

    latex = generate_paper(
        results_df, checkpoint_results, checkpoint_config,
        dataset_name, figures_dir, output_path, final_eval,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    print(f"Paper draft written to: {output_path}")

    if args.compile:
        pdflatex = shutil.which("pdflatex")
        if not pdflatex:
            print("pdflatex not found — .tex file is ready for manual compilation.")
            return
        result = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", output_path.name],
            cwd=str(output_path.parent), capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"PDF generated: {output_path.with_suffix('.pdf')}")
        else:
            print("LaTeX compilation failed. Check the .tex file for errors.")


if __name__ == "__main__":
    main()
