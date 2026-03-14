"""
Experimental paper generator.

Generates a draft LaTeX scaffold from results.tsv and, when available,
checkpoint metadata. The output is intended as a starting point for a paper,
not a camera-ready or automatically submission-ready manuscript.
"""

import argparse
import os
import shutil
import subprocess
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
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
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
    final_eval_path = str(final_eval_path)
    if not os.path.exists(final_eval_path):
        return None
    try:
        return torch.load(final_eval_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(final_eval_path, map_location="cpu")


def format_metric(metric_name, metric_value):
    metric_name = str(metric_name)
    if metric_name in {"kappa", "quadratic_kappa"}:
        return f"{metric_value:.4f}", metric_name
    return f"{metric_value:.2f}\\%", metric_name


def parse_published_chexnet_metric(dataset_name):
    if dataset_name != "chestxray14":
        return None
    benchmark_path = os.path.join(BENCHMARK_DIR, "chestxray14.md")
    if not os.path.exists(benchmark_path):
        return None
    text = Path(benchmark_path).read_text(encoding="utf-8")
    match = re.search(r"\|\s*\*\*CheXNet\*\*.*?\|\s*\*\*(\d+(?:\.\d+)?)\*\*", text)
    if not match:
        return None
    return float(match.group(1))


def generate_results_table(results_df):
    if results_df.empty:
        return "% No experiment rows found.\n"

    columns = [col for col in ["dataset", "session", "recipe", "backbone", "metric_name", "metric", "status"] if col in results_df.columns]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Run summary from the experiment ledger.}",
        r"\begin{tabular}{lllllll}",
        r"\toprule",
        r"Dataset & Session & Recipe & Backbone & Metric & Value & Status \\",
        r"\midrule",
    ]
    for _, row in results_df.head(8).iterrows():
        dataset = escape_latex(row.get("dataset", ""))
        session = escape_latex(row.get("session", ""))
        recipe = escape_latex(row.get("recipe", "")) or "manual"
        backbone = escape_latex(row.get("backbone", ""))
        metric_name = escape_latex(row.get("metric_name", "metric"))
        try:
            metric_value = float(row.get("metric", 0.0))
        except Exception:
            metric_value = 0.0
        metric_display, _ = format_metric(str(row.get("metric_name", "metric")), metric_value)
        status = escape_latex(row.get("status", "candidate"))
        lines.append(f"{dataset} & {session} & {recipe} & {backbone} & {metric_name} & {metric_display} & {status} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_per_class_table(checkpoint_results, final_eval=None):
    source_results = final_eval.get("results", {}) if final_eval else checkpoint_results
    split_name = str(final_eval.get("split", "validation")) if final_eval else "validation"
    per_class = source_results.get("per_class_auc")
    if not per_class:
        return ""

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{Per-class {escape_latex(split_name)} AUC from the saved run artifacts.}}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Class & AUC (\%) \\",
        r"\midrule",
    ]
    for class_name, auc in per_class.items():
        lines.append(f"{escape_latex(class_name.replace('_', ' '))} & {float(auc):.2f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_manual_section(template):
    lines = [
        r"\section{Manual Completion Notes}",
        r"This draft was generated from local experiment artifacts.",
        r"The following sections typically require human completion before submission:",
        r"\begin{itemize}",
        r"\item related work and citations",
        r"\item clinical context and dataset provenance narrative",
        r"\item limitations, failure analysis, and subgroup discussion",
        r"\item figures and qualitative examples",
        r"\end{itemize}",
    ]
    if template == "full":
        lines.extend(
            [
                r"\section{Experimental Positioning}",
                r"This repository is designed as an agent-first research framework.",
                r"The generated manuscript should be reviewed for factual accuracy and claim scope before sharing.",
            ]
        )
    return "\n".join(lines)


def generate_protocol_section(final_eval):
    if not final_eval:
        return ""
    split_name = final_eval.get("split", "test")
    return "\n".join(
        [
            r"\section{Evaluation Protocol}",
            r"During the autonomous search loop, model changes were selected using the validation split only.",
            rf"A separate held-out {escape_latex(split_name)} split was reserved for final reporting and was not used to rank intermediate experiments.",
            r"This mirrors the agent-first autoresearch workflow while improving the final reporting discipline relative to a validation-only benchmark.",
        ]
    )


def generate_final_result_section(final_eval, dataset_name):
    if not final_eval:
        return ""
    results = final_eval.get("results", {})
    metric_name = final_eval.get("metric_name", "metric")
    metric_value = float(final_eval.get("primary_metric", 0.0))
    metric_display, metric_name = format_metric(metric_name, metric_value)
    lines = [
        r"\section{Held-Out Final Evaluation}",
        rf"The best kept run was re-evaluated on the untouched {escape_latex(final_eval.get('split', 'test'))} split of {escape_latex(dataset_name)}.",
        rf"The held-out result was {metric_display} {escape_latex(metric_name)}.",
    ]
    if "per_class_auc" in results:
        lines.append(r"The per-class table in the manuscript figures should be interpreted as the held-out result when available.")
    published_chexnet = parse_published_chexnet_metric(dataset_name)
    if published_chexnet is not None:
        delta = metric_value - published_chexnet
        comparison = "above" if delta >= 0 else "below"
        lines.append(
            rf"For context, the original CheXNet paper reported 84.10\% mean\_auc, placing this run {abs(delta):.2f} points {comparison} that reference."
        )
    return "\n".join(lines)


def figure_block(path_str, caption, label, wide=False):
    env = "figure*" if wide else "figure"
    width = r"\textwidth" if wide else r"0.98\columnwidth"
    path = escape_latex(path_str.replace("\\", "/"))
    return "\n".join(
        [
            rf"\begin{{{env}}}[t]",
            r"\centering",
            rf"\includegraphics[width={width}]{{{path}}}",
            rf"\caption{{{escape_latex(caption)}}}",
            rf"\label{{fig:{escape_latex(label)}}}",
            rf"\end{{{env}}}",
        ]
    )


def generate_figure_section(figures_dir, output_path, final_eval=None):
    if not figures_dir:
        return ""

    figures_path = Path(figures_dir)
    if not figures_path.exists():
        return ""

    output_parent = output_path.parent
    split_name = str(final_eval.get("split", "validation")) if final_eval else "validation"
    figure_specs = [
        ("progress.png", "Autoresearch search trajectory across recorded experiments.", "progress", True),
        ("benchmark_comparison.png", "Benchmark comparison against published ChestX-ray14 baselines.", "benchmark", False),
        ("per_class_comparison.png", "Per-pathology AUC comparison with published baselines.", "perclass", True),
        ("roc_curves.png", f"Representative {split_name} ROC curves for selected pathologies.", "roc", True),
    ]

    blocks = [r"\section{Figures}"]
    found_any = False
    for filename, caption, label, wide in figure_specs:
        figure_path = figures_path / filename
        if not figure_path.exists():
            continue
        rel_path = os.path.relpath(figure_path, output_parent)
        blocks.append(figure_block(rel_path, caption, label, wide=wide))
        found_any = True

    return "\n\n".join(blocks) if found_any else ""


def generate_paper(
    results_df,
    checkpoint_results,
    checkpoint_config,
    dataset_name,
    template,
    figures_dir,
    output_path,
    final_eval,
):
    date_str = datetime.now().strftime("%B %Y")
    if not results_df.empty:
        best_row = results_df.iloc[0]
        metric_value = float(best_row.get("metric", 0.0))
        metric_name = str(best_row.get("metric_name", "metric"))
        metric_display, metric_name = format_metric(metric_name, metric_value)
        backbone = checkpoint_config.get("backbone", best_row.get("backbone", "unknown"))
        task = checkpoint_config.get("task", "classification")
        recipe = best_row.get("recipe", "") or "manual"
        dataset_label = best_row.get("dataset", dataset_name)
        if final_eval:
            final_metric_display, final_metric_name = format_metric(
                final_eval.get("metric_name", "metric"), float(final_eval.get("primary_metric", 0.0))
            )
            summary_sentence = (
                f"The best logged validation run used {escape_latex(backbone)} on {escape_latex(dataset_label)} "
                f"and reached {metric_display} {escape_latex(metric_name)}. "
                f"The same checkpoint achieved {final_metric_display} {escape_latex(final_metric_name)} on the held-out "
                f"{escape_latex(final_eval.get('split', 'test'))} split."
            )
        else:
            summary_sentence = (
                f"The best logged run used {escape_latex(backbone)} on {escape_latex(dataset_label)} "
                f"and reached {metric_display} {escape_latex(metric_name)}."
            )
    else:
        backbone = checkpoint_config.get("backbone", "unknown")
        task = checkpoint_config.get("task", "classification")
        recipe = "manual"
        summary_sentence = "No experiment rows were found, so this draft contains only structural placeholders."

    results_table = generate_results_table(results_df)
    per_class_table = generate_per_class_table(checkpoint_results, final_eval=final_eval)
    manual_section = generate_manual_section(template)
    figure_section = generate_figure_section(figures_dir, output_path, final_eval=final_eval)
    protocol_section = generate_protocol_section(final_eval)
    final_result_section = generate_final_result_section(final_eval, dataset_name)

    return rf"""\documentclass[10pt,twocolumn]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{booktabs}}
\usepackage{{geometry}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\geometry{{margin=0.85in}}

\title{{Experimental Draft: Agent-Driven Medical Imaging Research on {escape_latex(dataset_name)}}}
\author{{autoresearch-medvision}}
\date{{{date_str}}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This draft was generated from local experiment artifacts produced by the
autoresearch-medvision repository. It is intended as a scaffold for a paper,
not as a submission-ready manuscript. {summary_sentence}
\end{{abstract}}

\section{{Overview}}
The repository is an agent-first framework for short-horizon medical imaging
experiments. The inner loop preserves a fixed training budget and a simple
experiment ledger while allowing the agent to iterate on model and optimization
choices in \texttt{{train.py}}.

\section{{Run Summary}}
Dataset: {escape_latex(dataset_name)}\\
Task: {escape_latex(task)}\\
Recipe: {escape_latex(recipe)}\\
Backbone: {escape_latex(backbone)}

{results_table}

{per_class_table}

{protocol_section}

{final_result_section}

{figure_section}

\section{{Reproducibility}}
This draft was generated from \texttt{{results.tsv}} and optional checkpoint
metadata. To reproduce a run, prepare the dataset and launch the relevant
recipe or dataset configuration locally:
\begin{{verbatim}}
uv run prepare.py --dataset {dataset_name}
uv run train.py --dataset {dataset_name}
\end{{verbatim}}

{manual_section}

\end{{document}}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate an experimental paper scaffold from results.tsv and optional checkpoint metadata"
    )
    parser.add_argument("--results", type=str, default=RESULTS_FILE, help="Path to results.tsv")
    parser.add_argument("--template", type=str, default="full", choices=["short", "full"], help="Paper template")
    parser.add_argument("--output", type=str, default=None, help="Output .tex file path")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset filter")
    parser.add_argument("--session", type=str, default=None, help="Optional session filter")
    parser.add_argument("--figures-dir", type=str, default=None, help="Optional directory containing generated paper figures")
    parser.add_argument("--final-eval", type=str, default=None, help="Optional held-out final evaluation payload (.pt)")
    parser.add_argument("--compile", action="store_true", help="Compile LaTeX to PDF if pdflatex is available")
    args = parser.parse_args()

    results_df = load_results(args.results, dataset_filter=args.dataset, session_filter=args.session)
    checkpoint_results = {}
    checkpoint_config = {}
    dataset_name = args.dataset or "medical imaging dataset"
    final_eval = load_final_eval(args.final_eval)
    if not results_df.empty:
        checkpoint_results, checkpoint_config = load_checkpoint_metadata(results_df.iloc[0])
        dataset_name = str(results_df.iloc[0].get("dataset", dataset_name))

    output_path = Path(args.output or os.path.join(PAPER_DIR, "paper.tex"))
    default_figures_dir = output_path.parent / f"{dataset_name}_figures"
    figures_dir = args.figures_dir or str(default_figures_dir)
    latex = generate_paper(
        results_df,
        checkpoint_results,
        checkpoint_config,
        dataset_name,
        args.template,
        figures_dir,
        output_path,
        final_eval,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(latex)
    print(f"Experimental paper scaffold written to: {output_path}")

    if args.compile:
        pdflatex = shutil.which("pdflatex")
        if not pdflatex:
            print("pdflatex not found. Leaving the generated .tex file in place.")
            return
        result = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", output_path.name],
            cwd=str(output_path.parent),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"PDF generated: {output_path.with_suffix('.pdf')}")
        else:
            print("LaTeX compilation failed. Review the generated .tex file and local pdflatex setup.")


if __name__ == "__main__":
    main()
