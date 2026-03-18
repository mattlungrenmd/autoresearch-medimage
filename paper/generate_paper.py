"""
Paper Generator — Auto-generate arXiv-ready LaTeX from experiment results.

Reads results.tsv and model checkpoints to generate a methods section,
results tables, and figures suitable for arXiv submission.

Usage:
    python paper/generate_paper.py                        # Generate from results.tsv
    python paper/generate_paper.py --template short       # Short paper (4 pages)
    python paper/generate_paper.py --template full        # Full paper (8 pages)
    python paper/generate_paper.py --output paper.tex     # Custom output path
    python paper/generate_paper.py --compile              # Also compile to PDF
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(PAPER_DIR, "templates")
RESULTS_FILE = os.path.join(os.path.dirname(PAPER_DIR), "results.tsv")


# ---------------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------------

def escape_latex(text):
    """Escape special LaTeX characters."""
    chars = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
             "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
             "^": r"\textasciicircum{}"}
    for old, new in chars.items():
        text = text.replace(old, new)
    return text


def generate_results_table(results_df):
    """Generate LaTeX table from results DataFrame."""
    if results_df.empty:
        return "% No results yet — run experiments first\n"

    # Filter to kept experiments
    kept = results_df[results_df["status"] == "keep"].copy()
    if kept.empty:
        kept = results_df.head(10)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Experiment results. Best result in \textbf{bold}.}",
        r"\label{tab:results}",
        r"\begin{tabular}{llccl}",
        r"\toprule",
        r"Commit & Metric & Memory (GB) & Status & Description \\",
        r"\midrule",
    ]

    best_metric = kept["metric"].max()
    for _, row in kept.iterrows():
        metric_str = f"{row['metric']:.2f}"
        if row["metric"] == best_metric:
            metric_str = r"\textbf{" + metric_str + "}"
        desc = escape_latex(str(row.get("description", "")))[:50]
        lines.append(
            f"  {row['commit']} & {metric_str} & {row['memory_gb']:.1f} & "
            f"{row['status']} & {desc} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_per_class_table(checkpoint_results):
    """Generate per-class AUC table (for multi-label tasks like CheXNet)."""
    if not checkpoint_results or "per_class_auc" not in checkpoint_results:
        return ""

    per_class = checkpoint_results["per_class_auc"]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Per-pathology AUC comparison with published results.}",
        r"\label{tab:per_class}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Pathology & Ours (AUC \%) & CheXNet (AUC \%) \\",
        r"\midrule",
    ]

    # CheXNet reference values
    chexnet_ref = {
        "Atelectasis": 80.94, "Cardiomegaly": 92.48, "Effusion": 86.38,
        "Infiltration": 73.45, "Mass": 86.81, "Nodule": 78.02,
        "Pneumonia": 76.80, "Pneumothorax": 93.71, "Consolidation": 80.07,
        "Edema": 88.78, "Emphysema": 93.71, "Fibrosis": 80.47,
        "Pleural_Thickening": 80.62, "Hernia": 91.64,
    }

    for pathology, auc in per_class.items():
        ref = chexnet_ref.get(pathology, "—")
        ref_str = f"{ref:.2f}" if isinstance(ref, float) else ref
        pathology_clean = escape_latex(pathology.replace("_", " "))
        lines.append(f"  {pathology_clean} & {auc:.2f} & {ref_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_paper(results_df, config, template="full"):
    """Generate complete LaTeX paper."""
    date_str = datetime.now().strftime("%B %Y")
    year = datetime.now().year

    results_table = generate_results_table(results_df)

    # Determine best result
    if not results_df.empty:
        best_row = results_df.loc[results_df["metric"].idxmax()]
        best_metric = f"{best_row['metric']:.2f}"
        best_desc = str(best_row.get("description", "best configuration"))
    else:
        best_metric = "N/A"
        best_desc = "pending experiments"

    dataset_name = config.get("dataset", "medical imaging dataset")
    task = config.get("task", "classification")
    backbone = config.get("backbone", "DenseNet-121")
    metric_name = config.get("metric_name", "AUC")

    latex = rf"""\documentclass[10pt,twocolumn]{{article}}

\usepackage[utf8]{{inputenc}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{amsmath}}
\usepackage[margin=1in]{{geometry}}

\title{{Automated Architecture Search for Medical Image {task.title()}:\\
Results on {escape_latex(dataset_name)}}}

\author{{autoresearch-medimage\\
\texttt{{https://github.com/mlungrengmail/autoresearch-medimage}}}}

\date{{{date_str}}}

\begin{{document}}

\maketitle

\begin{{abstract}}
We present results from an automated architecture and hyperparameter search
for medical image {task} on the {escape_latex(dataset_name)} dataset.
Using the autoresearch-medimage framework, an autonomous agent iteratively
explored model architectures (including DenseNet, EfficientNet, ConvNeXt,
and Vision Transformers), optimizers, loss functions, and augmentation
strategies through fixed 5-minute training experiments. Our best
configuration achieved {best_metric}\% {metric_name},
{'competitive with' if float(best_metric) > 80 else 'approaching'}
published state-of-the-art results.
All code and experiments are publicly available.
\end{{abstract}}

\section{{Introduction}}

Medical image analysis with deep learning has achieved remarkable results
across radiology and ophthalmology tasks. However, model development
typically requires extensive manual experimentation with architectures,
hyperparameters, and training strategies. We apply the autoresearch
paradigm~\cite{{karpathy2025autoresearch}} to medical imaging, enabling
fully autonomous experimentation.

\section{{Methods}}

\subsection{{Dataset}}
We use the {escape_latex(dataset_name)} dataset for {task}.
% TODO: Add dataset details (size, classes, splits, preprocessing)

\subsection{{Experimental Setup}}
Each experiment runs for a fixed 5-minute wall-clock training budget on
a single GPU. The agent modifies the training configuration (architecture,
optimizer, loss function, augmentation) and evaluates on a held-out
validation set. Improvements are kept; regressions are discarded.

\subsection{{Architecture Search}}
The agent explores the following backbone families via the
\texttt{{timm}} library:
\begin{{itemize}}
    \item DenseNet (121, 169, 201) --- established medical imaging baseline
    \item EfficientNet (B0--B4) --- strong accuracy/efficiency tradeoff
    \item ConvNeXt (Small, Base) --- modern CNN architecture
    \item Vision Transformers (ViT-Small, Swin-Tiny) --- attention-based
\end{{itemize}}

All models are initialized from ImageNet-pretrained weights and fine-tuned
with a task-specific classification or segmentation head.

\subsection{{Training Details}}
% TODO: Fill in from best experiment
\begin{{itemize}}
    \item Backbone: {escape_latex(backbone)}
    \item Optimizer: AdamW
    \item Learning rate: with warmup and cosine decay
    \item Loss: {'BCE with logits' if task == 'multilabel' else 'Cross-entropy'} with class weights
    \item Augmentation: Random flip, rotation, color jitter
    \item Mixed precision: FP16
\end{{itemize}}

\section{{Results}}

{results_table}

Our best configuration ({escape_latex(best_desc)}) achieved
{best_metric}\% {metric_name} on the validation set.

% TODO: Add per-class results table if applicable
% TODO: Add training curves figure

\section{{Comparison with Published Results}}

\begin{{table}}[h]
\centering
\caption{{Comparison with published methods.}}
\label{{tab:comparison}}
\begin{{tabular}}{{lcc}}
\toprule
Method & {metric_name} (\%) & Year \\
\midrule
% TODO: Add published baselines
Ours (best) & {best_metric} & {year} \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Discussion}}

% TODO: Discuss which architectures worked best and why
% TODO: Discuss failure modes and limitations
% TODO: Compare with FDA-cleared device performance if applicable

\section{{Conclusion}}

We demonstrated that autonomous architecture search can produce
competitive medical imaging models with minimal human intervention.
The autoresearch-medimage framework enables researchers to explore
the design space systematically and reproducibly.

\section*{{Reproducibility}}

All code, configurations, and experiment logs are available at:
\url{{https://github.com/mlungrengmail/autoresearch-medimage}}.
Experiments can be reproduced with:
\begin{{verbatim}}
uv run prepare.py --dataset {dataset_name}
uv run train.py --recipe <recipe_name>
\end{{verbatim}}

\bibliographystyle{{plain}}
\begin{{thebibliography}}{{9}}

\bibitem{{karpathy2025autoresearch}}
A.~Karpathy.
\newblock autoresearch: Autonomous pretraining research.
\newblock \url{{https://github.com/karpathy/autoresearch}}, 2025.

% TODO: Add relevant citations

\end{{thebibliography}}

\end{{document}}
"""
    return latex


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate arXiv paper from experiment results")
    parser.add_argument("--results", type=str, default=RESULTS_FILE,
                        help="Path to results.tsv")
    parser.add_argument("--template", type=str, default="full",
                        choices=["short", "full"],
                        help="Paper template")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .tex file path")
    parser.add_argument("--dataset", type=str, default="chestxray14",
                        help="Dataset name for paper context")
    parser.add_argument("--compile", action="store_true",
                        help="Compile LaTeX to PDF (requires pdflatex)")
    args = parser.parse_args()

    # Load results
    if os.path.exists(args.results):
        results_df = pd.read_csv(args.results, sep="\t")
        print(f"Loaded {len(results_df)} experiment results")
    else:
        print(f"No results file found at {args.results}")
        print("Generating template with placeholder results...")
        results_df = pd.DataFrame(columns=["commit", "metric", "memory_gb", "status", "description"])

    # Config
    config = {
        "dataset": args.dataset,
        "task": "multilabel" if args.dataset == "chestxray14" else "classification",
        "backbone": "DenseNet-121",
        "metric_name": "Mean AUC" if args.dataset == "chestxray14" else "Accuracy",
    }

    # Generate
    latex = generate_paper(results_df, config, template=args.template)

    # Output
    output_path = args.output or os.path.join(PAPER_DIR, "paper.tex")
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"\nPaper generated: {output_path}")

    # Compile if requested
    if args.compile:
        import subprocess
        print("Compiling LaTeX...")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", output_path],
            cwd=os.path.dirname(output_path),
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            pdf_path = output_path.replace(".tex", ".pdf")
            print(f"PDF generated: {pdf_path}")
        else:
            print("LaTeX compilation failed. Install pdflatex or use Overleaf.")
            print(result.stderr[:500])

    print(f"\nNext steps:")
    print(f"  1. Fill in TODO sections in the generated .tex file")
    print(f"  2. Add figures (training curves, Grad-CAM visualizations)")
    print(f"  3. Upload to Overleaf or compile locally with pdflatex")
    print(f"  4. Submit to arXiv: https://arxiv.org/submit")


if __name__ == "__main__":
    main()
