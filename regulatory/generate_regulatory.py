"""
Experimental regulatory scaffold generator.

These documents are research templates intended to organize evidence gaps and
submission preparation work. They are not submission-ready filings.
"""

import argparse
import os
from datetime import datetime

import pandas as pd

REGULATORY_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(REGULATORY_DIR, "output")
RESULTS_FILE = os.path.join(os.path.dirname(REGULATORY_DIR), "results.tsv")


def load_best_result(results_path, dataset_filter=None, session_filter=None):
    if not results_path or not os.path.exists(results_path):
        return {}
    df = pd.read_csv(results_path, sep="\t")
    if dataset_filter and "dataset" in df.columns:
        df = df[df["dataset"].astype(str) == str(dataset_filter)]
    if "session" in df.columns:
        session_series = df["session"].fillna("").astype(str).str.strip()
        if session_filter:
            df = df[session_series == str(session_filter)]
        else:
            non_empty = session_series[session_series != ""]
            if not non_empty.empty:
                df = df[session_series == str(non_empty.iloc[-1])]
    elif session_filter:
        raise ValueError("results.tsv does not contain a 'session' column")
    if "status" in df.columns:
        keep_like = df[df["status"].isin(["keep", "candidate"])]
        if not keep_like.empty:
            df = keep_like
    if df.empty or "metric" not in df.columns:
        return {}
    df["metric"] = pd.to_numeric(df["metric"], errors="coerce")
    return df.sort_values("metric", ascending=False).iloc[0].to_dict()


def format_metric_value(value):
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def infer_defaults(best_row, args):
    dataset = args.dataset or best_row.get("dataset", "Custom medical imaging dataset")
    metric_name = args.metric or best_row.get("metric_name", "metric")
    best_metric = format_metric_value(args.best_metric or best_row.get("metric", "[TBD]"))
    backbone = best_row.get("backbone", "unknown")
    recipe = best_row.get("recipe", "")

    dataset_lower = str(dataset).lower()
    if "chest" in dataset_lower or "xray" in dataset_lower:
        intended_use = (
            args.intended_use
            or "Research-use-only model for assisting the review of chest radiographs in retrospective evaluation workflows."
        )
        device_name = args.device_name or "Experimental Chest Imaging Research Model"
    else:
        intended_use = (
            args.intended_use
            or "Research-use-only model for evaluating medical imaging data within retrospective experimentation workflows."
        )
        device_name = args.device_name or "Experimental Medical Imaging Research Model"

    return {
        "device_name": device_name,
        "intended_use": intended_use,
        "dataset": dataset,
        "metric_name": metric_name,
        "best_metric": best_metric,
        "backbone": backbone,
        "recipe": recipe,
    }


def shared_gap_section():
    return """## Evidence Gaps / Not Submission Ready

This scaffold is intentionally conservative. Before any submission, at minimum
the following evidence and process gaps must be closed:

- prospective or real-world validation
- subgroup and fairness analysis
- reader study or workflow-impact study where clinically relevant
- software lifecycle, risk management, and QMS evidence
- cybersecurity, interoperability, and deployment documentation
- labeling, intended-use, and contraindication review by regulatory counsel
"""


def generate_fda(config):
    return f"""# FDA Research Scaffold - {config['device_name']}
# Generated: {datetime.now().strftime('%Y-%m-%d')}
# STATUS: Experimental template only

## Device Summary

- Device name: {config['device_name']}
- Intended use: {config['intended_use']}
- Validation dataset: {config['dataset']}
- Primary metric: {config['metric_name']}
- Best logged value: {config['best_metric']}
- Backbone: {config['backbone']}
- Recipe or run label: {config['recipe'] or 'manual'}

## FDA-Facing Questions To Resolve

- product code and regulation number for the actual intended use
- whether the device is CADe, CADx, triage, quantification, or another SaMD category
- predicate strategy versus De Novo strategy
- required standalone testing and any reader-study expectations
- software documentation scope under IEC 62304 and related guidance

## Draft Submission Workstreams

- intended-use and indications-for-use statement
- validation dataset description and inclusion/exclusion criteria
- model architecture, training controls, and change-management plan
- statistical analysis plan and confidence intervals
- labeling, warnings, and deployment environment assumptions

{shared_gap_section()}

## Disclaimer

This file is a research planning scaffold. It is not legal, regulatory, or
submission advice.
"""


def generate_ce(config):
    return f"""# CE / EU MDR Research Scaffold - {config['device_name']}
# Generated: {datetime.now().strftime('%Y-%m-%d')}
# STATUS: Experimental template only

## Device Summary

- Device name: {config['device_name']}
- Intended use: {config['intended_use']}
- Validation dataset: {config['dataset']}
- Primary metric: {config['metric_name']}
- Best logged value: {config['best_metric']}
- Backbone: {config['backbone']}

## EU MDR Workstreams

- classification under Rule 11 based on actual intended use
- technical documentation and software lifecycle evidence
- clinical evaluation report scope and literature strategy
- risk management and post-market surveillance planning
- usability, interoperability, and cybersecurity evidence

## Evidence Package Notes

- this scaffold assumes a research model, not a market-ready device
- any clinical, workflow, or economic claims require dedicated supporting evidence
- literature benchmarking does not replace device-specific validation

{shared_gap_section()}

## Disclaimer

This file is a research planning scaffold. It is not a CE-mark-ready dossier.
"""


def generate_ukca(config):
    return f"""# UKCA / MHRA Research Scaffold - {config['device_name']}
# Generated: {datetime.now().strftime('%Y-%m-%d')}
# STATUS: Experimental template only

## Device Summary

- Device name: {config['device_name']}
- Intended use: {config['intended_use']}
- Validation dataset: {config['dataset']}
- Primary metric: {config['metric_name']}
- Best logged value: {config['best_metric']}
- Backbone: {config['backbone']}

## UK-Specific Planning Notes

- confirm MHRA pathway and UKCA/CE applicability at the planned launch date
- map NICE evidence expectations to the actual use case
- define NHS interoperability and deployment assumptions if relevant
- define UK-specific population validation needs

## Documentation Tracks

- technical documentation and software lifecycle controls
- clinical evaluation and any UK population evidence
- safety case, risk management, and post-market surveillance
- deployment, interoperability, cybersecurity, and service ownership

{shared_gap_section()}

## Disclaimer

This file is a research planning scaffold. It is not a UKCA-ready submission pack.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate experimental regulatory scaffolds from results.tsv and run metadata"
    )
    parser.add_argument("--type", type=str, default="all", choices=["all", "fda", "ce", "ukca"], help="Documentation type to generate")
    parser.add_argument("--device-name", type=str, default=None, help="Optional device name override")
    parser.add_argument("--intended-use", type=str, default=None, help="Optional intended-use override")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset override")
    parser.add_argument("--session", type=str, default=None, help="Optional session filter")
    parser.add_argument("--metric", type=str, default=None, help="Optional primary metric override")
    parser.add_argument("--best-metric", type=str, default=None, help="Optional best metric override")
    parser.add_argument("--results", type=str, default=RESULTS_FILE, help="Path to results.tsv")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory for generated docs")
    args = parser.parse_args()

    best_row = load_best_result(args.results, dataset_filter=args.dataset, session_filter=args.session)
    config = infer_defaults(best_row, args)
    os.makedirs(args.output_dir, exist_ok=True)

    generators = {
        "fda": ("fda_510k_scaffold.md", generate_fda),
        "ce": ("ce_mdr_scaffold.md", generate_ce),
        "ukca": ("ukca_scaffold.md", generate_ukca),
    }
    doc_types = generators.keys() if args.type == "all" else [args.type]

    for doc_type in doc_types:
        filename, generator = generators[doc_type]
        output_path = os.path.join(args.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(generator(config))
        print(f"Generated: {output_path}")

    print("\nThese outputs are experimental scaffolds. Review with qualified regulatory professionals before any external use.")


if __name__ == "__main__":
    main()
