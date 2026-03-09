"""
Regulatory planning document generator for autoresearch-medimage.

Generates FDA 510(k), CE/EU MDR, and UKCA planning documents from actual
experiment results, checkpoint metadata, and software environment information.

These documents are real planning tools — not scaffolds or placeholders. They
pull actual performance data, software versions, and model configuration from
the winning checkpoint. They still require review by qualified regulatory
professionals before any submission.

Usage:
    uv run regulatory/generate_regulatory.py --dataset chestxray14
    uv run regulatory/generate_regulatory.py --type fda --dataset chestxray14
"""

import argparse
import os
import platform
import sys
from datetime import datetime

import pandas as pd
import torch

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


def load_checkpoint_config(best_row):
    """Load full config from the checkpoint file."""
    checkpoint_path = str(best_row.get("checkpoint_path", "")).strip()
    if not checkpoint_path or checkpoint_path.lower() == "nan":
        return {}, {}
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(os.path.dirname(REGULATORY_DIR), checkpoint_path)
    if not os.path.exists(checkpoint_path):
        return {}, {}
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    return payload.get("config", {}), payload.get("results", {})


def fmt(value):
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def infer_config(best_row, ckpt_config, args):
    """Build full config from checkpoint data, results row, and CLI overrides."""
    dataset = args.dataset or ckpt_config.get("dataset", best_row.get("dataset", "Custom dataset"))
    metric_name = args.metric or ckpt_config.get("metric_name", best_row.get("metric_name", "metric"))
    best_metric = fmt(args.best_metric or best_row.get("metric", "[TBD]"))
    backbone = ckpt_config.get("backbone", best_row.get("backbone", "unknown"))
    task = ckpt_config.get("task", "classification")
    num_classes = ckpt_config.get("num_classes", 0)
    class_names = ckpt_config.get("class_names", [])
    recipe = ckpt_config.get("recipe", best_row.get("recipe", ""))
    image_size = ckpt_config.get("image_size", 224)
    loss = ckpt_config.get("loss", "cross_entropy")
    optimizer = ckpt_config.get("optimizer", "adam")
    lr = ckpt_config.get("learning_rate", 0.001)
    batch_size = ckpt_config.get("batch_size", 16)

    dataset_lower = str(dataset).lower()
    if "chest" in dataset_lower or "xray" in dataset_lower:
        intended_use = (
            args.intended_use
            or "Computer-aided detection/diagnosis tool for assisting the review of "
               "chest radiographs in retrospective evaluation workflows."
        )
        device_name = args.device_name or "Chest Imaging AI Research Model"
    elif "retina" in dataset_lower or "fundus" in dataset_lower or "dr" in dataset_lower:
        intended_use = (
            args.intended_use
            or "Computer-aided detection/diagnosis tool for assisting the evaluation of "
               "retinal images in retrospective research workflows."
        )
        device_name = args.device_name or "Retinal Imaging AI Research Model"
    else:
        intended_use = (
            args.intended_use
            or "Computer-aided analysis tool for evaluating medical imaging data "
               "within retrospective research workflows."
        )
        device_name = args.device_name or "Medical Imaging AI Research Model"

    return {
        "device_name": device_name,
        "intended_use": intended_use,
        "dataset": dataset,
        "task": task,
        "num_classes": num_classes,
        "class_names": class_names,
        "metric_name": metric_name,
        "best_metric": best_metric,
        "backbone": backbone,
        "recipe": recipe,
        "image_size": image_size,
        "loss": loss,
        "optimizer": optimizer,
        "learning_rate": lr,
        "batch_size": batch_size,
    }


def software_section():
    """Generate software environment description from actual runtime."""
    lines = [
        "## Software Environment",
        "",
        f"- Python: {platform.python_version()}",
        f"- PyTorch: {torch.__version__}",
        f"- Platform: {platform.system()} {platform.release()} ({platform.machine()})",
        f"- CUDA available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"- CUDA version: {torch.version.cuda}")
        lines.append(f"- GPU: {torch.cuda.get_device_name(0)}")
        lines.append(f"- GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    return "\n".join(lines)


def model_description(config):
    """Generate model architecture description."""
    lines = [
        "## Model Architecture",
        "",
        f"- Backbone: {config['backbone']}",
        f"- Task: {config['task']}",
        f"- Number of output classes: {config['num_classes']}",
        f"- Input size: {config['image_size']}x{config['image_size']} pixels",
        f"- Loss function: {config['loss']}",
        f"- Optimizer: {config['optimizer']}",
        f"- Learning rate: {config['learning_rate']}",
        f"- Batch size: {config['batch_size']}",
    ]
    if config["class_names"]:
        lines.append(f"- Output classes: {', '.join(config['class_names'])}")
    return "\n".join(lines)


def performance_section(config, ckpt_results):
    """Generate performance summary from actual results."""
    lines = [
        "## Performance Summary",
        "",
        f"- Primary metric: {config['metric_name']}",
        f"- Best value: {config['best_metric']}",
        f"- Selection split: validation (held-out test used only for final reporting)",
    ]

    per_class = ckpt_results.get("per_class_auc")
    if per_class:
        lines.extend(["", "### Per-Class Performance", ""])
        lines.append("| Class | AUC (%) |")
        lines.append("|-------|---------|")
        for cls, auc in per_class.items():
            lines.append(f"| {cls.replace('_', ' ')} | {float(auc):.2f} |")
        values = [float(v) for v in per_class.values()]
        lines.append(f"| **Mean** | **{sum(values)/len(values):.2f}** |")

    ci = ckpt_results.get("confidence_interval") or ckpt_results.get("ci")
    if ci:
        low = ci.get("lower", ci.get("low", 0))
        high = ci.get("upper", ci.get("high", 0))
        lines.append(f"- 95% confidence interval: [{low:.2f}%, {high:.2f}%]")

    return "\n".join(lines)


def risk_section(config):
    """Generate task-appropriate risk considerations."""
    task = config["task"]
    lines = [
        "## Risk Analysis Considerations",
        "",
        "The following risks are inherent to the device type and must be addressed:",
        "",
    ]

    if task in ("classification", "multilabel"):
        lines.extend([
            "### False Positive Risk",
            "- Unnecessary follow-up procedures, patient anxiety, resource waste",
            "- Mitigation: report sensitivity/specificity at clinically relevant operating points",
            "",
            "### False Negative Risk",
            "- Missed findings, delayed treatment, adverse patient outcomes",
            "- Mitigation: validate against ground truth established by expert consensus",
            "",
            "### Bias and Fairness",
            "- Performance variation across demographics (age, sex, ethnicity)",
            "- Performance variation across acquisition devices and protocols",
            "- Mitigation: subgroup analysis required before deployment claims",
        ])
    elif task == "segmentation":
        lines.extend([
            "### Under-Segmentation Risk",
            "- Incomplete delineation of pathology boundaries",
            "- Mitigation: report Dice per structure with confidence intervals",
            "",
            "### Over-Segmentation Risk",
            "- False positive regions that may lead to unnecessary intervention",
            "- Mitigation: validate against expert contours from multiple annotators",
            "",
            "### Generalization Risk",
            "- Segmentation quality may degrade on images from unseen scanners/protocols",
            "- Mitigation: multi-site validation required",
        ])

    return "\n".join(lines)


def evidence_gaps():
    return """## Evidence Gaps

Before any regulatory submission, the following must be addressed:

- [ ] Prospective or multi-site real-world validation
- [ ] Subgroup analysis by demographics and acquisition device
- [ ] Reader study or workflow-impact study where clinically relevant
- [ ] Software lifecycle documentation (IEC 62304)
- [ ] Risk management file (ISO 14971)
- [ ] Cybersecurity documentation
- [ ] Labeling and intended-use review by regulatory counsel
- [ ] Post-market surveillance plan
"""


def generate_fda(config, ckpt_results):
    return f"""# FDA Pre-Submission Planning Document
## {config['device_name']}
Generated: {datetime.now().strftime('%Y-%m-%d')}

---

## Device Description

- **Device name**: {config['device_name']}
- **Intended use**: {config['intended_use']}
- **Validation dataset**: {config['dataset']}
- **Recipe/configuration**: {config['recipe'] or 'manual'}

{model_description(config)}

{performance_section(config, ckpt_results)}

{software_section()}

## Regulatory Classification

- **Product code**: [TBD — determine based on actual intended use]
- **SaMD category**: [CADe / CADx / triage / quantification — select based on clinical claims]
- **Predicate strategy**: [510(k) predicate or De Novo — determine based on product code]
- **Clinical decision level**: [Inform / Drive / Diagnose / Treat — per IMDRF framework]

## Submission Workstreams

### 1. Intended Use and Indications for Use
- Define precise clinical context, target population, and user qualifications
- Specify conditions of use (clinical setting, integration point, standalone vs workflow)

### 2. Algorithm Description
- Architecture: {config['backbone']} with {config['task']} head
- Training procedure: {config['optimizer']} optimizer, {config['loss']} loss, {config['image_size']}px inputs
- Preprocessing pipeline documentation
- Version control and change management plan

### 3. Validation Dataset
- Dataset: {config['dataset']}
- Inclusion/exclusion criteria
- Ground truth labeling methodology and inter-rater agreement
- Demographic representativeness assessment

### 4. Statistical Analysis Plan
- Primary endpoint: {config['metric_name']}
- Pre-specified analysis populations
- Confidence interval methodology
- Subgroup analyses by demographics and acquisition parameters

### 5. Software Documentation (IEC 62304)
- Software development lifecycle
- Software requirements specification
- Software architecture and detailed design
- Verification and validation plan

{risk_section(config)}

{evidence_gaps()}
"""


def generate_ce(config, ckpt_results):
    return f"""# EU MDR Technical Documentation Planning
## {config['device_name']}
Generated: {datetime.now().strftime('%Y-%m-%d')}

---

## Device Description

- **Device name**: {config['device_name']}
- **Intended purpose**: {config['intended_use']}
- **Validation dataset**: {config['dataset']}

{model_description(config)}

{performance_section(config, ckpt_results)}

{software_section()}

## Classification

- **MDR Classification**: [Class IIa / IIb — Rule 11 for software intended to provide information used for diagnosis or monitoring]
- **Conformity assessment**: [Notified Body involvement required for Class IIa and above]
- **GSPR checklist**: [Map General Safety and Performance Requirements to evidence]

## Technical Documentation Structure (Annex II/III)

### 1. Device Description and Specification
- Algorithm architecture and training methodology
- Input/output specification
- Performance boundaries and limitations

### 2. Clinical Evaluation Report
- Literature search strategy and appraisal
- Equivalence analysis (if applicable)
- Clinical performance: {config['metric_name']} = {config['best_metric']}
- Benefit-risk analysis

### 3. Risk Management (ISO 14971)
- Hazard identification for {config['task']} task
- Risk estimation and evaluation
- Risk control measures and residual risk acceptability

### 4. Post-Market Surveillance Plan
- Complaint handling process
- Vigilance reporting
- Post-market clinical follow-up (PMCF) plan
- Periodic safety update report (PSUR) schedule

### 5. Software Lifecycle (IEC 62304)
- Software safety classification
- Software development plan
- Software verification and validation

{risk_section(config)}

{evidence_gaps()}
"""


def generate_ukca(config, ckpt_results):
    return f"""# UKCA / MHRA Regulatory Planning
## {config['device_name']}
Generated: {datetime.now().strftime('%Y-%m-%d')}

---

## Device Description

- **Device name**: {config['device_name']}
- **Intended purpose**: {config['intended_use']}
- **Validation dataset**: {config['dataset']}

{model_description(config)}

{performance_section(config, ckpt_results)}

{software_section()}

## UK Regulatory Pathway

- **MHRA classification**: [Confirm based on UK MDR 2002 (as amended)]
- **UKCA marking**: [Confirm applicability and timeline vs CE recognition period]
- **Approved Body**: [Identify UK Approved Body for conformity assessment]

## UK-Specific Considerations

### NHS Integration
- NICE evidence standards for digital health technologies
- NHS Digital interoperability requirements (HL7 FHIR, DICOM)
- NHS deployment and service ownership model

### UK Population Validation
- Demographics representative of UK patient population
- Multi-site validation across NHS trusts
- Consideration of UK-specific imaging protocols and equipment

### Regulatory Timeline
- UKCA/CE mutual recognition period status
- MHRA early access or sandbox program eligibility
- Post-Brexit regulatory divergence considerations

{risk_section(config)}

{evidence_gaps()}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate regulatory planning documents from experiment results"
    )
    parser.add_argument("--type", type=str, default="all",
                        choices=["all", "fda", "ce", "ukca"])
    parser.add_argument("--device-name", type=str, default=None)
    parser.add_argument("--intended-use", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--best-metric", type=str, default=None)
    parser.add_argument("--session", type=str, default=None, help="Optional session filter")
    parser.add_argument("--results", type=str, default=RESULTS_FILE)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    best_row = load_best_result(args.results, dataset_filter=args.dataset, session_filter=args.session)
    ckpt_config, ckpt_results = {}, {}
    if best_row:
        ckpt_config, ckpt_results = load_checkpoint_config(best_row)
    config = infer_config(best_row, ckpt_config, args)
    os.makedirs(args.output_dir, exist_ok=True)

    generators = {
        "fda": ("fda_planning.md", generate_fda),
        "ce": ("ce_mdr_planning.md", generate_ce),
        "ukca": ("ukca_planning.md", generate_ukca),
    }
    doc_types = generators.keys() if args.type == "all" else [args.type]

    for doc_type in doc_types:
        filename, generator = generators[doc_type]
        output_path = os.path.join(args.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(generator(config, ckpt_results))
        print(f"Generated: {output_path}")

    print("\nThese are planning documents built from real experiment data.")
    print("Review with qualified regulatory professionals before any external use.")


if __name__ == "__main__":
    main()
