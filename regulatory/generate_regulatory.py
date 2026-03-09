"""
Regulatory Documentation Generator — Scaffold FDA 510(k), CE/UKCA docs from model results.

Generates regulatory documentation templates pre-filled with model performance data,
intended use statements, and predicate device comparisons.

Usage:
    python regulatory/generate_regulatory.py                     # Generate all templates
    python regulatory/generate_regulatory.py --type fda          # FDA 510(k) only
    python regulatory/generate_regulatory.py --type ce           # CE/MDR only
    python regulatory/generate_regulatory.py --type ukca         # UKCA only
    python regulatory/generate_regulatory.py --results results.tsv  # Use experiment data
    python regulatory/generate_regulatory.py --checkpoint checkpoints/best.pth  # Use checkpoint
"""

import os
import argparse
import csv
import subprocess
from datetime import datetime
from pathlib import Path

REGULATORY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(REGULATORY_DIR)
TEMPLATES_DIR = os.path.join(REGULATORY_DIR, "templates")
OUTPUT_DIR = os.path.join(REGULATORY_DIR, "output")
RESULTS_FILE = os.environ.get("AUTORESEARCH_RESULTS_FILE", os.path.join(PROJECT_DIR, "results.tsv"))
CHECKPOINT_DIR = os.environ.get("AUTORESEARCH_CHECKPOINT_DIR", os.path.join(PROJECT_DIR, "checkpoints"))


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------

def auto_detect_from_results(results_path, dataset_filter=None):
    """Read best metric and config from results.tsv."""
    info = {"best_metric": None, "backbone": None, "loss": None, "extras": None}
    if not os.path.exists(results_path):
        return info

    try:
        best = None
        best_desc = ""
        with open(results_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                desc = row.get("description", "")
                if dataset_filter and f"dataset={dataset_filter};" not in desc:
                    continue
                try:
                    metric = float(row["metric"])
                except (KeyError, TypeError, ValueError):
                    continue
                if best is None or metric > best:
                    best = metric
                    best_desc = desc

        if best is not None:
            info["best_metric"] = f"{best:.2f}"
            # Parse description
            for part in best_desc.split(";"):
                if "=" in part:
                    key, val = part.split("=", 1)
                    key = key.strip()
                    if key == "backbone":
                        info["backbone"] = val.strip()
                    elif key == "loss":
                        info["loss"] = val.strip()
                    elif key == "extras":
                        info["extras"] = val.strip()
    except Exception:
        pass
    return info


def auto_detect_from_checkpoint(checkpoint_path):
    """Read config and per-class results from a checkpoint."""
    info = {"backbone": None, "per_class_auc": None, "config": {}}
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return info
    try:
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        results = ckpt.get("results", {})
        info["config"] = cfg
        info["backbone"] = cfg.get("backbone")
        if "per_class_auc" in results:
            info["per_class_auc"] = results["per_class_auc"]
        return info
    except Exception:
        return info


def generate_sbom():
    """Generate SBOM from pyproject.toml and installed packages."""
    lines = []

    # Try reading pyproject.toml for declared dependencies
    pyproject = os.path.join(PROJECT_DIR, "pyproject.toml")
    declared_deps = []
    if os.path.exists(pyproject):
        with open(pyproject, encoding="utf-8") as f:
            in_deps = False
            for line in f:
                if "dependencies" in line and "=" in line:
                    in_deps = True
                    continue
                if in_deps:
                    line = line.strip()
                    if line.startswith("]"):
                        break
                    if line.startswith('"') or line.startswith("'"):
                        dep = line.strip('",\' ')
                        if dep:
                            declared_deps.append(dep)

    # Try getting installed versions via uv
    installed = {}
    try:
        result = subprocess.run(
            ["uv", "pip", "list", "--format=columns"],
            capture_output=True, text=True, timeout=10,
            cwd=PROJECT_DIR,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n")[2:]:  # skip header
                parts = line.split()
                if len(parts) >= 2:
                    installed[parts[0].lower()] = parts[1]
    except Exception:
        pass

    # Build SBOM table
    lines.append("| Package | Declared | Installed | Risk | Mitigation |")
    lines.append("|---------|----------|-----------|------|------------|")

    risk_map = {
        "torch": ("High", "Core ML framework — pinned version, widely validated"),
        "torchvision": ("Medium", "Image transforms — pinned to torch version"),
        "timm": ("Medium", "Pretrained model weights — checksums verified"),
        "transformers": ("Medium", "Foundation models — pinned version"),
        "numpy": ("Low", "Numerical computation — widely validated"),
        "pillow": ("Low", "Image I/O only"),
        "pandas": ("Low", "Data loading — no inference path"),
        "scikit-learn": ("Low", "Metrics computation only"),
        "pyyaml": ("Low", "Config parsing only"),
        "albumentations": ("Low", "Image augmentation only"),
    }

    for dep in declared_deps:
        # Extract package name from version spec
        pkg_name = dep.split(">=")[0].split("<=")[0].split("==")[0].split(">")[0].split("<")[0].strip()
        pkg_lower = pkg_name.lower()
        version_spec = dep[len(pkg_name):].strip() or "any"
        installed_ver = installed.get(pkg_lower, "[not installed]")
        risk, mitigation = risk_map.get(pkg_lower, ("Low", "Standard dependency"))
        lines.append(f"| {pkg_name} | {version_spec} | {installed_ver} | {risk} | {mitigation} |")

    return "\n".join(lines) if lines else "| [Run `uv sync` to populate] | | | | |"


def format_per_class_for_regulatory(per_class_auc):
    """Format per-class AUC as a markdown table for regulatory docs."""
    if not per_class_auc:
        return "[INSERT per_class_auc table from model evaluation]"

    lines = ["| Pathology | AUC (%) | 95% CI |"]
    lines.append("|-----------|--------|--------|")
    for pathology, auc in per_class_auc.items():
        pathology_clean = pathology.replace("_", " ")
        lines.append(f"| {pathology_clean} | {auc:.2f} | [CI] |")

    mean_auc = sum(per_class_auc.values()) / len(per_class_auc)
    lines.append(f"| **Mean** | **{mean_auc:.2f}** | [CI] |")
    return "\n".join(lines)


BACKBONE_PRETTY = {
    "densenet121": "DenseNet-121", "densenet169": "DenseNet-169",
    "densenet201": "DenseNet-201", "resnet50": "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0", "efficientnet_b3": "EfficientNet-B3",
    "efficientnet_b4": "EfficientNet-B4", "convnext_small": "ConvNeXt-Small",
    "swin_tiny_patch4_window7_224": "Swin-Tiny",
    "vit_small_patch16_224": "ViT-Small", "unet": "U-Net",
}


# ---------------------------------------------------------------------------
# FDA 510(k) Templates
# ---------------------------------------------------------------------------

def generate_fda_510k(config):
    """Generate FDA 510(k) predicate comparison and substantial equivalence argument."""
    device_name = config.get("device_name", "[YOUR DEVICE NAME]")
    intended_use = config.get("intended_use", "[INTENDED USE STATEMENT]")
    indication = config.get("indication", "[INDICATION FOR USE]")
    dataset = config.get("dataset", "[DATASET NAME]")
    metric_name = config.get("metric_name", "AUC")
    best_metric = config.get("best_metric", "[XX.XX]")
    backbone = config.get("backbone", "[MODEL ARCHITECTURE]")
    sbom = config.get("sbom", "")
    per_class_table = config.get("per_class_table", "[INSERT per_class_auc table from model evaluation]")
    torch_version = config.get("torch_version", "[VERSION]")

    return f"""# FDA 510(k) Presubmission — {device_name}
# Generated: {datetime.now().strftime('%Y-%m-%d')}
# STATUS: TEMPLATE — Requires regulatory counsel review

## 1. Device Description

**Device Name:** {device_name}
**Classification:** Class II Medical Device (Computer-Aided Detection/Diagnosis)
**Product Code:** QIH (AI-based Medical Image Analyzer, non-cancer)
**Regulation Number:** 21 CFR 892.2050

### 1.1 Intended Use
{intended_use}

### 1.2 Indications for Use
{indication}

### 1.3 Technology Description
- **Architecture:** {backbone}
- **Input:** DICOM / PNG / JPEG medical images
- **Output:** Probability scores per pathology class
- **Processing:** On-device inference (no cloud requirement)

---

## 2. Predicate Device Comparison

| Feature | This Device | Predicate 1 | Predicate 2 |
|---------|-------------|-------------|-------------|
| Intended Use | {intended_use[:50]}... | [PREDICATE USE] | [PREDICATE USE] |
| Technology | Deep learning ({backbone}) | [PREDICATE TECH] | [PREDICATE TECH] |
| Input | Digital radiograph | [PREDICATE INPUT] | [PREDICATE INPUT] |
| Output | Pathology probabilities | [PREDICATE OUTPUT] | [PREDICATE OUTPUT] |
| {metric_name} | {best_metric}% | [PREDICATE METRIC] | [PREDICATE METRIC] |

### 2.1 Known Predicate Devices (Computer-Aided Detection — Chest X-ray)

**IMPORTANT:** Verify all 510(k) numbers and clearance dates directly on the
[FDA 510(k) database](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm)
before any submission. Numbers below are provided as starting points only.

| Device | Company | 510(k) Number | Clearance Date | Notes |
|--------|---------|---------------|----------------|-------|
| qXR | Qure.ai | [VERIFY on FDA DB] | 2020+ | Multiple clearances; 26 indications across 9 products |
| Lunit INSIGHT CXR | Lunit | K211733 | 2021-11 | Triage for pleural effusion & pneumothorax |
| Annalise Enterprise CXR | Annalise.ai | K213941 | 2022-02 | Pneumothorax triage & notification |
| HealthPNX | Zebra Medical | K190362 | 2019-05 | Pneumothorax detection/triage |
| BriefCase | Aidoc Medical | [VERIFY on FDA DB] | 2022+ | AI triage, multiple indications |

### 2.2 Substantial Equivalence Argument
[FILL IN: Explain how your device is substantially equivalent to the predicate
in terms of intended use, technology, and performance. Reference specific
predicate devices and their 510(k) numbers.]

---

## 3. Performance Testing — Summary

### 3.1 Standalone Performance
| Metric | Value | 95% CI | Predicate Benchmark |
|--------|-------|--------|---------------------|
| Mean {metric_name} | {best_metric}% | [CI] | [BENCHMARK] |
| Sensitivity | [XX.X]% | [CI] | [BENCHMARK] |
| Specificity | [XX.X]% | [CI] | [BENCHMARK] |
| PPV | [XX.X]% | [CI] | [BENCHMARK] |
| NPV | [XX.X]% | [CI] | [BENCHMARK] |

### 3.2 Per-Pathology Performance
{per_class_table}

### 3.3 Reader Study Design
[REQUIRED: Design a multi-reader multi-case (MRMC) study]
- **Readers:** [N] board-certified radiologists
- **Cases:** [N] retrospective cases from [SITE]
- **Design:** Crossover — each reader reads with and without AI assist
- **Primary endpoint:** AUC improvement with AI assist
- **Sample size justification:** [FILL IN — use FDA guidance on MRMC studies]

---

## 4. Software Documentation (IEC 62304)

### 4.1 Software Level of Concern
**Level:** [A / B / C — likely B or C for CADe/CADx]

### 4.2 Software Description
| Component | Description |
|-----------|-------------|
| Framework | PyTorch {torch_version} |
| Model | {backbone} (pretrained ImageNet + fine-tuned) |
| Libraries | timm, torchvision |
| Input format | DICOM, PNG, JPEG |
| Output format | JSON (pathology probabilities) |
| OS | Linux (inference server) |

### 4.3 SOUP (Software of Unknown Provenance)
{sbom}

---

## 5. Cybersecurity (FDA Guidance 2023)

### 5.1 Threat Model
[FILL IN: STRIDE analysis for your deployment scenario]

### 5.2 SBOM (Software Bill of Materials)
See Section 4.3 above for declared dependencies with versions and risk assessment.

---

## 6. Labeling

### 6.1 Warnings and Precautions
- This device is intended to assist, not replace, clinical judgment
- Performance validated on {dataset} — may not generalize to all populations
- Not validated for pediatric patients (< 18 years)
- Not validated for images with implanted devices
- [ADD population-specific limitations]

### 6.2 Contraindications
- [FILL IN based on your validation data gaps]

---

## 7. Checklist — Pre-Submission

- [ ] Predicate device(s) identified with 510(k) numbers
- [ ] Substantial equivalence argument drafted
- [ ] Standalone performance on representative dataset
- [ ] Multi-reader multi-case (MRMC) study designed
- [ ] IEC 62304 software documentation
- [ ] Risk analysis (ISO 14971) completed
- [ ] Cybersecurity documentation (SBOM, threat model)
- [ ] Labeling reviewed by regulatory counsel
- [ ] Pre-submission meeting requested with FDA (recommended)

---

*This template is for guidance only. Consult with a regulatory affairs professional
and legal counsel before any FDA submission. FDA requirements change — check the
latest guidance documents at https://www.fda.gov/medical-devices.*
"""


def generate_ce_mdr(config):
    """Generate EU MDR / CE marking documentation scaffold."""
    device_name = config.get("device_name", "[YOUR DEVICE NAME]")
    intended_use = config.get("intended_use", "[INTENDED USE STATEMENT]")
    best_metric = config.get("best_metric", "[XX.XX]")
    metric_name = config.get("metric_name", "AUC")
    backbone = config.get("backbone", "[MODEL ARCHITECTURE]")

    return f"""# CE Marking — EU MDR 2017/745 Documentation
# Device: {device_name}
# Generated: {datetime.now().strftime('%Y-%m-%d')}
# STATUS: TEMPLATE — Requires Notified Body and regulatory counsel review

## 1. Device Classification (EU MDR)

**Classification:** Class IIa (Rule 11 — software intended to provide information
used to make decisions with diagnosis or therapeutic purposes)

**Note:** AI/ML-based diagnostic software is typically Class IIa under MDR Rule 11.
If the software is intended for screening (life-threatening conditions), it may be
Class IIb or III. Consult with your Notified Body.

**UDI-DI:** [To be assigned]
**EUDAMED registration:** [Required before market placement]

---

## 2. General Safety and Performance Requirements (GSPR) Checklist

### Annex I — Essential Requirements

| GSPR # | Requirement | Applicable | Evidence |
|--------|-------------|------------|----------|
| 1 | Device achieves intended performance | Yes | Clinical evaluation, validation study |
| 2 | Design minimizes risk | Yes | Risk management file (ISO 14971) |
| 3 | Device suitable for intended use | Yes | Usability evaluation (IEC 62366) |
| 4 | Transport/storage conditions | N/A | Software only |
| 5 | Acceptable benefit-risk ratio | Yes | Clinical evaluation report |
| 6 | Performance maintained during lifetime | Yes | Post-market surveillance plan |
| 8 | Device and manufacturing consistent | Yes | QMS (ISO 13485) |
| 9.1 | Design ensures safety | Yes | Risk management |
| 10 | Chemical/physical/biological properties | N/A | Software only |
| 11 | Infection and microbial contamination | N/A | Software only |
| 14 | Devices with measuring function | N/A | No measurement |
| 15 | Protection against radiation | N/A | No radiation emitted |
| 16 | Electronic programmable systems | Yes | IEC 62304, software validation |
| 17 | Software (AI/ML specific) | Yes | IEC 62304, AI/ML lifecycle |
| 23 | Labeling requirements | Yes | IFU documentation |

---

## 3. Clinical Evaluation Report (CER) — Outline

### 3.1 Scope
Clinical evaluation of {device_name} for {intended_use}, per MDR Article 61
and MEDDEV 2.7/1 Rev 4.

### 3.2 Device Technology
- **Architecture:** {backbone}
- **Training data:** ImageNet pretraining + medical dataset fine-tuning
- **Performance:** {best_metric}% {metric_name}

### 3.3 Clinical Data Sources
1. **Literature review:** Systematic search of PubMed, Embase for comparable AI devices
2. **Clinical investigation data:** [Validation study results]
3. **Post-market data:** [To be collected]
4. **Equivalent device data:** [If applicable]

### 3.4 State of the Art
[FILL IN: Current standard of care, existing AI devices, clinical need]

### 3.5 Performance Summary
| Metric | Our Device | Literature Benchmark | Clinical Significance |
|--------|-----------|---------------------|----------------------|
| {metric_name} | {best_metric}% | [BENCHMARK] | [SIGNIFICANCE] |
| Sensitivity | [XX.X]% | [BENCHMARK] | [SIGNIFICANCE] |
| Specificity | [XX.X]% | [BENCHMARK] | [SIGNIFICANCE] |

### 3.6 Benefit-Risk Analysis
[FILL IN: Benefits of AI-assisted diagnosis vs. risks of false positives/negatives]

### 3.7 Conclusions
[FILL IN: Device meets GSPR, acceptable benefit-risk ratio, suitable for intended use]

---

## 4. Post-Market Surveillance (PMS) Plan

### 4.1 Monitoring Activities
- Model performance drift monitoring (monthly {metric_name} recalculation)
- Vigilance reporting (serious incidents within 15 days)
- Periodic Safety Update Report (PSUR) — every [1-2] years for Class IIa
- Customer complaint handling

### 4.2 Post-Market Clinical Follow-up (PMCF)
- Prospective data collection from [N] clinical sites
- Annual {metric_name} benchmarking against updated state of the art
- Subgroup analysis (age, sex, ethnicity, device manufacturer)

---

## 5. Quality Management System

**Standard:** ISO 13485:2016
**Notified Body:** [SELECT — e.g., BSI, TUV SUD, SGS, DEKRA]

### 5.1 AI/ML Specific Processes
- Dataset management and versioning
- Model training, validation, and testing procedures
- Algorithm change management protocol
- Bias and fairness assessment

---

## 6. Checklist — CE Marking

- [ ] Device classification confirmed (Rule 11)
- [ ] QMS (ISO 13485) certified
- [ ] Technical documentation (Annex II/III) complete
- [ ] Clinical Evaluation Report written
- [ ] Risk management file (ISO 14971) complete
- [ ] Software lifecycle (IEC 62304) documented
- [ ] Usability engineering (IEC 62366) completed
- [ ] Labeling and IFU prepared
- [ ] Post-market surveillance plan
- [ ] PMCF plan
- [ ] EU Authorized Representative designated (if non-EU manufacturer)
- [ ] Notified Body audit scheduled
- [ ] EUDAMED registration prepared
- [ ] Declaration of Conformity drafted

---

*This template is for guidance only. EU MDR requirements are complex and
evolving. Engage a qualified regulatory consultant and Notified Body early
in the process.*
"""


def generate_ukca(config):
    """Generate UK MDR 2002 / UKCA documentation scaffold."""
    device_name = config.get("device_name", "[YOUR DEVICE NAME]")
    intended_use = config.get("intended_use", "[INTENDED USE STATEMENT]")
    best_metric = config.get("best_metric", "[XX.XX]")
    metric_name = config.get("metric_name", "AUC")
    backbone = config.get("backbone", "[MODEL ARCHITECTURE]")

    return f"""# UKCA Marking — UK MDR 2002 (as amended) Documentation
# Device: {device_name}
# Generated: {datetime.now().strftime('%Y-%m-%d')}
# STATUS: TEMPLATE — Requires UK Approved Body and regulatory counsel review

## 1. Regulatory Framework

Post-Brexit, the UK has its own medical device regulations:
- **UK MDR 2002** (SI 2002 No. 618, as amended by SI 2023 No. 627)
- **UKCA marking** required for devices placed on the GB market
- **CE marking** accepted until June 2028 (extended deadline)
- **UK Approved Bodies** replace EU Notified Bodies

### 1.1 Classification
**Classification:** Class IIa (analogous to EU MDR Rule 11)
**UK Approved Body:** [SELECT — e.g., BSI UK, SGS UK]

### 1.2 Device Technology
- **Architecture:** {backbone}
- **Performance:** {best_metric}% {metric_name}

---

## 2. Essential Requirements (UK MDR 2002, Schedule 1)

| Req # | Requirement | Evidence |
|-------|-------------|----------|
| 1 | Device does not compromise safety | Risk management (ISO 14971) |
| 2 | Design and construction principles | Technical documentation |
| 3 | Acceptable risk with benefit | Clinical evaluation |
| 6 | Performance as intended | Validation study |
| 12 | Software verification and validation | IEC 62304 |
| 13 | Labeling | IFU documentation |

---

## 3. Performance Summary

| Metric | Value | UK Population Validation |
|--------|-------|--------------------------|
| {metric_name} | {best_metric}% | [UK COHORT RESULTS] |
| Sensitivity | [XX.X]% | [UK COHORT] |
| Specificity | [XX.X]% | [UK COHORT] |

### 3.1 UK-Specific Considerations
- Validation on UK patient demographics (NHS population)
- Compatibility with NHS PACS systems
- NICE evidence standards framework for digital health technologies
- NHS Digital Technology Assessment Criteria (DTAC)

---

## 4. NICE Evidence Standards

For NHS adoption, the device should meet NICE Evidence Standards Framework:
- **Tier 1** (Inform): Relevant health condition, current pathway
- **Tier 2** (Communicate): Credible clinical evidence of effectiveness
- **Tier 3a** (Diagnose — low risk): High-quality observational/quasi-experimental
- **Tier 3b** (Diagnose — high risk): Randomized controlled evidence

### 4.1 NICE Submission Checklist
- [ ] Evidence standards tier identified
- [ ] Health economic analysis prepared
- [ ] NHS pathway integration documented
- [ ] Comparator analysis (current standard of care)
- [ ] Real-world evidence from UK clinical sites

---

## 5. NHS Digital — DTAC Compliance

| Domain | Requirement | Status |
|--------|-------------|--------|
| Clinical safety | DCB 0129 (manufacturer) / DCB 0160 (deployer) | [ ] |
| Data protection | UK GDPR / Data Protection Act 2018 | [ ] |
| Technical security | Cyber Essentials Plus / DSPT | [ ] |
| Interoperability | FHIR / HL7 / DICOM compliance | [ ] |
| Usability | Accessibility (WCAG 2.1 AA) | [ ] |

---

## 6. Post-Market Surveillance (UK)

- **MHRA reporting**: Serious incidents within 15 days
- **Field Safety Corrective Actions** (FSCA) as needed
- **Periodic Summary Reports** to MHRA
- **UK Responsible Person** designated (if non-UK manufacturer)

---

## 7. Checklist — UKCA Marking

- [ ] UK MDR 2002 classification confirmed
- [ ] UK Approved Body selected and engaged
- [ ] Technical documentation complete
- [ ] Clinical evaluation (UK population data)
- [ ] ISO 13485 QMS certification
- [ ] Risk management (ISO 14971)
- [ ] Software lifecycle (IEC 62304)
- [ ] UK Responsible Person designated (if needed)
- [ ] UKCA Declaration of Conformity
- [ ] Device registered with MHRA
- [ ] NICE evidence standards assessment
- [ ] DTAC compliance (for NHS deployment)

---

*This template is for guidance only. UK regulatory requirements are evolving
post-Brexit. Consult with MHRA guidance and qualified regulatory professionals.*
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate regulatory documentation from model results")
    parser.add_argument("--type", type=str, default="all",
                        choices=["all", "fda", "ce", "ukca"],
                        help="Documentation type to generate")
    parser.add_argument("--device-name", type=str, default="AI-Assisted Chest X-ray Analysis",
                        help="Device name")
    parser.add_argument("--intended-use", type=str,
                        default="Computer-aided detection of thoracic pathologies in chest radiographs",
                        help="Intended use statement")
    parser.add_argument("--dataset", type=str, default="ChestX-ray14",
                        help="Validation dataset name")
    parser.add_argument("--metric", type=str, default="Mean AUC",
                        help="Primary performance metric name")
    parser.add_argument("--best-metric", type=str, default=None,
                        help="Best metric value (auto-read from results.tsv if not provided)")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to results.tsv to auto-fill performance data")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint for per-class results and config")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Output directory for generated docs")
    args = parser.parse_args()

    # Auto-detect from results.tsv
    results_path = args.results or RESULTS_FILE
    # Use dataset arg for filtering (default "ChestX-ray14" display name, filter key is lowercase)
    dataset_filter_key = args.dataset.lower().replace("-", "").replace(" ", "")
    results_info = auto_detect_from_results(results_path, dataset_filter=dataset_filter_key)

    # Auto-detect from checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path and os.path.isdir(CHECKPOINT_DIR):
        ckpts = sorted(Path(CHECKPOINT_DIR).glob("*.pth"), key=os.path.getmtime, reverse=True)
        if ckpts:
            ckpt_path = str(ckpts[0])
    ckpt_info = auto_detect_from_checkpoint(ckpt_path)

    # Determine backbone (checkpoint > results > default)
    backbone_raw = ckpt_info.get("backbone") or results_info.get("backbone") or "densenet121"
    backbone = BACKBONE_PRETTY.get(backbone_raw, backbone_raw)

    # Determine best metric
    best_metric = args.best_metric or results_info.get("best_metric") or "[XX.XX]"
    if best_metric != "[XX.XX]":
        print(f"Best metric from results: {best_metric}%")

    # Generate SBOM
    sbom = generate_sbom()

    # Format per-class AUC
    per_class_table = format_per_class_for_regulatory(ckpt_info.get("per_class_auc"))

    # Get torch version
    torch_version = "[VERSION]"
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        pass

    config = {
        "device_name": args.device_name,
        "intended_use": args.intended_use,
        "indication": args.intended_use,
        "dataset": args.dataset,
        "metric_name": args.metric,
        "best_metric": best_metric,
        "backbone": backbone,
        "sbom": sbom,
        "per_class_table": per_class_table,
        "torch_version": torch_version,
    }

    print(f"Architecture: {backbone}")

    os.makedirs(args.output_dir, exist_ok=True)

    generators = {
        "fda": ("fda_510k_scaffold.md", generate_fda_510k),
        "ce": ("ce_mdr_scaffold.md", generate_ce_mdr),
        "ukca": ("ukca_scaffold.md", generate_ukca),
    }

    types_to_generate = generators.keys() if args.type == "all" else [args.type]

    for doc_type in types_to_generate:
        filename, generator = generators[doc_type]
        content = generator(config)
        output_path = os.path.join(args.output_dir, filename)
        with open(output_path, "w") as f:
            f.write(content)
        print(f"Generated: {output_path}")

    print(f"\nRegulatory documents generated in {args.output_dir}/")
    print(f"\nAuto-populated:")
    print(f"  - Architecture: {backbone}")
    print(f"  - Best metric: {best_metric}%")
    print(f"  - SBOM: {'yes' if 'Package' in sbom else 'run uv sync first'}")
    print(f"  - Per-class AUC: {'yes' if ckpt_info.get('per_class_auc') else 'use --checkpoint flag'}")
    print("\nIMPORTANT: These are templates only. Before any regulatory submission:")
    print("  1. Engage qualified regulatory affairs professionals")
    print("  2. Consult legal counsel experienced in SaMD regulation")
    print("  3. Contact the relevant authority (FDA/Notified Body/MHRA) for pre-submission")
    print("  4. Fill in all [BRACKETED] sections with your specific data")


if __name__ == "__main__":
    main()
