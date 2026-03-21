"""
Paper Tracker — Automatically discover new medical imaging papers and datasets.

Pulls from arXiv, tracks papers with code/data, and auto-generates recipe stubs
for reproducible experiments.

Usage:
    python tools/paper_tracker.py                     # Latest medical imaging papers
    python tools/paper_tracker.py --query "retinal"   # Search specific topic
    python tools/paper_tracker.py --with-data          # Only papers with datasets
    python tools/paper_tracker.py --generate-recipe    # Auto-create recipe YAML
    python tools/paper_tracker.py --days 7             # Papers from last 7 days
"""

import os
import re
import json
import time
import argparse
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# arXiv API
# ---------------------------------------------------------------------------

ARXIV_API = "http://export.arxiv.org/api/query"

# Medical imaging search categories
MEDICAL_IMAGING_QUERIES = [
    "medical image segmentation",
    "medical image classification",
    "medical image foundation model",
    "radiology artificial intelligence",
    "x-ray image classification",
    "retinal image analysis",
    "fundus image analysis",
    "ophthalmology deep learning",
    "pathology image classification",
    "ultrasound image segmentation",
    "CT scan deep learning",
    "MRI segmentation neural network",
    "dermatology image classification",
]

# Indicators that a paper has associated data/code
DATA_KEYWORDS = [
    "dataset", "kaggle", "github.com", "publicly available",
    "open source", "open-source", "code available", "data available",
    "huggingface", "zenodo", "grand-challenge", "physionet",
]

CODE_KEYWORDS = [
    "github.com", "code is available", "code available at",
    "implementation is available", "source code",
    "huggingface.co", "gitlab.com",
]


def search_arxiv(query, max_results=50, days_back=30):
    """Search arXiv for recent papers matching query."""
    # Build date filter
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
    except Exception as e:
        print(f"  Error fetching arXiv: {e}")
        return []

    # Parse XML
    root = ET.fromstring(data)
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    papers = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        arxiv_id = entry.find("atom:id", ns).text.strip().split("/abs/")[-1]
        published = entry.find("atom:published", ns).text[:10]

        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.find("atom:name", ns).text
            authors.append(name)

        # Extract links
        links = {}
        for link in entry.findall("atom:link", ns):
            rel = link.get("rel", "")
            href = link.get("href", "")
            if rel == "alternate":
                links["abstract"] = href
            elif link.get("title") == "pdf":
                links["pdf"] = href

        # Check for data/code indicators
        text = (title + " " + summary).lower()
        has_data = any(kw in text for kw in DATA_KEYWORDS)
        has_code = any(kw in text for kw in CODE_KEYWORDS)

        # Extract GitHub URLs from summary
        github_urls = re.findall(r"https?://github\.com/[\w\-]+/[\w\-]+", summary)

        # Extract dataset names
        datasets_mentioned = []
        for ds in ["ChestX-ray14", "APTOS", "DRIVE", "CHASE_DB1", "BraTS",
                    "ISIC", "CheXpert", "MIMIC-CXR", "NIH", "RSNA",
                    "MedMNIST", "Retinal", "OCT", "PathMNIST", "DermaMNIST",
                    "PneumoniaMNIST", "CIFAR", "ImageNet"]:
            if ds.lower() in text:
                datasets_mentioned.append(ds)

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors[:5],  # first 5 authors
            "summary": summary[:500],
            "published": published,
            "links": links,
            "has_data": has_data,
            "has_code": has_code,
            "github_urls": github_urls,
            "datasets": datasets_mentioned,
        })

    return papers


def search_medical_imaging_papers(days_back=30, max_per_query=20):
    """Search across all medical imaging topics."""
    all_papers = {}

    for query in MEDICAL_IMAGING_QUERIES:
        print(f"  Searching: {query}...")
        papers = search_arxiv(query, max_results=max_per_query, days_back=days_back)
        for paper in papers:
            # Deduplicate by arXiv ID
            all_papers[paper["arxiv_id"]] = paper
        time.sleep(3)  # Be nice to arXiv API

    return list(all_papers.values())


# ---------------------------------------------------------------------------
# Recipe generation
# ---------------------------------------------------------------------------

def detect_task_from_paper(paper):
    """Guess the task type from paper title/summary."""
    text = (paper["title"] + " " + paper["summary"]).lower()

    if any(w in text for w in ["segmentation", "segment", "u-net", "unet"]):
        return "segmentation"
    elif any(w in text for w in ["detection", "detect", "localization"]):
        return "classification"  # detection often uses classification backbone
    elif any(w in text for w in ["classification", "classify", "diagnosis", "grading"]):
        return "classification"
    elif any(w in text for w in ["multi-label", "multilabel"]):
        return "multilabel"
    return "classification"


def detect_backbone_from_paper(paper):
    """Guess the backbone from paper content."""
    text = (paper["title"] + " " + paper["summary"]).lower()

    if "densenet" in text:
        return "densenet121"
    elif "efficientnet" in text:
        if "b4" in text:
            return "efficientnet_b4"
        elif "b3" in text:
            return "efficientnet_b3"
        return "efficientnet_b0"
    elif "resnet" in text:
        if "101" in text:
            return "resnet101"
        return "resnet50"
    elif "vit" in text or "vision transformer" in text:
        return "vit_small_patch16_224"
    elif "swin" in text:
        return "swin_tiny_patch4_window7_224"
    elif "convnext" in text:
        return "convnext_small"
    return "efficientnet_b3"  # safe default for medical imaging


def generate_recipe_stub(paper):
    """Generate a recipe YAML stub from paper metadata."""
    task = detect_task_from_paper(paper)
    backbone = detect_backbone_from_paper(paper)

    authors_str = ", ".join(paper["authors"][:3])
    if len(paper["authors"]) > 3:
        authors_str += ", et al."

    recipe = f"""# {paper['title']}
# {authors_str} ({paper['published'][:4]})
# https://arxiv.org/abs/{paper['arxiv_id']}
#
# AUTO-GENERATED STUB — review and fill in details

paper:
  title: "{paper['title']}"
  authors: "{authors_str}"
  year: {paper['published'][:4]}
  arxiv: "{paper['arxiv_id']}"
  dataset: "{', '.join(paper['datasets']) if paper['datasets'] else 'TODO'}"
  reported_metric: "TODO — check paper"

dataset: TODO  # Set to a built-in dataset name or create a custom manifest under data/<dataset>/dataset.yaml
task: {task}

model:
  backbone: {backbone}
  pretrained: true
  dropout: 0.2

training:
  batch_size: {'8' if task == 'segmentation' else '16'}
  optimizer: adamw
  learning_rate: 0.0003
  weight_decay: 0.01
  warmup_ratio: 0.05
  warmdown_ratio: 0.3

loss:
  type: {'dice_bce' if task == 'segmentation' else 'focal' if task == 'multilabel' else 'cross_entropy'}
  class_weights: true

augmentation:
  horizontal_flip: true
  vertical_flip: {'true' if task == 'segmentation' else 'false'}
  rotation: 15
  color_jitter: false

notes: |
  Auto-generated from arXiv paper. Review the following:
  1. Verify the dataset name and create a custom dataset manifest if needed
  2. Check reported metrics in the paper
  3. Verify backbone architecture matches paper
  4. Adjust hyperparameters based on paper's methodology section
"""
    if paper["github_urls"]:
        recipe += f"  Original code: {paper['github_urls'][0]}\n"

    return recipe


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_papers(papers, show_summary=False):
    """Pretty-print paper list."""
    for i, paper in enumerate(papers, 1):
        flags = []
        if paper["has_data"]:
            flags.append("DATA")
        if paper["has_code"]:
            flags.append("CODE")
        if paper["datasets"]:
            flags.append(f"datasets: {', '.join(paper['datasets'])}")

        flag_str = f" [{', '.join(flags)}]" if flags else ""

        print(f"\n{i}. {paper['title']}")
        print(f"   {', '.join(paper['authors'][:3])} ({paper['published']})")
        print(f"   https://arxiv.org/abs/{paper['arxiv_id']}{flag_str}")

        if paper["github_urls"]:
            print(f"   Code: {paper['github_urls'][0]}")

        if show_summary:
            print(f"   {paper['summary'][:200]}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Track medical imaging papers from arXiv")
    parser.add_argument("--query", type=str, default=None,
                        help="Custom search query")
    parser.add_argument("--days", type=int, default=30,
                        help="Look back N days (default: 30)")
    parser.add_argument("--with-data", action="store_true",
                        help="Only show papers with datasets/code")
    parser.add_argument("--with-code", action="store_true",
                        help="Only show papers with code links")
    parser.add_argument("--generate-recipe", action="store_true",
                        help="Auto-generate recipe YAML stubs for papers with data")
    parser.add_argument("--max-results", type=int, default=20,
                        help="Max results per query")
    parser.add_argument("--summary", action="store_true",
                        help="Show paper summaries")
    parser.add_argument("--save", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    print("=== Medical Imaging Paper Tracker ===\n")

    if args.query:
        print(f"Searching arXiv for: {args.query}")
        papers = search_arxiv(args.query, max_results=args.max_results, days_back=args.days)
    else:
        print(f"Searching across {len(MEDICAL_IMAGING_QUERIES)} medical imaging topics...")
        papers = search_medical_imaging_papers(days_back=args.days, max_per_query=args.max_results)

    # Sort by date
    papers.sort(key=lambda p: p["published"], reverse=True)

    # Filter
    if args.with_data:
        papers = [p for p in papers if p["has_data"] or p["datasets"]]
    if args.with_code:
        papers = [p for p in papers if p["has_code"] or p["github_urls"]]

    print(f"\nFound {len(papers)} papers")

    if not papers:
        print("No papers found. Try broader search or longer date range.")
        return

    print_papers(papers, show_summary=args.summary)

    # Generate recipe stubs
    if args.generate_recipe:
        recipe_dir = Path(__file__).parent.parent / "recipes"
        recipe_dir.mkdir(exist_ok=True)

        generated = 0
        for paper in papers:
            if paper["has_data"] or paper["datasets"]:
                # Create safe filename
                safe_name = re.sub(r"[^a-z0-9]+", "_", paper["title"].lower())[:60]
                recipe_path = recipe_dir / f"auto_{safe_name}.yaml"

                if not recipe_path.exists():
                    recipe_content = generate_recipe_stub(paper)
                    recipe_path.write_text(recipe_content)
                    generated += 1
                    print(f"\n  Generated recipe: {recipe_path.name}")

        print(f"\n  Total recipes generated: {generated}")

    # Save to JSON
    if args.save:
        with open(args.save, "w") as f:
            json.dump(papers, f, indent=2)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
