#!/usr/bin/env python3
"""Results dashboard — generate HTML visualization from results.tsv.

Creates charts showing:
- Metric progression over experiments
- Backbone comparison (bar chart)
- Pareto frontier (metric vs VRAM)

Usage:
    python tools/results_dashboard.py
    python tools/results_dashboard.py --results-file results.tsv --output dashboard.html
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


def parse_results(results_file):
    """Parse results.tsv into structured data."""
    rows = []
    with open(results_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            desc = row.get("description", "")
            parts = {}
            for part in desc.split(";"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    parts[k] = v

            try:
                rows.append({
                    "commit": row.get("commit", ""),
                    "metric": float(row.get("metric", 0)),
                    "memory_gb": float(row.get("memory_gb", 0)),
                    "status": row.get("status", ""),
                    "dataset": parts.get("dataset", ""),
                    "task": parts.get("task", ""),
                    "backbone": parts.get("backbone", ""),
                    "loss": parts.get("loss", ""),
                    "extras": parts.get("extras", "-"),
                    "recipe": parts.get("recipe", "-"),
                })
            except (ValueError, KeyError):
                continue
    return rows


def generate_html(rows, output_path):
    """Generate HTML dashboard with inline charts (no external dependencies)."""
    if not rows:
        print("No results to visualize.")
        return

    # Group by dataset
    datasets = {}
    for r in rows:
        ds = r["dataset"]
        if ds not in datasets:
            datasets[ds] = []
        datasets[ds].append(r)

    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>autoresearch-medvision Results Dashboard</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
h1 { color: #2c3e50; }
h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 8px; }
.card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
table { border-collapse: collapse; width: 100%; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #ecf0f1; }
th { background: #3498db; color: white; }
tr:hover { background: #ebf5fb; }
.keep { color: #27ae60; font-weight: bold; }
.discard { color: #e74c3c; }
.metric { font-size: 2em; font-weight: bold; color: #2c3e50; }
.bar-container { display: flex; align-items: center; gap: 8px; }
.bar { height: 20px; background: #3498db; border-radius: 4px; }
.best { background: #27ae60; }
</style>
</head>
<body>
<h1>autoresearch-medvision Results Dashboard</h1>
"""

    for ds, ds_rows in datasets.items():
        best_row = max(ds_rows, key=lambda r: r["metric"])
        html += f'<h2>{ds} ({ds_rows[0]["task"]})</h2>\n'
        html += f'<div class="card"><span class="metric">{best_row["metric"]:.2f}</span>'
        html += f' best ({best_row["backbone"]}, {best_row["extras"]})</div>\n'

        # Backbone comparison
        backbone_best = {}
        for r in ds_rows:
            bb = r["backbone"]
            if bb not in backbone_best or r["metric"] > backbone_best[bb]["metric"]:
                backbone_best[bb] = r

        max_metric = max(r["metric"] for r in backbone_best.values()) if backbone_best else 1
        html += '<div class="card"><h3>Backbone Comparison</h3>\n'
        for bb, r in sorted(backbone_best.items(), key=lambda x: -x[1]["metric"]):
            pct = r["metric"] / max_metric * 100 if max_metric > 0 else 0
            cls = "best" if r["metric"] == best_row["metric"] else ""
            html += f'<div class="bar-container"><span style="width:120px">{bb}</span>'
            html += f'<div class="bar {cls}" style="width:{pct}%"></div>'
            html += f'<span>{r["metric"]:.2f} ({r["memory_gb"]:.1f}GB)</span></div>\n'
        html += '</div>\n'

        # Results table
        html += '<div class="card"><h3>All Experiments</h3>\n'
        html += '<table><tr><th>#</th><th>Backbone</th><th>Loss</th>'
        html += '<th>Extras</th><th>Metric</th><th>VRAM (GB)</th><th>Status</th></tr>\n'
        for i, r in enumerate(ds_rows):
            status_cls = "keep" if r["status"] == "keep" else "discard"
            html += f'<tr><td>{i+1}</td><td>{r["backbone"]}</td><td>{r["loss"]}</td>'
            html += f'<td>{r["extras"]}</td><td>{r["metric"]:.4f}</td>'
            html += f'<td>{r["memory_gb"]:.2f}</td>'
            html += f'<td class="{status_cls}">{r["status"]}</td></tr>\n'
        html += '</table></div>\n'

    html += '</body></html>'

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate results dashboard")
    parser.add_argument("--results-file", default="results.tsv",
                        help="Path to results.tsv")
    parser.add_argument("--output", "-o", default="results_dashboard.html",
                        help="Output HTML path")
    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"Results file not found: {args.results_file}")
        sys.exit(1)

    rows = parse_results(args.results_file)
    print(f"Parsed {len(rows)} experiment results")
    generate_html(rows, args.output)


if __name__ == "__main__":
    main()
