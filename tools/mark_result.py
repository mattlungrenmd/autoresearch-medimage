"""
Mark an experiment row in results.tsv as keep, discard, or candidate.

This is the lightweight helper for the search phase:
`train.py` appends rows automatically, and the agent/human can then mark the
latest result as kept or discarded after reading run.log.
"""

import argparse
import os

import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS = os.path.join(REPO_DIR, "results.tsv")


def pick_row(df, dataset=None, recipe=None, session=None, row_index=None):
    view = df.copy()
    if dataset:
        view = view[view["dataset"].astype(str) == str(dataset)]
    if recipe:
        view = view[view["recipe"].astype(str) == str(recipe)]
    if session:
        if "session" not in view.columns:
            raise ValueError("results.tsv does not contain a 'session' column")
        view = view[view["session"].fillna("").astype(str) == str(session)]
    if view.empty:
        raise ValueError("No matching rows found in results.tsv")

    if row_index is not None:
        if row_index < 0 or row_index >= len(view):
            raise ValueError(f"--row must be between 0 and {len(view) - 1}")
        return view.index[row_index]

    return view.index[-1]


def main():
    parser = argparse.ArgumentParser(description="Mark a results.tsv row as keep/discard/candidate")
    parser.add_argument("--results", type=str, default=DEFAULT_RESULTS, help="Path to results.tsv")
    parser.add_argument("--status", type=str, required=True, choices=["keep", "discard", "candidate"], help="New status")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset filter before selecting the row")
    parser.add_argument("--recipe", type=str, default=None, help="Optional recipe filter before selecting the row")
    parser.add_argument("--session", type=str, default=None, help="Optional session filter before selecting the row")
    parser.add_argument("--row", type=int, default=None, help="0-based row within the filtered view; defaults to the latest row")
    parser.add_argument("--append-note", type=str, default=None, help="Optional note appended to the description")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        raise FileNotFoundError(f"results.tsv not found at: {args.results}")

    df = pd.read_csv(args.results, sep="\t")
    if df.empty:
        raise ValueError("results.tsv is empty")
    if "status" not in df.columns:
        raise ValueError("results.tsv must contain a 'status' column")

    idx = pick_row(df, dataset=args.dataset, recipe=args.recipe, session=args.session, row_index=args.row)
    df.at[idx, "status"] = args.status

    if args.append_note:
        existing = str(df.at[idx, "description"]) if "description" in df.columns else ""
        if existing and existing.lower() != "nan":
            df.at[idx, "description"] = f"{existing} | {args.append_note}"
        elif "description" in df.columns:
            df.at[idx, "description"] = args.append_note

    df.to_csv(args.results, sep="\t", index=False)

    row = df.loc[idx]
    print("Updated row:")
    print(f"  dataset: {row.get('dataset', '')}")
    if "session" in row:
        print(f"  session: {row.get('session', '')}")
    print(f"  recipe: {row.get('recipe', '')}")
    print(f"  backbone: {row.get('backbone', '')}")
    print(f"  metric: {row.get('metric_name', 'metric')}={row.get('metric', '')}")
    print(f"  status: {row.get('status', '')}")
    if "description" in row:
        print(f"  description: {row.get('description', '')}")


if __name__ == "__main__":
    main()
