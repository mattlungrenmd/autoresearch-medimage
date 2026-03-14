import json
import tempfile
import unittest
from pathlib import Path

from tools.search_report import generate_search_report


class SearchReportTests(unittest.TestCase):
    def test_generate_search_report_writes_summary_and_lineage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_path = root / "results.tsv"
            search_dir = root / "logs" / "session_a" / "search"
            search_dir.mkdir(parents=True, exist_ok=True)

            results_path.write_text(
                "\t".join(
                    [
                        "timestamp",
                        "commit",
                        "session",
                        "dataset",
                        "recipe",
                        "backbone",
                        "optimizer",
                        "loss",
                        "image_size",
                        "batch_size",
                        "learning_rate",
                        "weight_decay",
                        "dropout",
                        "augmentations",
                        "seed",
                        "time_budget_seconds",
                        "metric_name",
                        "metric",
                        "memory_gb",
                        "status",
                        "checkpoint_path",
                        "description",
                        "candidate_id",
                        "parent_id",
                        "search_algo",
                        "mutation_op",
                        "base_recipe",
                        "replicate_group",
                        "failure_reason",
                    ]
                )
                + "\n"
                + "\t".join(
                    [
                        "2026-03-14T00:00:00",
                        "abc123",
                        "session_a",
                        "chestxray14",
                        "cand0001_seed",
                        "densenet121",
                        "adamw",
                        "focal",
                        "224",
                        "16",
                        "0.001",
                        "0.01",
                        "0.2",
                        "hflip",
                        "42",
                        "300.0",
                        "mean_auc",
                        "74.3",
                        "1.0",
                        "keep",
                        "",
                        "",
                        "cand0001",
                        "",
                        "adaevolve_lite",
                        "seed",
                        "chexnet_improved",
                        "cand0001",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (search_dir / "search_state.json").write_text(
                json.dumps(
                    {
                        "search_algo": "adaevolve_lite",
                        "dataset": "chestxray14",
                        "completed_runs": 1,
                        "non_improving_runs": 0,
                        "best_confirmed_metric": 74.3,
                    }
                ),
                encoding="utf-8",
            )
            (search_dir / "candidates.jsonl").write_text(
                json.dumps(
                    {
                        "sequence_id": 1,
                        "candidate_id": "cand0001",
                        "parent_id": "",
                        "recipe_name": "cand0001_seed",
                        "branch_key": "densenet",
                        "mutation_op": "seed",
                        "metric": 74.3,
                        "status": "keep",
                        "replicate_group": "cand0001",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary_path, lineage_path = generate_search_report(
                session="session_a",
                results_path=results_path,
                search_dir=search_dir,
            )

            self.assertTrue(summary_path.exists())
            self.assertTrue(lineage_path.exists())
            self.assertIn("Search Summary: session_a", summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
