import csv
import tempfile
import unittest
from pathlib import Path

import train


class TrainResultsHeaderTests(unittest.TestCase):
    def test_ensure_results_file_widens_legacy_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.tsv"
            legacy_header = [
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
            ]
            with open(results_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=legacy_header, delimiter="\t")
                writer.writeheader()
                writer.writerow(
                    {
                        "timestamp": "2026-03-14T00:00:00",
                        "dataset": "toy_binary",
                        "recipe": "toy_binary_smoke",
                        "backbone": "resnet18",
                        "metric_name": "auc",
                        "metric": "100.0",
                        "status": "candidate",
                    }
                )

            original_results = train.RESULTS_FILE
            try:
                train.RESULTS_FILE = str(results_path)
                train.ensure_results_file()
            finally:
                train.RESULTS_FILE = original_results

            with open(results_path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                self.assertIn("candidate_id", reader.fieldnames)
                self.assertIn("failure_reason", reader.fieldnames)
                row = next(reader)
            self.assertEqual(row["recipe"], "toy_binary_smoke")
            self.assertEqual(row["candidate_id"], "")
            self.assertEqual(row["failure_reason"], "")

    def test_current_results_metadata_includes_search_fields(self):
        original_metadata = dict(train.CURRENT_RUN_METADATA)
        try:
            train.CURRENT_RUN_METADATA = {
                "candidate_id": "cand0007",
                "parent_id": "cand0003",
                "search_algo": "adaevolve_lite",
                "mutation_op": "lr_wd",
                "base_recipe": "chexnet_improved",
                "replicate_group": "cand0007",
                "failure_reason": "",
            }
            metadata = train.current_results_metadata({"image_size": 224}, "focal", batch_size=16)
        finally:
            train.CURRENT_RUN_METADATA = original_metadata

        self.assertEqual(metadata["candidate_id"], "cand0007")
        self.assertEqual(metadata["mutation_op"], "lr_wd")
        self.assertEqual(metadata["batch_size"], "16")


if __name__ == "__main__":
    unittest.main()
