import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


class RunFullTrainQueueTests(unittest.TestCase):
    def test_default_train_session_uses_candidate_and_epoch_budget(self):
        from tools import run_full_train_queue as rftq

        self.assertEqual(rftq.default_train_session("cand0026", 10), "cand0026_fulltrain_ep10")

    def test_queue_plan_path_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)
            queue_dir = workspace_root / "logs" / "demo_session_fulltrain_queue"
            queue_dir.mkdir(parents=True, exist_ok=True)
            plan_path = queue_dir / "queue_plan.json"
            self.assertEqual(plan_path.name, "queue_plan.json")
            self.assertIn("fulltrain_queue", str(plan_path))

    def test_select_candidate_ids_prefers_diverse_branches_then_backfills(self):
        from tools import run_full_train_queue as rftq

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.tsv"
            df = pd.DataFrame(
                [
                    {"session": "demo", "dataset": "chestxray14", "candidate_id": "candA", "backbone": "efficientnet_b0", "metric": 82.5, "status": "discard"},
                    {"session": "demo", "dataset": "chestxray14", "candidate_id": "candB", "backbone": "efficientnet_b0", "metric": 82.2, "status": "discard"},
                    {"session": "demo", "dataset": "chestxray14", "candidate_id": "candC", "backbone": "densenet169", "metric": 82.1, "status": "discard"},
                    {"session": "demo", "dataset": "chestxray14", "candidate_id": "candD", "backbone": "resnet50", "metric": 81.9, "status": "discard"},
                    {"session": "demo", "dataset": "chestxray14", "candidate_id": "candE", "backbone": "convnext_tiny", "metric": 81.5, "status": "discard"},
                ]
            )
            df.to_csv(results_path, sep="\t", index=False)

            selected = rftq.select_candidate_ids(
                results_path=results_path,
                session="demo",
                dataset="chestxray14",
                top_k=5,
                diverse_by="branch",
            )

            self.assertEqual(selected[:4], ["candA", "candC", "candD", "candE"])
            self.assertEqual(selected[4], "candB")

    def test_select_candidate_ids_keeps_explicit_candidates_first(self):
        from tools import run_full_train_queue as rftq

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.tsv"
            df = pd.DataFrame(
                [
                    {"session": "demo", "dataset": "chestxray14", "candidate_id": "candA", "backbone": "efficientnet_b0", "metric": 82.5, "status": "discard"},
                    {"session": "demo", "dataset": "chestxray14", "candidate_id": "candB", "backbone": "densenet169", "metric": 82.2, "status": "discard"},
                    {"session": "demo", "dataset": "chestxray14", "candidate_id": "candC", "backbone": "resnet50", "metric": 81.9, "status": "discard"},
                ]
            )
            df.to_csv(results_path, sep="\t", index=False)

            selected = rftq.select_candidate_ids(
                results_path=results_path,
                session="demo",
                dataset="chestxray14",
                candidate_ids=["candC"],
                top_k=2,
                diverse_by="branch",
            )

            self.assertEqual(selected, ["candC", "candA"])

    def test_main_passes_python_executable_through_to_run_full_train(self):
        from tools import run_full_train_queue as rftq

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)
            results_path = workspace_root / "results.tsv"
            pd.DataFrame(
                [
                    {
                        "session": "demo",
                        "dataset": "chestxray14",
                        "candidate_id": "candA",
                        "backbone": "efficientnet_b0",
                        "metric": 82.5,
                        "status": "keep",
                    }
                ]
            ).to_csv(results_path, sep="\t", index=False)

            argv = [
                "run_full_train_queue.py",
                "--session",
                "demo",
                "--results",
                str(results_path),
                "--workspace-root",
                str(workspace_root),
                "--top-k",
                "1",
                "--epochs",
                "20",
                "--python-executable",
                "C:\\gpu\\python.exe",
                "--dry-run",
            ]
            with mock.patch("sys.argv", argv):
                rftq.main()

            plan_path = workspace_root / "logs" / "demo_fulltrain_queue" / "queue_plan.json"
            self.assertTrue(plan_path.exists())
            payload = plan_path.read_text(encoding="utf-8")
            self.assertIn("C:\\\\gpu\\\\python.exe", payload)


if __name__ == "__main__":
    unittest.main()
