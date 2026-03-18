import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import cleanup_artifacts as ca


class CleanupArtifactsTests(unittest.TestCase):
    def write_results(self, path: Path, rows: list[dict[str, object]]) -> None:
        fieldnames = ["metric", "checkpoint_path"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_load_best_checkpoint_names_keeps_top_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.tsv"
            self.write_results(
                results_path,
                [
                    {"metric": 84.8, "checkpoint_path": "checkpoints/winner_b.pth"},
                    {"metric": 82.1, "checkpoint_path": "checkpoints/winner_c.pth"},
                    {"metric": 84.2, "checkpoint_path": "checkpoints/winner_a.pth"},
                ],
            )

            keep_names = ca.load_best_checkpoint_names(results_path, keep_best=2)

            self.assertEqual(keep_names, {"winner_b.pth", "winner_a.pth"})

    def test_main_dry_run_plans_checkpoint_and_smoke_log_cleanup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir)
            checkpoints_dir = repo_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            (checkpoints_dir / "keep_me.pth").write_bytes(b"x" * 10)
            (checkpoints_dir / "drop_me.pth").write_bytes(b"x" * 20)

            logs_dir = repo_dir / "logs"
            (logs_dir / "publish_smoke_20260314").mkdir(parents=True, exist_ok=True)
            (logs_dir / "publish_smoke_20260314" / "run.log").write_text("smoke", encoding="utf-8")
            (logs_dir / "real_session").mkdir(parents=True, exist_ok=True)
            (logs_dir / "real_session" / "run.log").write_text("keep", encoding="utf-8")

            results_path = repo_dir / "results.tsv"
            self.write_results(
                results_path,
                [
                    {"metric": 99.0, "checkpoint_path": str(checkpoints_dir / "keep_me.pth")},
                    {"metric": 10.0, "checkpoint_path": str(checkpoints_dir / "drop_me.pth")},
                ],
            )

            argv = [
                "cleanup_artifacts.py",
                "--repo-root",
                str(repo_dir),
                "--results",
                str(results_path),
                "--keep-best",
                "1",
                "--remove-smoke-logs",
            ]
            with mock.patch("sys.argv", argv), mock.patch("builtins.print") as mocked_print:
                ca.main()

            output = "\n".join(" ".join(str(arg) for arg in call.args) for call in mocked_print.call_args_list)
            self.assertIn("drop_me.pth", output)
            self.assertIn("publish_smoke_20260314", output)
            self.assertIn("Keeping checkpoint files:", output)
            self.assertIn("keep_me.pth", output)
            self.assertNotIn("[checkpoint] " + str(checkpoints_dir / "keep_me.pth"), output)
            self.assertTrue((checkpoints_dir / "drop_me.pth").exists())
            self.assertTrue((logs_dir / "publish_smoke_20260314").exists())


if __name__ == "__main__":
    unittest.main()
