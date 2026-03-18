import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from tools import finalize_run as fr


class FinalizeRunTests(unittest.TestCase):
    def test_main_honors_custom_final_eval_and_figures_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir)
            results_path = repo_dir / "results.tsv"
            pd.DataFrame(
                [
                    {
                        "session": "cand0056_fulltrain_ep10",
                        "dataset": "chestxray14",
                        "recipe": "cand0056_cand0020_lr_wd",
                        "metric_name": "mean_auc",
                        "metric": 84.80239,
                        "status": "candidate",
                        "checkpoint_path": str(repo_dir / "checkpoints" / "winner.pth"),
                    }
                ]
            ).to_csv(results_path, sep="\t", index=False)

            commands = []
            old_repo_dir = fr.REPO_DIR
            try:
                fr.REPO_DIR = str(repo_dir)
                custom_eval = repo_dir / "paper" / "generated" / "cand0056_test_eval.pt"
                custom_figures = repo_dir / "paper" / "generated" / "cand0056_figures"
                argv = [
                    "finalize_run.py",
                    "--results",
                    str(results_path),
                    "--dataset",
                    "chestxray14",
                    "--session",
                    "cand0056_fulltrain_ep10",
                    "--final-eval-output",
                    str(custom_eval),
                    "--figures-output-dir",
                    str(custom_figures),
                ]
                with mock.patch("tools.finalize_run.dataset_has_split", return_value=True), mock.patch(
                    "tools.finalize_run.run_command",
                    side_effect=lambda command, workdir: commands.append(command) or "",
                ), mock.patch("sys.argv", argv):
                    fr.main()
            finally:
                fr.REPO_DIR = old_repo_dir

            eval_commands = [cmd for cmd in commands if "evaluate_checkpoint.py" in " ".join(cmd)]
            figure_commands = [cmd for cmd in commands if "generate_figures.py" in " ".join(cmd)]
            self.assertEqual(len(eval_commands), 1)
            self.assertIn(str(custom_eval), eval_commands[0])
            self.assertEqual(len(figure_commands), 1)
            self.assertIn(str(custom_figures), figure_commands[0])

    def test_main_uses_default_output_paths_when_custom_paths_omitted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir)
            results_path = repo_dir / "results.tsv"
            pd.DataFrame(
                [
                    {
                        "session": "demo_session",
                        "dataset": "toy_binary",
                        "recipe": "toy_binary_smoke",
                        "metric_name": "auc",
                        "metric": 100.0,
                        "status": "candidate",
                        "checkpoint_path": str(repo_dir / "checkpoints" / "winner.pth"),
                    }
                ]
            ).to_csv(results_path, sep="\t", index=False)

            commands = []
            old_repo_dir = fr.REPO_DIR
            try:
                fr.REPO_DIR = str(repo_dir)
                expected_eval = repo_dir / "paper" / "generated" / "toy_binary_final_eval.pt"
                expected_figures = repo_dir / "paper" / "generated" / "toy_binary_figures"
                argv = [
                    "finalize_run.py",
                    "--results",
                    str(results_path),
                    "--dataset",
                    "toy_binary",
                    "--session",
                    "demo_session",
                ]
                with mock.patch("tools.finalize_run.dataset_has_split", return_value=True), mock.patch(
                    "tools.finalize_run.run_command",
                    side_effect=lambda command, workdir: commands.append(command) or "",
                ), mock.patch("sys.argv", argv):
                    fr.main()
            finally:
                fr.REPO_DIR = old_repo_dir

            eval_commands = [cmd for cmd in commands if "evaluate_checkpoint.py" in " ".join(cmd)]
            figure_commands = [cmd for cmd in commands if "generate_figures.py" in " ".join(cmd)]
            self.assertEqual(len(eval_commands), 1)
            self.assertIn(str(expected_eval), eval_commands[0])
            self.assertEqual(len(figure_commands), 1)
            self.assertIn(str(expected_figures), figure_commands[0])


if __name__ == "__main__":
    unittest.main()
