import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from tools import run_full_train as rft


class RunFullTrainTests(unittest.TestCase):
    def test_select_winner_prefers_best_keep_row(self):
        df = pd.DataFrame(
            [
                {"recipe": "baseline", "metric": 80.1, "status_norm": "discard"},
                {"recipe": "winner", "metric": 81.7, "status_norm": "keep"},
                {"recipe": "better_but_discard", "metric": 82.0, "status_norm": "discard"},
            ]
        )
        winner = rft.select_winner_row(df)
        self.assertEqual(winner["recipe"], "winner")

    def test_filter_results_can_target_candidate_id(self):
        df = pd.DataFrame(
            [
                {"session": "demo", "dataset": "chestxray14", "candidate_id": "cand0003", "metric": 80.9},
                {"session": "demo", "dataset": "chestxray14", "candidate_id": "cand0026", "metric": 81.3},
            ]
        )
        filtered = rft.filter_results(df, dataset="chestxray14", session="demo", candidate_id="cand0026")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["candidate_id"], "cand0026")

    def test_resolve_recipe_source_uses_sanitized_search_recipe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir)
            old_repo_dir = rft.REPO_DIR
            old_recipes_dir = rft.RECIPES_DIR
            try:
                rft.REPO_DIR = repo_dir
                rft.RECIPES_DIR = repo_dir / "recipes"
                session = "demo_session"
                search_dir = repo_dir / "logs" / session / "search"
                recipe_dir = search_dir / "recipes"
                recipe_dir.mkdir(parents=True)
                recipe_path = recipe_dir / "cand0007.yaml"
                with open(recipe_path, "w", encoding="utf-8") as handle:
                    yaml.safe_dump(
                        {
                            "dataset": "chestxray14",
                            "model": {"backbone": "efficientnet_b0"},
                            "search_metadata": {"candidate_id": "cand0007"},
                        },
                        handle,
                        sort_keys=False,
                    )
                candidates_file = search_dir / "candidates.jsonl"
                candidates_file.write_text(
                    json.dumps(
                        {
                            "candidate_id": "cand0007",
                            "recipe_name": "cand0007",
                            "recipe_file": str(recipe_path),
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                winner = pd.Series({"candidate_id": "cand0007", "recipe": "cand0007"})
                output_dir = repo_dir / "logs" / "promoted" / "full_train"

                source_kind, source_value = rft.resolve_recipe_source(winner, session, output_dir, repo_dir)

                self.assertEqual(source_kind, "recipe-file")
                sanitized_path = Path(source_value)
                self.assertTrue(sanitized_path.exists())
                sanitized = yaml.safe_load(sanitized_path.read_text(encoding="utf-8"))
                self.assertNotIn("search_metadata", sanitized)
                self.assertEqual(sanitized["model"]["backbone"], "efficientnet_b0")
            finally:
                rft.REPO_DIR = old_repo_dir
                rft.RECIPES_DIR = old_recipes_dir

    def test_resolve_recipe_source_uses_full_train_selection_recipe_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir)
            old_repo_dir = rft.REPO_DIR
            old_recipes_dir = rft.RECIPES_DIR
            try:
                rft.REPO_DIR = repo_dir
                rft.RECIPES_DIR = repo_dir / "recipes"
                session = "cand0056_fulltrain_ep10"
                full_train_dir = repo_dir / "logs" / session / "full_train"
                full_train_dir.mkdir(parents=True)
                recipe_path = full_train_dir / "cand0056_cand0020_lr_wd.yaml"
                with open(recipe_path, "w", encoding="utf-8") as handle:
                    yaml.safe_dump(
                        {
                            "dataset": "chestxray14",
                            "model": {"backbone": "efficientnet_b0"},
                            "search_metadata": {"candidate_id": "cand0056"},
                        },
                        handle,
                        sort_keys=False,
                    )
                (full_train_dir / "selection.json").write_text(
                    json.dumps(
                        {
                            "source_kind": "recipe-file",
                            "source_value": str(recipe_path),
                            "train_session": session,
                        }
                    ),
                    encoding="utf-8",
                )
                winner = pd.Series({"recipe": "cand0056_cand0020_lr_wd"})
                output_dir = repo_dir / "logs" / "stage2" / "refine_seeds"

                source_kind, source_value = rft.resolve_recipe_source(winner, session, output_dir, repo_dir)

                self.assertEqual(source_kind, "recipe-file")
                sanitized_path = Path(source_value)
                self.assertTrue(sanitized_path.exists())
                sanitized = yaml.safe_load(sanitized_path.read_text(encoding="utf-8"))
                self.assertNotIn("search_metadata", sanitized)
                self.assertEqual(sanitized["model"]["backbone"], "efficientnet_b0")
            finally:
                rft.REPO_DIR = old_repo_dir
                rft.RECIPES_DIR = old_recipes_dir

    def test_build_command_uses_epoch_budget(self):
        command = rft.build_command(
            python_executable=Path("python.exe"),
            train_script=Path("train.py"),
            source_kind="recipe",
            source_value="chexnet",
            epochs=10,
            train_session="demo_fulltrain_ep10",
            dataset="chestxray14",
            dump_effective_config=Path("effective.json"),
            save_checkpoint=True,
        )
        self.assertIn("--epochs", command)
        self.assertIn("10", command)
        self.assertIn("--recipe", command)
        self.assertIn("chexnet", command)
        self.assertIn("--save-checkpoint", command)
        self.assertEqual(command[0], "python.exe")

    def test_resolve_workspace_python_prefers_workspace_venv_gpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            python_path = workspace / ".venv-gpu" / "Scripts" / "python.exe"
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")
            resolved = rft.resolve_workspace_python(workspace)
            self.assertEqual(resolved, python_path)

    def test_resolve_python_executable_honors_explicit_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            explicit_python = Path(tmpdir) / "custom-python.exe"
            explicit_python.write_text("", encoding="utf-8")

            resolved = rft.resolve_python_executable(workspace, explicit_python=str(explicit_python))

            self.assertEqual(resolved, explicit_python.resolve())

    def test_launcher_paths_live_under_full_train_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs" / "demo" / "full_train"
            launcher_log, launcher_err = rft.launcher_paths(logs_dir)
            self.assertEqual(launcher_log.name, "launcher.log")
            self.assertEqual(launcher_err.name, "launcher.err.log")
            self.assertIn("full_train", str(launcher_log))


if __name__ == "__main__":
    unittest.main()
