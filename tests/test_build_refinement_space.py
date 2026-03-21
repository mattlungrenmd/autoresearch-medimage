import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml

from tools import build_refinement_space as brs


class BuildRefinementSpaceTests(unittest.TestCase):
    def test_main_writes_refinement_space_with_seed_recipe_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            search_space_path = workspace / "search_spaces" / "demo.yaml"
            search_space_path.parent.mkdir(parents=True, exist_ok=True)
            search_space_path.write_text(
                yaml.safe_dump(
                    {
                        "dataset": "chestxray14",
                        "task": "multilabel",
                        "seed_recipes": ["chestxray14_advanced"],
                        "backbone_pool": ["efficientnet_b0", "densenet169"],
                        "mutation_families": ["loss", "lr_wd"],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            results_path = workspace / "results.tsv"
            pd.DataFrame(
                [
                    {
                        "session": "demo_search",
                        "dataset": "chestxray14",
                        "candidate_id": "cand0001",
                        "recipe": "cand0001_demo",
                        "backbone": "efficientnet_b0",
                        "metric": 82.5,
                        "status": "discard",
                    }
                ]
            ).to_csv(results_path, sep="\t", index=False)

            search_dir = workspace / "logs" / "demo_search" / "search"
            recipe_dir = search_dir / "recipes"
            recipe_dir.mkdir(parents=True, exist_ok=True)
            recipe_path = recipe_dir / "cand0001_demo.yaml"
            recipe_path.write_text(
                yaml.safe_dump(
                    {
                        "dataset": "chestxray14",
                        "model": {"backbone": "efficientnet_b0"},
                        "training": {"batch_size": 32, "learning_rate": 0.0003, "optimizer": "adamw"},
                        "loss": {"type": "bce"},
                        "search_metadata": {"candidate_id": "cand0001"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            (search_dir / "candidates.jsonl").write_text(
                json.dumps(
                    {
                        "candidate_id": "cand0001",
                        "recipe_name": "cand0001_demo",
                        "recipe_file": str(recipe_path),
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            output_path = workspace / "logs" / "demo_refine" / "refinement_space.yaml"
            argv = [
                "build_refinement_space.py",
                "--session",
                "demo_search",
                "--space",
                str(search_space_path),
                "--results",
                str(results_path),
                "--workspace-root",
                str(workspace),
                "--top-k",
                "1",
                "--output",
                str(output_path),
            ]
            with mock.patch("sys.argv", argv):
                brs.main()

            self.assertTrue(output_path.exists())
            payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["refinement_parent_session"], "demo_search")
            self.assertEqual(payload["refinement_candidates"], ["cand0001"])
            self.assertEqual(payload["seed_recipes"], [])
            self.assertEqual(len(payload["seed_recipe_files"]), 1)
            seed_path = output_path.parent / payload["seed_recipe_files"][0]
            self.assertTrue(seed_path.exists())

    def test_main_writes_refinement_space_from_seed_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            search_space_path = workspace / "search_spaces" / "demo.yaml"
            search_space_path.parent.mkdir(parents=True, exist_ok=True)
            search_space_path.write_text(
                yaml.safe_dump(
                    {
                        "dataset": "chestxray14",
                        "task": "multilabel",
                        "seed_recipes": ["chestxray14_advanced"],
                        "backbone_pool": ["efficientnet_b0", "densenet169", "resnet50"],
                        "mutation_families": ["backbone_swap", "loss", "lr_wd"],
                        "phase_families": {
                            "default": ["loss", "lr_wd"],
                            "refresh": ["loss", "lr_wd", "backbone_swap"],
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            results_path = workspace / "results.tsv"
            pd.DataFrame(
                [
                    {
                        "session": "cand0056_fulltrain_ep10",
                        "dataset": "chestxray14",
                        "recipe": "cand0056_cand0020_lr_wd",
                        "backbone": "efficientnet_b0",
                        "metric": 84.80,
                        "status": "candidate",
                    },
                    {
                        "session": "cand0026_fulltrain_ep10",
                        "dataset": "chestxray14",
                        "recipe": "cand0026_cand0002_loss",
                        "backbone": "densenet169",
                        "metric": 84.72,
                        "status": "candidate",
                    },
                ]
            ).to_csv(results_path, sep="\t", index=False)

            for session, recipe_name, backbone in [
                ("cand0056_fulltrain_ep10", "cand0056_cand0020_lr_wd", "efficientnet_b0"),
                ("cand0026_fulltrain_ep10", "cand0026_cand0002_loss", "densenet169"),
            ]:
                full_train_dir = workspace / "logs" / session / "full_train"
                full_train_dir.mkdir(parents=True, exist_ok=True)
                recipe_path = full_train_dir / f"{recipe_name}.yaml"
                recipe_path.write_text(
                    yaml.safe_dump(
                        {
                            "dataset": "chestxray14",
                            "model": {"backbone": backbone},
                            "training": {"batch_size": 16, "learning_rate": 0.0003, "optimizer": "adamw"},
                            "loss": {"type": "bce"},
                            "search_metadata": {"from_session": session},
                        },
                        sort_keys=False,
                    ),
                    encoding="utf-8",
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

            output_path = workspace / "logs" / "demo_refine" / "refinement_space.yaml"
            argv = [
                "build_refinement_space.py",
                "--seed-session",
                "cand0056_fulltrain_ep10",
                "--seed-session",
                "cand0026_fulltrain_ep10",
                "--space",
                str(search_space_path),
                "--results",
                str(results_path),
                "--workspace-root",
                str(workspace),
                "--output",
                str(output_path),
                "--mutation-family",
                "loss",
                "--mutation-family",
                "lr_wd",
            ]
            with mock.patch("sys.argv", argv):
                brs.main()

            payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["refinement_parent_sessions"],
                ["cand0056_fulltrain_ep10", "cand0026_fulltrain_ep10"],
            )
            self.assertFalse(payload["include_best_known_seed"])
            self.assertEqual(
                payload["refinement_candidates"],
                ["cand0056_cand0020_lr_wd", "cand0026_cand0002_loss"],
            )
            self.assertEqual(payload["seed_recipes"], [])
            self.assertEqual(payload["mutation_families"], ["loss", "lr_wd"])
            self.assertEqual(payload["phase_families"]["refresh"], ["loss", "lr_wd"])
            self.assertEqual(len(payload["seed_recipe_files"]), 2)
            for relative_seed in payload["seed_recipe_files"]:
                seed_path = output_path.parent / relative_seed
                self.assertTrue(seed_path.exists())
                seed_payload = yaml.safe_load(seed_path.read_text(encoding="utf-8"))
                self.assertNotIn("search_metadata", seed_payload)


if __name__ == "__main__":
    unittest.main()
