import csv
import random
import tempfile
import unittest
from pathlib import Path

import yaml

from tools import run_search_session as rss


class SearchSessionTests(unittest.TestCase):
    def test_branch_key_groups_backbone_families(self):
        self.assertEqual(rss.branch_key_for_backbone("densenet169"), "densenet")
        self.assertEqual(rss.branch_key_for_backbone("efficientnet_b0"), "efficientnet")
        self.assertEqual(rss.branch_key_for_backbone("convnext_tiny"), "convnext")
        self.assertEqual(rss.branch_key_for_backbone("resnet50"), "resnet")

    def test_force_explore_prefers_least_tried_family(self):
        operator_stats = {
            "backbone_swap": {"trials": 5, "reward_sum": 0.5},
            "lr_wd": {"trials": 1, "reward_sum": 0.2},
            "augmentation": {"trials": 0, "reward_sum": 0.0},
        }
        choice = rss.select_family(["backbone_swap", "lr_wd", "augmentation"], operator_stats, random.Random(7), force_explore=True)
        self.assertEqual(choice, "augmentation")

    def test_mutate_recipe_backbone_swap_changes_backbone(self):
        recipe = {
            "dataset": "chestxray14",
            "task": "multilabel",
            "model": {"backbone": "densenet121", "dropout": 0.2},
            "training": {"batch_size": 16, "learning_rate": 0.001, "weight_decay": 0.01},
            "loss": {"type": "focal", "class_weights": True},
            "augmentation": {},
            "advanced": {"lr_schedule": "cosine"},
        }
        space = {
            "task": "multilabel",
            "backbone_pool": ["densenet121", "efficientnet_b0"],
            "mutation_families": ["backbone_swap"],
            "ranges": {},
            "categorical_choices": {},
        }
        mutated, delta = rss.mutate_recipe(recipe, space, "backbone_swap", random.Random(3))
        self.assertEqual(mutated["model"]["backbone"], "efficientnet_b0")
        self.assertEqual(delta["backbone"]["from"], "densenet121")

    def test_choose_from_ladder_prefer_far_handles_string_ladders(self):
        choice = rss.choose_from_ladder(
            "densenet121",
            ["densenet121", "resnet50", "convnext_tiny"],
            random.Random(1),
            prefer_far=True,
        )
        self.assertEqual(choice, "convnext_tiny")

    def test_build_recipe_stem_stays_compact_for_long_candidate_names(self):
        stem = rss.build_recipe_stem(
            "cand0020",
            "cand0019_cand0016_cand0007_cand0005_chestxray14_frontier_efficientnet_b0_stronger_color_loss_augmentation_regularization",
            "confirm",
        )
        self.assertTrue(stem.startswith("cand0020_"))
        self.assertLessEqual(len(stem), rss.MAX_RECIPE_STEM)
        self.assertIn("confirm", stem)

    def test_sync_candidate_from_results_refreshes_pending_candidate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.tsv"
            fieldnames = [
                "session",
                "candidate_id",
                "recipe",
                "backbone",
                "metric_name",
                "metric",
                "status",
                "checkpoint_path",
                "description",
            ]
            with open(results_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                writer.writerow(
                    {
                        "session": "resume_demo",
                        "candidate_id": "cand0019",
                        "recipe": "cand0019_cand0016_regularization",
                        "backbone": "efficientnet_b0",
                        "metric_name": "mean_auc",
                        "metric": "82.348437",
                        "status": "candidate",
                        "checkpoint_path": "checkpoints/demo.pth",
                        "description": "demo result row",
                    }
                )

            candidate = {
                "candidate_id": "cand0019",
                "recipe_name": "old_pending_name",
                "backbone": "",
                "status": "pending",
            }
            synced = rss.sync_candidate_from_results(results_path, "resume_demo", candidate)
            self.assertTrue(synced)
            self.assertEqual(candidate["status"], "candidate")
            self.assertEqual(candidate["recipe_name"], "cand0019_cand0016_regularization")
            self.assertAlmostEqual(candidate["metric"], 82.348437)
            self.assertEqual(candidate["checkpoint_path"], "checkpoints/demo.pth")

    def test_filtered_families_for_phase_uses_phase_specific_gates(self):
        space = {
            "task": "multilabel",
            "mutation_families": ["loss", "augmentation", "regularization", "lr_wd", "optimizer_scheduler", "backbone_swap"],
            "phase_families": {
                "default": ["loss", "augmentation", "regularization", "lr_wd"],
                "explore": ["loss", "augmentation", "regularization", "lr_wd", "optimizer_scheduler"],
                "refresh": ["loss", "augmentation", "regularization", "lr_wd", "optimizer_scheduler", "backbone_swap"],
            },
        }
        recipe = {"task": "multilabel"}
        self.assertEqual(
            rss.filtered_families_for_phase(space, recipe),
            ["loss", "augmentation", "regularization", "lr_wd"],
        )
        self.assertEqual(
            rss.filtered_families_for_phase(space, recipe, force_explore=True),
            ["loss", "augmentation", "regularization", "lr_wd", "optimizer_scheduler"],
        )
        self.assertEqual(
            rss.filtered_families_for_phase(space, recipe, allow_large_jump=True),
            ["loss", "augmentation", "regularization", "lr_wd", "optimizer_scheduler", "backbone_swap"],
        )

    def test_bounded_operator_reward_clips_large_regressions_and_crashes(self):
        space = {"operator_reward": {"clip": 2.0, "crash_penalty": -1.5}}
        self.assertEqual(
            rss.bounded_operator_reward(82.3, float("nan"), {"status": "crash"}, space),
            -1.5,
        )
        self.assertEqual(
            rss.bounded_operator_reward(82.3, 70.0, {"status": "discard"}, space),
            -2.0,
        )

    def test_resolve_seed_recipe_file_uses_search_space_directory_for_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            space_dir = Path(tmpdir) / "search_spaces"
            space_dir.mkdir(parents=True, exist_ok=True)
            recipe_path = Path(tmpdir) / "logs" / "demo" / "seed.yaml"
            recipe_path.parent.mkdir(parents=True, exist_ok=True)
            recipe_path.write_text("dataset: chestxray14\n", encoding="utf-8")
            space = {"_path": str(space_dir / "demo_space.yaml")}
            resolved = rss.resolve_seed_recipe_file(space, "../logs/demo/seed.yaml")
            self.assertEqual(resolved, recipe_path.resolve())

    def test_resolve_space_path_accepts_absolute_yaml_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            space_path = Path(tmpdir) / "refinement_space.yaml"
            space_path.write_text("dataset: chestxray14\n", encoding="utf-8")
            resolved = rss.resolve_space_path(str(space_path.resolve()))
            self.assertEqual(resolved, space_path.resolve())

    def test_initialize_seed_candidates_can_skip_best_known_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            search_dir = Path(tmpdir) / "logs" / "demo" / "search"
            results_path = Path(tmpdir) / "results.tsv"
            with open(results_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["session", "dataset", "recipe", "backbone", "metric_name", "metric", "status"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "session": "older_session",
                        "dataset": "chestxray14",
                        "recipe": "chexnet_improved",
                        "backbone": "densenet121",
                        "metric_name": "mean_auc",
                        "metric": "90.0",
                        "status": "keep",
                    }
                )
            seed_recipe = Path(tmpdir) / "seed.yaml"
            seed_recipe.write_text(
                yaml.safe_dump(
                    {
                        "dataset": "chestxray14",
                        "task": "multilabel",
                        "model": {"backbone": "efficientnet_b0"},
                        "training": {"batch_size": 32, "learning_rate": 0.001, "optimizer": "adamw"},
                        "loss": {"type": "bce"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            space = {
                "dataset": "chestxray14",
                "task": "multilabel",
                "include_best_known_seed": False,
                "seed_recipes": [],
                "seed_recipe_files": [str(seed_recipe.resolve())],
            }
            state = {}
            candidates = []

            rss.initialize_seed_candidates(search_dir, "chestxray14", space, state, candidates, results_path)

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["base_recipe"], seed_recipe.stem)


if __name__ == "__main__":
    unittest.main()
