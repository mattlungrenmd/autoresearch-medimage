import csv
import random
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
