import unittest
import importlib.util
from pathlib import Path

import train


def load_module(module_name: str, relative_path: str):
    path = Path(train.__file__).resolve().parent / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


autoresearch_loop = load_module("autoresearch_loop_test", "tools/autoresearch_loop.py")
run_recipe_queue = load_module("run_recipe_queue_test", "tools/run_recipe_queue.py")


class QueueAndLoopSurfaceTests(unittest.TestCase):
    def test_queue_presets_only_reference_available_recipes(self):
        recipe_dir = Path(train.__file__).resolve().parent / "recipes"
        for preset_name, recipes in run_recipe_queue.PRESETS.items():
            self.assertTrue(recipes, msg=f"{preset_name} should not be empty")
            for recipe in recipes:
                resolved = train.resolve_recipe_name(recipe)
                recipe_path = recipe_dir / f"{resolved}.yaml"
                self.assertTrue(
                    recipe_path.exists(),
                    msg=f"{preset_name} references missing recipe '{recipe}'",
                )

    def test_recipe_spec_from_run_uses_dataset_and_task_from_run(self):
        run_row = {
            "dataset": "brain_mri",
            "task": "segmentation",
            "backbone": "resnet34",
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 1e-3,
            "batch_size": 8,
            "dropout": 0.1,
            "image_size": 256,
            "loss": "dice_bce",
        }
        spec = autoresearch_loop.recipe_spec_from_run(run_row, checkpoint_config={})

        self.assertEqual(spec["dataset"], "brain_mri")
        self.assertEqual(spec["task"], "segmentation")
        self.assertEqual(spec["paper"]["dataset"], "brain_mri")
        self.assertEqual(spec["loss"]["class_weights"], False)


if __name__ == "__main__":
    unittest.main()
