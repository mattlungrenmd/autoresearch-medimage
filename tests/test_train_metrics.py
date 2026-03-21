import math
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

import train


class IdentityLogitModel(torch.nn.Module):
    def forward(self, x):
        return x.float()


class TrainMetricTests(unittest.TestCase):
    def test_classification_eval_reports_f1(self):
        logits = torch.tensor(
            [
                [3.0, 1.0],
                [1.0, 3.0],
                [4.0, 1.0],
                [1.0, 4.0],
            ]
        )
        targets = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(TensorDataset(logits, targets), batch_size=2, shuffle=False)
        cfg = {"num_classes": 2}

        results = train.evaluate_classification_loader(
            IdentityLogitModel(),
            loader,
            cfg,
            torch.device("cpu"),
        )

        self.assertAlmostEqual(results["accuracy"], 100.0)
        self.assertAlmostEqual(results["auc"], 100.0)
        self.assertAlmostEqual(results["f1"], 100.0)
        self.assertAlmostEqual(results["macro_f1"], 100.0)

    def test_multilabel_eval_can_optimize_and_reuse_f1_thresholds(self):
        probs = torch.tensor(
            [
                [0.40, 0.10],
                [0.30, 0.20],
                [0.20, 0.70],
                [0.10, 0.80],
            ],
            dtype=torch.float32,
        )
        logits = torch.logit(probs, eps=1e-6)
        targets = torch.tensor(
            [
                [1, 0],
                [0, 0],
                [0, 1],
                [0, 1],
            ],
            dtype=torch.float32,
        )
        loader = DataLoader(TensorDataset(logits, targets), batch_size=2, shuffle=False)
        cfg = {"num_classes": 2, "class_names": ["FindingA", "FindingB"]}

        baseline = train.evaluate_multilabel_loader(
            IdentityLogitModel(),
            loader,
            cfg,
            torch.device("cpu"),
        )
        tuned = train.evaluate_multilabel_loader(
            IdentityLogitModel(),
            loader,
            cfg,
            torch.device("cpu"),
            optimize_thresholds_for_f1=True,
        )

        self.assertIn("macro_f1", tuned)
        self.assertIn("micro_f1", tuned)
        self.assertIn("per_class_f1", tuned)
        self.assertIn("f1_thresholds", tuned)
        self.assertGreater(tuned["macro_f1"], baseline["macro_f1"])
        self.assertLess(tuned["f1_thresholds"]["FindingA"], 0.5)

        reused = train.evaluate_multilabel_loader(
            IdentityLogitModel(),
            loader,
            cfg,
            torch.device("cpu"),
            thresholds=[
                tuned["f1_thresholds"]["FindingA"],
                tuned["f1_thresholds"]["FindingB"],
            ],
        )
        self.assertTrue(math.isclose(reused["macro_f1"], tuned["macro_f1"], rel_tol=1e-6))
        self.assertTrue(math.isclose(reused["micro_f1"], tuned["micro_f1"], rel_tol=1e-6))

    def test_primary_metric_supports_f1_variants(self):
        metric_name, metric_value = train.primary_metric_for_results(
            {"f1": 88.5, "macro_f1": 77.0},
            {"task": "classification", "metric": "f1"},
        )
        self.assertEqual(metric_name, "f1")
        self.assertEqual(metric_value, 88.5)

        metric_name, metric_value = train.primary_metric_for_results(
            {"mean_auc": 84.1, "macro_f1": 73.2, "micro_f1": 79.4},
            {"task": "multilabel", "metric": "macro_f1"},
        )
        self.assertEqual(metric_name, "macro_f1")
        self.assertEqual(metric_value, 73.2)


if __name__ == "__main__":
    unittest.main()
