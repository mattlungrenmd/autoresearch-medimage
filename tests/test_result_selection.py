import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from tools import finalize_run, run_full_train, run_full_train_queue
from tools.result_utils import normalize_status, valid_metric_rows
from tools.search_report import generate_search_report


def result_row(recipe, metric, status="candidate", metric_name="mean_auc"):
    return {
        "session": "demo",
        "dataset": "chestxray14",
        "candidate_id": recipe,
        "recipe": recipe,
        "backbone": "resnet50",
        "metric": metric,
        "metric_name": metric_name,
        "status": status,
    }


class ResultSelectionTests(unittest.TestCase):
    def test_status_normalization_handles_missing_values(self):
        for value in (None, pd.NA, float("nan"), "", "  "):
            with self.subTest(value=value):
                self.assertEqual(normalize_status(value), "")
        self.assertEqual(normalize_status(" KEEP "), "keep")

    def test_valid_rows_exclude_failures_and_nonfinite_scores_without_mutation(self):
        source = pd.DataFrame(
            [
                result_row("crashed", 100, " CRASH "),
                result_row("error", 100, metric_name="error"),
                result_row("nan", float("nan")),
                result_row("infinite", float("inf")),
                result_row("negative_infinite", float("-inf")),
                result_row("unparseable", "not-a-score"),
                result_row("valid", "81.5", " KEEP "),
            ]
        )
        before = source.copy(deep=True)

        selected = valid_metric_rows(source)

        self.assertEqual(selected["recipe"].tolist(), ["valid"])
        self.assertEqual(selected.iloc[0]["metric"], 81.5)
        self.assertEqual(selected.iloc[0]["status_norm"], "keep")
        pd.testing.assert_frame_equal(source, before)

    def test_missing_metric_column_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "'metric' column"):
            valid_metric_rows(pd.DataFrame([{"status": "keep"}]))

    def test_full_train_rejects_sessions_without_a_valid_score(self):
        for row in (
            result_row("crashed", 100, "crash"),
            result_row("unscored", float("nan"), "keep"),
            result_row("infinite", float("inf"), "keep"),
        ):
            with self.subTest(recipe=row["recipe"]):
                with self.assertRaisesRegex(ValueError, "No successful runs"):
                    run_full_train.select_winner_row(pd.DataFrame([row]))

    def test_full_train_prefers_valid_keep_over_higher_candidate(self):
        rows = pd.DataFrame(
            [
                result_row("invalid_keep", float("inf"), "keep"),
                result_row("candidate", 90),
                result_row("kept", 81.5, " KEEP "),
            ]
        )
        self.assertEqual(run_full_train.select_winner_row(rows)["recipe"], "kept")

    def test_finalize_uses_the_same_validity_rules(self):
        rows = pd.DataFrame(
            [result_row("crashed", 100, "crash"), result_row("kept", 81.5, " KEEP ")]
        )
        selected = finalize_run.select_rows(rows, dataset="chestxray14", session="demo")
        self.assertEqual(finalize_run.best_row(selected)["recipe"], "kept")
        with self.assertRaisesRegex(ValueError, "No successful runs"):
            finalize_run.select_rows(rows.iloc[:1])

    def test_queue_ignores_failed_and_infinite_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.tsv"
            pd.DataFrame(
                [
                    result_row("crashed", 100, "crash"),
                    result_row("infinite", float("inf")),
                    result_row("valid", 81.5),
                ]
            ).to_csv(results_path, sep="\t", index=False)
            selected = run_full_train_queue.select_candidate_ids(
                results_path, session="demo", top_k=3
            )
            self.assertEqual(selected, ["valid"])

    def test_finalize_skips_held_out_evaluation_when_checkpoint_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.tsv"
            row = result_row("valid", 81.5)
            row["checkpoint_path"] = None
            pd.DataFrame([row]).to_csv(results_path, sep="\t", index=False)
            argv = [
                "finalize_run.py",
                "--results",
                str(results_path),
                "--session",
                "demo",
            ]
            with (
                mock.patch.object(finalize_run, "REPO_DIR", tmpdir),
                mock.patch.object(finalize_run, "dataset_has_split", return_value=True),
                mock.patch.object(finalize_run, "run_command", return_value="") as run,
                mock.patch("sys.argv", argv),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                finalize_run.main()
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(len(commands), 3)
            self.assertFalse(
                any("evaluate_checkpoint.py" in " ".join(cmd) for cmd in commands)
            )
            self.assertFalse(any("--final-eval" in cmd for cmd in commands))

    def test_queue_does_not_backfill_past_an_explicit_top_k(self):
        selected = run_full_train_queue.select_candidate_ids(
            Path("unused.tsv"),
            session="demo",
            candidate_ids=["first", "second"],
            top_k=2,
        )
        self.assertEqual(selected, ["first", "second"])

    def test_report_does_not_advertise_an_invalid_winner(self):
        for valid in (True, False):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                results_path = root / "results.tsv"
                rows = [
                    result_row("infinite", float("inf")),
                    result_row("crashed", 100, "crash"),
                ]
                if valid:
                    rows.append(result_row("valid", 81.5))
                pd.DataFrame(rows).to_csv(results_path, sep="\t", index=False)

                summary, _ = generate_search_report(
                    "demo", results_path=results_path, search_dir=root / "search"
                )
                content = summary.read_text(encoding="utf-8")
                self.assertNotIn("- recipe: infinite", content)
                self.assertNotIn("- recipe: crashed", content)
                if valid:
                    self.assertIn("- recipe: valid", content)
                else:
                    self.assertNotIn("## Session Winner", content)


if __name__ == "__main__":
    unittest.main()
