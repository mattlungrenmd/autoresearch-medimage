import unittest

import train


class TrainSurfaceTests(unittest.TestCase):
    def test_results_metadata_helpers_exist(self):
        self.assertTrue(hasattr(train, "RESULTS_FILE"))
        self.assertTrue(hasattr(train, "CURRENT_RUN_METADATA"))
        self.assertTrue(callable(train.ensure_results_file))
        self.assertTrue(callable(train.current_results_metadata))

    def test_windows_worker_helper_exists(self):
        self.assertTrue(callable(train.recommended_windows_worker_count))
        self.assertEqual(
            train.recommended_windows_worker_count(4, batch_size=16, image_size=224),
            2,
        )


if __name__ == "__main__":
    unittest.main()
