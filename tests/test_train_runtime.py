import unittest

import train


class TrainRuntimeTests(unittest.TestCase):
    def test_recommended_windows_worker_count_downshifts_for_large_batches(self):
        self.assertEqual(train.recommended_windows_worker_count(4, batch_size=16, image_size=224), 2)
        self.assertEqual(train.recommended_windows_worker_count(4, batch_size=32, image_size=224), 1)
        self.assertEqual(train.recommended_windows_worker_count(4, batch_size=32, image_size=288), 0)


if __name__ == "__main__":
    unittest.main()
