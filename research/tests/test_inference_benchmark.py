from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_inference import choose_validation_windows, percentile_summary


class FakeSequenceDataset:
    def __init__(self, indices):
        self.indices = indices


class InferenceBenchmarkTests(unittest.TestCase):
    def test_percentile_summary_reports_expected_statistics(self):
        result = percentile_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["min"], 1.0)
        self.assertEqual(result["max"], 4.0)
        self.assertEqual(result["mean"], 2.5)
        self.assertAlmostEqual(result["p50"], 2.5)

    def test_percentile_summary_rejects_empty_or_nonfinite_input(self):
        for values in ([], [1.0, float("nan")], [float("inf")]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    percentile_summary(values)

    def test_window_selection_balances_samples_across_episodes(self):
        sequence = FakeSequenceDataset([
            (0, 3, 0), (0, 3, 1), (0, 3, 2),
            (3, 6, 3), (3, 6, 4), (3, 6, 5),
        ])
        selected = choose_validation_windows(sequence, np.array([3, 6]), max_samples=4)
        self.assertEqual(selected, [0, 2, 3, 5])

    def test_window_selection_caps_requested_count(self):
        sequence = FakeSequenceDataset([(0, 10, i) for i in range(10)])
        selected = choose_validation_windows(sequence, np.array([10]), max_samples=3)
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected, [0, 4, 9])


if __name__ == "__main__":
    unittest.main()
