import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class NoiseDiagnosticTests(unittest.TestCase):
    def test_seed_variation_bias_and_mae_remain_separate_and_padding_is_excluded(self):
        from autopicker_dp.diagnose import prediction_statistics
        target = np.zeros((1, 2, 4))
        predictions = np.array([[[[1., 2., 3., 4.], [999.]*4]],
                                [[[-1., -2., -3., -4.], [999.]*4]]])
        stats = prediction_statistics(predictions, target, np.array([[True, False]]))
        self.assertEqual(stats['mean_seed_std_by_joint'], [1., 2., 3., 4.])
        self.assertEqual(stats['mean_abs_bias_by_joint'], [0.]*4)
        self.assertEqual(stats['single_sample_mae_by_joint'], [1., 2., 3., 4.])


if __name__ == '__main__':
    unittest.main()
