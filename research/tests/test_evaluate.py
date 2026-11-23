from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ActionMetricTests(unittest.TestCase):
    def test_padding_is_excluded_and_joint_units_are_not_averaged(self):
        from autopicker_dp.evaluate import action_metrics
        target = np.zeros((2, 2, 4))
        predicted = np.array([[[1., 2., 3., 4.], [100.] * 4],
                              [[-1., -2., -3., -4.], [100.] * 4]])
        result = action_metrics(predicted, target, np.array([[1, 0], [1, 0]], dtype=bool),
                                np.full(4, -5.), np.full(4, 5.))
        self.assertEqual(result['valid_action_nodes'], 2)
        np.testing.assert_array_equal(result['mae_by_joint'], [1., 2., 3., 4.])
        np.testing.assert_array_equal(result['rmse_by_joint'], [1., 2., 3., 4.])
        self.assertEqual(result['out_of_limits_nodes'], 0)


if __name__ == '__main__':
    unittest.main()
