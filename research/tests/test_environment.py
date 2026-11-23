from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class CaptureGeometryTests(unittest.TestCase):
    def test_reset_does_not_move_previous_apple_during_initialization(self):
        from autopicker_dp.environment import PickingEnvironment
        previous = {'position': 'collected'}
        marker = SimpleNamespace(entry=previous)
        def initialize():
            if marker.entry is not None:
                marker.entry['position'] = 'arm_home'
        demo = SimpleNamespace(
            markers=[marker], pool=SimpleNamespace(reset=lambda: previous.update(position='tree')),
            interaction=Mock(), robot=Mock(), arms=[], records=[], initialize=initialize)
        environment = PickingEnvironment.__new__(PickingEnvironment)
        environment.demo = demo
        environment.world = Mock()
        environment._reset_controller()
        self.assertEqual(previous['position'], 'tree')
        self.assertIsNone(marker.entry)

    def test_near_apple_with_wrong_direction_is_not_an_accurate_capture(self):
        from autopicker_dp.environment import capture_errors
        pose = np.eye(4)
        error, angle = capture_errors(pose, [.001, 0, 0], [0, -1, 0])
        self.assertAlmostEqual(error, .001)
        self.assertAlmostEqual(angle, 0.)
        error, angle = capture_errors(pose, [.001, 0, 0], [0, 0, 1])
        self.assertAlmostEqual(angle, 90.)


if __name__ == '__main__':
    unittest.main()
