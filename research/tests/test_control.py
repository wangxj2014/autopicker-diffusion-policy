import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ControlTests(unittest.TestCase):
    def executor(self):
        from autopicker_dp.control import JointExecutor
        return JointExecutor([-1, 0, -.5, -1], [1, 1.2, 0, 1],
                             [.2, .2, .2, .2], [.4, .4, .4, .4], .01)

    def test_motion_obeys_position_velocity_and_acceleration_limits(self):
        executor = self.executor()
        executor.reset([0, .1, -.1, 0])
        previous_position = executor.position.copy()
        previous_velocity = np.zeros(4)
        for _ in range(500):
            position, velocity = executor.advance([1, 1, -.4, -1])
            self.assertTrue(np.all(position >= executor.lower))
            self.assertTrue(np.all(position <= executor.upper))
            np.testing.assert_allclose((position - previous_position) / .01, velocity, atol=1e-10)
            self.assertLessEqual(np.max(np.abs(velocity)), .2 + 1e-9)
            self.assertLessEqual(np.max(np.abs(velocity - previous_velocity) / .01), .4 + 1e-9)
            previous_position, previous_velocity = position, velocity

    def test_invalid_commands_rejected_without_mutation(self):
        executor = self.executor()
        executor.reset([0, .1, -.1, 0])
        for command in ([0, 2, 0, 0], [float('nan'), 0, 0, 0], [0, 0]):
            with self.assertRaises(ValueError):
                executor.advance(command)
        np.testing.assert_array_equal(executor.position, [0, .1, -.1, 0])

    def test_brakes_before_joint_boundary_and_can_reverse(self):
        executor = self.executor()
        executor.reset([0, .1, -.1, 0])
        previous_velocity = np.zeros(4)
        for target in ([1, 1.2, 0, 1], [-1, 0, -.5, -1]):
            for _ in range(1300):
                position, velocity = executor.advance(target)
                self.assertLessEqual(np.max(np.abs(velocity - previous_velocity) / .01), .4 + 1e-8)
                previous_velocity = velocity
            np.testing.assert_allclose(position, target, atol=1e-5)

    def test_node_interpolation_uses_command_not_future_measured_state(self):
        from autopicker_dp.control import CommandTrajectory
        trajectory = CommandTrajectory([0, .1, .2], [[0]*4, [.2]*4, [.4]*4])
        np.testing.assert_allclose(trajectory.at(.05), [.1]*4)
        np.testing.assert_allclose(trajectory.at(.2), [.4]*4)
        with self.assertRaises(ValueError):
            trajectory.at(.21)
        with self.assertRaises(ValueError):
            CommandTrajectory([0, .1, .1], np.zeros((3, 4)))


if __name__ == '__main__':
    unittest.main()
