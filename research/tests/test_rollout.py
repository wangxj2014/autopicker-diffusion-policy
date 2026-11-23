from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class RolloutTests(unittest.TestCase):
    def test_converted_replay_uses_next_node_labels_and_holds_last_node(self):
        from autopicker_dp.rollout import ConvertedReplay
        data = {'action_time_s': np.array([.1, .2]),
                'action': np.array([[1., 2., 3., 4.], [2., 4., 6., 8.]])}
        provider = ConvertedReplay(data, np.zeros(4))
        np.testing.assert_allclose(provider.command(SimpleNamespace(time=.05)), [.5, 1., 1.5, 2.])
        np.testing.assert_allclose(provider.command(SimpleNamespace(time=.2)), [2., 4., 6., 8.])
        np.testing.assert_allclose(provider.command(SimpleNamespace(time=1.)), [2., 4., 6., 8.])
        with self.assertRaisesRegex(RuntimeError, 'exhausted'):
            provider.command(SimpleNamespace(time=3.3))

    def test_live_observation_equals_raw_conversion_in_base_frame(self):
        from autopicker_dp.rollout import live_observation
        from autopicker_dp.dataset import _observation, _base_inverse
        transform = np.array([[0., 1., 0., 0.], [-1., 0., 0., 0.],
                              [0., 0., 1., 0.], [2., 3., 4., 1.]])
        inverse = _base_inverse({'world_from_base': transform,
                                 'transform_convention': 'USD row vectors, meters'})
        q = np.arange(4.)
        velocity = np.arange(16.)
        robot = SimpleNamespace(get_joint_velocities=lambda: velocity[None])
        env = SimpleNamespace(positions=lambda: q, demo=SimpleNamespace(robot=robot),
                              arm={'indices': [0, 1, 2, 3], 'attached': True},
                              entry={'world': np.array([3., 5., 7.])},
                              receiver=lambda: np.array([4., 6., 8.]))
        raw = {'q': q[None], 'qdot': velocity[None, :4],
               'goal_world_m': env.entry['world'][None], 'receiver_world_m': env.receiver()[None],
               'attached': np.array([True])}
        np.testing.assert_array_equal(live_observation(env, inverse), _observation(raw, [0], inverse)[0])


if __name__ == '__main__':
    unittest.main()
