from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopicker_dp.control import JointExecutor
from test_augmentation import recorded_episode


class DynamicStateTests(unittest.TestCase):
    def source(self):
        episode = recorded_episode('/apple')
        episode.metadata['mode'] = 'policy'
        episode.events = []
        episode.outcome = {'success': False, 'reason': 'timeout'}
        row = list(episode.states[19])
        row[3], row[4] = np.full(4, .11), np.full(4, .02)
        episode.states[19] = tuple(row)
        row = list(episode.states[20])
        row[2], row[3] = np.full(4, .03), np.full(4, .12)
        episode.states[20] = tuple(row)
        return episode

    def test_state_preserves_motion_and_uses_preceding_applied_command(self):
        from autopicker_dp.dynamic_recovery import state_from_policy
        state = state_from_policy(self.source(), .2, [0, 1, 2, 3])
        self.assertEqual(state['source_row'], 20)
        self.assertEqual(state['qdot'], [.03] * 4)
        self.assertEqual(state['executor_position'], [.11] * 4)
        self.assertEqual(state['executor_velocity'], [.02] * 4)
        for time in (0, .205, 2., float('nan')):
            with self.subTest(time=time), self.assertRaises(ValueError):
                state_from_policy(self.source(), time, [0, 1, 2, 3])

    def test_restore_does_not_settle_or_zero_active_velocity(self):
        from autopicker_dp.dynamic_recovery import state_from_policy, restore_dynamic_state
        state = state_from_policy(self.source(), .2, [0, 1, 2, 3])
        class Robot:
            def set_joint_positions(self, q): self.q = q.copy()
            def set_joint_velocities(self, v): self.v = v.copy()
            def set_joint_position_targets(self, q): self.target = q.copy()
            def set_joint_velocity_targets(self, v): self.velocity_target = v.copy()
            def get_joint_positions(self): return self.q.copy()
            def get_joint_velocities(self): return self.v.copy()
        robot = Robot()
        executor = JointExecutor([-2.]*4, [2.]*4, [1.]*4, [2.]*4, .01)
        env = SimpleNamespace(demo=SimpleNamespace(robot=robot), arm={'indices': np.arange(4)},
                              executor=executor, steps=0)
        audit = restore_dynamic_state(env, state)
        np.testing.assert_array_equal(robot.v[0, :4], [.03]*4)
        np.testing.assert_array_equal(robot.v[0, 4:], np.zeros(12))
        np.testing.assert_array_equal(executor.position, [.11]*4)
        np.testing.assert_array_equal(executor.velocity, [.02]*4)
        self.assertEqual(audit['max_active_velocity_error'], 0.)
        self.assertEqual(env.steps, 0)
        before = robot.q.copy()
        bad = deepcopy(state)
        bad['executor_velocity'][0] = 2.
        with self.assertRaises(ValueError):
            restore_dynamic_state(env, bad)
        np.testing.assert_array_equal(robot.q, before)
        for getter, value in (('get_joint_positions', np.full((1, 16), np.nan)),
                              ('get_joint_velocities', np.full((1, 16), np.nan))):
            original = getattr(robot, getter)
            setattr(robot, getter, lambda: value)
            with self.subTest(getter=getter), self.assertRaises(ValueError):
                restore_dynamic_state(env, state)
            setattr(robot, getter, original)

    @unittest.skipUnless(importlib.util.find_spec('zarr'), 'Zarr required')
    def test_dynamic_augmentation_binds_policy_identity_and_initial_velocity(self):
        from test_augmentation import augmentation_fixture
        from autopicker_dp.dataset import build_dataset, _sha256
        from autopicker_dp.dynamic_recovery import state_from_policy, validated_dynamic_state
        from autopicker_dp.augmentation import recovery_sequences
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent, _, digest, _, _ = augmentation_fixture(root)
            source = self.source()
            source.metadata.update(target_path='/apple_0', dataset_sha256=digest,
                                   source_episode_index=0, source_split='train', checkpoint_sha256='model')
            source.save(root / 'policy')
            state = state_from_policy(source, .2, [0, 1, 2, 3])
            specification = parent.manifest['episodes'][0]
            info = {'kind': 'policy_state_reset', 'source_dataset_sha256': digest,
                'source_episode_index': 0, 'source_split': 'train',
                'source_episode_json_sha256': specification['episode_json_sha256'],
                'source_raw_sha256': specification['raw_sha256'],
                'policy_episode_path': str(root / 'policy'),
                'policy_episode_json_sha256': _sha256(root / 'policy/episode.json'),
                'policy_raw_sha256': json.loads((root / 'policy/episode.json').read_text())['sha256'],
                'policy_checkpoint_sha256': 'model', 'state': state}
            corrected = recorded_episode('/apple_0')
            corrected.metadata.update(recovery=info, reset_joint_positions=state['q_all'])
            for key, value in (('source_split', 'val'), ('policy_checkpoint_sha256', 'other'),
                               ('policy_episode_json_sha256', 'changed'), ('source_episode_index', 1)):
                altered = {**info, key: value}
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'policy source identity'):
                    validated_dynamic_state(altered, corrected.metadata)
            row = list(corrected.states[0]); row[2] = np.array(state['qdot'])
            corrected.states[0] = tuple(row)
            corrected.save(root / 'dynamic_raw/episode_0000')
            build_dataset(root / 'dynamic_raw', root / 'dynamic', split_counts=(1, 0, 0))
            sequences, _ = recovery_sequences(parent, digest, root / 'dynamic')
            self.assertEqual(len(sequences), 30)
            wrong_velocity = deepcopy(corrected)
            row = list(wrong_velocity.states[0]); row[2] = np.zeros(4)
            wrong_velocity.states[0] = tuple(row)
            wrong_velocity.save(root / 'wrong_velocity_raw/episode_0000')
            build_dataset(root / 'wrong_velocity_raw', root / 'wrong_velocity', split_counts=(1, 0, 0))
            with self.assertRaisesRegex(ValueError, 'initial measured'):
                recovery_sequences(parent, digest, root / 'wrong_velocity')
            path = root / 'dynamic_raw/episode_0000/episode.json'
            document = json.loads(path.read_text())
            document['metadata']['recovery']['state']['qdot'][0] = .2
            path.write_text(json.dumps(document))
            build_dataset(root / 'dynamic_raw', root / 'forged', split_counts=(1, 0, 0))
            with self.assertRaisesRegex(ValueError, 'Dynamic recovery state'):
                recovery_sequences(parent, digest, root / 'forged')


if __name__ == '__main__':
    unittest.main()
