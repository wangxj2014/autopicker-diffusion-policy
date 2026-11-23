import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class RecoveryTests(unittest.TestCase):
    def test_rejected_trial_is_persisted_and_next_trial_remains_runnable(self):
        from autopicker_dp.recovery import collect_recovery_trial
        from autopicker_dp.episode import Episode
        from unittest.mock import Mock
        metadata = {'target_path': '/train-apple', 'physics_dt': .1,
                    'joint_lower': [-1.]*4, 'joint_upper': [1.]*4}
        source = Episode(metadata)
        for step in range(21):
            source.append_state(step * .1, np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4),
                                np.zeros(3), np.zeros(3), np.zeros(3), step >= 15, np.zeros(16))
        source.event('capture', 1.5, error_m=.001, angle_error_degrees=.1)
        source.event('collect', 2.)
        source.finish(True, 'collected')
        env = SimpleNamespace(arm={'indices': [0, 1, 2, 3]}, reset=Mock(), positions=lambda: np.full(4, .1))
        runner = Mock(return_value={'success': True, 'reason': 'collected'})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source.save(root / 'source')
            spec = {'path': 'source', 'episode_json_sha256': 'fixture', 'raw_sha256': {},
                    'metadata': {'target_path': '/train-apple'}}
            failed = collect_recovery_trial(env, root, spec, 1, 1, np.full(4, 2.), 'dataset',
                                           root / 'failed', .1, 35., runner=runner)
            self.assertFalse(failed['success'])
            env.reset.assert_not_called()
            runner.assert_not_called()
            failure = json.loads((root / 'failed/failure.json').read_text())
            self.assertEqual(failure['target_path'], '/train-apple')
            self.assertEqual(failure['recovery']['delta_q'], [2.]*4)
            self.assertIn('joint limits', failure['reason'])
            succeeded = collect_recovery_trial(env, root, spec, 1, 1, np.full(4, .1), 'dataset',
                                              root / 'next', .1, 35., runner=runner)
            self.assertTrue(succeeded['success'])
            self.assertEqual(runner.call_count, 1)
            def invalid_recording(environment, directory, *args, **kwargs):
                directory.mkdir()
                (directory / 'failure.json').write_text(json.dumps({'reason': 'Nonfinite state',
                    'audit_error': 'Incomplete episode', 'metadata': {
                        'target_path': '/train-apple', 'recovery': kwargs['metadata_extra']['recovery']}}))
                raise RuntimeError('Episode failed data audit: Nonfinite state')
            invalid = collect_recovery_trial(env, root, spec, 1, 1, np.full(4, .1), 'dataset',
                                            root / 'invalid', .1, 35., runner=invalid_recording)
            self.assertEqual(invalid['reason'], 'Nonfinite state')
            self.assertEqual(invalid['audit_error'], 'Incomplete episode')

    def test_selects_one_source_per_train_target_without_holdout_leakage(self):
        from autopicker_dp.recovery import recovery_sources
        episodes = [{'split': split, 'metadata': {'scene_sha256': 'scene', 'target_path': target}}
                    for target, split in [('a', 'val'), ('b', 'train'), ('b', 'train'),
                                          ('c', 'test'), ('d', 'train')]]
        self.assertEqual(recovery_sources({'episodes': episodes}), [1, 4])

    def test_reset_uses_unattached_source_and_changes_only_active_joints(self):
        from autopicker_dp.recovery import recovery_reset
        times = np.arange(0., 2.01, .1)
        q = np.tile([-.6, .24, -.36, 1.13], (len(times), 1))
        all_q = np.zeros((len(times), 16))
        indices = [2, 6, 10, 14]
        all_q[:, indices] = q
        arrays = {'time_s': times, 'q': q, 'q_all': all_q, 'attached': times >= 1.5}
        source = SimpleNamespace(events=[{'kind': 'capture', 'time_s': 1.5}],
                                 arrays=lambda: arrays, outcome={'success': True})
        lower = [-1.22, 0., -.54, -1.91]
        upper = [.26, 1.2, 0., 1.57]
        delta = np.array([.05, .025, .02, .08])
        reset, provenance = recovery_reset(source, indices, lower, upper, delta)
        np.testing.assert_allclose(reset[indices], q[10] + delta)
        np.testing.assert_array_equal(np.delete(reset, indices), np.zeros(12))
        self.assertAlmostEqual(provenance['source_time_s'], 1.)
        self.assertEqual(provenance['kind'], 'perturbed_pre_capture_reset')
        self.assertEqual(provenance['reset_velocity'], 'zero_then_settle')
        json.dumps(provenance, allow_nan=False)
        reset2, info2 = recovery_reset(source, indices, lower, upper, delta)
        np.testing.assert_array_equal(reset2, reset)
        self.assertEqual(info2, provenance)

    def test_out_of_limits_attached_or_mismatched_source_is_rejected(self):
        from autopicker_dp.recovery import recovery_reset
        arrays = {'time_s': np.array([0., 1., 2.]), 'q': np.zeros((3, 4)),
                  'q_all': np.zeros((3, 16)), 'attached': np.zeros(3, bool)}
        source = SimpleNamespace(events=[{'kind': 'capture', 'time_s': 1.5}],
                                 arrays=lambda: arrays, outcome={'success': True})
        for delta in ([2, 0, 0, 0], [float('nan'), 0, 0, 0]):
            with self.assertRaises(ValueError):
                recovery_reset(source, [0, 1, 2, 3], [-1]*4, [1]*4, delta)
        arrays['attached'][1] = True
        with self.assertRaises(ValueError):
            recovery_reset(source, [0, 1, 2, 3], [-1]*4, [1]*4, [.1]*4)
        arrays['attached'][1] = False
        arrays['q_all'][1, 1] = .3
        with self.assertRaises(ValueError):
            recovery_reset(source, [0, 1, 2, 3], [-1]*4, [1]*4, [.1]*4)


if __name__ == '__main__':
    unittest.main()
