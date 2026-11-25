from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopicker_dp.hybrid import HYBRID_CONFIG


def audit_fixture():
    times = np.arange(7) * .1
    positions = np.zeros((7, 4))
    arrays = {'time_s': times, 'q': positions, 'command': positions.copy(),
        'velocity_command': positions.copy(), 'attached': np.array([0, 0, 0, 1, 1, 1, 0], dtype=bool),
        'tip_world_m': np.array([[.04, 0, 0], [.03, 0, 0], [.01, 0, 0], [.002, 0, 0],
                                 [.01, 0, 0], [.03, 0, 0], [.04, 0, 0]]),
        'goal_world_m': np.zeros((7, 3))}
    arrays['receiver_world_m'] = np.tile([.04, 0., -.12], (7, 1))
    stages = [{'kind': 'controller_stage', 'time_s': .1, 'target_path': 'apple',
        'stage': 'precision', 'reason': 'tip_entered_precision_region', 'entry_distance_m': .03,
        'entry_angle_degrees': .5, 'start_command': [0.] * 4, 'goal_command': [0.] * 4, 'duration_s': .8},
        {'kind': 'capture', 'time_s': .3, 'target_path': 'apple', 'error_m': .002, 'angle_error_degrees': .5},
        {'kind': 'controller_stage', 'time_s': .3, 'target_path': 'apple', 'stage': 'return',
         'reason': 'capture_event', 'start_command': [0.] * 4, 'goal_command': [0.] * 4, 'duration_s': .8},
        {'kind': 'collect', 'time_s': .6, 'target_path': 'apple'}]
    metadata = {'hybrid_configuration': dict(HYBRID_CONFIG), 'physics_dt': .1,
        'joint_lower': [-1.] * 4, 'joint_upper': [1.] * 4, 'speed_scale': 1., 'target_path': 'apple',
        'inference_requests': [{'time_s': 0., 'request_id': 1}]}
    return SimpleNamespace(metadata=metadata, events=stages, arrays=lambda: arrays,
                           outcome={'success': True}), arrays


class M6AuditTests(unittest.TestCase):
    def geometry_fixture(self):
        episode, arrays = audit_fixture()
        arrays['q'][:, :3] = arrays['tip_world_m']
        episode.events[0]['entry_angle_degrees'] = 0.
        episode.events[1]['angle_error_degrees'] = 0.
        episode.events[2]['goal_command'] = [.04, 0., 0., 0.]
        def forward(q):
            pose = np.eye(4)
            pose[:3, 3] = q[:3]
            pose[:3, 1] = [-np.sin(q[3]), np.cos(q[3]), 0.]
            return pose
        return episode, arrays, SimpleNamespace(forward=forward), np.array([0., -1., 0.])

    def test_kinematics_audit_recomputes_pose_and_terminal_goals(self):
        from validate_m6 import kinematics_audit
        episode, _, chain, direction = self.geometry_fixture()
        result = kinematics_audit(episode, chain, direction)
        self.assertEqual(result['max_fk_measured_tip_error_m'], 0.)
        self.assertEqual(result['capture_pose']['angle_degrees'], 0.)
        for change in (
            lambda e, a: a['q'].__setitem__((3, 3), .1),
            lambda e, a: a['q'].__setitem__((2, 0), .1),
            lambda e, a: e.events[0]['goal_command'].__setitem__(0, .01),
            lambda e, a: e.events[2]['goal_command'].__setitem__(0, .1),
            lambda e, a: e.events[1].update(angle_error_degrees=.5),
        ):
            episode, arrays, chain, direction = self.geometry_fixture()
            change(episode, arrays)
            with self.subTest(change=change), self.assertRaises(ValueError):
                kinematics_audit(episode, chain, direction)

    def test_stage_and_command_audit_accepts_measured_transitions(self):
        from validate_m6 import stage_audit
        episode, _ = audit_fixture()
        result = stage_audit(episode)
        self.assertAlmostEqual(result['precision_duration_s'], .2)
        self.assertEqual(result['max_applied_command_error'], 0.)

    def test_stage_audit_rejects_ungated_or_unexecuted_transitions(self):
        from validate_m6 import stage_audit
        for change in (
            lambda e, a: e.events[2].update(time_s=.2),
            lambda e, a: e.events[1].update(error_m=.004),
            lambda e, a: e.events[1].update(angle_error_degrees=float('nan')),
            lambda e, a: e.events[0].update(entry_distance_m=.02),
            lambda e, a: e.events[0].update(duration_s=2.),
            lambda e, a: a['attached'].__setitem__(2, True),
            lambda e, a: a['receiver_world_m'].__setitem__((6, 0), 1.),
            lambda e, a: a['command'].__setitem__((2, 0), .01),
            lambda e, a: e.events[2]['goal_command'].__setitem__(1, .02),
            lambda e, a: e.metadata['inference_requests'].append({'time_s': .2, 'request_id': 2}),
            lambda e, a: e.metadata['hybrid_configuration'].update(precision_entry_distance_m=.2),
        ):
            episode, arrays = audit_fixture()
            change(episode, arrays)
            with self.subTest(change=change), self.assertRaises(ValueError):
                stage_audit(episode)

    def test_no_entry_timeout_is_a_valid_failure_not_a_dropped_trial(self):
        from validate_m6 import stage_audit
        episode, arrays = audit_fixture()
        episode.events = [{'kind': 'failure', 'time_s': .6}]
        episode.outcome = {'success': False}
        arrays['attached'][:] = False
        arrays['tip_world_m'][:, 0] = .04
        episode.metadata['inference_requests'] = [{'time_s': t, 'request_id': i + 1}
            for i, t in enumerate((0., .2, .4))]
        result = stage_audit(episode)
        self.assertIsNone(result['precision_entry_s'])
        self.assertEqual(result['stages'], [])
        episode.metadata['inference_requests'].pop()
        with self.assertRaises(ValueError):
            stage_audit(episode)


if __name__ == '__main__':
    unittest.main()
