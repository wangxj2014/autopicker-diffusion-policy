from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopicker_dp.hybrid import CORRIDOR_CONFIG
from test_m6_audit import audit_fixture


class M7AuditTests(unittest.TestCase):
    def module(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('validate_m7'), 'M7 audit is missing')
        import validate_m7
        return validate_m7

    def geometry(self):
        episode, arrays = audit_fixture()
        arrays['tip_world_m'][:] = [0., .2, 0.]
        arrays['tip_world_m'][1] = [0., .1, 0.]
        arrays['q'][:, :3] = arrays['tip_world_m']
        episode.metadata['hybrid_configuration'] = dict(CORRIDOR_CONFIG)
        episode.events[0].update(reason='tip_entered_approach_corridor', entry_distance_m=.1,
                                entry_angle_degrees=0., entry_axial_m=.1, entry_lateral_m=0.)
        def forward(q):
            pose = np.eye(4)
            pose[:3, 3] = q[:3]
            pose[:3, 1] = [-np.sin(q[3]), np.cos(q[3]), 0.]
            return pose
        return episode, arrays, SimpleNamespace(forward=forward), np.array([0., -1., 0.])

    def test_gate_is_recomputed_from_raw_geometry_not_event_claims(self):
        audit = self.module()
        episode, arrays, chain, direction = self.geometry()
        mask, reasons = audit.entry_evidence(episode, chain, direction)
        self.assertEqual(np.flatnonzero(mask).tolist(), [1])
        self.assertEqual(reasons[1], 'tip_entered_approach_corridor')
        for change in (
            lambda e, a: e.events[0].update(entry_angle_degrees=4.),
            lambda e, a: e.events[0].update(entry_angle_degrees=float('nan')),
            lambda e, a: e.events[0].update(entry_axial_m=float('nan')),
            lambda e, a: e.events[0].update(entry_lateral_m=.01),
            lambda e, a: a['q'].__setitem__((1, 3), .2),
            lambda e, a: a['tip_world_m'].__setitem__((1, 0), .03),
            lambda e, a: a['tip_world_m'].__setitem__((0, 1), .1),
            lambda e, a: e.events[0].update(time_s=.2),
        ):
            e, a, c, d = self.geometry()
            change(e, a)
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.entry_evidence(e, c, d)

    def test_seed_audit_rejects_global_coupling_or_forged_identity(self):
        audit = self.module()
        from autopicker_dp.inference import episode_noise_seed
        metadata = {'rollout_seed': 0, 'source_episode_index': 4, 'seed_mode': 'episode', 'generation': 7,
                    'inference_requests': [dict(request_id=23, generation=7, time_s=0.,
                     source_episode_index=4, episode_request_index=0, noise_seed=episode_noise_seed(0,4,0))]}
        metadata.update(inference_request_start=22, inference_request_end=23,
                        inference_attempts=deepcopy(metadata['inference_requests']))
        self.assertEqual(audit.seed_audit(metadata, 22), 23)
        for key in ('noise_seed', 'source_episode_index', 'episode_request_index', 'request_id', 'generation'):
            altered = deepcopy(metadata)
            altered['inference_requests'][0][key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.seed_audit(altered, 22)

    def test_failed_request_span_is_retained_before_next_episode(self):
        audit = self.module()
        from autopicker_dp.inference import episode_noise_seed
        meta = {'rollout_seed': 0, 'source_episode_index': 4, 'seed_mode': 'episode', 'generation': 7,
                'inference_requests': [], 'inference_request_start': 0, 'inference_request_end': 1,
                'inference_attempts': [dict(request_id=1, generation=7, time_s=0., source_episode_index=4,
                 episode_request_index=0, noise_seed=episode_noise_seed(0,4,0))]}
        # A failed first request consumes ID 1 despite having no applied action.
        self.assertEqual(audit.seed_audit(meta, 0, failed=True), 1)
        with self.assertRaises(ValueError):
            audit.seed_audit(meta, 0)
        meta.update(source_episode_index=5, generation=8, inference_request_start=1, inference_request_end=2)
        meta['inference_attempts'] = [dict(request_id=2, generation=8, time_s=0., source_episode_index=5,
                episode_request_index=0, noise_seed=episode_noise_seed(0,5,0))]
        meta['inference_requests'] = deepcopy(meta['inference_attempts'])
        self.assertEqual(audit.seed_audit(meta, 1), 2)
        meta['inference_request_end'] = 3
        with self.assertRaises(ValueError):
            audit.seed_audit(meta, 1)

    def test_paired_prefix_includes_all_raw_states_and_inference_content(self):
        audit = self.module()
        base, a = audit_fixture()
        candidate, b = audit_fixture()
        # The first command at handoff may differ; everything before it must match.
        base.events = [dict(kind='controller_stage', time_s=.3)]
        candidate.events = [dict(kind='controller_stage', time_s=.1)]
        for e in (base, candidate):
            e.metadata['inference_requests'] = [dict(time_s=0., generation=1, source_episode_index=4,
                episode_request_index=0, noise_seed=123, observation=[[0.]*15]*2, action=[[0.]*4]*2)]
        candidate.metadata['inference_requests'][0]['request_id'] = 50
        result = audit.paired_prefix(base, candidate)
        self.assertEqual(result['raw_rows'], 1)
        self.assertEqual(result['request_count'], 1)
        b['command'][0, 0] = .01
        with self.assertRaises(ValueError):
            audit.paired_prefix(base, candidate)
        b['command'][0, 0] = 0.
        candidate.metadata['inference_requests'][0]['action'][0][0] = .1
        with self.assertRaises(ValueError):
            audit.paired_prefix(base, candidate)


if __name__ == '__main__':
    unittest.main()
