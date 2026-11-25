from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopicker_dp.episode import Episode


def fixture():
    target = np.array([0., 0., 0.])
    pose = np.eye(4)
    pose[:3, 3] = [.04, 0., 0.]
    solved = np.eye(4)
    chain = SimpleNamespace(lower=np.full(4, -1.), upper=np.ones(4),
        inverse_approach=lambda *args: np.array([.1, .2, .3, .4]), forward=lambda q: solved)
    env = SimpleNamespace(generation=1, time=0., phase='reach', success=False,
        arm={'chain': chain, 'attached': False, 'release_goal': np.array([-.2, -.3, -.4, -.5])},
        executor=SimpleNamespace(position=np.zeros(4)),
        demo=SimpleNamespace(velocity=np.ones(4), acceleration=np.ones(4), config={'speed_scale': 1.}),
        entry={'world': target, 'path': 'apple'}, tip_pose=lambda: pose,
        positions=lambda: np.array([.01, .12, .03, .04]), direction=np.array([0., -1., 0.]),
        episode=Episode({'target_path': 'apple'}))
    calls = []
    policy = SimpleNamespace(requests=[], command=lambda env: calls.append(env.time) or np.full(4, .8))
    return env, pose, solved, policy, calls


class CorridorTests(unittest.TestCase):
    def provider(self):
        from autopicker_dp import hybrid
        self.assertTrue(hasattr(hybrid, 'CorridorHybridPolicy'), 'Corridor handoff is missing')
        env, pose, solved, policy, calls = fixture()
        pose[:3, 3] = [0., .1, 0.]
        return hybrid.CorridorHybridPolicy(policy, env), env, pose, calls

    def test_aligned_entry_latches_and_keeps_real_capture_gate(self):
        provider, env, pose, calls = self.provider()
        np.testing.assert_array_equal(provider.command(env), env.executor.position)
        self.assertEqual(provider.phase, 'precision')
        self.assertEqual(env.episode.events[0]['reason'], 'tip_entered_approach_corridor')
        self.assertAlmostEqual(env.episode.events[0]['entry_axial_m'], .1)
        env.time = 10.
        pose[:3, 3] = [.08, .1, 0.]
        provider.command(env)
        self.assertEqual(provider.phase, 'precision')
        self.assertEqual(calls, [])
        env.arm['attached'], env.phase = True, 'return'
        env.episode.event('capture', env.time, error_m=.002, angle_error_degrees=.5)
        provider.command(env)
        self.assertEqual(provider.phase, 'return')

    def test_off_axis_wrong_angle_behind_and_far_remain_learned(self):
        for position, angle in (([.021, .1, 0], 0), ([0, .1, 0], 5.1),
                                ([0, -.1, 0], 0), ([0, .121, 0], 0)):
            provider, env, pose, calls = self.provider()
            pose[:3, 3] = position
            radians = np.deg2rad(angle)
            pose[:3, 1] = [-np.sin(radians), np.cos(radians), 0.]
            np.testing.assert_array_equal(provider.command(env), np.full(4, .8))
            self.assertEqual(provider.phase, 'policy')
            self.assertEqual(len(calls), 1)

    def test_original_near_sphere_remains_a_fallback(self):
        provider, env, pose, calls = self.provider()
        pose[:3, 3] = [.03, 0., 0.]
        provider.command(env)
        self.assertEqual(provider.phase, 'precision')
        self.assertEqual(env.episode.events[0]['reason'], 'tip_entered_precision_region')
        self.assertEqual(calls, [])


class HybridTests(unittest.TestCase):
    def provider(self):
        from autopicker_dp.hybrid import HybridPolicy
        env, pose, solved, policy, calls = fixture()
        return HybridPolicy(policy, env), env, pose, solved, calls

    def test_precision_latches_and_holds_until_measured_capture(self):
        provider, env, pose, _, calls = self.provider()
        np.testing.assert_array_equal(provider.command(env), np.full(4, .8))
        env.time = 1.
        pose[0, 3] = .03
        np.testing.assert_array_equal(provider.command(env), env.executor.position)
        self.assertEqual(provider.phase, 'precision')
        # Even after the motion duration and a transient outward drift, neither
        # the old learned retreat nor the return trajectory may resume.
        env.time = 10.
        pose[0, 3] = .05
        np.testing.assert_allclose(provider.command(env), [.1, .2, .3, .4])
        self.assertEqual(calls, [0.])
        self.assertEqual([e['stage'] for e in env.episode.events], ['precision'])

    def test_capture_switches_immediately_from_current_command_and_preserves_hor(self):
        provider, env, pose, _, calls = self.provider()
        pose[0, 3] = .02
        provider.command(env)
        env.time = 1.0375  # Deliberately not a policy replan boundary.
        env.executor.position = np.array([.08, .13, .25, .32])
        env.arm['attached'], env.phase = True, 'return'
        env.episode.event('capture', env.time, error_m=.002, angle_error_degrees=.5)
        np.testing.assert_array_equal(provider.command(env), env.executor.position)
        self.assertEqual(provider.phase, 'return')
        env.time += 10.
        np.testing.assert_allclose(provider.command(env), [-.2, .12, -.4, -.5])
        self.assertEqual(calls, [])
        stages = [e for e in env.episode.events if e['kind'] == 'controller_stage']
        self.assertEqual([e['stage'] for e in stages], ['precision', 'return'])
        self.assertEqual(stages[1]['time_s'], 1.0375)

    def test_attachment_without_current_capture_cannot_authorize_return(self):
        for event_time in (None, -1.):
            provider, env, *_ = self.provider()
            env.arm['attached'], env.phase = True, 'return'
            if event_time is not None:
                env.episode.event('capture', event_time, error_m=.002, angle_error_degrees=.5)
            with self.assertRaisesRegex(RuntimeError, 'capture event'):
                provider.command(env)

    def test_attachment_loss_and_reset_generation_fail(self):
        provider, env, *_ = self.provider()
        env.arm['attached'], env.phase = True, 'return'
        env.episode.event('capture', env.time, error_m=.002, angle_error_degrees=.5)
        provider.command(env)
        env.arm['attached'] = False
        with self.assertRaisesRegex(RuntimeError, 'attachment'):
            provider.command(env)
        env.generation += 1
        with self.assertRaisesRegex(RuntimeError, 'generation'):
            provider.command(env)

    def test_invalid_or_inaccurate_ik_fails_without_issuing_a_retreat(self):
        for invalid in ('nan', 'limit', 'residual'):
            provider, env, pose, solved, calls = self.provider()
            pose[0, 3] = .02
            if invalid == 'nan':
                env.arm['chain'].inverse_approach = lambda *args: np.full(4, np.nan)
            elif invalid == 'limit':
                env.arm['chain'].inverse_approach = lambda *args: np.full(4, 2.)
            else:
                solved[0, 3] = .005
            with self.assertRaises((ValueError, RuntimeError)):
                provider.command(env)
            self.assertEqual(calls, [])
            self.assertEqual(env.episode.events, [])


if __name__ == '__main__':
    unittest.main()
