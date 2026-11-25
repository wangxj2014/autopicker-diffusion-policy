"""Explicit oracle-IK terminal control; results are not pure learned-policy results."""
import numpy as np

from .control import vector
from .environment import capture_errors
from autopicker_apple_pick import motion_duration, quintic_blend


HYBRID_CONFIG = {
    'kind': 'event_gated_hybrid_v1',
    'precision_entry_distance_m': .03,
    'approach_angle_degrees': 30.,
    'minimum_motion_duration_s': .8,
    'precision_controller': 'latched_oracle_ik_quintic',
    'return_controller': 'capture_event_release_goal_preserve_hor',
    'learned_control': 'coarse_approach_only',
}

CORRIDOR_CONFIG = {
    **HYBRID_CONFIG,
    'kind': 'event_gated_hybrid_corridor_v1',
    'corridor_entry_distance_m': .12,
    'corridor_lateral_error_m': .02,
    'corridor_angle_error_degrees': 5.,
    'corridor_minimum_axial_m': 0.,
}


class HybridPolicy:
    def __init__(self, policy, env):
        self.policy = policy
        self.requests = policy.requests
        self.generation = env.generation
        self.phase = 'policy'

    def inference_audit(self):
        return self.policy.inference_audit()

    def _begin(self, env, stage, goal, reason, **details):
        goal = vector(goal)
        chain = env.arm['chain']
        if np.any(goal < chain.lower) or np.any(goal > chain.upper):
            raise ValueError('Hybrid goal outside joint limits')
        start = vector(env.executor.position)
        duration = motion_duration(start, goal, env.demo.velocity, env.demo.acceleration,
                                   HYBRID_CONFIG['minimum_motion_duration_s'] / env.demo.config['speed_scale'])
        self.start, self.goal, self.started, self.duration = start, goal, env.time, duration
        self.phase = stage
        env.episode.event('controller_stage', env.time, stage=stage, reason=reason,
                          start_command=start.tolist(), goal_command=goal.tolist(),
                          duration_s=duration, **details)

    def _precision_entry(self, env):
        distance, angle = capture_errors(env.tip_pose(), env.entry['world'], env.direction)
        if not np.isfinite([distance, angle]).all():
            raise ValueError('Invalid precision entry pose')
        if distance <= HYBRID_CONFIG['precision_entry_distance_m']:
            return dict(reason='tip_entered_precision_region',
                        entry_distance_m=distance, entry_angle_degrees=angle)
        return None

    def command(self, env):
        if env.generation != self.generation:
            raise RuntimeError('Hybrid controller generation differs from environment reset')
        if env.success:
            return env.executor.position.copy()
        if env.arm['attached'] and self.phase != 'return':
            captures = [event for event in env.episode.events if event['kind'] == 'capture']
            if (env.phase != 'return' or len(captures) != 1
                    or captures[0]['target_path'] != env.entry['path']
                    or abs(captures[0]['time_s'] - env.time) > 1e-9):
                raise RuntimeError('Return requires a current measured capture event')
            goal = vector(env.arm['release_goal'])
            goal[1] = env.positions()[1]
            self._begin(env, 'return', goal, 'capture_event')
        elif not env.arm['attached'] and self.phase == 'return':
            raise RuntimeError('Hybrid return lost attachment before collect')
        elif self.phase == 'policy':
            entry = self._precision_entry(env)
            if entry is not None:
                chain = env.arm['chain']
                goal = vector(chain.inverse_approach(env.entry['world'], env.positions(),
                                                     HYBRID_CONFIG['approach_angle_degrees']))
                residual = capture_errors(chain.forward(goal), env.entry['world'], env.direction)
                if not np.isfinite(residual).all() or residual[0] > 1e-5 or residual[1] > .01:
                    raise ValueError('Precision IK did not satisfy the grasp pose')
                self._begin(env, 'precision', goal, **entry)
        if self.phase == 'policy':
            return self.policy.command(env)
        blend = quintic_blend((env.time - self.started) / self.duration)
        return self.start + blend * (self.goal - self.start)


class CorridorHybridPolicy(HybridPolicy):
    """Earlier handoff only when the measured pose lies in the approach corridor."""

    def _precision_entry(self, env):
        near = super()._precision_entry(env)
        if near is not None:
            return near
        pose = env.tip_pose()
        distance, angle = capture_errors(pose, env.entry['world'], env.direction)
        delta = np.asarray(env.entry['world']) - pose[:3, 3]
        axial = float(delta @ env.direction)
        lateral = float(np.linalg.norm(delta - axial * env.direction))
        if (distance <= CORRIDOR_CONFIG['corridor_entry_distance_m']
                and axial >= CORRIDOR_CONFIG['corridor_minimum_axial_m']
                and lateral <= CORRIDOR_CONFIG['corridor_lateral_error_m']
                and angle <= CORRIDOR_CONFIG['corridor_angle_error_degrees']):
            return dict(reason='tip_entered_approach_corridor', entry_distance_m=distance,
                        entry_angle_degrees=angle, entry_axial_m=axial, entry_lateral_m=lateral)
        return None
