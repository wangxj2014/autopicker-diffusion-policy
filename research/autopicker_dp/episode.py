"""Versioned raw episodes with explicit timing and task outcomes."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from .control import vector

SCHEMA = 'autopicker-raw-v1'
STATE_KEYS = ('time_s', 'q', 'qdot', 'command', 'velocity_command', 'tip_world_m',
              'goal_world_m', 'receiver_world_m', 'attached', 'q_all')


def validate_frame(frame, sim_time, physics_dt, previous_reference):
    reference = tuple(frame['reference_time'])
    if len(reference) != 2 or reference[1] <= 0:
        raise ValueError('Invalid render time')
    timestamp = Fraction(*reference)
    if previous_reference is not None and timestamp <= Fraction(*previous_reference):
        raise ValueError('Stale or nonmonotonic RGB-D frame')
    exposure_time = float(timestamp)
    if frame.get('clock_mapping') == 'lockstep_render_barrier':
        exposure_time = float(frame['exposure_sim_time_s'])
        if (not np.isfinite(frame['max_joint_change']) or frame['max_joint_change'] > 1e-8
                or abs(float(frame['physics_time_after_s']) - exposure_time) > 1e-8):
            raise ValueError('Physics changed during the render barrier')
    error = exposure_time - sim_time
    if not np.isfinite(error) or abs(error) > physics_dt + 1e-7:
        raise ValueError(f'Render/state time mismatch: {error:.6f}s')
    rgb, depth = frame['rgb'], frame['depth_m']
    if rgb.dtype != np.uint8 or rgb.shape != (*depth.shape, 3) or depth.ndim != 2:
        raise ValueError('RGB and depth shapes or types disagree')
    for name, shape in (('intrinsics', (3, 3)), ('world_from_camera', (4, 4))):
        if np.shape(frame[name]) != shape or not np.isfinite(frame[name]).all():
            raise ValueError(f'Invalid {name}')
    return error


class Episode:
    def __init__(self, metadata):
        self.metadata = dict(metadata)
        self.metadata['schema'] = SCHEMA
        self.states = []
        self.frames = []
        self.events = []
        self.outcome = None

    def append_state(self, time_s, q, qdot, command, velocity_command,
                     tip_world_m, goal_world_m, receiver_world_m, attached, q_all):
        if self.outcome is not None:
            raise ValueError('Episode is already finished')
        if not np.isfinite(time_s) or time_s < 0 or (self.states and time_s <= self.states[-1][0]):
            raise ValueError('State times must increase')
        row = (float(time_s), vector(q), vector(qdot), vector(command), vector(velocity_command),
               vector(tip_world_m, 3), vector(goal_world_m, 3), vector(receiver_world_m, 3),
               bool(attached), vector(q_all, 16))
        self.states.append(row)

    def append_frame(self, frame, sim_time_s, episode_time_s, physics_step, camera_pose_error_m):
        previous = self.frames[-1]['reference_time'] if self.frames else None
        error = validate_frame(frame, sim_time_s, self.metadata['physics_dt'], previous)
        if camera_pose_error_m > .001 or not np.isfinite(camera_pose_error_m):
            raise ValueError('Rendered camera pose does not match current physics pose')
        depth = np.asarray(frame['depth_m'], dtype=np.float32)
        valid = np.isfinite(depth) & (depth >= .1) & (depth <= 5.)
        self.frames.append({'rgb': frame['rgb'].copy(), 'depth_m': np.where(valid, depth, 0),
                            'depth_valid': valid, 'intrinsics': frame['intrinsics'].copy(),
                            'world_from_camera': frame['world_from_camera'].copy(),
                            'reference_time': np.array(frame['reference_time'], dtype=np.int64),
                            'capture_id': frame['capture_id'], 'sim_time_s': sim_time_s,
                            'episode_time_s': episode_time_s, 'physics_step': physics_step,
                            'sync_error_s': error, 'camera_pose_error_m': camera_pose_error_m,
                            'clock_mapping': frame.get('clock_mapping', 'render_time'),
                            'exposure_sim_time_s': frame.get('exposure_sim_time_s', float(Fraction(*frame['reference_time']))),
                            'physics_time_after_s': frame.get('physics_time_after_s', sim_time_s),
                            'max_joint_change': frame.get('max_joint_change', 0.)})

    def event(self, kind, time_s, **details):
        self.events.append({'kind': kind, 'time_s': float(time_s),
                            'target_path': self.metadata['target_path'], **details})

    def finish(self, success, reason):
        outcome = {'success': bool(success), 'reason': str(reason)}
        self._check_success(outcome)
        self.outcome = outcome

    def _check_success(self, outcome):
        if not outcome['success']:
            return
        captures = [event for event in self.events if event['kind'] == 'capture']
        collects = [event for event in self.events if event['kind'] == 'collect']
        if (len(captures) != 1 or len(collects) != 1
                or captures[0]['time_s'] >= collects[0]['time_s']
                or any(event['target_path'] != self.metadata['target_path'] for event in captures + collects)
                or captures[0]['error_m'] > .003 or captures[0]['angle_error_degrees'] > 1.):
            raise ValueError('Success requires one accurate capture followed by matching collect')

    def arrays(self):
        return {key: np.asarray([row[index] for row in self.states])
                for index, key in enumerate(STATE_KEYS)}

    def audit(self):
        if self.outcome is None or len(self.states) < 2:
            raise ValueError('Incomplete episode')
        self._check_success(self.outcome)
        arrays = self.arrays()
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError('Nonfinite state data')
        if not np.allclose(np.diff(arrays['time_s']), self.metadata['physics_dt'], rtol=0, atol=1e-7):
            raise ValueError('Raw physics samples are missing or nonuniform')
        lower, upper = np.array(self.metadata['joint_lower']), np.array(self.metadata['joint_upper'])
        for name in ('q', 'command'):
            if np.any(arrays[name] < lower - .001) or np.any(arrays[name] > upper + .001):
                raise ValueError(f'{name} crosses joint limits')
        if self.metadata.get('capture_required') and not self.frames:
            raise ValueError('Required RGB-D observations are missing')
        previous_reference = None
        for frame in self.frames:
            validate_frame(frame, frame['sim_time_s'], self.metadata['physics_dt'], previous_reference)
            previous_reference = frame['reference_time']
            if frame['camera_pose_error_m'] > .001:
                raise ValueError('Camera pose mismatch')
        return {'success': self.outcome['success'], 'reason': self.outcome['reason'],
                'states': len(self.states), 'frames': len(self.frames),
                'duration_s': float(arrays['time_s'][-1]),
                'max_sync_error_s': max((abs(frame['sync_error_s']) for frame in self.frames), default=0.)}

    def save(self, path):
        report = self.audit()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(path / 'states.npz', **self.arrays())
        if self.frames:
            np.savez_compressed(path / 'rgbd.npz', **{
                key: np.asarray([frame[key] for frame in self.frames]) for key in self.frames[0]})
        checksums = {item.name: hashlib.sha256(item.read_bytes()).hexdigest() for item in path.glob('*.npz')}
        document = {'metadata': self.metadata, 'events': self.events, 'outcome': self.outcome,
                    'audit': report, 'sha256': checksums}
        (path / 'episode.json').write_text(json.dumps(document, indent=2, allow_nan=False), encoding='utf-8')

    @classmethod
    def load(cls, path):
        path = Path(path)
        document = json.loads((path / 'episode.json').read_text(encoding='utf-8'))
        if document['metadata']['schema'] != SCHEMA:
            raise ValueError('Unsupported episode schema')
        for name, expected in document['sha256'].items():
            if name not in ('states.npz', 'rgbd.npz') or hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
                raise ValueError('Episode checksum mismatch')
        episode = cls(document['metadata'])
        with np.load(path / 'states.npz', allow_pickle=False) as archive:
            lengths = {len(archive[key]) for key in STATE_KEYS}
            if len(lengths) != 1:
                raise ValueError('Inconsistent array lengths')
            for values in zip(*(archive[key] for key in STATE_KEYS)):
                episode.append_state(*values)
        if (path / 'rgbd.npz').exists():
            with np.load(path / 'rgbd.npz', allow_pickle=False) as archive:
                frames = {key: archive[key] for key in archive.files}
                if len({len(value) for value in frames.values()}) != 1:
                    raise ValueError('Inconsistent frame array lengths')
                episode.frames = [{key: value[index].copy() for key, value in frames.items()}
                                  for index in range(len(frames['rgb']))]
        episode.events = document['events']
        episode.outcome = document['outcome']
        episode.audit()
        return episode
