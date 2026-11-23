"""Verify the recorded/replayed M0 pilot using ordinary Python and NumPy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopicker_dp.episode import Episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recordings', type=Path)
    parser.add_argument('replays', type=Path)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--targets', type=int, default=5)
    options = parser.parse_args()
    groups = []
    for folder in (options.recordings, options.replays):
        summary = json.loads((folder / 'summary.json').read_text())
        assert summary['requested'] == summary['attempted'] == summary['successes'] == options.episodes
        paths = sorted(folder.glob('episode_*/episode.json'))
        assert len(paths) == options.episodes
        episodes = [Episode.load(path.parent) for path in paths]
        for episode in episodes:
            reset = episode.metadata['reset_orchard_audit']
            assert reset['apple_count'] == reset['visible_available_count'] == reset['inactive_fallen_count'] > 0
            assert reset['max_world_error_m'] <= 1e-6 and reset['max_local_error_m'] <= 1e-6
        assert all(episode.outcome['success'] for episode in episodes)
        assert len({episode.metadata['target_path'] for episode in episodes}) >= options.targets
        groups.append(episodes)
    recorded, replayed = groups
    initial_errors = []
    for original, replay in zip(recorded, replayed):
        assert original.metadata['mode'] == 'expert' and replay.metadata['mode'] == 'replay'
        assert original.metadata['target_path'] == replay.metadata['target_path']
        assert original.metadata['scene_sha256'] == replay.metadata['scene_sha256']
        assert original.frames and original.metadata['capture_required']
        original_arrays, replay_arrays = original.arrays(), replay.arrays()
        assert replay.metadata['source_command_sha256'] == hashlib.sha256(original_arrays['command'].tobytes()).hexdigest()
        initial_error = float(np.max(np.abs(original_arrays['q_all'][0] - replay_arrays['q_all'][0])))
        assert initial_error < 1e-5, f'Reset is not reproducible: {initial_error}'
        initial_errors.append(initial_error)
        assert replay.metadata['max_inactive_joint_drift'] < .003
        for frame in original.frames:
            step = int(frame['physics_step'])
            assert 0 <= step < len(original.states)
            assert abs(float(frame['episode_time_s']) - original_arrays['time_s'][step]) < 1e-7
            assert frame['clock_mapping'] == 'lockstep_render_barrier'
            assert frame['max_joint_change'] <= 1e-8
            assert frame['camera_pose_error_m'] <= .001
            assert frame['depth_valid'].any()
            assert np.isfinite(frame['depth_m']).all()
    captures = [event for episode in recorded + replayed for event in episode.events if event['kind'] == 'capture']
    report = {
        'recorded_successes': len(recorded), 'replayed_successes': len(replayed),
        'distinct_targets': len({episode.metadata['target_path'] for episode in recorded}),
        'rgbd_frames': sum(len(episode.frames) for episode in recorded),
        'physics_samples': sum(len(episode.states) for episode in recorded),
        'max_capture_error_m': max(event['error_m'] for event in captures),
        'max_capture_angle_error_degrees': max(event['angle_error_degrees'] for event in captures),
        'max_reset_joint_error': max(initial_errors),
        'reset_apples_checked_per_episode': recorded[0].metadata['reset_orchard_audit']['apple_count'],
        'max_reset_apple_world_error_m': max(episode.metadata['reset_orchard_audit']['max_world_error_m']
                                            for episode in recorded + replayed),
        'max_inactive_joint_drift': max(episode.metadata['max_inactive_joint_drift'] for episode in recorded + replayed),
        'max_camera_pose_error_m': max(float(frame['camera_pose_error_m']) for episode in recorded for frame in episode.frames),
        'max_mapped_sync_error_s': max(abs(float(frame['sync_error_s'])) for episode in recorded for frame in episode.frames),
        'sample_dt': recorded[0].metadata['sample_dt'],
        'replay_dt': replayed[0].metadata['replay_dt'],
        'speed_scale': recorded[0].metadata['speed_scale'],
        'scope': 'Deterministic single-arm oracle demonstrations and command replay; no learned policy evaluation',
    }
    print(json.dumps(report, indent=2))
    print('AUTOPICKER_M0_ACCEPTANCE_PASS')


if __name__ == '__main__':
    main()
