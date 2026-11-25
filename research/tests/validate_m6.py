"""Audit the explicitly hybrid event gate against the frozen M5 learned baseline."""
import argparse
import json
from pathlib import Path

import numpy as np

from validate_m1 import rollout_report
from validate_m2 import ROOT, digest, training_audit
from validate_m3 import policy_rollout
from autopicker_dp.control import JointExecutor, vector
from autopicker_dp.dataset import load_dataset, _configuration
from autopicker_dp.episode import Episode
from autopicker_dp.environment import capture_errors
from autopicker_dp.hybrid import HYBRID_CONFIG
from autopicker_apple_pick import JointChain, motion_duration, quintic_blend


def kinematics_audit(episode, chain, direction):
    """Independently verify raw poses and controller goals with exported USD FK."""
    a = episode.arrays()
    poses = np.array([chain.forward(q) for q in a['q']])
    errors = np.linalg.norm(poses[:, :3, 3] - a['tip_world_m'], axis=1)
    if not np.isfinite(poses).all() or errors.max() > 1e-5:
        raise ValueError('Raw joint FK disagrees with measured tip positions')
    result = {'max_fk_measured_tip_error_m': float(errors.max()), 'capture_pose': None,
              'precision_goal_pose': None, 'return_goal_error_m': None}
    for event in episode.events:
        if event['kind'] not in ('capture', 'controller_stage'):
            continue
        row = int(np.abs(a['time_s'] - event['time_s']).argmin())
        if abs(a['time_s'][row] - event['time_s']) > 1e-8:
            raise ValueError('Kinematic event is not on a physics step')
        if event['kind'] == 'capture':
            distance, angle = capture_errors(poses[row], a['goal_world_m'][row], direction)
            if (not np.isfinite([distance, angle, event['angle_error_degrees']]).all()
                    or distance > .003 + 1e-5 or angle > 1. + .002
                    or abs(angle - event['angle_error_degrees']) > .002
                    or abs(distance - event['error_m']) > 1e-5):
                raise ValueError('Capture pose disagrees with raw joint FK')
            result['capture_pose'] = {'distance_m': distance, 'angle_degrees': angle}
        elif event['stage'] == 'precision':
            goal_pose = chain.forward(vector(event['goal_command']))
            distance, angle = capture_errors(goal_pose, a['goal_world_m'][row], direction)
            if not np.isfinite([distance, angle]).all() or distance > 1e-5 or angle > .01:
                raise ValueError('Precision goal does not solve the grasp pose')
            result['precision_goal_pose'] = {'distance_m': distance, 'angle_degrees': angle}
        else:
            goal_pose = chain.forward(vector(event['goal_command']))
            distance = float(np.linalg.norm(goal_pose[:3, 3] - a['receiver_world_m'][row] - [0., 0., .12]))
            if not np.isfinite(distance) or distance > 1e-5:
                raise ValueError('Return goal does not solve the receiver position')
            result['return_goal_error_m'] = distance
    return result


def stage_audit(episode):
    meta, events, a = episode.metadata, episode.events, episode.arrays()
    if meta.get('hybrid_configuration') != HYBRID_CONFIG:
        raise ValueError('Hybrid configuration changed')
    times = a['time_s']
    stages = [e for e in events if e['kind'] == 'controller_stage']
    names = [e['stage'] for e in stages]
    if names not in ([], ['precision'], ['return'], ['precision', 'return']):
        raise ValueError('Controller stages oscillated or repeated')
    captures = [e for e in events if e['kind'] == 'capture']
    collects = [e for e in events if e['kind'] == 'collect']
    if len(captures) > 1 or len(collects) > 1 or len(collects) > len(captures):
        raise ValueError('Invalid capture/collect sequence')
    if bool(collects) != bool(episode.outcome['success']):
        raise ValueError('Outcome disagrees with collect event')

    def row(timestamp):
        matches = np.flatnonzero(np.isclose(times, timestamp, rtol=0, atol=1e-8))
        if len(matches) != 1:
            raise ValueError('Event is not on a recorded physics step')
        return int(matches[0])

    distance = np.linalg.norm(a['tip_world_m'] - a['goal_world_m'], axis=1)
    expected_attached = np.zeros(len(times), dtype=bool)
    if captures:
        c = captures[0]
        measured = distance[row(c['time_s'])]
        if (not np.isfinite([c['error_m'], c['angle_error_degrees']]).all()
                or not 0 <= c['angle_error_degrees'] <= 1. or not 0 <= c['error_m'] <= .003
                or abs(measured - c['error_m']) > 1e-7 or measured > .003
                or c['target_path'] != meta['target_path']):
            raise ValueError('Capture did not meet measured criteria')
        expected_attached[row(c['time_s']):] = True
        if not stages or stages[-1]['stage'] != 'return' or abs(stages[-1]['time_s'] - c['time_s']) > 1e-9:
            raise ValueError('Return must begin on the capture step')
        if events.index(c) >= events.index(stages[-1]):
            raise ValueError('Return was logged before capture')
    elif 'return' in names:
        raise ValueError('Return without capture')
    if collects:
        c = collects[0]
        if c['time_s'] <= captures[0]['time_s'] or c['target_path'] != meta['target_path']:
            raise ValueError('Invalid collect ordering or target')
        i = row(c['time_s'])
        release_error = np.linalg.norm(a['tip_world_m'][i] - a['receiver_world_m'][i] - [0., 0., .12])
        if not np.isfinite(release_error) or release_error > .025:
            raise ValueError('Collect did not meet measured receiver criterion')
        expected_attached[row(c['time_s']):] = False
    if not np.array_equal(a['attached'], expected_attached):
        raise ValueError('Measured attachment disagrees with task events')
    precision = next((e for e in stages if e['stage'] == 'precision'), None)
    cutoff = captures[0]['time_s'] if captures else np.inf
    near = np.flatnonzero((distance <= HYBRID_CONFIG['precision_entry_distance_m']) & (times < cutoff - 1e-9))
    if precision:
        i = row(precision['time_s'])
        if (not len(near) or i != near[0]
                or not np.isclose(precision['entry_distance_m'], distance[i], rtol=0, atol=1e-9)
                or not np.isfinite(precision['entry_angle_degrees'])
                or precision['reason'] != 'tip_entered_precision_region'):
            raise ValueError('Precision entry differs from first measured entry')
    elif len(near):
        raise ValueError('Precision entry was skipped')
    requests = meta.get('inference_requests', [])
    if stages and any(r['time_s'] >= stages[0]['time_s'] - 1e-9 for r in requests):
        raise ValueError('Learned inference continued after hybrid handoff')
    stop = stages[0]['time_s'] if stages else times[-1]
    expected_requests = np.arange(0., stop - 1e-9, .2)
    if len(requests) != len(expected_requests) or not np.allclose(
            [r['time_s'] for r in requests], expected_requests, rtol=0, atol=1e-9):
        raise ValueError('Missing or mistimed inference before handoff')
    scale = meta['speed_scale']
    velocity = np.array([.18, .16, .14, .30]) * scale
    acceleration = np.array([.35, .30, .30, .60]) * scale**2
    max_error = 0.
    for n, stage in enumerate(stages):
        i = row(stage['time_s'])
        end = row(stages[n + 1]['time_s']) if n + 1 < len(stages) else len(times) - 1
        start, goal = vector(stage['start_command']), vector(stage['goal_command'])
        previous = a['command'][i - 1] if i else a['q'][0]
        if (not np.allclose(start, previous, rtol=0, atol=1e-12)
                or np.any(goal < meta['joint_lower']) or np.any(goal > meta['joint_upper'])
                or stage['target_path'] != meta['target_path']):
            raise ValueError('Stage starts discontinuously or has an invalid goal')
        duration = motion_duration(start, goal, velocity, acceleration, .8 / scale)
        if not np.isfinite(stage['duration_s']) or abs(duration - stage['duration_s']) > 1e-10:
            raise ValueError('Stage duration differs from the frozen controller')
        if stage['stage'] == 'return' and (stage['reason'] != 'capture_event'
                or abs(goal[1] - a['q'][i, 1]) > 1e-12):
            raise ValueError('Return did not preserve measured hor')
        executor = JointExecutor(meta['joint_lower'], meta['joint_upper'], velocity, acceleration, meta['physics_dt'])
        executor.reset(start)
        executor.velocity = a['velocity_command'][i - 1].copy() if i else np.zeros(4)
        for k in range(i, end):
            blend = quintic_blend((times[k] - stage['time_s']) / duration)
            command, speed = executor.advance(start + blend * (goal - start))
            error = max(float(np.max(np.abs(command - a['command'][k]))),
                        float(np.max(np.abs(speed - a['velocity_command'][k]))))
            max_error = max(max_error, error)
        if max_error > 1e-9:
            raise ValueError('Applied commands differ from the recorded hybrid trajectories')
    return {'stages': stages, 'precision_entry_s': precision['time_s'] if precision else None,
        'precision_duration_s': None if precision is None else (
            captures[0]['time_s'] if captures else times[-1]) - precision['time_s'],
        'max_applied_command_error': max_error,
        'classical_control_duration_s': float(times[-1] - stages[0]['time_s']) if stages else 0.}


def hybrid_rollout(folder, dataset, dataset_hash, indices, checkpoint_hash, split, baseline_folder):
    report = rollout_report(folder, dataset_hash)
    summary = json.loads((folder / 'summary.json').read_text())
    if (report['mode'] != 'hybrid' or report['split'] != split or report['seed'] != 0
            or report['checkpoint_sha256'] != checkpoint_hash or report['requested'] != len(indices)
            or summary.get('hybrid_configuration') != HYBRID_CONFIG
            or [r['source_episode_index'] for r in summary['results']] != indices):
        raise ValueError('Hybrid rollout protocol/model/selection mismatch')
    geometry_path = ROOT / 'docs/research/2026-11-23-m6-kinematics.json'
    geometry = json.loads(geometry_path.read_text())
    if (geometry['scene_sha256'] != dataset.manifest['source_configuration']['scene_sha256']
            or geometry['arm'] != dataset.manifest['source_configuration']['arm']):
        raise ValueError('Kinematic export belongs to a different scene or arm')
    chain = JointChain.__new__(JointChain)
    chain.base = np.array(geometry['base'])
    chain.frames = [(np.array(a), np.array(b), kind, np.array(axis))
                    for a, b, kind, axis in geometry['frames']]
    direction = np.array(geometry['direction'])
    report['kinematics_sha256'] = digest(geometry_path)
    request_id = 0
    for ordinal, index in enumerate(indices):
        path = folder / f'episode_{ordinal:04d}'
        failed = (path / 'failure.json').exists()
        document = json.loads((path / ('failure.json' if failed else 'episode.json')).read_text())
        meta = document['metadata']
        source = dataset.manifest['episodes'][index]
        base = Episode.load(baseline_folder / f'episode_{ordinal:04d}').metadata
        if (meta['reset_joint_positions'] != source['metadata']['reset_joint_positions']
                or _configuration(meta) != dataset.manifest['source_configuration']
                or meta['source_episode_index'] != index or meta['source_split'] != source['split']
                or meta['target_path'] != source['metadata']['target_path']
                or meta['mode'] != 'hybrid' or meta['rollout_seed'] != 0 or meta['policy_dt'] != .1
                or meta['hybrid_configuration'] != HYBRID_CONFIG
                or meta.get('dynamic_initial_state') is not None):
            raise ValueError('Hybrid reset/source/control protocol changed')
        for key, value in base['source_sha256'].items():
            if key != 'research/autopicker_dp/rollout.py' and meta['source_sha256'].get(key) != value:
                raise ValueError(f'Unrelated source changed: {key}')
        for key in ('research/autopicker_dp/rollout.py', 'research/autopicker_dp/hybrid.py'):
            if meta['source_sha256'].get(key) != digest(ROOT / key):
                raise ValueError(f'Running hybrid implementation differs: {key}')
        requests = meta.get('inference_requests', [])
        for r in requests:
            request_id += 1
            if r['request_id'] != request_id:
                raise ValueError('Inference sequence changed or requests were removed')
        if failed:
            report['episodes'][ordinal]['stage_audit_unavailable'] = document['audit_error']
            continue
        episode = Episode.load(path)
        duration = episode.arrays()['time_s'][-1]
        if duration > 35. + 1e-8 or (episode.outcome['reason'] == 'Episode timed out before collect'
                                   and abs(duration - 35.) > 1e-8):
            raise ValueError('Hybrid timeout differs from frozen 35 s budget')
        report['episodes'][ordinal]['hybrid_audit'] = stage_audit(episode)
        report['episodes'][ordinal]['kinematics_audit'] = kinematics_audit(episode, chain, direction)
        report['episodes'][ordinal]['episode_json_sha256'] = digest(path / 'episode.json')
        report['episodes'][ordinal]['states_sha256'] = digest(path / 'states.npz')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    data, runs = ROOT / 'datasets/autopicker/lowdim_v2', ROOT / 'runs/autopicker'
    dataset = load_dataset(data)
    data_hash = digest(data / 'manifest.json')
    training = training_audit(runs / 'm5_train_sample_long_v1', dataset, data_hash, 'physical', 'train')
    checkpoint_hash = training['checkpoint_sha256']
    if checkpoint_hash != 'd0d90421470e49ac5c3828e5486ade87da326907a33f994d725b5fc18d47f85a':
        raise ValueError('M6 must use the frozen M5 sample 30000 checkpoint')
    pairs = {}
    for label, split, indices in [('seen_targets', 'train', dataset.episode_indices('train')[:3]),
                                   ('validation', 'val', dataset.episode_indices('val'))]:
        base_folder = runs / f'm5_sample_long_{split}_v1'
        baseline = policy_rollout(base_folder, dataset, data_hash, indices, training['normalizer'], checkpoint_hash, split)
        hybrid = hybrid_rollout(runs / f'm6_hybrid_{split}_v1', dataset, data_hash, indices, checkpoint_hash, split, base_folder)
        pairs[label] = {'source_episode_indices': indices, 'pure_policy': baseline, 'hybrid': hybrid}
    report = {'schema': 'autopicker-m6-report-v1', 'dataset_sha256': data_hash,
        'checkpoint_sha256': checkpoint_hash, 'hybrid_configuration': HYBRID_CONFIG,
        'scope': 'oracle hybrid; one seed, three seen targets, one validation target repeated twice; visual attachment',
        'pairs': pairs, 'new_hybrid_trials': 5, 'reused_pure_policy_trials': 5, 'new_training_runs': 0,
        'pure_learned_collection_demonstrated': False, 'robust_policy_accepted': False,
        'data_and_protocol_audit_passed': True}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('AUTOPICKER_M6_AUDIT_PASS ' + json.dumps({k: [v['hybrid']['captures'], v['hybrid']['successes'],
          v['hybrid']['requested']] for k, v in pairs.items()}), flush=True)


if __name__ == '__main__':
    main()
