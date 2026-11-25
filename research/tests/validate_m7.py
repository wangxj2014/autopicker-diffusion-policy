"""Audit the fixed 20-trial M7 experiment; missing runs cannot yield a passing report."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from validate_m1 import rollout_report
from validate_m2 import ROOT, digest, training_audit
from validate_m6 import kinematics_audit, stage_audit
from autopicker_dp.dataset import load_dataset, _configuration
from autopicker_dp.episode import Episode
from autopicker_dp.hybrid import HYBRID_CONFIG, CORRIDOR_CONFIG
from autopicker_apple_pick import JointChain
from run_m7 import verify_completion

CHECKPOINT_HASH = 'd0d90421470e49ac5c3828e5486ade87da326907a33f994d725b5fc18d47f85a'
DATASET_HASH = 'f7d96a5015010969051af29532cbc0d2203b0cc852703fb0b5dec0a43c704fd5'


def geometry():
    path = ROOT / 'docs/research/2026-11-23-m6-kinematics.json'
    info = json.loads(path.read_text())
    chain = JointChain.__new__(JointChain)
    chain.base = np.array(info['base'])
    chain.frames = [(np.array(a), np.array(b), kind, np.array(axis))
                    for a, b, kind, axis in info['frames']]
    return chain, np.array(info['direction']), info, digest(path)


def pose_series(episode, chain, direction):
    a = episode.arrays()
    delta = a['goal_world_m'] - a['tip_world_m']
    axial = delta @ direction
    lateral = np.linalg.norm(delta - axial[:, None] * direction, axis=1)
    poses = np.array([chain.forward(q) for q in a['q']])
    angle = np.degrees(np.arccos(np.clip(-poses[:, :3, 1] @ direction, -1, 1)))
    return np.linalg.norm(delta, axis=1), axial, lateral, angle


def entry_evidence(episode, chain, direction):
    a = episode.arrays()
    distance, axial, lateral, angle = pose_series(episode, chain, direction)
    near = distance <= .03
    eligible = near | ((distance <= .12) & (axial >= 0) & (lateral <= .02) & (angle <= 5.))
    reasons = np.where(near, 'tip_entered_precision_region', 'tip_entered_approach_corridor')
    cutoff = next((e['time_s'] for e in episode.events if e['kind'] == 'capture'), np.inf)
    candidates = np.flatnonzero(eligible & (a['time_s'] < cutoff - 1e-9))
    precision = next((e for e in episode.events if e.get('stage') == 'precision'), None)
    if precision is not None:
        i = int(np.abs(a['time_s'] - precision['time_s']).argmin())
        fields = ['entry_distance_m', 'entry_angle_degrees']
        if not near[i]:
            fields += ['entry_axial_m', 'entry_lateral_m']
        if not np.isfinite([precision[key] for key in fields]).all():
            raise ValueError('Nonfinite corridor event geometry')
        if (abs(a['time_s'][i] - precision['time_s']) > 1e-8 or not len(candidates) or i != candidates[0]
                or precision['reason'] != reasons[i]
                or abs(precision['entry_distance_m'] - distance[i]) > 1e-9
                or abs(precision['entry_angle_degrees'] - angle[i]) > .002):
            raise ValueError('Corridor entry differs from first qualifying measured pose')
        if not near[i] and (abs(precision['entry_axial_m'] - axial[i]) > 1e-9
                            or abs(precision['entry_lateral_m'] - lateral[i]) > 1e-9):
            raise ValueError('Corridor event geometry disagrees with raw state')
    elif len(candidates):
        raise ValueError('Eligible corridor entry was skipped')
    return eligible, reasons


def seed_audit(metadata, previous_request_id, *, failed=False):
    if metadata['seed_mode'] != 'episode':
        raise ValueError('M7 requires independent episode seeds')
    attempts = metadata['inference_attempts']
    requests = metadata.get('inference_requests', [])
    if (metadata['inference_request_start'] != previous_request_id
            or metadata['inference_request_end'] != previous_request_id + len(attempts)
            or not 0 <= len(attempts) - len(requests) <= int(failed)):
        raise ValueError('Incomplete or invalid inference attempt span')
    for index, request in enumerate(attempts):
        previous_request_id += 1
        # Calculate independently from the serving implementation.
        payload = f"autopicker-episode-v1:{metadata['rollout_seed']}:{metadata['source_episode_index']}:{index}".encode()
        expected = {'source_episode_index': metadata['source_episode_index'], 'episode_request_index': index,
                    'noise_seed': int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big'),
                    'request_id': previous_request_id, 'generation': metadata['generation']}
        if any(request.get(key) != value for key, value in expected.items()):
            raise ValueError('Episode noise/request identity differs from paired protocol')
        if index < len(requests) and any(requests[index].get(key) != value for key, value in request.items()):
            raise ValueError('Successful inference differs from attempted identity')
    return previous_request_id


def paired_prefix(baseline, candidate):
    stops = [next((e['time_s'] for e in ep.events if e['kind'] == 'controller_stage'), ep.arrays()['time_s'][-1])
             for ep in (baseline, candidate)]
    cutoff = min(stops)
    arrays = [ep.arrays() for ep in (baseline, candidate)]
    indices = [np.flatnonzero(a['time_s'] < cutoff - 1e-9) for a in arrays]
    if not np.array_equal(indices[0], indices[1]) or set(arrays[0]) != set(arrays[1]):
        raise ValueError('Paired raw prefix length/schema differs')
    for key in arrays[0]:
        if not np.array_equal(arrays[0][key][indices[0]], arrays[1][key][indices[1]]):
            raise ValueError(f'Paired raw prefix differs: {key}')
    requests = [[r for r in ep.metadata['inference_requests'] if r['time_s'] < cutoff - 1e-9]
                for ep in (baseline, candidate)]
    if len(requests[0]) != len(requests[1]):
        raise ValueError('Paired inference prefix length differs')
    for a, b in zip(*requests):
        for key in ('time_s', 'generation', 'source_episode_index', 'episode_request_index',
                    'noise_seed', 'observation', 'action'):
            if a.get(key) != b.get(key):
                raise ValueError(f'Paired inference prefix differs: {key}')
    return {'until_time_s_exclusive': float(cutoff), 'raw_rows': len(indices[0]),
            'request_count': len(requests[0]), 'bitwise_equal': True}


def audit_run(folder, dataset, mode, split, indices, seed, chain, direction):
    verify_completion(folder)
    summary = json.loads((folder / 'summary.json').read_text())
    configuration = HYBRID_CONFIG if mode == 'hybrid' else CORRIDOR_CONFIG
    expected = {'mode': mode, 'split': split, 'seed': seed, 'seed_mode': 'episode',
                'checkpoint_sha256': CHECKPOINT_HASH, 'hybrid_configuration': configuration,
                'requested': len(indices)}
    if (any(summary.get(key) != value for key, value in expected.items())
            or [r['source_episode_index'] for r in summary['results']] != indices):
        raise ValueError('M7 model/selection/controller/seed protocol mismatch')
    report = rollout_report(folder, DATASET_HASH)
    report['seed_mode'] = 'episode'
    request_id = 0
    from autopicker_dp.run import SOURCE_HASHES
    for ordinal, index in enumerate(indices):
        directory = folder / f'episode_{ordinal:04d}'
        failed = (directory / 'failure.json').exists()
        document = json.loads((directory / ('failure.json' if failed else 'episode.json')).read_text())
        meta = document['metadata']
        source = dataset.manifest['episodes'][index]
        expected = {'source_episode_index': index, 'source_split': split, 'mode': mode, 'rollout_seed': seed,
            'reset_joint_positions': source['metadata']['reset_joint_positions'],
            'target_path': source['metadata']['target_path'], 'hybrid_configuration': configuration,
            'policy_dt': .1, 'source_sha256': SOURCE_HASHES}
        if (any(meta.get(key) != value for key, value in expected.items())
                or _configuration(meta) != dataset.manifest['source_configuration']
                or meta.get('dynamic_initial_state') is not None):
            raise ValueError('M7 reset/source/implementation identity changed')
        request_id = seed_audit(meta, request_id, failed=failed or not document['outcome']['success'])
        report['episodes'][ordinal]['source_episode_index'] = index
        report['episodes'][ordinal]['episode_json_sha256'] = digest(directory / ('failure.json' if failed else 'episode.json'))
        if failed:
            report['episodes'][ordinal]['audit_unavailable'] = document['audit_error']
            continue  # Keep the attempt in the denominator; never count as success.
        episode = Episode.load(directory)
        a = episode.arrays()
        result = summary['results'][ordinal]
        if (result['reason'] != episode.outcome['reason'] or result['target_path'] != meta['target_path']
                or abs(result['duration_s'] - a['time_s'][-1]) > 1e-9
                or a['time_s'][-1] > 35. + 1e-8
                or (episode.outcome['reason'] == 'Episode timed out before collect'
                    and abs(a['time_s'][-1] - 35.) > 1e-8)):
            raise ValueError('M7 outcome or 35 s budget mismatch')
        kine = kinematics_audit(episode, chain, direction)
        kwargs = {}
        if mode == 'hybrid-corridor':
            mask, reasons = entry_evidence(episode, chain, direction)
            kwargs = dict(configuration=CORRIDOR_CONFIG, entry_mask=mask, entry_reasons=reasons)
        stage = stage_audit(episode, **kwargs)
        report['episodes'][ordinal].update(stage_audit=stage, kinematics_audit=kine,
            states_sha256=digest(directory / 'states.npz'))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = ROOT / 'runs/autopicker'
    dataset = load_dataset(ROOT / 'datasets/autopicker/lowdim_v2')
    if digest(ROOT / 'datasets/autopicker/lowdim_v2/manifest.json') != DATASET_HASH:
        raise ValueError('M7 dataset changed')
    training = training_audit(runs / 'm5_train_sample_long_v1', dataset, DATASET_HASH, 'physical', 'train')
    if training['checkpoint_sha256'] != CHECKPOINT_HASH:
        raise ValueError('M7 frozen checkpoint changed')
    chain, direction, info, geometry_hash = geometry()
    if (info['scene_sha256'] != dataset.manifest['source_configuration']['scene_sha256']
            or info['arm'] != dataset.manifest['source_configuration']['arm']):
        raise ValueError('Kinematic export differs from scene/arm')
    pairs = []
    for seed in (0, 1):
        for split, indices in (('train', [1, 2, 4]), ('val', [0, 5])):
            reports, folders = [], []
            for mode in ('hybrid', 'hybrid-corridor'):
                folder = runs / f'm7_{mode}_{split}_s{seed}'
                folders.append(folder)
                reports.append(audit_run(folder, dataset, mode, split, indices, seed, chain, direction))
            prefixes = []
            for i in range(len(indices)):
                paths = [folder / f'episode_{i:04d}' for folder in folders]
                if any((path / 'failure.json').exists() for path in paths):
                    prefixes.append({'unavailable': 'raw episode failed audit; attempt retained'})
                else:
                    prefixes.append(paired_prefix(*(Episode.load(path) for path in paths)))
            pairs.append(dict(seed=seed, split=split, source_episode_indices=indices,
                              baseline=reports[0], corridor=reports[1], paired_prefixes=prefixes))
    report = {'schema': 'autopicker-m7-report-v1', 'dataset_sha256': DATASET_HASH,
        'checkpoint_sha256': CHECKPOINT_HASH, 'kinematics_sha256': geometry_hash,
        'new_closed_loop_trials': sum(p[k]['attempted'] for p in pairs for k in ('baseline', 'corridor')),
        'new_training_runs': 0, 'pairs': pairs, 'data_and_protocol_audit_passed': True,
        'scope': 'oracle hybrid, two inference seeds, three seen targets and one development validation target repeated twice',
        'pure_learned_collection_demonstrated': False, 'robust_policy_accepted': False}
    if report['new_closed_loop_trials'] != 20:
        raise ValueError('Incomplete fixed experiment')
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('AUTOPICKER_M7_AUDIT_PASS', flush=True)


if __name__ == '__main__':
    main()
