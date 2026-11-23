"""Audit M1 artifacts, preserving failed policy episodes in the denominator."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopicker_dp.dataset import load_dataset
from autopicker_dp.episode import Episode


def rollout_report(folder, dataset_hash):
    folder = Path(folder)
    summary = json.loads((folder / 'summary.json').read_text())
    if (not summary['completed'] or summary['dataset_sha256'] != dataset_hash
            or summary['attempted'] != summary['requested']
            or len(summary['results']) != summary['requested']):
        raise ValueError('Incomplete or mismatched rollout summary')
    episodes, latencies, captures, successes = [], [], 0, 0
    for index, result in enumerate(summary['results']):
        directory = folder / f'episode_{index:04d}'
        if (directory / 'failure.json').exists():
            failure = json.loads((directory / 'failure.json').read_text())
            if (result['success'] or failure['metadata']['dataset_sha256'] != dataset_hash
                    or failure['metadata']['checkpoint_sha256'] != summary['checkpoint_sha256']
                    or failure['metadata']['target_path'] != result['target_path']
                    or failure['reason'] != result['reason']):
                raise ValueError('Failure record identity or outcome differs from summary')
            episodes.append({'target_path': result['target_path'], 'success': False, 'reason': result['reason']})
            continue
        episode = Episode.load(directory)
        if episode.outcome['success'] != result['success'] or episode.metadata['dataset_sha256'] != dataset_hash:
            raise ValueError('Episode outcome or data identity differs from summary')
        if episode.metadata['checkpoint_sha256'] != summary['checkpoint_sha256']:
            raise ValueError('Episode checkpoint differs from summary')
        reset = episode.metadata['reset_orchard_audit']
        if (reset['apple_count'] != reset['visible_available_count']
                or reset['apple_count'] != reset['inactive_fallen_count']
                or reset['max_world_error_m'] > 1e-6 or episode.metadata['max_inactive_joint_drift'] > .003):
            raise ValueError('Invalid reset or inactive joint drift')
        arrays = episode.arrays()
        before_capture = ~arrays['attached']
        distance = np.linalg.norm(arrays['tip_world_m'] - arrays['goal_world_m'], axis=1)
        events = [event for event in episode.events if event['kind'] in ('capture', 'collect')]
        captured = any(event['kind'] == 'capture' for event in events)
        captures += int(captured)
        successes += int(episode.outcome['success'])
        requests = episode.metadata.get('inference_requests', [])
        if requests:
            timestamps = np.array([request['time_s'] for request in requests])
            if (timestamps[0] != 0 or not np.allclose(np.diff(timestamps), .2, atol=1e-8)
                    or any(request['generation'] != episode.metadata['generation'] for request in requests)):
                raise ValueError('Incorrect inference timing or generation')
            latencies.extend(request['latency_s'] for request in requests)
        episodes.append({'target_path': episode.metadata['target_path'],
                         'success': episode.outcome['success'], 'reason': episode.outcome['reason'],
                         'captured': captured, 'duration_s': result['duration_s'],
                         'wall_time_s': result['wall_time_s'],
                         'min_unattached_tip_distance_m': float(distance[before_capture].min()),
                         'inference_requests': len(requests), 'limited_steps': episode.metadata['limited_steps'],
                         'events': events})
    if successes != summary['successes']:
        raise ValueError('Success denominator disagrees with audited episodes')
    return {'path': str(folder), 'mode': summary['mode'], 'scope': summary['scope'], 'split': summary['split'],
            'requested': summary['requested'], 'attempted': summary['attempted'],
            'captures': captures, 'successes': successes, 'checkpoint_sha256': summary['checkpoint_sha256'],
            'seed': summary['seed'], 'episodes': episodes,
            'latency_s': None if not latencies else {'count': len(latencies), 'mean': float(np.mean(latencies)),
                                                    'p95': float(np.quantile(latencies, .95)),
                                                    'max': float(np.max(latencies))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--overfit-training', type=Path, required=True)
    parser.add_argument('--train-training', type=Path, required=True)
    parser.add_argument('--overfit-rollout', type=Path, required=True)
    parser.add_argument('--val-rollout', type=Path, required=True)
    parser.add_argument('--rgbd-smoke', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    dataset = load_dataset(options.dataset)
    digest = hashlib.sha256((options.dataset / 'manifest.json').read_bytes()).hexdigest()
    replay = rollout_report(options.replay, digest)
    if replay['successes'] != replay['requested'] or replay['requested'] != len(dataset.episode_ends):
        raise ValueError('Converted replay failed acceptance')
    training = {}
    for split, folder in [('all-pilot', options.overfit_training), ('train', options.train_training)]:
        report = json.loads((folder / 'report.json').read_text())
        if (report['split'] != split or report['dataset_sha256'] != digest
                or report['checkpoint_reload_max_error'] != 0
                or hashlib.sha256((folder / 'checkpoint.pt').read_bytes()).hexdigest() != report['checkpoint_sha256']):
            raise ValueError('Training artifact audit failed')
        expected = (list(range(len(dataset.episode_ends))) if split == 'all-pilot'
                    else dataset.episode_indices('train'))
        if report['normalizer']['fit_metadata']['episode_indices'] != expected:
            raise ValueError('Training normalization scope differs from declared split')
        training[split] = {key: report[key] for key in
                           ('scope', 'seed', 'steps', 'batch_size', 'learning_rate', 'training_windows',
                            'initial_loss_mean', 'final_loss_mean', 'checkpoint_reload_max_error',
                            'wall_time_s', 'peak_cuda_memory_bytes', 'checkpoint_sha256', 'versions')}
        training[split]['offline'] = json.loads((folder / 'offline.json').read_text())
    overfit = rollout_report(options.overfit_rollout, digest)
    validation = rollout_report(options.val_rollout, digest)
    if (overfit['checkpoint_sha256'] != training['all-pilot']['checkpoint_sha256']
            or validation['checkpoint_sha256'] != training['train']['checkpoint_sha256']):
        raise ValueError('Rollout did not use the audited checkpoint')
    action_support = {'train_min': dataset.normalizer.fields['action']['min'].tolist(),
                      'train_max': dataset.normalizer.fields['action']['max'].tolist(), 'splits': {}}
    starts = np.r_[0, dataset.episode_ends[:-1]]
    for split in ('val', 'test'):
        actions = np.concatenate([dataset.data['action'][starts[index]:dataset.episode_ends[index]]
                                  for index in dataset.episode_indices(split)])
        exceedance = np.maximum(np.maximum(np.array(action_support['train_min']) - actions,
                                          actions - np.array(action_support['train_max'])), 0.)
        action_support['splits'][split] = {'rows': len(actions),
            'rows_outside_by_joint_atol_1e_5': (exceedance > 1e-5).sum(0).tolist(),
            'max_outside_by_joint_native_units': exceedance.max(0).tolist()}
    # The trained scheduler clips normalized samples to [-1, 1]. With min/max
    # action normalization that restricts outputs to the train action envelope.
    action_support['sampler_clip_sample'] = True
    report = {'schema': 'autopicker-m1-report-v1',
              'scope': 'single scene, five oracle targets, one training seed, visual attachment; pilot only',
              'dataset': {'path': str(options.dataset), 'sha256': digest, 'rows': dataset.manifest['rows'],
                          'episodes': len(dataset.episode_ends), 'target_splits': dataset.manifest['target_splits'],
                          'dt': dataset.manifest['dt'], 'splits': {key: dataset.episode_indices(key)
                                                                for key in ('train', 'val', 'test')}},
              'training': training, 'action_support': action_support, 'converted_replay': replay,
              'overfit_rollout': overfit, 'validation_rollout': validation,
              'data_and_interface_audit_passed': True,
              'learned_collection_demonstrated': overfit['successes'] + validation['successes'] > 0,
              'robust_policy_accepted': False}
    if options.rgbd_smoke:
        episode = Episode.load(options.rgbd_smoke / 'episode_0000')
        arrays = episode.arrays()
        collected = next(event['time_s'] for event in episode.events if event['kind'] == 'collect')
        if not episode.outcome['success'] or not episode.metadata['capture_required'] or not episode.frames:
            raise ValueError('RGB-D smoke episode did not collect with camera recording')
        for frame in episode.frames:
            if (abs(arrays['time_s'][int(frame['physics_step'])] - frame['episode_time_s']) > 1e-8
                    or not frame['depth_valid'].any()):
                raise ValueError('RGB-D smoke has mismatched physics or invalid depth')
        post_collect_frames = int(sum(frame['episode_time_s'] > collected for frame in episode.frames))
        if post_collect_frames < 1:
            raise ValueError('RGB-D smoke lacks real post-collect frames')
        report['rgbd_tail_smoke'] = {'path': str(options.rgbd_smoke), **episode.audit(),
                                     'post_collect_frames': post_collect_frames,
                                     'max_camera_pose_error_m': max(float(frame['camera_pose_error_m'])
                                                                    for frame in episode.frames)}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('AUTOPICKER_M1_AUDIT_PASS ' + json.dumps({'rows': report['dataset']['rows'],
          'converted_replay': [replay['successes'], replay['requested']],
          'overfit_rollout': [overfit['successes'], overfit['requested']],
          'validation_rollout': [validation['successes'], validation['requested']]}))


if __name__ == '__main__':
    main()
