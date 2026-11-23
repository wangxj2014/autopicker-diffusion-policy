"""Audit dynamic correction lineage, state restoration and fixed-budget comparison."""
import argparse
import json
from pathlib import Path

import numpy as np

from validate_m2 import ROOT, compact_training, comparable_scheduler, digest, training_audit
from validate_m3 import policy_rollout
from validate_m1 import rollout_report
from autopicker_dp.dataset import load_dataset, _configuration
from autopicker_dp.dynamic_recovery import source_trials, validate_dynamic_provenance
from autopicker_dp.episode import Episode
from autopicker_dp.rollout import verified_sources
from autopicker_dp.train import training_inputs


def training_pair(baseline, dynamic, dataset, dataset_hash, static_path, dynamic_path):
    result = {}
    for label, report, path in (('static', baseline, static_path), ('dynamic', dynamic, dynamic_path)):
        normalizer, sequences, info = training_inputs(dataset, dataset_hash, 'train', 'physical', path)
        if (report.get('recovery_augmentation') != info
                or report['training_windows'] != len(sequences)
                or report['normalizer'] != normalizer.to_dict()
                or report['dataset_manifest'] != dataset.manifest):
            raise ValueError('Recovery lineage, window count or frozen normalizer mismatch')
        result[label] = {'base_windows': info['base_windows'], 'recovery_windows': info['recovery_windows'],
            'combined_windows': len(sequences), 'recovery_window_fraction': info['recovery_windows'] / len(sequences),
            'sampling': info['sampling'], 'normalizer_source': info['normalizer_source'],
            'recovery_dataset_sha256': info['dataset_sha256']}
    if comparable_scheduler(baseline['scheduler']) != comparable_scheduler(dynamic['scheduler']):
        raise ValueError('Dynamic correction changed scheduler')
    for key in ('seed', 'steps', 'batch_size', 'learning_rate', 'dt', 'model_config',
                'ema_decay', 'optimizer', 'versions', 'normalizer'):
        if baseline[key] != dynamic[key]:
            raise ValueError(f'Dynamic correction changed another training factor: {key}')
    for key, value in {'seed': 0, 'steps': 3000, 'batch_size': 64, 'learning_rate': 1e-4,
                       'dt': .1, 'ema_decay': .995, 'optimizer': 'AdamW'}.items():
        if baseline[key] != value:
            raise ValueError(f'Training differs from fixed M4 protocol: {key}')
    # Augmentation adds provenance validation; rollout adds restoration only for
    # dynamic source episodes. Ordinary parent-data learned control is unchanged.
    for key, value in baseline['source_sha256'].items():
        if key not in ('augmentation.py', 'rollout.py') and dynamic['source_sha256'].get(key) != value:
            raise ValueError(f'Unrelated training source changed: {key}')
    result['normalizer'] = baseline['normalizer']
    return result


def restored_initial(episode, state, audit=None):
    arrays = episode.arrays()
    errors = {'position': float(np.max(np.abs(arrays['q_all'][0] - state['q_all']))),
              'active_velocity': float(np.max(np.abs(arrays['qdot'][0] - state['qdot'])))}
    values = list(errors.values())
    if audit is not None:
        values += [audit['max_position_error'], audit['max_active_velocity_error']]
    if arrays['time_s'][0] != 0 or not np.isfinite(values).all() or max(values) > 1e-6:
        raise ValueError('Dynamic initial state or restore readback mismatch')
    return errors


def dynamic_audit(folder, exported, replay_folder, parent, parent_hash, policy_folder, checkpoint):
    trials = source_trials(parent, parent_hash, policy_folder, checkpoint)
    summary = json.loads((folder / 'summary.json').read_text())
    if (not summary['completed'] or summary['mode'] != 'expert-dynamic-recovery'
            or summary['dataset_sha256'] != parent_hash
            or summary['policy_checkpoint_sha256'] != digest(checkpoint)
            or summary['requested'] != 6 or summary['attempted'] != 6
            or len(summary['results']) != 6 or summary['successes'] != 6):
        raise ValueError('Dynamic pilot requires all six attempts to succeed before export/training')
    dataset = load_dataset(exported)
    verified_sources(exported)
    if (len(dataset.episode_ends) != 6 or dataset.episode_indices('train') != list(range(6))
            or Path(dataset.manifest['source_root']).resolve() != folder.resolve()
            or [e['episode_json_sha256'] for e in dataset.manifest['episodes']]
            != [digest(folder / f'episode_{i:04d}/episode.json') for i in range(6)]):
        raise ValueError('Dynamic export omits attempts or changes their order/split')
    rows = []
    for ordinal, (source, expected, time_s) in enumerate(trials):
        result = summary['results'][ordinal]
        episode = Episode.load(folder / f'episode_{ordinal:04d}')
        info = episode.metadata['recovery']
        state = validate_dynamic_provenance(info, episode.metadata, parent, parent_hash)
        if (any(info.get(key) != value for key, value in expected.items())
                or state['source_time_s'] != time_s
                or result['source_time_s'] != time_s
                or result['source_episode_index'] != info['source_episode_index']
                or result['target_path'] != source.metadata['target_path']
                or not result['success'] or not episode.outcome['success']
                or result['reason'] != episode.outcome['reason']
                or episode.metadata['mode'] != 'expert' or episode.metadata['post_collect_hold_s'] != .2
                or episode.metadata['max_inactive_joint_drift'] > .003
                or result['duration_s'] > 35.):
            raise ValueError('Dynamic correction source, protocol or outcome mismatch')
        initial_error = restored_initial(episode, state, info['restore_audit'])
        rows.append({'source_episode_index': info['source_episode_index'], 'source_time_s': time_s,
            'target_path': episode.metadata['target_path'], 'recovery': info,
            'initial_error': initial_error, **episode.audit()})
    export_hash = digest(exported / 'manifest.json')
    replay = rollout_report(replay_folder, export_hash)
    replay_summary = json.loads((replay_folder / 'summary.json').read_text())
    if (replay['mode'] != 'converted-replay' or replay['split'] != 'train' or replay['seed'] != 0
            or replay['successes'] != 6 or replay['requested'] != 6 or replay['captures'] != 6
            or replay['checkpoint_sha256'] is not None
            or [r['source_episode_index'] for r in replay_summary['results']] != list(range(6))):
        raise ValueError('Dynamic converted replay acceptance failed')
    for ordinal, row in enumerate(rows):
        episode = Episode.load(replay_folder / f'episode_{ordinal:04d}')
        state = row['recovery']['state']
        if (episode.metadata.get('dynamic_initial_state') != state
                or episode.metadata['reset_joint_positions'] != state['q_all']
                or episode.metadata['source_episode_index'] != ordinal
                or episode.metadata['source_split'] != 'train'
                or episode.metadata['target_path'] != row['target_path']
                or _configuration(episode.metadata) != parent.manifest['source_configuration']
                or episode.metadata['mode'] != 'converted-replay'
                or episode.metadata['policy_dt'] != .1
                or replay['episodes'][ordinal]['duration_s'] > 35.):
            raise ValueError('Dynamic replay initial state or protocol differs from expert correction')
        row['replay_initial_error'] = restored_initial(episode, state)
    return {'requested': 6, 'attempted': 6, 'successes': 6, 'episodes': rows,
        'export_accepted': True, 'export_rows': dataset.manifest['rows'], 'export_sha256': export_hash,
        'converted_replay': replay, 'used_for_training': True,
        'restore_limit': 'Measured q_all, active qdot and previous executor state only; inactive velocity zero, no solver checkpoint'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    data = ROOT / 'datasets/autopicker'
    runs = ROOT / 'runs/autopicker'
    parent = load_dataset(data / 'lowdim_v2')
    parent_hash = digest(data / 'lowdim_v2/manifest.json')
    baseline = training_audit(runs / 'm3_train_recovery_v1', parent, parent_hash, 'physical', 'train')
    dynamic = training_audit(runs / 'm4_train_dynamic_v1', parent, parent_hash, 'physical', 'train')
    comparison = training_pair(baseline, dynamic, parent, parent_hash,
                              data / 'm2_recovery_lowdim_v1', data / 'm4_dynamic_lowdim_v1')
    recovery = dynamic_audit(data / 'm4_dynamic_raw_v1', data / 'm4_dynamic_lowdim_v1',
        runs / 'm4_dynamic_replay_v1', parent, parent_hash, runs / 'm3_recovery_train_v1',
        runs / 'm3_train_recovery_v1/checkpoint.pt')
    if recovery['export_sha256'] != comparison['dynamic']['recovery_dataset_sha256']:
        raise ValueError('Dynamic training did not use accepted recovery export')
    pairs = {}
    for label, split, indices, old_name, new_name in (
        ('seen_targets', 'train', parent.episode_indices('train')[:3],
         'm3_recovery_train_v1', 'm4_dynamic_train_v1'),
        ('validation', 'val', parent.episode_indices('val'),
         'm3_recovery_val_v1', 'm4_dynamic_val_v1')):
        old = policy_rollout(runs / old_name, parent, parent_hash, indices, baseline['normalizer'],
                             baseline['checkpoint_sha256'], split)
        new = policy_rollout(runs / new_name, parent, parent_hash, indices, dynamic['normalizer'],
                             dynamic['checkpoint_sha256'], split)
        pairs[label] = {'source_episode_indices': indices, 'baseline_rollout': old, 'dynamic_rollout': new,
                        'baseline_reused_from_m3': True}
    report = {'schema': 'autopicker-m4-report-v1', 'dataset_sha256': parent_hash,
        'scope': 'one seed, three seen targets, one validation target repeated twice; oracle fixed scene',
        'baseline_training': compact_training(baseline), 'dynamic_training': compact_training(dynamic),
        'comparison': comparison, 'recovery': recovery, 'pairs': pairs,
        'new_policy_trials': 5, 'reused_policy_trials': 5, 'data_and_protocol_audit_passed': True,
        'learned_collection_demonstrated': any(p['dynamic_rollout']['successes'] for p in pairs.values()),
        'robust_policy_accepted': False}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('AUTOPICKER_M4_AUDIT_PASS ' + json.dumps({label: {
        'baseline': [p['baseline_rollout']['successes'], p['baseline_rollout']['requested']],
        'dynamic': [p['dynamic_rollout']['successes'], p['dynamic_rollout']['requested']]}
        for label, p in pairs.items()}), flush=True)


if __name__ == '__main__':
    main()
