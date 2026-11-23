"""Audit recovery-only augmentation and paired closed-loop outcomes."""
import argparse
import json
from pathlib import Path

from validate_m2 import (ROOT, compact_training, comparable_scheduler, digest,
                         recovery_audit, rollout_protocol, training_audit)
from validate_m1 import rollout_report
from autopicker_dp.dataset import load_dataset
from autopicker_dp.train import training_inputs


def training_pair(baseline, augmented, dataset, dataset_hash, recovery_path):
    normalizer, sequences, info = training_inputs(dataset, dataset_hash, 'train', 'physical', recovery_path)
    if (baseline.get('recovery_augmentation') is not None
            or augmented.get('recovery_augmentation') != info
            or baseline['training_windows'] != info['base_windows']
            or augmented['training_windows'] != len(sequences)
            or baseline['normalizer'] != normalizer.to_dict()
            or augmented['normalizer'] != normalizer.to_dict()
            or baseline['dataset_manifest'] != dataset.manifest
            or augmented['dataset_manifest'] != dataset.manifest):
        raise ValueError('Augmentation lineage, window count or frozen normalizer mismatch')
    if comparable_scheduler(baseline['scheduler']) != comparable_scheduler(augmented['scheduler']):
        raise ValueError('Augmentation changed scheduler')
    for key in ('seed', 'steps', 'batch_size', 'learning_rate', 'dt', 'model_config',
                'ema_decay', 'optimizer', 'versions'):
        if baseline[key] != augmented[key]:
            raise ValueError(f'Augmentation changed another training factor: {key}')
    for key, value in {'seed': 0, 'steps': 3000, 'batch_size': 64, 'learning_rate': 1e-4,
                       'dt': .1, 'ema_decay': .995, 'optimizer': 'AdamW'}.items():
        if baseline[key] != value:
            raise ValueError(f'Training differs from fixed M3 protocol: {key}')
    # train.py adds the optional data path. recovery.py changed failure handling
    # after M2 training; it is not imported by the policy or sampling loop.
    for key, value in baseline['source_sha256'].items():
        if key not in ('train.py', 'recovery.py') and augmented['source_sha256'].get(key) != value:
            raise ValueError(f'Unrelated training source changed: {key}')
    return {'base_windows': info['base_windows'], 'recovery_windows': info['recovery_windows'],
            'combined_windows': len(sequences),
            'recovery_window_fraction': info['recovery_windows'] / len(sequences),
            'sampling': info['sampling'], 'normalizer_source': info['normalizer_source'],
            'recovery_dataset_sha256': info['dataset_sha256'],
            'normalizer': normalizer.to_dict()}


def policy_rollout(folder, dataset, dataset_hash, indices, normalizer, checkpoint_hash, split):
    report = rollout_report(folder, dataset_hash)
    report['velocity_support'] = rollout_protocol(folder, dataset, indices, normalizer)
    if (report['checkpoint_sha256'] != checkpoint_hash or report['seed'] != 0
            or report['split'] != split or report['mode'] != 'policy'
            or report['requested'] != len(indices)
            or [row['target_path'] for row in report['episodes']] !=
            [dataset.manifest['episodes'][index]['metadata']['target_path'] for index in indices]):
        raise ValueError('Rollout model, protocol or target identity mismatch')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    dataset_path = ROOT / 'datasets/autopicker/lowdim_v2'
    recovery_path = ROOT / 'datasets/autopicker/m2_recovery_lowdim_v1'
    runs = ROOT / 'runs/autopicker'
    dataset = load_dataset(dataset_path)
    dataset_hash = digest(dataset_path / 'manifest.json')
    baseline = training_audit(runs / 'm2_train_physical_v1', dataset, dataset_hash, 'physical', 'train')
    augmented = training_audit(runs / 'm3_train_recovery_v1', dataset, dataset_hash, 'physical', 'train')
    augmentation = training_pair(baseline, augmented, dataset, dataset_hash, recovery_path)
    recovery = recovery_audit(ROOT / 'datasets/autopicker/m2_recovery_raw_v1', recovery_path,
                              runs / 'm2_recovery_replay_v1', dataset, dataset_hash)
    if not recovery['export_accepted'] or recovery['export_sha256'] != augmentation['recovery_dataset_sha256']:
        raise ValueError('Recovery pilot was not fully accepted')
    recovery['used_for_training_in_this_ablation'] = True
    pairs = {}
    for label, split, indices, old_name, new_name in (
        ('seen_targets', 'train', dataset.episode_indices('train')[:3],
         'm3_baseline_train_v1', 'm3_recovery_train_v1'),
        ('validation', 'val', dataset.episode_indices('val'),
         'm2_policy_val_physical_v1', 'm3_recovery_val_v1')):
        old = policy_rollout(runs / old_name, dataset, dataset_hash, indices, baseline['normalizer'],
                             baseline['checkpoint_sha256'], split)
        new = policy_rollout(runs / new_name, dataset, dataset_hash, indices, augmented['normalizer'],
                             augmented['checkpoint_sha256'], split)
        pairs[label] = {'source_episode_indices': indices,
                        'baseline_rollout': old, 'augmented_rollout': new,
                        'baseline_reused_from_m2': label == 'validation'}
    report = {'schema': 'autopicker-m3-report-v1', 'dataset_sha256': dataset_hash,
              'scope': 'one seed, three seen targets, one validation target repeated twice; oracle fixed scene',
              'baseline_training': compact_training(baseline), 'augmented_training': compact_training(augmented),
              'augmentation': augmentation, 'recovery': recovery, 'pairs': pairs,
              'new_policy_trials': 8, 'reused_policy_trials': 2,
              'data_and_protocol_audit_passed': True,
              'learned_collection_demonstrated': any(pair['augmented_rollout']['successes'] for pair in pairs.values()),
              'robust_policy_accepted': False}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('AUTOPICKER_M3_AUDIT_PASS ' + json.dumps({label: {
        'baseline': [pair['baseline_rollout']['successes'], pair['baseline_rollout']['requested']],
        'augmented': [pair['augmented_rollout']['successes'], pair['augmented_rollout']['requested']]}
        for label, pair in pairs.items()}), flush=True)


if __name__ == '__main__':
    main()
