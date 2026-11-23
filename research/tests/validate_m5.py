"""Audit clean-action prediction with equal and separately labeled longer budgets."""
import argparse
import json
from pathlib import Path

import numpy as np

from validate_m2 import ROOT, compact_training, comparable_scheduler, digest, training_audit
from validate_m3 import policy_rollout
from autopicker_dp.dataset import load_dataset, SequenceDataset
from autopicker_dp.train import training_inputs


def training_comparison(baseline, sample, dataset, dataset_hash, recovery_path, expected_steps):
    normalizer, sequences, info = training_inputs(dataset, dataset_hash, 'train', 'physical', recovery_path)
    for report, objective, steps in ((baseline, 'epsilon', 3000), (sample, 'sample', expected_steps)):
        if (report.get('recovery_augmentation') != info
                or report['training_windows'] != len(sequences)
                or report['normalizer'] != normalizer.to_dict()
                or report['dataset_manifest'] != dataset.manifest):
            raise ValueError('Recovery lineage, windows or frozen normalizer mismatch')
        if (report['model_config'].get('prediction_type', 'epsilon') != objective
                or report['scheduler']['prediction_type'] != objective):
            raise ValueError('Prediction objective mismatch')
        for key, value in {'seed': 0, 'steps': steps, 'batch_size': 64, 'learning_rate': 1e-4,
                           'dt': .1, 'ema_decay': .995, 'optimizer': 'AdamW'}.items():
            if report[key] != value:
                raise ValueError(f'Training differs from fixed M5 protocol: {key}')
    if expected_steps not in (3000, 30000):
        raise ValueError('Unplanned training budget')
    for key in ('versions', 'normalizer'):
        if baseline[key] != sample[key]:
            raise ValueError(f'Another training factor changed: {key}')
    configs = [{k: v for k, v in r['model_config'].items() if k != 'prediction_type'}
               for r in (baseline, sample)]
    schedulers = [comparable_scheduler({k: v for k, v in r['scheduler'].items()
                                       if k != 'prediction_type'}) for r in (baseline, sample)]
    if configs[0] != configs[1] or schedulers[0] != schedulers[1]:
        raise ValueError('Model or schedule changed beyond the prediction objective')
    # M4 added dynamic-source validation/restoration. M5 uses the original static
    # recovery data and ordinary reset, so those branches are inactive here.
    allowed = {'policy.py', 'train.py', 'augmentation.py', 'rollout.py'}
    for key, value in baseline['source_sha256'].items():
        if key not in allowed and sample['source_sha256'].get(key) != value:
            raise ValueError(f'Unrelated source changed: {key}')
    return {'base_windows': info['base_windows'], 'recovery_windows': info['recovery_windows'],
        'combined_windows': len(sequences), 'sampling': info['sampling'],
        'normalizer_source': info['normalizer_source'], 'recovery_dataset_sha256': info['dataset_sha256'],
        'baseline_steps': 3000, 'sample_steps': expected_steps, 'equal_budget': expected_steps == 3000,
        'source_hash_changes': [key for key in sorted(set(baseline['source_sha256']) | set(sample['source_sha256']))
            if baseline['source_sha256'].get(key) != sample['source_sha256'].get(key)]}


def diagnostic_audit(path, dataset, dataset_hash, training):
    report = json.loads(path.read_text())
    sequences = SequenceDataset(dataset.data, dataset.episode_ends, dataset.episode_indices('train'))
    indices = list(range(0, len(sequences), 8))
    valid_nodes = sum(int(sequences[i]['valid_mask'][1:3].sum()) for i in indices)
    expected = {'dataset_sha256': dataset_hash, 'checkpoint_sha256': training['checkpoint_sha256'],
        'training_split': 'train', 'parent_train_episode_indices': dataset.episode_indices('train'),
        'window_indices': indices, 'noise_seeds': list(range(8)), 'valid_action_nodes': valid_nodes,
        'prediction_type': training['model_config'].get('prediction_type', 'epsilon'),
        'joint_units': dataset.manifest['source_configuration']['joint_units']}
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError('Diagnostic identity or probe protocol mismatch')
    for key in ('mean_seed_std_by_joint', 'mean_abs_bias_by_joint', 'single_sample_mae_by_joint'):
        values = np.asarray(report[key])
        if values.shape != (4,) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Invalid diagnostic statistics')
    return report


def budget_source_comparison(short, longer):
    original, extended = short['source_sha256'], longer['source_sha256']
    # The diagnostic CLI was added after the equal-budget training. It is not
    # imported by training or inference. Existing implementation must be exact.
    if (any(extended.get(key) != value for key, value in original.items())
            or set(extended) - set(original) - {'diagnose.py'}):
        raise ValueError('Longer budget also changed implementation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-longer', action='store_true')
    options = parser.parse_args()
    data, runs = ROOT / 'datasets/autopicker', ROOT / 'runs/autopicker'
    parent = load_dataset(data / 'lowdim_v2')
    parent_hash = digest(data / 'lowdim_v2/manifest.json')
    baseline = training_audit(runs / 'm3_train_recovery_v1', parent, parent_hash, 'physical', 'train')
    baseline_diagnostic = diagnostic_audit(runs / 'm3_train_recovery_v1/m5_diagnostic.json',
                                          parent, parent_hash, baseline)
    baseline_rollouts = {}
    split_specs = [('seen_targets', 'train', parent.episode_indices('train')[:3]),
                   ('validation', 'val', parent.episode_indices('val'))]
    for label, split, indices in split_specs:
        baseline_rollouts[label] = policy_rollout(runs / f'm3_recovery_{split}_v1', parent, parent_hash,
            indices, baseline['normalizer'], baseline['checkpoint_sha256'], split)
    experiments = {}
    specs = [('equal_budget', '', 3000)]
    if options.include_longer:
        specs.append(('longer_budget', '_long', 30000))
    for label, suffix, steps in specs:
        folder = runs / f'm5_train_sample{suffix}_v1'
        training = training_audit(folder, parent, parent_hash, 'physical', 'train')
        comparison = training_comparison(baseline, training, parent, parent_hash,
                                         data / 'm2_recovery_lowdim_v1', steps)
        diagnostic = diagnostic_audit(folder / 'diagnostic.json', parent, parent_hash, training)
        rollouts = {name: policy_rollout(runs / f'm5_sample{suffix}_{split}_v1', parent, parent_hash,
            indices, training['normalizer'], training['checkpoint_sha256'], split)
            for name, split, indices in split_specs}
        experiments[label] = {'training': compact_training(training), 'comparison': comparison,
                              'diagnostic': diagnostic, 'rollouts': rollouts}
    if options.include_longer:
        budget_source_comparison(experiments['equal_budget']['training'],
                                 experiments['longer_budget']['training'])
    report = {'schema': 'autopicker-m5-report-v1', 'dataset_sha256': parent_hash,
        'scope': 'one seed, three seen targets, one validation target repeated twice; oracle fixed scene',
        'baseline_training': compact_training(baseline), 'baseline_diagnostic': baseline_diagnostic,
        'baseline_rollouts': baseline_rollouts, 'baseline_reused_from_m3': True,
        'experiments': experiments, 'new_policy_trials': 5 * len(experiments), 'reused_policy_trials': 5,
        'data_and_protocol_audit_passed': True,
        'learned_collection_demonstrated': any(r['successes'] for e in experiments.values()
                                              for r in e['rollouts'].values()),
        'robust_policy_accepted': False}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('AUTOPICKER_M5_AUDIT_PASS ' + json.dumps({label: {
        split: [r['successes'], r['requested']] for split, r in e['rollouts'].items()}
        for label, e in experiments.items()}), flush=True)


if __name__ == '__main__':
    main()
