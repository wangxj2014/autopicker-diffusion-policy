"""Audit the fixed-budget physical-normalization ablation and recovery pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research'))
from autopicker_dp.dataset import load_dataset, _configuration
from autopicker_dp.episode import Episode
from autopicker_dp.recovery import recovery_reset, recovery_sources
from autopicker_dp.rollout import verified_sources
from autopicker_dp.train import training_normalizer
from validate_m1 import rollout_report


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_audit(folder, dataset, dataset_hash, normalization, split):
    import torch
    report = json.loads((folder / 'report.json').read_text())
    offline = json.loads((folder / 'offline.json').read_text())
    indices = dataset.episode_indices('train') if split == 'train' else list(range(len(dataset.episode_ends)))
    expected = training_normalizer(dataset, indices, normalization).to_dict()
    checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
    if (report['dataset_sha256'] != dataset_hash or report['split'] != split
            or report.get('action_normalization', 'data') != normalization
            or report['normalizer'] != expected or report['checkpoint_reload_max_error'] != 0
            or digest(folder / 'checkpoint.pt') != report['checkpoint_sha256']
            or checkpoint['metadata']['normalizer'] != expected
            or any(checkpoint['metadata'][key] != value for key, value in report.items()
                   if key not in ('checkpoint_reload_max_error', 'checkpoint_sha256'))
            or offline['dataset_sha256'] != dataset_hash
            or offline['checkpoint_sha256'] != report['checkpoint_sha256']
            or offline['seed'] != 0 or offline['training_split'] != split):
        raise ValueError(f'Training/checkpoint/normalization identity mismatch: {folder}')
    report['offline'] = offline
    report['model_config'] = checkpoint['model_config']
    return report


def compact_training(report):
    keys = ('split', 'seed', 'steps', 'batch_size', 'learning_rate', 'initial_loss_mean',
            'final_loss_mean', 'checkpoint_sha256', 'checkpoint_reload_max_error', 'wall_time_s',
            'training_windows', 'model_config', 'scheduler', 'source_sha256', 'offline')
    return {key: report[key] for key in keys}


def comparable_scheduler(configuration):
    # Diffusers serializes a set of names in process-dependent order. Every
    # numeric/behavioral setting must still compare exactly.
    return {**configuration, '_use_default_values': sorted(configuration.get('_use_default_values', []))}


def rollout_protocol(folder, dataset, indices, normalizer):
    summary = json.loads((folder / 'summary.json').read_text())
    if [result['source_episode_index'] for result in summary['results']] != indices:
        raise ValueError('Rollout selected different source episodes')
    diagnostics = []
    for ordinal, index in enumerate(indices):
        source = dataset.manifest['episodes'][index]
        directory = folder / f'episode_{ordinal:04d}'
        failed = (directory / 'failure.json').exists()
        document = json.loads((directory / ('failure.json' if failed else 'episode.json')).read_text())
        metadata = document['metadata']
        if (metadata['reset_joint_positions'] != source['metadata']['reset_joint_positions']
                or _configuration(metadata) != dataset.manifest['source_configuration']
                or metadata['source_episode_index'] != index or metadata['source_split'] != source['split']
                or metadata['rollout_seed'] != 0 or metadata['mode'] != 'policy'
                or metadata['policy_dt'] != .1):
            raise ValueError('Rollout reset, source selection or control configuration changed')
        if not failed:
            result = summary['results'][ordinal]
            if (result['duration_s'] > 35. + 1e-8 or
                    result['reason'] == 'Episode timed out before collect' and abs(result['duration_s'] - 35.) > 1e-8):
                raise ValueError('Observed rollout timeout budget differs from 35 seconds')
        requests = metadata.get('inference_requests', [])
        if requests:
            obs = np.array([request['observation'][-1] for request in requests])
            lower = np.array(normalizer['fields']['obs']['min'])[4:8]
            upper = np.array(normalizer['fields']['obs']['max'])[4:8]
            outside = (obs[:, 4:8] < lower - 1e-5) | (obs[:, 4:8] > upper + 1e-5)
            diagnostics.append({'target_path': source['metadata']['target_path'], 'requests': len(requests),
                'velocity_observations_outside_fit_range_by_joint': outside.sum(0).tolist()})
    return diagnostics


def recovery_audit(folder, exported, replay_folder, baseline, baseline_hash):
    summary = json.loads((folder / 'summary.json').read_text())
    sources = recovery_sources(baseline.manifest)
    expected_trials = [(index, sign) for index in sources for sign in (-1, 1)]
    if (not summary['completed'] or summary['dataset_sha256'] != baseline_hash
            or summary['requested'] != len(expected_trials) or summary['attempted'] != len(expected_trials)
            or len(summary['results']) != len(expected_trials)):
        raise ValueError('Incomplete recovery pilot')
    successes, rows = 0, []
    for ordinal, (index, sign) in enumerate(expected_trials):
        result = summary['results'][ordinal]
        source_spec = baseline.manifest['episodes'][index]
        source = Episode.load(Path(baseline.manifest['source_root']) / source_spec['path'])
        directory = folder / f'episode_{ordinal:04d}'
        amplitude = np.array([np.deg2rad(3.), .025, .02, np.deg2rad(5.)])
        failed = (directory / 'failure.json').exists()
        document = json.loads((directory / ('failure.json' if failed else 'episode.json')).read_text())
        provenance = document.get('recovery', document.get('metadata', {}).get('recovery'))
        if (provenance is None or provenance['source_split'] != 'train'
                or provenance['source_episode_index'] != index or provenance['source_dataset_sha256'] != baseline_hash
                or provenance['source_episode_json_sha256'] != source_spec['episode_json_sha256']
                or provenance['source_raw_sha256'] != source_spec['raw_sha256']
                or provenance['delta_q'] != (sign * amplitude).tolist()
                or result['sign'] != sign or result['source_episode_index'] != index
                or result['target_path'] != source.metadata['target_path']):
            raise ValueError('Recovery source identity, delta or trial order mismatch')
        if failed:
            target = document.get('target_path', document.get('metadata', {}).get('target_path'))
            if result['success'] or target != result['target_path'] or document['reason'] != result['reason']:
                raise ValueError('Recovery failed trial outcome mismatch')
            rows.append({**result, 'recovery': provenance})
            continue
        episode = Episode.load(directory)
        provenance = episode.metadata['recovery']
        reset, expected = recovery_reset(source, provenance['active_indices'], source.metadata['joint_lower'],
                                         source.metadata['joint_upper'], sign * amplitude)
        if (any(provenance[key] != value for key, value in expected.items())
                or provenance['source_split'] != 'train' or provenance['source_episode_index'] != index
                or provenance['source_dataset_sha256'] != baseline_hash
                or provenance['source_episode_json_sha256'] != source_spec['episode_json_sha256']
                or provenance['source_raw_sha256'] != source_spec['raw_sha256']
                or result['sign'] != sign or result['source_episode_index'] != index
                or episode.metadata['target_path'] != source.metadata['target_path']
                or result['target_path'] != source.metadata['target_path']
                or episode.metadata['reset_joint_positions'] != reset.tolist()
                or result['success'] != episode.outcome['success']
                or episode.metadata['max_inactive_joint_drift'] > .003):
            raise ValueError('Recovery provenance, reset or split mismatch')
        successes += int(episode.outcome['success'])
        rows.append({'target_path': episode.metadata['target_path'], 'sign': sign,
                     'source_episode_index': index, 'recovery': provenance, **episode.audit()})
    if summary['successes'] != successes:
        raise ValueError('Recovery denominator mismatch')
    report = {'path': str(folder), 'requested': len(expected_trials), 'successes': successes,
              'scope': summary['scope'], 'episodes': rows, 'export_accepted': False,
              'used_for_training_in_this_ablation': False}
    if successes != len(expected_trials):
        if exported.exists() or replay_folder.exists():
            raise ValueError('Failed recovery pilot must not silently export a successful subset')
        return report
    dataset = load_dataset(exported)
    verified_sources(exported)
    if (len(dataset.episode_ends) != len(expected_trials)
            or Path(dataset.manifest['source_root']).resolve() != folder.resolve()
            or [episode['episode_json_sha256'] for episode in dataset.manifest['episodes']]
            != [digest(folder / f'episode_{index:04d}/episode.json') for index in range(len(expected_trials))]
            or any(group['split'] != 'train' for group in dataset.manifest['target_splits'])
            or {group['target_path'] for group in dataset.manifest['target_splits']}
            != {baseline.manifest['episodes'][index]['metadata']['target_path'] for index in sources}):
        raise ValueError('Recovery export leaks target splits or omits attempts')
    replay = rollout_report(replay_folder, digest(exported / 'manifest.json'))
    replay_summary = json.loads((replay_folder / 'summary.json').read_text())
    if (replay['successes'] != len(expected_trials) or replay['requested'] != len(expected_trials)
            or [result['source_episode_index'] for result in replay_summary['results']] != list(range(len(expected_trials)))
            or [episode['target_path'] for episode in replay['episodes']] != [row['target_path'] for row in rows]):
        raise ValueError('Recovery converted replay acceptance failed')
    report.update(export_accepted=True, export_rows=dataset.manifest['rows'],
                  export_sha256=digest(exported / 'manifest.json'), converted_replay=replay)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-recovery', action='store_true')
    options = parser.parse_args()
    dataset_path = ROOT / 'datasets/autopicker/lowdim_v2'
    dataset = load_dataset(dataset_path)
    dataset_hash = digest(dataset_path / 'manifest.json')
    runs = ROOT / 'runs/autopicker'
    pairs = {}
    for label, split, baseline_name, physical_name, base_rollout, physical_rollout in (
        ('all-pilot', 'all-pilot', 'm1_overfit_v2', 'm2_overfit_physical_v1',
         'm1_policy_overfit_v2', 'm2_policy_overfit_physical_v1'),
        ('validation', 'train', 'm1_train_v2', 'm2_train_physical_v1',
         'm1_policy_val_v2', 'm2_policy_val_physical_v1')):
        base = training_audit(runs / baseline_name, dataset, dataset_hash, 'data', split)
        physical = training_audit(runs / physical_name, dataset, dataset_hash, 'physical', split)
        if comparable_scheduler(base['scheduler']) != comparable_scheduler(physical['scheduler']):
            raise ValueError('Ablation changed scheduler configuration')
        for key in ('seed', 'steps', 'batch_size', 'learning_rate', 'dt', 'model_config',
                    'ema_decay', 'optimizer', 'training_windows', 'versions'):
            if base[key] != physical[key]:
                raise ValueError(f'Ablation changed another training factor: {key}')
        if base['normalizer']['fields']['obs'] != physical['normalizer']['fields']['obs']:
            raise ValueError('Ablation changed observation normalization')
        old = rollout_report(runs / base_rollout, dataset_hash)
        new = rollout_report(runs / physical_rollout, dataset_hash)
        indices = list(range(5)) if split == 'all-pilot' else dataset.episode_indices('val')
        old['velocity_support'] = rollout_protocol(runs / base_rollout, dataset, indices, base['normalizer'])
        new['velocity_support'] = rollout_protocol(runs / physical_rollout, dataset, indices, physical['normalizer'])
        if (old['checkpoint_sha256'] != base['checkpoint_sha256']
                or new['checkpoint_sha256'] != physical['checkpoint_sha256']
                or old['seed'] != new['seed'] or old['requested'] != new['requested']
                or [e['target_path'] for e in old['episodes']] != [e['target_path'] for e in new['episodes']]):
            raise ValueError('Ablation rollout protocol or checkpoint changed')
        pairs[label] = {'baseline_training': compact_training(base), 'physical_training': compact_training(physical),
                        'baseline_rollout': old, 'physical_rollout': new}
    report = {'schema': 'autopicker-m2-report-v1', 'dataset_sha256': dataset_hash,
              'scope': 'one seed, five oracle pilot targets; normalization ablation only',
              'pairs': pairs, 'data_and_protocol_audit_passed': True,
              'robust_policy_accepted': False,
              'physical_action_lower': dataset.manifest['source_configuration']['joint_lower'],
              'physical_action_upper': dataset.manifest['source_configuration']['joint_upper']}
    if options.include_recovery:
        report['recovery'] = recovery_audit(ROOT / 'datasets/autopicker/m2_recovery_raw_v1',
            ROOT / 'datasets/autopicker/m2_recovery_lowdim_v1', runs / 'm2_recovery_replay_v1', dataset, dataset_hash)
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('AUTOPICKER_M2_AUDIT_PASS ' + json.dumps({label: {
        'baseline': [pair['baseline_rollout']['successes'], pair['baseline_rollout']['requested']],
        'physical': [pair['physical_rollout']['successes'], pair['physical_rollout']['requested']]}
        for label, pair in pairs.items()}), flush=True)


if __name__ == '__main__':
    main()
