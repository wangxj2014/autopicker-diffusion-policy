"""Separate seed variation and mean action bias on fixed parent-train observations."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def prediction_statistics(predictions, target, valid):
    predictions, target, valid = np.asarray(predictions), np.asarray(target), np.asarray(valid, dtype=bool)
    if (predictions.ndim != 4 or predictions.shape[0] < 2 or predictions.shape[1:] != target.shape
            or target.shape[-1] != 4 or valid.shape != target.shape[:-1] or not valid.any()
            or not np.isfinite(predictions).all() or not np.isfinite(target).all()):
        raise ValueError('Expected finite seed x window x node x joint predictions and valid mask')
    return {'valid_action_nodes': int(valid.sum()),
            'mean_seed_std_by_joint': predictions.std(0)[valid].mean(0).tolist(),
            'mean_abs_bias_by_joint': np.abs(predictions.mean(0) - target)[valid].mean(0).tolist(),
            'single_sample_mae_by_joint': np.abs(predictions - target)[:, valid].mean((0, 1)).tolist()}


def main():
    import torch
    from autopicker_dp.dataset import load_dataset, SequenceDataset, Normalizer, _sha256
    from autopicker_dp.train import load_policy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError(options.output)
    torch.set_num_threads(2)
    dataset = load_dataset(options.dataset)
    policy, metadata = load_policy(options.checkpoint, options.upstream, 'cuda')
    digest = _sha256(options.dataset / 'manifest.json')
    if metadata['dataset_sha256'] != digest or metadata['split'] != 'train':
        raise ValueError('Diagnostic requires the matching train checkpoint')
    normalizer = Normalizer.from_dict(metadata['normalizer'])
    sequences = SequenceDataset(dataset.data, dataset.episode_ends, dataset.episode_indices('train'))
    indices = list(range(0, len(sequences), 8))
    batch = [sequences[i] for i in indices]
    obs = np.stack([item['obs'] for item in batch])
    target = np.stack([item['action'][1:3] for item in batch])
    valid = np.stack([item['valid_mask'][1:3] for item in batch])
    tensor = torch.as_tensor(normalizer.normalize(obs, key='obs'), device='cuda')
    predictions = np.stack([normalizer.denormalize(
        policy.action_chunk(policy.predict(tensor, seed)).cpu().numpy(), key='action') for seed in range(8)])
    report = {'scope': 'teacher-forced parent-train states; seed variation is not collection performance',
        'dataset_sha256': digest, 'checkpoint_sha256': _sha256(options.checkpoint),
        'training_split': 'train', 'parent_train_episode_indices': dataset.episode_indices('train'),
        'window_indices': indices, 'noise_seeds': list(range(8)),
        'prediction_type': policy.config['prediction_type'],
        'joint_units': dataset.manifest['source_configuration']['joint_units'],
        **prediction_statistics(predictions, target, valid)}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('AUTOPICKER_PREDICTION_DIAGNOSTIC_COMPLETE ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
