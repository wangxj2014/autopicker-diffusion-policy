"""Teacher-forced action errors; these are not closed-loop task success rates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def action_metrics(prediction, target, valid, lower, upper):
    prediction, target, valid = np.asarray(prediction), np.asarray(target), np.asarray(valid, dtype=bool)
    if (prediction.shape != target.shape or prediction.shape[-1] != 4
            or valid.shape != prediction.shape[:-1] or not valid.any()
            or not np.isfinite(prediction).all() or not np.isfinite(target).all()):
        raise ValueError('Invalid action metric inputs')
    error = (prediction - target)[valid]
    outside = ((prediction < np.asarray(lower) - 1e-7) | (prediction > np.asarray(upper) + 1e-7)).any(-1)
    return {'valid_action_nodes': int(valid.sum()),
            'mae_by_joint': np.abs(error).mean(0).tolist(),
            'rmse_by_joint': np.sqrt(np.square(error).mean(0)).tolist(),
            'max_abs_error_by_joint': np.abs(error).max(0).tolist(),
            'out_of_limits_nodes': int(outside[valid].sum())}


def main():
    import torch
    from autopicker_dp.dataset import load_dataset, Normalizer, SequenceDataset
    from autopicker_dp.train import load_policy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError(options.output)
    torch.set_num_threads(2)
    dataset = load_dataset(options.dataset)
    policy, metadata = load_policy(options.checkpoint, options.upstream, options.device)
    digest = hashlib.sha256((options.dataset / 'manifest.json').read_bytes()).hexdigest()
    if metadata['dataset_sha256'] != digest:
        raise ValueError('Checkpoint and evaluation dataset identities differ')
    normalizer = Normalizer.from_dict(metadata['normalizer'])
    config = dataset.manifest['source_configuration']
    report = {'scope': 'teacher-forced oracle observations; not task success or generalization evidence',
              'training_split': metadata['split'], 'seed': options.seed,
              'dataset_sha256': digest, 'checkpoint_sha256': hashlib.sha256(options.checkpoint.read_bytes()).hexdigest(),
              'joint_names': config['joint_names'], 'joint_units': config['joint_units'],
              'splits': {}}
    splits = ['all-pilot'] if metadata['split'] == 'all-pilot' else ['train', 'val', 'test']
    for split in splits:
        indices = (list(range(len(dataset.episode_ends))) if split == 'all-pilot'
                   else dataset.episode_indices(split))
        sequences = SequenceDataset(dataset.data, dataset.episode_ends, indices)
        predictions, targets, masks = [], [], []
        for start in range(0, len(sequences), 64):
            batch = [sequences[index] for index in range(start, min(start + 64, len(sequences)))]
            obs = normalizer.normalize(np.stack([item['obs'] for item in batch]), key='obs')
            prediction = policy.predict(torch.as_tensor(obs, device=options.device), options.seed + start)
            predictions.append(normalizer.denormalize(policy.action_chunk(prediction).cpu().numpy(), key='action'))
            targets.append(np.stack([item['action'][1:3] for item in batch]))
            masks.append(np.stack([item['valid_mask'][1:3] for item in batch]))
        report['splits'][split] = action_metrics(np.concatenate(predictions), np.concatenate(targets),
                                                np.concatenate(masks), config['joint_lower'], config['joint_upper'])
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print('AUTOPICKER_OFFLINE_EVALUATION_COMPLETE ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
