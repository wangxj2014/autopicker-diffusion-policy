"""Bounded low-dimensional diffusion training with reproducible checkpoints."""
import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research'))
from autopicker_dp.policy import DiffusionPolicy, UPSTREAM_REVISION


def add_velocity_observation_noise(observation, normalized_std):
    """Add Gaussian noise only to qdot channels of normalized Bx2x15 observations.

    normalized_std is a four-element standard deviation already converted
    from physical rad/s using the training normalizer. This is observation
    corruption regularization, not a substitute for expert recovery data.
    """
    if (not torch.is_tensor(observation) or observation.ndim != 3
            or observation.shape[1:] != (2, 15)
            or not torch.isfinite(observation).all()):
        raise ValueError('Expected finite normalized Bx2x15 observations')
    std = torch.as_tensor(normalized_std, dtype=observation.dtype, device=observation.device)
    if std.shape != (4,) or not torch.isfinite(std).all() or torch.any(std < 0):
        raise ValueError('Velocity noise standard deviation must be four finite nonnegative values')
    if not torch.any(std):
        return observation
    result = observation.clone()
    result[..., 4:8] += torch.randn_like(result[..., 4:8]) * std.view(1, 1, 4)
    return result


def save_checkpoint(path, policy, ema, optimizer, metadata):
    torch.save({'schema': 'autopicker-diffusion-v1', 'upstream_revision': UPSTREAM_REVISION,
                'model_config': policy.config, 'model': policy.state_dict(), 'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(), 'metadata': metadata,
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, path)


def load_policy(path, upstream, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint['schema'] != 'autopicker-diffusion-v1' or checkpoint['upstream_revision'] != UPSTREAM_REVISION:
        raise ValueError('Unsupported policy checkpoint')
    policy = DiffusionPolicy(upstream, **checkpoint['model_config']).to(device).eval()
    policy.load_state_dict(checkpoint['ema'])
    if 'scheduler' in checkpoint['metadata']:
        from diffusers import DDIMScheduler
        policy.scheduler = DDIMScheduler.from_config(checkpoint['metadata']['scheduler'])
    if policy.scheduler.config.prediction_type != policy.config['prediction_type']:
        raise ValueError('Checkpoint scheduler and model prediction objective disagree')
    return policy, checkpoint['metadata']


def training_normalizer(dataset, indices, action_normalization):
    from autopicker_dp.dataset import Normalizer
    normalizer = Normalizer.fit(dataset.data, dataset.episode_ends, indices)
    if action_normalization == 'physical':
        configuration = dataset.manifest['source_configuration']
        return normalizer.with_action_limits(configuration['joint_lower'], configuration['joint_upper'])
    if action_normalization != 'data':
        raise ValueError('Unknown action normalization')
    return normalizer


def training_inputs(dataset, dataset_hash, split, action_normalization, recovery_path=None):
    from autopicker_dp.dataset import SequenceDataset
    if split not in ('train', 'all-pilot'):
        raise ValueError('Unknown training split')
    if recovery_path is not None and (split != 'train' or action_normalization != 'physical'):
        raise ValueError('Recovery augmentation requires train split and physical action normalization')
    indices = dataset.episode_indices('train') if split == 'train' else list(range(len(dataset.episode_ends)))
    normalizer = training_normalizer(dataset, indices, action_normalization)
    augmentation = None
    if recovery_path is not None:
        from autopicker_dp.augmentation import recovery_sequences
        sequences, augmentation = recovery_sequences(dataset, dataset_hash, recovery_path)
    else:
        sequences = SequenceDataset(dataset.data, dataset.episode_ends, indices)
    return normalizer, sequences, augmentation


def main():
    from autopicker_dp.dataset import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--recovery-dataset', type=Path,
                        help='Append audited recovery windows; requires train split and physical normalization')
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'all-pilot'), default='train')
    parser.add_argument('--action-normalization', choices=('data', 'physical'), default='data',
                        help='Action affine bounds: selected data extrema or physical joint limits')
    parser.add_argument('--prediction-type', choices=('epsilon', 'sample'), default='epsilon',
                        help='Predict diffusion noise or the clean normalized action sequence')
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--qdot-noise-std-rad-s', type=float, default=0.0,
                        help='Train-only Gaussian qdot observation noise in rad/s; 0 disables it')
    options = parser.parse_args()
    if (options.steps < 1 or options.batch_size < 1 or not 0 < options.learning_rate < 1
            or not np.isfinite(options.qdot_noise_std_rad_s) or options.qdot_noise_std_rad_s < 0):
        parser.error('Require positive training budget, learning rate below one, and finite nonnegative qdot noise')
    dataset = load_dataset(options.dataset)
    dataset_hash = hashlib.sha256((options.dataset / 'manifest.json').read_bytes()).hexdigest()
    normalizer, sequences, augmentation = training_inputs(
        dataset, dataset_hash, options.split, options.action_normalization, options.recovery_dataset)
    batches = [normalizer.normalize(sequences[index]) for index in range(len(sequences))]
    tensors = {key: torch.as_tensor(np.stack([item[key] for item in batches]), device=options.device)
               for key in ('obs', 'action', 'valid_mask')}
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    torch.set_num_threads(4)
    policy = DiffusionPolicy(options.upstream, prediction_type=options.prediction_type).to(options.device)
    ema = deepcopy(policy).eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=options.learning_rate, weight_decay=1e-6)
    options.output.mkdir(parents=True, exist_ok=False)
    if options.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    qdot_noise_normalized_std = (options.qdot_noise_std_rad_s
                                  * normalizer.fields['obs']['scale'][4:8]).astype(np.float32)
    metadata = {'scope': 'all-pilot overfit diagnostic; no held-out evaluation' if options.split == 'all-pilot'
                else 'target-grouped train-only fitting; pilot-sized split',
                'split': options.split, 'seed': options.seed, 'steps': options.steps,
                'batch_size': options.batch_size, 'learning_rate': options.learning_rate,
                'normalizer': normalizer.to_dict(), 'action_normalization': options.action_normalization,
                'dt': dataset.manifest['dt'],
                'dataset_manifest': dataset.manifest,
                'dataset_sha256': dataset_hash,
                'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
                'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (ROOT / 'research/autopicker_dp').glob('*.py')},
                'versions': {**{name: version(name) for name in
                                ('numpy', 'torch', 'diffusers', 'zarr', 'numcodecs', 'einops')},
                             'python': sys.version, 'cuda': torch.version.cuda},
                'scheduler': dict(policy.scheduler.config), 'ema_decay': .995,
                'observation_augmentation': {
                    'kind': 'train_only_independent_gaussian_qdot_noise',
                    'enabled': bool(options.qdot_noise_std_rad_s > 0),
                    'qdot_noise_std_rad_s': float(options.qdot_noise_std_rad_s),
                    'normalized_std': qdot_noise_normalized_std.tolist(),
                    'channels': 'obs[..., 4:8] only; q, goal, receiver and attached unchanged',
                    'evaluation_augmentation': False},
                'optimizer': 'AdamW', 'training_windows': len(sequences)}
    if augmentation is not None:
        metadata['recovery_augmentation'] = augmentation
    started = time.monotonic()
    losses = []
    with (options.output / 'metrics.jsonl').open('w', encoding='utf-8') as stream:
        for step in range(1, options.steps + 1):
            selected = torch.randint(0, len(sequences), (options.batch_size,), device=options.device)
            batch_observation = tensors['obs'][selected].float()
            if options.qdot_noise_std_rad_s > 0:
                batch_observation = add_velocity_observation_noise(
                    batch_observation, qdot_noise_normalized_std)
            loss = policy.loss(batch_observation, tensors['action'][selected].float(),
                               tensors['valid_mask'][selected].bool())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
            optimizer.step()
            with torch.no_grad():
                decay = min(.995, (1. + step) / (10. + step))
                for averaged, current in zip(ema.parameters(), policy.parameters()):
                    averaged.lerp_(current, 1. - decay)
            losses.append(float(loss.detach()))
            if step % 100 == 0 or step == options.steps:
                metrics = {'step': step, 'loss_mean_last100': float(np.mean(losses[-100:])),
                           'wall_time_s': time.monotonic() - started}
                stream.write(json.dumps(metrics) + '\n')
                stream.flush()
                print('TRAIN_PROGRESS ' + json.dumps(metrics), flush=True)
    metadata.update(wall_time_s=time.monotonic() - started, initial_loss_mean=float(np.mean(losses[:100])),
                    final_loss_mean=float(np.mean(losses[-100:])),
                    peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if options.device.startswith('cuda') else 0)
    checkpoint = options.output / 'checkpoint.pt'
    save_checkpoint(checkpoint, policy, ema, optimizer, metadata)
    reloaded, _ = load_policy(checkpoint, options.upstream, options.device)
    probe = tensors['obs'][:4].float()
    expected = ema.predict(probe, seed=1234)
    actual = reloaded.predict(probe, seed=1234)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    metadata['checkpoint_reload_max_error'] = float((actual - expected).abs().max())
    metadata['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (options.output / 'report.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    print('AUTOPICKER_TRAIN_AND_RELOAD_PASS ' + json.dumps({key: metadata[key] for key in
          ('scope', 'initial_loss_mean', 'final_loss_mean', 'checkpoint_reload_max_error', 'wall_time_s')}), flush=True)


if __name__ == '__main__':
    main()
