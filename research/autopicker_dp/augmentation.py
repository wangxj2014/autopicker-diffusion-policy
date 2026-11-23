"""Audit and append recovery windows while preserving primary data identity."""
from pathlib import Path

import numpy as np

from .dataset import SequenceDataset, _sha256, load_dataset, prepare_dataset
from .episode import Episode
from .recovery import recovery_reset


def reconstruct(dataset):
    """Bind exported arrays and metadata to the audited raw episodes."""
    manifest = dataset.manifest
    counts = [sum(group['split'] == split for group in manifest['target_splits'])
              for split in ('train', 'val', 'test')]
    raw = prepare_dataset(manifest['source_root'], manifest['dt'], counts)
    if (any(manifest.get(key) != value for key, value in raw.manifest.items())
            or not np.array_equal(raw.episode_ends, dataset.episode_ends)
            or any(not np.array_equal(raw.data[key], dataset.data[key]) for key in raw.data)):
        raise ValueError('Export differs from reconstructed raw data')


def validate_recovery(parent, parent_hash, recovery):
    """Reject held-out targets and verify the exact source-state perturbation."""
    manifest = recovery.manifest
    if (manifest['dt'] != parent.manifest['dt']
            or manifest['source_configuration'] != parent.manifest['source_configuration']
            or not manifest['episodes']
            or any(group['split'] != 'train' for group in manifest['target_splits'])):
        raise ValueError('Recovery rate, control configuration or target split mismatch')
    seen, seen_trajectories = set(), set()
    for episode in manifest['episodes']:
        metadata = episode['metadata']
        info = metadata.get('recovery', {})
        index = info.get('source_episode_index')
        if (not isinstance(index, int) or isinstance(index, bool)
                or not 0 <= index < len(parent.manifest['episodes'])):
            raise ValueError('Invalid recovery parent episode index')
        source = parent.manifest['episodes'][index]
        if (episode['split'] != 'train' or source['split'] != 'train'
                or info.get('source_split') != 'train' or info.get('source_dataset_sha256') != parent_hash
                or info.get('source_episode_json_sha256') != source['episode_json_sha256']
                or info.get('source_raw_sha256') != source['raw_sha256']
                or metadata['target_path'] != source['metadata']['target_path']
                or episode['episode_json_sha256'] in seen
                or episode['raw_sha256']['states.npz'] in seen_trajectories):
            raise ValueError('Recovery target, parent identity or duplicate source mismatch')
        seen.add(episode['episode_json_sha256'])
        seen_trajectories.add(episode['raw_sha256']['states.npz'])
        raw_source = Episode.load(Path(parent.manifest['source_root']) / source['path'])
        if _sha256(Path(parent.manifest['source_root']) / source['path'] / 'episode.json') != source['episode_json_sha256']:
            raise ValueError('Parent raw episode identity changed')
        try:
            reset, expected = recovery_reset(raw_source, info['active_indices'], metadata['joint_lower'],
                                             metadata['joint_upper'], info['delta_q'])
        except (KeyError, TypeError) as error:
            raise ValueError('Missing or malformed recovery provenance') from error
        if (any(info.get(key) != value for key, value in expected.items())
                or metadata.get('reset_joint_positions') != reset.tolist()):
            raise ValueError('Recovery source state or reset provenance mismatch')
        recorded = Episode.load(Path(manifest['source_root']) / episode['path'])
        settled = np.asarray(info.get('settled_q'), dtype=float)
        if (settled.shape != (4,) or not np.isfinite(settled).all()
                or not np.allclose(settled, recorded.arrays()['q'][0], atol=1e-8, rtol=0)):
            raise ValueError('Recovery settled state differs from recorded initial state')


def recovery_sequences(parent, parent_hash, recovery_path):
    recovery_path = Path(recovery_path)
    recovery = load_dataset(recovery_path)
    reconstruct(parent)
    reconstruct(recovery)
    validate_recovery(parent, parent_hash, recovery)
    count = len(parent.episode_ends)
    indices = parent.episode_indices('train') + list(range(count, count + len(recovery.episode_ends)))
    arrays = {key: np.concatenate((parent.data[key], recovery.data[key])) for key in ('obs', 'action')}
    ends = np.r_[parent.episode_ends, parent.episode_ends[-1] + recovery.episode_ends]
    sequences = SequenceDataset(arrays, ends, indices)
    base_windows = len(SequenceDataset(parent.data, parent.episode_ends, parent.episode_indices('train')))
    return sequences, {'dataset_path': str(recovery_path.resolve()),
        'dataset_sha256': _sha256(recovery_path / 'manifest.json'), 'dataset_manifest': recovery.manifest,
        'parent_dataset_sha256': parent_hash, 'base_episode_indices': parent.episode_indices('train'),
        'recovery_episode_indices': list(range(len(recovery.episode_ends))),
        'base_windows': base_windows, 'recovery_windows': int(recovery.episode_ends[-1]),
        'sampling': 'uniform over combined window anchors',
        'normalizer_source': 'primary dataset train only; frozen before augmentation'}
