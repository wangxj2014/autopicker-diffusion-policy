"""Convert audited expert episodes to fixed-rate, oracle lowdim sequences.

Storage uses Zarr v2, imported only by build_dataset and load_dataset. Conversion,
normalization and sequence sampling require only NumPy and ordinary Python.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autopicker_dp.episode import Episode

SCHEMA = 'autopicker-lowdim-v1'
OBS_FIELDS = [('q', 4), ('qdot', 4), ('goal_base_m', 3),
              ('receiver_base_m', 3), ('attached', 1)]
ACTION_SEMANTICS = 'applied absolute joint position command at observation time + dt'
SPLITS = ('train', 'val', 'test')


def _base_inverse(metadata):
    transform = np.asarray(metadata.get('world_from_base'), dtype=np.float64)
    if (metadata.get('transform_convention') != 'USD row vectors, meters'
            or transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[:, 3], [0., 0., 0., 1.], rtol=0, atol=1e-10)):
        raise ValueError('Expected finite affine USD row-vector transform in meters')
    try:
        inverse = np.linalg.inv(transform)
    except np.linalg.LinAlgError as error:
        raise ValueError('Base transform must be invertible') from error
    if not np.isfinite(inverse).all():
        raise ValueError('Base transform inverse is nonfinite')
    return inverse


def _check_source(episode):
    episode.audit()
    if not episode.outcome['success'] or episode.metadata.get('mode') != 'expert':
        raise ValueError('Dataset requires successful expert sources')
    audit = episode.metadata.get('reset_orchard_audit')
    if not isinstance(audit, dict):
        raise ValueError('Missing reset orchard audit')
    required = ('apple_count', 'visible_available_count', 'inactive_fallen_count',
                'max_world_error_m', 'max_local_error_m')
    if any(key not in audit or not np.isfinite(audit[key]) for key in required):
        raise ValueError('Incomplete reset orchard audit')
    if (audit['apple_count'] < 1 or audit['visible_available_count'] != audit['apple_count']
            or audit['inactive_fallen_count'] != audit['apple_count']
            or not 0 <= audit['max_world_error_m'] <= 1e-6
            or not 0 <= audit['max_local_error_m'] <= 1e-6):
        raise ValueError('Failed reset orchard audit')


def _observation(arrays, indices, inverse):
    goal = arrays['goal_world_m'][indices] @ inverse[:3, :3] + inverse[3, :3]
    receiver = arrays['receiver_world_m'][indices] @ inverse[:3, :3] + inverse[3, :3]
    return np.concatenate((arrays['q'][indices], arrays['qdot'][indices], goal, receiver,
                           arrays['attached'][indices, None]), axis=-1).astype(np.float32)


def convert_episode(episode, dt=.1):
    """Sample exact physics rows; label t with the applied command at t + dt.

    Returns obs (N,15), action (N,4), time_s/action_time_s (N,), and terminal
    metadata. The last raw state is retained only in terminal, never labeled
    with an extrapolated action. A fractional final interval is omitted.
    """
    _check_source(episode)
    physics_dt = float(episode.metadata['physics_dt'])
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError('dt and physics_dt must be finite positive intervals')
    stride = round(dt / physics_dt)
    if stride < 1 or abs(stride * physics_dt - dt) > 1e-9:
        raise ValueError('dt must be an integer multiple of physics_dt')
    arrays = episode.arrays()
    count = (len(arrays['time_s']) - 1) // stride
    if count == 0:
        raise ValueError('Episode has no complete dt interval')
    current = np.arange(count, dtype=np.int64) * stride
    following = current + stride
    times, action_times = arrays['time_s'][current], arrays['time_s'][following]
    if not np.allclose(action_times - times, dt, rtol=0, atol=1e-8):
        raise ValueError('Raw physics samples do not align with dt')
    collect_time = next(event['time_s'] for event in episode.events if event['kind'] == 'collect')
    if action_times[-1] + 1e-9 < collect_time:
        raise ValueError('Sampling removes the collect command; record a real post-collect hold')
    inverse = _base_inverse(episode.metadata)
    retained_duration = float(action_times[-1] - times[0])
    duration = float(arrays['time_s'][-1] - times[0])
    terminal = {'time_s': float(arrays['time_s'][-1]),
                'obs': _observation(arrays, np.array([-1]), inverse)[0].tolist(),
                'command': arrays['command'][-1].tolist(),
                'duration_s': duration, 'retained_duration_s': retained_duration,
                'omitted_duration_s': max(0., duration - retained_duration),
                'retained_end_time_s': float(action_times[-1]),
                'outcome': dict(episode.outcome), 'events': episode.events}
    return {'obs': _observation(arrays, current, inverse),
            'action': arrays['command'][following].astype(np.float32),
            'time_s': times.astype(np.float64),
            'action_time_s': action_times.astype(np.float64), 'terminal': terminal}


def _episode_layout(data, episode_ends, episode_indices):
    arrays = {key: np.asarray(data[key]) for key in ('obs', 'action')}
    if any(value.ndim != 2 or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Expected finite two-dimensional obs and action arrays')
    count = len(arrays['obs'])
    ends = np.asarray(episode_ends)
    selected = np.asarray(list(episode_indices))
    if (len(arrays['action']) != count or ends.ndim != 1 or len(ends) == 0
            or not np.issubdtype(ends.dtype, np.integer)
            or ends[-1] != count or np.any(np.diff(np.r_[0, ends]) <= 0)):
        raise ValueError('Invalid episode ends or inconsistent data lengths')
    if (selected.ndim != 1 or (len(selected) and (
            not np.issubdtype(selected.dtype, np.integer) or np.any(selected < 0)
            or np.any(selected >= len(ends)) or len(np.unique(selected)) != len(selected)))):
        raise ValueError('Invalid episode selection')
    return arrays, ends.astype(np.int64), selected.astype(np.int64)


class Normalizer:
    """Invertible affine scaling with float32-aware near-constant detection.

    Variable dimensions map training min/max to [-1, 1]. Constant or
    float32-precision near-constant dimensions use scale=1 and bias=-minimum,
    preserving meaningful held-out deviations without amplifying quantization
    noise. No input or output is clipped.
    """
    def __init__(self, fields, fit_metadata):
        self.fields = fields
        self.fit_metadata = dict(fit_metadata)

    @classmethod
    def fit(cls, data, episode_ends, episode_indices):
        arrays, ends, selected = _episode_layout(data, episode_ends, episode_indices)
        if len(selected) == 0:
            raise ValueError('Normalizer requires at least one training episode')
        starts = np.r_[0, ends[:-1]]
        fields = {}
        for key, values in arrays.items():
            train = np.concatenate([values[starts[index]:ends[index]] for index in selected])
            minimum = train.min(axis=0).astype(np.float64)
            maximum = train.max(axis=0).astype(np.float64)
            spread = maximum - minimum
            # Dataset observations/actions are stored as float32. Several ULPs
            # of spread can arise even when a physical field is effectively
            # fixed (for example, a receiver axis in a planar setup). Treat
            # that representational jitter as constant instead of multiplying
            # it by an enormous min/max scale. Held-out deviations remain
            # un-clipped and are expressed in the original physical units.
            precision_atol = (8. * np.finfo(np.float32).eps
                              * np.maximum(1., np.maximum(np.abs(minimum), np.abs(maximum))))
            constant = spread <= precision_atol
            scale = np.ones_like(spread)
            np.divide(2., spread, out=scale, where=~constant)
            bias = np.where(constant, -minimum, -1. - minimum * scale)
            fields[key] = {'min': minimum, 'max': maximum, 'scale': scale,
                           'bias': bias, 'constant': constant}
        return cls(fields, {'episode_indices': selected.tolist(),
                            'row_count': int(np.sum(ends[selected] - starts[selected])),
                            'method': 'train-selected per-dimension affine minmax with float32 near-constant detection',
                            'constant_dimensions': 'spread <= 8*float32_eps*max(1, abs(min), abs(max)); scale=1, bias=-training_min',
                            'clip': False})

    def with_action_limits(self, lower, upper):
        """Copy this normalizer, mapping physical action limits to [-1, 1].

        Observation statistics retain their selected-episode provenance. The
        fitted action range must lie within these limits (float32 tolerance).
        """
        from copy import deepcopy
        lower, upper = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
        action = self.fields['action']
        if (lower.shape != action['min'].shape or upper.shape != lower.shape
                or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                or np.any(upper <= lower)
                or np.any(action['min'] < lower - 1e-7) or np.any(action['max'] > upper + 1e-7)):
            raise ValueError('Invalid physical action limits or fitted actions outside limits')
        fields = deepcopy(self.fields)
        scale = 2. / (upper - lower)
        fields['action'] = {'min': lower.copy(), 'max': upper.copy(), 'scale': scale,
                            'bias': -1. - lower * scale, 'constant': np.zeros_like(lower, dtype=bool)}
        metadata = deepcopy(self.fit_metadata)
        metadata['action_normalization'] = 'physical_joint_limits'
        metadata['action_limits_source'] = 'dataset source_configuration joint_lower/joint_upper'
        return type(self)(fields, metadata)

    def _transform(self, data, key, inverse):
        if isinstance(data, dict):
            return {name: self._transform(value, name, inverse) if name in self.fields else value
                    for name, value in data.items()}
        if key not in self.fields:
            raise ValueError('Array normalization requires key obs or action')
        values = np.asarray(data)
        field = self.fields[key]
        if (values.ndim == 0 or values.shape[-1] != len(field['scale'])
                or not np.isfinite(values).all()):
            raise ValueError('Invalid normalization input shape or values')
        result = ((values - field['bias']) / field['scale'] if inverse
                  else values * field['scale'] + field['bias'])
        return result.astype(np.float32)

    def normalize(self, data, key=None):
        return self._transform(data, key, inverse=False)

    def denormalize(self, data, key=None):
        return self._transform(data, key, inverse=True)

    def to_dict(self):
        return {'fields': {key: {name: value.tolist() for name, value in field.items()}
                           for key, field in self.fields.items()},
                'fit_metadata': self.fit_metadata}

    @classmethod
    def from_dict(cls, document):
        fields = {}
        for key in ('obs', 'action'):
            field = {name: np.asarray(document['fields'][key][name],
                                     dtype=bool if name == 'constant' else np.float64)
                     for name in ('min', 'max', 'scale', 'bias', 'constant')}
            if (any(value.ndim != 1 or not np.isfinite(value).all() for value in field.values())
                    or len({value.shape for value in field.values()}) != 1
                    or np.any(field['scale'] <= 0)):
                raise ValueError('Invalid normalizer parameters')
            fields[key] = field
        return cls(fields, document['fit_metadata'])


class SequenceDataset:
    """Raw NumPy windows with replicated boundaries and an action validity mask.

    Every selected row is one current index k. Windows start at k-n_obs_steps+1;
    action[n_obs_steps-1] is the command for current observation k. No window
    reads another episode. Call Normalizer.normalize on returned dictionaries.
    """
    def __init__(self, data, episode_ends, episode_indices, horizon=16, n_obs_steps=2):
        self.data, self.episode_ends, selected = _episode_layout(data, episode_ends, episode_indices)
        if (not isinstance(horizon, int) or not isinstance(n_obs_steps, int)
                or not 1 <= n_obs_steps <= horizon):
            raise ValueError('Require integer horizon >= n_obs_steps >= 1')
        self.horizon, self.n_obs_steps = horizon, n_obs_steps
        starts = np.r_[0, self.episode_ends[:-1]]
        self.indices = [(int(starts[episode]), int(self.episode_ends[episode]), current)
                        for episode in selected
                        for current in range(starts[episode], self.episode_ends[episode])]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start, end, current = self.indices[index]
        requested = np.arange(self.horizon) + current - self.n_obs_steps + 1
        valid = (requested >= start) & (requested < end)
        clipped = np.clip(requested, start, end - 1)
        return {'obs': self.data['obs'][clipped[:self.n_obs_steps]].copy(),
                'action': self.data['action'][clipped].copy(), 'valid_mask': valid}


class LoadedDataset:
    def __init__(self, data, episode_ends, manifest, normalizer):
        self.data = data
        self.episode_ends = np.asarray(episode_ends, dtype=np.int64)
        self.manifest = manifest
        self.normalizer = normalizer

    def episode_indices(self, split):
        if split not in SPLITS:
            raise ValueError('Unknown dataset split')
        return [index for index, episode in enumerate(self.manifest['episodes'])
                if episode['split'] == split]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _configuration(metadata):
    required = ('arm', 'scene_sha256', 'physics_dt', 'joint_names', 'joint_units',
                'joint_lower', 'joint_upper', 'controller_config', 'executor',
                'speed_scale', 'drive_speed_scale', 'grasp_error_every',
                'action_semantics', 'read_phase', 'world_from_base', 'transform_convention')
    if any(key not in metadata for key in required):
        raise ValueError('Missing source control configuration')
    if not isinstance(metadata.get('target_path'), str) or not metadata['target_path']:
        raise ValueError('Missing source target identity')
    return {key: metadata[key] for key in required}


def _target_splits(episodes, split_counts):
    groups = sorted({(source['metadata']['scene_sha256'], source['metadata']['target_path'])
                     for source in episodes})
    counts = (3, 1, 1) if split_counts is None else tuple(split_counts)
    if (len(counts) != 3 or any(not isinstance(count, int) or count < 0 for count in counts)
            or counts[0] == 0 or sum(counts) != len(groups)):
        raise ValueError('Split counts must cover all target groups; default pilot requires five')
    assigned = {}
    offset = 0
    for split, count in zip(SPLITS, counts):
        for group in groups[offset:offset + count]:
            assigned[group] = split
        offset += count
    return assigned


def prepare_dataset(input_root, dt=.1, split_counts=None):
    """Audit sources and construct an in-memory dataset without importing Zarr.

    The default sorted target split requires exactly five target groups and
    assigns 3/1/1 groups to train/val/test. Explicit counts may be supplied.
    """
    root = Path(input_root)
    if list(root.rglob('failure.json')):
        raise ValueError('Input contains failed source episodes')
    paths = sorted(root.rglob('episode.json'))
    if not paths:
        raise ValueError('No source episodes found')
    sources, converted, ends = [], [], []
    configuration = None
    for path in paths:
        document = json.loads(path.read_text(encoding='utf-8'))
        raw_files = {item.name for item in path.parent.glob('*.npz')}
        if ('states.npz' not in document.get('sha256', {})
                or raw_files != set(document['sha256'])):
            raise ValueError('Source raw archive checksum inventory is incomplete')
        source = Episode.load(path.parent)
        current_configuration = _configuration(source.metadata)
        if configuration is not None and current_configuration != configuration:
            raise ValueError('Dataset requires uniform configuration; mixed source arm/scene/control')
        configuration = current_configuration
        result = convert_episode(source, dt)
        converted.append(result)
        ends.append((ends[-1] if ends else 0) + len(result['obs']))
        sources.append({'path': path.parent.relative_to(root).as_posix(),
                        'episode_json_sha256': _sha256(path),
                        'raw_sha256': dict(document['sha256']),
                        'metadata': source.metadata, 'terminal': result['terminal'],
                        'rows': len(result['obs'])})
    assigned = _target_splits(sources, split_counts)
    for source in sources:
        source['split'] = assigned[(source['metadata']['scene_sha256'], source['metadata']['target_path'])]
    arrays = {key: np.concatenate([result[key] for result in converted])
              for key in ('obs', 'action', 'time_s', 'action_time_s')}
    manifest = {'schema': SCHEMA, 'dt': float(dt), 'arm': configuration['arm'],
                'source_root': str(root.resolve()), 'source_configuration': configuration,
                'observation_fields': [{'name': name, 'size': size} for name, size in OBS_FIELDS],
                'observation_semantics': 'q, qdot, oracle goal/receiver in base frame, oracle attached',
                'action_semantics': ACTION_SEMANTICS,
                'sampling': 'exact raw physics rows; complete dt intervals only; no interpolation',
                'split_semantics': 'sorted (scene_sha256, target_path) groups; pilot only',
                'target_splits': [{'scene_sha256': scene, 'target_path': target, 'split': split}
                                  for (scene, target), split in sorted(assigned.items())],
                'episodes': sources, 'rows': int(ends[-1]), 'episode_ends': ends}
    train = [index for index, source in enumerate(sources) if source['split'] == 'train']
    return LoadedDataset(arrays, ends, manifest, Normalizer.fit(arrays, ends, train))


def _zarr_v2():
    import zarr
    if zarr.__version__.split('.')[0] != '2':
        raise ValueError('Dataset storage requires Zarr v2')
    return zarr


def _write_json(path, document):
    path.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + '\n',
                    encoding='utf-8')


def build_dataset(input_root, output, dt=.1, split_counts=None):
    """Create an immutable Zarr-v2 export and return its verified LoadedDataset.

    output/data.zarr contains data/{obs,action,time_s,action_time_s} and
    meta/{episode_ends,episode_split}. JSON manifests, train normalization
    statistics and file checksums are adjacent. Existing output is refused.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Dataset output already exists: {output}')
    zarr = _zarr_v2()
    loaded = prepare_dataset(input_root, dt=dt, split_counts=split_counts)
    loaded.manifest['converter_sha256'] = _sha256(__file__)
    loaded.manifest['storage'] = {'format': 'Zarr v2', 'zarr_version': zarr.__version__,
                                  'path': 'data.zarr'}
    output.mkdir(parents=True, exist_ok=False)
    store = zarr.open_group(str(output / 'data.zarr'), mode='w')
    data = store.create_group('data')
    for key, values in loaded.data.items():
        chunks = (min(1024, len(values)),) + values.shape[1:]
        data.create_dataset(key, data=values, chunks=chunks)
    meta = store.create_group('meta')
    meta.create_dataset('episode_ends', data=loaded.episode_ends)
    meta.create_dataset('episode_split', data=np.asarray(
        [source['split'] for source in loaded.manifest['episodes']], dtype='<U5'))
    _write_json(output / 'manifest.json', loaded.manifest)
    _write_json(output / 'normalizer.json', loaded.normalizer.to_dict())
    checksums = {path.relative_to(output).as_posix(): _sha256(path)
                 for path in sorted(output.rglob('*')) if path.is_file()}
    _write_json(output / 'checksums.json', checksums)
    return load_dataset(output)


def load_dataset(path):
    """Verify every exported file before loading arrays and frozen metadata."""
    root = Path(path)
    checksums = json.loads((root / 'checksums.json').read_text(encoding='utf-8'))
    inventory = {item.relative_to(root).as_posix() for item in root.rglob('*')
                 if item.is_file() and item != root / 'checksums.json'}
    if set(checksums) != inventory:
        raise ValueError('Dataset checksum file inventory mismatch')
    for name, expected in checksums.items():
        if (Path(name).is_absolute() or '..' in Path(name).parts
                or _sha256(root / name) != expected):
            raise ValueError(f'Dataset checksum mismatch: {name}')
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != SCHEMA:
        raise ValueError('Unsupported dataset schema')
    zarr = _zarr_v2()
    store = zarr.open_group(str(root / 'data.zarr'), mode='r')
    arrays = {key: np.asarray(store['data'][key])
              for key in ('obs', 'action', 'time_s', 'action_time_s')}
    ends = np.asarray(store['meta']['episode_ends'])
    rows = manifest['rows']
    if any(arrays[key].shape != (rows, width) or arrays[key].dtype != np.float32
           for key, width in (('obs', 15), ('action', 4))):
        raise ValueError('Invalid stored observation/action shape or dtype')
    if any(arrays[key].shape != (rows,) or arrays[key].dtype != np.float64
           or not np.isfinite(arrays[key]).all() for key in ('time_s', 'action_time_s')):
        raise ValueError('Invalid stored time arrays')
    _episode_layout(arrays, ends, range(len(ends)))
    expected_splits = [source['split'] for source in manifest['episodes']]
    if (ends.tolist() != manifest['episode_ends'] or len(ends) != len(expected_splits)
            or np.asarray(store['meta']['episode_split']).tolist() != expected_splits):
        raise ValueError('Stored episode metadata differs from manifest')
    if not np.allclose(arrays['action_time_s'] - arrays['time_s'], manifest['dt'],
                       rtol=0, atol=1e-8):
        raise ValueError('Stored action timestamps differ from dt')
    starts = np.r_[0, ends[:-1]]
    for start, end in zip(starts, ends):
        if not np.allclose(np.diff(arrays['time_s'][start:end]), manifest['dt'], rtol=0, atol=1e-8):
            raise ValueError('Stored observation timestamps are not uniform')
    normalizer = Normalizer.from_dict(json.loads((root / 'normalizer.json').read_text(encoding='utf-8')))
    loaded = LoadedDataset(arrays, ends, manifest, normalizer)
    expected = Normalizer.fit(arrays, ends, loaded.episode_indices('train'))
    if normalizer.to_dict() != expected.to_dict():
        raise ValueError('Stored normalizer does not match train-only statistics')
    return loaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dt', type=float, default=.1)
    parser.add_argument('--split-counts', type=int, nargs=3, metavar=('TRAIN', 'VAL', 'TEST'))
    options = parser.parse_args()
    loaded = build_dataset(options.input, options.output, options.dt, options.split_counts)
    print(json.dumps({'path': str(options.output), 'rows': len(loaded.data['obs']),
                      'episodes': len(loaded.episode_ends),
                      'splits': {split: loaded.episode_indices(split) for split in SPLITS}}, indent=2))
    print('AUTOPICKER_DATASET_AUDIT_PASS', flush=True)


if __name__ == '__main__':
    main()
