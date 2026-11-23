from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopicker_dp.episode import Episode
from test_dataset import example_episode


def recorded_episode(target, value=.1):
    metadata = example_episode(target).metadata
    metadata.update(reset_joint_positions=([value] * 4 + [0.] * 12), max_inactive_joint_drift=0.)
    episode = Episode(metadata)
    for step in range(151):
        q = np.full(4, value)
        episode.append_state(step * .01, q, np.zeros(4), q, np.zeros(4),
                             np.zeros(3), np.ones(3), np.zeros(3), step >= 120,
                             np.r_[q, np.zeros(12)])
    episode.event('capture', 1.2, error_m=.001, angle_error_degrees=.1)
    episode.event('collect', 1.4)
    episode.finish(True, 'collected')
    return episode


def augmentation_fixture(root):
    from autopicker_dp.dataset import build_dataset, _sha256
    from autopicker_dp.recovery import recovery_reset
    for index in range(3):
        recorded_episode(f'/apple_{index}').save(root / 'base_raw' / f'episode_{index:04d}')
    base_path = root / 'base'
    base = build_dataset(root / 'base_raw', base_path, split_counts=(1, 1, 1))
    base_hash = _sha256(base_path / 'manifest.json')
    specification = base.manifest['episodes'][0]
    source = Episode.load(root / 'base_raw' / specification['path'])
    for ordinal, sign in enumerate((-1, 1)):
        reset, info = recovery_reset(source, [0, 1, 2, 3], [-2.] * 4, [2.] * 4, np.full(4, sign * .1))
        info.update(source_dataset_sha256=base_hash, source_episode_index=0, source_split='train',
                    source_episode_json_sha256=specification['episode_json_sha256'],
                    source_raw_sha256=specification['raw_sha256'], settled_q=reset[:4].tolist())
        episode = recorded_episode('/apple_0', .1 + sign * .1)
        episode.metadata.update(recovery=info, reset_joint_positions=reset.tolist())
        episode.save(root / 'recovery_raw' / f'episode_{ordinal:04d}')
    recovery_path = root / 'recovery'
    recovery = build_dataset(root / 'recovery_raw', recovery_path, split_counts=(1, 0, 0))
    return base, base_path, base_hash, recovery, recovery_path


@unittest.skipUnless(importlib.util.find_spec('zarr'), 'Zarr training environment required')
class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.base, self.base_path, self.base_hash, self.recovery, self.recovery_path = augmentation_fixture(Path(self.folder.name))

    def test_append_preserves_boundaries_parent_and_training_only_selection(self):
        from autopicker_dp.augmentation import recovery_sequences
        before = deepcopy(self.base.normalizer.to_dict())
        sequence, info = recovery_sequences(self.base, self.base_hash, self.recovery_path)
        self.assertEqual(len(sequence), 45)
        self.assertEqual(info['base_windows'], 15)
        self.assertEqual(info['recovery_windows'], 30)
        self.assertEqual(info['sampling'], 'uniform over combined window anchors')
        np.testing.assert_array_equal(sequence[14]['obs'][:, :4], np.full((2, 4), .1, np.float32))
        np.testing.assert_array_equal(sequence[14]['action'], np.full((16, 4), .1, np.float32))
        np.testing.assert_array_equal(sequence[15]['obs'][:, :4], np.zeros((2, 4)))
        np.testing.assert_array_equal(sequence[29]['action'], np.zeros((16, 4)))
        np.testing.assert_array_equal(sequence[30]['obs'][:, :4], np.full((2, 4), .2, np.float32))
        self.assertEqual(self.base.normalizer.to_dict(), before)
        self.assertEqual(len(self.base.episode_ends), 3)
        self.assertEqual(info['normalizer_source'], 'primary dataset train only; frozen before augmentation')

    def test_rejects_parent_identity_holdout_configuration_and_provenance_changes(self):
        from autopicker_dp.augmentation import validate_recovery
        validate_recovery(self.base, self.base_hash, self.recovery)
        mutations = [
            lambda m: m.update(dt=.2),
            lambda m: m['source_configuration'].update(arm='R_up'),
            lambda m: m['episodes'][0].update(split='val'),
            lambda m: m['episodes'][0]['metadata'].update(target_path='/apple_1'),
            lambda m: m['episodes'][0]['metadata']['recovery'].update(source_dataset_sha256='wrong'),
            lambda m: m['episodes'][0]['metadata']['recovery'].update(source_episode_index=1),
            lambda m: m['episodes'][0]['metadata']['recovery'].update(source_episode_json_sha256='wrong'),
            lambda m: m['episodes'][0]['metadata']['recovery'].update(source_raw_sha256={}),
            lambda m: m['episodes'][0]['metadata']['recovery'].update(source_row=0),
            lambda m: m['episodes'][0]['metadata'].update(reset_joint_positions=[.7] * 16),
            lambda m: m['episodes'][1].update(episode_json_sha256=m['episodes'][0]['episode_json_sha256']),
        ]
        for mutate in mutations:
            changed = deepcopy(self.recovery)
            mutate(changed.manifest)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                validate_recovery(self.base, self.base_hash, changed)

    def test_reconstruction_detects_changed_raw_archive(self):
        from autopicker_dp.augmentation import recovery_sequences
        raw = Path(self.recovery.manifest['source_root']) / 'episode_0000/states.npz'
        raw.write_bytes(raw.read_bytes() + b'altered')
        with self.assertRaises(ValueError):
            recovery_sequences(self.base, self.base_hash, self.recovery_path)

    def test_rejects_forged_settled_state_after_valid_reexport(self):
        from autopicker_dp.dataset import build_dataset
        from autopicker_dp.augmentation import recovery_sequences
        raw_root = Path(self.recovery.manifest['source_root'])
        path = raw_root / 'episode_0000/episode.json'
        document = json.loads(path.read_text())
        document['metadata']['recovery']['settled_q'] = [1.] * 4
        path.write_text(json.dumps(document))
        exported = Path(self.folder.name) / 'forged_settled'
        build_dataset(raw_root, exported, split_counts=(1, 0, 0))
        with self.assertRaisesRegex(ValueError, 'settled'):
            recovery_sequences(self.base, self.base_hash, exported)

    def test_rejects_duplicate_trajectory_with_different_metadata(self):
        from autopicker_dp.dataset import build_dataset
        from autopicker_dp.augmentation import recovery_sequences
        raw_root = Path(self.recovery.manifest['source_root'])
        duplicate = raw_root / 'episode_0002'
        shutil.copytree(raw_root / 'episode_0000', duplicate)
        path = duplicate / 'episode.json'
        document = json.loads(path.read_text())
        document['metadata']['generation'] = 99
        path.write_text(json.dumps(document))
        exported = Path(self.folder.name) / 'duplicated'
        build_dataset(raw_root, exported, split_counts=(1, 0, 0))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            recovery_sequences(self.base, self.base_hash, exported)


if __name__ == '__main__':
    unittest.main()
