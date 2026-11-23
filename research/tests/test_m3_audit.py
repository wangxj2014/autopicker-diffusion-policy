from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from test_m2_audit import AVAILABLE


@unittest.skipUnless(AVAILABLE and importlib.util.find_spec('zarr'), 'Training environment required')
class M3AuditTests(unittest.TestCase):
    def test_pair_rejects_changed_factor_or_augmentation_lineage(self):
        from test_augmentation import augmentation_fixture
        from autopicker_dp.train import training_inputs
        from validate_m3 import training_pair
        with tempfile.TemporaryDirectory() as folder:
            parent, _, digest, _, recovery = augmentation_fixture(Path(folder))
            normalizer, sequences, info = training_inputs(parent, digest, 'train', 'physical', recovery)
            baseline = {'seed': 0, 'steps': 3000, 'batch_size': 64, 'learning_rate': 1e-4,
                'dt': .1, 'model_config': {}, 'ema_decay': .995, 'optimizer': 'AdamW',
                'versions': {}, 'scheduler': {}, 'normalizer': normalizer.to_dict(),
                'dataset_manifest': parent.manifest, 'training_windows': info['base_windows'],
                'source_sha256': {'policy.py': 'same'}}
            augmented = deepcopy(baseline)
            augmented.update(training_windows=len(sequences), recovery_augmentation=info)
            result = training_pair(baseline, augmented, parent, digest, recovery)
            self.assertEqual(result['recovery_window_fraction'], 2 / 3)
            for change in (
                lambda r: r.update(steps=6000),
                lambda r: r.update(training_windows=1),
                lambda r: r['normalizer']['fields']['obs']['min'].__setitem__(0, -99.),
                lambda r: r['recovery_augmentation'].update(dataset_sha256='wrong'),
                lambda r: r['source_sha256'].update(**{'policy.py': 'different'}),
            ):
                altered = deepcopy(augmented)
                change(altered)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    training_pair(baseline, altered, parent, digest, recovery)

    def test_early_failures_remain_in_paired_denominator(self):
        from test_augmentation import augmentation_fixture
        from autopicker_dp.train import training_normalizer
        from validate_m3 import policy_rollout
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent, _, digest, _, _ = augmentation_fixture(root)
            source = parent.manifest['episodes'][0]
            rollout = root / 'rollout'
            episode = rollout / 'episode_0000'
            episode.mkdir(parents=True)
            metadata = {**source['metadata'], 'dataset_sha256': digest, 'checkpoint_sha256': 'model',
                'source_episode_index': 0, 'source_split': 'train', 'rollout_seed': 0,
                'policy_dt': .1, 'mode': 'policy'}
            summary = {'completed': True, 'dataset_sha256': digest, 'checkpoint_sha256': 'model',
                'requested': 1, 'attempted': 1, 'successes': 0, 'mode': 'policy',
                'scope': 'fixture', 'split': 'train', 'seed': 0,
                'results': [{'success': False, 'target_path': source['metadata']['target_path'],
                             'reason': 'invalid action', 'source_episode_index': 0}]}
            (episode / 'failure.json').write_text(json.dumps({'reason': 'invalid action', 'metadata': metadata}))
            (rollout / 'summary.json').write_text(json.dumps(summary))
            normalizer = training_normalizer(parent, [0], 'physical').to_dict()
            report = policy_rollout(rollout, parent, digest, [0], normalizer, 'model', 'train')
            self.assertEqual((report['successes'], report['requested']), (0, 1))
            for key, value in [('checkpoint_sha256', 'wrong'), ('seed', 1), ('split', 'val')]:
                altered = {**summary, key: value}
                (rollout / 'summary.json').write_text(json.dumps(altered))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    policy_rollout(rollout, parent, digest, [0], normalizer, 'model', 'train')


if __name__ == '__main__':
    unittest.main()
