from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from test_m2_audit import AVAILABLE


@unittest.skipUnless(AVAILABLE and importlib.util.find_spec('zarr'), 'Training environment required')
class M5AuditTests(unittest.TestCase):
    def test_longer_budget_cannot_also_change_policy_implementation(self):
        from validate_m5 import budget_source_comparison
        short = {'source_sha256': {'policy.py': 'same', 'train.py': 'same'}}
        long = {'source_sha256': {**short['source_sha256'], 'diagnose.py': 'new diagnostic'}}
        budget_source_comparison(short, long)
        long['source_sha256']['policy.py'] = 'changed'
        with self.assertRaises(ValueError):
            budget_source_comparison(short, long)

    def test_comparison_rejects_objective_mismatch_budget_and_data_changes(self):
        from test_augmentation import augmentation_fixture
        from autopicker_dp.train import training_inputs
        from validate_m5 import training_comparison
        with tempfile.TemporaryDirectory() as directory:
            dataset, _, digest, _, path = augmentation_fixture(Path(directory))
            normalizer, seq, info = training_inputs(dataset, digest, 'train', 'physical', path)
            base = {'seed': 0, 'steps': 3000, 'batch_size': 64, 'learning_rate': 1e-4,
                'dt': .1, 'model_config': {'horizon': 16}, 'ema_decay': .995, 'optimizer': 'AdamW',
                'versions': {}, 'scheduler': {'prediction_type': 'epsilon'},
                'normalizer': normalizer.to_dict(), 'dataset_manifest': dataset.manifest,
                'training_windows': len(seq), 'recovery_augmentation': info,
                'source_sha256': {'control.py': 'same'}}
            sample = deepcopy(base)
            sample['model_config']['prediction_type'] = 'sample'
            sample['scheduler']['prediction_type'] = 'sample'
            training_comparison(base, sample, dataset, digest, path, 3000)
            for change in (
                lambda r: r['scheduler'].update(prediction_type='epsilon'),
                lambda r: r.update(steps=30000),
                lambda r: r['recovery_augmentation'].update(dataset_sha256='forged'),
                lambda r: r['normalizer']['fields']['action']['min'].__setitem__(0, -5.),
                lambda r: r['source_sha256'].update(**{'control.py': 'changed'}),
            ):
                altered = deepcopy(sample)
                change(altered)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    training_comparison(base, altered, dataset, digest, path, 3000)
            longer = deepcopy(sample)
            longer['steps'] = 30000
            training_comparison(base, longer, dataset, digest, path, 30000)

    def test_diagnostic_rejects_wrong_checkpoint_probes_and_nonfinite_statistics(self):
        from test_augmentation import augmentation_fixture
        from autopicker_dp.dataset import SequenceDataset
        from validate_m5 import diagnostic_audit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _, digest, _, _ = augmentation_fixture(root)
            sequences = SequenceDataset(dataset.data, dataset.episode_ends, dataset.episode_indices('train'))
            indices = list(range(0, len(sequences), 8))
            training = {'checkpoint_sha256': 'model', 'model_config': {'prediction_type': 'sample'}}
            report = {'dataset_sha256': digest, 'checkpoint_sha256': 'model', 'training_split': 'train',
                'parent_train_episode_indices': dataset.episode_indices('train'), 'window_indices': indices,
                'noise_seeds': list(range(8)), 'prediction_type': 'sample',
                'valid_action_nodes': sum(int(sequences[i]['valid_mask'][1:3].sum()) for i in indices),
                'joint_units': dataset.manifest['source_configuration']['joint_units'],
                'mean_seed_std_by_joint': [0.]*4, 'mean_abs_bias_by_joint': [0.]*4,
                'single_sample_mae_by_joint': [0.]*4}
            path = root / 'diagnostic.json'
            path.write_text(json.dumps(report))
            diagnostic_audit(path, dataset, digest, training)
            for change in (
                lambda r: r.update(checkpoint_sha256='another-model'),
                lambda r: r.update(window_indices=[999]),
                lambda r: r.update(noise_seeds=[0, 1]),
                lambda r: r.update(valid_action_nodes=999),
                lambda r: r.update(mean_seed_std_by_joint=[float('nan')]*4),
            ):
                altered = deepcopy(report)
                change(altered)
                path.write_text(json.dumps(altered))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    diagnostic_audit(path, dataset, digest, training)


if __name__ == '__main__':
    unittest.main()
