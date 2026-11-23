import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'diffusers', 'einops'))


@unittest.skipUnless(AVAILABLE, 'Training environment required')
class M2AuditTests(unittest.TestCase):
    def test_scheduler_default_name_order_is_not_a_training_factor(self):
        from validate_m2 import comparable_scheduler
        baseline = {'clip_sample': True, '_use_default_values': ['beta_start', 'beta_end']}
        reordered = {'clip_sample': True, '_use_default_values': ['beta_end', 'beta_start']}
        self.assertEqual(comparable_scheduler(baseline), comparable_scheduler(reordered))
        reordered['clip_sample'] = False
        self.assertNotEqual(comparable_scheduler(baseline), comparable_scheduler(reordered))

    def test_offline_seed_must_match_frozen_ablation(self):
        import torch
        from autopicker_dp.train import training_normalizer
        from validate_m2 import digest, training_audit
        dataset = SimpleNamespace(data={'obs': np.zeros((2, 15)), 'action': np.zeros((2, 4))},
            episode_ends=[2], episode_indices=lambda split: [0],
            manifest={'source_configuration': {'joint_lower': [-1.]*4, 'joint_upper': [1.]*4}})
        metadata = {'dataset_sha256': 'fixture', 'split': 'train', 'action_normalization': 'physical',
                    'normalizer': training_normalizer(dataset, [0], 'physical').to_dict()}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            torch.save({'metadata': metadata, 'model_config': {}}, folder / 'checkpoint.pt')
            checkpoint_hash = digest(folder / 'checkpoint.pt')
            (folder / 'report.json').write_text(json.dumps({**metadata,
                'checkpoint_sha256': checkpoint_hash, 'checkpoint_reload_max_error': 0}))
            offline = {'dataset_sha256': 'fixture', 'checkpoint_sha256': checkpoint_hash,
                       'seed': 0, 'training_split': 'train'}
            (folder / 'offline.json').write_text(json.dumps(offline))
            training_audit(folder, dataset, 'fixture', 'physical', 'train')
            offline['seed'] = 1
            (folder / 'offline.json').write_text(json.dumps(offline))
            with self.assertRaises(ValueError):
                training_audit(folder, dataset, 'fixture', 'physical', 'train')

    def test_recovery_failures_are_reported_without_success_only_export(self):
        from test_dataset import example_episode
        from validate_m2 import digest, recovery_audit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw'
            raw.mkdir()
            source = example_episode()
            source.save(root / 'source')
            spec = {'path': 'source', 'split': 'train', 'metadata': source.metadata,
                    'episode_json_sha256': digest(root / 'source/episode.json'),
                    'raw_sha256': json.loads((root / 'source/episode.json').read_text())['sha256']}
            baseline = SimpleNamespace(manifest={'source_root': str(root), 'episodes': [spec]})
            results = []
            for ordinal, sign in enumerate((-1, 1)):
                folder = raw / f'episode_{ordinal:04d}'
                folder.mkdir()
                provenance = {'source_dataset_sha256': 'fixture', 'source_episode_index': 0,
                    'source_split': 'train', 'source_episode_json_sha256': spec['episode_json_sha256'],
                    'source_raw_sha256': spec['raw_sha256'],
                    'delta_q': (sign * np.array([np.deg2rad(3.), .025, .02, np.deg2rad(5.)])).tolist()}
                (folder / 'failure.json').write_text(json.dumps({'reason': 'rejected', 'recovery': provenance,
                    'target_path': source.metadata['target_path']}))
                results.append({'success': False, 'reason': 'rejected', 'source_episode_index': 0,
                                'sign': sign, 'target_path': source.metadata['target_path']})
            (raw / 'summary.json').write_text(json.dumps({'completed': True, 'dataset_sha256': 'fixture',
                'requested': 2, 'attempted': 2, 'successes': 0, 'results': results, 'scope': 'fixture'}))
            report = recovery_audit(raw, root / 'export', root / 'replay', baseline, 'fixture')
            self.assertEqual((report['successes'], report['requested']), (0, 2))
            self.assertFalse(report['export_accepted'])
            self.assertEqual(len(report['episodes']), 2)


if __name__ == '__main__':
    unittest.main()
