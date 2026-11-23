from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from test_m2_audit import AVAILABLE


@unittest.skipUnless(AVAILABLE and importlib.util.find_spec('zarr'), 'Training environment required')
class M4AuditTests(unittest.TestCase):
    def test_pair_rejects_changed_budget_normalizer_data_and_controller(self):
        from test_augmentation import augmentation_fixture
        from autopicker_dp.train import training_inputs
        from validate_m4 import training_pair
        with tempfile.TemporaryDirectory() as directory:
            parent, _, digest, _, recovery = augmentation_fixture(Path(directory))
            normalizer, sequences, info = training_inputs(parent, digest, 'train', 'physical', recovery)
            baseline = {'seed': 0, 'steps': 3000, 'batch_size': 64, 'learning_rate': 1e-4,
                'dt': .1, 'model_config': {}, 'ema_decay': .995, 'optimizer': 'AdamW',
                'versions': {}, 'scheduler': {}, 'normalizer': normalizer.to_dict(),
                'dataset_manifest': parent.manifest, 'training_windows': len(sequences),
                'recovery_augmentation': info, 'source_sha256': {'policy.py': 'same'}}
            result = training_pair(baseline, deepcopy(baseline), parent, digest, recovery, recovery)
            self.assertEqual(result['dynamic']['combined_windows'], len(sequences))
            for change in (
                lambda r: r.update(steps=6000),
                lambda r: r.update(training_windows=1),
                lambda r: r['normalizer']['fields']['obs']['max'].__setitem__(0, 99.),
                lambda r: r['recovery_augmentation'].update(dataset_sha256='wrong'),
                lambda r: r['source_sha256'].update(**{'policy.py': 'different'}),
            ):
                altered = deepcopy(baseline)
                change(altered)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    training_pair(baseline, altered, parent, digest, recovery, recovery)

    def test_restoration_audit_rejects_lost_velocity_and_nonfinite_error(self):
        from validate_m4 import restored_initial
        state = {'q_all': [0.]*16, 'qdot': [.2]*4}
        arrays = {'q_all': np.zeros((1, 16)), 'qdot': np.full((1, 4), .2), 'time_s': np.array([0.])}
        episode = SimpleNamespace(arrays=lambda: arrays)
        audit = {'max_position_error': 0., 'max_active_velocity_error': 0.}
        restored_initial(episode, state, audit)
        arrays['qdot'][0] = 0
        with self.assertRaises(ValueError):
            restored_initial(episode, state, audit)
        arrays['qdot'][0] = .2
        audit['max_active_velocity_error'] = float('nan')
        with self.assertRaises(ValueError):
            restored_initial(episode, state, audit)


if __name__ == '__main__':
    unittest.main()
