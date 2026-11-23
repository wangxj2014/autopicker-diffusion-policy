import json
from pathlib import Path
import tempfile
import unittest

from validate_m1 import rollout_report


class RolloutReportTests(unittest.TestCase):
    def test_failed_first_action_stays_in_denominator_and_identity_is_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            episode = root / 'episode_0000'
            episode.mkdir()
            summary = {'completed': True, 'dataset_sha256': 'data', 'checkpoint_sha256': 'model',
                       'requested': 1, 'attempted': 1, 'successes': 0, 'mode': 'policy',
                       'scope': 'fixture', 'split': 'val', 'seed': 0,
                       'results': [{'success': False, 'target_path': '/apple', 'reason': 'invalid action'}]}
            (root / 'summary.json').write_text(json.dumps(summary))
            failure = {'metadata': {'dataset_sha256': 'data', 'checkpoint_sha256': 'model',
                                     'target_path': '/apple'}, 'reason': 'invalid action'}
            (episode / 'failure.json').write_text(json.dumps(failure))
            report = rollout_report(root, 'data')
            self.assertEqual(report['requested'], 1)
            self.assertEqual(report['successes'], 0)
            self.assertEqual(report['episodes'][0]['reason'], 'invalid action')
            failure['metadata']['checkpoint_sha256'] = 'different model'
            (episode / 'failure.json').write_text(json.dumps(failure))
            with self.assertRaises(ValueError):
                rollout_report(root, 'data')


if __name__ == '__main__':
    unittest.main()
