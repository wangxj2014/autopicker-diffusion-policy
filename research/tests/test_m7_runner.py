import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class M7RunnerTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('run_m7'), 'Fixed M7 runner is missing')
        import run_m7
        return run_m7

    def test_fixed_matrix_has_twenty_attempts_without_test_split(self):
        runner = self.module()
        trials = runner.trial_specs()
        self.assertEqual(len(trials), 8)
        self.assertEqual(sum(t['max_episodes'] for t in trials), 20)
        self.assertEqual({t['seed'] for t in trials}, {0, 1})
        self.assertEqual({t['split'] for t in trials}, {'train', 'val'})
        self.assertEqual({t['mode'] for t in trials}, {'hybrid', 'hybrid-corridor'})
        self.assertTrue(all(t['seed_mode'] == 'episode' and t['timeout'] == 35. for t in trials))

    def test_resume_refuses_partial_or_changed_source_results(self):
        runner = self.module()
        spec = runner.trial_specs()[0]
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            summary = {'completed': True, 'attempted': 3, 'requested': 3,
                       **{key: spec[key] for key in ('mode', 'split', 'seed', 'seed_mode')},
                       'dataset_sha256': runner.DATASET_HASH, 'checkpoint_sha256': runner.CHECKPOINT_HASH,
                       'results': [{'source_episode_index': i} for i in [1, 2, 4]]}
            (folder / 'summary.json').write_text(json.dumps(summary))
            for i in range(3):
                path = folder / f'episode_{i:04d}'
                path.mkdir()
                (path / 'episode.json').write_text(json.dumps({'metadata': {'source_sha256': {'a': 'abc'}}}))
            self.assertTrue(hasattr(runner, 'record_completion'), 'Process completion evidence is missing')
            log = folder / 'process.log'
            log.write_text('AUTOPICKER_ROLLOUT_COMPLETE {}\n')
            runner.record_completion(folder, 0)
            runner.verify_resume(folder, spec, {'a': 'abc'})
            with self.assertRaises(ValueError):
                runner.verify_resume(folder, spec, {'a': 'changed'})
            summary['completed'] = False
            (folder / 'summary.json').write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                runner.verify_resume(folder, spec, {'a': 'abc'})

    def test_cleanup_failure_missing_or_changed_receipt_cannot_resume(self):
        runner = self.module()
        self.assertTrue(hasattr(runner, 'record_completion'), 'Process completion evidence is missing')
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'summary.json').write_text('{"completed": true}')
            log = folder / 'process.log'
            log.write_text('AUTOPICKER_ROLLOUT_COMPLETE {}\n')
            with self.assertRaises(ValueError):
                runner.record_completion(folder, 1)
            self.assertFalse((folder / 'completion.json').exists())
            with self.assertRaises((ValueError, FileNotFoundError)):
                runner.verify_completion(folder)
            log.write_text('AUTOPICKER_ROLLOUT_COMPLETE {}\nTraceback (most recent call last): cleanup failed\n')
            with self.assertRaises(ValueError):
                runner.record_completion(folder, 0)
            log.write_text('AUTOPICKER_ROLLOUT_COMPLETE {}\n')
            runner.record_completion(folder, 0)
            runner.verify_completion(folder)
            log.write_text('changed log\n')
            with self.assertRaises(ValueError):
                runner.verify_completion(folder)


if __name__ == '__main__':
    unittest.main()
