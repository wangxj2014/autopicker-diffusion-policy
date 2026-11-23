import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FailurePersistenceTests(unittest.TestCase):
    def test_post_collect_records_actual_physics_without_new_expert_commands(self):
        from autopicker_dp.run import record_post_collect
        command = np.array([.1, .2, .3, .4])
        env = SimpleNamespace(dt=.01, steps=15, success=True,
                              executor=SimpleNamespace(position=command.copy()), update_task=Mock())
        requested = []
        def step(value):
            requested.append(value.copy())
            env.steps += 1
        env.step = step
        record_post_collect(env, .2, None, 10)
        self.assertEqual(env.steps, 35)
        np.testing.assert_array_equal(requested, np.tile(command, (20, 1)))
        with self.assertRaises(ValueError):
            record_post_collect(env, .205, None, 10)

    def test_invalid_terminal_measurement_still_writes_failure_record(self):
        from autopicker_dp.episode import Episode
        from autopicker_dp.run import finish_episode
        environment = SimpleNamespace(
            episode=Episode({'target_path': '/apple'}), time=.1, success=False,
            executor=SimpleNamespace(position=np.zeros(4), velocity=np.zeros(4), limited_steps=0),
            max_inactive_drift=0., record_state=Mock(side_effect=ValueError('Nonfinite joint state')))
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / 'failed_episode'
            with self.assertRaisesRegex(RuntimeError, 'data audit'):
                finish_episode(environment, directory, 'Nonfinite joint state', time.monotonic())
            failure = json.loads((directory / 'failure.json').read_text())
            self.assertIn('Nonfinite joint state', failure['reason'])
            self.assertEqual(failure['events'][-1]['kind'], 'failure')
            self.assertFalse((directory / 'episode.json').exists())


if __name__ == '__main__':
    unittest.main()
