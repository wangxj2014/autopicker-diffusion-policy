from pathlib import Path
from io import StringIO
import json
from queue import Queue
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class OnlineActionTests(unittest.TestCase):
    def test_replanning_anchors_to_current_executor_position(self):
        from autopicker_dp.inference import ActionLoop
        calls = []
        def predict(observations, generation, timestamp):
            calls.append((observations.copy(), generation, timestamp))
            offset = .1 * len(calls)
            return np.array([[offset] * 4, [offset + .1] * 4])
        loop = ActionLoop(predict, .1, np.full(4, -2.), np.full(4, 2.))
        loop.reset(7, np.zeros(4))
        first = loop.command(0, np.zeros(15), np.zeros(4), .1)
        np.testing.assert_array_equal(first, np.zeros(4))
        # Advance one observation without replanning so the two-frame history
        # at the next request contains the current observation twice.
        loop.command(1, np.ones(15), np.zeros(4), .1)
        # The executor is still short of the previous predicted endpoint. The
        # new chunk must begin at its measured position, not the stale endpoint.
        replanned = loop.command(2, np.ones(15), np.full(4, .15), .1)
        np.testing.assert_allclose(replanned, np.full(4, .15))
        np.testing.assert_array_equal(calls[0][0], np.zeros((2, 15)))
        np.testing.assert_array_equal(calls[1][0], np.stack([np.ones(15), np.ones(15)]))
        self.assertEqual(calls[1][1:], (7, .2))

    def test_action_loop_reports_latest_prediction_nodes(self):
        from autopicker_dp.inference import ActionLoop
        loop = ActionLoop(lambda *args: np.array([[.1] * 4, [.2] * 4]),
                          .1, np.full(4, -1.), np.ones(4))
        loop.reset(1, np.zeros(4))
        loop.command(0, np.zeros(15), np.zeros(4), .1)
        np.testing.assert_allclose(loop.actions, [[.1] * 4, [.2] * 4])
        np.testing.assert_allclose(loop.anchor, np.zeros(4))

    def test_invalid_predictions_fail_and_reset_clears_history(self):
        from autopicker_dp.inference import ActionLoop
        for response in (np.full((2, 4), np.nan), np.ones((4,)), np.full((2, 4), 2.)):
            loop = ActionLoop(lambda *args: response, .1, np.full(4, -1.), np.ones(4))
            loop.reset(1, np.zeros(4))
            with self.assertRaises(ValueError):
                loop.command(0, np.zeros(15), np.zeros(4), 1/480)

    def test_episode_reset_removes_old_observations_and_action_nodes(self):
        from autopicker_dp.inference import ActionLoop
        loop = ActionLoop(lambda *args: np.ones((2, 4)), .1, np.full(4, -2.), np.full(4, 2.))
        loop.reset(1, np.zeros(4))
        loop.command(0, np.ones(15), np.zeros(4), 1/480)
        loop.reset(2, np.full(4, .5))
        command = loop.command(0, np.zeros(15), np.full(4, .5), 1/480)
        np.testing.assert_array_equal(command, np.full(4, .5))
        np.testing.assert_array_equal(loop.requests[0]['observation'], np.zeros((2, 15)))
        self.assertEqual(len(loop.requests), 1)
        self.assertEqual(loop.requests[0]['generation'], 2)


class InferenceProtocolTests(unittest.TestCase):
    def test_inference_server_command_carries_step_override(self):
        from autopicker_dp.inference import inference_process_command
        command = inference_process_command('/venv/python', '/model.pt', '/upstream', 7, 5)
        self.assertEqual(command[-2:], ['--inference-steps', '5'])
        self.assertIn('/model.pt', command)
        self.assertIn('/upstream', command)
        with self.assertRaises(ValueError):
            inference_process_command('/venv/python', '/model.pt', '/upstream', 7, 0)
        with self.assertRaises(ValueError):
            inference_process_command('/venv/python', '/model.pt', '/upstream', 7, 101)

    def client(self):
        from autopicker_dp.inference import PolicyClient
        client = PolicyClient.__new__(PolicyClient)
        client.responses, client.timeout, client.sequence = Queue(), .005, 0
        client.attempts = []
        client.process = SimpleNamespace(stdin=StringIO())
        return client

    def test_stale_generation_and_time_and_request_are_rejected(self):
        for key, invalid in [('request_id', 0), ('generation', 6), ('time_s', .1)]:
            client = self.client()
            message = {'request_id': 1, 'generation': 7, 'time_s': .2, 'action': [[0.] * 4] * 2}
            message[key] = invalid
            client.responses.put(json.dumps(message))
            with self.assertRaisesRegex(RuntimeError, 'Stale'):
                client.predict(np.zeros((2, 15)), 7, .2)

    def test_inference_timeout_and_child_exit_are_reported(self):
        client = self.client()
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            client._receive()
        client.responses.put(None)
        with self.assertRaisesRegex(RuntimeError, 'exited'):
            client._receive()


if __name__ == '__main__':
    unittest.main()
