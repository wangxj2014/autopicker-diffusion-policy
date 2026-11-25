from io import StringIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopicker_dp.inference import PolicyClient


class EpisodeSeedTests(unittest.TestCase):
    def client(self, sequence=0, seed=0):
        client = PolicyClient.__new__(PolicyClient)
        client.sequence, client.base_seed = sequence, seed
        client.attempts = []
        client.process = SimpleNamespace(stdin=StringIO())
        client._receive = lambda: dict(json.loads(client.process.stdin.getvalue().splitlines()[-1]),
                                       action=[[0.] * 4] * 2)
        return client

    def predict(self, client):
        client.predict(np.zeros((2, 15)), 7, 0.)
        return json.loads(client.process.stdin.getvalue().splitlines()[-1])

    def test_episode_noise_is_independent_of_previous_requests_and_order(self):
        a, b = self.client(7), self.client(175)
        self.assertTrue(hasattr(a, 'reset_episode'), 'Episode-local inference reset is missing')
        a.reset_episode(4)
        b.reset_episode(4)
        first_a, first_b = self.predict(a), self.predict(b)
        self.assertNotEqual(first_a['request_id'], first_b['request_id'])
        self.assertEqual(first_a['noise_seed'], first_b['noise_seed'])
        self.assertEqual(first_a['episode_request_index'], 0)
        second = self.predict(a)
        self.assertNotEqual(first_a['noise_seed'], second['noise_seed'])
        a.reset_episode(5)
        self.assertNotEqual(first_a['noise_seed'], self.predict(a)['noise_seed'])
        a.reset_episode(4)
        self.assertEqual(first_a['noise_seed'], self.predict(a)['noise_seed'])
        c = self.client(seed=1)
        c.reset_episode(4)
        self.assertNotEqual(first_a['noise_seed'], self.predict(c)['noise_seed'])

    def test_seed_identity_is_echoed_and_verified(self):
        for key in ('noise_seed', 'source_episode_index', 'episode_request_index'):
            client = self.client()
            self.assertTrue(hasattr(client, 'reset_episode'), 'Episode-local inference reset is missing')
            client.reset_episode(4)
            original = client._receive
            client._receive = lambda: dict(original(), **{key: -1})
            with self.assertRaisesRegex(RuntimeError, 'Stale'):
                self.predict(client)

    def test_legacy_protocol_does_not_add_seed_fields(self):
        request = self.predict(self.client())
        self.assertEqual(set(request), {'request_id', 'generation', 'time_s', 'obs'})

    def test_attempt_identity_survives_response_failure(self):
        client = self.client()
        client.reset_episode(4)
        client._receive = lambda: {'request_id': -1}
        with self.assertRaises(RuntimeError):
            self.predict(client)
        self.assertEqual(len(client.attempts), 1)
        self.assertEqual(client.attempts[0]['request_id'], 1)
        self.assertEqual(client.attempts[0]['episode_request_index'], 0)

    def test_seed_formula_is_frozen_and_inputs_validated(self):
        from autopicker_dp import inference
        self.assertTrue(hasattr(inference, 'episode_noise_seed'), 'Seed formula is missing')
        import hashlib
        expected = int.from_bytes(hashlib.sha256(b'autopicker-episode-v1:0:4:0').digest()[:4], 'big')
        self.assertEqual(inference.episode_noise_seed(0, 4, 0), expected)
        for args in ((0, -1, 0), (0, 4, -1), (0, 4, .5), (True, 4, 0)):
            with self.assertRaises(ValueError):
                inference.episode_noise_seed(*args)


if __name__ == '__main__':
    unittest.main()
