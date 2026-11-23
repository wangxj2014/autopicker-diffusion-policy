import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'diffusers', 'einops'))


@unittest.skipUnless(AVAILABLE and os.environ.get('AUTOPICKER_DP_UPSTREAM'), 'Isolated training environment and pinned upstream required')
class PolicyTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def policy(self):
        from autopicker_dp.policy import DiffusionPolicy
        return DiffusionPolicy(os.environ['AUTOPICKER_DP_UPSTREAM'], down_dims=(16, 32),
                               diffusion_step_embed_dim=16, inference_steps=5)

    def test_noise_loss_backward_and_padding_mask(self):
        import torch
        policy = self.policy()
        observation = torch.randn(2, 2, 15)
        action = torch.randn(2, 16, 4).clamp(-1, 1)
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[:, -3:] = False
        loss = policy.loss(observation, action, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in policy.parameters()))
        with self.assertRaises(ValueError):
            policy.loss(observation, action, torch.zeros_like(mask))

    def test_reload_reproduces_denoised_sequence_and_action_slice(self):
        import torch
        policy = self.policy().eval()
        observation = torch.zeros(2, 2, 15)
        predicted = policy.predict(observation, seed=42)
        self.assertEqual(predicted.shape, (2, 16, 4))
        self.assertTrue(torch.isfinite(predicted).all())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.pt'
            torch.save(policy.state_dict(), path)
            reloaded = self.policy().eval()
            reloaded.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(reloaded.predict(observation, seed=42), predicted, atol=0, rtol=0)
        torch.testing.assert_close(policy.action_chunk(predicted), predicted[:, 1:3])

    def test_checkpoint_restores_ema_config_and_metadata(self):
        import torch
        from autopicker_dp.train import save_checkpoint, load_policy
        policy = self.policy().eval()
        optimizer = torch.optim.AdamW(policy.parameters())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint.pt'
            save_checkpoint(path, policy, policy, optimizer, {'seed': 9, 'dataset_sha256': 'fixture'})
            loaded, metadata = load_policy(path, os.environ['AUTOPICKER_DP_UPSTREAM'], 'cpu')
            self.assertEqual(metadata['seed'], 9)
            self.assertEqual(loaded.config, policy.config)
            observation = torch.zeros(1, 2, 15)
            torch.testing.assert_close(loaded.predict(observation, 11), policy.predict(observation, 11), atol=0, rtol=0)

    def test_checkpoint_restores_saved_scheduler_not_constructor_defaults(self):
        import torch
        from diffusers import DDIMScheduler
        from autopicker_dp.train import save_checkpoint, load_policy
        policy = self.policy().eval()
        policy.scheduler = DDIMScheduler.from_config(policy.scheduler.config, clip_sample_range=.5)
        optimizer = torch.optim.AdamW(policy.parameters())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint.pt'
            save_checkpoint(path, policy, policy, optimizer, {'scheduler': dict(policy.scheduler.config)})
            loaded, _ = load_policy(path, os.environ['AUTOPICKER_DP_UPSTREAM'], 'cpu')
            self.assertEqual(loaded.scheduler.config.clip_sample_range, .5)

    def test_training_selection_and_checkpoint_preserve_physical_action_mapping(self):
        import numpy as np
        import torch
        from autopicker_dp.dataset import LoadedDataset, Normalizer
        from autopicker_dp.train import training_normalizer, save_checkpoint, load_policy
        data = {'obs': np.arange(45).reshape(3, 15).astype(np.float32),
                'action': np.array([[.1] * 4, [.3] * 4, [.8] * 4], np.float32)}
        original = Normalizer.fit(data, [2, 3], [0])
        dataset = LoadedDataset(data, [2, 3], {'source_configuration': {
            'joint_lower': [0.] * 4, 'joint_upper': [1.] * 4}}, original)
        physical = training_normalizer(dataset, [0], 'physical')
        self.assertEqual(physical.to_dict()['fields']['obs'], original.to_dict()['fields']['obs'])
        self.assertEqual(training_normalizer(dataset, [0], 'data').to_dict(), original.to_dict())
        with self.assertRaises(ValueError):
            training_normalizer(dataset, [0], 'invalid')
        policy = self.policy().eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint.pt'
            save_checkpoint(path, policy, policy, torch.optim.AdamW(policy.parameters()),
                            {'normalizer': physical.to_dict(), 'action_normalization': 'physical'})
            loaded, metadata = load_policy(path, os.environ['AUTOPICKER_DP_UPSTREAM'], 'cpu')
            restored = Normalizer.from_dict(metadata['normalizer'])
            observation = torch.zeros(1, 2, 15)
            actual = restored.denormalize(loaded.predict(observation, 11).detach().numpy(), key='action')
            expected = physical.denormalize(policy.predict(observation, 11).detach().numpy(), key='action')
            np.testing.assert_array_equal(actual, expected)
            self.assertTrue(np.all((actual >= 0) & (actual <= 1)))

    @unittest.skipUnless(importlib.util.find_spec('zarr'), 'Zarr required')
    def test_recovery_training_freezes_normalizer_and_saves_both_identities(self):
        import torch
        from test_augmentation import augmentation_fixture
        from autopicker_dp.train import training_inputs, training_normalizer, save_checkpoint, load_policy
        with tempfile.TemporaryDirectory() as folder:
            base, _, base_hash, recovery, recovery_path = augmentation_fixture(Path(folder))
            normalizer, sequences, info = training_inputs(base, base_hash, 'train', 'physical', recovery_path)
            self.assertEqual(normalizer.to_dict(), training_normalizer(base, [0], 'physical').to_dict())
            self.assertEqual(len(sequences), 45)
            for split, mapping in [('all-pilot', 'physical'), ('train', 'data')]:
                with self.assertRaises(ValueError):
                    training_inputs(base, base_hash, split, mapping, recovery_path)
            policy = self.policy()
            path = Path(folder) / 'checkpoint.pt'
            save_checkpoint(path, policy, policy, torch.optim.AdamW(policy.parameters()),
                            {'dataset_sha256': base_hash, 'normalizer': normalizer.to_dict(),
                             'recovery_augmentation': info})
            _, metadata = load_policy(path, os.environ['AUTOPICKER_DP_UPSTREAM'], 'cpu')
            self.assertEqual(metadata['recovery_augmentation'], info)
            self.assertEqual(metadata['dataset_sha256'], base_hash)
            self.assertNotEqual(info['dataset_sha256'], base_hash)


if __name__ == '__main__':
    unittest.main()
