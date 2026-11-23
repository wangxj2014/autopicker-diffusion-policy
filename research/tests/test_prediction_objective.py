import os
from pathlib import Path
import tempfile
import unittest

from test_policy import AVAILABLE


@unittest.skipUnless(AVAILABLE and os.environ.get('AUTOPICKER_DP_UPSTREAM'), 'Pinned training environment required')
class PredictionObjectiveTests(unittest.TestCase):
    def policy(self, prediction_type='sample'):
        from autopicker_dp.policy import DiffusionPolicy
        return DiffusionPolicy(os.environ['AUTOPICKER_DP_UPSTREAM'], down_dims=(16, 32),
            diffusion_step_embed_dim=16, inference_steps=5, prediction_type=prediction_type)

    def test_clean_action_loss_uses_action_target_and_masks_padding(self):
        import torch
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.))
            def forward(self, sample, timestep, global_cond):
                return self.value.expand_as(sample)
        policy = self.policy()
        policy.model = Constant()
        actions = torch.full((2, 16, 4), .25)
        actions[:, -3:] = 100.
        valid = torch.ones((2, 16), dtype=torch.bool)
        valid[:, -3:] = False
        loss = policy.loss(torch.zeros((2, 2, 15)), actions, valid)
        self.assertAlmostEqual(loss.item(), .25**2)
        loss.backward()
        self.assertAlmostEqual(policy.model.value.grad.item(), -.5, places=6)
        self.assertEqual(policy.scheduler.config.prediction_type, 'sample')

    def test_sample_checkpoint_preserves_objective_and_predictions(self):
        import torch
        from autopicker_dp.train import save_checkpoint, load_policy
        torch.set_num_threads(1)
        policy = self.policy().eval()
        obs = torch.zeros((1, 2, 15))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint.pt'
            save_checkpoint(path, policy, policy, torch.optim.AdamW(policy.parameters()),
                            {'scheduler': dict(policy.scheduler.config)})
            loaded, _ = load_policy(path, os.environ['AUTOPICKER_DP_UPSTREAM'], 'cpu')
            self.assertEqual(loaded.config['prediction_type'], 'sample')
            torch.testing.assert_close(loaded.predict(obs, 42), policy.predict(obs, 42), atol=0, rtol=0)
            checkpoint = torch.load(path, weights_only=True)
            checkpoint['metadata']['scheduler']['prediction_type'] = 'epsilon'
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'objective'):
                load_policy(path, os.environ['AUTOPICKER_DP_UPSTREAM'], 'cpu')

    def test_sample_ddim_decodes_known_clean_action_for_any_noise_seed(self):
        import torch
        class KnownAction(torch.nn.Module):
            def forward(self, sample, timestep, global_cond):
                return torch.full_like(sample, .25)
        policy = self.policy()
        policy.model = KnownAction()
        for seed in (0, 42):
            predicted = policy.predict(torch.zeros((1, 2, 15)), seed)
            torch.testing.assert_close(predicted, torch.full_like(predicted, .25), atol=0, rtol=0)

    def test_legacy_checkpoint_defaults_to_epsilon_and_rejects_unknown_objective(self):
        import torch
        from autopicker_dp.train import save_checkpoint, load_policy
        with self.assertRaises(ValueError):
            self.policy('invalid')
        policy = self.policy('epsilon')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint.pt'
            save_checkpoint(path, policy, policy, torch.optim.AdamW(policy.parameters()), {})
            checkpoint = torch.load(path, weights_only=True)
            del checkpoint['model_config']['prediction_type']
            torch.save(checkpoint, path)
            loaded, _ = load_policy(path, os.environ['AUTOPICKER_DP_UPSTREAM'], 'cpu')
            self.assertEqual(loaded.scheduler.config.prediction_type, 'epsilon')
            torch.testing.assert_close(loaded.predict(torch.zeros((1, 2, 15)), 9),
                                       policy.predict(torch.zeros((1, 2, 15)), 9), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
