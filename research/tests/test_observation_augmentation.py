import importlib.util
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
AVAILABLE = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(AVAILABLE, 'PyTorch required')
class ObservationAugmentationTests(unittest.TestCase):
    def test_noise_changes_only_velocity_channels(self):
        import torch
        from autopicker_dp.train import add_velocity_observation_noise
        torch.manual_seed(12)
        observation = torch.zeros((3, 2, 15))
        augmented = add_velocity_observation_noise(observation, [0.1, 0.2, 0.3, 0.4])
        self.assertFalse(torch.equal(augmented[..., 4:8], observation[..., 4:8]))
        torch.testing.assert_close(augmented[..., :4], observation[..., :4])
        torch.testing.assert_close(augmented[..., 8:], observation[..., 8:])
        torch.testing.assert_close(observation, torch.zeros_like(observation))

    def test_zero_noise_is_identity(self):
        import torch
        from autopicker_dp.train import add_velocity_observation_noise
        observation = torch.randn((2, 2, 15))
        self.assertIs(add_velocity_observation_noise(observation, [0., 0., 0., 0.]), observation)

    def test_rejects_bad_shape_or_noise(self):
        import torch
        from autopicker_dp.train import add_velocity_observation_noise
        with self.assertRaises(ValueError):
            add_velocity_observation_noise(torch.zeros((2, 15)), [0.] * 4)
        with self.assertRaises(ValueError):
            add_velocity_observation_noise(torch.zeros((1, 2, 15)), [-0.1] * 4)
        with self.assertRaises(ValueError):
            add_velocity_observation_noise(torch.zeros((1, 2, 15)), [0.] * 3)


if __name__ == '__main__':
    unittest.main()
