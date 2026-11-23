"""Pinned upstream temporal U-Net with explicit low-dimensional DDIM semantics."""
from pathlib import Path
import subprocess
import sys

import torch
from diffusers import DDIMScheduler

UPSTREAM_REVISION = '5ba07ac6661db573af695b419a7947ecb704690f'


def upstream_model(path):
    path = Path(path).resolve()
    revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError('Upstream Diffusion Policy revision differs from the frozen version')
    subprocess.run(['git', '-C', str(path), 'diff', '--exit-code', 'HEAD', '--', 'diffusion_policy'],
                   check=True, stdout=subprocess.DEVNULL)
    sys.path.insert(0, str(path))
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    module_path = Path(sys.modules[ConditionalUnet1D.__module__].__file__).resolve()
    if path not in module_path.parents:
        raise ValueError('A different upstream package was already imported')
    return ConditionalUnet1D


class DiffusionPolicy(torch.nn.Module):
    def __init__(self, upstream, down_dims=(64, 128, 256), diffusion_step_embed_dim=64,
                 horizon=16, n_obs_steps=2, n_action_steps=2, inference_steps=20,
                 prediction_type='epsilon'):
        super().__init__()
        if horizon != 16 or n_obs_steps != 2 or n_action_steps != 2 or not 1 <= inference_steps <= 100:
            raise ValueError('This pilot requires horizon/observation/action lengths 16/2/2')
        if prediction_type not in ('epsilon', 'sample'):
            raise ValueError('Unknown diffusion prediction objective')
        self.config = dict(down_dims=list(down_dims), diffusion_step_embed_dim=diffusion_step_embed_dim,
                           horizon=horizon, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps,
                           inference_steps=inference_steps, prediction_type=prediction_type)
        self.model = upstream_model(upstream)(input_dim=4, global_cond_dim=15*n_obs_steps,
                                              diffusion_step_embed_dim=diffusion_step_embed_dim,
                                              down_dims=list(down_dims), kernel_size=5, n_groups=8,
                                              cond_predict_scale=True)
        self.scheduler = DDIMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2',
                                        clip_sample=True, prediction_type=prediction_type,
                                        set_alpha_to_one=True, steps_offset=0)

    def loss(self, observation, action, valid_mask):
        if (observation.shape != (len(action), 2, 15) or action.shape[1:] != (16, 4)
                or valid_mask.shape != action.shape[:2] or not valid_mask.any()
                or not torch.isfinite(observation).all() or not torch.isfinite(action).all()):
            raise ValueError('Invalid diffusion training batch')
        noise = torch.randn_like(action)
        timesteps = torch.randint(0, 100, (len(action),), device=action.device)
        noisy = self.scheduler.add_noise(action, noise, timesteps)
        prediction = self.model(noisy, timesteps, global_cond=observation.flatten(1))
        target = noise if self.config['prediction_type'] == 'epsilon' else action
        return ((prediction - target).square() * valid_mask[..., None]).sum() / (valid_mask.sum() * 4)

    @torch.no_grad()
    def predict(self, observation, seed):
        if observation.ndim != 3 or observation.shape[1:] != (2, 15) or not torch.isfinite(observation).all():
            raise ValueError('Expected finite Bx2x15 observations')
        generator = torch.Generator(device=observation.device).manual_seed(int(seed))
        action = torch.randn((len(observation), 16, 4), generator=generator,
                              device=observation.device, dtype=observation.dtype)
        self.scheduler.set_timesteps(self.config['inference_steps'], device=observation.device)
        for timestep in self.scheduler.timesteps:
            prediction = self.model(action, timestep, global_cond=observation.flatten(1))
            action = self.scheduler.step(prediction, timestep, action, eta=0.).prev_sample
        return action

    def action_chunk(self, prediction):
        return prediction[:, self.config['n_obs_steps'] - 1:
                          self.config['n_obs_steps'] - 1 + self.config['n_action_steps']]
