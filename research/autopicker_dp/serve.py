"""Dedicated low-dimensional GPU inference process; stdout is JSON only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autopicker_dp.dataset import Normalizer
from autopicker_dp.train import load_policy
from autopicker_dp.inference import SEED_FIELDS, episode_noise_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    options = parser.parse_args()
    torch.set_num_threads(2)
    policy, metadata = load_policy(options.checkpoint, options.upstream, options.device)
    normalizer = Normalizer.from_dict(metadata['normalizer'])
    policy.predict(torch.zeros((1, 2, 15), device=options.device), seed=options.seed)
    print(json.dumps({'status': 'ready', 'checkpoint_sha256': hashlib.sha256(options.checkpoint.read_bytes()).hexdigest(),
                      'metadata': metadata, 'policy_config': policy.config}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        observation = np.asarray(request['obs'], dtype=np.float32)
        if observation.shape != (2, 15) or not np.isfinite(observation).all():
            raise ValueError('Invalid observation message')
        normalized = normalizer.normalize(observation, key='obs')
        tensor = torch.as_tensor(normalized[None], device=options.device, dtype=torch.float32)
        noise_seed = options.seed + int(request['request_id'])
        if any(key in request for key in SEED_FIELDS):
            noise_seed = episode_noise_seed(options.seed, request['source_episode_index'],
                                           request['episode_request_index'])
            if request['noise_seed'] != noise_seed:
                raise ValueError('Episode inference seed mismatch')
        prediction = policy.predict(tensor, seed=noise_seed)
        action = normalizer.denormalize(policy.action_chunk(prediction)[0].cpu().numpy(), key='action')
        response = {key: request[key] for key in ('request_id', 'generation', 'time_s')}
        response.update({key: request[key] for key in SEED_FIELDS if key in request})
        response['action'] = action.tolist()
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
