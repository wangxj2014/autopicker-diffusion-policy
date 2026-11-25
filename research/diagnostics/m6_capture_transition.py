"""Reproduce three selected M5 retreat probes; counterfactuals never control Isaac."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research'))
from autopicker_dp.dataset import Normalizer, load_dataset
from autopicker_dp.environment import capture_errors
from autopicker_dp.episode import Episode
from autopicker_dp.train import load_policy
from autopicker_apple_pick import JointChain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    geometry_path = ROOT / 'docs/research/2026-11-23-m6-kinematics.json'
    info = json.loads(geometry_path.read_text())
    chain = JointChain.__new__(JointChain)
    chain.base = np.array(info['base'])
    chain.frames = [(np.array(a), np.array(b), kind, np.array(axis)) for a, b, kind, axis in info['frames']]
    data = load_dataset(ROOT / 'datasets/autopicker/lowdim_v2')
    if info['scene_sha256'] != data.manifest['source_configuration']['scene_sha256']:
        raise ValueError('Kinematic export belongs to a different scene')
    checkpoint = ROOT / 'runs/autopicker/m5_train_sample_long_v1/checkpoint.pt'
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != 'd0d90421470e49ac5c3828e5486ade87da326907a33f994d725b5fc18d47f85a':
        raise ValueError('Expected frozen M5 sample 30000 checkpoint')
    policy, metadata = load_policy(checkpoint, ROOT / 'work/diffusion_policy_upstream', args.device)
    norm = Normalizer.from_dict(metadata['normalizer'])
    starts = np.r_[0, data.episode_ends[:-1]]
    train = np.concatenate([data.data['obs'][starts[i]:data.episode_ends[i]] for i in data.episode_indices('train')])
    probes = []
    for ordinal, timestamp in ((0, 18.6), (1, 13.8), (2, 12.6)):
        folder = ROOT / 'runs/autopicker/m5_sample_long_train_v1' / f'episode_{ordinal:04d}'
        episode = Episode.load(folder)
        arrays = episode.arrays()
        request = min(episode.metadata['inference_requests'], key=lambda r: abs(r['time_s'] - timestamp))
        if abs(request['time_s'] - timestamp) > 1e-9:
            raise ValueError('Selected diagnostic request missing')
        obs = np.array(request['observation'], dtype=np.float32)
        goal = arrays['goal_world_m'][0]
        direction = np.array(info['direction'])
        fk_error = max(np.linalg.norm(chain.forward(q)[:3, 3] - tip)
                       for q, tip in zip(arrays['q'][::48], arrays['tip_world_m'][::48]))
        if fk_error > 1e-5:
            raise ValueError('Exported FK does not match the simulated robot')
        variants = {key: obs.copy() for key in ('actual', 'zero_velocity', 'nearest_expert_velocity', 'attached_true')}
        variants['zero_velocity'][:, 4:8] = 0.
        variants['attached_true'][:, -1] = 1.
        neighbors = []
        for i in range(2):
            # Diagnostic metric: normalized joint position squared distance +
            # 100 * goal position squared distance, same attachment state.
            metric = (((train[:, :4] - obs[i, :4]) * norm.fields['obs']['scale'][:4])**2).sum(-1)
            metric += ((train[:, 8:11] - obs[i, 8:11])**2).sum(-1) * 100
            metric[train[:, -1] != obs[i, -1]] = np.inf
            neighbor = int(metric.argmin())
            neighbors.append(neighbor)
            variants['nearest_expert_velocity'][i, 4:8] = train[neighbor, 4:8]
        outputs = {}
        for label, modified in variants.items():
            prediction = policy.action_chunk(policy.predict(torch.as_tensor(
                norm.normalize(modified, key='obs')[None], device=args.device), request['request_id']))[0].cpu().numpy()
            actions = norm.denormalize(prediction, key='action')
            outputs[label] = {'action': actions.tolist(), 'observation': modified.tolist(),
                'distance_m_and_angle_degrees': [list(capture_errors(chain.forward(q), goal, direction)) for q in actions]}
        # CPU and CUDA use different random generators and numerical kernels.
        # Original logged actions are authoritative for the measured rollout;
        # counterfactual comparisons use the same selected device and seed.
        outputs['logged_actual'] = {'action': request['action'],
            'distance_m_and_angle_degrees': [list(capture_errors(chain.forward(q), goal, direction))
                                           for q in request['action']]}
        raw_row = round(timestamp / episode.metadata['physics_dt'])
        probes.append({'target_path': episode.metadata['target_path'], 'time_s': timestamp,
            'request_id': request['request_id'], 'measured_distance_m': float(np.linalg.norm(arrays['tip_world_m'][raw_row] - goal)),
            'attached': bool(arrays['attached'][raw_row]), 'fk_measured_max_error_m': float(fk_error),
            'parent_train_neighbor_rows': neighbors, 'outputs': outputs,
            'episode_json_sha256': hashlib.sha256((folder / 'episode.json').read_bytes()).hexdigest()})
    report = {'schema': 'autopicker-m6-transition-diagnostic-v1', 'device': args.device,
        'checkpoint_sha256': checkpoint_hash, 'kinematics_sha256': hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
        'selection': 'Three M5 train retreat requests selected during investigation, not an independent statistical sample',
        'scope': 'offline counterfactual only; nearest expert velocity is never supplied to the live controller',
        'probes': probes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('AUTOPICKER_M6_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
