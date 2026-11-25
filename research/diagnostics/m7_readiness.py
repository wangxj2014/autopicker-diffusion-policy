"""CPU-only archived-trajectory and real inference-protocol checks; no new closed-loop outcome."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'research'), str(ROOT / 'research/tests')]
from autopicker_dp.episode import Episode
from autopicker_dp.environment import capture_errors
from autopicker_dp.hybrid import CORRIDOR_CONFIG
from autopicker_dp.inference import episode_noise_seed, SEED_FIELDS
from validate_m6 import stage_audit, kinematics_audit
from validate_m7 import geometry, pose_series, CHECKPOINT_HASH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    chain, direction, info, geometry_hash = geometry()
    archived = []
    observation = None
    for split, count in (('train', 3), ('val', 2)):
        for ordinal in range(count):
            directory = ROOT / f'runs/autopicker/m6_hybrid_{split}_v1/episode_{ordinal:04d}'
            episode = Episode.load(directory)
            old_stage = stage_audit(episode)
            kine = kinematics_audit(episode, chain, direction)
            a = episode.arrays()
            distance, axial, lateral, angle = pose_series(episode, chain, direction)
            cutoff = next((e['time_s'] for e in episode.events if e['kind'] == 'controller_stage'), np.inf)
            before = a['time_s'] < cutoff
            eligible = before & ((distance <= .03) | ((distance <= .12) & (axial >= 0) & (lateral <= .02) & (angle <= 5.)))
            candidates = np.flatnonzero(eligible)
            best = np.flatnonzero(before)[np.argmin(distance[before])]
            def point(i):
                return {key: float(values[i]) for key, values in (
                    ('time_s', a['time_s']), ('distance_m', distance), ('axial_m', axial),
                    ('lateral_m', lateral), ('angle_degrees', angle))}
            candidate = None
            if len(candidates):
                row = candidates[0]
                candidate = point(row)
                chain.prefix = info['arm']
                chain.lower, chain.upper = np.array(episode.metadata['joint_lower']), np.array(episode.metadata['joint_upper'])
                goal = chain.inverse_approach(a['goal_world_m'][row], a['q'][row], 30.)
                residual = capture_errors(chain.forward(goal), a['goal_world_m'][row], direction)
                if (not np.isfinite(residual).all() or residual[0] > 1e-5 or residual[1] > .01
                        or np.any(goal < chain.lower) or np.any(goal > chain.upper)):
                    raise ValueError('Archived entry pose does not admit an accurate terminal IK goal')
                candidate.update(ik_goal=goal.tolist(), ik_residual_m=residual[0], ik_angle_degrees=residual[1])
            archived.append({'source_split': split, 'source_episode_index': episode.metadata['source_episode_index'],
                'target_path': episode.metadata['target_path'], 'm6_precision_entry_s': old_stage['precision_entry_s'],
                'hypothetical_corridor_entry': candidate, 'closest_pre_handoff': point(best),
                'max_fk_measured_tip_error_m': kine['max_fk_measured_tip_error_m'],
                'm6_command_reconstruction_error': old_stage['max_applied_command_error'],
                'episode_json_sha256': hashlib.sha256((directory / 'episode.json').read_bytes()).hexdigest(),
                'states_sha256': hashlib.sha256((directory / 'states.npz').read_bytes()).hexdigest()})
            if observation is None:
                observation = episode.metadata['inference_requests'][0]['observation']
    requests = []
    for request_id, source, index in ((1, 4, 0), (2, 4, 1), (3, 5, 0), (37, 4, 0)):
        requests.append(dict(request_id=request_id, generation=7, time_s=.2*index, obs=observation,
                             source_episode_index=source, episode_request_index=index,
                             noise_seed=episode_noise_seed(0, source, index)))
    command = [sys.executable, '-I', str(ROOT/'research/autopicker_dp/serve.py'),
        '--checkpoint', str(ROOT/'runs/autopicker/m5_train_sample_long_v1/checkpoint.pt'),
        '--upstream', str(ROOT/'work/diffusion_policy_upstream'), '--seed', '0', '--device', 'cpu']
    process = subprocess.run(command, input=''.join(json.dumps(r)+'\n' for r in requests),
                             capture_output=True, text=True, timeout=120, cwd=ROOT)
    if process.returncode:
        raise RuntimeError(process.stderr)
    responses = [json.loads(line) for line in process.stdout.splitlines()]
    if len(responses) != len(requests) + 1 or responses[0]['checkpoint_sha256'] != CHECKPOINT_HASH:
        raise ValueError('Wrong checkpoint or incomplete protocol smoke')
    for request, response in zip(requests, responses[1:]):
        if any(response.get(key) != request[key] for key in ('request_id', 'generation', 'time_s') + SEED_FIELDS):
            raise ValueError('Inference process failed to echo request identity')
        action = np.array(response['action'])
        if action.shape != (2, 4) or not np.isfinite(action).all():
            raise ValueError('Invalid real inference response')
    error = float(np.max(np.abs(np.array(responses[1]['action'])-responses[-1]['action'])))
    if error != 0.:
        raise ValueError('CPU inference changed with global request sequence')
    report = {'schema': 'autopicker-m7-readiness-v1', 'new_closed_loop_trials': 0,
        'scope': 'CPU protocol and hypothetical gates on archived M6 states; not new rollout success evidence',
        'checkpoint_sha256': CHECKPOINT_HASH, 'kinematics_sha256': geometry_hash,
        'corridor_configuration': CORRIDOR_CONFIG, 'archived_episodes': archived,
        'real_serve_cpu_smoke': {'device': 'cpu', 'requests': len(requests), 'seed': 0,
            'requests_and_responses': [{'request_identity': {k:v for k,v in r.items() if k!='obs'}, 'action': s['action']}
                                      for r,s in zip(requests,responses[1:])],
            'same_episode_different_global_id_max_action_error': error,
            'scope': 'CPU generator only; does not establish CUDA numerical equivalence'}}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('AUTOPICKER_M7_CPU_READINESS_PASS', flush=True)


if __name__ == '__main__':
    main()
