"""Measured Isaac rollouts for converted commands or a separate diffusion process."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research'))
from autopicker_dp.control import CommandTrajectory
from autopicker_dp.dataset import prepare_dataset, _base_inverse, _observation, _sha256
from autopicker_dp.inference import ActionLoop, PolicyClient
from autopicker_dp.telemetry import ResourceMonitor, write_run_manifest


def verified_sources(path):
    """Verify the immutable export and reconstruct its audited sources without Zarr.

    Isaac need not import training-environment packages. Raw archive hashes and
    all reconstructed manifest fields must agree with the export used to fit.
    """
    path = Path(path)
    checksums = json.loads((path / 'checksums.json').read_text())
    inventory = {item.relative_to(path).as_posix() for item in path.rglob('*')
                 if item.is_file() and item != path / 'checksums.json'}
    if set(checksums) != inventory:
        raise ValueError('Dataset checksum inventory mismatch')
    for name, expected in checksums.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or _sha256(path / name) != expected:
            raise ValueError(f'Dataset checksum mismatch: {name}')
    manifest = json.loads((path / 'manifest.json').read_text())
    counts = [sum(group['split'] == split for group in manifest['target_splits'])
              for split in ('train', 'val', 'test')]
    loaded = prepare_dataset(manifest['source_root'], manifest['dt'], counts)
    if any(manifest.get(key) != value for key, value in loaded.manifest.items()):
        raise ValueError('Raw sources or conversion differ from the frozen training dataset')
    return loaded, _sha256(path / 'manifest.json')


def live_observation(env, inverse):
    arrays = {'q': env.positions()[None],
              'qdot': env.demo.robot.get_joint_velocities()[:, env.arm['indices']],
              'goal_world_m': np.asarray(env.entry['world'])[None],
              'receiver_world_m': env.receiver()[None],
              'attached': np.array([env.arm['attached']])}
    return _observation(arrays, [0], inverse)[0]


class ConvertedReplay:
    def __init__(self, converted, initial_command):
        self.trajectory = CommandTrajectory(np.r_[0., converted['action_time_s']],
                                            np.vstack([initial_command, converted['action']]))
        self.requests = []

    def command(self, env):
        if env.time > self.trajectory.times[-1] + 3.:
            raise RuntimeError('Converted replay exhausted without collecting the target')
        return self.trajectory.at(min(env.time, self.trajectory.times[-1]))


class LivePolicy:
    def __init__(self, client, env, dt, inverse):
        self.inverse, self.client = inverse, client
        self.request_start, self.attempt_start = client.sequence, len(client.attempts)
        self.loop = ActionLoop(client.predict, dt, env.arm['chain'].lower, env.arm['chain'].upper)
        self.loop.reset(env.generation, env.executor.position)
        self.requests = self.loop.requests

    def inference_audit(self):
        return {'inference_request_start': self.request_start, 'inference_request_end': self.client.sequence,
                'inference_attempts': self.client.attempts[self.attempt_start:]}

    def command(self, env):
        generation, steps, world_time = env.generation, env.steps, float(env.world.current_time)
        command = self.loop.command(steps, live_observation(env, self.inverse),
                                    env.executor.position, env.dt)
        if self.requests and self.requests[-1]['time_s'] == env.time:
            self.requests[-1].update(self.client.last_request_identity)
        if (env.generation != generation or env.steps != steps
                or float(env.world.current_time) != world_time):
            raise RuntimeError('Simulation advanced during policy inference')
        return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('converted-replay', 'policy', 'hybrid', 'hybrid-corridor'), required=True,
                        help='hybrid uses diffusion only for coarse approach, then oracle IK and event-gated return')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--inference-python', type=Path)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--split', choices=('train', 'val', 'test', 'all-pilot'), default='val')
    parser.add_argument('--max-episodes', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed-mode', choices=('global', 'episode'), default='global',
                        help='episode fixes noise by trial seed, source episode and local request index')
    parser.add_argument('--timeout', type=float, default=35.)
    parser.add_argument('--scene', type=Path, default=ROOT / 'outputs/autopicker_apple_orchard.usd')
    parser.add_argument('--headless', action='store_true')
    options = parser.parse_args()
    if (not np.isfinite(options.timeout) or options.timeout <= 0
            or options.max_episodes is not None and options.max_episodes < 1):
        parser.error('Require positive finite timeout and positive episode count')
    if options.mode != 'converted-replay' and any(value is None for value in
                                       (options.checkpoint, options.inference_python, options.upstream)):
        parser.error('Policy needs --checkpoint, --inference-python, --upstream')
    dataset, dataset_hash = verified_sources(options.dataset)
    config = dataset.manifest['source_configuration']
    if hashlib.sha256(options.scene.read_bytes()).hexdigest() != config['scene_sha256']:
        raise ValueError('Rollout scene differs from training scene')
    indices = (list(range(len(dataset.episode_ends))) if options.split == 'all-pilot'
               else dataset.episode_indices(options.split))
    if options.max_episodes is not None:
        indices = indices[:options.max_episodes]
    if not indices:
        raise ValueError('No episodes selected')
    options.output.mkdir(parents=True, exist_ok=False)
    checkpoint_sha256 = hashlib.sha256(options.checkpoint.read_bytes()).hexdigest() if options.checkpoint else None
    input_paths = {'dataset_manifest': options.dataset / 'manifest.json', 'scene': options.scene}
    if options.checkpoint is not None:
        input_paths['checkpoint'] = options.checkpoint
    run_manifest = write_run_manifest(
        options.output, ROOT, 'isaac_rollout', configuration=vars(options), input_paths=input_paths,
        extra={'dataset_sha256': dataset_hash, 'checkpoint_sha256': checkpoint_sha256,
               'selected_episode_indices': indices,
               'inference_python': str(options.inference_python) if options.inference_python else None})
    run_manifest_sha256 = hashlib.sha256(
        (options.output / 'run_manifest.json').read_bytes()).hexdigest()
    summary = {'mode': options.mode, 'split': options.split, 'seed': options.seed, 'seed_mode': options.seed_mode,
               'requested': len(indices), 'attempted': 0, 'successes': 0, 'results': [],
               'dataset_sha256': dataset_hash, 'checkpoint_sha256': checkpoint_sha256,
               'run_manifest': 'run_manifest.json', 'run_manifest_sha256': run_manifest_sha256,
               'resource_metrics': 'system_metrics.csv',
               'scope': 'oracle lowdim, fixed scene/reset, pilot only', 'completed': False}
    if options.mode in ('hybrid', 'hybrid-corridor'):
        from autopicker_dp.hybrid import HYBRID_CONFIG, CORRIDOR_CONFIG, HybridPolicy, CorridorHybridPolicy
        hybrid_config = HYBRID_CONFIG if options.mode == 'hybrid' else CORRIDOR_CONFIG
        hybrid_class = HybridPolicy if options.mode == 'hybrid' else CorridorHybridPolicy
        summary.update(hybrid_configuration=dict(hybrid_config),
                       scope='hybrid: diffusion coarse approach + oracle IK precision + event-gated classical return; fixed scene')
    client = app = env = None
    monitor = ResourceMonitor(options.output, interval_s=1.0).start()
    try:
        if options.mode != 'converted-replay':
            client = PolicyClient(options.inference_python, options.checkpoint, options.upstream,
                                  options.seed, options.output / 'inference.log')
            trained = client.ready['metadata']
            if trained['dataset_sha256'] != dataset_hash or trained['dt'] != dataset.manifest['dt']:
                raise ValueError('Checkpoint data identity or action rate differs from rollout')
            summary.update(checkpoint_sha256=client.ready['checkpoint_sha256'],
                           training_split=trained['split'], training_seed=trained['seed'],
                           training_scope=trained['scope'],
                           inference_runtime=trained.get('versions'),
                           observation_augmentation=trained.get('observation_augmentation'))
            if trained['split'] == 'all-pilot':
                summary['scope'] += '; all-pilot overfit diagnostic; no held-out evaluation'
        sys.argv = sys.argv[:1]
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': options.headless, 'width': 640, 'height': 480})
        from autopicker_dp.environment import PickingEnvironment
        from autopicker_dp.run import run_episode
        from isaacsim.core.version import get_version
        from pxr import UsdGeom
        summary['isaac_version'] = list(get_version())
        env = PickingEnvironment(options.scene, config['arm'], config['speed_scale'])
        starts = np.r_[0, dataset.episode_ends[:-1]]
        for ordinal, index in enumerate(indices):
            source = dataset.manifest['episodes'][index]
            env.reset(source['metadata']['target_path'], source['metadata']['reset_joint_positions'])
            recovery = source['metadata'].get('recovery', {})
            if recovery.get('kind') == 'policy_state_reset':
                from autopicker_dp.dynamic_recovery import validated_dynamic_state, restore_dynamic_state
                state = validated_dynamic_state(recovery, source['metadata'])
                restore_dynamic_state(env, state)
            actual_transform = np.array(UsdGeom.Xformable(env.stage.GetPrimAtPath('/World/Autopicker'))
                                        .ComputeLocalToWorldTransform(0))
            if (not np.allclose(actual_transform, config['world_from_base'], rtol=0, atol=1e-9)
                    or env.dt != config['physics_dt'] or env.drive_speed_scale != config['drive_speed_scale']
                    or json.loads(json.dumps(env.demo.config)) != config['controller_config']):
                print('CONFIG_DIAGNOSTIC ' + repr({'actual_transform': actual_transform.tolist(),
                      'expected_transform': config['world_from_base'], 'physics_dt': env.dt,
                      'drive_speed_scale': env.drive_speed_scale,
                      'config_differences': {key: (value, config['controller_config'].get(key))
                                             for key, value in env.demo.config.items()
                                             if value != config['controller_config'].get(key)}}), flush=True)
                raise ValueError('Rollout base transform or control configuration differs from source')
            if client is None:
                converted = {key: values[starts[index]:dataset.episode_ends[index]]
                             for key, values in dataset.data.items()}
                provider = ConvertedReplay(converted, env.executor.position)
            else:
                if options.seed_mode == 'episode':
                    client.reset_episode(int(index))
                provider = LivePolicy(client, env, dataset.manifest['dt'], _base_inverse(config))
                if options.mode in ('hybrid', 'hybrid-corridor'):
                    provider = hybrid_class(provider, env)
            extra = {'mode': options.mode, 'dataset_sha256': dataset_hash,
                     'checkpoint_sha256': summary['checkpoint_sha256'], 'rollout_seed': options.seed,
                     'seed_mode': options.seed_mode,
                     'source_episode_index': index, 'source_split': source['split'],
                     'evaluation_scope': summary['scope'], 'policy_dt': dataset.manifest['dt'],
                     'observation_semantics': dataset.manifest['observation_semantics']}
            if options.mode in ('hybrid', 'hybrid-corridor'):
                extra['hybrid_configuration'] = dict(hybrid_config)
            if recovery.get('kind') == 'policy_state_reset':
                extra['dynamic_initial_state'] = state
            summary['attempted'] += 1
            directory = options.output / f'episode_{ordinal:04d}'
            try:
                result = run_episode(env, directory, None, dataset.manifest['dt'], options.timeout,
                                     action_provider=provider, metadata_extra=extra)
            except RuntimeError:
                # Even an invalid first action can fail before two states exist.
                # Keep its failure record in the denominator, then reset/audit.
                failure_path = directory / 'failure.json'
                if not failure_path.exists():
                    raise
                failure = json.loads(failure_path.read_text())
                result = {'path': str(directory), 'success': False, 'reason': failure['reason'],
                          'audit_error': failure['audit_error'], 'duration_s': env.time}
            result.update(source_episode_index=index, target_path=source['metadata']['target_path'])
            summary['results'].append(result)
            summary['successes'] += int(result['success'])
            (options.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        summary['completed'] = True
        print('AUTOPICKER_ROLLOUT_COMPLETE ' + json.dumps(summary), flush=True)
    except BaseException as error:
        summary['fatal_error'] = str(error)
        import traceback
        traceback.print_exc()
        print('AUTOPICKER_ROLLOUT_FAILED', flush=True)
        raise
    finally:
        try:
            if client is not None:
                client.close()
            if env is not None:
                env.close()
            if app is not None:
                app.close()
        finally:
            summary['resource_monitor'] = monitor.stop()
            (options.output / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
