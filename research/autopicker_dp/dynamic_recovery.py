"""Expert correction from measured failed-policy states, preserving active velocity."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research'))
from autopicker_dp.control import vector
from autopicker_dp.dataset import _configuration, _sha256
from autopicker_dp.episode import Episode


def state_from_policy(source, time_s, indices):
    source.audit()
    arrays = source.arrays()
    indices = np.asarray(indices)
    if (source.metadata.get('mode') != 'policy' or source.outcome['success']
            or not np.isfinite(time_s) or indices.shape != (4,)
            or not np.issubdtype(indices.dtype, np.integer)
            or len(set(indices.tolist())) != 4 or np.any(indices < 0) or np.any(indices >= 16)):
        raise ValueError('Dynamic recovery requires a failed policy and valid active indices/time')
    row = int(np.searchsorted(arrays['time_s'], time_s))
    if (row < 1 or row >= len(arrays['time_s']) - 1
            or abs(arrays['time_s'][row] - time_s) > 1e-8 or arrays['attached'][:row + 1].any()
            or not np.allclose(arrays['q_all'][row, indices], arrays['q'][row], atol=1e-8, rtol=0)):
        raise ValueError('Recovery time must be an exact unattached raw state with preceding command')
    # Row r records state BEFORE applying command[r]. The live executor at that
    # instant still holds command[r-1] and its velocity, not command[r].
    return {'source_row': row, 'source_time_s': float(arrays['time_s'][row]),
            'active_indices': indices.tolist(), 'q_all': arrays['q_all'][row].tolist(),
            'qdot': arrays['qdot'][row].tolist(),
            'executor_position': arrays['command'][row - 1].tolist(),
            'executor_velocity': arrays['velocity_command'][row - 1].tolist(),
            'inactive_velocity': 'zero; unrecorded by legacy source',
            'restore_semantics': 'articulated state only; no PhysX solver checkpoint or settling'}


def restore_dynamic_state(env, state):
    """Restore before the first recorded step; never advance or settle physics."""
    q_all = vector(state['q_all'], 16)
    qdot = vector(state['qdot'])
    command = vector(state['executor_position'])
    velocity = vector(state['executor_velocity'])
    indices = np.asarray(state['active_indices'])
    executor = env.executor
    if (env.steps != 0 or indices.shape != (4,)
            or not np.issubdtype(indices.dtype, np.integer)
            or not np.array_equal(indices, env.arm['indices'])
            or np.any(command < executor.lower) or np.any(command > executor.upper)
            or np.any(np.abs(velocity) > executor.max_velocity + 1e-8)
            or np.any(q_all[indices] < executor.lower - .001)
            or np.any(q_all[indices] > executor.upper + .001)):
        raise ValueError('Invalid dynamic reset state or executor bounds')
    full_velocity = np.zeros((1, 16))
    full_velocity[0, indices] = qdot
    full_command = q_all[None].copy()
    full_command[0, indices] = command
    velocity_command = np.zeros((1, 16))
    velocity_command[0, indices] = velocity
    robot = env.demo.robot
    robot.set_joint_positions(q_all[None])
    robot.set_joint_velocities(full_velocity)
    robot.set_joint_position_targets(full_command)
    robot.set_joint_velocity_targets(velocity_command)
    executor.reset(command)
    executor.velocity = velocity.copy()
    env.full_command = full_command
    env.initial_all = q_all.copy()
    env.reset_requested = q_all.copy()
    measured_q = vector(robot.get_joint_positions()[0], 16)
    measured_velocity = vector(robot.get_joint_velocities()[0], 16)
    audit = {'max_position_error': float(np.max(np.abs(measured_q - q_all))),
             'max_active_velocity_error': float(np.max(np.abs(measured_velocity[indices] - qdot)))}
    if max(audit.values()) > 1e-6:
        raise RuntimeError('Simulator did not restore the requested articulated state')
    return audit


def validated_dynamic_state(info, metadata):
    """Bind the restored state to a recorded train-policy trajectory."""
    source_path = Path(info['policy_episode_path'])
    source = Episode.load(source_path)
    index = info['source_episode_index']
    if (info['kind'] != 'policy_state_reset' or info['source_split'] != 'train'
            or source.metadata.get('dataset_sha256') != info['source_dataset_sha256']
            or source.metadata.get('source_episode_index') != index or source.metadata.get('source_split') != 'train'
            or source.metadata.get('checkpoint_sha256') != info['policy_checkpoint_sha256']
            or metadata['target_path'] != source.metadata['target_path']
            or _configuration(source.metadata) != _configuration(metadata)
            or _sha256(source_path / 'episode.json') != info['policy_episode_json_sha256']
            or json.loads((source_path / 'episode.json').read_text())['sha256'] != info['policy_raw_sha256']):
        raise ValueError('Dynamic recovery policy source identity or train split mismatch')
    expected = state_from_policy(source, info['state']['source_time_s'], info['state']['active_indices'])
    if info['state'] != expected or metadata['reset_joint_positions'] != expected['q_all']:
        raise ValueError('Dynamic recovery state differs from policy source')
    return expected


def validate_dynamic_provenance(info, metadata, parent, parent_hash):
    state = validated_dynamic_state(info, metadata)
    specification = parent.manifest['episodes'][info['source_episode_index']]
    if (info['source_dataset_sha256'] != parent_hash or specification['split'] != 'train'
            or metadata['target_path'] != specification['metadata']['target_path']
            or _configuration(metadata) != parent.manifest['source_configuration']):
        raise ValueError('Dynamic recovery parent target or split mismatch')
    return state


def source_trials(dataset, dataset_hash, folder, checkpoint):
    summary = json.loads((folder / 'summary.json').read_text())
    model = json.loads((checkpoint.parent / 'report.json').read_text())
    if (not summary['completed'] or summary['mode'] != 'policy' or summary['split'] != 'train'
            or summary['dataset_sha256'] != dataset_hash or summary['requested'] != 3
            or summary['attempted'] != 3 or len(summary['results']) != 3
            or _sha256(checkpoint) != summary['checkpoint_sha256']
            or model['checkpoint_sha256'] != summary['checkpoint_sha256']
            or model['dataset_sha256'] != dataset_hash or model['split'] != 'train'):
        raise ValueError('Dynamic correction requires three completed train-only policy trials')
    trials = []
    for ordinal, result in enumerate(summary['results']):
        index = result['source_episode_index']
        if index != dataset.episode_indices('train')[:3][ordinal] or result['success']:
            raise ValueError('Unexpected policy source target/order or successful trial')
        path = folder / f'episode_{ordinal:04d}'
        source = Episode.load(path)
        parent = dataset.manifest['episodes'][index]
        info = {'kind': 'policy_state_reset', 'source_dataset_sha256': dataset_hash,
                'source_episode_index': index, 'source_split': 'train',
                'source_episode_json_sha256': parent['episode_json_sha256'],
                'source_raw_sha256': parent['raw_sha256'],
                'policy_episode_path': str(path.resolve()), 'policy_episode_json_sha256': _sha256(path / 'episode.json'),
                'policy_raw_sha256': json.loads((path / 'episode.json').read_text())['sha256'],
                'policy_checkpoint_sha256': summary['checkpoint_sha256']}
        for time_s in (2., 6.):
            trials.append((source, info.copy(), time_s))
    return trials


def main():
    from autopicker_dp.rollout import verified_sources
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--policy-rollout', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=35.)
    parser.add_argument('--headless', action='store_true')
    options = parser.parse_args()
    if not np.isfinite(options.timeout) or options.timeout <= 0:
        parser.error('Require a positive finite timeout')
    dataset, dataset_hash = verified_sources(options.dataset)
    config = dataset.manifest['source_configuration']
    trials = source_trials(dataset, dataset_hash, options.policy_rollout, options.checkpoint)
    scene = ROOT / 'outputs/autopicker_apple_orchard.usd'
    if _sha256(scene) != config['scene_sha256']:
        raise ValueError('Scene differs from source')
    options.output.mkdir(parents=True, exist_ok=False)
    summary = {'mode': 'expert-dynamic-recovery', 'dataset_sha256': dataset_hash,
        'policy_checkpoint_sha256': _sha256(options.checkpoint), 'requested': len(trials),
        'attempted': 0, 'successes': 0, 'completed': False, 'results': []}
    app = env = None
    try:
        sys.argv = sys.argv[:1]
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': options.headless, 'width': 640, 'height': 480})
        from autopicker_dp.environment import PickingEnvironment
        from autopicker_dp.run import run_episode
        env = PickingEnvironment(scene, config['arm'], config['speed_scale'])
        for ordinal, (source, info, time_s) in enumerate(trials):
            directory = options.output / f'episode_{ordinal:04d}'
            summary['attempted'] += 1
            try:
                info['state'] = state_from_policy(source, time_s, env.arm['indices'])
                state = info['state']
                validate_dynamic_provenance(info, {**source.metadata, 'reset_joint_positions': state['q_all']},
                                            dataset, dataset_hash)
                env.reset(source.metadata['target_path'], state['q_all'])
                info['restore_audit'] = restore_dynamic_state(env, state)
                result = run_episode(env, directory, None, dataset.manifest['dt'], options.timeout,
                                     metadata_extra={'recovery': info}, post_collect_hold=.2)
            except (ValueError, RuntimeError) as error:
                if not directory.exists():
                    directory.mkdir()
                    (directory / 'failure.json').write_text(json.dumps({
                        'reason': str(error), 'recovery': info, 'target_path': source.metadata['target_path']}, indent=2))
                failure = directory / 'failure.json'
                reason = json.loads(failure.read_text())['reason'] if failure.exists() else str(error)
                result = {'path': str(directory), 'success': False, 'reason': reason}
            result.update(source_episode_index=info['source_episode_index'], source_time_s=time_s,
                          target_path=source.metadata['target_path'])
            summary['results'].append(result)
            summary['successes'] += int(result['success'])
            (options.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        summary['completed'] = True
        print('AUTOPICKER_DYNAMIC_RECOVERY_COMPLETE ' + json.dumps(summary), flush=True)
    except BaseException as error:
        summary['fatal_error'] = str(error)
        import traceback
        traceback.print_exc()
        raise
    finally:
        (options.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == '__main__':
    main()
