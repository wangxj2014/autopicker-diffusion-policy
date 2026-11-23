"""Collect expert recovery from perturbed, unattached training-target resets.

This is a static reset augmentation pilot, not on-policy DAgger: velocities
are zeroed and physically settled before recording. No held-out target or
learned evaluation trajectory is used to generate these demonstrations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research'))


def recovery_sources(manifest):
    """First recorded episode per training (scene, target), in export order."""
    selected, seen = [], set()
    for index, episode in enumerate(manifest['episodes']):
        key = (episode['metadata']['scene_sha256'], episode['metadata']['target_path'])
        if episode['split'] == 'train' and key not in seen:
            selected.append(index)
            seen.add(key)
    if not selected:
        raise ValueError('Recovery needs training targets')
    return selected


def recovery_reset(source, indices, lower, upper, delta):
    """Perturb a measured state 0.5 s before capture; reject clipping or attachment."""
    arrays = source.arrays()
    captures = [event['time_s'] for event in source.events if event['kind'] == 'capture']
    if not source.outcome['success'] or len(captures) != 1 or captures[0] < .5:
        raise ValueError('Recovery needs a successful source with pre-capture history')
    row = int(np.searchsorted(arrays['time_s'], captures[0] - .5, side='right') - 1)
    indices = np.asarray(indices)
    lower, upper, delta = (np.asarray(value, dtype=float) for value in (lower, upper, delta))
    if (row < 0 or indices.shape != (4,) or not np.issubdtype(indices.dtype, np.integer)
            or len(set(indices.tolist())) != 4 or np.any(indices < 0) or np.any(indices >= 16)
            or any(value.shape != (4,) or not np.isfinite(value).all() for value in (lower, upper, delta))
            or np.any(upper <= lower) or arrays['attached'][row]):
        raise ValueError('Invalid recovery source state, indices or perturbation')
    reset = arrays['q_all'][row].copy()
    reference = arrays['q'][row]
    if reset.shape != (16,) or not np.isfinite(reset).all() or not np.allclose(reset[indices], reference, atol=1e-8, rtol=0):
        raise ValueError('Source active joints disagree with full robot state')
    reset[indices] = reference + delta
    if np.any(reset[indices] < lower) or np.any(reset[indices] > upper):
        raise ValueError('Recovery perturbation crosses joint limits; no silent clipping')
    return reset, {'kind': 'perturbed_pre_capture_reset', 'source_row': row,
                   'source_time_s': float(arrays['time_s'][row]), 'source_capture_time_s': float(captures[0]),
                   'reference_q': reference.tolist(), 'delta_q': delta.tolist(),
                   'reset_q': reset[indices].tolist(), 'active_indices': indices.tolist(),
                   'reset_velocity': 'zero_then_settle', 'clipped': False}


def collect_recovery_trial(env, source_root, specification, index, sign, amplitude, dataset_hash,
                           directory, dt, timeout, runner=None):
    """Persist reset preparation failures in the same denominator as executed trials."""
    from autopicker_dp.episode import Episode
    if runner is None:
        from autopicker_dp.run import run_episode
        runner = run_episode
    provenance = {'source_dataset_sha256': dataset_hash, 'source_episode_index': index, 'source_split': 'train',
                  'source_episode_json_sha256': specification['episode_json_sha256'],
                  'source_raw_sha256': specification['raw_sha256'], 'delta_q': (sign * amplitude).tolist()}
    target = specification['metadata']['target_path']
    try:
        source = Episode.load(Path(source_root) / specification['path'])
        reset, details = recovery_reset(source, env.arm['indices'], source.metadata['joint_lower'],
                                        source.metadata['joint_upper'], sign * amplitude)
        provenance.update(details)
        env.reset(target, reset)
        provenance['settled_q'] = env.positions().tolist()
        result = runner(env, directory, None, dt, timeout,
                        metadata_extra={'recovery': provenance}, post_collect_hold=.2)
    except (RuntimeError, ValueError) as error:
        if not directory.exists():
            directory.mkdir()
            (directory / 'failure.json').write_text(json.dumps({
                'reason': str(error), 'recovery': provenance, 'target_path': target}, indent=2))
        result = {'path': str(directory), 'success': False, 'reason': str(error)}
        failure_path = directory / 'failure.json'
        if failure_path.exists():
            failure = json.loads(failure_path.read_text())
            result['reason'] = failure['reason']
            if 'audit_error' in failure:
                result['audit_error'] = failure['audit_error']
    result.update(source_episode_index=index, target_path=target, sign=sign)
    return result


def main():
    from autopicker_dp.rollout import verified_sources

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', type=Path, default=ROOT / 'outputs/autopicker_apple_orchard.usd')
    parser.add_argument('--timeout', type=float, default=35.)
    parser.add_argument('--headless', action='store_true')
    options = parser.parse_args()
    if not np.isfinite(options.timeout) or options.timeout <= 0:
        parser.error('Require finite positive timeout')
    dataset, digest = verified_sources(options.dataset)
    configuration = dataset.manifest['source_configuration']
    if hashlib.sha256(options.scene.read_bytes()).hexdigest() != configuration['scene_sha256']:
        raise ValueError('Recovery scene differs from source')
    selected = recovery_sources(dataset.manifest)
    # A fixed paired perturbation per target makes the pilot deterministic.
    amplitude = np.array([np.deg2rad(3.), .025, .02, np.deg2rad(5.)])
    trials = [(index, sign) for index in selected for sign in (-1, 1)]
    options.output.mkdir(parents=True, exist_ok=False)
    summary = {'mode': 'expert-recovery', 'dataset_sha256': digest,
               'scope': 'train targets only; static perturbed reset; zero velocity then settle',
               'requested': len(trials), 'attempted': 0, 'successes': 0, 'results': [],
               'completed': False, 'perturbation_amplitude_native_units': amplitude.tolist()}
    app = env = None
    try:
        sys.argv = sys.argv[:1]
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': options.headless, 'width': 640, 'height': 480})
        from autopicker_dp.environment import PickingEnvironment
        from pxr import UsdGeom
        env = PickingEnvironment(options.scene, configuration['arm'], configuration['speed_scale'])
        actual_transform = np.array(UsdGeom.Xformable(env.stage.GetPrimAtPath('/World/Autopicker'))
                                    .ComputeLocalToWorldTransform(0))
        if (env.dt != configuration['physics_dt'] or env.drive_speed_scale != configuration['drive_speed_scale']
                or not np.allclose(actual_transform, configuration['world_from_base'], atol=1e-9, rtol=0)
                or json.loads(json.dumps(env.demo.config)) != configuration['controller_config']
                or list(env.arm['chain'].names) != configuration['joint_names']
                or not np.array_equal(env.arm['chain'].lower, configuration['joint_lower'])
                or not np.array_equal(env.arm['chain'].upper, configuration['joint_upper'])):
            raise ValueError('Recovery environment configuration differs from source')
        for ordinal, (index, sign) in enumerate(trials):
            specification = dataset.manifest['episodes'][index]
            directory = options.output / f'episode_{ordinal:04d}'
            summary['attempted'] += 1
            result = collect_recovery_trial(env, dataset.manifest['source_root'], specification,
                index, sign, amplitude, digest, directory, dataset.manifest['dt'], options.timeout)
            summary['results'].append(result)
            summary['successes'] += int(result['success'])
            (options.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        summary['completed'] = True
        print('AUTOPICKER_RECOVERY_COMPLETE ' + json.dumps(summary), flush=True)
    except BaseException as error:
        summary['fatal_error'] = str(error)
        import traceback
        traceback.print_exc()
        print('AUTOPICKER_RECOVERY_FAILED', flush=True)
        raise
    finally:
        (options.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == '__main__':
    main()
