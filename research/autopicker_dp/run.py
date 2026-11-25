"""Collect, replay or audit single-arm Autopicker research episodes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research'))
SOURCE_FILES = sorted((ROOT / 'research/autopicker_dp').glob('*.py')) + [
    ROOT / 'outputs/autopicker_apple_pick.py', ROOT / 'outputs/autopicker_depth_fusion.py',
    ROOT / 'outputs/autopicker_camera_viewer.py']
SOURCE_HASHES = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in SOURCE_FILES}

from autopicker_dp.control import CommandTrajectory
from autopicker_dp.episode import Episode


def repository_revision():
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    return revision, dirty


def record_post_collect(env, duration, camera, sample_steps):
    steps = round(duration / env.dt)
    if not np.isfinite(duration) or duration < 0 or abs(steps * env.dt - duration) > 1e-8:
        raise ValueError('Post-collect hold must be a nonnegative multiple of physics_dt')
    if not env.success:
        raise ValueError('Post-collect recording requires a collected target')
    command = env.executor.position.copy()
    for _ in range(steps):
        env.step(command)
        env.update_task()
        if camera is not None and env.steps % sample_steps == 0:
            camera.capture()


def run_episode(environment, directory, camera, sample_dt, timeout_s, source=None, replay_dt=.1,
                action_provider=None, metadata_extra=None, post_collect_hold=0.):
    from autopicker_dp.environment import ExpertPlanner
    from pxr import UsdGeom
    from isaacsim.core.version import get_version

    env = environment
    if source is not None and action_provider is not None:
        raise ValueError('Source replay and external policy are mutually exclusive')
    revision, dirty = repository_revision()
    metadata = {'arm': env.arm_name, 'target_path': env.entry['path'], 'physics_dt': env.dt,
                'joint_names': env.arm['chain'].names, 'joint_units': ['rad', 'm', 'm', 'rad'],
                'joint_lower': env.arm['chain'].lower.tolist(), 'joint_upper': env.arm['chain'].upper.tolist(),
                'reset_joint_positions': env.reset_requested.tolist(), 'generation': env.generation,
                'reset_orchard_audit': env.reset_orchard_audit,
                'git_revision': revision, 'git_dirty': dirty, 'isaac_version': list(get_version()),
                'scene_sha256': hashlib.sha256(env.scene.read_bytes()).hexdigest(),
                'speed_scale': env.demo.config['speed_scale'], 'grasp_error_every': 0,
                'drive_speed_scale': env.drive_speed_scale,
                'capture_required': camera is not None, 'sample_dt': sample_dt,
                'action_semantics': '480Hz applied position and derived velocity targets',
                'read_phase': 'before command and next physics step; last row is terminal state',
                'mode': 'replay' if source else 'expert', 'executor': 'bounded_position_v1',
                'replay_dt': replay_dt if source else None,
                'post_collect_hold_s': post_collect_hold}
    metadata['source_sha256'] = SOURCE_HASHES
    metadata['controller_config'] = env.demo.config
    metadata['world_from_base'] = np.array(UsdGeom.Xformable(
        env.stage.GetPrimAtPath('/World/Autopicker')).ComputeLocalToWorldTransform(0)).tolist()
    metadata['transform_convention'] = 'USD row vectors, meters'
    metadata['camera_clock_mapping'] = 'lockstep_render_barrier' if camera is not None else None
    if metadata_extra:
        metadata.update(metadata_extra)
    episode = Episode(metadata)
    env.episode = episode
    planner = None if source is not None or action_provider is not None else ExpertPlanner(env)
    trajectory = None
    if source:
        arrays = source.arrays()
        trajectory = CommandTrajectory(arrays['time_s'], arrays['command']).resample(replay_dt)
        episode.metadata['source_target_path'] = source.metadata['target_path']
        episode.metadata['source_command_sha256'] = hashlib.sha256(arrays['command'].tobytes()).hexdigest()
    sample_steps = round(sample_dt / env.dt)
    if sample_steps < 1 or abs(sample_steps * env.dt - sample_dt) > 1e-8:
        raise ValueError('sample-dt must be a positive multiple of physics_dt')
    started = time.monotonic()
    next_progress = started + 15.
    failure = None
    try:
        while env.time < timeout_s:
            env.update_task()
            if camera is not None and env.steps % sample_steps == 0:
                camera.capture()
            if env.success:
                if post_collect_hold:
                    record_post_collect(env, post_collect_hold, camera, sample_steps)
                break
            if action_provider is not None:
                requested = action_provider.command(env)
            elif trajectory is None:
                requested = planner.command()
            else:
                requested = trajectory.at(min(env.time, trajectory.times[-1]))
                if env.time > trajectory.times[-1] + 3.:
                    raise RuntimeError('Replay exhausted without collecting the target')
            env.step(requested)
            if time.monotonic() >= next_progress:
                distance = float(np.linalg.norm(env.tip_pose()[:3, 3] - env.entry['world']))
                print(f'EPISODE_PROGRESS mode={episode.metadata["mode"]} t={env.time:.2f}s '
                      f'phase={env.phase} planner={planner.phase if planner else episode.metadata["mode"]} '
                      f'target_distance={distance:.5f}m frames={len(episode.frames)}', flush=True)
                next_progress = time.monotonic() + 15.
        if not env.success:
            raise RuntimeError('Episode timed out before collect')
    except Exception as error:
        failure = str(error)
        print('EPISODE_FAILURE ' + failure, flush=True)
        episode.event('failure', env.time, reason=failure)
    return finish_episode(env, directory, failure, started, action_provider)


def _policy_provider(provider):
    """Unwrap optional hybrid controllers to find the learned policy provider."""
    current = provider
    for _ in range(4):
        if current is None or hasattr(current, 'telemetry_rows'):
            return current
        current = getattr(current, 'policy', None)
    return None


def _write_trajectory_csv(directory, rows):
    if not rows:
        return None
    path = Path(directory) / 'trajectory.csv'
    fieldnames = list(rows[0])
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)
    return path


def finish_episode(env, directory, failure, started, action_provider=None):
    episode = env.episode
    policy_provider = _policy_provider(action_provider)
    trajectory_rows = list(getattr(policy_provider, 'telemetry_rows', []))
    if action_provider is not None:
        episode.metadata['inference_requests'] = action_provider.requests
        if hasattr(action_provider, 'inference_audit'):
            episode.metadata.update(action_provider.inference_audit())
    if policy_provider is not None:
        episode.metadata['trajectory_diagnostics'] = {
            'file': 'trajectory.csv', 'sample_period_s': 0.1,
            'sample_count': len(trajectory_rows),
            'columns': list(trajectory_rows[0]) if trajectory_rows else [],
        }
    try:
        if not episode.states or env.time > episode.states[-1][0]:
            env.record_state(env.executor.position, env.executor.velocity)
    except Exception as error:
        failure = f'{failure}; terminal state: {error}' if failure else f'Terminal state: {error}'
        episode.event('failure', env.time, reason=failure)
    episode.metadata.update(wall_time_s=time.monotonic() - started,
                            limited_steps=env.executor.limited_steps,
                            max_inactive_joint_drift=env.max_inactive_drift)
    try:
        episode.finish(env.success and failure is None, failure or 'collected')
        episode.save(directory)
    except ValueError as error:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'failure.json').write_text(json.dumps({
            'metadata': episode.metadata, 'events': episode.events,
            'reason': failure or str(error), 'audit_error': str(error)}, indent=2), encoding='utf-8')
        raise RuntimeError(f'Episode failed data audit: {failure or error}') from error
    trajectory_path = _write_trajectory_csv(directory, trajectory_rows)
    result = {'path': str(directory), **episode.audit(), 'wall_time_s': time.monotonic() - started,
              'max_inactive_joint_drift': env.max_inactive_drift,
              'trajectory_csv': str(trajectory_path) if trajectory_path else None,
              'trajectory_samples': len(trajectory_rows)}
    print('EPISODE_RESULT ' + json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('collect', 'replay', 'audit'))
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets/autopicker/pilot')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--scene', type=Path, default=ROOT / 'outputs/autopicker_apple_orchard.usd')
    parser.add_argument('--arm', default='R_down', choices=('L_down', 'L_up', 'R_down', 'R_up'))
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--targets', type=int, default=5,
                        help='Maximum distinct reachable apples preselected per tree/approach-angle combination')
    parser.add_argument('--trees', default='Tree_01,Tree_02,Tree_03',
                        help='Comma-separated orchard trees to cover (Tree_01,Tree_02,Tree_03)')
    parser.add_argument('--approach-angles', default='30,45,60',
                        help='Comma-separated approach angles in degrees')
    parser.add_argument('--approach-distance', type=float, default=0.12)
    parser.add_argument('--initial-variation', type=float, default=0.5,
                        help='Bounded random initial-joint variation in [0,1]; 0 disables it')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--speed-scale', type=float, default=1.)
    parser.add_argument('--sample-dt', type=float, default=.1)
    parser.add_argument('--replay-dt', type=float, default=.1)
    parser.add_argument('--timeout', type=float, default=90.)
    parser.add_argument('--post-collect-hold', type=float, default=.2,
                        help='Collect only: record actual held-command physics after collect')
    parser.add_argument('--no-rgbd', action='store_true')
    parser.add_argument('--headless', action='store_true')
    options = parser.parse_args()
    if options.mode in ('replay', 'audit') and options.input is None:
        parser.error('--input is required for replay/audit')
    if options.episodes < 1 or options.targets < 1:
        parser.error('Require episodes and targets >= 1')
    try:
        tree_names = [name.strip() for name in options.trees.split(',') if name.strip()]
        approach_angles = [float(value.strip()) for value in options.approach_angles.split(',') if value.strip()]
    except ValueError as error:
        parser.error(f'Invalid trees or approach angles: {error}')
    if (not tree_names or len(set(tree_names)) != len(tree_names)
            or any(name not in ('Tree_01', 'Tree_02', 'Tree_03') for name in tree_names)):
        parser.error('--trees must contain unique Tree_01, Tree_02 and/or Tree_03 names')
    if (not approach_angles or any(not np.isfinite(value) or not 0 < value < 90
                                   for value in approach_angles)):
        parser.error('--approach-angles must contain values strictly between 0 and 90')
    if (not 0.01 <= options.approach_distance <= 0.3
            or not 0 <= options.initial_variation <= 1 or not np.isfinite(options.initial_variation)):
        parser.error('Invalid approach distance or initial variation')
    if not 0 < options.speed_scale <= 45 or any(not np.isfinite(value) or value <= 0 for value in
                                               (options.sample_dt, options.replay_dt, options.timeout)):
        parser.error('Require finite positive time intervals and speed-scale in (0, 45]')
    if not np.isfinite(options.post_collect_hold) or options.post_collect_hold < 0:
        parser.error('Require finite nonnegative post-collect-hold')
    if options.mode == 'audit':
        paths = sorted(options.input.glob('episode_*/episode.json'))
        if (options.input / 'episode.json').exists():
            paths = [options.input / 'episode.json']
        if not paths:
            parser.error('No episodes found')
        reports = [{'path': str(path.parent), **Episode.load(path.parent).audit()} for path in paths]
        print(json.dumps(reports, indent=2))
        print('AUTOPICKER_DATA_AUDIT_PASS', flush=True)
        return
    options.output.mkdir(parents=True, exist_ok=False)
    sys.argv = sys.argv[:1]
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': options.headless, 'width': 640, 'height': 480})
    environment = camera = None
    results = []
    try:
        from autopicker_dp.environment import PickingEnvironment
        from autopicker_dp.capture import SynchronizedCamera
        sources = []
        if options.mode == 'replay':
            paths = sorted(options.input.glob('episode_*/episode.json'))
            if (options.input / 'episode.json').exists():
                paths = [options.input / 'episode.json']
            sources = [Episode.load(path.parent) for path in paths]
            if not sources or any(not source.outcome['success'] for source in sources):
                raise ValueError('Replay needs successful, audited source episodes')
            options.arm = sources[0].metadata['arm']
            options.speed_scale = sources[0].metadata['speed_scale']
            if any(source.metadata['arm'] != options.arm or source.metadata['speed_scale'] != options.speed_scale for source in sources):
                raise ValueError('Replay source arm/config must be uniform')
            scene_hash = hashlib.sha256(options.scene.read_bytes()).hexdigest()
            if any(source.metadata['scene_sha256'] != scene_hash for source in sources):
                raise ValueError('Replay scene differs from recording')
        environment = PickingEnvironment(options.scene, options.arm, options.speed_scale)
        if not options.no_rgbd:
            camera = SynchronizedCamera(environment)
        count = len(sources) if sources else options.episodes
        coverage_plans = []
        if not sources:
            for tree_name in tree_names:
                base_offset = environment.base_x_offset_for_tree(tree_name)
                for angle in approach_angles:
                    paths = environment.targets(options.targets, base_offset, tree_name, angle,
                                                 options.approach_distance, allow_fewer=True)
                    if paths:
                        coverage_plans.append({'tree': tree_name, 'base_x_offset_m': base_offset,
                                               'approach_angle_degrees': angle, 'targets': paths})
                        print(f'COVERAGE_PLAN tree={tree_name} base_x_offset_m={base_offset:.3f} '
                              f'angle={angle:g} reachable_targets={len(paths)}', flush=True)
                    else:
                        print(f'COVERAGE_PLAN_SKIP tree={tree_name} angle={angle:g} '
                              'reason=no_reachable_targets', flush=True)
            if not coverage_plans:
                raise RuntimeError('No reachable tree/approach-angle combinations were found')
        for index in range(count):
            source = sources[index] if sources else None
            if source:
                target_path = source.metadata['target_path']
                base_offset = float(source.metadata.get('base_x_offset_m', 0.0))
                angle = float(source.metadata.get('approach_angle_degrees', 30.0))
                distance = float(source.metadata.get('approach_distance_m', 0.12))
                initial = source.metadata['reset_joint_positions']
                metadata_extra = {'tree_name': source.metadata.get('tree_name', 'legacy_replay'),
                                  'base_x_offset_m': base_offset,
                                  'approach_angle_degrees': angle,
                                  'approach_distance_m': distance,
                                  'initial_variation': 0.0}
            else:
                plan = coverage_plans[index % len(coverage_plans)]
                target_path = plan['targets'][(index // len(coverage_plans)) % len(plan['targets'])]
                base_offset = plan['base_x_offset_m']
                angle = plan['approach_angle_degrees']
                distance = options.approach_distance
                initial = environment.varied_initial_positions(options.seed + index,
                                                                options.initial_variation)
                metadata_extra = {'tree_name': plan['tree'], 'base_x_offset_m': base_offset,
                                  'approach_angle_degrees': angle, 'approach_distance_m': distance,
                                  'initial_variation': options.initial_variation,
                                  'collection_seed': options.seed + index}
            environment.reset(target_path, initial, base_offset, angle, distance)
            result = run_episode(environment, options.output / f'episode_{index:04d}', camera,
                                 options.sample_dt, options.timeout, source, options.replay_dt,
                                 metadata_extra=metadata_extra,
                                 post_collect_hold=options.post_collect_hold if source is None else 0.)
            results.append(result)
            if not result['success']:
                break
        summary = {'mode': options.mode, 'requested': count, 'attempted': len(results),
                   'successes': sum(result['success'] for result in results), 'results': results}
        (options.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        if summary['successes'] != count:
            raise RuntimeError('Episode acceptance failed; inspect summary.json')
        print('AUTOPICKER_' + options.mode.upper() + '_PASS ' + json.dumps(summary), flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        print('AUTOPICKER_RESEARCH_FAILED', flush=True)
        raise
    finally:
        if camera is not None:
            camera.close()
        if environment is not None:
            environment.close()
        app.close()


if __name__ == '__main__':
    main()
