"""Single-arm task with shared execution and measured capture/collection events."""
from pathlib import Path
import sys

import numpy as np

from .control import JointExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'outputs'))


def capture_errors(tip_pose, target, direction):
    distance = float(np.linalg.norm(tip_pose[:3, 3] - target))
    angle = float(np.degrees(np.arccos(np.clip(-tip_pose[:3, 1] @ direction, -1, 1))))
    return distance, angle


class PickingEnvironment:
    def __init__(self, scene, arm_name='R_down', speed_scale=1.):
        import omni.usd
        from isaacsim.core.api import World
        from autopicker_apple_pick import AutopickerApplePick, PHYSICS_DT, JointChain, ARM_PREFIXES

        self.scene = Path(scene).resolve()
        if arm_name not in ('L_down', 'L_up', 'R_down', 'R_up'):
            raise ValueError('Unknown arm name')
        if not omni.usd.get_context().open_stage(str(self.scene)):
            raise RuntimeError('Cannot open scene')
        self.stage = omni.usd.get_context().get_stage()
        if self.stage.GetPrimAtPath('/World/ApplePickingGraph'):
            raise ValueError('Research requires the development scene without embedded control')
        self.world = World(stage_units_in_meters=1., backend='numpy', device='cpu',
                           physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT)
        self.dt = PHYSICS_DT
        self.arm_name = arm_name
        self.drive_speed_scale = 45.
        self.demo = AutopickerApplePick({'asset_path': str(self.scene), 'speed_scale': self.drive_speed_scale,
                                         'grasp_error_every': 0, 'cycles': 1})
        self.demo.prepare(self.stage)
        from pxr import UsdGeom
        root = self.stage.GetPrimAtPath('/World/Autopicker')
        translate_ops = [op for op in UsdGeom.Xformable(root).GetOrderedXformOps()
                         if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        if len(translate_ops) != 1:
            raise RuntimeError('Expected one root translate op on /World/Autopicker')
        self.base_root_translate = np.asarray(translate_ops[0].Get(), dtype=float)
        self.base_x_offset = 0.0
        self.demo.config['speed_scale'] = speed_scale
        self.world.reset()
        self.demo.initialize()
        self.arm = next(arm for arm in self.demo.arms if arm['chain'].prefix == arm_name)
        self.reset_positions = self.demo.robot.get_joint_positions()[0].copy()
        self.reset_positions[self.arm['indices']] = self.arm['home']
        self.orchard_rest_world = {entry['path']: entry['world'].copy() for entry in self.demo.pool.entries}
        self.generation = 0
        self.episode = None

    def base_x_offset_for_tree(self, tree_name):
        if tree_name not in ('Tree_01', 'Tree_02', 'Tree_03'):
            raise ValueError(f'Unknown orchard tree: {tree_name}')
        side = 'Left' if self.arm_name.startswith('L_') else 'Right'
        centers = {}
        for name in ('Tree_01', 'Tree_02', 'Tree_03'):
            values = [entry['world'][0] for entry in self.demo.pool.entries
                      if entry['side'] == side and f'/{name}/' in entry['path']]
            if not values:
                raise ValueError(f'No apples found for {side}/{name}')
            centers[name] = float(np.median(values))
        # Tree_02 is the original calibrated docking position (zero offset).
        return centers[tree_name] - centers['Tree_02']

    def targets(self, count, base_x_offset=0.0, tree_name=None, approach_angle=30.0,
                approach_distance=0.12, allow_fewer=False):
        self._reset_controller(base_x_offset)
        self.arm = next(arm for arm in self.demo.arms if arm['chain'].prefix == self.arm_name)
        selected = []
        original_entries = self.demo.pool.entries
        if tree_name is not None:
            if tree_name not in ('Tree_01', 'Tree_02', 'Tree_03'):
                raise ValueError(f'Unknown orchard tree: {tree_name}')
            side = 'Left' if self.arm_name.startswith('L_') else 'Right'
            self.demo.pool.entries = [entry for entry in original_entries
                                      if entry['side'] == side and f'/{tree_name}/' in entry['path']]
        try:
            for _ in range(count):
                result = self.demo.pool.reserve(self.arm['chain'], self.arm['home'],
                                                approach_angle=approach_angle,
                                                approach_distance=approach_distance)
                if result is None:
                    break
                selected.append(result[0]['path'])
        finally:
            self.demo.pool.entries = original_entries
            self.demo.pool.reset()
        if len(selected) < count and not allow_fewer:
            raise ValueError(f'Only {len(selected)} reachable targets found for {tree_name or "all trees"} '
                             f'at {approach_angle:g} degrees; requested {count}')
        return selected

    def varied_initial_positions(self, seed, variation=0.5):
        if not np.isfinite(variation) or not 0.0 <= variation <= 1.0:
            raise ValueError('initial variation must be in [0, 1]')
        initial = self.reset_positions.copy()
        rng = np.random.default_rng(seed)
        scales = np.array([0.06, 0.025, 0.025, 0.12]) * variation
        candidate = initial[self.arm['indices']] + rng.uniform(-scales, scales)
        candidate = np.clip(candidate, self.arm['chain'].lower + 0.04,
                            self.arm['chain'].upper - 0.04)
        initial[self.arm['indices']] = candidate
        return initial

    def _set_base_x_offset(self, base_x_offset):
        from pxr import Gf, UsdGeom
        if not np.isfinite(base_x_offset):
            raise ValueError('Base X offset must be finite')
        root = self.stage.GetPrimAtPath('/World/Autopicker')
        ops = [op for op in UsdGeom.Xformable(root).GetOrderedXformOps()
               if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        if len(ops) != 1:
            raise RuntimeError('Expected one root translate op on /World/Autopicker')
        original = self.base_root_translate
        ops[0].Set(Gf.Vec3d(float(original[0] + base_x_offset),
                            float(original[1]), float(original[2])))
        self.base_x_offset = float(base_x_offset)

    def _reset_controller(self, base_x_offset=0.0):
        from autopicker_apple_pick import JointChain, ARM_PREFIXES
        for marker in self.demo.markers:
            marker.entry = None
        self.demo.pool.reset()
        self.demo.interaction.reset()
        self.demo.robot.destroy()
        self._set_base_x_offset(base_x_offset)
        self.world.reset()
        container = self.stage.GetPrimAtPath(self.demo.robot_path)
        self.demo.chains = [JointChain(self.stage, container, prefix) for prefix in ARM_PREFIXES]
        self.demo.arms.clear()
        self.demo.records.clear()
        self.demo.initialize()

    def _audit_orchard_reset(self):
        from pxr import UsdGeom

        max_world_error = max_local_error = 0.
        restored = inactive_fallen = 0
        for entry in self.demo.pool.entries:
            prim = self.stage.GetPrimAtPath(entry['path'])
            transform = UsdGeom.Xformable(prim)
            world_position = np.array(transform.ComputeLocalToWorldTransform(0).ExtractTranslation())
            local_position = next(op.Get() for op in transform.GetOrderedXformOps()
                                  if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
            world_error = float(np.linalg.norm(world_position - self.orchard_rest_world[entry['path']]))
            local_error = float(np.linalg.norm(np.array(local_position) - entry['rest']))
            if not np.isfinite([world_error, local_error]).all() or max(world_error, local_error) > 1e-6:
                raise RuntimeError(f'Apple reset position mismatch: {entry["path"]}')
            if (entry['claimed'] or entry['state'] != 'available'
                    or not prim.GetAttribute('picking:available').Get()
                    or UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible):
                raise RuntimeError(f'Apple reset availability mismatch: {entry["path"]}')
            fallen = entry['fall_prim']
            if (UsdGeom.Imageable(fallen).ComputeVisibility() != UsdGeom.Tokens.invisible
                    or entry['collider'].Get() or not fallen.GetAttribute('physics:kinematicEnabled').Get()):
                raise RuntimeError(f'Fallen apple still active after reset: {entry["path"]}')
            max_world_error = max(max_world_error, world_error)
            max_local_error = max(max_local_error, local_error)
            restored += 1
            inactive_fallen += 1
        if self.demo.interaction.active:
            raise RuntimeError('Orchard motion remains active after reset')
        return {'apple_count': len(self.demo.pool.entries), 'visible_available_count': restored,
                'inactive_fallen_count': inactive_fallen, 'max_world_error_m': max_world_error,
                'max_local_error_m': max_local_error}

    def reset(self, target_path, reset_positions=None, base_x_offset=0.0, approach_angle=30.0, approach_distance=0.12):
        from pxr import Gf
        import omni.timeline

        self._reset_controller(base_x_offset)
        self.arm = next(arm for arm in self.demo.arms if arm['chain'].prefix == self.arm_name)
        for arm in self.demo.arms:
            arm['phase'] = 'done'
        initial = np.asarray(self.reset_positions if reset_positions is None else reset_positions, dtype=float)
        if initial.shape != (16,) or not np.isfinite(initial).all():
            raise ValueError('Reset requires all sixteen joint positions')
        self.demo.robot.set_joint_positions(initial[None])
        self.demo.robot.set_joint_velocities(np.zeros((1, 16)))
        self.full_command = initial[None].copy()
        self.demo.robot.set_joint_position_targets(self.full_command)
        self.demo.robot.set_joint_velocity_targets(np.zeros_like(self.full_command))
        for _ in range(240):
            self.world.step(render=False)
        self.reset_orchard_audit = self._audit_orchard_reset()
        self.reset_requested = initial.copy()
        self.entry = next((entry for entry in self.demo.pool.entries if entry['path'] == target_path), None)
        if self.entry is None:
            raise ValueError('Target does not exist in this scene')
        self.entry.update(claimed=True, state='reserved')
        self.stage.GetPrimAtPath(target_path).GetAttribute('picking:available').Set(False)
        self.arm['target'] = self.entry['world'].copy()
        self.arm['apple_path'] = target_path
        self.demo.markers[self.arm['index']].bind(self.entry)
        self.arm['attached'] = False
        self.approach_angle_degrees = float(approach_angle)
        self.approach_distance_m = float(approach_distance)
        if not 0.0 < self.approach_angle_degrees < 90.0 or not 0.01 <= self.approach_distance_m <= 0.3:
            raise ValueError('Invalid approach angle or distance')
        self.direction = self.arm['chain'].approach_direction(self.approach_angle_degrees)
        self.executor = JointExecutor(self.arm['chain'].lower, self.arm['chain'].upper,
                                      self.demo.velocity, self.demo.acceleration, self.dt)
        self.executor.reset(self.positions())
        self.phase = 'reach'
        self.steps = 0
        self.generation += 1
        self.start_time = float(self.world.current_time)
        self.initial_all = self.demo.robot.get_joint_positions()[0].copy()
        self.success = False
        self.episode = None
        self.max_inactive_drift = 0.
        self.demo._update_receiver(self.arm, self.positions())
        self.demo.markers[self.arm['index']].Set(Gf.Vec3d(*self.entry['world']))
        omni.timeline.get_timeline_interface().set_auto_update(False)
        self.world.render()

    @property
    def time(self):
        return self.steps * self.dt

    def positions(self):
        return self.demo.robot.get_joint_positions()[0, self.arm['indices']].copy()

    def tip_pose(self):
        from autopicker_apple_pick import pose_matrix
        positions, orientations = self.demo._tip_poses()
        index = self.demo.tip_indices[self.arm['index']]
        return pose_matrix(positions[index], orientations[index])

    def receiver(self):
        return np.array(self.demo.drop_markers[self.arm['index']].Get())

    def update_task(self):
        from pxr import Gf
        from autopicker_apple_pick import RECEIVER_EXT_OFFSET

        self.demo._update_receiver(self.arm, self.positions())
        pose = self.tip_pose()
        if self.arm['attached']:
            fruit = pose[:3, 3] + pose[:3, :3] @ self.arm['offset']
            self.demo.markers[self.arm['index']].Set(Gf.Vec3d(*fruit))
            release = self.receiver() + [0., 0., RECEIVER_EXT_OFFSET]
            if np.linalg.norm(pose[:3, 3] - release) <= .025:
                self.demo._release_fruit(self.arm)
                self.arm.update(attached=False, count=1)
                self.success = True
                self.phase = 'done'
                self.episode.event('collect', self.time, receiver_world_m=self.receiver().tolist())
        elif not self.success:
            distance, angle = capture_errors(pose, self.entry['world'], self.direction)
            if distance <= .003 and angle <= 1.:
                if not self.demo._try_capture(self.arm, pose[:3, 3]):
                    raise RuntimeError('Capture rejected by orchard state')
                self.arm['offset'] = pose[:3, :3].T @ (self.entry['world'] - pose[:3, 3])
                self.arm['attached'] = True
                self.phase = 'return'
                self.episode.event('capture', self.time, error_m=distance, angle_error_degrees=angle)
        all_positions = self.demo.robot.get_joint_positions()[0]
        inactive = np.ones(16, dtype=bool)
        inactive[self.arm['indices']] = False
        drift = float(np.max(np.abs(all_positions[inactive] - self.initial_all[inactive])))
        self.max_inactive_drift = max(self.max_inactive_drift, drift)
        if drift > .003:
            raise RuntimeError(f'Inactive arms moved: {drift:.6f}')

    def record_state(self, command, velocity):
        self.episode.append_state(self.time, self.positions(),
                                  self.demo.robot.get_joint_velocities()[0, self.arm['indices']],
                                  command, velocity, self.tip_pose()[:3, 3], self.entry['world'],
                                  self.receiver(), self.arm['attached'], self.demo.robot.get_joint_positions()[0])

    def step(self, requested):
        position, velocity = self.executor.advance(requested)
        self.record_state(position, velocity)
        self.full_command[0, self.arm['indices']] = position
        velocity_command = np.zeros_like(self.full_command)
        velocity_command[0, self.arm['indices']] = velocity
        self.demo.robot.set_joint_position_targets(self.full_command)
        self.demo.robot.set_joint_velocity_targets(velocity_command)
        self.world.step(render=False)
        self.steps += 1
        if abs(float(self.world.current_time) - self.start_time - self.time) > 1e-5:
            raise RuntimeError('Physics advanced outside the controlled step loop')

    def close(self):
        self.demo.stop(pause=False)
        self.world.stop()


class ExpertPlanner:
    def __init__(self, environment):
        self.env = environment
        arm = environment.arm
        self.path = arm['chain'].plan_approach(environment.entry['world'], environment.positions(),
                                                 environment.approach_angle_degrees,
                                                 environment.approach_distance_m)
        self.phase = 'pregrasp'
        self.begin(self.path[0])

    def begin(self, goal):
        from autopicker_apple_pick import motion_duration
        self.start = self.env.executor.position.copy()
        self.goal = np.asarray(goal).copy()
        self.started = self.env.time
        self.duration = motion_duration(self.start, self.goal, self.env.demo.velocity,
                                         self.env.demo.acceleration, .8 / self.env.demo.config['speed_scale'])

    def command(self):
        from autopicker_apple_pick import motion_duration, quintic_blend

        if self.env.phase == 'return' and self.phase != 'return':
            goal = self.env.arm['release_goal'].copy()
            goal[1] = self.env.positions()[1]
            self.phase = 'return'
            self.begin(goal)
        elapsed = self.env.time - self.started
        if self.phase == 'pregrasp' and elapsed >= self.duration:
            target = self.env.arm['chain'].forward(self.path[0])
            distance, angle = capture_errors(self.env.tip_pose(), target[:3, 3], self.env.direction)
            if distance <= .003 and angle <= 1.:
                self.phase = 'reach'
                self.started = self.env.time
                delta = np.max(np.abs(np.diff(self.path, axis=0)), axis=0) * (len(self.path) - 1)
                length = np.linalg.norm(self.env.arm['chain'].forward(self.path[-1])[:3, 3] - target[:3, 3])
                self.duration = max(motion_duration(np.zeros(4), delta, self.env.demo.velocity,
                                                     self.env.demo.acceleration, .1),
                                    1.875 * length / (.01 * self.env.demo.config['speed_scale']))
                elapsed = 0.
        blend = quintic_blend(elapsed / self.duration)
        if self.phase == 'reach':
            progress = blend * (len(self.path) - 1)
            segment = min(int(progress), len(self.path) - 2)
            return self.path[segment] + (progress - segment) * (self.path[segment + 1] - self.path[segment])
        return self.start + blend * (self.goal - self.start)
