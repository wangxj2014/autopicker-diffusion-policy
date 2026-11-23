"""Lockstep RGB-D capture with render-time and camera-pose verification."""
import time

import numpy as np


class SynchronizedCamera:
    def __init__(self, environment, resolution=(160, 120)):
        import omni.replicator.core as rep
        from autopicker_camera_viewer import _make_sync_writer
        from autopicker_depth_fusion import DepthFusion

        self.environment = environment
        rep.orchestrator.set_capture_on_play(False)
        self.path = f'/World/Autopicker/root_joint/{environment.arm_name}_cam_link/Camera'
        self.product = rep.create.render_product(self.path, resolution)
        self.writer = _make_sync_writer([self.product])
        self.fusion = DepthFusion({environment.arm_name: self.path})
        self.previous_payload = None
        self.generation = None
        self.body = None

    def capture(self, timeout_s=20.):
        from pxr import UsdGeom
        import omni.timeline
        import omni.replicator.core as rep
        from isaacsim.core.simulation_manager import SimulationManager
        from autopicker_apple_pick import pose_matrix

        env = self.environment
        if self.generation != env.generation:
            self.body = SimulationManager.get_physics_sim_view().create_rigid_body_view(self.path.rsplit('/', 1)[0])
            if self.body.count != 1:
                raise RuntimeError('Expected one camera rigid body')
            self.generation = env.generation
            self.previous_payload = self.writer.get_data()
        before_time = float(env.world.current_time)
        before_positions = env.demo.robot.get_joint_positions().copy()
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_auto_update(False)
        timeline.set_current_time(before_time)
        deadline = time.monotonic() + timeout_s
        last_error = 'No render batch'
        while time.monotonic() < deadline:
            env.world.render()
            rep.orchestrator.step(delta_time=0., rt_subframes=2, pause_timeline=False)
            if (abs(float(env.world.current_time) - before_time) > 1e-8
                    or not np.allclose(env.demo.robot.get_joint_positions(), before_positions, atol=1e-8, rtol=0)):
                raise RuntimeError('Rendering advanced physics during lockstep capture')
            payload = self.writer.get_data()
            if payload is None or payload is self.previous_payload or not self.fusion.update(payload):
                continue
            self.previous_payload = payload
            frame = self.fusion.get_depth_frame(env.arm_name)
            frame.update(clock_mapping='lockstep_render_barrier', exposure_sim_time_s=before_time,
                         physics_time_after_s=float(env.world.current_time),
                         max_joint_change=float(np.max(np.abs(env.demo.robot.get_joint_positions() - before_positions))))
            transform = self.body.get_transforms()[0]
            body_pose = pose_matrix(transform[:3], transform[[6, 3, 4, 5]]).T
            local_pose = np.array(UsdGeom.Xformable(env.stage.GetPrimAtPath(self.path)).GetLocalTransformation())
            expected = local_pose @ body_pose
            pose_error = float(np.linalg.norm(expected[3, :3] - frame['world_from_camera'][3, :3]))
            rotation_error = float(np.max(np.abs(expected[:3, :3] - frame['world_from_camera'][:3, :3])))
            if rotation_error > 1e-3:
                last_error = f'Rendered camera rotation mismatch: {rotation_error}'
                continue
            try:
                env.episode.append_frame(frame, before_time, env.time, env.steps, pose_error)
                return
            except ValueError as error:
                last_error = str(error)
        raise RuntimeError(f'RGB-D timeout: {last_error}')

    def close(self):
        self.writer.detach()
        self.product.destroy()
