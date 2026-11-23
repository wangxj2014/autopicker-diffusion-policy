"""Pick existing orchard apples with the original four-arm autopicker (Isaac Sim 5.1)."""

import asyncio
from pathlib import Path

import numpy as np


ASSET_PATH = str(Path(globals().get("__file__", ".")).resolve().parent / "autopicker_apple_orchard.usd")
ARM_PREFIXES = ("L_down", "L_up", "R_down", "R_up")
JOINT_SUFFIXES = ("roll", "hor", "ver", "rot")
PHYSICS_DT = 1.0 / 480.0
DRIVE_REFERENCE_SPEED_SCALE = 15.0
ANGULAR_DRIVE_STIFFNESS = 500.0
LINEAR_DRIVE_STIFFNESS = 2000.0
DRIVE_DAMPING = 300.0
RECEIVER_ROOT = "/World/AppleReceivers"
RECEIVER_SIZE = 0.15
RECEIVER_EXT_OFFSET = 0.12
URDF_LIMITS = {
    "roll": (-1.2217, 0.261799),
    "hor": (0.0, 1.2),
    "ver": (-0.54, 0.0),
    "rot": (-1.91, 1.57),
}
COLORS = ((1.0, 0.3, 0.1), (1.0, 0.75, 0.1), (0.15, 0.7, 1.0), (0.65, 0.3, 1.0))
DEFAULTS = {
    "asset_path": ASSET_PATH,
    "robot_path": None,
    "seed": 42,
    "cycles": 0,
    "speed_scale": 45.0,
    "position_tolerance": 0.025,
    "settle_timeout": 12.0,
    "release_offset_down": 0.30,
    "release_offset_inward": 0.40,
    "release_extra_down_right": 0.03,
    "min_target_distance": 0.4,
    "approach_angle_degrees": 30.0,
    "approach_distance_m": 0.12,
    "grasp_error_every": 4,
    "grasp_error_m": 0.045,
    "fruit_contact_radius": 0.060,
    "tool_contact_radius": 0.018,
    "shake_degrees": 1.4,
    "shake_frequency_hz": 3.5,
    "shake_decay": 2.8,
    "detach_delay_s": 0.18,
    "sample_lower": tuple(URDF_LIMITS[suffix][0] for suffix in JOINT_SUFFIXES),
    "sample_upper": tuple(URDF_LIMITS[suffix][1] for suffix in JOINT_SUFFIXES),
}


def quintic_blend(fraction):
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return fraction**3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))


def motion_duration(start, goal, velocity, acceleration, minimum_duration=0.8):
    distance = np.abs(np.asarray(goal) - np.asarray(start))
    return max(minimum_duration, float(np.max(1.875 * distance / velocity)),
               float(np.max(np.sqrt(5.774 * distance / acceleration))))


class WorldTranslateHandle:
    """Read and write a translation in world coordinates under a transformed parent."""
    def __init__(self, translate):
        from pxr import UsdGeom
        self.translate = translate
        parent = translate.GetAttr().GetPrim().GetParent()
        self.parent_world = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(0)
        self.parent_inverse = self.parent_world.GetInverse()

    def Set(self, position):
        from pxr import Gf
        self.translate.Set(self.parent_inverse.Transform(Gf.Vec3d(*position)))

    def Get(self):
        return self.parent_world.Transform(self.translate.Get())


def define_receiver(stage, prefix, color, position=None):
    """Create or reuse a visible receiver that persists independently of the demo markers."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    root = UsdGeom.Xform.Define(stage, RECEIVER_ROOT)
    root.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
    cube = UsdGeom.Cube.Define(stage, f"{RECEIVER_ROOT}/{prefix}")
    cube.GetSizeAttr().Set(RECEIVER_SIZE)
    half = RECEIVER_SIZE / 2.0
    cube.GetExtentAttr().Set([Gf.Vec3f(-half), Gf.Vec3f(half)])
    cube.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
    cube.GetPurposeAttr().Set(UsdGeom.Tokens.default_)
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    material = UsdShade.Material.Define(stage, f"/World/Looks/Receiver_{prefix}")
    shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.1)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    translate = next((op for op in cube.GetOrderedXformOps()
                      if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    if translate is None:
        translate = cube.AddTranslateOp()
    handle = WorldTranslateHandle(translate)
    if position is not None:
        handle.Set(position)
    return handle


def validate_stage(stage):
    from pxr import UsdGeom

    if stage is None:
        raise ValueError("Open a USD or create a new stage before starting the demo.")
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0) or UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError("Open the robot in a Z-up stage with metersPerUnit=1.")


def pose_matrix(position, quaternion):
    quaternion = np.asarray(quaternion, dtype=float)
    quaternion = quaternion / np.linalg.norm(quaternion)
    scalar = quaternion[0]
    vector = quaternion[1:]
    skew = np.array([[0.0, -vector[2], vector[1]],
                     [vector[2], 0.0, -vector[0]],
                     [-vector[1], vector[0], 0.0]])
    result = np.eye(4)
    result[:3, :3] = np.eye(3) + 2.0 * scalar * skew + 2.0 * skew @ skew
    result[:3, 3] = position
    return result


class JointChain:
    def __init__(self, stage, container, prefix):
        from pxr import UsdGeom, UsdPhysics

        self.prefix = prefix
        self.names = [f"{prefix}_{suffix}_joint" for suffix in JOINT_SUFFIXES]
        prims = {prim.GetName(): prim for prim in stage.Traverse()
                 if prim.GetPath().HasPrefix(container.GetPath())}
        self.tip_path = str(prims[f"{prefix}_ext_link"].GetPath())
        self.frames = []
        lower = []
        upper = []
        expected_parent = prims["base_link"].GetPath()
        self.base = np.array(UsdGeom.Xformable(prims["base_link"]).ComputeLocalToWorldTransform(0)).T
        if not np.allclose(self.base[:3, :3].T @ self.base[:3, :3], np.eye(3), atol=1e-5):
            raise ValueError("Robot scaling is unsupported; use a meter-scale USD with scale 1.")
        for name in self.names + [f"{prefix}_ext_joint"]:
            prim = prims[name]
            joint = UsdPhysics.Joint(prim)
            if joint.GetBody0Rel().GetTargets() != [expected_parent]:
                raise ValueError(f"Unexpected chain topology at {prim.GetPath()}")
            expected_parent = joint.GetBody1Rel().GetTargets()[0]
            frames = []
            for position_attr, rotation_attr in (
                (joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()),
                (joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr()),
            ):
                rotation = rotation_attr.Get()
                frames.append(pose_matrix(position_attr.Get(),
                              [rotation.GetReal(), *rotation.GetImaginary()]))
            kind = "fixed"
            axis = np.zeros(3)
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                kind = "angular" if prim.IsA(UsdPhysics.RevoluteJoint) else "linear"
                axis["XYZ".index(prim.GetAttribute("physics:axis").Get())] = 1.0
                scale = np.pi / 180.0 if kind == "angular" else 1.0
                lower.append(prim.GetAttribute("physics:lowerLimit").Get() * scale)
                upper.append(prim.GetAttribute("physics:upperLimit").Get() * scale)
                drive = UsdPhysics.DriveAPI.Get(prim, kind)
                if not drive or (drive.GetStiffnessAttr().Get() or 0.0) <= 0.0:
                    raise ValueError(f"No position drive at {prim.GetPath()}; open autopicker_v4.usd.")
                if (drive.GetDampingAttr().Get() or 0.0) <= 0.0:
                    raise ValueError(f"Zero drive damping at {prim.GetPath()}; use the tuned USD.")
            self.frames.append((frames[0], np.linalg.inv(frames[1]), kind, axis))
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)

    def forward(self, positions):
        transform = self.base.copy()
        for index, (parent, child_inverse, kind, axis) in enumerate(self.frames):
            motion = np.eye(4)
            if kind == "angular":
                angle = positions[index]
                motion = pose_matrix(np.zeros(3), [np.cos(angle / 2), *(axis * np.sin(angle / 2))])
            elif kind == "linear":
                motion[:3, 3] = axis * positions[index]
            transform = transform @ parent @ motion @ child_inverse
        return transform

    def inverse_position(self, target, seed, wrist=None):
        positions = np.clip(np.asarray(seed, dtype=float).copy(), self.lower + 0.01, self.upper - 0.01)
        if wrist is None:
            from scipy.optimize import least_squares

            result = least_squares(lambda candidate: self.forward(candidate)[:3, 3] - target,
                                   positions, bounds=(self.lower + 0.01, self.upper - 0.01),
                                   max_nfev=200, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            if np.linalg.norm(result.fun) < 1e-5:
                return result.x
            raise ValueError(f"{self.prefix}: Cartesian target is unreachable with free wrist")
        positions[3] = wrist
        for _ in range(80):
            current = self.forward(positions)[:3, 3]
            error = np.asarray(target) - current
            if np.linalg.norm(error) < 1e-5:
                return positions
            jacobian = np.zeros((3, 3))
            for index in range(3):
                shifted = positions.copy()
                shifted[index] += 1e-5
                jacobian[:, index] = (self.forward(shifted)[:3, 3] - current) / 1e-5
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-5 * np.eye(3), error)
            positions[:3] += np.clip(delta, -0.08, 0.08)
            positions = np.clip(positions, self.lower + 0.01, self.upper - 0.01)
        raise ValueError(f"{self.prefix}: Cartesian target is unreachable with wrist={wrist:.3f}")

    def approach_direction(self, angle_degrees):
        angle = np.deg2rad(angle_degrees)
        side = 1.0 if self.prefix.startswith("L_") else -1.0
        return self.base[:3, :3] @ np.array([0.0, side * np.cos(angle), np.sin(angle)])

    def inverse_approach(self, target, seed, angle_degrees):
        from scipy.optimize import least_squares

        direction = self.approach_direction(angle_degrees)

        def residual(candidate):
            pose = self.forward(candidate)
            return np.concatenate((pose[:3, 3] - target, -pose[:3, 1] - direction))

        initial = np.clip(seed, self.lower + 0.01, self.upper - 0.01)
        result = least_squares(residual, initial, bounds=(self.lower + 0.01, self.upper - 0.01),
                               max_nfev=100, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        position_error = np.linalg.norm(result.fun[:3])
        direction_error = np.linalg.norm(result.fun[3:])
        if position_error < 1e-5 and direction_error < 2.0 * np.sin(np.deg2rad(0.01) / 2.0):
            return result.x
        raise ValueError(f"{self.prefix}: target is unreachable at {angle_degrees:g} degrees")

    def plan_approach(self, target, seed, angle_degrees, distance):
        goal = self.inverse_approach(target, seed, angle_degrees)
        direction = self.approach_direction(angle_degrees)
        lengths = sorted({distance, *[length for length in (.08, .04, .02, .01) if length < distance]},
                         reverse=True)
        for length in lengths:
            path = [goal]
            try:
                for offset in np.linspace(0.0, length, 17)[1:]:
                    path.append(self.inverse_approach(target - direction * offset, path[-1], angle_degrees))
            except ValueError:
                continue
            return np.asarray(path[::-1])
        raise ValueError(f"{self.prefix}: no reachable upward approach to target")


class _AutopickerMotion:
    def __init__(self, config=None):
        self.config = {**DEFAULTS, **(config or {})}
        self.random = np.random.default_rng(self.config["seed"])
        self.running = False
        self.completed = False
        self.error = None
        self.records = []
        self.arms = []
        self.robot = None
        self.tips = None
        self._physics_subscription = None
        self._timeline_subscription = None
        self._stage_subscription = None
        self._stop_requested = False
        self._timeline_has_played = False
        self.marker_root = "/AutopickerRandomPick"
        self.time = 0.0
        if not 0.0 < self.config["speed_scale"] <= 45.0:
            raise ValueError("speed_scale must be in (0, 45].")
        if not 0.0 < self.config["position_tolerance"] <= 0.05:
            raise ValueError("position_tolerance must be in (0, 0.05] meters.")
        if self.config["settle_timeout"] <= 0.0:
            raise ValueError("settle_timeout must be positive.")
        if self.config["cycles"] < 0:
            raise ValueError("cycles must be nonnegative; 0 means unlimited.")
        if int(self.config["grasp_error_every"]) != self.config["grasp_error_every"] or self.config["grasp_error_every"] < 0:
            raise ValueError("grasp_error_every must be a nonnegative integer; 0 disables injected error.")
        for name in ("grasp_error_m", "fruit_contact_radius", "tool_contact_radius", "shake_degrees",
                     "shake_frequency_hz", "shake_decay", "detach_delay_s"):
            if not np.isfinite(self.config[name]) or self.config[name] <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(self.config["min_target_distance"]) or self.config["min_target_distance"] < 0.0:
            raise ValueError("min_target_distance must be finite and nonnegative, in meters.")
        if not 0.0 < self.config["approach_angle_degrees"] < 90.0:
            raise ValueError("approach_angle_degrees must be between 0 and 90 degrees.")
        if not 0.01 <= self.config["approach_distance_m"] <= 0.3:
            raise ValueError("approach_distance_m must be between 0.01 and 0.3 meters.")
        for name in ("release_offset_down", "release_offset_inward", "release_extra_down_right"):
            if not np.isfinite(self.config[name]) or self.config[name] < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative, in meters.")

    def prepare(self, stage):
        from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
        from isaacsim.core.utils.stage import add_reference_to_stage

        self.stage = stage
        validate_stage(stage)
        candidates = []
        required_names = {f"{prefix}_{suffix}_joint" for prefix in ARM_PREFIXES for suffix in JOINT_SUFFIXES}
        for prim in stage.Traverse():
            if prim.GetName() == "base_link" and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                container = prim.GetParent()
                names = {item.GetName() for item in stage.Traverse()
                         if item.GetPath().HasPrefix(container.GetPath())}
                if required_names <= names:
                    candidates.append(container)
        requested_path = self.config["robot_path"]
        if requested_path:
            candidates = [prim for prim in candidates if str(prim.GetPath()).startswith(requested_path.rstrip("/") + "/")
                          or str(prim.GetPath()) == requested_path]
        created = False
        if not candidates and not requested_path:
            asset = Path(self.config["asset_path"])
            if not asset.is_file():
                raise FileNotFoundError(f"Set config['asset_path'] to your USD: {asset}")
            if stage.GetPrimAtPath("/World/Autopicker"):
                raise ValueError("/World/Autopicker already exists but is not the expected robot.")
            add_reference_to_stage(str(asset), "/World/Autopicker")
            candidates = [stage.GetPrimAtPath("/World/Autopicker/root_joint")]
            created = True
        if len(candidates) != 1:
            raise ValueError("Expected one four-arm robot; set config['robot_path'] to its parent prim.")
        container = candidates[0]
        limits_by_name = {f"{prefix}_{suffix}_joint": URDF_LIMITS[suffix]
                         for prefix in ARM_PREFIXES for suffix in JOINT_SUFFIXES}
        drive_speed = max(1.0, self.config["speed_scale"] / DRIVE_REFERENCE_SPEED_SCALE)
        for prim in stage.Traverse():
            if prim.GetPath().HasPrefix(container.GetPath()) and prim.GetName() in limits_by_name:
                lower, upper = limits_by_name[prim.GetName()]
                if prim.IsA(UsdPhysics.RevoluteJoint):
                    lower, upper = np.degrees([lower, upper])
                prim.GetAttribute("physics:lowerLimit").Set(float(lower))
                prim.GetAttribute("physics:upperLimit").Set(float(upper))
                linear = prim.IsA(UsdPhysics.PrismaticJoint)
                drive = UsdPhysics.DriveAPI.Get(prim, "linear" if linear else "angular")
                stiffness = LINEAR_DRIVE_STIFFNESS if linear else ANGULAR_DRIVE_STIFFNESS
                # Compress drive response along with the trajectory: K scales by s², D by s.
                # Set absolute gains so repeated Play/Stop cannot compound the speedup.
                drive.GetStiffnessAttr().Set(stiffness * drive_speed**2)
                drive.GetDampingAttr().Set(DRIVE_DAMPING * drive_speed)
        self.chains = [JointChain(stage, container, prefix) for prefix in ARM_PREFIXES]
        self.robot_path = str(container.GetPath())
        existing = stage.GetPrimAtPath(self.marker_root)
        if existing:
            if existing.GetCustomDataByKey("autopickerRandomPick") is not True:
                raise ValueError(f"{self.marker_root} is occupied by another object.")
            stage.RemovePrim(self.marker_root)
        root = UsdGeom.Xform.Define(stage, self.marker_root)
        root.GetPrim().SetCustomDataByKey("autopickerRandomPick", True)
        self.markers = []
        self.drop_markers = []
        for prefix, color in zip(ARM_PREFIXES, COLORS):
            sphere = UsdGeom.Sphere.Define(stage, f"{self.marker_root}/{prefix}/fruit")
            sphere.GetRadiusAttr().Set(0.035)
            sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
            self.markers.append(sphere.AddTranslateOp())
            self.drop_markers.append(define_receiver(stage, prefix, color))
        if created:
            light = UsdLux.DomeLight.Define(stage, f"{self.marker_root}/Light")
            light.GetIntensityAttr().Set(700.0)
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=np.array([5.5, -5.0, 4.0]), target=np.array([0.8, 0.6, 0.9]))

    def initialize(self):
        from pxr import Gf
        from isaacsim.core.prims import Articulation
        from isaacsim.core.simulation_manager import SimulationManager

        self.robot = Articulation(prim_paths_expr=self.robot_path, name="autopicker_random_pick",
                                  reset_xform_properties=False)
        self.robot.initialize()
        if not self.robot.is_physics_handle_valid():
            raise RuntimeError("Physics handle is invalid; initialize simulation before the demo.")
        if self.robot.count != 1 or self.robot.num_dof != 16:
            raise RuntimeError("Expected one articulation with 16 DOFs.")
        self.command = self.robot.get_joint_positions().copy()
        self.robot.set_joint_velocity_targets(np.zeros_like(self.command))
        self.robot.set_joint_efforts(np.zeros_like(self.command))
        self.robot.set_joint_position_targets(self.command)
        self.tips = SimulationManager.get_physics_sim_view().create_rigid_body_view(
            [chain.tip_path for chain in self.chains])
        self.tip_indices = [self.tips.prim_paths.index(chain.tip_path) for chain in self.chains]
        tip_positions, _ = self._tip_poses()
        speed_scale = self.config["speed_scale"]
        self.velocity = np.array([0.18, 0.16, 0.14, 0.30]) * speed_scale
        self.acceleration = np.array([0.35, 0.30, 0.30, 0.60]) * speed_scale**2
        self.initial_fk_error = []
        for index, chain in enumerate(self.chains):
            indices = np.array([self.robot.get_dof_index(name) for name in chain.names], dtype=np.int32)
            positions = self.command[0, indices]
            error = float(np.linalg.norm(chain.forward(positions)[:3, 3] - tip_positions[self.tip_indices[index]]))
            self.initial_fk_error.append(error)
            if error > 0.003:
                raise RuntimeError(f"{chain.prefix}: USD/physics FK mismatch: {error:.4f} m")
            home = np.clip(np.array([-0.12, 0.12, -0.12, 0.0]), chain.lower + 0.02, chain.upper - 0.02)
            home_position = chain.forward(home)[:3, 3]
            hor_probe = home.copy()
            hor_probe[1] += 0.1
            hor_world_direction = (chain.forward(hor_probe)[:3, 3] - home_position) / 0.1
            hor_world_direction[2] = 0.0
            inward_sign = -1.0 if chain.prefix.startswith("L_") else 1.0
            down_offset = self.config["release_offset_down"]
            if chain.prefix.startswith("R_"):
                down_offset += self.config["release_extra_down_right"]
            release_offset = np.array([0.0, inward_sign * self.config["release_offset_inward"],
                                       -down_offset])
            release_position = home_position + chain.base[:3, :3] @ release_offset
            release_goal = chain.inverse_position(release_position, home)
            low = np.maximum(self.config["sample_lower"], chain.lower + 0.03)
            high = np.minimum(self.config["sample_upper"], chain.upper - 0.03)
            if low.shape != (4,) or high.shape != (4,) or np.any(low >= high):
                raise ValueError(f"{chain.prefix}: invalid sampling limits")
            arm = {"chain": chain, "indices": indices, "home": home, "low": low, "high": high,
                   "release_goal": release_goal, "release_position": release_position,
                   "hor_world_direction": hor_world_direction,
                   "count": 0, "attached": False, "offset": np.zeros(3), "target": None,
                   "index": index}
            self.arms.append(arm)
            self._update_receiver(arm, positions)
            self.markers[index].Set(Gf.Vec3d(*home_position))
            print(f"{chain.prefix} release XYZ(m)={np.round(release_position, 3).tolist()}", flush=True)
            self._move(arm, "home", home)
        self.running = True
        print("AUTOPICKER_READY: four arms; visual attachment; use autopicker_demo.stop() to stop.", flush=True)

    def _tip_poses(self):
        transforms = self.tips.get_transforms()
        return transforms[:, :3].copy(), transforms[:, [6, 3, 4, 5]].copy()

    def _update_receiver(self, arm, positions):
        from pxr import Gf

        # Follow the robot's horizontal slide axis, including its current world heading.
        # The cross-slide offset, height and receiver orientation remain fixed.
        position = arm["release_position"] - [0.0, 0.0, RECEIVER_EXT_OFFSET]
        position += arm["hor_world_direction"] * (positions[1] - arm["release_goal"][1])
        self.drop_markers[arm["index"]].Set(Gf.Vec3d(*position))

    def _move(self, arm, phase, goal, start=None):
        if start is None:
            start = self.command[0, arm["indices"]].copy()
        arm.update(phase=phase, start=np.asarray(start).copy(), goal=np.asarray(goal).copy(),
                   elapsed=0.0,
                   duration=motion_duration(start, goal, self.velocity, self.acceleration,
                                            minimum_duration=0.8 / self.config["speed_scale"]))

    def _release_fruit(self, arm):
        """Allow the orchard controller to collect the fruit before selecting another one."""

    def _try_capture(self, arm, tip_position):
        return True

    def _effects_active(self):
        return False

    def _new_target(self, arm, positions):
        from pxr import Gf

        chain = arm["chain"]
        for _ in range(100):
            sampled = self.random.uniform(arm["low"], arm["high"])
            target = chain.forward(sampled)[:3, 3]
            if arm["target"] is not None and np.linalg.norm(target - arm["target"]) < self.config["min_target_distance"]:
                continue
            if any(other["target"] is not None and other is not arm
                   and np.linalg.norm(target - other["target"]) < 0.25 for other in self.arms):
                continue
            try:
                goal = chain.inverse_position(target, positions, sampled[3])
            except ValueError:
                continue
            arm["target"] = target
            self.markers[arm["index"]].Set(Gf.Vec3d(*target))
            self._move(arm, "reach", goal)
            print(f"{chain.prefix} target XYZ(m)={np.round(target, 3).tolist()}", flush=True)
            return
        raise RuntimeError(f"{chain.prefix}: could not sample a reachable, separated target; "
                           "reduce min_target_distance or widen the sampling limits.")

    def step(self, time_step):
        from pxr import Gf

        if not self.running:
            return
        if not self.robot.is_physics_handle_valid():
            self.stop(pause=False)
            return
        self.time += time_step
        positions = self.robot.get_joint_positions()[0]
        velocities = self.robot.get_joint_velocities()[0]
        tip_positions, tip_orientations = self._tip_poses()
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise RuntimeError("Non-finite joint state; simulation paused.")
        velocity_command = np.zeros_like(self.command)
        for arm in self.arms:
            index = arm["index"]
            chain = arm["chain"]
            selected = positions[arm["indices"]]
            if np.any(selected < chain.lower - 0.01) or np.any(selected > chain.upper + 0.01):
                raise RuntimeError(f"{chain.prefix}: joint limit violation")
            self._update_receiver(arm, selected)
            tip_index = self.tip_indices[index]
            tip_position = tip_positions[tip_index]
            tip_pose = pose_matrix(tip_position, tip_orientations[tip_index])
            if arm["attached"]:
                fruit = tip_position + tip_pose[:3, :3] @ arm["offset"]
                self.markers[index].Set(Gf.Vec3d(*fruit))
            arm["elapsed"] += time_step
            phase = arm["phase"]
            if phase == "done":
                continue
            blend = quintic_blend(arm["elapsed"] / arm["duration"])
            fraction = float(np.clip(arm["elapsed"] / arm["duration"], 0.0, 1.0))
            blend_rate = 30.0 * fraction**2 * (1.0 - fraction)**2 / arm["duration"]
            if phase == "reach" and "approach_path" in arm:
                path = arm["approach_path"]
                progress = blend * (len(path) - 1)
                segment = min(int(progress), len(path) - 2)
                self.command[0, arm["indices"]] = path[segment] + (progress - segment) * (path[segment + 1] - path[segment])
                velocity_command[0, arm["indices"]] = (path[segment + 1] - path[segment]) * (len(path) - 1) * blend_rate
            else:
                self.command[0, arm["indices"]] = arm["start"] + blend * (arm["goal"] - arm["start"])
                if phase == "pregrasp":
                    velocity_command[0, arm["indices"]] = blend_rate * (arm["goal"] - arm["start"])
            if arm["elapsed"] < arm["duration"]:
                continue
            expected_tip = chain.forward(arm["goal"])[:3, 3]
            error = float(np.linalg.norm(tip_position - expected_tip))
            joint_error = np.max(np.abs(selected - arm["goal"]))
            joint_speed = np.max(np.abs(velocities[arm["indices"]]))
            # Visual attachment needs accurate arrival, without a timed or low-speed hold.
            arrived = error <= self.config["position_tolerance"] and joint_error <= 0.03
            if phase in ("pregrasp", "reach") and "approach_path" in arm:
                angular_error = np.rad2deg(np.arccos(np.clip(
                    -tip_pose[:3, 1] @ arm["approach_direction"], -1.0, 1.0)))
                arrived = arrived and angular_error <= 1.0
                if phase == "pregrasp":
                    entry_tolerance = min(0.003, 0.1 * arm["approach_length"])
                    arrived = arrived and error <= entry_tolerance and joint_error <= 0.01
                else:
                    arrived = arrived and error <= 0.003
            if not arrived:
                if arm["elapsed"] > arm["duration"] + self.config["settle_timeout"]:
                    raise RuntimeError(f"{chain.prefix} {phase}: did not reach target (tip error {error:.3f} m, "
                                       f"joint error {joint_error:.3f}, joint speed {joint_speed:.3f}). "
                                       "Disable other controllers/JointCommandArray and check drive gains.")
                continue
            if phase == "home":
                self._new_target(arm, selected)
            elif phase == "pregrasp":
                self._move(arm, "reach", arm["approach_path"][-1])
                path_delta = np.max(np.abs(np.diff(arm["approach_path"], axis=0)), axis=0) * (len(arm["approach_path"]) - 1)
                arm["duration"] = motion_duration(np.zeros(4), path_delta, self.velocity, self.acceleration,
                                                   minimum_duration=0.8 / self.config["speed_scale"])
                approach_start = chain.forward(arm["approach_path"][0])[:3, 3]
                approach_end = chain.forward(arm["approach_path"][-1])[:3, 3]
                approach_speed = 0.01 * self.config["speed_scale"]
                arm["duration"] = max(arm["duration"], 1.875 * np.linalg.norm(approach_end - approach_start) / approach_speed)
            elif phase == "reach":
                if not self._try_capture(arm, tip_position):
                    return_goal = arm["release_goal"].copy()
                    return_goal[1] = arm["goal"][1]
                    self._move(arm, "return_empty", return_goal)
                    continue
                capture_error = float(np.linalg.norm(tip_position - arm["target"]))
                if capture_error > self.config["position_tolerance"]:
                    raise RuntimeError(f"{chain.prefix}: capture distance exceeds tolerance.")
                arm["offset"] = tip_pose[:3, :3].T @ (arm["target"] - tip_position)
                arm["attached"] = True
                self.records.append({"arm": chain.prefix, "cycle": arm["count"] + 1,
                                     "error_m": capture_error, "time_s": self.time,
                                     "target_xyz": arm["target"].tolist()})
                print(f"{chain.prefix} CAPTURE error={capture_error * 1000:.1f} mm", flush=True)
                return_goal = arm["release_goal"].copy()
                return_goal[1] = arm["goal"][1]
                self._move(arm, "return", return_goal)
            elif phase == "return":
                self._release_fruit(arm)
                arm.update(attached=False, count=arm["count"] + 1)
                print(f"{chain.prefix} RELEASE cycle={arm['count']}", flush=True)
                if self.config["cycles"] and arm["count"] >= self.config["cycles"]:
                    arm["phase"] = "done"
                else:
                    self._new_target(arm, selected)
            elif phase == "return_empty":
                self._new_target(arm, selected)
        self.robot.set_joint_position_targets(self.command)
        self.robot.set_joint_velocity_targets(velocity_command)
        if all(arm["phase"] == "done" for arm in self.arms) and not self._effects_active():
            self.completed = True
            self.stop(pause=False)
            print("AUTOPICKER_COMPLETE", flush=True)

    def _on_step(self, time_step):
        try:
            self.step(time_step)
        except Exception as error:
            import carb
            self.error = str(error)
            carb.log_error(f"AUTOPICKER_STOPPED: {error}")
            self.stop()

    def _subscribe_lifecycle(self):
        import omni.timeline
        import omni.usd

        def on_timeline(event):
            if event.type == int(omni.timeline.TimelineEventType.PLAY):
                self._timeline_has_played = True
            elif event.type == int(omni.timeline.TimelineEventType.STOP) and (self._timeline_has_played or self.running):
                self.stop(pause=False)

        def on_stage(event):
            if event.type in (int(omni.usd.StageEventType.CLOSING), int(omni.usd.StageEventType.OPENED)):
                self.stop(pause=False)

        self._timeline_subscription = omni.timeline.get_timeline_interface().get_timeline_event_stream().create_subscription_to_pop(on_timeline)
        self._stage_subscription = omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(on_stage)

    def subscribe(self):
        import omni.physx

        if self._timeline_subscription is None:
            self._subscribe_lifecycle()
        self._physics_subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(self._on_step)

    def stop(self, pause=True):
        self._stop_requested = True
        self.running = False
        self._physics_subscription = None
        self._timeline_subscription = None
        self._stage_subscription = None
        if self.robot is not None:
            self.robot.destroy()
            self.robot = None
        self.tips = None
        if pause:
            import omni.timeline
            omni.timeline.get_timeline_interface().pause()


class AppleHandle:
    """Move a complete apple in world space, preserving its tree parent."""
    def __init__(self, stage):
        self.stage = stage
        self.entry = None

    def bind(self, entry):
        from pxr import UsdGeom
        self.entry = entry
        prim = self.stage.GetPrimAtPath(entry["path"])
        self.translate = next(op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
                              if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
        self.parent_inverse = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(0).GetInverse()

    def Set(self, world_position):
        from pxr import Gf, UsdGeom
        if self.entry is not None:
            # The tree may still sway while a successfully picked fruit is carried.
            parent = self.translate.GetAttr().GetPrim().GetParent()
            inverse = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(0).GetInverse()
            self.translate.Set(inverse.Transform(Gf.Vec3d(*world_position)))

    def collect(self, receiver_center, receiver_size, release_height=None):
        from pxr import UsdGeom

        prim = self.stage.GetPrimAtPath(self.entry["path"])
        position = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation())
        offset = position - receiver_center
        half = receiver_size / 2.0
        upper = half if release_height is None else max(half, release_height)
        if np.any(np.abs(offset[:2]) > half) or not -half <= offset[2] <= upper:
            raise RuntimeError(f"Apple is outside its receiver opening: {self.entry['path']}")
        # Deposit vertically from the release point into the smaller receiver, then hide.
        self.Set(receiver_center)
        UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)


class ApplePool:
    """Reserve real, reachable apples once; never synthesize or reposition targets."""
    def __init__(self, stage):
        from pxr import UsdGeom
        self.stage = stage
        self.entries = []
        self.solutions = {}
        for prim in stage.Traverse():
            rest = prim.GetAttribute("picking:restTranslate")
            if rest and rest.HasAuthoredValueOpinion():
                self.entries.append({"path": str(prim.GetPath()),
                                     "side": prim.GetAttribute("picking:side").Get(),
                                     "rest": rest.Get(), "claimed": False})
        if not self.entries:
            raise ValueError("Open autopicker_apple_orchard.usd before running the apple demo.")
        self.reset()

    def reset(self):
        from pxr import Gf, UsdGeom
        for path in {entry["path"].split("/Fruit/")[0] for entry in self.entries}:
            shake = self.stage.GetPrimAtPath(path).GetAttribute("xformOp:rotateXYZ:contactShake")
            if shake:
                shake.Set(Gf.Vec3f(0))
        for entry in self.entries:
            prim = self.stage.GetPrimAtPath(entry["path"])
            translate = next(op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
                             if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
            translate.Set(entry["rest"])
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            entry["world"] = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation())
            entry["claimed"] = False
            entry["state"] = "available"
            prim.GetAttribute("picking:available").Set(True)
        self.solutions.clear()

    def reserve(self, chain, seed, approach_angle=None, approach_distance=0.12):
        side = "Left" if chain.prefix.startswith("L_") else "Right"
        current = chain.forward(seed)[:3, 3]
        # Filter along the robot's own horizontal axis, regardless of scene heading.
        horizontal_axis = chain.base[:3, 0]
        x_samples = [float(chain.forward(q)[:3, 3] @ horizontal_axis) for q in
                     (chain.lower + .01, chain.upper - .01)]
        candidates = [entry for entry in self.entries if not entry["claimed"] and entry["side"] == side
                      and min(x_samples) - .04 <= float(entry["world"] @ horizontal_axis) <= max(x_samples) + .04]
        candidates.sort(key=lambda entry: float(np.linalg.norm(entry["world"] - current)))
        for entry in candidates:
            key = (chain.prefix, entry["path"], approach_angle, approach_distance)
            if key not in self.solutions:
                try:
                    if approach_angle is None:
                        self.solutions[key] = chain.inverse_position(entry["world"], seed)
                    else:
                        self.solutions[key] = chain.plan_approach(
                            entry["world"], seed, approach_angle, approach_distance)[-1]
                except ValueError:
                    self.solutions[key] = None
            goal = self.solutions[key]
            if goal is None:
                continue
            entry["claimed"] = True
            entry["state"] = "reserved"
            self.stage.GetPrimAtPath(entry["path"]).GetAttribute("picking:available").Set(False)
            return entry, goal.copy()
        return None


class OrchardInteraction:
    """Geometric contact and damped tree motion, with PhysX rigid bodies for fallen fruit."""
    FALL_ROOT = "/World/FallenApples"

    def __init__(self, stage, pool, config):
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
        self.stage, self.pool, self.config = stage, pool, config
        self.events, self.shakes, self.pending = [], {}, []
        self.trees = {}
        self.fall_view = None
        root = UsdGeom.Xform.Define(stage, self.FALL_ROOT)
        self.fall_parent = root.ComputeLocalToWorldTransform(0)
        self.fall_inverse = self.fall_parent.GetInverse()
        material = UsdShade.Material.Define(stage, self.FALL_ROOT + "/ContactMaterial")
        surface = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        surface.CreateStaticFrictionAttr(.6)
        surface.CreateDynamicFrictionAttr(.5)
        surface.CreateRestitutionAttr(.15)
        layer = stage.GetEditTarget().GetLayer()
        # Flatten once so CopySpec also works with referenced or session-layer scenes.
        flattened = stage.Flatten()
        for index, entry in enumerate(pool.entries):
            geometry_path = entry["path"].split("/Fruit/")[0]
            if geometry_path not in self.trees:
                geometry = stage.GetPrimAtPath(geometry_path)
                geometry.SetTypeName("Xform")
                xform = UsdGeom.Xformable(geometry)
                rotate = next((op for op in xform.GetOrderedXformOps()
                               if op.GetOpName() == "xformOp:rotateXYZ:contactShake"), None)
                if rotate is None:
                    rotate = xform.AddRotateXYZOp(opSuffix="contactShake")
                rotate.Set(Gf.Vec3f(0))
                self.trees[geometry_path] = {"op": rotate, "entries": [], "wood": None}
            entry["tree"] = geometry_path
            self.trees[geometry_path]["entries"].append(entry)
            fall_path = f"{self.FALL_ROOT}/Fruit_{index:03d}"
            entry["fall_path"] = fall_path
            prim = stage.GetPrimAtPath(fall_path)
            if not prim:
                Sdf.CopySpec(flattened, entry["path"], layer, fall_path)
                prim = stage.GetPrimAtPath(fall_path)
                # Falling copies must not be discovered as new orchard targets on restart.
                for attr in prim.GetAttributes():
                    if attr.GetName().startswith("picking:"):
                        prim.RemoveProperty(attr.GetName())
                xform = UsdGeom.Xformable(prim)
                xform.ClearXformOpOrder()
                xform.AddTransformOp(opSuffix="fallPose")
            body = UsdPhysics.RigidBodyAPI.Apply(prim)
            body.CreateRigidBodyEnabledAttr(True)
            body.CreateKinematicEnabledAttr(True)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(.18)
            # Smooth sphere proxies otherwise roll indefinitely after fruit-to-fruit
            # impacts. Damping approximates an irregular apple rolling on grass.
            prim.AddAppliedSchema("PhysxRigidBodyAPI")
            prim.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(2.5)
            prim.CreateAttribute("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float).Set(.25)
            collider = UsdGeom.Sphere.Define(stage, fall_path + "/ContactShape")
            skin = UsdGeom.Mesh(stage.GetPrimAtPath(entry["path"] + "/Skin"))
            extent = np.array(skin.GetExtentAttr().Get())
            # Conservative smooth proxy matched to this fruit's visible dimensions.
            radius = float(np.max(np.abs(extent)))
            collider.CreateRadiusAttr(radius)
            collider.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(False)
            UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(material, materialPurpose="physics")
            entry["fall_prim"] = prim
            entry["fall_pose"] = next(op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
                                      if op.GetOpName() == "xformOp:transform:fallPose")
            entry["collider"] = collider.GetPrim().GetAttribute("physics:collisionEnabled")
        self.reset()

    def initialize_physics(self):
        from isaacsim.core.simulation_manager import SimulationManager
        self.fall_view = SimulationManager.get_physics_sim_view().create_rigid_body_view(
            self.FALL_ROOT + "/Fruit_*")
        indices = {path: i for i, path in enumerate(self.fall_view.prim_paths)}
        for entry in self.pool.entries:
            entry["fall_index"] = indices[entry["fall_path"]]

    def reset(self):
        from pxr import Gf, UsdGeom
        self.shakes.clear()
        self.pending.clear()
        self.events.clear()
        for tree in self.trees.values():
            tree["op"].Set(Gf.Vec3f(0))
        for entry in self.pool.entries:
            prim = entry["fall_prim"]
            prim.GetAttribute("physics:kinematicEnabled").Set(True)
            entry["collider"].Set(False)
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            prim.GetAttribute("physics:velocity").Set(Gf.Vec3f(0))
            prim.GetAttribute("physics:angularVelocity").Set(Gf.Vec3f(0))
            original = self.stage.GetPrimAtPath(entry["path"])
            world = UsdGeom.Xformable(original).ComputeLocalToWorldTransform(0)
            entry["fall_pose"].Set(world * self.fall_inverse)

    @property
    def active(self):
        return bool(self.shakes or self.pending)

    def position(self, entry):
        from pxr import UsdGeom
        return np.array(UsdGeom.Xformable(self.stage.GetPrimAtPath(entry["path"])).ComputeLocalToWorldTransform(0).ExtractTranslation())

    def contact(self, target, tip, now):
        from pxr import Gf, UsdGeom
        if target["state"] not in ("available", "reserved"):
            return False
        tree_path = target["tree"]
        tree = self.trees[tree_path]
        standing = [e for e in tree["entries"] if e["state"] in ("available", "reserved")]
        distances = [(float(np.linalg.norm(self.position(e) - tip)), e) for e in standing]
        distance, hit = min(distances, key=lambda item: item[0])
        kind = "fruit"
        if distance > self.config["fruit_contact_radius"]:
            # The existing wood consists of tessellated tubes: nearest surface vertices
            # provide a lightweight tool-radius contact proxy without branch rigid bodies.
            if tree["wood"] is None:
                from scipy.spatial import cKDTree
                wood_points = [np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get())
                               for p in self.stage.Traverse()
                               if str(p.GetPath()).startswith(tree_path + "/Wood/") and p.IsA(UsdGeom.Mesh)]
                tree["wood"] = cKDTree(np.concatenate(wood_points))
            inverse = UsdGeom.Xformable(self.stage.GetPrimAtPath(tree_path)).ComputeLocalToWorldTransform(0).GetInverse()
            local_tip = np.array(inverse.Transform(Gf.Vec3d(*np.asarray(tip, dtype=float).tolist())))
            wood_distance = float(tree["wood"].query(local_tip)[0])
            if wood_distance > self.config["tool_contact_radius"]:
                return False
            kind = "branch"
        self.shakes[tree_path] = {"start": now}
        event = {"time_s": now, "kind": kind, "target": target["path"],
                 "tip_xyz": np.asarray(tip).tolist(), "dropped": []}
        # Nearby fruit only: never detach carried or already collected apples.
        affected = [hit] if distance <= .32 else []
        if kind == "branch":
            affected = [e for d, e in sorted(distances, key=lambda item: item[0])[:2] if d <= .32]
        for entry in affected:
            entry.update(state="pending", claimed=True)
            self.stage.GetPrimAtPath(entry["path"]).GetAttribute("picking:available").Set(False)
            self.pending.append((now + self.config["detach_delay_s"], entry, event))
        self.events.append(event)
        return True

    def _detach(self, entry, event, now):
        from pxr import Gf, UsdGeom
        source = self.stage.GetPrimAtPath(entry["path"])
        world = UsdGeom.Xformable(source).ComputeLocalToWorldTransform(0)
        entry["fall_pose"].Set(world * self.fall_inverse)
        prim = entry["fall_prim"]
        UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        UsdGeom.Imageable(source).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        entry["collider"].Set(True)
        prim.GetAttribute("physics:kinematicEnabled").Set(False)
        # Initial pose/velocity are written through PhysX too: USD kinematic changes
        # and dynamics must agree in the same simulation step.
        velocity = np.zeros(6, dtype=np.float32)
        velocity[3:] = [1.5, -.6, .8]
        if self.fall_view is not None:
            transforms = self.fall_view.get_transforms().copy()
            velocities = self.fall_view.get_velocities().copy()
            index = entry["fall_index"]
            rotation = world.ExtractRotationQuat()
            transforms[index] = [*world.ExtractTranslation(), *rotation.GetImaginary(), rotation.GetReal()]
            velocities[index] = velocity
            selected = np.array([index], dtype=np.uint32)
            self.fall_view.set_transforms(transforms, selected)
            self.fall_view.set_velocities(velocities, selected)
        else:
            prim.GetAttribute("physics:angularVelocity").Set(Gf.Vec3f(*np.degrees(velocity[3:]).tolist()))
        entry["state"] = "fallen"
        event["dropped"].append({"path": entry["path"], "fall_path": entry["fall_path"],
                                 "time_s": now, "xyz": list(world.ExtractTranslation())})
        print(f"APPLE_FALL {entry['path']}", flush=True)

    def step(self, now):
        from pxr import Gf
        for path, shake in list(self.shakes.items()):
            age = now - shake["start"]
            if age >= 2.5:
                self.trees[path]["op"].Set(Gf.Vec3f(0))
                del self.shakes[path]
                continue
            amplitude = self.config["shake_degrees"] * np.exp(-self.config["shake_decay"] * age)
            phase = 2 * np.pi * self.config["shake_frequency_hz"] * age
            self.trees[path]["op"].Set(Gf.Vec3f(amplitude * np.sin(phase),
                                              .45 * amplitude * np.sin(phase * 1.13), 0))
        remaining = []
        for deadline, entry, event in self.pending:
            if now >= deadline:
                self._detach(entry, event, now)
            else:
                remaining.append((deadline, entry, event))
        self.pending = remaining


class AutopickerApplePick(_AutopickerMotion):
    def prepare(self, stage):
        # Validate the orchard before modifying the robot or creating any markers.
        self.pool = ApplePool(stage)
        super().prepare(stage)
        for prefix in ARM_PREFIXES:
            stage.RemovePrim(f"{self.marker_root}/{prefix}/fruit")
        self.markers = [AppleHandle(stage) for _ in ARM_PREFIXES]
        self.interaction = OrchardInteraction(stage, self.pool, self.config)
        self.failures = []

    def initialize(self):
        super().initialize()
        self.interaction.initialize_physics()

    def _effects_active(self):
        return self.interaction.active

    def _try_capture(self, arm, tip_position):
        entry = self.markers[arm["index"]].entry
        actual = self.interaction.position(entry)
        error = float(np.linalg.norm(tip_position - actual))
        if entry["state"] == "reserved" and error <= self.config["position_tolerance"]:
            arm["target"] = actual
            entry["state"] = "carried"
            return True
        contact = self.interaction.contact(entry, tip_position, self.time)
        if entry["state"] == "reserved":
            entry["state"] = "missed"
        self.failures.append({"arm": arm["chain"].prefix, "apple_path": entry["path"],
                              "error_m": error, "contact": contact, "time_s": self.time})
        print(f"{arm['chain'].prefix} GRASP_MISS error={error:.3f}m contact={contact}", flush=True)
        return False

    def _release_fruit(self, arm):
        center = np.array(self.drop_markers[arm["index"]].Get())
        self.markers[arm["index"]].collect(
            center, RECEIVER_SIZE,
            release_height=RECEIVER_EXT_OFFSET + 2.0 * self.config["position_tolerance"])
        self.markers[arm["index"]].entry["state"] = "collected"

    def _new_target(self, arm, positions):
        angle = self.config["approach_angle_degrees"]
        distance = self.config["approach_distance_m"]
        selection = self.pool.reserve(arm["chain"], positions, approach_angle=angle, approach_distance=distance)
        if selection is None:
            arm["phase"] = "done"
            print(f"{arm['chain'].prefix}: no reachable unpicked apples remain.", flush=True)
            return
        entry, goal = selection
        path = arm["chain"].plan_approach(entry["world"], goal, angle, distance)
        arm["attempt"] = arm.get("attempt", 0) + 1
        arm["apple_path"] = entry["path"]
        arm["target"] = entry["world"].copy()
        every = self.config["grasp_error_every"]
        arm["injected_error_m"] = 0.0
        if every and arm["attempt"] % every == 0:
            for sign in (1.0, -1.0):
                offset = sign * self.config["grasp_error_m"] * arm["hor_world_direction"]
                try:
                    path = arm["chain"].plan_approach(arm["target"] + offset, goal, angle, distance)
                except ValueError:
                    continue
                arm["injected_error_m"] = float(np.linalg.norm(offset))
                break
        self.markers[arm["index"]].bind(entry)
        arm["approach_path"] = path
        arm["approach_direction"] = arm["chain"].approach_direction(angle)
        arm["approach_length"] = float(np.linalg.norm(
            arm["chain"].forward(path[-1])[:3, 3] - arm["chain"].forward(path[0])[:3, 3]))
        self._move(arm, "pregrasp", path[0], start=positions)
        print(f"{arm['chain'].prefix} APPLE {entry['path']}", flush=True)

    def step(self, time_step):
        if self.running:
            self.interaction.step(self.time + time_step)
        first_record = len(self.records)
        super().step(time_step)
        arms = {arm["chain"].prefix: arm for arm in self.arms}
        for record in self.records[first_record:]:
            arm = arms[record["arm"]]
            record["apple_path"] = arm["apple_path"]
            record["approach_angle_degrees"] = self.config["approach_angle_degrees"]


async def start_demo(config=None):
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.api import World

    global autopicker_demo, _autopicker_start_lock
    previous = globals().get("autopicker_demo")
    if previous is not None:
        previous.stop()
    demo = AutopickerApplePick(config)
    autopicker_demo = demo
    if "_autopicker_start_lock" not in globals():
        _autopicker_start_lock = asyncio.Lock()
    async with _autopicker_start_lock:
        if demo._stop_requested:
            return demo
        stage = omni.usd.get_context().get_stage()

        def stopped():
            same_stage = stage == omni.usd.get_context().get_stage()
            if demo._stop_requested or not same_stage:
                demo.stop(pause=same_stage)
                return True
            return False

        try:
            validate_stage(stage)
            timeline = omni.timeline.get_timeline_interface()
            timeline.stop()
            await omni.kit.app.get_app().next_update_async()
            if stopped():
                return demo
            demo._subscribe_lifecycle()
            world = World.instance()
            if world is None:
                world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
            await world.initialize_simulation_context_async()
            if stopped():
                return demo
            world.set_simulation_dt(physics_dt=PHYSICS_DT)
            demo.prepare(stage)
            await world.reset_async()
            if stopped():
                return demo
            demo.initialize()
            demo.subscribe()
            await world.play_async()
            stopped()
            return demo
        except BaseException:
            demo.stop(pause=stage == omni.usd.get_context().get_stage())
            raise


def launch_demo(config=None):
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    graph = stage.GetPrimAtPath('/World/ApplePickingGraph') if stage else None
    if graph:
        graph.SetActive(False)
    global autopicker_start_task
    autopicker_start_task = asyncio.ensure_future(start_demo(config))
    return autopicker_start_task


if __name__ == "__main__":
    launch_demo()
