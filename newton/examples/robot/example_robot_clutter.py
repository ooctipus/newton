# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Four compact Franka contact scenes with low-polygon CAD and cyclic motion.

Run ``python -m newton.examples robot_clutter --scene hardware``. Other scenes
are ``coins``, ``dishes`` and ``cube``. These are contact demos, not trained
pick-and-place policies. The cube follows MuJoCo's contact-coupled cubie model,
not commercial internal tracks. No object poses are prescribed at runtime.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton.examples.robot._clutter_coins import build_coins
from newton.examples.robot._clutter_cube import build_cube
from newton.examples.robot._clutter_dishes import build_dishes
from newton.examples.robot._clutter_hardware import build_hardware
from newton.examples.robot._clutter_mesh import cook_sdf_meshes

SCENES = {
    "hardware": (build_hardware, 256),
    "coins": (build_coins, 520),
    "dishes": (build_dishes, 40),
    "cube": (build_cube, 26),
}


@wp.kernel
def _joint_targets(
    path: wp.array2d[float],
    phase: float,
    phase_rate: float,
    targets: wp.array2d[float],
    velocity_targets: wp.array2d[float],
):
    world, dof = wp.tid()
    segment = int(wp.floor(phase)) % (path.shape[0] - 1)
    u = phase - wp.floor(phase)
    # Quintic interpolation has zero velocity and acceleration at both ends.
    s = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
    targets[world, dof] = wp.lerp(path[segment, dof], path[segment + 1, dof], s)
    velocity_targets[world, dof] = (
        (path[segment + 1, dof] - path[segment, dof]) * 30.0 * u * u * (1.0 - u) * (1.0 - u) * phase_rate
    )


def interaction_path(
    scene: str,
    target: tuple[float, float, float],
    *,
    shake_amplitude: float = 0.06,
    shake_cycles: int = 4,
    shake_axis: str = "x",
) -> np.ndarray:
    """Return xyz, wrist yaw [m, rad] and finger opening [m] at cyclic knots."""
    p = np.asarray(target, dtype=float)
    lift = np.array([0.0, 0.0, 0.16])
    # The dish bay is narrow in Y: sweep along the cutlery, clear of its front rail.
    sweep = np.array([0.035, 0.0, 0.0] if scene == "dishes" else [0.0, 0.035, 0.0])
    high = p + lift
    if scene in ("coins", "hardware", "dishes"):
        # Close around the attached handle, slide the free container, release.
        # All object motion comes from contact; no grasp weld or pose writes.
        if not np.isfinite(shake_amplitude) or shake_amplitude <= 0 or shake_cycles < 1:
            raise ValueError("Shake amplitude and cycle count must be positive")
        if shake_axis not in ("x", "y"):
            raise ValueError("Shake axis must be x or y")
        stroke = np.array([shake_amplitude, 0.0, 0.0] if shake_axis == "x" else [0.0, shake_amplitude, 0.0])
        positions = [high, p, p, *([p + stroke, p - stroke] * shake_cycles), p, p, high]
        yaw = [0.0] * len(positions)
        fingers = [0.030, 0.030, *([0.007] * (2 * shake_cycles + 2)), 0.030, 0.030]
    elif scene == "cube":
        # Endpoint-only joint interpolation sweeps the TCP through an arc.
        # Sample the fixed-position wrist turn in task space before solving IK.
        turn = np.linspace(0.0, np.pi / 2, 9)[1:]
        positions = [high, p, p, *([p] * len(turn)), p, high, high, high]
        yaw = [0.0, 0.0, 0.0, *turn, np.pi / 2, np.pi / 2, 0.0, 0.0]
        fingers = [0.04, 0.04, 0.026, *([0.026] * len(turn)), 0.04, 0.04, 0.04, 0.04]
    else:
        # Descend within the container, sweep across the contents, then retract.
        a, b = p - sweep, p + sweep
        positions = [high, a + lift, a, b, b + lift, high]
        yaw = [0.0] * len(positions)
        fingers = [0.025] * len(positions)
    return np.column_stack((positions, yaw, fingers)).astype(np.float32)


def interaction_times(scene: str, *, shake_cycles: int = 4) -> np.ndarray:
    """Keep approach/release timing fixed while fitting repeated strokes into the grip interval."""
    if scene in ("coins", "hardware", "dishes"):
        if shake_cycles < 1:
            raise ValueError("Shake cycle count must be positive")
        return np.concatenate(([0.0, 0.22, 0.30], np.linspace(0.30, 0.66, 2 * shake_cycles + 2)[1:], [0.76, 1.0]))
    return np.concatenate(([0.0, 1.0, 2.0], np.linspace(2.0, 3.0, 9)[1:], [4, 5, 6, 7])) / 7


def interaction_phase(
    time: float, *, settle: float, period: float, times: np.ndarray, motion: bool
) -> tuple[float, float]:
    """Return waypoint phase and its rate [1/s], with zero rate while holding still."""
    if not motion or time < settle:
        return 0.0, 0.0
    cycle = ((time - settle) / period) % 1.0
    segment = min(int(np.searchsorted(times, cycle, side="right")) - 1, len(times) - 2)
    duration = times[segment + 1] - times[segment]
    return segment + (cycle - times[segment]) / duration, 1.0 / (period * duration)


def build_franka(
    *, container_grasp: bool = False, cube_grasp: bool = False, arm_stiffness_scale: float = 1.0
) -> tuple[newton.ModelBuilder, int]:
    """Load the standard Franka once; keep all nine joints torque-driven."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.gap = 0.0003
    builder.add_urdf(
        newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
        floating=False,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
        force_position_velocity_actuation=container_grasp,
    )
    q = [0.0, 0.024, 0.0, -2.368, 0.0, 2.392, np.pi / 4, 0.04, 0.04]
    builder.joint_q[:9] = q
    builder.joint_target_q[:9] = q
    builder.joint_target_ke[:9] = [2000.0] * 4 + [1000.0] * 3 + [250.0] * 2
    builder.joint_target_kd[:9] = [150.0] * 4 + [80.0] * 3 + [8.0] * 2
    builder.joint_effort_limit[:9] = [87.0] * 4 + [12.0] * 3 + [20.0] * 2
    builder.joint_armature[:9] = [0.1] * 7 + [0.02] * 2
    if not np.isfinite(arm_stiffness_scale) or arm_stiffness_scale <= 0:
        raise ValueError("Arm stiffness scale must be finite and positive")
    # Reduce loaded tracking sag without increasing the actuator effort limits.
    # Scaling damping by sqrt(stiffness) approximately preserves damping ratio.
    builder.joint_target_ke[:7] = [arm_stiffness_scale * value for value in builder.joint_target_ke[:7]]
    builder.joint_target_kd[:7] = [np.sqrt(arm_stiffness_scale) * value for value in builder.joint_target_kd[:7]]
    if cube_grasp:
        # Tight cubie layers require better tracking than a broad bin handle;
        # retain torque limits and leave every object entirely contact-driven.
        builder.joint_target_ke[:7] = [4.0 * value for value in builder.joint_target_ke[:7]]
        builder.joint_target_kd[:7] = [2.0 * value for value in builder.joint_target_kd[:7]]
    if container_grasp or cube_grasp:
        # The loaded container remains table-supported. Rubber pads transmit
        # horizontal force through friction against its 24 mm handle bar.
        builder.joint_target_ke[7:9] = [6000.0 if container_grasp else 1500.0] * 2
        builder.joint_target_kd[7:9] = [20.0] * 2
        builder.joint_effort_limit[7:9] = [30.0] * 2
        for shape, body in enumerate(builder.shape_body):
            if body >= 0 and "finger" in builder.body_label[body]:
                builder.shape_material_mu[shape] = 1.0
    tcp = next(i for i, label in enumerate(builder.body_label) if label.endswith("/fr3_hand_tcp"))
    return builder, tcp


def solve_waypoints(model: newton.Model, tcp: int, waypoints: np.ndarray, *, grasp_tilt: float = 0.0) -> np.ndarray:
    """Solve IK only at setup; the simulation merely interpolates joint targets."""
    position = ik.IKObjectivePosition(
        link_index=tcp,
        link_offset=wp.vec3(),
        target_positions=wp.zeros(1, dtype=wp.vec3, device=model.device),
    )
    rotation = ik.IKObjectiveRotation(
        link_index=tcp,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.zeros(1, dtype=wp.vec4, device=model.device),
    )
    limits = ik.IKObjectiveJointLimit(model.joint_limit_lower, model.joint_limit_upper)
    solver = ik.IKSolver(
        model=model,
        n_problems=1,
        objectives=[position, rotation, limits],
        lambda_initial=0.1,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    q = wp.clone(model.joint_q.reshape((1, -1)))
    result = []
    state = model.state()
    for xyz_yaw_grip in waypoints:
        xyz, yaw, grip = xyz_yaw_grip[:3], float(xyz_yaw_grip[3]), float(xyz_yaw_grip[4])
        quat = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw) * wp.quat_from_axis_angle(
            wp.vec3(1, 0, 0), float(np.pi + grasp_tilt)
        )
        position.target_positions.assign(np.asarray([xyz], dtype=np.float32))
        rotation.target_rotations.assign(np.asarray([list(quat)], dtype=np.float32))
        solver.step(q, q, iterations=64)
        solved = q.numpy()[0]
        solved[7:9] = grip
        q.assign(solved.reshape((1, -1)))
        newton.eval_fk(model, q.flatten(), model.joint_qd, state)
        pose = state.body_q.numpy()[tcp]
        error = float(np.linalg.norm(pose[:3] - xyz))
        angle_error = 2.0 * np.arccos(np.clip(abs(np.dot(pose[3:], list(quat))), 0.0, 1.0))
        if error > 0.008 or angle_error > 0.08:
            raise ValueError(
                f"Franka waypoint {xyz.tolist()} is not reachable (IK error {error:.4f} m, {angle_error:.4f} rad)"
            )
        result.append(solved.copy())
    result[-1] = result[0].copy()
    return np.asarray(result, dtype=np.float32)


class Example:
    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        self.sim_time, self.frame = 0.0, 0
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = args.substeps
        build, default_count = SCENES[args.scene]
        self.object_count = default_count if args.object_count is None else args.object_count
        if min(self.object_count, args.world_count, args.substeps, args.iterations) < 1:
            raise ValueError("Object/world counts, substeps and iterations must be positive")
        self.sim_dt = self.frame_dt / self.sim_substeps
        if (
            not np.isfinite(args.period)
            or args.period <= 0
            or (args.settle_seconds is not None and (not np.isfinite(args.settle_seconds) or args.settle_seconds < 0))
        ):
            raise ValueError("Period must be positive and settling time non-negative")
        if args.screenshot and args.viewer != "gl":
            raise ValueError("--screenshot requires --viewer gl")
        if any(
            value is not None and value < 1
            for value in (args.dense_rows, args.mf_rows, args.contacts_per_world, args.triangle_pairs_per_world)
        ):
            raise ValueError("Contact and constraint capacities must be positive")
        self.body_pair_reduction = args.body_pair_reduction
        if self.body_pair_reduction is None:
            self.body_pair_reduction = args.solver == "fpgs" and args.scene == "hardware"
        if self.body_pair_reduction and args.solver != "fpgs":
            raise ValueError("Body-pair reduction currently supports FPGS only")

        container_grasp = args.scene in ("coins", "hardware", "dishes")
        # Pull the heavy bin along its handle-to-COM line instead of yawing it.
        # Inclining the jaws shares that load between normal and friction forces.
        self.shake_axis = args.shake_axis or ("y" if args.scene == "hardware" else "x")
        self.grasp_tilt = (30.0 if args.scene == "hardware" else 0.0) if args.grasp_tilt is None else args.grasp_tilt
        if not np.isfinite(self.grasp_tilt):
            raise ValueError("Grasp tilt must be finite")
        if not container_grasp and self.grasp_tilt != 0.0:
            raise ValueError("Grasp tilt is supported only for container scenes")
        robot, tcp = build_franka(
            container_grasp=container_grasp,
            cube_grasp=args.scene == "cube",
            arm_stiffness_scale=args.arm_stiffness_scale,
        )
        self.tcp = tcp
        robot_model = robot.finalize()
        self.robot_body_count = robot.body_count
        scene_z = 0.08 if args.scene == "cube" else 0.0
        self.scene_info = build(
            robot,
            center=(0.5, 0.0, scene_z),
            count=self.object_count,
            seed=args.seed,
        )
        self.settle_seconds = (
            self.scene_info.get("settle_time", 2.0) if args.settle_seconds is None else args.settle_seconds
        )
        self.sdf_report = cook_sdf_meshes(self.scene_info.get("sdf_meshes", []), device=robot_model.device)
        cfg = newton.ModelBuilder.ShapeConfig(gap=0.0003, mu=0.08 if container_grasp else 0.7)
        robot.add_shape_box(
            body=-1,
            hx=0.65,
            hy=0.40,
            hz=0.025,
            xform=wp.transform((0.3, 0.0, -0.025), wp.quat_identity()),
            cfg=cfg,
            color=(0.43, 0.34, 0.23),
            label="workbench",
        )
        self.path = wp.array(
            solve_waypoints(
                robot_model,
                tcp,
                interaction_path(
                    args.scene,
                    self.scene_info["target"],
                    shake_amplitude=args.shake_amplitude,
                    shake_cycles=args.shake_cycles,
                    shake_axis=self.shake_axis,
                ),
                grasp_tilt=np.deg2rad(self.grasp_tilt),
            ),
            dtype=float,
            device=robot_model.device,
        )
        self.path_times = interaction_times(args.scene, shake_cycles=args.shake_cycles)
        self.container_trace = []
        robot.joint_q[:9] = self.path.numpy()[0].tolist()
        robot.joint_target_q[:9] = robot.joint_q[:9]
        self.bodies_per_world = robot.body_count
        scene = newton.ModelBuilder()
        scene.replicate(robot, args.world_count)
        self.model = scene.finalize()

        contacts_per_object = {"hardware": 48, "dishes": 48, "coins": 10}.get(args.scene, 16)
        default_contacts = 768 if args.scene == "cube" else contacts_per_object * self.object_count + 128
        self.contact_capacity = args.contacts_per_world or default_contacts
        self.triangle_pair_capacity = args.triangle_pairs_per_world or (131072 if args.scene == "hardware" else 65536)
        # FeatherPGS solves separated rows as speculative contacts, so velocity-adapted gaps give a falling part
        # its row before it reaches the part below; the 0.3 mm gap alone is a fraction of a substep's travel at
        # impact speeds, and parts landing deeper than a thin feature's half thickness wedge.
        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            body_pair_reduction=(
                newton.CollisionPipeline.BodyPairReductionConfig(cell_size=0.04, hysteresis=0.0, verify=args.test)
                if self.body_pair_reduction
                else None
            ),
            speculative_contact_gap_max=0.1 if args.solver == "fpgs" else None,
            rigid_contact_max=self.contact_capacity * args.world_count,
            # FeatherPGS follows the contact rows as they grow; MuJoCo holds the fixed capacity.
            growable_contacts=args.solver == "fpgs",
            max_triangle_pairs=self.triangle_pair_capacity * args.world_count,
        )
        self.contacts = self.pipeline.contacts()
        if args.solver == "fpgs":
            # Immediate MF uses world DOFs, avoiding split's 64-free-body map.
            # Compound hardware benefits from patch reduction before the solve;
            # raw collision storage must still fit all unreduced contacts.
            self.mf_rows = args.mf_rows or (
                max(256, 88 * self.object_count) if self.body_pair_reduction else 3 * self.contact_capacity
            )
            self.solver = newton.solvers.SolverFeatherPGS(
                self.model,
                pgs_iterations=args.iterations,
                dense_max_constraints=args.dense_rows or (2304 if args.scene == "cube" else 256),
                mf_max_constraints=self.mf_rows,
                enable_joint_limits=True,
                diagnostics=newton.solvers.SolverFeatherPGS.Diagnostics(row_watermark=True),
            )
        else:
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_contacts=False,
                solver="newton",
                integrator="implicitfast",
                iterations=args.iterations,
                ls_iterations=40,
                cone="elliptic",
                nconmax=self.contact_capacity,
                njmax=3 * self.contact_capacity + self.model.joint_dof_count // args.world_count,
            )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        wp.copy(self.control.joint_target_q, self.model.joint_q)
        self.targets = self.control.joint_target_q.reshape((args.world_count, -1))
        self.velocity_targets = self.control.joint_target_qd.reshape((args.world_count, -1))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((1.5, 1.2, 0.0))
        self.viewer.set_camera(wp.vec3(1.23, -1.55, 1.10), pitch=-26.3, yaw=121.8)
        self.graph = None
        if self.body_pair_reduction:
            self.pipeline.collide(self.state_0, self.contacts, dt=self.sim_dt)
        self.capture()

    def capture(self):
        # Even substep counts leave the same state buffer at frame boundaries.
        if self.args.cuda_graph and self.model.device.is_cuda and self.sim_substeps % 2 == 0:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts, dt=self.sim_dt)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        phase, phase_rate = interaction_phase(
            self.sim_time,
            settle=self.settle_seconds,
            period=self.args.period,
            times=self.path_times,
            motion=self.args.motion,
        )
        # Velocity feed-forward avoids damping the intended container stroke.
        # Cube control is deliberately unchanged while its mechanics are studied.
        if self.args.scene == "cube":
            phase_rate = 0.0
        wp.launch(
            _joint_targets,
            dim=(self.args.world_count, 9),
            inputs=[self.path, phase, phase_rate, self.targets, self.velocity_targets],
        )
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.frame += 1
        if self.frame % 60 == 0:
            self.check_capacity()
        if self.args.test and "container_body" in self.scene_info:
            q = self.state_0.body_q.numpy()
            self.container_trace.append((self.sim_time, *q[self.scene_info["container_body"], :3], *q[self.tcp, :3]))
        if self.args.test and not self.args.quiet and self.frame % 120 == 0:
            print(f"{self.args.scene}: {self.frame}/{self.args.num_frames} frames ({self.sim_time:.1f} s)", flush=True)

    def check_capacity(self):
        """Fail on omitted constraints instead of silently accepting truncated physics."""
        self.pipeline.narrow_phase.check_buffer_capacity()
        if self.body_pair_reduction:
            stats = self.pipeline.body_pair_reduction_stats()
            if stats["invariant_violations"] or stats["outranked_discards"] or stats["input_overflow_frames"]:
                raise RuntimeError(f"Body-pair contact reduction failed: {stats}")
        if self.args.solver == "fpgs":
            self.solver.check_constraint_capacity()
        else:
            self.solver.check_contact_capacity()
            overflow = self.solver.mjw_data.overflow.numpy()
            if np.any(overflow):
                raise RuntimeError(f"MJWarp overflow/warning flags per world: {overflow.tolist()}")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()
        if self.args.screenshot and self.frame == self.args.num_frames:
            from PIL import Image  # noqa: PLC0415

            self.args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(self.viewer.get_frame().numpy()).save(self.args.screenshot)

    def test_post_step(self):
        """Reject non-finite states and escaped objects during smoke tests."""
        q = self.state_0.body_q.numpy()
        v = self.state_0.body_qd.numpy()
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            invalid = np.flatnonzero(~np.isfinite(q).all(axis=1) | ~np.isfinite(v).all(axis=1))
            raise ValueError(f"Frame {self.frame}: non-finite state of {self.model.body_label[int(invalid[0])]}")
        escaped = np.flatnonzero((np.max(np.abs(q[:, :3]), axis=1) > 3.0) | (q[:, 2] < -0.10))
        if len(escaped):
            body = int(escaped[0])
            raise ValueError(
                f"Frame {self.frame}: {self.model.body_label[body]} escaped the tabletop "
                f"at {q[body, :3].tolist()} with velocity {v[body].tolist()}"
            )
        self.check_capacity()

    def test_final(self):
        """Check state and contact capacities at the end of a rollout."""
        self.test_post_step()

    def report(self) -> dict:
        """Summarize the actual constructed scene, without making speedup claims."""
        self.check_capacity()
        result = {
            "scene": self.args.scene,
            "solver": self.args.solver,
            "device": str(self.model.device),
            "worlds": self.args.world_count,
            "objects_per_world": self.object_count,
            "bodies_per_world": self.bodies_per_world,
            "shapes_per_world": self.model.shape_count // self.args.world_count,
            "frames": self.frame,
            "simulated_seconds": self.sim_time,
            "substeps": self.sim_substeps,
            "iterations": self.args.iterations,
            "settle_seconds": self.settle_seconds,
            "motion_period_seconds": self.args.period,
            "shake_amplitude_m": self.args.shake_amplitude if self.args.scene != "cube" else None,
            "shake_cycles": self.args.shake_cycles if self.args.scene != "cube" else None,
            "shake_axis": self.shake_axis if self.args.scene != "cube" else None,
            "grasp_tilt_degrees": self.grasp_tilt,
            "arm_stiffness_scale": self.args.arm_stiffness_scale,
            "joint_velocity_feedforward": self.args.scene != "cube",
            "contact_capacity_per_world": self.contact_capacity,
            "triangle_pair_capacity_per_world": self.triangle_pair_capacity,
            "collision_buffers": self.pipeline.narrow_phase.buffer_capacity_status(),
            "cad_sdf_meshes": self.sdf_report,
        }
        if self.container_trace:
            trace = np.asarray(self.container_trace)
            moving = trace[trace[:, 0] >= self.settle_seconds]
            if len(moving):
                result["container_x_travel_m"] = float(np.ptp(moving[:, 1]))
                result["container_y_travel_m"] = float(np.ptp(moving[:, 2]))
                result["container_z_range_m"] = [float(moving[:, 3].min()), float(moving[:, 3].max())]
        if self.args.solver == "fpgs":
            result["mf_row_capacity_per_world"] = self.mf_rows
            result["rows"] = self.solver.constraint_row_watermarks()
        result["body_pair_reduction"] = self.body_pair_reduction
        if self.body_pair_reduction:
            result["reduction"] = self.pipeline.body_pair_reduction_stats()
        return result


def create_parser():
    """Extend the normal example CLI with scene and solver choices."""
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=720)
    parser.add_argument("--scene", choices=tuple(SCENES), default="hardware")
    parser.add_argument("--solver", choices=("fpgs", "mjwarp"), default="fpgs")
    parser.add_argument("--world-count", type=int, default=1)
    parser.add_argument("--object-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--settle-seconds", type=float, default=None, help="Default: hardware 3 s, otherwise 2 s.")
    parser.add_argument("--period", type=float, default=10.0)
    parser.add_argument("--shake-amplitude", type=float, default=0.06, help="Container stroke half-span [m].")
    parser.add_argument("--shake-cycles", type=int, default=4, help="Back-and-forth cycles per container grasp.")
    parser.add_argument("--shake-axis", choices=("x", "y"), default=None, help="Default: hardware y, otherwise x.")
    parser.add_argument(
        "--grasp-tilt", type=float, default=None, help="Gripper tilt from downward [degrees]: hardware 30, otherwise 0."
    )
    parser.add_argument(
        "--arm-stiffness-scale",
        type=float,
        default=1.0,
        help="Scale arm stiffness; damping uses sqrt(scale). Effort limits stay fixed.",
    )
    parser.add_argument("--contacts-per-world", type=int, default=None)
    parser.add_argument(
        "--triangle-pairs-per-world",
        type=int,
        default=None,
        help="Mesh candidate storage: hardware 131072, otherwise 65536.",
    )
    parser.add_argument(
        "--dense-rows", type=int, default=None, help="Articulated row storage: cube 2304, otherwise 256."
    )
    parser.add_argument("--mf-rows", type=int, default=None, help="Free-body row storage, independent of raw contacts.")
    parser.add_argument(
        "--body-pair-reduction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reduce compound contact patches; defaults on for FPGS hardware only (not backend-matched).",
    )
    parser.add_argument("--report", type=Path, default=None, help="Optional rollout summary JSON.")
    parser.add_argument("--motion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--screenshot", type=Path, default=None, help="Save the last frame (GL viewer only).")
    return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
    result = example.report()
    print(json.dumps(result, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
