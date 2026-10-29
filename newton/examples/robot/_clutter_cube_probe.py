# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Apply measured virtual wrenches to isolated cubes, without a robot or pose control.

Run ``python -m newton.examples.robot._clutter_cube_probe --device cuda:1``.
Each world receives one load amplitude. These are diagnostic sweeps, not measured
real-cube calibration targets. The model has no screw preload, dry joint friction,
magnetic detents, or axial core compliance; rigid retainers cannot validate those.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.robot._clutter_cube import build_cube
from newton.examples.robot._clutter_mesh import cook_sdf_meshes


@wp.kernel
def _load(
    mode: int,
    bodies_per_world: int,
    body_indices: wp.array[int],
    points: wp.array[wp.vec3],
    amplitudes: wp.array[float],
    scale: wp.array[float],
    radius: float,
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_f: wp.array[wp.spatial_vector],
    measured_load: wp.array2d[float],
    stop_reason: wp.array[int],
    peak_load: wp.array2d[float],
):
    world = wp.tid()
    first = world * bodies_per_world + body_indices[0]
    second = world * bodies_per_world + body_indices[1]
    amount = amplitudes[world] * scale[0]
    if stop_reason[world] != 0:
        amount = 0.0
    if mode == 0:
        body_f[first] = wp.spatial_vector(wp.vec3(), wp.vec3(0.0, 0.0, amount))
        measured_load[world, 0] = 0.0
        measured_load[world, 1] = 0.0
        measured_load[world, 2] = amount
        measured_load[world, 3] = 0.0
    elif mode == 1:
        p = wp.transform_point(body_q[first], points[0])
        q = wp.transform_point(body_q[second], points[1])
        direction = wp.normalize(wp.cross(wp.vec3(0.0, 0.0, 1.0), p - q))
        force = direction * (amount / (2.0 * radius))
        cp = wp.transform_point(body_q[first], body_com[first])
        cq = wp.transform_point(body_q[second], body_com[second])
        body_f[first] = wp.spatial_vector(force, wp.cross(p - cp, force))
        body_f[second] = wp.spatial_vector(-force, wp.cross(q - cq, -force))
        moment = wp.cross(p - q, force)
        for axis in range(3):
            measured_load[world, axis] = moment[axis]
        measured_load[world, 3] = wp.length(force)
    else:
        force = wp.vec3(amount, 0.0, 0.0)
        body_f[first] = wp.spatial_vector(force, wp.vec3())
        center = wp.transform_get_translation(body_q[world * bodies_per_world])
        point = wp.transform_point(body_q[first], body_com[first])
        moment = wp.cross(point - center, force)
        for axis in range(3):
            measured_load[world, axis] = moment[axis]
        measured_load[world, 3] = wp.abs(amount)
    for component in range(4):
        peak_load[world, component] = wp.max(peak_load[world, component], wp.abs(measured_load[world, component]))


@wp.kernel
def _guard_load(
    bodies_per_world: int,
    top_center: int,
    loaded_body: int,
    pitch: float,
    max_speed: float,
    max_error: float,
    max_angle: float,
    clock: wp.array[float],
    time_offset: float,
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    stop_reason: wp.array[int],
    stop_time: wp.array[float],
    measured_load: wp.array2d[float],
    trip_load: wp.array2d[float],
    onset_time: wp.array[float],
    onset_load: wp.array2d[float],
    trip_state: wp.array2d[float],
):
    world = wp.tid()
    if stop_reason[world] != 0:
        return
    offset = world * bodies_per_world
    core, top = body_q[offset], body_q[offset + top_center]
    rotation = wp.transform_get_rotation(top)
    reason = int(0)
    signed_angle = 2.0 * wp.atan2(rotation[2], rotation[3])
    angle = wp.abs(signed_angle)
    max_position_error = float(0.0)
    max_orientation_error = float(0.0)
    max_angular_speed = float(0.0)
    if angle >= 0.5 * wp.pi:
        reason = 1
    for local in range(1, bodies_per_world):
        body = offset + local
        speed = wp.length(wp.spatial_bottom(body_qd[body]))
        max_angular_speed = wp.max(max_angular_speed, speed)
        if speed > max_speed:
            reason = 2
        expected = wp.transform_point(core, body_com[body])
        expected_rotation = wp.transform_get_rotation(core)
        if body_com[body][2] > 0.5 * pitch:
            expected = wp.transform_point(top, body_com[body])
            expected_rotation = rotation
        actual = wp.transform_point(body_q[body], body_com[body])
        position_error = wp.length(actual - expected)
        max_position_error = wp.max(max_position_error, position_error)
        if position_error > max_error:
            reason = 3
        dot = wp.abs(wp.dot(wp.transform_get_rotation(body_q[body]), expected_rotation))
        orientation_error = 2.0 * wp.acos(wp.clamp(dot, 0.0, 1.0))
        max_orientation_error = wp.max(max_orientation_error, orientation_error)
        if orientation_error > max_angle:
            reason = 4
    if onset_time[world] < 0.0 and angle >= (2.0 * wp.pi / 180.0) and reason == 0:
        onset_time[world] = clock[0] + time_offset
        for component in range(4):
            onset_load[world, component] = measured_load[world, component]
    if reason != 0:
        stop_reason[world] = reason
        stop_time[world] = clock[0] + time_offset
        for component in range(4):
            trip_load[world, component] = measured_load[world, component]
        trip_state[world, 0] = signed_angle
        trip_state[world, 1] = body_qd[offset + top_center][5]
        trip_state[world, 2] = max_position_error
        trip_state[world, 3] = max_orientation_error
        trip_state[world, 4] = max_angular_speed
        loaded = offset + loaded_body
        displacement = wp.transform_point(body_q[loaded], body_com[loaded]) - wp.transform_point(core, body_com[loaded])
        for axis in range(3):
            trip_state[world, 5 + axis] = displacement[axis]


def _rotation(quaternion):
    return np.asarray(wp.quat_to_matrix(wp.quat(*quaternion))).reshape(3, 3)


class Example:
    """Measure passive cube response under ramp, hold, unload, and relaxation."""

    def __init__(self, args):
        values = [
            *args.amplitudes,
            args.settle,
            args.ramp,
            args.hold,
            args.relax,
            args.max_speed,
            args.max_error,
            args.max_angle,
            args.gravity,
        ]
        if not args.amplitudes or not np.isfinite(values).all():
            raise ValueError("Provide finite load amplitudes and timing parameters")
        if args.ramp <= 0 or min(args.settle, args.hold, args.relax) < 0:
            raise ValueError("Ramp duration must be positive and other durations nonnegative")
        if min(args.substeps, args.iterations, args.contacts, args.rows) < 1:
            raise ValueError("Substeps, iterations, and capacities must be positive")
        if min(args.max_speed, args.max_error, args.max_angle) <= 0:
            raise ValueError("Diagnostic speed and coherence limits must be positive")
        newton.use_coord_layout_targets = True
        wp.set_device(args.device)
        self.args, self.frame = args, 0
        self.time, self.dt = 0.0, 1.0 / (60 * args.substeps)
        self.samples = []
        cube = newton.ModelBuilder(gravity=(0.0, 0.0, -args.gravity))
        self.info = build_cube(
            cube,
            center=(0.0, 0.0, 0.08),
            armature=args.armature,
            damping=args.damping,
            internal_friction=args.friction,
            clearance=args.clearance,
            collision=args.collision,
        )
        self.cad = cook_sdf_meshes(self.info["sdf_meshes"], device=args.device)
        self.data = self.info["metadata"]
        self.bodies_per_world = cube.body_count
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -args.gravity))
        builder.replicate(cube, len(args.amplitudes))
        self.model = builder.finalize()
        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            rigid_contact_max=args.contacts * len(args.amplitudes),
            max_triangle_pairs=32768 * len(args.amplitudes),
        )
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverFeatherPGS(
            self.model,
            pgs_mode="matrix_free",
            pgs_iterations=args.iterations,
            dense_max_constraints=args.rows,
            mf_max_constraints=256,
            row_watermark=True,
        )
        self.current, self.next = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.current)
        self.mode = ("torque", "couple", "off-axis").index(args.mode)
        lattice, bodies = self.data["lattice"], self.info["object_bodies"]
        sites = ((0, 0, 1), (0, 0, 1))
        if self.mode == 1:
            sites = ((0, 1, 1), (0, -1, 1))
        elif self.mode == 2:
            sites = ((1, 1, 1), (1, 1, 1))
        self.indices = wp.array([bodies[lattice.index(site)] for site in sites], dtype=int)
        self.loaded_piece = lattice.index(sites[0])
        self.loaded_body = bodies[self.loaded_piece]
        self.top_center = bodies[lattice.index((0, 0, 1))]
        radius, pitch = self.data["half_extent"], self.data["pitch"]
        self.points = wp.array([(0.0, radius, pitch), (0.0, -radius, pitch)], dtype=wp.vec3)
        self.amplitudes = wp.array(args.amplitudes, dtype=float)
        self.scale = wp.zeros(1, dtype=float)
        self.measured_load = wp.zeros((len(args.amplitudes), 4), dtype=float)
        self.peak_load = wp.zeros((len(args.amplitudes), 4), dtype=float)
        self.trip_load = wp.zeros((len(args.amplitudes), 4), dtype=float)
        self.trip_state = wp.zeros((len(args.amplitudes), 8), dtype=float)
        self.onset_time = wp.full(len(args.amplitudes), -1.0, dtype=float)
        self.onset_load = wp.zeros((len(args.amplitudes), 4), dtype=float)
        self.clock = wp.zeros(1, dtype=float)
        self.stop_reason = wp.zeros(len(args.amplitudes), dtype=int)
        self.stop_time = wp.full(len(args.amplitudes), -1.0, dtype=float)
        self.com = self.model.body_com.numpy()
        self.shape_body = self.model.shape_body.numpy()
        self.graph = None
        if args.substeps % 2 == 0:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for substep in range(self.args.substeps):
            self.current.clear_forces()
            wp.launch(
                _guard_load,
                dim=len(self.args.amplitudes),
                inputs=[
                    self.bodies_per_world,
                    self.top_center,
                    self.loaded_body,
                    self.data["pitch"],
                    self.args.max_speed,
                    self.args.max_error,
                    float(np.deg2rad(self.args.max_angle)),
                    self.clock,
                    substep * self.dt,
                    self.model.body_com,
                    self.current.body_q,
                    self.current.body_qd,
                    self.stop_reason,
                    self.stop_time,
                    self.measured_load,
                    self.trip_load,
                    self.onset_time,
                    self.onset_load,
                    self.trip_state,
                ],
            )
            wp.launch(
                _load,
                dim=len(self.args.amplitudes),
                inputs=[
                    self.mode,
                    self.bodies_per_world,
                    self.indices,
                    self.points,
                    self.amplitudes,
                    self.scale,
                    self.data["half_extent"],
                    self.model.body_com,
                    self.current.body_q,
                    self.current.body_f,
                    self.measured_load,
                    self.stop_reason,
                    self.peak_load,
                ],
            )
            self.pipeline.collide(self.current, self.contacts)
            self.solver.step(self.current, self.next, self.control, self.contacts, self.dt)
            self.current, self.next = self.next, self.current

    def envelope(self):
        t = self.time - self.args.settle
        ramp, hold = self.args.ramp, self.args.hold
        return float(np.clip(min(t / ramp, (2 * ramp + hold - t) / ramp), 0.0, 1.0))

    def step(self):
        self.load_fraction = self.envelope()
        self.scale.assign([self.load_fraction])
        self.clock.assign([self.time])
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.time += 1.0 / 60
        self.frame += 1
        self.test_post_step()
        if self.frame % 6 == 0 or self.frame == 1:
            self.measure()

    def test_post_step(self):
        """Reject omitted contacts, solver rows, and non-finite physical states."""
        self.pipeline.narrow_phase.check_buffer_capacity()
        self.solver.check_constraint_capacity()
        if not np.isfinite(self.current.body_q.numpy()).all() or not np.isfinite(self.current.body_qd.numpy()).all():
            raise RuntimeError(f"Non-finite cube state at frame {self.frame}")

    def measure(self):
        poses = self.current.body_q.numpy()
        velocities = self.current.body_qd.numpy()
        lattice = np.asarray(self.data["lattice"])
        ideal = lattice * self.data["pitch"]
        top = lattice[:, 2] == 1
        center = np.asarray(self.data["cube_center"])
        loads = self.measured_load.numpy()
        stopped = self.stop_reason.numpy()
        self.solver.update_contacts(self.contacts)
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        body_a = self.shape_body[self.contacts.rigid_contact_shape0.numpy()[:count]]
        body_b = self.shape_body[self.contacts.rigid_contact_shape1.numpy()[:count]]
        forces = self.contacts.rigid_contact_force.numpy()[:count]
        magnitudes = np.linalg.norm(forces, axis=1)
        local_a = self.contacts.rigid_contact_point0.numpy()[:count]
        local_b = self.contacts.rigid_contact_point1.numpy()[:count]
        contact_poses = self.next.body_q.numpy()
        for world, amplitude in enumerate(self.args.amplitudes):
            bodies = np.asarray(self.info["object_bodies"]) + world * self.bodies_per_world
            rotations = np.asarray([_rotation(poses[body, 3:]) for body in bodies])
            positions = np.asarray(
                [poses[body, :3] + rotation @ self.com[body] for body, rotation in zip(bodies, rotations, strict=True)]
            )
            relative = positions - center
            u, _, vh = np.linalg.svd(ideal[top].T @ relative[top])
            fitted = vh.T @ np.diag((1, 1, np.linalg.det(vh.T @ u.T))) @ u.T
            error = relative[top] - ideal[top] @ fitted.T
            orientation_residual = np.arccos(
                np.clip((np.trace(rotations[top] @ fitted.T, axis1=1, axis2=2) - 1) / 2, -1, 1)
            )
            top_center = world * self.bodies_per_world + self.top_center
            center_rotation = _rotation(poses[top_center, 3:])
            in_world = ((body_a >= world * self.bodies_per_world) & (body_a < (world + 1) * self.bodies_per_world)) | (
                (body_b >= world * self.bodies_per_world) & (body_b < (world + 1) * self.bodies_per_world)
            )
            a_top, b_top = np.isin(body_a, bodies[top]), np.isin(body_b, bodies[top])
            cross_layer = (a_top ^ b_top) & (magnitudes > 1.0e-6)
            contact_torque_z = 0.0
            for contact in np.flatnonzero(cross_layer):
                body = body_a[contact] if a_top[contact] else body_b[contact]
                local = local_a[contact] if a_top[contact] else local_b[contact]
                force = forces[contact] if a_top[contact] else -forces[contact]
                point = np.asarray(wp.transform_point(wp.transform(*contact_poses[body]), wp.vec3(*local)))
                contact_torque_z += float(np.cross(point - center, force)[2])
            self.samples.append(
                {
                    "time_s": self.time,
                    "world": world,
                    "amplitude_nm_or_n": amplitude,
                    "load_fraction": self.load_fraction,
                    "load_stop_reason": int(stopped[world]),
                    "applied_core_torque_nm": loads[world, :3].tolist(),
                    "applied_torque_z_nm": float(loads[world, 2]),
                    "force_per_application_point_n": float(loads[world, 3]),
                    "top_yaw_deg": float(np.rad2deg(np.arctan2(fitted[1, 0], fitted[0, 0]))),
                    "top_mean_omega_z_rad_s": float(velocities[bodies[top], 5].mean()),
                    "top_center_yaw_deg": float(np.rad2deg(np.arctan2(center_rotation[1, 0], center_rotation[0, 0]))),
                    "top_center_omega_z_rad_s": float(velocities[top_center, 5]),
                    "top_orientation_residual_deg": np.rad2deg(orientation_residual).tolist(),
                    "active_contact_count": int(np.count_nonzero(in_world & (magnitudes > 1.0e-6))),
                    "active_cross_layer_contacts": int(np.count_nonzero(cross_layer)),
                    "cross_layer_force_magnitude_sum_n": float(magnitudes[cross_layer].sum()),
                    "cross_layer_contact_torque_z_nm": contact_torque_z,
                    "top_fit_rms_mm": float(np.sqrt(np.mean(np.sum(error * error, axis=1))) * 1000),
                    "lower_max_drift_mm": float(np.linalg.norm(relative[~top] - ideal[~top], axis=1).max() * 1000),
                    "loaded_body_displacement_mm": float(
                        np.linalg.norm(relative[self.loaded_piece] - ideal[self.loaded_piece]) * 1000
                    ),
                    "loaded_body_displacement_vector_mm": (
                        (relative[self.loaded_piece] - ideal[self.loaded_piece]) * 1000
                    ).tolist(),
                    "loaded_body_radial_distance_change_mm": float(
                        (np.linalg.norm(relative[self.loaded_piece]) - np.linalg.norm(ideal[self.loaded_piece])) * 1000
                    ),
                }
            )
        if self.frame % 60 == 0:
            print(json.dumps(self.samples[-len(self.args.amplitudes) :]), flush=True)

    def report(self, error=None):
        return {
            "probe_version": "wrench-audit-v3",
            "mode": self.args.mode,
            "error": error,
            "configuration": vars(self.args) | {"report": str(self.args.report)},
            "physical_top_layer_izz_kg_m2": self.data["top_layer_physical_izz_kg_m2"],
            "added_top_layer_izz_kg_m2": self.data["top_layer_added_izz_kg_m2"],
            "unsupported": [
                "screw preload",
                "dry joint friction",
                "axial core compliance",
                "real measured torque calibration",
            ],
            "load_stop_codes": {
                0: "active",
                1: "first quarter turn",
                2: "diagnostic speed limit",
                3: "diagnostic COM-coherence limit",
                4: "diagnostic orientation-coherence limit",
            },
            "load_stop_reason": self.stop_reason.numpy().tolist(),
            "load_stop_time_s": self.stop_time.numpy().tolist(),
            "load_stop_during_unloaded_settle": (
                (self.stop_time.numpy() >= 0) & (self.stop_time.numpy() < self.args.settle)
            ).tolist(),
            "wrench_columns": ["core_Mx_Nm", "core_My_Nm", "core_Mz_Nm", "force_per_point_N"],
            "peak_absolute_applied_load": self.peak_load.numpy().tolist(),
            "last_applied_load_before_stop": self.trip_load.numpy().tolist(),
            "load_stop_state_columns": [
                "top_center_yaw_rad",
                "top_center_omega_z_rad_s",
                "max_cubie_com_error_m",
                "max_cubie_orientation_error_rad",
                "max_cubie_angular_speed_rad_s",
                "loaded_body_displacement_x_m",
                "loaded_body_displacement_y_m",
                "loaded_body_displacement_z_m",
            ],
            "load_stop_state": self.trip_state.numpy().tolist(),
            "first_two_degree_motion_time_s": self.onset_time.numpy().tolist(),
            "load_at_first_two_degree_motion": self.onset_load.numpy().tolist(),
            "cad": self.cad,
            "rows": self.solver.constraint_row_watermarks(),
            "collision_buffers": self.pipeline.narrow_phase.buffer_capacity_status(),
            "samples": self.samples,
        }


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--mode", choices=("torque", "couple", "off-axis"), default="couple")
    parser.add_argument("--collision", choices=("sdf", "convex"), default="sdf")
    parser.add_argument("--amplitudes", nargs="+", type=float, default=[0.001, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--armature", type=float, default=0.0)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--friction", type=float, default=0.0)
    parser.add_argument(
        "--gravity", type=float, default=9.81, help="Gravity magnitude [m/s²]; zero is a different diagnostic case."
    )
    parser.add_argument("--clearance", type=float, default=0.00008)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--contacts", type=int, default=768)
    parser.add_argument("--rows", type=int, default=2304)
    parser.add_argument("--settle", type=float, default=0.5)
    parser.add_argument("--ramp", type=float, default=0.25)
    parser.add_argument("--hold", type=float, default=2.0)
    parser.add_argument("--relax", type=float, default=1.0)
    parser.add_argument(
        "--max-speed", type=float, default=5.0, help="Diagnostic angular-speed guard [rad/s], not a real-cube limit."
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=0.001,
        help="Diagnostic cubie-coherence guard [m], not calibrated compliance.",
    )
    parser.add_argument("--max-angle", type=float, default=5.0, help="Diagnostic cubie-orientation guard [degrees].")
    parser.add_argument("--report", type=Path, default=Path("cube_probe.json"))
    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()
    example = Example(args)
    failure = None
    try:
        duration = args.settle + 2 * args.ramp + args.hold + args.relax
        for _ in range(round(60 * duration)):
            example.step()
    except RuntimeError as error:
        failure = str(error)
        print(failure, flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(example.report(failure), indent=2) + "\n")
    print(args.report, flush=True)
    if failure:
        raise SystemExit(1)
