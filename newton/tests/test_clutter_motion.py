# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check contact-driven container targets without a physics rollout."""

import unittest
from itertools import pairwise
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.examples.robot.example_robot_clutter import (
    Example,
    _joint_targets,
    build_franka,
    create_parser,
    interaction_path,
    interaction_phase,
    interaction_times,
    solve_waypoints,
)


class TestClutterMotion(unittest.TestCase):
    def setUp(self):
        self._device_scope = wp.ScopedDevice("cpu")
        self._device_scope.__enter__()
        self.addCleanup(self._device_scope.__exit__, None, None, None)

    def _targets(self, path, phase, rate):
        # Two replicated worlds with free-joint-style unequal q/qd strides.
        # This tests the target writer, not the solver's multiworld support.
        q = wp.full((2, 23), -99.0, dtype=float)
        qd = wp.full((2, 21), -99.0, dtype=float)
        wp.launch(
            _joint_targets,
            dim=(2, 9),
            inputs=[wp.array(path, dtype=float), phase, rate, q, qd],
            device="cpu",
        )
        return q.numpy(), qd.numpy()

    def test_analytic_velocity_and_independent_array_strides(self):
        """Different coordinate/DOF strides must not redirect velocity writes."""
        path = np.sin(np.arange(36).reshape(4, 9)).astype(np.float32)
        phase, rate = 1.3, 2.4
        q, qd = self._targets(path, phase, rate)
        u = phase - int(phase)
        blend = 10 * u**3 - 15 * u**4 + 6 * u**5
        delta = path[2] - path[1]
        expected_q = path[1] + blend * delta
        expected_qd = delta * 30 * u**2 * (1 - u) ** 2 * rate
        np.testing.assert_allclose(q[:, :9], np.tile(expected_q, (2, 1)), atol=2.0e-6)
        np.testing.assert_allclose(qd[:, :9], np.tile(expected_qd, (2, 1)), atol=2.0e-6)
        np.testing.assert_array_equal(q[:, 9:], -99.0)
        np.testing.assert_array_equal(qd[:, 9:], -99.0)
        dt = 1.0e-3
        before, _ = self._targets(path, phase - dt * rate, rate)
        after, _ = self._targets(path, phase + dt * rate, rate)
        np.testing.assert_allclose((after[:, :9] - before[:, :9]) / (2 * dt), qd[:, :9], atol=2.0e-4)

    def test_knot_velocities_are_zero(self):
        """Stops at every knot and the cycle seam keep zero commanded velocity."""
        path = np.vstack((np.zeros(9), np.ones(9), np.full(9, -1), np.zeros(9))).astype(np.float32)
        for phase in range(len(path)):
            with self.subTest(phase=phase):
                q, qd = self._targets(path, float(phase), 3.0)
                np.testing.assert_array_equal(q[:, :9], np.tile(path[phase], (2, 1)))
                np.testing.assert_array_equal(qd[:, :9], 0.0)

    def test_container_cycle_counts_and_grasp_schedule(self):
        """Keep all strokes inside the closed-gripper interval for any cycle count."""
        target = np.array((0.5, -0.215, 0.0246))
        for scene in ("coins", "hardware", "dishes"):
            for cycles in (1, 2, 4, 7):
                with self.subTest(scene=scene, cycles=cycles):
                    path = interaction_path(scene, target, shake_amplitude=0.06, shake_cycles=cycles)
                    times = interaction_times(scene, shake_cycles=cycles)
                    self.assertEqual(path.shape, (2 * cycles + 6, 5))
                    self.assertEqual(len(times), len(path))
                    self.assertTrue(np.all(np.diff(times) > 0))
                    np.testing.assert_array_equal(times[[0, -1]], (0, 1))
                    np.testing.assert_allclose(path[0], path[-1])
                    np.testing.assert_allclose(path[2:-2, 4], 0.007)
                    np.testing.assert_allclose(path[[0, 1, -2, -1], 4], 0.030)
                    strokes = path[3 : 3 + 2 * cycles, :3] - target
                    np.testing.assert_allclose(strokes[:, 0], np.tile((0.06, -0.06), cycles), atol=1.0e-7)
                    np.testing.assert_allclose(strokes[:, 1:], 0.0, atol=1.0e-7)
        with self.assertRaises(ValueError):
            interaction_path("coins", target, shake_cycles=0)
        with self.assertRaises(ValueError):
            interaction_times("coins", shake_cycles=0)

    def test_nonuniform_phase_rate_and_stationary_intervals(self):
        """Use each segment's physical duration, with no feed-forward while settling."""
        times, settle, period = interaction_times("hardware"), 3.0, 10.0
        for segment, (begin, end) in enumerate(pairwise(times)):
            time = settle + period * (0.75 * begin + 0.25 * end)
            phase, rate = interaction_phase(time, settle=settle, period=period, times=times, motion=True)
            self.assertAlmostEqual(phase, segment + 0.25)
            self.assertAlmostEqual(rate, 1 / (period * (end - begin)))
        path = np.vstack((np.zeros(9), np.ones(9))).astype(np.float32)
        for time, motion in ((0.0, True), (settle - 0.001, True), (settle + 5.0, False)):
            phase, rate = interaction_phase(time, settle=settle, period=period, times=times, motion=motion)
            self.assertEqual((phase, rate), (0.0, 0.0))
            _, qd = self._targets(path, phase, rate)
            np.testing.assert_array_equal(qd[:, :9], 0.0)
        phase, _ = interaction_phase(settle + period, settle=settle, period=period, times=times, motion=True)
        self.assertEqual(phase, 0.0)

    def test_shake_axis_preserves_closure_and_timing(self):
        """Only the requested horizontal stroke direction changes between X and Y."""
        target = np.array((0.5, -0.145, 0.0496))
        for scene in ("hardware", "coins", "dishes"):
            for cycles in (1, 4):
                with self.subTest(scene=scene, cycles=cycles):
                    x_path = interaction_path(scene, target, shake_cycles=cycles, shake_axis="x")
                    y_path = interaction_path(scene, target, shake_cycles=cycles, shake_axis="y")
                    np.testing.assert_array_equal(x_path, interaction_path(scene, target, shake_cycles=cycles))
                    self.assertEqual(len(y_path), len(interaction_times(scene, shake_cycles=cycles)))
                    np.testing.assert_array_equal(y_path[:, 2:], x_path[:, 2:])
                    np.testing.assert_array_equal(y_path[[0, 1, 2, -3, -2, -1]], x_path[[0, 1, 2, -3, -2, -1]])
                    stroke = y_path[3 : 3 + 2 * cycles, :3] - target
                    np.testing.assert_allclose(stroke[:, 1], np.tile((0.06, -0.06), cycles), atol=1.0e-7)
                    np.testing.assert_allclose(stroke[:, [0, 2]], 0.0, atol=1.0e-7)
                    np.testing.assert_array_equal(y_path[0], y_path[-1])
        with self.assertRaises(ValueError):
            interaction_path("hardware", target, shake_axis="z")

    def test_container_velocity_actuators_do_not_change_cube_modes(self):
        """MJWarp must install velocity actuators; the cube's baseline stays unchanged."""
        container, _ = build_franka(container_grasp=True)
        cube, _ = build_franka(cube_grasp=True)
        np.testing.assert_array_equal(container.joint_target_mode[:9], newton.JointTargetMode.POSITION_VELOCITY)
        np.testing.assert_array_equal(cube.joint_target_mode[:9], newton.JointTargetMode.POSITION)
        self.assertEqual(len(interaction_path("cube", (0.5, 0, 0.08))), len(interaction_times("cube")))

    def test_arm_stiffness_scale_preserves_fingers_and_physical_limits(self):
        """Scale only arm PD gains, never actuator authority or robot mass properties."""
        baseline, _ = build_franka(container_grasp=True, arm_stiffness_scale=1.0)
        scaled, _ = build_franka(container_grasp=True, arm_stiffness_scale=2.0)
        np.testing.assert_allclose(scaled.joint_target_ke[:7], 2 * np.asarray(baseline.joint_target_ke[:7]))
        np.testing.assert_allclose(scaled.joint_target_kd[:7], np.sqrt(2) * np.asarray(baseline.joint_target_kd[:7]))
        np.testing.assert_array_equal(scaled.joint_target_ke[7:], baseline.joint_target_ke[7:])
        np.testing.assert_array_equal(scaled.joint_target_kd[7:], baseline.joint_target_kd[7:])
        for attribute in ("joint_effort_limit", "joint_armature", "body_mass", "body_inertia"):
            np.testing.assert_array_equal(getattr(scaled, attribute), getattr(baseline, attribute))
        for invalid in (0.0, -1.0, float("nan"), float("inf"), -float("inf")):
            with self.subTest(scale=invalid), self.assertRaisesRegex(ValueError, "finite and positive"):
                build_franka(container_grasp=True, arm_stiffness_scale=invalid)

    def test_scene_specific_cli_defaults_and_explicit_overrides(self):
        """Resolve hardware-only changes before loading assets; preserve other scenes."""
        cases = [
            (["--scene", scene], axis, tilt)
            for scene, axis, tilt in (
                ("hardware", "y", 30.0),
                ("coins", "x", 0.0),
                ("dishes", "x", 0.0),
                ("cube", "x", 0.0),
            )
        ]
        cases.append((["--scene", "hardware", "--shake-axis", "x", "--grasp-tilt", "0"], "x", 0.0))
        for command, expected_axis, expected_tilt in cases:
            with self.subTest(command=command):
                args = create_parser().parse_args(command)
                example = Example.__new__(Example)
                with (
                    patch(
                        "newton.examples.robot.example_robot_clutter.build_franka",
                        side_effect=RuntimeError("Stop before asset construction"),
                    ) as build,
                ):
                    with self.assertRaisesRegex(RuntimeError, "Stop before asset construction"):
                        Example.__init__(example, None, args)
                    build.assert_called_once()
                self.assertEqual(example.shake_axis, expected_axis)
                self.assertEqual(example.grasp_tilt, expected_tilt)

    def test_tilted_hardware_waypoints_have_requested_tcp_orientation(self):
        """Check CPU FK after setup IK; this does not qualify contact or grasp forces."""
        builder, tcp = build_franka(container_grasp=True)
        model = builder.finalize(device="cpu")
        waypoints = interaction_path("hardware", (0.5, -0.145, 0.0496), shake_cycles=1, shake_axis="y")
        tilt = np.deg2rad(30.0)
        solved = solve_waypoints(model, tcp, waypoints, grasp_tilt=tilt)
        state = model.state()
        for q, waypoint in zip(solved, waypoints, strict=True):
            state.joint_q.assign(q)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            pose = state.body_q.numpy()[tcp]
            desired = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(waypoint[3])) * wp.quat_from_axis_angle(
                wp.vec3(1, 0, 0), float(np.pi + tilt)
            )
            angle = 2 * np.arccos(np.clip(abs(np.dot(pose[3:], np.asarray(desired))), 0, 1))
            self.assertLess(np.linalg.norm(pose[:3] - waypoint[:3]), 0.008)
            self.assertLess(angle, 0.08)
            np.testing.assert_allclose(q[7:9], waypoint[4], atol=1.0e-7)
        # The API's omitted tilt remains exactly equivalent to explicit zero.
        closed_path = waypoints[[0, 1, -1]]
        np.testing.assert_array_equal(
            solve_waypoints(model, tcp, closed_path), solve_waypoints(model, tcp, closed_path, grasp_tilt=0.0)
        )

    def test_fpgs_replicated_scene_reaches_asset_construction(self):
        """Accept replicated scenes after the solver's target-layout correction."""
        args = create_parser().parse_args(["--solver", "fpgs", "--world-count", "2"])
        with (
            patch(
                "newton.examples.robot.example_robot_clutter.build_franka",
                side_effect=RuntimeError("Stop before asset construction"),
            ) as build,
        ):
            with self.assertRaisesRegex(RuntimeError, "Stop before asset construction"):
                Example(None, args)
            build.assert_called_once()


if __name__ == "__main__":
    unittest.main()
