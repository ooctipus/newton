# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prescale joint velocities in parallel lanes, bit for bit like the serial PhysX prescale.

``_serial_prescale`` is ``kernels.prescale_joint_velocity_limits`` as it was
before it ran in lanes (main b7932853): one thread per articulation scaled a
copy of the joint velocities in place. It stays here as the reference.
``_emulate_lanes`` replays the threads of a CUDA lane launch in NumPy, warp
shuffles included, so every lane width is checked on CPU as well.
"""

import itertools
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.sim import BodyFlags, JointType
from newton._src.solvers.feather_pgs import field_group
from newton._src.solvers.feather_pgs.joint_table import JointTable
from newton._src.solvers.feather_pgs.kernels import prescale_joint_velocity_limits
from newton._src.solvers.feather_pgs.solver_feather_pgs import _velocity_prescale_launch
from newton.solvers import SolverFeatherPGS
from newton.tests import test_feather_pgs_mixed_tree_free_body as mixed_tree


@wp.kernel
def _serial_prescale(
    articulation_start: wp.array[int],
    joint_type: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_velocity_limit: wp.array[float],
    body_flags: wp.array[wp.int32],
    drive_slot: wp.array[int],
    skip_driven: int,
    # in/out
    joint_qd: wp.array[float],
):
    art = wp.tid()
    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]

    ratio = float(1.0)
    for j in range(joint_start, joint_end):
        if (body_flags[joint_child[j]] & BodyFlags.KINEMATIC) != 0:
            continue
        jtype = joint_type[j]
        if jtype != JointType.PRISMATIC and jtype != JointType.REVOLUTE and jtype != JointType.D6:
            continue

        lin_count = joint_dof_dim[j, 0]
        ang_count = joint_dof_dim[j, 1]
        axis_count = lin_count + ang_count
        qd_start = joint_qd_start[j]

        for axis in range(axis_count):
            dof = qd_start + axis
            if skip_driven != 0:
                if drive_slot[dof] >= 0:
                    continue
            limit = joint_velocity_limit[dof]
            qd_abs = wp.abs(joint_qd[dof])
            if limit > 0.0 and wp.isfinite(limit) and qd_abs > 0.0:
                scale = limit / qd_abs
                if scale < ratio:
                    ratio = scale

    if ratio >= 1.0:
        return

    for j in range(joint_start, joint_end):
        if (body_flags[joint_child[j]] & BodyFlags.KINEMATIC) != 0:
            continue
        jtype = joint_type[j]
        if jtype != JointType.PRISMATIC and jtype != JointType.REVOLUTE and jtype != JointType.D6:
            continue

        lin_count = joint_dof_dim[j, 0]
        ang_count = joint_dof_dim[j, 1]
        axis_count = lin_count + ang_count
        qd_start = joint_qd_start[j]

        for axis in range(axis_count):
            dof = qd_start + axis
            if skip_driven != 0:
                if drive_slot[dof] >= 0:
                    continue
            joint_qd[dof] = joint_qd[dof] * ratio


_FIXED_DOF_DIMS = {
    JointType.PRISMATIC: (1, 0),
    JointType.REVOLUTE: (0, 1),
    JointType.BALL: (0, 3),
    JointType.FIXED: (0, 0),
    JointType.FREE: (3, 3),
    JointType.DISTANCE: (3, 3),
    JointType.ROD: (2, 2),
}
# Finite, zero, negative, infinite, NaN, tiny and huge limits.
_LIMITS = np.array((0.5, 1.0, 2.5, 4.0, 0.0, -1.0, np.inf, np.nan, 1.0e-30, 1.0e30), dtype=np.float32)
# No input velocity and no scaled velocity reaches this value, so it marks a DOF the kernel failed to write.
_UNWRITTEN = np.float32(-1.5e38)
_ARRAYS = (
    "articulation_start",
    "joint_type",
    "joint_child",
    "joint_qd_start",
    "joint_dof_dim",
    "joint_velocity_limit",
    "body_flags",
    "drive_slot",
)
# Random layouts: seed, articulation count, joints ahead of the first articulation and the
# interval of articulations longer than a warp (0: none).
_CASES = (
    (0, 1, 0, 0),
    (1, 3, 2, 0),
    (2, 37, 0, 5),
    (3, 37, 3, 7),
    (4, 101, 1, 0),
    (5, 2, 0, 1),
)
# Every lane width a CUDA launch can use.
_LANE_WIDTHS = (1, 2, 4, 8, 16, 32)
# Joint types whose leading DOFs the prescale reads and scales.
_LIMITED_TYPES = (int(JointType.PRISMATIC), int(JointType.REVOLUTE), int(JointType.D6))


def _layout(seed: int, articulations: int, *, leading_joints: int = 0, long_every: int = 0) -> dict:
    """Return the joint arrays of random articulations.

    The articulations mix every joint type with free roots, kinematic
    children, empty articulations and, every ``long_every`` articulations,
    one longer than a warp. ``leading_joints`` joints precede the first
    articulation.
    """
    rng = np.random.default_rng(seed)
    types, dims, starts = [], [], []

    def add(joint_type):
        if joint_type == JointType.D6:
            dims.append((int(rng.integers(0, 4)), int(rng.integers(0, 4))))
        else:
            dims.append(_FIXED_DOF_DIMS[joint_type])
        types.append(int(joint_type))

    for _ in range(leading_joints):
        add(JointType.REVOLUTE)
    kinds = (
        JointType.PRISMATIC,
        JointType.REVOLUTE,
        JointType.D6,
        JointType.BALL,
        JointType.FIXED,
        JointType.DISTANCE,
        JointType.ROD,
    )
    for art in range(articulations):
        starts.append(len(types))
        if rng.random() < 0.1:
            continue
        if rng.random() < 0.3:
            add(JointType.FREE)
        count = 40 if long_every and art % long_every == 0 else int(rng.integers(1, 24))
        for _ in range(count):
            add(kinds[int(rng.integers(0, len(kinds)))])
    starts.append(len(types))

    joint_count = len(types)
    dof_dim = np.asarray(dims, dtype=np.int32).reshape(joint_count, 2)
    qd_start = np.concatenate(([0], np.cumsum(dof_dim.sum(axis=1)))).astype(np.int32)
    dof_count = int(qd_start[-1])
    qd = (rng.normal(size=dof_count) * 4.0).astype(np.float32)
    special = rng.random(dof_count)
    qd[special < 0.02] = np.inf
    qd[(special >= 0.02) & (special < 0.04)] = -np.inf
    qd[(special >= 0.04) & (special < 0.06)] = np.nan
    qd[(special >= 0.06) & (special < 0.08)] = -0.0
    qd[(special >= 0.08) & (special < 0.10)] = 0.0
    qd[(special >= 0.10) & (special < 0.12)] = 1.0e-40
    limit = _LIMITS[rng.integers(0, len(_LIMITS), size=dof_count)] * rng.uniform(0.5, 2.0, size=dof_count)
    kinematic = rng.random(joint_count) < 0.15
    return {
        "articulation_start": np.asarray(starts, dtype=np.int32),
        "joint_type": np.asarray(types, dtype=np.int32),
        # Each joint has its own child body.
        "joint_child": np.arange(joint_count, dtype=np.int32),
        "joint_qd_start": qd_start,
        "joint_dof_dim": dof_dim,
        "joint_velocity_limit": limit.astype(np.float32),
        "body_flags": np.where(kinematic, int(BodyFlags.KINEMATIC), int(BodyFlags.DYNAMIC)).astype(np.int32),
        "drive_slot": np.where(rng.random(max(dof_count, 1)) < 0.3, 1, -1).astype(np.int32),
        "joint_qd": qd,
    }


def _serial(layout: dict, skip_driven: int, device) -> np.ndarray:
    """Scale a copy of the velocities in place with the serial reference."""
    arrays = [wp.array(layout[name], device=device) for name in _ARRAYS]
    reference = wp.array(layout["joint_qd"], device=device)
    wp.launch(
        _serial_prescale,
        dim=len(layout["articulation_start"]) - 1,
        inputs=[*arrays, skip_driven],
        outputs=[reference],
        device=device,
    )
    return reference.numpy()


def _lanes(layout: dict, skip_driven: int, device, lanes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the lane kernel's output, every DOF prefilled as unwritten, and its input afterwards.

    The launch covers complete warps. CPU threads cannot exchange ratios, so with more than one lane
    a CPU launch scales each lane's joints by that lane's own ratio.
    """
    count = len(layout["articulation_start"]) - 1
    joints = field_group.upload(JointTable, device, **{name: layout[name] for name in _ARRAYS[:-1]})
    drive_slot = wp.array(layout["drive_slot"], device=device)
    joint_qd = wp.array(layout["joint_qd"], device=device)
    output = wp.full(len(layout["joint_qd"]), _UNWRITTEN, dtype=float, device=device)
    wp.launch(
        prescale_joint_velocity_limits,
        dim=(count * lanes + 31) // 32 * 32,
        inputs=[joints, count, lanes, drive_slot, skip_driven, joint_qd],
        outputs=[output],
        device=device,
    )
    return output.numpy(), joint_qd.numpy()


def _emulated_axis_count(layout: dict, joint: int) -> int:
    """Return how many leading DOFs of ``joint`` the prescale reads and scales."""
    if int(layout["joint_type"][joint]) not in _LIMITED_TYPES:
        return 0
    if int(layout["body_flags"][layout["joint_child"][joint]]) & int(BodyFlags.KINEMATIC):
        return 0
    return int(layout["joint_dof_dim"][joint].sum())


def _emulate_lanes(
    layout: dict, skip_driven: int, lanes: int, *, combine: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay in NumPy, in thread order, every thread of a complete-warp lane launch.

    Thread ``tid`` is lane ``tid % lanes`` of articulation ``tid // lanes``; threads past the last
    articulation pad the launch. With ``combine`` each aligned group of lanes takes its minimum ratio
    through the XOR butterfly of a 32-lane warp, as on CUDA. Without it each lane keeps its own ratio,
    as on CPU, where the lane minimum returns its input.

    Returns:
        The written velocities (``_UNWRITTEN`` where no thread wrote), the number of writes of each
        DOF and the ratio of each thread.
    """
    start, qd_start = layout["articulation_start"], layout["joint_qd_start"]
    qd, limit, drive = layout["joint_qd"], layout["joint_velocity_limit"], layout["drive_slot"]
    count = len(start) - 1
    threads = (count * lanes + 31) // 32 * 32
    joints = [
        (int(start[tid // lanes]), int(start[tid // lanes + 1])) if tid // lanes < count else (0, 0)
        for tid in range(threads)
    ]

    def skipped(dof: int) -> bool:
        return skip_driven != 0 and drive[dof] >= 0

    ratios = np.ones(threads, dtype=np.float32)
    # A tiny speed under a huge limit overflows to an infinite quotient, as in the kernel.
    with np.errstate(over="ignore"):
        for tid, (first, end) in enumerate(joints):
            for joint in range(first + tid % lanes, end, lanes):
                begin = int(qd_start[joint])
                for dof in range(begin, begin + _emulated_axis_count(layout, joint)):
                    speed = np.abs(qd[dof])
                    if not skipped(dof) and limit[dof] > 0.0 and np.isfinite(limit[dof]) and speed > 0.0:
                        scale = limit[dof] / speed
                        if scale < ratios[tid]:
                            ratios[tid] = scale
    if combine:
        lane = np.arange(32)
        for warp in range(0, threads, 32):
            ratio = ratios[warp : warp + 32]
            offset = lanes >> 1
            while offset:
                other = ratio[lane ^ offset]
                ratio[:] = np.where(other < ratio, other, ratio)
                offset >>= 1

    output = np.full(len(qd), _UNWRITTEN, dtype=np.float32)
    writes = np.zeros(len(qd), dtype=np.int64)
    # An infinite speed times a zero ratio is NaN, as in the kernel.
    with np.errstate(invalid="ignore"):
        for tid, (first, end) in enumerate(joints):
            lane, ratio = tid % lanes, ratios[tid]
            dofs = []
            if tid // lanes == 0:
                # Joints ahead of the first articulation.
                dofs += [(dof, False) for dof in range(lane, int(qd_start[first]), lanes)]
            if ratio < 1.0:
                for joint in range(first + lane, end, lanes):
                    begin = int(qd_start[joint])
                    scaled_end = begin + _emulated_axis_count(layout, joint)
                    dofs += [
                        (dof, dof < scaled_end and not skipped(dof)) for dof in range(begin, int(qd_start[joint + 1]))
                    ]
            else:
                dofs += [(dof, False) for dof in range(int(qd_start[first]) + lane, int(qd_start[end]), lanes)]
            for dof, scaled in dofs:
                output[dof] = qd[dof] * ratio if scaled else qd[dof]
                writes[dof] += 1
    return output, writes, ratios


def _assert_bits_equal(actual: np.ndarray, expected: np.ndarray, message: str = ""):
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32), err_msg=message)


def _solver_model(device) -> newton.Model:
    """Build a fixed-base arm, a free root with a D6 child, a ball chain and a pendulum on a kinematic root."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(np.eye(3, dtype=np.float32) * 0.1)
    Dof = newton.ModelBuilder.JointDofConfig

    parent, joints = -1, []
    for add in (builder.add_joint_revolute, builder.add_joint_prismatic, builder.add_joint_revolute):
        child = builder.add_link(mass=1.0, inertia=inertia)
        joints.append(add(parent=parent, child=child, axis=newton.Axis.Z))
        parent = child
    builder.add_articulation(joints)

    root = builder.add_link(mass=1.0, inertia=inertia, xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()))
    child = builder.add_link(mass=1.0, inertia=inertia)
    builder.add_articulation(
        [
            builder.add_joint_free(parent=-1, child=root),
            builder.add_joint_d6(
                parent=root,
                child=child,
                linear_axes=[Dof(axis=newton.Axis.X)],
                angular_axes=[Dof(axis=newton.Axis.Y), Dof(axis=newton.Axis.Z)],
            ),
        ]
    )

    ball = builder.add_link(mass=1.0, inertia=inertia)
    tip = builder.add_link(mass=1.0, inertia=inertia)
    builder.add_articulation(
        [
            builder.add_joint_ball(parent=-1, child=ball),
            builder.add_joint_revolute(parent=ball, child=tip, axis=newton.Axis.X),
        ]
    )

    root = builder.add_link(mass=1.0, inertia=inertia, is_kinematic=True)
    tip = builder.add_link(mass=1.0, inertia=inertia)
    builder.add_articulation(
        [
            builder.add_joint_revolute(parent=-1, child=root, axis=newton.Axis.Z),
            builder.add_joint_revolute(parent=root, child=tip, axis=newton.Axis.Y),
        ]
    )
    model = builder.finalize(device=device)
    dofs = model.joint_dof_count
    model.joint_velocity_limit.assign(np.linspace(0.5, 2.0, dofs, dtype=np.float32))
    return model


def _model_layout(model: newton.Model, joint_qd: np.ndarray, drive_slot: np.ndarray) -> dict:
    layout = {name: getattr(model, name).numpy() for name in _ARRAYS if name != "drive_slot"}
    layout["drive_slot"] = drive_slot
    layout["joint_qd"] = joint_qd
    return layout


def _devices() -> list:
    return ["cpu", *wp.get_cuda_devices()]


def _lane_width(device) -> int:
    """Return a lane width the device can run: one on CPU, a partial warp on CUDA."""
    return 1 if wp.get_device(device).is_cpu else 8


def _recording_copies(destinations: list):
    """Patch ``wp.copy`` to record each destination array."""
    copy = wp.copy

    def recording(destination, source, *args, **kwargs):
        destinations.append(destination)
        return copy(destination, source, *args, **kwargs)

    return patch.object(wp, "copy", recording)


class TestFeatherPGSVelocityPrescale(unittest.TestCase):
    def _check_layouts(self, device, lane_widths):
        scaled = unscaled = scaled_articulations = copied_articulations = 0
        for seed, articulations, leading, long_every in _CASES:
            layout = _layout(seed, articulations, leading_joints=leading, long_every=long_every)
            dof_ranges = list(itertools.pairwise(layout["joint_qd_start"][layout["articulation_start"]].tolist()))
            for skip_driven in (0, 1):
                expected = _serial(layout, skip_driven, device)
                changed = expected.view(np.uint32) != layout["joint_qd"].view(np.uint32)
                scaled += int(np.count_nonzero(changed))
                unscaled += int(np.count_nonzero(~changed))
                for first, end in dof_ranges:
                    if end > first:
                        if changed[first:end].any():
                            scaled_articulations += 1
                        else:
                            copied_articulations += 1
                for lanes in lane_widths:
                    actual, source = _lanes(layout, skip_driven, device, lanes)
                    message = f"seed={seed} skip_driven={skip_driven} lanes={lanes}"
                    _assert_bits_equal(actual, expected, message)
                    _assert_bits_equal(source, layout["joint_qd"], message)
        # Both the scaling and the copying branches ran.
        self.assertGreater(scaled, 100)
        self.assertGreater(unscaled, 100)
        # A ratio below 1 changes the DOF that set it, so an articulation whose DOFs all keep their
        # bits took the copy branch. Unscaled DOFs of scaled articulations do not count here.
        self.assertGreaterEqual(scaled_articulations, 10)
        self.assertGreaterEqual(copied_articulations, 10)

    def test_lanes_match_serial_prescale_bitwise_cpu(self):
        """Write every DOF bit for bit like the serial prescale of a copy, with one lane per articulation."""
        self._check_layouts("cpu", (1,))

    @unittest.skipUnless(wp.is_cuda_available(), "Lane groups exchange ratios with CUDA warp shuffles")
    def test_lanes_match_serial_prescale_bitwise_cuda(self):
        """Write every DOF bit for bit like the serial prescale for every lane width of a warp."""
        for device in wp.get_cuda_devices():
            with self.subTest(device=str(device)):
                self._check_layouts(device, _LANE_WIDTHS)

    def test_every_lane_width_in_emulation_and_on_cpu(self):
        """Match the serial prescale at every lane width in emulation, and the kernel's lane mapping on CPU.

        With the warp butterfly, every lane of a group holds the same ratio bits and each DOF is
        written once. A CPU launch of the kernel, whose lanes keep their own ratios, must match the
        emulation without the butterfly wherever at most one thread writes a DOF: lanes of a group
        can then take different branches, which split the DOFs differently. That output differs from
        the serial prescale, so the CUDA test of these layouts catches a missing or broken butterfly.
        """
        uncombined_errors = dict.fromkeys(_LANE_WIDTHS[1:], 0)
        for seed, articulations, leading, long_every in _CASES:
            layout = _layout(seed, articulations, leading_joints=leading, long_every=long_every)
            count = len(layout["articulation_start"]) - 1
            for skip_driven in (0, 1):
                expected = _serial(layout, skip_driven, "cpu")
                for lanes in _LANE_WIDTHS:
                    message = f"seed={seed} skip_driven={skip_driven} lanes={lanes}"
                    combined, writes, ratios = _emulate_lanes(layout, skip_driven, lanes, combine=True)
                    _assert_bits_equal(combined, expected, message)
                    np.testing.assert_array_equal(writes, 1, err_msg=message)
                    group_bits = ratios[: count * lanes].view(np.uint32).reshape(count, lanes)
                    np.testing.assert_array_equal(
                        group_bits, np.broadcast_to(group_bits[:, :1], group_bits.shape), err_msg=message
                    )

                    uncombined, writes, _ = _emulate_lanes(layout, skip_driven, lanes, combine=False)
                    actual, source = _lanes(layout, skip_driven, "cpu", lanes)
                    single = writes <= 1
                    _assert_bits_equal(actual[single], uncombined[single], message)
                    _assert_bits_equal(source, layout["joint_qd"], message)
                    if lanes > 1:
                        errors = uncombined.view(np.uint32) != expected.view(np.uint32)
                        uncombined_errors[lanes] += int(np.count_nonzero(errors))
        for lanes, errors in uncombined_errors.items():
            self.assertGreater(errors, 100, f"lanes={lanes}")

    def test_dofless_and_unarticulated_layouts(self):
        """Copy the velocities of joints outside every articulation and write nothing for DOF-less joints."""
        for device in _devices():
            with self.subTest(device=str(device)):
                layout = _layout(0, 4)
                layout["joint_dof_dim"][:] = 0
                layout["joint_qd_start"][:] = 0
                layout["joint_qd"] = np.zeros(0, dtype=np.float32)
                actual, _ = _lanes(layout, 0, device, _lane_width(device))
                self.assertEqual(actual.shape, (0,))

                # Every joint precedes the only articulation, which is empty.
                layout = _layout(1, 0, leading_joints=5)
                layout["articulation_start"] = np.array([5, 5], dtype=np.int32)
                layout["joint_velocity_limit"][:] = 1.0e-3
                actual, _ = _lanes(layout, 0, device, _lane_width(device))
                _assert_bits_equal(actual, _serial(layout, 0, device))
                _assert_bits_equal(actual, layout["joint_qd"])

    def test_launch_covers_complete_warps(self):
        """Size lane groups by the mean articulation, capped at a warp, and launch complete CUDA warps."""

        def launch(starts, *, cuda):
            # The articulations end at the last joint.
            return _velocity_prescale_launch(np.asarray(starts), joint_count=int(starts[-1]), cuda=cuda)

        self.assertEqual(launch(np.array([0], dtype=np.int32), cuda=True), (1, 0))
        self.assertEqual(launch([0, 21, 22, 43, 44], cuda=False), (1, 4))
        # A 21-joint and a one-joint articulation average 11 joints.
        self.assertEqual(launch([0, 21, 22, 43, 44], cuda=True), (16, 64))
        self.assertEqual(launch([0, 8, 16, 24], cuda=True), (8, 32))
        self.assertEqual(launch([2, 2, 3], cuda=True), (1, 32))
        self.assertEqual(launch([0, 100, 200], cuda=True), (32, 64))
        lanes, threads = launch(np.arange(0, 1001, 10), cuda=True)
        self.assertEqual((lanes, threads % 32), (16, 0))
        self.assertGreaterEqual(threads, 100 * lanes)

    def test_launch_requires_articulations_to_end_at_the_last_joint(self):
        """Reject joints past the last articulation, whose DOFs no lane would write."""
        self.assertEqual(_velocity_prescale_launch(np.array([0, 4, 7]), joint_count=7, cuda=True), (4, 32))
        # Without articulations the launcher copies every DOF.
        self.assertEqual(_velocity_prescale_launch(np.array([0]), joint_count=7, cuda=True), (1, 0))
        for cuda in (False, True):
            with self.subTest(cuda=cuda), self.assertRaisesRegex(ValueError, "articulation_start"):
                _velocity_prescale_launch(np.array([0, 4, 7]), joint_count=9, cuda=cuda)

    def test_launcher_writes_every_dof_without_a_copy(self):
        """Write each DOF of a solver model's work velocity in one launch, bit for bit like the serial prescale."""
        for device in _devices():
            with self.subTest(device=str(device)):
                model = _solver_model(device)
                # CPU solvers reject velocity limits (matrix-free mode needs CUDA), but the launcher is the same.
                solver = SolverFeatherPGS(model)
                self.assertEqual(
                    (solver._velocity_prescale_lanes, solver._velocity_prescale_threads),
                    _velocity_prescale_launch(
                        model.articulation_start.numpy(),
                        joint_count=model.joint_count,
                        cuda=wp.get_device(device).is_cuda,
                    ),
                )
                joint_qd = (np.random.default_rng(3).normal(size=model.joint_dof_count) * 3.0).astype(np.float32)
                state = model.state()
                state.joint_qd.assign(joint_qd)
                solver.qd_work.fill_(_UNWRITTEN)
                destinations = []
                with _recording_copies(destinations):
                    solver._launch_velocity_prescale(state.joint_qd, solver.qd_work)
                self.assertEqual(destinations, [])
                drive_slot = solver._vel_limit_drive_slot_arg.numpy()
                expected = _serial(_model_layout(model, joint_qd, drive_slot), solver._vel_limit_skip_driven, device)
                self.assertFalse(np.array_equal(expected, joint_qd))
                _assert_bits_equal(solver.qd_work.numpy(), expected)
                _assert_bits_equal(state.joint_qd.numpy(), joint_qd)

    @unittest.skipUnless(wp.is_cuda_available(), "Joint velocity limits require the CUDA matrix-free mode")
    def test_stage1_prescales_without_a_separate_copy_cuda(self):
        """Write the stage-1 velocity in the prescale launch alone and record the lane width."""
        for device in wp.get_cuda_devices():
            for drive_mode in ("augmented", "physx_pgs"):
                with self.subTest(device=str(device), drive_mode=drive_mode):
                    model = _solver_model(device)
                    solver = SolverFeatherPGS(
                        model,
                        pgs_mode="matrix_free",
                        enable_joint_velocity_limits=True,
                        drive_mode=drive_mode,
                    )
                    # The fused clamp skips driven DOFs; only physx_pgs drives allocate drive rows.
                    self.assertEqual(solver._vel_limit_skip_driven, int(drive_mode == "physx_pgs"))
                    lanes, threads = _velocity_prescale_launch(
                        model.articulation_start.numpy(), joint_count=model.joint_count, cuda=True
                    )
                    self.assertEqual(
                        (solver._velocity_prescale_lanes, solver._velocity_prescale_threads), (lanes, threads)
                    )
                    self.assertEqual(threads % 32, 0)
                    routes = solver.get_path_selection()["routes"]
                    self.assertEqual(routes["velocity_limits"]["prescale_lanes"], lanes)

                    dofs = model.joint_dof_count
                    joint_qd = (np.random.default_rng(7).normal(size=dofs) * 3.0).astype(np.float32)
                    drive_slot = np.full(dofs, -1, dtype=np.int32)
                    if solver._vel_limit_skip_driven:
                        # A previous step allocated drive rows for every other DOF.
                        drive_slot[::2] = np.arange((dofs + 1) // 2, dtype=np.int32)
                        solver.drive_slot.assign(drive_slot)
                    state, following = model.state(), model.state()
                    state.joint_qd.assign(joint_qd)
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                    solver._prepare_augmented_state(state, following, model.control())
                    solver.qd_work.fill_(_UNWRITTEN)
                    destinations = []
                    with _recording_copies(destinations):
                        stage3_qd = solver._stage1_fk_id(state, solver, following)
                    self.assertIs(stage3_qd, solver.qd_work)
                    self.assertFalse(any(destination is solver.qd_work for destination in destinations))
                    expected = _serial(
                        _model_layout(model, joint_qd, drive_slot), solver._vel_limit_skip_driven, device
                    )
                    self.assertFalse(np.array_equal(expected, joint_qd))
                    _assert_bits_equal(solver.qd_work.numpy(), expected)
                    _assert_bits_equal(state.joint_qd.numpy(), joint_qd)

    @unittest.skipUnless(wp.is_cuda_available(), "Joint velocity limits require the CUDA matrix-free mode")
    def test_captured_owner_route_prescales_bitwise_cuda(self):
        """Prescale each substep of a captured env step bit for bit on the held-tree route under both friction laws.

        This is the route of the velocity-limited suite tasks: a held-tree owner, parallel streams
        and one CUDA graph of a collision and two substeps. Copies inside the graph record each
        substep's input velocity and the velocity that the owner's predictor consumes.
        """
        for device in wp.get_cuda_devices():
            for friction_anchor_beta in (0.0, 0.2):
                with self.subTest(device=str(device), friction_anchor_beta=friction_anchor_beta):
                    self._check_captured_owner_route(device, friction_anchor_beta)

    def _check_captured_owner_route(self, device, friction_anchor_beta):
        # Per world, a fixed-base arm with 0.8 rad/s revolute limits over a free box on the ground.
        model = mixed_tree._build(device, speed_limit=0.8)
        with patch.object(SolverFeatherPGS, "_route_pins", {"storage": "dense"}):
            solver = SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                drive_mode="augmented",
                enable_joint_limits=True,
                enable_joint_velocity_limits=True,
                friction_anchor_beta=friction_anchor_beta,
            )
        self.assertIsNotNone(solver._tree_kinetic_state)
        pipeline = newton.CollisionPipeline(model, deterministic=friction_anchor_beta > 0.0)
        contacts = pipeline.contacts()
        states, control = (model.state(), model.state()), model.control()
        revolute = np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.REVOLUTE))
        fast = np.zeros(model.joint_dof_count, dtype=np.float32)
        # Most arm joints start above their limit.
        fast[model.joint_qd_start.numpy()[revolute]] = np.random.default_rng(11).uniform(-1.5, 1.5, len(revolute))

        def set_fast_velocities():
            states[0].joint_qd.assign(fast)
            newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])

        def env_step():
            pipeline.collide(states[0], contacts)
            for state_in, state_out in (states, states[::-1]):
                state_in.clear_forces()
                solver.step(state_in, state_out, control, contacts, mixed_tree.DT)

        set_fast_velocities()
        # Allocate the lazily created buffers before the capture.
        env_step()

        # Prefilled: a record that no copy in the graph wrote fails the bitwise check below.
        inputs = [wp.full(model.joint_dof_count, _UNWRITTEN, dtype=float, device=device) for _ in range(2)]
        consumed = [wp.full(model.joint_dof_count, _UNWRITTEN, dtype=float, device=device) for _ in range(2)]
        launches, predicted = [], []
        launch = solver._launch_velocity_prescale
        predict = solver._tree_kinetic_state.predict

        def recording_launch(joint_qd, prescaled_joint_qd):
            destinations = []
            with _recording_copies(destinations):
                launch(joint_qd, prescaled_joint_qd)
            self.assertFalse(any(destination is prescaled_joint_qd for destination in destinations))
            wp.copy(inputs[len(launches)], joint_qd)
            launches.append(prescaled_joint_qd)

        def recording_predict(state, augmented, step_control, velocity, dt):
            # Each substep predicts once, after its own prescale, from the prescaled velocity.
            self.assertEqual(len(predicted), len(launches) - 1)
            self.assertIs(velocity, launches[-1])
            wp.copy(consumed[len(predicted)], velocity)
            predicted.append(velocity)
            predict(state, augmented, step_control, velocity, dt)

        solver._launch_velocity_prescale = recording_launch
        solver._tree_kinetic_state.predict = recording_predict
        # Finish the uncaptured steps first: the seeded events below do not order their buffer clears.
        wp.synchronize_device(device)
        with wp.ScopedCapture(device=device) as capture:
            # The uncaptured steps recorded each buffer slot's memset event outside the graph, and a
            # captured wait on such an event fails (CUDA error 905). Seed both slots, as every capture must.
            solver.seed_double_buffer_events()
            env_step()
        self.assertEqual(len(launches), 2)
        self.assertEqual(len(predicted), 2)
        self.assertTrue(all(prescaled is solver.qd_work for prescaled in launches))

        drive_slot = solver._vel_limit_drive_slot_arg.numpy()
        scaled = 0
        for replay in range(3):
            # An episode reset between env steps, as a training loop does.
            set_fast_velocities()
            solver.reset(states[0])
            wp.capture_launch(capture.graph)
            for substep in range(2):
                source = inputs[substep].numpy()
                expected = _serial(_model_layout(model, source, drive_slot), solver._vel_limit_skip_driven, device)
                _assert_bits_equal(consumed[substep].numpy(), expected, f"replay={replay} substep={substep}")
                scaled += int(np.count_nonzero(expected != source))
        self.assertGreater(scaled, 0)
        self.assertTrue(np.isfinite(states[0].body_q.numpy()).all())

    def test_route_records_no_prescale_without_velocity_limits(self):
        """Record lane width 0 when no prescale kernel runs."""
        solver = SolverFeatherPGS(_solver_model("cpu"))
        self.assertEqual(solver.get_path_selection()["routes"]["velocity_limits"]["prescale_lanes"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
