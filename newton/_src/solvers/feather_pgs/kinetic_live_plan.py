# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Checked live-model ownership for the experimental compact Kuka operator.

Only immutable topology is downloaded here, once at admission. Dynamic force,
pose, contact and held-factor values are never constructor inputs.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from . import kuka_joint_world

PARENTS = np.array(
    [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 9, 15, 16, 17, 18, 9, 20, 21, 22, 23, 9, 25, 26, 27, 28, -1, -1],
    dtype=np.int32,
)
SCALAR_BODIES = np.array(
    [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28],
    dtype=np.int32,
)


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def validate_gravity(model, worlds):
    """Validate owned local gravity; the separate global-world tail is unused."""
    values = model.gravity
    _require(
        values is not None
        and values.is_contiguous
        and (values.shape == (worlds + 1,) or (worlds == 1 and values.shape == (1,))),
        "Kinetic owner requires the model's contiguous local/global gravity layout",
    )
    gravity = values.numpy()
    _require(np.isfinite(gravity[:worlds]).all(), "Kinetic owner requires finite owned-world gravity")


@dataclass
class LivePlan:
    """Static host proof and immutable device maps, allocated only once."""

    host: dict
    data: object
    base: object
    worlds: int
    arts: np.ndarray
    primary_offset: wp.array
    secondary_offset: wp.array
    drive_row_by_dof: wp.array
    drive_q_index_by_dof: wp.array

    def device_data(self, device=None):
        """Return the already-owned maps; moving them requires new admission."""
        if device is not None and wp.get_device(device) != self.data.body_ids.device:
            raise ValueError("Live plan cannot change device after admission")
        return self.data


def build_live_plan(solver):
    """Validate the actual model and retain its exact global coordinate indices."""
    from .kinetic_types import KineticPlan  # noqa: PLC0415 -- optional experimental factories stay lazy

    model, device = solver.model, solver.model.device
    base_host = kuka_joint_world.bind_plan(solver)
    host = {name: np.asarray(getattr(base_host, name)).copy() for name in base_host.__dataclass_fields__}
    worlds = int(solver.world_count)
    joints, bodies, dofs = host["joint_ids"], host["body_ids"], host["dof_ids"]
    _require(
        np.array_equal(host["body_parent"], np.broadcast_to(PARENTS, (worlds, 32))),
        "Kinetic owner requires the exact trunk/four-finger topology",
    )
    types = model.joint_type.numpy()[joints]
    expected_types = np.full(32, 3, np.int32)
    expected_types[SCALAR_BODIES], expected_types[30:] = 1, 4
    _require(
        np.array_equal(types, np.broadcast_to(expected_types, types.shape)),
        "Kinetic owner requires the original revolute/fixed/free joint types",
    )
    expected_dims = np.zeros((32, 2), np.int32)
    expected_dims[SCALAR_BODIES, 1], expected_dims[30:] = 1, (3, 3)
    _require(
        np.array_equal(model.joint_dof_dim.numpy()[joints], np.broadcast_to(expected_dims, (worlds, 32, 2))),
        "Kinetic owner requires complete global35 coordinates",
    )
    _require(
        np.array_equal(model.joint_qd_start.numpy()[joints[:, SCALAR_BODIES]], dofs[:, :23]),
        "Scalar ordering differs from the compact held operator",
    )
    local = np.full((worlds, 32), -1, np.int32)
    local[:, SCALAR_BODIES] = np.arange(23, dtype=np.int32)
    dof_joint = np.empty((worlds, 35), np.int32)
    dof_joint[:, :23], dof_joint[:, 23:29], dof_joint[:, 29:] = (
        joints[:, SCALAR_BODIES],
        joints[:, 30, None],
        joints[:, 31, None],
    )
    q_index = model.joint_q_start.numpy()[dof_joint[:, :23]].astype(np.int32)
    _require(len(np.unique(q_index)) == 23 * worlds, "Primary position ownership aliases")
    _require(
        np.array_equal(dofs[:, :29], solver.world_dof_indices.numpy()),
        "Original response29 ordering differs from live physical35",
    )
    _require(
        np.array_equal(model.body_world.numpy()[bodies], np.broadcast_to(np.arange(worlds)[:, None], bodies.shape)),
        "Kinetic ownership crosses worlds",
    )
    _require(
        np.array_equal(model.body_flags.numpy()[bodies], np.broadcast_to([1] * 31 + [2], bodies.shape)),
        "Kinetic owner requires dynamic primary/free and prescribed root bodies",
    )
    arts = solver.body_to_articulation.numpy()[bodies[:, [0, 30, 31]]]
    _require(
        np.array_equal(arts, np.arange(3 * worlds).reshape(worlds, 3)),
        "Current origin view requires world-major articulation ownership",
    )
    validate_gravity(model, worlds)
    _require(
        not np.any(solver._kinematic_dof_mask.numpy()[dofs[:, :29]]),
        "Response coordinates cannot include prescribed DOFs",
    )
    host.update(body_local_dof=local, dof_joint=dof_joint, q_index=q_index)
    existing = getattr(getattr(solver, "_joint_world", None), "plan", None)
    base = existing if existing is not None else base_host.device_data(device)
    data = KineticPlan()
    for name, values in host.items():
        setattr(data, name, getattr(base, name) if hasattr(base, name) else wp.array(values, dtype=int, device=device))
    offsets = solver.articulation_world_dof_offset.numpy()
    primary_offset = wp.array(offsets[arts[:, 0]], dtype=int, device=device)
    secondary_offset = wp.array(offsets[arts[:, 1]], dtype=int, device=device)
    _require(
        np.all(
            ((offsets[arts[:, 0]] == 0) & (offsets[arts[:, 1]] == 23))
            | ((offsets[arts[:, 0]] == 6) & (offsets[arts[:, 1]] == 0))
        ),
        "Current response offsets must preserve original primary/free order",
    )
    # This is the original augmented-drive topology, including disabled gains.
    # CPU original setup omits these maps because async drive construction is CUDA-only.
    drive_rows = np.full(model.joint_dof_count, -1, np.int32)
    drive_q = np.full(model.joint_dof_count, -1, np.int32)
    for world in range(worlds):
        drive_rows[dofs[world, :23]] = int(arts[world, 0]) * solver.articulation_max_dofs + np.arange(23)
        drive_q[dofs[world, :23]] = q_index[world]
    row_array = getattr(solver, "_augmented_drive_row_by_dof", None)
    q_array = getattr(solver, "_augmented_drive_q_index_by_dof", None)
    if row_array is None or not np.array_equal(row_array.numpy(), drive_rows):
        row_array = wp.array(drive_rows, dtype=int, device=device)
    if q_array is None or not np.array_equal(q_array.numpy(), drive_q):
        q_array = wp.array(drive_q, dtype=int, device=device)
    return LivePlan(host, data, base, worlds, arts, primary_offset, secondary_offset, row_array, q_array)


build_plan = build_live_plan
