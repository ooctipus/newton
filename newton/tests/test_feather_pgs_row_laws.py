# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The row laws against every row builder and pass that forms a row's right-hand side or candidate.

A replaced implementation is launched as the solver launches it, or compiled from
its text anchored in its module; a unified rounding or literal variant is bounded
in units of the terms and recorded in ``UNIFICATIONS``.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import kernels, row_laws
from newton.tests.feather_pgs_law_harness import (
    anchored,
    call,
    dedent_block,
    joint_type,
    mismatches,
    native_kernel,
    reference_kernel,
    register_device_tests,
    require_cuda,
    run_native,
    scalars,
    unified,
)

# ---------------------------------------------------------------------------------------------------------------------
# Row laws against the kernels they replace
# ---------------------------------------------------------------------------------------------------------------------


@wp.kernel
def _law_world_bias(
    phi: wp.array2d[float],
    beta: wp.array2d[float],
    kind: wp.array2d[int],
    target: wp.array2d[float],
    dt: float,
    bias_scale: float,
    contact_speculative_scale: float,
    limit_speculative_scale: float,
    contact_w: float,
    rhs: wp.array2d[float],
    row_w: wp.array2d[float],
):
    world, i = wp.tid()
    inv_dt = 1.0 / dt
    value, w = row_laws.row_rhs(
        kind[world, i],
        phi[world, i],
        beta[world, i],
        target[world, i],
        inv_dt,
        bias_scale,
        contact_speculative_scale,
        limit_speculative_scale,
        contact_w,
    )
    rhs[world, i] = value
    row_w[world, i] = w


def _rows(rng, worlds, rows):
    """Random dense rows of every kind, with the phi edge cases the bias branches test."""
    phi = scalars(rng, worlds * rows, 0.02).reshape(worlds, rows)
    kind = rng.integers(0, 8, (worlds, rows)).astype(np.int32)
    beta = rng.uniform(0.0, 1.0, (worlds, rows)).astype(np.float32)
    target = scalars(rng, worlds * rows).reshape(worlds, rows)
    return phi, kind, beta, target


def test_row_rhs_matches_world_contact_bias(test, device):
    """Equal ``kernels.compute_world_contact_bias`` bitwise for every row kind and bias setting."""
    rng = np.random.default_rng(11)
    worlds, rows = 16, 64
    phi, kind, beta, target = _rows(rng, worlds, rows)
    for dt, bias_scale, contact_spec, limit_spec, contact_w in (
        (1.0 / 60.0, 1.0, 1.0, 1.0, 1.0),
        (1.0 / 120.0, 0.0, 0.0, 1.0, 0.5),
        (7.0e-3, 0.7, 0.3, 0.2, 0.98),
    ):
        with test.subTest(dt=dt, bias_scale=bias_scale, contact_w=contact_w):
            out = call(
                kernels.compute_world_contact_bias,
                worlds * rows,
                device,
                world_constraint_count=np.full(worlds, rows, dtype=np.int32),
                max_constraints=rows,
                world_phi=phi,
                world_row_beta=beta,
                world_row_type=kind,
                world_target_velocity=target,
                dt=dt,
                bias_scale=bias_scale,
                contact_speculative_scale=contact_spec,
                joint_limit_speculative_scale=limit_spec,
                contact_w=contact_w,
                mf_constraint_count=np.zeros(worlds, dtype=np.int32),
                owned_rows_le=0,
                world_rhs=np.zeros((worlds, rows), dtype=np.float32),
                world_row_w=np.ones((worlds, rows), dtype=np.float32),
            )
            law = call(
                _law_world_bias,
                (worlds, rows),
                device,
                phi=phi,
                beta=beta,
                kind=kind,
                target=target,
                dt=dt,
                bias_scale=bias_scale,
                contact_speculative_scale=contact_spec,
                limit_speculative_scale=limit_spec,
                contact_w=contact_w,
                rhs=np.zeros((worlds, rows), dtype=np.float32),
                row_w=np.zeros((worlds, rows), dtype=np.float32),
            )
            test.assertFalse(mismatches(out["world_rhs"].numpy(), law["rhs"].numpy()).any())
            if contact_w < 1.0:
                test.assertFalse(mismatches(out["world_row_w"].numpy(), law["row_w"].numpy()).any())


@wp.kernel
def _law_restitution_fires(phi: wp.array[float], u: wp.array[float], dt: float, threshold: float, out: wp.array[int]):
    i = wp.tid()
    out[i] = wp.where(row_laws.contact_restitution_fires(phi[i], u[i], dt, threshold), 1, 0)


@wp.kernel
def _kernel_restitution_fires(
    phi: wp.array[float], u: wp.array[float], dt: float, threshold: float, out: wp.array[int]
):
    i = wp.tid()
    out[i] = wp.where(kernels.contact_restitution_fires(phi[i], u[i], dt, threshold), 1, 0)


def test_restitution_fires_matches_kernels(test, device):
    """Fire exactly where ``kernels.contact_restitution_fires`` fires, end-gap slop included."""
    rng = np.random.default_rng(12)
    n = 8192
    phi = scalars(rng, n, 0.01)
    u = scalars(rng, n, 1.0)
    u[::5] = -0.5  # at the threshold
    values = {"phi": phi, "u": u, "dt": 1.0 / 60.0, "threshold": 0.5, "out": np.zeros(n, dtype=np.int32)}
    law = call(_law_restitution_fires, n, device, **values)["out"].numpy()
    reference = call(_kernel_restitution_fires, n, device, **values)["out"].numpy()
    np.testing.assert_array_equal(law, reference)
    test.assertGreater(law.sum(), 0)


def _mf_layout(rng, worlds, rows):
    """Matrix-free rows of contacts between free bodies: each world owns two six-coordinate bodies."""
    bodies = 2 * worlds
    body_a = np.full((worlds, rows), -1, dtype=np.int32)
    body_b = np.full((worlds, rows), -1, dtype=np.int32)
    for w in range(worlds):
        body_a[w] = np.where(rng.uniform(size=rows) < 0.85, 2 * w, -1)
        body_b[w] = np.where(rng.uniform(size=rows) < 0.5, 2 * w + 1, -1)
    return {
        "mf_body_a": body_a,
        "mf_body_b": body_b,
        "mf_J_a": rng.normal(size=(worlds, rows, 6)).astype(np.float32),
        "mf_J_b": rng.normal(size=(worlds, rows, 6)).astype(np.float32),
        "body_to_articulation": np.arange(bodies, dtype=np.int32),
        "articulation_dof_start": (6 * np.arange(bodies)).astype(np.int32),
        "incident_velocity": rng.normal(size=6 * bodies).astype(np.float32),
        "rigid_body_max_depenetration_velocity": rng.choice(
            np.array([0.5, 2.0, 0.0, np.inf, 1.0e30, 0.05], dtype=np.float32), bodies
        ),
    }


@wp.kernel
def _law_mf_position_rhs(
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_phi: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_target_velocity: wp.array2d[float],
    mf_row_restitution: wp.array2d[float],
    has_target_velocity: int,
    body_to_articulation: wp.array[int],
    articulation_dof_start: wp.array[int],
    incident_velocity: wp.array[float],
    rigid_body_max_depenetration_velocity: wp.array[float],
    pgs_beta: float,
    contact_w: float,
    dt: float,
    contact_speculative_scale: float,
    restitution_velocity_threshold: float,
    rhs: wp.array2d[float],
    row_w: wp.array2d[float],
):
    world, i = wp.tid()
    ba = mf_body_a[world, i]
    bb = mf_body_b[world, i]
    target = float(0.0)
    if has_target_velocity != 0:
        target = mf_target_velocity[world, i]
    phi = mf_phi[world, i]
    kind = mf_row_type[world, i]
    inv_dt = 1.0 / dt
    value = float(0.0)
    w = float(1.0)
    if kind == row_laws.ROW_CONTACT:
        cap = row_laws.contact_depenetration_cap(ba, bb, rigid_body_max_depenetration_velocity)
        value, w = row_laws.contact_rhs(
            phi, pgs_beta, target, inv_dt, 1.0, contact_speculative_scale, contact_w, cap, 0.0
        )
        restitution = mf_row_restitution[world, i]
        if restitution > 0.0:
            relative_incident = float(0.0)
            if ba >= 0:
                dof_start = articulation_dof_start[body_to_articulation[ba]]
                for k in range(6):
                    relative_incident += mf_J_a[world, i, k] * incident_velocity[dof_start + k]
            if bb >= 0:
                dof_start = articulation_dof_start[body_to_articulation[bb]]
                for k in range(6):
                    relative_incident += mf_J_b[world, i, k] * incident_velocity[dof_start + k]
            relative_incident -= target
            fires, bounce = row_laws.contact_restitution_rhs(
                phi, relative_incident, target, restitution, dt, restitution_velocity_threshold
            )
            if fires:
                value = bounce
                w = 1.0
    elif kind == row_laws.ROW_FRICTION:
        # Matrix-free friction rows carry beta * phi in phi.
        value, w = row_laws.row_rhs(kind, phi, 1.0, target, inv_dt, 1.0, 0.0, 0.0, 1.0)
    else:
        value = phi - target  # a free body's speed row carries its limit in phi
    rhs[world, i] = value
    row_w[world, i] = w


def test_contact_rhs_matches_mf_effective_mass_rhs(test, device):
    """Equal ``compute_mf_effective_mass_and_rhs`` up to its ``x / dt`` rounding, with the same caps and weights.

    Unified variant: the matrix-free bias divides by ``dt`` where the canonical law multiplies by ``1 / dt``
    (and penetration starts at ``phi <= 0``, where it was ``phi < 0``, which only flips the sign of a zero).
    """
    rng = np.random.default_rng(13)
    worlds, rows = 8, 48
    layout = _mf_layout(rng, worlds, rows)
    phi = scalars(rng, worlds * rows, 0.02).reshape(worlds, rows)
    kind = rng.choice(np.array([0, 0, 0, 2, 4], dtype=np.int32), (worlds, rows))
    target = scalars(rng, worlds * rows).reshape(worlds, rows)
    restitution = rng.choice(np.array([0.0, 0.0, 0.5, 1.0], dtype=np.float32), (worlds, rows))
    shared = {
        **layout,
        "mf_phi": phi,
        "mf_row_type": kind,
        "mf_target_velocity": target,
        "mf_row_restitution": restitution,
        "has_target_velocity": 1,
        "pgs_beta": 0.2,
        "contact_w": 0.8,
        "dt": 1.0 / 60.0,
        "contact_speculative_scale": 0.9,
        "restitution_velocity_threshold": 0.2,
    }
    hinv = np.tile(np.eye(6, dtype=np.float32), (2 * worlds, 1, 1))
    out = call(
        kernels.compute_mf_effective_mass_and_rhs,
        worlds * rows,
        device,
        mf_constraint_count=np.full(worlds, rows, dtype=np.int32),
        mf_body_Hinv=hinv,
        pgs_cfm=1.0e-6,
        mf_max_constraints=rows,
        mf_eff_mass_inv=np.zeros((worlds, rows), dtype=np.float32),
        mf_MiJt_a=np.zeros((worlds, rows, 6), dtype=np.float32),
        mf_MiJt_b=np.zeros((worlds, rows, 6), dtype=np.float32),
        mf_rhs=np.zeros((worlds, rows), dtype=np.float32),
        mf_row_w=np.ones((worlds, rows), dtype=np.float32),
        **shared,
    )
    law = call(
        _law_mf_position_rhs,
        (worlds, rows),
        device,
        rhs=np.zeros((worlds, rows), dtype=np.float32),
        row_w=np.zeros((worlds, rows), dtype=np.float32),
        **shared,
    )
    reference_rhs, law_rhs = out["mf_rhs"].numpy(), law["rhs"].numpy()
    np.testing.assert_array_equal(out["mf_row_w"].numpy(), law["row_w"].numpy())
    magnitude = np.abs(phi) * 60.0 + np.abs(target)
    test.assertLessEqual(
        unified(
            "mf rhs (compute_mf_effective_mass_and_rhs)", reference_rhs.ravel(), law_rhs.ravel(), magnitude.ravel(), 2.0
        ),
        2.0,
    )
    # Rows whose bias the variant does not touch (speed rows, restitution) stay bitwise.
    untouched = (kind == 4).ravel()
    test.assertFalse(mismatches(reference_rhs.ravel()[untouched, None], law_rhs.ravel()[untouched, None]).any())


@wp.kernel
def _law_mf_pass_rhs(
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_dof_a: wp.array2d[int],
    mf_dof_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    world_dof_indices: wp.array2d[int],
    mf_phi: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_target_velocity: wp.array2d[float],
    mf_row_restitution: wp.array2d[float],
    rigid_body_max_depenetration_velocity: wp.array[float],
    pgs_beta: float,
    dt: float,
    bias_scale: float,
    speculative_scale: float,
    position_velocity: wp.array[float],
    incident_velocity: wp.array[float],
    preserve_unreached_speculative: int,
    apply_restitution: int,
    restitution_velocity_threshold: float,
    rhs: wp.array2d[float],
):
    world, i = wp.tid()
    target = mf_target_velocity[world, i]
    phi = mf_phi[world, i]
    kind = mf_row_type[world, i]
    inv_dt = 1.0 / dt
    value = float(0.0)
    if kind == row_laws.ROW_CONTACT:
        end_gap = float(0.0)
        if preserve_unreached_speculative != 0 and phi > 0.0:
            jv = kernels.mf_contact_row_dot(
                mf_J_a, mf_J_b, mf_dof_a[world, i], mf_dof_b[world, i], world_dof_indices, position_velocity, world, i
            )
            end_gap = phi + dt * (jv - target)
        cap = row_laws.contact_depenetration_cap(
            mf_body_a[world, i], mf_body_b[world, i], rigid_body_max_depenetration_velocity
        )
        value, _w = row_laws.contact_rhs(
            phi, pgs_beta, target, inv_dt, bias_scale, speculative_scale, 1.0, cap, end_gap
        )
        restitution = float(0.0)
        if apply_restitution != 0:
            restitution = mf_row_restitution[world, i]
        if restitution > 0.0:
            relative_incident = kernels.mf_contact_row_dot(
                mf_J_a, mf_J_b, mf_dof_a[world, i], mf_dof_b[world, i], world_dof_indices, incident_velocity, world, i
            )
            relative_incident -= target
            fires, bounce = row_laws.contact_restitution_rhs(
                phi, relative_incident, target, restitution, dt, restitution_velocity_threshold
            )
            if fires:
                value = bounce
    elif kind == row_laws.ROW_FRICTION:
        value, _w = row_laws.row_rhs(kind, phi, 1.0, target, inv_dt, bias_scale, 0.0, 0.0, 1.0)
    else:
        value = phi - target
    rhs[world, i] = value


def test_contact_rhs_matches_mf_rhs_bias(test, device):
    """Equal ``compute_mf_rhs_bias`` in its position and velocity passes up to its ``x / dt`` rounding.

    The velocity pass keeps a separated row's speculative bias while the position
    solve's linearized end gap is open; the law reads that gap as ``end_gap``.
    """
    rng = np.random.default_rng(14)
    worlds, rows = 8, 48
    layout = _mf_layout(rng, worlds, rows)
    phi = scalars(rng, worlds * rows, 0.02).reshape(worlds, rows)
    kind = rng.choice(np.array([0, 0, 0, 2, 4], dtype=np.int32), (worlds, rows))
    target = scalars(rng, worlds * rows, 0.1).reshape(worlds, rows)
    dof_a = np.where(layout["mf_body_a"] >= 0, 0, -1).astype(np.int32)
    dof_b = np.where(layout["mf_body_b"] >= 0, 6, -1).astype(np.int32)
    indices = (12 * np.arange(worlds)[:, None] + np.arange(12)[None, :]).astype(np.int32)
    common = {
        "mf_body_a": layout["mf_body_a"],
        "mf_body_b": layout["mf_body_b"],
        "mf_dof_a": dof_a,
        "mf_dof_b": dof_b,
        "mf_J_a": layout["mf_J_a"],
        "mf_J_b": layout["mf_J_b"],
        "world_dof_indices": indices,
        "mf_phi": phi,
        "mf_row_type": kind,
        "mf_target_velocity": target,
        "mf_row_restitution": rng.choice(np.array([0.0, 0.6], dtype=np.float32), (worlds, rows)),
        "rigid_body_max_depenetration_velocity": layout["rigid_body_max_depenetration_velocity"],
        "pgs_beta": 0.3,
        "dt": 1.0 / 60.0,
        "position_velocity": rng.normal(size=12 * worlds).astype(np.float32),
        "incident_velocity": rng.normal(size=12 * worlds).astype(np.float32),
        "restitution_velocity_threshold": 0.1,
    }
    incident = np.abs(common["incident_velocity"]).reshape(worlds, 12)
    incident_terms = np.where(dof_a >= 0, (np.abs(layout["mf_J_a"]) * incident[:, None, :6]).sum(axis=2), 0.0)
    incident_terms += np.where(dof_b >= 0, (np.abs(layout["mf_J_b"]) * incident[:, None, 6:]).sum(axis=2), 0.0)
    for bias_scale, speculative, preserve, restitution in ((1.0, 0.8, 0, 1), (0.0, 0.0, 1, 1), (0.5, 1.0, 1, 0)):
        with test.subTest(bias_scale=bias_scale, preserve=preserve, restitution=restitution):
            settings = {
                "bias_scale": bias_scale,
                "speculative_scale": speculative,
                "preserve_unreached_speculative": preserve,
                "apply_restitution": restitution,
            }
            out = call(
                kernels.compute_mf_rhs_bias,
                worlds * rows,
                device,
                mf_constraint_count=np.full(worlds, rows, dtype=np.int32),
                has_target_velocity=1,
                mf_max_constraints=rows,
                mf_rhs=np.zeros((worlds, rows), dtype=np.float32),
                **common,
                **settings,
            )
            law = call(
                _law_mf_pass_rhs,
                (worlds, rows),
                device,
                rhs=np.zeros((worlds, rows), dtype=np.float32),
                **common,
                **settings,
            )
            # A restitution row's target is -target + e (J v - target): its terms include the row's J v products.
            magnitude = np.abs(phi) * 60.0 + 2.0 * np.abs(target) + incident_terms
            worst = unified(
                f"mf rhs passes (compute_mf_rhs_bias, bias_scale {bias_scale}, preserve {preserve})",
                out["mf_rhs"].numpy().ravel(),
                law["rhs"].numpy().ravel(),
                magnitude.ravel(),
                2.0,
            )
            test.assertLessEqual(worst, 2.0)


def _dense_layout(rng, worlds, rows, dofs):
    return {
        "world_dof_count": np.full(worlds, dofs, dtype=np.int32),
        "world_dof_indices": (dofs * np.arange(worlds)[:, None] + np.arange(dofs)[None, :]).astype(np.int32),
        "world_J": rng.normal(size=(worlds, rows, dofs)).astype(np.float32),
    }


@wp.kernel
def _law_velocity_pass(
    world_dof_count: wp.array[int],
    world_dof_indices: wp.array2d[int],
    world_J: wp.array3d[float],
    world_phi: wp.array2d[float],
    world_row_type: wp.array2d[int],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
    world_position_velocity: wp.array[float],
    world_incident_velocity: wp.array[float],
    dt: float,
    apply_restitution: int,
    restitution_velocity_threshold: float,
    rhs: wp.array2d[float],
):
    world, i = wp.tid()
    inv_dt = 1.0 / dt
    phi = world_phi[world, i]
    kind = world_row_type[world, i]
    target = world_target_velocity[world, i]
    value = -target
    if kind == row_laws.ROW_CONTACT:
        fires = bool(False)
        bounce = float(0.0)
        restitution = float(0.0)
        if apply_restitution != 0:
            restitution = world_row_restitution[world, i]
        if restitution > 0.0:
            relative_incident = (
                kernels.world_contact_row_dot(
                    world_dof_count, world_dof_indices, world_J, world_incident_velocity, world, i
                )
                - target
            )
            fires, bounce = row_laws.contact_restitution_rhs(
                phi, relative_incident, target, restitution, dt, restitution_velocity_threshold
            )
        if fires:
            value = bounce
        else:
            end_gap = float(0.0)
            if phi > 0.0:
                jv = kernels.world_contact_row_dot(
                    world_dof_count, world_dof_indices, world_J, world_position_velocity, world, i
                )
                end_gap = phi + dt * (jv - target)
            value, _w = row_laws.contact_rhs(phi, 0.0, target, inv_dt, 0.0, 0.0, 1.0, 0.0, end_gap)
    elif kind == row_laws.ROW_JOINT_LIMIT:
        value = row_laws.limit_rhs(phi, 0.0, target, inv_dt, 0.0, 1.0)
    elif kind == row_laws.ROW_JOINT_TARGET:
        value = 0.0
    rhs[world, i] = value


def test_velocity_pass_matches_world_contact_velocity_bias(test, device):
    """Equal ``compute_world_contact_velocity_bias`` (the dense velocity pass), apart from signs of zero.

    The pass is the contact law with no Baumgarte and no speculative scale, keeping
    a separated row's bias while its end gap is open, and the limit law with no
    Baumgarte; a closed row's ``-target + 0`` may turn a ``-0`` target into ``+0``.
    """
    rng = np.random.default_rng(15)
    worlds, rows, dofs = 8, 32, 12
    phi, kind, _, target = _rows(rng, worlds, rows)
    common = {
        **_dense_layout(rng, worlds, rows, dofs),
        "world_phi": phi,
        "world_row_type": kind,
        "world_target_velocity": target,
        "world_row_restitution": rng.choice(np.array([0.0, 0.5], dtype=np.float32), (worlds, rows)),
        "world_position_velocity": rng.normal(scale=0.1, size=dofs * worlds).astype(np.float32),
        "world_incident_velocity": rng.normal(size=dofs * worlds).astype(np.float32),
        "dt": 1.0 / 60.0,
        "apply_restitution": 1,
        "restitution_velocity_threshold": 0.2,
    }
    out = call(
        kernels.compute_world_contact_velocity_bias,
        worlds * rows,
        device,
        world_constraint_count=np.full(worlds, rows, dtype=np.int32),
        max_constraints=rows,
        world_rhs=np.zeros((worlds, rows), dtype=np.float32),
        **common,
    )
    law = call(_law_velocity_pass, (worlds, rows), device, rhs=np.zeros((worlds, rows), dtype=np.float32), **common)
    np.testing.assert_array_equal(out["world_rhs"].numpy(), law["rhs"].numpy())
    unified(
        "dense velocity pass (compute_world_contact_velocity_bias), signs of zero",
        out["world_rhs"].numpy().ravel(),
        law["rhs"].numpy().ravel(),
        np.abs(phi.ravel()) * 60.0 + np.abs(target.ravel()),
        0.0,
    )


@wp.kernel
def _law_restitution_matrix_free(
    world_dof_count: wp.array[int],
    world_dof_indices: wp.array2d[int],
    world_J: wp.array3d[float],
    world_phi: wp.array2d[float],
    world_row_type: wp.array2d[int],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
    world_incident_velocity: wp.array[float],
    dt: float,
    restitution_velocity_threshold: float,
    rhs: wp.array2d[float],
    row_w: wp.array2d[float],
):
    world, i = wp.tid()
    if world_row_type[world, i] != row_laws.ROW_CONTACT:
        return
    target = world_target_velocity[world, i]
    relative_incident = (
        kernels.world_contact_row_dot(world_dof_count, world_dof_indices, world_J, world_incident_velocity, world, i)
        - target
    )
    fires, bounce = row_laws.contact_restitution_rhs(
        world_phi[world, i],
        relative_incident,
        target,
        world_row_restitution[world, i],
        dt,
        restitution_velocity_threshold,
    )
    if fires:
        rhs[world, i] = bounce
        row_w[world, i] = 1.0


def test_restitution_matches_matrix_free_restitution(test, device):
    """Replace exactly the rows ``apply_world_contact_restitution_matrix_free`` replaces, with the same rhs bits."""
    rng = np.random.default_rng(16)
    worlds, rows, dofs = 8, 32, 12
    phi, kind, _, target = _rows(rng, worlds, rows)
    rhs0 = rng.normal(size=(worlds, rows)).astype(np.float32)
    common = {
        **_dense_layout(rng, worlds, rows, dofs),
        "world_phi": phi,
        "world_row_type": kind,
        "world_target_velocity": target,
        "world_row_restitution": rng.choice(np.array([0.0, 0.5, 1.0], dtype=np.float32), (worlds, rows)),
        "world_incident_velocity": rng.normal(size=dofs * worlds).astype(np.float32),
        "dt": 1.0 / 60.0,
        "restitution_velocity_threshold": 0.2,
    }
    out = call(
        kernels.apply_world_contact_restitution_matrix_free,
        worlds * rows,
        device,
        world_constraint_count=np.full(worlds, rows, dtype=np.int32),
        max_constraints=rows,
        write_row_w=1,
        mf_constraint_count=np.zeros(worlds, dtype=np.int32),
        owned_rows_le=0,
        world_rhs=rhs0.copy(),
        world_row_w=np.full((worlds, rows), 0.5, dtype=np.float32),
        **common,
    )
    law = call(
        _law_restitution_matrix_free,
        (worlds, rows),
        device,
        rhs=rhs0.copy(),
        row_w=np.full((worlds, rows), 0.5, dtype=np.float32),
        **common,
    )
    test.assertFalse(mismatches(out["world_rhs"].numpy(), law["rhs"].numpy()).any())
    test.assertFalse(mismatches(out["world_row_w"].numpy(), law["row_w"].numpy()).any())
    test.assertTrue((out["world_row_w"].numpy() == 1.0).any())


@wp.kernel
def _law_restitution_accumulated(
    world_phi: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_type: wp.array2d[int],
    world_row_restitution: wp.array2d[float],
    dt: float,
    contact_speculative_scale: float,
    restitution_velocity_threshold: float,
    rhs: wp.array2d[float],
):
    world, i = wp.tid()
    restitution = world_row_restitution[world, i]
    if world_row_type[world, i] != row_laws.ROW_CONTACT or restitution <= 0.0:
        return
    phi = world_phi[world, i]
    # The geometric bias the position pass added: the contact law with no target.
    bias, _w = row_laws.contact_rhs(
        phi, world_row_beta[world, i], 0.0, 1.0 / dt, 1.0, contact_speculative_scale, 1.0, 0.0, 0.0
    )
    relative_incident = rhs[world, i] - bias
    if row_laws.contact_restitution_fires(phi, relative_incident, dt, restitution_velocity_threshold):
        rhs[world, i] = (1.0 + restitution) * relative_incident


def test_restitution_matches_accumulated_restitution(test, device):
    """Equal ``apply_world_contact_restitution_accumulated`` up to its geometric bias's ``x / dt`` rounding.

    Unified variant: the split route recovers the incident velocity by removing a
    bias it recomputes as ``beta * phi / dt``; the law removes the bias the position
    pass added, ``bias_scale * beta * phi * (1 / dt)``.
    """
    rng = np.random.default_rng(17)
    worlds, rows = 8, 64
    phi, kind, beta, _ = _rows(rng, worlds, rows)
    rhs0 = rng.normal(size=(worlds, rows)).astype(np.float32)
    common = {
        "world_phi": phi,
        "world_row_beta": beta,
        "world_row_type": kind,
        "world_row_restitution": rng.choice(np.array([0.0, 0.5, 1.0], dtype=np.float32), (worlds, rows)),
        "dt": 1.0 / 60.0,
        "contact_speculative_scale": 0.9,
        "restitution_velocity_threshold": 0.2,
    }
    out = call(
        kernels.apply_world_contact_restitution_accumulated,
        worlds * rows,
        device,
        world_constraint_count=np.full(worlds, rows, dtype=np.int32),
        max_constraints=rows,
        write_row_w=0,
        world_rhs=rhs0.copy(),
        world_row_w=np.ones((worlds, rows), dtype=np.float32),
        **common,
    )
    law = call(_law_restitution_accumulated, (worlds, rows), device, rhs=rhs0.copy(), **common)
    magnitude = 2.0 * (np.abs(rhs0) + np.abs(phi) * 60.0)
    worst = unified(
        "split restitution (apply_world_contact_restitution_accumulated)",
        out["world_rhs"].numpy().ravel(),
        law["rhs"].numpy().ravel(),
        magnitude.ravel(),
        4.0,
    )
    test.assertLessEqual(worst, 4.0)


@wp.kernel
def _law_fused_limits(
    world_dof_indices: wp.array2d[int],
    max_world_dofs: int,
    fused_limit_dof_mask: wp.array[int],
    limit_q_index: wp.array[int],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_q: wp.array[float],
    activation_gap: float,
    pgs_beta: float,
    dt: float,
    active_sides: wp.array2d[int],
    lower_rhs: wp.array2d[float],
    upper_rhs: wp.array2d[float],
):
    element = wp.tid()
    world = element // max_world_dofs
    local_dof = element - world * max_world_dofs
    global_dof = world_dof_indices[world, local_dof]
    active = int(0)
    rhs = wp.vec2(0.0)
    if global_dof >= 0 and fused_limit_dof_mask[global_dof] != 0:
        q_index = limit_q_index[global_dof]
        if q_index >= 0:
            for side in range(2):
                on, phi, _sign = row_laws.joint_limit_side(
                    joint_q[q_index], joint_limit_lower[global_dof], joint_limit_upper[global_dof], side, activation_gap
                )
                if on:
                    rhs[side] = row_laws.limit_rhs(phi, pgs_beta, 0.0, 1.0 / dt, 1.0, 1.0)
                    active |= 1 << side
    active_sides[world, local_dof] = active
    lower_rhs[world, local_dof] = rhs[0]
    upper_rhs[world, local_dof] = rhs[1]


def test_limit_laws_match_fused_diagonal_limits(test, device):
    """Equal ``prepare_fused_diagonal_joint_limits``: the same active sides and bitwise rhs."""
    rng = np.random.default_rng(18)
    worlds, dofs = 16, 8
    count = worlds * dofs
    lower = scalars(rng, count, 0.5)
    upper = lower + np.abs(scalars(rng, count, 0.5))
    lower[::7] = -np.inf
    upper[::9] = np.inf
    values = {
        "world_dof_indices": np.arange(count, dtype=np.int32).reshape(worlds, dofs),
        "max_world_dofs": dofs,
        "fused_limit_dof_mask": (rng.uniform(size=count) < 0.9).astype(np.int32),
        "limit_q_index": np.where(rng.uniform(size=count) < 0.95, np.arange(count), -1).astype(np.int32),
        "joint_limit_lower": lower,
        "joint_limit_upper": upper,
        "joint_q": (lower + rng.normal(scale=0.3, size=count)).astype(np.float32),
        "activation_gap": 0.05,
        "pgs_beta": 0.25,
        "dt": 1.0 / 60.0,
        "active_sides": np.zeros((worlds, dofs), dtype=np.int32),
        "lower_rhs": np.zeros((worlds, dofs), dtype=np.float32),
        "upper_rhs": np.zeros((worlds, dofs), dtype=np.float32),
    }
    out = call(kernels.prepare_fused_diagonal_joint_limits, count, device, **values)
    law = call(_law_fused_limits, count, device, **values)
    for name in ("active_sides", "lower_rhs", "upper_rhs"):
        test.assertFalse(mismatches(out[name].numpy(), law[name].numpy()).any(), name)
    test.assertTrue((out["active_sides"].numpy() != 0).any())


@wp.kernel
def _law_mimic_phi(
    mimic_q0: wp.array[int],
    mimic_q1: wp.array[int],
    mimic_coef0: wp.array[float],
    mimic_coef1: wp.array[float],
    joint_q: wp.array[float],
    phi: wp.array[float],
):
    k = wp.tid()
    phi[k] = row_laws.mimic_phi(joint_q[mimic_q0[k]], joint_q[mimic_q1[k]], mimic_coef0[k], mimic_coef1[k])


def test_mimic_phi_matches_mimic_rows(test, device):
    """Equal the violation ``populate_mimic_J_for_size`` writes for every mimic row."""
    rng = np.random.default_rng(19)
    arts, per_art, dofs_per_art = 8, 3, 6
    count = arts * per_art
    dof0 = np.array([a * dofs_per_art + (k % dofs_per_art) for a in range(arts) for k in range(per_art)], np.int32)
    dof1 = np.array(
        [a * dofs_per_art + ((k + 2) % dofs_per_art) for a in range(arts) for k in range(per_art)], np.int32
    )
    coef0 = scalars(rng, count, 0.2)
    coef1 = scalars(rng, count, 1.5)
    joint_q = scalars(rng, arts * dofs_per_art, 1.0)
    out = call(
        kernels.populate_mimic_J_for_size,
        arts,
        device,
        articulation_dof_start=(dofs_per_art * np.arange(arts)).astype(np.int32),
        art_to_world=np.arange(arts, dtype=np.int32),
        group_to_art=np.arange(arts, dtype=np.int32),
        mimic_slot=np.tile(np.arange(per_art, dtype=np.int32), arts),
        mimic_art_start=(per_art * np.arange(arts + 1)).astype(np.int32),
        mimic_art_list=np.arange(count, dtype=np.int32),
        mimic_dof0=dof0,
        mimic_dof1=dof1,
        mimic_q0=dof0,
        mimic_q1=dof1,
        mimic_coef0=coef0,
        mimic_coef1=coef1,
        joint_q=joint_q,
        pgs_beta=0.2,
        pgs_cfm=0.0,
        J_group=np.zeros((arts, per_art, dofs_per_art), dtype=np.float32),
        world_row_type=np.zeros((arts, per_art), dtype=np.int32),
        world_row_parent=np.zeros((arts, per_art), dtype=np.int32),
        world_row_mu=np.zeros((arts, per_art), dtype=np.float32),
        world_row_beta=np.zeros((arts, per_art), dtype=np.float32),
        world_row_cfm=np.zeros((arts, per_art), dtype=np.float32),
        world_phi=np.zeros((arts, per_art), dtype=np.float32),
        world_target_velocity=np.zeros((arts, per_art), dtype=np.float32),
    )
    law = call(
        _law_mimic_phi,
        count,
        device,
        mimic_q0=dof0,
        mimic_q1=dof1,
        mimic_coef0=coef0,
        mimic_coef1=coef1,
        joint_q=joint_q,
        phi=np.zeros(count, dtype=np.float32),
    )
    test.assertFalse(mismatches(out["world_phi"].numpy().reshape(-1), law["phi"].numpy()).any())


@wp.kernel
def _law_joint_limit_candidates(
    joint_q: wp.array[float],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    activation_gap: float,
    dofs: int,
    phi: wp.array2d[float],
    count: wp.array[int],
):
    art = wp.tid()
    rows = int(0)
    for d in range(dofs):
        dof = art * dofs + d
        for side in range(2):
            on, value, _sign = row_laws.joint_limit_side(
                joint_q[dof], joint_limit_lower[dof], joint_limit_upper[dof], side, activation_gap
            )
            if on:
                phi[art, rows] = value
                rows += 1
    count[art] = rows


def test_joint_limit_side_matches_limit_rows(test, device):
    """Activate the sides ``build_joint_limit_rows_for_size`` builds rows for, with bitwise phi, in row order."""
    rng = np.random.default_rng(20)
    arts, dofs = 8, 6
    count = arts * dofs
    lower = scalars(rng, count, 0.5)
    upper = lower + np.abs(scalars(rng, count, 0.5))
    lower[::7] = -np.inf
    upper[::9] = np.inf
    joint_q = (lower + rng.normal(scale=0.2, size=count)).astype(np.float32)
    joint_q[~np.isfinite(joint_q)] = 0.0
    rows = 2 * dofs
    revolute = int(joint_type("REVOLUTE"))
    out = call(
        kernels.build_joint_limit_rows_for_size,
        arts,
        device,
        articulation_start=(dofs * np.arange(arts + 1)).astype(np.int32),
        articulation_dof_start=(dofs * np.arange(arts)).astype(np.int32),
        joint_type=np.full(count, revolute, dtype=np.int32),
        joint_q_start=np.arange(count + 1, dtype=np.int32),
        joint_qd_start=np.arange(count + 1, dtype=np.int32),
        joint_dof_dim=np.tile(np.array([[0, 1]], dtype=np.int32), (count, 1)),
        joint_limit_lower=lower,
        joint_limit_upper=upper,
        joint_q=joint_q,
        joint_limit_activation_gap=0.05,
        art_to_world=np.arange(arts, dtype=np.int32),
        group_to_art=np.arange(arts, dtype=np.int32),
        max_constraints=rows,
        pgs_beta=0.2,
        pgs_cfm=0.0,
        resolved_worlds=np.zeros(0, dtype=np.int32),
        world_slot_counter=np.zeros(arts, dtype=np.int32),
        first_rejected_slot=np.full(arts, np.iinfo(np.int32).max, dtype=np.int32),
        J_group=np.zeros((arts, rows, dofs), dtype=np.float32),
        world_row_type=np.zeros((arts, rows), dtype=np.int32),
        world_row_parent=np.zeros((arts, rows), dtype=np.int32),
        world_row_mu=np.zeros((arts, rows), dtype=np.float32),
        world_row_beta=np.zeros((arts, rows), dtype=np.float32),
        world_row_cfm=np.zeros((arts, rows), dtype=np.float32),
        world_phi=np.zeros((arts, rows), dtype=np.float32),
        world_target_velocity=np.zeros((arts, rows), dtype=np.float32),
    )
    law = call(
        _law_joint_limit_candidates,
        arts,
        device,
        joint_q=joint_q,
        joint_limit_lower=lower,
        joint_limit_upper=upper,
        activation_gap=0.05,
        dofs=dofs,
        phi=np.zeros((arts, rows), dtype=np.float32),
        count=np.zeros(arts, dtype=np.int32),
    )
    np.testing.assert_array_equal(out["world_slot_counter"].numpy(), law["count"].numpy())
    test.assertFalse(mismatches(out["world_phi"].numpy(), law["phi"].numpy()).any())


@wp.kernel
def _law_velocity_limit_active(
    joint_velocity_limit: wp.array[float], joint_qd: wp.array[float], fraction: float, active: wp.array[int]
):
    dof = wp.tid()
    active[dof] = wp.where(row_laws.velocity_limit_active(joint_velocity_limit[dof], joint_qd[dof], fraction), 1, 0)


def test_velocity_limit_active_matches_velocity_limit_slots(test, device):
    """Give rows to the coordinates ``allocate_joint_velocity_limit_slots`` gives rows to."""
    rng = np.random.default_rng(21)
    arts, dofs = 8, 6
    count = arts * dofs
    limit = scalars(rng, count, 2.0)
    qd = scalars(rng, count, 2.0)
    for fraction in (0.0, 0.8):
        with test.subTest(fraction=fraction):
            out = call(
                kernels.allocate_joint_velocity_limit_slots,
                arts,
                device,
                articulation_start=(dofs * np.arange(arts + 1)).astype(np.int32),
                articulation_dof_start=(dofs * np.arange(arts)).astype(np.int32),
                articulation_H_rows=np.full(arts, dofs, dtype=np.int32),
                joint_type=np.full(count, int(joint_type("REVOLUTE")), dtype=np.int32),
                joint_qd_start=np.arange(count + 1, dtype=np.int32),
                joint_dof_dim=np.tile(np.array([[0, 1]], dtype=np.int32), (count, 1)),
                joint_velocity_limit=limit,
                joint_qd=qd,
                velocity_limit_activation_fraction=fraction,
                drive_slot=np.full(1, -1, dtype=np.int32),
                skip_driven=0,
                art_to_world=np.arange(arts, dtype=np.int32),
                max_constraints=4 * dofs,
                velocity_limit_slot=np.full(2 * count, -1, dtype=np.int32),
                velocity_limit_sign=np.zeros(2 * count, dtype=np.float32),
                world_slot_counter=np.zeros(arts, dtype=np.int32),
                first_rejected_slot=np.full(arts, np.iinfo(np.int32).max, dtype=np.int32),
            )
            law = call(
                _law_velocity_limit_active,
                count,
                device,
                joint_velocity_limit=limit,
                joint_qd=qd,
                fraction=fraction,
                active=np.zeros(count, dtype=np.int32),
            )
            rows = out["velocity_limit_slot"].numpy().reshape(count, 2)[:, 0] >= 0
            np.testing.assert_array_equal(rows.astype(np.int32), law["active"].numpy())


# The contact and limit rhs of the Warp row producers that the laws unify, as each writes it.
_ROW_PACKETS_RHS = """
                    if phi <= 0.0:
                        rhs += data.bias_scale * beta * phi / data.dt
                        weight = data.contact_w
                    else:
                        rhs += data.speculative_scale * phi / data.dt
"""
_SPARSE_CONTACT_RHS = """
                    inv_dt = 1.0 / dt
                    bias = beta * phi * inv_dt
                    if lane == 0 and phi > 0.0:
                        bias = rows.speculative_scale * phi * inv_dt
"""
_SPARSE_LIMIT_RHS = """
        inv_dt = 1.0 / dt
        bias = phi * inv_dt
        if phi < 0.0 or rows.type[world, row] == PGS_CONSTRAINT_TYPE_MIMIC:
            bias = rows.beta_row[world, row] * phi * inv_dt
"""
_RESOLVED_RESIDUAL = """
    bias = data.speculative_scale * phi / data.dt
    if phi < 0.0:
        bias = data.beta * phi / data.dt
        if mf:
            cap = float(1.0e20)
            if responds_a:
                cap = data.max_depenetration_velocity[body_a]
            if responds_b:
                cap_b = data.max_depenetration_velocity[body_b]
                if cap_b > 0.0 and wp.isfinite(cap_b) and cap_b < cap:
                    cap = cap_b
            if cap > 0.0 and wp.isfinite(cap):
                bias = wp.max(bias, -cap)
"""


def test_contact_rhs_unifies_warp_row_producers(test, device):
    """Unify the row producers' rhs variants with the canonical laws, within 2 units of the bias terms.

    * ``row_packets.produce_contacts``: ``x / dt`` where the law has ``x * (1 / dt)``;
    * ``sparse_contact``: the bias without ``bias_scale`` (always 1 there), selected before the add, so the
      law's fused multiply-add rounds once less; its limit and mimic prefix the same way;
    * ``resolved_worlds._check_normal``: the predicted residual's ``x / dt`` and ``phi < 0`` boundary.
    """
    for module, text in (
        ("row_packets.py", _ROW_PACKETS_RHS),
        ("sparse_contact.py", _SPARSE_CONTACT_RHS),
        ("sparse_contact.py", _SPARSE_LIMIT_RHS),
        ("resolved_worlds.py", _RESOLVED_RESIDUAL),
    ):
        anchored(test, module, text)
    signature = (
        "phi_in: wp.array[float], beta_in: wp.array[float], target_in: wp.array[float], kind_in: wp.array[int], "
        "cap_in: wp.array[wp.vec2], incident_in: wp.array[float], out: wp.array[wp.vec4]"
    )
    settings = "dt = 1.0 / 60.0\nbias_scale = 0.8\nspeculative = 0.9\n"
    reference_body = (
        "i = wp.tid()\n"
        + settings
        + "phi = phi_in[i]\nbeta = beta_in[i]\ntarget = target_in[i]\nlane = int(0)\n"
        + "rhs = -target\nweight = float(1.0)\n"
        + dedent_block(_ROW_PACKETS_RHS)
        .replace("data.bias_scale", "bias_scale")
        .replace("data.dt", "dt")
        .replace("data.contact_w", "0.5")
        .replace("data.speculative_scale", "speculative")
        + "packets = rhs\n"
        + dedent_block(_SPARSE_CONTACT_RHS).replace("rows.speculative_scale", "speculative")
        + "sparse = -target + bias\nworld = int(0)\nrow = i\n"
        + dedent_block(_SPARSE_LIMIT_RHS)
        .replace("rows.type[world, row]", "kind_in[i]")
        .replace("rows.beta_row[world, row]", "beta")
        + "limit = -target + bias\n"
        + "mf = True\nresponds_a = cap_in[i][0] >= 0.0\nresponds_b = cap_in[i][1] >= 0.0\nbody_a = int(0)\nbody_b = int(1)\n"
        + dedent_block(_RESOLVED_RESIDUAL)
        .replace("data.speculative_scale", "speculative")
        .replace("data.dt", "dt")
        .replace("data.beta", "beta")
        .replace("data.max_depenetration_velocity[body_a]", "cap_in[i][0]")
        .replace("data.max_depenetration_velocity[body_b]", "cap_in[i][1]")
        + "residual = incident_in[i] + bias\n"
        + "out[i] = wp.vec4(packets, sparse, limit, residual)\n"
    )
    reference = reference_kernel(
        "producer_rhs_reference",
        signature,
        reference_body,
        {"PGS_CONSTRAINT_TYPE_MIMIC": kernels.PGS_CONSTRAINT_TYPE_MIMIC},
    )
    law_body = (
        "i = wp.tid()\n"
        + settings
        + "phi = phi_in[i]\nbeta = beta_in[i]\ntarget = target_in[i]\ninv_dt = 1.0 / dt\n"
        + "packets, w = row_laws.contact_rhs(phi, beta, target, inv_dt, bias_scale, speculative, 0.5, 0.0, 0.0)\n"
        + "sparse, w2 = row_laws.contact_rhs(phi, beta, target, inv_dt, 1.0, speculative, 1.0, 0.0, 0.0)\n"
        + "limit, w3 = row_laws.row_rhs(kind_in[i], phi, beta, target, inv_dt, 1.0, 0.0, 1.0, 1.0)\n"
        + "body_a = int(-1)\nif cap_in[i][0] >= 0.0:\n    body_a = 2 * i\n"
        + "body_b = int(-1)\nif cap_in[i][1] >= 0.0:\n    body_b = 2 * i + 1\n"
        + "cap = row_laws.contact_depenetration_cap(body_a, body_b, caps_flat)\n"
        + "bias, w4 = row_laws.contact_rhs(phi, beta, 0.0, inv_dt, 1.0, speculative, 1.0, cap, 0.0)\n"
        + "residual = incident_in[i] + bias\n"
        + "out[i] = wp.vec4(packets, sparse, limit, residual)\n"
    )
    law = reference_kernel(
        "producer_rhs_law", signature + ", caps_flat: wp.array[float]", law_body, {"row_laws": row_laws}
    )
    rng = np.random.default_rng(55)
    n = 32768
    caps = rng.choice(np.array([-1.0, 0.5, 2.0, 0.0, np.inf], dtype=np.float32), (n, 2))
    values = {
        "phi_in": scalars(rng, n, 0.02),
        "beta_in": rng.uniform(0.0, 1.0, n).astype(np.float32),
        "target_in": scalars(rng, n),
        "kind_in": rng.choice(np.array([3, 5], dtype=np.int32), n),
        "cap_in": caps,
        "incident_in": scalars(rng, n),
        "out": np.zeros((n, 4), dtype=np.float32),
    }
    a = call(reference, n, device, **values)["out"].numpy()
    b = call(law, n, device, caps_flat=caps.reshape(-1), **values)["out"].numpy()
    magnitude = np.abs(values["phi_in"]) * 60.0 + np.abs(values["target_in"]) + np.abs(values["incident_in"])
    for column, name in enumerate(
        (
            "row_packets.produce_contacts rhs",
            "sparse_contact contact rhs",
            "sparse_contact limit and mimic prefix rhs",
            "resolved_worlds predicted residual",
        )
    ):
        test.assertLessEqual(unified(name, a[:, column], b[:, column], magnitude, 2.0), 2.0, name)


def test_keyed_rows_rhs_matches_native_rows(test, device):
    """Unify the keyed rows' native rhs (``x / dt``) and literal ``1e-6`` slop with the canonical row laws.

    Contact and limit biases move by the rounding of ``x / dt`` against ``x * (1 / dt)``;
    the restitution test with the literal slop fires exactly where the law fires.
    """
    require_cuda(test, device)
    contact = anchored(
        test, "keyed_rows.py", "rhs += (phi <= 0.0f ? c.bias_scale * c.beta : c.speculative_scale) * phi / c.dt;"
    )
    fires = anchored(
        test,
        "keyed_rows.py",
        "if (relative < -c.restitution_threshold && (phi <= 1.0e-6f || phi + c.dt * relative <= 1.0e-6f))",
    )
    limit = anchored(test, "keyed_row_layout.py", "((phi < 0.0f ? p.pgs_beta : 1.0f) * phi / p.dt)")
    rng = np.random.default_rng(54)
    n = 32768
    values = {
        "phi": scalars(rng, n, 0.02),
        "target": scalars(rng, n),
        "relative": scalars(rng, n),
        "beta": rng.uniform(0.0, 1.0, n).astype(np.float32),
    }
    inputs = dict.fromkeys(values, float)
    outputs = {"contact_rhs": float, "fired": float, "limit_rhs": float}
    constants = "    const float bias_scale = 0.7f, speculative_scale = 0.9f, dt = 1.0f / 60.0f, threshold = 0.2f;\n"
    reference_body = (
        constants
        + "    float rhs = -target;\n    "
        + contact.replace("c.bias_scale", "bias_scale")
        .replace("c.beta", "beta")
        .replace("c.speculative_scale", "speculative_scale")
        .replace("c.dt", "dt")
        + "\n    const float contact_rhs = rhs;\n    float fired = 0.0f;\n    "
        + fires.replace("c.restitution_threshold", "threshold").replace("c.dt", "dt")
        + " fired = 1.0f;\n    const float limit_rhs = "
        + limit.replace("p.pgs_beta", "beta").replace("p.dt", "dt")
        + ";"
    )
    law_body = (
        constants
        + "    const float contact_rhs = RowLaw::contact_rhs(phi, beta, target, 1.0f / dt, bias_scale, speculative_scale, 1.0f, 0.0f, 0.0f).v0;\n"
        "    const float fired = RowLaw::contact_restitution_fires(phi, relative, dt, threshold) ? 1.0f : 0.0f;\n"
        "    const float limit_rhs = RowLaw::limit_rhs(phi, beta, 0.0f, 1.0f / dt, 1.0f, 1.0f);"
    )
    reference = native_kernel("native_keyed_rhs", inputs, outputs, reference_body)
    law = native_kernel("law_keyed_rhs", inputs, outputs, law_body, preludes=row_laws.ROW_LAW.preludes())
    a = run_native(reference, inputs, outputs, values, n, device)
    b = run_native(law, inputs, outputs, values, n, device)
    np.testing.assert_array_equal(a["fired"], b["fired"])
    magnitude = np.abs(values["phi"]) * 60.0 + np.abs(values["target"])
    test.assertLessEqual(
        unified("keyed rows contact rhs (native)", a["contact_rhs"], b["contact_rhs"], magnitude, 2.0), 2.0
    )
    limit_magnitude = np.abs(values["phi"]) * 60.0
    test.assertLessEqual(
        unified("keyed rows limit prefix rhs (native)", a["limit_rhs"], b["limit_rhs"], limit_magnitude, 2.0), 2.0
    )


class TestFeatherPGSRowLaws(unittest.TestCase):
    """Each row law against the implementations it replaces."""


register_device_tests(TestFeatherPGSRowLaws, globals())


if __name__ == "__main__":
    unittest.main(verbosity=2)
