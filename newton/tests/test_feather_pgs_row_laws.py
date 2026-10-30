# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The row builders and passes that form a row's right-hand side or candidate, against the row laws.

Each kernel is launched as the solver launches it and compared with the laws'
composition written out here: bitwise on the CPU, and on CUDA within rounding of the
terms (``UNIFICATIONS`` records how many values moved).
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import kernels, response_builds_rows, row_laws
from newton.tests.feather_pgs_law_harness import (
    call,
    joint_type,
    mismatches,
    register_device_tests,
    scalars,
    unified,
    world_row_support,
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
    """Equal ``kernels.compute_world_contact_bias`` bitwise for every row kind and setting."""
    rng = np.random.default_rng(11)
    worlds, rows = 16, 64
    phi, kind, beta, target = _rows(rng, worlds, rows)
    for dt, contact_spec, contact_w in ((1.0 / 60.0, 1.0, 1.0), (1.0 / 120.0, 0.0, 0.5), (7.0e-3, 0.3, 0.98)):
        with test.subTest(dt=dt, contact_spec=contact_spec, contact_w=contact_w):
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
                contact_speculative_scale=contact_spec,
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
                bias_scale=1.0,
                contact_speculative_scale=contact_spec,
                limit_speculative_scale=1.0,
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


def test_restitution_fires_where_the_impact_reaches_the_surface(test, device):
    """Fire for a contact closing faster than the threshold that is at the surface or reaches it during the step.

    The end-gap slop absorbs float32 residuals of rows that land exactly at contact.
    """
    rng = np.random.default_rng(12)
    n = 8192
    phi = scalars(rng, n, 0.01)
    u = scalars(rng, n, 1.0)
    u[::5] = -0.5  # at the threshold
    dt = np.float32(1.0 / 60.0)
    values = {"phi": phi, "u": u, "dt": float(dt), "threshold": 0.5, "out": np.zeros(n, dtype=np.int32)}
    law = call(_law_restitution_fires, n, device, **values)["out"].numpy()
    slop = np.float32(row_laws.CONTACT_END_GAP_SLOP)
    expected = (u < np.float32(-0.5)) & ((phi <= slop) | (phi + dt * u <= slop))
    np.testing.assert_array_equal(law, expected.astype(np.int32))
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
        "mf_Z_a": rng.normal(size=(worlds, rows, 6)).astype(np.float32),
        "mf_Z_b": rng.normal(size=(worlds, rows, 6)).astype(np.float32),
        "body_to_articulation": np.arange(bodies, dtype=np.int32),
        "articulation_dof_start": (6 * np.arange(bodies)).astype(np.int32),
        "incident_velocity": rng.normal(size=6 * bodies).astype(np.float32),
        "rigid_body_max_depenetration_velocity": rng.choice(
            np.array([0.5, 2.0, 0.0, np.inf, 1.0e30, 0.05], dtype=np.float32), bodies
        ),
    }


def _body_factors(rng, bodies, device, *, identity=False):
    """Each body's leaf-first factor of an SPD 6x6 mass matrix (``L_by_size[6]``, one per body: ``P H P^T = L L^T``
    with ``P`` the coordinates reversed), its articulation the body."""
    if identity:
        lower = np.tile(np.eye(6), (bodies, 1, 1))
    else:
        a = rng.normal(size=(bodies, 6, 6))
        lower = np.linalg.cholesky(a @ np.transpose(a, (0, 2, 1)) + 6.0 * np.eye(6))
    factors = kernels.BodyFactors()
    factors.body_to_articulation = wp.array(np.arange(bodies, dtype=np.int32), device=device)
    factors.factor_index = wp.array(np.arange(bodies, dtype=np.int32), device=device)
    factors.dof_start = wp.array((6 * np.arange(bodies)).astype(np.int32), device=device)
    factors.dof_factor = wp.array(np.full(6 * bodies, -1, dtype=np.int32), device=device)
    factors.L = wp.array(lower.astype(np.float32), dtype=float, device=device)
    return factors, lower.astype(np.float32).astype(np.float64)


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
        value, w = row_laws.contact_rhs(phi, pgs_beta, target, inv_dt, 1.0, contact_speculative_scale, contact_w, cap)
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
    """Equal the free-body rows' build (``free_body_rows_kernel``): the contact law with its bodies' depenetration cap
    and weights."""
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
    factors, lower = _body_factors(rng, 2 * worlds, device)
    out = call(
        response_builds_rows.free_body_rows_kernel,
        worlds * rows,
        device,
        mf_constraint_count=np.full(worlds, rows, dtype=np.int32),
        factors=factors,
        pgs_cfm=1.0e-6,
        mf_max_constraints=rows,
        mf_eff_mass_inv=np.zeros((worlds, rows), dtype=np.float32),
        mf_row_incident=np.zeros((worlds, rows), dtype=np.float32),
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
        mf_J_a=shared["mf_Z_a"],
        mf_J_b=shared["mf_Z_b"],
        **shared,
    )
    # The whitened rows: Z = L^-1 P J^T per body (factor coordinate i at position 5 - i; a speed row keeps its axis),
    # and A_ii = cfm + J H^-1 J^T with H = P^T L L^T P.
    Z = {side: out[f"mf_Z_{side}"].numpy().astype(np.float64) for side in "ab"}
    diagonal = np.full((worlds, rows), 1.0e-6)
    for side, bodies in (("a", layout["mf_body_a"]), ("b", layout["mf_body_b"])):
        J = shared[f"mf_Z_{side}"].astype(np.float64)
        for world, row in zip(*np.nonzero(bodies >= 0), strict=True):
            factor = lower[bodies[world, row]]
            whitened = np.linalg.solve(factor, J[world, row][::-1])[::-1]
            expected = J[world, row] if kind[world, row] == 4 else whitened
            np.testing.assert_allclose(Z[side][world, row], expected, rtol=1e-5, atol=1e-5)
            diagonal[world, row] += J[world, row] @ np.linalg.solve((factor @ factor.T)[::-1, ::-1], J[world, row])
    np.testing.assert_allclose(out["mf_eff_mass_inv"].numpy(), 1.0 / diagonal, rtol=1e-5)
    reference_rhs, law_rhs = out["mf_rhs"].numpy(), law["rhs"].numpy()
    np.testing.assert_array_equal(out["mf_row_w"].numpy(), law["row_w"].numpy())
    magnitude = np.abs(phi) * 60.0 + np.abs(target)
    test.assertLessEqual(
        unified("mf rhs (free_body_rows_kernel)", reference_rhs.ravel(), law_rhs.ravel(), magnitude.ravel(), 2.0),
        2.0,
    )
    # Rows whose bias the variant does not touch (speed rows, restitution) stay bitwise.
    untouched = (kind == 4).ravel()
    test.assertFalse(mismatches(reference_rhs.ravel()[untouched, None], law_rhs.ravel()[untouched, None]).any())


def _dense_layout(rng, worlds, rows, dofs, device):
    """Packed rows (:mod:`.packed_rows`) over every world coordinate."""
    return {
        "world_dof_indices": (dofs * np.arange(worlds)[:, None] + np.arange(dofs)[None, :]).astype(np.int32),
        "row_support": world_row_support(worlds, rows, dofs, device),
        "packed_J": rng.normal(size=(worlds, rows, dofs)).astype(np.float32),
    }


@wp.kernel
def _law_restitution_matrix_free(
    world_dof_indices: wp.array2d[int],
    row_support: kernels.RowSupport,
    packed_J: wp.array3d[float],
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
        kernels.world_contact_row_dot(world_dof_indices, row_support, packed_J, world_incident_velocity, world, i)
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
        **_dense_layout(rng, worlds, rows, dofs, device),
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
        sparse_diagonal=0,
        dense_offsets=np.zeros(worlds, dtype=np.int32),
        dense_groups=np.zeros(worlds, dtype=np.int32),
        dense_dofs=0,
        dense_L=np.zeros((1, 1, 1), dtype=np.float32),
        dense_Z=np.zeros((1, 1, 1), dtype=np.float32),
        sparse_row_dof=np.full((1, 1, 2), -1, dtype=np.int32),
        sparse_row_z=np.zeros((1, 1, 2), dtype=np.float32),
        diagonal_inverse_mass=np.zeros(1, dtype=np.float32),
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
        "prescribed": np.zeros(count, dtype=np.int32),
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
        prescribed=np.zeros(count, dtype=np.int32),
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
                prescribed=np.zeros(count, dtype=np.int32),
                joint_qd=qd,
                velocity_limit_activation_fraction=fraction,
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


class TestFeatherPGSRowLaws(unittest.TestCase):
    """The row-rhs producers against the row laws they compose."""


register_device_tests(TestFeatherPGSRowLaws, globals())


if __name__ == "__main__":
    unittest.main(verbosity=2)
