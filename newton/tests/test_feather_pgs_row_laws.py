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

from newton._src.solvers.feather_pgs import kernels, row_laws
from newton._src.solvers.feather_pgs.tree_factor import WalkTrees
from newton._src.solvers.feather_pgs.walk_rows import free_body_headers_kernel
from newton.tests.feather_pgs_law_harness import (
    call,
    mismatches,
    prefix_warp_rows,
    register_device_tests,
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
    """Equal ``kernels.compute_world_contact_bias`` bitwise for every row kind and setting."""
    rng = np.random.default_rng(11)
    worlds, rows = 16, 64
    phi, kind, beta, target = _rows(rng, worlds, rows)
    for dt, contact_spec, contact_w in ((1.0 / 60.0, 1.0, 1.0), (1.0 / 120.0, 0.0, 0.5), (7.0e-3, 0.3, 0.98)):
        with test.subTest(dt=dt, contact_spec=contact_spec, contact_w=contact_w):
            out = call(
                kernels.compute_world_contact_bias,
                (worlds, rows),
                device,
                world_constraint_count=np.full(worlds, rows, dtype=np.int32),
                max_constraints=rows,
                lanes=rows,
                world_phi=phi,
                world_row_beta=beta,
                world_row_type=kind,
                world_target_velocity=target,
                world_row_restitution=np.zeros((worlds, rows), dtype=np.float32),
                row_incident=np.zeros((worlds, rows), dtype=np.float32),
                dt=dt,
                contact_speculative_scale=contact_spec,
                contact_w=contact_w,
                restitution_velocity_threshold=0.0,
                weighted=int(contact_w < 1.0),
                material=0,
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


@wp.kernel
def _row_incident(
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    body_to_articulation: wp.array[int],
    articulation_dof_start: wp.array[int],
    incident_velocity: wp.array[float],
    incident: wp.array2d[float],
):
    """Each free-body row's ``J v_hat``, its sides in turn (the order :func:`_law_mf_position_rhs` sums)."""
    world, i = wp.tid()
    value = float(0.0)
    ba = mf_body_a[world, i]
    if ba >= 0:
        dof_start = articulation_dof_start[body_to_articulation[ba]]
        for k in range(6):
            value += mf_J_a[world, i, k] * incident_velocity[dof_start + k]
    bb = mf_body_b[world, i]
    if bb >= 0:
        dof_start = articulation_dof_start[body_to_articulation[bb]]
        for k in range(6):
            value += mf_J_b[world, i, k] * incident_velocity[dof_start + k]
    incident[world, i] = value


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
        beta = row_laws.contact_capped_beta(phi, pgs_beta, cap, dt)
        value, w = row_laws.contact_rhs(phi, beta, target, inv_dt, 1.0, contact_speculative_scale, contact_w)
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
    """Equal the free-body rows' headers (``walk_rows.free_body_headers_kernel``): the contact law with its bodies'
    depenetration cap and weights, from the incident velocity the walk summed."""
    rng = np.random.default_rng(13)
    worlds, rows = 8, 48
    layout = _mf_layout(rng, worlds, rows)
    phi = scalars(rng, worlds * rows, 0.02).reshape(worlds, rows)
    speed = int(row_laws.ROW_SPEED)
    kind = rng.choice(np.array([0, 0, 0, 2, speed], dtype=np.int32), (worlds, rows))
    # A speed row's diagonal walks its body's covector, which needs the body's walk: here its rows name no body.
    for side in "ab":
        layout[f"mf_body_{side}"] = np.where(kind == speed, -1, layout[f"mf_body_{side}"]).astype(np.int32)
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
    # Each row's incident velocity J v_hat, which the one writer sums along its walks (here in the law's order).
    incident = call(
        _row_incident,
        (worlds, rows),
        device,
        mf_J_a=shared["mf_Z_a"],
        mf_J_b=shared["mf_Z_b"],
        incident=np.zeros((worlds, rows), dtype=np.float32),
        **shared,
    )["incident"]
    out = call(
        free_body_headers_kernel,
        worlds * rows,
        device,
        mf_constraint_count=np.full(worlds, rows, dtype=np.int32),
        walk=WalkTrees(),
        v_hat=layout["incident_velocity"],
        pgs_cfm=1.0e-6,
        mf_max_constraints=rows,
        weighted=1,
        material=0,
        mf_eff_mass_inv=np.zeros((worlds, rows), dtype=np.float32),
        mf_row_incident=incident,
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
    # The walked rows' diagonal: A_ii = cfm + |Z_a|^2 + |Z_b|^2 over the bodies a row names.
    diagonal = np.full((worlds, rows), 1.0e-6)
    for side in "ab":
        bodies, Z = layout[f"mf_body_{side}"], layout[f"mf_Z_{side}"].astype(np.float64)
        diagonal += np.where(bodies >= 0, (Z * Z).sum(axis=2), 0.0)
    np.testing.assert_allclose(out["mf_eff_mass_inv"].numpy(), 1.0 / diagonal, rtol=1e-5)
    reference_rhs, law_rhs = out["mf_rhs"].numpy(), law["rhs"].numpy()
    np.testing.assert_array_equal(out["mf_row_w"].numpy(), law["row_w"].numpy())
    magnitude = np.abs(phi) * 60.0 + np.abs(target)
    test.assertLessEqual(
        unified("mf rhs (free_body_headers_kernel)", reference_rhs.ravel(), law_rhs.ravel(), magnitude.ravel(), 2.0),
        2.0,
    )
    # Rows whose bias the variant does not touch (speed rows, restitution) stay bitwise.
    untouched = (kind == speed).ravel()
    test.assertFalse(mismatches(reference_rhs.ravel()[untouched, None], law_rhs.ravel()[untouched, None]).any())


@wp.kernel
def _law_restitution_matrix_free(
    world_phi: wp.array2d[float],
    world_row_type: wp.array2d[int],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
    row_incident: wp.array2d[float],
    dt: float,
    restitution_velocity_threshold: float,
    rhs: wp.array2d[float],
    row_w: wp.array2d[float],
):
    world, i = wp.tid()
    if world_row_type[world, i] != row_laws.ROW_CONTACT:
        return
    target = world_target_velocity[world, i]
    relative_incident = row_incident[world, i] - target
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
    """Rebound exactly the rows the restitution law selects, with the same rhs bits, and leave the rest their bias."""
    rng = np.random.default_rng(16)
    worlds, rows = 8, 32
    phi, kind, beta, target = _rows(rng, worlds, rows)
    common = {
        "world_phi": phi,
        "world_row_type": kind,
        "world_target_velocity": target,
        "world_row_restitution": rng.choice(np.array([0.0, 0.5, 1.0], dtype=np.float32), (worlds, rows)),
        "row_incident": rng.normal(size=(worlds, rows)).astype(np.float32),
        "dt": 1.0 / 60.0,
        "restitution_velocity_threshold": 0.2,
    }
    out = call(
        kernels.compute_world_contact_bias,
        (worlds, rows),
        device,
        world_constraint_count=np.full(worlds, rows, dtype=np.int32),
        max_constraints=rows,
        lanes=rows,
        world_row_beta=beta,
        contact_speculative_scale=1.0,
        contact_w=0.5,
        weighted=1,
        material=0,
        mf_constraint_count=np.zeros(worlds, dtype=np.int32),
        owned_rows_le=0,
        world_rhs=np.zeros((worlds, rows), dtype=np.float32),
        world_row_w=np.zeros((worlds, rows), dtype=np.float32),
        **common,
    )
    bias = call(
        _law_world_bias,
        (worlds, rows),
        device,
        phi=phi,
        beta=beta,
        kind=kind,
        target=target,
        dt=common["dt"],
        bias_scale=1.0,
        contact_speculative_scale=1.0,
        limit_speculative_scale=1.0,
        contact_w=0.5,
        rhs=np.zeros((worlds, rows), dtype=np.float32),
        row_w=np.zeros((worlds, rows), dtype=np.float32),
    )
    law = call(_law_restitution_matrix_free, (worlds, rows), device, rhs=bias["rhs"], row_w=bias["row_w"], **common)
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
    v_hat: wp.array[float],
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
                    # The row's incident velocity is its J = +1 or -1 on the predictor.
                    incident = wp.where(side == 0, v_hat[global_dof], -v_hat[global_dof])
                    rhs[side] = row_laws.limit_rhs(phi, pgs_beta, 0.0, 1.0 / dt, 1.0, 1.0) + incident
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
        "v_hat": scalars(rng, count, 1.0),
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
    """Equal the violation the mimic warp (``rows.mimic_warp_kernel``) writes for every mimic row, in mimic order."""
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
    out = prefix_warp_rows(
        device,
        arts,
        dofs_per_art,
        per_art,
        family="mimics",
        mimics=per_art,
        mimic_start=(per_art * np.arange(arts + 1)).astype(np.int32),
        mimic_list=np.arange(count, dtype=np.int32),
        mimic_valid=np.ones(count, dtype=np.int32),
        mimic_enabled=np.ones(count, dtype=bool),
        mimic_dof0=dof0,
        mimic_dof1=dof1,
        mimic_q0=dof0,
        mimic_q1=dof1,
        mimic_coef0=coef0,
        mimic_coef1=coef1,
        mimic_slot=np.full(count, -1, dtype=np.int32),
        q=joint_q,
        beta=0.2,
        cfm=0.0,
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
    np.testing.assert_array_equal(out["count"], np.full(arts, per_art))
    np.testing.assert_array_equal(out["mimic_slot"], np.tile(np.arange(per_art), arts))
    test.assertFalse(mismatches(out["phi"].reshape(-1), law["phi"].numpy()).any())


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
    """Activate the sides the joint-limit warp (``rows.joint_limit_warp_kernel``) builds rows for, with bitwise phi, in
    row order."""
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
    out = prefix_warp_rows(
        device,
        arts,
        dofs,
        rows,
        lower=lower,
        upper=upper,
        q=joint_q,
        activation_gap=0.05,
        beta=0.2,
        cfm=0.0,
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
    np.testing.assert_array_equal(out["count"], law["count"].numpy())
    test.assertFalse(mismatches(out["phi"], law["phi"].numpy()).any())


@wp.kernel
def _law_velocity_limit_active(
    joint_velocity_limit: wp.array[float], joint_qd: wp.array[float], fraction: float, active: wp.array[int]
):
    dof = wp.tid()
    active[dof] = wp.where(row_laws.velocity_limit_active(joint_velocity_limit[dof], joint_qd[dof], fraction), 1, 0)


def test_velocity_limit_active_matches_velocity_limit_rows(test, device):
    """Give rows to the coordinates the joint velocity-limit warp (``rows.velocity_limit_warp_kernel``) gives rows
    to: a pair each, lower (``J = +1``) then upper, in coordinate order."""
    rng = np.random.default_rng(21)
    arts, dofs = 8, 6
    count = arts * dofs
    limit = scalars(rng, count, 2.0)
    qd = scalars(rng, count, 2.0)
    for fraction in (0.0, 0.8):
        with test.subTest(fraction=fraction):
            out = prefix_warp_rows(
                device,
                arts,
                dofs,
                2 * dofs,
                family="velocity",
                velocity_limit=limit,
                vhat=qd,
                activation=fraction,
                cfm=0.0,
            )
            law = (
                call(
                    _law_velocity_limit_active,
                    count,
                    device,
                    joint_velocity_limit=limit,
                    joint_qd=qd,
                    fraction=fraction,
                    active=np.zeros(count, dtype=np.int32),
                )
                .get("active")
                .numpy()
                .reshape(arts, dofs)
            )
            np.testing.assert_array_equal(out["count"], 2 * law.sum(axis=1))
            for art in range(arts):
                coordinates = np.flatnonzero(law[art])
                J = out["J"][art, : out["count"][art]]
                expected = np.zeros_like(J)
                expected[0::2][np.arange(coordinates.size), coordinates] = 1.0
                expected[1::2][np.arange(coordinates.size), coordinates] = -1.0
                np.testing.assert_array_equal(J, expected)


class TestFeatherPGSRowLaws(unittest.TestCase):
    """The row-rhs producers against the row laws they compose."""


register_device_tests(TestFeatherPGSRowLaws, globals())


if __name__ == "__main__":
    unittest.main(verbosity=2)
