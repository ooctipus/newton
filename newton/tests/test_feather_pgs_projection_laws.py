# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The projection laws against the sweeps' implementations, in Warp and as native CUDA text.

Native references are the blocks the generators emit, extracted from their output
and compiled; the pair law unifies the native sweeps' rounding with the Warp one.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import friction, kernels, projection_laws
from newton.tests.feather_pgs_law_harness import (
    anchored,
    between,
    call,
    dedent_block,
    mismatches,
    native_kernel,
    normalized,
    reference_kernel,
    register_device_tests,
    require_cuda,
    run_native,
    scalars,
    spd3,
    twin_inputs,
    unified,
)

# ---------------------------------------------------------------------------------------------------------------------
# Projection laws against the sweeps' implementations
# ---------------------------------------------------------------------------------------------------------------------


@wp.kernel
def _kernel_friction_pair(
    a: wp.array[float],
    c: wp.array[float],
    d: wp.array[float],
    residual: wp.array[wp.vec2],
    old: wp.array[wp.vec2],
    radius: wp.array[float],
    omega: wp.array[float],
    out: wp.array[wp.vec2],
):
    i = wp.tid()
    out[i] = friction.friction_pair_candidate(a[i], c[i], d[i], residual[i], old[i], radius[i], omega[i])


@wp.kernel
def _law_friction_pair(
    a: wp.array[float],
    c: wp.array[float],
    d: wp.array[float],
    residual: wp.array[wp.vec2],
    old: wp.array[wp.vec2],
    radius: wp.array[float],
    omega: wp.array[float],
    out: wp.array[wp.vec2],
):
    i = wp.tid()
    out[i] = projection_laws.friction_pair(a[i], c[i], d[i], residual[i], old[i], radius[i], omega[i])


def test_friction_pair_matches_warp_pair_law(test, device):
    """Equal ``friction.friction_pair_candidate``, the Warp loop sweeps' pair law, bitwise."""
    rng = np.random.default_rng(41)
    n = 8192
    values = {**twin_inputs("friction_pair", rng, n, device), "out": np.zeros((n, 2), dtype=np.float32)}
    a = call(_kernel_friction_pair, n, device, **values)["out"].numpy()
    b = call(_law_friction_pair, n, device, **values)["out"].numpy()
    test.assertFalse(mismatches(a, b).any())


def _contact_triples(rng, worlds):
    """One matrix-free contact (normal and two tangent rows) per world between bodies 2w and 2w + 1."""
    body_a = np.where(rng.uniform(size=worlds) < 0.9, 2 * np.arange(worlds), -1).astype(np.int32)
    body_b = np.where(rng.uniform(size=worlds) < 0.6, 2 * np.arange(worlds) + 1, -1).astype(np.int32)
    J_a = rng.normal(size=(worlds, 3, 6)).astype(np.float32)
    J_b = rng.normal(size=(worlds, 3, 6)).astype(np.float32)
    mass = rng.uniform(0.2, 3.0, size=(worlds, 1, 6)).astype(np.float32)
    rhs = np.zeros((worlds, 3), dtype=np.float32)
    rhs[:, 0] = rng.normal(scale=0.5, size=worlds)
    old = np.abs(rng.normal(scale=0.3, size=(worlds, 3))).astype(np.float32)
    old[:, 1:] = rng.normal(scale=0.2, size=(worlds, 2))
    return {
        "mf_body_a": np.repeat(body_a[:, None], 3, axis=1),
        "mf_body_b": np.repeat(body_b[:, None], 3, axis=1),
        "mf_J_a": J_a,
        "mf_J_b": J_b,
        "mf_MiJt_a": J_a * mass,
        "mf_MiJt_b": J_b * mass,
        "mf_row_parent": np.tile(np.array([[-1, 0, 0]], dtype=np.int32), (worlds, 1)),
        "mf_row_mu": np.repeat(rng.uniform(0.0, 1.2, (worlds, 1)), 3, axis=1).astype(np.float32),
        "mf_rhs": rhs,
        "body_to_articulation": np.arange(2 * worlds, dtype=np.int32),
        "art_dof_start": (6 * np.arange(2 * worlds)).astype(np.int32),
        "mf_impulses": old,
        "v_out": rng.normal(size=12 * worlds).astype(np.float32),
    }


@wp.func
def _gather_contact_block(
    world: int,
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    v_out: wp.array[float],
):
    """The sweeps' gather of a contact's row velocities and Delassus block (the friction steps' own loops)."""
    ba = mf_body_a[world, 0]
    bb = mf_body_b[world, 0]
    u_n = float(0.0)
    u_t1 = float(0.0)
    u_t2 = float(0.0)
    if ba >= 0:
        ds_a = art_dof_start[body_to_articulation[ba]]
        for k in range(6):
            va_k = v_out[ds_a + k]
            u_n = u_n + mf_J_a[world, 0, k] * va_k
            u_t1 = u_t1 + mf_J_a[world, 1, k] * va_k
            u_t2 = u_t2 + mf_J_a[world, 2, k] * va_k
    if bb >= 0:
        ds_b = art_dof_start[body_to_articulation[bb]]
        for k in range(6):
            vb_k = v_out[ds_b + k]
            u_n = u_n + mf_J_b[world, 0, k] * vb_k
            u_t1 = u_t1 + mf_J_b[world, 1, k] * vb_k
            u_t2 = u_t2 + mf_J_b[world, 2, k] * vb_k
    G_nn = float(0.0)
    G_nt1 = float(0.0)
    G_nt2 = float(0.0)
    G_t1t1 = float(0.0)
    G_t1t2 = float(0.0)
    G_t2t2 = float(0.0)
    if ba >= 0:
        for k in range(6):
            G_nn = G_nn + mf_J_a[world, 0, k] * mf_MiJt_a[world, 0, k]
            G_nt1 = G_nt1 + mf_J_a[world, 0, k] * mf_MiJt_a[world, 1, k]
            G_nt2 = G_nt2 + mf_J_a[world, 0, k] * mf_MiJt_a[world, 2, k]
            G_t1t1 = G_t1t1 + mf_J_a[world, 1, k] * mf_MiJt_a[world, 1, k]
            G_t1t2 = G_t1t2 + mf_J_a[world, 1, k] * mf_MiJt_a[world, 2, k]
            G_t2t2 = G_t2t2 + mf_J_a[world, 2, k] * mf_MiJt_a[world, 2, k]
    if bb >= 0:
        for k in range(6):
            G_nn = G_nn + mf_J_b[world, 0, k] * mf_MiJt_b[world, 0, k]
            G_nt1 = G_nt1 + mf_J_b[world, 0, k] * mf_MiJt_b[world, 1, k]
            G_nt2 = G_nt2 + mf_J_b[world, 0, k] * mf_MiJt_b[world, 2, k]
            G_t1t1 = G_t1t1 + mf_J_b[world, 1, k] * mf_MiJt_b[world, 1, k]
            G_t1t2 = G_t1t2 + mf_J_b[world, 1, k] * mf_MiJt_b[world, 2, k]
            G_t2t2 = G_t2t2 + mf_J_b[world, 2, k] * mf_MiJt_b[world, 2, k]
    W = wp.mat33(G_nn, G_nt1, G_nt2, G_nt1, G_t1t1, G_t1t2, G_nt2, G_t1t2, G_t2t2)
    return W, wp.vec3(u_n, u_t1, u_t2)


@wp.kernel
def _kernel_coulomb_steps(
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_rhs: wp.array2d[float],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    mode: int,
    mf_impulses: wp.array2d[float],
    v_out: wp.array[float],
    out: wp.array[wp.vec3],
):
    world = wp.tid()
    new_t1 = float(0.0)
    if mode == 2:
        new_t1 = kernels.friction_step_coulomb_newton(
            world,
            1,
            mf_body_a,
            mf_body_b,
            mf_J_a,
            mf_J_b,
            mf_MiJt_a,
            mf_MiJt_b,
            mf_row_parent,
            mf_row_mu,
            mf_rhs,
            body_to_articulation,
            art_dof_start,
            mf_impulses,
            v_out,
        )
    else:
        new_t1 = kernels.friction_step_bisection(
            world,
            1,
            mf_body_a,
            mf_body_b,
            mf_J_a,
            mf_J_b,
            mf_MiJt_a,
            mf_MiJt_b,
            mf_row_parent,
            mf_row_mu,
            mf_rhs,
            body_to_articulation,
            art_dof_start,
            mode,
            mf_impulses,
            v_out,
        )
    out[world] = wp.vec3(mf_impulses[world, 0], new_t1, mf_impulses[world, 2])


@wp.kernel
def _law_coulomb_steps(
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_row_mu: wp.array2d[float],
    mf_rhs: wp.array2d[float],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    mode: int,
    mf_impulses: wp.array2d[float],
    v_out: wp.array[float],
    out: wp.array[wp.vec3],
):
    world = wp.tid()
    W, u = _gather_contact_block(
        world, mf_body_a, mf_body_b, mf_J_a, mf_J_b, mf_MiJt_a, mf_MiJt_b, body_to_articulation, art_dof_start, v_out
    )
    old = wp.vec3(mf_impulses[world, 0], mf_impulses[world, 1], mf_impulses[world, 2])
    mu = mf_row_mu[world, 1]
    target = -mf_rhs[world, 0]
    if mode == 2:
        out[world] = projection_laws.coulomb_newton(W, u, old, mu, target)
    else:
        if mode == 1:
            target = projection_laws.de_saxce_target(target, u, mu)
        out[world] = projection_laws.coulomb_bisection(W, u, old, mu, target)


def test_coulomb_laws_match_mf_friction_steps(test, device):
    """Equal ``friction_step_bisection`` (with and without de Saxce) and ``friction_step_coulomb_newton`` bitwise."""
    rng = np.random.default_rng(42)
    worlds = 2048
    for mode, name in ((0, "bisection"), (1, "bisection_desaxce"), (2, "coulomb_newton")):
        with test.subTest(friction_mode=name):
            values = _contact_triples(rng, worlds)
            out = call(
                _kernel_coulomb_steps, worlds, device, mode=mode, out=np.zeros((worlds, 3), np.float32), **values
            )
            law_values = {k: v for k, v in values.items() if k != "mf_row_parent"}
            law = call(
                _law_coulomb_steps, worlds, device, mode=mode, out=np.zeros((worlds, 3), np.float32), **law_values
            )
            test.assertFalse(mismatches(out["out"].numpy(), law["out"].numpy()).any())
            test.assertTrue(np.any(out["out"].numpy() != values["mf_impulses"]))


@wp.kernel
def _kernel_coulomb_row(
    W: wp.array[wp.mat33], b: wp.array[wp.vec3], mu: wp.array[float], out: wp.array[kernels.FPGSCoulombNewtonResult]
):
    i = wp.tid()
    out[i] = kernels.solve_coulomb_row(W[i], b[i], mu[i])


@wp.kernel
def _law_coulomb_row(
    W: wp.array[wp.mat33], b: wp.array[wp.vec3], mu: wp.array[float], out: wp.array[kernels.FPGSCoulombNewtonResult]
):
    i = wp.tid()
    out[i] = projection_laws.coulomb_row(W[i], b[i], mu[i])


def test_coulomb_row_matches_daviet_solve(test, device):
    """Equal ``kernels.solve_coulomb_row`` (Daviet's Newton on one block) bitwise, iteration counts included."""
    rng = np.random.default_rng(43)
    n = 8192
    values = {**twin_inputs("coulomb_row", rng, n, device), "out": np.zeros((n, 6), dtype=np.float32)}
    a = call(_kernel_coulomb_row, n, device, **values)["out"].numpy()
    b = call(_law_coulomb_row, n, device, **values)["out"].numpy()
    test.assertFalse(mismatches(a, b).any())


@wp.kernel
def _kernel_speed_row(
    velocity: wp.array[float],
    limit: wp.array[float],
    effective_mass: wp.array[float],
    old: wp.array[float],
    accumulated: int,
    out: wp.array[wp.vec2],
):
    row = wp.tid()
    out[row] = kernels.project_free_body_speed(
        velocity, 3 * row, limit[row], effective_mass[row], old[row], accumulated
    )


@wp.kernel
def _law_speed_row(
    velocity: wp.array[float],
    limit: wp.array[float],
    effective_mass: wp.array[float],
    old: wp.array[float],
    accumulated: int,
    out: wp.array[wp.vec2],
):
    row = wp.tid()
    dof = 3 * row
    v = wp.vec3(velocity[dof], velocity[dof + 1], velocity[dof + 2])
    result = projection_laws.speed_row(wp.length(v), limit[row], effective_mass[row], old[row], accumulated)
    if result[2] != 0.0:
        for k in range(3):
            velocity[dof + k] = velocity[dof + k] + v[k] * result[2]
    out[row] = wp.vec2(result[0], result[1])


def test_speed_row_matches_free_body_speed_projection(test, device):
    """Equal ``project_free_body_speed`` bitwise in impulse, change and the projected velocity, under both laws."""
    rng = np.random.default_rng(44)
    rows = 4096
    for accumulated in (0, 1):
        with test.subTest(accumulated=accumulated):
            velocity = rng.normal(scale=2.0, size=3 * rows).astype(np.float32)
            velocity[: 3 * 16] = 0.0
            values = {
                "limit": rng.uniform(0.1, 3.0, rows).astype(np.float32),
                "effective_mass": rng.uniform(0.1, 2.0, rows).astype(np.float32),
                "old": np.abs(rng.normal(size=rows)).astype(np.float32),
                "accumulated": accumulated,
            }
            a = call(
                _kernel_speed_row, rows, device, velocity=velocity.copy(), out=np.zeros((rows, 2), np.float32), **values
            )
            b = call(
                _law_speed_row, rows, device, velocity=velocity.copy(), out=np.zeros((rows, 2), np.float32), **values
            )
            test.assertFalse(mismatches(a["out"].numpy(), b["out"].numpy()).any())
            test.assertFalse(mismatches(a["velocity"].numpy()[:, None], b["velocity"].numpy()[:, None]).any())


# The Warp loop sweeps' update, unilateral clamp and disk clamp, as kernels.py writes them.
_MF_LOOP_STEP = """
            if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
                # Proximal regularization in w-form (w = 1/(1+g)): identical
                # algebra, but no g*lambda product that could overflow for
                # extreme finite g; w = 1 (g = 0) is the exact hard update.
                w = float(1.0)
                if regularize != 0:
                    w = mf_row_w[world, i]
                delta = -residual * eff_inv * w - (1.0 - w) * old_impulse
            new_impulse = old_impulse + omega * delta

            # Project
            if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
                if new_impulse < 0.0:
                    new_impulse = 0.0
"""
_LOOP_DISK = """
                mag = wp.sqrt(a * a + b * b)
                if mag > radius:
                    scale = radius / mag
                    a *= scale
                    b *= scale
"""


def test_step_and_clamps_match_warp_loop_sweeps(test, device):
    """Equal the Warp loop sweeps' regularized update, unilateral clamp and disk clamp bitwise."""
    anchored(test, "kernels.py", _MF_LOOP_STEP)
    anchored(test, "kernels.py", _LOOP_DISK)
    signature = (
        "residual_in: wp.array[float], eff_inv_in: wp.array[float], w_in: wp.array[float], "
        "old_in: wp.array[float], pair: wp.array[wp.vec2], radius_in: wp.array[float], "
        "omega: float, out: wp.array[wp.vec3]"
    )
    body = (
        "i = wp.tid()\nworld = 0\nmf_row_w = w_in\nregularize = 1\nrow_type = PGS_CONSTRAINT_TYPE_CONTACT\n"
        "residual = residual_in[i]\neff_inv = eff_inv_in[i]\nold_impulse = old_in[i]\ndelta = -residual * eff_inv\n"
        + dedent_block(_MF_LOOP_STEP).replace("mf_row_w[world, i]", "mf_row_w[i]")
        + "a = pair[i][0]\nb = pair[i][1]\nradius = radius_in[i]\n"
        + dedent_block(_LOOP_DISK)
        + "out[i] = wp.vec3(new_impulse, a, b)\n"
    )
    reference = reference_kernel(
        "loop_step_reference", signature, body, {"PGS_CONSTRAINT_TYPE_CONTACT": kernels.PGS_CONSTRAINT_TYPE_CONTACT}
    )
    law_body = (
        "i = wp.tid()\n"
        "new_impulse = projection_laws.unilateral(projection_laws.regularized_step(old_in[i], -residual_in[i] * eff_inv_in[i], w_in[i], omega))\n"
        "pair_out = projection_laws.disk(pair[i], radius_in[i])\n"
        "out[i] = wp.vec3(new_impulse, pair_out[0], pair_out[1])\n"
    )
    law = reference_kernel("loop_step_law", signature, law_body, {"projection_laws": projection_laws})
    rng = np.random.default_rng(45)
    n = 8192
    values = {
        "residual_in": scalars(rng, n),
        "eff_inv_in": rng.uniform(0.1, 3.0, n).astype(np.float32),
        "w_in": rng.choice(np.array([1.0, 0.5, 0.98, 0.2], dtype=np.float32), n),
        "old_in": np.abs(scalars(rng, n)),
        "pair": rng.normal(size=(n, 2)).astype(np.float32),
        "radius_in": np.abs(scalars(rng, n)),
        "omega": 0.9,
        "out": np.zeros((n, 3), dtype=np.float32),
    }
    a = call(reference, n, device, **values)["out"].numpy()
    b = call(law, n, device, **values)["out"].numpy()
    test.assertFalse(mismatches(a, b).any())


def test_friction_pair_matches_native_pair_law(test, device):
    """Unify the native sweeps' pair law (``friction.FRICTION_PAIR``) with the Warp one: equal within 1e-5 of the disk.

    Today's CUDA text and Warp function are two writings of one law whose
    roundings differ. The law is the Warp writing; its CUDA rendering moves the
    native sweeps by that rounding, a few float32 units of the disk's radius.
    """
    require_cuda(test, device)
    inputs = {"a": float, "c": float, "d": float, "residual": wp.vec2, "old": wp.vec2, "radius": float, "omega": float}
    outputs = {"result": wp.vec2}
    hand = native_kernel(
        "pair_hand",
        inputs,
        outputs,
        "    const float2 p = friction_pair_candidate(a, c, d, residual[0], residual[1], old[0], old[1], radius, omega);\n"
        "    const wp::vec2 result(p.x, p.y);",
        preludes=(friction.FRICTION_PAIR,),
    )
    law = native_kernel(
        "pair_law",
        inputs,
        outputs,
        "    const wp::vec2 result = ProjectionLaw::friction_pair(a, c, d, residual, old, radius, omega);",
        preludes=projection_laws.PROJECTION_LAW.preludes(),
    )
    rng = np.random.default_rng(51)
    n = 65536
    values = twin_inputs("friction_pair", rng, n, device)
    a = run_native(hand, inputs, outputs, values, n, device)["result"]
    b = run_native(law, inputs, outputs, values, n, device)["result"]
    scale = np.maximum(values["radius"], 1.0e-6)[:, None]
    unified("native pair law (friction.FRICTION_PAIR)", a, b, scale, 0.0)
    test.assertLess(np.max(np.abs(a - b) / scale), 1.0e-5)


def test_coulomb_laws_match_native_friction_modes(test, device):
    """Equal the MF-GS sweep's native RAISim bisection (dense and matrix-free copies) and Daviet Newton.

    Each reference is the block the generator emits, extracted from its output.
    """
    require_cuda(test, device)
    from newton._src.solvers.feather_pgs import mf_gs_source  # noqa: PLC0415

    inputs = {
        "u_n": float,
        "u_t1": float,
        "u_t2": float,
        "G_nn": float,
        "G_nt1": float,
        "G_nt2": float,
        "G_t1t1": float,
        "G_t1t2": float,
        "G_t2t2": float,
        "old_lambda_n": float,
        "old_lambda_t1": float,
        "old_lambda_t2": float,
        "mu": float,
        "target_vel_n": float,
    }
    outputs = {"result": wp.vec3}
    law_body = (
        "    const wp::mat33 W(G_nn, G_nt1, G_nt2, G_nt1, G_t1t1, G_t1t2, G_nt2, G_t1t2, G_t2t2);\n"
        "    const wp::vec3 u(u_n, u_t1, u_t2), old(old_lambda_n, old_lambda_t1, old_lambda_t2);\n"
        "    const wp::vec3 result = CoulombLaw::{law}(W, u, old, mu, target_vel_n);"
    )
    rng = np.random.default_rng(52)
    n = 32768
    w = spd3(rng, n)
    values = {
        "u_n": scalars(rng, n),
        "u_t1": scalars(rng, n),
        "u_t2": scalars(rng, n),
        "G_nn": w[:, 0, 0],
        "G_nt1": w[:, 0, 1],
        "G_nt2": w[:, 0, 2],
        "G_t1t1": w[:, 1, 1],
        "G_t1t2": w[:, 1, 2],
        "G_t2t2": w[:, 2, 2],
        "old_lambda_n": np.abs(scalars(rng, n)),
        "old_lambda_t1": scalars(rng, n, 0.3),
        "old_lambda_t2": scalars(rng, n, 0.3),
        "mu": rng.uniform(0.0, 1.2, n).astype(np.float32),
        "target_vel_n": scalars(rng, n),
    }
    cases = []
    for mode, law_name, occurrences in (
        ("bisection", "coulomb_bisection", (0, 1)),
        ("coulomb_newton", "coulomb_newton", (0,)),
    ):
        source = mf_gs_source.get_source(16, 16, 12, "sm_80", friction_mode=mode)
        for occurrence in occurrences:
            block = between(source, "if (G_nn >= 1.0e-20f) {", "d_n_total = new_lambda_n - old_lambda_n;", occurrence)
            cases.append((f"{mode}_{occurrence}", law_name, block))
    for name, law_name, block in cases:
        with test.subTest(block=name):
            reference = native_kernel(
                f"native_{name}",
                inputs,
                outputs,
                "    float new_lambda_n = old_lambda_n, new_lambda_t1 = old_lambda_t1, new_lambda_t2 = old_lambda_t2;\n"
                + block
                + "}\n    const wp::vec3 result(new_lambda_n, new_lambda_t1, new_lambda_t2);",
            )
            law = native_kernel(
                f"law_{name}",
                inputs,
                outputs,
                law_body.replace("{law}", law_name),
                preludes=projection_laws.COULOMB_LAW.preludes(),
            )
            a = run_native(reference, inputs, outputs, values, n, device)["result"]
            b = run_native(law, inputs, outputs, values, n, device)["result"]
            scale = np.maximum(np.abs(a).max(axis=1, keepdims=True), 1.0e-3)
            unified(f"native {name} (mf_gs_source)", a, b, scale, 0.0)
            test.assertLess(np.max(np.abs(a - b) / scale), 1.0e-4)


def test_drive_and_velocity_limit_laws_match_native_text(test, device):
    """Equal the native PhysX drive update and both velocity-limit laws, as the generators emit them, bitwise."""
    require_cuda(test, device)
    from newton._src.solvers.feather_pgs import mf_gs_source, velocity_limit_law  # noqa: PLC0415

    source = mf_gs_source.get_source(16, 16, 12, "sm_80")
    drive = between(source, "new_impulse = old_impulse * s_drive_imp_mul_dense[i]", "delta_impulse = new_impulse")
    stateless = between(source, "float delta_vlim = -residual_vlim / denom;", "if (lane == 0) s_lam_dense[i]")
    accumulated = velocity_limit_law.accumulated_update(
        "old_vlim", "residual_vlim / denom", new="new_vlim", delta="delta_impulse_vlim", indent="    ", declare="float "
    )
    test.assertIn(
        normalized(accumulated),
        normalized(mf_gs_source.get_source(16, 16, 12, "sm_80", velocity_limit_law="accumulated")),
    )
    rng = np.random.default_rng(53)
    n = 32768
    values = {
        "old_impulse": scalars(rng, n),
        "jv": scalars(rng, n),
        "imp_mul": rng.uniform(0.0, 1.0, n).astype(np.float32),
        "vel_mul": scalars(rng, n),
        "target": scalars(rng, n),
        "max_impulse": np.abs(scalars(rng, n)),
        "residual_vlim": scalars(rng, n),
        "denom": rng.uniform(0.1, 3.0, n).astype(np.float32),
        "old_vlim": np.abs(scalars(rng, n)),
    }
    inputs = dict.fromkeys(values, float)
    outputs = {"drive": float, "stateless_new": float, "accumulated_new": float, "accumulated_delta": float}
    # Each extracted text in its own block: both velocity-limit laws declare ``delta_impulse_vlim``.
    reference_body = (
        "    float drive, stateless_new, accumulated_new, accumulated_delta;\n    {\n    float new_impulse = 0.0f;\n    "
        + drive.replace("s_drive_imp_mul_dense[i]", "imp_mul")
        .replace("s_drive_vel_mul_dense[i]", "vel_mul")
        .replace("s_drive_target_dense[i]", "target")
        .replace("s_drive_max_imp_dense[i]", "max_impulse")
        + "\n    drive = new_impulse;\n    }\n    {\n    "
        + stateless
        + "\n    stateless_new = delta_impulse_vlim;\n    }\n    {\n"
        + accumulated
        + "    accumulated_new = new_vlim;\n    accumulated_delta = delta_impulse_vlim;\n    }"
    )
    law_body = (
        "    const float drive = ProjectionLaw::physx_drive(old_impulse, jv, imp_mul, vel_mul, target, max_impulse);\n"
        "    const float step = residual_vlim / denom;\n"
        "    const float stateless_new = ProjectionLaw::velocity_limit_update(0.0f, residual_vlim, step, 0).v0;\n"
        "    const auto accumulated = ProjectionLaw::velocity_limit_update(old_vlim, residual_vlim, step, 1);\n"
        "    const float accumulated_new = accumulated.v0, accumulated_delta = accumulated.v1;"
    )
    reference = native_kernel("native_drive_vlim", inputs, outputs, reference_body)
    law = native_kernel("law_drive_vlim", inputs, outputs, law_body, preludes=projection_laws.PROJECTION_LAW.preludes())
    a = run_native(reference, inputs, outputs, values, n, device)
    b = run_native(law, inputs, outputs, values, n, device)
    for name in outputs:
        test.assertFalse(mismatches(a[name][:, None], b[name][:, None]).any(), name)


def test_cached_friction_pair_matches_native_cached_law(test, device):
    """Unify the sparse sweep's cached pair law (``friction.FRICTION_PAIR_CACHED``) with the law's metric form.

    The law's :func:`friction_pair_metric` and :func:`friction_pair_from_metric` compose
    :func:`friction_pair`; the native text rounds as the native pair law does.
    """
    require_cuda(test, device)
    inputs = {"a": float, "c": float, "d": float, "residual": wp.vec2, "old": wp.vec2, "radius": float, "omega": float}
    outputs = {"result": wp.vec2, "whole": wp.vec2}
    hand = native_kernel(
        "pair_cached_hand",
        inputs,
        outputs,
        "    const FrictionPairMetric metric = friction_pair_metric(a, c, d);\n"
        "    const float2 p = friction_pair_from_metric(metric, residual[0], residual[1], old[0], old[1], radius, omega);\n"
        "    const wp::vec2 result(p.x, p.y), whole = result;",
        preludes=(friction.FRICTION_PAIR_CACHED,),
    )
    law = native_kernel(
        "pair_cached_law",
        inputs,
        outputs,
        "    const auto metric = ProjectionLaw::friction_pair_metric(a, c, d);\n"
        "    const wp::vec2 result = ProjectionLaw::friction_pair_from_metric(\n"
        "        metric.v0, metric.v1, metric.v2, metric.v3, residual, old, radius, omega);\n"
        "    const wp::vec2 whole = ProjectionLaw::friction_pair(a, c, d, residual, old, radius, omega);",
        preludes=projection_laws.PROJECTION_LAW.preludes(),
    )
    rng = np.random.default_rng(56)
    n = 65536
    values = twin_inputs("friction_pair", rng, n, device)
    a = run_native(hand, inputs, outputs, values, n, device)
    b = run_native(law, inputs, outputs, values, n, device)
    test.assertFalse(mismatches(b["result"], b["whole"]).any(), "the metric form composes the whole law")
    scale = np.maximum(values["radius"], 1.0e-6)[:, None]
    unified("native cached pair law (friction.FRICTION_PAIR_CACHED)", a["result"], b["result"], scale, 0.0)
    test.assertLess(np.max(np.abs(a["result"] - b["result"]) / scale), 1.0e-5)


class TestFeatherPGSProjectionLaws(unittest.TestCase):
    """Each projection law against the implementations it replaces."""


register_device_tests(TestFeatherPGSProjectionLaws, globals())


if __name__ == "__main__":
    unittest.main(verbosity=2)
