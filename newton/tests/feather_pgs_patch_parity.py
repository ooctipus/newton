# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare persistent patch friction across FeatherPGS routes, independent of slot order.

Contact slots come from per-world atomic reservations, and a patch normal row's
parent names the slot of the next normal of its patch. These helpers key rows
and anchor history by contact index instead, so two solvers that build the same
rows in a different slot order compare equal. This module holds shared test
helpers; it defines no tests.
"""

import numpy as np

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION

# Per route: constraint count, row type, parent, mu, right-hand side, impulse, phi and
# target arrays. Route 1 (matrix-free rows) folds its target into the right-hand side.
_ROUTE_ARRAYS = {
    0: ("constraint_count", "row_type", "row_parent", "row_mu", "rhs", "impulses", "phi", "target_velocity"),
    1: ("mf_constraint_count", "mf_row_type", "mf_row_parent", "mf_row_mu", "mf_rhs", "mf_impulses", "mf_phi", None),
}


def route_arrays(solver, route=0):
    """Return a route's row arrays as host copies, keyed by role."""
    names = ("count", "type", "parent", "mu", "rhs", "impulse", "phi", "target")
    return {
        role: None if attribute is None else getattr(solver, attribute).numpy()
        for role, attribute in zip(names, _ROUTE_ARRAYS[route], strict=True)
    }


def _jacobian(solver, route, world, count):
    """Physical row Jacobians in world coordinates (route 0) or body-space pairs (route 1)."""
    if route == 1:
        return np.concatenate([solver.mf_J_a.numpy()[world, :count], solver.mf_J_b.numpy()[world, :count]], axis=1)
    if solver._sparse_mass_matrix_size is not None:
        from newton.tests.test_feather_pgs_sparse_solver import TestFeatherPGSSparseSolver  # noqa: PLC0415

        return TestFeatherPGSSparseSolver()._sparse_rows_physical(solver, world, count)[1]
    if solver.pgs_mode == "matrix_free":
        return solver.J_world.numpy()[world, :count]
    # Split mode keeps each articulation's columns in its size group.
    jacobian = np.zeros((count, solver.max_world_dofs))
    worlds, offsets = solver.art_to_world.numpy(), solver.articulation_world_dof_offset.numpy()
    for size in solver.size_groups:
        group_jacobian = solver.J_by_size[size].numpy()
        for group, art in enumerate(solver.group_to_art[size].numpy()):
            if worlds[art] == world:
                offset = int(offsets[art])
                jacobian[:, offset : offset + size] = group_jacobian[group, :count]
    return jacobian


def contact_rows(solver, contacts, route=0, *, incident=False, jacobian=False):
    """Return each routed contact's rows keyed by its contact index.

    Each record holds the contact's row types; row parents as contact indices
    (the next normal of the patch ring for the normal row, the owning contact
    for friction rows, -1 for none); mu and the right-hand side per row; the
    normal row's phi and target; and the patch view's weight, tangential phi
    and friction location when patches are enabled. ``incident`` adds the
    native sparse route's separately stored ``J v_hat`` to the right-hand
    side, matching the assembled split rows. ``jacobian`` adds the physical
    row Jacobians.
    """
    count = min(int(contacts.rigid_contact_count.numpy()[0]), solver.contact_slot.shape[0])
    slot, path = solver.contact_slot.numpy()[:count], solver.contact_path.numpy()[:count]
    world, needed = solver.contact_world.numpy()[:count], solver.contact_slots_needed.numpy()[:count]
    arrays = route_arrays(solver, route)
    if incident and route == 0 and solver._sparse_mass_matrix_size is not None:
        arrays["rhs"] = arrays["rhs"] + solver._sparse_row_incident.numpy()
    patches = getattr(solver, "_friction_patches", None)
    view = patches.view if patches is not None and patches.view.weight.shape[0] else None
    routed = [c for c in range(count) if path[c] == route and slot[c] >= 0 and needed[c] > 0]
    owner = {}
    for c in routed:
        for row in range(int(slot[c]), int(slot[c]) + int(needed[c])):
            owner[(int(world[c]), row)] = c
    jacobians = {}
    records = {}
    for c in routed:
        w, first = int(world[c]), int(slot[c])
        rows = slice(first, first + int(needed[c]))
        parents = arrays["parent"][w, rows]
        record = {
            "types": arrays["type"][w, rows].copy(),
            "parents": np.array([owner.get((w, int(p)), -2) if p >= 0 else -1 for p in parents], dtype=np.int64),
            "mu": arrays["mu"][w, rows].copy(),
            "rhs": arrays["rhs"][w, rows].copy(),
            "phi": float(arrays["phi"][w, first]),
            "target": None if arrays["target"] is None else float(arrays["target"][w, first]),
            "patch": None,
        }
        if view is not None:
            record["patch"] = np.concatenate(
                ([view.weight.numpy()[c]], view.phi.numpy()[c], view.point_a.numpy()[c])
            ).astype(np.float64)
        if jacobian:
            if w not in jacobians:
                jacobians[w] = _jacobian(solver, route, w, int(arrays["count"][w]))
            record["jacobian"] = jacobians[w][rows].copy()
        records[c] = record
    return records


def assert_rows_close(a, b, *, atol=3e-5, rtol=3e-4):
    """Require the same contacts, row types and ring links, and close row values.

    Float fields that one record does not hold (for example a route without a
    stored target) are skipped.
    """
    if sorted(a) != sorted(b):
        raise AssertionError(f"Routed contacts differ: {sorted(set(a) ^ set(b))}")
    for c in sorted(a):
        first, second = a[c], b[c]
        for name in ("types", "parents"):
            np.testing.assert_array_equal(first[name], second[name], err_msg=f"contact {c} {name}")
        for name in ("mu", "rhs", "phi", "target", "patch", "jacobian"):
            if first.get(name) is None or second.get(name) is None:
                continue
            np.testing.assert_allclose(first[name], second[name], atol=atol, rtol=rtol, err_msg=f"contact {c} {name}")


def ring_load(parents, impulses, first):
    """Sum the normal impulses of the patch ring that starts at row ``first``."""
    load, row = float(impulses[first]), int(parents[first])
    while row >= 0 and row != first:
        load += float(impulses[row])
        row = int(parents[row])
    return load


def assert_ring_coulomb(solver, route=0, *, atol=2e-5):
    """Bound every friction pair by its coefficient times the pooled ring normal load."""
    arrays = route_arrays(solver, route)
    pairs = 0
    for w, count in enumerate(arrays["count"]):
        kinds, parents = arrays["type"][w, :count], arrays["parent"][w, :count]
        mu, impulses = arrays["mu"][w, :count], arrays["impulse"][w, :count]
        for row in np.flatnonzero(kinds == PGS_CONSTRAINT_TYPE_FRICTION):
            parent = int(parents[row])
            if row != parent + 1 or kinds[parent] != PGS_CONSTRAINT_TYPE_CONTACT:
                continue
            pairs += 1
            budget = float(mu[row]) * ring_load(parents, impulses, parent)
            magnitude = float(np.linalg.norm(impulses[row : row + 2]))
            if magnitude > budget + atol:
                raise AssertionError(
                    f"World {w} friction rows {row}, {row + 1}: |lambda_t| {magnitude} exceeds pooled budget {budget}"
                )
    return pairs


def assert_history_close(a, b, *, atol=2e-5, rtol=7e-4):
    """Require identical anchor validity and matches, and close carried history.

    ``a`` and ``b`` are solvers or their patch states. Frames are indexed by
    contact, so both must have stepped the same contacts.
    """
    first, second = (getattr(x, "_friction_patches", x) for x in (a, b))
    valid = first.previous.valid.numpy()
    np.testing.assert_array_equal(valid, second.previous.valid.numpy(), err_msg="previous.valid")
    np.testing.assert_array_equal(
        first.current.source.numpy() >= 0, second.current.source.numpy() >= 0, err_msg="current.source >= 0"
    )
    live = valid != 0
    for name in ("displacement", "tangent_impulse"):
        np.testing.assert_allclose(
            getattr(first.previous, name).numpy()[live],
            getattr(second.previous, name).numpy()[live],
            atol=atol,
            rtol=rtol,
            err_msg=f"previous.{name}",
        )


def make_loaded(model, state, contacts, *, margin=0.005, speed=0.01):
    """Start near rest inside a loaded contact envelope, so anchors stick and carry impulse.

    Applies to explicitly written contacts; a collision pipeline rewrites the
    margins at its next call.
    """
    state.joint_qd.assign(state.joint_qd.numpy() * np.float32(speed))
    contacts.rigid_contact_margin0.fill_(margin)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
