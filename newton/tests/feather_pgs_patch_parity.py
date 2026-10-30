# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare persistent patch friction across FeatherPGS routes, independent of slot order.

Contact slots come from per-world atomic reservations, and a patch normal row's
parent names the slot of the next normal of its patch. These helpers key rows
and anchor history by contact index instead, so two solvers that build the same
rows in a different slot order compare equal.

Keying anchor rows by contact is sound only where one member carries each
anchor. On a rectangular footprint the members of a support edge tie, and
float32 rounding of their positions picks the member that carries the edge's
friction rows; the anchor's location, the edge mean, does not move. The
:func:`patch_rows` family keys normal rows by contact and anchor rows by their
friction location, and compares impulses and history where they are well posed.
This module holds shared test helpers; it defines no tests.
"""

import contextlib
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.friction_patches import FrictionPatches, PatchRowLaw, seed_patch_routes
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION
from newton.tests.feather_pgs_free_body_rows import free_body_jacobians
from newton.tests.feather_pgs_law_harness import unpacked_rows

# Per route: constraint count, row type, parent, mu, right-hand side, impulse, phi and
# target arrays. Route 1 (matrix-free rows) folds its target into the right-hand side.
_ROUTE_ARRAYS = {
    0: ("constraint_count", "row_type", "row_parent", "row_mu", "rhs", "impulses", "phi", "target_velocity"),
    1: ("mf_constraint_count", "mf_row_type", "mf_row_parent", "mf_row_mu", "mf_rhs", "mf_impulses", "mf_phi", None),
}


def row_law(patches, beta=0.0):
    """The friction row law of a patch view and its beta (``friction_anchor_beta``)."""
    law = PatchRowLaw()
    law.patches = patches
    law.beta = beta
    return law


def point_row_law(device, count):
    """Point friction's row law: the patch law with beta 0, a disabled view and a zero phi for ``count`` contacts."""
    patches = FrictionPatches()
    patches.phi = wp.zeros(count, dtype=wp.vec2, device=device)
    return row_law(patches)


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
        J_a, J_b = free_body_jacobians(solver)
        return np.concatenate([J_a[world, :count], J_b[world, :count]], axis=1)
    if solver._sparse_mass_matrix_size is not None:
        from newton.tests.test_feather_pgs_sparse_solver import TestFeatherPGSSparseSolver  # noqa: PLC0415

        return TestFeatherPGSSparseSolver()._sparse_rows_physical(solver, world, count)[1]
    if solver._matrix_free:
        return unpacked_rows(solver)[0][world, :count]
    # Split mode keeps each articulation's columns in its size group; the world's
    # articulations follow in index order (split construction maps no world offsets).
    worlds = solver.forest.art_to_world.numpy()
    blocks = []
    for size in solver.forest.size_groups:
        group_jacobian = solver.J_by_size[size].numpy()
        for group, art in enumerate(solver.forest.group_to_art[size].numpy()):
            if worlds[art] == world:
                blocks.append((int(art), group_jacobian[group, :count]))
    return np.concatenate(
        [block for _art, block in sorted(blocks, key=lambda item: item[0])] or [np.zeros((count, 0))], axis=1
    )


def contact_rows(solver, contacts, route=0, *, incident=False, jacobian=False, every_row=False):
    """Return each routed contact's rows keyed by its contact index.

    Each record holds the contact's row types; row parents as contact indices
    (the next normal of the patch ring for the normal row, the owning contact
    for friction rows, -1 for none); mu, the right-hand side and the solved
    impulse per row; the normal row's phi and target; and the patch view's
    weight, tangential phi and friction location when patches are enabled.
    ``incident`` adds the native sparse route's separately stored ``J v_hat``
    to the right-hand side, matching the assembled split rows. ``jacobian``
    adds the physical row Jacobians. ``every_row`` records phi and the target
    of each of the contact's rows, tangent rows included, instead of the
    normal row's only.
    """
    count = min(int(contacts.rigid_contact_count.numpy()[0]), solver.contact_slot.shape[0])
    slot, path = solver.contact_slot.numpy()[:count], solver.contact_path.numpy()[:count]
    world, needed = solver.contact_world.numpy()[:count], solver.contact_slots_needed.numpy()[:count]
    arrays = route_arrays(solver, route)
    if incident and route == 0 and solver._sparse_mass_matrix_size is not None:
        arrays["rhs"] = arrays["rhs"] + solver._sparse_row_incident.numpy()
    patches = solver.laws.friction_patches
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
        phi, target = arrays["phi"], arrays["target"]
        record = {
            "types": arrays["type"][w, rows].copy(),
            "parents": np.array([owner.get((w, int(p)), -2) if p >= 0 else -1 for p in parents], dtype=np.int64),
            "mu": arrays["mu"][w, rows].copy(),
            "rhs": arrays["rhs"][w, rows].copy(),
            "impulses": arrays["impulse"][w, rows].copy(),
            "phi": phi[w, rows].copy() if every_row else float(phi[w, first]),
            "target": None if target is None else (target[w, rows].copy() if every_row else float(target[w, first])),
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
    stored target) are skipped. Solved impulses are left to
    :func:`assert_contact_impulses_close`.
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


def assert_contact_impulses_close(a, b, *, atol=2e-5, rtol=7e-4, load_rtol=0.0):
    """Require the same contacts and close solved impulses on each contact's rows.

    ``a`` and ``b`` are :func:`contact_rows` records, so impulses compare by
    contact rather than by slot. The default tolerance is the patch history
    contract of :func:`assert_history_close`. ``load_rtol`` widens each
    contact's absolute tolerance by that fraction of its normal impulse (the
    larger of the two records'), for contacts whose saturated friction follows
    a changing normal load.
    """
    if sorted(a) != sorted(b):
        raise AssertionError(f"Routed contacts differ: {sorted(set(a) ^ set(b))}")
    for c in sorted(a):
        load = max(abs(float(a[c]["impulses"][0])), abs(float(b[c]["impulses"][0])))
        np.testing.assert_allclose(
            a[c]["impulses"],
            b[c]["impulses"],
            atol=atol + load_rtol * load,
            rtol=rtol,
            err_msg=f"contact {c} impulses",
        )


def ring_load(parents, impulses, first):
    """Sum the normal impulses of the patch ring that starts at row ``first``.

    A ring visits each live row at most once, so a walk that has not returned
    to ``first`` (or ended at -1) after that many links is a malformed ring:
    raise instead of following it forever.
    """
    load, row = float(impulses[first]), int(parents[first])
    for _ in range(len(parents)):
        if row < 0 or row == first:
            return load
        if row >= len(parents):
            raise AssertionError(f"Patch ring from row {first} links to row {row}, past the {len(parents)} live rows")
        load += float(impulses[row])
        row = int(parents[row])
    raise AssertionError(f"Patch ring from row {first} does not close within {len(parents)} rows")


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
    contact, so both must have stepped the same contacts with the same contact
    capacity.
    """
    first, second = (x.laws.friction_patches if hasattr(x, "laws") else x for x in (a, b))
    if first.capacity != second.capacity:
        raise AssertionError(
            f"Patch histories hold {first.capacity} and {second.capacity} contact slots; construct both solvers "
            "with the same model.rigid_contact_max (build the collision pipeline before the solver)"
        )
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


def assert_loaded_anchors(solver, contacts, *, carried=True, reset_worlds=()):
    """Require every ring-linked anchor to hold a valid anchor and, where expected, carried history.

    On a loaded sticking fixture each patch keeps its anchors from step to
    step, so the anchor history that parity checks compare stays populated.
    ``carried`` expects every anchor outside ``reset_worlds`` to have matched
    an anchor of the previous step (``current.source >= 0``); anchors of reset
    worlds start over. Route 0 rows only. Returns the number of ring-linked
    anchors.
    """
    rows = contact_rows(solver, contacts)
    patches = solver.laws.friction_patches
    valid, source = patches.previous.valid.numpy(), patches.current.source.numpy()
    worlds = solver.contact_world.numpy()
    anchors = [c for c, record in rows.items() if len(record["types"]) == 3 and record["parents"][0] not in (-1, c)]
    if not anchors:
        raise AssertionError("No ring-linked contact holds an anchor")
    for c in anchors:
        if not valid[c]:
            raise AssertionError(f"Ring-linked contact {c} lost its anchor")
        expected = carried and int(worlds[c]) not in reset_worlds
        if bool(source[c] >= 0) != expected:
            raise AssertionError(f"Contact {c} carried history: {bool(source[c] >= 0)}, expected {expected}")
    return len(anchors)


def _tangent_basis(normal):
    """Return the tangent pair of FeatherPGS friction rows for ``normal`` (``contact_tangent_basis``)."""
    first = np.cross(normal, (1.0, 0.0, 0.0))
    if np.dot(first, first) < 1.0e-12:
        first = np.cross(normal, (0.0, 1.0, 0.0))
    first = first / np.linalg.norm(first)
    second = np.cross(normal, first)
    return first, second / np.linalg.norm(second)


def patch_rows(solver, contacts, dt, *, jacobian=False):
    """Describe route 0's patch rows independent of which support-edge member carries each anchor.

    ``normals`` keys every routed contact's normal row by contact index: its type,
    its ring successor (a contact index, -1 for none), mu, bias, phi, target and
    solved impulse. ``regions`` keys each patch region by its sorted member
    contacts and lists its anchors, each with the patch view's friction location,
    weight and tangential phi and its friction rows' types, mu, biases and
    impulses (``own`` when both rows link to the anchor's own normal row), plus
    the region's normal load and friction vector. A row's bias is its right-hand
    side without the predicted velocity's ``J v_hat``, which only dense split
    rows add in (matrix-free and sparse split rows keep it apart): ``-target``
    plus the position bias of the row's phi. The record keeps the largest
    position-bias gain of step ``dt`` for normal and anchor rows
    (:func:`assert_patch_rows_close`). ``jacobian`` also records each row's
    physical Jacobian (:func:`contact_rows`).
    """
    baked = not solver._matrix_free
    rows = contact_rows(solver, contacts, jacobian=baked or jacobian)
    patches = solver.laws.friction_patches
    count = min(int(contacts.rigid_contact_count.numpy()[0]), solver.contact_slot.shape[0])
    owner = patches.current.owner.numpy()[:count]
    weight = patches.view.weight.numpy()[:count]
    location = patches.view.point_a.numpy()[:count].astype(np.float64)
    tangent_phi = patches.view.phi.numpy()[:count].astype(np.float64)
    normal = contacts.rigid_contact_normal.numpy()[:count].astype(np.float64)
    worlds = solver.contact_world.numpy()[:count]
    velocities = {}

    def bias(c, row):
        if not baked:
            return row["rhs"].astype(np.float64)
        w = int(worlds[c])
        if w not in velocities:
            velocities[w] = _predicted_velocity(solver, w)
        return row["rhs"].astype(np.float64) - row["jacobian"].astype(np.float64) @ velocities[w]

    record = {
        "normals": {},
        "regions": {},
        "gain": max(float(solver.options.pgs_beta), float(solver.options.contact_speculative_scale)) / dt,
        "friction_gain": float(solver.laws.friction_anchor_beta) / dt,
    }
    members = {}
    for c, row in rows.items():
        members.setdefault(int(owner[c]), []).append(c)
        row["bias"] = bias(c, row)
        record["normals"][c] = {
            "type": int(row["types"][0]),
            "next": int(row["parents"][0]),
            "mu": float(row["mu"][0]),
            "bias": float(row["bias"][0]),
            "phi": row["phi"],
            "target": row["target"],
            "impulse": float(row["impulses"][0]),
            "jacobian": row["jacobian"][0] if jacobian else None,
        }
    for region in members.values():
        anchors, friction = [], np.zeros(3)
        for c in region:
            row = rows[c]
            if len(row["types"]) != 3:
                continue
            first, second = _tangent_basis(normal[c])
            vector = float(row["impulses"][1]) * first + float(row["impulses"][2]) * second
            friction += vector
            anchors.append(
                {
                    "contact": c,
                    "location": location[c],
                    "weight": float(weight[c]),
                    "phi": tangent_phi[c],
                    "types": row["types"][1:].copy(),
                    "own": bool(np.all(row["parents"][1:] == c)),
                    "mu": row["mu"][1:].astype(np.float64),
                    "bias": row["bias"][1:],
                    "impulses": row["impulses"][1:].astype(np.float64),
                    "jacobian": row["jacobian"][1:] if jacobian else None,
                }
            )
        record["regions"][tuple(sorted(region))] = {
            "anchors": anchors,
            "load": sum(float(rows[c]["impulses"][0]) for c in region),
            "friction": friction,
        }
    return record


def _predicted_velocity(solver, world):
    """Return a world's predicted generalized velocity in the column order of its dense split row Jacobians."""
    v_hat = solver.v_hat.numpy().astype(np.float64)
    # The world's articulations in index order, as :func:`_jacobian` lays them out.
    worlds, starts = solver.forest.art_to_world.numpy(), solver.forest.articulation_dof_start.numpy()
    blocks = []
    for size in solver.forest.size_groups:
        for art in solver.forest.group_to_art[size].numpy():
            if worlds[art] == world:
                blocks.append((int(art), v_hat[int(starts[art]) : int(starts[art]) + size]))
    return np.concatenate([block for _art, block in sorted(blocks, key=lambda item: item[0])] or [np.zeros(0)])


def _match_anchors(first, second):
    """Pair two regions' anchors one to one, each with the nearest remaining friction location."""
    pairs, remaining = [], list(range(len(second)))
    for anchor in first:
        nearest = min(remaining, key=lambda j: float(np.linalg.norm(second[j]["location"] - anchor["location"])))
        remaining.remove(nearest)
        pairs.append((anchor, second[nearest]))
    return pairs


def assert_patch_rows_close(a, b, *, atol=3e-5, rtol=3e-4, lifecycle=True):
    """Require the same patch rows on two routes, whichever member of a support edge carries an anchor.

    ``a`` and ``b`` are :func:`patch_rows` records. Normal rows compare by contact
    and anchor rows by friction location: row types, ring links, mu (divided on
    anchor rows), phi, target, patch weight, location and, where both records
    hold them, row Jacobians within ``atol`` and ``rtol``. A row's bias follows its phi through the position-bias gain, so its
    tolerance adds the gain times the records' phi difference: one float32 ulp of
    a foot's pose moves phi by about 4e-7 and the bias of a gap row by 400 times
    that. ``lifecycle`` also compares each anchor's tangential phi, which its
    carried history decides; leave it off where anchors release at the friction
    cone's boundary (sliding feet).
    """
    if sorted(a["normals"]) != sorted(b["normals"]):
        raise AssertionError(f"Routed contacts differ: {sorted(set(a['normals']) ^ set(b['normals']))}")
    for c in sorted(a["normals"]):
        first, second = a["normals"][c], b["normals"][c]
        for name in ("type", "next"):
            if first[name] != second[name]:
                raise AssertionError(f"contact {c} normal row {name}: {first[name]} != {second[name]}")
        for name in ("mu", "phi", "target", "jacobian"):
            if first[name] is not None and second[name] is not None:
                np.testing.assert_allclose(
                    first[name], second[name], atol=atol, rtol=rtol, err_msg=f"contact {c} {name}"
                )
        follow = a["gain"] * abs(first["phi"] - second["phi"])
        np.testing.assert_allclose(
            first["bias"], second["bias"], atol=atol + follow, rtol=rtol, err_msg=f"contact {c} bias"
        )
    if sorted(a["regions"]) != sorted(b["regions"]):
        raise AssertionError(f"Patch regions differ: {sorted(set(a['regions']) ^ set(b['regions']))}")
    for key in sorted(a["regions"]):
        anchors = a["regions"][key]["anchors"], b["regions"][key]["anchors"]
        if len(anchors[0]) != len(anchors[1]):
            raise AssertionError(f"Region {key} holds {len(anchors[0])} and {len(anchors[1])} anchors")
        for first, second in _match_anchors(*anchors):
            label = f"region {key} anchor at {np.round(first['location'], 4).tolist()}"
            np.testing.assert_allclose(
                first["location"], second["location"], atol=atol, rtol=rtol, err_msg=f"{label} location"
            )
            np.testing.assert_array_equal(first["types"], second["types"], err_msg=f"{label} types")
            if first["jacobian"] is not None and second["jacobian"] is not None:
                np.testing.assert_allclose(
                    first["jacobian"], second["jacobian"], atol=atol, rtol=rtol, err_msg=f"{label} jacobian"
                )
            if not (first["own"] and second["own"]):
                raise AssertionError(f"{label}: friction rows must link to the anchor's own normal row")
            np.testing.assert_allclose(
                [first["weight"], *first["mu"]],
                [second["weight"], *second["mu"]],
                atol=atol,
                rtol=rtol,
                err_msg=f"{label} weight and mu",
            )
            follow = a["friction_gain"] * float(np.abs(first["phi"] - second["phi"]).max())
            np.testing.assert_allclose(
                first["bias"], second["bias"], atol=atol + follow, rtol=rtol, err_msg=f"{label} bias"
            )
            if lifecycle:
                np.testing.assert_allclose(
                    first["phi"], second["phi"], atol=atol, rtol=rtol, err_msg=f"{label} tangential phi"
                )


def assert_patch_impulses_close(a, b, *, atol=2e-5, rtol=7e-4):
    """Require close solved impulses per normal row and per anchor, keyed as by :func:`assert_patch_rows_close`.

    Sound where each region's impulses are determined, such as point feet or a
    single bouncing foot. On a stance whose contact rows outnumber its degrees
    of freedom, float order redistributes load among the rows; compare region
    sums there (:func:`assert_region_impulses_close`). Call after
    :func:`assert_patch_rows_close`, which matches contacts and anchors.
    """
    for c in sorted(a["normals"]):
        np.testing.assert_allclose(
            a["normals"][c]["impulse"], b["normals"][c]["impulse"], atol=atol, rtol=rtol, err_msg=f"contact {c} impulse"
        )
    for key in sorted(a["regions"]):
        for first, second in _match_anchors(a["regions"][key]["anchors"], b["regions"][key]["anchors"]):
            np.testing.assert_allclose(
                first["impulses"],
                second["impulses"],
                atol=atol,
                rtol=rtol,
                err_msg=f"region {key} anchor at {np.round(first['location'], 4).tolist()} impulses",
            )


def assert_region_impulses_close(a, b, *, load_atol, friction_atol):
    """Bound the difference of each patch region's normal load and friction vector between two routes.

    ``a`` and ``b`` are :func:`patch_rows` records; the bounds are spreads
    measured between routes whose inputs differ by float32 ulps (see
    :data:`STANDING_REGION_BOUNDS`). ``friction_atol`` None skips the friction
    vectors, which a stance whose contact rows outnumber its degrees of freedom
    does not determine.
    """
    if sorted(a["regions"]) != sorted(b["regions"]):
        raise AssertionError(f"Patch regions differ: {sorted(set(a['regions']) ^ set(b['regions']))}")
    for key in sorted(a["regions"]):
        first, second = a["regions"][key], b["regions"][key]
        if abs(first["load"] - second["load"]) > load_atol:
            raise AssertionError(
                f"Region {key} normal load {first['load']} vs {second['load']} differs by more than {load_atol}"
            )
        if friction_atol is None:
            continue
        difference = float(np.linalg.norm(first["friction"] - second["friction"]))
        if difference > friction_atol:
            raise AssertionError(
                f"Region {key} friction {first['friction'].tolist()} vs {second['friction'].tolist()}: "
                f"|difference| {difference} exceeds {friction_atol}"
            )


def anchor_sites(solver, contacts):
    """List the stored anchors by site: body, body-frame location, carried flag and carried history.

    A site does not change when float32 rounding moves an anchor's rows to
    another member of its support edge, unlike the contact index that carries it.
    """
    patches = solver.laws.friction_patches
    previous = patches.previous
    count = min(int(contacts.rigid_contact_count.numpy()[0]), patches.capacity)
    valid = previous.valid.numpy()[:count]
    body = previous.body_a.numpy()[:count]
    anchor = previous.anchor_a.numpy()[:count].astype(np.float64)
    displacement = previous.displacement.numpy()[:count].astype(np.float64)
    tangent = previous.tangent_impulse.numpy()[:count].astype(np.float64)
    source = patches.current.source.numpy()[:count]
    return [
        {
            "body": int(body[c]),
            "anchor": anchor[c],
            "carried": bool(source[c] >= 0),
            "displacement": displacement[c],
            "tangent": tangent[c],
        }
        for c in np.flatnonzero(valid)
    ]


def assert_anchor_sites_close(a, b, *, atol=3e-5, rtol=3e-4, history=(2e-5, 7e-4), tangent=True):
    """Require the same stored anchors on two routes and close carried history.

    ``a`` and ``b`` are :func:`anchor_sites` lists, paired by body and nearest
    location; locations agree within ``atol`` and ``rtol`` and carried flags
    exactly. ``history`` is the (atol, rtol) of the stored displacement and,
    with ``tangent``, of the stored tangent impulse; None skips both. Leave
    ``tangent`` off where float order redistributes friction among a stance's
    anchors; :func:`assert_history_matches_rows` still ties each route's stored
    impulses to its solved rows.
    """
    if len(a) != len(b):
        raise AssertionError(f"Stored anchors differ in number: {len(a)} and {len(b)}")
    remaining = list(range(len(b)))
    for site in a:
        candidates = [j for j in remaining if b[j]["body"] == site["body"]]
        if not candidates:
            raise AssertionError(f"No stored anchor of body {site['body']} pairs with {site['anchor'].tolist()}")
        nearest = min(candidates, key=lambda j: float(np.linalg.norm(b[j]["anchor"] - site["anchor"])))
        remaining.remove(nearest)
        other = b[nearest]
        label = f"body {site['body']} anchor at {np.round(site['anchor'], 4).tolist()}"
        np.testing.assert_allclose(site["anchor"], other["anchor"], atol=atol, rtol=rtol, err_msg=f"{label} location")
        if site["carried"] != other["carried"]:
            raise AssertionError(f"{label} carried history: {site['carried']} and {other['carried']}")
        if history is None:
            continue
        names = ("displacement", "tangent") if tangent else ("displacement",)
        for name in names:
            np.testing.assert_allclose(
                site[name], other[name], atol=history[0], rtol=history[1], err_msg=f"{label} stored {name}"
            )


def assert_history_matches_rows(solver, contacts, *, atol=1e-6, rtol=1e-6):
    """Require each stored anchor with route 0 rows to hold the tangent impulse its rows solved.

    Compares magnitudes, which the frame of the stored impulse does not change
    beyond float32 rounding of its rotation into the body frame. Returns the
    number of anchors checked.
    """
    patches = solver.laws.friction_patches
    count = min(int(contacts.rigid_contact_count.numpy()[0]), patches.capacity)
    valid = patches.previous.valid.numpy()[:count]
    tangent = patches.previous.tangent_impulse.numpy()[:count]
    slot, world = solver.contact_slot.numpy()[:count], solver.contact_world.numpy()[:count]
    needed, path = solver.contact_slots_needed.numpy()[:count], solver.contact_path.numpy()[:count]
    impulses = solver.impulses.numpy()
    checked = 0
    for c in np.flatnonzero(valid):
        if path[c] != 0 or slot[c] < 0 or needed[c] != 3:
            continue
        w, s = int(world[c]), int(slot[c])
        solved = float(np.hypot(impulses[w, s + 1], impulses[w, s + 2]))
        stored = float(np.linalg.norm(tangent[c]))
        if abs(stored - solved) > atol + rtol * solved:
            raise AssertionError(f"Contact {c} stored tangent impulse {stored}; its friction rows solved {solved}")
        checked += 1
    return checked


def _rotate(quaternion, vector):
    """Rotate ``vector`` by a warp quaternion (x, y, z, w)."""
    axis, w = np.asarray(quaternion[:3], dtype=np.float64), float(quaternion[3])
    twice = 2.0 * np.cross(axis, vector)
    return vector + w * twice + np.cross(axis, twice)


def _seed_record(inputs):
    """Describe one ``seed_patch_routes`` launch: route 0's scale and each carried route-0 anchor's seed."""
    extent, frame, previous, body_q, world, slot, path, needed, routes = inputs[:9]
    record = {"scale": None, "anchors": [], "impulses": None}
    if not routes.enabled[0]:
        return record
    record["scale"], record["impulses"] = float(routes.scale[0]), routes.impulses0
    live = int(extent.numpy()[1])
    source, body = frame.source.numpy()[:live], frame.body_a.numpy()[:live]
    flipped, normal = frame.flipped.numpy()[:live], frame.normal.numpy()[:live].astype(np.float64)
    world, slot, path, needed = (array.numpy()[:live] for array in (world, slot, path, needed))
    stored, poses = previous.tangent_impulse.numpy().astype(np.float64), body_q.numpy().astype(np.float64)
    parents, mu, impulses = routes.parents0.numpy(), routes.mu0.numpy(), routes.impulses0.numpy()
    for c in np.flatnonzero((source >= 0) & (slot >= 0) & (path == 0) & (needed == 3)):
        tangent = stored[source[c]] * record["scale"]
        if body[c] >= 0:
            tangent = _rotate(poses[body[c], 3:], tangent)
        n = normal[c]
        if flipped[c] != 0:
            n, tangent = -n, -tangent
        first, second = _tangent_basis(n)
        value = np.array([tangent @ first, tangent @ second])
        w, s = int(world[c]), int(slot[c])
        budget = max(float(mu[w, s + 1]) * ring_load(parents[w], impulses[w], s), 0.0)
        magnitude = float(np.linalg.norm(value))
        if magnitude > budget and magnitude > 0.0:
            value *= budget / magnitude
        record["anchors"].append(
            {
                "contact": int(c),
                "world": w,
                "slot": s,
                "expected": value,
                "seeded": impulses[w, s + 1 : s + 3].astype(np.float64),
                "swept": None,
            }
        )
    return record


@contextlib.contextmanager
def recorded_patch_seeds():
    """Record each patch seed launch inside the block and the friction rows the next PGS sweep starts from.

    Yields a list that receives one record per ``seed_patch_routes`` launch:
    route 0's warm-start ``scale`` (None when the launch leaves route 0 out)
    and, for each anchor that carries history into a route-0 row, its
    ``expected`` seed, the ``seeded`` friction rows the launch wrote and the
    ``swept`` rows the first ``pgs_solve*`` kernel launched after it reads.
    The expected seed follows the patch seeding law: the stored tangent
    impulse times the scale, rotated by the body pose, projected on the
    contact's tangent pair and clamped to the anchor's pooled budget (divided
    mu times its ring's seeded normal load).
    """
    records = []

    def spy(launch):
        def run(kernel, *args, **kwargs):
            if records and records[-1]["impulses"] is not None and kernel.key.startswith("sweep_"):
                record = records[-1]
                impulses, record["impulses"] = record["impulses"].numpy(), None
                for anchor in record["anchors"]:
                    s = anchor["slot"]
                    anchor["swept"] = impulses[anchor["world"], s + 1 : s + 3].astype(np.float64)
            result = launch(kernel, *args, **kwargs)
            if kernel is seed_patch_routes:
                records.append(_seed_record(kwargs["inputs"]))
            return result

        return run

    # The solver modules call warp through its module attributes, and launch_tiled does not go through
    # wp.launch, so both are spied on.
    with mock.patch.object(wp, "launch", spy(wp.launch)), mock.patch.object(wp, "launch_tiled", spy(wp.launch_tiled)):
        yield records


def assert_patch_seeds(records, *, scale, carried=True):
    """Require one step's patch seed to follow the seeding law and to start the sweep.

    ``records`` come from :func:`recorded_patch_seeds` around one step: the
    seed launches once with route 0's warm-start ``scale``, each carried
    anchor's friction rows hold its expected seed within float32 rounding of
    the law, and the PGS sweep reads them unchanged. ``carried`` expects at
    least one anchor to carry history into the seed. Returns the largest
    expected seed magnitude.
    """
    if len(records) != 1:
        raise AssertionError(f"A step must launch the patch seed once, not {len(records)} times")
    (record,) = records
    if record["scale"] is None:
        raise AssertionError("The patch seed launch leaves out route 0's rows")
    np.testing.assert_allclose(record["scale"], scale, rtol=1e-6, err_msg="route 0 warm-start scale")
    if carried and not record["anchors"]:
        raise AssertionError("No route-0 anchor carries history into the seed")
    largest = 0.0
    for anchor in record["anchors"]:
        label = f"contact {anchor['contact']}"
        np.testing.assert_allclose(
            anchor["seeded"], anchor["expected"], atol=1e-6, rtol=1e-5, err_msg=f"{label} seeded friction rows"
        )
        if anchor["swept"] is None:
            raise AssertionError("No PGS sweep launched after the patch seed")
        np.testing.assert_array_equal(
            anchor["swept"], anchor["seeded"], err_msg=f"{label} friction rows the sweep starts from"
        )
        largest = max(largest, float(np.linalg.norm(anchor["expected"])))
    return largest


def assert_factor_rows_consistent(z, jacobian, impulses, incident, delta, v_hat, v_out, *, label=""):
    """Tie one world's native sparse rows, factor delta and velocity to its solved impulses.

    ``z`` holds the rows in factor coordinates and ``jacobian`` the physical
    rows recovered from them, both ``[rows, world dofs]``. The sparse sweep
    reads each row's ``incident`` (``J v_hat``) apart from its bias, keeps the
    impulses' response as the factor-coordinate ``delta`` (seeded from the
    carried impulses when warm starting) and decodes ``v_out`` from it. So the
    incident must equal ``jacobian @ v_hat``, ``delta`` must equal
    ``z.T @ impulses``, and ``jacobian @ v_out`` must equal ``z @ delta`` plus
    the incident. Tolerances are those of the sparse solver's own state check.
    """
    np.testing.assert_allclose(jacobian @ v_hat, incident, atol=2e-5, rtol=2e-5, err_msg=f"{label} row incident")
    np.testing.assert_allclose(delta, z.T @ impulses, atol=3e-5, rtol=3e-4, err_msg=f"{label} factor delta")
    np.testing.assert_allclose(
        jacobian @ v_out, z @ delta + incident, atol=3e-5, rtol=3e-5, err_msg=f"{label} decoded velocity"
    )


def assert_sparse_rows_consistent(solver):
    """Apply :func:`assert_factor_rows_consistent` to every world of a native sparse split step.

    Returns the number of rows checked.
    """
    from newton.tests.test_feather_pgs_sparse_solver import TestFeatherPGSSparseSolver  # noqa: PLC0415

    recover = TestFeatherPGSSparseSolver()._sparse_rows_physical
    impulses, incident = solver.impulses.numpy(), solver._sparse_row_incident.numpy()
    delta, dofs = solver._sparse_factor_velocity_delta.numpy(), solver.forest.world_dof_indices.numpy()
    v_hat, v_out = solver.v_hat.numpy(), solver.v_out.numpy()
    checked = 0
    for world, live in enumerate(solver.constraint_count.numpy()):
        count = int(live)
        z, jacobian = recover(solver, world, count)
        velocity = dofs[world]
        assert_factor_rows_consistent(
            z,
            jacobian,
            impulses[world, :count],
            incident[world, :count],
            delta[world],
            v_hat[velocity],
            v_out[velocity],
            label=f"world {world}",
        )
        checked += count
    return checked


def ring_budget_ratios(solver, route=0):
    """Return each friction pair's |lambda_t| over its coefficient times the pooled ring normal load."""
    arrays = route_arrays(solver, route)
    ratios = []
    for w, count in enumerate(arrays["count"]):
        kinds, parents = arrays["type"][w, :count], arrays["parent"][w, :count]
        mu, impulses = arrays["mu"][w, :count], arrays["impulse"][w, :count]
        for row in np.flatnonzero(kinds == PGS_CONSTRAINT_TYPE_FRICTION):
            parent = int(parents[row])
            if row != parent + 1 or kinds[parent] != PGS_CONSTRAINT_TYPE_CONTACT:
                continue
            budget = float(mu[row]) * ring_load(parents, impulses, parent)
            if budget > 0.0:
                ratios.append(float(np.linalg.norm(impulses[row : row + 2])) / budget)
    return np.asarray(ratios)


def assert_states_close(first, second, *, velocity_atol=2.0e-3):
    """Require close states: poses within 2e-4 and twists within ``velocity_atol``, both relative 2e-4.

    The default is the sparse-solver state contract; the standing feet compare
    twists at :data:`STANDING_VELOCITY_ATOL`.
    """
    for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
        np.testing.assert_allclose(
            getattr(first, name).numpy(),
            getattr(second, name).numpy(),
            rtol=2.0e-4,
            atol=velocity_atol if name.endswith("qd") else 2.0e-4,
            err_msg=name,
        )


# The box feet rest on their corners at zero gap, where the position correction's 1/dt gain turns float32
# rounding of a foot's pose into joint velocity, momentum and contact impulse. Two routes that differ in float
# order drift apart from step to step; on CPU, two dense split arms of which one moves every body pose
# component its rows read by +-1 ulp at every step spread, at most over 32 noise seeds:
# - standing (8 PGS iterations, the 24-step masked-reset schedule): joint velocity 3.6e-3, body twist 3.1e-3,
#   momentum 3.7e-3, region normal load 1.6e-3, region friction 4.2e-3, poses 8.1e-6, row Jacobians 5.3e-6
#   (a torso started one ulp higher without further noise: joint velocity 2.3e-3, load 1.4e-3);
# - sliding (SLIDING_FEET, SLIDING_ITERATIONS, 24 steps): joint velocity 1.7e-3, region load 1.1e-3, region
#   friction 1.0e-4.
# The bounds below are at least twice these. They are CPU measurements: re-measure on CUDA (dense against
# sparse split rows, the ordinary against the incremental matrix-free sweep) before tightening them. They
# assume that both arms reserve each world's contact rows in one order, as sorted contacts within one CUDA
# warp do: dealing one arm's rows in a random per-world order alone moves standing joint velocity by up to
# 1.2e-2 and region load by 5.9e-3 on CPU.
# Twists and momentum of the standing and sliding feet.
STANDING_VELOCITY_ATOL = 7.5e-3
# The stance's contact rows outnumber its degrees of freedom, so float order shifts friction between its
# feet by more than any region carries (3.5e-3 at most): stance regions compare normal load only, and the
# sliding feet check friction.
STANDING_REGION_BOUNDS = {"load_atol": 3.5e-3, "friction_atol": None}
# Every sliding anchor sits at its pooled budget, so kinetic friction fixes each region's friction at its
# coefficient times the region's load: an anchor budget that pools only its own normal moves it by 6.5e-3.
SLIDING_REGION_BOUNDS = {"load_atol": 2.5e-3, "friction_atol": 3.0e-4}
# Low-friction feet (feet and ground mu 0.1) start sliding forward at 0.5 m/s and keep sliding.
SLIDING_FEET = {"mu": 0.1, "speed": 0.5}
# Iterations that keep every sliding anchor at its pooled budget (8 leave a third of them inside).
SLIDING_ITERATIONS = 32


def standing_model(device="cuda:0", worlds=2, *, mu=None, speed=0.0):
    """Stand a floating torso on two position-held legs with box feet, one articulation per world.

    Each foot rests on four corners, so patch friction links their normals in
    one ring and keeps two loaded, sticking anchors per foot from step to step.
    The tree is branched with a free root, as the sparse routes require.
    ``mu`` sets the friction coefficient of the feet and the ground (default:
    0.8 feet on the default ground), and ``speed`` starts the torso moving
    along +x [m/s], so low-friction feet slide (:data:`SLIDING_FEET`).
    """
    builder = newton.ModelBuilder()
    cfg = builder.default_shape_cfg.copy()
    cfg.density, cfg.mu = 0.0, 0.8 if mu is None else mu
    for _ in range(worlds):
        builder.begin_world()
        torso = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.48), wp.quat_identity()),
            mass=2.0,
            inertia=wp.mat33(np.eye(3) * 0.1),
            lock_inertia=True,
        )
        joints = [builder.add_joint_free(child=torso)]
        builder.joint_qd[-6] = speed
        for side in (-1.0, 1.0):
            parent = torso
            for link in range(2):
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.05), lock_inertia=True)
                offset = wp.vec3(0.0, 0.15 * side, 0.0) if link == 0 else wp.vec3(0.0, 0.0, -0.25)
                joints.append(
                    builder.add_joint_revolute(
                        parent=parent,
                        child=body,
                        axis=newton.Axis.Y if link else newton.Axis.X,
                        parent_xform=wp.transform(offset, wp.quat_identity()),
                        target_ke=400.0,
                        target_kd=20.0,
                    )
                )
                parent = body
            foot = wp.transform(wp.vec3(0.0, 0.0, -0.2), wp.quat_identity())
            builder.add_shape_box(parent, hx=0.06, hy=0.03, hz=0.03, xform=foot, cfg=cfg)
        builder.add_articulation(joints)
        builder.end_world()
    if mu is None:
        builder.add_ground_plane()
    else:
        ground = builder.default_shape_cfg.copy()
        ground.mu = mu
        builder.add_ground_plane(cfg=ground)
    return builder.finalize(device=device)


def make_loaded(model, state, contacts, *, margin=0.005, speed=0.01):
    """Start near rest inside a loaded contact envelope, so anchors stick and carry impulse.

    Applies to explicitly written contacts; a collision pipeline rewrites the
    margins at its next call.
    """
    state.joint_qd.assign(state.joint_qd.numpy() * np.float32(speed))
    contacts.rigid_contact_margin0.fill_(margin)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
