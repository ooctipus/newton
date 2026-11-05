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

import operator

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.friction_patches import PatchRowLaw
from newton._src.solvers.feather_pgs.row_kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION
from newton.tests.feather_pgs_free_body_rows import free_body_jacobians
from newton.tests.feather_pgs_law_harness import unpacked_rows

# Per route: constraint count, row type, parent, mu, right-hand side, impulse, phi and
# target arrays. Route 1 (matrix-free rows, the row store's free bodies' rows) folds its target into the right-hand side.
_ROUTE_ARRAYS = {
    0: ("constraint_count", "row_type", "row_parent", "row_mu", "rhs", "impulses", "phi", "target_velocity"),
    1: ("mf_constraint_count", "row_type", "row_parent", "row_mu", "rhs", "impulses", "phi", None),
}


def row_law(patches, beta=0.0):
    """The friction row law of a patch view and its beta (``friction_anchor_beta``)."""
    law = PatchRowLaw()
    law.patches = patches
    law.beta = beta
    return law


def route_arrays(solver, route=0):
    """Return a route's row arrays as host copies, keyed by role, by the route's own row (the free bodies' from the
    store's body base, their parents too); ``offset`` is the route's first row in the world's line."""
    names = ("count", "type", "parent", "mu", "rhs", "impulse", "phi", "target")
    offset = solver.placement.body_base if route == 1 else 0
    arrays = {
        role: None if attribute is None else operator.attrgetter(attribute)(solver.placement).numpy()
        for role, attribute in zip(names, _ROUTE_ARRAYS[route], strict=True)
    }
    for role in names[1:]:
        if arrays[role] is not None:
            arrays[role] = arrays[role][:, offset:]
    arrays["parent"] = np.where(arrays["parent"] >= 0, arrays["parent"] - offset, arrays["parent"])
    arrays["offset"] = offset
    return arrays


def _jacobian(solver, route, world, count):
    """Physical row Jacobians in world coordinates (route 0) or body-space pairs (route 1)."""
    if route == 1:
        J_a, J_b = free_body_jacobians(solver)
        return np.concatenate([J_a[world, :count], J_b[world, :count]], axis=1)
    if solver.placement.sparse_mass_matrix_size is not None:
        from newton.tests.test_feather_pgs_sparse_solver import TestFeatherPGSSparseSolver  # noqa: PLC0415

        return TestFeatherPGSSparseSolver()._sparse_rows_physical(solver, world, count)[1]
    return unpacked_rows(solver)[0][world, :count]


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
    count = min(int(contacts.rigid_contact_count.numpy()[0]), solver.placement.contact_slot.shape[0])
    slot, path = solver.placement.contact_slot.numpy()[:count], solver.placement.contact_path.numpy()[:count]
    world, needed = (
        solver.placement.contact_world.numpy()[:count],
        solver.placement.contact_slots_needed.numpy()[:count],
    )
    arrays = route_arrays(solver, route)
    if incident and route == 0 and solver.placement.sparse_mass_matrix_size is not None:
        arrays["rhs"] = arrays["rhs"] + solver.placement.row_incident.numpy()
    patches = solver.laws.friction_patches
    view = patches.view if patches is not None and patches.view.weight.shape[0] else None
    routed = [c for c in range(count) if path[c] == route and slot[c] >= 0 and needed[c] > 0]
    # The route's own rows: a slot less the route's first row in the line.
    slot = slot - arrays["offset"]
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
                ([view.weight.numpy()[c]], view.phi.numpy()[c], view.point.numpy()[c])
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
    np.testing.assert_allclose(
        first.previous.displacement.numpy()[live],
        second.previous.displacement.numpy()[live],
        atol=atol,
        rtol=rtol,
        err_msg="previous.displacement",
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
    worlds = solver.placement.contact_world.numpy()
    anchors = [c for c, record in rows.items() if len(record["types"]) >= 3 and record["parents"][0] not in (-1, c)]
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
    side, which keeps the predicted velocity's ``J v_hat`` apart: ``-target``
    plus the position bias of the row's phi. The record keeps the largest
    position-bias gain of step ``dt`` for normal and anchor rows
    (:func:`assert_patch_rows_close`). ``jacobian`` also records each row's
    physical Jacobian (:func:`contact_rows`).
    """
    rows = contact_rows(solver, contacts, jacobian=jacobian)
    patches = solver.laws.friction_patches
    count = min(int(contacts.rigid_contact_count.numpy()[0]), solver.placement.contact_slot.shape[0])
    owner = patches.current.owner.numpy()[:count]
    weight = patches.view.weight.numpy()[:count]
    location = patches.view.point.numpy()[:count].astype(np.float64)
    tangent_phi = patches.view.phi.numpy()[:count].astype(np.float64)
    normal = contacts.rigid_contact_normal.numpy()[:count].astype(np.float64)

    def bias(c, row):
        return row["rhs"].astype(np.float64)

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
            if len(row["types"]) < 3:
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
                first["location"], second["location"], atol=ANCHOR_ATOL, rtol=rtol, err_msg=f"{label} location"
            )
            np.testing.assert_array_equal(first["types"], second["types"], err_msg=f"{label} types")
            if first["jacobian"] is not None and second["jacobian"] is not None:
                np.testing.assert_allclose(
                    first["jacobian"], second["jacobian"], atol=ANCHOR_ATOL, rtol=rtol, err_msg=f"{label} jacobian"
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
    source = patches.current.source.numpy()[:count]
    return [
        {
            "body": int(body[c]),
            "anchor": anchor[c],
            "carried": bool(source[c] >= 0),
            "displacement": displacement[c],
        }
        for c in np.flatnonzero(valid)
    ]


def assert_anchor_sites_close(a, b, *, atol=None, rtol=3e-4, history=(2e-5, 7e-4)):
    """Require the same stored anchors on two routes and close carried history.

    ``a`` and ``b`` are :func:`anchor_sites` lists, paired by body and nearest
    location; locations agree within ``atol`` and ``rtol`` and carried flags
    exactly. ``history`` is the (atol, rtol) of the stored displacement; None
    skips it.
    """
    atol = ANCHOR_ATOL if atol is None else atol
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
        if history is not None:
            np.testing.assert_allclose(
                site["displacement"],
                other["displacement"],
                atol=history[0],
                rtol=history[1],
                err_msg=f"{label} stored displacement",
            )


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
    impulses, incident = solver.placement.impulses.numpy(), solver.placement.row_incident.numpy()
    delta, dofs = solver.placement.sparse_factor_velocity_delta.numpy(), solver.forest.world_dof_indices.numpy()
    v_hat, v_out = solver.v_hat.numpy(), solver.v_out.numpy()
    checked = 0
    for world, live in enumerate(solver.placement.constraint_count.numpy()):
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
ANCHOR_ATOL = 2.0e-3
"""How far two arms' anchors of one region may sit apart [m] (and their rows' Jacobians differ): a region's anchor is its
centre of pressure from the last solve, which a statically indeterminate stance leaves to the solve's float order, so a
one-ulp twin's moves it (measured on the feet: 0.09 mm standing, 1.4 mm sliding)."""

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
