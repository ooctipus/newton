# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The contact front end: what every contact-row producer computes before it writes a row.

One law each, written once for Warp kernels and native CUDA (:mod:`.law_source`):

* endpoints: :func:`contact_bodies` (shape to body), :func:`contact_endpoints`
  (to articulation and world) and :func:`contact_responds`;
* geometry: :func:`contact_geometry` reads a contact from the collision arrays;
  :func:`contact_points` gives the normal from body B to body A, the
  world contact points offset by the shapes' margins along it, and the gap
  ``phi = n . (p_a - p_b)``. A second arithmetic contract is kept as an explicit
  mode: :func:`contact_points_endpoint_relative` subtracts body translations
  apart from the rotated offsets (the resolved-world classifier's rounding);
* materials: :func:`contact_materials`, the friction coefficient
  (:func:`contact_friction`, also the torsional one's combination) and the
  restitution (:func:`contact_restitution`) averaged over the present shapes;
* row frame: :func:`contact_tangent_basis`, :func:`contact_friction_points` (the
  point law's friction location) and :func:`contact_row_frame`, each row's
  direction and endpoints under the point or patch law (``patch_row_frame``);
* :func:`prescribed_target`, the row target that prescribed bodies' motion
  imposes, and the spin row's, :func:`prescribed_spin_target`;
* :func:`jacobian_entry`, ``J = d . (S_lin + S_ang x (p - o))`` for one motion
  subspace column ``S`` of a body whose twists are taken about ``o``, and its
  closed form for a free body's six unit columns, :func:`free_body_jacobian`;
  the spin row's (contact torsion, :data:`CONTACT_SPIN_ROW`), ``J = n . S_ang``,
  :func:`spin_entry` and :func:`free_body_spin_jacobian`.
"""

import warp as wp

from ...sim import BodyFlags
from .law_source import LawFamily
from .spatial_laws import SPATIAL, point_velocity

CONTACT_LAW = LawFamily("ContactLaw", requires=(SPATIAL,))
"""The contact front end's family; native code calls ``ContactLaw::contact_points(...)``."""

CONTACT_SPIN_ROW = wp.constant(3)
"""A contact's spin row under contact torsion: its fourth row, after the normal (0) and the friction pair (1, 2)."""

CONTACT_WORLDS_DIFFER = wp.constant(-2)
"""World of a contact whose endpoints belong to different worlds (:func:`contact_endpoints`)."""

_KINEMATIC = wp.constant(int(BodyFlags.KINEMATIC))


@CONTACT_LAW.law
def contact_bodies(shape_a: int, shape_b: int, shape_body: wp.array[int]) -> tuple[int, int]:
    """Return the bodies of a contact's two shapes, -1 for a static shape or a missing one."""
    body_a = int(-1)
    body_b = int(-1)
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]
    return body_a, body_b


@CONTACT_LAW.law
def contact_endpoints(
    shape_a: int,
    shape_b: int,
    shape_body: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
) -> tuple[int, int, int, int, int]:
    """Return a contact's bodies, articulations and world: ``(body_a, body_b, art_a, art_b, world)``.

    The world is the endpoints' common world, -1 when neither endpoint belongs to
    an articulation and :data:`CONTACT_WORLDS_DIFFER` when they belong to
    different worlds.
    """
    body_a, body_b = contact_bodies(shape_a, shape_b, shape_body)
    art_a = int(-1)
    art_b = int(-1)
    if body_a >= 0:
        art_a = body_to_articulation[body_a]
    if body_b >= 0:
        art_b = body_to_articulation[body_b]
    world = int(-1)
    if art_a >= 0:
        world = art_to_world[art_a]
    if art_b >= 0:
        world_b = art_to_world[art_b]
        if world >= 0 and world_b != world:
            return body_a, body_b, art_a, art_b, CONTACT_WORLDS_DIFFER
        world = world_b
    return body_a, body_b, art_a, art_b, world


@CONTACT_LAW.law
def contact_responds(
    art: int,
    body: int,
    articulation_response_dof_count: wp.array[int],
    body_has_response_dofs: wp.array[int],
    body_flags: wp.array[wp.int32],
) -> bool:
    """Return whether a contact endpoint moves under the contact's impulses.

    It must belong to an articulation with response coordinates, reach some of
    them and not be kinematic.
    """
    if art < 0:
        return False
    if articulation_response_dof_count[art] <= 0:
        return False
    return body_has_response_dofs[body] != 0 and (body_flags[body] & _KINEMATIC) == 0


@CONTACT_LAW.law
def contact_points(
    normal_ab: wp.vec3,
    point0: wp.vec3,
    point1: wp.vec3,
    margin0: float,
    margin1: float,
    body_a: int,
    body_b: int,
    body_q: wp.array[wp.transform],
) -> tuple[wp.vec3, wp.vec3, wp.vec3, float]:
    """Return a contact's normal from B to A, its world points on each surface and its gap ``phi``.

    The collision pipeline stores the normal from A to B and each witness point in
    its body's frame (in world coordinates for a static shape). Each point moves to
    its shape's surface by its margin along the normal; ``phi = n . (p_a - p_b)`` is
    negative in penetration.
    """
    normal = -normal_ab
    point_a = point0 - margin0 * normal
    point_b = point1 + margin1 * normal
    if body_a >= 0:
        point_a = wp.transform_point(body_q[body_a], point0) - margin0 * normal
    if body_b >= 0:
        point_b = wp.transform_point(body_q[body_b], point1) + margin1 * normal
    phi = wp.dot(normal, point_a - point_b)
    return normal, point_a, point_b, phi


@CONTACT_LAW.law
def contact_geometry(
    c: int,
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
) -> tuple[int, int, wp.vec3, wp.vec3, wp.vec3, float]:
    """Return contact ``c``'s bodies and geometry from the contact arrays: ``(body_a, body_b, normal, p_a, p_b, phi)``.

    :func:`contact_bodies` then :func:`contact_points`, for the producers that read a
    contact straight from the collision pipeline's arrays.
    """
    body_a, body_b = contact_bodies(shape0[c], shape1[c], shape_body)
    n, point_a, point_b, phi = contact_points(
        normal[c], point0[c], point1[c], margin0[c], margin1[c], body_a, body_b, body_q
    )
    return body_a, body_b, n, point_a, point_b, phi


@CONTACT_LAW.law
def contact_points_endpoint_relative(
    normal_ab: wp.vec3,
    point0: wp.vec3,
    point1: wp.vec3,
    margin0: float,
    margin1: float,
    body_a: int,
    body_b: int,
    body_q: wp.array[wp.transform],
) -> tuple[wp.vec3, wp.vec3, wp.vec3, wp.vec3, wp.vec3, wp.vec3, float]:
    """Return the contact geometry with body translations kept apart from the rotated offsets.

    ``(normal, translation_a, offset_a, translation_b, offset_b, separation, phi)``,
    with ``p = translation + offset`` on each side and
    ``separation = (translation_a - translation_b) + (offset_a - offset_b)``: the
    gap and lever arms avoid subtracting rounded large world points (the
    resolved-world classifier's arithmetic contract).
    """
    normal = -normal_ab
    translation_a = wp.vec3(0.0)
    translation_b = wp.vec3(0.0)
    offset_a = point0 - margin0 * normal
    offset_b = point1 + margin1 * normal
    if body_a >= 0:
        pose_a = body_q[body_a]
        translation_a = wp.transform_get_translation(pose_a)
        offset_a = wp.quat_rotate(wp.transform_get_rotation(pose_a), point0) - margin0 * normal
    if body_b >= 0:
        pose_b = body_q[body_b]
        translation_b = wp.transform_get_translation(pose_b)
        offset_b = wp.quat_rotate(wp.transform_get_rotation(pose_b), point1) + margin1 * normal
    separation = (translation_a - translation_b) + (offset_a - offset_b)
    phi = wp.dot(normal, separation)
    return normal, translation_a, offset_a, translation_b, offset_b, separation, phi


@CONTACT_LAW.law
def contact_friction(shape_a: int, shape_b: int, shape_mu: wp.array[float]) -> float:
    """Return a contact's friction coefficient: the plain mean over its present shapes."""
    mu = float(0.0)
    count = int(0)
    if shape_a >= 0:
        mu += shape_mu[shape_a]
        count += 1
    if shape_b >= 0:
        mu += shape_mu[shape_b]
        count += 1
    if count > 0:
        mu /= float(count)
    return mu


@CONTACT_LAW.law
def contact_restitution(shape_a: int, shape_b: int, shape_restitution: wp.array[float]) -> float:
    """Return a contact's restitution: the mean over its present shapes of each one clamped to ``[0, 1]``.

    A non-finite value counts as 0.
    """
    restitution = float(0.0)
    count = int(0)
    if shape_a >= 0:
        value_a = shape_restitution[shape_a]
        if wp.isfinite(value_a):
            restitution += wp.clamp(value_a, 0.0, 1.0)
        count += 1
    if shape_b >= 0:
        value_b = shape_restitution[shape_b]
        if wp.isfinite(value_b):
            restitution += wp.clamp(value_b, 0.0, 1.0)
        count += 1
    if count > 0:
        restitution /= float(count)
    return restitution


@CONTACT_LAW.law
def contact_materials(
    shape_a: int,
    shape_b: int,
    shape_mu: wp.array[float],
    shape_restitution: wp.array[float],
) -> tuple[float, float]:
    """Return a contact's friction coefficient and restitution (:func:`contact_friction`, :func:`contact_restitution`)."""
    return contact_friction(shape_a, shape_b, shape_mu), contact_restitution(shape_a, shape_b, shape_restitution)


@CONTACT_LAW.law
def contact_tangent_basis(n: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    """Return the deterministic tangent pair every FeatherPGS friction row uses for ``n``."""
    tangent0 = wp.cross(n, wp.vec3(1.0, 0.0, 0.0))
    if wp.length_sq(tangent0) < 1.0e-12:
        tangent0 = wp.cross(n, wp.vec3(0.0, 1.0, 0.0))
    tangent0 = wp.normalize(tangent0)
    tangent1 = wp.normalize(wp.cross(n, tangent0))
    return tangent0, tangent1


@CONTACT_LAW.law
def contact_friction_points(
    point_a: wp.vec3, point_b: wp.vec3, shared_anchor: int, friction_shared_anchor: int
) -> tuple[wp.vec3, wp.vec3]:
    """Return the point law's friction location on each side: the contact points or their shared midpoint."""
    if shared_anchor != 0 or friction_shared_anchor != 0:
        anchor = 0.5 * (point_a + point_b)
        return anchor, anchor
    return point_a, point_b


@CONTACT_LAW.law
def contact_row_frame(
    row: int,
    shared_anchor: int,
    normal: wp.vec3,
    tangent0: wp.vec3,
    tangent1: wp.vec3,
    point_a: wp.vec3,
    point_b: wp.vec3,
    friction_a: wp.vec3,
    friction_b: wp.vec3,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    """Return contact row ``row``'s direction and its endpoints on each side.

    The normal row (row 0) acts at the contact points, or at their midpoint with
    ``shared_anchor``. Tangent rows 1 and 2 act along ``tangent0`` and ``tangent1``
    at the friction location: :func:`contact_friction_points` under the point law,
    the patch's friction location under the patch law. The spin row
    (:data:`CONTACT_SPIN_ROW`) turns about the normal, wherever its bodies are.
    """
    direction = normal
    p0 = point_a
    p1 = point_b
    if row == 0:
        if shared_anchor != 0:
            p0 = 0.5 * (point_a + point_b)
            p1 = p0
    elif row != CONTACT_SPIN_ROW:
        direction = tangent0
        if row == 2:
            direction = tangent1
        p0 = friction_a
        p1 = friction_b
    return direction, p0, p1


@CONTACT_LAW.law
def jacobian_entry(motion: wp.spatial_vector, direction: wp.vec3, point: wp.vec3, origin: wp.vec3) -> float:
    """Return one Jacobian entry ``d . (S_lin + S_ang x (p - o))``.

    ``motion`` is a motion subspace column (or a body twist) about ``origin``,
    linear part first; the entry is the velocity of ``point`` along ``direction``
    that a unit rate of the column produces (:func:`.spatial_laws.point_velocity`).
    """
    return wp.dot(direction, point_velocity(motion, point - origin))


@CONTACT_LAW.law
def free_body_jacobian(direction: wp.vec3, point: wp.vec3, origin: wp.vec3) -> wp.spatial_vector:
    """Return a free body's Jacobian row ``[d, (p - o) x d]``: :func:`jacobian_entry` over its six unit columns."""
    return wp.spatial_vector(direction, wp.cross(point - origin, direction))


@CONTACT_LAW.law
def spin_entry(motion: wp.spatial_vector, normal: wp.vec3) -> float:
    """Return one entry of a contact's spin row, ``J = n . S_ang``: the spin about the normal that a unit rate of
    the motion subspace column (or body twist) ``motion`` produces, wherever its twist is taken about."""
    return wp.dot(normal, wp.spatial_bottom(motion))


@CONTACT_LAW.law
def free_body_spin_jacobian(normal: wp.vec3) -> wp.spatial_vector:
    """Return a free body's spin row ``[0, n]``: :func:`spin_entry` over its six unit columns."""
    return wp.spatial_vector(wp.vec3(0.0), normal)


@CONTACT_LAW.law
def prescribed_target(
    body_a: int,
    art_a: int,
    body_b: int,
    art_b: int,
    point_a: wp.vec3,
    point_b: wp.vec3,
    direction: wp.vec3,
    prescribed_articulation: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_v_s: wp.array[wp.spatial_vector],
) -> float:
    """Return the row target that prescribed endpoints impose: minus their relative velocity along the row.

    An endpoint of a prescribed articulation moves with its body twist
    ``body_v_s`` (about the articulation origin) whatever the impulses; the row's
    unknown velocity is measured relative to that motion.
    """
    known = float(0.0)
    if body_a >= 0 and art_a >= 0:
        if prescribed_articulation[art_a] != 0:
            known = jacobian_entry(body_v_s[body_a], direction, point_a, articulation_origin[art_a])
    if body_b >= 0 and art_b >= 0:
        if prescribed_articulation[art_b] != 0:
            known -= jacobian_entry(body_v_s[body_b], direction, point_b, articulation_origin[art_b])
    return -known


@CONTACT_LAW.law
def prescribed_spin_target(
    body_a: int,
    art_a: int,
    body_b: int,
    art_b: int,
    normal: wp.vec3,
    prescribed_articulation: wp.array[int],
    body_v_s: wp.array[wp.spatial_vector],
) -> float:
    """Return the spin row's target that prescribed endpoints impose: minus their relative spin about the normal
    (:func:`prescribed_target` with :func:`spin_entry`)."""
    known = float(0.0)
    if body_a >= 0 and art_a >= 0:
        if prescribed_articulation[art_a] != 0:
            known = spin_entry(body_v_s[body_a], normal)
    if body_b >= 0 and art_b >= 0:
        if prescribed_articulation[art_b] != 0:
            known -= spin_entry(body_v_s[body_b], normal)
    return -known
