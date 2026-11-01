# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The contact front end: what every contact-row producer computes before it writes a row.

One law each, written once for Warp kernels and native CUDA (:mod:`.law_source`):

* endpoints: :func:`contact_bodies` (shape to body), :func:`contact_endpoints`
  (to articulation and world) and :func:`contact_responds`;
* geometry: :func:`contact_geometry` reads a contact from the collision arrays;
  :func:`contact_points` gives the normal from body B to body A, each surface
  point as a translation and an offset, and the gap ``phi`` formed from
  translations and offsets apart, never from rounded world points;
* materials: :func:`contact_materials`, the friction coefficient
  (:func:`contact_friction`, also the torsional one's combination) and the
  restitution (:func:`contact_restitution`) averaged over the present shapes;
* row frame: :func:`contact_tangent_basis`, :func:`contact_point` (the one point
  a contact's rows act at) and :func:`contact_row_frame`, each row's direction and
  point under the point or patch law;
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
) -> tuple[wp.vec3, wp.vec3, wp.vec3, wp.vec3, wp.vec3, wp.vec3, float]:
    """Return a contact's normal from B to A, each surface point as a translation and an offset, and its gap.

    ``(normal, translation_a, offset_a, translation_b, offset_b, separation, phi)``. The collision pipeline stores the
    normal from A to B and each witness point in its body's frame (in world coordinates for a static shape). A body's
    surface point is its pose's translation plus the witness point rotated into world axes, moved to the surface by
    its margin along the normal; a static shape's witness point is its own translation, the margin its offset. The
    gap is ``phi = n . separation`` with ``separation = (translation_a - translation_b) + (offset_a - offset_b)``,
    negative in penetration: the large parts subtract first, so the gap does not carry the rounding of world
    points, a spacing of the coordinate far from the origin (Q20). Each world surface point is
    ``translation + offset``; the rows act at :func:`contact_point`.
    """
    normal = -normal_ab
    translation_a = point0
    translation_b = point1
    offset_a = -margin0 * normal
    offset_b = margin1 * normal
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
) -> tuple[int, int, wp.vec3, wp.vec3, float]:
    """Return contact ``c``'s bodies, normal, the point its rows act at and its gap from the contact arrays:
    ``(body_a, body_b, normal, point, phi)``.

    :func:`contact_bodies`, :func:`contact_points` and :func:`contact_point`, for the producers that read a contact
    straight from the collision pipeline's arrays.
    """
    body_a, body_b = contact_bodies(shape0[c], shape1[c], shape_body)
    n, translation_a, offset_a, _translation_b, _offset_b, separation, phi = contact_points(
        normal[c], point0[c], point1[c], margin0[c], margin1[c], body_a, body_b, body_q
    )
    return body_a, body_b, n, contact_point(translation_a, offset_a, separation), phi


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
def contact_point(translation_a: wp.vec3, offset_a: wp.vec3, separation: wp.vec3) -> wp.vec3:
    """Return the one point a contact's rows act at: the midpoint of its surface points, A's surface point less half
    the separation (:func:`contact_points`). Equal and opposite impulses at one point put no torque on the pair, so a
    contact conserves angular momentum."""
    return translation_a + (offset_a - 0.5 * separation)


@CONTACT_LAW.law
def contact_row_frame(
    row: int,
    normal: wp.vec3,
    tangent0: wp.vec3,
    tangent1: wp.vec3,
    point: wp.vec3,
    friction_point: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
    """Return contact row ``row``'s direction and point.

    The normal row (row 0) acts at the contact's ``point``. Tangent rows 1 and 2 act
    along ``tangent0`` and ``tangent1`` at the friction location: ``point`` under the
    point law, the patch's under the patch law. The spin row (:data:`CONTACT_SPIN_ROW`)
    turns about the normal, wherever its bodies are.
    """
    direction = normal
    row_point = point
    if row != 0 and row != CONTACT_SPIN_ROW:
        direction = tangent0
        if row == 2:
            direction = tangent1
        row_point = friction_point
    return direction, row_point


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


@wp.func
def contact_row_entry(row: int, motion: wp.spatial_vector, direction: wp.vec3, point: wp.vec3, origin: wp.vec3):
    """Return contact row ``row``'s entry for the motion column ``motion`` (:func:`contact_row_frame`'s ``direction``
    and ``point``): the spin row's :func:`spin_entry` about the normal, any other row's
    :func:`jacobian_entry`."""
    if row == CONTACT_SPIN_ROW:
        return spin_entry(motion, direction)
    return jacobian_entry(motion, direction, point, origin)


@CONTACT_LAW.law
def prescribed_target(
    body_a: int,
    art_a: int,
    body_b: int,
    art_b: int,
    point: wp.vec3,
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
            known = jacobian_entry(body_v_s[body_a], direction, point, articulation_origin[art_a])
    if body_b >= 0 and art_b >= 0:
        if prescribed_articulation[art_b] != 0:
            known -= jacobian_entry(body_v_s[body_b], direction, point, articulation_origin[art_b])
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
