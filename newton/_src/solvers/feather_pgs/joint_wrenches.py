# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The joints' reaction wrenches: Newton's ``BODY_PARENT_F`` observable, each body's incoming joint wrench.

The solve is the step's joint-space dynamics at the input pose: ``H(q) (qd+ - qd) / h + c(q, qd) = tau + J^T lambda / h``.
Its joint wrenches are inverse dynamics at that pose with ``qdd = (qd+ - qd) / h`` (Newton-Euler from the leaves): each
body's acceleration is its twist's change at the input pose, ``J(q) qdd`` (forward kinematics of ``qd+`` and of ``qd``
at the input coordinates), plus the velocity product, ``a_vp(child) = a_vp(parent) + v(parent) x v(child)`` in world
Plucker coordinates (a free root on the world, whose speeds are its centre's, has none at its centre). Its net wrench is
``m a_com`` and ``I alpha + w x I w`` about its centre; the world puts on it its weight (less its gravity compensation),
its applied wrench (``body_f``) and its rows' impulses over ``h``: the contacts' (each row's direction at its point, the
spin row's torque about the normal) and a cut body's link's. What remains, summed from the leaves to the root, is the
wrench each joint passes to its child: its constraint wrench with its drive, limit and passive forces, averaged over the
step. A joint that holds a body at rest carries its weight, a spinning pendulum's hinge its centripetal pull.

The readout runs after the step from its input coordinates and both speeds, so it needs the input state: a step in
place (``state_in is state_out``) has no input left and raises.
"""

import warp as wp

from ...sim import JointType
from ...sim.articulation import eval_fk
from .contact_input import ContactInput, contact_frame
from .contact_laws import CONTACT_SPIN_ROW, contact_row_frame
from .friction_patches import PatchRowLaw
from .spatial_laws import weight_fraction

# FeatherPGS takes no gradients: its kernels compile no adjoints.
wp.set_module_options({"enable_backward": False})


@wp.func
def _centre(body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], body: int) -> wp.vec3:
    return wp.transform_point(body_q[body], body_com[body])


@wp.func
def _take(
    acc: wp.array[wp.spatial_vector],
    body: int,
    force: wp.vec3,
    torque: wp.vec3,
    point: wp.vec3,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
):
    """Take the world's wrench ``(force, torque)`` at ``point`` on ``body`` out of its accumulator (which holds the
    joint's share: net less external), about its centre of mass."""
    if body >= 0:
        moment = torque + wp.cross(point - _centre(body_q, body_com, body), force)
        wp.atomic_sub(acc, body, wp.spatial_vector(force, moment))


@wp.kernel
def contact_row_wrenches(
    data: ContactInput,
    law: PatchRowLaw,
    impulses: wp.array2d[wp.float32],
    inv_dt: float,
    body_com: wp.array[wp.vec3],
    # outputs
    acc: wp.array[wp.spatial_vector],
):
    """Each contact's rows' wrenches on its two bodies, as the one writer formed the rows (:func:`.contact_input.
    contact_frame`, :func:`.contact_laws.contact_row_frame`): ``+lambda`` along a row's direction on body a, its
    opposite on body b; the spin row a torque about the normal."""
    c = wp.tid()
    if c >= data.count[0]:
        return
    slot = data.slot[c]
    if slot < 0 or data.path[c] < 0:
        return
    frame = contact_frame(data, law, c)
    line = data.first[data.world[c]]
    for row in range(wp.min(data.slots_needed[c], 4)):
        impulse = impulses[line, slot + row] * inv_dt
        direction, point = contact_row_frame(
            row, frame.normal, frame.tangent0, frame.tangent1, frame.point, frame.friction_point
        )
        force = impulse * direction
        torque = wp.vec3(0.0)
        if row == CONTACT_SPIN_ROW:
            torque = force
            force = wp.vec3(0.0)
        _take(acc, frame.body_a, force, torque, point, data.body_q, body_com)
        _take(acc, frame.body_b, -force, -torque, point, data.body_q, body_com)


@wp.kernel
def link_row_wrenches(
    primary: wp.array[int],
    shadow: wp.array[int],
    link_slot: wp.array[int],
    link_world: wp.array[int],
    first: wp.array[int],
    impulses: wp.array2d[wp.float32],
    inv_dt: float,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    # outputs
    acc: wp.array[wp.spatial_vector],
):
    """Each link's six rows' wrench (``row_laws.LINK_ROWS``: a force then a torque along the world's axes, at the
    primary piece's centre) on its primary piece, its opposite on its shadow."""
    k = wp.tid()
    slot = link_slot[k]
    if slot < 0:
        return
    line = first[link_world[k]]
    force = wp.vec3(impulses[line, slot + 0], impulses[line, slot + 1], impulses[line, slot + 2]) * inv_dt
    torque = wp.vec3(impulses[line, slot + 3], impulses[line, slot + 4], impulses[line, slot + 5]) * inv_dt
    point = _centre(body_q, body_com, primary[k])
    _take(acc, primary[k], force, torque, point, body_q, body_com)
    _take(acc, shadow[k], -force, -torque, point, body_q, body_com)


@wp.func
def _plucker(twist: wp.spatial_vector, centre: wp.vec3) -> wp.spatial_vector:
    """A body twist ``(v_com, w)`` in world Plucker coordinates, its linear part the velocity at the world origin."""
    w = wp.spatial_bottom(twist)
    return wp.spatial_vector(wp.spatial_top(twist) - wp.cross(w, centre), w)


@wp.func
def _cross_motion(a: wp.spatial_vector, b: wp.spatial_vector) -> wp.spatial_vector:
    """The motion cross product ``a x b`` of two Plucker twists (linear first)."""
    wa = wp.spatial_bottom(a)
    wb = wp.spatial_bottom(b)
    return wp.spatial_vector(wp.cross(wa, wp.spatial_top(b)) + wp.cross(wp.spatial_top(a), wb), wp.cross(wa, wb))


@wp.kernel
def tree_wrenches(
    articulation_start: wp.array[int],
    articulation_end: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_qd_out: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_com: wp.array[wp.vec3],
    body_world: wp.array[int],
    gravity: wp.array[wp.vec3],
    compensation: wp.array[float],
    inv_dt: float,
    # outputs
    velocity_product: wp.array[wp.spatial_vector],
    acc: wp.array[wp.spatial_vector],
    wrench: wp.array[wp.spatial_vector],
):
    """Each tree's joints' wrenches (the module docstring): the velocity products from its root (its joints are
    parent-first), then from its leaves a body's net wrench less the world's on it, plus its children's, each child's
    moved to the body's centre. ``body_qd`` and ``body_qd_out`` are the input pose's twists of the speeds before and
    after the solve."""
    tree = wp.tid()
    for j in range(articulation_start[tree], articulation_end[tree]):
        body = joint_child[j]
        parent = joint_parent[j]
        product = wp.spatial_vector()
        if parent >= 0:
            twist = _plucker(body_qd[parent], _centre(body_q, body_com, parent))
            product = velocity_product[parent] + _cross_motion(
                twist, _plucker(body_qd[body], _centre(body_q, body_com, body))
            )
        elif joint_type[j] == JointType.FREE or joint_type[j] == JointType.DISTANCE:
            # A free root's speeds are its centre's velocity and its spin: its centre has no velocity product.
            v = wp.spatial_top(body_qd[body])
            product = wp.spatial_vector(-wp.cross(wp.spatial_bottom(body_qd[body]), v), wp.vec3(0.0))
        velocity_product[body] = product
    for i in range(articulation_end[tree] - articulation_start[tree]):
        j = articulation_end[tree] - 1 - i
        body = joint_child[j]
        pose = body_q[body]
        rotation = wp.transform_get_rotation(pose)
        inertia = wp.quat_to_matrix(rotation) * body_inertia[body] * wp.transpose(wp.quat_to_matrix(rotation))
        mass = body_mass[body]
        centre = _centre(body_q, body_com, body)
        v0 = wp.spatial_top(body_qd[body])
        w0 = wp.spatial_bottom(body_qd[body])
        product = velocity_product[body]
        alpha = (wp.spatial_bottom(body_qd_out[body]) - w0) * inv_dt + wp.spatial_bottom(product)
        dv = (wp.spatial_top(body_qd_out[body]) - v0) * inv_dt
        accel = dv + wp.spatial_top(product) + wp.cross(wp.spatial_bottom(product), centre) + wp.cross(w0, v0)
        world = body_world[body]
        if world < 0:
            world = gravity.shape[0] - 1
        weight = mass * weight_fraction(compensation[body]) * gravity[world]
        force = mass * accel - weight - wp.spatial_top(body_f[body])
        torque = inertia * alpha + wp.cross(w0, inertia * w0) - wp.spatial_bottom(body_f[body])
        total = acc[body] + wp.spatial_vector(force, torque)
        wrench[body] = total
        parent = joint_parent[j]
        if parent >= 0:
            arm = centre - _centre(body_q, body_com, parent)
            acc[parent] = acc[parent] + wp.spatial_vector(
                wp.spatial_top(total), wp.spatial_bottom(total) + wp.cross(arm, wp.spatial_top(total))
            )


def parent_wrenches(solver, state_in, state_out, contacts, body_parent_f: wp.array) -> None:
    """Write each body's incoming joint wrench of the last step into ``body_parent_f`` (world frame, at the body's centre
    of mass [N, N m]; zero for a body no joint moves). See the module docstring."""
    if state_in is state_out:
        raise ValueError("BODY_PARENT_F needs the step's input and output states apart (state_in is not state_out).")
    body_parent_f.zero_()
    dt = solver.factor.last_step_dt
    cut, model, placement = solver.cut, solver.model, solver.placement
    if dt is None or dt <= 0.0 or model.body_count == 0 or model.articulation_count == 0:
        return
    inv_dt, device = 1.0 / dt, model.device
    forest_in, forest_out = cut.mirror(state_in), cut.mirror(state_out)
    acc = solver._parent_wrench_acc
    wrench = solver._parent_wrench
    acc.zero_()
    wrench.zero_()
    if contacts is not None and contacts.rigid_contact_count is not None and contacts.rigid_contact_max > 0:
        from .contact_input import bind as bind_contact_input  # noqa: PLC0415
        from .friction_patches import patch_row_law  # noqa: PLC0415

        data = bind_contact_input(solver, forest_in, solver.factor, contacts, dt, 1.0)
        wp.launch(
            contact_row_wrenches,
            dim=contacts.rigid_contact_max,
            inputs=[data, patch_row_law(solver), placement.impulses, inv_dt, model.body_com],
            outputs=[acc],
            device=device,
        )
    laws = solver.laws
    if laws.link_count > 0 and laws.link_slot is not None:
        wp.launch(
            link_row_wrenches,
            dim=laws.link_count,
            inputs=[
                laws._link_primary,
                laws._link_shadow,
                laws.link_slot,
                laws._link_world,
                placement.dense_first,
                placement.impulses,
                inv_dt,
                forest_in.body_q,
                model.body_com,
            ],
            outputs=[acc],
            device=device,
        )
    # The input pose's twists of the speeds before and after the solve (forward kinematics at the input coordinates).
    before, after = solver._parent_wrench_twists
    eval_fk(model, forest_in.joint_q, forest_in.joint_qd, before)
    eval_fk(model, forest_in.joint_q, forest_out.joint_qd, after)
    wp.launch(
        tree_wrenches,
        dim=model.articulation_count,
        inputs=[
            model.articulation_start,
            model.articulation_end,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            before.body_q,
            before.body_qd,
            after.body_qd,
            forest_in.body_f,
            model.body_mass,
            model.body_inertia,
            model.body_com,
            model.body_world,
            model.gravity,
            solver.factor.body_gravity_compensation,
            inv_dt,
        ],
        outputs=[solver._parent_wrench_product, acc, wrench],
        device=device,
    )
    wp.copy(body_parent_f, wrench, count=cut.source.body_count)
