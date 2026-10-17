# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""State/publication adapter for a primary tree and independent free roots.

The primary tree is followed by independent free/prescribed root services.
Dimensions, root counts and storage come from the
owner; the common body law and integration live in cached_state.
"""

from string import Template
from textwrap import dedent, indent

from .cached_state import StateOperands, StateStages, make_state_kernel

_RELATIVE = """
if local == 0:
    _store_shift(address, 0, wp.transform_get_translation(relative))
    relative = wp.transform(wp.vec3(), wp.transform_get_rotation(relative))
elif local >= $PRIMARY_BODIES:
    _store_shift(address, local - $ROOT_OFFSET, wp.vec3())
"""

_ORIGINS = """
_scan_poses(address, world, plan)
for root in range(lane, $ROOTS, stride):
    root_lane = int(0)
    if root > 0:
        root_lane = $ROOT_OFFSET + root
    body = plan.body_ids[world, root_lane]
    origin = wp.transform_point(_load_pose(address, root_lane), data.body_com[body])
    _store_origin(address, root, origin)
$ORIGIN_PUBLICATION
_sync()
"""

_MOTION = """
for local in range(lane, $BODIES, stride):
    joint = plan.joint_ids[world, local]
    root = int(0)
    if local >= $PRIMARY_BODIES:
        root = local - $ROOT_OFFSET
    origin = _load_origin(address, root)
    parent = plan.body_parent[world, local]
    anchor = data.joint_X_p[joint]
    if parent >= 0:
        anchor = _load_pose(address, parent) * anchor
    else:
        anchor = wp.transform(
            wp.transform_get_translation(anchor) - _load_shift(address, root),
            wp.transform_get_rotation(anchor),
        )
    anchor_local = wp.transform(
        wp.transform_get_translation(anchor) - origin,
        wp.transform_get_rotation(anchor),
    )
$JOINT_MOTION
    _store_motion(address, local, velocity)
_sync()
$MOTION_PADDING
_scan_motion(address, world, plan)
"""

_BODIES = """
for local in range(lane, $BODIES, stride):
    body = plan.body_ids[world, local]
    root = int(0)
    if local >= $PRIMARY_BODIES:
        root = local - $ROOT_OFFSET
    pose = _load_pose(address, local)
    pose_com = pose * data.body_X_com[body]
    origin = _load_origin(address, root)
    velocity = _load_motion(address, local, 0)
    acceleration = _load_motion(address, local, 1)
    shift = _load_shift(address, root)
    data.body_q[body] = wp.transform(
        wp.transform_get_translation(pose) + shift,
        wp.transform_get_rotation(pose),
    )
    if local < $PRIMARY_BODIES:
        _primary_terms(
            address,
            local,
            world,
            body,
            pose,
            pose_com,
            origin,
            velocity,
            acceleration,
            data,
$TERMS_ARGS
        )
    else:
        data.body_q_com[body] = pose_com
        data.body_v_s[body] = velocity
        data.body_a_s[body] = acceleration
        kernels.finalize_body_dynamics_body(
            body,
            data.body_to_articulation,
            data.body_q,
            data.body_q_com,
            data.body_com,
            data.body_mass,
            data.body_inertia,
            data.is_free_rigid,
            data.articulation_origin,
            data.materialize_all_body_inertia,
            data.materialize_body_inertia_terms,
            data.gravity,
            data.body_v_s,
            data.body_a_s,
            data.body_I_s,
            data.body_inertia_terms,
            data.body_f_s,
            data.body_qd,
        )
        $CACHE.com_offset[world, local] = $FREE_COM_OFFSET
        if local == $PRIMARY_BODIES:
            $CACHE.free_bias[world] = data.body_f_s[body]
_sync()
"""

_COLLECT = """
success = $COLLECT
for root in range(lane, $ROOTS, stride):
    root_lane = int(0)
    if root > 0:
        root_lane = $ROOT_OFFSET + root
    data.fk_id_cache_valid[data.body_to_articulation[plan.body_ids[world, root_lane]]] = int(
        $VALID
    )
"""


def make_forest_state(
    name,
    namespace,
    *,
    finish,
    primary_bodies,
    free_roots,
    dofs,
    signature,
    setup,
    joint_motion,
    origin_publication,
    terms_args,
    cache,
    free_com_offset,
    collect,
    valid,
    stamp="",
    padded=False,
    clocked=False,
):
    """Bind forest publication to its explicit current/geometric cache policy."""
    if primary_bodies <= 0 or free_roots < 0:
        raise ValueError("A forest requires a primary tree and nonnegative free-root count")
    bodies = primary_bodies + free_roots

    def padding(store, value):
        if not padded:
            return ""
        return (
            f"for padding in range(lane, _scan_width(), stride):\n"
            f"    if padding >= {bodies}:\n        {store}(address, padding, {value})\n_sync()"
        )

    values = {
        "PRIMARY_BODIES": primary_bodies,
        "ROOT_OFFSET": primary_bodies - 1,
        "ROOTS": free_roots + 1,
        "BODIES": bodies,
        "ORIGIN_PUBLICATION": origin_publication,
        "JOINT_MOTION": indent(dedent(joint_motion).strip(), "    "),
        "MOTION_PADDING": padding("_store_motion", "wp.spatial_vector()"),
        "TERMS_ARGS": indent(dedent(terms_args).strip(), "            "),
        "CACHE": cache,
        "FREE_COM_OFFSET": free_com_offset,
        "COLLECT": collect,
        "VALID": valid,
    }

    # Values may themselves contain topology operands, but never producer statements.
    def render(source):
        return Template(Template(source).substitute(values)).substitute(values)

    stages = StateStages(
        setup=setup,
        relative=render(_RELATIVE),
        poses=padding("_store_pose", "wp.transform_identity()") + "\n" + render(_ORIGINS),
        motion=render(_MOTION),
        bodies=render(_BODIES),
        collect=render(_COLLECT) + stamp,
    )
    operands = StateOperands(
        bodies,
        dofs,
        "plan.joint_ids[world, local]",
        "plan.dof_ids[world, local]",
        f"for root in range(lane, {free_roots}, stride)",
        "plan.root_slots[world, root]",
    )
    return make_state_kernel(name, signature, operands, stages, namespace, finish=finish, clocked=clocked)
