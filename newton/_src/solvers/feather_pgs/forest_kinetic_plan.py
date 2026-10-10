# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Complete topology ownership for a scalar tree and independent free roots.

The primary scalar tree and its scratch dimensions come from one layout. Two
free roots remain independent, one dynamic and one prescribed. Body, articulation,
world and factor group indices are independent.
"""

import numpy as np

from ...sim import BodyFlags, JointType
from .forest_kinetic_layout import from_solver


def build_plan(solver, layout=None):
    """Build complete physical IDs, rejecting omissions and changed scalar topology."""
    layout = from_solver(solver) if layout is None else layout
    n, primary_bodies, body_count = layout.primary_dofs, layout.primary_bodies, layout.bodies
    model, worlds = solver.model, int(solver.world_count)
    art_world = solver.art_to_world.numpy()
    starts, ends = model.articulation_start.numpy(), solver.articulation_joint_end.numpy()
    dof_starts = solver.articulation_dof_start.numpy()
    counts = np.diff(np.r_[dof_starts, model.joint_dof_count])
    responses = solver.articulation_response_dof_count.numpy()
    prescribed = solver._prescribed_articulation.numpy()
    types, parents, children = model.joint_type.numpy(), model.joint_parent.numpy(), model.joint_child.numpy()
    qstarts, dstarts = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    dimensions = model.joint_dof_dim.numpy()
    body_art, masks = solver.body_to_articulation.numpy(), solver.body_response_dof_mask.numpy()
    primary_group, group6 = solver.group_to_art[n].numpy(), solver.group_to_art[6].numpy()
    if n == 6:
        complete = group6
        primary_group = complete[(ends[complete] - starts[complete]) == primary_bodies]
        group6 = complete[(ends[complete] - starts[complete]) == 1]
        if len(complete) != 2 * worlds or len(np.unique(complete)) != len(complete):
            raise ValueError("Same-width scalar forest requires complete distinct scalar/free roles")
        original_group = {int(art): group for group, art in enumerate(complete)}
        primary = {int(art): original_group[int(art)] for art in primary_group}
        secondary = {int(art): original_group[int(art)] for art in group6}
    else:
        primary = {int(a): g for g, a in enumerate(primary_group)}
        secondary = {int(a): g for g, a in enumerate(group6)}
    if (
        primary_group.shape != (worlds,)
        or group6.shape != (worlds,)
        or len(np.unique(primary_group)) != worlds
        or len(np.unique(group6)) != worlds
    ):
        raise ValueError("Cached scalar forest requires one complete factor role per world")
    flags = model.body_flags.numpy()
    roots = {int(j): k for k, j in enumerate(solver._free_root_joint_indices.numpy())}
    fields = {
        name: np.full((worlds, width), -1, np.int32)
        for name, width in (
            ("body_ids", 32),
            ("body_parent", 32),
            ("joint_ids", 32),
            ("dof_ids", layout.physical_dofs),
            ("root_slots", 2),
            ("arts", 3),
            ("body_local_dof", 32),
            ("q_index", n),
        )
    }
    fields.update(primary_group=np.full(worlds, -1, np.int32), secondary_group=np.full(worlds, -1, np.int32))
    expected_parent = expected_dof_body = expected_source = None
    if (
        len(art_world) != 3 * worlds
        or model.body_count != body_count * worlds
        or model.joint_count != body_count * worlds
        or model.joint_dof_count != layout.physical_dofs * worlds
    ):
        raise ValueError("Cached scalar forest requires exact layout body/coordinate/three-articulation worlds")
    by_world = [[] for _ in range(worlds)]
    for art, world in enumerate(art_world):
        if not 0 <= world < worlds:
            raise ValueError("Scalar forest articulation outside complete world ownership")
        by_world[int(world)].append(art)
    for world, arts in enumerate(by_world):
        a = [art for art in arts if art in primary]
        b = [art for art in arts if art in secondary]
        c = [art for art in arts if prescribed[art] != 0]
        if len(arts) != 3 or (len(a), len(b), len(c)) != (1, 1, 1):
            raise ValueError(
                "Cached scalar forest requires a scalar primary, dynamic free root and prescribed free root"
            )
        ordered = [a[0], b[0], c[0]]
        if len(set(ordered)) != 3 or list(counts[ordered]) != [n, 6, 6] or list(responses[ordered]) != [n, 6, 0]:
            raise ValueError("Cached scalar forest physical and response coordinates do not match")
        if list(ends[ordered] - starts[ordered]) != [primary_bodies, 1, 1]:
            raise ValueError("Cached scalar forest requires all primary joints and both free roots")
        joints = np.concatenate([np.arange(starts[art], ends[art]) for art in ordered])
        bodies = children[joints]
        if (
            np.any(bodies < 0)
            or len(np.unique(bodies)) != body_count
            or not np.array_equal(body_art[bodies], np.repeat(ordered, [primary_bodies, 1, 1]))
        ):
            raise ValueError("Scalar forest joint/body ownership mismatch")
        lookup = {int(body): i for i, body in enumerate(bodies)}
        parent = np.array([lookup.get(int(parents[joint]), -1) for joint in joints], np.int32)
        if (
            parents[joints[0]] != -1
            or np.any(parent[1:primary_bodies] < 0)
            or np.any(parent[1:primary_bodies] >= np.arange(1, primary_bodies))
            or np.any(parents[joints[primary_bodies:]] >= 0)
        ):
            raise ValueError("Cached scalar forest requires a topologically ordered tree and independent free roots")
        if np.any((parents[joints] >= 0) & (parent < 0)):
            raise ValueError("Cached scalar forest parent escapes its world")
        # The retained binding handles scalar axes, welds and independent free
        # roots. Its layout dimensions are shared while parent edges vary.
        primary_types = types[joints[:primary_bodies]]
        primary_counts = dimensions[joints[:primary_bodies]].sum(axis=1)
        scalar = np.isin(primary_types, [int(JointType.PRISMATIC), int(JointType.REVOLUTE)])
        weld = primary_types == int(JointType.FIXED)
        locked_root = primary_types[0] == int(JointType.D6) and primary_counts[0] == 0
        if locked_root:
            weld[0] = True
        if (
            not weld[0]
            or not np.all(scalar | weld)
            or np.any(primary_counts != scalar.astype(int))
            or np.any(types[joints[primary_bodies:]] != int(JointType.FREE))
            or np.any(dimensions[joints[primary_bodies:]].sum(axis=1) != 6)
        ):
            raise ValueError(
                "Cached scalar forest requires a fixed root, scalar descendants and six-coordinate free roots"
            )
        dof_body = np.flatnonzero(scalar).astype(np.int32)
        if len(dof_body) != n:
            raise ValueError("Cached scalar forest requires the layout primary coordinate count")
        if np.any(flags[bodies[: primary_bodies + 1]] & int(BodyFlags.KINEMATIC)):
            raise ValueError("Cached scalar forest primary and response free bodies must be dynamic")
        if not (flags[bodies[primary_bodies + 1]] & int(BodyFlags.KINEMATIC)):
            raise ValueError("Cached scalar forest requires the final free root to remain prescribed")
        if any(int(j) not in roots for j in joints[primary_bodies:]):
            raise ValueError("Cached scalar forest is missing an original free-root transport slot")
        dofs = np.concatenate([np.arange(dof_starts[art], dof_starts[art] + counts[art]) for art in ordered])
        ancestor_mask = np.zeros(primary_bodies, np.uint32)
        for local in range(primary_bodies):
            own = -1
            if local in dof_body:
                own = int(np.flatnonzero(dof_body == local)[0])
                if dstarts[joints[local]] != dofs[own] or dimensions[joints[local]].sum() != 1:
                    raise ValueError("Scalar forest scalar coordinate mapping changed")
                fields["body_local_dof"][world, local] = own
                fields["q_index"][world, own] = qstarts[joints[local]]
            elif dimensions[joints[local]].sum() != 0:
                raise ValueError("Scalar forest weld gained physical coordinates")
            inherited = int(ancestor_mask[parent[local]]) if parent[local] >= 0 else 0
            ancestor_mask[local] = inherited | (1 << own if own >= 0 else 0)
        if (
            not np.array_equal(masks[bodies[:primary_bodies]], ancestor_mask)
            or masks[bodies[primary_bodies]] != 63
            or masks[bodies[primary_bodies + 1]] != 0
        ):
            raise ValueError("Cached scalar forest must preserve original ancestor response masks")
        source = np.full((n, n), -1, np.int32)
        for row, row_body in enumerate(dof_body):
            for col, col_body in enumerate(dof_body):
                if ancestor_mask[col_body] & (1 << row):
                    source[row, col] = col
                elif ancestor_mask[row_body] & (1 << col):
                    source[row, col] = row
        if expected_parent is None:
            expected_parent, expected_dof_body, expected_source = (
                parent[:primary_bodies].copy(),
                dof_body.copy(),
                source,
            )
        elif (
            not np.array_equal(parent[:primary_bodies], expected_parent)
            or not np.array_equal(dof_body, expected_dof_body)
            or not np.array_equal(source, expected_source)
        ):
            raise ValueError("Cached scalar forest requires a shared primary topology and scalar coordinate ordering")
        fields["arts"][world] = ordered
        fields["body_ids"][world, : layout.bodies], fields["joint_ids"][world, : layout.bodies] = bodies, joints
        fields["body_parent"][world, : layout.bodies], fields["dof_ids"][world] = parent, dofs
        fields["primary_group"][world], fields["secondary_group"][world] = primary[a[0]], secondary[b[0]]
        fields["root_slots"][world] = [roots[int(j)] for j in joints[primary_bodies:]]
    if (
        len(np.unique(fields["body_ids"][:, : layout.bodies])) != model.body_count
        or len(np.unique(fields["dof_ids"])) != model.joint_dof_count
    ):
        raise ValueError("Cached scalar forest must cover every public body and physical coordinate exactly once")
    fields["dof_body"] = dof_body
    if n == 6:
        if n in solver._crba_source_dof_by_size:
            raise ValueError("Same-width scalar/free roles must not reuse homogeneous CRBA metadata")
        fields["source"] = expected_source
    else:
        fields["source"] = solver._crba_source_dof_by_size[n].numpy().astype(np.int32)
        if not np.array_equal(fields["source"], expected_source):
            raise ValueError("Cached scalar forest CRBA source schedule disagrees with the physical parent tree")
    return fields
