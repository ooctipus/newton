# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Complete topology ownership for a scalar tree and independent free roots.

The first binding retains an eleven-body/nine-coordinate primary tree and two
free roots, one dynamic and one prescribed. Body, articulation, world and factor
group indices are independent; no particular robot parent tree is required.
"""

import numpy as np

from ...sim import BodyFlags, JointType


def build_plan(solver):
    """Build complete physical IDs, rejecting omissions and changed scalar topology."""
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
    group9, group6 = solver.group_to_art[9].numpy(), solver.group_to_art[6].numpy()
    if (
        group9.shape != (worlds,)
        or group6.shape != (worlds,)
        or len(np.unique(group9)) != worlds
        or len(np.unique(group6)) != worlds
    ):
        raise ValueError("Cached scalar forest requires one complete factor group per world and size")
    primary = {int(a): g for g, a in enumerate(group9)}
    secondary = {int(a): g for g, a in enumerate(group6)}
    flags = model.body_flags.numpy()
    roots = {int(j): k for k, j in enumerate(solver._free_root_joint_indices.numpy())}
    fields = {
        name: np.full((worlds, width), -1, np.int32)
        for name, width in (
            ("body_ids", 32),
            ("body_parent", 32),
            ("joint_ids", 32),
            ("dof_ids", 21),
            ("root_slots", 2),
            ("arts", 3),
            ("body_local_dof", 32),
            ("q_index", 9),
        )
    }
    fields.update(primary_group=np.full(worlds, -1, np.int32), secondary_group=np.full(worlds, -1, np.int32))
    expected_parent = expected_dof_body = expected_source = None
    if (
        len(art_world) != 3 * worlds
        or model.body_count != 13 * worlds
        or model.joint_count != 13 * worlds
        or model.joint_dof_count != 21 * worlds
    ):
        raise ValueError("Cached scalar forest requires exact 13-body/21-coordinate/three-articulation worlds")
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
            raise ValueError("Cached scalar forest requires primary9/free6/prescribed6")
        ordered = [a[0], b[0], c[0]]
        if len(set(ordered)) != 3 or list(counts[ordered]) != [9, 6, 6] or list(responses[ordered]) != [9, 6, 0]:
            raise ValueError("Cached scalar forest physical and response coordinates do not match")
        if list(ends[ordered] - starts[ordered]) != [11, 1, 1]:
            raise ValueError("Cached scalar forest requires all11 primary joints and both free roots")
        joints = np.concatenate([np.arange(starts[art], ends[art]) for art in ordered])
        bodies = children[joints]
        if (
            np.any(bodies < 0)
            or len(np.unique(bodies)) != 13
            or not np.array_equal(body_art[bodies], np.repeat(ordered, [11, 1, 1]))
        ):
            raise ValueError("Scalar forest joint/body ownership mismatch")
        lookup = {int(body): i for i, body in enumerate(bodies)}
        parent = np.array([lookup.get(int(parents[joint]), -1) for joint in joints], np.int32)
        if (
            parents[joints[0]] != -1
            or np.any(parent[1:11] < 0)
            or np.any(parent[1:11] >= np.arange(1, 11))
            or np.any(parents[joints[11:]] >= 0)
        ):
            raise ValueError("Cached scalar forest requires a topologically ordered tree and independent free roots")
        if np.any((parents[joints] >= 0) & (parent < 0)):
            raise ValueError("Cached scalar forest parent escapes its world")
        # The retained binding handles scalar axes, welds and independent free
        # roots. Its factor dimensions remain fixed while parent edges vary.
        primary_types = types[joints[:11]]
        primary_counts = dimensions[joints[:11]].sum(axis=1)
        scalar = np.isin(primary_types, [int(JointType.PRISMATIC), int(JointType.REVOLUTE)])
        weld = primary_types == int(JointType.FIXED)
        locked_root = primary_types[0] == int(JointType.D6) and primary_counts[0] == 0
        if locked_root:
            weld[0] = True
        if (
            not weld[0]
            or not np.all(scalar | weld)
            or np.any(primary_counts != scalar.astype(int))
            or np.any(types[joints[11:]] != int(JointType.FREE))
            or np.any(dimensions[joints[11:]].sum(axis=1) != 6)
        ):
            raise ValueError(
                "Cached scalar forest requires a fixed root, scalar descendants and six-coordinate free roots"
            )
        dof_body = np.flatnonzero(scalar).astype(np.int32)
        if len(dof_body) != 9:
            raise ValueError("Cached scalar forest requires the retained nine-coordinate primary operator")
        if np.any(flags[bodies[:12]] & int(BodyFlags.KINEMATIC)):
            raise ValueError("Cached scalar forest primary and response free bodies must be dynamic")
        if not (flags[bodies[12]] & int(BodyFlags.KINEMATIC)):
            raise ValueError("Cached scalar forest requires the final free root to remain prescribed")
        if any(int(j) not in roots for j in joints[11:]):
            raise ValueError("Cached scalar forest is missing an original free-root transport slot")
        dofs = np.concatenate([np.arange(dof_starts[art], dof_starts[art] + counts[art]) for art in ordered])
        ancestor_mask = np.zeros(11, np.uint32)
        for local in range(11):
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
        if not np.array_equal(masks[bodies[:11]], ancestor_mask) or masks[bodies[11]] != 63 or masks[bodies[12]] != 0:
            raise ValueError("Cached scalar forest must preserve original ancestor response masks")
        source = np.full((9, 9), -1, np.int32)
        for row, row_body in enumerate(dof_body):
            for col, col_body in enumerate(dof_body):
                if ancestor_mask[col_body] & (1 << row):
                    source[row, col] = col
                elif ancestor_mask[row_body] & (1 << col):
                    source[row, col] = row
        if expected_parent is None:
            expected_parent, expected_dof_body, expected_source = parent[:11].copy(), dof_body.copy(), source
        elif (
            not np.array_equal(parent[:11], expected_parent)
            or not np.array_equal(dof_body, expected_dof_body)
            or not np.array_equal(source, expected_source)
        ):
            raise ValueError("Cached scalar forest requires a shared primary topology and scalar coordinate ordering")
        fields["arts"][world] = ordered
        fields["body_ids"][world, :13], fields["joint_ids"][world, :13] = bodies, joints
        fields["body_parent"][world, :13], fields["dof_ids"][world] = parent, dofs
        fields["primary_group"][world], fields["secondary_group"][world] = primary[a[0]], secondary[b[0]]
        fields["root_slots"][world] = [roots[int(j)] for j in joints[11:]]
    if (
        len(np.unique(fields["body_ids"][:, :13])) != model.body_count
        or len(np.unique(fields["dof_ids"])) != model.joint_dof_count
    ):
        raise ValueError("Cached scalar forest must cover every public body and physical coordinate exactly once")
    fields["dof_body"] = dof_body
    fields["source"] = solver._crba_source_dof_by_size[9].numpy().astype(np.int32)
    if not np.array_equal(fields["source"], expected_source):
        raise ValueError("Cached scalar forest CRBA source schedule disagrees with the physical parent tree")
    return fields
