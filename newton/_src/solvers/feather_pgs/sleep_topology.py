# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Model-derived dynamic components for experimental FPGS sleeping.

A world-anchored zero-DOF skeleton does not couple its movable branches.
Once an ancestor can respond, its entire descendant tree stays together.
This topology is independent of the narrower capability of any sleep consumer.
As with the solver's other schedules, topology changes require reconstruction.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from ...sim import BodyFlags, JointType


@dataclass(frozen=True)
class SleepTopology:
    """Own immutable component maps; ``-1`` denotes an immobile skeleton."""

    body_component_host: np.ndarray
    joint_component_host: np.ndarray
    component_world_host: np.ndarray
    component_eligible_host: np.ndarray
    component_body_start_host: np.ndarray
    component_bodies_host: np.ndarray
    component_joint_start_host: np.ndarray
    component_joints_host: np.ndarray
    body_component: wp.array
    joint_component: wp.array
    component_world: wp.array
    component_eligible: wp.array

    @property
    def component_count(self) -> int:
        """Return the number of independent dynamic components."""
        return len(self.component_world_host)


def build_sleep_topology(solver) -> SleepTopology:
    """Build a constructor-owned map from the existing validated tree plan.

    Kinematic/proxy ancestry and loop or mimic coupling are never sleep-eligible
    in this first implementation. Coupled endpoints are still merged so that
    the map cannot suggest independent ownership to a future consumer.
    """
    model = solver.model
    body_count, joint_count = int(model.body_count), int(model.joint_count)
    starts = model.articulation_start.numpy()
    ends = solver._model_plan.articulation_joint_end
    art_world = solver._model_plan.articulation_world
    parents = model.joint_parent.numpy()
    children = model.joint_child.numpy()
    types = model.joint_type.numpy()
    dims = model.joint_dof_dim.numpy()
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    flags = model.body_flags.numpy()
    body_world = model.body_world.numpy() if model.body_world is not None else None
    body_component = np.full(body_count, -1, dtype=np.int32)
    joint_component = np.full(joint_count, -1, dtype=np.int32)
    body_art = np.full(body_count, -1, dtype=np.int32)
    immobile = np.zeros(body_count, dtype=bool)
    ancestry_unsafe = np.zeros(body_count, dtype=bool)
    seen = np.zeros(body_count, dtype=bool)
    component_parent: list[int] = []
    component_world: list[int] = []
    component_unsafe: list[bool] = []
    bad_articulations: set[int] = set()

    def add_component(world: int, unsafe: bool = False) -> int:
        component = len(component_parent)
        component_parent.append(component)
        component_world.append(world)
        component_unsafe.append(unsafe or world < 0 or world >= solver.world_count)
        return component

    def root(component: int) -> int:
        while component_parent[component] != component:
            component_parent[component] = component_parent[component_parent[component]]
            component = component_parent[component]
        return component

    def couple(a: int, b: int) -> int:
        components = [root(c) for c in (a, b) if c >= 0]
        if not components:
            return -1
        first = min(components)
        if any(component_world[c] != component_world[first] for c in components):
            component_world[first] = -1
        for component in components:
            component_parent[component] = first
        component_unsafe[first] = True
        return first

    for art in range(int(model.articulation_count)):
        start, end = int(starts[art]), int(ends[art])
        for joint in range(start, end):
            body, parent = int(children[joint]), int(parents[joint])
            if body < 0 or body >= body_count or seen[body]:
                bad_articulations.add(art)
                if 0 <= body < body_count and seen[body]:
                    bad_articulations.add(int(body_art[body]))
                continue
            parent_valid = parent == -1 or (0 <= parent < body_count and seen[parent] and body_art[parent] == art)
            if not parent_valid:
                bad_articulations.add(art)
                if 0 <= parent < body_count and seen[parent]:
                    bad_articulations.add(int(body_art[parent]))
            parent_static = parent == -1 or (parent_valid and immobile[parent])
            unsafe = not parent_valid or int(flags[body]) != int(BodyFlags.DYNAMIC)
            if parent_valid and parent >= 0:
                unsafe = unsafe or bool(ancestry_unsafe[parent])
            dofs = int(qd_start[joint + 1] - qd_start[joint])
            coords = int(q_start[joint + 1] - q_start[joint])
            zero_joint = (
                types[joint] in (JointType.FIXED, JointType.D6)
                and np.all(dims[joint] == 0)
                and dofs == 0
                and coords == 0
            )
            immobile[body] = parent_valid and parent_static and zero_joint
            ancestry_unsafe[body] = unsafe
            body_art[body] = art
            seen[body] = True
            if immobile[body]:
                continue
            if parent_valid and parent >= 0 and not parent_static:
                component = int(body_component[parent])
            else:
                component = add_component(int(art_world[art]), unsafe)
            # Unsupported zero-DOF representations are not static separators.
            component_unsafe[component] |= unsafe or (dofs == 0 and not zero_joint)
            body_component[body] = component
            joint_component[joint] = component

    # Trailing loop joints do not own their child a second time. Their row
    # support connects the two existing dynamic components, not the ground.
    for art in range(int(model.articulation_count)):
        for joint in range(int(ends[art]), int(starts[art + 1])):
            body, parent = int(children[joint]), int(parents[joint])
            valid = (
                0 <= body < body_count
                and seen[body]
                and body_art[body] == art
                and (parent == -1 or (0 <= parent < body_count and seen[parent] and body_art[parent] == art))
            )
            if not valid:
                bad_articulations.add(art)
                continue
            a = int(body_component[parent]) if parent >= 0 else -1
            joint_component[joint] = couple(a, int(body_component[body]))
            if types[joint] != JointType.BALL:
                # The current solver ignores non-BALL loop closures. Do not
                # extend that unsupported configuration with sleeping.
                bad_articulations.add(art)

    # Enabled/coefficient values are live inputs. Merge even disabled mimics;
    # changing their enable state must never turn two sleeping owners into one.
    mimic_count = int(getattr(model, "constraint_mimic_count", 0) or 0)
    if mimic_count:
        joint0 = model.constraint_mimic_joint0.numpy()
        joint1 = model.constraint_mimic_joint1.numpy()
        for a, b in zip(joint0, joint1, strict=True):
            if 0 <= a < joint_count and 0 <= b < joint_count:
                couple(int(joint_component[a]), int(joint_component[b]))
            else:
                bad_articulations.update(range(int(model.articulation_count)))

    for body in range(body_count):
        component = int(body_component[body])
        if component >= 0 and int(body_art[body]) in bad_articulations:
            component_unsafe[root(component)] = True
        if not seen[body]:
            # FPGS does not simulate jointless dynamic bodies. Keep them mapped
            # but ineligible, instead of confusing them with fixed ground.
            world = int(body_world[body]) if body_world is not None else -1
            body_component[body] = add_component(world, unsafe=True)

    roots = sorted({root(c) for c in range(len(component_parent))})
    dense_index = {component: index for index, component in enumerate(roots)}
    for mapping in (body_component, joint_component):
        for index in np.flatnonzero(mapping >= 0):
            mapping[index] = dense_index[root(int(mapping[index]))]
    worlds = np.asarray([component_world[c] for c in roots], dtype=np.int32)
    eligible = np.asarray([not component_unsafe[c] for c in roots], dtype=np.int32)

    def members(mapping: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        indices = np.flatnonzero(mapping >= 0)
        order = np.argsort(mapping[indices], kind="stable")
        entries = indices[order].astype(np.int32)
        offsets = np.zeros(len(roots) + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(np.bincount(mapping[indices], minlength=len(roots)))
        return offsets, entries

    body_start, bodies = members(body_component)
    joint_start, joints = members(joint_component)
    host_arrays = (body_component, joint_component, worlds, eligible, body_start, bodies, joint_start, joints)
    for array in host_arrays:
        array.setflags(write=False)
    return SleepTopology(
        *host_arrays,
        *(wp.array(array, dtype=wp.int32, device=model.device) for array in host_arrays[:4]),
    )
