# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sleeping, the rest law of FeatherPGS: a tree at rest is held, and its contacts with what holds it are not made.

**The law.** A tree (a coordinate tree of the forest plan: a free body, or the moving part of an articulation) is
*quiet* in a substep when its kinetic energy is under ``M V^2 / 2`` (``M`` its mass, ``V`` = :data:`SLEEP_SPEED`, so
the mass-weighted RMS speed of its material points is under ``V``), its contacts stick (the friction force of its
contacts within :data:`STICK` of their cone, lumped over the tree), and no *agent* touches it: a body of a tree that
never rests (a robot), a moving kinematic body, or a tree that moved in the last substep. A tree a joint force or a
body wrench acts on is *acted on* (:data:`~newton._src.geometry.sleep_filter.ACTED`), as is a tree that never rests;
an island sleeps as a whole and an acted-on member never rests, so every other tree of an island that held an
acted-on tree at the last pass is *loaded* and counts no quiet substep until a pass finds its island free of one (a
cube in a hand, a part near a robot). After :data:`QUIET_SUBSTEPS` quiet substeps a tree is *ready*. At the next collision pass an island of ready trees
(trees joined by candidate pairs whose grown boxes overlap) falls asleep, and an asleep tree whose island holds an
awake tree that is not ready, or a moving kinematic body, wakes, before the narrow phase makes its contacts
(:mod:`newton._src.geometry.sleep_filter`, which the collision pipeline runs on :meth:`Sleep.collision_sleep_filter`'s
arrays). An asleep tree's solved velocity is zero (:meth:`Sleep.hold`), so its coordinates stay bit for bit; no contact
is made between two held sides, so it holds no row. A reset or a notified state write wakes the trees of its worlds,
a model change every tree. The law's per-contact load runs only for contacts of trees that can count: a scene whose
free bodies are held by robots pays a per-tree test. Each world's trees count on that world's data alone.

A tree never rests where a row joins it to another tree (a loop's link, a mimic), a coordinate of it is kinematic,
carries a drive gain or has an actuator (``Model.joint_target_mode``: MuJoCo's rule, an actuated tree never sleeps); a
model notification reads these again. A model in which no tree can rest builds none of this.

**Why these numbers.** ``V`` = 3 mm/s is under one substep of free fall at the 1.25 ms substeps of the tasks
(``g h`` = 12 mm/s), so a body that loses its support leaves rest within a substep, and above what an unconverged
solve leaves on a resting part. 16 quiet substeps (two collision passes at 8 substeps a pass) is MuJoCo's order (10
steps) and rejects the apex of a flight, under ``V`` for less than a substep. The sticking test keeps a body that
creeps from rest (friction at its cone, any speed) awake. The agent rule is NIST's measured need: a part a gripper
holds still in its seat was quiet, so the moment the gripper's boxes left it, it slept where the gripper left it,
before settling its last millimetre (0.12 fewer such seatings a world on 1,024 worlds than without sleeping).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...geometry.sleep_filter import ACTED, LOADED
from ...sim.enums import BodyFlags, JointTargetMode
from .contact_forces import contact_linear_force
from .contact_input import ContactRoutes, bind_routes
from .contact_laws import contact_friction

wp.set_module_options({"enable_backward": False})

SLEEP_SPEED = 3.0e-3
"""The rest speed ``V`` [m/s]: a tree is quiet when its mass-weighted RMS material speed is under it."""
QUIET_SUBSTEPS = 16
"""Quiet substeps before a tree is ready to sleep."""
STICK = 0.9
"""A tree's contacts stick while their friction force stays within this share of their cone."""

READY = wp.constant(-1)
AWAKE = wp.constant(-(QUIET_SUBSTEPS + 1))
"""An awake tree's state with no quiet substep counted (``tree_asleep``)."""
_KINEMATIC = wp.constant(int(BodyFlags.KINEMATIC))
AGENT = wp.constant(1)
"""A body's kind in ``body_agent``: a movable body of a tree that never rests (it acts on what it touches)."""
KINEMATIC_AGENT = wp.constant(2)
"""A body's kind in ``body_agent``: a kinematic body (it acts on what it touches while it moves)."""


@wp.func
def _agent(
    body: int,
    body_agent: wp.array[wp.int32],
    body_slot: wp.array[wp.int32],
    body_qd: wp.array[wp.spatial_vector],
    moving: wp.array[wp.int32],
) -> bool:
    """Whether ``body`` acts on what it touches: a movable body of a tree that never rests, a kinematic body with a
    velocity, or a body of a tree that moved in the last substep."""
    if body < 0:
        return False
    kind = body_agent[body]
    if kind == AGENT:
        return True
    if kind == KINEMATIC_AGENT:
        v = body_qd[body]
        return v[0] != 0.0 or v[1] != 0.0 or v[2] != 0.0 or v[3] != 0.0 or v[4] != 0.0 or v[5] != 0.0
    s = body_slot[body]
    return s >= 0 and moving[s] != 0


@wp.func
def _counting(s: int, trees: int, tree_asleep: wp.array2d[wp.int32]) -> int:
    """``s`` where its tree can count a quiet substep (awake, not acted on, not loaded), else -1."""
    if s < 0:
        return -1
    state = tree_asleep[s // trees, s % trees]
    if state >= 0 or state == ACTED or state == LOADED:
        return -1
    return s


@wp.kernel
def _contact_load(
    dense_first: wp.array[int],
    body_base: int,
    routes: ContactRoutes,
    impulses: wp.array2d[wp.float32],
    constraint_count: wp.array[wp.int32],
    mf_constraint_count: wp.array[wp.int32],
    row_type: wp.array2d[wp.int32],
    row_parent: wp.array2d[wp.int32],
    enable_friction: int,
    inv_dt: float,
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_mu: wp.array[float],
    body_slot: wp.array[wp.int32],
    body_agent: wp.array[wp.int32],
    body_qd: wp.array[wp.spatial_vector],
    moving: wp.array[wp.int32],
    trees: int,
    tree_asleep: wp.array2d[wp.int32],
    rigid_contact_force: wp.array[wp.vec3],
    load_n: wp.array[float],
    load_t: wp.array[wp.vec3],
    pressed: wp.array[wp.int32],
):
    """Export the force of each contact a tree that can count touches (the solver's export) and lump it on such trees:
    each tree's cone capacity ``sum mu f_n`` and its friction force ``sum f_t``, and whether an agent (:func:`_agent`)
    touches it. Nothing for a contact between trees that cannot count (acted on, loaded, or never resting)."""
    c = wp.tid()
    if c >= routes.count[0]:
        return
    sa = shape0[c]
    sb = shape1[c]
    body_a = int(-1)
    body_b = int(-1)
    if sa >= 0:
        body_a = shape_body[sa]
    if sb >= 0:
        body_b = shape_body[sb]
    a = int(-1)
    b = int(-1)
    if body_a >= 0:
        a = _counting(body_slot[body_a], trees, tree_asleep)
    if body_b >= 0:
        b = _counting(body_slot[body_b], trees, tree_asleep)
    if a < 0 and b < 0:
        return
    contact_linear_force(
        dense_first,
        body_base,
        routes,
        c,
        impulses,
        constraint_count,
        mf_constraint_count,
        row_type,
        row_parent,
        enable_friction,
        inv_dt,
        rigid_contact_force,
    )
    if a >= 0 and _agent(body_b, body_agent, body_slot, body_qd, moving):
        pressed[a] = 1
    if b >= 0 and _agent(body_a, body_agent, body_slot, body_qd, moving):
        pressed[b] = 1
    f = rigid_contact_force[c]
    n = routes.normal[c]
    f_n = -wp.dot(f, n)
    f_t = f + f_n * n
    capacity = contact_friction(sa, sb, shape_mu) * wp.max(f_n, 0.0)
    if a >= 0:
        wp.atomic_add(load_n, a, capacity)
        wp.atomic_add(load_t, a, f_t)
    if b >= 0:
        wp.atomic_add(load_n, b, capacity)
        wp.atomic_add(load_t, b, -f_t)


@wp.kernel
def _rest_test(
    slot_never: wp.array[wp.int32],
    slot_mass: wp.array[float],
    slot_body_start: wp.array[wp.int32],
    slot_bodies: wp.array[wp.int32],
    slot_dof_start: wp.array[wp.int32],
    slot_dofs: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    joint_f: wp.array[float],
    trees: int,
    speed_squared: float,
    stick: float,
    load_n: wp.array[float],
    load_t: wp.array[wp.vec3],
    pressed: wp.array[wp.int32],
    moving: wp.array[wp.int32],
    tree_asleep: wp.array2d[wp.int32],
):
    """Count each awake tree's quiet substeps (see the module docstring); consume its contact load; say whether it
    moved (for the next substep's :func:`_agent`). A tree that can count had its contacts' load this substep, so every
    counted substep is checked."""
    i = wp.tid()
    capacity = load_n[i]
    friction = load_t[i]
    touched = pressed[i] != 0
    load_n[i] = 0.0
    load_t[i] = wp.vec3(0.0)
    pressed[i] = 0
    moving[i] = 0
    if slot_never[i] != 0:
        return
    w = i // trees
    t = i % trees
    state = tree_asleep[w, t]
    if state >= 0:
        return
    moving[i] = 1
    if state == LOADED:
        # Its island held an acted-on tree at the last pass: it counts nothing until the pass releases it.
        return
    acted = bool(False)
    for k in range(slot_dof_start[i], slot_dof_start[i + 1]):
        if joint_f.shape[0] > 0:
            if joint_f[slot_dofs[k]] != 0.0:
                acted = True
    twice_energy = float(0.0)
    if not acted:
        for k in range(slot_body_start[i], slot_body_start[i + 1]):
            b = slot_bodies[k]
            v = body_qd[b]
            linear = wp.spatial_top(v)
            angular = wp.quat_rotate_inv(wp.transform_get_rotation(body_q[b]), wp.spatial_bottom(v))
            twice_energy += body_mass[b] * wp.dot(linear, linear) + wp.dot(angular, body_inertia[b] * angular)
            if body_f.shape[0] > 0:
                if wp.length_sq(wp.spatial_top(body_f[b])) + wp.length_sq(wp.spatial_bottom(body_f[b])) != 0.0:
                    acted = True
    if acted:
        # A joint force or a body wrench acts on it: it never rests while one does, nor does its island.
        tree_asleep[w, t] = ACTED
        return
    still = wp.length(friction) <= stick * capacity and twice_energy < slot_mass[i] * speed_squared
    moving[i] = wp.where(still, 0, 1)
    # A tree an agent touches is loaded by it, not at rest: its count restarts until the agent lets go.
    if still and not touched:
        tree_asleep[w, t] = wp.min(wp.max(state, AWAKE) + 1, READY)
    else:
        tree_asleep[w, t] = AWAKE


@wp.kernel
def _hold(dof_slot: wp.array[wp.int32], trees: int, tree_asleep: wp.array2d[wp.int32], v_out: wp.array[float]):
    """An asleep tree's solved velocity is zero: its coordinates stay as they are."""
    d = wp.tid()
    s = dof_slot[d]
    if s >= 0:
        if tree_asleep[s // trees, s % trees] >= 0:
            v_out[d] = 0.0


@wp.func
def _awake(i: int, slot_never: wp.array[wp.int32]) -> int:
    """A tree's state when it wakes with its count restarted: acted on where it never rests."""
    if slot_never[i] != 0:
        return ACTED
    return AWAKE


@wp.kernel
def _wake_worlds(
    world_mask: wp.array[wp.bool], slot_never: wp.array[wp.int32], trees: int, tree_asleep: wp.array2d[wp.int32]
):
    i = wp.tid()
    w = i // trees
    if world_mask.shape[0] == 0 or world_mask[w]:
        tree_asleep[w, i % trees] = _awake(i, slot_never)


@wp.kernel
def _wake_articulation_worlds(
    articulations: wp.array[wp.int32],
    art_to_world: wp.array[wp.int32],
    slot_never: wp.array[wp.int32],
    trees: int,
    tree_asleep: wp.array2d[wp.int32],
):
    w = art_to_world[articulations[wp.tid()]]
    if w >= 0:
        for t in range(trees):
            tree_asleep[w, t] = _awake(w * trees + t, slot_never)


class Sleep:
    """The rest law's state and launches of one solver (see the module docstring); ``solver.sleep``, built by
    :meth:`build` where a tree can rest."""

    def __init__(self, solver, layout: dict):
        model = solver.model
        device = model.device
        self.solver = solver
        self.world_count, self.trees = layout["shape"]
        with wp.ScopedDevice(device):
            never = layout["slot_never"].reshape(layout["shape"]) != 0
            self.tree_asleep = wp.array(np.where(never, int(ACTED), int(AWAKE)).astype(np.int32), dtype=wp.int32)
            self._every_world = wp.zeros(0, dtype=wp.bool)
            self.shape_sleep_index = wp.array(layout["shape_sleep_index"], dtype=wp.vec2i)
            self.slot_never = wp.array(layout["slot_never"], dtype=wp.int32)
            self.slot_mass = wp.array(layout["slot_mass"], dtype=wp.float32)
            self.slot_body_start = wp.array(layout["slot_body_start"], dtype=wp.int32)
            self.slot_bodies = wp.array(layout["slot_bodies"], dtype=wp.int32)
            self.slot_dof_start = wp.array(layout["slot_dof_start"], dtype=wp.int32)
            self.slot_dofs = wp.array(layout["slot_dofs"], dtype=wp.int32)
            self.body_slot = wp.array(layout["body_slot"], dtype=wp.int32)
            self.dof_slot = wp.array(layout["dof_slot"], dtype=wp.int32)
            slots = int(self.world_count * self.trees)
            self.body_agent = wp.array(layout["body_agent"], dtype=wp.int32)
            self.load_n = wp.zeros(slots, dtype=wp.float32)
            self.load_t = wp.zeros(slots, dtype=wp.vec3)
            self.pressed = wp.zeros(slots, dtype=wp.int32)
            self.moving = wp.zeros(slots, dtype=wp.int32)
            self._no_forces = wp.zeros(0, dtype=wp.float32)
            self._no_wrenches = wp.zeros(0, dtype=wp.spatial_vector)

    @classmethod
    def build(cls, solver) -> Sleep | None:
        """The law's state over ``solver``'s forest, or ``None`` where no tree can rest."""
        layout = cls.layout(solver)
        return None if layout is None or layout["slot_never"].all() else cls(solver, layout)

    def refresh(self) -> None:
        """Take a model change: the trees that never rest (drive gains, kinematic flags) and the shapes' sleep index
        are read again into the same arrays, and every tree wakes. A model built with no tree that can rest keeps
        none."""
        layout = self.layout(self.solver)
        for name in ("shape_sleep_index", "slot_never", "body_slot", "dof_slot", "body_agent"):
            getattr(self, name).assign(layout[name])
        self.wake(None)

    @staticmethod
    def layout(solver) -> dict | None:
        """Each tree's slot ``(world, index in its world)``, the trees that never rest, each shape's sleep index and
        the slots' bodies, coordinates and masses (host), or ``None`` without trees."""
        model, forest, cut = solver.model, solver.forest, solver.cut
        trees = forest.tree_root.size
        if trees == 0 or model.shape_count == 0:
            return None
        body_tree = forest.body_tree
        kinematic_body = (model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) != 0
        joint_child = model.joint_child.numpy()
        qd_start = model.joint_qd_start.numpy()
        joint_tree = np.where(joint_child >= 0, body_tree[np.maximum(joint_child, 0)], -1)
        dof_tree = np.repeat(joint_tree, np.diff(qd_start))
        never = np.zeros(trees, dtype=bool)
        np.logical_or.at(never, body_tree[kinematic_body & (body_tree >= 0)], True)
        kinematic_dofs = solver.factor.kinematic_dof_mask_host
        if kinematic_dofs is not None:
            np.logical_or.at(never, dof_tree[(np.asarray(kinematic_dofs) != 0) & (dof_tree >= 0)], True)
        # A drive acts on these trees: a coordinate with a drive gain never rests.
        driven = (model.joint_target_ke.numpy() != 0.0) | (model.joint_target_kd.numpy() != 0.0)
        if model.joint_target_mode is not None:
            # An actuated coordinate (a target or an effort actuator, JointTargetMode): MuJoCo's rule, an actuated
            # tree never sleeps.
            driven |= model.joint_target_mode.numpy() != int(JointTargetMode.NONE)
        np.logical_or.at(never, dof_tree[driven & (dof_tree >= 0)], True)
        # A row joins these trees to another: a loop's link (its primary piece and its shadow) or a mimic.
        linked = np.concatenate(
            (cut.primary, cut.shadow, joint_child[forest.mimics.follower], joint_child[forest.mimics.reference])
        )
        linked = body_tree[linked[linked >= 0].astype(np.int64)]
        never[linked[linked >= 0]] = True
        # Each tree's slot: its world and its index among its world's trees.
        world = forest.articulation_world[forest.tree_articulation].astype(np.int64)
        world_count = int(max(forest.world_count, world.max() + 1))
        order = np.lexsort((np.arange(trees), world))
        rank = np.empty(trees, dtype=np.int64)
        starts = np.searchsorted(world[order], np.arange(world_count))
        rank[order] = np.arange(trees) - starts[world[order]]
        width = int(rank.max() + 1)
        slot = world * width + rank
        slots = world_count * width
        slot_never = np.ones(slots, dtype=np.int32)
        slot_never[slot] = never.astype(np.int32)
        resting = np.where(never, -1, slot).astype(np.int32)

        def members(of_tree):
            keep = of_tree >= 0
            index = np.flatnonzero(keep)
            owner = slot[of_tree[keep]]
            order = np.argsort(owner, kind="stable")
            start = np.zeros(slots + 1, dtype=np.int32)
            np.add.at(start, owner + 1, 1)
            return np.cumsum(start).astype(np.int32), index[order].astype(np.int32)

        slot_body_start, slot_bodies = members(body_tree)
        slot_dof_start, slot_dofs = members(dof_tree)
        mass = model.body_mass.numpy()
        slot_mass = np.zeros(slots, dtype=np.float32)
        np.add.at(slot_mass, slot[body_tree[body_tree >= 0]], mass[body_tree >= 0])
        # Each shape: its tree's slot, (world, -1) static, (world, -2) on a body never held (kinematic).
        shape_body = model.shape_body.numpy()
        shape_world = model.shape_world.numpy()
        index = np.stack((shape_world, np.full_like(shape_world, -1)), axis=1).astype(np.int32)
        on_body = shape_body >= 0
        index[on_body & kinematic_body[np.maximum(shape_body, 0)], 1] = -2
        tree_of = np.where(on_body, body_tree[np.maximum(shape_body, 0)], -1)
        on_tree = (tree_of >= 0) & ~(on_body & kinematic_body[np.maximum(shape_body, 0)])
        index[on_tree, 0] = world[tree_of[on_tree]]
        index[on_tree, 1] = rank[tree_of[on_tree]]
        body_slot = np.where(body_tree >= 0, resting[np.maximum(body_tree, 0)], -1).astype(np.int32)
        return {
            "shape": (world_count, width),
            "shape_sleep_index": index,
            "slot_never": slot_never,
            "slot_mass": slot_mass,
            "slot_body_start": slot_body_start,
            "slot_bodies": slot_bodies,
            "slot_dof_start": slot_dof_start,
            "slot_dofs": slot_dofs,
            "body_slot": body_slot,
            "body_agent": np.where(
                kinematic_body, int(KINEMATIC_AGENT), np.where((body_tree >= 0) & (body_slot < 0), int(AGENT), 0)
            ).astype(np.int32),
            "dof_slot": np.where(dof_tree >= 0, resting[np.maximum(dof_tree, 0)], -1).astype(np.int32),
        }

    def collision_sleep_filter(self) -> tuple[wp.array, wp.array]:
        """The shapes' sleep index and the live sleep state, for :meth:`~newton.CollisionPipeline.configure_sleep_filter`."""
        return self.shape_sleep_index, self.tree_asleep

    def hold(self, v_out: wp.array) -> None:
        """Zero the solved velocity of every asleep tree (after the sweep, before the finish)."""
        wp.launch(
            _hold,
            dim=self.dof_slot.shape[0],
            inputs=[self.dof_slot, self.trees, self.tree_asleep],
            outputs=[v_out],
            device=self.solver.model.device,
        )

    def test(self, state_in, state_out, control, contacts, dt: float) -> None:
        """After the finish: lump the substep's contact forces on the trees, then count each tree's quiet substeps."""
        solver = self.solver
        model, placement = solver.model, solver.placement
        device = model.device
        if contacts is not None and contacts.rigid_contact_count is not None and placement.contact_pool is not None:
            pool = placement.contact_pool
            pool.launch(
                _contact_load,
                pool.launch_rows(),
                inputs=[
                    placement.dense_first,
                    placement.body_base,
                    bind_routes(placement, contacts),
                    placement.impulses,
                    placement.constraint_count,
                    placement.mf_constraint_count,
                    placement.row_type,
                    placement.row_parent,
                    int(solver.options.enable_contact_friction),
                    1.0 / dt,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    model.shape_body,
                    solver.laws.shape_material_mu,
                    self.body_slot,
                    self.body_agent,
                    state_out.body_qd,
                    self.moving,
                    self.trees,
                    self.tree_asleep,
                ],
                outputs=[contacts.rigid_contact_force, self.load_n, self.load_t, self.pressed],
                extent=contacts.rigid_contact_count,
                max_blocks=placement.contact_capacity_blocks,
            )
        joint_f = getattr(control, "joint_f", None) if control is not None else None
        body_f = state_in.body_f
        wp.launch(
            _rest_test,
            dim=self.slot_never.shape[0],
            inputs=[
                self.slot_never,
                self.slot_mass,
                self.slot_body_start,
                self.slot_bodies,
                self.slot_dof_start,
                self.slot_dofs,
                state_out.body_q,
                state_out.body_qd,
                self._no_wrenches if body_f is None else body_f,
                model.body_mass,
                model.body_inertia,
                self._no_forces if joint_f is None else joint_f,
                self.trees,
                SLEEP_SPEED * SLEEP_SPEED,
                STICK,
            ],
            outputs=[self.load_n, self.load_t, self.pressed, self.moving, self.tree_asleep],
            device=device,
        )

    def wake(self, world_mask: wp.array | None) -> None:
        """Wake every tree of the worlds ``world_mask`` selects (``None``: every tree), its quiet count restarted."""
        wp.launch(
            _wake_worlds,
            dim=self.world_count * self.trees,
            inputs=[self._every_world if world_mask is None else world_mask, self.slot_never, self.trees],
            outputs=[self.tree_asleep],
            device=self.solver.model.device,
        )

    def wake_articulations(self, articulations: wp.array | None) -> None:
        """Wake every tree of the worlds of ``articulations`` (``None``: every tree)."""
        if articulations is None:
            self.wake(None)
            return
        wp.launch(
            _wake_articulation_worlds,
            dim=articulations.shape[0],
            inputs=[articulations, self.solver.forest.art_to_world, self.slot_never, self.trees],
            outputs=[self.tree_asleep],
            device=self.solver.model.device,
        )
