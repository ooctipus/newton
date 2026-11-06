# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scalar branches: world-fixed roots over one-coordinate prismatic leaves (``scalar_branches.py``).

The CPU tests compare the scalar kernels, bound through ``ScalarBranchState``,
with the held star on the same inputs, at rounding level, and their inverse masses
with the float64 mass matrix (:mod:`.feather_pgs_mass_reference`). Stars with
revolute leaves run in the held tree's diagonal-mass group. ``TestGripperContactBound``
checks the CUDA arm-key press bound on the held star, the law both CUDA routes
integrate.
"""

import dataclasses
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import path_selection, tree_factor
from newton._src.solvers.feather_pgs import scalar_branches as sb
from newton._src.solvers.feather_pgs.device_proof import DeviceProof
from newton._src.solvers.feather_pgs.friction_patches import CONTACT_PATH_RESOLVED
from newton._src.solvers.feather_pgs.held_tree import HeldTree
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_mass_reference import MassReference
from newton.tests.test_feather_pgs_cache_validity import _set_clock

DT = 1.0 / 240.0
GRAVITY = (0.7, -1.2, -9.1)
# Fields the scalar family stops writing for its bodies; no consumer on its route reads them.
DROPPED_FIELDS = ("body_q_com", "body_v_s", "body_a_s", "body_f_s", "body_I_s", "body_inertia_terms")
# Fields the held owner writes only for free bodies, not for a fixed-base arm (it publishes body_v_s).
ARM_DROPPED_FIELDS = ("body_q_com", "body_a_s", "body_f_s", "body_I_s", "body_inertia_terms")
# CUDA bounds against ordinary dynamics: the keys' are FMA contraction and the rounding of the
# inverse-mass projection and gravity force, which ordinary dynamics evaluates in another order;
# the arm leaves the serial route for the held owner's float order.
KEY_BOUND = {"rtol": 3.0e-5, "atol": 5.0e-6}
# Key-only contact impulses [N·s] (1e-4 to 1e-2 per step in these tests) at the split's rounding. A key's normal
# row and its two tangent rows act on its one coordinate, and the tangents' diagonal is small (the key's axis is
# nearly normal to the floor), so the rounding of a contact point moves the split between them: by up to 0.3% of
# the normal impulse between the two routes, and by 0.13% within the ordinary route when only its pose arithmetic
# rounds differently. The keys' coordinates and velocities, which the split leaves unchanged, keep KEY_BOUND.
IMPULSE_BOUND = {"rtol": 5.0e-3, "atol": 1.0e-8}
# The arm moves from ordinary dynamics (every reference here: no cached owner takes these worlds,
# and cached_dynamics selects only that owner) to the held owner: the same law at the same mass
# cadence in another float order. It is bounded by the held-solve tests' 1e-5 floor (ae6baadb) plus
# the scalar route's own run-to-run spread. Their other term, twice the ordinary solver's distance
# from a held law, measures this very move, so it cannot bound it.
ARM_FLOOR = 1.0e-5
# The arm-key press (_press_key_with_tip, _gripper_switch). The pose floor is the first step's
# response at the frictionless position-correction gain pgs_beta / (D dt) to four float32 spacings
# of the tips' extent, one per body and route. The friction cone scales that gain (on the CPU twin
# by 1.40 in the loaded world and 0.86 in the other), so the floor admits about three spacings, and
# only while the contact stays penetrating: it starts three spacings deep, and a shift beyond that
# turns the row speculative and steepens the response toward five times the gain.
GRIPPER_POSE_SPACINGS = 4
# The fraction of the head route's iteration-budget error (its distance from the converged
# reference) a route switch may move the contact impulses by: five times CUDA's 2% (9.4e-6 N s,
# against the CPU twin's 4.8e-4 N s budget). TestGripperContactBound pins it on the CPU twin between
# about 0.03, which an ARM_FLOOR-sized difference of the arm's coordinates needs, and 0.16, above
# which a doubled drive stiffness passes in the loaded world.
GRIPPER_BUDGET_FRACTION = 0.1
# Sweeps of the converged reference: on the CPU twin 256 sweeps come within 1.7e-8 N s of it, and
# 1024 and 2048 match it bitwise.
GRIPPER_REFERENCE_ITERATIONS = 512
# Coordinates of the fixtures' arm. A star of as many leaves shares the arm's size group, which is
# then not diagonal: no scalar plan exists.
ARM_DOFS = 6
# Leaves of the CUDA owner fixtures: the key presses, the small stars and the keyboard-shaped worlds.
PRESS_LEAVES = 7
STAR_LEAVES = 12
KEYBOARD_LEAVES = 108


def build_branch_model(
    device="cpu",
    *,
    worlds=3,
    leaves=7,
    kinds=("prismatic",),
    root="fixed",
    arm=True,
    shapes=False,
    chain=False,
    multi=False,
    descendant=False,
    particle=False,
):
    """Per world: a tilted root over one-coordinate leaves (a keyboard's class) beside a fixed-base six-joint arm.

    Leaves carry nontrivial axes, anchors, COMs, drives, limits, armature,
    springs and damping. ``kinds`` cycles the leaves' joint kinds. ``root`` is
    ``"fixed"``, ``"d6"`` (zero coordinates), ``"revolute"`` or ``"free"``.
    ``chain`` hangs the second leaf from the first, ``multi`` makes the last
    leaf a two-coordinate D6 joint and ``descendant`` welds a body to the first
    leaf. ``shapes`` adds a box to each leaf and a sphere to the arm's tip;
    ``particle`` adds one particle to the model.
    """
    template = newton.ModelBuilder(gravity=GRAVITY)
    case = template.add_link(
        mass=1.2, com=wp.vec3(0.0, 0.01, -0.005), inertia=wp.mat33(0.02, 0.001, 0.0, 0.001, 0.05, 0.0, 0.0, 0.0, 0.06)
    )
    frame = wp.transform(wp.vec3(0.25, 0.0, 0.08), wp.quat_rpy(0.3, -0.2, 0.4))
    child_frame = wp.transform(wp.vec3(0.02, -0.01, 0.004), wp.quat_rpy(-0.1, 0.2, 0.0))
    if root == "fixed":
        joints = [template.add_joint_fixed(parent=-1, child=case, parent_xform=frame, child_xform=child_frame)]
    elif root == "d6":
        joints = [template.add_joint_d6(parent=-1, child=case, parent_xform=frame, child_xform=child_frame)]
    elif root == "revolute":
        joints = [
            template.add_joint_revolute(
                parent=-1, child=case, axis=newton.Axis.Z, parent_xform=frame, child_xform=child_frame
            )
        ]
    else:
        joints = [template.add_joint_free(child=case, parent_xform=frame)]
    leaf_bodies = []
    for index in range(leaves):
        kind = kinds[index % len(kinds)]
        body = template.add_link(
            mass=0.012 + 1.0e-3 * index,
            com=wp.vec3(0.0005, -0.0003, 0.002),
            inertia=wp.mat33(2.0e-6, 1.0e-7, 0.0, 1.0e-7, 2.0e-6, 0.0, 0.0, 0.0, 3.0e-6),
        )
        common = {
            "parent": leaf_bodies[0] if chain and index == 1 else case,
            "child": body,
            "parent_xform": wp.transform(
                wp.vec3(-0.17 + 0.019 * index, 0.01, 0.012), wp.quat_rpy(0.2, -0.1 * index, 0.3)
            ),
            "child_xform": wp.transform(wp.vec3(-0.009, 0.004, 0.008), wp.quat_rpy(0.4, 0.2, -0.1)),
        }
        drive = {
            "target_ke": 300.0 + 10.0 * index,
            "target_kd": 1.5,
            "armature": 1.0e-4,
            "damping": 0.02 + 0.001 * index,
            "spring_stiffness": 0.4,
            "spring_ref": -0.001,
        }
        if multi and index == leaves - 1:
            axis = newton.ModelBuilder.JointDofConfig
            joints.append(
                template.add_joint_d6(
                    linear_axes=[axis(axis=newton.Axis.X), axis(axis=newton.Axis.Z)],
                    **common,
                )
            )
        elif kind == "prismatic":
            joints.append(
                template.add_joint_prismatic(
                    axis=wp.vec3(0.2 + 0.1 * index, 0.5, 0.9), limit_lower=-0.004, limit_upper=0.0, **common, **drive
                )
            )
        else:
            joints.append(
                template.add_joint_revolute(
                    axis=wp.vec3(0.3, -0.4 + 0.1 * index, 0.8), limit_lower=-0.5, limit_upper=0.5, **common, **drive
                )
            )
        leaf_bodies.append(body)
        if shapes:
            template.add_shape_box(body, hx=0.008, hy=0.008, hz=0.004)
    if descendant:
        tip = template.add_link(mass=0.004, inertia=wp.mat33(1.0e-7, 0.0, 0.0, 0.0, 1.0e-7, 0.0, 0.0, 0.0, 1.0e-7))
        joints.append(
            template.add_joint_fixed(
                parent=leaf_bodies[0], child=tip, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.01), wp.quat_identity())
            )
        )
    template.add_articulation(joints)
    if arm:
        base = template.add_link(mass=0.6, com=wp.vec3(0.0, 0.0, 0.02), inertia=wp.mat33(np.eye(3) * 1.0e-3))
        chain_joints = [
            template.add_joint_fixed(
                parent=-1, child=base, parent_xform=wp.transform(wp.vec3(0.0, 0.25, 0.0), wp.quat_identity())
            )
        ]
        parent = base
        for index in range(6):
            link = template.add_link(
                mass=0.15,
                com=wp.vec3(0.01, 0.0, 0.03),
                inertia=wp.mat33(2.0e-4, 0.0, 0.0, 0.0, 2.0e-4, 0.0, 0.0, 0.0, 1.0e-4),
            )
            chain_joints.append(
                template.add_joint_revolute(
                    parent=parent,
                    child=link,
                    axis=newton.Axis.Z if index % 2 == 0 else newton.Axis.Y,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()),
                    target_ke=50.0,
                    target_kd=2.0,
                    armature=1.0e-3,
                    limit_lower=-2.0,
                    limit_upper=2.0,
                )
            )
            parent = link
        if shapes:
            template.add_shape_sphere(
                parent, xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()), radius=0.01
            )
        template.add_articulation(chain_joints)
    builder = newton.ModelBuilder(gravity=GRAVITY)
    builder.replicate(template, worlds, spacing=(1.0, 0.0, 0.0))
    if particle:
        builder.add_particle(wp.vec3(0.0, 0.0, 1.0), wp.vec3(), 0.01)
    model = builder.finalize(device=device)
    rng = np.random.default_rng(5)
    model.joint_q.assign(rng.uniform(-0.004, 0.0, model.joint_coord_count).astype(np.float32))
    model.joint_qd.assign(rng.uniform(-0.05, 0.05, model.joint_dof_count).astype(np.float32))
    return model


def _excess(actual, desired, rtol=0.0):
    """Largest ``|actual - desired| - rtol * |desired|``: within ``atol`` exactly when assert_allclose passes."""
    actual, desired = np.asarray(actual, np.float64), np.asarray(desired, np.float64)
    return max(float(np.max(np.abs(actual - desired) - rtol * np.abs(desired))), 0.0)


def _assert_switch_within(test, label, switch, again, tolerance):
    """Bound an A/B difference by the tolerance plus the default arm's own run-to-run (A/A) spread.

    The A/A spread must itself stay within the tolerance: a larger one means the rollout's
    nondeterminism, not the switch, decides the comparison.
    """
    test.assertLessEqual(
        again,
        tolerance,
        f"{label}: the default arm differs from itself by {again:.3g} (> {tolerance:.3g}); "
        "the rollout cannot resolve the switch",
    )
    test.assertLessEqual(
        switch,
        tolerance + again,
        f"{label}: the switch changes it by {switch:.3g}, above {tolerance:.3g} plus the A/A spread {again:.3g}",
    )


def branch_articulations(model):
    """The first articulation of every world: the branch root's."""
    return np.arange(0, model.articulation_count, model.articulation_count // model.world_count)


def cpu_solver(model, size):
    """A CPU solver that carries the CUDA route's compact diagonal size, so that the scalar kernels and the head
    kernels bind the arrays the CUDA route binds."""
    solver = SolverFeatherPGS(model, enable_joint_limits=True, update_mass_matrix_interval=2)
    solver._compact_diagonal_mass_size = size
    return solver


def branch_size(model):
    """Coordinates of the branch articulation (the first of each world)."""
    start = model.articulation_start.numpy()[:2]
    return int(np.diff(model.joint_qd_start.numpy()[start])[0])


def same(test, expected, actual, index, name):
    """The same law in another float order: within the held-solve floor."""
    expected, actual = expected.numpy()[index], actual.numpy()[index]
    test.assertTrue(np.all(np.isfinite(actual)), name)
    np.testing.assert_allclose(actual, expected, rtol=ARM_FLOOR, atol=2.0e-6, err_msg=name)


def recipe_solver(model, *, scalar, pins=None, **options):
    """The keyboard recipe's solver, plus scalar branches if asked; ``pins`` are more route pins."""
    arguments = {
        "pgs_iterations": 8,
        "update_mass_matrix_interval": 2,
        "enable_joint_limits": True,
        "dense_max_constraints": 64,
        "mf_max_constraints": 64,
        **options,
    }
    pins = {**({} if scalar else {"star": "held"}), **(pins or {})}
    return SolverFeatherPGS(model, pins=SolverFeatherPGS.Pins(**pins), **arguments)


def dense_head(model, *, rows=64, **options):
    """The A/B head: the keyboard recipe on the general placement and dense storage, with the star in the held tree.

    ``rows`` is its dense row capacity: every key's limit rows are dense rows there. The general rows are inside
    the resolved-world classifier's law, the sparse-diagonal rows of the scalar route are not: the head alone leaves
    a world whose rows all separate without rows (:func:`_normal_impulses`), which changes the rows the arms keep,
    not their physics (the pusher press: key travel within 5.6e-7 and normal impulses within 3.0e-6 relative).
    """
    return recipe_solver(model, scalar=False, pins={"placement": "general"}, dense_max_constraints=rows, **options)


def install_held_contacts(model, state, contacts, pairs):
    """Hold one contact per ``(body, other body or -1 for the world)`` pair with fixed local witnesses.

    Both witnesses start at the first body's origin with normal ``-z`` (as
    ``install_contacts`` of the compact-contact tests), so a contact starts at
    zero gap and pushes back once its first body moves down against the other.
    """
    shape_body = model.shape_body.numpy()
    poses = state.body_q.numpy().astype(np.float64)
    capacity = contacts.rigid_contact_shape0.shape[0]
    shapes = np.full((2, capacity), -1, dtype=np.int32)
    points = np.zeros((2, capacity, 3), dtype=np.float32)
    for contact, bodies in enumerate(pairs):
        witness = poses[bodies[0], :3]
        for end, body in enumerate(bodies):
            if body < 0:
                points[end, contact] = witness
                continue
            shapes[end, contact] = int(np.flatnonzero(shape_body == body)[0])
            pose = poses[body]
            offset, rotation = witness - pose[:3], -pose[3:6]
            points[end, contact] = offset + 2.0 * np.cross(rotation, np.cross(rotation, offset) + pose[6] * offset)
    for end in range(2):
        getattr(contacts, f"rigid_contact_shape{end}").assign(shapes[end])
        getattr(contacts, f"rigid_contact_point{end}").assign(points[end])
        getattr(contacts, f"rigid_contact_margin{end}").zero_()
    contacts.rigid_contact_normal.assign(np.tile(np.array([0.0, 0.0, -1.0], np.float32), (capacity, 1)))
    contacts.rigid_contact_count.assign(np.array([len(pairs)], dtype=np.int32))


class _Layout:
    """Index sets of a branch model: the scalar family's coordinates, DOFs and bodies, and the arm's."""

    def __init__(self, model):
        starts = model.articulation_start.numpy()
        child, q_start, qd_start = model.joint_child.numpy(), model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        self.branches = branch_articulations(model)
        self.arms = self.branches + 1
        size = branch_size(model)
        leaves = [starts[a] + 1 + np.arange(size) for a in self.branches]
        self.key_coordinates = np.concatenate([q_start[joints] for joints in leaves])
        self.key_dofs = np.concatenate([qd_start[joints] for joints in leaves])
        self.key_bodies = np.concatenate([child[starts[a] : starts[a + 1]] for a in self.branches])
        self.key_leaves = np.concatenate([child[joints] for joints in leaves])
        self.arm_coordinates = np.concatenate(
            [np.arange(q_start[starts[a]], q_start[starts[a + 1]]) for a in self.arms]
        )
        self.arm_dofs = np.concatenate([np.arange(qd_start[starts[a]], qd_start[starts[a + 1]]) for a in self.arms])
        self.arm_bodies = np.concatenate([child[starts[a] : starts[a + 1]] for a in self.arms])
        self.arm_tips = child[starts[self.arms + 1] - 1]


class _Pair:
    """A held-star solver and a scalar-branch solver over one model, each with its own states and inputs."""

    def __init__(self, model, *, poison=False):
        size = branch_size(model)
        self.model = model
        self.head = SolverFeatherPGS(model, enable_joint_limits=True, update_mass_matrix_interval=2)
        self.mine = cpu_solver(model, size)
        self.reference = MassReference(model)
        self.plan, self.reason = sb.plan_scalar_branches(self.mine)
        if self.plan is None:
            raise AssertionError(self.reason)
        self.state = sb.ScalarBranchState(self.mine, self.plan)
        self.states = {name: [model.state(), model.state()] for name in ("head", "mine")}
        self.controls = {name: model.control() for name in ("head", "mine")}
        if poison:
            for name in DROPPED_FIELDS:
                getattr(self.mine.factor, name).fill_(float("nan"))
        starts = model.articulation_start.numpy()
        self.arts = self.mine.forest.group_to_art[size].numpy()
        self.bodies = np.concatenate([model.joint_child.numpy()[starts[a] : starts[a + 1]] for a in self.arts])
        self.coordinates = np.concatenate(
            [model.joint_q_start.numpy()[starts[a] + 1] + np.arange(size) for a in self.arts]
        )
        self.dofs = self.plan.dofs
        self.rng = np.random.default_rng(17)
        self.step_index = 0

    def inputs(self):
        """Draw the same targets, control forces and external wrenches for both routes."""
        model, rng = self.model, self.rng
        target_q = rng.uniform(-0.003, 0.0, self.controls["head"].joint_target_q.shape[0]).astype(np.float32)
        target_qd = rng.uniform(-0.01, 0.01, self.controls["head"].joint_target_qd.shape[0]).astype(np.float32)
        joint_f = rng.uniform(-2.0e-3, 2.0e-3, model.joint_dof_count).astype(np.float32)
        body_f = rng.uniform(-2.0e-2, 2.0e-2, (model.body_count, 6)).astype(np.float32)
        for name in ("head", "mine"):
            self.controls[name].joint_target_q.assign(target_q)
            self.controls[name].joint_target_qd.assign(target_qd)
            self.controls[name].joint_f.assign(joint_f)
            self.states[name][0].body_f.assign(body_f)

    def mass_mask(self, refresh):
        """Select every articulation for the inverse-mass refresh, or none."""
        values = np.full(self.model.articulation_count, int(refresh), dtype=np.int32)
        for solver in (self.head, self.mine):
            solver.factor.mass_update_mask.assign(values)

    def check_inverse_mass(self, test, joint_q, tag):
        """Hold the scalar inverse masses to the float64 mass matrix, armature and drive stiffness at ``joint_q``."""
        inverse = self.mine.factor.diagonal_inverse_mass.numpy()
        # The branches' drives stay inside their kernels: the head prepared the same stiffness.
        rows = self.head.factor.drive_row_by_dof.numpy()
        K = self.head.factor.aug_row_K.numpy()
        armature = self.model.joint_armature.numpy()
        # A kinematic coordinate's inverse mass is its prescription, not its inertia.
        kinematic = self.mine.factor.kinematic_dof_mask.numpy()
        for art in self.arts:
            start = int(self.mine.forest.articulation_dof_start_host[art])
            H = self.reference.mass_matrix(joint_q, int(art))
            for local in range(H.shape[0]):
                dof = start + local
                if kinematic[dof]:
                    continue
                drive = max(float(K[rows[dof]]), 0.0) if rows[dof] >= 0 else 0.0
                expected = 1.0 / (H[local, local] + armature[dof] + drive)
                test.assertAlmostEqual(inverse[dof] / expected, 1.0, delta=2.0e-6, msg=f"{tag} dof {dof}")

    def step(self, test):
        """Advance both routes one substep and compare what each phase publishes.

        Both routes integrate the same solved velocity: the head's predictor
        plus a random impulse response, standing in for the contact solve.
        """
        head_in, head_out = self.states["head"]
        mine_in, mine_out = self.states["mine"]
        head, held = self.head.factor, self.head.factor.held
        tag = f"step {self.step_index}"
        held.begin(head_in, head, head_in.joint_qd, DT)
        self.state.begin(mine_in, self.mine.factor, mine_in.joint_qd, DT)
        same(test, head_in.body_q, mine_in.body_q, self.bodies, f"{tag} input body_q")
        same(test, head.joint_S_s, self.mine.factor.joint_S_s, self.dofs, f"{tag} input joint_S_s")
        same(test, head.articulation_origin, self.mine.factor.articulation_origin, self.arts, f"{tag} input origin")
        head.prepare_drives(head_in, self.controls["head"], DT, self.head.v_hat, self.head.v_out)
        held.refresh(head)
        # The held predictor solves by the walk (Z4.2): the walk's refresh, as the step's factor stage runs it.
        head.walk_step(head_in, DT)
        held.predict(head_in, head, self.controls["head"], head_in.joint_qd, DT)
        self.state.predict(mine_in, self.mine.factor, self.controls["mine"], mine_in.joint_qd, DT)
        self.check_inverse_mass(test, mine_in.joint_q.numpy(), tag)
        same(test, self.head.v_hat, self.mine.v_hat, self.dofs, f"{tag} v_hat")
        solved = self.head.v_hat.numpy() + self.rng.uniform(-0.03, 0.03, self.model.joint_dof_count).astype(np.float32)
        for solver in (self.head, self.mine):
            solver.v_out.assign(solved)
        held.finish(head_in, head, head_out, DT)
        self.state.finish(mine_in, self.mine.factor, mine_out, DT)
        for name, index in (("joint_q", self.coordinates), ("joint_qd", self.dofs)):
            same(test, getattr(head_out, name), getattr(mine_out, name), index, f"{tag} {name}")
        for name in ("body_q", "body_qd"):
            same(test, getattr(head_out, name), getattr(mine_out, name), self.bodies, f"{tag} {name}")
        same(test, head.articulation_origin, self.mine.factor.articulation_origin, self.arts, f"{tag} origin")
        np.testing.assert_array_equal(self.mine.factor.fk_id_cache_valid.numpy()[self.arts], 1)
        self.states["head"].reverse()
        self.states["mine"].reverse()
        self.step_index += 1


def check_owner_launches(test, device):
    """Launch every kernel of an owner beside scalar branches; each launch must match its kernel's parameters.

    The owner composition of the held-tree launch check (``check_owner_launches`` of the cache-validity
    tests) that the keyboard class takes: a scalar group beside an arm's tree group, whose repair compares
    input banks. The owner takes two steps, the other DOFs' drive preparation included, and a copy-back begin.
    Every held-tree and scalar launch is checked before it runs, the scalar kernels' outputs included, and every kernel
    each group bound must have run: a tree group's by the cache its launches pass, the scalar group's by
    kernel. On CPU the owner is built directly, as ``cached_dynamics`` builds it on CUDA, where the
    recipe's solver installs it (construction's lane timing included).
    """
    from newton.tests.test_feather_pgs_cache_validity import _input_mismatches  # noqa: PLC0415

    device = wp.get_device(device)
    scalar_kernels = {
        sb.repair_scalar_branches: "repair",
        sb.predict_prismatic_branches: "predict",
        sb.finish_prismatic_branches: "finish",
        tree_factor.prepare_augmented_drives: "drives",
    }
    launched = []

    def checked(launch):
        def launch_checked(kernel, *args, **kwargs):
            if kernel.key.startswith("tree_kinetic_") or kernel in scalar_kernels:
                # Checked before the launch: Warp would run a struct passed in another struct's place.
                inputs = [*kwargs.get("inputs", ()), *kwargs.get("outputs", ())]
                test.assertEqual(_input_mismatches(kernel, inputs), [], kernel.key)
                launched.append((kernel, inputs))
            return launch(kernel, *args, **kwargs)

        return launch_checked

    with mock.patch.multiple(wp, launch=checked(wp.launch), launch_tiled=checked(wp.launch_tiled)):
        if device.is_cpu:
            model = build_branch_model(device, worlds=2)
            solver = cpu_solver(model, branch_size(model))
            solver._scalar_branches, _reason = sb.plan_scalar_branches(solver)
            owner = HeldTree(solver)
        else:
            model = build_branch_model(device, worlds=2, leaves=PRESS_LEAVES, shapes=True)
            solver = recipe_solver(model, scalar=True)
            owner = solver.factor.held
        test.assertIsInstance(owner, HeldTree)
        test.assertIsInstance(owner.scalar, sb.ScalarBranchState)
        groups = []
        for group in owner.groups:
            if group is owner.scalar:
                kernels = {
                    group.repair_kernel: "repair",
                    group.predictor_kernel: "predict",
                    group.finish_kernel: "finish",
                    tree_factor.prepare_augmented_drives: "drives",
                }
            else:
                test.assertTrue(group.bank_compare)
                kernels = {
                    group.repair_kernel: "repair",
                    group.finish_kernel: "finish",
                    group.predictor_kernel: "predict",
                }
            groups.append((group, kernels))
        test.assertEqual(sum(group is owner.scalar for group, _ in groups), 1)
        states, control = [model.state(), model.state()], model.control()
        for source, refresh in ((0, True), (1, False)):
            state, output = states[source], states[1 - source]
            _set_clock(solver.factor, refresh)
            owner.begin(state, solver.factor, state.joint_qd, DT)
            if owner.scalar.other_drive_dofs is not None:
                solver.factor.prepare_drives(
                    state, control, DT, solver.v_hat, solver.v_out, owner.scalar.other_drive_dofs
                )
            solver.factor.mass_update_mask.fill_(1)
            owner.refresh(solver.factor)
            solver.factor.walk_step(state, DT)
            owner.predict(state, solver.factor, control, state.joint_qd, DT)
            solver.v_out.assign(solver.v_hat)
            _set_clock(solver.factor, refresh, ahead=1)
            owner.finish(state, solver.factor, output, DT)
        # A copy of the bank the last finish published, one coordinate changed: the tree group's repair compares
        # the banks and repairs, the scalar group's stamp names the other bank.
        states[1].assign(states[0])
        coordinates = states[1].joint_q.numpy()
        coordinates[0] += np.float32(0.01)
        states[1].joint_q.assign(coordinates)
        _set_clock(solver.factor, False)
        owner.begin(states[1], solver.factor, states[1].joint_qd, DT, state_out=states[0])
    for group, kernels in groups:
        if group is owner.scalar:
            ran = {kernel for kernel, _ in launched}
        else:
            ran = {kernel for kernel, inputs in launched if any(value is group.data for value in inputs)}
        unlaunched = sorted(kind for kernel, kind in kernels.items() if kernel not in ran)
        test.assertEqual(unlaunched, [], type(group).__name__)
    test.assertTrue(np.isfinite(states[0].joint_q.numpy()).all())


class TestScalarBranchRule(unittest.TestCase):
    def test_world_fixed_one_coordinate_stars_are_admitted(self):
        """Admit prismatic leaf stars on FIXED and zero-coordinate D6 roots; leave revolute leaves to the held tree."""
        for kinds in (("revolute",), ("prismatic", "revolute")):
            with self.subTest(kinds=kinds):
                plan, reason = sb.plan_scalar_branches(cpu_solver(build_branch_model(kinds=kinds), 7))
                self.assertIsNone(plan)
                self.assertIn("revolute leaves run in the held tree's diagonal-mass group", reason)
        for root in ("fixed", "d6"):
            with self.subTest(root=root):
                model = build_branch_model(root=root)
                solver = cpu_solver(model, 7)
                plan, reason = sb.plan_scalar_branches(solver)
                self.assertIsNotNone(plan, reason)
                self.assertEqual((plan.size, plan.group_count), (7, model.world_count))
                self.assertIn("run as scalar branches", reason)
                starts = model.articulation_start.numpy()
                expected = np.concatenate(
                    [
                        model.joint_qd_start.numpy()[starts[a] + 1] + np.arange(7)
                        for a in solver.forest.group_to_art[7].numpy()
                    ]
                )
                np.testing.assert_array_equal(plan.dofs, np.sort(expected))
                # A plan holds an array, so it compares and hashes by identity.
                self.assertNotEqual(plan, dataclasses.replace(plan, dofs=plan.dofs.copy()))
                self.assertEqual(len({plan, plan}), 1)

    def test_admits_the_keyboard_shaped_fixture(self):
        """Admit the 108-key group of the two-world 108+6 fixture, whatever the arm's root."""
        from newton.tests.test_feather_pgs_star_rows import build_solver_fixture  # noqa: PLC0415

        model = build_solver_fixture("cpu")
        kinds, reason = sb.branch_topology(model, np.array([0, 2]), model.articulation_start.numpy()[1:])
        self.assertIsNotNone(kinds, reason)
        self.assertEqual(kinds.shape, (2, 108))
        self.assertTrue(np.all(kinds == int(newton.JointType.PRISMATIC)))

    def test_owner_fixtures_give_the_keys_their_own_group(self):
        """Give the keys of every CUDA owner fixture a diagonal-mass size group apart from the arm's."""
        for leaves in (PRESS_LEAVES, STAR_LEAVES, KEYBOARD_LEAVES):
            with self.subTest(leaves=leaves):
                model = build_branch_model(worlds=2, leaves=leaves, shapes=True)
                solver = SolverFeatherPGS(model, enable_joint_limits=True)
                self.assertEqual(sorted(solver.forest.size_groups), sorted((leaves, ARM_DOFS)))
                self.assertEqual(set(solver.forest.diagonal_mass_sizes), {leaves})
        # An arm-sized star joins the arm's group: no diagonal-mass size is left to admit.
        model = build_branch_model(worlds=2, leaves=ARM_DOFS, shapes=True)
        solver = SolverFeatherPGS(model, enable_joint_limits=True)
        self.assertEqual((solver.forest.size_groups, set(solver.forest.diagonal_mass_sizes)), ([ARM_DOFS], set()))

    def test_other_topologies_are_declined(self):
        """Decline chains, multi-coordinate leaves, welded descendants and roots that move."""
        for variant, words in (
            ({"chain": True}, "hang from the root body"),
            ({"descendant": True}, "hang from the root body"),
            ({"multi": True}, "one-coordinate"),
            ({"root": "revolute"}, "FIXED or zero-coordinate D6 root"),
            ({"root": "free"}, "FIXED or zero-coordinate D6 root"),
        ):
            with self.subTest(**variant):
                model = build_branch_model(worlds=2, **variant)
                joint_end = model.articulation_start.numpy()[1:]
                kinds, reason = sb.branch_topology(model, branch_articulations(model), joint_end)
                self.assertIsNone(kinds)
                self.assertIn(words, reason)

    def test_tree_groups_beside_scalar_branches_compare_banks(self):
        """Compare input banks in the tree groups beside scalar branches, as without them, and record it.

        The scalar group's cache is stamped with the bank its finish published
        into, and its repair compares the stamp alone ("identity"). After a
        finish into ``s1`` a step that reads ``s0`` (a copy of ``s1`` after an
        odd substep count) launches the arm's comparing repair and the scalar
        repair. Without scalar branches the same solver's tree groups compare
        that bank too. Launches are recorded, not run.
        """
        model = build_branch_model(worlds=2)
        solver = cpu_solver(model, 7)
        solver._scalar_branches, _reason = sb.plan_scalar_branches(solver)
        owners = {"scalar": HeldTree(solver)}
        with mock.patch.object(solver, "_scalar_branches", None):
            owners["plain"] = HeldTree(solver)
        s0, s1 = model.state(), model.state()
        kinds = {}
        for owner in owners.values():
            for group in owner.groups:
                kinds[group.repair_kernel] = "compare" if group.bank_compare else "repair"

        def launches(call):
            launched = []

            def record(kernel, *args, **kwargs):
                launched.append(kernel)

            with mock.patch.multiple(wp, launch=record, launch_tiled=record):
                call()
            return [kinds[kernel] for kernel in launched if kernel in kinds]

        expected = {"scalar": ["compare", "repair"], "plain": ["compare", "compare"]}
        for name, owner in owners.items():
            copy_back = expected[name]
            with self.subTest(owner=name):
                for group in owner.groups:
                    if group is owner.scalar:
                        self.assertEqual(path_selection._scalar_group(group)["input_validation"], "identity")
                        continue
                    record = path_selection._tree_group(group)
                    self.assertEqual(record["input_validation"], "content")
                    self.assertTrue(group.bank_compare)
                    self.assertIn("_banked", record["kernels"]["repair"])
                launches(lambda owner=owner: owner.finish(s0, solver.factor, s1, DT))
                self.assertEqual(
                    launches(lambda owner=owner: owner.begin(s0, solver.factor, s0.joint_qd, DT, state_out=s1)),
                    copy_back,
                )

    def test_owner_launches_match_their_kernels(self):
        """Launch every kernel of an owner beside scalar branches with one input of each parameter's type, on CPU.

        See :func:`check_owner_launches`.
        """
        check_owner_launches(self, "cpu")

    def test_owner_proves_the_scalar_topology_through_its_tree_groups(self):
        """Validate a notification with the owner's and the arm group's proofs alone, and still refuse new topology.

        The arm group's proof holds the model topology, the joint axes and
        art_to_world and watches every joint frame, and the owner's coverage
        proof every size's group map, so the scalar group holds no proof of its
        own: a moved key root is reported through the arm group's proof, so the
        solver invalidates every cache, and a turned key axis or a changed
        parent is refused there.
        """
        joint = newton.ModelFlags.JOINT_PROPERTIES
        model = build_branch_model(worlds=2)
        solver = cpu_solver(model, 7)
        solver._scalar_branches, _reason = sb.plan_scalar_branches(solver)
        owner = HeldTree(solver)
        validate = DeviceProof.validate
        with mock.patch.object(DeviceProof, "validate", autospec=True, side_effect=validate) as proofs:
            self.assertTrue(owner.validate_model(joint))
        self.assertEqual(proofs.call_count, 2)
        frames = model.joint_X_p.numpy()
        frames[model.articulation_start.numpy()[owner.scalar.group_to_art.numpy()[1]], 2] += 0.01
        model.joint_X_p.assign(frames)
        self.assertFalse(owner.validate_model(joint))
        self.assertTrue(owner.validate_model(joint))
        axes = model.joint_axis.numpy().copy()
        turned = axes.copy()
        turned[owner.scalar.plan.dofs[0]] = (0.0, 0.0, 1.0)
        model.joint_axis.assign(turned)
        with self.assertRaisesRegex(RuntimeError, "joint_axis changed"):
            owner.validate_model(joint)
        model.joint_axis.assign(axes)
        self.assertTrue(owner.validate_model(joint))
        parents = model.joint_parent.numpy()
        parents[model.articulation_start.numpy()[owner.scalar.group_to_art.numpy()[0]] + 2] = -1
        model.joint_parent.assign(parents)
        with self.assertRaisesRegex(RuntimeError, "joint_parent changed"):
            owner.validate_model(newton.ModelFlags.JOINT_PROPERTIES)


class TestScalarBranchKernels(unittest.TestCase):
    def test_prismatic_branches_match_the_held_star(self):
        """Match the held star over steps, mass cadence, a masked reset and a re-pose.

        Every field the family publishes is compared: input and output poses,
        coordinates, body velocities, motion axes, predictors and the articulation
        origin, and the inverse masses with the float64 mass matrix. The dropped
        FK/ID fields of the scalar solver hold NaN throughout, so no scalar kernel
        reads them.
        """
        pair = _Pair(build_branch_model(), poison=True)
        worlds = pair.model.world_count
        for step in range(8):
            pair.inputs()
            pair.mass_mask(step % 2 == 0)
            if step == 2:
                # A masked reset of world 0 after writing its coordinates (Isaac Lab's reset).
                mask = wp.array([True] + [False] * (worlds - 1), dtype=wp.bool, device=pair.model.device)
                for name in ("head", "mine"):
                    current = pair.states[name][0]
                    q = current.joint_q.numpy()
                    q[pair.coordinates[: pair.plan.size]] -= 0.001
                    current.joint_q.assign(q)
                pair.head.factor.held.invalidate(mask)
                pair.state.invalidate(mask)
                np.testing.assert_array_equal(
                    pair.mine.factor.fk_id_cache_valid.numpy()[pair.arts], [0] + [1] * (worlds - 1)
                )
            if step == 4:
                # Re-pose the branch roots through their frames and notify, as Isaac Lab's resets do.
                roots = pair.model.articulation_start.numpy()[pair.arts]
                frames = pair.model.joint_X_p.numpy()
                frames[roots, :3] += (0.01, -0.02, 0.03)
                frames[roots, 3:] = np.asarray(wp.quat_rpy(0.25, -0.1, 0.5))
                pair.model.joint_X_p.assign(frames)
                # The owner's tree groups report the moved frames; the solver then invalidates every cache.
                pair.state.invalidate()
                pair.head.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
                np.testing.assert_array_equal(pair.mine.factor.fk_id_cache_valid.numpy()[pair.arts], 0)
            pair.step(self)

    def test_kinematic_and_held_mass_steps_match(self):
        """Match a kinematic leaf (a prescribed coordinate) and held inverse masses."""
        model = build_branch_model(worlds=2)
        flags = model.body_flags.numpy()
        flags[_Pair(model).bodies[3]] |= int(newton.BodyFlags.KINEMATIC)
        model.body_flags.assign(flags)
        pair = _Pair(model)
        kinematic = pair.dofs[np.flatnonzero(pair.mine.factor.kinematic_dof_mask.numpy()[pair.dofs])]
        self.assertEqual(len(kinematic), 1)
        for step in range(4):
            pair.inputs()
            pair.mass_mask(step == 0)
            pair.step(self)
        np.testing.assert_array_equal(pair.mine.factor.joint_qdd.numpy()[kinematic], 0.0)

    def test_other_drives_match_the_drive_kernel(self):
        """Prepare the drives outside the branches as the full drive kernel does and leave the branches' alone.

        The branches' predict evaluates their own drives (the twins above), so
        drive preparation covers only the other DOFs: their stiffness and
        forces carry the full kernel's bits, including a clamped force and a
        nonpositive stiffness, and nothing writes the branches' entries.
        """
        model = build_branch_model(worlds=3)
        pair = _Pair(model)
        layout = _Layout(model)
        rng = np.random.default_rng(23)
        ke, kd = model.joint_target_ke.numpy(), model.joint_target_kd.numpy()
        effort = model.joint_effort_limit.numpy()
        ke[layout.arm_dofs[0]] = kd[layout.arm_dofs[0]] = 0.0
        effort[layout.arm_dofs[1]] = 1.0e-3
        model.joint_target_ke.assign(ke)
        model.joint_target_kd.assign(kd)
        model.joint_effort_limit.assign(effort)
        state, control = model.state(), model.control()
        state.joint_q.assign(rng.uniform(-0.3, 0.3, model.joint_coord_count).astype(np.float32))
        state.joint_qd.assign(rng.uniform(-0.5, 0.5, model.joint_dof_count).astype(np.float32))
        control.joint_target_q.assign(rng.uniform(-0.2, 0.2, control.joint_target_q.shape[0]).astype(np.float32))
        control.joint_target_qd.assign(rng.uniform(-0.1, 0.1, control.joint_target_qd.shape[0]).astype(np.float32))
        head = pair.head
        head.factor.prepare_drives(state, control, DT, head.v_hat, head.v_out)
        pair.mine.factor.aug_row_K.fill_(-7.0)
        pair.mine.factor.joint_tau.fill_(-7.0)
        mine = pair.mine
        mine.factor.prepare_drives(state, control, DT, mine.v_hat, mine.v_out, pair.state.other_drive_dofs)
        others = pair.state.other_drive_dofs.numpy()
        np.testing.assert_array_equal(np.sort(np.concatenate([others, pair.dofs])), np.arange(model.joint_dof_count))
        self.assertEqual(path_selection._tree_group(pair.state)["drive_dofs"], len(others))
        rows = pair.mine.factor.drive_row_by_dof.numpy()
        np.testing.assert_array_equal(pair.mine.factor.joint_tau.numpy()[others], head.factor.joint_tau.numpy()[others])
        other_rows = rows[others][rows[others] >= 0]
        np.testing.assert_array_equal(
            pair.mine.factor.aug_row_K.numpy()[other_rows], head.factor.aug_row_K.numpy()[other_rows]
        )
        self.assertIn(0.0, head.factor.aug_row_K.numpy()[other_rows])
        self.assertIn(np.float32(1.0e-3), np.abs(head.factor.joint_tau.numpy()[others]))
        np.testing.assert_array_equal(pair.mine.factor.joint_tau.numpy()[pair.dofs], -7.0)
        np.testing.assert_array_equal(pair.mine.factor.aug_row_K.numpy()[rows[pair.dofs]], -7.0)


def _close(test, expected, actual, index, tag, *, rtol, atol):
    actual, expected = actual.numpy()[index], expected.numpy()[index]
    test.assertTrue(np.all(np.isfinite(actual)), tag)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=tag)


def _arm_within(test, reference, actual, bodies, tag, *, again=None):
    """Bound the arm bodies' A/B difference by ``ARM_FLOOR`` plus the scalar route's A/A spread ``again``."""
    actual = actual.numpy()[bodies]
    test.assertTrue(np.all(np.isfinite(actual)), tag)
    spread = 0.0 if again is None else _excess(again.numpy()[bodies], actual)
    _assert_switch_within(test, tag, _excess(actual, reference.numpy()[bodies]), spread, ARM_FLOOR)


def _contact_rows(test, solver, count):
    """The ``(world, slot)`` normal row of each of the first ``count`` contacts, in contact order."""
    world, slot = solver.placement.contact_world.numpy()[:count], solver.placement.contact_slot.numpy()[:count]
    test.assertTrue(np.all(slot >= 0), "every held contact owns a row")
    return world, slot


def _normal_impulses(test, solver, count):
    """The solved normal impulse [N·s] of each of the first ``count`` contacts, in contact order.

    A contact of a world the resolved-world classifier resolved (every row separates at the predictor) has no
    rows and carries none.
    """
    world, slot = solver.placement.contact_world.numpy()[:count], solver.placement.contact_slot.numpy()[:count]
    resolved = solver.placement.contact_path.numpy()[:count] == int(CONTACT_PATH_RESOLVED)
    test.assertTrue(np.all((slot >= 0) | resolved), "every contact owns a row or sits in a resolved world")
    return np.where(resolved, 0.0, solver.placement.impulses.numpy()[world, np.maximum(slot, 0)]).astype(np.float64)


def _key_inertia(model, leaves, dofs, axes):
    """A prismatic key's mass along its coordinate [kg]: ``m |S|^2 + armature + ke dt^2 + (kd + damping) dt``.

    ``axes`` are the keys' motion axes ``S`` in float64; the last terms are the implicit drive's and passive damper's.
    """
    m = model.body_mass.numpy()[leaves].astype(np.float64)
    ke = model.joint_target_ke.numpy()[dofs].astype(np.float64)
    kd = model.joint_target_kd.numpy()[dofs].astype(np.float64)
    armature = model.joint_armature.numpy()[dofs].astype(np.float64)
    damping = model.joint_damping.numpy()[dofs].astype(np.float64)
    return m * np.sum(axes * axes, axis=1) + armature + ke * DT * DT + (kd + damping) * DT


def _key_law(model, leaves, dofs, axes, q, qd, steps, push=None):
    """Integrate keys by their scalar law in float64 and return their coordinates after each step.

    Without contacts or active limits a prismatic key of mass ``m`` and unit
    motion axis ``S`` on a fixed root follows the implicit drive of zero targets
    with its passive spring and damping and gravity along the axis:
    ``(m |S|^2 + armature + ke dt^2 + (kd + damping) dt) (qd' - qd) / dt = m S.g + push
    + spring (ref - q) - damping qd - (ke (q + dt qd) + kd qd)``, then
    ``q' = q + dt qd'``. ``push(step)`` gives a step's external generalized force.
    """
    m = model.body_mass.numpy()[leaves].astype(np.float64)
    ke = model.joint_target_ke.numpy()[dofs].astype(np.float64)
    kd = model.joint_target_kd.numpy()[dofs].astype(np.float64)
    spring = model.joint_spring_stiffness.numpy()[dofs].astype(np.float64)
    reference = model.joint_spring_ref.numpy()[dofs].astype(np.float64)
    damping = model.joint_damping.numpy()[dofs].astype(np.float64)
    inertia = _key_inertia(model, leaves, dofs, axes)
    weight = m * (axes @ model.gravity.numpy()[0].astype(np.float64))
    q, qd = np.array(q, dtype=np.float64), np.array(qd, dtype=np.float64)
    coordinates = []
    for step in range(steps):
        force = 0.0 if push is None else push(step)
        drive = -(ke * (q + DT * qd) + kd * qd)
        qd = qd + DT * (weight + force + spring * (reference - q) - damping * qd + drive) / inertia
        q = q + DT * qd
        coordinates.append(q.copy())
    return np.asarray(coordinates)


def _press_key_with_tip(test, model, layout, solver, *, shift=0.0, steps=12):
    """Press every world's fifth key with its arm's tip through a held contact while 2 N push the tip down.

    ``shift`` raises the tip's contact witness along the contact normal [m]. Returns the contacts'
    normal impulses [N·s] and the keys' coordinates [m] after every step (``steps x worlds``), the
    normal rows' Delassus diagonal at the first step and its ``closed_form`` [1/kg] and the tips'
    largest world coordinate at the start [m]. The closed form is the key's inverse mass along the
    normal, ``(n.S)^2 / (m |S|^2 + armature + ke dt^2 + kd dt)``, plus the solver's CFM: the
    upright tip does not move along the normal.
    """
    keys = layout.key_leaves.reshape(model.world_count, -1)[:, 4]
    dofs = layout.key_dofs.reshape(model.world_count, -1)[:, 4]
    coordinates = layout.key_coordinates.reshape(model.world_count, -1)[:, 4]
    pairs = [(int(tip), int(key)) for tip, key in zip(layout.arm_tips, keys, strict=True)]
    states, control = [model.state(), model.state()], model.control()
    contacts = newton.Contacts(len(pairs), 0, device=model.device)
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    install_held_contacts(model, states[0], contacts, pairs)
    if shift:
        # The tip's witness sits at the upright tip's origin, whose z axis is the contact normal's.
        points = contacts.rigid_contact_point0.numpy()
        points[: len(pairs), 2] += np.float32(shift)
        contacts.rigid_contact_point0.assign(points)
    extent = float(np.abs(states[0].body_q.numpy()[layout.arm_tips, :3]).max())
    force = np.zeros((model.body_count, 6), dtype=np.float32)
    force[layout.arm_tips, 2] = -2.0
    impulses, travel, diagonal, closed_form = [], [], None, None
    for _step in range(steps):
        current, following = states
        current.body_f.assign(force)
        solver.step(current, following, control, contacts, DT)
        states = [following, current]
        rows = _contact_rows(test, solver, len(pairs))
        impulses.append(solver.placement.impulses.numpy()[rows].astype(np.float64))
        travel.append(following.joint_q.numpy()[coordinates].astype(np.float64))
        if diagonal is None:
            diagonal = solver.placement.diag.numpy()[rows].astype(np.float64)
            axes = solver.factor.joint_S_s.numpy()[dofs, :3].astype(np.float64)
            normals = contacts.rigid_contact_normal.numpy()[: len(pairs)].astype(np.float64)
            along = np.sum(normals * axes, axis=1)
            closed_form = along * along / _key_inertia(model, keys, dofs, axes) + solver.options.pgs_cfm
    return {
        "impulses": np.asarray(impulses),
        "travel": np.asarray(travel),
        "diagonal": diagonal,
        "closed_form": closed_form,
        "extent": extent,
    }


def _gripper_switch(head, other, reference, *, beta, dt):
    """Measure a route switch in :func:`_press_key_with_tip` against the pose floor and the head's iteration budget.

    Returns per world [N·s]: ``first``, the first step's impulse difference, and ``floor``,
    ``GRIPPER_POSE_SPACINGS`` float32 spacings of the tips' largest coordinate through the contact's
    position correction ``beta / (D dt)``; ``switch``, the largest difference over every step, and
    ``bound``, ``GRIPPER_BUDGET_FRACTION`` of ``budget`` (the head's largest distance from the
    converged ``reference``) plus the floor.
    """
    spacing = float(np.spacing(np.float32(head["extent"])))
    floor = GRIPPER_POSE_SPACINGS * beta * spacing / (head["diagonal"] * dt)
    budget = np.abs(head["impulses"] - reference["impulses"]).max(axis=0)
    return {
        "first": np.abs(other["impulses"][0] - head["impulses"][0]),
        "floor": floor,
        "switch": np.abs(other["impulses"] - head["impulses"]).max(axis=0),
        "budget": budget,
        "bound": GRIPPER_BUDGET_FRACTION * budget + floor,
    }


def _assert_gripper_switch(test, head, other, reference, *, beta, dt):
    """Hold a route switch in :func:`_press_key_with_tip` to the head's response, pose, budget and travel bounds.

    The head's normal rows' Delassus diagonal matches its closed form within 1e-4 (the tip adds
    under 2e-6 of it on the CPU twin), which keeps the pose floor finite, and the switch's matches
    the head's at the keys' FMA bound. The converged ``reference`` must stay within a quarter of the
    head's impulse at every step (the CPU twin's head comes within 11% and 8e-5 of it), so a broken
    reference cannot widen the budget. The impulses then compare at :func:`_gripper_switch`'s
    bounds and the travel at 1e-3 relative.
    """
    for press in (head, other, reference):
        test.assertTrue(np.all(np.isfinite(press["impulses"])) and np.all(np.isfinite(press["travel"])))
    test.assertGreater(float(head["impulses"].max()), 0.0, "The tip must press the key")
    np.testing.assert_allclose(head["diagonal"], head["closed_form"], rtol=1.0e-4, err_msg="diagonal law")
    np.testing.assert_allclose(other["diagonal"], head["diagonal"], rtol=KEY_BOUND["rtol"], err_msg="diagonal")
    switch = _gripper_switch(head, other, reference, beta=beta, dt=dt)
    test.assertTrue(
        np.all(np.abs(reference["impulses"] - head["impulses"]) <= 0.25 * np.abs(head["impulses"]) + switch["floor"]),
        f"the converged reference is {switch['budget']} N s from the head, beyond a quarter of its impulse",
    )
    test.assertTrue(
        np.all(switch["first"] <= switch["floor"]),
        f"first impulses differ by {switch['first']} N s, above the pose floor {switch['floor']} N s",
    )
    test.assertTrue(
        np.all(switch["switch"] <= switch["bound"]),
        f"impulses differ by {switch['switch']} N s, above {GRIPPER_BUDGET_FRACTION} of the iteration budget's "
        f"error {switch['budget']} N s plus the pose floor",
    )
    np.testing.assert_allclose(other["travel"], head["travel"], rtol=1.0e-3, atol=1.0e-7)


class TestGripperContactBound(unittest.TestCase):
    """The arm-key press bound of the CUDA A/B on ordinary dynamics, the law both CUDA routes integrate.

    Each test holds one stand-in against the head's eight sweeps and the converged reference:
    another float order of the tip's pose, the held owner's distance from ordinary dynamics in
    the arm, or a defect of the keys' response.
    """

    def _press(self, *, iterations=8, shift=0.0, arm_seed=None, defect=None):
        """Press keys on CPU ordinary dynamics (split mode).

        ``arm_seed`` moves every arm coordinate by ``ARM_FLOOR``, with signs drawn from that seed;
        ``defect`` changes the keys' response.
        """
        model = build_branch_model(worlds=2, leaves=STAR_LEAVES, shapes=True)
        layout = _Layout(model)
        if arm_seed is not None:
            signs = np.random.default_rng(arm_seed).choice([-1.0, 1.0], len(layout.arm_coordinates))
            q = model.joint_q.numpy()
            q[layout.arm_coordinates] += (ARM_FLOOR * signs).astype(np.float32)
            model.joint_q.assign(q)
        if defect == "double_stiffness":
            # The keys' inverse mass counts the drive stiffness K = ke dt^2 + kd dt twice.
            armature = model.joint_armature.numpy()
            ke, kd = model.joint_target_ke.numpy(), model.joint_target_kd.numpy()
            armature[layout.key_dofs] += ke[layout.key_dofs] * DT * DT + kd[layout.key_dofs] * DT
            model.joint_armature.assign(armature)
        elif defect == "no_drive":
            for name in ("joint_target_ke", "joint_target_kd"):
                values = getattr(model, name).numpy()
                values[layout.key_dofs] = 0.0
                getattr(model, name).assign(values)
        # The bound reads every step's rows: the complete path builds them in a world whose contacts separate too.
        solver = recipe_solver(model, scalar=False, pgs_iterations=iterations, pins={"claims": ()})
        return solver, _press_key_with_tip(self, model, layout, solver, shift=shift)

    def _head_and_reference(self):
        """The head's press at eight sweeps, the converged reference's, ``pgs_beta`` and the loaded world."""
        solver, head = self._press()
        _, reference = self._press(iterations=GRIPPER_REFERENCE_ITERATIONS)
        return head, reference, solver.options.pgs_beta, int(np.argmax(np.abs(head["impulses"]).max(axis=0)))

    def test_bound_admits_a_pose_shift(self):
        """Admit a one-spacing shift of the tip's contact witness, another float order of its pose.

        The first impulses move linearly with the shift, at the contact's position-correction gain
        ``pgs_beta / (D dt)`` scaled by the friction cone (1.40 and 0.86), within a factor of two of
        that gain, and every bound of ``test_gripper_contact_matches_ordinary_dynamics`` holds
        (0.35 and 0.21 of the pose floor). The loaded world's later impulses move by 3.7 times the
        floor alone: the bound needs its budget term.
        """
        head, reference, beta, loaded = self._head_and_reference()
        spacing = float(np.spacing(np.float32(head["extent"])))
        _, shifted = self._press(shift=spacing)
        gain = beta / (head["diagonal"] * DT)
        moved = head["impulses"][0] - shifted["impulses"][0]
        self.assertTrue(np.all((moved >= 0.5 * gain * spacing) & (moved <= 2.0 * gain * spacing)), moved)
        _assert_gripper_switch(self, head, shifted, reference, beta=beta, dt=DT)
        switch = _gripper_switch(head, shifted, reference, beta=beta, dt=DT)
        self.assertGreater(switch["switch"][loaded], switch["floor"][loaded], switch)

    def test_bound_admits_the_held_arm_floor(self):
        """Admit the held owner's documented distance from ordinary dynamics in the arm: ``ARM_FLOOR`` per coordinate.

        Moving every arm coordinate by ``ARM_FLOOR`` (1e-5 rad, four sign draws) moves the upright
        tip sideways, perpendicular to the contact normal, so the first impulses move by at most
        0.11 of the pose floor, but the friction history moves the loaded world's later impulses by
        7.8e-6 to 1.4e-5 N s, CUDA's size: 20 to 35 times the floor alone and up to 0.28 of the
        bound, which admits them for any ``GRIPPER_BUDGET_FRACTION`` above about 0.03.
        """
        head, reference, beta, loaded = self._head_and_reference()
        for seed in range(4):
            with self.subTest(seed=seed):
                _, moved = self._press(arm_seed=seed)
                _assert_gripper_switch(self, head, moved, reference, beta=beta, dt=DT)
                switch = _gripper_switch(head, moved, reference, beta=beta, dt=DT)
                self.assertGreater(switch["switch"][loaded], switch["floor"][loaded], switch)

    def test_bound_refuses_defects(self):
        """Refuse defects of the keys' response in every world, which holds ``GRIPPER_BUDGET_FRACTION`` below 0.16.

        An inverse mass that counts the drive stiffness twice, or keys without drives, stand in for
        defects of the scalar family: their contact rows' diagonal moves by tens of percent, their
        first impulses miss the pose floor by more than tenfold, and every world's impulses miss the
        budget bound, the doubled stiffness's by 1.6 times in the loaded world.
        """
        head, reference, beta, _ = self._head_and_reference()
        for defect in ("double_stiffness", "no_drive"):
            with self.subTest(defect=defect):
                _, broken = self._press(defect=defect)
                self.assertTrue(np.all(np.abs(broken["diagonal"] / head["diagonal"] - 1.0) > 0.1), broken["diagonal"])
                switch = _gripper_switch(head, broken, reference, beta=beta, dt=DT)
                self.assertTrue(np.all(switch["first"] > 10.0 * switch["floor"]), switch)
                self.assertTrue(np.all(switch["switch"] > switch["bound"]), switch)


@unittest.skipUnless(wp.is_cuda_available(), "Scalar branches run on the CUDA held-tree owner")
class TestScalarBranchOwner(unittest.TestCase):
    def _poison(self, solver, layout):
        """Fill what the scalar owner neither writes nor reads with NaN.

        For the keys: the FK/ID fields the scalar family stops writing, and the
        drive stiffness and forces its predict evaluates itself. For the arm:
        the FK/ID fields the head's serial route writes and the held owner does
        not for a fixed-base tree.
        """
        for names, bodies in ((DROPPED_FIELDS, layout.key_bodies), (ARM_DROPPED_FIELDS, layout.arm_bodies)):
            for name in names:
                array = getattr(solver.factor, name)
                values = array.numpy()
                values[bodies] = np.nan
                array.assign(values)
        tau, stiffness = solver.factor.joint_tau.numpy(), solver.factor.aug_row_K.numpy()
        tau[layout.key_dofs] = np.nan
        rows = solver.factor.drive_row_by_dof.numpy()[layout.key_dofs]
        # A coordinate without a drive row (-1) must not poison the last row, which is an arm's.
        stiffness[rows[rows >= 0]] = np.nan
        solver.factor.joint_tau.assign(tau)
        solver.factor.aug_row_K.assign(stiffness)

    def _assert_routes(self, head, scalar):
        """Require the routes an A/B compares: the head's ordinary dynamics and the scalar owner.

        The keys must form their own size group: an arm-sized star would share
        the arm's group, which the recipe cannot run (see
        ``test_owner_fixtures_give_the_keys_their_own_group``).
        """
        self.assertIsNone(head.factor.held.scalar)
        self.assertIsNone(head._scalar_branches)
        owner = scalar.factor.held
        self.assertIsInstance(owner, HeldTree)
        self.assertIsInstance(owner.scalar, sb.ScalarBranchState)
        self.assertIs(owner.scalar.predictor_kernel, sb.predict_prismatic_branches)
        self.assertNotEqual(owner.scalar.size, ARM_DOFS)
        self.assertEqual(sorted(group.sparse.size for group in owner.groups), sorted((ARM_DOFS, owner.scalar.size)))
        return owner

    def test_owner_matches_ordinary_dynamics(self):
        """Own 108-key worlds as scalar branches beside a held arm and track ordinary dynamics.

        Keys and their floor-contact impulses stay within the keys' bound of
        ordinary dynamics through the mass cadence, an inertial and drive
        notification, a masked reset after a coordinate write, a root re-pose, a
        state bank copied back into the input (which every cache must repair)
        and captured graphs with masked and full resets. The arm moves to the held owner's float order,
        within the held-solve floor (``ARM_FLOOR``) plus the twin's spread.
        An eager twin of the scalar route does the captured work step by step:
        each replay must give what it gives. The FK/ID fields that neither the
        family writes for the keys nor the held owner for the arm hold NaN
        throughout, so nothing on the route reads them.
        """
        device = wp.get_device("cuda:0")
        model = build_branch_model(device, worlds=4, leaves=KEYBOARD_LEAVES, shapes=True)
        layout = _Layout(model)
        starts, child = model.articulation_start.numpy(), model.joint_child.numpy()
        floors = [int(child[starts[a] + 1 + k]) for a in layout.branches for k in (0, 17, 53, 107)]
        # The head, the scalar route (captured below) and the scalar route's eager twin.
        solvers = [
            # Every key's limit rows are dense rows on the head: with the arm's and the contacts', 240 rows a world.
            dense_head(model, rows=256),
            recipe_solver(model, scalar=True),
            recipe_solver(model, scalar=True),
        ]
        head, scalar, twin = solvers
        owners = [self._assert_routes(head, solver) for solver in (scalar, twin)]
        for solver in (scalar, twin):
            self._poison(solver, layout)
        states = [[model.state(), model.state()] for _ in solvers]
        controls = [model.control() for _ in solvers]
        contacts = [newton.Contacts(len(floors), 0, device=device) for _ in solvers]
        for index, pair in enumerate(states):
            newton.eval_fk(model, pair[0].joint_q, pair[0].joint_qd, pair[0])
            install_held_contacts(model, pair[0], contacts[index], [(body, -1) for body in floors])
        worlds = model.world_count

        def inputs(sequence):
            force = np.zeros((model.body_count, 6), dtype=np.float32)
            force[floors[1::4], 2] = -0.3 * sequence
            force[layout.arm_tips, :3] = (0.05 * sequence, -0.1, 0.02)
            for index, pair in enumerate(states):
                for state in pair:
                    state.body_f.assign(force)
                controls[index].joint_f.assign(np.linspace(-0.01, 0.02, model.joint_dof_count, dtype=np.float32))
                controls[index].joint_target_q.fill_(-0.0004 * sequence)

        def write_coordinates(reset):
            for pair in states:
                q, qd = pair[0].joint_q.numpy(), pair[0].joint_qd.numpy()
                for world, selected in enumerate(reset):
                    if selected:
                        q[layout.key_coordinates.reshape(worlds, -1)[world]] -= 0.0005
                        qd[layout.arm_dofs.reshape(worlds, -1)[world]] *= -0.5
                pair[0].joint_q.assign(q)
                pair[0].joint_qd.assign(qd)

        def compare(tag):
            for name, index in (("joint_q", layout.key_coordinates), ("joint_qd", layout.key_dofs)):
                _close(
                    self, getattr(states[0][0], name), getattr(states[1][0], name), index, f"{tag} {name}", **KEY_BOUND
                )
            for name in ("body_q", "body_qd"):
                _close(
                    self,
                    getattr(states[0][0], name),
                    getattr(states[1][0], name),
                    layout.key_bodies,
                    f"{tag} key {name}",
                    **KEY_BOUND,
                )
                _arm_within(
                    self,
                    getattr(states[0][0], name),
                    getattr(states[1][0], name),
                    layout.arm_bodies,
                    f"{tag} arm {name}",
                    again=getattr(states[2][0], name),
                )
            _close(self, head.v_hat, scalar.v_hat, layout.key_dofs, f"{tag} v_hat", **KEY_BOUND)
            # The floor contacts are key-only rows: their impulses follow the head's within the keys' bound.
            impulses = [_normal_impulses(self, solver, len(floors)) for solver in solvers]
            np.testing.assert_allclose(impulses[1], impulses[0], err_msg=f"{tag} impulses", **IMPULSE_BOUND)
            loads.append(float(impulses[0].max()))
            # The twin does the scalar route's work eagerly: it must give the same state and impulses.
            for name, index in (("joint_q", layout.key_coordinates), ("joint_qd", layout.key_dofs)):
                _close(
                    self,
                    getattr(states[1][0], name),
                    getattr(states[2][0], name),
                    index,
                    f"{tag} twin {name}",
                    **KEY_BOUND,
                )
            for name in ("body_q", "body_qd"):
                bodies = np.concatenate([layout.key_bodies, layout.arm_bodies])
                _close(
                    self,
                    getattr(states[1][0], name),
                    getattr(states[2][0], name),
                    bodies,
                    f"{tag} twin {name}",
                    **KEY_BOUND,
                )
            np.testing.assert_allclose(impulses[2], impulses[1], err_msg=f"{tag} twin impulses", **IMPULSE_BOUND)
            for solver in solvers:
                solver.check_constraint_capacity()
                self.assertGreater(int(solver.placement.constraint_count.numpy().sum()), 0)
            for owner in owners:
                self.assertEqual(int(sum(int(group.status.numpy().sum()) for group in owner.groups[:-1])), 0)

        loads = []
        mask = wp.array([True] + [False] * (worlds - 1), dtype=wp.bool, device=device)
        for step in range(10):
            inputs(step + 1)
            if step == 3:
                model.body_mass.assign(model.body_mass.numpy() * 1.1)
                model.joint_armature.assign(model.joint_armature.numpy() * 1.2)
                model.joint_target_ke.assign(model.joint_target_ke.numpy() * 1.15)
                for solver in solvers:
                    solver.notify_model_changed(
                        newton.ModelFlags.BODY_INERTIAL_PROPERTIES | newton.ModelFlags.JOINT_DOF_PROPERTIES
                    )
            if step == 5:
                write_coordinates((True,) + (False,) * (worlds - 1))
                for index, solver in enumerate(solvers):
                    solver.reset(states[index][0], mask)
            if step == 7:
                # Lift the keyboards through their fixed roots, as Isaac Lab's resets re-pose them.
                frames = model.joint_X_p.numpy()
                frames[starts[layout.branches], 2] += 0.001
                model.joint_X_p.assign(frames)
                for solver in solvers:
                    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
            for index, solver in enumerate(solvers):
                current, following = states[index]
                solver.step(current, following, controls[index], contacts[index], DT)
                if step == 8:
                    # Copy the output back into the input bank, as Isaac Lab does after an odd number of
                    # substeps: the next step reads a bank no cache published, so every route repairs.
                    current.assign(following)
                else:
                    states[index] = [following, current]
            compare(f"step {step}")

        graphs = []
        for index, solver in enumerate(solvers[:2]):
            current, following = states[index]
            with wp.ScopedCapture(device=device) as capture:
                solver.reset(current, mask)
                solver.step(current, following, controls[index], contacts[index], DT)
                solver.step(following, current, controls[index], contacts[index], DT)
            graphs.append(capture.graph)
        # Mask contents, coordinates and forces change between replays without rebinding graph inputs.
        for replay, reset in enumerate(((True, False, False, False), (False, True, True, False), (True,) * worlds)):
            mask.assign(np.asarray(reset, dtype=np.bool_))
            inputs(replay + 11)
            write_coordinates(reset)
            for solver in (scalar, twin):
                self._poison(solver, layout)
            for graph in graphs:
                wp.capture_launch(graph)
            # The eager twin of the captured work.
            current, following = states[2]
            twin.reset(current, mask)
            twin.step(current, following, controls[2], contacts[2], DT)
            twin.step(following, current, controls[2], contacts[2], DT)
            compare(f"replay {replay}")
        # The comparison is not vacuous: the floors carried the keys.
        self.assertGreater(max(loads), 0.0)

    def test_cuda_owner_launches_match_their_kernels(self):
        """Launch every kernel the recipe's owner beside scalar branches binds, construction's included, with matching inputs."""
        check_owner_launches(self, "cuda:0")

    def test_drive_kernel_covers_only_the_other_dofs(self):
        """Launch the drive preparation over the DOFs outside the branches only; the branches fold theirs.

        Without this restriction the full drive kernel runs over every DOF,
        the branches' 1.77 M of keyboard-shaped worlds included.
        """
        device = wp.get_device("cuda:0")
        model = build_branch_model(device, worlds=2, leaves=STAR_LEAVES, shapes=True)
        solver = recipe_solver(model, scalar=True)
        states, control = [model.state(), model.state()], model.control()
        contacts = newton.Contacts(1, 0, device=device)
        contacts.rigid_contact_count.zero_()
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        launched = []
        original = wp.launch

        def record(kernel, *args, **kwargs):
            launched.append((kernel, kwargs.get("dim")))
            return original(kernel, *args, **kwargs)

        with mock.patch.object(wp, "launch", record):
            solver.step(states[0], states[1], control, contacts, DT)
        dims = [dim for kernel, dim in launched if kernel is tree_factor.prepare_augmented_drives]
        self.assertEqual(dims, [model.joint_dof_count - STAR_LEAVES * model.world_count])
        self.assertTrue(np.isfinite(states[1].joint_q.numpy()).all())

    def test_key_press_follows_the_drive_law(self):
        """Press keys with an external force and release them: both routes follow the drive's closed form.

        Without contacts or active limits each key is a scalar system: the
        implicit drive gives ``(m |S|^2 + armature + K) qdd = u0 + forces`` with
        ``K = ke dt^2 + kd dt`` and ``u0 = -(ke (q + dt qd) + kd qd)`` for zero
        targets, plus the passive spring and damping and gravity along the axis.
        The press and the spring-back are integrated in float64 from the model's
        values and compared with both routes, which also match each other.
        """
        device = wp.get_device("cuda:0")
        model = build_branch_model(device, worlds=2, leaves=PRESS_LEAVES, shapes=True)
        layout = _Layout(model)
        lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
        lower[layout.key_dofs], upper[layout.key_dofs] = -0.5, 0.5
        model.joint_limit_lower.assign(lower)
        model.joint_limit_upper.assign(upper)
        model.joint_q.zero_()
        model.joint_qd.zero_()
        solvers = [recipe_solver(model, scalar=False, pins={"placement": "general"}), recipe_solver(model, scalar=True)]
        self._assert_routes(*solvers)
        states = [[model.state(), model.state()] for _ in solvers]
        controls = [model.control() for _ in solvers]
        contacts = [newton.Contacts(1, 0, device=device) for _ in solvers]
        for control, contact, pair in zip(controls, contacts, states, strict=True):
            control.joint_target_q.zero_()
            control.joint_target_qd.zero_()
            contact.rigid_contact_count.zero_()
            newton.eval_fk(model, pair[0].joint_q, pair[0].joint_qd, pair[0])
        press, release = 20, 60
        travel = [[], []]
        axes = None
        for step in range(press + release):
            force = np.zeros((model.body_count, 6), dtype=np.float32)
            if 0 < step <= press:
                # Along each key's motion axis, which the first step published.
                force[layout.key_leaves, :3] = -1.5 * axes / np.linalg.norm(axes, axis=1, keepdims=True)
            for index, solver in enumerate(solvers):
                current, following = states[index]
                current.body_f.assign(force)
                solver.step(current, following, controls[index], contacts[index], DT)
                states[index] = [following, current]
                travel[index].append(following.joint_q.numpy()[layout.key_coordinates].copy())
            if step == 0:
                axes = solvers[1].factor.joint_S_s.numpy()[layout.key_dofs, :3].astype(np.float64)
                np.testing.assert_allclose(
                    solvers[0].factor.joint_S_s.numpy()[layout.key_dofs, :3], axes, rtol=1.0e-6, atol=1.0e-7
                )
        travel = np.asarray(travel, dtype=np.float64)
        np.testing.assert_allclose(travel[1], travel[0], rtol=1.0e-5, atol=1.0e-8)
        # The same discrete law in float64; the press is the force's projection on each axis.
        pressing = -1.5 * np.linalg.norm(axes, axis=1)
        rest = np.zeros(len(layout.key_dofs))
        expected = _key_law(
            model,
            layout.key_leaves,
            layout.key_dofs,
            axes,
            rest,
            rest,
            press + release,
            push=lambda step: pressing if 0 < step <= press else 0.0,
        )
        self.assertGreater(float(np.abs(expected).max()), 2.0e-3)
        for index in range(2):
            np.testing.assert_allclose(travel[index], expected, rtol=1.0e-4, atol=2.0e-6)

    def test_pusher_press_matches_ordinary_dynamics(self):
        """Press a key with a kinematic pusher: travel and impulses follow the head, the spring-back the key law.

        The pusher is a frictionless world contact above one key per world; its
        witness descends at constant speed, holds and then leaves. The contact is
        key-only (one sparse coordinate), so both routes solve the same row from
        the same motion axis, origin and inverse mass and differ by FMA
        contraction and the inverse-mass projection's rounding alone: travel and
        normal impulses within 1e-5 relative. Once
        the pusher leaves, each key springs back by its scalar law, integrated in
        float64 from the state it was released in.
        """
        device = wp.get_device("cuda:0")
        model = build_branch_model(device, worlds=2, leaves=PRESS_LEAVES, shapes=True)
        layout = _Layout(model)
        worlds = model.world_count
        lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
        lower[layout.key_dofs], upper[layout.key_dofs] = -0.5, 0.5
        model.joint_limit_lower.assign(lower)
        model.joint_limit_upper.assign(upper)
        model.joint_q.zero_()
        model.joint_qd.zero_()
        # Frictionless keys: the pusher's normal impulse alone moves a key along its axis.
        mu = model.shape_material_mu.numpy()
        mu[np.isin(model.shape_body.numpy(), layout.key_leaves)] = 0.0
        model.shape_material_mu.assign(mu)
        solvers = [dense_head(model), recipe_solver(model, scalar=True)]
        self._assert_routes(*solvers)
        states = [[model.state(), model.state()] for _ in solvers]
        controls = [model.control() for _ in solvers]
        contacts = [newton.Contacts(worlds, 0, device=device) for _ in solvers]
        for control, contact, pair in zip(controls, contacts, states, strict=True):
            control.joint_target_q.zero_()
            control.joint_target_qd.zero_()
            contact.rigid_contact_count.zero_()
            newton.eval_fk(model, pair[0].joint_q, pair[0].joint_qd, pair[0])

        def advance():
            for index, solver in enumerate(solvers):
                current, following = states[index]
                current.clear_forces()
                solver.step(current, following, controls[index], contacts[index], DT)
                states[index] = [following, current]

        # A free step publishes the motion axes; the pusher presses the key closest to vertical.
        advance()
        axes = solvers[1].factor.joint_S_s.numpy()[layout.key_dofs, :3].astype(np.float64)
        np.testing.assert_allclose(
            solvers[0].factor.joint_S_s.numpy()[layout.key_dofs, :3], axes, rtol=1.0e-6, atol=1.0e-7
        )
        key = int(np.argmax(np.abs(axes[:PRESS_LEAVES, 2])))
        self.assertGreater(abs(axes[key, 2]), 0.5)
        pressed = layout.key_leaves.reshape(worlds, -1)[:, key]
        coordinates = layout.key_coordinates.reshape(worlds, -1)[:, key]
        dofs = layout.key_dofs.reshape(worlds, -1)[:, key]
        # Both routes see one pusher: zero gap at the head's key, its normal from the key up to the pusher.
        for contact in contacts:
            install_held_contacts(model, states[0][0], contact, [(int(body), -1) for body in pressed])
            contact.rigid_contact_normal.assign(np.tile(np.array([0.0, 0.0, 1.0], np.float32), (worlds, 1)))
        top = contacts[0].rigid_contact_point1.numpy().copy()
        rate, descend, hold, release = 1.0e-4, 16, 8, 40
        travel, impulses = [[], []], [[], []]
        for step in range(descend + hold):
            depth = rate * min(step + 1, descend)
            for contact in contacts:
                contact.rigid_contact_point1.assign(top - np.array([0.0, 0.0, depth], np.float32))
            advance()
            for index, solver in enumerate(solvers):
                impulses[index].append(_normal_impulses(self, solver, worlds))
                travel[index].append(states[index][0].joint_q.numpy()[coordinates].copy())
        released = [(state[0].joint_q.numpy()[coordinates], state[0].joint_qd.numpy()[dofs]) for state in states]
        for contact in contacts:
            contact.rigid_contact_count.zero_()
        spring = [[], []]
        for _step in range(release):
            advance()
            for index in range(2):
                spring[index].append(states[index][0].joint_q.numpy()[coordinates].copy())
        travel, impulses, spring = (np.asarray(values, dtype=np.float64) for values in (travel, impulses, spring))
        for values in (travel, impulses, spring):
            self.assertTrue(np.all(np.isfinite(values)))
        # The pusher holds each key against its drive, deeper than the key sags under gravity.
        self.assertTrue(np.all(impulses[:, descend:] > 0.0))
        self.assertTrue(np.all(np.abs(travel[:, -1]) > 1.0e-3))
        np.testing.assert_allclose(impulses[1], impulses[0], rtol=1.0e-5, atol=1.0e-9)
        np.testing.assert_allclose(travel[1], travel[0], rtol=1.0e-5, atol=1.0e-9)
        np.testing.assert_allclose(spring[1], spring[0], rtol=1.0e-5, atol=1.0e-9)
        pressed_axes = axes.reshape(worlds, -1, 3)[:, key]
        for index in range(2):
            expected = _key_law(model, pressed, dofs, pressed_axes, *released[index], release)
            np.testing.assert_allclose(spring[index], expected, rtol=1.0e-4, atol=1.0e-7)

    def test_gripper_contact_matches_ordinary_dynamics(self):
        """Press a key with the arm's tip: travel and impulses stay within the head's pose and budget bounds.

        The scalar route changes two float orders at once: the key's (the scalar kernels instead of
        ordinary dynamics) and the arm's (the held owner instead of ordinary dynamics). Both
        integrate the same law, so the bound is what that law leaves undetermined at the recipe's
        eight sweeps (``_assert_gripper_switch``):

        - The contact response is determined: the normal row's Delassus diagonal ``D``, 29 1/kg,
          is the key's inverse mass along the normal (the upright arm's tip adds under 2e-6 of it on
          the CPU twin). The head's matches that closed form within 1e-4, and the scalar route's the
          head's within the keys' bound.
        - The first step starts both routes from one state. They differ by the float order of the
          tip's and the key's poses, about one float32 spacing of the contact's world position each,
          which the contact's position correction turns into impulse at ``pgs_beta / (D dt)``
          without friction: 1.6 N s/m, 1e-7 N s per spacing of 0.5 m. The first impulses must agree
          within the pose floor, that gain's response to four spacings. On the CPU twin the friction
          cone scales the gain by 1.40 and 0.86 while the contact stays penetrating (it starts three
          spacings deep), so the floor admits about three spacings. Measured on CUDA against the
          linear-state law the scalar kernels follow: 6.8e-8 and 4.2e-8 N s, 0.17 and 0.11 of the floor. The response is linear: on the CPU twin a 3e-8 m
          shift of the tip's witness (one spacing of 0.3 m) gives 6.9e-8 and 4.2e-8 N s, and
          ``TestGripperContactBound``'s one-spacing shift of the 0.5 m extent (6e-8 m) gives 1.4e-7
          and 8.4e-8 N s (0.35 and 0.21 of the floor).
        - Later steps inherit a friction history that eight sweeps leave unconverged: on the CPU
          twin the loaded world's key slides at the friction cone for four steps and then sticks
          1e-5 m deeper inside it, where the same head solver at ``GRIPPER_REFERENCE_ITERATIONS``
          sweeps holds it in place at the cone, and its normal impulse ends about 10% (4.8e-4 N s)
          short of that converged run. The history amplifies the first difference (CUDA: 9.4e-6
          N s, 2% of that error if CUDA's matches; a one-spacing shift on the CPU twin: 0.3%). Every
          step's impulses must agree within ``GRIPPER_BUDGET_FRACTION`` (a tenth) of the head's
          distance from the converged run plus the pose floor, so the scalar route stays within 1.1
          times the head's distance from that run plus the floor. On the CPU twin an
          ``ARM_FLOOR``-sized difference of the arm's coordinates, the held owner's documented
          distance from ordinary dynamics, moves the loaded world's impulses by up to 1.4e-5 N s,
          0.28 of the bound. The converged run must stay within a quarter of the head's impulse at
          every step, so a broken reference cannot widen the bound.
        - The key's travel moves far less (under 7e-8 m for a one-spacing shift on the CPU twin,
          against the budget's 1e-5 m) and keeps a 1e-3 relative bound.

        A defect in the scalar family's response misses these by orders of magnitude: on the CPU
        twin an inverse mass that counts the drive stiffness twice moves the diagonal by 29%, the
        first impulses by 135 and 620 times the pose floor and the later ones by 1.6 and 620 times
        the bound, and keys without drives carry no load.
        """
        device = wp.get_device("cuda:0")
        model = build_branch_model(device, worlds=2, leaves=STAR_LEAVES, shapes=True)
        layout = _Layout(model)
        head = recipe_solver(model, scalar=False, pins={"placement": "general"})
        scalar = recipe_solver(model, scalar=True)
        self._assert_routes(head, scalar)
        # The head route converged: the same solver with many more sweeps.
        reference = recipe_solver(
            model, scalar=False, pins={"placement": "general"}, pgs_iterations=GRIPPER_REFERENCE_ITERATIONS
        )
        presses = [_press_key_with_tip(self, model, layout, solver) for solver in (head, scalar, reference)]
        _assert_gripper_switch(self, *presses, beta=head.options.pgs_beta, dt=DT)

    def test_revolute_leaves_run_in_the_held_diagonal_group(self):
        """A star with revolute leaves stays out of scalar branches: the held tree's diagonal-mass group owns it."""
        device = wp.get_device("cuda:0")
        model = build_branch_model(device, worlds=2, leaves=STAR_LEAVES, kinds=("prismatic", "revolute"), shapes=True)
        solver = recipe_solver(model, scalar=True)
        self.assertIsNone(solver._scalar_branches)
        self.assertIn("revolute leaves run in the held tree's diagonal-mass group", solver._scalar_branches_reason)
        owner = solver.factor.held
        self.assertIsNone(owner.scalar)
        responses = sorted(group._response for group in owner.groups)
        self.assertEqual(responses, ["cholesky", "diagonal"])


if __name__ == "__main__":
    unittest.main()
