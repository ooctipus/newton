# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cached tree dynamics treat kinematic bodies as ordinary dynamics does.

Body flags are live model data. A body made kinematic or dynamic at runtime
changes only the kinematic masks and effective armature, which the solver
re-derives in place on a body-property notification and the held-tree owner
reads at every launch. The owner therefore stays installed across such
notifications, also through replays of a graph captured before them, and the
changes ordinary dynamics rejects raise its error before anything changes.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import joint_table
from newton._src.solvers.feather_pgs.held_tree import HeldTree, operator_metadata
from newton._src.solvers.feather_pgs.row_kernels import (
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
)
from newton._src.solvers.feather_pgs.tree_kinetic_plan import make_plan
from newton._src.solvers.feather_pgs.tree_plan import TreePlan
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_sparse_scene as sparse_scene
from newton.tests.feather_pgs_law_harness import recomputing

DT = 1.0 / 240.0
KINEMATIC = int(newton.BodyFlags.KINEMATIC)
DYNAMIC = int(newton.BodyFlags.DYNAMIC)
# Point friction and Newton's default patch friction admit the owner through different contact paths.
FRICTION_LAWS = {"point": 0.0, "patch": 0.2}
STATE_FIELDS = ("body_q", "body_qd", "joint_q", "joint_qd")
PRESCRIBED_ERROR = (
    "A prescribed-response articulation changed kinematic membership; "
    "reconstruct the solver and recapture its CUDA graph."
)


def _set_flag(model, body, flag):
    flags = model.body_flags.numpy()
    flags[body] = flag
    model.body_flags.assign(flags)


def _solver(model, beta, cached, **options):
    """A default solver under one friction law; ``cached="off"`` recomputes its state from every step's input."""
    solver = SolverFeatherPGS(model, angular_damping=0.0, friction_anchor_beta=beta, **options)
    return recomputing(solver) if cached == "off" else solver


def _toggle_scene():
    """One kinematic free body in zero gravity: the runtime-toggle scene of test_kinematic_links."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(xform=wp.transform_identity(), mass=1.0, is_kinematic=True)
    builder.add_shape_sphere(body, radius=0.1)
    return builder.finalize(device="cuda:0"), body


def _pair_scene():
    """A kinematic box beside a dynamic probe sphere; the box is a prescribed root in matrix-free mode."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    box = builder.add_body(xform=wp.transform(wp.vec3(-0.3, 0.0, 0.0), wp.quat_identity()), mass=1.0, is_kinematic=True)
    builder.add_shape_box(box, hx=0.25, hy=0.15, hz=0.15)
    probe = builder.add_body(xform=wp.transform(wp.vec3(0.45, 0.0, 0.0), wp.quat_identity()), mass=1.0)
    builder.add_shape_sphere(probe, radius=0.1)
    return builder.finalize(device="cuda:0"), box, probe


def _chain(*, floating):
    """A moving three-link revolute chain on a fixed or floating base, in zero gravity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    links, joints = [], []
    for i in range(3):
        link = builder.add_link(xform=wp.transform(wp.vec3(0.3 * i, 0.0, 1.0), wp.quat_identity()), mass=1.0)
        builder.add_shape_box(link, hx=0.1, hy=0.05, hz=0.05)
        if not links:
            base = wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity())
            joint = builder.add_joint_free(link) if floating else builder.add_joint_fixed(-1, link, parent_xform=base)
        else:
            offset = wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity())
            joint = builder.add_joint_revolute(links[-1], link, axis=newton.Axis.Y, parent_xform=offset)
        links.append(link)
        joints.append(joint)
    builder.add_articulation(joints)
    builder.joint_qd[:] = [0.3] * len(builder.joint_qd)
    return builder.finalize(device="cuda:0"), links


def _arm_and_box():
    """A fixed-base arm of two revolute links and a free box 6 mm below its last link, in zero gravity.

    Returns the model, the box and the arm's last link.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    mount = wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity())
    base = builder.add_link(xform=mount, mass=1.0)
    builder.add_shape_sphere(base, radius=0.02)
    joints = [builder.add_joint_fixed(-1, base, parent_xform=mount)]
    middle = builder.add_link(xform=wp.transform(wp.vec3(0.1, 0.0, 0.5), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(middle, hx=0.08, hy=0.03, hz=0.03)
    joints.append(
        builder.add_joint_revolute(
            base, middle, axis=newton.Axis.Z, child_xform=wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity())
        )
    )
    # The last link spans y in [-0.05, 0.05]; the box's upper face sits at y = -0.056.
    link = builder.add_link(xform=wp.transform(wp.vec3(0.4, 0.0, 0.5), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(link, hx=0.1, hy=0.05, hz=0.05)
    joints.append(
        builder.add_joint_revolute(
            middle,
            link,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
        )
    )
    builder.add_articulation(joints)
    box = builder.add_body(xform=wp.transform(wp.vec3(0.45, -0.106, 0.5), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
    return builder.finalize(device="cuda:0"), box, link


def _kinematic_root_and_box():
    """A tree with a kinematic floating root and a dynamic link, and a free box 6 mm ahead of the root along x.

    Returns the model, the root and the box.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    root = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), mass=1.0, is_kinematic=True)
    builder.add_shape_box(root, hx=0.1, hy=0.1, hz=0.1)
    child = builder.add_link(xform=wp.transform(wp.vec3(-0.3, 0.0, 0.5), wp.quat_identity()), mass=0.5)
    builder.add_shape_box(child, hx=0.1, hy=0.05, hz=0.05)
    hinge = wp.transform(wp.vec3(-0.3, 0.0, 0.0), wp.quat_identity())
    joints = [
        builder.add_joint_free(root),
        builder.add_joint_revolute(root, child, axis=newton.Axis.Y, parent_xform=hinge),
    ]
    builder.add_articulation(joints)
    box = builder.add_body(xform=wp.transform(wp.vec3(0.156, 0.0, 0.5), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
    return builder.finalize(device="cuda:0"), root, box


def _sparse_case(beta, cached, *, kinematic_body=None, **options):
    """The branched scene of feather_pgs_sparse_scene on its native sparse route."""
    model = sparse_scene.branched_model()
    if kinematic_body is not None:
        _set_flag(model, kinematic_body, KINEMATIC)
    solver = sparse_scene.make_solver(model, cached != "off", friction_anchor_beta=beta, **options)
    state, out, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control.joint_f.assign(np.linspace(-0.08, 0.11, model.joint_dof_count, dtype=np.float32))
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
    return model, solver, state, out, control, pipeline, pipeline.contacts()


def _snapshot(state):
    return {name: getattr(state, name).numpy().copy() for name in STATE_FIELDS}


def _solver_snapshot(solver):
    """Everything a body-flag notification may change in the solver and its owner."""
    arrays = {
        "kinematic_dof_mask_host": solver.factor.kinematic_dof_mask_host,
        "kinematic_dof_mask": solver.factor.kinematic_dof_mask,
        "kinematic_joint_mask": solver.factor.kinematic_joint_mask,
        "joint_armature_effective": solver.factor.armature,
        "fk_id_cache_valid": solver.factor.fk_id_cache_valid,
        "mass_update_requested": solver.factor.mass_update_requested,
        "dof_armature": solver.factor.dof_armature,
    }
    owner = solver.factor.held
    for index, group in enumerate(getattr(owner, "groups", ())):
        for name in ("geometry_valid", "generation", "geometry_generation", "status"):
            arrays[f"group {index} {name}"] = getattr(group.data, name)
    return {key: (value.numpy() if isinstance(value, wp.array) else value).copy() for key, value in arrays.items()}


class _Rollout:
    """Step one solver under a constant wrench on one body, eagerly or by replaying one captured step."""

    def __init__(self, model, solver, body, wrench, *, graph=False):
        self.model, self.solver = model, solver
        self.pipeline = newton.CollisionPipeline(model)
        self.contacts = self.pipeline.contacts()
        self.control = model.control()
        self.state, self.next_state = model.state(), model.state()
        newton.eval_fk(model, self.state.joint_q, self.state.joint_qd, self.state)
        # Both banks carry the wrench, so the copy that closes each step keeps it.
        wrenches = np.zeros((model.body_count, 6), np.float32)
        wrenches[body] = wrench
        self.state.body_f.assign(wrenches)
        self.next_state.body_f.assign(wrenches)
        self.graph, self._captured = graph, None

    def step(self):
        self.pipeline.collide(self.state, self.contacts)
        self.solver.step(self.state, self.next_state, self.control, self.contacts, DT)
        self.state.assign(self.next_state)

    def run(self, steps):
        if self.graph and self._captured is None:
            # One eager step, then capture the next; replays take every later step.
            self.step()
            steps -= 1
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.step()
            self._captured = capture.graph
        for _ in range(steps):
            if self.graph:
                wp.capture_launch(self._captured)
            else:
                self.step()
        return _snapshot(self.state)


@unittest.skipUnless(wp.is_cuda_available(), "Cached tree owners are a CUDA path")
class TestFeatherPGSTreeBodyFlags(unittest.TestCase):
    def _assert_states_close(self, actual, expected, atol=1.0e-5, what=""):
        for name in STATE_FIELDS:
            np.testing.assert_allclose(actual[name], expected[name], atol=atol, rtol=0.0, err_msg=f"{what} {name}")

    def _assert_phases_close(self, actual, expected, atol=1.0e-5):
        self.assertEqual(len(actual), len(expected))
        for phase, (a, e) in enumerate(zip(actual, expected, strict=True)):
            self._assert_states_close(a, e, atol=atol, what=f"phase {phase}")

    def _notify(self, model, solver, body, flag):
        """Set one body's flag and notify; any owner and its route record must survive."""
        owner = solver.factor.held
        route = solver.get_path_selection()["routes"]["held_tree"]
        _set_flag(model, body, flag)
        solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
        self.assertIs(solver.factor.held, owner)
        self.assertEqual(solver.get_path_selection()["routes"]["held_tree"], route)

    def _phases(self, model, solver, body, flags, *, load, wrench, steps, graph=False):
        """State after each phase of ``steps``; between phases ``body`` takes the next of ``flags``."""
        rollout = _Rollout(model, solver, load, wrench, graph=graph)
        phases = [rollout.run(steps)]
        for flag in flags:
            self._notify(model, solver, body, flag)
            phases.append(rollout.run(steps))
        return phases

    def _drive(self, model, solver, body, axis, *, steps):
        """State after each step while ``body``'s free-joint coordinate ``axis`` advances at 1 m/s.

        A kinematic joint keeps its coordinates, so the pose and the velocity
        are set before every step, as a user drives a kinematic body.
        """
        rollout = _Rollout(model, solver, body, np.zeros(6, np.float32))
        state = rollout.state
        joint = int(np.flatnonzero(model.joint_child.numpy() == body)[0])
        q_index = int(model.joint_q_start.numpy()[joint]) + axis
        qd_index = int(model.joint_qd_start.numpy()[joint]) + axis
        start = float(state.joint_q.numpy()[q_index])
        snapshots = []
        for step in range(1, steps + 1):
            q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
            q[q_index], qd[qd_index] = start + step * DT, 1.0
            state.joint_q.assign(q)
            state.joint_qd.assign(qd)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            snapshots.append(rollout.run(1))
        return snapshots

    def test_wrench_on_a_kinematic_link_stays_off_its_ancestors(self):
        """An external wrench on a kinematic link moves nothing under the owner, as under ordinary dynamics.

        Model body flags are live data: the leaf of a moving chain is made
        kinematic after finalize, before the solver is built (the builder
        only marks roots kinematic). The held tree's predictor reduced the
        leaf's wrench into its ancestors' joints; ordinary dynamics drops it.
        """
        wrench = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0], np.float32)
        for floating in (False, True):
            for law, beta in FRICTION_LAWS.items():
                with self.subTest(base="floating" if floating else "fixed", law=law):
                    runs = {}
                    for cached, load in (("off", wrench), ("auto", wrench), ("auto", 0.0 * wrench)):
                        model, links = _chain(floating=floating)
                        _set_flag(model, links[2], KINEMATIC)
                        solver = _solver(model, beta, cached)
                        if cached == "auto":
                            self.assertIsInstance(solver.factor.held, HeldTree)
                        runs[cached, bool(load.any())] = _Rollout(model, solver, links[2], load).run(40)
                    moving = runs["off", True]
                    self.assertGreater(float(np.abs(moving["joint_qd"]).max()), 0.1, "the chain keeps moving")
                    self._assert_states_close(runs["auto", True], moving, what="owner against ordinary")
                    # The kinematic leaf's wrench changes nothing at all.
                    self._assert_states_close(runs["auto", True], runs["auto", False], atol=1.0e-6, what="wrench")

    def test_wrench_on_a_kinematic_link_stays_off_its_ancestors_on_sparse_factors(self):
        """The sparse tree owner (chain-scan predictor) drops a kinematic leaf's wrench as ordinary dynamics does."""
        for law, beta in FRICTION_LAWS.items():
            with self.subTest(law=law):
                runs = {}
                for cached in ("off", "auto"):
                    # Body 1 is the contacting leaf of the first world's tree; every body carries a wrench.
                    case = _sparse_case(beta, cached, kinematic_body=1)
                    solver = case[1]
                    # Ordinary dynamics take the dense factors: native sparse factors are the held tree's.
                    self.assertEqual(solver.placement.sparse_mass_matrix_size is not None, cached == "auto")
                    if cached == "auto":
                        self.assertEqual(solver.get_path_selection()["routes"]["held_tree"]["owner"], "tree")
                    for _ in range(40):
                        sparse_scene.step(case)
                    runs[cached] = _snapshot(case[2])
                self.assertTrue(np.isfinite(runs["auto"]["joint_qd"]).all())
                self._assert_states_close(runs["auto"], runs["off"], what="owner against ordinary")

    def test_a_kinematic_coordinate_past_its_limit_takes_no_row(self):
        """A prescribed coordinate is outside the solver's unknowns: its joint limit takes no row on any route.

        The scene starts the kinematic leaf's joint at 0.23 against its 0.2 limit. A limit row on that coordinate alone
        has a response of the 1e10 armature's size against its cfm and no fixed point: its impulse grew with every
        sweep and leaked into the ancestors, so the owner and the ordinary dense rows parted as the budget grew. Without
        it, every row converges and the two meet at every budget.
        """
        for iterations in (8, 64, 512):
            with self.subTest(iterations=iterations):
                runs = {}
                for cached in ("off", "auto"):
                    case = _sparse_case(0.0, cached, kinematic_body=1, pgs_iterations=iterations)
                    for _ in range(10):
                        sparse_scene.step(case)
                    runs[cached] = _snapshot(case[2])
                    if cached == "off":
                        solver = case[1]
                        rows = int(solver.placement.constraint_count.numpy()[0])
                        impulses = np.abs(solver.placement.impulses.numpy()[0, :rows])
                        self.assertLess(float(impulses.max(initial=0.0)), 1.0e3, "a row grew without a fixed point")
                self._assert_states_close(runs["auto"], runs["off"], what="owner against ordinary")

    def test_toggled_free_body_keeps_the_owner_and_ordinary_dynamics(self):
        """A free body made dynamic and kinematic again keeps the owner and tracks ordinary dynamics.

        This is test_kinematic_links' runtime toggle on the default solver,
        whose held-tree owner froze body flags in its static proof and
        refused the first notification.
        """
        wrench = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)
        for law, beta in FRICTION_LAWS.items():
            with self.subTest(law=law):
                runs = {}
                for cached in ("off", "auto"):
                    model, body = _toggle_scene()
                    solver = _solver(model, beta, cached, update_mass_matrix_interval=100)
                    if cached == "auto":
                        self.assertIsInstance(solver.factor.held, HeldTree)
                    runs[cached] = self._phases(
                        model, solver, body, (DYNAMIC, KINEMATIC), load=body, wrench=wrench, steps=60
                    )
                self._assert_phases_close(runs["auto"], runs["off"])
                held, moved, held_again = (phase["body_q"][body, :3] for phase in runs["auto"])
                self.assertLess(float(np.linalg.norm(held)), 1.0e-3)
                self.assertGreater(float(moved[0]), 0.05)
                self.assertLess(float(np.linalg.norm(held_again - moved)), 1.0e-3)

    def test_graph_replayed_across_toggles_matches_eager_stepping(self):
        """A step captured before the toggles replays with the inputs each notification re-derives.

        The notifications run between replays, outside capture. They rewrite
        the kinematic masks, the held factors' armature and the mass-update
        request in place, so the graph needs no recapture and the replayed
        owner matches eager stepping on both routes.
        """
        wrench = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)
        for law, beta in FRICTION_LAWS.items():
            with self.subTest(law=law):
                runs = {}
                for name, cached, graph in (
                    ("eager ordinary", "off", False),
                    ("eager owner", "auto", False),
                    ("replayed owner", "auto", True),
                ):
                    model, body = _toggle_scene()
                    solver = _solver(model, beta, cached, update_mass_matrix_interval=100)
                    runs[name] = self._phases(
                        model, solver, body, (DYNAMIC, KINEMATIC), load=body, wrench=wrench, steps=60, graph=graph
                    )
                replayed = runs["replayed owner"]
                self.assertIsInstance(solver.factor.held, HeldTree)
                self._assert_phases_close(replayed, runs["eager owner"])
                self._assert_phases_close(replayed, runs["eager ordinary"])
                self.assertGreater(float(replayed[1]["body_q"][body, 0]), 0.05)

    def test_forest_and_per_size_groups_absorb_a_probe_toggle(self):
        """Both group forms of a world with a prescribed root keep the owner while its probe toggles."""
        wrench = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.5], np.float32)
        for law, beta in FRICTION_LAWS.items():
            for forest in ("1", "0"):
                with (
                    self.subTest(law=law, forest=forest),
                    mock.patch.object(
                        SolverFeatherPGS, "_route_pins", {"grouping": "forest" if forest == "1" else "per_size"}
                    ),
                ):
                    runs = {}
                    for cached in ("off", "auto"):
                        model, box, probe = _pair_scene()
                        solver = _solver(model, beta, cached)
                        self.assertTrue(solver.forest.has_prescribed_response)
                        if cached == "auto":
                            groups = solver.factor.held.groups
                            if forest == "1":
                                self.assertEqual([group.layout.root_families for group in groups], [(0, 6)])
                            else:
                                operators = sorted(group.sparse.kind for group in groups)
                                self.assertEqual(operators, ["prescribed", "tree"])
                        runs[cached] = self._phases(
                            model, solver, probe, (KINEMATIC, DYNAMIC), load=probe, wrench=wrench, steps=30
                        )
                    self._assert_phases_close(runs["auto"], runs["off"])
                    # The prescribed box holds its pose; the dynamic probe (5.2 kg with its
                    # sphere) is pushed 3 mm along x before it first becomes kinematic.
                    for phase in runs["auto"]:
                        np.testing.assert_allclose(phase["body_q"][box, :3], (-0.3, 0.0, 0.0), atol=1.0e-6)
                    self.assertGreater(float(runs["auto"][0]["body_q"][probe, 0]), 0.452)

    def test_chain_links_toggled_at_runtime_match_ordinary_dynamics(self):
        """Roots and leaves made kinematic mid-run keep the held-tree and sparse-tree owners on ordinary dynamics."""
        wrench = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0], np.float32)
        cases = (
            # (base, toggled link, flags after the first phase, loaded link)
            ("floating", 0, (KINEMATIC, DYNAMIC), 1),
            ("fixed", 2, (KINEMATIC,), 2),
            ("floating", 2, (KINEMATIC,), 2),
        )
        for law, beta in FRICTION_LAWS.items():
            for base, link, flags, load in cases:
                with self.subTest(law=law, base=base, link=link):
                    runs = {}
                    for cached in ("off", "auto"):
                        model, links = _chain(floating=base == "floating")
                        solver = _solver(model, beta, cached)
                        if cached == "auto":
                            self.assertIsInstance(solver.factor.held, HeldTree)
                        runs[cached] = self._phases(
                            model, solver, links[link], flags, load=links[load], wrench=wrench, steps=20
                        )
                    self._assert_phases_close(runs["auto"], runs["off"])
            # The sparse scene's floating root (body 0) and contacting leaf (body 1) of the first world.
            for body, flags in ((0, (KINEMATIC, DYNAMIC)), (1, (KINEMATIC,))):
                with self.subTest(law=law, owner="sparse tree", body=body):
                    runs = {}
                    for cached in ("off", "auto"):
                        case = _sparse_case(beta, cached)
                        model, solver = case[0], case[1]
                        if cached == "auto":
                            self.assertEqual(solver.get_path_selection()["routes"]["held_tree"]["owner"], "tree")
                        phases = []
                        for flag in (None, *flags):
                            if flag is not None:
                                self._notify(model, solver, body, flag)
                            for _ in range(20):
                                sparse_scene.step(case)
                            phases.append(_snapshot(case[2]))
                        runs[cached] = phases
                    self._assert_phases_close(runs["auto"], runs["off"])

    def test_moving_kinematic_body_keeps_its_contact_rows(self):
        """A kinematic body closing on a dynamic one keeps its world's contact rows under the owner.

        In matrix-free mode the resolved-world classifier reads the twists the
        owner's predictor publishes, under point friction and, with its row
        bound, under persistent patch friction. It took a kinematic body of a
        responding tree as still, where the ordinary route keeps the world's
        rows: a free box made kinematic at runtime, which the owner now
        absorbs, and a kinematic floating root built with a dynamic link. The
        owner skipped the first contact step and over-corrected after it.
        """
        cases = (
            # (case, scene, made kinematic at runtime, driven coordinate of the kinematic free joint)
            ("free box made kinematic", _arm_and_box, True, 1),
            ("kinematic root built", _kinematic_root_and_box, False, 0),
        )
        for case, scene, toggle, axis in cases:
            for law, beta in FRICTION_LAWS.items():
                with self.subTest(case=case, law=law):
                    runs = {}
                    for cached in ("off", "auto"):
                        model, driver, struck = scene()
                        solver = _solver(model, beta, cached)
                        if cached == "auto":
                            self.assertIsInstance(solver.factor.held, HeldTree)
                            classifier = solver.sweep.resolved_worlds_owner
                            self.assertIsNotNone(classifier, solver.sweep.resolved_worlds_reason)
                            self.assertTrue(classifier.published)
                            # Patch friction takes the classifier's capacity-bounded decision.
                            self.assertEqual(classifier.row_bound is not None, law == "patch")
                        if toggle:
                            self._notify(model, solver, driver, KINEMATIC)
                        runs[cached] = self._drive(model, solver, driver, axis, steps=8)
                    # The arm's last link turns away, or the box is pushed ahead of the root.
                    self.assertGreater(float(np.abs(runs["off"][-1]["body_qd"][struck]).max()), 0.5)
                    self._assert_phases_close(runs["auto"], runs["off"], atol=1.0e-4)

    def test_rejected_toggle_raises_the_ordinary_error_before_any_change(self):
        """A prescribed root made dynamic raises the ordinary error on both routes and changes no solver state."""
        wrench = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.5], np.float32)
        for law, beta in FRICTION_LAWS.items():
            for cached in ("off", "auto"):
                with self.subTest(law=law, cached_dynamics=cached):
                    model, box, probe = _pair_scene()
                    solver = _solver(model, beta, cached)
                    owner = solver.factor.held
                    _Rollout(model, solver, probe, wrench).run(5)
                    before = _solver_snapshot(solver)
                    if cached == "auto":
                        # The held tree's caches are valid, so an invalidation would show.
                        self.assertTrue(before["fk_id_cache_valid"].all())
                    _set_flag(model, box, DYNAMIC)
                    with self.assertRaises(RuntimeError) as raised:
                        solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
                    self.assertEqual(str(raised.exception), PRESCRIBED_ERROR)
                    self.assertIs(solver.factor.held, owner)
                    after = _solver_snapshot(solver)
                    self.assertEqual(after.keys(), before.keys())
                    for key, value in before.items():
                        np.testing.assert_array_equal(after[key], value, err_msg=key)


INERTIA = wp.mat33(np.eye(3) * 0.01)


def _replicated(scene, worlds=2):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(scene, worlds)
    return builder.finalize(device="cpu")


def _held_operator(model, arts, parents, counts, **attributes):
    """A cached-tree operator over articulations ``arts`` that share one topology, for make_plan alone."""
    mass = TreePlan.build(np.asarray(parents, np.int32), np.asarray(counts, np.int32)).factor
    plan = SimpleNamespace(
        group_to_art=wp.array(np.asarray(arts, np.int32), dtype=int, device=model.device),
        art_to_world=wp.array(model.articulation_world.numpy(), dtype=int, device=model.device),
    )
    return SimpleNamespace(plan=plan, host=operator_metadata(mass), **attributes)


def _rows_of_kind(solver, model, states, kind):
    """Step ``solver`` once from ``states[0]`` (then swap the states) and count its dense rows of ``kind``."""
    solver.step(states[0], states[1], model.control(), None, 1.0e-3)
    states.reverse()
    count = int(solver.placement.constraint_count.numpy()[0])
    return int(np.count_nonzero(solver.placement.row_type.numpy()[0, :count] == kind))


class TestPrescribedCoordinateRows(unittest.TestCase):
    """The rows read the model's live limits and skip a prescribed coordinate where they read them
    (:data:`.joint_table.JOINTS`)."""

    def test_model_writes_reach_the_rows_and_a_prescribed_coordinate_takes_none(self):
        """Limits written to the model mid-run reach the rows at the next step, with no notification, and a prescribed
        coordinate takes no limit row whatever the model holds.

        A chain of three hinges whose root link is kinematic starts every joint past its upper limit: the default
        activation gap makes both sides of each finite limit a row, so the two dynamic coordinates take four
        position-limit rows (and, with velocity limits, four velocity-limit rows) and the prescribed one none. Writing
        an unlimited position and velocity limit to the last coordinate leaves two of each.
        """
        for device in ("cpu", *(("cuda:0",) if wp.is_cuda_available() else ())):
            with self.subTest(device=device):
                scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                links = [scene.add_link(mass=1.0, inertia=INERTIA, is_kinematic=index == 0) for index in range(3)]
                joints = [
                    scene.add_joint_revolute(
                        links[index - 1] if index else -1, link, axis=newton.Axis.Y, limit_lower=-0.5, limit_upper=0.5
                    )
                    for index, link in enumerate(links)
                ]
                scene.add_articulation(joints)
                model = scene.finalize(device=device)
                velocity_limits = model.device.is_cuda
                solver = SolverFeatherPGS(
                    model, angular_damping=0.0, enable_joint_limits=True, enable_joint_velocity_limits=velocity_limits
                )
                for name in ("joint_limit_lower", "joint_limit_upper", "joint_velocity_limit"):
                    # The joint table holds the model's live arrays.
                    self.assertEqual(getattr(joint_table.bind(solver), name).ptr, getattr(model, name).ptr, name)
                states = [model.state(), model.state()]
                states[0].joint_q.assign(np.full(3, 0.8, np.float32))
                model.joint_velocity_limit.assign(np.full(3, 2.0, np.float32))
                self.assertEqual(_rows_of_kind(solver, model, states, PGS_CONSTRAINT_TYPE_JOINT_LIMIT), 4)
                if velocity_limits:
                    self.assertEqual(_rows_of_kind(solver, model, states, PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT), 4)
                model.joint_limit_lower.assign(np.asarray([-0.5, -0.5, -np.inf], np.float32))
                model.joint_limit_upper.assign(np.asarray([0.5, 0.5, np.inf], np.float32))
                model.joint_velocity_limit.assign(np.asarray([1.0, 1.0, 0.0], np.float32))
                self.assertEqual(_rows_of_kind(solver, model, states, PGS_CONSTRAINT_TYPE_JOINT_LIMIT), 2)
                if velocity_limits:
                    self.assertEqual(_rows_of_kind(solver, model, states, PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT), 2)


class TestTreePlanKinematicBodies(unittest.TestCase):
    """make_plan's admission of kinematic bodies, on the host (no kernels)."""

    def test_held_groups_admit_kinematic_coordinate_bodies(self):
        """A kinematic free body and a chain with a kinematic root or link are admitted into held groups."""
        scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        scene.add_body(mass=1.0, inertia=INERTIA, is_kinematic=True)
        model = _replicated(scene)
        layout, _ = make_plan(model, _held_operator(model, [0, 1], [-1], [6]))
        self.assertEqual(layout.root_coordinates, (6,))
        for link in (0, 2):
            with self.subTest(link=link):
                scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                links = [scene.add_link(mass=1.0, inertia=INERTIA) for _ in range(3)]
                joints = [scene.add_joint_free(links[0])]
                joints += [scene.add_joint_revolute(links[i - 1], links[i], axis=newton.Axis.Y) for i in (1, 2)]
                scene.add_articulation(joints)
                model = _replicated(scene)
                # Body flags are live data; a link may be made kinematic after finalize.
                _set_flag(model, link, KINEMATIC)
                layout, _ = make_plan(model, _held_operator(model, [0, 1], [-1, 0, 1], [6, 1, 1]))
                self.assertEqual((layout.bodies, layout.dofs), (3, 8))

    def test_factorless_roots_must_be_kinematic_free_bodies(self):
        """A prescribed group and a forest's zero family still refuse a dynamic body."""
        scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        scene.add_body(mass=1.0, inertia=INERTIA)
        model = _replicated(scene)
        with self.assertRaisesRegex(ValueError, "Prescribed cached roots must be single kinematic free bodies"):
            make_plan(model, _held_operator(model, [0, 1], [-1], [6], prescribed=True))
        scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        box = scene.add_body(mass=1.0, inertia=INERTIA, is_kinematic=True)
        scene.add_body(mass=1.0, inertia=INERTIA)
        model = _replicated(scene)
        forest = {"forest_arts": np.array([[0, 1], [2, 3]]), "root_families": (0, 6)}
        layout, _ = make_plan(model, _held_operator(model, [0, 2], [-1, -1], [6, 6], **forest), complete=False)
        self.assertEqual(layout.root_families, (0, 6))
        for world in range(2):
            _set_flag(model, box + 2 * world, DYNAMIC)
        with self.assertRaisesRegex(ValueError, "A zero-family forest tree must be a kinematic free body"):
            make_plan(model, _held_operator(model, [0, 2], [-1, -1], [6, 6], **forest), complete=False)


if __name__ == "__main__":
    unittest.main()
