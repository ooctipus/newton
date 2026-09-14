# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the default-off ancestor-sparse factor boundary without a GPU."""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import sparse_factor as sf
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS

ASSET = Path(
    "/tmp/https/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd"
)


def fixture(device="cpu", *, capacity=100):
    """Bind the actual complete USD tree and independent COM kinetic operator."""
    b = newton.ModelBuilder()
    b.begin_world()
    b.add_usd(str(ASSET), load_visual_shapes=False)
    b.end_world()
    b.add_ground_plane()
    m = b.finalize(device="cpu")
    q = m.joint_q.numpy().copy()
    q[2] = 0.9
    q[7:] += np.random.default_rng(14).uniform(-0.1, 0.1, 37).astype(np.float32)
    state = m.state()
    twists = []
    for dof in range(43):
        v = np.zeros(43, np.float32)
        v[dof] = 1
        newton.eval_fk(m, wp.array(q, dtype=float, device="cpu"), wp.array(v, dtype=float, device="cpu"), state)
        twists.append(state.body_qd.numpy().copy())
    twists = np.asarray(twists, np.float64).transpose(1, 2, 0)
    poses = state.body_q.numpy().copy()
    host = sf.make_plan(m)
    mass = m.body_mass.numpy().astype(float)
    inertia = m.body_inertia.numpy().astype(float)
    com = m.body_com.numpy().astype(float)
    centers = np.array(
        [np.asarray(wp.transform_point(wp.transform(*pose), wp.vec3(*c))) for pose, c in zip(poses, com, strict=False)]
    )
    origin = centers[0]
    S = np.zeros((43, 6), np.float64)
    Is = np.zeros((44, 6, 6), np.float64)
    H = np.zeros((43, 43), np.float64)
    for body in range(44):
        rotation = np.asarray(wp.quat_to_matrix(wp.quat(*poses[body, 3:])), float).reshape(3, 3)
        ic = rotation @ inertia[body] @ rotation.T
        r = centers[body] - origin
        cross = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        Is[body, 3:, 3:] = ic + mass[body] * ((r @ r) * np.eye(3) - np.outer(r, r))
        Is[body, :3, 3:] = -mass[body] * cross
        Is[body, 3:, :3] = mass[body] * cross
        Is[body, :3, :3] = mass[body] * np.eye(3)
        H += mass[body] * twists[body, :3].T @ twists[body, :3] + twists[body, 3:].T @ ic @ twists[body, 3:]
    for dof, joint in enumerate(host["dof_joint"]):
        angular = twists[joint, 3:, dof]
        S[dof, 3:] = angular
        S[dof, :3] = twists[joint, :3, dof] - np.cross(angular, centers[joint] - origin)
    composite = Is.copy()
    parents = m.joint_parent.numpy()
    for body in range(43, 0, -1):
        composite[parents[body]] += composite[body]
    R = np.r_[np.full(6, 1e-5), np.linspace(0.001, 0.02, 37)].astype(np.float32)
    K = np.linspace(0.001, 0.03, 43).astype(np.float32)
    H += np.diag(R + K)
    if device != "cpu":
        m = b.finalize(device=device)
    m.rigid_contact_max = 8
    m.joint_q.assign(q)
    state = m.state()
    newton.eval_fk(m, m.joint_q, m.joint_qd, state)
    with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_FACTOR": "0", "FEATHER_PGS_SINGLE_FACTOR": "0"}):
        solver = SolverFeatherPGS(
            m,
            pgs_mode="split" if device == "cpu" else "matrix_free",
            pgs_iterations=8,
            dense_max_constraints=capacity,
            enable_joint_limits=True,
            joint_limit_activation_gap=0.01,
            update_mass_matrix_interval=2,
            mf_gs_incremental_rows=0,
            fuse_joint_velocity_limits=False,
            use_parallel_streams=False,
            double_buffer=False,
        )
    plan, host = sf.make_plan(m, solver)
    owner = sf.SparseFactor(solver, plan, host)
    solver.joint_S_s.assign(S.astype(np.float32))
    solver.body_I_c.assign(composite.astype(np.float32))
    solver.articulation_origin.assign(np.asarray([origin], np.float32))
    solver.R_by_size[43].assign(R[None])
    solver._augmented_drive_row_by_dof = wp.array(np.arange(43, dtype=np.int32), dtype=int, device=device)
    solver.aug_row_K = wp.array(K, dtype=float, device=device)
    solver.mass_update_mask.fill_(1)
    solver.body_v_s.zero_()
    contacts = newton.Contacts(rigid_contact_max=8, soft_contact_max=0, device=device)
    shape_bodies = m.shape_body.numpy()
    ids = [int(np.flatnonzero(shape_bodies == i)[0]) for i in (6, 13, 14)]
    floor = int(np.flatnonzero(shape_bodies < 0)[0])
    contacts.rigid_contact_count.assign(np.array([3], np.int32))
    sh0 = np.full(8, -1, np.int32)
    sh1 = sh0.copy()
    sh0[:3] = ids
    sh1[:3] = floor
    p0 = np.zeros((8, 3), np.float32)
    p1 = p0.copy()
    n = p0.copy()
    for i, body in enumerate((6, 13, 14)):
        p0[i] = [0.02, -0.01, -0.05]
        p1[i] = np.asarray(wp.transform_point(wp.transform(*poses[body]), wp.vec3(*p0[i])))
        p1[i, 2] += 0.003
        n[i] = [0, 0, -1]
    for name, data in [("shape0", sh0), ("shape1", sh1), ("point0", p0), ("point1", p1), ("normal", n)]:
        getattr(contacts, "rigid_contact_" + name).assign(data)
    contacts.rigid_contact_margin0.zero_()
    contacts.rigid_contact_margin1.zero_()
    return {
        "model": m,
        "state": state,
        "solver": solver,
        "owner": owner,
        "H": H,
        "S": S,
        "composite": composite,
        "host": host,
        "contacts": contacts,
    }


def unpack(owner):
    """Decode only for the independent test, never in the runtime path."""
    W = np.zeros((43, 43), float)
    W[owner.host["row"], owner.host["col"]] = owner.data.W.numpy()[0]
    return W


def physical_rows(f):
    """Form current physical J independently of the native sparse packet."""
    s, o, c = f["solver"], f["owner"], f["contacts"]
    count = int(s.constraint_count.numpy()[0])
    J = np.zeros((count, 43))
    row = 0
    q = f["state"].joint_q.numpy()
    lower = f["model"].joint_limit_lower.numpy()
    upper = f["model"].joint_limit_upper.numpy()
    for dof, qi in enumerate(s._joint_limit_q_index.numpy()):
        if qi < 0 or not s.enable_joint_limits:
            continue
        for sign, bound in ((1.0, lower[dof]), (-1.0, upper[dof])):
            if np.isfinite(bound) and sign * (q[qi] - bound) <= s.joint_limit_activation_gap:
                J[row, dof] = sign
                row += 1
    poses = f["state"].body_q.numpy()
    orig = s.articulation_origin.numpy()[0].astype(float)
    shape = f["model"].shape_body.numpy()
    S = s.joint_S_s.numpy().astype(float)
    shape0 = c.rigid_contact_shape0.numpy()
    shape1 = c.rigid_contact_shape1.numpy()
    p0 = c.rigid_contact_point0.numpy()
    p1 = c.rigid_contact_point1.numpy()
    normal = c.rigid_contact_normal.numpy()
    m0 = c.rigid_contact_margin0.numpy()
    m1 = c.rigid_contact_margin1.numpy()
    slots = s.contact_slot.numpy()
    paths = s.contact_path.numpy()
    needed = s.contact_slots_needed.numpy()
    for raw in range(int(c.rigid_contact_count.numpy()[0])):
        if paths[raw] != 0:
            continue
        ba, bb = int(shape[shape0[raw]]), int(shape[shape1[raw]])
        n = -normal[raw].astype(float)
        pa = (
            np.asarray(wp.transform_point(wp.transform(*poses[ba]), wp.vec3(*p0[raw])), float)
            if ba >= 0
            else p0[raw].astype(float)
        ) - m0[raw] * n
        pb = (
            np.asarray(wp.transform_point(wp.transform(*poses[bb]), wp.vec3(*p1[raw])), float)
            if bb >= 0
            else p1[raw].astype(float)
        ) + m1[raw] * n
        t0 = np.cross(n, [1.0, 0.0, 0.0])
        if t0 @ t0 < 1e-12:
            t0 = np.cross(n, [0.0, 1.0, 0.0])
        t0 /= np.linalg.norm(t0)
        t1 = np.cross(n, t0)
        t1 /= np.linalg.norm(t1)
        for r, direction in enumerate((n, t0, t1)[: needed[raw]]):
            a, b = pa, pb
            if s.contact_shared_anchor or (r and s.contact_friction_shared_anchor):
                a = b = 0.5 * (pa + pb)
            for body, point, sign in ((ba, a, 1), (bb, b, -1)):
                if body < 0:
                    continue
                mask = int(o.host["body_mask"][body])
                for d in range(43):
                    if mask & (1 << d):
                        J[slots[raw] + r, d] += sign * (direction @ (S[d, :3] + np.cross(S[d, 3:], point - orig)))
    return J


def original_eight(J, Y, diag, rhs, types, parent, mu, vhat):
    """Independent physical-velocity recurrence, original current friction law."""
    v = np.array(vhat, float)
    lam = np.zeros(len(J), float)
    for _ in range(8):
        for row in range(len(J)):
            old = lam[row]
            new = old - (J[row] @ v + rhs[row]) / diag[row]
            if types[row] in (0, 3):
                new = max(new, 0.0)
            elif types[row] == 2:
                p = parent[row]
                radius = max(mu[row] * lam[p], 0.0)
                if radius <= 0:
                    new = 0.0
                else:
                    sib = p + 2 if row == p + 1 else p + 1
                    mag = np.hypot(new, lam[sib])
                    if mag > radius:
                        scale = radius / mag
                        new *= scale
                        other = lam[sib] * scale
                        v += Y[sib] * (other - lam[sib])
                        lam[sib] = other
            lam[row] = new
            v += Y[row] * (new - old)
    return v, lam


class TestSparseFactor(unittest.TestCase):
    def test_reject_other_canonical_producers(self):
        """Reject optional producers before they can touch retired buffers."""
        solver = SimpleNamespace(
            _execution_plan=SimpleNamespace(use_tiled_cholesky=lambda size: True),
            trisolve_kernel="tiled",
            small_dof_threshold=32,
            _mimic_count=0,
            _connect_count=0,
            _joint_world=None,
            _row_packets=None,
            _debug_buffers_enabled=False,
            _grouped_tau_mass=False,
            _grouped_mass=False,
            _wr_world_contacts=None,
            _ink_sizes=None,
        )
        with patch("newton._src.solvers.feather_pgs.single_factor.supported", return_value=True):
            self.assertTrue(sf.supported(solver))
            for name, value in (
                ("_grouped_tau_mass", True),
                ("_grouped_mass", True),
                ("_wr_world_contacts", object()),
                ("_ink_sizes", (43,)),
            ):
                old = getattr(solver, name)
                setattr(solver, name, value)
                with self.subTest(name=name):
                    self.assertFalse(sf.supported(solver))
                setattr(solver, name, old)

    def test_private_api(self):
        """Require the complete sparse producer and consumer API."""
        self.assertTrue(callable(sf.create_owner))
        self.assertTrue(callable(sf.make_plan))

    def test_actual_plan_and_original_metadata_binding(self):
        """Keep actual topology and original allocation/metadata argument coverage."""
        f = fixture()
        o = f["owner"]
        s = f["solver"]
        self.assertEqual(len(o.host["row"]), 434)
        self.assertEqual(int(o.host["support_count"].max()), 18)
        # Reconstruct every original ancestor H coefficient from current COM
        # inertia and screw inputs independently of native factorization.
        rebuilt = np.zeros((43, 43))
        for a, b, src in zip(o.host["row"], o.host["col"], o.host["source"], strict=False):
            i, j = 42 - a, 42 - b
            other = j if src == i else i
            val = f["S"][other] @ f["composite"][o.host["dof_joint"][src]] @ f["S"][src]
            rebuilt[i, j] = rebuilt[j, i] = val
        rebuilt += np.diag(s.R_by_size[43].numpy()[0] + s.aug_row_K.numpy())
        np.testing.assert_allclose(rebuilt, f["H"], rtol=3e-5, atol=3e-6)
        L = np.linalg.cholesky(f["H"][::-1, ::-1])
        W = np.linalg.solve(L, np.eye(43))
        o.data.W.assign(W[o.host["row"], o.host["col"]][None].astype(np.float32))
        o.data.valid.fill_(1)
        s.v_hat.zero_()
        o.build_rows(f["state"], s, f["contacts"], 0.0025)
        s.check_constraint_capacity()
        self.assertTrue(np.all(s.contact_path.numpy()[:3] == 0))
        self.assertGreaterEqual(int(s.constraint_count.numpy()[0]), 9)
        flags = f["model"].body_flags.numpy().copy()
        flags[6] |= int(newton.BodyFlags.KINEMATIC)
        f["model"].body_flags.assign(flags)
        with self.assertRaisesRegex(ValueError, "kinematic"):
            sf.make_plan(f["model"])


class TestSparseLevelUpdate(unittest.TestCase):
    """Check exact level-owned updates without changing the held representation."""

    def test_level_schedule_owns_exact_schur_terms(self):
        """Cover each actual Schur product once with conflict-free destinations."""
        self.assertTrue(callable(getattr(sf, "_level_update_schedule", None)))
        f = fixture()
        index = f["host"]["index"]
        rows, cols = f["host"]["row"], f["host"]["col"]
        schedule = sf._level_update_schedule(index)
        offsets = schedule["level_offsets"]
        pivots, scales = schedule["level_pivots"], schedule["level_scales"]
        entries, starts = schedule["level_entries"], schedule["level_term_offsets"]
        terms = schedule["level_terms"]
        self.assertEqual(offsets.shape, (16, 3))
        self.assertEqual(offsets[-1].tolist(), [43, 391, 1142])
        self.assertEqual(terms.shape, (2242, 2))
        self.assertEqual(sorted(pivots.tolist()), list(range(43)))
        self.assertEqual(sorted(scales.tolist()), np.flatnonzero(rows > cols).tolist())
        levels = np.empty(43, np.int32)
        for level in range(15):
            levels[pivots[offsets[level, 0] : offsets[level + 1, 0]]] = level
        coverage, longest = [], []
        for level in range(15):
            current = pivots[offsets[level, 0] : offsets[level + 1, 0]]
            self.assertEqual(current.tolist(), sorted(current.tolist()))
            for col in current:
                self.assertTrue(np.all(levels[np.flatnonzero(index[col, :col] >= 0)] < level))
            tasks = range(int(offsets[level, 2]), int(offsets[level + 1, 2]))
            destinations = [int(entries[task]) for task in tasks]
            self.assertEqual(len(destinations), len(set(destinations)))
            longest.append(max((int(starts[t + 1] - starts[t]) for t in tasks), default=0))
            for task in tasks:
                entry = int(entries[task])
                r, c = int(rows[entry]), int(cols[entry])
                self.assertGreater(levels[c], level)
                expected = [
                    (int(index[r, k]), int(index[c, k])) for k in current if index[r, k] >= 0 and index[c, k] >= 0
                ]
                self.assertEqual([tuple(x) for x in terms[starts[task] : starts[task + 1]]], expected)
                for x, y in expected:
                    self.assertEqual(cols[x], cols[y])
                    coverage.append((entry, int(cols[x])))
        expected = [
            (e, k)
            for e, (r, c) in enumerate(zip(rows, cols, strict=True))
            for k in range(c)
            if index[r, k] >= 0 and index[c, k] >= 0
        ]
        self.assertEqual(sorted(coverage), expected)
        self.assertEqual(sum(longest), 42)
        self.assertEqual(max(longest), 8)
        broken = index.copy()
        broken[42, 42] = -1
        with self.assertRaises(ValueError):
            sf._level_update_schedule(broken)

    def test_level_factor_actual_operator_and_owner_admission(self):
        """Reconstruct actual augmented H and retain the unchanged inverse action."""
        with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_LEVEL_UPDATE": "1"}):
            f = fixture()
        o, host = f["owner"], f["host"]
        self.assertTrue(o.level_update)
        self.assertEqual(o.kernels.refresh.key, "sparse_factor_level_update43_434")
        self.assertEqual(o.data.W.shape, (1, 434))
        self.assertEqual(o.data.Z.shape, (1, 100, 18))
        schedule = sf._level_update_schedule(host["index"])
        rows, cols, index = host["row"], host["col"], host["index"]
        original = f["H"][::-1, ::-1]
        a = original[rows, cols].astype(np.float32)
        offsets = schedule["level_offsets"]
        for level in range(15):
            for k in schedule["level_pivots"][offsets[level, 0] : offsets[level + 1, 0]]:
                a[index[k, k]] = np.sqrt(a[index[k, k]])
            for entry in schedule["level_scales"][offsets[level, 1] : offsets[level + 1, 1]]:
                a[entry] /= a[index[cols[entry], cols[entry]]]
            for task in range(int(offsets[level, 2]), int(offsets[level + 1, 2])):
                entry = schedule["level_entries"][task]
                start, end = schedule["level_term_offsets"][task : task + 2]
                for x, y in schedule["level_terms"][start:end]:
                    a[entry] -= a[x] * a[y]
        L = np.zeros((43, 43), np.float64)
        L[rows, cols] = a
        np.testing.assert_allclose(L @ L.T, original, rtol=3e-5, atol=3e-6)
        W = np.linalg.solve(L, np.eye(43))
        tau = np.random.default_rng(38).normal(0, 0.2, 43)
        np.testing.assert_allclose(W.T @ (W @ tau), np.linalg.solve(original, tau), rtol=2e-4, atol=3e-5)
        with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_LEVEL_UPDATE": "0"}):
            original_plan, _ = sf.make_plan(f["model"], f["solver"])
            original_owner = sf.SparseFactor(f["solver"], original_plan, host)
        self.assertFalse(original_owner.level_update)
        for name in schedule:
            self.assertIsNone(getattr(original_owner.plan, name))
        self.assertEqual(original_owner.kernels.refresh.key, "sparse_factor_refresh43_434")
        self.assertIs(original_owner.kernels.predictor, o.kernels.predictor)
        self.assertIs(original_owner.kernels.contacts, o.kernels.contacts)
        self.assertIs(original_owner.kernels.solve, o.kernels.solve)


class TestSparseParallelLimits(unittest.TestCase):
    """Check the isolated global-row prefix on the original actual-tree fixture."""

    device = "cpu"

    def test_parallel_limit_owner_admission(self):
        """Require actual opt-in dispatch, original row storage and c100 admission."""
        flags = {"FEATHER_PGS_SPARSE_PACKETS": "0", "FEATHER_PGS_SPARSE_PARALLEL_LIMITS": "1"}
        with patch.dict(os.environ, flags):
            for capacity in (100, 128):
                o = fixture(self.device, capacity=capacity)["owner"]
                self.assertEqual(o.parallel_limit_prefix, capacity == 100)
                self.assertFalse(o.packet_rows)
                self.assertEqual(o.data.W.shape, (1, 434))
                self.assertEqual(o.data.Z.shape, (1, capacity, 18))
                self.assertEqual(o.data.incident.shape, (1, capacity))
                self.assertEqual(o.data.support.shape, (1, capacity))
                self.assertEqual(o.kernels.contacts.key, "sparse_factor_contact_triplet18")
                self.assertEqual(o.kernels.solve.key, f"sparse_factor_gs43_s18_c{capacity}")
                expected = "sparse_factor_parallel_limit_prefix43" if capacity == 100 else "build_limit_prefix"
                self.assertEqual(o.kernels.prefix.key, expected)
        with patch.dict(os.environ, {**flags, "FEATHER_PGS_SPARSE_PACKETS": "1"}):
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                fixture(self.device)
        with patch.dict(os.environ, {**flags, "FEATHER_PGS_SPARSE_PARALLEL_LIMITS": "0"}):
            self.assertFalse(fixture(self.device)["owner"].parallel_limit_prefix)

    def test_parallel_limit_prefix_order_and_capacity(self):
        """Preserve current and held W columns, stable boundaries and uncapped counts."""
        from newton._src.solvers.feather_pgs.sparse_factor_rows import (  # noqa: PLC0415
            build_limit_prefix,
            get_parallel_limit_kernel,
        )

        flags = {"FEATHER_PGS_SPARSE_PACKETS": "0", "FEATHER_PGS_SPARSE_PARALLEL_LIMITS": "1"}
        for capacity in (100, 7):
            with patch.dict(os.environ, flags):
                f = fixture(self.device, capacity=capacity)
            s, o, m = f["solver"], f["owner"], f["model"]
            W = np.linalg.solve(np.linalg.cholesky(f["H"][::-1, ::-1]), np.eye(43))
            o.data.W.assign(W[o.host["row"], o.host["col"]][None].astype(np.float32))
            held = o.data.W.numpy().copy()
            qi, q = s._joint_limit_q_index.numpy(), f["state"].joint_q.numpy()
            lower = np.full(43, -np.inf, np.float32)
            upper = -lower
            lower[qi >= 0] = q[qi[qi >= 0]] - 0.001
            upper[qi >= 0] = q[qi[qi >= 0]] + 0.001
            m.joint_limit_lower.assign(lower)
            m.joint_limit_upper.assign(upper)
            outputs = [
                o.data.Z,
                o.data.incident,
                o.data.support,
                s.slot_counter,
                s.row_type,
                s.row_parent,
                s.row_mu,
                s.row_beta,
                s.row_cfm,
                s.phi,
                s.target_velocity,
                s.diag,
                s.dense_phase_bounds,
            ]
            for enabled, velocity in ((1, 0.2), (1, -0.3), (0, 0.1)):
                s.v_hat.assign(np.linspace(-velocity, velocity, 43, dtype=np.float32))
                args = [
                    o.plan,
                    o.data,
                    s._joint_limit_q_index,
                    m.joint_limit_lower,
                    m.joint_limit_upper,
                    f["state"].joint_q,
                    s.v_hat,
                    enabled,
                    s.joint_limit_activation_gap,
                    s.pgs_beta,
                    s.pgs_cfm,
                    s.slot_counter,
                    s.row_type,
                    s.row_parent,
                    s.row_mu,
                    s.row_beta,
                    s.row_cfm,
                    s.phi,
                    s.target_velocity,
                    s.diag,
                    s.dense_phase_bounds,
                ]
                for output in outputs:
                    output.fill_(-777)
                wp.launch(build_limit_prefix, dim=1, inputs=args, device=self.device)
                expected = [output.numpy().copy() for output in outputs]
                for output in outputs:
                    output.fill_(-777)
                wp.launch_tiled(get_parallel_limit_kernel(), dim=[1], inputs=args, block_dim=32, device=self.device)
                for output, reference in zip(outputs, expected, strict=True):
                    np.testing.assert_allclose(output.numpy(), reference, rtol=3e-5, atol=3e-6)
                self.assertEqual(int(s.slot_counter.numpy()[0]), 74 if enabled else 0)
                np.testing.assert_array_equal(o.data.W.numpy(), held)
                if enabled:
                    # The first two candidates are lower/upper of the same DOF;
                    # boundaries at candidates32/64 retain the original order.
                    np.testing.assert_allclose(o.data.Z.numpy()[0, 0], -o.data.Z.numpy()[0, 1])


@unittest.skipUnless(wp.is_cuda_available(), "Native parallel-limit controls require CUDA")
class TestSparseParallelLimitsCUDA(TestSparseParallelLimits):
    """Run the same current/held prefix and capacity controls on the leased device."""

    device = "cuda:0"


@unittest.skipUnless(wp.is_cuda_available(), "Native sparse physical controls require CUDA")
class TestSparseFactorCUDA(unittest.TestCase):
    def test_complete_owner_two_steps_and_graph(self):
        """Exercise actual constructor and every retained original service."""
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                self._complete_owner_two_steps_and_graph(parallel)

    def _complete_owner_two_steps_and_graph(self, parallel):
        """Retain the failed serial control and also exercise real async drives."""
        f = fixture("cuda:0")
        m = f["model"]
        state = f["state"]
        contacts = f["contacts"]
        options = {
            "pgs_mode": "matrix_free",
            "pgs_iterations": 8,
            "dense_max_constraints": 100,
            "enable_joint_limits": True,
            "joint_limit_activation_gap": 0.01,
            "update_mass_matrix_interval": 2,
            "mf_gs_incremental_rows": 0,
            "fuse_joint_velocity_limits": False,
            "use_parallel_streams": parallel,
            "double_buffer": False,
        }
        with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_FACTOR": "0", "FEATHER_PGS_SINGLE_FACTOR": "0"}):
            original = SolverFeatherPGS(m, **options)
        with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_FACTOR": "1", "FEATHER_PGS_SINGLE_FACTOR": "0"}):
            candidate = SolverFeatherPGS(m, **options)
        self.assertIsNotNone(candidate._sparse_factor)
        if os.environ.get("FEATHER_PGS_SPARSE_PARALLEL_LIMITS") == "1":
            self.assertTrue(candidate._sparse_factor.parallel_limit_prefix)
            self.assertEqual(candidate._sparse_factor.kernels.prefix.key, "sparse_factor_parallel_limit_prefix43")
        if os.environ.get("FEATHER_PGS_SPARSE_LEVEL_UPDATE") == "1":
            self.assertTrue(candidate._sparse_factor.level_update)
            self.assertEqual(candidate._sparse_factor.kernels.refresh.key, "sparse_factor_level_update43_434")
        original_states = [m.state(), m.state()]
        candidate_states = [m.state(), m.state()]
        original_states[0].assign(state)
        candidate_states[0].assign(state)
        control = m.control()
        held = None
        for epoch in range(2):
            src, dst = epoch % 2, (epoch + 1) % 2
            original.step(original_states[src], original_states[dst], control, contacts, 0.0025)
            candidate.step(candidate_states[src], candidate_states[dst], control, contacts, 0.0025)
            candidate.check_constraint_capacity()
            np.testing.assert_allclose(
                candidate_states[dst].joint_qd.numpy(), original_states[dst].joint_qd.numpy(), rtol=3e-4, atol=3e-5
            )
            np.testing.assert_allclose(
                candidate_states[dst].joint_q.numpy(), original_states[dst].joint_q.numpy(), rtol=3e-5, atol=3e-6
            )
            W = candidate._sparse_factor.data.W.numpy().copy()
            if epoch:
                self.assertTrue(np.array_equal(W, held))
            held = W
        # Both owners continue their own produced states. Each graph's second
        # output is its first input, so every replay advances two more steps.
        with wp.ScopedCapture(device="cuda:0") as original_capture:
            original.step(original_states[0], original_states[1], control, contacts, 0.0025)
            original.step(original_states[1], original_states[0], control, contacts, 0.0025)
        with wp.ScopedCapture(device="cuda:0") as capture:
            candidate.step(candidate_states[0], candidate_states[1], control, contacts, 0.0025)
            candidate.step(candidate_states[1], candidate_states[0], control, contacts, 0.0025)
        for _ in range(3):
            wp.capture_launch(original_capture.graph)
            wp.capture_launch(capture.graph)
            candidate.check_constraint_capacity()
            np.testing.assert_allclose(
                candidate_states[0].joint_qd.numpy(), original_states[0].joint_qd.numpy(), rtol=3e-4, atol=3e-5
            )
            np.testing.assert_allclose(
                candidate_states[0].joint_q.numpy(), original_states[0].joint_q.numpy(), rtol=3e-5, atol=3e-6
            )
        self.assertEqual(candidate.J_world.shape, (1, 1, 1))
        self.assertIsNone(candidate._memset_stream)

    def test_actual_operator_current_rows_and_held_reuse(self):
        """Check actual-tree native refresh/action, all rows and reuse without a GPU-only oracle."""
        f = fixture("cuda:0")
        s = f["solver"]
        o = f["owner"]
        h = f["H"]
        if os.environ.get("FEATHER_PGS_SPARSE_PARALLEL_LIMITS") == "1":
            self.assertTrue(o.parallel_limit_prefix)
            self.assertFalse(o.packet_rows)
            self.assertEqual(o.kernels.prefix.key, "sparse_factor_parallel_limit_prefix43")
        if os.environ.get("FEATHER_PGS_SPARSE_LEVEL_UPDATE") == "1":
            self.assertTrue(o.level_update)
            self.assertEqual(o.kernels.refresh.key, "sparse_factor_level_update43_434")
        o.refresh(s)
        o.check()
        W = unpack(o)
        A = W.T @ W
        defect = np.linalg.norm(h @ A[::-1, ::-1] - np.eye(43), np.inf) / (
            np.linalg.norm(h, np.inf) * np.linalg.norm(A, np.inf) + 1
        )
        self.assertLess(defect, 2e-6)
        tau = np.random.default_rng(38).normal(0, 0.2, 43).astype(np.float32)
        s.joint_tau.assign(tau)
        o.predict(s)
        expected = np.linalg.solve(h, tau)
        np.testing.assert_allclose(s.joint_qdd.numpy(), expected, rtol=2e-4, atol=3e-5)
        saved = o.data.W.numpy().copy()
        s.mass_update_mask.zero_()
        s.body_I_c.fill_(wp.spatial_matrix(np.nan))
        o.refresh(s)
        np.testing.assert_array_equal(o.data.W.numpy(), saved)
        s.v_hat.assign(np.random.default_rng(12).normal(0, 0.02, 43).astype(np.float32))
        o.build_rows(f["state"], s, f["contacts"], 0.0025)
        o.check()
        count = int(s.constraint_count.numpy()[0])
        self.assertGreaterEqual(count, 9)
        if o.packet_rows:
            from tools.fpgs_bench.test_sparse_packet_rows import check_packet_operator  # noqa: PLC0415

            check_packet_operator(f)
            # The native refresh/current-force/reuse checks above remain unchanged;
            # packet rows have no global Z panel for the old coefficient snapshot.
            return
        z = o.data.Z.numpy()[0, :count]
        templates = o.data.support.numpy()[0, :count]
        full = np.zeros((count, 43))
        for row, tpl in enumerate(templates):
            n = o.host["support_count"][tpl]
            full[row, o.host["support_nodes"][tpl, :n]] = z[row, :n]
        J = physical_rows(f)
        np.testing.assert_allclose(full, (W @ J[:, ::-1].T).T, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(o.data.incident.numpy()[0, :count], J @ s.v_hat.numpy(), rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(
            s.diag.numpy()[0, :count], np.sum(full * full, axis=1) + s.row_cfm.numpy()[0, :count], rtol=2e-5, atol=2e-6
        )
        before = s.v_hat.numpy().copy()
        s.v_out.assign(before)
        s.impulses.zero_()
        s._stage4_compute_rhs_world(0.0025)
        o.restitution(0.0025)
        rhs = s.rhs.numpy()[0, :count].astype(float)
        types = s.row_type.numpy()[0, :count]
        parent = s.row_parent.numpy()[0, :count]
        mu = s.row_mu.numpy()[0, :count]
        Y = np.linalg.solve(h, J.T).T
        diagonal = np.sum(J * Y, axis=1) + s.row_cfm.numpy()[0, :count]
        expected_v, expected_lam = original_eight(J, Y, diagonal, rhs, types, parent, mu, before)
        o.solve(s.rhs, 8, 1.0, 0)
        lam = s.impulses.numpy()[0, :count]
        delta = (W.T @ (full.T @ lam))[::-1]
        np.testing.assert_allclose(s.v_out.numpy(), before + delta, rtol=3e-5, atol=3e-6)
        self.assertTrue(np.isfinite(s.v_out.numpy()).all())
        np.testing.assert_allclose(s.v_out.numpy(), expected_v, rtol=3e-4, atol=3e-5)
        np.testing.assert_allclose(lam, expected_lam, rtol=3e-4, atol=3e-5)
        # Finite-capacity prefix, normal-only and both responsive endpoints.
        q = f["state"].joint_q.numpy()
        qi = s._joint_limit_q_index.numpy()
        lower = np.full(43, -np.inf, np.float32)
        upper = -lower
        lower[qi >= 0] = q[qi[qi >= 0]] - 0.001
        upper[qi >= 0] = q[qi[qi >= 0]] + 0.001
        f["model"].joint_limit_lower.assign(lower)
        f["model"].joint_limit_upper.assign(upper)
        s.enable_contact_friction = False
        o.build_rows(f["state"], s, f["contacts"], 0.0025)
        self.assertEqual(int(s.constraint_count.numpy()[0]), 77)
        s.enable_contact_friction = True
        s.contact_shared_anchor = True
        shape = f["model"].shape_body.numpy()
        ids = f["contacts"].rigid_contact_shape1.numpy().copy()
        ids[0] = int(np.flatnonzero(shape == 13)[0])
        f["contacts"].rigid_contact_shape1.assign(ids)
        s.contact_gap_gate = float("inf")
        o.build_rows(f["state"], s, f["contacts"], 0.0025)
        count = int(s.constraint_count.numpy()[0])
        self.assertEqual(count, 83)
        current = physical_rows(f)
        pack = o.data.Z.numpy()[0, :count]
        tpl = o.data.support.numpy()[0, :count]
        actual = np.zeros((count, 43))
        for r, t in enumerate(tpl):
            n = o.host["support_count"][t]
            actual[r, o.host["support_nodes"][t, :n]] = pack[r, :n]
        np.testing.assert_allclose(actual, (W @ current[:, ::-1].T).T, rtol=3e-5, atol=3e-6)
        self.assertIn(18, o.host["support_count"][tpl])
        # A row failure remains visible after a later successful refresh.
        o.data.status.fill_(4)
        s.mass_update_mask.fill_(1)
        s.body_I_c.assign(f["composite"].astype(np.float32))
        o.refresh(s)
        self.assertEqual(int(o.data.status.numpy()[0]), 4)
        with self.assertRaisesRegex(RuntimeError, "guard failed"):
            o.check()


if __name__ == "__main__":
    unittest.main()
