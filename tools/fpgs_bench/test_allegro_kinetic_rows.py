# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Matched-input controls for current Allegro rows in held kinetic coordinates."""

import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton import JointType
from newton._src.solvers.feather_pgs import allegro_kinetic_rows as rows
from newton._src.solvers.feather_pgs.kernels import compute_world_contact_bias
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_parallel_kernel

CAPTURE_ROOT = Path("/tmp/fpgs-ten-hour-20260911-gc5FsH")
CAPTURE_HASHES = (
    "ad8902e9ded8b490ee1107e2a415324611ee53904c6afa751571a798c1100b2a",
    "8d6b1a4d95c16f3ea55700c8a453bf823b2ffdec1e783c5540954efda2af36b3",
    "310269af117cd537d7098d6530f2d49396f283632b2c4f6409c50d544af6dbbc",
    "b4929d30df22ef2a19a87dc2c874823d6021d835595f42ca9486c0246a7eef3e",
)


def captures():
    """Reuse the four clean actual current-row captures, not a new simulation."""
    for gpu in (0, 1):
        for index in (0, 1):
            path = (
                CAPTURE_ROOT
                / f"current_whole_owner_allegro512_gpu{gpu}"
                / f"whole_step{1600 + index}_capture0{index}.npz"
            )
            if hashlib.sha256(path.read_bytes()).hexdigest() != CAPTURE_HASHES[gpu * 2 + index]:
                raise ValueError(f"Pinned Allegro fixture changed: {path}")
            yield path


def fixture(snapshot, device):
    """Bind only the saved input arrays required by the existing row owner."""
    vectors = {"articulation_origin", "shape_material_mu_unused"}

    def array(key, dtype=None):
        value = snapshot[key]
        if dtype is None:
            dtype = wp.uint32 if value.dtype == np.uint32 else (int if value.dtype.kind in "iu" else float)
        return wp.array(value, dtype=dtype, device=device)

    model = SimpleNamespace(device=wp.get_device(device), requires_grad=False)
    for key in snapshot.files:
        if key.startswith("model_"):
            setattr(model, key[6:], array(key))
    solver = SimpleNamespace(model=model, **json.loads(str(snapshot["metadata_json"])))
    for key in snapshot.files:
        if key.startswith("solver_"):
            name = key[7:]
            setattr(solver, name, array(key, wp.vec3 if name in vectors else None))
    solver.size_groups = {16: None, 6: None}
    solver.group_to_art = {n: array(f"group_to_art_{n}") for n in (16, 6)}
    solver.n_arts_by_size = {n: len(snapshot[f"group_to_art_{n}"]) for n in (16, 6)}
    solver.L_by_size = {n: array(f"L_{n}") for n in (16, 6)}
    solver.J_by_size = {n: wp.full(snapshot[f"J_{n}"].shape, 91.0, dtype=float, device=device) for n in (16, 6)}
    solver.rhs = wp.full(snapshot["solve_rhs"].shape, 73.0, dtype=float, device=device)
    solver.diag = wp.zeros_like(solver.rhs)
    solver.row_w = wp.zeros_like(solver.rhs)
    solver._contact_w = 1.0
    state = SimpleNamespace(body_q=array("state_body_q", wp.transform), joint_q=array("state_joint_q"))
    aug = SimpleNamespace(
        joint_S_s=array("aug_joint_S_s", wp.spatial_vector), body_v_s=array("aug_body_v_s", wp.spatial_vector)
    )
    contact = SimpleNamespace()
    for key in snapshot.files:
        if key.startswith("contact_"):
            name = key[8:]
            dtype = wp.vec3 if name.endswith(("point0", "point1", "normal")) else None
            setattr(contact, name, array(key, dtype))
    contact.rigid_contact_max = len(snapshot["contact_rigid_contact_point0"])
    owner = rows.AllegroKineticRows(solver)
    return solver, state, aug, contact, owner


def expand(owner):
    """Expand the two kinetic segments solely for the independent test oracle."""
    coefficients = owner.data.coefficients.numpy()
    code = owner.data.encoding.numpy()
    result = np.zeros((*code.shape, 22), dtype=np.float64)
    count = owner.solver.constraint_count.numpy()
    for world, n in enumerate(count):
        for row in range(int(n)):
            value = int(code[world, row])
            o0, n0, o1, n1 = value & 31, (value >> 5) & 7, (value >> 8) & 31, (value >> 13) & 7
            result[world, row, o0 : o0 + n0] = coefficients[world, row, :n0]
            result[world, row, o1 : o1 + n1] = coefficients[world, row, n0 : n0 + n1]
    return result


def run_rows(snapshot, device, *, fallback=False):
    """Run the actual map/prefix/contact/fallback producer sequence."""
    solver, state, aug, contact, owner = fixture(snapshot, device)
    owner.begin_rows(state, aug, solver.dt)
    owner.produce_contacts(state, aug, contact, solver.dt)
    if fallback:
        solver.mf_constraint_count.fill_(1)
    owner.finish_rows()
    return solver, state, aug, contact, owner


def rhs_reference(solver, owner):
    """Apply the original bias kernel to identical current geometry inputs."""

    result = wp.zeros_like(solver.rhs)
    wp.launch(
        compute_world_contact_bias,
        dim=solver.world_count * solver.dense_max_constraints,
        inputs=[
            solver.constraint_count,
            solver.dense_max_constraints,
            solver.phi,
            solver.row_beta,
            solver.row_type,
            solver.target_velocity,
            solver.dt,
            1.0,
            solver.contact_speculative_scale,
            1.0,
            1.0,
            solver.mf_constraint_count,
            0,
        ],
        outputs=[result, wp.zeros_like(solver.row_w)],
        device=solver.model.device,
    )
    expected = result.numpy().astype(np.float64)
    kind, rest, phi = solver.row_type.numpy(), solver.row_restitution.numpy(), solver.phi.numpy()
    target = solver.target_velocity.numpy()
    relative = np.einsum("wrd,wd->wr", expand(owner), owner.maps.kinetic_incident.numpy().astype(np.float64)) - target
    fire = (
        (kind == 0)
        & (rest > 0)
        & (relative < -solver._effective_restitution_velocity_threshold)
        & ((phi <= 1e-6) | (phi + solver.dt * relative <= 1e-6))
    )
    expected[fire] = (-target + rest * relative)[fire]
    return expected


def bind_parallel(snapshot, solver, owner, device, *, kinetic):
    """Bind the unchanged original parallel law and its direct-Z input successor."""

    factory = rows.get_parallel_factory(_get_pgs_solve_parallel_kernel) if kinetic else _get_pgs_solve_parallel_kernel

    def array(value, dtype=float):
        return wp.array(value, dtype=dtype, device=device)

    worlds, capacity = solver.world_count, solver.dense_max_constraints
    meta = np.zeros((worlds, 4), dtype=np.int32)
    meta[:, 0], meta[:, 2], meta[:, 3] = np.arange(worlds), np.arange(worlds), 16
    impulse = array(snapshot["solve_impulses"])
    velocity = array(snapshot["solve_v_out"])
    values = {
        "general_world_grid_stride": worlds,
        "use_general_world_queue": 0,
        "world_constraint_count": solver.constraint_count,
        "dense_phase_bounds": solver.dense_phase_bounds,
        "world_dof_indices": solver.world_dof_indices,
        "rhs_bias": solver.rhs,
        "world_impulses": impulse,
        "world_row_type": solver.row_type,
        "world_row_parent": solver.row_parent,
        "world_row_mu": solver.row_mu,
        "world_row_cfm": solver.row_cfm,
        "iterations": 12,
        "omega": 1.0,
        "regularize": 0,
        "row_phase": 0,
        "friction_start_iteration": 0,
        "iteration_offset": 0,
        "freeze_drive_rows": 0,
        "defer_dense_response": 0,
        "ink_meta": array(meta.reshape(-1), int),
        "ink_L_a": solver.L_by_size[16],
        "ink_L_b": solver.L_by_size[6],
        "ink_J_a": array(snapshot["J_16"]),
        "ink_J_b": array(snapshot["J_6"]),
        "kinetic_rows": owner.data,
        "v_out": velocity,
        "mf_constraint_count": solver.mf_constraint_count,
    }
    calls = []
    for hi, lo in ((32, 0), (64, 32), (96, 64), (128, 96)):
        kernel = factory(
            capacity,
            64,
            22,
            "cuda",
            rows=hi,
            min_rows=lo,
            sweeps=24,
            matrix_free=True,
            inkernel_response=(16, 6, 0, 16),
        )
        args = []
        for argument in kernel.adj.args:
            name = argument.label
            if name not in values:
                annotation = argument.type
                if hasattr(annotation, "dtype") and hasattr(annotation, "ndim"):
                    shape = (1,) * annotation.ndim
                    if name in ("local_solve_owner", "rb_class", "world_deferred_dof_mask"):
                        shape = (worlds,) if annotation.ndim == 1 else (worlds, 22)
                    values[name] = wp.zeros(shape, dtype=annotation.dtype, device=device)
                elif annotation in (int, wp.int32):
                    values[name] = 0
                else:
                    values[name] = 0.0
            args.append(values[name])
        calls.append((kernel, args))
    return calls, velocity, impulse


def launch_parallel(calls, worlds, device):
    """Execute the same four original row tiers with no altered sweep allowance."""
    for kernel, arguments in calls:
        wp.launch_tiled(kernel, dim=[worlds], inputs=arguments, block_dim=kernel._fpgs_block_dim, device=device)


class TestAllegroKineticRowsCPU(unittest.TestCase):
    def test_unsupported_moving_joint_kind(self):
        """Reject a moving type absent from the original velocity-limit allocator."""
        with np.load(next(captures())) as snapshot:
            solver, _state, _aug, _contact, _owner = fixture(snapshot, "cpu")
            types = solver.model.joint_type.numpy()
            moving = np.flatnonzero(np.diff(solver.model.joint_qd_start.numpy()) > 0)
            types[moving[0]] = int(JointType.BALL)
            solver.model.joint_type.assign(types)
            with self.assertRaisesRegex(ValueError, "velocity-limit"):
                rows.topology_plan(solver)

    def test_module_contract(self):
        """Expose the complete typed-row owner and direct kinetic consumer."""
        self.assertTrue(callable(rows.create_owner))
        self.assertTrue(callable(rows.get_ink_stage))

    def test_current_and_held_maps(self):
        """Check current motion maps against independent held-factor actions."""
        for path in captures():
            with self.subTest(path=path.name), np.load(path) as snapshot:
                solver, state, aug, _contact, owner = fixture(snapshot, "cpu")
                owner.begin_rows(state, aug, solver.dt)
                maps = owner.maps.maps.numpy().astype(np.float64)
                motion = aug.joint_S_s.numpy().astype(np.float64)
                starts = owner.maps.starts.numpy()
                for world in (0, 1, 117, 511):
                    for finger in range(4):
                        L = snapshot["L_16"][world, finger * 4 : finger * 4 + 4, finger * 4 : finger * 4 + 4].astype(
                            np.float64
                        )
                        for depth in range(4):
                            S = motion[starts[world, 0] + finger * 4 : starts[world, 0] + finger * 4 + 4].copy()
                            S[depth + 1 :] = 0.0
                            expected = np.linalg.solve(L, S)
                            actual = maps[world, (finger * 4 + depth) * 24 : (finger * 4 + depth + 1) * 24].reshape(
                                4, 6
                            )
                            np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
                    expected = np.linalg.solve(
                        snapshot["L_6"][world].astype(np.float64), motion[starts[world, 1] : starts[world, 1] + 6]
                    )
                    np.testing.assert_allclose(maps[world, 384:].reshape(6, 6), expected, rtol=3e-6, atol=3e-6)
                # Deliberately alter current motion while preserving held L.
                held = {n: solver.L_by_size[n].numpy().copy() for n in (16, 6)}
                changed = aug.joint_S_s.numpy()
                changed[:, :3] += np.float32(0.0125)
                aug.joint_S_s.assign(changed)
                owner.begin_rows(state, aug, solver.dt)
                self.assertGreater(np.max(np.abs(owner.maps.maps.numpy() - maps)), 1e-3)
                for n in (16, 6):
                    np.testing.assert_array_equal(solver.L_by_size[n].numpy(), held[n])

    def test_actual_prefix_and_contact_rows(self):
        """Preserve current row metadata and physical J through direct held Z."""
        for path in captures():
            with self.subTest(path=path.name), np.load(path) as snapshot:
                solver, _state, _aug, _contact, owner = run_rows(snapshot, "cpu")
                np.testing.assert_array_equal(owner.prefix.bounds.numpy(), snapshot["solver_dense_phase_bounds"])
                np.testing.assert_array_equal(
                    solver.velocity_limit_slot.numpy(), snapshot["solver_velocity_limit_slot"]
                )
                counts = snapshot["solver_constraint_count"]
                active = np.arange(solver.dense_max_constraints)[None, :] < counts[:, None]
                for name in (
                    "row_type",
                    "row_parent",
                    "row_mu",
                    "row_beta",
                    "row_cfm",
                    "phi",
                    "target_velocity",
                    "row_restitution",
                ):
                    np.testing.assert_allclose(
                        getattr(solver, name).numpy()[active], snapshot[f"solver_{name}"][active], rtol=2e-5, atol=2e-6
                    )
                Z = expand(owner)
                J = np.concatenate((snapshot["J_16"], snapshot["J_6"]), axis=2).astype(np.float64)
                rebuilt = np.empty_like(J)
                rebuilt[:, :, :16] = Z[:, :, :16] @ np.swapaxes(snapshot["L_16"].astype(np.float64), 1, 2)
                rebuilt[:, :, 16:] = Z[:, :, 16:] @ np.swapaxes(snapshot["L_6"].astype(np.float64), 1, 2)
                defect = np.linalg.norm((rebuilt - J)[active]) / max(np.linalg.norm(J[active]), 1.0)
                self.assertLess(defect, 2e-6)
                print(json.dumps({"fixture": path.name, "current_J_defect": float(defect)}))
                # CPU/GPU point-transform cancellation can differ by <0.5 micrometre,
                # amplified by 1/dt. Test the law on identical current inputs.
                np.testing.assert_allclose(
                    solver.rhs.numpy()[active], rhs_reference(solver, owner)[active], rtol=2e-5, atol=2e-5
                )
                # Owned worlds never wrote canonical J.
                for n in (16, 6):
                    self.assertTrue(np.all(solver.J_by_size[n].numpy() == 91.0))

    def test_complete_fallback_materialization(self):
        """Keep exact physical rows when matrix-free rows force the original fallback."""
        path = next(captures())
        with np.load(path) as snapshot:
            solver, _state, _aug, _contact, _owner = run_rows(snapshot, "cpu", fallback=True)
            active = np.arange(solver.dense_max_constraints)[None, :] < snapshot["solver_constraint_count"][:, None]
            for n in (16, 6):
                np.testing.assert_allclose(
                    solver.J_by_size[n].numpy()[active], snapshot[f"J_{n}"][active], rtol=2e-5, atol=2e-6
                )

    def test_original_parallel_abi_is_unchanged(self):
        """Create an isolated kinetic successor without modifying ordinary signatures."""

        factory = rows.get_parallel_factory(_get_pgs_solve_parallel_kernel)
        kwargs = {"rows": 32, "sweeps": 24, "matrix_free": True, "inkernel_response": (16, 6, 0, 16)}
        ordinary = _get_pgs_solve_parallel_kernel(192, 64, 22, "cuda", **kwargs)
        candidate = factory(192, 64, 22, "cuda", **kwargs)
        self.assertFalse(getattr(ordinary, "_fpgs_kinetic_rows", False))
        self.assertTrue(candidate._fpgs_kinetic_rows)
        self.assertIn("kinetic", candidate.key)

    def test_parallel_cpu_bindings(self):
        """Compile both native ABIs on CPU without claiming CUDA solve execution."""
        with np.load(next(captures())) as snapshot:
            solver, _state, _aug, _contact, owner = run_rows(snapshot, "cpu")
            for kinetic in (False, True):
                calls, _velocity, _impulse = bind_parallel(snapshot, solver, owner, "cpu", kinetic=kinetic)
                launch_parallel(calls, solver.world_count, "cpu")
                self.assertEqual(len(calls), 4)


class TestAllegroKineticRowsCUDA(unittest.TestCase):
    def setUp(self):
        """Require a real visible CUDA device for the explicit native selectors."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA unavailable")
        self.device = wp.get_device("cuda:0")

    def test_native_current_held_rows_and_fallback(self):
        """Check actual current/held rows and complete fallback on both device types."""
        for path in captures():
            with self.subTest(path=str(path)), np.load(path) as snapshot:
                solver, _state, _aug, _contact, owner = run_rows(snapshot, self.device)
                Z = expand(owner)
                active = np.arange(solver.dense_max_constraints)[None, :] < snapshot["solver_constraint_count"][:, None]
                J = np.concatenate((snapshot["J_16"], snapshot["J_6"]), axis=2).astype(np.float64)
                reconstructed = np.concatenate(
                    (
                        Z[:, :, :16] @ np.swapaxes(snapshot["L_16"].astype(np.float64), 1, 2),
                        Z[:, :, 16:] @ np.swapaxes(snapshot["L_6"].astype(np.float64), 1, 2),
                    ),
                    axis=2,
                )
                defect = np.linalg.norm((reconstructed - J)[active]) / max(np.linalg.norm(J[active]), 1.0)
                self.assertLess(defect, 2e-6)
                np.testing.assert_allclose(
                    solver.rhs.numpy()[active], rhs_reference(solver, owner)[active], rtol=2e-5, atol=2e-5
                )
                solver.mf_constraint_count.fill_(1)
                owner.finish_rows()
                for n in (16, 6):
                    np.testing.assert_allclose(
                        solver.J_by_size[n].numpy()[active], snapshot[f"J_{n}"][active], rtol=2e-5, atol=2e-6
                    )

    def test_native_parallel_matched_current_inputs(self):
        """Preserve the original 24-step impulse response with current rows and held L."""
        for path in captures():
            with self.subTest(path=str(path)), np.load(path) as snapshot:
                solver, _state, _aug, _contact, owner = run_rows(snapshot, self.device)
                original, vo, lo = bind_parallel(snapshot, solver, owner, self.device, kinetic=False)
                candidate, vc, lc = bind_parallel(snapshot, solver, owner, self.device, kinetic=True)
                launch_parallel(original, solver.world_count, self.device)
                launch_parallel(candidate, solver.world_count, self.device)
                va, vb = vo.numpy(), vc.numpy()
                la, lb = lo.numpy(), lc.numpy()
                self.assertTrue(np.isfinite(vb).all() and np.isfinite(lb).all())
                np.testing.assert_allclose(vb, va, rtol=3e-5, atol=3e-5)
                np.testing.assert_allclose(lb, la, rtol=3e-5, atol=3e-6)
                # Independent physical impulse action, including nonzero incoming impulses.
                vref = snapshot["solve_v_out"].astype(np.float64).copy()
                delta = lb - snapshot["solve_impulses"]
                for n in (16, 6):
                    L = snapshot[f"L_{n}"].astype(np.float64)
                    J = snapshot[f"J_{n}"].astype(np.float64)
                    force = np.einsum("wrd,wr->wd", J, delta)
                    response = np.linalg.solve(np.swapaxes(L, 1, 2), np.linalg.solve(L, force[..., None]))[..., 0]
                    for group, art in enumerate(snapshot[f"group_to_art_{n}"]):
                        start = snapshot["solver_articulation_dof_start"][art]
                        vref[start : start + n] += response[group]
                defect = np.linalg.norm(vb - vref) / max(np.linalg.norm(vref), 1.0)
                self.assertLess(defect, 2e-6)
                print(
                    json.dumps(
                        {
                            "fixture": str(path),
                            "velocity_action_defect": float(defect),
                            "max_velocity_delta": float(np.max(np.abs(vb - va))),
                        }
                    )
                )

    def test_native_graph_refresh_and_grow_shrink(self):
        """Replay current producers and tier dispatch after count and held-state changes."""
        with np.load(next(captures())) as snapshot:
            solver, state, aug, contact, owner = run_rows(snapshot, self.device)
            calls, velocity, _impulse = bind_parallel(snapshot, solver, owner, self.device, kinetic=True)
            launch_parallel(calls, solver.world_count, self.device)

            def run():
                owner.begin_rows(state, aug, solver.dt)
                owner.produce_contacts(state, aug, contact, solver.dt)
                owner.finish_rows()
                launch_parallel(calls, solver.world_count, self.device)

            with wp.ScopedCapture(device=self.device) as captured:
                run()
            wp.capture_launch(captured.graph)
            self.assertTrue(np.isfinite(velocity.numpy()).all())
            held = {n: solver.L_by_size[n].numpy().copy() for n in (16, 6)}
            changed = aug.joint_S_s.numpy()
            changed[:, :3] += np.float32(0.0075)
            aug.joint_S_s.assign(changed)
            solver.mf_constraint_count.fill_(1)
            wp.capture_launch(captured.graph)
            for n in (16, 6):
                np.testing.assert_array_equal(solver.L_by_size[n].numpy(), held[n])
            contact.rigid_contact_count.zero_()
            solver.constraint_count.zero_()
            solver.mf_constraint_count.zero_()
            wp.capture_launch(captured.graph)
            self.assertTrue(np.isfinite(velocity.numpy()).all())


if __name__ == "__main__":
    unittest.main()
