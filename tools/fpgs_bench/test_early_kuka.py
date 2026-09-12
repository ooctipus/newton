# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent controls for disjoint current ZERO-world publication."""

import ast
import hashlib
import importlib
import inspect
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import kernels

SNAPSHOTS = Path("/tmp/fpgs-kuka-live-512-20260911-02/gpu0")
PINS = (
    "5e79694fee997003ef299b33eed223d861244200bd57ac9ee6ef69c8d3b74c34",
    "85084c49880a217e3ab52cbc3340bd5f174349fab8baa77cf0e099e1a9c6d815",
)
CHAIN = (
    kernels.update_qdd_from_velocity,
    kernels.remove_free_root_transport_from_qdd,
    kernels.integrate_generalized_joints,
    kernels.eval_rigid_fk_kinematics,
    kernels.finalize_body_dynamics,
)
OUTPUTS = {
    "v_new": "post_solve_v_out",
    "joint_qdd": "post3_aug_joint_qdd",
    "joint_q_new": "pre_state_joint_q",
    "joint_qd_new": "pre_state_joint_qd",
    "body_q": "pre_state_body_q",
    "body_qd": "pre_state_body_qd",
    "body_q_com": "post3_aug_body_q_com",
    "articulation_origin": "post3_solver_articulation_origin",
    "joint_S_s": "post3_aug_joint_S_s",
    "body_v_s": "post3_aug_body_v_s",
    "body_a_s": "post3_aug_body_a_s",
    "body_I_s": "post3_aug_body_I_s",
    "body_inertia_terms": "post3_solver__body_inertia_terms",
    "body_f_s": "post3_aug_body_f_s",
    "fk_id_cache_valid": "post3_solver__fk_id_cache_valid",
}


def load_snapshot(index):
    """Load exact retained physical Kuka inputs, preserving their original source pin."""
    path = SNAPSHOTS / f"kuka_step{1600 + index}_capture0{index}.npz"
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != PINS[index]:
            raise RuntimeError("Saved Kuka snapshot changed")
    return np.load(path, allow_pickle=False)


def bind(snapshot, index, device):
    """Bind independent original publication outputs and actual global35/response29 maps."""
    values = {
        "inv_dt": 240.0,
        "dt": 1 / 240,
        "angular_damping": 0.0,
        "materialize_all_body_inertia": 0,
        "materialize_body_inertia_terms": index,
    }
    aliases = {
        "joint_q": "pre_state_joint_q",
        "joint_qd": "pre_state_joint_qd",
        "kinematic_dof_mask": "post3_solver__kinematic_dof_mask",
        "kinematic_joint_mask": "post3_solver__kinematic_joint_mask",
        "free_root_joint_indices": "post3_solver__free_root_joint_indices",
    }
    for kernel in CHAIN:
        for arg in kernel.adj.args:
            name = arg.label
            if name in values:
                continue
            key = OUTPUTS.get(name, aliases.get(name))
            if key is None:
                key = next(key for key in ("full_model_" + name, "post3_solver_" + name) if key in snapshot)
            values[name] = wp.array(snapshot[key], dtype=arg.type.dtype, device=device)
    dims = (
        len(snapshot["pre_state_joint_qd"]),
        len(snapshot["post3_solver__free_root_joint_indices"]),
        len(snapshot["full_model_joint_type"]),
        len(snapshot["post3_solver_art_to_world"]),
        len(snapshot["full_model_body_mass"]),
    )
    model = SimpleNamespace(
        device=wp.get_device(device),
        articulation_count=dims[3],
        body_count=dims[4],
        joint_count=dims[2],
        joint_dof_count=dims[0],
        articulation_start=values["articulation_start"],
        joint_child=values["joint_child"],
        joint_parent=values["joint_parent"],
    )
    starts = snapshot["post3_solver_articulation_dof_start"]
    solver = SimpleNamespace(
        model=model,
        world_count=512,
        _model_plan=SimpleNamespace(articulation_dof_count=np.diff(np.r_[starts, dims[0]])),
        _resolved_simple_worlds=wp.zeros(512, dtype=int, device=device),
        art_to_world=wp.array(snapshot["post3_solver_art_to_world"], dtype=int, device=device),
        body_to_articulation=values["body_to_articulation"],
        articulation_joint_end=values["articulation_joint_end"],
        articulation_dof_start=wp.array(starts, dtype=int, device=device),
    )
    return SimpleNamespace(values=values, dims=dims, solver=solver, device=device)


def operands(bundle, index):
    """Read the newly integrated coordinates only in the original FK stage."""
    values = dict(bundle.values)
    if index == 3:
        values["joint_q"], values["joint_qd"] = values["joint_q_new"], values["joint_qd_new"]
    return [values[arg.label] for arg in CHAIN[index].adj.args]


def original(bundle):
    """Run the complete original five-kernel publication in its original order."""
    for index, kernel in enumerate(CHAIN):
        wp.launch(kernel, dim=bundle.dims[index], inputs=operands(bundle, index), device=bundle.device)


def partition(bundle, data, selected, *, finalize=None):
    """Run the actual ownership adapter, optionally separating FK and inertia publication."""
    early = importlib.import_module("newton._src.solvers.feather_pgs.early_kuka")
    early_dims = (
        bundle.dims[3] * data.max_dofs,
        bundle.dims[1],
        bundle.dims[3] * data.max_joints,
        bundle.dims[3],
        bundle.dims[3] * data.max_joints,
    )
    for index, kernel in enumerate(CHAIN):
        if finalize is not None and (index == 4) != finalize:
            continue
        wp.launch(
            early.get_publication_kernel(kernel.key, selected),
            dim=early_dims[index] if selected else bundle.dims[index],
            inputs=[*operands(bundle, index), data],
            device=bundle.device,
        )


def classify(bundle, data, mask):
    """Refresh actual cohort storage, with stale unused tails deliberately poisoned."""
    early = importlib.import_module("newton._src.solvers.feather_pgs.early_kuka")
    data.resolved.assign(np.asarray(mask, dtype=np.int32))
    data.counts.zero_()
    data.early_arts.fill_(-71)
    data.late_arts.fill_(-71)
    wp.launch(early.compact_publication, dim=bundle.dims[3], inputs=[data], device=bundle.device)


def check_outputs(test, candidate, reference):
    """Compare all public/cache/dynamics outputs without a bit-identity gate."""
    for name in OUTPUTS:
        actual, expected = candidate.values[name].numpy(), reference.values[name].numpy()
        test.assertTrue(np.isfinite(actual).all(), name)
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6, err_msg=name)


class TestEarlyKuka(unittest.TestCase):
    def test_default_off_owner_exists(self):
        """Require the distinct ZERO publication owner before testing its equations."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.early_kuka")
        self.assertTrue(callable(module.create_owner))

    def test_original_statement_recovery(self):
        """Recover every original numerical statement after deleting the index-only prelude."""
        early = importlib.import_module("newton._src.solvers.feather_pgs.early_kuka")
        for name in early.PUBLICATION:
            original_ast = ast.parse(textwrap.dedent(inspect.getsource(getattr(kernels, name).func))).body[0]
            for selected in (False, True):
                adapted = ast.parse(early.publication_source(name, selected)).body[0]

                class Restore(ast.NodeTransformer):
                    def visit_Name(self, node):
                        if node.id == "mapped_index":
                            return ast.Call(
                                func=ast.Attribute(value=ast.Name(id="wp", ctx=ast.Load()), attr="tid", ctx=ast.Load()),
                                args=[],
                                keywords=[],
                            )
                        return node

                actual = Restore().visit(ast.Module(body=adapted.body[-len(original_ast.body) :], type_ignores=[]))
                self.assertEqual(ast.dump(actual), ast.dump(ast.Module(body=original_ast.body, type_ignores=[])))

    def test_loaded_refresh_reuse_and_changed_cohort(self):
        """Preserve actual Kuka publication under empty, full, growing and shrinking ownership."""
        early = importlib.import_module("newton._src.solvers.feather_pgs.early_kuka")
        for index in (0, 1):
            with load_snapshot(index) as snapshot:
                reference = bind(snapshot, index, "cpu")
                original(reference)
                candidate = bind(snapshot, index, "cpu")
                data = early.allocate_cohort(candidate.solver)
                self.assertEqual(candidate.dims[0], 512 * 35)
                self.assertEqual(data.max_dofs, 23)
                self.assertEqual(candidate.dims[3], 512 * 3)
                for mask in (np.arange(512) % 2, np.zeros(512), np.ones(512), np.arange(512) % 3 == 0):
                    # Keep immutable current inputs; restore every writable field between calls.
                    fresh = bind(snapshot, index, "cpu")
                    candidate.values = fresh.values
                    classify(candidate, data, mask)
                    counts = data.counts.numpy()
                    self.assertEqual(int(counts.sum()), candidate.dims[3])
                    self.assertEqual(int(counts[0]), int(np.count_nonzero(mask)) * 3)
                    ids = np.r_[data.early_arts.numpy()[: counts[0]], data.late_arts.numpy()[: counts[1]]]
                    np.testing.assert_array_equal(np.sort(ids), np.arange(candidate.dims[3]))
                    # Poisoned tails must never become original kernel indices.
                    partition(candidate, data, True)
                    partition(candidate, data, False)
                    check_outputs(self, candidate, reference)

    def test_selected_outputs_and_mf_reader_order(self):
        """Leave late output owners untouched and retain the old free-body inverse before finalization."""
        early = importlib.import_module("newton._src.solvers.feather_pgs.early_kuka")
        with load_snapshot(0) as snapshot:
            candidate, reference = bind(snapshot, 0, "cpu"), bind(snapshot, 0, "cpu")
            data = early.allocate_cohort(candidate.solver)
            mask = np.arange(512) % 2
            classify(candidate, data, mask)
            saved_inertia = candidate.values["body_I_s"].numpy().copy()
            saved_q = candidate.values["joint_q_new"].numpy().copy()
            partition(candidate, data, True, finalize=False)
            np.testing.assert_array_equal(candidate.values["body_I_s"].numpy(), saved_inertia)
            joint_world = snapshot["post3_solver_art_to_world"][snapshot["full_model_joint_articulation"]]
            coord_world = np.repeat(joint_world, np.diff(snapshot["full_model_joint_q_start"]))
            np.testing.assert_array_equal(
                candidate.values["joint_q_new"].numpy()[mask[coord_world] == 0], saved_q[mask[coord_world] == 0]
            )
            inverse_inputs = [
                wp.array(snapshot["solve_free_rigid_body_indices"], dtype=int, device="cpu"),
                candidate.values["body_I_s"],
                candidate.values["is_free_rigid"],
                candidate.values["body_to_articulation"],
                wp.array(snapshot["full_model_body_flags"], dtype=int, device="cpu"),
            ]
            first = wp.zeros(candidate.dims[4], dtype=wp.spatial_matrix, device="cpu")
            second = wp.zeros_like(first)
            wp.launch(
                kernels.compute_mf_body_Hinv,
                dim=inverse_inputs[0].shape[0],
                inputs=inverse_inputs,
                outputs=[first],
                device="cpu",
            )
            inverse_inputs[1] = reference.values["body_I_s"]
            wp.launch(
                kernels.compute_mf_body_Hinv,
                dim=inverse_inputs[0].shape[0],
                inputs=inverse_inputs,
                outputs=[second],
                device="cpu",
            )
            np.testing.assert_allclose(first.numpy(), second.numpy(), atol=3e-6, rtol=3e-6)
            partition(candidate, data, True, finalize=True)
            partition(candidate, data, False)
            original(reference)
            check_outputs(self, candidate, reference)

    def test_lifecycle_guards_and_final_join(self):
        """Reject aliased states and require recorded reader completion before final cache readiness."""
        early = importlib.import_module("newton._src.solvers.feather_pgs.early_kuka")
        owner = object.__new__(early.EarlyKuka)
        owner.active, owner.finalized = False, False
        array = wp.zeros(4, dtype=float, device="cpu")
        state = SimpleNamespace(**dict.fromkeys(("joint_q", "joint_qd", "body_q", "body_qd"), array))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            owner.begin(state, state)
        owner.active = True
        with self.assertRaises(RuntimeError):
            owner.wait()
        with self.assertRaisesRegex(RuntimeError, "reader event"):
            owner.finish(None, None, None, 1 / 240)
        owner.finalized = True
        owner.solver = SimpleNamespace(model=SimpleNamespace(device="cpu"))
        owner.done, owner.current = object(), object()
        owner.publish = Mock()
        stream = Mock()
        with patch.object(wp, "get_stream", return_value=stream):
            owner.finish(None, None, state, 1 / 240)
        stream.wait_event.assert_called_once_with(owner.done)
        self.assertEqual(owner.publish.call_count, 2)
        self.assertIs(owner.solver._fk_id_cache_source_state, state)
        self.assertFalse(owner.active)
        self.assertIsNone(owner.current)
        owner.wait()  # No stale captured-event wait is carried into eager or next capture.

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA requires the root GPU lease")
    def test_cuda_disjoint_stream_graph_replays(self):
        """Match original loaded publication with early/late streams and changed replay ownership."""
        early = importlib.import_module("newton._src.solvers.feather_pgs.early_kuka")
        device = wp.get_device("cuda:0")
        with load_snapshot(1) as snapshot:
            reference, candidate = bind(snapshot, 1, device), bind(snapshot, 1, device)
            data = early.allocate_cohort(candidate.solver)
            mask = wp.array(np.arange(512) % 2, dtype=int, device=device)
            stream, ready, done = wp.Stream(device), wp.Event(device), wp.Event(device)

            def both():
                data.counts.zero_()
                wp.copy(data.resolved, mask)
                wp.launch(early.compact_publication, dim=candidate.dims[3], inputs=[data], device=device)
                wp.get_stream(device).record_event(ready)
                stream.wait_event(ready)
                with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
                    partition(candidate, data, True)
                    stream.record_event(done)
                partition(candidate, data, False)
                wp.get_stream(device).wait_event(done)

            original(reference)
            both()  # Compile every actual kernel outside capture.
            check_outputs(self, candidate, reference)
            with wp.ScopedCapture(device=device) as capture:
                both()
            for value in (np.zeros(512), np.ones(512), np.arange(512) % 3 == 0):
                fresh = bind(snapshot, 1, device)
                for name in OUTPUTS:
                    wp.copy(candidate.values[name], fresh.values[name])
                mask.assign(np.asarray(value, np.int32))
                wp.capture_launch(capture.graph)
                check_outputs(self, candidate, reference)


if __name__ == "__main__":
    unittest.main()
