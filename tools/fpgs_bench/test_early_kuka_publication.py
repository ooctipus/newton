# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent ZERO-classifier, complete host publication and event-order controls.

Reuse pinned historical Kuka512 physical inputs, not a current rollout oracle.
The original and selected publication use the same solved velocity and equations;
these tests do not claim that every changed cohort is physically ZERO-admissible.
"""

import inspect
import json
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import early_kuka as early
from newton._src.solvers.feather_pgs import kernels
from newton._src.solvers.feather_pgs import simple_world as zero
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from tools.fpgs_bench import test_early_kuka as fixture


def classifier(snapshot):
    """Run the actual unchanged ZERO kernels on saved current geometry/predictor."""
    metadata = json.loads(str(snapshot["post3_metadata_json"]))
    data, raw, scratch = zero._SimpleWorldInput(), zero._SimpleRawContacts(), zero._SimpleWorldScratch()
    aliases = {
        "limit_q_index": "post3_solver__joint_limit_q_index",
        "lower": "full_model_joint_limit_lower",
        "upper": "full_model_joint_limit_upper",
        "q": "pre_state_joint_q",
        "v_hat": "pre_rows_v_hat",
        "prescribed_articulation": "post3_solver__prescribed_articulation",
        "body_q": "post3_state_body_q",
        "max_linear_velocity": "post3_solver_rigid_body_max_linear_velocity",
        "max_angular_velocity": "post3_solver_rigid_body_max_angular_velocity",
        "max_depenetration_velocity": "solve_rigid_body_max_depenetration_velocity",
    }
    scalar = {
        "dt": metadata["dt"],
        "beta": metadata["pgs_beta"],
        "speculative_scale": metadata["contact_speculative_scale"],
        "activation_gap": metadata["joint_limit_activation_gap"],
        "restitution_threshold": metadata["_effective_restitution_velocity_threshold"],
        "shared_anchor": int(metadata["contact_shared_anchor"]),
        "absolute_margin": 1e-5,
        "relative_margin": 2.0**-20,
    }
    for name, field in zero._SimpleWorldInput.vars.items():
        if name in scalar:
            setattr(data, name, scalar[name])
            continue
        if name == "world_dof_count":
            value = np.sum(snapshot["solve_world_dof_indices"] >= 0, axis=1, dtype=np.int32)
        else:
            key = aliases.get(name)
            if key is None:
                key = next(
                    key
                    for key in ("post3_solver_" + name, "post3_aug_" + name, "full_model_" + name, "solver_" + name)
                    if key in snapshot
                )
            value = snapshot[key]
        setattr(data, name, wp.array(value, dtype=field.type.dtype, device="cpu"))
    for name, field in zero._SimpleRawContacts.vars.items():
        key = {
            "shape_body": "full_model_shape_body",
            "shape_mu": "solver_shape_material_mu",
            "shape_restitution": "solver_shape_material_restitution",
        }.get(name, "contact_rigid_contact_" + name)
        setattr(raw, name, wp.array(snapshot[key], dtype=field.type.dtype, device="cpu"))
    worlds, bodies = data.world_dof_indices.shape[0], data.body_q.shape[0]
    for name, field in zero._SimpleWorldScratch.vars.items():
        count = 1 if name == "global_invalid" else bodies if name in ("body_twist", "body_valid") else worlds
        setattr(scratch, name, wp.zeros(count, dtype=field.type.dtype, device="cpu"))
    wp.launch(zero.prepare_simple_worlds, dim=worlds, inputs=[1, data, scratch], device="cpu")
    wp.launch(zero.build_predicted_body_twists, dim=bodies, inputs=[data, scratch], device="cpu")
    threads = max(1, min(raw.shape0.shape[0], 65536))
    wp.launch(zero.check_raw_contact_normals, dim=threads, inputs=[threads, raw, data, scratch], device="cpu")
    wp.launch(zero.finalize_simple_worlds, dim=worlds, inputs=[scratch], device="cpu")
    return data, raw, scratch


def host_owner(bundle, cohort, index):
    """Bind the actual publish method to independent original operand owners."""
    values, solver, model = bundle.values, bundle.solver, bundle.solver.model
    for name in (
        "joint_type",
        "joint_q_start",
        "joint_qd_start",
        "joint_X_p",
        "joint_X_c",
        "joint_axis",
        "joint_dof_dim",
        "body_com",
        "body_mass",
        "body_inertia",
        "gravity",
    ):
        setattr(model, name, values[name])
    for name, value in {
        "v_out": values["v_new"],
        "_kinematic_dof_mask": values["kinematic_dof_mask"],
        "_kinematic_joint_mask": values["kinematic_joint_mask"],
        "_free_root_joint_count": bundle.dims[1],
        "_free_root_joint_indices": values["free_root_joint_indices"],
        "angular_damping": values["angular_damping"],
        "body_X_com": values["body_X_com"],
        "articulation_origin": values["articulation_origin"],
        "_fk_id_cache_valid": values["fk_id_cache_valid"],
        "is_free_rigid": values["is_free_rigid"],
        "_body_inertia_terms": values["body_inertia_terms"],
        "_step": 1600 + index,
        "update_mass_matrix_interval": 2,
        "_global_inertia_stream": object(),
    }.items():
        setattr(solver, name, value)
    state_in = SimpleNamespace(joint_q=values["joint_q"], joint_qd=values["joint_qd"])
    state_out = SimpleNamespace(
        joint_q=values["joint_q_new"],
        joint_qd=values["joint_qd_new"],
        body_q=values["body_q"],
        body_qd=values["body_qd"],
    )
    state_aug = SimpleNamespace(
        **{
            name: values[name]
            for name in (
                "joint_qdd",
                "body_q_com",
                "joint_S_s",
                "body_v_s",
                "body_a_s",
                "body_I_s",
                "body_f_s",
            )
        }
    )
    owner = object.__new__(early.EarlyKuka)
    owner.solver, owner.data = solver, cohort
    owner.kernels = {
        (name, selected): early.get_publication_kernel(name, selected)
        for name in early.PUBLICATION
        for selected in (False, True)
    }
    return owner, (state_in, state_aug, state_out, values["dt"])


class TestEarlyKukaPublication(unittest.TestCase):
    def test_actual_zero_selection_and_global35_host_publication(self):
        """Publish every selected global DOF, including both six-DOF root families."""
        for index in (0, 1):
            with self.subTest(index=index), fixture.load_snapshot(index) as snapshot:
                _, _, scratch = classifier(snapshot)
                mask = scratch.resolved.numpy().copy()
                self.assertEqual(int(scratch.global_invalid.numpy()[0]), 0)
                self.assertGreater(np.count_nonzero(mask), 0)
                self.assertLess(np.count_nonzero(mask), len(mask))
                reference, candidate = fixture.bind(snapshot, index, "cpu"), fixture.bind(snapshot, index, "cpu")
                data = early.allocate_cohort(candidate.solver)
                fixture.classify(candidate, data, mask)
                dof_art, art_world = data.dof_art.numpy(), data.art_world.numpy()
                self.assertTrue(np.all(dof_art >= 0))
                np.testing.assert_array_equal(np.bincount(art_world[dof_art]), np.full(512, 35))
                prescribed = snapshot["post3_solver__prescribed_articulation"] != 0
                selected = data.art_mask.numpy() != 0
                self.assertEqual(np.count_nonzero(selected & prescribed), np.count_nonzero(mask))
                roots = snapshot["post3_solver__free_root_joint_indices"]
                joint_art = snapshot["full_model_joint_articulation"]
                self.assertEqual(np.count_nonzero(selected[joint_art[roots]]), 2 * np.count_nonzero(mask))
                owner, args = host_owner(candidate, data, index)
                fixture.original(reference)
                inertia_before = candidate.values["body_I_s"].numpy().copy()
                owner.publish(*args, early=True, finalize=False)
                np.testing.assert_array_equal(candidate.values["body_I_s"].numpy(), inertia_before)
                owner.publish(*args, early=True, finalize=True)
                owner.publish(*args, early=False, finalize=False)
                owner.publish(*args, early=False, finalize=True)
                fixture.check_outputs(self, candidate, reference)

    def test_cross_field_alias_and_active_lifecycle(self):
        """Reject every source/target alias before any early cache writes."""
        fields = ("joint_q", "joint_qd", "body_q", "body_qd")
        for source_name in fields:
            for target_name in fields:
                with self.subTest(source=source_name, target=target_name):
                    owner = object.__new__(early.EarlyKuka)
                    owner.active, owner.finalized = False, False
                    source = SimpleNamespace(**{name: wp.zeros(8, device="cpu") for name in fields})
                    target = SimpleNamespace(**{name: wp.zeros(8, device="cpu") for name in fields})
                    setattr(target, target_name, getattr(source, source_name))
                    with self.assertRaisesRegex(ValueError, "disjoint"):
                        owner.begin(source, target)
        owner.active = True
        with self.assertRaises(RuntimeError):
            owner.begin(source, target)
        with self.assertRaises(RuntimeError):
            owner.wait()

    def test_actual_host_fork_reader_and_join_order(self):
        """Require full v copy, reader completion and final join in actual host methods."""
        events = []

        class Stream:
            def __init__(self, name):
                self.name = name

            def record_event(self, event):
                events.append((self.name, "record", event))

            def wait_event(self, event):
                events.append((self.name, "wait", event))

        class Counter:
            def zero_(self):
                events.append("counts_zero")

        owner = object.__new__(early.EarlyKuka)
        owner.active, owner.finalized = False, False
        resolved = object()
        owner.data = SimpleNamespace(resolved=resolved, counts=Counter())
        owner.solver = SimpleNamespace(
            model=SimpleNamespace(device="cpu", articulation_count=3),
            _resolved_simple_worlds=resolved,
            v_out="vout",
            v_hat="vhat",
        )
        main, owner.stream = Stream("main"), Stream("early")
        owner.ready, owner.reader_done, owner.done = "ready", "reader", "done"
        owner.publish = lambda *args, **kwargs: events.append(("publish", kwargs["early"], kwargs["finalize"]))
        with (
            patch.object(wp, "get_stream", return_value=main),
            patch.object(wp, "ScopedStream", side_effect=lambda *args, **kwargs: nullcontext()),
            patch.object(wp, "launch", side_effect=lambda *args, **kwargs: events.append("compact")),
            patch.object(wp, "copy", side_effect=lambda *args: events.append(("copy", *args))),
        ):
            owner.launch("in", "aug", "out", 1 / 240)
            events.append("original_mf_inverse")
            owner.after_mf_inverse()
            with self.assertRaises(RuntimeError):
                owner.after_mf_inverse()
            owner.finish("in", "aug", "out", 1 / 240)
        self.assertEqual(
            events,
            [
                "counts_zero",
                "compact",
                ("copy", "vout", "vhat"),
                ("main", "record", "ready"),
                ("early", "wait", "ready"),
                ("publish", True, False),
                "original_mf_inverse",
                ("main", "record", "reader"),
                ("early", "wait", "reader"),
                ("publish", True, True),
                ("early", "record", "done"),
                ("publish", False, False),
                ("publish", False, True),
                ("main", "wait", "done"),
            ],
        )
        self.assertFalse(owner.active)
        self.assertIsNone(owner.current)
        self.assertEqual(owner.solver._fk_id_cache_source_state, "out")
        owner.wait()

    def test_source_reader_and_final_state_seams(self):
        """Keep cache readers behind current routing and final state readers behind the join."""
        allocation = inspect.getsource(kernels.allocate_world_contact_slots.func)
        self.assertLess(
            allocation.index("resolved_worlds[world] != 0"), allocation.index("_allocate_world_contact_slot(")
        )
        build = inspect.getsource(kernels.build_joint_limit_rows_for_size.func)
        self.assertLess(build.index("resolved_worlds[world] != 0"), build.index("q_val = joint_q[q_start + axis]"))
        setup = inspect.getsource(SolverFeatherPGS._mf_pgs_setup)
        self.assertLess(setup.index("compute_mf_body_Hinv"), setup.index("after_mf_inverse"))
        self.assertLess(setup.index("after_mf_inverse"), setup.index("compute_mf_effective_mass_and_rhs"))
        step = inspect.getsource(SolverFeatherPGS.step)
        self.assertLess(step.index("self._early_kuka.finish"), step.index("S7_Dense_Warmstart_Carry"))
        prepare = inspect.getsource(SolverFeatherPGS._stage6_prepare_world_velocity)
        self.assertLess(prepare.index("self._early_kuka.active"), prepare.index("wp.copy"))


if __name__ == "__main__":
    unittest.main()
