# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Saved-input controls for the light joint-world predictor and ZERO ownership."""

import ast
import hashlib
import inspect
import json
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import kernels
from newton._src.solvers.feather_pgs import kuka_joint_world as light
from newton._src.solvers.feather_pgs import simple_world as zero
from newton._src.solvers.feather_pgs.raw_world_contacts import RawWorldContactBuckets

PINS = (
    "5e79694fee997003ef299b33eed223d861244200bd57ac9ee6ef69c8d3b74c34",
    "85084c49880a217e3ab52cbc3340bd5f174349fab8baa77cf0e099e1a9c6d815",
    "8a798762f69fd20dd66c4e0ebd61cc59e62328fc65f7ceaf1dc687f514c84dc3",
    "21ff99f3a56f4425d0d4772a1cd3c567a1c3a439b575d336f93ea1e62c8ca377",
)


def snapshots():
    """Load both GPUs and both held-factor phases from the exact historical capture."""
    for index, pin in enumerate(PINS):
        gpu, phase = divmod(index, 2)
        path = Path(f"/tmp/fpgs-kuka-live-512-20260911-02/gpu{gpu}/kuka_step{1600 + phase}_capture0{phase}.npz")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != pin:
                raise RuntimeError("Saved original Kuka inputs changed")
        with np.load(path, allow_pickle=False) as snapshot:
            yield snapshot


def plan_inputs(snapshot):
    """Bind actual global35 and current ancestry metadata, not response29 aliases."""
    starts = snapshot["post3_solver_articulation_dof_start"]
    count = len(snapshot["pre_state_joint_qd"])
    return {
        "world_count": snapshot["solve_world_dof_indices"].shape[0],
        "art_world": snapshot["post3_solver_art_to_world"],
        "art_start": snapshot["full_model_articulation_start"],
        "art_end": snapshot["post3_solver_articulation_joint_end"],
        "dof_start": starts,
        "dof_count": np.diff(np.r_[starts, count]),
        "response_count": snapshot["post3_solver_articulation_response_dof_count"],
        "primary_arts": snapshot["group_to_art_23"],
        "secondary_arts": snapshot["group_to_art_6"],
        "prescribed": snapshot["post3_solver__prescribed_articulation"],
        "joint_parent": snapshot["full_model_joint_parent"],
        "joint_child": snapshot["full_model_joint_child"],
        "joint_qd_start": snapshot["full_model_joint_qd_start"],
        "joint_dof_dim": snapshot["full_model_joint_dof_dim"],
        "body_art": snapshot["post3_solver_body_to_articulation"],
        "body_response_mask": snapshot["post3_solver_body_response_dof_mask"],
        "free_root_joints": snapshot["post3_solver__free_root_joint_indices"],
    }


def bind(snapshot, device):
    """Bind complete original current inputs and independent output storage."""
    plan = light.build_plan(**plan_inputs(snapshot))
    metadata = json.loads(str(snapshot["post3_metadata_json"]))
    data, raw = zero._SimpleWorldInput(), zero._SimpleRawContacts()
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
    scalars = {
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
        if name in scalars:
            setattr(data, name, scalars[name])
            continue
        if name == "world_dof_count":
            value = np.sum(snapshot["solve_world_dof_indices"] >= 0, axis=1, dtype=np.int32)
        else:
            key = aliases.get(name)
            if key is None:
                key = next(
                    k
                    for k in ("post3_solver_" + name, "post3_aug_" + name, "full_model_" + name, "solver_" + name)
                    if k in snapshot
                )
            value = snapshot[key]
        setattr(data, name, wp.array(value, dtype=field.type.dtype, device=device))
    for name, field in zero._SimpleRawContacts.vars.items():
        key = {
            "shape_body": "full_model_shape_body",
            "shape_mu": "solver_shape_material_mu",
            "shape_restitution": "solver_shape_material_restitution",
        }.get(name, "contact_rigid_contact_" + name)
        setattr(raw, name, wp.array(snapshot[key], dtype=field.type.dtype, device=device))
    predictor = light.PredictorData()
    pred_keys = {
        "lower23": "post3_L_23",
        "lower6": "post3_L_6",
        "inverse23": "post3_Linv_23",
        "inverse6": "post3_Linv_6",
        "tau": "post3_aug_joint_tau",
        "qd": "post3_stage3_qd",
        "qdd": "post3_aug_joint_qdd",
        "kinematic_dof": "post3_solver__kinematic_dof_mask",
        "kinematic_joint": "post3_solver__kinematic_joint_mask",
        "free_root_joints": "post3_solver__free_root_joint_indices",
        "joint_qd_start": "full_model_joint_qd_start",
    }
    for name, field in light.PredictorData.vars.items():
        setattr(predictor, name, wp.array(snapshot[pred_keys[name]], dtype=field.type.dtype, device=device))
    integration = light.IntegrationData()
    for name, field in light.IntegrationData.vars.items():
        if name == "angular_damping":
            integration.angular_damping = 0.0
            continue
        key = {
            "kinematic_joint_mask": "post3_solver__kinematic_joint_mask",
            "joint_q": "pre_state_joint_q",
            "joint_qd": "pre_state_joint_qd",
            "joint_q_new": "pre_state_joint_q",
            "joint_qd_new": "pre_state_joint_qd",
        }.get(name, "full_model_" + name)
        setattr(integration, name, wp.array(snapshot[key], dtype=field.type.dtype, device=device))
    worlds, bodies = plan.body_ids.shape[0], data.body_q.shape[0]
    output = light.LightOutput()
    for name, field in light.LightOutput.vars.items():
        count = (
            1
            if name == "active_count"
            else bodies
            if name == "endpoint_twists"
            else data.v_hat.shape[0]
            if name == "v_out"
            else worlds
        )
        setattr(output, name, wp.zeros(count, dtype=field.type.dtype, device=device))
    buckets = RawWorldContactBuckets(worlds, raw.shape0.shape[0], device=device)
    buckets.build(raw.count, raw.shape0, raw.shape1, raw.shape_body, data.body_to_articulation, data.art_to_world)
    return SimpleNamespace(
        plan=plan,
        device_plan=plan.device_data(device),
        data=data,
        raw=raw,
        predictor=predictor,
        output=output,
        integration=integration,
        buckets=buckets,
        device=device,
    )


def launch(bundle):
    """Charge and execute the complete one-warp predictor/selector/generalized owner."""
    bundle.output.active_count.zero_()
    wp.launch_tiled(
        light.get_kernel(str(wp.get_device(bundle.device).arch)),
        dim=[len(bundle.plan.body_ids)],
        inputs=[
            bundle.device_plan,
            bundle.predictor,
            bundle.data,
            bundle.raw,
            bundle.buckets.data,
            bundle.output,
            bundle.integration,
        ],
        block_dim=32,
        device=bundle.device,
    )


def original_selection(bundle):
    """Evaluate every original signed limit and normal at the candidate's actual predictor."""
    data, raw, device = bundle.data, bundle.raw, bundle.device
    scratch = zero._SimpleWorldScratch()
    for name, field in zero._SimpleWorldScratch.vars.items():
        count = (
            1
            if name == "global_invalid"
            else data.body_q.shape[0]
            if name in ("body_twist", "body_valid")
            else data.world_dof_count.shape[0]
        )
        setattr(scratch, name, wp.zeros(count, dtype=field.type.dtype, device=device))
    wp.launch(zero.prepare_simple_worlds, dim=data.world_dof_count.shape[0], inputs=[1, data, scratch], device=device)
    wp.launch(zero.build_predicted_body_twists, dim=data.body_q.shape[0], inputs=[data, scratch], device=device)
    threads = max(raw.shape0.shape[0], 1)
    wp.launch(zero.check_raw_contact_normals, dim=threads, inputs=[threads, raw, data, scratch], device=device)
    wp.launch(zero.finalize_simple_worlds, dim=data.world_dof_count.shape[0], inputs=[scratch], device=device)
    return scratch


def original_publication(bundle):
    """Use original functions after original generalized-velocity ownership."""
    p, i, o, d = bundle.predictor, bundle.integration, bundle.output, bundle.data
    wp.copy(o.v_out, d.v_hat)
    wp.launch(
        kernels.update_qdd_from_velocity,
        dim=d.v_hat.shape[0],
        inputs=[i.joint_qd, p.kinematic_dof, 1.0 / d.dt, o.v_out, p.qdd],
        device=bundle.device,
    )
    wp.launch(
        kernels.remove_free_root_transport_from_qdd,
        dim=p.free_root_joints.shape[0],
        inputs=[p.free_root_joints, p.joint_qd_start, p.kinematic_joint, i.joint_qd, p.qdd],
        device=bundle.device,
    )
    wp.launch(
        kernels.integrate_generalized_joints,
        dim=i.joint_type.shape[0],
        inputs=[
            i.joint_type,
            i.joint_parent,
            i.joint_child,
            i.joint_q_start,
            i.joint_qd_start,
            i.kinematic_joint_mask,
            i.joint_dof_dim,
            i.body_com,
            i.joint_X_c,
            i.joint_q,
            i.joint_qd,
            p.qdd,
            d.dt,
            i.angular_damping,
            i.joint_q_new,
            i.joint_qd_new,
        ],
        device=bundle.device,
    )


class TestKukaJointWorld(unittest.TestCase):
    def test_native_entrypoint_exists(self):
        """Require the complete light owner before claiming runtime readiness."""
        self.assertTrue(callable(getattr(light, "get_kernel", None)))

    def test_actual_static_maps_and_counterexamples(self):
        """Cover every physical DOF and reject incorrect ancestry or prescribed ownership."""
        iterator = snapshots()
        snapshot = next(iterator)
        inputs = plan_inputs(snapshot)
        plan = light.build_plan(**inputs)
        np.testing.assert_array_equal(np.sort(plan.dof_ids.ravel()), np.arange(512 * 35))
        np.testing.assert_array_equal(np.sort(plan.body_ids.ravel()), np.arange(512 * 32))
        self.assertEqual(plan.body_dof[plan.body_dof >= 0].size, 512 * 23)
        for field in ("body_response_mask", "prescribed", "joint_parent"):
            with self.subTest(field=field):
                wrong = {**inputs, field: inputs[field].copy()}
                if field == "body_response_mask":
                    wrong[field][plan.body_ids[0, 20]] ^= np.uint32(1)
                elif field == "prescribed":
                    wrong[field][:] = 0
                else:
                    wrong[field][plan.joint_ids[0, 10]] = plan.body_ids[1, 0]
                with self.assertRaises(ValueError):
                    light.build_plan(**wrong)
        model = SimpleNamespace(joint_dof_count=len(snapshot["pre_state_joint_qd"]))
        solver = SimpleNamespace(model=model, world_count=512)
        model_fields = ("articulation_start", "joint_parent", "joint_child", "joint_qd_start", "joint_dof_dim")
        for name in model_fields:
            setattr(model, name, wp.array(snapshot["full_model_" + name], dtype=int, device="cpu"))
        for name in (
            "art_to_world",
            "articulation_joint_end",
            "articulation_dof_start",
            "articulation_response_dof_count",
            "_prescribed_articulation",
            "body_to_articulation",
            "body_response_dof_mask",
            "_free_root_joint_indices",
        ):
            dtype = wp.uint32 if name == "body_response_dof_mask" else int
            setattr(solver, name, wp.array(snapshot["post3_solver_" + name], dtype=dtype, device="cpu"))
        solver.group_to_art = {n: wp.array(snapshot[f"group_to_art_{n}"], dtype=int, device="cpu") for n in (23, 6)}
        np.testing.assert_array_equal(light.bind_plan(solver).dof_ids, plan.dof_ids)

    def test_parent_prefix_against_current_mask_contraction(self):
        """Match original current-S body action without reusing incoming body velocity."""
        for snapshot in snapshots():
            plan = light.build_plan(**plan_inputs(snapshot))
            axes, velocity = snapshot["post3_aug_joint_S_s"], snapshot["pre_rows_v_hat"]
            actual = light.prefix_reference(plan, axes, velocity)
            expected = np.zeros_like(actual)
            masks = snapshot["post3_solver_body_response_dof_mask"][plan.body_ids[:, :30]]
            for k in range(23):
                dofs = plan.dof_ids[:, k]
                contribution = axes[dofs] * velocity[dofs, None]
                expected[:, :30] += np.where(((masks >> k) & 1)[..., None], contribution[:, None, :], 0.0)
            np.testing.assert_allclose(actual[:, :30], expected[:, :30], atol=3e-6, rtol=3e-6)

    def test_two_dot_predictor_against_held_physical_operator(self):
        """Keep held R+K mass authority and compare action, not exact FP32 iterates."""
        for snapshot in snapshots():
            plan = light.build_plan(**plan_inputs(snapshot))
            for size, offset in ((23, 0), (6, 23)):
                ids = plan.dof_ids[:, offset : offset + size]
                tau = snapshot["post3_aug_joint_tau"][ids]
                lower = np.tril(snapshot[f"post3_L_{size}"]).astype(np.float64)
                inverse = np.tril(snapshot[f"post3_Linv_{size}"])
                actual = light.predictor_reference(inverse, tau)
                reference = np.linalg.solve(np.swapaxes(lower, 1, 2), np.linalg.solve(lower, tau[..., None]))[..., 0]
                self.assertTrue(np.isfinite(actual).all())
                # This bound is checked on acceleration action, not ZERO eligibility.
                np.testing.assert_allclose(actual, reference, atol=3e-5, rtol=3e-6)

    def check_native(self, device, all_cases):
        """Check original physical action, conservative decisions and disjoint generalized outputs."""
        for index, snapshot in enumerate(snapshots()):
            if not all_cases and index > 0:
                break
            candidate, reference = bind(snapshot, device), bind(snapshot, device)
            before_q = candidate.integration.joint_q_new.numpy().copy()
            before_qd = candidate.integration.joint_qd_new.numpy().copy()
            launch(candidate)
            expected_velocity = snapshot["pre_rows_v_hat"]
            np.testing.assert_allclose(candidate.data.v_hat.numpy(), expected_velocity, atol=3e-6, rtol=3e-6)
            original = original_selection(candidate)
            selected = candidate.output.resolved.numpy().astype(bool)
            self.assertGreater(np.count_nonzero(selected), 0)
            self.assertLess(np.count_nonzero(selected), len(selected))
            self.assertTrue(np.all(original.resolved.numpy()[selected] == 1))
            active = candidate.output.active_worlds.numpy()[: candidate.output.active_count.numpy()[0]]
            np.testing.assert_array_equal(np.sort(active), np.flatnonzero(~selected))
            reference.data.v_hat.assign(candidate.data.v_hat.numpy())
            original_publication(reference)
            joint_world = snapshot["post3_solver_art_to_world"][snapshot["full_model_joint_articulation"]]
            coord_world = np.repeat(joint_world, np.diff(snapshot["full_model_joint_q_start"]))
            dof_world = np.repeat(joint_world, np.diff(snapshot["full_model_joint_qd_start"]))
            for name, owners, before in (
                ("joint_q_new", coord_world, before_q),
                ("joint_qd_new", dof_world, before_qd),
            ):
                actual, expected = (
                    getattr(candidate.integration, name).numpy(),
                    getattr(reference.integration, name).numpy(),
                )
                np.testing.assert_allclose(actual[selected[owners]], expected[selected[owners]], atol=3e-6, rtol=3e-6)
                np.testing.assert_array_equal(actual[~selected[owners]], before[~selected[owners]])
            ids = candidate.plan.dof_ids[selected].ravel()
            np.testing.assert_allclose(
                candidate.predictor.qdd.numpy()[ids], reference.predictor.qdd.numpy()[ids], atol=3e-6, rtol=3e-6
            )

    def test_actual_native_cpu(self):
        """Check the complete native owner on all four saved inputs using CPU execution."""
        self.check_native("cpu", True)

    def test_invalid_and_nonfinite_inverse_original_lower_fallback(self):
        """Invalid buckets and a nonfinite W retain original L predictor without partial q/qd publication."""
        iterator = snapshots()
        snapshot = next(iterator)
        for mode in ("raw", "inverse"):
            candidate = bind(snapshot, "cpu")
            before_q = candidate.integration.joint_q_new.numpy().copy()
            before_qd = candidate.integration.joint_qd_new.numpy().copy()
            if mode == "raw":
                candidate.buckets.data.invalid.fill_(1)
                candidate.buckets.data.offsets.fill_(-1000)
            else:
                inverse = candidate.predictor.inverse23.numpy().copy()
                inverse[candidate.plan.primary_group[0], 0, 0] = np.nan
                candidate.predictor.inverse23.assign(inverse)
            launch(candidate)
            fallback = candidate.output.predictor_fallback.numpy() != 0
            self.assertEqual(np.count_nonzero(fallback), 512 if mode == "raw" else 1)
            self.assertTrue(np.all(candidate.output.resolved.numpy()[fallback] == 0))
            np.testing.assert_allclose(candidate.data.v_hat.numpy(), snapshot["pre_rows_v_hat"], atol=3e-6, rtol=3e-6)
            joint_world = snapshot["post3_solver_art_to_world"][snapshot["full_model_joint_articulation"]]
            for name, starts, before in (
                ("joint_q_new", "full_model_joint_q_start", before_q),
                ("joint_qd_new", "full_model_joint_qd_start", before_qd),
            ):
                owners = np.repeat(joint_world, np.diff(snapshot[starts]))
                np.testing.assert_array_equal(
                    getattr(candidate.integration, name).numpy()[fallback[owners]], before[fallback[owners]]
                )

    def test_original_normal_and_publication_source_closure(self):
        """Nonresponding pairs are neutral, and original generalized numerical bodies are preserved."""
        tree = ast.parse(light.normal_source())
        normal = tree.body[-1]
        neutral = next(
            node
            for node in normal.body
            if isinstance(node, ast.If)
            and ast.dump(node.test) == ast.dump(ast.parse("not responds_a and not responds_b", mode="eval").body)
        )
        self.assertIs(neutral.body[0].value.value, True)
        self.assertNotIn("scratch", light.normal_source())
        for name in ("update_qdd_from_velocity", "remove_free_root_transport_from_qdd", "integrate_generalized_joints"):
            source = ast.parse(textwrap.dedent(inspect.getsource(getattr(kernels, name).func))).body[0]
            adapted = ast.parse(light.operation_source(name)).body[0]

            class Restore(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id == "physical_index":
                        return ast.parse("wp.tid()", mode="eval").body
                    return node

            actual = Restore().visit(ast.Module(body=adapted.body, type_ignores=[]))
            self.assertEqual(ast.dump(actual), ast.dump(ast.Module(body=source.body, type_ignores=[])))

    @unittest.skipUnless(wp.is_cuda_available(), "Actual CUDA device required")
    def test_actual_native_cuda(self):
        """Check the actual warp mapping and numerical action on the selected CUDA device."""
        self.check_native("cuda:0", True)

    @unittest.skipUnless(wp.is_cuda_available(), "Actual CUDA device required")
    def test_two_actual_graphs_empty_invalid_and_regrowth(self):
        """Replay two graph objects through complete raw-prefix and ownership changes."""
        iterator = snapshots()
        snapshot = next(iterator)
        candidate = bind(snapshot, "cuda:0")
        launch(candidate)
        count = int(candidate.raw.count.numpy()[0])
        raw, data = candidate.raw, candidate.data
        for _ in range(2):
            with wp.ScopedCapture(device="cuda:0") as capture:
                candidate.buckets.build(
                    raw.count, raw.shape0, raw.shape1, raw.shape_body, data.body_to_articulation, data.art_to_world
                )
                launch(candidate)
            for current in (0, count, count + 1, count):
                candidate.raw.count.fill_(current)
                candidate.integration.joint_q_new.assign(snapshot["pre_state_joint_q"])
                candidate.integration.joint_qd_new.assign(snapshot["pre_state_joint_qd"])
                candidate.output.active_worlds.fill_(-71)
                wp.capture_launch(capture.graph)
                selected = candidate.output.resolved.numpy() != 0
                active_count = int(candidate.output.active_count.numpy()[0])
                active = candidate.output.active_worlds.numpy()[:active_count]
                np.testing.assert_array_equal(np.sort(active), np.flatnonzero(~selected))
                if current > count:
                    self.assertEqual(active_count, 512)
                    self.assertEqual(np.count_nonzero(selected), 0)
                    np.testing.assert_array_equal(
                        candidate.integration.joint_q_new.numpy(), snapshot["pre_state_joint_q"]
                    )
                else:
                    original = original_selection(candidate)
                    self.assertGreater(np.count_nonzero(selected), 0)
                    self.assertTrue(np.all(original.resolved.numpy()[selected] == 1))
                self.assertTrue(np.isfinite(candidate.data.v_hat.numpy()).all())


if __name__ == "__main__":
    unittest.main()
