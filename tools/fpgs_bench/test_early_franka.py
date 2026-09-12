# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent early-cohort native solve and complete publication controls.

Saved source inputs are pinned by the reused packet-local fixture. CPU tests
exercise original and masked publication; native GS and graph ownership require
CUDA. This does not establish rollout or end-to-end performance equivalence.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import early_franka as early
from newton._src.solvers.feather_pgs import kernels
from tools.fpgs_bench import test_franka_packet_local as local

KERNELS = (
    kernels.update_qdd_from_velocity,
    kernels.remove_free_root_transport_from_qdd,
    kernels.integrate_generalized_joints,
    kernels.eval_rigid_fk_kinematics,
    kernels.finalize_body_dynamics,
)
OUTPUT_KEYS = {
    "v_new": "solved__v_out",
    "joint_qdd": "state_aug__joint_qdd",
    "joint_q_new": "initial_state_out__joint_q",
    "joint_qd_new": "initial_state_out__joint_qd",
    "body_q": "initial_state_out__body_q",
    "body_qd": "initial_state_out__body_qd",
    "body_q_com": "state_aug__body_q_com",
    "articulation_origin": "solver__articulation_origin",
    "joint_S_s": "state_aug__joint_S_s",
    "body_v_s": "state_aug__body_v_s",
    "body_a_s": "state_aug__body_a_s",
    "body_f_s": "state_aug__body_f_s",
    "body_I_s": "state_aug__body_I_s",
    "body_inertia_terms": "solver___body_inertia_terms",
    "fk_id_cache_valid": "solver___fk_id_cache_valid",
}


def bind(snapshot, capture, device):
    """Bind exact original publication inputs and independent writable arrays."""
    aliases = {
        "joint_q": "state_in__joint_q",
        "joint_qd": "state_in__joint_qd",
        "kinematic_dof_mask": "solver___kinematic_dof_mask",
        "kinematic_joint_mask": "solver___kinematic_joint_mask",
        "free_root_joint_indices": "solver___free_root_joint_indices",
    }
    next_refresh = (capture["solver_step"] + 1) % 2 == 0
    values = {
        "inv_dt": 1.0 / capture["dt"],
        "dt": capture["dt"],
        "angular_damping": capture["settings"]["solver"]["angular_damping"],
        "materialize_all_body_inertia": 0,
        "materialize_body_inertia_terms": int(next_refresh),
    }
    for kernel in KERNELS:
        for argument in kernel.adj.args:
            name = argument.label
            if name in values:
                continue
            key = OUTPUT_KEYS.get(name, aliases.get(name))
            if key is None:
                key = next(key for key in ("model__" + name, "solver__" + name) if key in snapshot)
            values[name] = wp.array(snapshot[key], dtype=argument.type.dtype, device=device)
    outputs = {name: values[name] for name in OUTPUT_KEYS}
    seeds = {name: wp.clone(value) for name, value in outputs.items()}
    dimensions = (
        snapshot["state_in__joint_qd"].shape[0],
        snapshot["solver___free_root_joint_indices"].shape[0],
        snapshot["model__joint_type"].shape[0],
        snapshot["solver__articulation_joint_end"].shape[0],
        snapshot["model__body_mass"].shape[0],
    )
    return SimpleNamespace(
        values=values, outputs=outputs, seeds=seeds, dimensions=dimensions, device=device, capture=capture
    )


def restore(bundle):
    """Restore exact pre-publication state, including held inertia fields."""
    for name, value in bundle.outputs.items():
        wp.copy(value, bundle.seeds[name])


def arguments(bundle, index):
    """Switch FK to newly integrated coordinates, preserving original input q."""
    values = dict(bundle.values)
    if index == 3:
        values["joint_q"] = bundle.values["joint_q_new"]
        values["joint_qd"] = bundle.values["joint_qd_new"]
    return values


def run(bundle):
    """Execute the original complete publication chain from saved solved v."""
    for index, (kernel, dim) in enumerate(zip(KERNELS, bundle.dimensions, strict=True)):
        values = arguments(bundle, index)
        wp.launch(kernel, dim=dim, inputs=[values[arg.label] for arg in kernel.adj.args], device=bundle.device)


def expected_hot(snapshot):
    """Derive the conservative raw-endpoint cohort without accepted-row shortcuts."""
    owners = snapshot["solver___local_solve_owner"]
    primary = snapshot["solver___local_primary_articulation"]
    count = snapshot["solver__constraint_count"]
    hot = (owners == 1) & (snapshot["solver__mf_constraint_count"] == 0)
    hot &= (count > 0) & (count <= 9)
    hot &= snapshot["solver__dense_phase_bounds"][:, 1] == count
    total = int(snapshot["contacts__rigid_contact_count"][0])
    shapes = [snapshot["contacts__rigid_contact_shape" + str(side)] for side in (0, 1)]
    if total < 0 or total > min(len(value) for value in shapes):
        return np.zeros_like(hot)
    body_art = snapshot["solver__body_to_articulation"]
    shape_body = snapshot["model__shape_body"]
    incident = set()
    for shape_ids in shapes:
        for shape in shape_ids[:total]:
            if shape < 0:
                continue
            body = shape_body[shape]
            if body >= 0:
                incident.add(int(body_art[body]))
    return hot & ~np.isin(primary, list(incident))


def output_masks(snapshot, hot):
    """Map the world cohort to primary-only physical output indices."""
    primary = snapshot["solver___local_primary_articulation"][hot]
    art_hot = np.zeros_like(snapshot["solver__art_to_world"], dtype=bool)
    art_hot[primary] = True
    body_art = snapshot["solver__body_to_articulation"]
    body_hot = (body_art >= 0) & art_hot[np.maximum(body_art, 0)]
    dof_hot = np.zeros_like(snapshot["state_in__joint_qd"], dtype=bool)
    coord_hot = np.zeros_like(snapshot["state_in__joint_q"], dtype=bool)
    for art in primary:
        start = snapshot["solver__articulation_dof_start"][art]
        dof_hot[start : start + 9] = True
        start, end = snapshot["model__articulation_start"][art : art + 2]
        first = snapshot["model__joint_q_start"][start]
        last = snapshot["model__joint_q_start"][end]
        coord_hot[first:last] = True
    masks = dict.fromkeys(OUTPUT_KEYS, body_hot)
    for name in ("v_new", "joint_qdd", "joint_qd_new", "joint_S_s"):
        masks[name] = dof_hot
    masks["joint_q_new"] = coord_hot
    for name in ("articulation_origin", "fk_id_cache_valid"):
        masks[name] = art_hot
    return masks


def assert_publication(test, candidate, reference):
    """Compare complete physical/public/cache outputs on the same backend."""
    for name in OUTPUT_KEYS:
        actual, expected = candidate.outputs[name].numpy(), reference.outputs[name].numpy()
        test.assertTrue(np.isfinite(actual).all(), name)
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6, err_msg=name)


def bind_cohort(snapshot, device):
    """Bind actual maps independently of the CUDA-stream owning constructor."""
    data = early.CohortData()
    worlds = len(snapshot["solver__constraint_count"])
    arts = len(snapshot["solver__art_to_world"])
    for name in ("incidence", "early_owner", "early_count", "zero_mf", "late_owner", "early_arts"):
        setattr(data, name, wp.zeros(worlds, dtype=int, device=device))
    data.invalid_raw = wp.zeros(1, dtype=int, device=device)
    data.counts = wp.zeros(2, dtype=int, device=device)
    data.art_mask = wp.zeros(arts, dtype=int, device=device)
    data.late_arts = wp.full(arts, -71, dtype=int, device=device)
    for name, key in {
        "art_world": "solver__art_to_world",
        "primary": "solver___local_primary_articulation",
        "joint_art": "model__joint_articulation",
        "body_art": "solver__body_to_articulation",
        "art_start": "model__articulation_start",
        "art_end": "solver__articulation_joint_end",
        "dof_start": "solver__articulation_dof_start",
        "joint_child": "model__joint_child",
    }.items():
        setattr(data, name, wp.array(snapshot[key], dtype=int, device=device))
    starts = snapshot["solver__articulation_dof_start"]
    ends = np.r_[starts[1:], len(snapshot["state_in__joint_qd"])]
    data.dof_art = wp.array(np.repeat(np.arange(arts), ends - starts), dtype=int, device=device)
    primary = snapshot["solver___local_primary_articulation"]
    data.max_joints = int(
        np.max(snapshot["solver__articulation_joint_end"][primary] - snapshot["model__articulation_start"][primary])
    )
    return data


def classify(snapshot, cohort, device, *, shapes=None, mf=None):
    """Run actual raw-incidence admission and valid-list/late-owner publication."""
    arrays = {}
    for name, key in {
        "count": "contacts__rigid_contact_count",
        "shape0": "contacts__rigid_contact_shape0",
        "shape1": "contacts__rigid_contact_shape1",
        "shape_body": "model__shape_body",
        "slots": "solver__slot_counter",
        "bounds": "solver__dense_phase_bounds",
        "mf": "solver__mf_slot_counter",
        "owner": "solver___local_solve_owner",
    }.items():
        value = snapshot[key]
        if shapes is not None and name in shapes:
            value = shapes[name]
        if name == "mf" and mf is not None:
            value = mf
        arrays[name] = wp.array(value, dtype=int, device=device)
    update_cohort(cohort, arrays, device)
    return arrays


def update_cohort(cohort, arrays, device):
    """Refresh current ownership without host reads, including poisoned list tails."""
    for name in ("incidence", "invalid_raw", "counts", "art_mask"):
        getattr(cohort, name).zero_()
    cohort.early_arts.fill_(-71)
    cohort.late_arts.fill_(-71)
    workers = 128
    wp.launch(
        early.mark_raw_incidence,
        dim=workers,
        inputs=[
            arrays["count"],
            arrays["shape0"],
            arrays["shape1"],
            arrays["shape_body"],
            workers,
            cohort,
        ],
        device=device,
    )
    wp.launch(
        early.classify_early,
        dim=cohort.early_owner.shape[0],
        inputs=[
            arrays["slots"],
            arrays["bounds"],
            arrays["mf"],
            cohort,
        ],
        device=device,
    )
    wp.launch(early.compact_publication, dim=cohort.art_mask.shape[0], inputs=[cohort], device=device)
    wp.launch(
        early.prepare_late_owner, dim=cohort.early_owner.shape[0], inputs=[arrays["owner"], cohort], device=device
    )


def publish_partition(bundle, cohort, selected):
    """Call the actual owner publication method, not a test-only launch ordering."""
    values = bundle.values
    model = SimpleNamespace(**values)
    model.device = wp.get_device(bundle.device)
    model.joint_dof_count, _, model.joint_count, model.articulation_count, model.body_count = bundle.dimensions
    solver = SimpleNamespace(**values)
    solver.model = model
    solver.world_count = cohort.early_owner.shape[0]
    solver._kinematic_dof_mask = values["kinematic_dof_mask"]
    solver._kinematic_joint_mask = values["kinematic_joint_mask"]
    solver._fk_id_cache_valid = values["fk_id_cache_valid"]
    solver._body_inertia_terms = values["body_inertia_terms"]
    solver._step = bundle.capture["solver_step"]
    solver.update_mass_matrix_interval = 2
    solver._global_inertia_stream = object()
    solver.v_out = values["v_new"]

    def remove_transport(_state_in, _state_aug):
        kernel = KERNELS[1]
        wp.launch(
            kernel,
            dim=bundle.dimensions[1],
            inputs=[values[arg.label] for arg in kernel.adj.args],
            device=bundle.device,
        )

    solver._remove_free_root_transport = remove_transport
    controller = SimpleNamespace(
        solver=solver,
        data=cohort,
        kernels={
            (kernel.key, selected): early.get_publication_kernel(kernel.key, selected)
            for index, kernel in enumerate(KERNELS)
            if index != 1
        },
    )
    state_in = SimpleNamespace(joint_q=values["joint_q"], joint_qd=values["joint_qd"])
    state_aug = SimpleNamespace(**values)
    state_out = SimpleNamespace(
        joint_q=values["joint_q_new"],
        joint_qd=values["joint_qd_new"],
        body_q=values["body_q"],
        body_qd=values["body_qd"],
    )
    early.EarlyFranka.publish(controller, state_in, state_aug, state_out, values["dt"], early=selected)


def native_fixture(snapshot, capture, device):
    """Alias actual current packet readers to the cache overwritten by early FK."""
    prefix, packet, data = local.bind_current(snapshot, capture, device)
    local.produce(prefix, packet, data, device)
    solve = local.bind_calls(snapshot, prefix, packet, data, device, private=True)
    publication = bind(snapshot, capture, device)
    publication.values["v_new"] = publication.outputs["v_new"] = solve.outputs[2]
    publication.seeds["v_new"] = wp.clone(solve.seed[2])
    data.motion = publication.outputs["joint_S_s"]
    data.origin = publication.outputs["articulation_origin"]
    data.body_v = publication.outputs["body_v_s"]
    cohort = bind_cohort(snapshot, device)
    selection = classify(snapshot, cohort, device)
    early_calls, late_calls = [], []
    for index, (kernel, dim, inputs) in enumerate(solve.calls):
        if index == 0:
            early_arguments, late_arguments = list(inputs), list(inputs)
            names = [arg.label for arg in kernel.adj.args]
            for name, value in {
                "local_solve_owner": cohort.early_owner,
                "world_constraint_count": cohort.early_count,
                "mf_constraint_count": cohort.zero_mf,
            }.items():
                early_arguments[names.index(name)] = value
            late_arguments[names.index("local_solve_owner")] = cohort.late_owner
            early_calls.append((kernel, dim, early_arguments))
            late_calls.append((kernel, dim, late_arguments))
        else:
            late_calls.append((kernel, dim, inputs))
    return SimpleNamespace(
        solve=solve,
        publication=publication,
        cohort=cohort,
        selection=selection,
        early_calls=early_calls,
        late_calls=late_calls,
        device=device,
        counts=wp.array(snapshot["solver__constraint_count"], dtype=int, device=device),
        stream=wp.Stream(device) if wp.get_device(device).is_cuda else None,
    )


def launch_calls(calls, device):
    """Execute unchanged native local solver signatures and original launch widths."""
    for kernel, dim, inputs in calls:
        wp.launch_tiled(kernel, dim=[dim], inputs=inputs, block_dim=64, device=device)


def run_original(fixture):
    """Run the complete original local solve and original final publication."""
    restore(fixture.publication)
    local.run(fixture.solve)
    run(fixture.publication)


def run_split(fixture, *, negative=None):
    """Fork early solve/publication against real contact production and late work."""
    solve, pub, cohort, device = fixture.solve, fixture.publication, fixture.cohort, fixture.device
    restore(pub)
    for output, seed in zip(solve.outputs, solve.seed, strict=True):
        wp.copy(output, seed)
    wp.launch_tiled(
        local.rows.get_prefix_kernel(),
        dim=[(solve.prefix.group_to_art.shape[0] + 3) // 4],
        inputs=[solve.prefix],
        block_dim=128,
        device=device,
    )
    update_cohort(cohort, fixture.selection, device)
    main = wp.get_stream(device)
    ready = main.record_event()
    fixture.stream.wait_event(ready)
    with wp.ScopedStream(fixture.stream, sync_enter=False, sync_exit=False):
        launch_calls(fixture.early_calls, device)
        publish_partition(pub, cohort, True)
        done = fixture.stream.record_event()
    workers = max(1, min(solve.data.point0.shape[0], 512))
    wp.launch_tiled(
        local.contact.produce_contacts,
        dim=[workers],
        inputs=[workers, solve.data, solve.packet],
        block_dim=32,
        device=device,
    )
    clear = early.get_publication_kernel("prepare_world_impulses", False)
    wp.launch(
        clear,
        dim=cohort.early_owner.shape[0],
        inputs=[
            fixture.counts,
            solve.outputs[1].shape[1],
            0,
            solve.outputs[1],
            cohort,
        ],
        device=device,
    )
    if negative is not None:
        main.wait_event(done)
        if negative == "velocity_copy":
            wp.copy(solve.outputs[2], solve.seed[2])
        elif negative == "impulse_clear":
            solve.outputs[1].zero_()
        elif negative == "duplicate_solve":
            launch_calls(fixture.early_calls, device)
        else:
            raise ValueError(negative)
    launch_calls(fixture.late_calls, device)
    publish_partition(pub, cohort, False)
    main.wait_event(done)


class TestEarlyFranka(unittest.TestCase):
    """Check the current-input publication boundary independently."""

    def test_actual_constructor_without_dof_sentinel(self):
        """Build the real owner from A-entry DOF starts, including its last art."""
        capture = next(local.captures())
        with np.load(capture["path"]) as snapshot:
            solver = SimpleNamespace(world_count=len(snapshot["solver__constraint_count"]))
            solver.model = SimpleNamespace(
                device=wp.get_device("cpu"),
                articulation_count=len(snapshot["solver__art_to_world"]),
                joint_count=len(snapshot["model__joint_type"]),
                joint_dof_count=len(snapshot["state_in__joint_qd"]),
            )
            for name in ("articulation_start", "joint_child", "joint_parent", "joint_dof_dim", "joint_articulation"):
                setattr(solver.model, name, wp.array(snapshot["model__" + name], dtype=int, device="cpu"))
            for name in (
                "art_to_world",
                "_local_primary_articulation",
                "body_to_articulation",
                "articulation_joint_end",
                "articulation_dof_start",
            ):
                setattr(solver, name, wp.array(snapshot["solver__" + name], dtype=int, device="cpu"))
            self.assertEqual(solver.articulation_dof_start.shape[0], solver.model.articulation_count)
            with (
                patch.object(early.wp, "Stream", return_value=object()),
                patch.object(early.wp, "Event", return_value=object()),
            ):
                controller = early.EarlyFranka(solver)
            expected = bind_cohort(snapshot, "cpu")
            np.testing.assert_array_equal(controller.data.dof_art.numpy(), expected.dof_art.numpy())
            np.testing.assert_array_equal(controller.data.joint_art.numpy(), expected.joint_art.numpy())
            self.assertEqual(controller.data.max_joints, 11)
            self.assertEqual(controller.data.dof_art.numpy()[-1], solver.model.articulation_count - 1)

    def test_loaded_cohort_and_overwrite_negative_controls(self):
        """Reject meaningful late overwrites and retain every same-world free6."""
        for capture in local.captures():
            with self.subTest(capture=capture["path"]), np.load(capture["path"]) as snapshot:
                hot = expected_hot(snapshot)
                masks = output_masks(snapshot, hot)
                self.assertGreater(np.count_nonzero(hot), 450)
                free_bodies = snapshot["solver__free_rigid_body_indices"]
                self.assertFalse(np.any(masks["body_q"][free_bodies]))
                free_arts = snapshot["solver__group_to_art__6"]
                self.assertFalse(np.any(masks["articulation_origin"][free_arts]))
                for name, values in (
                    ("late impulse clear", snapshot["solved__impulses"][hot]),
                    ("late velocity copy", (snapshot["solved__v_out"] - snapshot["solver__v_hat"])[masks["v_new"]]),
                ):
                    with self.assertRaises(AssertionError, msg=name):
                        np.testing.assert_allclose(values, np.zeros_like(values), rtol=3e-6, atol=3e-6)

    def test_primary_raw_incidence_and_current_transitions(self):
        """Keep free6 raw endpoints late without excluding an unrelated primary."""
        capture = next(local.captures())
        with np.load(capture["path"]) as snapshot:
            cohort = bind_cohort(snapshot, "cpu")
            eligible = snapshot["solver___local_solve_owner"] == 1
            worlds = np.flatnonzero(eligible)[:2]
            primary = snapshot["solver___local_primary_articulation"][worlds]

            def shape_of(art, *, fixed=False):
                bodies = np.flatnonzero(snapshot["solver__body_to_articulation"] == art)
                if fixed:
                    joints = (snapshot["model__joint_articulation"] == art) & (snapshot["model__joint_type"] == 3)
                    bodies = snapshot["model__joint_child"][joints]
                return int(np.flatnonzero(np.isin(snapshot["model__shape_body"], bodies))[0])

            free = next(
                art for art in snapshot["solver__group_to_art__6"] if snapshot["solver__art_to_world"][art] == worlds[0]
            )
            cases = (
                ("free6_ground", 1, shape_of(free), -1, []),
                ("primary_fixed", 1, shape_of(primary[0], fixed=True), -1, [worlds[0]]),
                ("cross_world", 1, shape_of(primary[0]), shape_of(primary[1]), list(worlds)),
                ("empty_reentry", 0, -1, -1, []),
            )
            for label, count, a, b, excluded in cases:
                with self.subTest(case=label):
                    classify(
                        snapshot,
                        cohort,
                        "cpu",
                        shapes={
                            "count": np.array([count], dtype=np.int32),
                            "shape0": np.array([a], dtype=np.int32),
                            "shape1": np.array([b], dtype=np.int32),
                        },
                    )
                    expected = eligible.copy()
                    expected[excluded] = False
                    np.testing.assert_array_equal(cohort.early_owner.numpy(), expected)
            classify(
                snapshot,
                cohort,
                "cpu",
                shapes={
                    "count": np.array([2], dtype=np.int32),
                    "shape0": np.array([-1], dtype=np.int32),
                    "shape1": np.array([-1], dtype=np.int32),
                },
            )
            self.assertFalse(np.any(cohort.early_owner.numpy()))
            arrays = classify(snapshot, cohort, "cpu")
            previous = cohort.early_owner.numpy().copy()
            changed = snapshot["solver__mf_slot_counter"].copy()
            changed[worlds[0]] = 1
            arrays["mf"].assign(changed)
            update_cohort(cohort, arrays, "cpu")
            self.assertEqual(cohort.early_owner.numpy()[worlds[0]], 0)
            arrays["mf"].assign(snapshot["solver__mf_slot_counter"])
            update_cohort(cohort, arrays, "cpu")
            np.testing.assert_array_equal(cohort.early_owner.numpy(), previous)

    def test_original_publication_bindings(self):
        """Reproduce all four recorded publications at original CPU/GPU scale."""
        for capture in local.captures():
            with self.subTest(capture=capture["path"]), np.load(capture["path"]) as snapshot:
                bundle = bind(snapshot, capture, "cpu")
                run(bundle)
                for name, field in (
                    ("joint_q_new", "joint_q"),
                    ("joint_qd_new", "joint_qd"),
                    ("body_q", "body_q"),
                    ("body_qd", "body_qd"),
                ):
                    actual = bundle.outputs[name].numpy()
                    self.assertTrue(np.isfinite(actual).all())
                    np.testing.assert_allclose(
                        actual,
                        snapshot["state_out__" + field],
                        rtol=3e-6,
                        atol=1e-5,
                        err_msg=name,
                    )

    def test_masked_publication_on_current_saved_inputs(self):
        """Match all original public/cache fields and preserve free6 until late."""
        for capture in local.captures():
            with self.subTest(capture=capture["path"]), np.load(capture["path"]) as snapshot:
                cohort = bind_cohort(snapshot, "cpu")
                classify(snapshot, cohort, "cpu")
                hot = expected_hot(snapshot)
                np.testing.assert_array_equal(cohort.early_owner.numpy(), hot)
                reference, candidate = bind(snapshot, capture, "cpu"), bind(snapshot, capture, "cpu")
                run(reference)
                publish_partition(candidate, cohort, True)
                masks = output_masks(snapshot, hot)
                for name in OUTPUT_KEYS:
                    actual = candidate.outputs[name].numpy()
                    np.testing.assert_array_equal(
                        actual[~masks[name]], candidate.seeds[name].numpy()[~masks[name]], err_msg=name
                    )
                publish_partition(candidate, cohort, False)
                assert_publication(self, candidate, reference)

    def test_late_clear_preserves_actual_hot_impulses(self):
        """Clear only late active rows after nonzero early impulses publish."""
        capture = next(local.captures())
        with np.load(capture["path"]) as snapshot:
            cohort = bind_cohort(snapshot, "cpu")
            classify(snapshot, cohort, "cpu")
            counts = wp.array(snapshot["solver__constraint_count"], dtype=int, device="cpu")
            values = wp.array(snapshot["solved__impulses"], dtype=float, device="cpu")
            wp.launch(
                early.get_publication_kernel("prepare_world_impulses", False),
                dim=counts.shape[0],
                inputs=[counts, values.shape[1], 0, values, cohort],
                device="cpu",
            )
            expected = snapshot["solved__impulses"].copy()
            active = np.arange(values.shape[1])[None, :] < snapshot["solver__constraint_count"][:, None]
            expected[active & ~expected_hot(snapshot)[:, None]] = 0.0
            np.testing.assert_array_equal(values.numpy(), expected)

    def test_native_argument_binding(self):
        """Bind original/early/late native ABIs without changing valid candidates."""
        capture = next(local.captures())
        with np.load(capture["path"]) as snapshot:
            fixture = native_fixture(snapshot, capture, "cpu")
            launch_calls(fixture.early_calls + fixture.late_calls, "cpu")
            for kernel, _dim, inputs in fixture.early_calls + fixture.late_calls:
                self.assertEqual(len(inputs), len(kernel.adj.args))
                self.assertGreaterEqual(int(inputs[0].numpy().min()), 0)

    @unittest.skipUnless(wp.is_cuda_available(), "Native GS and real stream graphs require CUDA")
    def test_actual_native_fork_and_two_graph_transitions(self):
        """Preserve current solve/publication under live early-cache overlap."""
        device = wp.get_device("cuda")
        for capture in local.captures():
            with self.subTest(capture=capture["path"]), np.load(capture["path"]) as snapshot:
                reference = native_fixture(snapshot, capture, device)
                candidate = native_fixture(snapshot, capture, device)
                run_original(reference)
                run_split(candidate)
                local.check_outputs(self, snapshot, candidate.solve, reference.solve)
                assert_publication(self, candidate.publication, reference.publication)
                with wp.ScopedCapture(device=device) as original_capture:
                    run_original(reference)
                with wp.ScopedCapture(device=device) as split_capture:
                    run_split(candidate)
                # These are admission-only withdrawals, not invented physical
                # contact/MF rows. All original physics inputs stay identical.
                seed_mf = snapshot["solver__mf_slot_counter"].copy()
                seed_slots = snapshot["solver__slot_counter"].copy()
                for mode in ("normal", "mf_withdraw", "forty_withdraw", "normal"):
                    changed_mf, changed_slots = seed_mf.copy(), seed_slots.copy()
                    if mode == "mf_withdraw":
                        changed_mf[:] = np.maximum(changed_mf, 1)
                    elif mode == "forty_withdraw":
                        changed_slots[::2] = 40
                    candidate.selection["mf"].assign(changed_mf)
                    candidate.selection["slots"].assign(changed_slots)
                    wp.capture_launch(original_capture.graph)
                    wp.capture_launch(split_capture.graph)
                    local.check_outputs(self, snapshot, candidate.solve, reference.solve)
                    assert_publication(self, candidate.publication, reference.publication)
                for negative in ("velocity_copy", "impulse_clear", "duplicate_solve"):
                    run_split(candidate, negative=negative)
                    with self.assertRaises(AssertionError, msg=negative):
                        local.check_outputs(self, snapshot, candidate.solve, reference.solve)


if __name__ == "__main__":
    unittest.main()
