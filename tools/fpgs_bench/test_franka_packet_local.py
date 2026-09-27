# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Loaded current-packet/held-factor local consumer and fallback controls.

These research fixtures retain their exact capture hashes. CPU tests exercise
the actual producer/fallback and bind all six native ABIs; native GS execution
and graph checks require CUDA. They do not certify rollout or sensor quality.
"""

import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import franka_contact_packet as contact
from newton._src.solvers.feather_pgs import franka_row_packets as rows
from newton._src.solvers.feather_pgs import solver_feather_pgs as original
from tools.fpgs_bench.test_franka_row_packets import CAPTURES, bind_prefix, bind_queue

PINS = (
    "062ec9bfe157c104241839e156be74e510b11263fd946232e3d41128ba11f85c",
    "da2823021101ad6cdaaac370dd94d81af22795a3290c22218c685b4177418f21",
    "4e4f33906f1d0570703bf6524e4c92af08062607f7c44fb7c1394aa6338e3126",
    "b12146b5e446cfdeb887d0402426f7520cc9be820123d88874bc75c393fa01b5",
)
OUTPUT_ALIASES = {
    "row_type": "kind",
    "row_parent": "parent",
    "row_mu": "mu",
    "row_beta": "row_beta",
    "row_cfm": "row_cfm",
    "phi": "phi",
    "target": "target",
    "restitution": "restitution",
    "rhs": "rhs",
    "diag": "diag",
}


def captures():
    """Reject a replaced snapshot, rather than accepting its newly recorded hash."""
    for gpu in (0, 1):
        audit = json.loads((CAPTURES / f"gpu{gpu}/audit.json").read_text())
        assert audit["complete"] and audit["success"] and audit["source_guard_pass"]
        assert len(audit["captures"]) == 2
        for index, capture in enumerate(audit["captures"]):
            expected = PINS[gpu * 2 + index]
            assert capture["sha256"] == expected
            with Path(capture["path"]).open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
            assert capture["mass_refresh"] == (index == 0)
            assert capture["solve_arguments"]["iterations"] == 8
            yield dict(capture, hardware=audit["hardware"])


def bind_current(snapshot, capture, device):
    """Bind production geometry inputs and alias its outputs to the real prefix."""
    prefix = bind_prefix(snapshot, capture["settings"]["solver"], device)
    prefix.dt = capture["dt"]
    packet = contact.allocate_packet(prefix.kind.shape[0], device, row_contact=prefix.row_contact)
    packet.jacobian.fill_(np.nan)
    total = int(snapshot["contacts__rigid_contact_count"][0])
    solver = SimpleNamespace(**capture["settings"]["solver"])
    solver.model = SimpleNamespace()
    state, augmented, contacts = SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
    aliases = {
        "body_mask": "body_response_dof_mask",
        "response_dofs": "articulation_response_dof_count",
        "dof_start": "articulation_dof_start",
        "origin": "articulation_origin",
        "prescribed": "_prescribed_articulation",
        "is_free": "is_free_rigid",
        "material_mu": "shape_material_mu",
        "material_restitution": "shape_material_restitution",
        "incident": "v_hat",
        "target": "target_velocity",
        "restitution": "row_restitution",
    }
    for name, variable in contact.ContactInput.vars.items():
        if not hasattr(variable.type, "dtype"):
            continue
        if name in ("count", "point0", "point1", "normal", "shape0", "shape1", "margin0", "margin1"):
            owner, key, field = contacts, "contacts__", "rigid_contact_" + name
        elif name in ("world", "slot", "art0", "art1", "path", "slots_needed"):
            field = "contact_" + {"art0": "art_a", "art1": "art_b"}.get(name, name)
            owner, key = solver, "solver__"
        elif name == "shape_body":
            owner, key, field = solver.model, "model__", name
        elif name == "body_q":
            owner, key, field = state, "state_in__", name
        elif name in ("body_v", "motion"):
            owner, key, field = augmented, "state_aug__", {"body_v": "body_v_s", "motion": "joint_S_s"}[name]
        else:
            owner, key, field = solver, "solver__", aliases.get(name, name)
        if name in OUTPUT_ALIASES:
            value = getattr(prefix, OUTPUT_ALIASES[name])
        else:
            value = snapshot[key + field]
            if (key == "contacts__" and name != "count") or field.startswith("contact_"):
                value = value[:total]
            value = wp.array(value, dtype=variable.type.dtype, device=device)
        setattr(owner, field, value)
    data = contact.bind(solver, state, augmented, contacts, capture["dt"], 1.0)
    return prefix, packet, data


def produce(prefix, packet, data, device):
    """Execute the two current producers, with no canonical grouped-J dependency."""
    wp.launch_tiled(
        rows.get_prefix_kernel(),
        dim=[(prefix.group_to_art.shape[0] + 3) // 4],
        inputs=[prefix],
        block_dim=128,
        device=device,
    )
    workers = max(1, min(data.point0.shape[0], 512))
    wp.launch_tiled(
        contact.produce_contacts, dim=[workers], inputs=[workers, data, packet], block_dim=32, device=device
    )


def grouped_jacobians(snapshot, prefix, packet):
    """Expand actual packet/prefix values into original physical group coordinates."""
    values, tokens = packet.jacobian.numpy(), packet.row_contact.numpy()
    dofs, weights = prefix.dofs.numpy(), prefix.weights.numpy()
    counts = snapshot["solver__constraint_count"]
    result = {}
    for size in (9, 6):
        matrix = np.zeros_like(snapshot[f"solver__J_by_size__{size}"])
        for group, art in enumerate(snapshot[f"solver__group_to_art__{size}"]):
            world = snapshot["solver__art_to_world"][art]
            for row in range(counts[world]):
                if tokens[world, row] >= 0:
                    start = 0 if size == 9 else 9
                    matrix[group, row] = values[world, row, start : start + size]
                elif size == 9:
                    for component in (0, 1):
                        dof = dofs[world, row, component]
                        if dof >= 0:
                            matrix[group, row, dof] += weights[world, row, component]
        result[size] = matrix
    return result


def bind_calls(snapshot, prefix, packet, data, device, *, private, matching_packet=True):
    """Bind all actual native signatures; MF fields are float3D, not vector2D."""

    def array(name, dtype=float):
        return wp.array(snapshot["solver__" + name], dtype=dtype, device=device)

    count, capacity = snapshot["solver__row_type"].shape
    outputs = [
        wp.clone(prefix.diag) if matching_packet else array("diag"),
        array("impulses"),
        array("v_hat"),
        array("mf_impulses"),
    ]
    if private:
        prefix.diag = data.diag = outputs[0]
    seed = [wp.clone(value) for value in outputs]
    jacobians = grouped_jacobians(snapshot, prefix, packet) if matching_packet else None
    groups = {}
    for size in (9, 6):
        values = jacobians[size] if matching_packet else snapshot[f"solver__J_by_size__{size}"]
        groups[size] = wp.array(np.full_like(values, np.nan) if private else values, dtype=float, device=device)
    owner = array("_local_solve_owner", int)
    common = [
        array("art_group_idx", int),
        array("art_to_world", int),
        array("articulation_dof_start", int),
        owner,
        array("constraint_count", int),
        array("L_by_size__9"),
        groups[9],
    ]
    row_inputs = [wp.clone(getattr(prefix, name)) for name in ("rhs", "kind", "parent", "mu")]
    if private:
        row_inputs = [prefix.rhs, prefix.kind, prefix.parent, prefix.mu]
    elif not matching_packet:
        row_inputs = [array("rhs"), array("row_type", int), array("row_parent", int), array("row_mu")]
    suffix = [
        *row_inputs,
        array("mf_constraint_count", int),
        array("mf_meta_packed", int),
        outputs[3],
        *[array(name) for name in ("mf_J_a", "mf_J_b", "mf_MiJt_a", "mf_MiJt_b", "mf_row_mu")],
        1.0,
        8,
        1.0,
        0,
        0,
    ]
    factory = rows.get_local_factory() if private else original._get_pgs_solve_local_owned_kernel
    calls = []
    for tier, max_rows, secondary, mf_rows in ((1, 9, 0, 0), (2, 20, 6, 0), (3, 40, 6, 12)):
        options = {
            "warps_per_block": 2,
            "lanes_per_world": 8 if tier == 1 else 32,
            "contact_capable": tier != 1,
            "dense_response_matrix": True,
        }
        if secondary:
            options.update(paired_dof_count=6, persistent_queue=True)
        if mf_rows:
            options.update(
                mf_max_constraints=snapshot["solver__mf_impulses"].shape[1], local_mf_max_constraints=mf_rows
            )
        kernel = factory(capacity, max_rows, 9, wp.get_device(device).arch or "cpu", **options)
        if tier == 1:
            candidates = array("_local_internal_candidates__9", int)
            inputs = [candidates, candidates]
        else:
            name = "pair" if tier == 2 else "residual"
            inputs = [
                array(f"_local_{name}_active_candidates__9", int),
                array(f"_local_{name}_active_secondaries__9", int),
                array(f"_local_{name}_active_counts__9", int),
                count,
            ]
        inputs += [tier, *common, array(f"L_by_size__{secondary or 9}"), groups[secondary or 9], *suffix, *outputs[:3]]
        if private:
            inputs += [prefix, packet]
        divisor = 8 if tier == 1 else 2
        calls.append((kernel, (count + divisor - 1) // divisor, inputs))
    return SimpleNamespace(
        calls=calls,
        outputs=outputs,
        seed=seed,
        owner=owner,
        groups=groups,
        prefix=prefix,
        packet=packet,
        data=data,
        private=private,
        device=device,
    )


def run(bundle):
    """Restore all solver outputs before every eager or captured complete solve."""
    for output, seed in zip(bundle.outputs, bundle.seed, strict=True):
        wp.copy(output, seed)
    if bundle.private:
        produce(bundle.prefix, bundle.packet, bundle.data, bundle.device)
    for kernel, dim, inputs in bundle.calls:
        wp.launch_tiled(kernel, dim=[dim], inputs=inputs, block_dim=64, device=bundle.device)


def check_outputs(test, snapshot, candidate, reference):
    """Check current rows, MF impulses, complete velocity and untouched tails."""
    counts = snapshot["solver__constraint_count"]
    dense = np.arange(snapshot["solver__row_type"].shape[1])[None, :] < counts[:, None]
    mf = np.arange(snapshot["solver__mf_impulses"].shape[1])[None, :] < snapshot["solver__mf_constraint_count"][:, None]
    for index, (actual, expected) in enumerate(zip(candidate.outputs, reference.outputs, strict=True)):
        a, b = actual.numpy(), expected.numpy()
        test.assertTrue(np.isfinite(a).all())
        np.testing.assert_allclose(a, b, rtol=3e-6, atol=3e-6, err_msg=f"output {index}")
        if index != 2:
            mask = mf if index == 3 else dense
            np.testing.assert_array_equal(a[~mask], candidate.seed[index].numpy()[~mask])


def physical_action(snapshot, jacobians, impulses, mf_impulses):
    """Reconstruct physical velocity and denominators independently in FP64.

    CFM belongs only in the denominator; it is not an extra force in this
    action. Current MF response is retained separately from held dense L.
    """
    names = (
        "v_hat",
        "constraint_count",
        "art_to_world",
        "articulation_dof_start",
        "row_cfm",
        "mf_constraint_count",
        "mf_dof_a",
        "mf_dof_b",
        "mf_MiJt_a",
        "mf_MiJt_b",
        "world_dof_indices",
        "group_to_art__9",
        "group_to_art__6",
        "L_by_size__9",
        "L_by_size__6",
    )
    source = {name: snapshot["solver__" + name] for name in names}
    velocity = source["v_hat"].astype(np.float64)
    diagonal = source["row_cfm"].astype(np.float64)
    for size in (9, 6):
        for group, art in enumerate(source[f"group_to_art__{size}"]):
            world = source["art_to_world"][art]
            count = source["constraint_count"][world]
            start = source["articulation_dof_start"][art]
            jacobian = jacobians[size][group, :count].astype(np.float64)
            factor = np.tril(source[f"L_by_size__{size}"][group]).astype(np.float64)
            response = np.linalg.solve(factor @ factor.T, jacobian.T).T
            velocity[start : start + size] += response.T @ impulses[world, :count]
            diagonal[world, :count] += np.einsum("ij,ij->i", jacobian, response)
    for world, count in enumerate(source["mf_constraint_count"]):
        for row in range(count):
            for side in ("a", "b"):
                offset = source["mf_dof_" + side][world, row]
                if offset >= 0:
                    indices = source["world_dof_indices"][world, offset : offset + 6]
                    assert np.all(indices >= 0)
                    velocity[indices] += source["mf_MiJt_" + side][world, row] * float(mf_impulses[world, row])
    return velocity, diagonal


@unittest.skipUnless(CAPTURES.exists(), "Pinned local Franka captures are unavailable")
class TestFrankaPacketLocal(unittest.TestCase):
    def test_saved_physical_action_reference(self):
        """The independent held-dense/current-MF action agrees with all four saved solves."""
        for capture in captures():
            with self.subTest(source=capture["sha256"][:8]), np.load(capture["path"]) as snapshot:
                velocity, diagonal = physical_action(
                    snapshot,
                    {size: snapshot[f"solver__J_by_size__{size}"] for size in (9, 6)},
                    snapshot["solved__impulses"],
                    snapshot["solved__mf_impulses"],
                )
                active = np.arange(diagonal.shape[1])[None, :] < snapshot["solver__constraint_count"][:, None]
                np.testing.assert_allclose(velocity, snapshot["solved__v_out"], rtol=3e-6, atol=3e-6)
                np.testing.assert_allclose(diagonal[active], snapshot["solved__diag"][active], rtol=3e-6, atol=3e-6)

    def test_all_native_cpu_bindings(self):
        """Build all six original/private native ABIs without claiming CPU GS execution."""
        capture = next(captures())
        with np.load(capture["path"]) as snapshot:
            prefix, packet, data = bind_current(snapshot, capture, "cpu")
            produce(prefix, packet, data, "cpu")
            for private in (False, True):
                bundle = bind_calls(snapshot, prefix, packet, data, "cpu", private=private)
                run(bundle)
                self.assertEqual(len(bundle.calls), 3)
                for kernel, _, inputs in bundle.calls:
                    self.assertEqual(len(inputs), len(kernel.adj.args))

    def test_general_fallback_current_action_all_captures(self):
        """Force every actual local world through complete current-J/held-L fallback."""
        for capture in captures():
            with (
                self.subTest(step=capture["solver_step"], source=capture["sha256"][:8]),
                np.load(capture["path"]) as snapshot,
            ):
                prefix, packet, data = bind_current(snapshot, capture, "cpu")
                produce(prefix, packet, data, "cpu")
                expected_j = grouped_jacobians(snapshot, prefix, packet)
                queue = bind_queue(snapshot)
                worlds, capacity = prefix.kind.shape
                wp.copy(queue.count, wp.array(snapshot["solver__constraint_count"], dtype=int, device="cpu"))
                queue.owner.zero_()
                wp.copy(queue.general, wp.array(np.arange(worlds), dtype=int, device="cpu"))
                queue.general_count.fill_(worlds)
                world_j = wp.full((worlds, capacity, 15), -77.0, dtype=float, device="cpu")
                world_y = wp.full_like(world_j, -77.0)
                expected_diag = np.zeros((worlds, capacity), dtype=np.float64)

                def a(name, dtype=int):
                    return wp.array(snapshot["solver__" + name], dtype=dtype, device="cpu")

                for size in (9, 6):
                    jacobian = wp.full_like(a(f"J_by_size__{size}", float), -77.0)
                    response = wp.full_like(jacobian, -77.0)
                    start, arts = (
                        a(f"world_response_group_art_start__{size}"),
                        a(f"world_response_group_to_art__{size}"),
                    )
                    mapping = a("art_group_idx")
                    wp.launch_tiled(
                        rows.materialize_general_prefix,
                        dim=[32],
                        inputs=[32, prefix, queue, start, arts, mapping, size, jacobian],
                        block_dim=32,
                        device="cpu",
                    )
                    wp.launch_tiled(
                        contact.materialize_fallback,
                        dim=[32],
                        inputs=[32, data, queue.owner, size, mapping, jacobian],
                        block_dim=32,
                        device="cpu",
                    )
                    wp.launch_tiled(
                        rows.general_response,
                        dim=[32],
                        inputs=[
                            32,
                            queue,
                            start,
                            arts,
                            mapping,
                            a("articulation_world_dof_offset"),
                            size,
                            a(f"L_by_size__{size}", float),
                            jacobian,
                            response,
                            world_j,
                            world_y,
                        ],
                        block_dim=32,
                        device="cpu",
                    )
                    actual_j, actual_y = jacobian.numpy(), response.numpy()
                    published_j, published_y = world_j.numpy(), world_y.numpy()
                    for group, art in enumerate(snapshot[f"solver__group_to_art__{size}"]):
                        world = snapshot["solver__art_to_world"][art]
                        count = snapshot["solver__constraint_count"][world]
                        j = expected_j[size][group, :count].astype(np.float64)
                        factor = np.tril(snapshot[f"solver__L_by_size__{size}"][group]).astype(np.float64)
                        h = factor @ factor.T
                        y = np.linalg.solve(h, j.T).T
                        np.testing.assert_array_equal(actual_j[group, :count], j)
                        np.testing.assert_allclose(actual_y[group, :count], y, rtol=3e-6, atol=3e-6)
                        np.testing.assert_array_equal(actual_j[group, count:], -77)
                        np.testing.assert_array_equal(actual_y[group, count:], -77)
                        offset = snapshot["solver__articulation_world_dof_offset"][art]
                        np.testing.assert_array_equal(published_j[world, :count, offset : offset + size], j)
                        np.testing.assert_array_equal(
                            published_y[world, :count, offset : offset + size], actual_y[group, :count]
                        )
                        expected_diag[world, :count] += np.einsum("ij,ij->i", j, y)
                wp.launch_tiled(
                    rows.general_diag,
                    dim=[32],
                    inputs=[32, prefix, queue, a("world_dof_count"), world_j, world_y],
                    block_dim=32,
                    device="cpu",
                )
                active = np.arange(capacity)[None, :] < snapshot["solver__constraint_count"][:, None]
                expected_diag += prefix.row_cfm.numpy()
                np.testing.assert_allclose(prefix.diag.numpy()[active], expected_diag[active], rtol=3e-6, atol=3e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "Actual native local GS needs CUDA")
    def test_current_packet_complete_solve_and_graph_transitions(self):
        """Current geometry, held factors, owner withdrawal/re-admission and two graphs."""
        device = wp.get_device("cuda:0")
        for capture in captures():
            with (
                self.subTest(step=capture["solver_step"], source=capture["sha256"][:8]),
                np.load(capture["path"]) as snapshot,
            ):
                prefix, packet, data = bind_current(snapshot, capture, device)
                produce(prefix, packet, data, device)
                baseline = bind_calls(snapshot, prefix, packet, data, device, private=False)
                candidate = bind_calls(snapshot, prefix, packet, data, device, private=True)
                run(baseline)
                run(candidate)
                check_outputs(self, snapshot, candidate, baseline)
                velocity, diagonal = physical_action(
                    snapshot,
                    grouped_jacobians(snapshot, prefix, packet),
                    candidate.outputs[1].numpy(),
                    candidate.outputs[3].numpy(),
                )
                active = np.arange(diagonal.shape[1])[None, :] < snapshot["solver__constraint_count"][:, None]
                np.testing.assert_allclose(candidate.outputs[2].numpy(), velocity, rtol=3e-6, atol=3e-6)
                np.testing.assert_allclose(candidate.outputs[0].numpy()[active], diagonal[active], rtol=3e-6, atol=3e-6)
                wp.synchronize_device(device)
                for bundle in (baseline, candidate):
                    with wp.ScopedCapture(device=device) as capture_graph:
                        run(bundle)
                    for _ in range(2):
                        wp.capture_launch(capture_graph.graph)
                        check_outputs(self, snapshot, candidate, baseline)
                    bundle.owner.zero_()
                    wp.capture_launch(capture_graph.graph)
                    for output, seed in zip(bundle.outputs, bundle.seed, strict=True):
                        if bundle.private and output is bundle.outputs[0]:
                            continue  # Current producer still owns CFM seeds; no GS executes.
                        np.testing.assert_array_equal(output.numpy(), seed.numpy())
                    wp.copy(bundle.owner, wp.array(snapshot["solver___local_solve_owner"], dtype=int, device=device))
                    wp.capture_launch(capture_graph.graph)
                    check_outputs(self, snapshot, candidate, baseline)
                for matrix in candidate.groups.values():
                    self.assertTrue(np.isnan(matrix.numpy()).all())

    @unittest.skipUnless(wp.is_cuda_available(), "Actual native local GS needs CUDA")
    def test_recorded_original_complete_solve(self):
        """The unchanged original reproduces complete outputs on the source GPU only."""
        device = wp.get_device("cuda:0")
        matched = 0
        for capture in captures():
            if capture["hardware"]["uuid"] != device.uuid:
                continue
            matched += 1
            with self.subTest(source=capture["sha256"][:8]), np.load(capture["path"]) as snapshot:
                prefix, packet, data = bind_current(snapshot, capture, device)
                produce(prefix, packet, data, device)
                bundle = bind_calls(snapshot, prefix, packet, data, device, private=False, matching_packet=False)
                run(bundle)
                for value, name in zip(bundle.outputs, ("diag", "impulses", "v_out", "mf_impulses"), strict=True):
                    np.testing.assert_allclose(
                        value.numpy(),
                        snapshot["solved__" + name],
                        rtol=3e-6,
                        atol=3e-6,
                        err_msg=f"recorded original {name}",
                    )
        self.assertEqual(matched, 2, "Saved-original fidelity requires one of the two source GPUs")


if __name__ == "__main__":
    unittest.main()
