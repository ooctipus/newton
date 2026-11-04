# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise MF metadata: parent indices beyond the signed-16-bit boundary and the pack launch policy."""

import dataclasses
import inspect
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import response_builds_rows, sweep_routes, sweep_routes_local
from newton._src.solvers.feather_pgs import solver_feather_pgs as fpgs
from newton._src.solvers.feather_pgs.placement import Placement
from newton.tests.feather_pgs_law_harness import row_lines
from newton.tests.feather_pgs_paired_fixture import solve_fixture
from newton.tests.test_feather_pgs_native_prelude import _native_snippets


def _build_pendulums(device, worlds=5):
    """Fixed-base double pendulums whose tips touch the ground through the collision pipeline."""
    builder = newton.ModelBuilder()
    shape_cfg = builder.default_shape_cfg.copy()
    shape_cfg.density = 0.0
    shape_cfg.mu = 0.5
    for world in range(worlds):
        builder.begin_world()
        parent = -1
        joints = []
        for link in range(2):
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.05), lock_inertia=True)
            joints.append(
                builder.add_joint_revolute(
                    parent=parent,
                    child=body,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.42 if link == 0 else -0.2), wp.quat_identity()),
                    target_ke=5.0 * world,
                    target_kd=1.0,
                )
            )
            parent = body
        builder.add_shape_sphere(
            parent, radius=0.03, xform=wp.transform(wp.vec3(0.0, 0.0, -0.2), wp.quat_identity()), cfg=shape_cfg
        )
        builder.add_articulation(joints)
        builder.end_world()
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _parent_decoder(source, *, name="parent", packed="packed_tp"):
    """Compile the actual generated GS parent expression on the CPU: the first one that decodes ``packed``."""
    expression = (
        re.search(rf"\b{name}\s*=\s*([^;]*{re.escape(packed)}[^;]*);", source).group(1).replace(packed, "value")
    )

    @wp.func_native(f"return {expression};")
    def decode(value: int) -> int: ...

    @wp.kernel(module="unique", enable_backward=False)
    def decode_rows(packed: wp.array[int], parents: wp.array[int]):
        index = wp.tid()
        parents[index] = decode(packed[index])

    return decode_rows


def _metadata_packer(source):
    """Compile the production packed-word expression without CUDA-only row setup."""
    expression = re.search(r"packed.w = ([^;]+);", source).group(1)

    @wp.func_native(f"return {expression};")
    def pack(rt: int, par: int) -> int: ...

    @wp.kernel(module="unique", enable_backward=False)
    def pack_rows(parents: wp.array[int], packed: wp.array[int]):
        index = wp.tid()
        packed[index] = pack(2, parents[index])

    return pack_rows


def _pack_state(has_free_rigid_bodies):
    """Stand in for a solver with opaque metadata-pack operands."""
    state = SimpleNamespace(
        pack_mf_meta_kernel=object(),
        mf_meta_lanes=32,
        model=SimpleNamespace(device="mock-device"),
        forest=SimpleNamespace(
            has_free_rigid_bodies=has_free_rigid_bodies, world_count=7, articulation_world_dof_offset=object()
        ),
    )
    for name in (
        "dense_first",
        "mf_constraint_count",
        "row_sides",
        "diag",
        "rhs",
        "row_incident",
        "row_type",
        "row_parent",
        "mf_meta_packed",
    ):
        setattr(state, name, object())
    return state


class TestFeatherPGSMFMetadata(unittest.TestCase):
    def test_pack_launch_requires_free_rigid_bodies(self):
        """Launch the metadata pack only when free rigid bodies can produce matrix-free rows."""
        for has_free in (False, True):
            state = _pack_state(has_free)
            rhs = state.rhs
            with (
                self.subTest(has_free_rigid_bodies=has_free),
                patch.object(fpgs.wp, "launch_tiled") as launch,
            ):
                Placement.pack_free_body_meta(state)
                if not has_free:
                    launch.assert_not_called()
                else:
                    # The launch contract, not its operand layout: the pack kernel packs this RHS into the metadata.
                    launch.assert_called_once()
                    args, kwargs = launch.call_args
                    self.assertIs(args[0], state.pack_mf_meta_kernel)
                    self.assertTrue(any(operand is rhs for operand in kwargs["inputs"]))
                    self.assertEqual(kwargs["outputs"], [state.mf_meta_packed])

    def test_pack_preserves_parent_bits_and_no_parent(self):
        """Preserve the existing word layout using defined unsigned shift arithmetic."""
        parents = np.array([-1, 0, 32767, 32768, 37247, 65534], np.int32)
        expected = (((parents.astype(np.int64) & 65535) << 16) | 2).astype(np.uint32).view(np.int32)
        kernel = response_builds_rows.free_body_meta_kernel(65535, 8 + 65535, 8, 32, "cpu")
        source = max(_native_snippets(kernel), key=len)
        result = wp.zeros(len(parents), dtype=int, device="cpu")
        wp.launch(
            _metadata_packer(source),
            dim=len(parents),
            inputs=[wp.array(parents, dtype=int, device="cpu")],
            outputs=[result],
            device="cpu",
        )
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_generated_parent_decode_preserves_unsigned_range_and_sentinel(self):
        """Decode actual native GS expressions at signed and unsigned boundaries."""
        parents = np.array([-1, 0, 32767, 32768, 37247, 65534], np.int32)
        packed = (((parents.astype(np.int64) & 65535) << 16) | 2).astype(np.uint32).view(np.int32)
        for stream in (False, True):
            with self.subTest(stream_mf_impulses=stream):
                # The generated text (no device: at this row capacity the staged impulses fit none).
                source, _ = sweep_routes.mf_gs_source(
                    4,
                    65535,
                    6,
                    has_dense_velocity_limit_rows=False,
                    shared_metadata=False,
                    stream_mf_impulses=stream,
                )
                result = wp.zeros(len(parents), dtype=int, device="cpu")
                wp.launch(
                    _parent_decoder(source),
                    dim=len(parents),
                    inputs=[wp.array(packed, dtype=int, device="cpu")],
                    outputs=[result],
                    device="cpu",
                )
                np.testing.assert_array_equal(result.numpy(), parents)

    def test_other_generated_solve_paths_decode_high_parents(self):
        """Decode high parents in local-pair consumers too."""
        parents = np.array([-1, 0, 32767, 32768, 37247, 65534], np.int32)
        packed = (((parents.astype(np.int64) & 65535) << 16) | 2).astype(np.uint32).view(np.int32)
        kernels = (
            (
                sweep_routes_local.local_owned_kernel(
                    8,
                    8,
                    23,
                    "cpu",
                    paired_dof_count=6,
                    mf_max_constraints=65535,
                    local_mf_max_constraints=3,
                ),
                "sweep_local_native",
                "parent",
                "packed_tp",
            ),
        )
        for kernel, native_name, name, word in kernels:
            with self.subTest(kernel=kernel.key):
                closure = inspect.getclosurevars(kernel.func)
                source = (closure.globals | closure.nonlocals)[native_name].native_snippet
                result = wp.zeros(len(parents), dtype=int, device="cpu")
                wp.launch(
                    _parent_decoder(source, name=name, packed=word),
                    dim=len(parents),
                    inputs=[wp.array(packed, dtype=int, device="cpu")],
                    outputs=[result],
                    device="cpu",
                )
                np.testing.assert_array_equal(result.numpy(), parents)

    def test_capacity_guard_does_not_alias_the_parent_sentinel(self):
        """Reject unsupported packed capacity before allocating solver row buffers."""
        for capacity in (65535, 0):
            response_builds_rows.free_body_meta_kernel(capacity, 8 + capacity, 8, 32, "cpu")
        for capacity in (65536, 100000):
            with self.subTest(capacity=capacity):
                with self.assertRaisesRegex(ValueError, "mf_max_constraints.*65535"):
                    response_builds_rows.free_body_meta_kernel(capacity, 8 + capacity, 8, 32, "cpu")
                # Only the CUDA sweeps pack the row metadata.
                if wp.is_cuda_available():
                    with self.assertRaisesRegex(ValueError, "mf_max_constraints.*65535"):
                        fpgs.SolverFeatherPGS(
                            newton.ModelBuilder().finalize(device="cuda:0"),
                            mf_max_constraints=capacity,
                        )

    def test_patch_links_decode_high_normal_rows(self):
        """Use the same unsigned layout when following normal-load patch links."""
        parents = np.array([-1, 0, 32767, 32768, 37247, 65534], np.int32)
        packed = ((parents.astype(np.int64) & 65535) << 16).astype(np.uint32).view(np.int32)
        local = sweep_routes_local.local_owned_kernel(
            8, 8, 23, "cpu", paired_dof_count=6, mf_max_constraints=65535, local_mf_max_constraints=3
        )
        closure = inspect.getclosurevars(local.func)
        cases = (
            # The matrix-free sweep's generated text (no device: its staged impulses at this capacity fit none).
            ("sweep_mf_gs", sweep_routes.mf_gs_source(8, 65535, 29)[0]),
            (local.key, (closure.globals | closure.nonlocals)["sweep_local_native"].native_snippet),
        )
        word = "mf_meta.data[off_meta + patch_row * 4 + 3]"
        for name, source in cases:
            with self.subTest(kernel=name):
                # Dense-row links use a different, unpacked array. The ring walk's parent accessor returns the
                # decoded link of its row.
                start = source.index(word)
                start = source.rfind("return ", 0, start)
                result = wp.zeros(len(parents), dtype=int, device="cpu")
                wp.launch(
                    _parent_decoder(
                        "int patch_row = " + source[start + len("return ") :], name="patch_row", packed=word
                    ),
                    dim=len(parents),
                    inputs=[wp.array(packed, dtype=int, device="cpu")],
                    outputs=[result],
                    device="cpu",
                )
                np.testing.assert_array_equal(result.numpy(), parents)


def _unit_first_art(worlds, dofs, device):
    """The walk's tree lookup (:attr:`.factor.walk.WalkTrees.first_art`) the sweeps read for each world's free body at
    its first coordinate (body ``w`` at global coordinate ``w * dofs``, six coordinates, at world coordinate 0)."""
    first_art = np.full(worlds * dofs, -1, np.int32)
    first_art[np.arange(worlds) * dofs] = np.arange(worlds)
    return wp.array(first_art, dtype=wp.int32, device=device)


def _solve_parent_rows(device, parents):
    """Place identical active contact triples at different MF storage offsets."""
    _, values = solve_fixture(device)
    worlds, capacity = len(parents), max(parents) + 3
    assert worlds == 5
    metadata = np.zeros((worlds, capacity * 4), np.int32)
    jacobian = np.zeros((worlds, capacity, 6), np.float32)
    velocity = np.zeros((worlds, 29), np.float32)
    for world, parent in enumerate(parents):
        velocity[world, 1:3] = np.array([-2.0, -3.0]) * (1 + 0.1 * world)
        for axis in range(3):
            row = parent + axis
            kind = 0 if axis == 0 else 2
            normal = -1 if axis == 0 else parent
            packed_parent = np.uint32(((normal & 65535) << 16) | kind).view(np.int32)
            metadata[world, 4 * row : 4 * row + 4] = (
                65535,
                np.float32(1).view(np.int32),
                # The word's right-hand side holds the row's incident velocity: its unit Jacobian on the body.
                np.float32((-(1 + 0.1 * world) if axis == 0 else 0) + velocity[world, axis]).view(np.int32),
                packed_parent,
            )
            jacobian[world, row, axis] = 1.0
    values.update(
        world_constraint_count=wp.zeros(worlds, dtype=int, device=device),
        dense_first=row_lines(device),
        mf_constraint_count=wp.array(np.asarray(parents) + 3, dtype=int, device=device),
        mf_contact_rows_end=wp.array(np.asarray(parents) + 3, dtype=int, device=device),
        mf_meta=wp.array(metadata, dtype=int, device=device),
        # The row store's lines: 8 dense rows, then the free bodies'.
        world_impulses=wp.zeros((worlds, 8 + capacity), device=device),
        world_row_mu=wp.full((worlds, 8 + capacity), 0.4, device=device),
        world_row_w=wp.ones((worlds, 8 + capacity), device=device),
        mf_Z=wp.array(np.concatenate([jacobian, np.zeros_like(jacobian)], axis=-1), device=device),
        v_out=wp.array(velocity.reshape(-1), device=device),
        general_world_grid_stride=worlds,
        use_general_world_queue=0,
        iterations=8,
        omega=1.0,
        regularize=0,
    )
    values["walk"].first_art = _unit_first_art(worlds, 29, device)
    kernel = sweep_routes.mf_gs_kernel(
        8,
        capacity,
        29,
        str(device.arch),
        has_dense_velocity_limit_rows=False,
        shared_metadata=False,
        stream_mf_impulses=True,
    )
    wp.launch_tiled(
        kernel,
        dim=[worlds],
        block_dim=32,
        device=device,
        inputs=[values[name] for name in inspect.signature(kernel.func).parameters],
    )
    impulses = values["world_impulses"].numpy()
    # The velocity change the sweep leaves in the walk's coordinates (the step's one leave walks it down).
    return values["du"].numpy(), np.asarray([impulses[w, 8 + p : 8 + p + 3] for w, p in enumerate(parents)])


@unittest.skipUnless(wp.is_cuda_available(), "MF-GS native solve requires CUDA")
class TestFeatherPGSMFMetadataCUDA(unittest.TestCase):
    def test_active_high_parents_match_rebased_contact_solve(self):
        """Solve high-parent contact triples with the supported streaming projection."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("Select a CUDA device for native MF-GS execution")
        low = _solve_parent_rows(device, [0] * 5)
        high = _solve_parent_rows(device, [0, 32767, 32768, 37247, 65532])
        for expected, actual in zip(low, high, strict=True):
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.all(high[1][:, 0] > 0))
        self.assertTrue(np.all(np.linalg.norm(high[1][:, 1:], axis=1) > 0))
        self.assertTrue(np.all(np.linalg.norm(high[1][:, 1:], axis=1) <= 0.4 * high[1][:, 0] + 1e-6))


def _record_pack_launches(solver):
    """Record the RHS operand of each metadata-pack launch while still launching it."""
    launch_tiled = wp.launch_tiled
    packed = []

    def record(kernel, *args, **kwargs):
        if kernel is solver.placement.pack_mf_meta_kernel:
            packed.append(kwargs["inputs"][5])
        return launch_tiled(kernel, *args, **kwargs)

    return packed, patch.object(fpgs.wp, "launch_tiled", record)


def _launch_pack_unconditionally(solver):
    """Restore the launch that ran before the skip, whether or not matrix-free rows exist.

    The solver builds the pack kernel only where it launches, so the arm builds it here.
    """
    pack = Placement.pack_free_body_meta
    solver.placement.pack_mf_meta_kernel = response_builds_rows.free_body_meta_kernel(
        solver.placement.mf_max_constraints,
        solver.placement.row_pitch,
        solver.placement.body_base,
        solver.placement.mf_meta_lanes,
        solver.model.device.arch,
    )

    def unconditional():
        # The production launch, with the skip's only gate held open for the call.
        placement = solver.placement
        with patch.object(placement, "forest", dataclasses.replace(placement.forest, has_free_rigid_bodies=True)):
            pack(placement)

    solver.placement.pack_free_body_meta = unconditional


def _step_matrix_free(model, solver, *, steps, joint_qd=0.0, dt=1.0 / 240.0):
    """Step through the collision pipeline; return the pack launches and the final values."""
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    states = [model.state() for _ in range(2)]
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    states[0].joint_qd.fill_(joint_qd)
    control = model.control()
    packed, recording = _record_pack_launches(solver)
    # The pendulum tips touch the ground intermittently, so the final step alone may carry no
    # contact impulse; the per-step peak shows the solve exercised contact rows during the run.
    impulse_peak = []
    with recording:
        for _ in range(steps):
            pipeline.collide(states[0], contacts)
            states[0].clear_forces()
            solver.step(states[0], states[1], control, contacts, dt)
            states.reverse()
            impulse_peak.append(np.abs(solver.placement.impulses.numpy()).max())
    values = {name: getattr(states[0], name).numpy() for name in ("joint_q", "joint_qd", "body_q", "body_qd")}
    values["impulse_peak"] = np.array(impulse_peak)
    for name in ("impulses", "mf_constraint_count", "mf_meta_packed"):
        values[name] = getattr(solver.placement, name).numpy()
    values["v_out"] = solver.v_out.numpy()
    values["contact_count"] = contacts.rigid_contact_count.numpy()
    return packed, values


@unittest.skipUnless(wp.is_cuda_available(), "Matrix-free stepping requires CUDA")
class TestFeatherPGSMFMetadataPackCUDA(unittest.TestCase):
    def test_no_free_rigid_bodies_skip_pack_and_keep_steps(self):
        """Skip the pack launch without free rigid bodies and match the unskipped steps bit for bit."""
        model = _build_pendulums(wp.get_device("cuda:0"))
        steps = 16
        runs = []
        for unconditional in (False, True):
            solver = fpgs.SolverFeatherPGS(model, pgs_iterations=6)
            self.assertFalse(solver.forest.has_free_rigid_bodies)
            # The pack kernel is built only where it launches.
            self.assertIsNone(solver.placement.pack_mf_meta_kernel)
            if unconditional:
                _launch_pack_unconditionally(solver)
            runs.append((solver, *_step_matrix_free(model, solver, steps=steps, joint_qd=0.2)))
        (skipping, skipped_packs, skipped), (legacy, legacy_packs, unskipped) = runs
        self.assertEqual(skipped_packs, [])
        # The route record says what the step did. The legacy arm's pack is replaced on the
        # instance, so only the skipping arm's record describes its launches.
        pack = skipping.get_path_selection()["routes"]["sweep"]["mf_meta_pack"]
        self.assertFalse(pack["launched"])
        self.assertIsNone(pack["kernel"])
        # The legacy arm proves the call site ran on every step.
        self.assertEqual(len(legacy_packs), steps)
        for actual in legacy_packs:
            self.assertIs(actual, legacy.placement.rhs)
        self.assertTrue(np.all(skipped["contact_count"] > 0))
        self.assertTrue(np.any(skipped["impulse_peak"] > 0.0))
        np.testing.assert_array_equal(skipped["mf_constraint_count"], 0)
        np.testing.assert_array_equal(skipped["mf_meta_packed"], 0)
        for name, value in skipped.items():
            np.testing.assert_array_equal(value, unskipped[name], err_msg=name)

    def test_free_rigid_bodies_pack_every_step(self):
        """Pack the problem on every step when free rigid bodies exist."""
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.01
        builder.add_ground_plane()
        box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()))
        builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
        model = builder.finalize(device=wp.get_device("cuda:0"))
        solver = fpgs.SolverFeatherPGS(model, pgs_iterations=6)
        self.assertTrue(solver.forest.has_free_rigid_bodies)
        steps = 4
        packed, values = _step_matrix_free(model, solver, steps=steps)
        self.assertGreater(int(values["mf_constraint_count"].max()), 0)
        self.assertEqual(len(packed), steps)
        pack = solver.get_path_selection()["routes"]["sweep"]["mf_meta_pack"]
        self.assertTrue(pack["launched"])
        self.assertEqual(pack["kernel"], solver.placement.pack_mf_meta_kernel.key)
        for rhs in packed:
            self.assertIs(rhs, solver.placement.rhs)


if __name__ == "__main__":
    unittest.main()
