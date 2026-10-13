# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise MF parent indices beyond the signed-16-bit boundary."""

import inspect
import re
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import solver_feather_pgs as fpgs
from newton.tests.test_feather_pgs_independent_components import (
    bind_components,
    component_fixture,
    launch_components,
    solve_fixture,
)


def _parent_decoder(source, *, name="mf_par", packed="packed_tp"):
    """Compile the actual generated GS parent expression on the CPU."""
    expression = re.search(rf"int {name}\s*=\s*([^;]+);", source).group(1).replace(packed, "value")

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


class TestFeatherPGSMFMetadata(unittest.TestCase):
    def test_pack_preserves_parent_bits_and_no_parent(self):
        """Preserve the existing word layout using defined unsigned shift arithmetic."""
        parents = np.array([-1, 0, 32767, 32768, 37247, 65534], np.int32)
        expected = (((parents.astype(np.int64) & 65535) << 16) | 2).astype(np.uint32).view(np.int32)
        kernel = fpgs._get_pack_mf_meta_kernel(65535, "cpu")
        source = inspect.getclosurevars(kernel.func).nonlocals["pack_mf_meta_native"].native_snippet
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
        for friction in ("current", "bisection", "bisection_desaxce", "coulomb_newton"):
            with self.subTest(friction=friction):
                kernel = fpgs._get_pgs_solve_mf_gs_kernel(
                    4,
                    65535,
                    6,
                    "cpu",
                    friction_mode=friction,
                    has_drive_rows=False,
                    has_dense_velocity_limit_rows=False,
                    shared_metadata=False,
                    stream_mf_impulses=friction == "current",
                )
                native = inspect.getclosurevars(kernel.func).nonlocals["pgs_solve_mf_gs_native"]
                result = wp.zeros(len(parents), dtype=int, device="cpu")
                wp.launch(
                    _parent_decoder(native.native_snippet),
                    dim=len(parents),
                    inputs=[wp.array(packed, dtype=int, device="cpu")],
                    outputs=[result],
                    device="cpu",
                )
                np.testing.assert_array_equal(result.numpy(), parents)

    def test_other_generated_solve_paths_decode_high_parents(self):
        """Decode high parents in propagation-full and local-pair consumers too."""
        parents = np.array([-1, 0, 32767, 32768, 37247, 65534], np.int32)
        packed = (((parents.astype(np.int64) & 65535) << 16) | 2).astype(np.uint32).view(np.int32)
        kernels = (
            (
                fpgs._get_pgs_solve_propagation_full_iteration_kernel(8, 65535, 29, 64, 8, 6, "cpu"),
                "pgs_solve_propagation_full_iteration_native",
                "mf_par",
                "packed_tp",
            ),
            (
                fpgs._get_pgs_solve_local_owned_kernel(
                    8,
                    8,
                    23,
                    "cpu",
                    paired_dof_count=6,
                    mf_max_constraints=65535,
                    local_mf_max_constraints=3,
                ),
                "native",
                "parent_mf",
                "packed_type_parent",
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
            fpgs._get_pack_mf_meta_kernel(capacity, "cpu")
        for capacity in (65536, 100000):
            with self.subTest(capacity=capacity):
                with self.assertRaisesRegex(ValueError, "mf_max_constraints.*65535"):
                    fpgs._get_pack_mf_meta_kernel(capacity, "cpu")
                with self.assertRaisesRegex(ValueError, "mf_max_constraints.*65535"):
                    fpgs.SolverFeatherPGS(
                        newton.ModelBuilder().finalize(device="cpu"),
                        pgs_mode="matrix_free",
                        mf_max_constraints=capacity,
                    )

    def test_patch_links_decode_high_normal_rows(self):
        """Use the same unsigned layout when following normal-load patch links."""
        parents = np.array([-1, 0, 32767, 32768, 37247, 65534], np.int32)
        packed = ((parents.astype(np.int64) & 65535) << 16).astype(np.uint32).view(np.int32)
        cases = (
            (
                fpgs._get_pgs_solve_mf_gs_kernel(8, 65535, 29, "cpu"),
                "pgs_solve_mf_gs_native",
                "mf_meta.data[off_meta + parent * 4 + 3]",
            ),
            (
                fpgs._get_pgs_solve_propagation_full_iteration_kernel(8, 65535, 29, 64, 8, 6, "cpu"),
                "pgs_solve_propagation_full_iteration_native",
                "mf_meta.data[off_meta + mf_par * 4 + 3]",
            ),
            (
                fpgs._get_pgs_solve_local_owned_kernel(
                    8, 8, 23, "cpu", paired_dof_count=6, mf_max_constraints=65535, local_mf_max_constraints=3
                ),
                "native",
                "mf_meta.data[mf_meta_offset + parent_mf * 4 + 3]",
            ),
        )
        for kernel, native_name, word in cases:
            with self.subTest(kernel=kernel.key):
                closure = inspect.getclosurevars(kernel.func)
                source = (closure.globals | closure.nonlocals)[native_name].native_snippet
                # Dense-row links use a different, unpacked array.
                start = source.index(word)
                start = source.rfind("int patch_row", 0, start)
                result = wp.zeros(len(parents), dtype=int, device="cpu")
                wp.launch(
                    _parent_decoder(source[start:], name="patch_row", packed=word),
                    dim=len(parents),
                    inputs=[wp.array(packed, dtype=int, device="cpu")],
                    outputs=[result],
                    device="cpu",
                )
                np.testing.assert_array_equal(result.numpy(), parents)

    def test_independent_component_admission_reads_high_parent_rows(self):
        """Admit complete valid contact triples above row32767 without truncation."""
        host = component_fixture(worlds=1, mf_capacity=37248)
        host["mf_counts"][0] = 37248
        data = bind_components(host, "cpu")
        launch_components(data, "cpu")
        np.testing.assert_array_equal(data.selector.numpy(), [-1])


def _solve_parent_rows(device, friction, parents):
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
                np.float32(-(1 + 0.1 * world) if axis == 0 else 0).view(np.int32),
                packed_parent,
            )
            jacobian[world, row, axis] = 1.0
    values.update(
        world_constraint_count=wp.zeros(worlds, dtype=int, device=device),
        mf_constraint_count=wp.array(np.asarray(parents) + 3, dtype=int, device=device),
        mf_contact_rows_end=wp.array(np.asarray(parents) + 3, dtype=int, device=device),
        mf_meta=wp.array(metadata, dtype=int, device=device),
        mf_impulses=wp.zeros((worlds, capacity), device=device),
        mf_row_mu=wp.full((worlds, capacity), 0.4, device=device),
        mf_row_w=wp.ones((worlds, capacity), device=device),
        mf_J_a=wp.array(jacobian, device=device),
        mf_MiJt_a=wp.array(jacobian, device=device),
        mf_J_b=wp.zeros((worlds, capacity, 6), device=device),
        mf_MiJt_b=wp.zeros((worlds, capacity, 6), device=device),
        v_out=wp.array(velocity.reshape(-1), device=device),
        general_world_grid_stride=worlds,
        use_general_world_queue=0,
        iterations=8,
        omega=1.0,
        regularize=0,
        row_phase=0,
        friction_start_iteration=0,
        iteration_offset=0,
        freeze_drive_rows=0,
        defer_dense_response=0,
    )
    kernel = fpgs._get_pgs_solve_mf_gs_kernel(
        8,
        capacity,
        29,
        str(device.arch),
        friction_mode=friction,
        has_drive_rows=False,
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
    impulses = values["mf_impulses"].numpy()
    return values["v_out"].numpy(), np.asarray([impulses[w, p : p + 3] for w, p in enumerate(parents)])


@unittest.skipUnless(wp.is_cuda_available(), "MF-GS native solve requires CUDA")
class TestFeatherPGSMFMetadataCUDA(unittest.TestCase):
    def test_active_high_parents_match_rebased_contact_solve(self):
        """Solve high-parent contact triples with the supported streaming projection."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("Select a CUDA device for native MF-GS execution")
        for friction in ("current",):
            with self.subTest(friction=friction):
                low = _solve_parent_rows(device, friction, [0] * 5)
                high = _solve_parent_rows(device, friction, [0, 32767, 32768, 37247, 65532])
                for expected, actual in zip(low, high, strict=True):
                    self.assertTrue(np.isfinite(actual).all())
                    np.testing.assert_array_equal(actual, expected)
                self.assertTrue(np.all(high[1][:, 0] > 0))
                self.assertTrue(np.all(np.linalg.norm(high[1][:, 1:], axis=1) > 0))
                self.assertTrue(np.all(np.linalg.norm(high[1][:, 1:], axis=1) <= 0.4 * high[1][:, 0] + 1e-6))


if __name__ == "__main__":
    unittest.main()
