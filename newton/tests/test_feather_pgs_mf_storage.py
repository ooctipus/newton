# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check capacity-safe storage selection without changing the MF-GS schedule."""

import inspect
import itertools
import re
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import solver_feather_pgs as fpgs
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_COUNT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
)
from newton._src.solvers.feather_pgs.mf_gs_source import get_source
from newton._src.solvers.feather_pgs.row_metadata import _DENSE_META_MAX_PARENT
from newton.tests.feather_pgs_paired_fixture import pack_endpoints, solve_fixture

# The cube puzzle's streamed-metadata shape: 2304 dense and matrix-free rows over 75 coordinates.
_CUBE = {
    "max_constraints": 2304,
    "mf_max_constraints": 2304,
    "max_world_dofs": 75,
    "device_arch": "120",
    "has_drive_rows": False,
    "has_dense_velocity_limit_rows": False,
    "skip_incremental_rows": 32,
}
# Streamed-metadata option sets the resident row metadata must leave otherwise unchanged.
_STREAMED_OPTIONS = (
    {},
    {"has_drive_rows": True, "has_dense_velocity_limit_rows": True},
    {"has_drive_rows": True, "fuse_vel_limits": True},
    {"has_drive_rows": True, "has_dense_velocity_limit_rows": True, "fuse_vel_limits": True},
    {"skip_incremental_rows": 0, "contact_torsion": True},
    {"friction_mode": "bisection"},
    {"friction_mode": "bisection_desaxce"},
    {"friction_mode": "coulomb_newton"},
    {"stream_mf_impulses": True},
    {"skip_local_internal_worlds": True, "local_internal_max_constraints": 19},
    {"skip_response_block": True},
    {
        "has_dense_velocity_limit_rows": False,
        "factor_coordinates": True,
        "independent_components": True,
        "skip_incremental_rows": 0,
    },
    {"max_constraints": 1024, "mf_max_constraints": 4096, "max_world_dofs": 604},
)


def _streamed_source(**options):
    """Generate a streamed-metadata MF-GS source for the cube shape with ``options`` applied."""
    return get_source(**{**_CUBE, "shared_metadata": False, **options}).native


def _row_metadata_read_globally(source):
    """Replace each read of the packed resident row metadata with the global read it decodes.

    A packed word holds the row type in its low bits and the parent plus one above them, so
    ``s_meta_dense[r] & mask`` reads ``world_row_type[r]`` and ``(s_meta_dense[r] >> bits) - 1``
    reads ``world_row_parent[r]``. The declaration and the load-time packing then go.
    """
    source = re.sub(r"\s*// Low 3 bits[^\n]*\n\s*__shared__ int   s_meta_dense\[\d+\];\n", "\n", source)
    source = re.sub(
        r"\s*int row_type = world_row_type\.data\[off_dense \+ i\];"
        r"\s*int row_parent = world_row_parent\.data\[off_dense \+ i\];"
        r"\s*s_meta_dense\[i\] = [^\n]*\n",
        "\n",
        source,
    )
    source = re.sub(r"\(\(s_meta_dense\[(\w+)\] >> \d+\) - 1\)", r"world_row_parent.data[off_dense + \1]", source)
    source = re.sub(r"\(s_meta_dense\[(\w+)\] >> \d+\) - 1", r"world_row_parent.data[off_dense + \1]", source)
    return re.sub(r"s_meta_dense\[(\w+)\] & \d+", r"world_row_type.data[off_dense + \1]", source)


def _row_metadata_codec(source):
    """Compile the source's own packing and decoding of the resident row metadata for the CPU."""
    pack = re.search(r"s_meta_dense\[i\] = ([^;]+);", source).group(1)
    row_type = re.search(r"int row_type = (s_meta_dense\[i\] & \d+);", source).group(1)
    parent = re.search(r"int parent_idx = (\(s_meta_dense\[i\] >> \d+\) - 1);", source).group(1)
    ring = re.search(r"patch_row = (\(\(s_meta_dense\[patch_row\] >> \d+\) - 1\))\)", source).group(1)

    @wp.func_native(f"int s_meta_dense[1]; const int i = 0; s_meta_dense[i] = {pack}; return s_meta_dense[i];")
    def pack_row(row_type: int, row_parent: int) -> int: ...

    @wp.func_native(f"int s_meta_dense[1] = {{packed}}; const int i = 0; return {row_type};")
    def decode_type(packed: int) -> int: ...

    @wp.func_native(f"int s_meta_dense[1] = {{packed}}; const int i = 0; return {parent};")
    def decode_parent(packed: int) -> int: ...

    @wp.func_native(f"int s_meta_dense[1] = {{packed}}; const int patch_row = 0; return {ring};")
    def decode_ring(packed: int) -> int: ...

    @wp.kernel(module="unique", enable_backward=False)
    def round_trip(types: wp.array[int], parents: wp.array[int], decoded: wp.array2d[int]):
        k = wp.tid()
        packed = pack_row(types[k], parents[k])
        decoded[k, 0] = decode_type(packed)
        decoded[k, 1] = decode_parent(packed)
        decoded[k, 2] = decode_ring(packed)

    return round_trip


class TestFeatherPGSMFStorage(unittest.TestCase):
    def test_hardware_capacity_streams_only_when_required(self):
        """Preserve all hardware rows when resident impulses exceed the device limit."""
        options = {"has_drive_rows": False, "fuse_vel_limits": False}
        self.assertEqual(fpgs._select_mfgs_storage(256, 37248, 1551, 101376, **options), (False, True))
        self.assertEqual(fpgs._select_mfgs_storage(256, 37248, 1551, 232448, **options), (True, False))
        self.assertEqual(fpgs._select_mfgs_storage(256, 15984, 3135, 101376, **options), (True, False))

    def test_storage_boundary_and_remaining_state(self):
        """Include tile storage and reject an unrepresentable velocity working set."""
        options = {"has_drive_rows": False, "fuse_vel_limits": False}
        # Four streamed metadata arrays leave velocities and two impulse arrays.
        required = 128 + 4 * (29 + 1024 + 4096)
        self.assertEqual(fpgs._select_mfgs_storage(1024, 4096, 29, required, **options), (False, False))
        self.assertEqual(fpgs._select_mfgs_storage(1024, 4096, 29, required - 1, **options), (False, True))
        with self.assertRaisesRegex(ValueError, "MF-GS.*shared memory"):
            fpgs._select_mfgs_storage(256, 37248, 26000, 101376, **options)

    def test_resident_row_metadata_reads_what_the_streamed_source_reads(self):
        """Read every row type and parent from the packed array where the streamed source reads global memory.

        Decoding each packed read back into the global read it stands for must give the streamed
        source byte for byte: the variants differ in the storage of row types and parents only.
        The end-of-iteration limit scans visit rows a split row-phase launch does not stage, so
        both read the row type from global memory there.
        """
        for options in _STREAMED_OPTIONS:
            with self.subTest(options=options):
                streamed = _streamed_source(**options)
                resident = _streamed_source(**options, resident_row_metadata=True)
                rows = options.get("max_constraints", _CUBE["max_constraints"])
                self.assertNotIn("s_meta_dense", streamed)
                self.assertEqual(resident.count(f"__shared__ int   s_meta_dense[{rows}];"), 1)
                self.assertIn("int row_type = s_meta_dense[i] &", resident)
                self.assertNotIn("(s_meta_dense[i] &", resident)
                self.assertEqual(
                    resident.count("(world_row_type.data[off_dense + i])"),
                    streamed.count("(world_row_type.data[off_dense + i])"),
                )
                # The float metadata streams in both.
                shared = ["s_v", "s_lam_dense"] + ([] if options.get("stream_mf_impulses") else ["s_lam_mf"])
                self.assertEqual(re.findall(r"__shared__ float\s+(\w+)", resident), shared)
                self.assertEqual(_row_metadata_read_globally(resident), streamed)
        # The cube's patch rings walk the resident array.
        self.assertIn("patch_row = ((s_meta_dense[patch_row] >> 3) - 1)", _streamed_source(resident_row_metadata=True))
        with self.assertRaisesRegex(ValueError, "streamed metadata"):
            get_source(**_CUBE, resident_row_metadata=True)

    def test_packed_row_metadata_decodes_every_row_type_and_parent(self):
        """Run the source's own packing and decoding on the CPU: every type and parent comes back exactly."""
        source = _streamed_source(resident_row_metadata=True)
        pairs = itertools.product(range(PGS_CONSTRAINT_TYPE_COUNT), (-1, 0, 1, 2302, 2303, _DENSE_META_MAX_PARENT))
        types, parents = np.array(list(pairs), np.int32).T.copy()
        decoded = wp.zeros((len(types), 3), dtype=int, device="cpu")
        wp.launch(
            _row_metadata_codec(source),
            dim=len(types),
            inputs=[wp.array(types, dtype=int, device="cpu"), wp.array(parents, dtype=int, device="cpu")],
            outputs=[decoded],
            device="cpu",
        )
        np.testing.assert_array_equal(decoded.numpy(), np.stack([types, parents, parents], axis=1))

    def test_resident_row_metadata_kernel_names_its_storage(self):
        """Name the resident variant apart from the streamed kernel, whose name stays as it was."""
        factory = fpgs._get_pgs_solve_mf_gs_kernel.__wrapped__
        shape = (2304, 2304, 75, "120")
        options = {"has_drive_rows": False, "has_dense_velocity_limit_rows": False, "skip_incremental_rows": 32}
        streamed = factory(*shape, shared_metadata=False, **options)
        resident = factory(*shape, shared_metadata=False, resident_row_metadata=True, **options)
        self.assertEqual(streamed.key, "pgs_solve_mf_gs_2304_2304_75_current_drive0_densevlim0_gmeta_skipinc32")
        self.assertEqual(resident.key, "pgs_solve_mf_gs_2304_2304_75_current_drive0_densevlim0_gmeta_smeta_skipinc32")
        self.assertEqual(resident.mf_native.native_snippet.count("__shared__ int   s_meta_dense[2304];"), 1)
        self.assertEqual(
            list(inspect.signature(resident.func).parameters), list(inspect.signature(streamed.func).parameters)
        )
        with self.assertRaisesRegex(ValueError, "streamed metadata"):
            factory(*shape, resident_row_metadata=True, **options)

    def test_streaming_factory_owns_global_impulses_on_one_lane(self):
        """Stream the uniform paired projection; reject single-lane nonlinear paths."""
        for friction in ("current", "bisection", "bisection_desaxce", "coulomb_newton"):
            for shared_metadata in (False, True):
                with self.subTest(friction=friction, shared_metadata=shared_metadata):
                    if friction != "current":
                        with self.assertRaisesRegex(ValueError, "warp-uniform current-friction"):
                            fpgs._get_pgs_solve_mf_gs_kernel(
                                16,
                                64,
                                12,
                                "cpu",
                                friction_mode=friction,
                                shared_metadata=shared_metadata,
                                stream_mf_impulses=True,
                            )
                        continue
                    kernel = fpgs._get_pgs_solve_mf_gs_kernel(
                        16,
                        64,
                        12,
                        "cpu",
                        friction_mode=friction,
                        shared_metadata=shared_metadata,
                        stream_mf_impulses=True,
                    )
                    native = inspect.getclosurevars(kernel.func).nonlocals["pgs_solve_mf_gs_native"]
                    source = native.native_snippet
                    self.assertIn("_glambda", kernel.key)
                    self.assertNotIn("__shared__ float s_lam_mf", source)
                    self.assertNotIn("mf_impulses.data[off_mf + i] =", source)
                    self.assertIn("__shfl_sync(MASK, lane == 0 ? g_lam_mf[i] : 0.0f, 0)", source)
                    self.assertIn(
                        "if (mf_friction_radius == 0.0f && "
                        "__shfl_sync(MASK, lane == 0 ? g_lam_mf[i] : 0.0f, 0) == 0.0f && "
                        "__shfl_sync(MASK, lane == 0 ? g_lam_mf[i + 1] : 0.0f, 0) == 0.0f)",
                        source,
                    )
                    for line in source.splitlines():
                        if re.search(r"g_lam_mf\[[^]]+\]\s*=(?!=)", line):
                            self.assertIn("if (lane == 0)", line)

    def test_limit_only_phase_does_not_stage_contact_prefix(self):
        """Keep contacts untouched and retain work for empty-segment fused clamps."""
        for fused in (False, True):
            kernel = fpgs._get_pgs_solve_mf_gs_kernel(16, 4096, 128, "cpu", fuse_vel_limits=fused)
            native = inspect.getclosurevars(kernel.func).nonlocals["pgs_solve_mf_gs_native"]
            source = native.native_snippet
            self.assertIn("int mf_load_lo = (row_phase == 5) ? mf_contact_end : 0;", source)
            self.assertEqual(source.count("for (int i = mf_load_lo + lane; i < m_mf; i += 32)"), 2)
            guard = "if (row_phase == 5 && dense_lo == dense_hi && mf_contact_end == m_mf) return;"
            if fused:
                self.assertNotIn(guard, source)
            else:
                self.assertIn(guard, source)
                self.assertLess(source.index(guard), source.index("// LOAD PHASE"))


def _patch_ring_rows(seed, *, worlds, dense_capacity, mf_capacity, dofs, drives, velocity_limits, rings):
    """Build MF-GS operands whose dense contacts form patch rings, as ``link_patch_rows`` leaves them.

    Each world holds a prefix of drive rows (with ``drives``) and position-limit rows, then
    velocity-limit rows (with ``velocity_limits``), recorded in its phase bounds, then contacts in a
    random allocation order. A contact is a normal row, followed by two tangent rows naming it as
    parent when it owns an anchor (at most two per ring). The normals of a ring link circularly
    through their parents; a ring of one ends with -1 or links to itself. ``rings`` bounds the rings
    per world. The last six coordinates are a free body with matrix-free contact triples of their
    own, linked the same way; the last world has none. Impulses start warm on some rows, signed
    zeros included.
    """
    rng = np.random.default_rng(seed)
    w_count, capacity, free = worlds, dense_capacity, dofs - 6
    host = {
        "world_constraint_count": np.zeros(w_count, np.int32),
        "dense_phase_bounds": np.zeros((w_count, 2), np.int32),
        "world_row_type": np.full((w_count, capacity), PGS_CONSTRAINT_TYPE_CONTACT, np.int32),
        "world_row_parent": np.full((w_count, capacity), -1, np.int32),
        "world_row_mu": np.zeros((w_count, capacity), np.float32),
        "rhs_bias": np.zeros((w_count, capacity), np.float32),
        "world_diag": np.zeros((w_count, capacity), np.float32),
        "world_row_w": rng.uniform(0.6, 1.0, (w_count, capacity)).astype(np.float32),
        "world_impulses": np.zeros((w_count, capacity), np.float32),
        "J_world": np.zeros((w_count, capacity, dofs), np.float32),
        "Y_world": np.zeros((w_count, capacity, dofs), np.float32),
        "world_drive_target_vel_bias": np.zeros((w_count, capacity), np.float32),
        "world_drive_vel_multiplier": np.zeros((w_count, capacity), np.float32),
        "world_drive_impulse_multiplier": np.zeros((w_count, capacity), np.float32),
        "world_drive_max_impulse": np.zeros((w_count, capacity), np.float32),
        "world_drive_vel_limit": np.zeros((w_count, capacity), np.float32),
        "mf_constraint_count": np.zeros(w_count, np.int32),
        "mf_meta": np.zeros((w_count, 4 * mf_capacity), np.int32),
        "mf_impulses": np.zeros((w_count, mf_capacity), np.float32),
        "mf_J_a": np.zeros((w_count, mf_capacity, 6), np.float32),
        "mf_J_b": np.zeros((w_count, mf_capacity, 6), np.float32),
        "mf_MiJt_a": np.zeros((w_count, mf_capacity, 6), np.float32),
        "mf_MiJt_b": np.zeros((w_count, mf_capacity, 6), np.float32),
        "mf_row_mu": np.zeros((w_count, mf_capacity), np.float32),
        "mf_row_w": rng.uniform(0.6, 1.0, (w_count, mf_capacity)).astype(np.float32),
        "world_deferred_dof_mask": (rng.random((w_count, dofs)) < 0.2).astype(np.int32),
        "v_out": rng.normal(0.0, 0.3, w_count * dofs).astype(np.float32),
    }

    def units(count):
        """Rings of one to four contacts; True marks a contact that owns an anchor."""
        layout = []
        for ring in range(count):
            members = int(rng.integers(1, 5))
            anchors = rng.choice(members, size=min(members, int(rng.integers(0, 3))), replace=False)
            layout += [(ring, k in anchors) for k in range(members)]
        return [layout[k] for k in rng.permutation(len(layout))]

    def link(parents, heads):
        for members in heads.values():
            if len(members) == 1:
                parents[members[0]] = members[0] if rng.random() < 0.5 else -1
                continue
            order = rng.permutation(members)
            for k, head in enumerate(order):
                parents[head] = order[(k + 1) % len(order)]

    for w in range(w_count):
        types, parents = host["world_row_type"][w], host["world_row_parent"][w]
        row = 0
        if drives:
            count = int(rng.integers(1, 4))
            types[:count] = PGS_CONSTRAINT_TYPE_JOINT_TARGET
            host["world_drive_target_vel_bias"][w, :count] = rng.uniform(-0.2, 0.2, count)
            host["world_drive_vel_multiplier"][w, :count] = rng.uniform(-0.8, -0.2, count)
            host["world_drive_impulse_multiplier"][w, :count] = rng.uniform(0.8, 1.0, count)
            host["world_drive_max_impulse"][w, :count] = rng.uniform(0.2, 1.0, count)
            host["world_drive_vel_limit"][w, :count] = rng.uniform(0.05, 0.3, count)
            row = count
        count = int(rng.integers(0, 4))
        types[row : row + count] = PGS_CONSTRAINT_TYPE_JOINT_LIMIT
        row += count
        host["dense_phase_bounds"][w, 0] = row
        if velocity_limits:
            count = int(rng.integers(1, 3))
            types[row : row + count] = PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
            row += count
        host["dense_phase_bounds"][w, 1] = row
        host["rhs_bias"][w, :row] = rng.uniform(-0.2, 0.1, row)
        heads = {}
        for ring, anchor in units(int(rng.integers(*rings))):
            needed = 3 if anchor else 1
            if row + needed > capacity:
                break
            heads.setdefault(ring, []).append(row)
            host["rhs_bias"][w, row] = rng.uniform(-0.3, 0.0) if rng.random() < 0.6 else rng.uniform(0.0, 0.3)
            if rng.random() < 0.4:
                host["world_impulses"][w, row] = rng.uniform(0.0, 0.2)
            if anchor:
                tangents = slice(row + 1, row + 3)
                types[tangents] = PGS_CONSTRAINT_TYPE_FRICTION
                parents[tangents] = row
                host["world_row_mu"][w, tangents] = rng.uniform(0.2, 0.9)
                host["rhs_bias"][w, tangents] = rng.uniform(-0.15, 0.15, 2)
                draw = rng.random()
                if draw < 0.3:
                    host["world_impulses"][w, tangents] = rng.uniform(-0.02, 0.02, 2)
                elif draw < 0.45:
                    host["world_impulses"][w, tangents] = -0.0
            row += needed
        link(parents, heads)
        host["world_constraint_count"][w] = row
        jacobian = rng.normal(0.0, 0.3, (row, dofs)).astype(np.float32)
        host["J_world"][w, :row] = jacobian
        host["Y_world"][w, :row] = jacobian * rng.uniform(0.5, 2.0, dofs).astype(np.float32)
        host["world_diag"][w, :row] = np.sum(jacobian * host["Y_world"][w, :row], axis=1) + np.float32(0.01)

        if w == w_count - 1:
            continue
        mf_parents, mf_types, mf_heads, row = {}, {}, {}, 0
        for ring, anchor in units(int(rng.integers(1, 4))):
            needed = 3 if anchor else 1
            if row + needed > mf_capacity:
                break
            mf_heads.setdefault(ring, []).append(row)
            mf_types[row] = PGS_CONSTRAINT_TYPE_CONTACT
            if anchor:
                for tangent in (row + 1, row + 2):
                    mf_types[tangent], mf_parents[tangent] = PGS_CONSTRAINT_TYPE_FRICTION, row
                host["mf_row_mu"][w, row + 1 : row + 3] = rng.uniform(0.2, 0.9)
            row += needed
        normals = np.full(row, -1, np.int32)
        link(normals, mf_heads)
        for r in range(row):
            rhs = rng.uniform(-0.3, 0.1) if mf_types[r] == PGS_CONSTRAINT_TYPE_CONTACT else rng.uniform(-0.1, 0.1)
            parent = int(normals[r]) if mf_types[r] == PGS_CONSTRAINT_TYPE_CONTACT else mf_parents[r]
            host["mf_meta"][w, 4 * r : 4 * r + 4] = (
                pack_endpoints(free, -1),
                np.float32(rng.uniform(0.5, 1.0)).view(np.int32),
                np.float32(rhs).view(np.int32),
                pack_endpoints(parent, mf_types[r]),
            )
            host["mf_J_a"][w, r] = rng.normal(0.0, 0.3, 6)
            host["mf_MiJt_a"][w, r] = host["mf_J_a"][w, r] * rng.uniform(0.5, 1.5)
        host["mf_impulses"][w, :row] = np.where(rng.random(row) < 0.3, rng.uniform(0.0, 0.1, row), 0.0)
        host["mf_constraint_count"][w] = row
    host["mf_contact_rows_end"] = host["mf_constraint_count"].copy()
    host["world_dof_indices"] = np.arange(w_count * dofs, dtype=np.int32).reshape(w_count, dofs)
    return host


def _bind_rows(host, device):
    """Bind the operands as the MF-GS launch takes them, with neutral queue, owner and torsion inputs."""

    def array(value):
        dtype = wp.int32 if value.dtype == np.int32 else wp.float32
        return wp.array(np.ascontiguousarray(value), dtype=dtype, device=device)

    worlds = len(host["world_constraint_count"])
    values = {name: array(value) for name, value in host.items()}
    values.update(
        general_world_count=array(np.array([worlds], np.int32)),
        general_worlds=array(np.random.default_rng(5).permutation(worlds).astype(np.int32)),
        rb_class=array(np.zeros(worlds, np.int32)),
        local_solve_owner=array(np.zeros(worlds, np.int32)),
        world_torsion_group=array(np.full((1, 1), -1, np.int32)),
        contact_torsion_radius=0.0,
    )
    return values


@unittest.skipUnless(wp.is_cuda_available(), "MF-GS native solve requires CUDA")
class TestFeatherPGSMFStorageCUDA(unittest.TestCase):
    def test_resident_row_metadata_matches_streamed_on_patch_rings(self):
        """Solve dense patch rings with resident row types and parents: identical bits out in every row phase.

        Random patch layouts at a small and at the cube puzzle's dense capacity run every row
        phase, from warm and signed-zero impulses, with delayed friction, iteration offsets,
        regularization and deferred responses, both one world per block and through a two-block
        queue whose blocks reuse their shared memory across worlds, then once more from a captured
        graph. The drive and limit case holds every end-of-iteration limit scan, whose rows a
        split row-phase launch does not stage.
        """
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("Select a CUDA device for native MF-GS execution")
        cube = {"has_drive_rows": False, "has_dense_velocity_limit_rows": False, "skip_incremental_rows": 32}
        limits = {"has_drive_rows": True, "has_dense_velocity_limit_rows": True, "fuse_vel_limits": True}
        cases = (
            ("cube_capacity", (2304, 48, 75), cube, (220, 300)),
            ("cube_options", (48, 12, 29), cube, (5, 9)),
            ("drives_and_limits", (48, 12, 29), limits, (5, 9)),
            ("bisection", (48, 12, 29), {"friction_mode": "bisection", "has_drive_rows": False}, (5, 9)),
            ("streamed_impulses", (48, 12, 29), {"stream_mf_impulses": True, "has_drive_rows": False}, (5, 9)),
        )
        settings = (
            {"iterations": 8, "omega": 1.0, "friction_start_iteration": 0, "iteration_offset": 0},
            {
                "iterations": 6,
                "omega": 0.9,
                "friction_start_iteration": 3,
                "iteration_offset": 1,
                "regularize": 1,
                "freeze_drive_rows": 1,
                "defer_dense_response": 1,
            },
        )
        outputs = ("v_out", "world_impulses", "mf_impulses")
        for name, (dense_capacity, mf_capacity, dofs), options, rings in cases:
            host = _patch_ring_rows(
                dense_capacity + dofs,
                worlds=5,
                dense_capacity=dense_capacity,
                mf_capacity=mf_capacity,
                dofs=dofs,
                drives=options.get("has_drive_rows", True),
                velocity_limits=options.get("has_dense_velocity_limit_rows", True),
                rings=rings,
            )
            values = _bind_rows(host, device)
            kernels = [
                fpgs._get_pgs_solve_mf_gs_kernel(
                    dense_capacity,
                    mf_capacity,
                    dofs,
                    str(device.arch),
                    shared_metadata=False,
                    resident_row_metadata=resident,
                    **options,
                )
                for resident in (False, True)
            ]
            self.assertTrue(kernels[1].key.startswith(kernels[0].key.split("_gmeta")[0] + "_gmeta_smeta"))
            banks = [{output: wp.clone(values[output]) for output in outputs} for _ in kernels]

            def run(launch, banks=banks, kernels=kernels, values=values):
                for kernel, bank in zip(kernels, banks, strict=True):
                    for output in outputs:
                        wp.copy(bank[output], values[output])
                    bound = {**values, **bank, **launch}
                    wp.launch_tiled(
                        kernel,
                        dim=[launch["general_world_grid_stride"]],
                        block_dim=32,
                        device=device,
                        inputs=[bound[parameter] for parameter in inspect.signature(kernel.func).parameters],
                    )

            launches = [
                {
                    "row_phase": phase,
                    "general_world_grid_stride": grid,
                    "use_general_world_queue": int(grid < 5),
                    "regularize": 0,
                    "freeze_drive_rows": 0,
                    "defer_dense_response": 0,
                    **setting,
                }
                for phase, grid, setting in itertools.product(range(7), (5, 2), settings)
            ]
            for launch in launches:
                run(launch)
                with self.subTest(case=name, **launch):
                    for output in outputs:
                        np.testing.assert_array_equal(
                            banks[1][output].numpy().view(np.uint32), banks[0][output].numpy().view(np.uint32)
                        )
                if launch is launches[0]:
                    # The fixture exercises the rings: the full sweep loads tangent rows and moves impulses.
                    impulses = banks[0]["world_impulses"].numpy()
                    tangents = host["world_row_type"] == PGS_CONSTRAINT_TYPE_FRICTION
                    self.assertFalse(np.array_equal(impulses, host["world_impulses"]), name)
                    self.assertTrue(np.any(impulses[tangents] != 0.0), name)
            with wp.ScopedCapture(device=device) as capture:
                run(launches[0])
            wp.capture_launch(capture.graph)
            with self.subTest(case=name, graph=True):
                for output in outputs:
                    np.testing.assert_array_equal(
                        banks[1][output].numpy().view(np.uint32), banks[0][output].numpy().view(np.uint32)
                    )

    def test_streamed_impulses_match_resident_with_friction_and_limits(self):
        """Match mixed contact triples, warm impulses, limits and graph replay exactly."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("Select a CUDA device for native MF-GS execution")
        _, arrays = solve_fixture(device)
        values = dict(arrays)
        values.update(
            general_world_grid_stride=5,
            use_general_world_queue=0,
            iterations=8,
            omega=1.0,
            regularize=0,
            row_phase=0,
            friction_start_iteration=2,
            iteration_offset=0,
            freeze_drive_rows=0,
            defer_dense_response=0,
        )
        outputs = ("v_out", "world_impulses", "mf_impulses")
        for friction in ("current",):
            for shared_metadata in (False, True):
                with self.subTest(friction=friction, shared_metadata=shared_metadata):
                    variants = []
                    for stream in (False, True):
                        kernel = fpgs._get_pgs_solve_mf_gs_kernel(
                            8,
                            6,
                            29,
                            str(device.arch),
                            friction_mode=friction,
                            has_drive_rows=False,
                            has_dense_velocity_limit_rows=False,
                            shared_metadata=shared_metadata,
                            stream_mf_impulses=stream,
                        )
                        bound = dict(values)
                        bound.update({name: wp.clone(values[name]) for name in outputs})
                        variants.append((kernel, bound))

                    def run(variants=variants):
                        for kernel, bound in variants:
                            for name in outputs:
                                wp.copy(bound[name], values[name])
                            wp.launch_tiled(
                                kernel,
                                dim=[5],
                                block_dim=32,
                                device=device,
                                inputs=[bound[name] for name in inspect.signature(kernel.func).parameters],
                            )

                    for phase in (0, 1, 2, 3, 4, 5):
                        for _, bound in variants:
                            bound["row_phase"] = phase
                        run()
                        for name in outputs:
                            np.testing.assert_array_equal(variants[0][1][name].numpy(), variants[1][1][name].numpy())
                    for _, bound in variants:
                        bound["row_phase"] = 0
                    with wp.ScopedCapture(device=device) as capture:
                        run()
                    wp.capture_launch(capture.graph)
                    for name in outputs:
                        np.testing.assert_array_equal(variants[0][1][name].numpy(), variants[1][1][name].numpy())


if __name__ == "__main__":
    unittest.main()
