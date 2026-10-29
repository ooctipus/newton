# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check capacity-safe storage selection without changing the MF-GS schedule."""

import functools
import inspect
import itertools
import os
import re
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import execution_plan, sweep_routes
from newton._src.solvers.feather_pgs import solver_feather_pgs as fpgs
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_COUNT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    BodyFactors,
)
from newton._src.solvers.feather_pgs.row_metadata import _DENSE_META_MAX_PARENT
from newton.tests.feather_pgs_paired_fixture import pack_endpoints, solve_fixture
from newton.tests.test_feather_pgs_masked_patch_rows import _three_size_model
from newton.tests.test_feather_pgs_native_prelude import _native_snippets

# A large streamed-metadata shape without drive or velocity-limit rows: 2304 dense and matrix-free rows
# over 75 coordinates.
_LARGE_STREAMED_SHAPE = {
    "max_constraints": 2304,
    "mf_max_constraints": 2304,
    "max_world_dofs": 75,
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
    {"stream_mf_impulses": True},
    {"skip_local_internal_worlds": True, "local_internal_max_constraints": 19},
    {
        "has_dense_velocity_limit_rows": False,
        "factor_coordinates": True,
        "independent_components": True,
        "skip_incremental_rows": 0,
    },
    {"max_constraints": 1024, "mf_max_constraints": 4096, "max_world_dofs": 604},
    # The accumulated velocity-limit law, in the dense limit pass and in streamed free-body totals.
    {"has_drive_rows": True, "has_dense_velocity_limit_rows": True, "velocity_limit_law": "accumulated"},
    {"stream_mf_impulses": True, "velocity_limit_law": "accumulated"},
)


def _streamed_source(**options):
    """Generate a streamed-metadata MF-GS source for the large shape with ``options`` applied."""
    options = {key: value for key, value in options.items() if key != "local_internal_max_constraints"}
    return sweep_routes.mf_gs_source(**{**_LARGE_STREAMED_SHAPE, "shared_metadata": False, **options})[0]


def _row_metadata_read_globally(source):
    """Replace each read of the packed resident row metadata with the global read it decodes.

    A packed word holds the row type in its low bits and the parent plus one above them, so
    ``s_meta_dense[r] & mask`` reads ``world_row_type[r]`` and ``(s_meta_dense[r] >> bits) - 1``
    reads ``world_row_parent[r]``. The declaration and the load-time packing then go.
    """
    source = re.sub(r"\n\s*__shared__ int s_meta_dense\[\d+\];\n", "\n", source)
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
    parent = re.search(r"parent = (\(s_meta_dense\[i\] >> \d+\) - 1);", source).group(1)
    ring = re.search(r"return (\(\(s_meta_dense\[patch_row\] >> \d+\) - 1\));", source).group(1)

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


def _host(array):
    """Copy a Warp array to a host array of its own (a CPU array's ``numpy()`` is a view of it)."""
    return np.array(array.numpy(), copy=True)


def _bits(values):
    """View a host array's raw bytes, so equality also tells signed zeros and NaN payloads apart."""
    return np.ascontiguousarray(values).view(np.uint8)


class _LaunchReplay:
    """Replay each launch of a production kernel through a twin and compare the two right away.

    :meth:`wrap` returns a stand-in for ``wp.launch_tiled`` that passes other kernels through. Before a
    production launch it clones every array operand once, so operands bound to one array stay bound to
    one clone. The production launch then runs on the caller's arrays and the twin on the clones, and
    both results are copied to the host before the stand-in returns: later stages and steps rewrite the
    caller's arrays, so a comparison made after the run would see another state. Each record names the
    operands whose bits differ and holds the launch grid, the scalar operands and host copies of the
    ``keep`` operands before and after the production launch. Production launches must pass ``dim``,
    ``inputs`` and ``outputs`` by keyword.
    """

    def __init__(self, production, twin, keep=()):
        self.production, self.twin, self.keep = production, twin, tuple(keep)
        self.names = list(inspect.signature(production.func).parameters)
        self.records = []

    def wrap(self, launch_tiled):
        """Return the ``wp.launch_tiled`` stand-in that replays and records the production launches."""

        def replay(kernel, *args, **kwargs):
            if kernel is not self.production:
                return launch_tiled(kernel, *args, **kwargs)
            inputs, outputs = list(kwargs["inputs"]), list(kwargs.get("outputs") or ())
            operands = dict(zip(self.names, inputs + outputs, strict=True))
            arrays = {name: value for name, value in operands.items() if isinstance(value, wp.array)}
            clones = {}
            for value in arrays.values():
                if id(value) not in clones:
                    clones[id(value)] = wp.clone(value)
            twin_operands = [clones[id(value)] if isinstance(value, wp.array) else value for value in inputs + outputs]
            before = {name: _host(arrays[name]) for name in self.keep}
            result = launch_tiled(kernel, *args, **kwargs)
            twin_kwargs = {**kwargs, "inputs": twin_operands[: len(inputs)], "outputs": twin_operands[len(inputs) :]}
            launch_tiled(self.twin, *args, **twin_kwargs)
            ours = {name: _host(value) for name, value in arrays.items()}
            theirs = {name: _host(clones[id(value)]) for name, value in arrays.items()}
            self.records.append(
                {
                    "differ": [name for name in arrays if not np.array_equal(_bits(ours[name]), _bits(theirs[name]))],
                    "dim": list(kwargs["dim"]),
                    "scalars": {name: value for name, value in operands.items() if name not in arrays},
                    "before": before,
                    "after": {name: ours[name] for name in self.keep},
                }
            )
            return result

        return replay


@wp.kernel(module="unique", enable_backward=False)
def _replay_production(source: wp.array[float], shifted: wp.array[float], velocity: wp.array[float]):
    i, _lane = wp.tid()
    shifted[i] = source[i] + 1.0
    velocity[i] = 0.5 * velocity[i] + shifted[i]


@wp.kernel(module="unique", enable_backward=False)
def _replay_twin(source: wp.array[float], shifted: wp.array[float], velocity: wp.array[float]):
    i, _lane = wp.tid()
    shifted[i] = source[i] + 1.0
    velocity[i] = 0.5 * velocity[i] + shifted[i]


@wp.kernel(module="unique", enable_backward=False)
def _replay_offset_twin(source: wp.array[float], shifted: wp.array[float], velocity: wp.array[float]):
    i, _lane = wp.tid()
    shifted[i] = source[i] + 1.0
    velocity[i] = 0.5 * velocity[i] + shifted[i] + 1.0


@wp.kernel(module="unique", enable_backward=False)
def _replay_rewrite(source: wp.array[float], velocity: wp.array[float]):
    i, _lane = wp.tid()
    source[i] = 2.0 * source[i] + 1.0
    velocity[i] = velocity[i] - 3.0


def _storage(shape, budget, *, block_count=16384, sm_count=188, has_drive_rows=False, fuse_vel_limits=False, **request):
    """The storage selection of a sweep of ``shape`` (rows, free-body rows, coordinates) under a static ``budget``."""
    staging = functools.partial(
        sweep_routes.mf_gs_staging, *shape, has_drive_rows=has_drive_rows, fuse_vel_limits=fuse_vel_limits
    )
    return execution_plan._select_mfgs_storage(staging, budget, block_count=block_count, sm_count=sm_count, **request)


class TestFeatherPGSMFStorage(unittest.TestCase):
    def test_hardware_capacity_streams_only_when_required(self):
        """Preserve all hardware rows when resident impulses exceed the device limit."""
        self.assertEqual(_storage((256, 37248, 1551), 100352)[:2], (False, True))
        self.assertEqual(_storage((256, 37248, 1551), 231424)[:2], (True, False))
        self.assertEqual(_storage((256, 15984, 3135), 100352)[:2], (True, False))

    def test_storage_boundary_and_remaining_state(self):
        """Include tile storage and reject an unrepresentable velocity working set."""
        # Four streamed metadata arrays leave velocities, two impulse arrays and the free bodies' two one-word
        # bitmaps and flag.
        required = 128 + 4 * (29 + 1024 + 4096) + 4 * (2 * 1 + 1)
        self.assertEqual(_storage((1024, 4096, 29), required)[:2], (False, False))
        self.assertEqual(_storage((1024, 4096, 29), required - 1)[:2], (False, True))
        with self.assertRaisesRegex(ValueError, "MF-GS.*shared memory"):
            _storage((256, 37248, 26000), 100352)

    def test_metadata_reasons_report_the_footprint_they_compared(self):
        """Report the metadata and block bytes the residency decision compared, for each outcome."""
        options = {"has_drive_rows": False, "fuse_vel_limits": False}
        cases = (
            ((192, 64, 29), 100352, True, "Row metadata is resident: {metadata} bytes fit the 4096-byte budget."),
            ((1024, 4096, 604), 100352, False, "Row metadata streams: {metadata} bytes exceed the 4096-byte budget."),
            (
                (192, 64, 29),
                4096,
                False,
                "the block would declare {total} bytes of static shared memory; the device's budget is 4096.",
            ),
        )
        for shape, budget, resident, text in cases:
            with self.subTest(shape=shape, budget=budget):
                total = sweep_routes.mf_gs_staging(*shape, **options).bytes
                metadata = total - sweep_routes.mf_gs_staging(*shape, shared_metadata=False, **options).bytes
                storage = _storage(shape, budget)
                self.assertEqual(storage.shared_metadata, resident)
                self.assertIn(text.format(metadata=metadata, total=total), storage.reason)

    def test_single_wave_launches_keep_row_types_and_parents_resident(self):
        """Keep packed row metadata resident only for grids of one block per SM within 48 KiB of static memory."""

        def select(shape, budget, blocks, sms=188, **request):
            return _storage(shape, budget, block_count=blocks, sm_count=sms, **request)

        large = (2304, 2304, 75)
        # 128 worlds, one block each, on 188 SMs: velocities, dense and MF impulses, the packed row
        # metadata, the dense rows' support (3 words of 75 coordinates: 2 * 3 + 1 entries) and the free bodies'
        # bitmaps (as many) take 128 + 4 * (75 + 3 * 2304) + 2 * 4 * (2 * 3 + 1) = 28132 bytes; the float metadata
        # streams.
        storage = select(large, 100352, 128)
        self.assertEqual(storage[:3], (False, False, True))
        self.assertIn("28132 <= 49152 bytes", storage.reason)
        self.assertEqual(select(large, 231424, 128)[:3], (False, False, True))
        # A grid of one block per SM is the largest that cannot lose concurrency to the extra bytes.
        self.assertTrue(select(large, 100352, 188).resident_row_metadata)
        for blocks in (189, 16384):
            with self.subTest(blocks=blocks):
                storage = select(large, 100352, blocks)
                self.assertEqual(storage[:3], (False, False, False))
                self.assertIn(f"{blocks}-block grid exceeds one block per SM (188 SMs)", storage.reason)
        # 48 KiB of static shared memory: 128 + 4 * (4746 + 2 * 2304 + 2304) + 2 * 4 * (2 * 149 + 1) bytes (the
        # support's and the free bodies' 149 words) is exactly the budget.
        self.assertTrue(select((2304, 2304, 4746), 100352, 128).resident_row_metadata)
        storage = select((2304, 2304, 4747), 100352, 128)
        self.assertFalse(storage.resident_row_metadata)
        self.assertIn("49156 bytes of static shared memory, over the 49152-byte budget", storage.reason)
        # A device that exposes less than 48 KiB sets the budget.
        self.assertTrue(select(large, 28132, 128).resident_row_metadata)
        self.assertFalse(select(large, 28131, 128).resident_row_metadata)
        # Streamed MF impulses leave room: 128 + 4 * (1551 + 2 * 256) + 2 * 4 * (2 * 49 + 1) = 9172 bytes.
        storage = select((256, 37248, 1551), 100352, 32)
        self.assertEqual(storage[:3], (False, True, True))
        self.assertIn("9172 <= 49152", storage.reason)
        # Resident MF impulses do not: 22528 of them hold the block at 99284 bytes.
        self.assertEqual(select((256, 22528, 1551), 100352, 32)[:3], (False, False, False))
        # Streaming them on request takes their bytes out of the block before the budget is checked.
        storage = select((256, 22528, 1551), 100352, 32, force_stream_mf_impulses=True)
        self.assertEqual(storage[:3], (False, True, True))
        self.assertIn("MF impulses stream on request (FEATHER_PGS_MFGS_STREAM_LAMBDA)", storage.reason)
        self.assertIn("9172 <= 49152", storage.reason)
        # The request changes nothing where capacity streams them already, or where nothing is resident to free.
        for shape, blocks in (((256, 37248, 1551), 32), ((2304, 2304, 75), 16384)):
            with self.subTest(shape=shape, blocks=blocks):
                requested = select(shape, 100352, blocks, force_stream_mf_impulses=True)
                self.assertTrue(requested.stream_mf_impulses)
                self.assertEqual(requested.resident_row_metadata, select(shape, 100352, blocks).resident_row_metadata)
        # Resident metadata already holds the packed rows.
        storage = select((192, 64, 22), 100352, 16384)
        self.assertEqual(storage[:3], (True, False, False))
        self.assertIn("Row metadata is resident", storage.reason)

    def test_resident_row_metadata_reads_what_the_streamed_source_reads(self):
        """Read every row type and parent from the packed array where the streamed source reads global memory.

        Decoding each packed read back into the global read it stands for must give the streamed
        source byte for byte: the variants differ in the storage of row types and parents only.
        The end-of-iteration limit scans keep the streamed source's global row-type reads, so the
        variants differ only in the row loop and the friction projectors (packed reads would be
        exact there too: every row a scan visits is staged).
        """
        for options in _STREAMED_OPTIONS:
            with self.subTest(options=options):
                streamed = _streamed_source(**options)
                resident = _streamed_source(**options, resident_row_metadata=True)
                rows = options.get("max_constraints", _LARGE_STREAMED_SHAPE["max_constraints"])
                self.assertNotIn("s_meta_dense", streamed)
                self.assertEqual(resident.count(f"__shared__ int s_meta_dense[{rows}];"), 1)
                self.assertIn("int row_type = s_meta_dense[i] &", resident)
                # The limit scans' row-type reads stay global, as in the streamed source.
                self.assertNotIn("(s_meta_dense[i] &", resident)
                self.assertEqual(
                    resident.count("(world_row_type.data[off_dense + i])"),
                    streamed.count("(world_row_type.data[off_dense + i])"),
                )
                # The float metadata streams in both.
                shared = ["s_v", "s_lam_dense"] + ([] if options.get("stream_mf_impulses") else ["s_lam_mf"])
                self.assertEqual(re.findall(r"__shared__ float\s+(\w+)", resident), shared)
                self.assertEqual(_row_metadata_read_globally(resident), streamed)
        # Patch rings walk the resident array.
        self.assertIn("return ((s_meta_dense[patch_row] >> 3) - 1);", _streamed_source(resident_row_metadata=True))
        with self.assertRaisesRegex(ValueError, "streamed metadata"):
            sweep_routes.mf_gs_source(**_LARGE_STREAMED_SHAPE, resident_row_metadata=True)

    def test_packed_row_metadata_decodes_row_types_and_parents(self):
        """Run the source's own packing and decoding on the CPU: row types and parents come back exactly.

        Every row type is packed with the boundary parents (none, the first rows, the shape's last
        rows and the largest the packing holds), and 2^16 random rows pair a random type with a
        random parent from -1 to the largest.
        """
        source = _streamed_source(resident_row_metadata=True)
        boundary = itertools.product(range(PGS_CONSTRAINT_TYPE_COUNT), (-1, 0, 1, 2302, 2303, _DENSE_META_MAX_PARENT))
        rng = np.random.default_rng(3)
        random_types = rng.integers(0, PGS_CONSTRAINT_TYPE_COUNT, 1 << 16)
        random_parents = rng.integers(-1, _DENSE_META_MAX_PARENT, 1 << 16, endpoint=True)
        pairs = np.concatenate([np.array(list(boundary)), np.stack([random_types, random_parents], axis=1)])
        types, parents = pairs.astype(np.int32).T.copy()
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
        factory = sweep_routes.mf_gs_kernel.__wrapped__
        shape = (2304, 2304, 75, "120")
        options = {"has_drive_rows": False, "has_dense_velocity_limit_rows": False, "skip_incremental_rows": 32}
        streamed = factory(*shape, shared_metadata=False, **options)
        resident = factory(*shape, shared_metadata=False, resident_row_metadata=True, **options)
        self.assertEqual(streamed.key, "sweep_mf_gs_2304_2304_75_drive0_densevlim0_gmeta_skipinc32")
        self.assertEqual(resident.key, "sweep_mf_gs_2304_2304_75_drive0_densevlim0_gmeta_smeta_skipinc32")
        self.assertEqual(max(_native_snippets(resident), key=len).count("__shared__ int s_meta_dense[2304];"), 1)
        self.assertEqual(
            list(inspect.signature(resident.func).parameters), list(inspect.signature(streamed.func).parameters)
        )
        with self.assertRaisesRegex(ValueError, "streamed metadata"):
            factory(*shape, resident_row_metadata=True, **options)

    def test_launch_replay_compares_each_launch_when_it_runs(self):
        """Compare each replayed launch with its twin before later stages rewrite the operands they share.

        Three steps launch a kernel on persistent arrays, one of them bound to two operands, and then
        rewrite those arrays with another kernel, which the replay passes through. An identical twin
        matches every launch, the shared operand included, and a twin that writes other values differs
        in that operand at every launch.
        """
        device = wp.get_device("cpu")
        for twin, differ in ((_replay_twin, []), (_replay_offset_twin, ["velocity"])):
            source = wp.array(np.arange(4, dtype=np.float32), device=device)
            velocity = wp.zeros(4, dtype=float, device=device)
            replay = _LaunchReplay(_replay_production, twin, keep=("velocity",))
            with mock.patch.object(wp, "launch_tiled", replay.wrap(wp.launch_tiled)):
                for _ in range(3):
                    wp.launch_tiled(
                        _replay_production,
                        dim=[4],
                        inputs=[source, source],
                        outputs=[velocity],
                        block_dim=32,
                        device=device,
                    )
                    wp.launch_tiled(_replay_rewrite, dim=[4], inputs=[source, velocity], block_dim=32, device=device)
            with self.subTest(twin=twin.key):
                self.assertEqual([record["differ"] for record in replay.records], [differ] * 3)
                for record in replay.records:
                    self.assertFalse(np.array_equal(record["before"]["velocity"], record["after"]["velocity"]))
                # The launches wrote through the shared operand and the rewrites ran between them: s -> 2 (s + 1) + 1.
                np.testing.assert_array_equal(source.numpy(), [21.0, 29.0, 37.0, 45.0])

    def test_streaming_factory_owns_global_impulses_on_one_lane(self):
        """Stream the uniform paired projection."""
        for shared_metadata in (False, True):
            with self.subTest(shared_metadata=shared_metadata):
                kernel = sweep_routes.mf_gs_kernel(
                    16,
                    64,
                    12,
                    "cpu",
                    shared_metadata=shared_metadata,
                    stream_mf_impulses=True,
                )
                source = max(_native_snippets(kernel), key=len)
                self.assertIn("_glambda", kernel.key)
                self.assertNotIn("__shared__ float s_lam_mf", source)
                self.assertNotIn("mf_impulses.data[off_mf + i] =", source)
                self.assertIn("READ(lane == 0 ? g_lam_mf[i] : 0.0f, 0)", source)
                for line in source.splitlines():
                    if re.search(r"g_lam_mf\[[^]]+\]\s*=(?!=)", line):
                        self.assertIn("if (lane == 0)", line)

    def test_paired_dense_impulses_have_one_owner(self):
        """Keep the paired tangent law and fenced publication in the dense projector."""
        for stream in (False, True):
            with self.subTest(stream_mf_impulses=stream):
                # The generated text (no device: staged impulses at this capacity exceed the CPU build's budget).
                snippet, _ = sweep_routes.mf_gs_source(
                    256,
                    15984,
                    3135,
                    has_drive_rows=False,
                    has_dense_velocity_limit_rows=False,
                    stream_mf_impulses=stream,
                )
                self.assertIn("READ(lane == 0 ? s_lam_dense[i] : 0.0f, 0)", snippet)
                self.assertIn("if (lane == 0) s_lam_dense[i] = 0.0f;", snippet)
                start = snippet.index("const float sibling_impulse = READ(lane == 0 ? s_lam_dense[sib] : 0.0f, 0);")
                end = snippet.index("if (sib_delta != 0.0f) {", start)
                owner = snippet[start:end]
                self.assertIn("ProjectionLaw::friction_pair(denom, cross, s_diag_dense[sib]", owner)
                self.assertIn("sib_delta = sib_new - sibling_impulse;", owner)
                # Every lane read the impulses before the first lane stores the pair, and stored before any reads.
                stores = owner[owner.index("SYNC();") :]
                self.assertIn("if (lane == 0) s_lam_dense[i] = new_impulse;", stores)
                self.assertIn("if (lane == 0) s_lam_dense[sib] = sib_new;", stores)
                self.assertTrue(stores.rstrip().endswith("SYNC();"))

    def test_limit_only_phase_does_not_stage_contact_prefix(self):
        """Keep contacts untouched and retain work for empty-segment fused clamps."""
        for fused in (False, True):
            kernel = sweep_routes.mf_gs_kernel(16, 4096, 128, "cpu", fuse_vel_limits=fused)
            source = max(_native_snippets(kernel), key=len)
            self.assertIn("int mf_load_lo = (row_phase == 5) ? mf_contact_end : 0;", source)
            # The impulses' staging and store, and the flags of the bodies in factor coordinates.
            self.assertEqual(source.count("for (int i = mf_load_lo + lane; i < m_mf; i += 32)"), 3)
            guard = "if (row_phase == 5 && dense_lo == dense_hi && mf_contact_end == m_mf) return;"
            if fused:
                self.assertNotIn(guard, source)
            else:
                self.assertIn(guard, source)
                # The guard returns before the world stages its free-body rows.
                self.assertLess(source.index(guard), source.index("for (int i = mf_load_lo + lane; i < m_mf; i += 32)"))


def _patch_ring_rows(seed, *, worlds, dense_capacity, mf_capacity, dofs, drives, velocity_limits, rings):
    """Build MF-GS operands whose dense contacts form patch rings, as ``link_patch_routes`` leaves them.

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
        "mf_Z_a": np.zeros((w_count, mf_capacity, 6), np.float32),
        "mf_Z_b": np.zeros((w_count, mf_capacity, 6), np.float32),
        "mf_row_mu": np.zeros((w_count, mf_capacity), np.float32),
        "mf_row_w": rng.uniform(0.6, 1.0, (w_count, mf_capacity)).astype(np.float32),
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
            host["mf_Z_a"][w, r] = rng.normal(0.0, 0.3, 6) * rng.uniform(0.7, 1.2)
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
        local_solve_owner=array(np.zeros(worlds, np.int32)),
        world_torsion_group=array(np.full((1, 1), -1, np.int32)),
        contact_torsion_radius=0.0,
        factors=_body_factors(host, device),
    )
    return values


def _body_factors(host, device):
    """Each world's free bodies' factors (a scaled identity per body) and their first global coordinates."""
    indices = host["world_dof_indices"]
    dof_factor = np.full(indices.size, -1, np.int32)
    lower = []
    for world, count in enumerate(host["mf_constraint_count"]):
        for row in range(count):
            packed = int(host["mf_meta"][world, 4 * row])
            for half in (packed >> 16, ((packed << 16) & 0xFFFFFFFF) >> 16):
                first = half - 65536 if half >= 32768 else half
                if first >= 0 and dof_factor[indices[world, first]] < 0:
                    dof_factor[indices[world, first]] = len(lower)
                    lower.append(np.eye(6) * (0.8 + 0.1 * (len(lower) % 5)))
    factors = BodyFactors()
    factors.dof_factor = wp.array(dof_factor, dtype=wp.int32, device=device)
    factors.L = wp.array(np.array(lower or [np.eye(6)], np.float32), dtype=wp.float32, device=device)
    return factors


@unittest.skipUnless(wp.is_cuda_available(), "MF-GS native solve requires CUDA")
class TestFeatherPGSMFStorageCUDA(unittest.TestCase):
    def test_resident_row_metadata_matches_streamed_on_patch_rings(self):
        """Solve dense patch rings with resident row types and parents: identical bits out in every row phase.

        Random patch layouts at a small and at a large dense capacity run every row
        phase, from warm and signed-zero impulses, with delayed friction, iteration offsets
        and regularization, both one world per block and through a two-block
        queue whose blocks reuse their shared memory across worlds, then once more from a captured
        graph. The drive and limit cases run the end-of-iteration limit scans (row phases 0, 2 and
        5), which read row types from global memory in both kernels, under either velocity-limit law.
        """
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("Select a CUDA device for native MF-GS execution")
        contacts = {"has_drive_rows": False, "has_dense_velocity_limit_rows": False, "skip_incremental_rows": 32}
        limits = {"has_drive_rows": True, "has_dense_velocity_limit_rows": True, "fuse_vel_limits": True}
        cases = (
            ("large_capacity", (2304, 48, 75), contacts, (220, 300)),
            ("small_capacity", (48, 12, 29), contacts, (5, 9)),
            ("drives_and_limits", (48, 12, 29), limits, (5, 9)),
            (
                "accumulated_limits",
                (48, 12, 29),
                {"has_drive_rows": True, "has_dense_velocity_limit_rows": True, "velocity_limit_law": "accumulated"},
                (5, 9),
            ),
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
                sweep_routes.mf_gs_kernel(
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
                    eager = [{output: _host(bank[output]) for output in outputs} for bank in banks]
            with wp.ScopedCapture(device=device) as capture:
                run(launches[0])
            # NaN in both banks: a graph that replayed nothing, or only the copies, cannot give the eager bits.
            for bank in banks:
                for output in outputs:
                    bank[output].fill_(float("nan"))
            wp.capture_launch(capture.graph)
            with self.subTest(case=name, graph=True):
                for bank, expected in zip(banks, eager, strict=True):
                    for output in outputs:
                        np.testing.assert_array_equal(_bits(_host(bank[output])), _bits(expected[output]), output)

    def _replay_solver_launches(self, model, options):
        """Build a solver and replay every MF-GS launch of four steps through its streamed twin.

        Returns the solver and the replay records (see :class:`_LaunchReplay`).
        """
        factory, calls = sweep_routes.mf_gs_kernel, []

        def recording(*args, **kwargs):
            kernel = factory(*args, **kwargs)
            calls.append((args, kwargs, kernel))
            return kernel

        with mock.patch.object(sweep_routes, "mf_gs_kernel", recording):
            solver = fpgs.SolverFeatherPGS(model, **options)
        production = solver._pgs_solve_mf_gs_kernel
        self.assertIn("_gmeta_smeta", production.key)
        args, kwargs = next((args, kwargs) for args, kwargs, kernel in calls if kernel is production)
        twin = factory(*args, **{**kwargs, "resident_row_metadata": False})
        self.assertEqual(twin.key, production.key.replace("_gmeta_smeta", "_gmeta", 1))
        keep = ("world_constraint_count", "world_row_type", "world_row_parent", "world_impulses")
        replay = _LaunchReplay(production, twin, keep=keep)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False, deterministic=True)
        contacts = pipeline.contacts()
        states = [model.state(), model.state()]
        control = model.control()
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        with mock.patch.object(wp, "launch_tiled", replay.wrap(wp.launch_tiled)):
            for _ in range(4):
                states[0].clear_forces()
                pipeline.collide(states[0], contacts)
                solver.step(states[0], states[1], control, contacts, 1.0 / 240.0)
                states.reverse()
        return solver, replay.records

    def test_solver_selects_resident_row_metadata_and_replays_its_launches_bitwise(self):
        """Keep row types and parents resident on a one-block-per-SM grid; its launches match the streamed twin.

        A dense row capacity of 320 streams the row metadata of a two-world patch scene (a standing
        robot, an arm resting its tip on the ground and a free box). Every MF-GS launch of four steps
        is replayed from its operands through the streamed kernel and compared bit for bit as it
        runs: the interleaved schedule's full sweeps (row phase 0) and the split schedule's limit,
        contact and velocity-limit passes (row phases 3, 4 and 5), whose contact pass stages only the
        contact rows. With fewer SMs than worlds the solver keeps the streamed kernel.
        """
        device = wp.get_device("cuda:0")
        model = _three_size_model(device)
        options = {
            "pgs_mode": "matrix_free",
            "pgs_iterations": 8,
            "enable_joint_limits": True,
            "dense_max_constraints": 320,
            "mf_max_constraints": 32,
            "mf_gs_incremental_rows": 0,
            "sparse_mass_matrix": "off",
        }
        for schedule, phases in (("interleaved", {0}), ("physx_grasp", {3, 4, 5})):
            with self.subTest(schedule=schedule):
                solver, records = self._replay_solver_launches(model, {**options, "pgs_schedule": schedule})
                sweep = solver.get_path_selection()["routes"]["sweep"]
                self.assertEqual(sweep["mf_gs"], solver._pgs_solve_mf_gs_kernel.key)
                storage = sweep["mf_gs_storage"]
                self.assertEqual((storage["metadata"], storage["row_types_and_parents"]), ("streamed", "resident"))
                self.assertIn("2-block grid fits one block per SM", storage["reason"])
                # The selection judged the grid the launches use.
                self.assertEqual(solver._mf_gs_block_count, 2)
                self.assertEqual(
                    {(*record["dim"], record["scalars"]["general_world_grid_stride"]) for record in records}, {(2, 2)}
                )
                self.assertEqual({record["scalars"]["row_phase"] for record in records}, phases)
                for index, record in enumerate(records):
                    self.assertEqual(
                        record["differ"], [], f"launch {index}, row phase {record['scalars']['row_phase']}"
                    )
                # The launches solve patch rings: some move dense impulses, and dense normal rows link rings.
                solved = sum(
                    not np.array_equal(record["before"]["world_impulses"], record["after"]["world_impulses"])
                    for record in records
                )
                self.assertGreater(solved, 0, "no replayed launch changed a dense impulse")
                rings = 0
                for record in records:
                    before = record["before"]
                    types, parents = before["world_row_type"], before["world_row_parent"]
                    for world, count in enumerate(before["world_constraint_count"]):
                        normal = types[world, :count] == PGS_CONSTRAINT_TYPE_CONTACT
                        linked = (parents[world, :count] >= 0) & (parents[world, :count] != np.arange(count))
                        rings += int(np.count_nonzero(normal & linked))
                self.assertGreater(rings, 0, "no dense normal row linked a patch ring")
        with mock.patch.object(device, "sm_count", 1):
            declined = fpgs.SolverFeatherPGS(model, **options)
        storage = declined.get_path_selection()["routes"]["sweep"]["mf_gs_storage"]
        self.assertIn("_gmeta", declined._pgs_solve_mf_gs_kernel.key)
        self.assertNotIn("_smeta", declined._pgs_solve_mf_gs_kernel.key)
        self.assertEqual((storage["metadata"], storage["row_types_and_parents"]), ("streamed", "streamed"))
        self.assertIn("2-block grid exceeds one block per SM (1 SMs)", storage["reason"])

    def test_solver_sizes_row_metadata_with_requested_streamed_impulses(self):
        """Take requested streamed MF impulses out of the block before keeping row types and parents resident.

        With 12288 matrix-free rows resident, the two-world block would declare more than 48 KiB of
        static shared memory and keeps its row types and parents streamed. Requested through
        ``FEATHER_PGS_MFGS_STREAM_LAMBDA``, the impulses stream instead and the remaining block keeps
        the row types and parents resident.
        """
        device = wp.get_device("cuda:0")
        model = _three_size_model(device)
        options = {
            "pgs_mode": "matrix_free",
            "dense_max_constraints": 320,
            "mf_max_constraints": 12288,
            "mf_gs_incremental_rows": 0,
            "sparse_mass_matrix": "off",
        }
        solvers = {}
        for request in ("0", "1"):
            with mock.patch.dict(os.environ, {"FEATHER_PGS_MFGS_STREAM_LAMBDA": request}):
                solvers[request] = fpgs.SolverFeatherPGS(model, **options)
        dofs = solvers["0"].max_world_dofs
        # The free bodies' coordinate bitmaps (sweep_routes.FreeBodyCoordinates): two words a 32 coordinates, one flag.
        bitmaps = 2 * ((dofs + 31) // 32) + 1
        if 128 + 4 * (dofs + bitmaps + 320 + 12288) > device.max_shared_memory_per_block:
            self.skipTest("The device streams these MF impulses by capacity")
        records = {request: solver.get_path_selection()["routes"]["sweep"] for request, solver in solvers.items()}
        self.assertEqual(
            (records["0"]["mf_gs_storage"]["mf_impulses"], records["0"]["mf_gs_storage"]["row_types_and_parents"]),
            ("resident", "streamed"),
        )
        self.assertIn(
            f"declare {128 + 4 * (dofs + bitmaps + 2 * 320 + 12288)} bytes", records["0"]["mf_gs_storage"]["reason"]
        )
        self.assertNotIn("_smeta", records["0"]["mf_gs"])
        self.assertNotIn("_glambda", records["0"]["mf_gs"])
        self.assertEqual(
            (records["1"]["mf_gs_storage"]["mf_impulses"], records["1"]["mf_gs_storage"]["row_types_and_parents"]),
            ("streamed", "resident"),
        )
        reason = records["1"]["mf_gs_storage"]["reason"]
        self.assertIn("MF impulses stream on request (FEATHER_PGS_MFGS_STREAM_LAMBDA)", reason)
        self.assertIn(f"declares {128 + 4 * (dofs + bitmaps + 2 * 320)} <= 49152 bytes", reason)
        self.assertIn("_gmeta_smeta_glambda", records["1"]["mf_gs"])

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
        )
        outputs = ("v_out", "world_impulses", "mf_impulses")
        for shared_metadata in (False, True):
            with self.subTest(shared_metadata=shared_metadata):
                variants = []
                for stream in (False, True):
                    kernel = sweep_routes.mf_gs_kernel(
                        8,
                        6,
                        29,
                        str(device.arch),
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
                    if phase == 0:
                        eager = [{name: _host(bound[name]) for name in outputs} for _, bound in variants]
                for _, bound in variants:
                    bound["row_phase"] = 0
                with wp.ScopedCapture(device=device) as capture:
                    run()
                # NaN in both variants: a graph that replayed nothing cannot give the eager bits.
                for _, bound in variants:
                    for name in outputs:
                        bound[name].fill_(float("nan"))
                wp.capture_launch(capture.graph)
                for (_, bound), expected in zip(variants, eager, strict=True):
                    for name in outputs:
                        np.testing.assert_array_equal(_bits(_host(bound[name])), _bits(expected[name]), name)


if __name__ == "__main__":
    unittest.main()
