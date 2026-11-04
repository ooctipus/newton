# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check capacity-safe storage selection without changing the MF-GS schedule."""

import functools
import inspect
import itertools
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
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
)
from newton._src.solvers.feather_pgs.row_metadata import _DENSE_META_MAX_PARENT
from newton._src.solvers.feather_pgs.tree_factor import WalkTrees
from newton.tests.feather_pgs_law_harness import row_lines
from newton.tests.feather_pgs_paired_fixture import pack_endpoints, solve_fixture
from newton.tests.test_feather_pgs_native_prelude import _native_snippets
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _add_chain(builder, cfg, parent, links, half_width):
    """Append revolute links given as (axis, offset) and a box foot under the last; return the joints."""
    joints = []
    for axis, offset in links:
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.05), lock_inertia=True)
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=body,
                axis=axis,
                parent_xform=wp.transform(wp.vec3(*offset), wp.quat_identity()),
                target_ke=400.0,
                target_kd=20.0,
            )
        )
        parent = body
    foot = wp.transform(wp.vec3(0.0, 0.0, -0.2), wp.quat_identity())
    builder.add_shape_box(parent, xform=foot, hx=0.06, hy=half_width, hz=0.03, cfg=cfg)
    return joints


def _add_robot(builder, cfg):
    """Stand a floating torso on two legs of three joints (12 DOFs), each box foot on its four corners at z = 0."""
    torso = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.48), wp.quat_identity()),
        mass=2.0,
        inertia=wp.mat33(np.eye(3) * 0.1),
        lock_inertia=True,
    )
    joints = [builder.add_joint_free(child=torso)]
    for side in (-1.0, 1.0):
        # Hip roll and pitch, then the knee.
        hip, knee = (0.0, 0.15 * side, 0.0), (0.0, 0.0, -0.25)
        legs = ((newton.Axis.X, hip), (newton.Axis.Y, (0.0, 0.0, 0.0)), (newton.Axis.Y, knee))
        joints += _add_chain(builder, cfg, torso, legs, 0.03)
    builder.add_articulation(joints)


def _three_size_model(device, worlds=2):
    """Per world, a standing robot (12 DOFs), a fixed arm resting its tip on the ground (2) and a free box (6): three
    size groups of at most 32 DOFs. The robot's 12 DOFs keep CUDA off the compact contact producer (up to 10 DOFs)."""
    builder = newton.ModelBuilder()
    cfg = builder.default_shape_cfg.copy()
    cfg.density, cfg.mu = 0.0, 0.8
    for _ in range(worlds):
        builder.begin_world()
        _add_robot(builder, cfg)
        # A two-joint arm on a fixed shoulder, its tip box resting on the ground.
        arm = ((newton.Axis.Y, (0.6, 0.0, 0.33)), (newton.Axis.Y, (0.0, 0.0, -0.1)))
        builder.add_articulation(_add_chain(builder, cfg, -1, arm, 0.06))
        box = builder.add_link(
            xform=wp.transform(wp.vec3(-0.6, 0.0, 0.05), wp.quat_identity()),
            mass=0.5,
            inertia=wp.mat33(np.eye(3) * 0.01),
        )
        builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
        builder.add_articulation([builder.add_joint_free(parent=-1, child=box)])
        builder.end_world()
    builder.add_ground_plane()
    return builder.finalize(device=device)


# A large streamed-metadata shape without velocity-limit rows: 2304 dense and matrix-free rows over 75 coordinates.
_LARGE_STREAMED_SHAPE = {
    "max_constraints": 2304,
    "mf_max_constraints": 2304,
    "max_world_dofs": 75,
    "has_dense_velocity_limit_rows": False,
    "skip_tier_rows": 32,
}
# Streamed-metadata option sets the resident row metadata must leave otherwise unchanged.
_STREAMED_OPTIONS = (
    {},
    {"has_dense_velocity_limit_rows": True},
    {"skip_tier_rows": 0, "contact_torsion": True},
    {"stream_mf_impulses": True},
    {"skip_local_internal_worlds": True, "local_internal_max_constraints": 19},
    {
        "has_dense_velocity_limit_rows": False,
        "factor_coordinates": True,
        "independent_components": True,
        "skip_tier_rows": 0,
    },
    {"max_constraints": 1024, "mf_max_constraints": 4096, "max_world_dofs": 604},
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
    source = re.sub(r"\n\s*SHARED int s_meta_dense\[\d+\];\n", "\n", source)
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


def _storage(shape, budget, *, block_count=16384, sm_count=188, **request):
    """The storage selection of a sweep of ``shape`` (rows, free-body rows, coordinates) under a static ``budget``."""
    staging = functools.partial(sweep_routes.mf_gs_staging, *shape)
    return execution_plan._select_mfgs_storage(staging, budget, block_count=block_count, sm_count=sm_count, **request)


class TestFeatherPGSMFStorage(unittest.TestCase):
    def test_hardware_capacity_streams_only_when_required(self):
        """Preserve all hardware rows when resident impulses exceed the device limit."""
        self.assertEqual(_storage((256, 37248, 1551), 100352)[:2], (False, True))
        self.assertEqual(_storage((256, 37248, 1551), 231424)[:2], (True, False))
        self.assertEqual(_storage((256, 15984, 3135), 100352)[:2], (True, False))

    def test_storage_boundary_and_remaining_state(self):
        """Include tile storage and reject an unrepresentable velocity working set."""
        # Four streamed metadata arrays leave velocities and two impulse arrays.
        required = 128 + 4 * (29 + 1024 + 4096)
        self.assertEqual(_storage((1024, 4096, 29), required)[:2], (False, False))
        self.assertEqual(_storage((1024, 4096, 29), required - 1)[:2], (False, True))
        with self.assertRaisesRegex(ValueError, "MF-GS.*shared memory"):
            _storage((256, 37248, 26000), 100352)

    def test_metadata_reasons_report_the_footprint_they_compared(self):
        """Report the metadata and block bytes the residency decision compared, for each outcome."""
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
                total = sweep_routes.mf_gs_staging(*shape).bytes
                metadata = total - sweep_routes.mf_gs_staging(*shape, shared_metadata=False).bytes
                storage = _storage(shape, budget)
                self.assertEqual(storage.shared_metadata, resident)
                self.assertIn(text.format(metadata=metadata, total=total), storage.reason)

    def test_single_wave_launches_keep_row_types_and_parents_resident(self):
        """Keep packed row metadata resident only for grids of one block per SM within 48 KiB of static memory."""

        def select(shape, budget, blocks, sms=188, **request):
            return _storage(shape, budget, block_count=blocks, sm_count=sms, **request)

        large = (2304, 2304, 75)
        # 128 worlds, one block each, on 188 SMs: velocities, dense and MF impulses and the packed row metadata take
        # 128 + 4 * (75 + 3 * 2304) = 28076 bytes; the float metadata streams.
        storage = select(large, 100352, 128)
        self.assertEqual(storage[:3], (False, False, True))
        self.assertIn("28076 <= 49152 bytes", storage.reason)
        self.assertEqual(select(large, 231424, 128)[:3], (False, False, True))
        # A grid of one block per SM is the largest that cannot lose concurrency to the extra bytes.
        self.assertTrue(select(large, 100352, 188).resident_row_metadata)
        for blocks in (189, 16384):
            with self.subTest(blocks=blocks):
                storage = select(large, 100352, blocks)
                self.assertEqual(storage[:3], (False, False, False))
                self.assertIn(f"{blocks}-block grid exceeds one block per SM (188 SMs)", storage.reason)
        # 48 KiB of static shared memory: 128 + 4 * (5344 + 2 * 2304 + 2304) bytes is exactly the budget.
        self.assertTrue(select((2304, 2304, 5344), 100352, 128).resident_row_metadata)
        storage = select((2304, 2304, 5345), 100352, 128)
        self.assertFalse(storage.resident_row_metadata)
        self.assertIn("49156 bytes of static shared memory, over the 49152-byte budget", storage.reason)
        # A device that exposes less than 48 KiB sets the budget.
        self.assertTrue(select(large, 28076, 128).resident_row_metadata)
        self.assertFalse(select(large, 28075, 128).resident_row_metadata)
        # Streamed MF impulses leave room: 128 + 4 * (1551 + 2 * 256) = 8380 bytes.
        storage = select((256, 37248, 1551), 100352, 32)
        self.assertEqual(storage[:3], (False, True, True))
        self.assertIn("8380 <= 49152", storage.reason)
        # Resident MF impulses do not: 22528 of them hold the block at 99088 bytes.
        self.assertEqual(select((256, 22528, 1551), 100352, 32)[:3], (False, False, False))
        # Streaming them on request takes their bytes out of the block before the budget is checked.
        storage = select((256, 22528, 1551), 100352, 32, force_stream_mf_impulses=True)
        self.assertEqual(storage[:3], (False, True, True))
        self.assertIn("MF impulses stream on request (test pin mf_storage='streamed')", storage.reason)
        self.assertIn("8380 <= 49152", storage.reason)
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
                self.assertEqual(resident.count(f"SHARED int s_meta_dense[{rows}];"), 1)
                self.assertIn("int row_type = s_meta_dense[i] &", resident)
                # The limit scans' row-type reads stay global, as in the streamed source.
                self.assertNotIn("(s_meta_dense[i] &", resident)
                self.assertEqual(
                    resident.count("(world_row_type.data[off_dense + i])"),
                    streamed.count("(world_row_type.data[off_dense + i])"),
                )
                # The float metadata streams in both.
                shared = ["s_v", "s_lam_dense"] + ([] if options.get("stream_mf_impulses") else ["s_lam_mf"])
                self.assertEqual(re.findall(r"SHARED float\s+(\w+)", resident), shared)
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
        options = {"has_dense_velocity_limit_rows": False, "skip_tier_rows": 32}
        streamed = factory(*shape, shared_metadata=False, **options)
        resident = factory(*shape, shared_metadata=False, resident_row_metadata=True, **options)
        self.assertEqual(streamed.key, "sweep_mf_gs_2304_2304_75_s75_densevlim0_gmeta_skiptier32_walk")
        self.assertEqual(resident.key, "sweep_mf_gs_2304_2304_75_s75_densevlim0_gmeta_smeta_skiptier32_walk")
        self.assertEqual(max(_native_snippets(resident), key=len).count("SHARED int s_meta_dense[2304];"), 1)
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
                self.assertNotIn("SHARED float s_lam_mf", source)
                self.assertNotIn("world_impulses.data[off_mf + i] =", source)
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
                    has_dense_velocity_limit_rows=False,
                    stream_mf_impulses=stream,
                )
                self.assertIn("READ(lane == 0 ? s_lam_dense[i] : 0.0f, 0)", snippet)
                start = snippet.index("const float sibling_impulse = READ(lane == 0 ? s_lam_dense[sib] : 0.0f, 0);")
                end = snippet.index("if (delta_impulse != 0.0f || sib_delta != 0.0f) {", start)
                owner = snippet[start:end]
                self.assertIn("ProjectionLaw::friction_pair(denom, cross, s_diag_dense[sib]", owner)
                self.assertIn("sib_delta = sib_new - sibling_impulse;", owner)
                # Every lane read the impulses before the first lane stores the pair, and stored before any reads.
                stores = owner[owner.index("SYNC();") :]
                self.assertIn("if (lane == 0) s_lam_dense[i] = new_impulse;", stores)
                self.assertIn("if (lane == 0) s_lam_dense[sib] = sib_new;", stores)
                self.assertTrue(stores.rstrip().endswith("SYNC();"))


def _patch_ring_rows(seed, *, worlds, dense_capacity, mf_capacity, dofs, velocity_limits, rings):
    """Build MF-GS operands whose dense contacts form patch rings, as ``link_patch_routes`` leaves them.

    Each world holds a prefix of position-limit rows, then
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
        "row_incident": rng.uniform(-0.1, 0.1, (w_count, capacity)).astype(np.float32),
        "world_diag": np.zeros((w_count, capacity), np.float32),
        "world_row_w": rng.uniform(0.6, 1.0, (w_count, capacity)).astype(np.float32),
        "world_impulses": np.zeros((w_count, capacity), np.float32),
        # Packed rows (packed_rows) over every coordinate: an articulated block, then the free body.
        "response_dofs": np.tile(np.array([free, 6], np.int32), w_count),
        "articulation_world_dof_offset": np.tile(np.array([0, free], np.int32), w_count),
        "packed_dense": 1,
        "packed_Z": np.zeros((w_count, capacity, dofs), np.float32),
        "row_sides": np.stack(
            [
                np.repeat(2 * np.arange(w_count, dtype=np.int32)[:, None], capacity, axis=1),
                np.repeat(2 * np.arange(w_count, dtype=np.int32)[:, None] + 1, capacity, axis=1),
            ],
            axis=-1,
        ),
        "mf_constraint_count": np.zeros(w_count, np.int32),
        "mf_meta": np.zeros((w_count, 4 * mf_capacity), np.int32),
        "mf_impulses": np.zeros((w_count, mf_capacity), np.float32),
        "mf_Z": np.zeros((w_count, mf_capacity, 12), np.float32),
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
        whitened = rng.normal(0.0, 0.3, (row, dofs)).astype(np.float32)
        host["packed_Z"][w, :row] = whitened
        host["world_diag"][w, :row] = np.sum(whitened * whitened, axis=1) + np.float32(0.01)

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
            host["mf_Z"][w, r, :6] = rng.normal(0.0, 0.3, 6) * rng.uniform(0.7, 1.2)
        host["mf_impulses"][w, :row] = np.where(rng.random(row) < 0.3, rng.uniform(0.0, 0.1, row), 0.0)
        host["mf_constraint_count"][w] = row
    host["mf_contact_rows_end"] = host["mf_constraint_count"].copy()
    host["world_dof_indices"] = np.arange(w_count * dofs, dtype=np.int32).reshape(w_count, dofs)
    return host


_STORE = {
    "world_row_type": None,
    "world_row_parent": None,
    "world_row_mu": "mf_row_mu",
    "rhs_bias": None,
    "row_incident": None,
    "world_diag": None,
    "world_row_w": "mf_row_w",
    "world_impulses": "mf_impulses",
    "row_sides": None,
}
"""The row store's records and, where the free bodies' rows hold values, their free-body arrays of the host."""


def _store(host):
    """The host's rows in one row store: each world's line holds the dense rows, then the free bodies' (the host's
    ``mf_impulses``, ``mf_row_mu``, ``mf_row_w``) from the dense capacity (:mod:`.dense_rows`)."""
    stored = dict(host)
    body = host["mf_impulses"].shape[1]
    for name, free in _STORE.items():
        dense = host[name]
        region = np.zeros((dense.shape[0], body, *dense.shape[2:]), dense.dtype) if free is None else host[free]
        stored[name] = np.concatenate((dense, region), axis=1)
    return stored


def _bind_rows(host, device):
    """Bind the operands as the MF-GS launch takes them (one row store, :func:`_store`), with neutral queue and owner
    inputs."""

    def array(value):
        dtype = wp.int32 if value.dtype == np.int32 else wp.float32
        return wp.array(np.ascontiguousarray(value), dtype=dtype, device=device)

    host = _store(host)
    worlds = len(host["world_constraint_count"])
    # Arrays bind as arrays; the scalars (packed_dense) pass as they are.
    values = {
        name: array(value) if isinstance(value, np.ndarray) else value
        for name, value in host.items()
        if name != "row_sides"
    }
    values["row_sides"] = wp.array(np.ascontiguousarray(host["row_sides"]), dtype=wp.vec2i, device=device)
    values.update(
        general_world_count=array(np.array([worlds], np.int32)),
        general_worlds=array(np.random.default_rng(5).permutation(worlds).astype(np.int32)),
        local_solve_owner=array(np.zeros(worlds, np.int32)),
        dense_first=row_lines(device),
        du=wp.zeros(host["world_dof_indices"].size, dtype=wp.float32, device=device),
    )
    values["walk"] = _walk(values["du"], _first_art(host, device))
    return values


def _first_art(host, device):
    """Every world's two articulations (packed_rows' sides), its articulated block and then its free body, found from
    their first global coordinates (:attr:`.tree_factor.WalkTrees.first_art`)."""
    indices = host["world_dof_indices"]
    worlds, dofs = indices.shape
    first_art = np.full(indices.size, -1, np.int32)
    first_art[indices[:, 0]] = 2 * np.arange(worlds)
    first_art[indices[:, dofs - 6]] = 2 * np.arange(worlds) + 1
    return wp.array(first_art, dtype=wp.int32, device=device)


def _walk(du, first_art):
    """A walk view whose ``du`` takes the sweep's velocity change in the walk's coordinates (the fixtures' free-body
    rows hold no speed row, so no tree is walked in a sweep) over the trees ``first_art`` finds."""
    walk = WalkTrees()
    walk.du = du
    walk.first_art = first_art
    return walk


@unittest.skipUnless(wp.is_cuda_available(), "MF-GS native solve requires CUDA")
class TestFeatherPGSMFStorageCUDA(unittest.TestCase):
    def test_resident_row_metadata_matches_streamed_on_patch_rings(self):
        """Solve dense patch rings with resident row types and parents: identical bits out.

        Random patch layouts at a small and at a large dense capacity run from warm and signed-zero
        impulses, with regularization, both one world per block
        and through a two-block queue whose blocks reuse their shared memory across worlds, then once
        more from a captured graph. The limit cases run the end-of-iteration limit scans, which read row
        types from global memory in both kernels.
        """
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("Select a CUDA device for native MF-GS execution")
        contacts = {"has_dense_velocity_limit_rows": False, "skip_tier_rows": 32}
        cases = (
            ("large_capacity", (2304, 48, 75), contacts, (220, 300)),
            ("small_capacity", (48, 12, 29), contacts, (5, 9)),
            ("limits", (48, 12, 29), {"has_dense_velocity_limit_rows": True}, (5, 9)),
            ("streamed_impulses", (48, 12, 29), {"stream_mf_impulses": True}, (5, 9)),
        )
        settings = (
            {"iterations": 8, "omega": 1.0},
            {"iterations": 6, "omega": 0.9, "regularize": 1},
        )
        # The velocity change the sweep leaves in the walk's coordinates (du), and the impulses.
        outputs = ("du", "world_impulses")
        for name, (dense_capacity, mf_capacity, dofs), options, rings in cases:
            host = _patch_ring_rows(
                dense_capacity + dofs,
                worlds=5,
                dense_capacity=dense_capacity,
                mf_capacity=mf_capacity,
                dofs=dofs,
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
                    bound = {**values, **bank, **launch, "walk": _walk(bank["du"], values["walk"].first_art)}
                    wp.launch_tiled(
                        kernel,
                        dim=[launch["general_world_grid_stride"]],
                        block_dim=32,
                        device=device,
                        inputs=[bound[parameter] for parameter in inspect.signature(kernel.func).parameters],
                    )

            launches = [
                {
                    "general_world_grid_stride": grid,
                    "use_general_world_queue": int(grid < 5),
                    "regularize": 0,
                    **setting,
                }
                for grid, setting in itertools.product((5, 2), settings)
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
                    impulses = banks[0]["world_impulses"].numpy()[:, :dense_capacity]
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

    def test_rows_without_a_friction_pair_read_no_sibling_diagonal(self):
        """Vote a row without a friction pair on its own change: NaN diagonals outside the worlds' rows change no bit.

        Each world holds one contact on a zero response whose warm impulse takes a nonzero step below its bound, so
        the first sweep is quiet and ends the solve. Every diagonal outside the worlds' rows is NaN, the one before
        the first world's included. The fused pair vote names the second tangent ``sib`` = -1 at such a row: reading
        its diagonal reads the row before the world's first (before the array for the first world, an illegal
        access when that address is unmapped), votes loud on NaN and keeps the worlds sweeping.
        """
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("Select a CUDA device for native MF-GS execution")
        worlds, capacity, mf_capacity, dofs = 4, 8, 4, 12
        host = _patch_ring_rows(
            0,
            worlds=worlds,
            dense_capacity=capacity,
            mf_capacity=mf_capacity,
            dofs=dofs,
            velocity_limits=False,
            rings=(1, 2),
        )
        host["world_constraint_count"][:] = 1
        host["dense_phase_bounds"][:] = 0
        host["world_row_type"][:] = PGS_CONSTRAINT_TYPE_CONTACT
        host["world_row_parent"][:] = -1
        host["packed_Z"][:] = 0.0
        host["rhs_bias"][:] = 0.0
        host["rhs_bias"][:, 0] = 1.0e-6
        host["row_incident"][:] = 0.0
        host["world_impulses"][:] = 0.0
        host["world_impulses"][:, 0] = 4.0
        host["mf_constraint_count"][:] = 0
        host["mf_contact_rows_end"][:] = 0
        values = _bind_rows(host, device)
        # One step from the warm impulse: |dlambda| A = 2^-20 against the bound 2^-20 (|b| + A |lambda|) > 2^-18.
        step = np.float32(4.0) + np.float32(-1.0e-6)
        self.assertNotEqual(step, np.float32(4.0))

        def diagonal(outside):
            """The worlds' diagonals with ``outside`` past each world's row and before the first world's (each world's
            line of the row store holds its dense rows, then its free bodies')."""
            pitch = capacity + mf_capacity
            padded = np.full(worlds * pitch + 1, outside, np.float32)
            padded[1::pitch] = 1.0
            storage = wp.array(padded, dtype=wp.float32, device=device)
            return storage, wp.array(ptr=storage.ptr + 4, shape=(worlds, pitch), dtype=wp.float32, device=device)

        guarded, finite = diagonal(np.nan), diagonal(1.0)
        # The velocity change the sweep leaves in the walk's coordinates (du), and the impulses.
        outputs = ("du", "world_impulses")
        # Streamed and resident row metadata read the diagonal from global memory; staged metadata from shared memory.
        streamed = {"shared_metadata": False}
        for metadata in (streamed, {**streamed, "resident_row_metadata": True}, {"shared_metadata": True}):
            kernel = sweep_routes.mf_gs_kernel(
                capacity,
                mf_capacity,
                dofs,
                str(device.arch),
                has_dense_velocity_limit_rows=False,
                **metadata,
            )
            for grid in (worlds, 2):
                launch = {
                    "row_phase": 0,
                    "general_world_grid_stride": grid,
                    "use_general_world_queue": int(grid < worlds),
                    "iterations": 8,
                    "omega": 1.0,
                    "regularize": 0,
                }
                results = []
                for _storage, world_diag in (guarded, finite):
                    bank = {output: wp.clone(values[output]) for output in outputs}
                    bound = {
                        **values,
                        **bank,
                        **launch,
                        "world_diag": world_diag,
                        "walk": _walk(bank["du"], values["walk"].first_art),
                    }
                    wp.launch_tiled(
                        kernel,
                        dim=[grid],
                        block_dim=32,
                        device=device,
                        inputs=[bound[parameter] for parameter in inspect.signature(kernel.func).parameters],
                    )
                    results.append({output: _host(bank[output]) for output in outputs})
                with self.subTest(**metadata, grid=grid):
                    np.testing.assert_array_equal(results[0]["world_impulses"][:, 0], np.full(worlds, step))
                    for output in outputs:
                        np.testing.assert_array_equal(_bits(results[0][output]), _bits(results[1][output]), output)

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
        production = solver.sweep.mf_gs_kernel
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
        runs. With fewer SMs than worlds the solver keeps the streamed kernel.
        """
        device = wp.get_device("cuda:0")
        model = _three_size_model(device)
        options = {
            "pgs_iterations": 8,
            "enable_joint_limits": True,
            "dense_max_constraints": 320,
            "mf_max_constraints": 32,
        }
        with mock.patch.dict(fpgs.SolverFeatherPGS._route_pins, {"storage": "dense"}):
            solver, records = self._replay_solver_launches(model, options)
            sweep = solver.get_path_selection()["routes"]["sweep"]
            self.assertEqual(sweep["mf_gs"], solver.sweep.mf_gs_kernel.key)
            storage = sweep["mf_gs_storage"]
            self.assertEqual((storage["metadata"], storage["row_types_and_parents"]), ("streamed", "resident"))
            self.assertIn("2-block grid fits one block per SM", storage["reason"])
            # The selection judged the grid the launches use.
            self.assertEqual(solver.sweep.mf_gs_block_count, 2)
            self.assertEqual(
                {(*record["dim"], record["scalars"]["general_world_grid_stride"]) for record in records}, {(2, 2)}
            )
            for index, record in enumerate(records):
                self.assertEqual(record["differ"], [], f"launch {index}")
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
        self.assertIn("_gmeta", declined.sweep.mf_gs_kernel.key)
        self.assertNotIn("_smeta", declined.sweep.mf_gs_kernel.key)
        self.assertEqual((storage["metadata"], storage["row_types_and_parents"]), ("streamed", "streamed"))
        self.assertIn("2-block grid exceeds one block per SM (1 SMs)", storage["reason"])

    def test_solver_sizes_row_metadata_with_requested_streamed_impulses(self):
        """Take requested streamed MF impulses out of the block before keeping row types and parents resident.

        With 12288 matrix-free rows resident, the two-world block would declare more than 48 KiB of
        static shared memory and keeps its row types and parents streamed. Requested through the test pin
        ``mf_storage='streamed'``, the impulses stream instead and the remaining block keeps
        the row types and parents resident.
        """
        device = wp.get_device("cuda:0")
        model = _three_size_model(device)
        options = {
            "dense_max_constraints": 320,
            "mf_max_constraints": 12288,
        }
        solvers = {}
        for request in ("0", "1"):
            pins = {"storage": "dense", **({"mf_storage": "streamed"} if request == "1" else {})}
            with mock.patch.dict(fpgs.SolverFeatherPGS._route_pins, pins):
                solvers[request] = fpgs.SolverFeatherPGS(model, **options)
        dofs = solvers["0"].forest.max_world_dofs
        if 128 + 4 * (dofs + 320 + 12288) > device.max_shared_memory_per_block:
            self.skipTest("The device streams these MF impulses by capacity")
        records = {request: solver.get_path_selection()["routes"]["sweep"] for request, solver in solvers.items()}
        self.assertEqual(
            (records["0"]["mf_gs_storage"]["mf_impulses"], records["0"]["mf_gs_storage"]["row_types_and_parents"]),
            ("resident", "streamed"),
        )
        self.assertIn(f"declare {128 + 4 * (dofs + 2 * 320 + 12288)} bytes", records["0"]["mf_gs_storage"]["reason"])
        self.assertNotIn("_smeta", records["0"]["mf_gs"])
        self.assertNotIn("_glambda", records["0"]["mf_gs"])
        self.assertEqual(
            (records["1"]["mf_gs_storage"]["mf_impulses"], records["1"]["mf_gs_storage"]["row_types_and_parents"]),
            ("streamed", "resident"),
        )
        reason = records["1"]["mf_gs_storage"]["reason"]
        self.assertIn("MF impulses stream on request (test pin mf_storage='streamed')", reason)
        self.assertIn(f"declares {128 + 4 * (dofs + 2 * 320)} <= 49152 bytes", reason)
        self.assertIn("_gmeta_smeta_glambda", records["1"]["mf_gs"])

    def test_streamed_impulses_match_resident_with_friction_and_limits(self):
        """Match mixed contact triples, warm impulses, limits and graph replay exactly."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("Select a CUDA device for native MF-GS execution")
        _, arrays = solve_fixture(device)
        values = dict(arrays)
        values.update(
            dense_first=row_lines(device),
            general_world_grid_stride=5,
            use_general_world_queue=0,
            iterations=8,
            omega=1.0,
            regularize=0,
        )
        # The velocity change the sweep leaves in the walk's coordinates (du), and the impulses.
        outputs = ("du", "world_impulses")
        for shared_metadata in (False, True):
            with self.subTest(shared_metadata=shared_metadata):
                variants = []
                for stream in (False, True):
                    kernel = sweep_routes.mf_gs_kernel(
                        8,
                        6,
                        29,
                        str(device.arch),
                        has_dense_velocity_limit_rows=False,
                        shared_metadata=shared_metadata,
                        stream_mf_impulses=stream,
                    )
                    bound = dict(values)
                    bound.update({name: wp.clone(values[name]) for name in outputs})
                    bound["walk"] = _walk(bound["du"], values["walk"].first_art)
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

                run()
                for name in outputs:
                    np.testing.assert_array_equal(variants[0][1][name].numpy(), variants[1][1][name].numpy())
                eager = [{name: _host(bound[name]) for name in outputs} for _, bound in variants]
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


def _pressed_probe(device):
    """Two worlds, each a prismatic probe pressed between a kinematic free box and a welded block: articulations
    without response coordinates (the forest plan elides the box, the block has none), so the probe's contacts are
    dense rows between a responding articulation and one without coordinates."""
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    box = scene.add_body(xform=wp.transform(wp.vec3(-0.14, 0.0, 0.0), wp.quat_identity()), mass=1.0, is_kinematic=True)
    scene.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    probe = scene.add_link(mass=1.0)
    scene.add_shape_sphere(probe, radius=0.05)
    scene.add_articulation([scene.add_joint_prismatic(parent=-1, child=probe, axis=newton.Axis.X)])
    block = scene.add_link(mass=1.0)
    scene.add_shape_box(block, hx=0.1, hy=0.1, hz=0.1)
    weld = wp.transform(wp.vec3(0.14, 0.0, 0.0), wp.quat_identity())
    scene.add_articulation([scene.add_joint_fixed(parent=-1, child=block, parent_xform=weld)])
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(scene, 2, spacing=(3.0, 0.0, 0.0))
    return builder.finalize(device=device)


def test_dense_rows_name_only_sides_with_coordinates(test, device):
    """A dense row's sides name only articulations that have response coordinates: a contact between a responding
    articulation and one without coordinates (a kinematic box, a welded block) is a row of the responding side alone,
    and the sweep, which adds the change at each side's coordinates to the walk's du, solves the rows to a finite
    step."""
    model = _pressed_probe(device)
    solver = newton.solvers.SolverFeatherPGS(model, angular_damping=0.0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    pipeline.collide(state_0, contacts)
    with mock.patch.object(solver.sweep, "solve"):
        solver.step(state_0, state_1, None, contacts, 1.0 / 240.0)

    response = solver.forest.articulation_response_dof_count.numpy()
    test.assertEqual(response.tolist(), [0, 1, 0] * 2)  # box, probe, block per world
    counts, sides = solver.placement.constraint_count.numpy(), solver.placement.row_sides.numpy()
    capacity = sides.shape[1]
    test.assertGreaterEqual(counts.min(), 2)  # each world's two contacts reach the probe through dense rows
    for world in range(model.world_count):
        rows = sides[world, : min(int(counts[world]), capacity)]
        test.assertTrue(all(response[art] > 0 for art in rows.ravel() if art >= 0))

    solver.step(state_0, state_1, None, contacts, 1.0 / 240.0)
    test.assertTrue(np.isfinite(state_1.joint_qd.numpy()).all())


class TestFeatherPGSDenseRowSides(unittest.TestCase):
    pass


add_function_test(
    TestFeatherPGSDenseRowSides,
    "test_dense_rows_name_only_sides_with_coordinates",
    test_dense_rows_name_only_sides_with_coordinates,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
