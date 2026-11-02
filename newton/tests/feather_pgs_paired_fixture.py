# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Paired articulation/free-body solve fixture shared by the matrix-free kernel tests.

The fixture is a 23+6 coordinate pair (one held robot factor and one free body
per world) with dense contact triples, matrix-free rows and a position-limit
prefix, bound as the arrays the paired-factor and matrix-free solve kernels take.
"""

import inspect

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import paired_components as components
from newton._src.solvers.feather_pgs import response_builds_rows, sweep_routes, sweep_routes_local
from newton._src.solvers.feather_pgs.kernels import BodyFactors
from newton._src.solvers.feather_pgs.tree_factor import WalkTrees
from newton.tests.feather_pgs_law_harness import row_lines


def pack_endpoints(a, b):
    """Encode the signed int16 endpoint pair without overflow casts."""
    return np.array(((int(a) & 65535) << 16) | (int(b) & 65535), dtype=np.uint32).view(np.int32).item()


def component_fixture(worlds=4, capacity=8, mf_capacity=6):
    """Build nonidentity held factors and complete dense/MF contact triples."""
    rng = np.random.default_rng(713)

    def floats(shape):
        return np.zeros(shape, np.float32)

    def integers(shape):
        return np.zeros(shape, np.int32)

    host = {
        "counts": np.full(worlds, 4, np.int32),
        "mf_counts": np.full(worlds, 3, np.int32),
        "primary_offset": np.where(np.arange(worlds) % 2, 6, 0).astype(np.int32),
        "secondary_offset": np.where(np.arange(worlds) % 2, 0, 23).astype(np.int32),
        "primary_group": np.arange(worlds - 1, -1, -1, dtype=np.int32),
        "selector": np.full(worlds, 91, np.int32),
        "J": floats((worlds, capacity, 29)),
        "Y": floats((worlds, capacity, 29)),
        "L": np.tril(rng.normal(0, 0.025, (worlds, 23, 23))).astype(np.float32),
        "rhs": floats((worlds, capacity)),
        "diag": np.ones((worlds, capacity), np.float32),
        "mu": np.full((worlds, capacity), 0.6, np.float32),
        "row_type": np.full((worlds, capacity), 3, np.int32),
        "row_parent": np.full((worlds, capacity), -1, np.int32),
        "mf_meta": integers((worlds, mf_capacity * 4)),
        "mf_mu": np.full((worlds, mf_capacity), 0.4, np.float32),
    }
    host["L"][:, np.arange(23), np.arange(23)] += 1.7
    for name in ("mf_Z_a", "mf_Z_b"):
        host[name] = floats((worlds, mf_capacity, 6))
    for world in range(worlds):
        offset, secondary = host["primary_offset"][world], host["secondary_offset"][world]
        factor = host["L"][host["primary_group"][world]].astype(np.float64)
        jacobian = rng.normal(0, 0.2, (capacity, 23)).astype(np.float32)
        host["J"][world, :, offset : offset + 23] = jacobian
        # The held factor is leaf-first: P H P^T = L L^T with P the coordinates reversed.
        mass = (factor @ factor.T)[::-1, ::-1]
        host["Y"][world, :, offset : offset + 23] = np.linalg.solve(mass, jacobian.T).T
        host["rhs"][world] = rng.normal(0, 0.2, capacity)
        host["row_type"][world, :3] = [0, 2, 2]
        host["row_parent"][world, :3] = [-1, 0, 0]
        for row in range(mf_capacity):
            endpoint_a, endpoint_b = (secondary, -1) if world % 2 == 0 else (-1, secondary)
            normal = row // 3 * 3
            kind, parent = (0, -1) if row % 3 == 0 else (2, normal)
            host["mf_meta"][world, row * 4 : row * 4 + 4] = [
                pack_endpoints(endpoint_a, endpoint_b),
                np.float32(0.8).view(np.int32),
                np.float32(0.1 * (row + 1)).view(np.int32),
                pack_endpoints(parent, kind),
            ]
            side = "a" if endpoint_a >= 0 else "b"
            # Whitened by the body's factor L = I / sqrt(0.7) (the solve fixture's secondary factor): Z = sqrt(0.7) J.
            host[f"mf_Z_{side}"][world, row] = rng.normal(0, 0.2, 6) * np.sqrt(0.7)
    return host


def bind_components(host, device="cpu"):
    """Bind the fixture as the paired-component descriptor (packed two-dimensional metadata ABI)."""
    data = components.PairedComponentData()
    data.world_count = len(host["counts"])
    for name, value in host.items():
        if name not in ("J", "L"):  # Only the whitened rows are stored; the factor is the walk's records below.
            dtype = wp.int32 if value.dtype == np.int32 else wp.float32
            setattr(data, name, wp.array(value, dtype=dtype, device=device))
    data.first = data.mf_first = row_lines(device)
    # The primary articulation of a world is its group's; its walk records hold the factor's rows (16 floats a
    # coordinate, as the walk's), so a factor entry the qualification must reject is a record entry.
    arts, n = host["L"].shape[0], host["L"].shape[1]
    data.primary_art = wp.array(host["primary_group"], dtype=wp.int32, device=device)
    data.dof_start = wp.array(np.arange(arts, dtype=np.int32) * n, dtype=wp.int32, device=device)
    data.walk = wp.array(host["L"][:, :, :16].reshape(arts * n, 16), dtype=wp.float32, device=device)
    return data


def launch_components(data, device="cpu"):
    """Run the production worker-stride qualification launch (native CPU branch on CPU)."""
    workers = min(data.world_count, 512)
    wp.launch_tiled(
        response_builds_rows.paired_components_kernel(23, 6),
        dim=[workers],
        inputs=[workers, data],
        block_dim=128,
        device=device,
    )


def solve_kernels(enabled, arch="120", *, contact_triples=False):
    """Construct the two production sweep kernels, plain or with independent components."""
    paired = sweep_routes_local.paired_factor_kernel(
        8, 29, 23, 6, arch, contact_triples=contact_triples, independent_components=enabled
    )
    mixed = sweep_routes.mf_gs_kernel(
        8,
        6,
        29,
        arch,
        factor_coordinates=True,
        has_dense_velocity_limit_rows=False,
        independent_components=enabled,
    )
    return paired, mixed


def solve_fixture(device):
    """Build disjoint hand/object, genuinely coupled fallback and no-MF worlds."""
    host = component_fixture(5)
    host["mf_counts"][3] = 0
    host["mf_counts"][1] = 4
    host["mf_meta"][1, 15] = pack_endpoints(-1, 4)
    host["counts"][4] = 0
    host["J"][2, 0, 23] = 0.35
    # The paired path owns a position-limit prefix then contact triples.
    for name in ("J", "rhs", "mu", "row_type", "row_parent"):
        host[name][:, :4] = host[name][:, [3, 0, 1, 2]]
    host["row_parent"][:, :4] = [-1, -1, 1, 1]
    # The rows the sweeps read: packed at their world coordinates and whitened (packed_rows).
    host["Y"] = _whitened(host)
    host["diag"] = np.sum(host["Y"] * host["Y"], axis=2) + np.float32(0.03)
    data = bind_components(host, device)

    def array(value):
        value = np.asarray(value)
        return wp.array(value, dtype=wp.int32 if value.dtype == np.int32 else wp.float32, device=device)

    rng = np.random.default_rng(784)
    arrays = {
        "general_world_count": array(np.array([5], np.int32)),
        "general_worlds": array(np.arange(5, dtype=np.int32)),
        "dense_phase_bounds": array(np.array([[1, 1]] * 4 + [[0, 0]], np.int32)),
        "local_solve_owner": array(np.zeros(5, np.int32)),
        "world_dof_indices": array(np.arange(145, dtype=np.int32).reshape(5, 29)),
        "row_incident": array(rng.normal(0, 0.2, (5, 8)).astype(np.float32)),
        "world_impulses": array(np.full((5, 8), 0.01, np.float32)),
        "mf_impulses": array(np.full((5, 6), 0.02, np.float32)),
        "world_row_w": array(np.ones((5, 8), np.float32)),
        "mf_row_w": array(np.ones((5, 6), np.float32)),
        "mf_contact_rows_end": array(np.full(5, 3, np.int32)),
        "primary_L": array(host["L"]),
        "secondary_L": array(np.repeat((np.eye(6) / np.sqrt(0.7))[None], 5, axis=0).astype(np.float32)),
        "primary_group_by_world": data.primary_group,
        "secondary_group_by_world": array(np.arange(5, dtype=np.int32)),
        "primary_offset_by_world": data.primary_offset,
        "secondary_offset_by_world": data.secondary_offset,
        "v_out": array(rng.normal(0, 0.2, 145).astype(np.float32)),
    }
    for target, field in (
        ("world_constraint_count", "counts"),
        ("mf_constraint_count", "mf_counts"),
        ("world_mf_constraint_count", "mf_counts"),
        ("rhs_bias", "rhs"),
        ("world_diag", "diag"),
        ("world_row_type", "row_type"),
        ("world_row_parent", "row_parent"),
        ("world_row_mu", "mu"),
        ("factor_rows", "Y"),
        ("mf_row_mu", "mf_mu"),
        ("mf_meta", "mf_meta"),
        ("mf_Z_a", "mf_Z_a"),
        ("mf_Z_b", "mf_Z_b"),
    ):
        arrays[target] = getattr(data, field)
    # Each world's free body holds the secondary factor of its world; its first global coordinate maps to it.
    factors = BodyFactors()
    dof_factor = np.full(145, -1, np.int32)
    for world in range(5):
        dof_factor[world * 29 + host["secondary_offset"][world]] = world
    factors.dof_factor = array(dof_factor)
    factors.L = arrays["secondary_L"]
    arrays["factors"] = factors
    # The sweeps add their velocity change in the walk's coordinates to the walk's du, from zero (the step's one leave
    # walks it down): the fixture's rows hold no free-body speed row, so no tree is walked in a sweep.
    walk = WalkTrees()
    walk.du = array(np.zeros(145, np.float32))
    arrays["walk"] = walk
    arrays["du"] = walk.du
    _packed_rows(host, arrays, device)
    return data, arrays


_SECONDARY_FACTOR = np.eye(6) / np.sqrt(0.7)
"""Each world's free body's factor (the solve fixture's secondary ``L``)."""


def _whitened(host):
    """Each world's rows at their world coordinates, whitened per component through its leaf-first factor: ``z =
    L^-1 P j`` in the kit's positions (:mod:`.packed_rows`' dense layout)."""
    worlds, capacity = host["J"].shape[:2]
    Z = np.zeros((worlds, capacity, 29), np.float32)
    for world in range(worlds):
        primary = host["L"][host["primary_group"][world]].astype(np.float64)
        for factor, offset, n in (
            (primary, int(host["primary_offset"][world]), 23),
            (_SECONDARY_FACTOR, int(host["secondary_offset"][world]), 6),
        ):
            j = host["J"][world, :, offset : offset + n].astype(np.float64)
            Z[world, :, offset : offset + n] = np.linalg.solve(np.tril(factor), j[:, ::-1].T).T[:, ::-1]
    return Z


def _packed_rows(host, arrays, device):
    """The dense rows packed at their world coordinates (:mod:`.packed_rows`, dense) and every articulation's factor,
    as the matrix-free sweep reads them: each world's primary (articulation ``2 w``, 23 coordinates) and its free body
    (``2 w + 1``)."""
    worlds, capacity = host["J"].shape[:2]
    sides = np.zeros((worlds, capacity, 2), np.int32)
    # Each world's articulated side (23 coordinates) and free body (6), at their world coordinates (dense rows).
    sizes = np.tile(np.array([23, 6], np.int32), worlds)
    offsets = np.zeros(2 * worlds, np.int32)
    first_art = np.full(145, -1, np.int32)
    blocks = []
    for world in range(worlds):
        primary_offset, secondary_offset = int(host["primary_offset"][world]), int(host["secondary_offset"][world])
        sides[world] = (2 * world, 2 * world + 1)
        offsets[2 * world], offsets[2 * world + 1] = primary_offset, secondary_offset
        first_art[world * 29 + primary_offset] = 2 * world
        first_art[world * 29 + secondary_offset] = 2 * world + 1
        blocks += [host["L"][host["primary_group"][world]].ravel(), _SECONDARY_FACTOR.astype(np.float32).ravel()]
    arrays["response_dofs"] = wp.array(sizes, dtype=wp.int32, device=device)
    arrays["articulation_world_dof_offset"] = wp.array(offsets, dtype=wp.int32, device=device)
    arrays["packed_dense"] = 1
    arrays["packed_Z"] = wp.array(host["Y"], dtype=wp.float32, device=device)
    arrays["row_sides"] = wp.array(sides, dtype=wp.vec2i, device=device)
    storage = wp.array(np.concatenate(blocks), dtype=wp.float32, device=device)
    starts = np.concatenate([[0], np.cumsum([block.size for block in blocks])[:-1]]).astype(np.uint64)
    factors = arrays["factors"]
    factors.first_art = wp.array(first_art, dtype=wp.int32, device=device)
    factors.size = wp.array(np.tile(np.array([23, 6], np.int32), worlds), dtype=wp.int32, device=device)
    factors.world_offset = wp.array(
        np.stack([host["primary_offset"], host["secondary_offset"]], axis=1).reshape(-1).astype(np.int32),
        dtype=wp.int32,
        device=device,
    )
    factors.address = wp.array(storage.ptr + starts * np.uint64(4), dtype=wp.uint64, device=device)
    arrays["factor_storage"] = storage  # The addresses stay valid while the arrays live.


def solve_pair_launches(data, arrays, *, enabled, arch, iterations=8):
    """The two sweeps' launches, ``(kernel, blocks, block size, arguments)``, on the fixture's data: the paired and
    mixed kernels for CUDA architecture ``arch``, each argument taken by the name the kernel's signature gives it.

    With ``enabled`` the components' selector, which :func:`launch_components` publishes, picks the worlds."""
    values = dict(arrays)
    values.update(
        dense_first=data.first,
        mf_first=data.mf_first,
        world_count=5,
        general_world_grid_stride=5,
        use_general_world_queue=0,
        iterations=iterations,
        omega=1.0,
        regularize=0,
    )
    if enabled:
        values["local_solve_owner"] = values["world_mf_constraint_count"] = data.selector
    return [
        (kernel, grid, block, [values[name] for name in inspect.signature(kernel.func).parameters])
        for kernel, grid, block in zip(
            solve_kernels(enabled, str(arch), contact_triples=True), (3, 5), (64, 32), strict=True
        )
    ]


def launch_solve_pair(data, arrays, *, enabled, device, iterations=8):
    """Solve with separate writable outputs but identical current data and the same law (CUDA: the sweeps are
    CUDA-only bodies)."""
    if enabled:
        launch_components(data, device)
    for kernel, grid, block, arguments in solve_pair_launches(
        data, arrays, enabled=enabled, arch=device.arch, iterations=iterations
    ):
        wp.launch_tiled(kernel, dim=[grid], inputs=arguments, block_dim=block, device=device)
