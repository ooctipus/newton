# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Paired articulation/free-body solve fixture shared by the matrix-free kernel tests.

The fixture is a 23+6 coordinate pair (one held robot factor and one free body
per world) with dense contact triples, matrix-free rows and a position-limit
prefix, bound as the arrays the paired-factor and matrix-free solve kernels take.
"""

from types import SimpleNamespace

import numpy as np
import warp as wp


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
    for name in ("mf_J_a", "mf_J_b", "mf_MiJt_a", "mf_MiJt_b"):
        host[name] = floats((worlds, mf_capacity, 6))
    for world in range(worlds):
        offset, secondary = host["primary_offset"][world], host["secondary_offset"][world]
        factor = host["L"][host["primary_group"][world]].astype(np.float64)
        jacobian = rng.normal(0, 0.2, (capacity, 23)).astype(np.float32)
        host["J"][world, :, offset : offset + 23] = jacobian
        host["Y"][world, :, offset : offset + 23] = np.linalg.solve(factor @ factor.T, jacobian.T).T
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
            host[f"mf_J_{side}"][world, row] = rng.normal(0, 0.2, 6)
            host[f"mf_MiJt_{side}"][world, row] = host[f"mf_J_{side}"][world, row] * 0.7
    return host


def bind_components(host, device="cpu"):
    """Upload the fixture as device arrays with the packed two-dimensional metadata ABI."""
    data = SimpleNamespace(world_count=len(host["counts"]))
    for name, value in host.items():
        setattr(data, name, wp.array(value, dtype=wp.int32 if value.dtype == np.int32 else wp.float32, device=device))
    return data


def solve_fixture(device):
    """Build disjoint hand/object, genuinely coupled fallback and no-MF worlds."""
    host = component_fixture(5)
    host["mf_counts"][3] = 0
    host["mf_counts"][1] = 4
    host["mf_meta"][1, 15] = pack_endpoints(-1, 4)
    host["counts"][4] = 0
    host["J"][2, 0, 23] = 0.35
    host["Y"][2, 0, 23] = 0.245
    host["diag"] = np.sum(host["J"] * host["Y"], axis=2) + np.float32(0.03)
    # The paired path owns a position-limit prefix then contact triples.
    for name in ("J", "Y", "rhs", "diag", "mu", "row_type", "row_parent"):
        host[name][:, :4] = host[name][:, [3, 0, 1, 2]]
    host["row_parent"][:, :4] = [-1, -1, 1, 1]
    # Paired row production already stores factor rows when MF is absent.
    for world in (3,):
        offset, group = host["primary_offset"][world], host["primary_group"][world]
        host["Y"][world, :, offset : offset + 23] = host["Y"][world, :, offset : offset + 23] @ host["L"][group]
    data = bind_components(host, device)

    def array(value):
        value = np.asarray(value)
        return wp.array(value, dtype=wp.int32 if value.dtype == np.int32 else wp.float32, device=device)

    rng = np.random.default_rng(784)
    arrays = {
        "general_world_count": array(np.array([5], np.int32)),
        "general_worlds": array(np.arange(5, dtype=np.int32)),
        "rb_class": array(np.zeros(5, np.int32)),
        "dense_phase_bounds": array(np.array([[1, 1]] * 4 + [[0, 0]], np.int32)),
        "local_solve_owner": array(np.zeros(5, np.int32)),
        "world_dof_indices": array(np.arange(145, dtype=np.int32).reshape(5, 29)),
        "world_deferred_dof_mask": array(np.zeros((5, 29), np.int32)),
        "world_impulses": array(np.full((5, 8), 0.01, np.float32)),
        "world_torsion_group": array(np.full((1, 1), -1, np.int32)),
        "contact_torsion_radius": 0.0,
        "mf_impulses": array(np.full((5, 6), 0.02, np.float32)),
        "world_row_w": array(np.ones((5, 8), np.float32)),
        "mf_row_w": array(np.ones((5, 6), np.float32)),
        "mf_contact_rows_end": array(np.full(5, 3, np.int32)),
        "primary_L": data.L,
        "primary_Linv": array(np.linalg.inv(host["L"].astype(np.float64)).astype(np.float32)),
        "secondary_L": array(np.repeat((np.eye(6) / np.sqrt(0.7))[None], 5, axis=0).astype(np.float32)),
        "secondary_Linv": array(np.repeat((np.eye(6) * np.sqrt(0.7))[None], 5, axis=0).astype(np.float32)),
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
        ("J_world", "J"),
        ("Y_world", "Y"),
        ("factor_rows", "Y"),
        ("mf_row_mu", "mf_mu"),
        ("mf_meta", "mf_meta"),
        ("mf_J_a", "mf_J_a"),
        ("mf_J_b", "mf_J_b"),
        ("mf_MiJt_a", "mf_MiJt_a"),
        ("mf_MiJt_b", "mf_MiJt_b"),
    ):
        arrays[target] = getattr(data, field)
    for name in ("target_vel_bias", "vel_multiplier", "impulse_multiplier", "max_impulse", "vel_limit"):
        arrays[f"world_drive_{name}"] = array(np.zeros((1, 1), np.float32))
    return data, arrays
