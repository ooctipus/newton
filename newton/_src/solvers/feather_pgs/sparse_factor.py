# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Default-off ancestor-sparse inverse-whitener for the checked G1 tree.

The held operator, current rows and solver have one representation owner.
Neither a dense canonical factor nor physical response rows are produced on
the admitted path. Unsupported constructors keep the original implementation.
"""

import os
from functools import cache
from types import SimpleNamespace

import numpy as np
import warp as wp

from ...geometry.flags import ShapeFlags
from ...sim import BodyFlags, JointType


@wp.struct
class SparsePlan:
    index: wp.array2d[int]
    row: wp.array[int]
    col: wp.array[int]
    source: wp.array[int]
    level_offsets: wp.array2d[int]
    level_pivots: wp.array[int]
    level_scales: wp.array[int]
    level_entries: wp.array[int]
    level_term_offsets: wp.array[int]
    level_terms: wp.array2d[int]
    inverse_nodes: wp.array2d[int]
    inverse_count: wp.array[int]
    support_nodes: wp.array2d[int]
    support_count: wp.array[int]
    limit_support: wp.array[int]
    pair_support: wp.array2d[int]
    body_mask: wp.array[wp.uint64]
    body_tag: wp.array[int]
    dof_joint: wp.array[int]
    group_to_art: wp.array[int]
    art_to_world: wp.array[int]
    art_dof_start: wp.array[int]
    art_joint_start: wp.array[int]
    joint_child: wp.array[int]


@wp.struct
class SparseData:
    W: wp.array2d[float]
    valid: wp.array[int]
    status: wp.array[int]
    Z: wp.array3d[float]
    support: wp.array2d[int]
    incident: wp.array2d[float]


def _level_update_schedule(index):
    """Compile exact right-looking Schur ownership for the admitted topology."""
    if index.shape != (43, 43):
        raise ValueError("Sparse level updates require the checked 43-coordinate pattern")
    return _level_update_schedule_cached(tuple(index.ravel()))


@cache
def _level_update_schedule_cached(pattern):
    """Share immutable exact-size schedule data across model notifications."""
    index = np.asarray(pattern, dtype=np.int32).reshape(43, 43)
    rows, cols = np.nonzero(index >= 0)
    if len(rows) != 434 or not np.array_equal(index[rows, cols], np.arange(434)) or np.any(rows < cols):
        raise ValueError("Sparse level updates require the checked packed lower triangle")
    levels = np.zeros(43, np.int32)
    for col in range(43):
        previous = np.flatnonzero(index[col, :col] >= 0)
        levels[col] = 0 if not len(previous) else 1 + max(levels[previous])
    if np.any(np.diag(index) < 0) or max(levels) != 14:
        raise ValueError("Sparse level dependency depth differs from the checked tree")
    offsets, pivots, scales, entries, starts, terms = [(0, 0, 0)], [], [], [], [0], []
    for level in range(15):
        current = np.flatnonzero(levels == level)
        pivots.extend(current)
        scales.extend(np.flatnonzero((rows > cols) & (levels[cols] == level)))
        for entry, (row, col) in enumerate(zip(rows, cols, strict=True)):
            if levels[col] <= level:
                continue
            update = [(index[row, k], index[col, k]) for k in current if index[row, k] >= 0 and index[col, k] >= 0]
            if update:
                entries.append(entry)
                terms.extend(update)
                starts.append(len(terms))
        offsets.append((len(pivots), len(scales), len(entries)))
    if offsets[-1] != (43, 391, 1142) or len(terms) != 2242:
        raise ValueError("Sparse level update work differs from the checked tree")
    result = {
        "level_offsets": np.asarray(offsets, dtype=np.int32),
        "level_pivots": np.asarray(pivots, dtype=np.int32),
        "level_scales": np.asarray(scales, dtype=np.int32),
        "level_entries": np.asarray(entries, dtype=np.int32),
        "level_term_offsets": np.asarray(starts, dtype=np.int32),
        "level_terms": np.asarray(terms, dtype=np.int32),
    }
    for values in result.values():
        values.setflags(write=False)
    return result


def make_plan(model, solver=None):
    """Prove the complete tree and possible dynamic endpoint support once."""
    worlds = int(model.world_count)
    if worlds <= 0 or (model.joint_count, model.joint_dof_count, model.body_count) != (
        44 * worlds,
        43 * worlds,
        44 * worlds,
    ):
        raise ValueError("Sparse G1 requires one complete 44-body/43-DOF tree per world")
    parent = model.joint_parent.numpy().reshape(worlds, 44)
    child = model.joint_child.numpy().reshape(worlds, 44)
    types = model.joint_type.numpy().reshape(worlds, 44)
    qd = model.joint_qd_start.numpy()[:-1].reshape(worlds, 44)
    expected_parent = np.array(
        [
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            0,
            0,
            8,
            9,
            10,
            11,
            12,
            0,
            14,
            14,
            14,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            22,
            25,
            22,
            27,
            28,
            14,
            14,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            36,
            39,
            36,
            41,
            42,
        ]
    )
    offset = np.arange(worlds)[:, None] * 44
    if not np.array_equal(child, offset + np.arange(44)) or not np.array_equal(
        parent, np.where(expected_parent < 0, -1, offset + expected_parent)
    ):
        raise ValueError("Sparse G1 body topology differs from the checked tree")
    if not np.all(types == types[0]) or types[0, 0] != int(JointType.FREE):
        raise ValueError("Sparse G1 requires a free root and identical joint types")
    fixed = {7, 15, 16, 22, 30, 36}
    for j in range(1, 44):
        if types[0, j] != int(JointType.FIXED if j in fixed else JointType.REVOLUTE):
            raise ValueError("Sparse G1 only supports the checked fixed/revolute tree")
    if not np.all(qd - np.arange(worlds)[:, None] * 43 == qd[0]):
        raise ValueError("Sparse G1 requires identical DOF order")
    if np.any(model.body_flags.numpy() & int(BodyFlags.KINEMATIC)):
        raise ValueError("Sparse G1 excludes prescribed/kinematic bodies")
    shape_body = model.shape_body.numpy()
    collidable = (model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_SHAPES)) != 0
    if np.any(~np.isin(shape_body[collidable & (shape_body >= 0)] % 44, [6, 13, 14])):
        raise ValueError("Sparse G1 only admits feet/torso collision participants")
    owned = [list(range(int(qd[0, j]), int(qd[0, j]) + (6 if j == 0 else 0 if j in fixed else 1))) for j in range(44)]
    ancestors = []
    for j in range(44):
        path, k = [], j
        while k >= 0:
            path.extend(owned[k])
            k = int(expected_parent[k])
        ancestors.append(sorted(path))
    pattern = np.eye(43, dtype=bool)
    dof_joint = np.empty(43, np.int32)
    for j, dofs in enumerate(owned):
        pattern[np.ix_(dofs, ancestors[j])] = True
        pattern[np.ix_(ancestors[j], dofs)] = True
        dof_joint[dofs] = j
    pattern = pattern[::-1, ::-1]
    rr, cc = np.nonzero(np.tril(pattern))
    if len(rr) != 434:
        raise ValueError("Unexpected G1 packed factor pattern")
    index = np.full((43, 43), -1, np.int32)
    index[rr, cc] = np.arange(434)
    inverse_nodes = np.full((43, 18), -1, np.int32)
    inverse_count = np.zeros(43, np.int32)
    for k in range(43):
        nodes = np.flatnonzero(np.tril(pattern)[:, k])
        inverse_nodes[k, : len(nodes)] = nodes
        inverse_count[k] = len(nodes)
    templates = []

    def add(support):
        nodes = tuple(sorted(42 - d for d in support))
        if len(nodes) > 18:
            raise ValueError("Current endpoint union exceeds the proved 18 coordinates")
        if nodes not in templates:
            templates.append(nodes)
        return templates.index(nodes)

    limit_support = np.array([add(ancestors[dof_joint[d]]) for d in range(43)], np.int32)
    endpoints = [set(), set(ancestors[6]), set(ancestors[13]), set(ancestors[14])]
    pair_support = np.array([[add(a | b) for b in endpoints] for a in endpoints], np.int32)
    support_nodes = np.full((len(templates), 18), -1, np.int32)
    for i, nodes in enumerate(templates):
        support_nodes[i, : len(nodes)] = nodes
    masks = np.array([sum(1 << d for d in support) for support in ancestors], np.uint64)
    tags = np.full(44, -1, np.int32)
    tags[[6, 13, 14]] = [1, 2, 3]
    host = {
        "index": index,
        "row": rr.astype(np.int32),
        "col": cc.astype(np.int32),
        "source": np.maximum(42 - rr, 42 - cc).astype(np.int32),
        "inverse_nodes": inverse_nodes,
        "inverse_count": inverse_count,
        "support_nodes": support_nodes,
        "support_count": np.array([len(x) for x in templates], np.int32),
        "limit_support": limit_support,
        "pair_support": pair_support,
        "body_mask": np.tile(masks, worlds),
        "body_tag": np.tile(tags, worlds),
        "dof_joint": dof_joint,
    }
    # Deeper physical DOF owns the composite inertia (same original source map).
    if solver is None:
        return host
    plan = SparsePlan()
    for name, values in host.items():
        setattr(plan, name, wp.array(values, dtype=wp.uint64 if name == "body_mask" else int, device=model.device))
    plan.group_to_art = solver.group_to_art[43]
    plan.art_to_world = solver.art_to_world
    plan.art_dof_start = solver.articulation_dof_start
    plan.art_joint_start = model.articulation_start
    plan.joint_child = model.joint_child
    return plan, host


@cache
def get_refresh_kernel(level_update=False, geometric=False):
    """Build the packed held inverse-whitener directly from current CRBA inputs."""
    source = r"""
#if defined(__CUDA_ARCH__)
    const int art = p.group_to_art.data[group];
    if (mask.data[art] == 0) return;
    const int world = p.art_to_world.data[art];
    const int ds = p.art_dof_start.data[art], js = p.art_joint_start.data[art];
    __shared__ float a[434], force[258];
    __shared__ int bad;
    if (threadIdx.x == 0) { bad = 0; d.valid.data[world] = 0; }
    for (int col = threadIdx.x; col < 43; col += blockDim.x) {
        const int body = p.joint_child.data[js + p.dof_joint.data[col]];
        const float* inertia = reinterpret_cast<const float*>(&I.data[body]);
        const float* screw = reinterpret_cast<const float*>(&S.data[ds + col]);
        for (int r = 0; r < 6; ++r) {
            float value = 0.0f;
            for (int c = 0; c < 6; ++c) value += inertia[6*r+c] * screw[c];
            force[6*col+r] = value;
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 434; e += blockDim.x) {
        const int row = p.row.data[e], col = p.col.data[e];
        const int original_row = 42-row, original_col = 42-col;
        const int src = p.source.data[e];
        const int projection = src == original_col ? original_row : original_col;
        const float* screw = reinterpret_cast<const float*>(&S.data[ds + projection]);
        float value = 0.0f;
        for (int k = 0; k < 6; ++k) value += screw[k] * force[6*src+k];
        if (row == col) {
            value += R.data[group*43+original_row];
            int drive = drive_row.data[ds+original_row];
            if (drive < 0 && !parallel_drives) {
                // Serial drive preparation compacts only its current rows.
                // Its all-minus-one immutable map is not a no-drive predicate.
                for (int r = 0; r < drive_counts.data[art]; ++r) {
                    const int entry = art * drive_stride + r;
                    if (drive_dofs.data[entry] == ds + original_row) { drive = entry; break; }
                }
            }
            if (drive >= 0 && K.data[drive] > 0.0f) value += K.data[drive];
        }
        a[e] = value;
    }
    __syncthreads();
    for (int k = 0; k < 43; ++k) {
        const int diag = p.index.data[k*43+k];
        if (threadIdx.x == 0) {
            if (!(a[diag] > 0.0f) || !isfinite(a[diag])) bad = 1;
            a[diag] = sqrtf(a[diag]);
        }
        __syncthreads();
        for (int row = k+1+threadIdx.x; row < 43; row += blockDim.x) {
            const int entry = p.index.data[row*43+k];
            if (entry >= 0) a[entry] /= a[diag];
        }
        __syncthreads();
        for (int e = threadIdx.x; e < 434; e += blockDim.x) {
            const int row = p.row.data[e], col = p.col.data[e];
            if (col <= k) continue;
            const int x = p.index.data[row*43+k], y = p.index.data[col*43+k];
            if (x >= 0 && y >= 0) a[e] -= a[x] * a[y];
        }
        __syncthreads();
    }
    for (int col = threadIdx.x; col < 43; col += blockDim.x) {
        float values[18];
        const int count = p.inverse_count.data[col];
        for (int i = 0; i < count; ++i) {
            const int row = p.inverse_nodes.data[col*18+i];
            float value = i == 0 ? 1.0f : 0.0f;
            for (int j = 0; j < i; ++j) {
                const int prev = p.inverse_nodes.data[col*18+j];
                value -= a[p.index.data[row*43+prev]] * values[j];
            }
            value /= a[p.index.data[row*43+row]];
            values[i] = value;
            if (!isfinite(value)) atomicExch(&bad, 1);
            d.W.data[group*434+p.index.data[row*43+col]] = value;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) { if (bad) atomicOr(&d.status.data[world], 1); d.valid.data[world] = bad == 0; }
#endif
"""
    if geometric:
        # Only replace mass assembly. The held factor, inverse, R/K policy and
        # status publication below keep their original arithmetic and cadence.
        start = source.index("    for (int col = threadIdx.x; col < 43; col += blockDim.x) {")
        end = source.index("    for (int e = threadIdx.x; e < 434; e += blockDim.x) {", start)
        source = source[:start] + source[end:]
        start = source.index("        const int src = p.source.data[e];")
        end = source.index("        if (row == col) {", start)
        source = source[:start] + "        float value = I.data[group*434+e];\n" + source[end:]
        source = source.replace("__shared__ float a[434], force[258];", "__shared__ float a[434];")
    if level_update:
        start = source.index("    for (int k = 0; k < 43; ++k) {")
        end = source.index("    for (int col = threadIdx.x; col < 43; col += blockDim.x) {", start)
        source = (
            source[:start]
            + r"""
    // Columns within a level are independent, but their ancestor Schur
    // destinations overlap. One destination owner gathers this level only.
    for (int level = 0; level < 15; ++level) {
        const int begin = 3*level, end = 3*(level+1);
        for (int task = p.level_offsets.data[begin]+threadIdx.x;
             task < p.level_offsets.data[end]; task += blockDim.x) {
            const int k = p.level_pivots.data[task];
            const int diag = p.index.data[k*43+k];
            if (!(a[diag] > 0.0f) || !isfinite(a[diag])) atomicExch(&bad, 1);
            a[diag] = sqrtf(a[diag]);
        }
        __syncthreads();
        for (int task = p.level_offsets.data[begin+1]+threadIdx.x;
             task < p.level_offsets.data[end+1]; task += blockDim.x) {
            const int entry = p.level_scales.data[task];
            const int col = p.col.data[entry];
            a[entry] /= a[p.index.data[col*43+col]];
        }
        __syncthreads();
        for (int task = p.level_offsets.data[begin+2]+threadIdx.x;
             task < p.level_offsets.data[end+2]; task += blockDim.x) {
            const int entry = p.level_entries.data[task];
            float value = a[entry];
            for (int term = p.level_term_offsets.data[task];
                 term < p.level_term_offsets.data[task+1]; ++term) {
                value -= a[p.level_terms.data[2*term]] * a[p.level_terms.data[2*term+1]];
            }
            a[entry] = value;
        }
        __syncthreads();
    }
"""
            + source[end:]
        )

    inertia_type = wp.array2d[float] if geometric else wp.array[wp.spatial_matrix]

    @wp.func_native(source)
    def native(
        group: int,
        p: SparsePlan,
        d: SparseData,
        mask: wp.array[int],
        S: wp.array[wp.spatial_vector],
        I: inertia_type,
        R: wp.array2d[float],
        drive_row: wp.array[int],
        K: wp.array[float],
        drive_counts: wp.array[int],
        drive_dofs: wp.array[int],
        drive_stride: int,
        parallel_drives: int,
    ): ...

    def refresh(
        p: SparsePlan,
        d: SparseData,
        mask: wp.array[int],
        S: wp.array[wp.spatial_vector],
        I: inertia_type,
        R: wp.array2d[float],
        drive_row: wp.array[int],
        K: wp.array[float],
        drive_counts: wp.array[int],
        drive_dofs: wp.array[int],
        drive_stride: int,
        parallel_drives: int,
    ):
        group, _ = wp.tid()
        native(group, p, d, mask, S, I, R, drive_row, K, drive_counts, drive_dofs, drive_stride, parallel_drives)

    refresh.__name__ = refresh.__qualname__ = (
        "sparse_factor_level_update43_434" if level_update else "sparse_factor_refresh43_434"
    )
    if geometric:
        refresh.__name__ = refresh.__qualname__ = "g1_kinetic_" + refresh.__name__
    return wp.kernel(enable_backward=False, module="unique")(refresh)


@cache
def get_predictor_kernel():
    """Apply both sparse current-force actions without grouped intermediates."""
    source = r"""
#if defined(__CUDA_ARCH__)
    const int art = p.group_to_art.data[group], world = p.art_to_world.data[art];
    const int start = p.art_dof_start.data[art];
    __shared__ float whitened[43];
    for (int row = threadIdx.x; row < 43; row += blockDim.x) {
        float value = 0.0f;
        for (int col = 0; col <= row; ++col) {
            const int entry = p.index.data[row*43+col];
            if (entry >= 0) value += d.W.data[group*434+entry] * tau.data[start+42-col];
        }
        whitened[row] = value;
    }
    __syncthreads();
    for (int col = threadIdx.x; col < 43; col += blockDim.x) {
        float value = 0.0f;
        for (int k = 0; k < p.inverse_count.data[col]; ++k) {
            const int row = p.inverse_nodes.data[col*18+k];
            value += d.W.data[group*434+p.index.data[row*43+col]] * whitened[row];
        }
        qdd.data[start+42-col] = d.valid.data[world] && !d.status.data[world] ? value : NAN;
    }
#endif
"""

    @wp.func_native(source)
    def native(group: int, p: SparsePlan, d: SparseData, tau: wp.array[float], qdd: wp.array[float]): ...

    def predictor(p: SparsePlan, d: SparseData, tau: wp.array[float], qdd: wp.array[float]):
        group, _ = wp.tid()
        native(group, p, d, tau, qdd)

    predictor.__name__ = predictor.__qualname__ = "sparse_factor_predict43"
    return wp.kernel(enable_backward=False, module="unique")(predictor)


def supported(solver):
    """Require every replaced stage to dispatch through this complete owner."""
    from .single_factor import supported as original_supported  # noqa: PLC0415

    return bool(
        original_supported(solver)
        and solver._execution_plan.use_tiled_cholesky(43)
        and (
            solver.trisolve_kernel == "tiled" or (solver.trisolve_kernel == "auto" and 43 > solver.small_dof_threshold)
        )
        and not solver._mimic_count
        and not solver._connect_count
        and solver._joint_world is None
        and solver._row_packets is None
        and not solver._debug_buffers_enabled
        and not solver._grouped_tau_mass
        and not solver._grouped_mass
        and getattr(solver, "_wr_world_contacts", None) is None
        and getattr(solver, "_ink_sizes", None) is None
    )


def create_owner(solver):
    """Keep unsupported constructors on the complete original representation."""
    if os.environ.get("FEATHER_PGS_SINGLE_FACTOR") == "1":
        raise ValueError("Sparse factor and single-factor owners are mutually exclusive")
    if not supported(solver):
        return None
    try:
        plan, host = make_plan(solver.model, solver)
    except ValueError:
        return None
    return SparseFactor(solver, plan, host)


class SparseFactor:
    """Own every consumer of the selected packed held/current representation."""

    def __init__(self, solver, plan, host):
        from .sparse_factor_rows import (  # noqa: PLC0415
            build_limit_prefix,
            get_contact_kernel,
            get_parallel_limit_kernel,
            get_solve_kernel,
        )

        self.solver, self.plan, self.host = solver, plan, host
        w, c, device = solver.world_count, solver.dense_max_constraints, solver.model.device
        self.packet_rows = os.environ.get("FEATHER_PGS_SPARSE_PACKETS") == "1" and c == 100
        self.block_contacts = os.environ.get("FEATHER_PGS_SPARSE_CONTACT_BLOCK") == "1"
        self.metric_tangents = os.environ.get("FEATHER_PGS_SPARSE_METRIC_TANGENTS") == "1" and c == 100
        spectral = os.environ.get("FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS", "0")
        if spectral not in ("0", "1"):
            raise ValueError("FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS must be 0 or 1")
        self.spectral_tangents = spectral == "1" and c == 100
        limit_jacobi = os.environ.get("FEATHER_PGS_SPARSE_LIMIT_JACOBI", "0")
        if limit_jacobi not in ("0", "1"):
            raise ValueError("FEATHER_PGS_SPARSE_LIMIT_JACOBI must be 0 or 1")
        self.limit_jacobi = limit_jacobi == "1"
        if self.limit_jacobi and not self.spectral_tangents:
            raise ValueError("Sparse limit Jacobi requires the capacity100 spectral owner")
        if self.spectral_tangents and (
            not self.metric_tangents
            or self.packet_rows
            or self.block_contacts
            or os.environ.get("FEATHER_PGS_SPARSE_PAIRED_GS") == "1"
        ):
            raise ValueError("Sparse spectral tangents require the original capacity100 metric owner")
        if self.metric_tangents and (self.packet_rows or self.block_contacts):
            raise ValueError("Sparse metric tangents, packets and contact-block rows are mutually exclusive")
        self.level_update = os.environ.get("FEATHER_PGS_SPARSE_LEVEL_UPDATE") == "1"
        if self.level_update:
            for name, values in _level_update_schedule(host["index"]).items():
                setattr(plan, name, wp.array(values, dtype=int, device=solver.model.device))
        self.data = SparseData()
        self.data.W = wp.empty((w, 434), dtype=float, device=device)
        self.data.valid = wp.zeros(w, dtype=int, device=device)
        self.data.status = wp.zeros(w, dtype=int, device=device)
        if self.packet_rows and self.block_contacts:
            raise ValueError("Sparse packets and contact-block rows are mutually exclusive")
        self.parallel_limit_prefix = os.environ.get("FEATHER_PGS_SPARSE_PARALLEL_LIMITS") == "1" and c == 100
        if self.packet_rows and self.parallel_limit_prefix:
            raise ValueError("Sparse packets and parallel global limit rows are mutually exclusive")
        self.data.Z = wp.empty((1, 1, 1) if self.packet_rows else (w, c, 18), dtype=float, device=device)
        self.data.support = wp.empty((w, c), dtype=int, device=device)
        self.data.incident = wp.empty((1, 1) if self.packet_rows else (w, c), dtype=float, device=device)
        self.kernels = SimpleNamespace(
            prefix=get_parallel_limit_kernel() if self.parallel_limit_prefix else build_limit_prefix,
            refresh=get_refresh_kernel(self.level_update),
            predictor=get_predictor_kernel(),
            contacts=get_contact_kernel(),
            solve=get_solve_kernel(c, self.block_contacts, metric_tangents=self.metric_tangents),
        )
        if self.spectral_tangents:
            from .sparse_spectral_tangents import get_solve_kernel as spectral_solve  # noqa: PLC0415

            self.kernels.solve = spectral_solve()
        if self.limit_jacobi:
            from .sparse_limit_jacobi import get_solve_kernel as limit_solve  # noqa: PLC0415

            self.kernels.solve = limit_solve()
        # Drop canonical matrix/row storage only after complete constructor admission.
        # Dummy shapes make accidental readers fail visibly, not reinterpret packed W/Z.
        dummy = wp.empty((1, 1, 1), dtype=float, device=device)
        for name in ("H_by_size", "L_by_size", "J_by_size", "Y_by_size", "tau_by_size", "qdd_by_size"):
            getattr(solver, name)[43] = dummy
        solver.J_world = dummy
        solver.Y_world = dummy
        solver._H_bufs = None
        solver._J_bufs = None
        solver._memset_stream = None
        if self.packet_rows:
            from .sparse_packet_rows import install  # noqa: PLC0415

            install(self)
        self.kinetic_state = None
        self.body_basis_rows = False
        if os.environ.get("FEATHER_PGS_BODY_BASIS_ROWS") == "1":
            from .body_basis_rows import install as install_body_basis_rows  # noqa: PLC0415

            install_body_basis_rows(self)
        self.paired_gs = False
        if os.environ.get("FEATHER_PGS_SPARSE_PAIRED_GS") == "1":
            from .sparse_paired_gs import install as install_paired_gs  # noqa: PLC0415

            install_paired_gs(self)

    def install_kinetic_state(self):
        """Admit the complete state producer only after solver construction."""
        from .g1_kinetic_state import G1KineticState  # noqa: PLC0415
        from .g1_kinetic_state import supported as kinetic_supported  # noqa: PLC0415

        if kinetic_supported(self.solver):
            self.kinetic_state = G1KineticState(self)
            self.kernels.refresh = get_refresh_kernel(self.level_update, geometric=True)

    def begin(self):
        """Reject changed execution ownership before any retired buffer is read."""
        if not supported(self.solver):
            raise RuntimeError("Sparse G1 configuration changed; reconstruct the solver and recapture before stepping")
        if self.kinetic_state is not None:
            from .g1_kinetic_state import supported as kinetic_supported  # noqa: PLC0415

            if not kinetic_supported(self.solver):
                raise RuntimeError("G1 kinetic-state ownership changed; reconstruct and recapture before stepping")

    def validate_notification(self, flags):
        """Re-prove structural ownership on model notifications before writes."""
        from ...sim import ModelFlags  # noqa: PLC0415

        if flags & (
            ModelFlags.BODY_PROPERTIES
            | ModelFlags.JOINT_PROPERTIES
            | ModelFlags.SHAPE_PROPERTIES
            | ModelFlags.CONSTRAINT_PROPERTIES
        ):
            try:
                make_plan(self.solver.model)
            except ValueError as error:
                raise RuntimeError("Sparse G1 structural ownership changed; reconstruct and recapture") from error
            if self.solver._mimic_count or self.solver._connect_count:
                raise RuntimeError("Sparse G1 constraint ownership changed; reconstruct and recapture")
        if self.kinetic_state is not None:
            self.kinetic_state.validate_model()

    def check(self):
        """Surface a failed packed operator/row guard at existing checked boundaries."""
        status = self.data.status.numpy()
        if np.any(status):
            raise RuntimeError(f"Sparse G1 representation guard failed: {np.unique(status[status != 0]).tolist()}")
        if self.kinetic_state is not None:
            status = self.kinetic_state.status.numpy()
            if np.any(status):
                raise RuntimeError(f"G1 kinetic-state guard failed: {np.unique(status[status != 0]).tolist()}")

    def refresh(self, state_aug):
        """Publish only the original requested held generation."""
        s = self.solver
        wp.launch_tiled(
            self.kernels.refresh,
            dim=[s.world_count],
            inputs=[
                self.plan,
                self.data,
                s.mass_update_mask,
                state_aug.joint_S_s,
                s.body_I_c if self.kinetic_state is None else self.kinetic_state.geometric,
                s.R_by_size[43],
                s._augmented_drive_row_by_dof,
                s.aug_row_K,
                s.aug_row_counts,
                s.aug_row_dof_index,
                s.articulation_max_dofs,
                int(s._parallel_augmented_drive_topology),
            ],
            block_dim=128,
            device=s.model.device,
        )

    def predict(self, state_aug):
        """Read current canonical force and write original generalized acceleration."""
        s = self.solver
        wp.launch_tiled(
            self.kernels.predictor,
            dim=[s.world_count],
            inputs=[self.plan, self.data, state_aug.joint_tau, state_aug.joint_qdd],
            block_dim=64,
            device=s.model.device,
        )

    def build_rows(self, state_in, state_aug, contacts, dt):
        """Retain original allocation/metadata while replacing all dense J production."""
        from . import kernels as k  # noqa: PLC0415

        s, model, device = self.solver, self.solver.model, self.solver.model.device
        c = s.dense_max_constraints
        if self.packet_rows:
            from .sparse_packet_rows import bind_current  # noqa: PLC0415

            bind_current(self, state_in, state_aug, contacts, dt)
        s.dense_contact_world_flag.zero_()
        if s._row_watermark:
            s._row_dropped_dense.zero_()
            s._row_dropped_mf.zero_()
            s._row_dropped_propagation.zero_()
        tiled_prefix = self.packet_rows or self.parallel_limit_prefix
        prefix_launch = wp.launch_tiled if tiled_prefix else wp.launch
        prefix_launch(
            self.kernels.prefix,
            dim=[s.world_count] if tiled_prefix else s.world_count,
            inputs=[
                self.plan,
                self.data,
                s._joint_limit_q_index,
                model.joint_limit_lower,
                model.joint_limit_upper,
                state_in.joint_q,
                s.v_hat,
                int(s.enable_joint_limits),
                s.joint_limit_activation_gap,
                s.pgs_beta,
                s.pgs_cfm,
                s.slot_counter,
                s.row_type,
                s.row_parent,
                s.row_mu,
                s.row_beta,
                s.row_cfm,
                s.phi,
                s.target_velocity,
                s.diag,
                s.dense_phase_bounds,
            ],
            block_dim=32 if tiled_prefix else 256,
            device=device,
        )
        if contacts is not None and contacts.rigid_contact_max > 0:
            threads = min(contacts.rigid_contact_max, 65536)
            is_free = s.is_free_rigid if s.is_free_rigid is not None else s._dummy_is_free_rigid
            dummy = s._dummy_mf_slot_counter
            drops = (
                [s._row_dropped_dense, s._row_dropped_mf, s._row_dropped_propagation]
                if s._row_watermark
                else [dummy, dummy, dummy]
            )
            wp.launch(
                k.allocate_world_contact_slots,
                dim=threads,
                inputs=[
                    contacts.rigid_contact_count,
                    threads,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_margin0,
                    contacts.rigid_contact_margin1,
                    state_in.body_q,
                    model.shape_transform,
                    model.shape_body,
                    s.body_to_articulation,
                    s.art_to_world,
                    s.articulation_response_dof_count,
                    model.body_flags,
                    s.body_has_response_dofs,
                    is_free,
                    0,
                    0,
                    int(s.propagation_same_articulation_rows),
                    0,
                    s.contact_gap_gate,
                    s.same_articulation_contact_gap_gate,
                    s.articulation_pair_contact_gap_gate,
                    c,
                    s.mf_max_constraints,
                    s.propagation_max_constraints,
                    int(s.enable_contact_friction),
                    s.contact_friction_gap_threshold,
                    s.contact_friction_anchor_limit,
                    int(s.contact_friction_articulation_pairs_only),
                    int(s._row_watermark),
                    s._resolved_simple_worlds,
                ],
                outputs=[
                    s.contact_world,
                    s.contact_slot,
                    s.contact_art_a,
                    s.contact_art_b,
                    s.slot_counter,
                    s.contact_path,
                    dummy,
                    dummy,
                    s.dense_contact_world_flag,
                    s.contact_slots_needed,
                    *drops,
                    s._constraint_capacity_status,
                ],
                device=device,
            )
            wp.launch(
                k.prepare_world_contact_rows,
                dim=threads,
                inputs=[
                    contacts.rigid_contact_count,
                    threads,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_margin0,
                    contacts.rigid_contact_margin1,
                    s.contact_world,
                    s.contact_slot,
                    s.contact_art_a,
                    s.contact_art_b,
                    s.contact_path,
                    s.contact_slots_needed,
                    model.shape_body,
                    state_in.body_q,
                    state_aug.body_v_s,
                    s._prescribed_articulation,
                    s.articulation_origin,
                    s.shape_material_mu,
                    s.shape_material_restitution,
                    int(s.enable_contact_friction),
                    s.contact_friction_gap_threshold,
                    int(s.contact_friction_shared_anchor),
                    s.contact_friction_anchor_limit,
                    int(s.contact_friction_articulation_pairs_only),
                    is_free,
                    s.contact_friction_scale,
                    int(s.contact_shared_anchor),
                    s.pgs_beta,
                    s.pgs_cfm,
                ],
                outputs=[
                    s.row_type,
                    s.row_parent,
                    s.row_mu,
                    s.row_beta,
                    s.row_cfm,
                    s.phi,
                    s.target_velocity,
                    s.row_restitution,
                ],
                device=device,
            )
            if self.packet_rows:
                wp.launch(
                    self.kernels.contacts,
                    dim=threads,
                    inputs=[
                        contacts.rigid_contact_count,
                        threads,
                        s.contact_path,
                        s.contact_slot,
                        s.contact_world,
                        s.contact_slots_needed,
                        self.data.support,
                    ],
                    device=device,
                )
            else:
                workers = min(contacts.rigid_contact_max, 16384)
                wp.launch_tiled(
                    self.kernels.contacts,
                    dim=[workers],
                    inputs=[
                        workers,
                        self.plan,
                        self.data,
                        contacts.rigid_contact_count,
                        s.contact_path,
                        s.contact_slot,
                        s.contact_world,
                        s.contact_art_a,
                        s.contact_art_b,
                        s.contact_slots_needed,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_margin0,
                        contacts.rigid_contact_margin1,
                        model.shape_body,
                        state_in.body_q,
                        state_aug.joint_S_s,
                        s.articulation_origin,
                        s.art_group_idx,
                        s.v_hat,
                        int(s.contact_shared_anchor),
                        int(s.contact_friction_shared_anchor),
                        s.row_cfm,
                        s.diag,
                    ],
                    block_dim=32,
                    device=device,
                )
        wp.launch(
            k.finalize_constraint_counts_with_status,
            dim=s.world_count,
            inputs=[s.slot_counter, c, 0],
            outputs=[s.constraint_count, s._constraint_capacity_status],
            device=device,
        )

    def restitution(self, dt):
        """Use the original current incident trigger after original bias construction."""
        if self.packet_rows:
            self.packet_input.dt = dt
            return  # Applied from the current incident during local row formation.
        from .sparse_factor_rows import apply_restitution  # noqa: PLC0415

        s = self.solver
        wp.launch(
            apply_restitution,
            dim=(s.world_count, s.dense_max_constraints),
            inputs=[
                self.data,
                s.constraint_count,
                s.row_type,
                s.phi,
                s.target_velocity,
                s.row_restitution,
                dt,
                s._effective_restitution_velocity_threshold,
                s.rhs,
            ],
            device=s.model.device,
        )

    def solve(self, rhs, iterations, omega, friction_start):
        """Visit every original row and return complete physical velocity once."""
        if self.packet_rows:
            from .sparse_packet_rows import solve  # noqa: PLC0415

            solve(self, rhs, iterations, omega, friction_start)
            return
        s = self.solver
        inputs = [
            self.plan,
            self.data,
            s.constraint_count,
            rhs,
            s.diag,
            s.impulses,
            s.row_type,
            s.row_parent,
            s.row_mu,
            iterations,
            omega,
            friction_start,
            s.v_hat,
            s.v_out,
        ]
        if self.spectral_tangents:
            inputs.insert(5, s.row_cfm)
        wp.launch_tiled(
            self.kernels.solve,
            dim=[s.world_count],
            inputs=inputs,
            block_dim=32,
            device=s.model.device,
        )
