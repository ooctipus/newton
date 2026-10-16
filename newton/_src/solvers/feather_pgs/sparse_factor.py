# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Ancestor-sparse inverse-whitener for one identical rooted tree per world.

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
from .sparse_factorization import _sparse_cholesky_source, _sparse_inverse_source
from .sparse_mass_matrix import (
    _factor_update_schedule as _level_update_schedule,
)
from .sparse_mass_matrix import (
    _SparseMassMatrixPlan,
    _SparseRowSupportPlan,
)


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
    physical: wp.array[int]
    sparse: wp.array[int]
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


def operator_metadata(mass, *, inverse_column_width=None):
    """Describe topology-derived factor coordinates for every cached tree adapter."""
    counts = np.diff(mass.column_offsets).astype(np.int32)
    width = int(counts.max()) if inverse_column_width is None else inverse_column_width
    if width < int(counts.max()):
        raise ValueError("Inverse-column storage does not cover the factor topology")
    nodes = np.full((mass.dof_count, width), -1, np.int32)
    for col in range(mass.dof_count):
        begin, end = mass.column_offsets[col : col + 2]
        nodes[col, : end - begin] = mass.entry_rows[mass.column_entries[begin:end]]
    return {
        "index": mass.lookup,
        "row": mass.entry_rows,
        "col": mass.columns,
        "source": mass.permutation[mass.columns],
        "inverse_nodes": nodes,
        "inverse_count": counts,
        "dof_joint": mass.dof_joint,
        "physical": mass.permutation,
        "sparse": mass.inverse_permutation,
    }


def bind_operator_plan(solver, size, host, *, group_to_art=None):
    """Bind factor metadata and articulation ownership without choosing a row law.

    ``group_to_art`` names the owned articulations explicitly; by default they
    are the solver's factor group of ``size`` coordinates.
    """
    model = solver.model
    plan = SparsePlan()
    for name, values in host.items():
        setattr(plan, name, wp.array(values, dtype=wp.uint64 if name == "body_mask" else int, device=model.device))
    plan.group_to_art = solver.group_to_art[size] if group_to_art is None else group_to_art
    plan.art_to_world = solver.art_to_world
    plan.art_dof_start = solver.articulation_dof_start
    plan.art_joint_start = model.articulation_start
    plan.joint_child = model.joint_child
    return plan


def make_plan(model, solver=None):
    """Plan one identical rooted tree per world from the model's topology.

    Every world holds the same tree in body order: a six-coordinate free root
    and at most one coordinate per further joint, no kinematic bodies, at most
    64 coordinates (the endpoint masks are 64-bit words) and contact supports
    of at most 32 coordinates (one lane per support node). Contact endpoints
    are the bodies that carry collidable shapes.
    """
    worlds = int(model.world_count)
    if worlds <= 0 or model.joint_count % worlds or model.body_count % worlds or model.joint_dof_count % worlds:
        raise ValueError("Packed sparse tree requires identical worlds")
    joints, bodies, size = model.joint_count // worlds, model.body_count // worlds, model.joint_dof_count // worlds
    if joints != bodies or not 7 <= size <= 64:
        raise ValueError("Packed sparse tree requires one body per joint and 7 to 64 coordinates per world")
    parent = model.joint_parent.numpy().reshape(worlds, joints)
    child = model.joint_child.numpy().reshape(worlds, joints)
    types = model.joint_type.numpy().reshape(worlds, joints)
    qd = model.joint_qd_start.numpy()[:-1].reshape(worlds, joints)
    tree_parent = parent[0]
    offset = np.arange(worlds)[:, None] * bodies
    if (
        tree_parent[0] != -1
        or np.any(tree_parent[1:] < 0)
        or np.any(tree_parent[1:] >= np.arange(1, joints))
        or not np.array_equal(child, offset + np.arange(joints))
        or not np.array_equal(parent, np.where(tree_parent < 0, -1, offset + tree_parent))
    ):
        raise ValueError("Packed sparse tree requires one rooted tree per world in body order, identical across worlds")
    if not np.all(types == types[0]) or types[0, 0] != int(JointType.FREE):
        raise ValueError("Packed sparse tree requires a free root and identical joint types")
    counts = np.diff(model.joint_qd_start.numpy())[:joints]
    if counts[0] != 6 or np.any(counts[1:] > 1):
        raise ValueError("Packed sparse tree requires a six-coordinate free root and at most one coordinate per joint")
    if not np.all(qd - np.arange(worlds)[:, None] * size == qd[0]):
        raise ValueError("Packed sparse tree requires identical DOF order")
    if np.any(model.body_flags.numpy() & int(BodyFlags.KINEMATIC)):
        raise ValueError("Packed sparse tree excludes prescribed/kinematic bodies")
    shape_body = model.shape_body.numpy()
    collidable = (model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_SHAPES)) != 0
    contact_bodies = np.unique(shape_body[collidable & (shape_body >= 0)] % bodies)
    # The representation comes from the same planner as the native sparse route.
    mass_plan = _SparseMassMatrixPlan.build(model.joint_ancestor.numpy()[:joints], counts)
    body_masks = np.zeros(bodies, dtype=np.uint64)
    body_masks[child[0]] = mass_plan.joint_ancestor_mask
    classes = np.unique(np.concatenate((np.zeros(1, np.uint64), body_masks[contact_bodies])))
    pairs = classes[:, None] | classes[None, :]
    max_row_dofs = max(int(mask).bit_count() for mask in np.concatenate((pairs.ravel(), mass_plan.ancestor_mask)))
    if max_row_dofs > 32:
        raise ValueError("Packed sparse tree contact supports exceed one lane per coordinate")
    support = _SparseRowSupportPlan.build(
        mass_plan, body_masks, max_row_dofs=max_row_dofs, contact_bodies=contact_bodies
    )
    if support is None:
        raise ValueError("Packed sparse tree endpoint union exceeds the planned support width")
    # Jacobian consumers address mask bits in physical coordinates.
    physical_masks = np.array(
        [sum(1 << int(mass_plan.permutation[d]) for d in range(size) if int(mask) & (1 << d)) for mask in body_masks],
        dtype=np.uint64,
    )
    host = {
        **operator_metadata(mass_plan),
        "support_nodes": support.class_dof,
        "support_count": np.sum(support.class_dof >= 0, axis=1).astype(np.int32),
        "limit_support": support.limit_class,
        "pair_support": support.pair_class,
        "body_mask": np.tile(physical_masks, worlds),
        "body_tag": np.tile(support.body_class, worlds),
    }
    if solver is None:
        return host
    return bind_operator_plan(solver, size, host), host


@cache
def get_refresh_kernel(
    *,
    dof_count: int,
    nonzero_count: int,
    inverse_column_width: int,
    level_count: int,
    level_update: bool = False,
    geometric: bool = False,
):
    """Bind current or cached reverse-ordered mass inputs to generated factors."""
    source = r"""
#if defined(__CUDA_ARCH__)
    const int art = p.group_to_art.data[group];
    if (mask.data[art] == 0) return;
    const int world = p.art_to_world.data[art];
    const int ds = p.art_dof_start.data[art], js = p.art_joint_start.data[art];
    __shared__ float a[PACKED_COUNT], force[FORCE_COUNT];
    __shared__ int bad;
    if (threadIdx.x == 0) { bad = 0; d.valid.data[world] = 0; }
    for (int col = threadIdx.x; col < DOF_COUNT; col += blockDim.x) {
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
    for (int e = threadIdx.x; e < PACKED_COUNT; e += blockDim.x) {
        const int row = p.row.data[e], col = p.col.data[e];
        const int original_row = p.physical.data[row], original_col = p.physical.data[col];
        const int src = p.source.data[e];
        const int projection = src == original_col ? original_row : original_col;
        const float* screw = reinterpret_cast<const float*>(&S.data[ds + projection]);
        float value = 0.0f;
        for (int k = 0; k < 6; ++k) value += screw[k] * force[6*src+k];
        if (row == col) {
            value += R.data[group*DOF_COUNT+original_row];
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
    const auto& indices = p;
    const int lane = threadIdx.x, stride = blockDim.x;
    float* factor = a;
    SPARSE_CHOLESKY
    float* inverse = d.W.data + group*PACKED_COUNT;
    SPARSE_INVERSE_COLUMNS
    __syncthreads();
    if (threadIdx.x == 0) { if (bad) atomicOr(&d.status.data[world], 1); d.valid.data[world] = bad == 0; }
#endif
"""
    if geometric:
        # Only replace mass assembly. The held factor, inverse, R/K policy and
        # status publication below keep their original arithmetic and cadence.
        start = source.index("    for (int col = threadIdx.x; col < DOF_COUNT; col += blockDim.x) {")
        end = source.index("    for (int e = threadIdx.x; e < PACKED_COUNT; e += blockDim.x) {", start)
        source = source[:start] + source[end:]
        start = source.index("        const int src = p.source.data[e];")
        end = source.index("        if (row == col) {", start)
        source = source[:start] + "        float value = I.data[group*PACKED_COUNT+e];\n" + source[end:]
        source = source.replace(
            "__shared__ float a[PACKED_COUNT], force[FORCE_COUNT];", "__shared__ float a[PACKED_COUNT];"
        )
    for token, value in (
        ("DOF_COUNT", dof_count),
        ("PACKED_COUNT", nonzero_count),
        ("FORCE_COUNT", 6 * dof_count),
    ):
        source = source.replace(token, str(value))
    source = source.replace(
        "SPARSE_CHOLESKY",
        _sparse_cholesky_source(
            dof_count,
            nonzero_count,
            algorithm="right_looking_levels" if level_update else "right_looking",
            level_count=level_count,
        ),
    )
    source = source.replace(
        "SPARSE_INVERSE_COLUMNS",
        _sparse_inverse_source(dof_count, local_column_width=inverse_column_width, column_layout="padded"),
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
        f"sparse_factor_level_update{dof_count}_{nonzero_count}"
        if level_update
        else f"sparse_factor_refresh{dof_count}_{nonzero_count}"
    )
    if geometric:
        refresh.__name__ = refresh.__qualname__ = "packed_tree_" + refresh.__name__
    return wp.kernel(enable_backward=False, module="unique")(refresh)


@cache
def get_predictor_kernel(*, dof_count: int, nonzero_count: int, inverse_column_width: int):
    """Apply both sparse current-force actions without grouped intermediates."""
    source = r"""
#if defined(__CUDA_ARCH__)
    const int art = p.group_to_art.data[group], world = p.art_to_world.data[art];
    const int start = p.art_dof_start.data[art];
    __shared__ float whitened[DOF_COUNT];
    for (int row = threadIdx.x; row < DOF_COUNT; row += blockDim.x) {
        float value = 0.0f;
        for (int col = 0; col <= row; ++col) {
            const int entry = p.index.data[row*DOF_COUNT+col];
            if (entry >= 0) value += d.W.data[group*PACKED_COUNT+entry] * tau.data[start+p.physical.data[col]];
        }
        whitened[row] = value;
    }
    __syncthreads();
    for (int col = threadIdx.x; col < DOF_COUNT; col += blockDim.x) {
        float value = 0.0f;
        for (int k = 0; k < p.inverse_count.data[col]; ++k) {
            const int row = p.inverse_nodes.data[col*INVERSE_WIDTH+k];
            value += d.W.data[group*PACKED_COUNT+p.index.data[row*DOF_COUNT+col]] * whitened[row];
        }
        qdd.data[start+p.physical.data[col]] = d.valid.data[world] && !d.status.data[world] ? value : NAN;
    }
#endif
"""
    for token, value in (
        ("DOF_COUNT", dof_count),
        ("PACKED_COUNT", nonzero_count),
        ("INVERSE_WIDTH", inverse_column_width),
    ):
        source = source.replace(token, str(value))

    @wp.func_native(source)
    def native(group: int, p: SparsePlan, d: SparseData, tau: wp.array[float], qdd: wp.array[float]): ...

    def predictor(p: SparsePlan, d: SparseData, tau: wp.array[float], qdd: wp.array[float]):
        group, _ = wp.tid()
        native(group, p, d, tau, qdd)

    predictor.__name__ = predictor.__qualname__ = f"sparse_factor_predict{dof_count}"
    return wp.kernel(enable_backward=False, module="unique")(predictor)


def supported(solver):
    """Require every replaced stage to dispatch through this complete owner."""
    size = getattr(solver, "max_world_dofs", None)
    if size is None:
        return False
    size = int(size)
    return bool(
        solver.model.device.is_cuda
        and not solver.model.requires_grad
        and solver.world_count > 0
        and solver.model.articulation_count == solver.world_count
        and 7 <= size <= 64
        and tuple(solver.size_groups) == (size,)
        and solver.n_arts_by_size[size] == solver.world_count
        and solver.tile_threads >= 32
        and solver.tile_threads % 32 == 0
        and solver._execution_plan.use_tiled_hinv_jt(size)
        and not solver._execution_plan.use_diagonal_mass(size)
        and solver._is_one_solve_art_per_world
        and not solver._has_free_rigid_bodies
        and not solver._preelim_active
        and not solver._regularization_enabled
        and not solver._local_internal_fast_path
        and not solver._paired_factor_coordinates
        and not solver._sparse_diagonal_contact_solve
        and not solver._mf_warmstart_enabled
        and (solver._jy_world_aliased or solver._hinv_jt_writes_world)
        and solver.pgs_mode == "matrix_free"
        and solver.pgs_schedule == "interleaved"
        and solver.pgs_iterations > 0
        and solver.pgs_velocity_iterations == 0
        and not solver.pgs_warmstart
        and not solver.pgs_debug
        and solver.drive_mode == "augmented"
        and solver.friction_mode == "current"
        and not solver.enable_joint_velocity_limits
        and not solver.fuse_joint_velocity_limits
        and solver.articulated_contact_response == "immediate"
        and solver.mf_gs_response_block_rows == 0
        and solver.mf_gs_incremental_rows == 0
        and solver.mf_gs_parallel_rows == 0
        and not solver._propagation_contacts_enabled()
        and solver._execution_plan.use_tiled_cholesky(size)
        and (
            solver.trisolve_kernel == "tiled"
            or (solver.trisolve_kernel == "auto" and size > solver.small_dof_threshold)
        )
        and not solver._mimic_count
        and not solver._connect_count
        and not solver._debug_buffers_enabled
        and not solver._grouped_tau_mass
        and not solver._grouped_mass
        and getattr(solver, "_wr_world_contacts", None) is None
        and getattr(solver, "_ink_sizes", None) is None
    )


def create_owner(solver):
    """Keep unsupported constructors on the complete original representation."""
    from .solver_feather_pgs import _reject_retired_experiments  # noqa: PLC0415

    _reject_retired_experiments()
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
        from .solver_feather_pgs import _reject_retired_experiments  # noqa: PLC0415
        from .sparse_factor_rows import (  # noqa: PLC0415
            build_limit_prefix,
            get_contact_kernel,
            get_parallel_limit_kernel,
            get_solve_kernel,
        )

        _reject_retired_experiments()
        self.solver, self.plan, self.host = solver, plan, host
        w, c, device = solver.world_count, solver.dense_max_constraints, solver.model.device
        # Retain false markers consumed by checked capture readers.
        self.packet_rows = False
        self.body_basis_rows = False
        self.paired_gs = False
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
        if self.spectral_tangents and (not self.metric_tangents or self.block_contacts):
            raise ValueError("Sparse spectral tangents require the original capacity100 metric owner")
        if self.metric_tangents and self.block_contacts:
            raise ValueError("Sparse metric tangents and contact-block rows are mutually exclusive")
        self.level_update = os.environ.get("FEATHER_PGS_SPARSE_LEVEL_UPDATE") == "1"
        schedule = _level_update_schedule(host["index"])
        self.factor_parameters = {
            "dof_count": host["index"].shape[0],
            "nonzero_count": len(host["row"]),
            "inverse_column_width": host["inverse_nodes"].shape[1],
            "level_count": len(schedule["level_offsets"]) - 1,
        }
        if self.level_update:
            for name, values in schedule.items():
                setattr(plan, name, wp.array(values, dtype=int, device=solver.model.device))
        self.size = host["index"].shape[0]
        self.support_width = host["support_nodes"].shape[1]
        self.data = SparseData()
        self.data.W = wp.empty((w, self.factor_parameters["nonzero_count"]), dtype=float, device=device)
        self.data.valid = wp.zeros(w, dtype=int, device=device)
        self.data.status = wp.zeros(w, dtype=int, device=device)
        # Prefix emission reads capacity from its buffers and preserves overflow counts.
        self.parallel_limit_prefix = os.environ.get("FEATHER_PGS_SPARSE_PARALLEL_LIMITS") == "1"
        self.data.Z = wp.empty((w, c, self.support_width), dtype=float, device=device)
        self.data.support = wp.empty((w, c), dtype=int, device=device)
        self.data.incident = wp.empty((w, c), dtype=float, device=device)
        dimensions = {
            "dof_count": self.size,
            "support_width": self.support_width,
            "inverse_width": self.factor_parameters["inverse_column_width"],
            "nonzero_count": self.factor_parameters["nonzero_count"],
        }
        self.kernels = SimpleNamespace(
            prefix=get_parallel_limit_kernel(
                dof_count=self.size, support_width=self.support_width, nonzero_count=dimensions["nonzero_count"]
            )
            if self.parallel_limit_prefix
            else build_limit_prefix,
            refresh=get_refresh_kernel(**self.factor_parameters, level_update=self.level_update),
            predictor=get_predictor_kernel(
                dof_count=self.size,
                nonzero_count=dimensions["nonzero_count"],
                inverse_column_width=dimensions["inverse_width"],
            ),
            contacts=get_contact_kernel(
                dof_count=self.size,
                support_width=self.support_width,
                factor_stride=self.data.W.shape[1],
            ),
            solve=get_solve_kernel(c, self.block_contacts, metric_tangents=self.metric_tangents, **dimensions),
        )
        if self.spectral_tangents:
            from .sparse_spectral_tangents import get_solve_kernel as spectral_solve  # noqa: PLC0415

            self.kernels.solve = spectral_solve(**dimensions)
        if self.limit_jacobi:
            from .sparse_limit_jacobi import get_solve_kernel as limit_solve  # noqa: PLC0415

            self.kernels.solve = limit_solve(**dimensions)
        # Drop canonical matrix/row storage only after complete constructor admission.
        # Dummy shapes make accidental readers fail visibly, not reinterpret packed W/Z.
        dummy = wp.empty((1, 1, 1), dtype=float, device=device)
        for name in ("H_by_size", "L_by_size", "J_by_size", "Y_by_size", "tau_by_size", "qdd_by_size"):
            getattr(solver, name)[self.size] = dummy
        solver.J_world = dummy
        solver.Y_world = dummy
        solver._H_bufs = None
        solver._J_bufs = None
        solver._memset_stream = None
        self.kinetic_state = None

    def install_kinetic_state(self):
        """Admit the complete state producer only after solver construction."""
        from .packed_tree_owner import PackedTreeState  # noqa: PLC0415
        from .packed_tree_owner import supported as kinetic_supported  # noqa: PLC0415

        if kinetic_supported(self.solver):
            self.kinetic_state = PackedTreeState(self)
            self.kernels.refresh = get_refresh_kernel(
                **self.factor_parameters, level_update=self.level_update, geometric=True
            )

    def begin(self):
        """Reject changed execution ownership before any retired buffer is read."""
        if not supported(self.solver):
            raise RuntimeError(
                "Packed sparse tree configuration changed; reconstruct the solver and recapture before stepping"
            )
        if self.kinetic_state is not None:
            from .packed_tree_owner import supported as kinetic_supported  # noqa: PLC0415

            if not kinetic_supported(self.solver):
                raise RuntimeError("Packed tree-state ownership changed; reconstruct and recapture before stepping")

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
                raise RuntimeError(
                    "Packed sparse tree structural ownership changed; reconstruct and recapture"
                ) from error
            if self.solver._mimic_count or self.solver._connect_count:
                raise RuntimeError("Packed sparse tree constraint ownership changed; reconstruct and recapture")
        if self.kinetic_state is not None:
            self.kinetic_state.validate_model()

    def check(self):
        """Surface a failed packed operator/row guard at existing checked boundaries."""
        status = self.data.status.numpy()
        if np.any(status):
            raise RuntimeError(
                f"Packed sparse tree representation guard failed: {np.unique(status[status != 0]).tolist()}"
            )
        if self.kinetic_state is not None:
            status = self.kinetic_state.status.numpy()
            if np.any(status):
                raise RuntimeError(f"Packed tree-state guard failed: {np.unique(status[status != 0]).tolist()}")

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
                s.R_by_size[self.size],
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
        s.dense_contact_world_flag.zero_()
        s._dense_first_rejected_slot.fill_(2147483647)
        s._row_dropped_all.zero_()
        tiled_prefix = self.parallel_limit_prefix
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
                self.size,
                self.support_width,
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
            drops = [s._row_dropped_dense, s._row_dropped_mf, s._row_dropped_propagation]
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
                    int(s.contact_friction_articulation_pairs_only),
                    1,
                    s._friction_patches.view,
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
                    s._dense_first_rejected_slot,
                    dummy,
                    dummy,
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
                    int(s.contact_friction_shared_anchor),
                    s.contact_friction_scale,
                    int(s.contact_shared_anchor),
                    s.pgs_beta,
                    s.pgs_cfm,
                    s._friction_patches.view,
                    s.friction_anchor_beta,
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
            inputs=[s.slot_counter, c, 0, s._dense_first_rejected_slot],
            outputs=[s.constraint_count, s._constraint_capacity_status],
            device=device,
        )

    def restitution(self, dt):
        """Use the original current incident trigger after original bias construction."""
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
