# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""FeatherPGS's sweep routes as configurations of the sweep emitter (:mod:`.sweep`).

Each function here builds one route's kernel from the emitter's parts and takes the arguments of the factory it
replaces, with that factory's kernel ABI, so a launch site can take either kernel unchanged.
"""

from __future__ import annotations

from functools import cache

import warp as wp

from . import kernel_template, sweep_units
from .contact_torsion import torque_sweep_source
from .device_preludes import RING, cuda_native, lane_group
from .kernel_template import KernelTemplate
from .kernels import PGS_LOCAL_SOLVE_OWNER_GENERAL
from .law_source import law_preludes
from .projection_laws import PROJECTION_LAW
from .response_operator import Block, DenseJY, FreeBody6, Lanes, Velocity
from .row_metadata import _DENSE_META_ROW_TYPE_MASK, _validate_dense_metadata_encoding
from .solve_policy import EXACT_POLICY, SolvePolicy, SolvePolicyUnsupported
from .staging import STATIC_SHARED_BYTES, Buffer, Staging, static_budget
from .sweep import (
    CONTACT,
    DRIVE,
    DRIVE_DATA,
    FRICTION,
    JOINT_LIMIT,
    VELOCITY_LIMIT,
    FreeBodyRows,
    RowContext,
    RowLaws,
    WorldRows,
    change_vote,
    dense_rows_block,
    end_sweep,
    free_body_rows_block,
    fused_clamp_pass,
    indent,
    row_visit,
    speed_rows_pass,
    velocity_limit_pass,
)
from .velocity_limit_law import check_velocity_limit_law, velocity_limit_law_suffix

# Dense row types each row phase visits (MF-GS ``row_phase``): 0 interleaved (every row; velocity limits in the end
# pass), 1/2 contact-then-internal, 3/4/5 the PhysX grasp schedule (internal, contacts, velocity limits), 6 dense
# rows only.
_MIMIC, _CONNECT = 5, 6
_PHASE_DENSE_KINDS = {
    1: (CONTACT, FRICTION),
    2: (DRIVE, JOINT_LIMIT, VELOCITY_LIMIT, _MIMIC, _CONNECT),
    3: (DRIVE, JOINT_LIMIT, _MIMIC, _CONNECT),
    4: (CONTACT, FRICTION),
    5: (VELOCITY_LIMIT,),
}


def _phase_admission() -> str:
    lines = []
    for phase, kinds in _PHASE_DENSE_KINDS.items():
        test = " && ".join(f"row_type != {kind}" for kind in kinds)
        lines.append(f"if (row_phase == {phase} && {test}) continue;")
    lines.append(f"if ((row_phase == 0 || row_phase == 2 || row_phase == 6) && row_type == {VELOCITY_LIMIT}) continue;")
    lines.append(f"if (freeze_drive_rows != 0 && row_type == {DRIVE}) continue;")
    return "\n".join(lines)


_MF_PHASE_ADMISSION = (
    f"if ((row_phase == 1 || row_phase == 4) && mf_rt != {CONTACT} && mf_rt != {FRICTION}) continue;\n"
    f"if (row_phase == 0 && mf_rt == {VELOCITY_LIMIT}) continue;"
)

MF_GS_ABI = kernel_template.params(
    ("world", int),
    ("world_constraint_count", wp.array[int]),
    ("dense_phase_bounds", wp.array2d[int]),
    ("local_solve_owner", wp.array[int]),
    ("world_dof_indices", wp.array2d[int]),
    ("rhs_bias", wp.array2d[float]),
    ("world_diag", wp.array2d[float]),
    ("world_row_w", wp.array2d[float]),
    ("world_impulses", wp.array2d[float]),
    ("J_world", wp.array3d[float]),
    ("Y_world", wp.array3d[float]),
    ("world_row_type", wp.array2d[int]),
    ("world_row_parent", wp.array2d[int]),
    ("world_row_mu", wp.array2d[float]),
    ("world_torsion_group", wp.array2d[int]),
    ("contact_torsion_radius", float),
    ("world_drive_target_vel_bias", wp.array2d[float]),
    ("world_drive_vel_multiplier", wp.array2d[float]),
    ("world_drive_impulse_multiplier", wp.array2d[float]),
    ("world_drive_max_impulse", wp.array2d[float]),
    ("world_drive_vel_limit", wp.array2d[float]),
    ("mf_constraint_count", wp.array[int]),
    ("mf_contact_rows_end", wp.array[int]),
    ("mf_meta", wp.array2d[int]),
    ("mf_impulses", wp.array2d[float]),
    ("mf_J_a", wp.array3d[float]),
    ("mf_J_b", wp.array3d[float]),
    ("mf_MiJt_a", wp.array3d[float]),
    ("mf_MiJt_b", wp.array3d[float]),
    ("mf_row_mu", wp.array2d[float]),
    ("mf_row_w", wp.array2d[float]),
    ("iterations", int),
    ("omega", float),
    ("regularize", int),
    ("row_phase", int),
    ("friction_start_iteration", int),
    ("iteration_offset", int),
    ("freeze_drive_rows", int),
    ("v_out", wp.array[float]),
)
"""The matrix-free sweep's native ABI: one world per call (the world queue's dispatch)."""


def mf_gs_source(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    friction_mode: str = "current",
    *,
    shared_metadata: bool = True,
    resident_row_metadata: bool = False,
    stream_mf_impulses: bool = False,
    has_drive_rows: bool = True,
    has_dense_velocity_limit_rows: bool = True,
    fuse_vel_limits: bool = False,
    skip_local_internal_worlds: bool = False,
    contact_torsion: bool = False,
    factor_coordinates: bool = False,
    skip_incremental_rows: int = 0,
    independent_components: bool = False,
    velocity_limit_law: str = "stateless",
    solve_policy: SolvePolicy = EXACT_POLICY,
) -> tuple[str, Staging]:
    """The native body of the matrix-free Gauss-Seidel sweep: a world's dense rows (:class:`.DenseJY`) and its
    free-body rows (:class:`.FreeBody6`) in one warp, then its end-of-sweep passes.

    Returns the body and its staging.
    """
    if resident_row_metadata and shared_metadata:
        raise ValueError("resident_row_metadata applies to streamed metadata (shared_metadata=False)")
    if friction_mode != "current":
        raise NotImplementedError(f"friction_mode={friction_mode!r} is not an emitter law")
    if fuse_vel_limits and not has_drive_rows:
        raise ValueError("fuse_vel_limits requires has_drive_rows")
    if factor_coordinates and (
        has_drive_rows or has_dense_velocity_limit_rows or fuse_vel_limits or skip_local_internal_worlds
    ):
        raise ValueError("factor coordinates require the paired augmented-drive contact solve")
    if independent_components and not factor_coordinates:
        raise ValueError("Independent components require the paired factor-coordinate solve")
    if (not solve_policy.exact or solve_policy.traces) and contact_torsion:
        raise SolvePolicyUnsupported(
            "The velocity exit law and trace cover the current-friction physical sweep without torsion"
        )
    _validate_dense_metadata_encoding(max_constraints)
    accumulated = check_velocity_limit_law(velocity_limit_law) == "accumulated"
    M_D, M_MF, D = max_constraints, mf_max_constraints, max_world_dofs
    lanes = Lanes(32)
    laws = RowLaws(
        drive=has_drive_rows,
        velocity_limit_law="accumulated" if accumulated else "stateless",
        regularize=True,
        torsion=contact_torsion,
        policy=solve_policy,
    )
    dense = WorldRows(
        M_D,
        shared_metadata=shared_metadata,
        packed_types=shared_metadata or resident_row_metadata,
        drive=has_drive_rows,
        fused_limits=fuse_vel_limits,
        lanes=lanes,
    )
    free = FreeBodyRows(M_MF, stream=stream_mf_impulses, lanes=lanes)
    dense_op = DenseJY(D, lanes)
    free_op = FreeBody6(lanes)
    staging = Staging((Buffer("s_v", "float", D), *dense.buffers(), *free.buffers()), threads=32)

    # Ownership: the paired owner's signed selector arrives in local_solve_owner under independent components.
    dense_count = "int m_dense = world_constraint_count.data[world];"
    component = owned = ""
    if independent_components:
        dense_count = (
            "const bool split_component = local_solve_owner.data[world] < 0;\n"
            "    int m_dense = split_component ? 0 : world_constraint_count.data[world];"
        )
        component = (
            "\n    const int first_dofs = split_component ? mf_meta.data[off_meta] : 0;"
            "\n    const int component_offset = split_component ? "
            "max(first_dofs >> 16, static_cast<int>(static_cast<short>(first_dofs & 65535))) : 0;"
        )
        owned = " && (!split_component || (d >= component_offset && d < component_offset + 6))"
    skips = []
    if factor_coordinates:
        skips.append("    if (m_mf == 0) return;")
    if skip_local_internal_worlds:
        skips.append(f"    if (local_solve_owner.data[world] != {PGS_LOCAL_SOLVE_OWNER_GENERAL}) return;")
    if skip_incremental_rows > 0:
        skips.append(f"    if (m_mf == 0 && m_dense <= {int(skip_incremental_rows)}) return;")
    load_lo = "    int load_lo = (row_phase == 5) ? 0 : dense_lo;" if fuse_vel_limits else ""
    empty_limit_phase = (
        "" if fuse_vel_limits else "    if (row_phase == 5 && dense_lo == dense_hi && mf_contact_end == m_mf) return;"
    )
    dense_loads = indent(dense.loads("i"), "        ")
    mf_load = free.load_line("i")
    mf_store = free.store_line("i")

    passes = []
    if has_dense_velocity_limit_rows:
        passes.append(
            "        if (row_phase == 0 || row_phase == 2) {\n"
            + velocity_limit_pass(dense, dense_op, laws, pad="            ")
            + "\n        }"
        )
    if fuse_vel_limits:
        passes.append(
            "        if (row_phase == 0 || row_phase == 2 || row_phase == 5) {\n"
            + fused_clamp_pass(dense, dense_op, laws, pad="            ")
            + "\n        }"
        )
    passes.append(
        "        if (row_phase == 0 || row_phase == 2 || row_phase == 5) {\n"
        + speed_rows_pass(free, free_op, laws, pad="            ")
        + "\n        }"
    )
    if contact_torsion:
        passes.append(torque_sweep_source(D))

    sweep_end = end_sweep(
        solve_policy,
        iteration="iter",
        exact="""        if (global_iter >= friction_start_iteration) {
            unsigned changed = BALLOT(iteration_changed != 0);
            if (changed == 0u) break;
        }
""",
    )
    body = f"""    const int lane = LANE;

    {dense_count}
    int m_mf = mf_constraint_count.data[world];
    if (m_dense == 0 && m_mf == 0) return;
    if (m_dense > {M_D}) m_dense = {M_D};
    if (m_mf > {M_MF}) m_mf = {M_MF};
{chr(10).join(skips)}
    int mf_contact_end = mf_contact_rows_end.data[world];
    if (mf_contact_end > m_mf) mf_contact_end = m_mf;

    // Row phases own contiguous dense ranges: 3 [0, bounds[0]), 5 [bounds[0], bounds[1]), 4 [bounds[1], m).
    int dense_lo = 0;
    int dense_hi = m_dense;
    if (row_phase == 3) {{
        dense_hi = min(dense_phase_bounds.data[world * 2 + 0], m_dense);
    }} else if (row_phase == 5) {{
        dense_lo = min(dense_phase_bounds.data[world * 2 + 0], m_dense);
        dense_hi = min(dense_phase_bounds.data[world * 2 + 1], m_dense);
    }} else if (row_phase == 4) {{
        dense_lo = min(dense_phase_bounds.data[world * 2 + 1], m_dense);
    }}
{load_lo}
{empty_limit_phase}

    int dof_map_base = world * {D};
    int off_dense = world * {M_D};
    int off_mf = world * {M_MF};
    int off_meta = off_mf * 4;{component}
    int jy_world_base = world * {M_D} * {D};
    int mf6_base = world * {M_MF} * 6;
    int mf_load_lo = (row_phase == 5) ? mf_contact_end : 0;

{staging.declarations("    ")}
{free.declaration()}

    for (int i = {"load_lo" if fuse_vel_limits else "dense_lo"} + lane; i < dense_hi; i += 32) {{
{dense_loads}
    }}
    if (row_phase != 3 && row_phase != 6) {{
        for (int i = mf_load_lo + lane; i < m_mf; i += 32) {{
            {mf_load}
        }}
    }}
    for (int d = lane; d < {D}; d += 32) {{
        int global_dof = world_dof_indices.data[dof_map_base + d];
        s_v[d] = global_dof >= 0{owned} ? v_out.data[global_dof] : 0.0f;
    }}
    SYNC();

{dense_op.pipeline_registers("    ")}

    for (int iter = 0; iter < iterations; iter++) {{
        int global_iter = iteration_offset + iter;
        int iteration_changed = 0;

{dense_rows_block(dense, dense_op, laws, admission=_phase_admission(), inline_velocity_limit=True, pad="        ")}

        if (row_phase == 0 || row_phase == 1 || row_phase == 4) {{
{free_body_rows_block(free, free_op, laws, admission=_MF_PHASE_ADMISSION, pad="        ")}
        }}

{chr(10).join(passes)}
{sweep_end}    }}

    for (int d = lane; d < {D}; d += 32) {{
        int global_dof = world_dof_indices.data[dof_map_base + d];
        if (global_dof >= 0{owned}) v_out.data[global_dof] = s_v[d];
    }}
    for (int i = dense_lo + lane; i < dense_hi; i += 32) {{
        world_impulses.data[off_dense + i] = s_lam_dense[i];
    }}
    if (row_phase != 3 && row_phase != 6) {{
        for (int i = mf_load_lo + lane; i < m_mf; i += 32) {{
            {mf_store}
        }}
    }}
#endif
"""
    preludes = (lane_group(32), RING, *law_preludes(PROJECTION_LAW), *solve_policy.preludes)
    return cuda_native(body, preludes=preludes), staging


@cache
def mf_gs_kernel(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    device_arch: str,
    friction_mode: str = "current",
    *,
    shared_metadata: bool = True,
    resident_row_metadata: bool = False,
    stream_mf_impulses: bool = False,
    has_drive_rows: bool = True,
    has_dense_velocity_limit_rows: bool = True,
    fuse_vel_limits: bool = False,
    skip_local_internal_worlds: bool = False,
    local_internal_max_constraints: int = 0,
    contact_torsion: bool = False,
    factor_coordinates: bool = False,
    skip_incremental_rows: int = 0,
    independent_components: bool = False,
    velocity_limit_law: str = "stateless",
    solve_policy: SolvePolicy = EXACT_POLICY,
) -> wp.Kernel:
    """The matrix-free Gauss-Seidel sweep kernel (the arguments and ABI of the factory it replaces)."""
    snippet, _staging = mf_gs_source(
        max_constraints,
        mf_max_constraints,
        max_world_dofs,
        friction_mode,
        shared_metadata=shared_metadata,
        resident_row_metadata=resident_row_metadata,
        stream_mf_impulses=stream_mf_impulses,
        has_drive_rows=has_drive_rows,
        has_dense_velocity_limit_rows=has_dense_velocity_limit_rows,
        fuse_vel_limits=fuse_vel_limits,
        skip_local_internal_worlds=skip_local_internal_worlds,
        contact_torsion=contact_torsion,
        factor_coordinates=factor_coordinates,
        skip_incremental_rows=skip_incremental_rows,
        independent_components=independent_components,
        velocity_limit_law=velocity_limit_law,
        solve_policy=solve_policy,
    )
    name = (
        f"sweep_mf_gs_{max_constraints}_{mf_max_constraints}_{max_world_dofs}_{friction_mode}"
        f"_drive{int(has_drive_rows)}_densevlim{int(has_dense_velocity_limit_rows)}"
        + velocity_limit_law_suffix(velocity_limit_law)
        + ("_fvl" if fuse_vel_limits else "")
        + ("" if shared_metadata else "_gmeta")
        + ("_smeta" if resident_row_metadata else "")
        + ("_glambda" if stream_mf_impulses else "")
        + (f"_local_fallback{local_internal_max_constraints}" if skip_local_internal_worlds else "")
        + ("_torsion" if contact_torsion else "")
        + ("_factor" if factor_coordinates else "")
        + (f"_skipinc{int(skip_incremental_rows)}" if skip_incremental_rows > 0 else "")
        + ("_components" if independent_components else "")
        + solve_policy.suffix
    )
    return KernelTemplate.from_abi(
        name,
        MF_GS_ABI,
        snippet,
        kernel_template.WORLD_QUEUE,
        extra=solve_policy.arguments,
        composed=True,
        native_name="sweep_mf_gs_native",
    ).kernel()


# ---------------------------------------------------------------------------------------------------------------------
# Multi-lane mapping: colored units (and split hubs) of a world's free-body rows
# ---------------------------------------------------------------------------------------------------------------------


def mf_colored_source(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    extra: int,
    colors: int,
    *,
    threads: int = 128,
    shared_metadata: bool = True,
    has_dense_velocity_limit_rows: bool = False,
    velocity_limit_law: str = "stateless",
    skip_local_internal_worlds: bool = False,
    skip_incremental_rows: int = 0,
    static_bytes: int = STATIC_SHARED_BYTES,
    cluster: int = 1,
) -> tuple[str, Staging]:
    """The colored sweep of a world's free-body rows, one block per world: dense rows by its first warp, then each
    color's units in parallel (one thread per unit), each color followed by the merge of the split bodies' copies,
    then the end passes.

    With ``cluster > 1`` a world takes a cluster of that many blocks (sm_90 and later): the first block holds the
    world's staged state and runs its dense rows, merges and end passes; every block's threads take the colors'
    units, reading and writing the first block's velocity and impulses through distributed shared memory; cluster
    barriers separate the colors.

    The world's velocity and the split copies' velocities (``extra`` copies of six coordinates) are staged, with the
    colors' offsets and the dense rows' support (the coordinates any dense row's ``J`` or ``Y`` touches: the dense
    rows run over it, not over the world's coordinates). The free-body impulses are staged in the shared memory the
    rest leaves of ``static_bytes`` when a world's rows fit it, and read and written in place otherwise (one block
    owns the world: a barrier publishes them either way).
    """
    M_D, M_MF, D = max_constraints, mf_max_constraints, max_world_dofs
    warp = Lanes(32)
    accumulated = check_velocity_limit_law(velocity_limit_law) == "accumulated"
    laws = RowLaws(drive=False, velocity_limit_law="accumulated" if accumulated else "stateless", regularize=True)
    dense = WorldRows(M_D, shared_metadata=shared_metadata, packed_types=shared_metadata, lanes=warp)
    free = FreeBodyRows(M_MF, stream=True, lanes=warp)
    dense_op = DenseJY(D, warp, support=("s_support", "s_support_count[0]"))
    free_op = FreeBody6(warp)
    velocity = D + 6 * extra
    words = (D + 31) // 32
    staging = Staging(
        (
            Buffer("s_v", "float", velocity),
            *dense.buffers(),
            Buffer("s_color_offset", "int", colors + 1),
            Buffer("s_support", "int", D),
            Buffer("s_support_bits", "unsigned", words),
            Buffer("s_support_count", "int", 1),
            Buffer("s_split_base", "float", 6 * (extra + 1)),
            Buffer("s_split_dof", "int", extra + 1),
            Buffer("s_split_copies", "int", extra + 1),
            Buffer("s_split_first", "int", extra + 1),
            Buffer("s_changed", "int", 1),
        ),
        threads=threads,
    )
    # The impulses take the room left, in whole warps' worth of rows, when it holds a useful share of the capacity.
    staged_rows = min(M_MF, max(0, static_bytes - staging.bytes) // 4 // 32 * 32)
    if staged_rows >= min(M_MF, 1024):
        staging = staging.plus(Buffer("s_lam", "float", staged_rows))
    else:
        staged_rows = 0
    # A unit's rows take the free-body row law of the matrix-free sweep, by one thread, on the unit's body copies.
    # The thread loads every field of its color slot (sweep_units.SLOT_FIELDS, field by field across the slots) into
    # registers at once, and visits the unit's rows unrolled: the visited row is the unit's row r, a friction pair's
    # second tangent is always its row 2, so every register index is a constant.
    unit_rows = FreeBodyRows(M_MF, stream=True, lanes=Lanes(1))
    arrays = {"mf_J_a": 0, "mf_J_b": 1, "mf_MiJt_a": 2, "mf_MiJt_b": 3}

    def unit_offset(row: str) -> str:
        offsets = {"i": "r", "sib": "2"}
        if row not in offsets:
            raise ValueError(f"the colored unit reads rows i and sib, not {row!r}")
        return offsets[row]

    unit_op = FreeBody6(
        Lanes(1),
        velocity=Velocity("V" if cluster > 1 else "s_v"),
        copies=("ka", "kb"),
        entry=lambda array, row, k: f"E[{unit_offset(row)} * 24 + {arrays[array] * 6} + {k}]",
        inverse=lambda row: (
            "mf_diag" if row == "i" else f"COPY_INVERSE({unit_offset(row)}, __int_as_float(W[{unit_offset(row)} * 3]))"
        ),
    )
    unit_ctx = RowContext(
        row="i",
        kind="mf_rt",
        operator=unit_op,
        load=unit_rows.load,
        store=unit_rows.store,
        rhs=lambda row: "__int_as_float(meta.z)" if row == "i" else f"__int_as_float(W[{unit_offset(row)} * 3 + 1])",
        mu=lambda row: f"E[{sweep_units.SLOT_MU}]",
        weight=lambda row: f"E[{sweep_units.SLOT_WEIGHT}]",
        parent="parent",
        ring_load=unit_rows.ring_load,
        lanes=Lanes(1),
        contact_kinds=(CONTACT,),
    )
    slot_fields = sweep_units.SLOT_ENTRIES + 2
    visit = row_visit(unit_ctx, laws, parent=unit_rows.parent_of("packed_tp"), check="mf_diag", pad=" " * 24)
    vlim_pass = ""
    if has_dense_velocity_limit_rows:
        vlim_pass = velocity_limit_pass(dense, dense_op, laws, pad="            ")
    skips = []
    if skip_local_internal_worlds:
        skips.append(f"    if (local_solve_owner.data[world] != {PGS_LOCAL_SOLVE_OWNER_GENERAL}) return;")
    if skip_incremental_rows > 0:
        skips.append(f"    if (m_mf == 0 && m_dense <= {int(skip_incremental_rows)}) return;")
    skips = "\n".join(skips)
    shared = "OWNER(s_lam)" if cluster > 1 else "s_lam"
    if staged_rows:
        lam_declaration = f"""    // The impulses: staged when the world's rows fit, else in place.
    const bool lam_staged = m_mf <= {staged_rows};
    float* g_lam_mf = lam_staged ? {shared} : &mf_impulses.data[off_mf];
    if (lam_staged && first_block) {{
        for (int i = tid; i < m_mf; i += blockDim.x) s_lam[i] = mf_impulses.data[off_mf + i];
    }}
"""
        lam_store = """    if (lam_staged) {
        for (int i = tid; i < m_mf; i += blockDim.x) mf_impulses.data[off_mf + i] = s_lam[i];
    }
"""
    else:
        lam_declaration = """    float* g_lam_mf = &mf_impulses.data[off_mf];
"""
        lam_store = ""
    if cluster > 1:
        # The cluster's barrier (release and acquire over the cluster's shared memory) and the first block's address
        # of a shared buffer.
        cluster_prelude = f"""    const int rank = world % {cluster};
    world /= {cluster};
    const bool first_block = rank == 0;
    const auto OWNER = [](auto* local) {{
        unsigned long long mapped;
        asm volatile("mapa.u64 %0, %1, %2;" : "=l"(mapped) : "l"(reinterpret_cast<unsigned long long>(local)), "r"(0));
        return reinterpret_cast<decltype(local)>(mapped);
    }};
#define WORLD_SYNC() asm volatile("barrier.cluster.arrive.release.aligned;\\n\\tbarrier.cluster.wait.acquire.aligned;" ::: "memory")
"""
        velocity_declaration = "    float* V = OWNER(s_v);\n    int* changed = OWNER(s_changed);\n"
        unit_first = "begin + rank * (int)blockDim.x + tid"
        unit_stride = f"{cluster} * (int)blockDim.x"
        world_undef = "#undef WORLD_SYNC\n"
    else:
        cluster_prelude = """    const bool first_block = true;
#define WORLD_SYNC() __syncthreads()
"""
        velocity_declaration = "    float* V = s_v;\n    int* changed = s_changed;\n"
        unit_first = "begin + tid"
        unit_stride = "(int)blockDim.x"
        world_undef = "#undef WORLD_SYNC\n"
    body = f"""    const int lane = LANE;
    const int warp = (int)(threadIdx.x >> 5);
    const int tid = (int)threadIdx.x;
{cluster_prelude}    int m_dense = world_constraint_count.data[world];
    int m_mf = mf_constraint_count.data[world];
    if (m_dense == 0 && m_mf == 0) return;
    if (m_dense > {M_D}) m_dense = {M_D};
    if (m_mf > {M_MF}) m_mf = {M_MF};
{skips}
    int mf_contact_end = mf_contact_rows_end.data[world];
    if (mf_contact_end > m_mf) mf_contact_end = m_mf;
    const int dense_lo = 0, dense_hi = m_dense;
    const int dof_map_base = world * {D};
    const int off_dense = world * {M_D};
    const int off_mf = world * {M_MF};
    const int off_meta = off_mf * 4;
    const int jy_world_base = world * {M_D} * {D};
    const int mf6_base = world * {M_MF} * 6;
    const int row_phase = 0, freeze_drive_rows = 0;

{staging.declarations("    ")}
{velocity_declaration}{lam_declaration}    // Field f of the unit in color slot p (sweep_units.SLOT_FIELDS), from slot_base / word_base.
#define SLOT_ENTRY(F) slot_data.data[slot_base + (F) * {M_MF}]
#define SLOT_WORD(F) slot_words.data[word_base + (F) * {M_MF}]
    for (int i = tid; i < dense_hi; i += blockDim.x) {{
{indent(dense.loads("i"), "        ")}
    }}
    for (int d = tid; d < {D}; d += blockDim.x) {{
        const int global_dof = world_dof_indices.data[dof_map_base + d];
        s_v[d] = global_dof >= 0 ? v_out.data[global_dof] : 0.0f;
    }}
    const int colors = color_count.data[world];
    const bool overflow = (color_overflow.data[world] & 1) != 0;
    const int splits = split_count.data[world];
    const int split_units_base = world * {extra + 1} * {colors};
    for (int e = tid; e < splits; e += blockDim.x) {{
        s_split_dof[e] = split_dof.data[off_mf + e];
        s_split_copies[e] = split_copies.data[off_mf + e];
        s_split_first[e] = split_first.data[off_mf + e];
    }}
    for (int c = tid; c <= colors; c += blockDim.x) s_color_offset[c] = color_offset.data[world * {colors + 1} + c];
    // The dense rows' support: a bitmap of the coordinates their J and Y rows touch, then its list, ascending.
    for (int w = tid; w < {words}; w += blockDim.x) s_support_bits[w] = 0u;
    __syncthreads();
    for (int e = tid; e < m_dense * {D}; e += blockDim.x) {{
        const int d = e % {D};
        if (J_world.data[jy_world_base + e] != 0.0f || Y_world.data[jy_world_base + e] != 0.0f)
            atomicOr(&s_support_bits[d >> 5], 1u << (d & 31));
    }}
    __syncthreads();
    if (warp == 0) {{
        int carry = 0;
        for (int base = 0; base < {words}; base += 32) {{
            const int w = base + lane;
            unsigned bits = w < {words} ? s_support_bits[w] : 0u;
            const int count = __popc(bits);
            int at = carry + SCAN(count);
            while (bits) {{
                s_support[at++] = w * 32 + __ffs(bits) - 1;
                bits &= bits - 1u;
            }}
            carry += READ(at - carry, 31);
        }}
        if (lane == 0) s_support_count[0] = carry;
    }}
    __syncthreads();

{dense_op.pipeline_registers("    ")}

    int iterations_run = 0;
    for (int iter = 0; iter < iterations; iter++) {{
        const int global_iter = iteration_offset + iter;
        int iteration_changed = 0;
        ++iterations_run;
        if (first_block && warp == 0) {{
{dense_rows_block(dense, dense_op, laws, admission="", inline_velocity_limit=False, pad="            ")}
        }}
        __syncthreads();
        // Every copy of a split body starts the colors at the body's velocity, which the merges then carry.
        if (first_block) {{
            for (int e = tid; e < splits * 6; e += blockDim.x) {{
                const int s = e / 6, k = e - 6 * s;
                const int dof = s_split_dof[s], first = s_split_first[s];
                const float body = s_v[dof + k];
                s_split_base[e] = body;
                for (int c = 1; c < s_split_copies[s]; ++c) s_v[{D} + 6 * (first + c - 1) + k] = body;
            }}
        }}
        WORLD_SYNC();
        for (int color = 0; color < colors; ++color) {{
            const int begin = s_color_offset[color];
            const int end_unit = s_color_offset[color + 1];
            // A world whose units did not fit the colors keeps its last color's units in row order on one thread.
            const bool serial = overflow && color == {colors - 1};
            const int first = serial ? (first_block && tid == 0 ? begin : end_unit) : {unit_first};
            const int stride = serial ? 1 : {unit_stride};
            for (int p = first; p < end_unit; p += stride) {{
                // One unit (a contact's normal row and its friction pair) by this thread alone: no other unit of
                // the color shares a body copy or a patch.
                const int4 slot = __ldg(reinterpret_cast<const int4*>(&color_slot.data[(off_mf + p) * 4]));
                const unsigned shape = (unsigned)slot.y;
                const float ka = (float)((shape >> 2) & 0x7fffu);
                const float kb = (float)(shape >> 17);
                const bool split = ka != 1.0f || kb != 1.0f;
                const int dof_a = slot.z, dof_b = slot.w;
                const int unit_row = slot.x;
                const int slot_base = world * {slot_fields * M_MF} + p;
                const int word_base = world * {sweep_units.SLOT_WORDS * M_MF} + p;
                const int rows = (int)(shape & 3u);
                // Every field of the slot at once: the first row's always, a friction pair's when the unit has one.
                float E[{slot_fields}];
                int W[{sweep_units.SLOT_WORDS}];
#pragma unroll
                for (int f = 0; f < {slot_fields}; ++f)
                    E[f] = (f < 24 || f >= {sweep_units.SLOT_ENTRIES} || rows == 3) ? SLOT_ENTRY(f) : 0.0f;
#pragma unroll
                for (int f = 0; f < {sweep_units.SLOT_WORDS}; ++f) W[f] = (f < 3 || rows == 3) ? SLOT_WORD(f) : 0;
                // A row's inverse diagonal on the copies: the stored one, or with a split side's j units in the
                // color 1 / (cfm + j_a J_a M_a + j_b J_b M_b).
                const auto COPY_INVERSE = [&](const int r, float stored) -> float {{
                    if (!split) return stored;
                    float d = cfm;
                    if (dof_a >= 0) {{
                        float side = 0.0f;
#pragma unroll
                        for (int k = 0; k < 6; ++k) side += E[r * 24 + k] * E[r * 24 + 12 + k];
                        d += ka * side;
                    }}
                    if (dof_b >= 0) {{
                        float side = 0.0f;
#pragma unroll
                        for (int k = 0; k < 6; ++k) side += E[r * 24 + 6 + k] * E[r * 24 + 18 + k];
                        d += kb * side;
                    }}
                    return d > 0.0f ? 1.0f / d : 0.0f;
                }};
#pragma unroll
                for (int r = 0; r < 3; ++r) {{
                    if (r >= rows) break;
                    // One pass of the row law; its `continue` leaves the row.
                    for (int once = 0; once < 1; ++once) {{
                        const int i = unit_row + r;
                        int4 meta;
                        meta.x = 0;
                        meta.y = W[r * 3];
                        meta.z = W[r * 3 + 1];
                        meta.w = W[r * 3 + 2];
                        const int packed_tp = meta.w;
                        const int mf_rt = packed_tp & 0xFFFF;
                        const float mf_diag = COPY_INVERSE(r, __int_as_float(meta.y));
{visit}
                    }}
                }}
            }}
            WORLD_SYNC();
            // The split bodies' copies merge: the body takes the mean of the changes of its j units in the color
            // (in copy order), and every copy takes the body's velocity.
            if (splits > 0) {{
                for (int e = first_block ? tid : splits * 6; e < splits * 6; e += blockDim.x) {{
                    const int s = e / 6, k = e - 6 * s;
                    const int j = split_units.data[split_units_base + s * {colors} + color];
                    if (j == 0) continue;
                    const int dof = s_split_dof[s], first = s_split_first[s], copies = s_split_copies[s];
                    const float base = s_split_base[e];
                    float change = s_v[dof + k] - base;
                    for (int c = 1; c < copies; ++c) change += s_v[{D} + 6 * (first + c - 1) + k] - base;
                    const float merged = base + change / (float)j;
                    s_split_base[e] = merged;
                    s_v[dof + k] = merged;
                    for (int c = 1; c < copies; ++c) s_v[{D} + 6 * (first + c - 1) + k] = merged;
                }}
                WORLD_SYNC();
            }}
        }}
        if (first_block && warp == 0) {{
{vlim_pass}
{speed_rows_pass(free, free_op, laws, pad="            ")}
        }}
        if (first_block && tid == 0) s_changed[0] = 0;
        WORLD_SYNC();
        if (iteration_changed != 0) atomicOr(changed, 1);
        WORLD_SYNC();
        const bool quiet = *reinterpret_cast<volatile int*>(changed) == 0;
        WORLD_SYNC();
        if (global_iter >= friction_start_iteration && quiet) break;
    }}
{world_undef}    if (!first_block) return;

    for (int d = tid; d < {D}; d += blockDim.x) {{
        const int global_dof = world_dof_indices.data[dof_map_base + d];
        if (global_dof >= 0) v_out.data[global_dof] = s_v[d];
    }}
    for (int i = tid; i < dense_hi; i += blockDim.x) world_impulses.data[off_dense + i] = s_lam_dense[i];
{lam_store}#undef SLOT_ENTRY
#undef SLOT_WORD
    // Readout: solves and iterations run, per world.
    if (tid == 0) {{
        sweep_stats.data[world * 2] += 1;
        sweep_stats.data[world * 2 + 1] += iterations_run;
    }}
#endif
"""
    preludes = (lane_group(32), RING, *law_preludes(PROJECTION_LAW))
    return cuda_native(body, preludes=preludes), staging


@cache
def mf_colored_kernel(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    extra: int,
    colors: int,
    *,
    threads: int = 128,
    shared_metadata: bool = True,
    has_dense_velocity_limit_rows: bool = False,
    velocity_limit_law: str = "stateless",
    skip_local_internal_worlds: bool = False,
    skip_incremental_rows: int = 0,
    static_bytes: int = STATIC_SHARED_BYTES,
    cluster: int = 1,
) -> wp.Kernel:
    """The colored sweep kernel: one block of ``threads`` per world (the world's units, leveled in
    :mod:`.sweep_units`), or a cluster of ``cluster`` blocks per world (launch ``worlds * cluster`` blocks)."""
    snippet, staging = mf_colored_source(
        max_constraints,
        mf_max_constraints,
        max_world_dofs,
        extra,
        colors,
        threads=threads,
        shared_metadata=shared_metadata,
        has_dense_velocity_limit_rows=has_dense_velocity_limit_rows,
        velocity_limit_law=velocity_limit_law,
        skip_local_internal_worlds=skip_local_internal_worlds,
        skip_incremental_rows=skip_incremental_rows,
        static_bytes=static_bytes,
        cluster=cluster,
    )
    staging.check("the colored sweep", static_bytes)
    params = kernel_template.params(
        ("world_constraint_count", wp.array[int]),
        ("local_solve_owner", wp.array[int]),
        ("world_dof_indices", wp.array2d[int]),
        ("rhs_bias", wp.array2d[float]),
        ("world_diag", wp.array2d[float]),
        ("world_row_w", wp.array2d[float]),
        ("world_impulses", wp.array2d[float]),
        ("J_world", wp.array3d[float]),
        ("Y_world", wp.array3d[float]),
        ("world_row_type", wp.array2d[int]),
        ("world_row_parent", wp.array2d[int]),
        ("world_row_mu", wp.array2d[float]),
        ("mf_constraint_count", wp.array[int]),
        ("mf_contact_rows_end", wp.array[int]),
        ("mf_meta", wp.array2d[int]),
        ("mf_impulses", wp.array2d[float]),
        ("mf_J_a", wp.array3d[float]),
        ("mf_J_b", wp.array3d[float]),
        ("mf_MiJt_a", wp.array3d[float]),
        ("mf_MiJt_b", wp.array3d[float]),
        ("mf_row_mu", wp.array2d[float]),
        ("mf_row_w", wp.array2d[float]),
        ("color_count", wp.array[int]),
        ("color_offset", wp.array2d[int]),
        ("color_slot", wp.array3d[int]),
        ("slot_data", wp.array3d[float]),
        ("slot_words", wp.array3d[int]),
        ("color_overflow", wp.array[int]),
        ("split_count", wp.array[int]),
        ("split_dof", wp.array2d[int]),
        ("split_copies", wp.array2d[int]),
        ("split_first", wp.array2d[int]),
        ("split_units", wp.array3d[int]),
        ("cfm", float),
        ("iterations", int),
        ("omega", float),
        ("regularize", int),
        ("friction_start_iteration", int),
        ("iteration_offset", int),
        ("sweep_stats", wp.array2d[int]),
        ("v_out", wp.array[float]),
    )
    name = (
        f"sweep_mf_colored_{max_constraints}_{mf_max_constraints}_{max_world_dofs}_x{extra}_c{colors}_t{threads}"
        + ("" if shared_metadata else "_gmeta")
        + ("_densevlim" if has_dense_velocity_limit_rows else "")
        + velocity_limit_law_suffix(velocity_limit_law)
        + ("_local" if skip_local_internal_worlds else "")
        + (f"_skipinc{int(skip_incremental_rows)}" if skip_incremental_rows > 0 else "")
        + next((f"_lam{buffer.count}" for buffer in staging if buffer.name == "s_lam"), "")
        + (f"_cl{cluster}" if cluster > 1 else "")
    )
    return KernelTemplate(
        name,
        params,
        snippet,
        dispatch=kernel_template.WORLDS,
        composed=True,
        native_name="sweep_mf_colored_native",
        launch_bounds=(threads, 1),
        cluster_dim=cluster,
    ).kernel()


class ColoredSweep:
    """The colored (and split) free-body sweep of a solver's general worlds, in place of its matrix-free sweep.

    Owns the units, colors and copies of every world (:mod:`.sweep_units`) and launches, per solve, the unit builder
    and then the colored sweep. :meth:`install` makes a solver launch it: the solver's own matrix-free sweep becomes a
    no-op of the same signature, so every other owner (tiers, local solves) still runs, and the colored sweep follows
    it on the same stream for the worlds the matrix-free sweep owned. Without a split the colors are the units' levels
    in row order and the sweep is the matrix-free sweep's row-order Gauss-Seidel sweep, a color's units at once.

    Args:
        solver: A matrix-free :class:`~newton.solvers.SolverFeatherPGS` with free-body rows.
        budget: Split budget ``B``: a body with more units is split into ``ceil(units / B)`` copies; 0 (the
            default) splits none; :data:`.sweep_units.AUTO_BUDGET` lets each world's data choose it. A split body
            converges like Jacobi over its copies: at the conformance gate's iterations a split hub misses the normal
            law (the gate's tray split in two: 1.25 to 1.41 times the bound), so the default splits none.
        extra: Capacity of extra body copies per world, which a budget needs (a body without room stays whole:
            status bit 2).
        colors: Color capacity per world (a multiple of 64 up to 512): a world takes about as many colors as its
            busiest body (copy) holds units; a unit past them runs in the last color, in row order (status bit 1).
        threads: Threads of a world's block (a color's units beyond them run in turns, or in a cluster).
        cluster: Blocks of a world (a thread-block cluster, sm_90 and later; 1: one block).
    """

    def __init__(
        self,
        solver,
        *,
        budget: int = 0,
        extra: int = 0,
        colors: int = sweep_units.MAX_COLORS,
        threads: int = 256,
        cluster: int = 1,
    ):
        self.solver = solver
        self.cluster = int(cluster)
        self.budget = int(budget)
        self.extra = int(extra)
        self.colors = int(colors)
        self.threads = int(threads)
        device = solver.model.device
        worlds, rows = solver.world_count, solver.mf_max_constraints
        dofs = solver.max_world_dofs
        plan = sweep_units.UnitPlan(rows, dofs, self.extra, self.colors)
        self.units = {}
        for field in sweep_units.UNIT_FIELDS:
            shape = field.shape({"world": worlds, **plan.sizes})
            self.units[field.name] = wp.zeros(shape, dtype=field.dtype, device=device)
        self.slots = {}
        for field in sweep_units.SLOT_FIELDS:
            shape = field.shape({"world": worlds, **plan.sizes})
            self.slots[field.name] = wp.zeros(shape, dtype=field.dtype, device=device)
        self.stats_array = wp.zeros((worlds, 2), dtype=wp.int32, device=device)
        static_bytes = static_budget(device.arch) if device.is_cuda else STATIC_SHARED_BYTES
        self.build_kernel = sweep_units.build_units_kernel(
            rows, dofs, self.extra, self.colors, static_bytes=static_bytes
        )
        self.sweep_kernel = mf_colored_kernel(
            solver.dense_max_constraints,
            rows,
            dofs,
            self.extra,
            self.colors,
            threads=self.threads,
            shared_metadata=True,
            has_dense_velocity_limit_rows=bool(solver.enable_joint_velocity_limits),
            velocity_limit_law=solver._velocity_limit_kernel_law(),
            skip_local_internal_worlds=bool(solver._local_internal_fast_path),
            skip_incremental_rows=_tier_rows(solver),
            static_bytes=static_bytes,
            cluster=self.cluster,
        )
        self._installed = None

    def launch(
        self,
        *,
        dense_rhs,
        mf_meta,
        iterations: int,
        omega: float,
        friction_start_iteration: int,
        iteration_offset: int,
        regularize: int,
    ) -> None:
        """Build the units and run the colored sweep over every world (a world without rows returns at once)."""
        solver, units = self.solver, self.units
        device = solver.model.device
        wp.launch_tiled(
            self.build_kernel,
            dim=[solver.world_count],
            inputs=[solver.mf_constraint_count, solver.mf_contact_rows_end, mf_meta, self.budget],
            outputs=[units[field.name] for field in sweep_units.UNIT_FIELDS],
            block_dim=32 * sweep_units.BUILD_WARPS,
            device=device,
        )
        wp.launch(
            sweep_units.stage_slots,
            dim=(solver.world_count, solver.mf_max_constraints),
            inputs=[
                units["unit_count"],
                units["color_slot"],
                mf_meta,
                solver.mf_J_a,
                solver.mf_J_b,
                solver.mf_MiJt_a,
                solver.mf_MiJt_b,
                solver.mf_row_mu,
                solver.mf_row_w,
            ],
            outputs=[self.slots["slot_data"], self.slots["slot_words"]],
            device=device,
        )
        wp.launch_tiled(
            self.sweep_kernel,
            dim=[solver.world_count * self.cluster],
            inputs=[
                solver.constraint_count,
                solver._local_solve_owner,
                solver.world_dof_indices,
                dense_rhs,
                solver.diag,
                solver.row_w,
                solver.impulses,
                solver.J_world,
                solver.Y_world,
                solver.row_type,
                solver.row_parent,
                solver.row_mu,
                solver.mf_constraint_count,
                solver.mf_contact_rows_end,
                mf_meta,
                solver.mf_impulses,
                solver.mf_J_a,
                solver.mf_J_b,
                solver.mf_MiJt_a,
                solver.mf_MiJt_b,
                solver.mf_row_mu,
                solver.mf_row_w,
                units["color_count"],
                units["color_offset"],
                units["color_slot"],
                self.slots["slot_data"],
                self.slots["slot_words"],
                units["color_overflow"],
                units["split_count"],
                units["split_dof"],
                units["split_copies"],
                units["split_first"],
                units["split_units"],
                float(solver.pgs_cfm),
                int(iterations),
                float(omega),
                int(regularize),
                int(friction_start_iteration),
                int(iteration_offset),
                self.stats_array,
            ],
            outputs=[solver.v_out],
            block_dim=self.threads,
            device=device,
        )

    def stats(self) -> dict:
        """The last build's units, colors and copies, and the solves and iterations run so far, over the worlds.

        Per world with units: units, colors, the largest color, split bodies, extra copies and the overflow bits;
        summed: solves and iterations (``iterations / solves`` is the mean a solve ran).
        """
        host = {
            name: self.units[name].numpy()
            for name in (
                "unit_count",
                "color_count",
                "color_offset",
                "color_overflow",
                "split_count",
                "extra_copies",
                "unit_budget",
            )
        }
        sweeps = self.stats_array.numpy()
        worlds = []
        for w in range(self.solver.world_count):
            units, colors = int(host["unit_count"][w]), int(host["color_count"][w])
            if units == 0:
                continue
            offsets = host["color_offset"][w, : colors + 1]
            sizes = offsets[1:] - offsets[:-1]
            worlds.append(
                {
                    "units": units,
                    "colors": colors,
                    "largest_color": int(sizes.max()) if colors else 0,
                    "serial_steps": int(((sizes + self.threads - 1) // self.threads).sum()),
                    "budget": int(host["unit_budget"][w]),
                    "splits": int(host["split_count"][w]),
                    "extra_copies": int(host["extra_copies"][w]),
                    "overflow": int(host["color_overflow"][w]),
                }
            )
        return {
            "budget": self.budget,
            "threads": self.threads,
            "solves": int(sweeps[:, 0].sum()),
            "iterations": int(sweeps[:, 1].sum()),
            "worlds": worlds,
        }

    def install(self) -> None:
        """Make the solver run this sweep for the worlds its matrix-free sweep owns (row phase 0 only)."""
        solver = self.solver
        if solver._paired_factor_solve_kernel is not None or solver._paired_components is not None:
            raise NotImplementedError("the colored sweep does not share worlds with the paired owner yet")
        if solver._contact_torsion_enabled or solver.pgs_schedule != "interleaved" or solver.drive_mode == "physx_pgs":
            raise NotImplementedError("the colored sweep covers the interleaved schedule without torsion or drive rows")
        original = solver._launch_matrix_free_gs_solve
        legacy_kernel = solver._pgs_solve_mf_gs_kernel
        solver._pgs_solve_mf_gs_kernel = _noop_mf_gs_kernel()

        def launch(
            *,
            dense_rhs,
            mf_meta,
            iterations,
            omega,
            friction_start_iteration=None,
            iteration_offset=0,
            freeze_drive_rows=False,
            soft_relax=False,
        ):
            original(
                dense_rhs=dense_rhs,
                mf_meta=mf_meta,
                iterations=iterations,
                omega=omega,
                friction_start_iteration=friction_start_iteration,
                iteration_offset=iteration_offset,
                freeze_drive_rows=freeze_drive_rows,
                soft_relax=soft_relax,
            )
            if iterations <= 0:
                return
            if friction_start_iteration is None:
                friction_start_iteration = solver._contact_friction_start_iteration(iterations)
            self.launch(
                dense_rhs=dense_rhs,
                mf_meta=mf_meta,
                iterations=iterations,
                omega=omega,
                friction_start_iteration=friction_start_iteration,
                iteration_offset=iteration_offset,
                regularize=int(solver._regularization_enabled and not soft_relax),
            )

        solver._launch_matrix_free_gs_solve = launch
        self._installed = (original, legacy_kernel)

    def uninstall(self) -> None:
        if self._installed is None:
            return
        original, legacy_kernel = self._installed
        self.solver._launch_matrix_free_gs_solve = original
        self.solver._pgs_solve_mf_gs_kernel = legacy_kernel
        self._installed = None


def _tier_rows(solver) -> int:
    """The dense-row budget of the tiers that take the worlds without free-body rows ahead of the matrix-free sweep
    (the matrix-free sweep's ``skip_incremental_rows``, as the solver's construction derives it)."""
    rows = int(getattr(solver, "_incremental_rows", 0) or 0)
    if solver._mf_gs_tiers and rows == 0 and solver.mf_gs_parallel_rows > 0:
        rows = min(
            int(solver.dense_max_constraints),
            int(solver.mf_gs_parallel_rows),
            128 if solver.mf_gs_parallel_matrix_free else 96,
        )
        rows -= rows % 4
    return rows


@cache
def _noop_mf_gs_kernel() -> wp.Kernel:
    """A matrix-free sweep of the legacy signature that solves nothing (its worlds go to another sweep)."""
    return KernelTemplate.from_abi(
        "sweep_mf_gs_noop",
        MF_GS_ABI,
        "\n#if defined(__CUDA_ARCH__)\n    return;\n#endif\n",
        kernel_template.WORLD_QUEUE,
        composed=True,
        native_name="sweep_mf_gs_noop_native",
    ).kernel()


# ---------------------------------------------------------------------------------------------------------------------
# The incremental tier: the response block of a world's dense rows (Block), its rows compacted at entry
# ---------------------------------------------------------------------------------------------------------------------


def incremental_source(
    max_constraints: int,
    max_world_dofs: int,
    *,
    incremental_rows: int,
    has_drive_rows: bool = True,
    has_dense_velocity_limit_rows: bool = True,
    skip_local_internal_worlds: bool = False,
    min_rows: int = 0,
    velocity_limit_law: str = "stateless",
    solve_policy: SolvePolicy = EXACT_POLICY,
) -> tuple[str, Staging]:
    """The Gauss-Seidel sweep of a world's dense rows (at most ``incremental_rows``) over their response block.

    For worlds without free-body rows: the warp builds ``A = J Y`` and the residuals once (:class:`.Block`),
    compacts the rows this phase visits in row order, and solves them with the shared row law; the velocity is
    published once after the sweeps.
    """
    _validate_dense_metadata_encoding(max_constraints)
    accumulated = check_velocity_limit_law(velocity_limit_law) == "accumulated"
    M_D, D, AM, MIN_ROWS = max_constraints, max_world_dofs, int(incremental_rows), int(min_rows)
    if AM <= 0 or AM > M_D or AM > 64:
        raise ValueError("incremental_rows must be in [1, min(max_constraints, 64)]")
    if MIN_ROWS < 0 or MIN_ROWS >= AM:
        raise ValueError("min_rows must be in [0, incremental_rows)")
    lanes = Lanes(32)
    laws = RowLaws(
        drive=has_drive_rows,
        velocity_limit_law="accumulated" if accumulated else "stateless",
        regularize=True,
        policy=solve_policy,
    )
    op = Block(D, AM, lanes, diagonal=lambda row: f"s_diag[{row}]", inverse=lambda row: f"s_inv_diag[{row}]")
    drive = DRIVE_DATA if has_drive_rows else {}
    staging = Staging(
        (
            Buffer("s_v", "float", D),
            *(
                Buffer(name, "float", AM)
                for name in ("s_lam", "s_app", "s_rhs", "s_diag", "s_inv_diag", "s_res", "s_w")
            ),
            Buffer("s_mu", "float", AM),
            Buffer("s_parent", "int", AM),
            Buffer("s_type", "int", AM),
            Buffer("s_A", "float", AM * AM),
            Buffer("s_rows", "int", AM),
            Buffer("s_vlim_rows", "int", AM),
            *(Buffer(name, "float", AM) for name, _ in drive.values()),
        ),
        threads=32,
    )
    ctx = RowContext(
        row="i",
        kind="row_type",
        operator=op,
        load=lambda row: f"s_lam[{row}]",
        store=lambda row, value: lanes.first(f"s_lam[{row}] = {value};"),
        rhs=lambda row: f"s_rhs[{row}]",
        mu=lambda row: f"s_mu[{row}]",
        weight=lambda row: f"s_w[{row}]",
        parent="parent",
        ring_load=lambda parent: (
            f"RING_LOAD(s_lam[{parent}], {parent}, {AM},\n"
            "    [&](int r) { return s_parent[r]; }, [&](int r) { return s_lam[r]; })"
        ),
        lanes=lanes,
        drive_data=(lambda name, row: f"{DRIVE_DATA[name][0]}[{row}]") if has_drive_rows else None,
        inline_velocity_limit=True,
        record=lambda row, delta: lanes.first(f"s_app[{row}] += {delta};"),
    )
    # The compaction admitted the rows on their diagonals.
    row_loop = "\n".join(
        [
            "        for (int k = 0; k < n_active; k++) {",
            "            const int i = s_rows[k];",
            "            const int row_type = s_type[i];",
            row_visit(ctx, laws, parent="s_parent[i]", check=None, pad="            "),
            "        }",
        ]
    )
    vlim_old = "s_lam[i]" if accumulated else "0.0f"
    vlim_pass = ""
    if has_dense_velocity_limit_rows:
        vlim_pass = f"""
        if (row_phase == 0 || row_phase == 2) {{
            for (int k = 0; k < n_vlim; k++) {{
                const int i = s_vlim_rows[k];
                const float residual_vlim = s_res[i];
                const auto vlim = ProjectionLaw::velocity_limit_update({vlim_old}, residual_vlim, {op.scale("residual_vlim", "i")}, {int(accumulated)});
                const float delta_impulse_vlim = vlim.v1;
                SYNC();
                if (lane == 0) {{
                    s_lam[i] = vlim.v0;
                    s_app[i] += delta_impulse_vlim;
                }}
                if (delta_impulse_vlim != 0.0f) {{
{change_vote(laws, (("delta_impulse_vlim", op.value("i"), str(VELOCITY_LIMIT)),), " " * 20)}
{op.apply_row("i", "delta_impulse_vlim", "                    ")}
                }}
                SYNC();
            }}
        }}"""
    sweep_end = end_sweep(solve_policy, iteration="iter")
    local_skip = (
        f"\n    if (local_solve_owner.data[world] != {PGS_LOCAL_SOLVE_OWNER_GENERAL}) return;"
        if skip_local_internal_worlds
        else ""
    )
    drive_loads = "".join(f"\n        {name}[i] = {source}.data[off_dense + i];" for name, source in drive.values())
    admit_drive = "1" if has_drive_rows else f"(row_type != {DRIVE})"
    phase_filters = "\n".join(
        f"            if (row_phase == {phase} && "
        + " && ".join(f"row_type != {kind}" for kind in kinds)
        + ") admit = 0;"
        for phase, kinds in _PHASE_DENSE_KINDS.items()
    )
    body = f"""    const int lane = LANE;

    int m_dense = world_constraint_count.data[world];
    const int m_mf = mf_constraint_count.data[world];
    if (m_dense == 0 || m_mf != 0) return;
    if (m_dense > {AM} || m_dense <= {MIN_ROWS}) return;{local_skip}

    int dense_lo = 0;
    int dense_hi = m_dense;
    if (row_phase == 3) {{
        dense_hi = min(dense_phase_bounds.data[world * 2 + 0], m_dense);
    }} else if (row_phase == 5) {{
        dense_lo = min(dense_phase_bounds.data[world * 2 + 0], m_dense);
        dense_hi = min(dense_phase_bounds.data[world * 2 + 1], m_dense);
    }} else if (row_phase == 4) {{
        dense_lo = min(dense_phase_bounds.data[world * 2 + 1], m_dense);
    }}
    if (dense_hi <= dense_lo) return;
    // The block holds rows [0, dense_hi): a friction row's parent precedes it, rows below dense_lo never change.
    const int n_rows_total = dense_hi;
    const int dof_map_base = world * {D};
    const int off_dense = world * {M_D};
    const int jy_world_base = world * {M_D} * {D};

{staging.declarations("    ")}

    for (int i = lane; i < n_rows_total; i += 32) {{
        s_lam[i] = world_impulses.data[off_dense + i];
        s_app[i] = 0.0f;
        s_rhs[i] = rhs_bias.data[off_dense + i];
        const float diag = world_diag.data[off_dense + i];
        s_diag[i] = diag;
        s_inv_diag[i] = diag > 0.0f ? 1.0f / diag : 0.0f;
        s_type[i] = world_row_type.data[off_dense + i] & {_DENSE_META_ROW_TYPE_MASK};
        s_parent[i] = world_row_parent.data[off_dense + i];
        s_mu[i] = world_row_mu.data[off_dense + i];
        s_w[i] = regularize ? world_row_w.data[off_dense + i] : 1.0f;{drive_loads}
    }}
    for (int d = lane; d < {D}; d += 32) {{
        const int global_dof = world_dof_indices.data[dof_map_base + d];
        s_v[d] = global_dof >= 0 ? v_out.data[global_dof] : 0.0f;
    }}
    SYNC();

    // The rows this phase visits, in row order; the phase's velocity-limit rows for the end pass.
    int n_active = 0;
    int n_vlim = 0;
    for (int base = dense_lo; base < dense_hi; base += 32) {{
        const int row = base + lane;
        int admit = 0;
        int vlim = 0;
        if (row < dense_hi && (s_inv_diag[row] > 0.0f || s_type[row] == {FRICTION})) {{
            const int row_type = s_type[row];
            admit = {admit_drive};
{phase_filters}
            if ((row_phase == 0 || row_phase == 2) && row_type == {VELOCITY_LIMIT}) {{
                admit = 0;
                vlim = 1;
            }}
            if (freeze_drive_rows != 0 && row_type == {DRIVE}) admit = 0;
        }}
        const int slot = ORDERED(n_active, admit != 0);
        if (admit) s_rows[slot] = row;
        const int vlim_slot = ORDERED(n_vlim, vlim != 0);
        if (vlim) s_vlim_rows[vlim_slot] = row;
    }}

{op.build()}
    SYNC();

    for (int iter = 0; iter < iterations; iter++) {{
        const int global_iter = iteration_offset + iter;
        int iteration_changed = 0;
{row_loop}
{vlim_pass}
{sweep_end}    }}

{op.publish()}
    for (int i = dense_lo + lane; i < dense_hi; i += 32) {{
        world_impulses.data[off_dense + i] = s_lam[i];
    }}
#endif
"""
    preludes = (lane_group(32), RING, *law_preludes(PROJECTION_LAW), *solve_policy.preludes)
    return cuda_native(body, preludes=preludes), staging


@cache
def incremental_kernel(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    device_arch: str,
    *,
    incremental_rows: int,
    has_drive_rows: bool = True,
    has_dense_velocity_limit_rows: bool = True,
    skip_local_internal_worlds: bool = False,
    min_rows: int = 0,
    velocity_limit_law: str = "stateless",
    solve_policy: SolvePolicy = EXACT_POLICY,
) -> wp.Kernel:
    """The incremental tier's kernel (the arguments and ABI of the factory it replaces: the matrix-free sweep's)."""
    snippet, _staging = incremental_source(
        max_constraints,
        max_world_dofs,
        incremental_rows=incremental_rows,
        has_drive_rows=has_drive_rows,
        has_dense_velocity_limit_rows=has_dense_velocity_limit_rows,
        skip_local_internal_worlds=skip_local_internal_worlds,
        min_rows=min_rows,
        velocity_limit_law=velocity_limit_law,
        solve_policy=solve_policy,
    )
    name = (
        f"sweep_incremental_{max_constraints}_{mf_max_constraints}_{max_world_dofs}_rows{int(incremental_rows)}"
        f"_min{int(min_rows)}_drive{int(has_drive_rows)}_densevlim{int(has_dense_velocity_limit_rows)}"
        + velocity_limit_law_suffix(velocity_limit_law)
        + ("_local_fallback" if skip_local_internal_worlds else "")
        + solve_policy.suffix
    )
    return KernelTemplate.from_abi(
        name,
        MF_GS_ABI,
        snippet,
        kernel_template.WORLD_QUEUE,
        extra=solve_policy.arguments,
        composed=True,
        native_name="sweep_incremental_native",
    ).kernel()
