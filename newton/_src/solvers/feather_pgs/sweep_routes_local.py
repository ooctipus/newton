# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The local owners' sweeps and the paired factor sweep as configurations of the sweep emitter (:mod:`.sweep`).

Every kernel's working set fits the static budget of the device it is built for (:func:`.staging.arch_budget`).

- :func:`local_owned_kernel`: one lane group (8, 16 or 32 lanes) per candidate articulation, or articulation and
  free body, with several groups per block. The group stages the factors and rows, whitens each row in the kernel
  and solves the rows over the packed Gram block ``A = Z^T Z`` (:class:`.response_operator.LocalBlock`) in factor
  coordinates, its residuals maintained and the velocity committed after the sweeps. The residual owner's free-body
  rows follow the dense rows in the block.
- :func:`paired_factor_kernel`: one warp per world of an articulation and a free body, rows in factor coordinates
  (:class:`.response_operator.FactorRows`), the velocity mapped in and out per component, under one of three
  schedules: row by row, contact triples after a prefix, or contact units.
"""

from __future__ import annotations

import dataclasses
from functools import cache

import warp as wp

from . import kernel_template
from .device_preludes import RING, cuda_native, lane_group
from .kernel_template import KernelTemplate
from .law_source import law_preludes
from .packed_rows import PACKED_SIDES
from .projection_laws import PROJECTION_LAW
from .response_operator import CholeskySolve, FactorComponent, FactorRows, FreeBodyZ, Lanes, LocalBlock
from .row_metadata import _mf_parent_expression
from .solve_policy import DEFAULT_POLICY, SolvePolicy
from .staging import Buffer, Staging, arch_budget
from .sweep import (
    CONTACT,
    FRICTION,
    JOINT_LIMIT,
    RowContext,
    RowLaws,
    admit_pair,
    end_sweep,
    friction_gate,
    friction_radius,
    indent,
    pair_radius,
    pair_update,
    row_update,
    row_visit,
)

# ---------------------------------------------------------------------------------------------------------------------
# The local owners: a lane group per articulation (and its free body), the response block built in the kernel
# ---------------------------------------------------------------------------------------------------------------------

LOCAL_ABI = kernel_template.params(
    ("candidate", int),
    ("candidate_articulations", wp.array[int]),
    ("candidate_secondary_articulations", wp.array[int]),
    ("expected_owner", int),
    ("articulation_group_index", wp.array[int]),
    ("articulation_world", wp.array[int]),
    ("articulation_dof_start", wp.array[int]),
    ("local_solve_owner", wp.array[int]),
    ("world_constraint_count", wp.array[int]),
    ("L_group", wp.array3d[float]),
    ("secondary_L_group", wp.array3d[float]),
    ("packed_J", wp.array3d[float]),
    ("row_sides", wp.array2d[wp.vec2i]),
    ("response_dofs", wp.array[int]),
    ("articulation_world_dof_offset", wp.array[int]),
    ("packed_dense", int),
    ("rhs_bias", wp.array2d[float]),
    ("world_row_type", wp.array2d[int]),
    ("world_row_parent", wp.array2d[int]),
    ("world_row_mu", wp.array2d[float]),
    ("mf_constraint_count", wp.array[int]),
    ("mf_meta", wp.array2d[int]),
    ("mf_impulses", wp.array2d[float]),
    ("mf_Z_a", wp.array3d[float]),
    ("mf_Z_b", wp.array3d[float]),
    ("mf_row_mu", wp.array2d[float]),
    ("regularize", int),
    ("world_row_w", wp.array2d[float]),
    ("mf_row_w", wp.array2d[float]),
    ("iterations", int),
    ("omega", float),
    ("friction_start_iteration", int),
    ("iteration_offset", int),
    ("world_diag", wp.array2d[float]),
    ("world_impulses", wp.array2d[float]),
    ("v_out", wp.array[float]),
)
"""The local owners' native ABI: one candidate per lane group (``candidates`` or ``candidate_queue`` dispatch)."""


def _check_local_options(
    warps_per_block, lanes_per_world, contact_capable, total_dofs, paired_dofs, mf_max, local_mf
) -> None:
    """Reject the options the local owners' factory rejects, with its messages."""
    if warps_per_block not in (1, 2):
        raise ValueError("warps_per_block must be 1 or 2")
    if lanes_per_world not in (8, 16, 32):
        raise ValueError("lanes_per_world must be 8, 16, or 32")
    if not contact_capable and paired_dofs:
        raise ValueError("only single-articulation kernels may omit contact handling")
    if local_mf and (not contact_capable or not paired_dofs or mf_max <= 0):
        raise ValueError("local MF rows require a contact-capable articulation pair and positive MF storage")
    if lanes_per_world < 32 and (contact_capable or total_dofs > 16):
        raise ValueError("subwarp local solve groups require a contactless world with at most 16 DoFs")


def local_staging(
    local_max_constraints: int,
    dof_count: int,
    paired_dof_count: int = 0,
    *,
    warps_per_block: int = 1,
    lanes_per_world: int = 32,
    contact_capable: bool = True,
    local_mf_max_constraints: int = 0,
) -> Staging:
    """The shared working set of a local owner's block: per lane group the factors, the rows' whitened entries,
    impulses and metadata, the velocity, the paired free body's impulses and the packed Gram block.

    The solver admits an owner, and sizes its blocks, from it (:meth:`.Staging.fits` against
    :func:`.staging.arch_budget`).
    """
    R, P, S, MFR, L = (
        int(local_max_constraints),
        int(dof_count),
        int(paired_dof_count),
        int(local_mf_max_constraints),
        int(lanes_per_world),
    )
    T, block_rows = P + S, R + MFR
    return Staging(
        (
            Buffer("s_L", "float", P * P),
            *((Buffer("s_L_secondary", "float", S * S),) if S else ()),
            Buffer("s_Z", "float", R * T),
            Buffer("s_v", "float", T),
            *(Buffer(name, "float", R) for name in ("s_lam", "s_rhs", "s_inv_diag")),
            Buffer("s_type", "int", R),
            Buffer("s_active", "int", R),
            *((Buffer("s_parent", "int", R), Buffer("s_mu", "float", R)) if contact_capable else ()),
            *((Buffer("s_mf_lam", "float", MFR),) if MFR else ()),
            Buffer("s_A", "float", block_rows * (block_rows + 1) // 2),
            Buffer("s_res", "float", block_rows),
            Buffer("s_app", "float", block_rows),
        ),
        groups=int(warps_per_block) * 32 // L,
        threads=32 * int(warps_per_block),
        name="LocalScratch",
    )


def local_owned_source(
    max_constraints: int,
    local_max_constraints: int,
    dof_count: int,
    paired_dof_count: int = 0,
    *,
    warps_per_block: int = 1,
    lanes_per_world: int = 32,
    contact_capable: bool = True,
    mf_max_constraints: int = 0,
    local_mf_max_constraints: int = 0,
    row_inputs=None,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
) -> tuple[str, Staging]:
    """The native body of a local owner's sweep: an articulation's rows (and a paired free body's coordinates and,
    for the residual owner, its free-body rows) over the response block the lane group builds in the kernel.

    The group stages the factors, the rows' impulses, right-hand sides and metadata and the velocity, which enters
    factor coordinates; whitens each row, ``Z = L^-1 P J^T``, from its Jacobian (``row_inputs``' packets, or the
    group's ``J``); stores the effective diagonal ``|Z|^2 + cfm`` back to ``world_diag`` and stages its inverse;
    builds the Gram block and the residuals (:class:`.LocalBlock`); then runs the sweeps with the shared row law, the
    dense rows in slot order and then the free-body rows, commits the velocity, which leaves factor coordinates, and
    the impulses of the rows with a Jacobian.

    Returns the body and its staging.
    """
    M, R, P, S = int(max_constraints), int(local_max_constraints), int(dof_count), int(paired_dof_count)
    T, MFS, MFR, L = P + S, int(mf_max_constraints), int(local_mf_max_constraints), int(lanes_per_world)
    _check_local_options(warps_per_block, L, contact_capable, T, S, MFS, MFR)
    lanes = Lanes(L)
    free = FreeBodyZ(lanes) if MFR else None
    laws = RowLaws(regularize=contact_capable, policy=solve_policy)
    factors = (CholeskySolve(P, "s_L", reciprocal=True, ascending=True),)
    if S:
        factors += (CholeskySolve(S, "s_L_secondary", reciprocal=True, ascending=True),)
    op = LocalBlock(
        T,
        factors,
        lanes,
        dense="row_count",
        free=free,
        inverse=lambda row: "inverse_diagonal" if row == "i" else f"s_inv_diag[{row}]",
    )
    staging = local_staging(
        R,
        P,
        S,
        warps_per_block=warps_per_block,
        lanes_per_world=L,
        contact_capable=contact_capable,
        local_mf_max_constraints=MFR,
    )

    # The rows' Jacobians: the producers' packed rows (packed_rows), each articulation's entries where it is a side.
    sides = ""
    primary_read = "(first_p >= 0 ? J_row[first_p + i] : 0.0f)"
    secondary_read = "(first_s >= 0 ? J_row[first_s + i] : 0.0f)"
    if row_inputs is not None:
        primary_read, secondary_read = row_inputs.read(P, S)
    else:
        sides = (
            "        const wp::vec2i sides_r = row_sides.data[world_row_base + row];\n"
            "        const float* J_row = &packed_J.data[(size_t)(world_row_base + row) * packed_J.shape[2]];\n"
            "        const int first_p = PACKED_FIRST(sides_r, art);\n"
            + ("        const int first_s = PACKED_FIRST(sides_r, secondary_art);\n" if S else "")
        )
    reads = (primary_read, secondary_read)[: len(factors)]

    pair = ""
    dof = "articulation_dof_start.data[art] + d"
    if S:
        pair = f"""
    const int secondary_art = candidate_secondary_articulations.data[candidate];
    if (articulation_world.data[secondary_art] != world) return;
    const int secondary_group = articulation_group_index.data[secondary_art];
    const int secondary_group_l_base = secondary_group * {S * S};"""
        dof = f"d < {P} ? articulation_dof_start.data[art] + d : articulation_dof_start.data[secondary_art] + d - {P}"
    factor_loads = [factors[0].stage(lambda i: f"L_group.data[group_l_base + {i}]", "lane", L)]
    if S:
        factor_loads.append(
            factors[1].stage(lambda i: f"secondary_L_group.data[secondary_group_l_base + {i}]", "lane", L)
        )
    metadata = (
        "\n        s_parent[row] = world_row_parent.data[world_row_base + row];"
        "\n        s_mu[row] = world_row_mu.data[world_row_base + row];"
        if contact_capable
        else ""
    )
    mf_setup = mf_load = mf_store = ""
    if MFR:
        mf_setup = f"""
    int mf_count = mf_constraint_count.data[world];
    if (mf_count > {MFR}) mf_count = {MFR};
    const int off_mf = world * {MFS};
    const int off_meta = off_mf * 4;
    const int mf6_base = world * {MFS * 6};"""
        mf_load = (
            f"\n    for (int row = lane; row < mf_count; row += {L}) s_mf_lam[row] = mf_impulses.data[off_mf + row];"
        )
        mf_store = (
            f"\n    for (int row = lane; row < mf_count; row += {L}) mf_impulses.data[off_mf + row] = s_mf_lam[row];"
        )

    sweeps = _local_sweeps(op, laws, lanes, R, MFS, MFR, contact_capable)
    body = f"""    const int lane = LANE;
    const int art = candidate_articulations.data[candidate];
    const int group = articulation_group_index.data[art];
    const int world = articulation_world.data[art];
    if (local_solve_owner.data[world] != expected_owner) return;
    const int row_count = world_constraint_count.data[world];
    if (row_count == 0 || row_count > {R}) return;

    const int world_row_base = world * {M};
    const int group_l_base = group * {P * P};{pair}
{staging.declarations("    ")}{mf_setup}

{indent(chr(10).join(factor_loads), "    ")}
    for (int row = lane; row < row_count; row += {L}) {{
        s_lam[row] = world_impulses.data[world_row_base + row];
        s_rhs[row] = rhs_bias.data[world_row_base + row];
        s_type[row] = world_row_type.data[world_row_base + row];{metadata}
    }}{mf_load}
    for (int d = lane; d < {T}; d += {L}) {{
        const int dof = {dof};
        s_v[d] = v_out.data[dof];
    }}
    SYNC();
    // The velocity enters factor coordinates, u = L^T P v, a lane per coordinate.
{op.enter("    ")}
    SYNC();

    // Each row whitened, Z = L^-1 P J^T, its effective diagonal |Z|^2 + cfm (stored back) and its inverse.
    for (int row = lane; row < row_count; row += {L}) {{
{sides}{op.respond(reads, "        ")}
        s_active[row] = active;
        const float effective_diagonal = row_diagonal + world_diag.data[world_row_base + row];
        s_inv_diag[row] = effective_diagonal > 0.0f ? 1.0f / effective_diagonal : 0.0f;
        world_diag.data[world_row_base + row] = effective_diagonal;
    }}
    SYNC();
{op.assemble(lambda row: f"s_rhs[{row}]", "    ")}
    SYNC();

{sweeps}

{op.commit("    ")}
    SYNC();
    // The velocity leaves factor coordinates, v = P^T L^-T u, block by block.
{op.leave("    ")}
    SYNC();
    for (int d = lane; d < {T}; d += {L}) {{
        const int dof = {dof};
        v_out.data[dof] = s_v[d];
    }}
    for (int row = lane; row < row_count; row += {L}) {{
        if (s_active[row] != 0) world_impulses.data[world_row_base + row] = s_lam[row];
    }}{mf_store}
{staging.undefine()}
#endif
"""
    preludes = (
        lane_group(L),
        *((RING,) if contact_capable else ()),
        PACKED_SIDES,
        *law_preludes(PROJECTION_LAW),
        *solve_policy.preludes,
    )
    return cuda_native(body, preludes=preludes), staging


def _local_sweeps(op: LocalBlock, laws: RowLaws, lanes: Lanes, R: int, MFS: int, MFR: int, contacts: bool) -> str:
    """The sweeps of a local owner: its dense rows in slot order, then its free-body rows, then the exit."""
    ctx = RowContext(
        row="i",
        kind="row_type",
        operator=op,
        load=lambda row: f"s_lam[{row}]",
        store=lambda row, value: lanes.first(f"s_lam[{row}] = {value};"),
        rhs=lambda row: f"s_rhs[{row}]",
        mu=lambda row: f"s_mu[{row}]",
        weight=lambda row: f"world_row_w.data[world_row_base + {row}]",
        parent="parent",
        ring_load=lambda parent: (
            f"RING_LOAD(s_lam[{parent}], {parent}, {R},\n"
            "    [&](int r) { return s_parent[r]; }, [&](int r) { return s_lam[r]; })"
        ),
        lanes=lanes,
        contact_kinds=(CONTACT, JOINT_LIMIT) if contacts else (JOINT_LIMIT,),
        record=lambda row, delta: lanes.first(f"s_app[{row}] += {delta};"),
        friction=contacts,
    )
    dense = [
        "for (int i = 0; i < row_count; ++i) {",
        "    if (s_active[i] == 0) continue;",
        "    const int row_type = s_type[i];",
        friction_gate("row_type", ctx.store("i", "0.0f"), ctx.sync(), "    ") if contacts else "",
        "    const float inverse_diagonal = s_inv_diag[i];",
        "    if (inverse_diagonal <= 0.0f) continue;",
        admit_pair("row_type", "s_parent[i]", "    ") if contacts else "",
        friction_radius("row_type", ctx.ring_load("parent"), "s_mu[i]", "s_lam[i]", "s_lam[i + 1]", "    ")
        if contacts
        else "",
        op.residual("jv", "    "),
        row_update(ctx, laws, residual="jv", pad="    "),
        "}",
    ]
    free = []
    if MFR:
        meta_parent = _mf_parent_expression("mf_meta.data[off_meta + patch_row * 4 + 3]", MFS)
        free_op = dataclasses.replace(
            op,
            first="row_count",
            inverse=lambda row: (
                "mf_inverse" if row == "i" else f"__int_as_float(mf_meta.data[off_meta + {row} * 4 + 1])"
            ),
        )
        free_ctx = dataclasses.replace(
            ctx,
            kind="mf_rt",
            operator=free_op,
            load=lambda row: f"s_mf_lam[{row}]",
            store=lambda row, value: lanes.first(f"s_mf_lam[{row}] = {value};"),
            mu=lambda row: f"mf_row_mu.data[off_mf + {row}]",
            weight=lambda row: f"mf_row_w.data[off_mf + {row}]",
            ring_load=lambda parent: (
                f"RING_LOAD(s_mf_lam[{parent}], {parent}, {MFR},\n"
                f"    [&](int patch_row) {{ return {meta_parent}; }},\n"
                "    [&](int patch_row) { return s_mf_lam[patch_row]; })"
            ),
            contact_kinds=(CONTACT,),
            record=lambda row, delta: lanes.first(f"s_app[row_count + {row}] += {delta};"),
        )
        free = [
            "for (int i = 0; i < mf_count; ++i) {",
            "    const int4 meta = *reinterpret_cast<const int4*>(&mf_meta.data[off_meta + i * 4]);",
            "    const float mf_inverse = __int_as_float(meta.y);",
            "    const int packed_tp = meta.w;",
            "    const int mf_rt = packed_tp & 0xFFFF;",
            friction_gate("mf_rt", free_ctx.store("i", "0.0f"), free_ctx.sync(), "    "),
            "    if (mf_inverse <= 0.0f) continue;",
            admit_pair("mf_rt", _mf_parent_expression("packed_tp", MFS), "    "),
            friction_radius(
                "mf_rt", free_ctx.ring_load("parent"), free_ctx.mu("i"), "s_mf_lam[i]", "s_mf_lam[i + 1]", "    "
            ),
            free_op.residual("jv", "    "),
            row_update(free_ctx, laws, residual="jv", pad="    "),
            "}",
        ]
    rows = "\n".join(line for line in (*dense, *free) if line)
    head = ["    for (int iteration = 0; iteration < iterations; ++iteration) {"]
    if contacts:
        head.append("        const int global_iter = iteration_offset + iteration;")
    head.append("        int iteration_changed = 0;")
    return "\n".join(
        [*head, indent(rows, "        "), end_sweep(laws.policy, iteration="iteration", gated=contacts) + "    }"]
    )


@cache
def local_owned_kernel(
    max_constraints: int,
    local_max_constraints: int,
    dof_count: int,
    device_arch: str,
    paired_dof_count: int = 0,
    persistent_queue: bool = False,
    *,
    warps_per_block: int = 1,
    lanes_per_world: int = 32,
    contact_capable: bool = True,
    mf_max_constraints: int = 0,
    local_mf_max_constraints: int = 0,
    row_inputs=None,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
) -> wp.Kernel:
    """A local owner's sweep kernel.

    ``row_inputs`` (:class:`.row_inputs.CompactLocalRows`) reads the rows' Jacobians from its packets, whose
    operands follow the ABI's; a solve readout's operand comes last.
    """
    snippet, staging = local_owned_source(
        max_constraints,
        local_max_constraints,
        dof_count,
        paired_dof_count,
        warps_per_block=warps_per_block,
        lanes_per_world=lanes_per_world,
        contact_capable=contact_capable,
        mf_max_constraints=mf_max_constraints,
        local_mf_max_constraints=local_mf_max_constraints,
        row_inputs=row_inputs,
        solve_policy=solve_policy,
    )
    staging.check("the local owner's sweep", arch_budget(device_arch))
    family = "sweep_local" if row_inputs is None else "sweep_local_packet"
    name = (
        f"{family}_{max_constraints}_{local_max_constraints}_{dof_count}_{paired_dof_count}"
        + (f"_w{warps_per_block}" if warps_per_block != 1 else "")
        + (f"_l{lanes_per_world}" if lanes_per_world != 32 else "")
        + ("" if contact_capable else "_contact0")
        + (f"_mf{local_mf_max_constraints}" if local_mf_max_constraints else "")
        + ("_queue" if persistent_queue else "")
        + solve_policy.suffix
    )
    queue = kernel_template.candidate_queue if persistent_queue else kernel_template.candidates
    arguments = () if row_inputs is None else row_inputs.arguments
    return KernelTemplate.from_abi(
        name,
        LOCAL_ABI,
        snippet,
        queue(staging.groups, lanes_per_world),
        extra=arguments + solve_policy.arguments,
        composed=True,
        native_name="sweep_local_native",
    ).kernel()


# ---------------------------------------------------------------------------------------------------------------------
# The paired factor sweep: rows in factor coordinates, a warp per world, three schedules
# ---------------------------------------------------------------------------------------------------------------------

PAIRED_FACTOR_ABI = kernel_template.params(
    ("world", int),
    ("world_constraint_count", wp.array[int]),
    ("dense_phase_bounds", wp.array2d[int]),
    ("world_dof_indices", wp.array2d[int]),
    ("rhs_bias", wp.array2d[float]),
    ("world_diag", wp.array2d[float]),
    ("world_impulses", wp.array2d[float]),
    ("factor_rows", wp.array3d[float]),
    ("world_row_type", wp.array2d[int]),
    ("world_row_parent", wp.array2d[int]),
    ("world_row_mu", wp.array2d[float]),
    ("world_mf_constraint_count", wp.array[int]),
    ("primary_L", wp.array3d[float]),
    ("secondary_L", wp.array3d[float]),
    ("primary_group_by_world", wp.array[int]),
    ("secondary_group_by_world", wp.array[int]),
    ("primary_offset_by_world", wp.array[int]),
    ("secondary_offset_by_world", wp.array[int]),
    ("iterations", int),
    ("omega", float),
    ("friction_start_iteration", int),
    ("iteration_offset", int),
    ("v_out", wp.array[float]),
)
"""The paired factor sweep's native ABI: one world per warp, two per block (``tiles(2, 32)``)."""

_PAIRED_WORLDS_PER_BLOCK = 2


def paired_factor_source(
    max_constraints: int,
    max_world_dofs: int,
    primary_dofs: int,
    secondary_dofs: int,
    contact_triples: bool = False,
    independent_components: bool = False,
    *,
    contact_units: bool = False,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
) -> tuple[str, Staging]:
    """The native body of the paired factor sweep: a world's rows in the factor coordinates of its articulation and
    free body (:class:`.FactorRows`), a warp per world, the factor velocity ``u = L^T v`` in a register per lane.

    The schedule visits the rows in slot order: row by row; with ``contact_triples``, when the world's rows are a
    prefix and then normal/tangent/tangent triples, the prefix row by row and each triple's normal then its tangent
    pair; with ``contact_units``, each row and then the tangent pair it owns (patch contacts own one when they hold
    an anchor). A pair solved as a unit reduces its sums in one ladder (:func:`.sweep.pair_update`). With
    ``independent_components`` a world whose signed free-body row count is negative solves only its articulation's
    coordinates (its free body's rows run in the matrix-free sweep).

    Returns the body and its staging.
    """
    M, D, P, S = int(max_constraints), int(max_world_dofs), int(primary_dofs), int(secondary_dofs)
    if M <= 0:
        raise ValueError("max_constraints must be positive")
    if P <= 0 or S <= 0 or P + S != D or D > 32:
        raise ValueError("paired factor PGS requires two positive components spanning at most 32 world DOFs")
    if contact_triples and contact_units:
        raise ValueError("contact triples and contact units are exclusive paired factor schedules")
    lanes = Lanes(32)
    laws = RowLaws(regularize=False, policy=solve_policy)
    op = FactorRows(
        D,
        (
            FactorComponent(P, "primary_offset", "primary_lane", "primary_L", "primary_matrix_base"),
            FactorComponent(S, "secondary_offset", "secondary_lane", "secondary_L", "secondary_matrix_base"),
        ),
        lanes,
        diagonal=lambda row: f"s_diag[{row}]",
    )
    staging = Staging(
        (
            Buffer("s_lam", "float", M),
            Buffer("s_rhs", "float", M),
            Buffer("s_diag", "float", M),
            Buffer("s_type", "unsigned char", M),
            *((Buffer("s_contact_mu", "float", (M + 2) // 3),) if contact_triples else ()),
        ),
        groups=_PAIRED_WORLDS_PER_BLOCK,
        threads=32 * _PAIRED_WORLDS_PER_BLOCK,
        name="PairedScratch",
    )
    ctx = RowContext(
        row="i",
        kind="row_type",
        operator=dataclasses.replace(op, diagonal=lambda row: "denom" if row == "i" else f"s_diag[{row}]"),
        load=lambda row: f"s_lam[{row}]",
        store=lambda row, value: lanes.first(f"s_lam[{row}] = {value};"),
        rhs=lambda row: f"s_rhs[{row}]",
        mu=lambda row: f"world_row_mu.data[off + {row}]",
        weight=lambda row: "1.0f",  # The paired route runs without regularization.
        parent="parent",
        ring_load=lambda parent: f"RING_LOAD(s_lam[{parent}], {parent}, m, ring_parent, ring_impulse)",
        lanes=lanes,
        inline_velocity_limit=True,
    )
    general = _paired_rows(ctx, laws, "m")
    if contact_triples:
        triples = _paired_rows(ctx, laws, "contact_start", friction=False) + "\n" + _paired_triples(ctx, laws)
        schedule = f"if (triple_layout) {{\n{indent(triples, '    ')}\n}} else {{\n{indent(general, '    ')}\n}}"
    elif contact_units:
        schedule = _paired_units(ctx, laws)
    else:
        schedule = general

    world_rows = "if (m == 0 || world_mf_constraint_count.data[world] != 0) return;"
    read = write = "global_dof >= 0"
    if independent_components:
        world_rows = (
            "if (m == 0 || world_mf_constraint_count.data[world] > 0) return;\n"
            "    const bool split_component = world_mf_constraint_count.data[world] < 0;"
        )
        read += f" && (!split_component || (lane >= primary_offset && lane < primary_offset + {P}))"
        write += f" && (!split_component || (primary_lane >= 0 && primary_lane < {P}))"
    layout = mu_load = ""
    if contact_triples:
        layout = """
    int contact_start = dense_phase_bounds.data[world * 2 + 1];
    if (contact_start < 0) contact_start = 0;
    if (contact_start > m) contact_start = m;
    // A prefix, then contacts that own exact normal/tangent/tangent triples.
    const bool triple_layout = dense_phase_bounds.data[world * 2] == contact_start && (m - contact_start) % 3 == 0;
    const int contact_count = (m - contact_start) / 3;"""
        mu_load = """
    for (int contact = lane; contact < contact_count; contact += 32)
        s_contact_mu[contact] = world_row_mu.data[off + contact_start + contact * 3 + 1];"""
    body = f"""{staging.declarations("    ")}
    const int lane = LANE;
    int m = world_constraint_count.data[world];
    if (m > {M}) m = {M};
    {world_rows}

    const int off = world * {M};
    const int row_base = off * {D};
    const int dof_map_base = world * {D};
    const int primary_group = primary_group_by_world.data[world];
    const int secondary_group = secondary_group_by_world.data[world];
    const int primary_offset = primary_offset_by_world.data[world];
    const int secondary_offset = secondary_offset_by_world.data[world];
    const int primary_matrix_base = primary_group * {P * P};
    const int secondary_matrix_base = secondary_group * {S * S};
    // A contact normal's patch ring links the region's next normal (-1 ends a point contact).
    const auto ring_parent = [&](int r) {{ return world_row_parent.data[off + r]; }};
    const auto ring_impulse = [&](int r) {{ return s_lam[r]; }};{layout}

    for (int i = lane; i < m; i += 32) {{
        s_lam[i] = world_impulses.data[off + i];
        s_rhs[i] = rhs_bias.data[off + i];
        s_diag[i] = world_diag.data[off + i];
        s_type[i] = static_cast<unsigned char>(world_row_type.data[off + i]);
    }}{mu_load}
    // The velocity in factor coordinates, u = L^T v per component, coordinate LANE in this lane's register.
    const int global_dof = lane < {D} ? world_dof_indices.data[dof_map_base + lane] : -1;
    const float physical_velocity = {read} ? v_out.data[global_dof] : 0.0f;
{op.enter("physical_velocity", "    ")}
    SYNC();

    for (int iter = 0; iter < iterations; ++iter) {{
        const int global_iter = iteration_offset + iter;
        int iteration_changed = 0;
{indent(schedule, "        ")}
{end_sweep(solve_policy, iteration="iter")}    }}

    // Back to physical coordinates, v = L^-T u per component.
{op.leave("physical_output", "    ")}
    if ({write}) v_out.data[global_dof] = physical_output;
    for (int i = lane; i < m; i += 32) world_impulses.data[off + i] = s_lam[i];
{staging.undefine()}
#endif
"""
    preludes = (lane_group(32), RING, *law_preludes(PROJECTION_LAW), *solve_policy.preludes)
    return cuda_native(body, preludes=preludes), staging


def _paired_rows(ctx: RowContext, laws: RowLaws, end: str, *, friction: bool = True) -> str:
    """Rows ``[0, end)`` one at a time in slot order, each row's factor row pipelined; a friction pair at its first
    tangent, its sibling's factor row loaded with the pair. Without ``friction`` the rows hold no friction pairs (a
    triple layout's prefix)."""
    op = ctx.operator
    sibling = f"const float {op.sibling} = row_type == {FRICTION} ? ({op.load('parent + 2')}) : 0.0f;\n"
    lines = [
        op.prefetch("0", ""),
        f"for (int i = 0; i < {end}; ++i) {{",
        op.consume("i", end, "    "),
        "    const int row_type = static_cast<int>(s_type[i]);",
        row_visit(
            dataclasses.replace(ctx, friction=friction),
            laws,
            parent="world_row_parent.data[off + i]",
            check="denom",
            declare=(sibling if friction else "") + "const float denom = s_diag[i];",
            pad="    ",
        ),
        "}",
    ]
    return "\n".join(line for line in lines if line)


def _pair_unit(ctx: RowContext, laws: RowLaws, mu: str) -> str:
    """The tangent pair ``(tangent1, tangent1 + 1)`` of the normal row ``i``, solved as a unit after it."""
    op = ctx.operator
    pair = dataclasses.replace(
        ctx,
        row="tangent1",
        kind=str(FRICTION),
        operator=dataclasses.replace(
            op, diagonal=lambda row: f"s_diag[{row}]", current="tangent1_z", sibling="tangent2_z"
        ),
        parent="i",
        inline_velocity_limit=False,
    )
    lines = [
        f"const float tangent1_z = {op.load('tangent1')};",
        f"const float tangent2_z = {op.load('tangent1 + 1')};",
        friction_gate(
            str(FRICTION), ctx.lanes.first("{ s_lam[tangent1] = 0.0f; s_lam[tangent1 + 1] = 0.0f; }"), ctx.sync(), ""
        ),
        "float radius = 0.0f;",
        pair_radius(ctx.ring_load("i"), mu, "s_lam[tangent1]", "s_lam[tangent1 + 1]", ""),
        pair_update(pair, laws, pad=""),
    ]
    return "\n".join(lines)


def _unit_row(ctx: RowContext, laws: RowLaws, kind: str) -> str:
    """Row ``i`` of a unit (its kind ``kind`` is never friction) by the row law, when its diagonal is positive."""
    op = ctx.operator
    row = dataclasses.replace(
        ctx,
        kind=kind,
        operator=dataclasses.replace(op, diagonal=lambda row: f"s_diag[{row}]"),
        friction=False,
        inline_velocity_limit=kind == "row_type",
    )
    lines = [
        f"const float {op.current} = {op.load('i')};",
        "if (s_diag[i] > 0.0f) {",
        op.residual("jv", "    "),
        row_update(row, laws, residual="jv", pad="    "),
        "}",
    ]
    return "\n".join(lines)


def _paired_triples(ctx: RowContext, laws: RowLaws) -> str:
    """The contact triples after the prefix: each normal row, then its tangent pair as a unit."""
    lines = [
        "for (int contact = 0; contact < contact_count; ++contact) {",
        "    const int i = contact_start + contact * 3;",
        "    const int tangent1 = i + 1;",
        indent(_unit_row(ctx, laws, str(CONTACT)), "    "),
        indent(_pair_unit(ctx, laws, "s_contact_mu[contact]"), "    "),
        "}",
    ]
    return "\n".join(lines)


def _paired_units(ctx: RowContext, laws: RowLaws) -> str:
    """Contact units in slot order: each row by the row law, then the tangent pair it owns as a unit.

    A row owns the next two rows when they are tangent rows naming it as parent. A tangent row outside a unit is
    zeroed while friction is delayed, and otherwise left alone.
    """
    lines = [
        "for (int i = 0, step = 1; i < m; i += step) {",
        "    const int row_type = static_cast<int>(s_type[i]);",
        "    step = 1;",
        f"    if (row_type == {FRICTION}) {{",
        friction_gate("row_type", ctx.store("i", "0.0f"), ctx.sync(), "        ", skip=False),
        "        continue;",
        "    }",
        "    const int tangent1 = i + 1;",
        f"    const bool owns_tangents = i + 2 < m && static_cast<int>(s_type[tangent1]) == {FRICTION}",
        "        && world_row_parent.data[off + tangent1] == i;",
        indent(_unit_row(ctx, laws, "row_type"), "    "),
        "    if (!owns_tangents) continue;",
        "    step = 3;",
        indent(_pair_unit(ctx, laws, "world_row_mu.data[off + tangent1]"), "    "),
        "}",
    ]
    return "\n".join(lines)


@cache
def paired_factor_kernel(
    max_constraints: int,
    max_world_dofs: int,
    primary_dofs: int,
    secondary_dofs: int,
    device_arch: str,
    contact_triples: bool = False,
    independent_components: bool = False,
    *,
    contact_units: bool = False,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
) -> wp.Kernel:
    """The paired factor sweep kernel; a readout's operand follows ``v_out``."""
    snippet, staging = paired_factor_source(
        max_constraints,
        max_world_dofs,
        primary_dofs,
        secondary_dofs,
        contact_triples,
        independent_components,
        contact_units=contact_units,
        solve_policy=solve_policy,
    )
    staging.check("the paired factor sweep", arch_budget(device_arch))
    name = (
        f"sweep_paired_factor_{max_constraints}_{max_world_dofs}_{primary_dofs}_{secondary_dofs}"
        + ("_contact3" if contact_triples else "")
        + ("_units" if contact_units else "")
        + ("_components" if independent_components else "")
        + solve_policy.suffix
    )
    return KernelTemplate.from_abi(
        name,
        PAIRED_FACTOR_ABI,
        snippet,
        kernel_template.tiles(_PAIRED_WORLDS_PER_BLOCK, 32),
        extra=solve_policy.arguments,
        composed=True,
        native_name="sweep_paired_factor_native",
    ).kernel()
