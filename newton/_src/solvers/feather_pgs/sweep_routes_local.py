# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The local owners' sweeps and the paired factor sweep as configurations of the sweep emitter (:mod:`.sweep`).

Every kernel's working set fits the static budget of the device it is built for (:func:`.staging.arch_budget`).

- :func:`local_owned_kernel`: one lane group (8, 16 or 32 lanes) per candidate articulation, or articulation and
  free body, with several groups per block. The group stages its rows, walks each row's seeds on the tree's factor in
  the kernel (or reads the one writer's rows) and solves the rows over the packed Gram block ``A = Z^T Z``
  (:class:`.response_operator.LocalBlock`), its residuals maintained and the velocity change committed after the
  sweeps, which leaves by the walk. The residual owner's free-body rows follow the dense rows in the block.
- :func:`paired_factor_kernel`: one warp per world of an articulation and a free body, rows in factor coordinates
  (:class:`.response_operator.FactorRows`) on the velocity change from the predictor, which leaves per component,
  under one of three schedules: row by row, contact triples after a prefix, or contact units.
"""

from __future__ import annotations

import dataclasses
from functools import cache

import warp as wp

from ...core.execution import Work, shape
from ...core.memory import Buffer, Staging, arch_budget, built_for
from . import field_group, kernel_template
from .device_preludes import RING, SHARED_STORAGE, cuda_native, lane_group
from .factor.walk import WalkTrees
from .field_group import Field
from .kernel_template import KernelTemplate
from .law_source import law_preludes
from .packed_rows import PACKED_SIDES
from .projection_laws import CONE_LAW, PROJECTION_LAW
from .response_operator import FactorComponent, FactorRows, FreeBodyZ, Lanes, LocalBlock
from .row_inputs import RowArgument, StoredLocalRows
from .row_kernels import PGS_CONSTRAINT_TYPE_SPEED
from .row_metadata import _mf_parent_expression
from .row_records import MfRows
from .solve_policy import DEFAULT_POLICY, SolvePolicy
from .sweep import (
    CONTACT,
    FRICTION,
    JOINT_LIMIT,
    PGS_LOCAL_SOLVE_OWNER_GENERAL,
    PGS_LOCAL_SOLVE_OWNER_PAIR,
    PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL,
    PGS_LOCAL_SOLVE_OWNER_SINGLE,
    TORSION,
    UNIT_HELD,
    ReducedPairCoupling,
    RowContext,
    RowLaws,
    UnitOperator,
    admit_pair,
    contact_unit,
    end_sweep,
    friction_radius,
    indent,
    material_arguments,
    pair_radius,
    pair_update,
    row_update,
    row_visit,
    unit_held_loads,
    weight_arguments,
)
from .sweep_routes_tiers import REGISTER_LINE

LOCAL_OWNERS = (
    Field("primary_articulation", wp.int32, ("world",), owner="sweep", attr="local_primary_articulation"),
    Field("pair_articulation", wp.int32, ("world",), owner="sweep", attr="local_pair_articulation"),
    Field("residual_pair_articulation", wp.int32, ("world",), owner="sweep", attr="local_residual_pair_articulation"),
    Field("max_rows", wp.int32, owner="sweep", attr="local_solve_max_rows"),
    Field("residual_max_rows", wp.int32, owner="sweep", attr="local_residual_max_rows"),
    Field("residual_mf_max_rows", wp.int32, owner="sweep", attr="local_residual_mf_max_rows"),
    Field("owner", wp.int32, ("world",), attr="local_solve_owner", owner="placement"),
    Field("general_world_count", wp.int32, (1,), owner="sweep", attr="local_general_world_count"),
    Field("general_worlds", wp.int32, ("world",), owner="sweep", attr="local_general_worlds"),
)
"""Each world's local-solve candidates (its primary articulation and paired articulations), the local row limits,
and the per-step owner of each world with the compact list of the general solver's worlds."""

LocalOwners = field_group.struct("LocalOwners", LOCAL_OWNERS, __name__)


# ---------------------------------------------------------------------------------------------------------------------
# The local owners: a lane group per articulation (and its free body), the response block built in the kernel
# ---------------------------------------------------------------------------------------------------------------------

LOCAL_ABI = kernel_template.params(
    ("candidate", int),
    ("candidate_articulations", wp.array[int]),
    ("candidate_secondary_articulations", wp.array[int]),
    ("expected_owner", int),
    ("articulation_world", wp.array[int]),
    ("articulation_dof_start", wp.array[int]),
    ("local_solve_owner", wp.array[int]),
    ("world_constraint_count", wp.array[int]),
    ("dense_first", wp.array[int]),
    ("packed_Z", wp.array3d[float]),
    ("row_sides", wp.array2d[wp.vec2i]),
    ("row_incident", wp.array2d[float]),
    ("response_dofs", wp.array[int]),
    ("articulation_world_dof_offset", wp.array[int]),
    ("packed_dense", int),
    ("rhs_bias", wp.array2d[float]),
    ("world_row_type", wp.array2d[int]),
    ("world_row_parent", wp.array2d[int]),
    ("world_row_mu", wp.array2d[float]),
    ("mf_constraint_count", wp.array[int]),
    ("mf_meta", wp.array2d[int]),
    ("mf_Z", wp.array3d[float]),
    ("regularize", int),
    ("world_row_w", wp.array2d[float]),
    ("iterations", int),
    ("omega", float),
    ("world_diag", wp.array2d[float]),
    ("world_impulses", wp.array2d[float]),
    ("v_out", wp.array[float]),
)
"""The local owners' native ABI: one candidate per lane group (a block's groups, or a queue of them). Every
local kernel also takes the walk (``walk``, :class:`.factor.walk.WalkTrees`), last of its row operands."""

WALK_ARGUMENT = RowArgument("walk", WalkTrees)
"""The tree factor's walk (:meth:`.tree_factor.TreeFactor.walk_view`), whose rows and leave the local kernels share."""


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
    walk_bodies: int = 0,
) -> Staging:
    """The shared working set of a local owner's block: per lane group the walk's data of its trees where the kernel
    walks seeds (``walk_bodies``, the primary tree's most bodies: :class:`.response_operator.LocalBlock`), the rows'
    whitened entries, impulses and metadata, the velocity, the paired free body's impulses and the packed Gram block.

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
    walk = LocalBlock(T, P, int(walk_bodies), staged=bool(walk_bodies), wrenches=contact_capable)
    return Staging(
        (
            *(
                (Buffer("s_walk", "float", walk.walk_floats()), Buffer("s_walk_path", "int", walk.walk_ints()))
                if walk_bodies
                else ()
            ),
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
    torsion: bool = False,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
    material: bool = False,
    walk_release: bool = False,
    walk_staged: bool = True,
    pitch: int | None = None,
) -> tuple[str, Staging]:
    """The native body of a local owner's sweep: an articulation's rows (and a paired free body's coordinates and,
    for the residual owner, its free-body rows) over the response block the lane group builds in the kernel. Under
    ``torsion`` a friction pair solves with its contact's spin row (the friction triple).

    The group stages the walk's data of its trees (where it walks seeds), the rows' impulses, right-hand sides and
    metadata and the predictor; takes each row's footprints on the walk of the trees' factors, ``z = R L^-1 P J^T``
    (``R`` the release of slipping drives, ``walk_release``; the walk staged in shared memory with ``walk_staged``
    where it walks seeds), and its incident velocity ``J v_hat`` into its
    right-hand side: from its seeds, walked in the kernel (``row_inputs``' packets,
    :class:`.row_inputs.CompactLocalRows`), or from the one writer's stored rows (:class:`.row_inputs.StoredLocalRows`);
    stores the effective diagonal ``|z|^2 + cfm`` back to ``world_diag`` and stages its inverse; builds the Gram block
    and the residuals (:class:`.LocalBlock`); then runs the sweeps with the shared row law, the dense rows in slot
    order and then the free-body rows, commits the velocity change, which leaves the walk's coordinates and adds to
    the predictor, and the impulses of the rows with a footprint.

    Returns the body and its staging.
    """
    M, R, P, S = int(max_constraints), int(local_max_constraints), int(dof_count), int(paired_dof_count)
    T, MFS, MFR, L = P + S, int(mf_max_constraints), int(local_mf_max_constraints), int(lanes_per_world)
    _check_local_options(warps_per_block, L, contact_capable, T, S, MFS, MFR)
    # The row store's line: the dense rows, then the free bodies' from M (:mod:`.row_records`).
    pitch = M + MFS if pitch is None else int(pitch)
    lanes = Lanes(L)
    free = FreeBodyZ(lanes) if MFR else None
    if torsion and not contact_capable:
        raise ValueError("spin rows (contact torsion) are contact rows")
    laws = RowLaws(
        regularize=contact_capable, material=material and contact_capable, torsion=torsion, policy=solve_policy
    )
    # The packets' seeds are walked here, on the world's walk staged in shared memory; the one writer's stored rows
    # are read, and the walk only for the leave, in place.
    bodies = 0 if row_inputs is None else row_inputs.bodies
    staged = row_inputs is not None and walk_staged
    op = LocalBlock(
        T,
        P,
        bodies,
        seeds=row_inputs is not None,
        staged=staged,
        wrenches=contact_capable,
        release=walk_release,
        lanes=lanes,
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
        walk_bodies=bodies if staged else 0,
    )

    # The rows: the packets' seeds walked here, or the one writer's stored rows (the class walks through it).
    source = StoredLocalRows() if row_inputs is None else row_inputs

    pair = ""
    dof = "articulation_dof_start.data[art] + d"
    if S:
        pair = """
    const int secondary_art = candidate_secondary_articulations.data[candidate];
    if (articulation_world.data[secondary_art] != world) return;"""
        dof = f"d < {P} ? articulation_dof_start.data[art] + d : articulation_dof_start.data[secondary_art] + d - {P}"
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
    const int off_mf = line * {pitch} + {M};
    const int off_meta = line * {MFS * 4};
    const int mf12_base = line * {MFS * 12};"""
        mf_load = (
            f"\n    for (int row = lane; row < mf_count; row += {L}) s_mf_lam[row] = world_impulses.data[off_mf + row];"
        )
        mf_store = (
            f"\n    for (int row = lane; row < mf_count; row += {L}) world_impulses.data[off_mf + row] = s_mf_lam[row];"
        )

    sweeps = _local_sweeps(op, laws, lanes, R, MFS, MFR, contact_capable)
    body = f"""    const int lane = LANE;
    const int art = candidate_articulations.data[candidate];
    const int world = articulation_world.data[art];
    if (local_solve_owner.data[world] != expected_owner) return;
    const int row_count = world_constraint_count.data[world];
    if (row_count == 0 || row_count > {R}) return;

    const int line = dense_first.data[world];
    __builtin_assume(line >= 0);
    const int world_row_base = line * {pitch};
    const int packed_row_base = line * {M};{pair}
{staging.declarations("    ")}{mf_setup}
{indent(op.prologue(), "    ")}
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

    // Each row's incident velocity and footprints on the walk, z = R L^-1 P J^T, its effective diagonal |z|^2 + cfm
    // (stored back) and its inverse.
    for (int row = lane; row < row_count; row += {L}) {{
{op.respond(source, "        ")}
        s_rhs[row] += incident;
        s_active[row] = active;
        const float effective_diagonal = {source.diagonal("row_diagonal")};
        s_inv_diag[row] = effective_diagonal > 0.0f ? 1.0f / effective_diagonal : 0.0f;
        world_diag.data[world_row_base + row] = effective_diagonal;
    }}
    SYNC();
{op.assemble(lambda row: f"s_rhs[{row}]", "    ")}
    SYNC();

{sweeps}

{op.commit("    ")}
    SYNC();
    // The velocity change in the walk's coordinates to the step's one leave (tree_factor.walk_leave_kernel), which
    // walks each tree down, dv = P^T L^-T R^T u.
    for (int d = lane; d < {T}; d += {L}) {{
        const int dof = {dof};
        walk.du.data[dof] += s_v[d];
    }}
    for (int row = lane; row < row_count; row += {L}) {{
        if (s_active[row] != 0) world_impulses.data[world_row_base + row] = s_lam[row];
    }}{mf_store}
{staging.undefine()}
#endif
"""
    preludes = (
        lane_group(L),
        SHARED_STORAGE,
        *((RING,) if contact_capable else ()),
        PACKED_SIDES,
        *law_preludes(PROJECTION_LAW, *((CONE_LAW,) if torsion else ())),
        *solve_policy.preludes,
        *op.walk_preludes(),
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
        eps=lambda row: f"world_row_eps.data[world_row_base + {row}]",
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
    torsion = laws.torsion

    def loop(count: str, head: tuple[str, ...], row: RowContext, parent: str, impulses: str, pairs: bool) -> list[str]:
        """One class's rows in slot order: ``head`` declares the row's kind (skipping a spin row, which its pair
        solves) and its stored inverse diagonal; with ``pairs`` a pair is admitted at its first tangent (``parent``)
        and visited on its disk (the class's ``impulses``)."""
        spin = {"load_spin": f"{impulses}[i + 2]"} if torsion else {}
        kind = row.kind
        return [
            f"for (int i = 0; i < {count}; ++i) {{",
            *head,
            f"    if ({row.operator.inverse('i')} <= 0.0f) continue;",
            admit_pair(kind, parent, "    ", torsion=torsion) if pairs else "",
            friction_radius(
                kind, row.ring_load("parent"), row.mu("i"), f"{impulses}[i]", f"{impulses}[i + 1]", "    ", **spin
            )
            if pairs
            else "",
            row.operator.residual("jv", "    "),
            row_update(row, laws, residual="jv", pad="    "),
            "}",
        ]

    dense = loop(
        "row_count",
        (
            "    if (s_active[i] == 0) continue;",
            "    const int row_type = s_type[i];",
            *((f"    if (row_type == {TORSION}) continue;",) if torsion else ()),
            "    const float inverse_diagonal = s_inv_diag[i];",
        ),
        ctx,
        "s_parent[i]",
        "s_lam",
        contacts,
    )
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
            mu=lambda row: f"world_row_mu.data[off_mf + {row}]",
            weight=lambda row: f"world_row_w.data[off_mf + {row}]",
            eps=lambda row: f"world_row_eps.data[off_mf + {row}]",
            ring_load=lambda parent: (
                f"RING_LOAD(s_mf_lam[{parent}], {parent}, {MFR},\n"
                f"    [&](int patch_row) {{ return {meta_parent}; }},\n"
                "    [&](int patch_row) { return s_mf_lam[patch_row]; })"
            ),
            contact_kinds=(CONTACT,),
            record=lambda row, delta: lanes.first(f"s_app[row_count + {row}] += {delta};"),
        )
        free = loop(
            "mf_count",
            (
                "    const int4 meta = *reinterpret_cast<const int4*>(&mf_meta.data[off_meta + i * 4]);",
                "    const float mf_inverse = __int_as_float(meta.y);",
                "    const int packed_tp = meta.w;",
                "    const int mf_rt = packed_tp & 0xFFFF;",
                *((f"    if (mf_rt == {TORSION}) continue;",) if torsion else ()),
            ),
            free_ctx,
            _mf_parent_expression("packed_tp", MFS),
            "s_mf_lam",
            True,
        )
    rows = "\n".join(line for line in (*dense, *free) if line)
    head = ["    for (int iteration = 0; iteration < iterations; ++iteration) {", "        int iteration_changed = 0;"]
    return "\n".join([*head, indent(rows, "        "), end_sweep(laws.policy, iteration="iteration") + "    }"])


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
    torsion: bool = False,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
    material: bool = False,
    walk_release: bool = False,
    walk_staged: bool = True,
    pitch: int | None = None,
) -> wp.Kernel:
    """A local owner's sweep kernel.

    ``row_inputs`` (:class:`.row_inputs.CompactLocalRows`) walks the rows' seeds from its packets, whose operands
    follow the ABI's, then the walk's; without it the kernel reads the one writer's rows. A solve readout's operand
    comes last.
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
        torsion=torsion,
        solve_policy=solve_policy,
        material=material,
        walk_release=walk_release,
        walk_staged=walk_staged,
        pitch=pitch,
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
        + ("_material" if material else "")
        + ("_torsion" if torsion else "")
        + (f"_walk{row_inputs.bodies}" if row_inputs is not None else "")
        + ("_inplace" if row_inputs is not None and not walk_staged else "")
        + ("_release" if walk_release else "")
        + ("" if pitch is None or pitch == max_constraints + mf_max_constraints else f"_p{int(pitch)}")
        + solve_policy.suffix
    )
    # The candidates a launch covers: each block's lane groups, or a device list walked with the grid's stride.
    queue = ("active_candidate_count", "candidate_grid_stride") if persistent_queue else None
    arguments = (
        *(() if row_inputs is None else row_inputs.arguments),
        WALK_ARGUMENT,
        *material_arguments(material),
    )
    return KernelTemplate.from_abi(
        name,
        LOCAL_ABI,
        snippet,
        kernel_template.items("candidate", groups=staging.groups, lanes=lanes_per_world, queue=queue),
        extra=arguments + solve_policy.arguments,
        composed=True,
        native_name="sweep_local_native",
        # ptxas fits the registers of the kernel's longest phase, the walk of a row's seeds, to the occupancy shared
        # memory admits under the line, so the walk costs the sweeps no resident block.
        launch_bounds=shape(
            Work(0, block=staging.threads, registers=REGISTER_LINE, shared=staging.bytes), built_for(device_arch)
        ).bounds,
    ).kernel()


# ---------------------------------------------------------------------------------------------------------------------
# The paired factor sweep: rows in factor coordinates, a warp per world, three schedules
# ---------------------------------------------------------------------------------------------------------------------

PAIRED_FACTOR_ABI = kernel_template.params(
    ("world", int),
    ("world_constraint_count", wp.array[int]),
    ("dense_first", wp.array[int]),
    ("dense_phase_bounds", wp.array2d[int]),
    ("world_dof_indices", wp.array2d[int]),
    ("rhs_bias", wp.array2d[float]),
    ("row_incident", wp.array2d[float]),
    ("world_diag", wp.array2d[float]),
    ("world_impulses", wp.array2d[float]),
    ("factor_rows", wp.array3d[float]),
    ("world_row_type", wp.array2d[int]),
    ("world_row_parent", wp.array2d[int]),
    ("world_row_mu", wp.array2d[float]),
    ("world_mf_constraint_count", wp.array[int]),
    ("primary_offset_by_world", wp.array[int]),
    ("secondary_offset_by_world", wp.array[int]),
    ("iterations", int),
    ("omega", float),
    ("v_out", wp.array[float]),
)
"""The paired factor sweep's native ABI: one world per warp, two per block (a tail-guarded tiled dispatch)."""

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
    torsion: bool = False,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
    weighted: bool = False,
    material: bool = False,
    pitch: int | None = None,
) -> tuple[str, Staging]:
    """The native body of the paired factor sweep: a world's rows in the factor coordinates of its articulation and
    free body (:class:`.FactorRows`), a warp per world, the velocity change from the predictor, ``u = L^T dv``, from
    zero in a register per lane.
    Under ``torsion`` every friction pair has its contact's spin row after it: a contact is four rows, and its unit
    solves the pair and the spin row as the friction triple.

    The schedule visits the rows in slot order: row by row; with ``contact_triples``, when the world's rows are a
    prefix and then normal/tangent/tangent triples, the prefix row by row and each triple as a contact unit; with
    ``contact_units``, each row, and a row with the tangent pair it owns (patch contacts own one when they hold an
    anchor) as a contact unit (:func:`_contact_unit`: a loaded unit on its Gram block, one ladder and one apply). With
    ``independent_components`` a world whose signed free-body row count is negative solves only its articulation's
    coordinates (its free body's rows run in the matrix-free sweep).

    The rows are the walk's (:mod:`.walk_rows`): the velocity change in the walk's coordinates goes to the step's one
    leave (:attr:`.factor.walk.WalkTrees.du`, :func:`.factor.walk.walk_leave_kernel`), which walks each component's
    tree down in the rows' frame.

    Returns the body and its staging.
    """
    M, D, P, S = int(max_constraints), int(max_world_dofs), int(primary_dofs), int(secondary_dofs)
    # The row store's line (its dense rows, then the free bodies'); the packed rows keep M rows a world.
    pitch = M if pitch is None else int(pitch)
    if M <= 0:
        raise ValueError("max_constraints must be positive")
    if P <= 0 or S <= 0 or P + S != D or D > 32:
        raise ValueError("paired factor PGS requires two positive components spanning at most 32 world DOFs")
    if contact_triples and contact_units:
        raise ValueError("contact triples and contact units are exclusive paired factor schedules")
    lanes = Lanes(32)
    laws = RowLaws(regularize=weighted, material=material, torsion=torsion, policy=solve_policy)
    stride = 3 + int(torsion)
    op = FactorRows(
        D,
        (
            FactorComponent(P, "primary_offset", "primary_lane"),
            FactorComponent(S, "secondary_offset", "secondary_lane"),
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
            *((Buffer("s_contact_mu", "float", (M + stride - 1) // stride),) if contact_triples else ()),
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
        weight=lambda row: f"world_row_w.data[off + {row}]",
        eps=lambda row: f"world_row_eps.data[off + {row}]",
        parent="parent",
        ring_load=lambda parent, load=None: (
            f"RING_LOAD({load or f's_lam[{parent}]'}, {parent}, m, ring_parent, ring_impulse)"
        ),
        lanes=lanes,
        inline_velocity_limit=True,
    )
    general = _paired_rows(ctx, laws, "m")
    if contact_triples:
        triples = _paired_rows(ctx, laws, "contact_start", friction=False) + "\n" + _paired_triples(ctx, laws, stride)
        schedule = f"if (triple_layout) {{\n{indent(triples, '    ')}\n}} else {{\n{indent(general, '    ')}\n}}"
    elif contact_units:
        schedule = _paired_units(ctx, laws)
    else:
        schedule = general

    world_rows = "if (m == 0 || world_mf_constraint_count.data[world] != 0) return;"
    write = "global_dof >= 0"
    if independent_components:
        world_rows = (
            "if (m == 0 || world_mf_constraint_count.data[world] > 0) return;\n"
            "    const bool split_component = world_mf_constraint_count.data[world] < 0;"
        )
        write += f" && (!split_component || (primary_lane >= 0 && primary_lane < {P}))"
    layout = mu_load = ""
    if contact_triples:
        rows = "normal/tangent/tangent/spin quads" if torsion else "normal/tangent/tangent triples"
        layout = f"""
    int contact_start = dense_phase_bounds.data[world * 2 + 1];
    if (contact_start < 0) contact_start = 0;
    if (contact_start > m) contact_start = m;
    // A prefix, then contacts that own exact {rows}.
    const bool triple_layout = dense_phase_bounds.data[world * 2] == contact_start && (m - contact_start) % {stride} == 0;
    const int contact_count = (m - contact_start) / {stride};"""
        mu_load = f"""
    for (int contact = lane; contact < contact_count; contact += 32)
        s_contact_mu[contact] = world_row_mu.data[off + contact_start + contact * {stride} + 1];"""
    leave = f"""    for (int i = lane; i < m; i += 32) world_impulses.data[off + i] = s_lam[i];
    // The velocity change in the walk's coordinates, each lane's coordinate, to the step's one leave
    // (tree_factor.walk_leave_kernel), which walks each component's tree down in the rows' frame.
    if ({write}) walk.du.data[global_dof] += {op.velocity};"""
    body = f"""{staging.declarations("    ")}
    const int lane = LANE;
    int m = world_constraint_count.data[world];
    if (m > {M}) m = {M};
    {world_rows}

    const int line = dense_first.data[world];
    __builtin_assume(line >= 0);
    const int off = line * {pitch};
    const int row_base = line * {M * D};
    const int dof_map_base = world * {D};
    const int primary_offset = primary_offset_by_world.data[world];
    const int secondary_offset = secondary_offset_by_world.data[world];
    // A contact normal's patch ring links the region's next normal (-1 ends a point contact).
    const auto ring_parent = [&](int r) {{ return world_row_parent.data[off + r]; }};
    const auto ring_impulse = [&](int r) {{ return s_lam[r]; }};{layout}

    for (int i = lane; i < m; i += 32) {{
        s_lam[i] = world_impulses.data[off + i];
        s_rhs[i] = rhs_bias.data[off + i] + row_incident.data[off + i];
        s_diag[i] = world_diag.data[off + i];
        s_type[i] = static_cast<unsigned char>(world_row_type.data[off + i]);
    }}{mu_load}
    // The velocity change in factor coordinates per component, coordinate LANE in this lane's register.
    const int global_dof = lane < {D} ? world_dof_indices.data[dof_map_base + lane] : -1;
{chr(10).join(op.lanes_of("    "))}
    float {op.velocity} = 0.0f;
    SYNC();

    for (int iter = 0; iter < iterations; ++iter) {{
        int iteration_changed = 0;
{indent(schedule, "        ")}
{end_sweep(solve_policy, iteration="iter")}    }}

{leave}
{staging.undefine()}
#endif
"""
    cone = (CONE_LAW,) if torsion else ()
    preludes = (
        lane_group(32),
        SHARED_STORAGE,
        RING,
        *law_preludes(PROJECTION_LAW, *cone),
        *solve_policy.preludes,
    )
    return cuda_native(body, preludes=preludes), staging


def _paired_rows(ctx: RowContext, laws: RowLaws, end: str, *, friction: bool = True) -> str:
    """Rows ``[0, end)`` one at a time in slot order, each row's factor row pipelined; a friction pair at its first
    tangent, its sibling's factor row loaded with the pair. Without ``friction`` the rows hold no friction pairs (a
    triple layout's prefix)."""
    op = ctx.operator
    sibling = f"const float {op.sibling} = row_type == {FRICTION} ? ({op.load('parent + 2')}) : 0.0f;\n"
    if laws.torsion:
        # The pair's contact's spin row, solved with it (the friction triple).
        sibling += f"const float {op.spin} = row_type == {FRICTION} ? ({op.load('parent + 3')}) : 0.0f;\n"
    lines = [
        op.prefetch("0", ""),
        f"for (int i = 0; i < {end}; ++i) {{",
        op.consume("i", end, "    "),
        "    const int row_type = static_cast<int>(s_type[i]);",
        f"    if (row_type == {TORSION}) continue;" if laws.torsion and friction else "",
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


@dataclasses.dataclass(frozen=True)
class _UnitFactorRows(UnitOperator, ReducedPairCoupling, FactorRows):
    """A contact unit's factor rows (:class:`.sweep.UnitOperator`): the unit's ladder (:func:`_contact_unit`) also
    reduces the pair's coupling and the rows' terms' magnitudes, so a moved tangent votes on a lower bound of its terms
    that is the row-by-row sweep's own when the normal did not move."""

    magnitudes = True


def _contact_unit(ctx: RowContext, laws: RowLaws, kind: str, mu: str) -> str:
    """The contact unit headed by row ``head`` (of kind ``kind``, never friction) with its tangent pair ``(head + 1,
    head + 2)`` (friction coefficient ``mu`` at the visited row ``i``), its three factor rows in registers, by the unit
    law (:func:`.sweep.contact_unit`); under torsion with its spin row ``head + 3`` too, its fourth.

    The unit's impulses choose its reductions, not its law. A loaded unit (a nonzero impulse among its rows) reduces its
    rows' velocity terms, the pair's couplings with the normal and their terms' magnitudes in one ladder, runs its rows
    on :class:`_UnitFactorRows` on the impulses its test loaded and applies their changes in one pass, after which the
    lanes synchronize once (:func:`.sweep.contact_unit`). An unloaded one visits its rows as the row-by-row
    sweep does: the normal on its own ladder (:func:`_unit_row`), then its pair on the pair's own (:func:`_pair_unit`),
    only on a nonzero disk, so a unit that stays inactive reduces the normal's terms alone.
    """
    op = ctx.operator
    velocity = op.velocity
    torsion = laws.torsion
    unit = _UnitFactorRows(**{field.name: getattr(op, field.name) for field in dataclasses.fields(op)})
    channels = (
        ("unit_jv0", "row_z", velocity),
        ("unit_jv1", "tangent1_z", velocity),
        ("unit_jv2", "tangent2_z", velocity),
        ("gram_n1", "row_z", "tangent1_z"),
        ("gram_n2", "row_z", "tangent2_z"),
        ("unit_cross", "tangent1_z", "tangent2_z"),
    )
    if torsion:
        channels += (
            ("unit_jv3", op.spin, velocity),
            ("gram_n3", "row_z", op.spin),
            ("unit_cross13", "tangent1_z", op.spin),
            ("unit_cross23", "tangent2_z", op.spin),
        )
    spin_mu = ctx.mu("head + 3")
    rows = dataclasses.replace(
        ctx, kind="unit_type", mu=lambda row: spin_mu if row == "spin" else mu, inline_velocity_limit=False
    )
    declare = (
        f"const int i = head + r;\nconst int unit_type = r == 0 ? {kind} : {FRICTION};\nconst float denom = s_diag[i];"
    )

    diagonal = lambda row: "denom" if row == "i" else f"s_diag[{row}]"  # noqa: E731
    unit_ctx = dataclasses.replace(rows, operator=dataclasses.replace(unit, diagonal=diagonal))
    unit_rows = contact_unit(
        unit_ctx, laws, rows="3", declare=declare, parent="head", check="denom", held=True, pad="    "
    )

    impulses, *staged = UNIT_HELD
    if torsion:
        lines = [
            f"const float row_z = {op.load('head')};",
            f"const float tangent1_z = {op.load('head + 1')};",
            f"const float tangent2_z = {op.load('head + 2')};",
            f"const float {op.spin} = {op.load('head + 3')};",
            unit_held_loads(unit_ctx, impulses, "head", "", torsion=True),
            "if (unit_lam0 != 0.0f || unit_lam1 != 0.0f || unit_lam2 != 0.0f || unit_lam3 != 0.0f) {",
            *(unit_held_loads(unit_ctx, name, "head", "    ", torsion=True) for name in staged),
            "    // The unit's velocity terms, its rows' Gram couplings and their magnitudes, in one ladder.",
            unit._sums(channels, "    ", magnitudes=("gram_n1", "gram_n2", "gram_n3")),
            unit_rows,
            "    // The unit's changes in one pass; then its stores are every lane's.",
            "    if (unit_dn != 0.0f || unit_dt1 != 0.0f || unit_dt2 != 0.0f || unit_ds != 0.0f)",
            f"        {velocity} += ((row_z * unit_dn + tangent1_z * unit_dt1) + tangent2_z * unit_dt2)"
            f" + {op.spin} * unit_ds;",
        ]
    else:
        lines = [
            f"const float row_z = {op.load('head')};",
            f"const float tangent1_z = {op.load('head + 1')};",
            f"const float tangent2_z = {op.load('head + 2')};",
            unit_held_loads(unit_ctx, impulses, "head", ""),
            "if (unit_lam0 != 0.0f || unit_lam1 != 0.0f || unit_lam2 != 0.0f) {",
            *(unit_held_loads(unit_ctx, name, "head", "    ") for name in staged),
            "    // The unit's velocity terms, the pair's Gram couplings with the normal and their magnitudes, in one "
            "ladder.",
            unit._sums(channels, "    ", magnitudes=("gram_n1", "gram_n2")),
            unit_rows,
            "    // The unit's changes in one pass: u += Z_n dn + Z_t1 dt1 + Z_t2 dt2; then its stores are every lane's.",
            "    if (unit_dn != 0.0f || unit_dt1 != 0.0f || unit_dt2 != 0.0f)",
            f"        {velocity} += (row_z * unit_dn + tangent1_z * unit_dt1) + tangent2_z * unit_dt2;",
        ]
    lines += [
        f"    {ctx.lanes.sync()}",
        "} else {",
        "    const int i = head, tangent1 = head + 1;",
        indent(_unit_row(ctx, laws, kind, load=False), "    "),
        indent(_pair_unit(ctx, laws, mu, spin_mu), "    "),
        "}",
    ]
    return "\n".join(lines)


def _pair_unit(ctx: RowContext, laws: RowLaws, mu: str, spin_mu: str) -> str:
    """The tangent pair ``(tangent1, tangent1 + 1)`` of the normal row ``i`` (its rows in ``tangent1_z``,
    ``tangent2_z``), solved as a unit after it on its own ladder (:func:`.sweep.pair_update`); under torsion with its
    spin row (``tangent1 + 2``, coefficient ``spin_mu``), the friction triple."""
    op = ctx.operator
    pair = dataclasses.replace(
        ctx,
        row="tangent1",
        kind=str(FRICTION),
        operator=dataclasses.replace(
            op, diagonal=lambda row: f"s_diag[{row}]", current="tangent1_z", sibling="tangent2_z"
        ),
        mu=lambda row: spin_mu if row == "spin" else mu,
        parent="i",
        inline_velocity_limit=False,
    )
    spin = {}
    if laws.torsion:
        spin = {"load_spin": "s_lam[tangent1 + 2]"}
    lines = [
        "float radius = 0.0f, load = 0.0f;" if laws.torsion else "float radius = 0.0f;",
        pair_radius(ctx.ring_load("i"), mu, "s_lam[tangent1]", "s_lam[tangent1 + 1]", "", **spin),
        pair_update(pair, laws, pad=""),
    ]
    return "\n".join(lines)


def _unit_row(ctx: RowContext, laws: RowLaws, kind: str, *, load: bool = True) -> str:
    """Row ``i`` of a unit (its kind ``kind`` is never friction) by the row law, when its diagonal is positive; its
    factor row loaded here, or (without ``load``) already."""
    op = ctx.operator
    row = dataclasses.replace(
        ctx,
        kind=kind,
        operator=dataclasses.replace(op, diagonal=lambda row: f"s_diag[{row}]"),
        friction=False,
        inline_velocity_limit=kind == "row_type",
    )
    lines = [
        f"const float {op.current} = {op.load('i')};" if load else "",
        "if (s_diag[i] > 0.0f) {",
        op.residual("jv", "    "),
        row_update(row, laws, residual="jv", pad="    "),
        "}",
    ]
    return "\n".join(line for line in lines if line)


def _paired_triples(ctx: RowContext, laws: RowLaws, stride: int) -> str:
    """The contact triples (quads under torsion: ``stride`` rows) after the prefix, each a contact unit
    (:func:`_contact_unit`)."""
    lines = [
        "for (int contact = 0; contact < contact_count; ++contact) {",
        f"    const int head = contact_start + contact * {stride};",
        indent(_contact_unit(ctx, laws, str(CONTACT), "s_contact_mu[contact]"), "    "),
        "}",
    ]
    return "\n".join(lines)


def _paired_units(ctx: RowContext, laws: RowLaws) -> str:
    """Contact units in slot order: a row by the row law, or, with the tangent pair it owns, a contact unit.

    A row owns the next two rows when they are tangent rows naming it as parent (and, under torsion, the spin row
    after them). A tangent row outside a unit is left alone.
    """
    lines = [
        "for (int i = 0, step = 1; i < m; i += step) {",
        "    const int row_type = static_cast<int>(s_type[i]);",
        "    step = 1;",
        *((f"    if (row_type == {TORSION}) continue;",) if laws.torsion else ()),
        f"    if (row_type == {FRICTION}) continue;",
        "    const int tangent1 = i + 1;",
        f"    const bool owns_tangents = i + 2 < m && static_cast<int>(s_type[tangent1]) == {FRICTION}",
        "        && world_row_parent.data[off + tangent1] == i;",
        "    if (!owns_tangents) {",
        indent(_unit_row(ctx, laws, "row_type"), "        "),
        "        continue;",
        "    }",
        f"    step = {4 if laws.torsion else 3};",
        "    const int head = i;",
        indent(_contact_unit(ctx, laws, "row_type", "world_row_mu.data[off + tangent1]"), "    "),
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
    torsion: bool = False,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
    weighted: bool = False,
    material: bool = False,
    pitch: int | None = None,
) -> wp.Kernel:
    """The paired factor sweep kernel; weighted rows' and material contacts' operands (:func:`.sweep.weight_arguments`,
    :func:`.sweep.material_arguments`), a readout's operand and the walk's view follow ``v_out``."""
    snippet, staging = paired_factor_source(
        max_constraints,
        max_world_dofs,
        primary_dofs,
        secondary_dofs,
        contact_triples,
        independent_components,
        contact_units=contact_units,
        torsion=torsion,
        solve_policy=solve_policy,
        weighted=weighted,
        material=material,
        pitch=pitch,
    )
    staging.check("the paired factor sweep", arch_budget(device_arch))
    name = (
        f"sweep_paired_factor_{max_constraints}_{max_world_dofs}_{primary_dofs}_{secondary_dofs}"
        + ("_contact3" if contact_triples else "")
        + ("_units" if contact_units else "")
        + ("_components" if independent_components else "")
        + ("_weighted" if weighted else "")
        + ("_material" if material else "")
        + ("_torsion" if torsion else "")
        + ("" if pitch is None or pitch == max_constraints else f"_p{int(pitch)}")
        + "_walk"
        + solve_policy.suffix
    )
    walked = (RowArgument("walk", WalkTrees),)
    return KernelTemplate.from_abi(
        name,
        PAIRED_FACTOR_ABI,
        snippet,
        kernel_template.items("world", groups=_PAIRED_WORLDS_PER_BLOCK, lanes=32, guard="world_count"),
        extra=(*weight_arguments(weighted), *material_arguments(material), *solve_policy.arguments, *walked),
        composed=True,
        native_name="sweep_paired_factor_native",
    ).kernel()


@wp.kernel
def local_solve_launch_gate():
    """Create a minimal graph dependency ahead of the bulk local solves."""

    pass


@wp.kernel
def classify_local_solve_worlds(
    mf: MfRows,
    owners: LocalOwners,
    world_constraint_count: wp.array[int],
    dense_phase_bounds: wp.array2d[int],
    mf_constraint_count: wp.array[int],
    articulation_dof_count: wp.array[int],
):
    """Assign exact solver ownership and compact active general worlds."""
    world = wp.tid()
    mf_line = mf.first[world]
    row_count = world_constraint_count[world]
    mf_count = mf_constraint_count[world]
    primary_articulation = owners.primary_articulation[world]
    pair_articulation = owners.pair_articulation[world]
    residual_pair_articulation = owners.residual_pair_articulation[world]
    single_phase = dense_phase_bounds[world, 1] == row_count

    local_mf = mf_count > 0 and mf_count <= owners.residual_mf_max_rows and residual_pair_articulation >= 0
    mf_row = int(0)
    while mf_row < mf_count and local_mf:
        sides = mf.sides[mf_line, mf.base + mf_row]
        if sides[0] >= 0 and sides[0] != residual_pair_articulation:
            local_mf = False
        if sides[1] >= 0 and sides[1] != residual_pair_articulation:
            local_mf = False
        # The residual local loop does not implement the speed law (projection_laws.speed_row): keep worlds with speed
        # rows on the general owner.
        if mf.row_type[mf_line, mf.base + mf_row] == PGS_CONSTRAINT_TYPE_SPEED:
            local_mf = False
        mf_row += 1

    owner = PGS_LOCAL_SOLVE_OWNER_GENERAL
    single_row_capacity = int(0)
    if primary_articulation >= 0:
        single_row_capacity = wp.min(owners.max_rows, articulation_dof_count[primary_articulation])
    if row_count > 0 and mf_count == 0:
        if single_phase and row_count <= single_row_capacity:
            owner = PGS_LOCAL_SOLVE_OWNER_SINGLE
        elif not single_phase and row_count <= owners.max_rows and pair_articulation >= 0:
            owner = PGS_LOCAL_SOLVE_OWNER_PAIR
    if owner == PGS_LOCAL_SOLVE_OWNER_GENERAL and (
        row_count > 0
        and row_count <= owners.residual_max_rows
        and residual_pair_articulation >= 0
        and ((mf_count == 0 and not single_phase) or local_mf)
    ):
        owner = PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL
    owners.owner[world] = owner
    if owner == PGS_LOCAL_SOLVE_OWNER_GENERAL and (row_count > 0 or mf_count > 0):
        general_index = wp.atomic_add(owners.general_world_count, 0, 1)
        owners.general_worlds[general_index] = world


@wp.kernel
def compact_local_pair_candidates(
    candidate_articulations: wp.array[int],
    candidate_secondary_articulations: wp.array[int],
    articulation_world: wp.array[int],
    local_solve_owner: wp.array[int],
    expected_owner: int,
    # outputs
    active_count: wp.array[int],
    active_articulations: wp.array[int],
    active_secondary_articulations: wp.array[int],
):
    """Compact topology candidates that selected paired local ownership."""
    candidate = wp.tid()
    articulation = candidate_articulations[candidate]
    world = articulation_world[articulation]
    if local_solve_owner[world] == expected_owner:
        active_index = wp.atomic_add(active_count, 0, 1)
        active_articulations[active_index] = articulation
        active_secondary_articulations[active_index] = candidate_secondary_articulations[candidate]
