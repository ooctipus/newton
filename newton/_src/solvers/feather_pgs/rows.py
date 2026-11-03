# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The row laws' tables: what every row writer reads besides the step's state, built once per model.

:meth:`RowLaws.build` resolves the laws from the options (:class:`.configuration.Options`) and the model:

- the contact law: the friction law (patch friction with ``friction_anchor_beta > 0``, else point friction), the
  regularization weight, the restitution threshold, the shape materials and contact torsion;
- the rigid bodies' speed limits;
- the joint-limit tables (limited sizes, each coordinate's limit coordinate) and the warp joint-limit row builders;
- the mimic and link plans (the bilateral rows' tables);
- the velocity prescale's launch;
- :attr:`RowLaws.set`, the route table's laws axis (:class:`.route_table.Laws`), once the bilateral pre-elimination
  is decided.

:meth:`RowLaws.write` is the step's one row pass: every row producer of every placement, in the launch order that
fixes the slot order, and :meth:`RowLaws.write_rhs` the dense rows' right-hand sides once their diagonals are known.
The device law functions stay in :mod:`.row_laws`, :mod:`.row_build` and :mod:`.prefix_rows`.
"""

from __future__ import annotations

import os
import warnings
from functools import cache

import numpy as np
import warp as wp

from ...sim import Contacts, Model, State
from ...sim.enums import JointType
from . import field_group, host_index, prefix_rows, walk_rows
from .configuration import Options
from .contact_input import ContactInput
from .contact_input import bind as bind_contact_input
from .contact_input import bind_routes as bind_contact_routes
from .contact_row_plan import SHARED_METADATA_PRODUCERS, contact_row_plan
from .dense_rows import UNIT_LIMIT, UNIT_MIMIC, UNIT_SHIFT, UNIT_VELOCITY, RowSupport
from .dense_rows import bind as bind_dense_rows
from .dense_rows import bind_mf as bind_mf_rows
from .device_preludes import block_groups, lane_group
from .field_group import SolverArrays
from .forest_plan import ForestPlan
from .friction_patches import _FrictionPatchState as FrictionPatchState
from .friction_patches import patch_row_law
from .joint_table import bind as bind_joint_table
from .joint_table import bind_free_bodies
from .kernel_template import KernelTemplate, params
from .kernels import (
    allocate_link_slots,
    allocate_rigid_velocity_limit_slots,
    allocate_world_contact_slots,
    compute_world_contact_bias,
    finalize_constraint_counts_with_status,
    gather_mf_warmstart,
    populate_rigid_velocity_limit_rows,
    populate_world_J_for_compact_size,
    populate_world_J_for_size,
    prepare_fused_diagonal_joint_limits,
    prepare_world_contact_rows,
    snapshot_dense_phase_bound,
    write_link_rows,
)
from .route_table import Laws
from .row_build import contact_row_prelude, populate_world_J_masked
from .row_laws import LINK_ROWS, ROW_LAW
from .solver_fields import (
    LINK_PLAN,
    LINK_SLOTS,
    MIMIC_PLAN,
    MIMIC_SLOTS,
    VELOCITY_LIMIT_SLOTS,
    Links,
    Preelim,
)

__all__ = [
    "RowLaws",
    "joint_limit_warp_kernel",
    "mimic_warp_kernel",
    "velocity_limit_warp_kernel",
    "velocity_prescale_launch",
]

_CALLER = 4
"""Warnings raised in :meth:`RowLaws.build` name the solver constructor's caller: build runs in the constructor body,
which runs inside the keyword-only shim (deprecate_nonkeyword_arguments)."""

JOINT_LIMIT_WARPS_PER_BLOCK = 4

_CONTACT_BUILD_THREAD_CAP = 65536
# The masked row builder's thread budget, in multiples of the contact-build cap.
_ROWS_THREADS_X = 4
_CONTACT_JACOBIAN_WORKER_CAP = 4096

BIAS_LANES = 256
"""The most threads a world's rows get in the bias launch (:func:`.kernels.compute_world_contact_bias`): a block."""


def velocity_prescale_launch(articulation_start: np.ndarray, *, joint_count: int, cuda: bool) -> tuple[int, int]:
    """Return the lanes per articulation and the thread count of the joint velocity prescale.

    A lane group as wide as the mean articulation's joint count, rounded up to a
    power of two of at most one warp, covers a typical articulation in one
    strided pass while many articulations share the device; larger ones take a
    few passes. CUDA launches cover complete warps. CPU threads cannot exchange
    values, so a CPU launch runs one lane per articulation.

    Raises:
        ValueError: If the last articulation does not end at ``joint_count``. The
            prescale writes no DOF of a joint past it.
    """
    count = len(articulation_start) - 1
    if count <= 0:
        return 1, 0
    if int(articulation_start[-1]) != joint_count:
        raise ValueError(
            "SolverFeatherPGS requires Model.articulation_start to end with the joint count "
            f"{joint_count}, got {int(articulation_start[-1])}; ModelBuilder.finalize() writes this sentinel."
        )
    if not cuda:
        return 1, count
    mean_joints = (int(articulation_start[-1]) - int(articulation_start[0])) / count
    lanes = 1
    while lanes < 32 and lanes < mean_joints:
        lanes *= 2
    return lanes, (count * lanes + 31) // 32 * 32


def _prefix_warp_params(law) -> tuple:
    """The warp prefix producers' operands: the family's law (:data:`.prefix_rows.LAW`, with the velocity or the mimic
    family's inputs), the rows' placement and Jacobian rows (:func:`.dense_rows.jacobian_row`: the packed rows, by
    world, or the size group's)."""
    return params(
        ("articulation_count", int),
        ("articulation_dof_start", wp.array[int]),
        ("art_to_world", wp.array[int]),
        ("group_to_art", wp.array[int]),
        ("resolved_worlds", wp.array[int]),
        ("law", law),
        ("support", RowSupport),
        ("J_group", wp.array3d[float]),
    )


# The joint row families of the prefix warp: (kernel name, native name, law, unit kind).
_PREFIX_FAMILIES = {
    "limits": ("build_joint_limit_rows_warp", "joint_limit_warp_native", prefix_rows.PrefixLaw, UNIT_LIMIT),
    "velocity": (
        "build_joint_velocity_limit_rows_warp",
        "velocity_limit_warp_native",
        prefix_rows.PrefixVelocityLaw,
        UNIT_VELOCITY,
    ),
    "mimics": ("build_mimic_rows_warp", "mimic_warp_native", prefix_rows.PrefixMimicLaw, UNIT_MIMIC),
}


@cache
def _prefix_warp_kernel(size: int, warps_per_block: int, family: str, mimics: int = 0) -> wp.Kernel:
    """A joint row family of the general route and the one writer, one warp per articulation of ``size`` coordinates
    (:func:`.prefix_rows.prefix_rows`) on CUDA, a block's articulations in turn on the CPU: its joint limits, its joint
    velocity limits, or its mimics (at most ``mimics`` an articulation). The warp takes each candidate's law, reserves
    the active rows on the world's counter, writes their records and publishes each row's side: its unit source for
    the one writer, else its Jacobian row (a mimic's two coordinates; its slot in ``mimic_slot`` too). The joint
    limits skip the resolved worlds; every world takes its velocity limits and mimics."""
    name, native, law, kind = _PREFIX_FAMILIES[family]
    kind = int(kind)
    if family == "mimics":
        source, second = f"({kind} << {int(UNIT_SHIFT)}) | mimic", "\n                J_side[d1] = w1;"
        extra = "            law.mimic_slot.data[mimic] = slot;\n"
    else:
        source, second, extra = f"({kind} << {int(UNIT_SHIFT)}) | (2 * (first_dof + d0) + (w0 < 0.0f ? 1 : 0))", "", ""
    # A row's side: its articulation's coordinates in the packed rows (cleared, then the coefficients), or the size
    # group's row (cleared each step).
    sink = f"""{extra}            if (support.walks != 0) {{
                // The one writer walks the row (walk_rows): its unit source, the coordinate's side, in the units' own
                // rows (the compact boundary keeps its limit prefix's alone).
                if (slot < support.units.shape[1])
                    support.units.data[(size_t)prefix_line * support.units.shape[1] + slot] =
                        {source};
            }} else {{
                float* J_side = &J_group.data[((size_t)row_index * J_group.shape[1] + slot) * J_group.shape[2] + first];
                if (support.packed != 0) {{
                    for (int e = 0; e < {size}; ++e) J_side[e] = 0.0f;
                }}
                J_side[d0] = w0;{second}
            }}"""
    if family == "mimics":
        candidates = prefix_rows.mimics(mimics, "law.")
    elif family == "velocity":
        candidates = prefix_rows.velocity_limits(tuple(range(size)), "first_dof + d", "law.")
    else:
        candidates = prefix_rows.limits(tuple(range(size)), "first_dof + d", "law.")
    rows = prefix_rows.prefix_rows(
        ((candidates,),),
        law="law.",
        sink=sink,
        ordered=False,
        rhs=False,
        locals_="int mimic = -1;" if family == "mimics" else "",
        side="art",
    )
    head = (
        "    if (world >= 0 && world < resolved_worlds.shape[0] && resolved_worlds.data[world] != 0) return;\n"
        if family == "limits"
        else ""
    )
    first_dof = "    const int first_dof = articulation_dof_start.data[art];\n"
    if family == "mimics":
        first_dof += (
            "    const int mimic_first = law.mimic_start.data[art], mimic_end = law.mimic_start.data[art + 1];\n"
        )
    body = f"""    const int art = group_to_art.data[group];
    const int world = art_to_world.data[art];
{head}{first_dof}    // The rows' first index and the side's first entry (dense_rows.jacobian_row; one side).
    const int row_index = support.packed != 0 ? support.first.data[world] : group;
    const int first = support.packed != 0 && support.dense != 0 ? support.offset.data[art] : 0;
{rows}"""
    suffix = f"_m{mimics}" if family == "mimics" else ""
    return KernelTemplate(
        f"{name}_{size}_{warps_per_block}{suffix}",
        _prefix_warp_params(law),
        block_groups(warps_per_block, "articulation_count", body),
        preludes=(lane_group(32), *ROW_LAW.preludes()),
        dual=True,
        native_name=native,
    ).kernel()


def joint_limit_warp_kernel(size: int, warps_per_block: int) -> wp.Kernel:
    """The general route's and the one writer's joint-limit rows, one warp per articulation (:func:`_prefix_warp_kernel`)."""
    return _prefix_warp_kernel(size, warps_per_block, "limits")


def velocity_limit_warp_kernel(size: int, warps_per_block: int) -> wp.Kernel:
    """The general route's and the one writer's joint velocity-limit rows, one warp per articulation
    (:func:`_prefix_warp_kernel`)."""
    return _prefix_warp_kernel(size, warps_per_block, "velocity")


def mimic_warp_kernel(size: int, warps_per_block: int, mimics: int) -> wp.Kernel:
    """The general route's and the one writer's mimic rows, at most ``mimics`` an articulation, one warp per
    articulation (:func:`_prefix_warp_kernel`)."""
    return _prefix_warp_kernel(size, warps_per_block, "mimics", mimics)


class RowLaws:
    """The row laws' tables of one model and its options (:meth:`build`).

    The tables keep the names they had on the solver, without the leading underscore, except the tables uploaded by
    field groups (:data:`.solver_fields.MIMIC_PLAN`, :data:`.solver_fields.LINK_PLAN`), which keep their fields'
    names.
    """

    set: Laws
    """The route table's laws axis (:class:`.route_table.Laws`), once the bilateral pre-elimination is decided
    (:meth:`decide`)."""

    @classmethod
    def build(
        cls,
        model: Model,
        forest: ForestPlan,
        options: Options,
    ) -> RowLaws:
        """Resolve the row laws and build their tables.

        Args:
            model: The model.
            forest: The forest plan.
            options: The solver's options.

        Raises:
            ValueError: For a negative or non-finite ``friction_anchor_beta`` or contact torsion with an option it does
                not compose with.
        """
        laws = cls()
        laws.model = model
        laws.forest = forest
        laws.options = options
        laws._arrays = SolverArrays(laws, model.device)

        # The friction law. How friction_anchor_beta was resolved, for get_path_selection().
        friction_anchor_beta = options.friction_anchor_beta
        laws.friction_anchor_beta_source = "argument"
        if friction_anchor_beta is None:
            # An explicit point algorithm remains a valid way to select point
            # friction. The ordinary constructor enables persistent patches.
            friction_anchor_beta = 0.2
            laws.friction_anchor_beta_source = "default"
        laws.friction_anchor_beta = float(friction_anchor_beta)
        if not np.isfinite(laws.friction_anchor_beta) or laws.friction_anchor_beta < 0.0:
            raise ValueError("friction_anchor_beta must be finite and non-negative")
        laws.friction_anchors_enabled = laws.friction_anchor_beta > 0.0
        # Kernels consume the float32 weight w = 1/(1+g). Values too small to
        # change w are the exact hard update and stay on the zero-cost path.
        laws.contact_w = float(np.float32(1.0 / (1.0 + options.pgs_contact_regularization)))
        laws.regularization_enabled = laws.contact_w < 1.0
        laws.effective_restitution_velocity_threshold = (
            options.restitution_velocity_threshold if options.enable_restitution else np.finfo(np.float32).max
        )

        # The rigid bodies' speed limits.
        laws.rigid_body_max_linear_velocity = getattr(model, "rigid_body_max_linear_velocity", None)
        laws.rigid_body_max_angular_velocity = getattr(model, "rigid_body_max_angular_velocity", None)
        laws.rigid_body_max_depenetration_velocity = getattr(model, "rigid_body_max_depenetration_velocity", None)
        if laws.rigid_body_max_depenetration_velocity is None:
            laws.rigid_body_max_depenetration_velocity = wp.full(
                model.body_count,
                float("inf"),
                dtype=wp.float32,
                device=model.device,
            )
        laws.has_rigid_body_velocity_limits = (
            laws.rigid_body_max_linear_velocity is not None and laws.rigid_body_max_angular_velocity is not None
        )

        laws._joint_limit_tables(model)
        # Mimic and link plans must exist before capacity selection so the row census
        # counts their rows (see Placement.internal_rows_per_world).
        laws._mimic_plan(model)
        laws._link_plan()
        laws._contact_torsion(model, options)

        # Shape materials.
        if model.shape_material_mu is not None:
            laws.shape_material_mu = model.shape_material_mu
        else:
            laws.shape_material_mu = wp.zeros((1,), dtype=wp.float32, device=model.device)
        if model.shape_material_restitution is not None:
            laws.shape_material_restitution = model.shape_material_restitution
        else:
            laws.shape_material_restitution = wp.zeros((1,), dtype=wp.float32, device=model.device)

        # The joint velocity prescale's lanes per articulation and threads (it writes every DOF of qd_work).
        if model.joint_count:
            laws.velocity_prescale_lanes, laws.velocity_prescale_threads = velocity_prescale_launch(
                model.articulation_start.numpy(), joint_count=model.joint_count, cuda=model.device.is_cuda
            )
        else:
            laws.velocity_prescale_lanes, laws.velocity_prescale_threads = 1, 0
        laws.patch_frame_reuse, laws.patch_frame_reuse_reason = laws._patch_frame_reuse()
        laws.friction_patches = None
        return laws

    def _contact_torsion(self, model, options) -> None:
        """Contact torsion: every friction contact's fourth row, its spin about the normal, on the elliptic cone with
        its friction pair. Its coefficient is the shapes' torsional friction (``Model.shape_material_mu_torsional``,
        a length), combined per contact as the friction coefficient is (:func:`.contact_laws.contact_friction`)."""
        self.contact_torsion_enabled = bool(options.contact_torsion)
        self.torsion_rows = int(self.contact_torsion_enabled)
        if self.contact_torsion_enabled:
            if not options.enable_contact_friction:
                raise ValueError(
                    "contact_torsion resists spin with the friction rows: it needs enable_contact_friction"
                )
        mu_torsional = model.shape_material_mu_torsional if model.shape_count else None
        if mu_torsional is None:
            mu_torsional = wp.zeros((1,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad)
        self.shape_material_mu_torsional = mu_torsional

    def decide(self, *, preelimination: bool) -> None:
        """Read the route table's laws axis (:attr:`set`) once the bilateral pre-elimination is decided."""
        self.set = Laws.build(self, self.options, self.model, preelimination=preelimination)

    # --- the joint-limit tables ----------------------------------------------------------------------------------
    def _joint_limit_tables(self, model):
        """The joint-limit tables: the limited sizes, each coordinate's limit coordinate, the warp limit and velocity-limit
        kernels."""
        if not model.articulation_count or not model.joint_count:
            self.joint_limit_sizes = frozenset()
            self.joint_limit_q_index = None
            self.joint_limit_q_index_host = np.zeros(model.joint_dof_count, dtype=np.int32)
            self.joint_limit_warp_kernels = {}
            self.velocity_limit_warp_kernels = {}
            return

        response_dof_count = self.forest.response_dof_count
        articulation_start = model.articulation_start.numpy()
        joint_type = model.joint_type.numpy()
        limit_joint = np.isin(
            joint_type,
            (int(JointType.PRISMATIC), int(JointType.REVOLUTE), int(JointType.D6)),
        )
        # Limit joints before each articulation's range ends, minus those before it starts.
        limit_joints_before = host_index.offsets(limit_joint)
        art_start = articulation_start[: model.articulation_count]
        art_end = articulation_start[1 : model.articulation_count + 1]
        limited_sizes = response_dof_count[
            (response_dof_count > 0) & (limit_joints_before[art_end] > limit_joints_before[art_start])
        ]
        # Insert each size where an articulation first has it, as the per-articulation set did.
        self.joint_limit_sizes = frozenset(int(size) for size in host_index.ordered_unique(limited_sizes))
        joint_q_start = model.joint_q_start.numpy().astype(np.int32, copy=False)
        joint_qd_start = model.joint_qd_start.numpy().astype(np.int32, copy=False)
        joint_dof_dim = model.joint_dof_dim.numpy().astype(np.int32, copy=False)
        limit_q_index = np.full(model.joint_dof_count, -1, dtype=np.int32)
        limit_joints = np.flatnonzero(limit_joint)
        limit_joint_of, axis = host_index.expand(joint_dof_dim[limit_joints, 0] + joint_dof_dim[limit_joints, 1])
        limit_q_index[joint_qd_start[limit_joints][limit_joint_of] + axis] = (
            joint_q_start[limit_joints][limit_joint_of] + axis
        )
        self.joint_limit_q_index = wp.array(limit_q_index, dtype=wp.int32, device=model.device)
        self.joint_limit_q_index_host = limit_q_index
        self.joint_limit_warp_kernels = {
            size: joint_limit_warp_kernel(size, warps_per_block=JOINT_LIMIT_WARPS_PER_BLOCK)
            for size in self.joint_limit_sizes
        }
        # The velocity limits sit on the same coordinates (limit_q_index >= 0), so on the same sizes.
        self.velocity_limit_warp_kernels = (
            {
                size: velocity_limit_warp_kernel(size, warps_per_block=JOINT_LIMIT_WARPS_PER_BLOCK)
                for size in self.joint_limit_sizes
            }
            if self.options.enable_joint_velocity_limits
            else {}
        )

    # --- the bilateral rows' plans ---------------------------------------------------------------------------------
    def _mimic_plan(self, model) -> None:
        """Precompute the static per-mimic lookup tables for the MIMIC row family.

        Resolves ``Model.constraint_mimic_*`` joint indices into articulation / DOF /
        coordinate indices and validates the supported subset: both joints 1-DoF
        (REVOLUTE or PRISMATIC), both in the same articulation, distinct DOFs — matching
        SolverMuJoCo's own restriction. Invalid entries are warned about and masked off;
        ``coef0``/``coef1``/``enabled`` stay live reads from the model arrays so runtime
        mutation needs no re-initialization. All buffers are allocated here, once, so the
        per-step launches are CUDA-graph-safe.
        """
        self.mimic_count = 0
        self.mimic_valid_np = None
        self.mimic_world_np = None
        self.mimic_art_start_np = None
        self._mimic_art_start = None
        self._mimic_art_list = None
        self.mimic_sizes = frozenset()
        self.mimic_slot = None
        self.mimic_warp_kernels = {}
        n = int(getattr(model, "constraint_mimic_count", 0) or 0)
        if n == 0 or not model.articulation_count or self.forest.art_to_world is None:
            return

        j0 = model.constraint_mimic_joint0.numpy().astype(np.int32, copy=False)
        j1 = model.constraint_mimic_joint1.numpy().astype(np.int32, copy=False)
        articulation_start = model.articulation_start.numpy().astype(np.int64, copy=False)
        joint_type = model.joint_type.numpy().astype(np.int32, copy=False)
        joint_qd_start = model.joint_qd_start.numpy().astype(np.int32, copy=False)
        joint_q_start = model.joint_q_start.numpy().astype(np.int32, copy=False)
        labels = list(getattr(model, "constraint_mimic_label", []) or [])

        one_dof = (int(JointType.REVOLUTE), int(JointType.PRISMATIC))
        articulation_bounds = articulation_start[: model.articulation_count + 1]
        a0, a1 = host_index.lookup(articulation_bounds, j0), host_index.lookup(articulation_bounds, j1)
        spans = (a0 < 0) | (a1 < 0) | (a0 != a1)
        # Joint checks apply to the constraints within one articulation, in order.
        within = np.flatnonzero(~spans)
        one_dof_pair = np.zeros(n, dtype=bool)
        one_dof_pair[within] = np.isin(joint_type[j0[within]], one_dof) & np.isin(joint_type[j1[within]], one_dof)
        self_coupled = np.zeros(n, dtype=bool)
        self_coupled[within] = joint_qd_start[j0[within]] == joint_qd_start[j1[within]]
        valid = (~spans & one_dof_pair & ~self_coupled).astype(np.int32)
        art = np.where(valid != 0, a0, -1).astype(np.int32)
        for k in np.flatnonzero(valid == 0):
            name = labels[k] if k < len(labels) else f"mimic_{k}"
            if spans[k]:
                warnings.warn(
                    f"SolverFeatherPGS: mimic constraint '{name}' spans articulations "
                    f"({int(a0[k])} vs {int(a1[k])}); cross-articulation mimic is unsupported and the "
                    "constraint is ignored.",
                    stacklevel=2,
                )
            elif not one_dof_pair[k]:
                warnings.warn(
                    f"SolverFeatherPGS: mimic constraint '{name}' references a non-1-DoF "
                    "joint (only REVOLUTE/PRISMATIC are supported); the constraint is ignored.",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"SolverFeatherPGS: mimic constraint '{name}' couples a DOF to itself; ignored.", stacklevel=2
                )

        valid_indices = np.flatnonzero(valid).astype(np.int32, copy=False)
        art_start, order = host_index.group_by(art[valid_indices], model.articulation_count)
        art_start = art_start.astype(np.int32)
        order = valid_indices[order]
        # A mimic row is reserved and solved in its articulation's solve world: a global articulation (and its
        # mimic, world -1 in the model) solves in world 0.
        mimic_world = np.zeros(n, dtype=np.int32)
        mimic_world[valid_indices] = self.forest.articulation_world[art[valid_indices]]

        self.mimic_valid_np = valid
        self.mimic_world_np = mimic_world
        self.mimic_art_start_np = art_start
        self.mimic_sizes = frozenset(
            int(self.forest.response_dof_count[a])
            for a in np.unique(art[valid_indices])
            if self.forest.response_dof_count[a] > 0
        )
        self._arrays.upload(
            MIMIC_PLAN,
            _mimic_art_start=art_start,
            _mimic_art_list=order,
            _mimic_valid=valid,
            _mimic_world=mimic_world,
            _mimic_dof0=joint_qd_start[j0],
            _mimic_dof1=joint_qd_start[j1],
            _mimic_q0=joint_q_start[j0],
            _mimic_q1=joint_q_start[j1],
        )
        self._arrays.allocate(MIMIC_SLOTS, mimic=n)
        self.mimic_count = n
        # A warp per articulation produces a size's mimic rows, its candidates the most mimics an articulation of that
        # size has.
        per_art = np.diff(art_start)
        sizes = self.forest.response_dof_count[: model.articulation_count]
        self.mimic_warp_kernels = {
            size: mimic_warp_kernel(size, JOINT_LIMIT_WARPS_PER_BLOCK, int(per_art[sizes == size].max()))
            for size in self.mimic_sizes
        }

    def _link_plan(self) -> None:
        """The links' rows (:mod:`.body_cut`): six bilateral rows per enabled link, whatever joint made them. Their
        tables are uploaded once (CUDA-graph-safe); a link's enable bit stays live."""
        cut = self.forest.cut
        self.link_count = n = int(cut.primary.size)
        self.link_slot = None
        self.link_world_np = None
        self.link_rows_per_world = 0
        if n == 0:
            return
        self.link_world_np = self.forest.articulation_world[cut.tree_shadow].astype(np.int32)
        self.link_rows_per_world = int(LINK_ROWS * np.bincount(self.link_world_np).max())
        self._arrays.upload(
            LINK_PLAN,
            _link_primary=cut.primary,
            _link_shadow=cut.shadow,
            _link_world=self.link_world_np,
            _link_enabled=cut.enabled,
        )
        self._arrays.allocate(LINK_SLOTS, link=n)

    def set_links_enabled(self) -> None:
        """Publish the links' enable bits (:meth:`.body_cut.BodyCut.set_enabled`); the next step reserves their rows."""
        self._link_enabled.assign(self.forest.cut.enabled)

    # --- the slots and the friction patches, allocated with the rows ----------------------------------------------
    def allocate(self, model, pool, phi) -> None:
        """Allocate the velocity-limit slots and the friction patch state.

        Args:
            model: The model.
            pool: The contact pool (its rows hold the patch state's per-contact fields).
            phi: The contact rows' patch phi view (a contact pool field).
        """
        options = self.options
        arrays = self._arrays
        # Persistent state is owned by the body-pair patch builder, independently
        # of collision matching and the contact warm-start tables.
        self.friction_patches = FrictionPatchState(
            model,
            pool.ready,
            self.friction_anchors_enabled,
            phi,
            reuse_frames=self.patch_frame_reuse,
            pool=pool,
        )
        # Joint velocity-limit buffers (per-DOF tracking). Independent from the
        # joint-position-limit buffers so the two flags can be used separately.
        # Both rows of a finite limit are allocated every step, so the velocity box
        # is present before the predicted velocity escapes.
        if options.enable_joint_velocity_limits and model.joint_dof_count > 0:
            arrays.allocate(VELOCITY_LIMIT_SLOTS, limit_row=2 * model.joint_dof_count)
        else:
            arrays.release(VELOCITY_LIMIT_SLOTS)

    def _patch_frame_reuse(self) -> tuple[bool, str]:
        """Return whether captured patch builds keep the sorted pairs of a reused contact frame, with the reason.

        A captured build keeps them when the previous build was captured in the same
        graph over the same contact stream, its ``contact_generation`` array, and the
        host issued no write of that stream in between (see :class:`.FrictionPatchState`).
        On by default;
        ``FEATHER_PGS_PATCH_FRAME_REUSE=0`` sorts every build.
        """
        if not self.friction_anchors_enabled:
            return False, "Point friction sorts no patch pairs."
        if os.environ.get("FEATHER_PGS_PATCH_FRAME_REUSE", "1") == "0":
            return False, "FEATHER_PGS_PATCH_FRAME_REUSE=0 sorts the patch pairs of every build."
        return True, (
            "A captured build keeps the sorted patch pairs of the previous build captured in the same graph over the "
            "same contact stream (its contact_generation array, which wrappers of one Contacts share) when the host "
            "noted no write of that stream in between (such as a collision pass, Contacts.clear() or a "
            "coupled-solver filter); other builds sort."
        )

    def patch_frame_status(self):
        """Return the device flag of patch frames left empty by a stale reuse, or ``None`` when none can be.

        Only a build that keeps its pairs through frame reuse sets the flag, so
        without frame reuse the host reads it never. A model without
        articulations allocates no patch state at all.
        """
        patches = self.friction_patches
        if patches is None or not patches.reuse_frames:
            return None
        return patches.frame_status

    # --- the row pass ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _launch_prefix_warp(solver, kernel, size: int, law, support) -> None:
        """Launch a prefix-rows warp producer (:func:`_prefix_warp_kernel`) over size group ``size``: a warp per
        articulation on CUDA, one thread per block's articulations on the CPU."""
        n_arts = solver.forest.n_arts_by_size[size]
        wp.launch_tiled(
            kernel,
            dim=[(n_arts + JOINT_LIMIT_WARPS_PER_BLOCK - 1) // JOINT_LIMIT_WARPS_PER_BLOCK],
            inputs=[
                n_arts,
                solver.forest.articulation_dof_start,
                solver.forest.art_to_world,
                solver.forest.group_to_art[size],
                solver.sweep.resolved_worlds,
                law,
                support,
            ],
            outputs=[solver.placement.jacobian_rows(size)],
            block_dim=32 * JOINT_LIMIT_WARPS_PER_BLOCK,
            device=solver.model.device,
        )

    def write(self, solver, state_in: State, state_aug: State, contacts: Contacts, dt: float) -> None:
        """The step's row pass: every row of every placement, with its records, its slot and its phase bounds.

        The launch order is the slot order: the allocators reserve slots with atomics, in the order mimic, link,
        joint limit, velocity limit, contacts, then the free bodies' rows, and the phase-bound snapshots are glued
        to that sequence. The pass then finalizes the counts with status, links the friction patches after every
        producer and claims the sweep's rows.

        Args:
            solver: The solver whose step this is: its model, options, plan and primitives.
            state_in: The step's input state.
            state_aug: The step's dynamics (the tree factor).
            contacts: The step's contacts, or ``None``.
            dt: The step size.
        """
        model = solver.model
        max_constraints = solver.placement.dense_max_constraints
        # Every row producer writes the dense row records through one struct, and the joint row producers read
        # the joint table through another, both bound for this step.
        dense_rows = bind_dense_rows(solver.placement)
        joints = bind_joint_table(solver)
        mf_active = solver.forest.has_free_rigid_bodies
        if solver.forest.has_prescribed_response and mf_active:
            solver.placement.body_rows.target_velocity.zero_()

        solver.placement.begin(solver._plan.jacobian_clear_sizes)

        has_free_rigid_flag = 1 if mf_active else 0
        # Dummy arrays when MF is not active (kernel still needs valid pointers).
        # Preallocated in __init__: a per-step wp.zeros here would add an
        # allocation + memset every step and, under CUDA graph capture, leave
        # the graph holding a pointer to a freed temporary.
        is_free_rigid = (
            solver.forest.is_free_rigid
            if solver.forest.is_free_rigid is not None
            else solver.placement.dummy_is_free_rigid
        )
        mf_slot_counter = solver.placement.mf_slot_counter if mf_active else solver.placement.dummy_mf_slot_counter
        dense_dropped_rows = solver.placement.row_dropped_dense
        mf_dropped_rows = solver.placement.row_dropped_mf
        mf_first_rejected_slot = (
            solver.placement.mf_first_rejected_slot if mf_active else solver.placement.dummy_mf_slot_counter
        )
        support = solver.placement.row_support()
        if solver.placement.row_packets is not None:
            solver.placement.row_packets.prepare_prefix(state_in, dt)
        if solver.placement.keyed_rows is not None:
            solver.placement.keyed_rows.begin_rows(state_in, state_aug, dt)

        # ── Dense row-family layout ──────────────────────────────────────
        # Slots are handed out in the order [mimic][joint-limit]
        # [joint-velocity-limit][contact/friction], so each family owns one
        # contiguous dense row range:
        #   mimic and position limits: [0, bounds[0])
        #   velocity limits:           [bounds[0], bounds[1])
        #   contacts and friction:     [bounds[1], m_dense)
        # The two snapshot_dense_phase_bound launches below record the family
        # boundaries and must stay glued to this allocation sequence
        # (slot_counter is zeroed at the top of this method). Mimic rows come
        # first, ahead of the (activation-gated, flickering) limit family, so
        # warm-started mimic lambdas keep a stable slot index across steps.

        # Mimic (joint coupling) equality rows — static count per model. One warp per articulation reserves, writes
        # and publishes them, as the joint limits (prefix_rows).
        if self.mimic_warp_kernels and solver.placement.row_packets is None:
            mimic_law = field_group.bind(
                prefix_rows.PrefixMimicLaw,
                solver=solver,
                placement=solver.placement,
                factor=solver.factor,
                laws=self,
                options=solver.options,
                model=model,
                state_in=state_in,
                dt=dt,
                mimics=self,
                mimic_model=model,
            )
            for size, warp_kernel in self.mimic_warp_kernels.items():
                self._launch_prefix_warp(solver, warp_kernel, size, mimic_law, support)

        # The links' rows: static per model, their sides the trees of the link's two pieces.
        if self.link_count > 0:
            links = field_group.bind(Links, laws=self)
            wp.launch(
                allocate_link_slots,
                dim=self.link_count,
                inputs=[links, max_constraints],
                outputs=[solver.placement.slot_counter, solver.placement.dense_first_rejected_slot],
                device=model.device,
            )
            wp.launch(
                write_link_rows,
                dim=self.link_count,
                inputs=[
                    joints,
                    links,
                    state_in.body_q,
                    model.body_com,
                    solver.forest.body_to_joint,
                    solver.forest.body_to_articulation,
                    model.joint_ancestor,
                    state_aug.joint_S_s,
                    solver.factor.articulation_origin,
                    solver.options.pgs_beta,
                    solver.options.pgs_cfm,
                    support,
                ],
                outputs=[solver.placement.packed_Z, dense_rows],
                device=model.device,
            )

        if solver.placement.fused_diagonal_joint_limits:
            wp.launch(
                prepare_fused_diagonal_joint_limits,
                dim=solver.forest.world_count * solver.forest.max_world_dofs,
                inputs=[
                    joints,
                    solver.forest.world_dof_indices,
                    solver.forest.max_world_dofs,
                    solver.placement.fused_diagonal_limit_dof_mask,
                    self.joint_limit_q_index,
                    state_in.joint_q,
                    solver.options.joint_limit_activation_gap,
                    solver.options.pgs_beta,
                    dt,
                    solver.v_hat,
                ],
                outputs=[
                    solver.placement.fused_diagonal_limit_active,
                    solver.placement.fused_diagonal_limit_lower_rhs,
                    solver.placement.fused_diagonal_limit_upper_rhs,
                ],
                device=model.device,
            )

        # Allocate and populate joint-limit rows per response-size group.
        if (
            solver.options.enable_joint_limits
            and self.joint_limit_sizes
            and solver.placement.row_packets is None
            and solver.placement.keyed_rows is None
        ):
            # The native limit rows (prefix_rows) read the law and write the records through one struct.
            law = field_group.bind(
                prefix_rows.PrefixLaw,
                solver=solver,
                placement=solver.placement,
                factor=solver.factor,
                laws=self,
                options=solver.options,
                model=model,
                state_in=state_in,
                dt=dt,
            )
            for size in solver.forest.size_groups:
                if size not in self.joint_limit_sizes or size in solver.placement.fused_diagonal_joint_limit_sizes:
                    continue
                self._launch_prefix_warp(solver, self.joint_limit_warp_kernels[size], size, law, support)

        # Unconditional: the first boundary must include position-limit
        # rows and remain valid when that family is disabled (when it is just
        # the current watermark at the start of the velocity-limit segment).
        if solver.placement.row_packets is None and solver.placement.keyed_rows is None:
            wp.launch(
                snapshot_dense_phase_bound,
                dim=solver.forest.world_count,
                inputs=[solver.placement.slot_counter, 0],
                outputs=[solver.placement.dense_phase_bounds],
                device=model.device,
            )

        # The joint velocity-limit rows (per-DOF clamp on |qdot_i| against model.joint_velocity_limit). Dense contact
        # rows are allocated after this block so contact/friction rows form the trailing per-world slot segment; the
        # sweeps visit velocity-limit rows in a dedicated final pass regardless of slot order. One warp per
        # articulation reserves, writes and publishes them, as the joint limits (prefix_rows).
        if self.velocity_limit_warp_kernels and solver.placement.keyed_rows is None:
            velocity_law = field_group.bind(
                prefix_rows.PrefixVelocityLaw,
                solver=solver,
                placement=solver.placement,
                factor=solver.factor,
                laws=self,
                options=solver.options,
                model=model,
                state_in=state_in,
                dt=dt,
                vhat=solver.v_hat,
            )
            for size, warp_kernel in self.velocity_limit_warp_kernels.items():
                self._launch_prefix_warp(solver, warp_kernel, size, velocity_law, support)

        # Unconditional velocity-limit boundary snapshot; dense contact rows start here.
        if solver.placement.row_packets is None and solver.placement.keyed_rows is None:
            wp.launch(
                snapshot_dense_phase_bound,
                dim=solver.forest.world_count,
                inputs=[solver.placement.slot_counter, 1],
                outputs=[solver.placement.dense_phase_bounds],
                device=model.device,
            )

        # The sparse diagonal on the walk (FL3.5): its rows read the slips' release (the predictor's border), its
        # contact producers among them.
        slots = solver.placement.walks and solver.placement.sparse_diagonal_contact_solve
        slots_written = False

        if (
            contacts is not None
            and getattr(contacts, "rigid_contact_count", None) is not None
            and contacts._rigid_contact_rows > 0
        ):
            enable_friction_flag = 1 if solver.options.enable_contact_friction else 0
            pool = solver.placement.contact_pool
            count = contacts.rigid_contact_count
            # Sized for the bound in a capture whose launches bind: every replay runs them at the live
            # count and strides with the captured threads (ContactPool.launch); otherwise for the ready rows.
            rows = pool.launch_rows()
            contact_build_threads = min(rows, _CONTACT_BUILD_THREAD_CAP)

            self._build_friction_patches(solver, state_in, contacts)

            # The allocator and the producers read the contacts, the bodies and the row law, and write the
            # contacts' allocation records and their row records, through one struct
            # (contact_input.ContactInput), bound for this step.
            contact_input = bind_contact_input(solver, state_in, state_aug, contacts, dt, 1.0)
            # The friction row law every producer writes: point friction is the patch law with beta 0.
            law = patch_row_law(solver)

            # The class's case names the producer that writes every contact row (contact_row_plan):
            # get_path_selection() records the same plan (routes.contact_rows). The one writer's contact rows are
            # its own to fill: their unit sources are written where their slots are reserved.
            plan = contact_row_plan(solver)
            units = solver.placement.unit_source if plan.producer == "walk" else solver.placement.no_units
            # Under persistent patch friction, contacts of resolved worlds that the complete path would
            # give rows are marked so the patch finish releases their anchors (kernels.py).
            pool.launch(
                allocate_world_contact_slots,
                contact_build_threads,
                inputs=[
                    contact_input,
                    contact_build_threads,
                    solver.forest.body_to_articulation,
                    solver.forest.art_to_world,
                    model.body_flags,
                    solver.forest.body_has_response_dofs,
                    is_free_rigid,
                    has_free_rigid_flag,
                    solver.options.contact_gap_gate,
                    solver.options.same_articulation_contact_gap_gate,
                    solver.options.articulation_pair_contact_gap_gate,
                    max_constraints,
                    solver.placement.mf_max_constraints,
                    enable_friction_flag,
                    solver.options.contact_friction_gap_threshold,
                    1,
                    self.friction_patches.view,
                    solver.sweep.resolved_worlds,
                ],
                outputs=[
                    solver.placement.slot_counter,
                    mf_slot_counter,
                    dense_dropped_rows,
                    mf_dropped_rows,
                    solver.placement.dense_first_rejected_slot,
                    mf_first_rejected_slot,
                    units,
                ],
                extent=count,
                clamp=True,
            )
            if contacts.rigid_contacts_pair_sorted:
                solver.placement.place_sorted_contact_rows(
                    contacts, contact_build_threads, mf_slot_counter, mf_first_rejected_slot, units
                )

            # The one writer's contact rows ("walk") take their records in the writer (walk_rows).
            if plan.producer == "local_packets":
                solver.placement.row_packets.produce_contacts(contact_input)
            elif plan.producer == "keyed_rows":
                solver.placement.keyed_rows.produce_contacts(contact_input)
            elif plan.producer == "compact_contact_boundary":
                from .compact_contact import launch_contacts  # noqa: PLC0415

                launch_contacts(solver, contact_input, dt)
            elif plan.producer in SHARED_METADATA_PRODUCERS:
                if plan.metadata_launch:
                    pool.launch(
                        prepare_world_contact_rows,
                        contact_build_threads,
                        inputs=[contact_input, contact_build_threads, law],
                        extent=count,
                        clamp=True,
                    )
                contact_jacobian_workers = min(rows, _CONTACT_JACOBIAN_WORKER_CAP)
                for size in solver.forest.size_groups:
                    method = plan.methods[size]
                    pool.launch(
                        populate_world_J_for_compact_size,
                        (contact_jacobian_workers, 32),
                        inputs=[
                            contact_input,
                            contact_jacobian_workers,
                            size,
                            solver.forest.art_group_idx,
                            law,
                            support,
                        ],
                        outputs=[solver.placement.jacobian_rows(size)],
                        extent=count,
                        clamp=True,
                    )
            elif plan.producer != "walk":
                # The masked builder: a prelude writes the row records and per-contact flags, then one
                # thread per (contact, DOF) writes the Jacobian; one launch pair serves the combined sizes.
                flags = solver.placement.rows_prelude_flags
                rows_threads = min(rows, _ROWS_THREADS_X * _CONTACT_BUILD_THREAD_CAP)
                combined_done = False
                for size in solver.forest.size_groups:
                    method = plan.methods[size]
                    if method in ("masked_combined", "masked"):
                        if method == "masked_combined":
                            if combined_done:
                                continue
                            combined_done = True
                            size_a = plan.combined[0]
                            size_b = plan.combined[1] if len(plan.combined) > 1 else -1
                            lanes = max(size_a, size_b)
                            jacobian_threads = rows_threads * lanes
                        else:
                            size_a, size_b, lanes, jacobian_threads = size, -1, size, rows_threads * 32
                        J_a = solver.placement.jacobian_rows(size_a)
                        J_b = solver.placement.jacobian_rows(size_b) if size_b > 0 else J_a
                        pool.launch(
                            contact_row_prelude,
                            rows_threads * 3,
                            inputs=[contact_input, rows_threads * 3, size_a, size_b, law],
                            outputs=[flags],
                            extent=count,
                            scale=3,
                            clamp=True,
                        )
                        pool.launch(
                            populate_world_J_masked,
                            jacobian_threads,
                            inputs=[
                                contact_input,
                                jacobian_threads,
                                size_a,
                                size_b,
                                solver.forest.art_group_idx,
                                law,
                                flags,
                                support,
                            ],
                            outputs=[J_a, J_b],
                            extent=count,
                            scale=lanes,
                            clamp=True,
                        )
                        continue
                    pool.launch(
                        populate_world_J_for_size,
                        contact_build_threads,
                        inputs=[
                            contact_input,
                            contact_build_threads,
                            size,
                            solver.forest.art_group_idx,
                            solver.forest.body_to_joint,
                            model.joint_ancestor,
                            model.joint_qd_start,
                            law,
                            support,
                        ],
                        outputs=[solver.placement.jacobian_rows(size)],
                        extent=count,
                        clamp=True,
                    )
            if slots:
                # The sparse diagonal's rows, every one or the compact boundary's limit prefix, then the independent
                # chains' marks in the slots they hold.
                self.write_slot_rows(solver, contact_input, law, dt)
                slots_written = True
                if solver.sweep.sparse_contact_marker is not None:
                    pool.launch(
                        solver.sweep.sparse_contact_marker,
                        contact_build_threads,
                        inputs=[solver.placement.dense_first, contact_input, contact_build_threads],
                        outputs=[solver.placement.sparse_diagonal_row_dof],
                        extent=count,
                        clamp=True,
                    )

            # The free bodies' contact rows, by the walk (walk_rows): a class with free bodies always walks.
            if mf_active:
                pool.launch(
                    walk_rows.write_free_body_rows,
                    contact_build_threads,
                    inputs=[
                        bind_mf_rows(solver.placement),
                        contact_input,
                        contact_build_threads,
                        int(solver.forest.has_prescribed_response),
                        law,
                        solver.factor.walk_view(),
                        solver.v_hat,
                    ],
                    extent=count,
                    clamp=True,
                )

            if mf_active:
                wp.launch(
                    finalize_constraint_counts_with_status,
                    dim=solver.forest.world_count,
                    inputs=[
                        solver.placement.mf_slot_counter,
                        solver.placement.mf_max_constraints,
                        1,
                        solver.placement.mf_first_rejected_slot,
                    ],
                    outputs=[solver.placement.mf_constraint_count, solver.placement.constraint_capacity_status],
                    device=model.device,
                )

                # ── MF warm-start seed ────────────────────────────────────
                # contact_slot + mf_row_type are now populated for THIS step.
                # mf_impulses was already memset at the top of this method
                # (covers vel-limit rows, padding tail, and any contact that
                # returns early in the gather); here we gather matched contacts'
                # previous impulses into their current slots by identity. The
                # carry itself is the row store's (ws_prev_dense_*), in its free-body region.
                if solver.options.pgs_warmstart:
                    match_index = getattr(contacts, "rigid_contact_match_index", None)
                    if match_index is None:
                        raise NotImplementedError(
                            "FeatherPGS mf_warmstart=True requires the Contacts buffer to be "
                            "created with contact_matching enabled (rigid_contact_match_index "
                            "is None). Build the CollisionPipeline / Contacts with "
                            'contact_matching="latest" (or "sticky").'
                        )
                    pool.launch(
                        gather_mf_warmstart,
                        rows,
                        inputs=[
                            bind_contact_routes(solver, contacts),
                            bind_mf_rows(solver.placement),
                            match_index,
                            solver.placement.ws_prev_slot_sorted,
                            solver.placement.body_region(solver.placement.ws_prev_dense_impulses),
                            solver.placement.body_region(solver.placement.ws_prev_dense_row_type),
                            solver.placement.body_region(solver.placement.ws_prev_dense_row_parent),
                            solver.placement.mf_constraint_count,
                            solver.placement.ws_prev_contact_normal,
                            solver.options.pgs_warmstart_decay,
                            # Carried support impulses are proportional to dt, so
                            # rescale by the exact step-size ratio (1 at fixed dt).
                            dt / solver.sweep.prev_dt if solver.sweep.prev_dt > 0.0 else 1.0,
                            solver.placement.mf_max_constraints,
                        ],
                        outputs=[solver.placement.body_rows.impulses],
                        extent=count,
                        max_blocks=solver.placement.contact_capacity_blocks,
                    )

        # Rigid-body velocity limits are matrix-free rows, not post-solve
        # clamps, in the MF path. Allocate after contacts so they occupy the
        # last MF slots; the fused PGS kernel also solves row_type=4 in a
        # final phase so articulated and rigid velocity-limit rows are both
        # visited after drive/contact/friction/position-limit rows.
        if mf_active:
            # Contact/friction rows all precede velocity-limit rows in the MF
            # slot space (the classifier completes before the vlim allocation
            # below). Record the partition boundary so the GS kernels can loop
            # only the contact prefix in contact phases and only the vlim tail
            # in velocity-limit phases instead of scanning all rows.
            wp.copy(solver.placement.mf_contact_rows_end, solver.placement.mf_slot_counter)
        if mf_active and solver.placement.rigid_velocity_limit_slot is not None:
            # Dummy when no articulation metadata exists; the kernel only
            # reads it for free-rigid roots, which then cannot occur.
            root_dof_start = (
                solver.forest.articulation_root_dof_start
                if solver.forest.articulation_root_dof_start is not None
                else wp.zeros((1,), dtype=wp.int32, device=model.device)
            )
            wp.launch(
                allocate_rigid_velocity_limit_slots,
                dim=solver.forest.free_rigid_body_count,
                inputs=[
                    bind_free_bodies(solver),
                    model.body_flags,
                    root_dof_start,
                    solver.v_hat,
                    solver.options.velocity_limit_activation_fraction,
                    solver.placement.mf_max_constraints,
                    solver.sweep.resolved_worlds,
                ],
                outputs=[
                    solver.placement.rigid_velocity_limit_slot,
                    solver.placement.mf_slot_counter,
                    solver.placement.mf_first_rejected_slot,
                ],
                device=model.device,
            )
            wp.launch(
                populate_rigid_velocity_limit_rows,
                dim=solver.forest.free_rigid_body_count,
                inputs=[
                    bind_free_bodies(solver),
                    bind_mf_rows(solver.placement),
                    solver.placement.rigid_velocity_limit_slot,
                ],
                device=model.device,
            )
            wp.launch(
                finalize_constraint_counts_with_status,
                dim=solver.forest.world_count,
                inputs=[
                    solver.placement.mf_slot_counter,
                    solver.placement.mf_max_constraints,
                    1,
                    solver.placement.mf_first_rejected_slot,
                ],
                outputs=[solver.placement.mf_constraint_count, solver.placement.constraint_capacity_status],
                device=model.device,
            )

        if slots and not slots_written:
            # Without contact rows the sparse diagonal's rows are its limits.
            self.write_slot_rows(solver, ContactInput(), patch_row_law(solver), dt)
        if solver.placement.row_packets is not None:
            solver.placement.row_packets.finish_rows()
        wp.launch(
            finalize_constraint_counts_with_status,
            dim=solver.forest.world_count,
            inputs=[solver.placement.slot_counter, max_constraints, 0, solver.placement.dense_first_rejected_slot],
            outputs=[solver.placement.constraint_count, solver.placement.constraint_capacity_status],
            device=model.device,
        )
        if solver.placement.keyed_rows is not None:
            solver.placement.keyed_rows.finish_rows()
        # The patch link runs after every row's records are written: the keyed rows' general worlds take theirs last
        # here; the one writer writes its contact rows' records after the row pass (write_walked), and links there.
        if not self._writer_records(solver):
            self._link_friction_patches(solver, contacts)
        solver.placement.latch_status(contacts)
        solver.sweep.claim()
        solver.placement.warn_overflow()

    def write_rhs(self, solver, dt: float) -> None:
        """Each dense row's rhs: its bias, or a bouncing contact's rebound (``compute_world_contact_bias``)."""
        if (
            solver.placement.row_packets is not None
            or solver.placement.keyed_rows is not None
            or solver.placement.walks
        ):
            return  # These rows write their own right-hand sides.
        # A world's lanes: its capacity, up to a block.
        lanes = min(solver.placement.dense_max_constraints, BIAS_LANES)
        wp.launch(
            compute_world_contact_bias,
            dim=(solver.forest.world_count, lanes),
            inputs=[
                bind_dense_rows(solver.placement),
                solver.placement.row_incident,
                solver.placement.constraint_count,
                solver.placement.dense_max_constraints,
                lanes,
                dt,
                solver.options.contact_speculative_scale,
                self.contact_w,
                self.effective_restitution_velocity_threshold,
                int(self.set.weighted),
                int(solver.options.contact_compliance),
            ],
            device=solver.model.device,
        )

    def write_walked(self, solver, state_in, contacts: Contacts | None, dt: float) -> None:
        """The one writer: every dense row's seed walked into its placement's sink, its diagonal, incident velocity and
        rhs, once the predictor's ``v_hat`` and its decided set are known. The general placement's
        (:func:`.walk_rows.walked_rows_kernel`): a world's lanes release its trees' slips, write and close its bilateral
        blocks, then walk every other row into the packed rows, corrected on the blocks it touches. The support
        placement's (case K1, :meth:`write_support`)."""
        if solver.routes.case == "K1":
            self.write_support(solver, state_in, contacts, dt)
        else:
            self.write_general(solver, state_in, contacts, dt)
        if self._writer_records(solver):
            self._link_friction_patches(solver, contacts)

    @staticmethod
    def _writer_records(solver) -> bool:
        """Whether the one writer writes the contact rows' records after the row pass (the general placement's and the
        support placement's: the patch link then runs after it)."""
        return bool(solver.placement.walks) and solver.routes.case != "K2"

    def joint_seeds(self, solver, state_in) -> walk_rows.JointSeeds:
        """What the one writer's seeds read besides the contact input (:class:`.walk_rows.JointSeeds`): the mimics' and
        links' tables where the model has them, else one-entry placeholders."""
        placement, model = solver.placement, solver.model
        mimics = self.mimic_count > 0
        links = self.link_count > 0
        dummy_int, dummy_float = placement.dummy_is_free_rigid, placement.contact_row_eps_dummy.flatten()
        seeds = walk_rows.JointSeeds()
        seeds.body_to_articulation = solver.forest.body_to_articulation
        seeds.mimic_dof0 = self._mimic_dof0 if mimics else dummy_int
        seeds.mimic_dof1 = self._mimic_dof1 if mimics else dummy_int
        seeds.mimic_coef1 = model.constraint_mimic_coef1 if mimics else dummy_float
        seeds.link_primary = self._link_primary if links else dummy_int
        seeds.link_shadow = self._link_shadow if links else dummy_int
        seeds.body_q = state_in.body_q
        seeds.body_com = model.body_com
        return seeds

    def write_general(self, solver, state_in, contacts: Contacts | None, dt: float) -> None:
        """The general placement's one writer (:func:`.walk_rows.walked_rows_kernel`)."""
        placement, factor, model = solver.placement, solver.factor, solver.model
        # Without contacts no row is a contact's: the writer reads no contact input.
        contact_input = (
            bind_contact_input(solver, state_in, factor, contacts, dt, 1.0) if contacts is not None else ContactInput()
        )
        dummy_int = placement.dummy_is_free_rigid
        blocks = factor.block_capacity if factor.eliminates else 0
        lanes = walk_rows.writer_lanes(placement.dense_internal_max_rows, placement.dense_max_constraints, model.device)
        inputs = [
            bind_dense_rows(placement),
            placement.row_support(),
            placement.constraint_count,
            # The local owners' packets and the keyed tiers keep their worlds' seeds: the writer walks the general
            # worlds' rows.
            placement.keyed_rows.owner if placement.keyed_rows is not None else placement.local_solve_owner,
            int(placement.row_packets is not None or placement.keyed_rows is not None),
            lanes,
            contact_input,
            patch_row_law(solver),
            walk_rows.packed_sink(solver),
            factor.blocks() if blocks else Preelim(),
            self.mimic_slot if blocks and self.mimic_slot is not None else dummy_int,
            self.link_slot if blocks and self.link_slot is not None else dummy_int,
            self.joint_seeds(solver, state_in),
            dt,
            solver.options.contact_speculative_scale,
            self.contact_w,
            self.effective_restitution_velocity_threshold,
            int(self.set.weighted),
            int(solver.options.contact_compliance),
        ]
        wp.launch(
            walk_rows.writer_kernel(solver),
            dim=(solver.forest.world_count, lanes),
            inputs=inputs,
            block_dim=walk_rows.WRITER_THREADS,
            device=model.device,
        )

    def write_support(self, solver, state_in, contacts: Contacts | None, dt: float) -> None:
        """Case K1's writer in two launches: the step's units listed, with the contact rows' records
        (:func:`.walk_rows.support_units_kernel`), then the walkers over the list into the support sink, a lane a unit
        (:func:`.walk_rows.support_rows_kernel`), each row of a slipping tree released by the predictor's border."""
        placement, model, sweep = solver.placement, solver.model, solver.sweep
        contact_input = (
            bind_contact_input(solver, state_in, solver.factor, contacts, dt, 1.0)
            if contacts is not None
            else ContactInput()
        )
        dense_rows, law = bind_dense_rows(placement), patch_row_law(solver)
        sink = sweep.support_sink(solver.v_hat)
        units = sweep.support_units
        units.count.zero_()
        wp.launch(
            sweep.support_units_kernel,
            dim=(solver.forest.world_count, sweep.support_lanes),
            inputs=[dense_rows, placement.unit_source, placement.constraint_count, contact_input, law, units],
            device=model.device,
        )
        wp.launch(
            sweep.support_rows,
            dim=sweep.support_threads,
            inputs=[
                dense_rows,
                placement.unit_source,
                units,
                sweep.support_threads,
                contact_input,
                law,
                sink,
                self.joint_seeds(solver, state_in),
                dt,
                solver.options.contact_speculative_scale,
                self.contact_w,
                self.effective_restitution_velocity_threshold,
                int(self.set.weighted),
                int(solver.options.contact_compliance),
            ],
            device=model.device,
        )

    def write_slot_rows(self, solver, contact_input: ContactInput, law, dt: float) -> None:
        """The sparse diagonal's one writer (:func:`.walk_rows.write_slot_rows`): every dense row, or under the compact
        boundary its limit prefix, walked into the slots sink with its diagonal, incident velocity and rhs."""
        placement = solver.placement
        prefix = bool(placement.compact_contact_boundary)
        rows = max(int(placement.dense_internal_max_rows), 1) if prefix else placement.dense_max_constraints
        lanes = min(rows, BIAS_LANES)
        wp.launch(
            walk_rows.write_slot_rows,
            dim=(solver.forest.world_count, lanes),
            inputs=[
                bind_dense_rows(placement),
                placement.row_support(),
                walk_rows.bind_slot_rows(solver),
                walk_rows.packed_sink(solver),
                placement.slot_counter,
                placement.dense_phase_bounds,
                int(prefix),
                lanes,
                contact_input,
                law,
                solver.v_hat,
                dt,
                solver.options.contact_speculative_scale,
                self.contact_w,
                self.effective_restitution_velocity_threshold,
                int(self.set.weighted),
                int(solver.options.contact_compliance),
            ],
            device=solver.model.device,
        )

    def _build_friction_patches(self, solver, state_in, contacts):
        """Construct the shared contact law independently of the dynamics owner."""
        if self.friction_anchors_enabled and contacts is not None:
            self.friction_patches.build(
                solver.model,
                state_in,
                contacts,
                body_to_articulation=solver.forest.body_to_articulation,
                is_free_rigid=solver.forest.is_free_rigid
                if solver.forest.is_free_rigid is not None
                else solver.placement.dummy_is_free_rigid,
                contact_gap_gate=solver.options.contact_gap_gate,
                same_articulation_gap_gate=solver.options.same_articulation_contact_gap_gate,
                articulation_pair_gap_gate=solver.options.articulation_pair_contact_gap_gate,
                friction_gap=solver.options.contact_friction_gap_threshold,
            )

    def _link_friction_patches(self, solver, contacts):
        """Link shared normal loads before any consumer packs row metadata."""
        if self.friction_anchors_enabled and solver.sweep.patch_frame_built(contacts):
            self.friction_patches.link(
                solver.placement.contact_world,
                solver.placement.contact_slot,
                solver.placement.contact_path,
                solver.placement.contact_slots_needed,
                solver.sweep.patch_routes(),
            )
