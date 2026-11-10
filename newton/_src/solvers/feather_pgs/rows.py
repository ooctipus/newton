# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The row laws' tables: what every row writer reads besides the step's state, built once per model.

:meth:`RowLaws.build` resolves the laws from the options (:func:`.configuration.check_options`) and the model:

- the contact law: the friction law (one wrench a friction region, its stick's gain ``friction_anchor_beta > 0``), the
  regularization weight, the restitution threshold, the shape materials and contact torsion;
- the rigid bodies' speed limits;
- the joint-limit tables (limited sizes, each coordinate's limit coordinate) and the warp joint-limit row builders;
- the mimic and link plans (the bilateral rows' tables);
- the velocity prescale's launch;
- :attr:`RowLaws.set`, the route table's laws axis (:class:`.route_table.Laws`), once the bilateral pre-elimination
  is decided.

The step's row pass is the families' methods, from :meth:`RowLaws.begin` to :meth:`RowLaws.write_contacts`, in the
launch order that fixes the slot order (the stage list runs those a class holds): each family records its rows and
their seeds, and the one writer walks them (:meth:`RowLaws.write_walked`, :mod:`.walk_rows`) into the class's sink with
their diagonals, incident velocities and right-hand sides.
The device law functions stay in :mod:`.row_laws`, :mod:`.row_build` and :mod:`.prefix_rows`.
"""

from __future__ import annotations

import warnings
from functools import cache

import numpy as np
import warp as wp

from ...core import memory
from ...core.execution import Shape, Work, launch, shape
from ...sim import Contacts, Model, State
from ...sim.enums import JointType
from . import field_group, host_index, prefix_rows, walk_rows
from .contact_input import ContactInput
from .contact_input import bind as bind_contact_input
from .device_preludes import lane_group
from .factor.fields import Preelim
from .field_group import Field, FieldGroup, SolverArrays
from .forest_plan import ForestPlan, RowLaw
from .friction_patches import _FrictionPatchState as FrictionPatchState
from .friction_patches import patch_row_law
from .joint_table import FreeBodies, bind_free_bodies
from .joint_table import bind as bind_joint_table
from .kernel_template import KernelTemplate, items, params
from .path_selection import json_value, kernel_key
from .route_table import Laws
from .row_kernels import (
    LINK_PLAN,
    LINK_SLOTS,
    Links,
    RowBias,
    allocate_link_slots,
    allocate_rigid_velocity_limit_slots,
    prepare_fused_diagonal_joint_limits,
    write_link_rows,
)
from .row_laws import LINK_ROWS, ROW_LAW
from .row_records import UNIT_LIMIT, UNIT_MIMIC, UNIT_SHIFT, UNIT_VELOCITY, RowSupport
from .row_records import bind as bind_dense_rows
from .row_slots import allocate_world_contact_slots, finalize_constraint_counts_with_status, snapshot_dense_phase_bound

VELOCITY_LIMIT_SLOTS = FieldGroup(
    Field("velocity_limit_slot", wp.int32, ("limit_row",), fill=-1, owner="laws"),
    Field("velocity_limit_sign", wp.float32, ("limit_row",), owner="laws"),
)
"""Two unilateral joint velocity-limit rows per DOF (lower ``+e_i``, upper ``-e_i``): their slots and signs."""

MIMIC_PLAN = FieldGroup(
    Field("_mimic_art_start", wp.int32, ("articulation_bound",), owner="laws"),
    Field("_mimic_art_list", wp.int32, ("mimic",), owner="laws"),
    Field("_mimic_valid", wp.int32, ("mimic",), owner="laws"),
    Field("_mimic_world", wp.int32, ("mimic",), owner="laws"),
    Field("_mimic_dof0", wp.int32, ("mimic",), owner="laws"),
    Field("_mimic_dof1", wp.int32, ("mimic",), owner="laws"),
    Field("_mimic_q0", wp.int32, ("mimic",), owner="laws"),
    Field("_mimic_q1", wp.int32, ("mimic",), owner="laws"),
    Field("_mimic_coef0", wp.float32, ("mimic",), owner="laws"),
    Field("_mimic_coef1", wp.float32, ("mimic",), owner="laws"),
    Field("_mimic_enabled", wp.bool, ("mimic",), owner="laws"),
)
"""The per-mimic tables: each articulation's valid mimics (CSR), each mimic's DOFs and coordinates (static), and its
coefficients and enable bit (read again at a constraint notification)."""

MIMIC_SLOTS = FieldGroup(Field("mimic_slot", wp.int32, ("mimic",), fill=-1, owner="laws"))


__all__ = [
    "RowLaws",
    "joint_limit_warp_kernel",
    "mimic_warp_kernel",
    "prefix_shape",
    "velocity_limit_warp_kernel",
    "velocity_prescale_launch",
]

JOINT_ROW_WARPS = 4
"""Articulations (warps) a block of the joint row families' producers on CUDA (:func:`prefix_shape`)."""

_CONTACT_BUILD_THREAD_CAP = 65536


def _holds_contact_rows(contacts: Contacts | None) -> bool:
    """Whether the step's contacts can hold rows: a counted stream with rows."""
    return (
        contacts is not None
        and getattr(contacts, "rigid_contact_count", None) is not None
        and contacts._rigid_contact_rows > 0
    )


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


PREFIX_LINE = 40
"""The joint row families' warps' register line (:func:`_prefix_warp_kernel`): twelve blocks of 128 a
multiprocessor."""


def _prefix_work(articulations: int) -> Work:
    """The joint row families' work: a warp an articulation, :data:`JOINT_ROW_WARPS` a block, at :data:`PREFIX_LINE`."""
    return Work(articulations, lanes=32, block=32 * JOINT_ROW_WARPS, registers=PREFIX_LINE)


def prefix_shape(articulations: int, device) -> Shape:
    """The joint row families' launch over ``articulations`` (:func:`_prefix_warp_kernel`): a warp an articulation,
    :data:`JOINT_ROW_WARPS` a block, on CUDA; an articulation a thread on the CPU (:func:`.execution.shape`)."""
    return shape(_prefix_work(articulations), memory.device(device))


def _prefix_warp_params(law) -> tuple:
    """The warp prefix producers' operands after the articulations' count (the dispatch's guard): the family's law
    (:data:`.prefix_rows.LAW`, with the velocity or the mimic family's inputs) and the rows' placement (its unit
    sources, :class:`.row_records.RowSupport`)."""
    return params(
        ("articulation_dof_start", wp.array[int]),
        ("art_to_world", wp.array[int]),
        ("group_to_art", wp.array[int]),
        ("law", law),
        ("support", RowSupport),
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
def _prefix_warp_kernel(size: int, family: str, device_arch, mimics: int = 0) -> wp.Kernel:
    """A joint row family of the one writer, one warp per articulation of ``size`` coordinates
    (:func:`.prefix_rows.prefix_rows`), built for ``device_arch`` (its groups a block and bound; one on the CPU): its
    joint limits, its joint velocity limits, or its mimics (at most ``mimics`` an articulation). The warp takes each
    candidate's law, reserves the active rows on the world's counter, writes their records and publishes each row's
    seed, its unit source, which the one writer walks (a mimic's slot in ``mimic_slot`` too); a limit that may act
    marks its world's acts word."""
    name, native, law, kind = _PREFIX_FAMILIES[family]
    kind = int(kind)
    if family == "mimics":
        source = f"({kind} << {int(UNIT_SHIFT)}) | mimic"
        extra = "            law.mimic_slot.data[mimic] = slot;\n"
    else:
        source, extra = f"({kind} << {int(UNIT_SHIFT)}) | (2 * (first_dof + d0) + (w0 < 0.0f ? 1 : 0))", ""
    # The row's seed: its unit source, the coordinate's side, in the units' own rows. The sink runs on an accepted slot
    # (below the capacity), and the unit sources are the capacity wide (Placement.hold_units).
    sink = f"""{extra}            support.units.data[(size_t)prefix_line * support.units.shape[1] + slot] = {source};"""
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
        locals_="int mimic = -1;" if family == "mimics" else "",
        side="art",
    )
    first_dof = "    const int first_dof = articulation_dof_start.data[art];\n"
    if family == "mimics":
        first_dof += (
            "    const int mimic_first = law.mimic_start.data[art], mimic_end = law.mimic_start.data[art + 1];\n"
        )
    body = f"""    const int art = group_to_art.data[group];
    const int world = art_to_world.data[art];
{first_dof}{rows}"""
    suffix = f"_m{mimics}" if family == "mimics" else ""
    found = shape(_prefix_work(0), memory.built_for(device_arch))
    return KernelTemplate(
        f"{name}_{size}_{found.groups}{suffix}",
        _prefix_warp_params(law),
        body,
        items("group", groups=found.groups, lanes=32, guard="articulation_count"),
        preludes=(lane_group(32), *ROW_LAW.preludes()),
        dual=True,
        native_name=native,
        launch_bounds=found.bounds,
    ).kernel()


def joint_limit_warp_kernel(size: int, device_arch) -> wp.Kernel:
    """The one writer's joint-limit rows, one warp per articulation (:func:`_prefix_warp_kernel`)."""
    return _prefix_warp_kernel(size, "limits", device_arch)


def velocity_limit_warp_kernel(size: int, device_arch) -> wp.Kernel:
    """The one writer's joint velocity-limit rows, one warp per articulation (:func:`_prefix_warp_kernel`)."""
    return _prefix_warp_kernel(size, "velocity", device_arch)


def mimic_warp_kernel(size: int, mimics: int, device_arch) -> wp.Kernel:
    """The one writer's mimic rows, at most ``mimics`` an articulation, one warp per articulation
    (:func:`_prefix_warp_kernel`)."""
    return _prefix_warp_kernel(size, "mimics", device_arch, mimics)


_BETA_REASONS = {
    "default": "friction_anchor_beta omitted: persistent patch friction at the 0.2 default.",
    "argument": "friction_anchor_beta passed explicitly.",
}


class RowLaws:
    """The row laws' tables of one model and its options (:meth:`build`).

    The tables keep the names they had on the solver, without the leading underscore, except the tables uploaded by
    field groups (:data:`MIMIC_PLAN`, :data:`.row_kernels.LINK_PLAN`), which keep their fields'
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
        options,
    ) -> RowLaws:
        """Resolve the row laws and build their tables.

        Args:
            model: The model.
            forest: The forest plan.
            options: The solver's options.

        Raises:
            ValueError: For a ``friction_anchor_beta`` that is not finite and positive, or contact torsion with an option
                it does not compose with.
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
            friction_anchor_beta = 0.2
            laws.friction_anchor_beta_source = "default"
        laws.friction_anchor_beta = float(friction_anchor_beta)
        # The stick's gain: every friction region carries its position-level stick (DECISIONS 11-04 23:18).
        if not np.isfinite(laws.friction_anchor_beta) or laws.friction_anchor_beta <= 0.0:
            raise ValueError("friction_anchor_beta, the stick's gain, must be finite and positive")
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
        laws.friction_patches = None
        return laws

    def _contact_torsion(self, model, options) -> None:
        """Contact torsion: every friction contact's fourth row, its spin about the normal, on the elliptic cone with
        its friction pair. Its coefficient is the shapes' torsional friction (``Model.shape_material_mu_torsional``,
        a length), combined per contact as the friction coefficient is (:func:`.contact_laws.contact_friction`)."""
        self.contact_torsion_enabled = bool(options.contact_torsion)
        self.torsion_rows = int(self.contact_torsion_enabled)
        # Spin rows appear on a contact whose region has a footprint (patch friction is one wrench a region,
        # core-docs/WRENCH/VERDICT.md) or under contact torsion on every friction contact: the sweeps then read each
        # contact's rows from the data (a pair, or a pair and a spin row). The model bounds both (RowLaw.SPIN: a body
        # pair that can touch at two points; RowLaw.CONTACT: one that can touch at all); where no world holds one, no
        # contact carries a spin row and every sweep, writer and lister runs its plain variant.
        present = RowLaw(int(np.bitwise_or.reduce(self.forest.world_laws, initial=0)))
        footprint = RowLaw.CONTACT if self.contact_torsion_enabled else RowLaw.SPIN
        self.spin_rows = bool(options.enable_contact_friction) and bool(present & footprint)
        # The producers read the same fact (ContactInput.spin): a contact the model cannot make carries no spin row.
        self.spin_rows_where = (2 if self.contact_torsion_enabled else 1) if self.spin_rows else 0
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
        arch = model.device.arch
        self.joint_limit_warp_kernels = {size: joint_limit_warp_kernel(size, arch) for size in self.joint_limit_sizes}
        # The velocity limits sit on the same coordinates (limit_q_index >= 0), so on the same sizes.
        self.velocity_limit_warp_kernels = (
            {size: velocity_limit_warp_kernel(size, arch) for size in self.joint_limit_sizes}
            if self.options.enable_joint_velocity_limits
            else {}
        )

    # --- the bilateral rows' plans ---------------------------------------------------------------------------------
    def _mimic_plan(self, model) -> None:
        """Precompute the static per-mimic lookup tables for the MIMIC row family.

        Every record of the model's mimics (:class:`.forest_plan.MimicRelations`: constraint and joint-owned, one
        relation a record) is resolved into articulation / DOF / coordinate indices, and the supported subset kept:
        both joints in one articulation, distinct DOFs, the record's joints admitting it. Invalid records are warned
        about and masked off. ``coef0``/``coef1``/``enabled`` are the records' own copies, read again at a
        constraint notification (:meth:`refresh_mimics`). All buffers are allocated here, once, so the per-step
        launches are CUDA-graph-safe.
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
        mimics = self.forest.mimics
        n = mimics.count
        if n == 0 or not model.articulation_count or self.forest.art_to_world is None:
            return

        j0, j1, component = mimics.follower, mimics.reference, mimics.component
        articulation_start = model.articulation_start.numpy().astype(np.int64, copy=False)
        joint_qd_start = model.joint_qd_start.numpy().astype(np.int32, copy=False)
        joint_q_start = model.joint_q_start.numpy().astype(np.int32, copy=False)
        labels = list(getattr(model, "constraint_mimic_label", []) or [])

        articulation_bounds = articulation_start[: model.articulation_count + 1]
        a0, a1 = host_index.lookup(articulation_bounds, j0), host_index.lookup(articulation_bounds, j1)
        spans = (a0 < 0) | (a1 < 0) | (a0 != a1)
        dof0, dof1 = joint_qd_start[j0] + component, joint_qd_start[j1] + component
        self_coupled = ~spans & (dof0 == dof1)
        valid = (~spans & mimics.supported & ~self_coupled).astype(np.int32)
        art = np.where(valid != 0, a0, -1).astype(np.int32)
        for k in np.flatnonzero(valid == 0):
            if k < mimics.constraints:
                what = f"mimic constraint '{labels[k] if k < len(labels) else f'mimic_{k}'}'"
            else:
                what = f"joint {int(j0[k])}'s mimic"
            if spans[k]:
                warnings.warn(
                    f"SolverFeatherPGS: {what} spans articulations ({int(a0[k])} vs {int(a1[k])}); "
                    "cross-articulation mimic is unsupported and the constraint is ignored.",
                    stacklevel=2,
                )
            elif not mimics.supported[k]:
                warnings.warn(
                    f"SolverFeatherPGS: {what} references a non-1-DoF "
                    "joint (only REVOLUTE/PRISMATIC are supported); the constraint is ignored.",
                    stacklevel=2,
                )
            else:
                warnings.warn(f"SolverFeatherPGS: {what} couples a DOF to itself; ignored.", stacklevel=2)

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
        coef0, coef1, enabled = mimics.coefficients(model)
        self._arrays.upload(
            MIMIC_PLAN,
            _mimic_art_start=art_start,
            _mimic_art_list=order,
            _mimic_valid=valid,
            _mimic_world=mimic_world,
            _mimic_dof0=dof0,
            _mimic_dof1=dof1,
            _mimic_q0=joint_q_start[j0] + component,
            _mimic_q1=joint_q_start[j1] + component,
            _mimic_coef0=coef0,
            _mimic_coef1=coef1,
            _mimic_enabled=enabled,
        )
        self._arrays.allocate(MIMIC_SLOTS, mimic=n)
        self.mimic_count = n
        # A warp per articulation produces a size's mimic rows, its candidates the most mimics an articulation of that
        # size has.
        per_art = np.diff(art_start)
        sizes = self.forest.response_dof_count[: model.articulation_count]
        self.mimic_warp_kernels = {
            size: mimic_warp_kernel(size, int(per_art[sizes == size].max()), model.device.arch)
            for size in self.mimic_sizes
        }

    def refresh_mimics(self, model) -> None:
        """Read the mimics' coefficients and enable bits again (a constraint notification), into the same arrays."""
        if self.mimic_count:
            for name, value in zip(
                ("_mimic_coef0", "_mimic_coef1", "_mimic_enabled"), self.forest.mimics.coefficients(model), strict=True
            ):
                getattr(self, name).assign(value)

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
        self.friction_patches = FrictionPatchState(model, pool.ready, phi, pool=pool)
        # Joint velocity-limit buffers (per-DOF tracking). Independent from the
        # joint-position-limit buffers so the two flags can be used separately.
        # Both rows of a finite limit are allocated every step, so the velocity box
        # is present before the predicted velocity escapes.
        if options.enable_joint_velocity_limits and model.joint_dof_count > 0:
            arrays.allocate(VELOCITY_LIMIT_SLOTS, limit_row=2 * model.joint_dof_count)
        else:
            arrays.release(VELOCITY_LIMIT_SLOTS)

    # --- the row pass ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _launch_prefix_warp(solver, kernel, size: int, law, support) -> None:
        """Launch a prefix-rows warp producer (:func:`_prefix_warp_kernel`) over size group ``size`` at
        :func:`prefix_shape`."""
        n_arts = solver.forest.n_arts_by_size[size]
        launch(
            kernel,
            prefix_shape(n_arts, solver.model.device),
            inputs=[
                n_arts,
                solver.forest.articulation_dof_start,
                solver.forest.art_to_world,
                solver.forest.group_to_art[size],
                law,
                support,
            ],
            device=solver.model.device,
        )

    # The row pass, in the launch order that is the slot order: the allocators reserve slots with atomics, in the order
    # mimic, link, joint limit, velocity limit, contacts, then the free bodies' rows, and the phase-bound snapshots are
    # glued to that sequence. The stage list (:func:`.step_plan._stages`) runs the families a class holds, in this
    # order: :meth:`begin`, :meth:`write_mimics`, :meth:`write_links`, :meth:`prepare_fused_limits`,
    # :meth:`write_limits`, :meth:`bound_phase` (0), :meth:`write_velocity_limits`, :meth:`bound_phase` (1),
    # :meth:`write_contacts`, :meth:`bound_free_body_contacts`, :meth:`allocate_free_body_limits`,
    # :meth:`write_free_bodies`, :meth:`finalize_dense`.
    #
    # Dense row-family layout: slots are handed out in the order [mimic][joint-limit][joint-velocity-limit]
    # [contact/friction], so each family owns one contiguous dense row range:
    #   mimic and position limits: [0, bounds[0])
    #   velocity limits:           [bounds[0], bounds[1])
    #   contacts and friction:     [bounds[1], m_dense)
    # Mimic rows come first, ahead of the (activation-gated, flickering) limit family, so warm-started mimic lambdas
    # keep a stable slot index across steps.

    def begin(self, solver, state_in: State, state_aug: State, dt: float) -> None:
        """Clear the rows' allocation state."""
        solver.placement.begin()

    def _prefix_law(self, solver, law, state_in: State, dt: float, **extra):
        """A joint row family's law (:mod:`.prefix_rows`), bound for this step."""
        return field_group.bind(
            law,
            solver=solver,
            placement=solver.placement,
            factor=solver.factor,
            laws=self,
            options=solver.options,
            model=solver.model,
            state_in=state_in,
            dt=dt,
            **extra,
        )

    def write_mimics(self, solver, state_in: State, dt: float) -> None:
        """The mimic (joint coupling) equality rows, a static count per model: one warp per articulation reserves,
        writes and publishes them, as the joint limits (:mod:`.prefix_rows`)."""
        law = self._prefix_law(solver, prefix_rows.PrefixMimicLaw, state_in, dt, mimics=self)
        support = solver.placement.row_support()
        for size, warp_kernel in self.mimic_warp_kernels.items():
            self._launch_prefix_warp(solver, warp_kernel, size, law, support)

    def write_links(self, solver, state_in: State) -> None:
        """The links' rows, static per model, their sides the trees of the link's two pieces (staged with links)."""
        model = solver.model
        links = field_group.bind(Links, laws=self)
        wp.launch(
            allocate_link_slots,
            dim=self.link_count,
            inputs=[links, solver.placement.dense_max_constraints],
            outputs=[solver.placement.slot_counter, solver.placement.dense_first_rejected_slot],
            device=model.device,
        )
        wp.launch(
            write_link_rows,
            dim=LINK_ROWS * self.link_count,
            inputs=[
                links,
                state_in.body_q,
                model.body_com,
                solver.forest.body_to_articulation,
                solver.options.pgs_beta,
                solver.placement.row_support(),
            ],
            outputs=[bind_dense_rows(solver.placement)],
            device=model.device,
        )

    def prepare_fused_limits(self, solver, state_in: State, dt: float) -> None:
        """The fused diagonal's joint limits: each limit coordinate's activity and its two sides' rhs (staged where the
        placement fuses them)."""
        placement = solver.placement
        wp.launch(
            prepare_fused_diagonal_joint_limits,
            dim=solver.forest.world_count * solver.forest.max_world_dofs,
            inputs=[
                bind_joint_table(solver),
                solver.forest.world_dof_indices,
                solver.forest.max_world_dofs,
                placement.fused_diagonal_limit_dof_mask,
                self.joint_limit_q_index,
                state_in.joint_q,
                solver.options.joint_limit_activation_gap,
                solver.options.pgs_beta,
                dt,
                solver.v_hat,
            ],
            outputs=[
                placement.fused_diagonal_limit_active,
                placement.fused_diagonal_limit_lower_rhs,
                placement.fused_diagonal_limit_upper_rhs,
            ],
            device=solver.model.device,
        )

    def limit_sizes(self, solver) -> tuple[int, ...]:
        """The size groups whose joint limits the warp producers write: the limited sizes the fused diagonal does not
        hold, in the forest's order."""
        fused = solver.spec.classes.fused_sizes
        return tuple(size for size in solver.forest.size_groups if size in self.joint_limit_sizes and size not in fused)

    def write_limits(self, solver, state_in: State, dt: float, sizes: tuple[int, ...]) -> None:
        """The joint position-limit rows of ``sizes`` (:meth:`limit_sizes`), one warp per articulation."""
        law = self._prefix_law(solver, prefix_rows.PrefixLaw, state_in, dt)
        support = solver.placement.row_support()
        for size in sizes:
            self._launch_prefix_warp(solver, self.joint_limit_warp_kernels[size], size, law, support)

    def bound_phase(self, solver, phase: int) -> None:
        """Snapshot the dense rows' family boundary ``phase`` (0: after the position limits, 1: after the velocity
        limits), where the row pass holds the dense rows' order; a family that is off leaves the current count."""
        wp.launch(
            snapshot_dense_phase_bound,
            dim=solver.forest.world_count,
            inputs=[solver.placement.slot_counter, phase],
            outputs=[solver.placement.dense_phase_bounds],
            device=solver.model.device,
        )

    def write_velocity_limits(self, solver, state_in: State, dt: float) -> None:
        """The joint velocity-limit rows (a per-DOF clamp on ``|qdot_i|``), one warp per articulation; the contact rows
        follow them, and the sweeps visit them in a final pass whatever their slots."""
        law = self._prefix_law(solver, prefix_rows.PrefixVelocityLaw, state_in, dt)
        support = solver.placement.row_support()
        for size, warp_kernel in self.velocity_limit_warp_kernels.items():
            self._launch_prefix_warp(solver, warp_kernel, size, law, support)

    def write_contacts(
        self, solver, state_in: State, state_aug: State, contacts: Contacts, dt: float, producer: str
    ) -> None:
        """The contact rows' slots, after the joint families: the dense rows' unit sources (the one writer's seeds) and
        the free bodies' slots (their writer's, :meth:`write_free_bodies`).

        Args:
            solver: The solver whose step this is: its model, options, plan and primitives.
            state_in: The step's input state.
            state_aug: The step's dynamics (the tree factor).
            contacts: The step's contacts, or ``None``.
            dt: The step size.
            producer: The route's producer of the contact rows (:attr:`.route_table.RouteRow.producer`), from the plan.
        """
        model = solver.model
        max_constraints = solver.placement.dense_max_constraints
        # The free bodies' contacts take their own run only where a colored world sweeps its units (Placement).
        has_free_rigid_flag = int(solver.placement.free_contact_rows)
        # Dummy arrays when MF is not active (kernel still needs valid pointers), preallocated: a per-step wp.zeros
        # would add an allocation and a memset every step and leave a captured graph a freed temporary.
        is_free_rigid = (
            solver.forest.is_free_rigid
            if solver.forest.is_free_rigid is not None
            else solver.placement.dummy_is_free_rigid
        )
        mf_slot_counter = solver.placement.mf_slot_counter
        dense_dropped_rows = solver.placement.row_dropped_dense
        mf_dropped_rows = solver.placement.row_dropped_mf
        mf_first_rejected_slot = solver.placement.mf_first_rejected_slot

        if _holds_contact_rows(contacts):
            enable_friction_flag = 1 if solver.options.enable_contact_friction else 0
            pool = solver.placement.contact_pool
            count = contacts.rigid_contact_count
            # Sized for the bound in a capture whose launches bind: every replay runs them at the live
            # count and strides with the captured threads (ContactPool.launch); otherwise for the ready rows.
            rows = pool.launch_rows()
            contact_build_threads = min(rows, _CONTACT_BUILD_THREAD_CAP)

            self._build_friction_patches(solver, state_in, contacts)

            # The allocator reads the contacts, the bodies and the row law, and writes the
            # contacts' allocation records, through one struct (contact_input.ContactInput), bound for this step.
            contact_input = bind_contact_input(solver, state_in, state_aug, contacts, dt, 1.0)

            # The route's producer writes every contact row (get_path_selection() records it, routes.contact_rows).
            # The one writer's contact rows are its own to fill: their unit sources are written where their slots are
            # reserved.
            units = solver.placement.unit_source if producer == "walk" else solver.placement.no_units
            # A contact whose rows may act at the predictor marks its world (Placement.world_acts).
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
                    solver.placement.acts_twists,
                ],
                outputs=[
                    solver.placement.world_acts,
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

    def bound_free_body_contacts(self, solver) -> None:
        """Record where the free bodies' contact rows end in their run: every contact and friction row precedes the
        speed rows in the run's slots, so the sweeps loop the contact prefix in the contact phases and the speed rows'
        tail in the last word (staged with the run, :attr:`.step_plan.Capacities.free_contact_rows`)."""
        wp.copy(solver.placement.mf_contact_rows_end, solver.placement.mf_slot_counter)

    def bound_speed_rows(self, solver) -> None:
        """Record where the line's speed rows start (``speed_first``): every contact row is reserved, so the free
        bodies' speed rows, reserved next, are the line's last rows (staged where the line holds them or the tiers
        read it)."""
        wp.copy(solver.placement.speed_first, solver.placement.slot_counter)

    def allocate_free_body_limits(self, solver) -> None:
        """The free bodies' speed limits as rows, not post-solve clamps: their slots, records and unit sources,
        reserved after the contacts' so they are the last rows of the line (or of the free bodies' run, where their
        contacts keep one), which the sweeps visit in their last word with the joints' velocity limits (staged with free
        bodies that carry speed-limit slots)."""
        placement, model = solver.placement, solver.model
        # Without articulation metadata no root is a free rigid body's, so the kernel reads no root's first DOF.
        root_dof_start = solver.forest.articulation_root_dof_start
        if root_dof_start is None:
            root_dof_start = placement.dummy_is_free_rigid
        if placement.free_contact_rows:
            counter, capacity, rejected = (
                placement.mf_slot_counter,
                placement.mf_max_constraints,
                placement.mf_first_rejected_slot,
            )
            base, units = placement.body_base, placement.no_units
        else:
            counter, capacity, rejected = (
                placement.slot_counter,
                placement.dense_max_constraints,
                placement.dense_first_rejected_slot,
            )
            # A class with free bodies walks (SolverFeatherPGS): the one writer takes the line's speed rows.
            base, units = 0, placement.unit_source
        wp.launch(
            allocate_rigid_velocity_limit_slots,
            dim=solver.forest.free_rigid_body_count,
            inputs=[
                bind_free_bodies(solver),
                model.body_flags,
                root_dof_start,
                solver.v_hat,
                solver.options.velocity_limit_activation_fraction,
                capacity,
                base,
            ],
            outputs=[
                placement.world_acts,
                placement.rigid_velocity_limit_slot,
                counter,
                rejected,
                bind_dense_rows(placement),
                units,
            ],
            device=model.device,
        )

    def write_free_bodies(self, solver, state_in: State, contacts: Contacts | None, dt: float) -> None:
        """The free bodies' rows, once their families' slots are reserved (staged with free bodies): one writer over
        their contacts' rows and their speed rows (:func:`.walk_rows.write_free_body_rows`), on its threads at its own
        line (:data:`.walk_rows.WRITERS`); their counts finalized with status."""
        placement, model = solver.placement, solver.model
        # The free bodies' contacts are theirs only where they keep their own run (Placement.free_contact_rows).
        if _holds_contact_rows(contacts) and placement.free_contact_rows:
            data = bind_contact_input(solver, state_in, solver.factor, contacts, dt, 1.0)
            rows = placement.contact_pool.launch_rows()
        else:
            # No contact: the writer reads only the count, zero.
            data, rows = ContactInput(), 0
            data.count = placement.dummy_contact_count
        speed = placement.rigid_velocity_limit_slot is not None
        speed_bodies = solver.forest.free_rigid_body_count if speed else 0
        threads = walk_rows.support_threads(rows + speed_bodies, model.device, walk_rows.WRITERS["free_body"])
        wp.launch(
            walk_rows.write_free_body_rows(model.device.arch),
            dim=threads,
            inputs=[
                walk_rows.free_body_sink(solver),
                bind_dense_rows(placement),
                data,
                patch_row_law(solver),
                bind_free_bodies(solver) if speed else FreeBodies(),
                placement.rigid_velocity_limit_slot if speed else placement.dummy_is_free_rigid,
                speed_bodies,
                threads,
                self.bias(solver.options, dt),
            ],
            block_dim=walk_rows.WRITERS["free_body"].block,
            device=model.device,
        )
        self._finalize(solver, 1)

    def _finalize(self, solver, family: int) -> None:
        """Finalize a row class's counts with status (0: the dense rows, 1: the free bodies')."""
        placement = solver.placement
        wp.launch(
            finalize_constraint_counts_with_status,
            dim=solver.forest.world_count,
            inputs=[
                placement.row_slot_counter[family],
                (placement.dense_max_constraints, placement.mf_max_constraints)[family],
                family,
                placement.row_first_rejected[family],
            ],
            outputs=[placement.row_count[family], placement.constraint_capacity_status],
            device=solver.model.device,
        )

    def finalize_dense(self, solver) -> None:
        """Finalize the dense rows' counts with status, after every dense producer."""
        self._finalize(solver, 0)

    def bias(self, options, dt: float) -> RowBias:
        """The rows' law's step parameters (:class:`.row_kernels.RowBias`) of a step of ``dt``."""
        bias = RowBias()
        bias.dt, bias.speculative_scale = dt, options.contact_speculative_scale
        bias.contact_w, bias.restitution_threshold = self.contact_w, self.effective_restitution_velocity_threshold
        bias.weighted, bias.material = int(self.set.weighted), int(options.contact_compliance)
        return bias

    def write_walked(
        self, solver, state_in, contacts: Contacts | None, dt: float, part: int = walk_rows.WRITE_ROWS
    ) -> None:
        """The one writer, after the row pass: every dense row's seed walked into its placement's sink, its diagonal,
        incident velocity and rhs, once the predictor's ``v_hat`` and its decided set are known. The general
        placement's (:func:`.walk_rows.walked_rows_kernel`): a world's lanes release its trees' slips, write and close
        its bilateral blocks, then walk every other row into the packed rows, corrected on the blocks it touches;
        ``part`` its rows (every row, the topology's, or the rest). The support placement's (case K1,
        :meth:`write_support`); the sparse diagonal's (case K2, :meth:`write_slot_rows`)."""
        contact_input = (
            bind_contact_input(solver, state_in, solver.factor, contacts, dt, 1.0)
            if contacts is not None
            else ContactInput()
        )
        if solver.spec.classes.support == "factor":
            self.write_support(solver, state_in, contact_input, dt)
        elif solver.spec.classes.support == "diagonal":
            self.write_slot_rows(solver, contacts, contact_input, dt)
        else:
            self.write_general(solver, state_in, contact_input, dt, part)

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
        seeds.mimic_coef1 = self._mimic_coef1 if mimics else dummy_float
        seeds.link_primary = self._link_primary if links else dummy_int
        seeds.link_shadow = self._link_shadow if links else dummy_int
        seeds.body_q = state_in.body_q
        seeds.body_com = model.body_com
        return seeds

    def write_general(self, solver, state_in, contact_input: ContactInput, dt: float, part: int) -> None:
        """The general placement's one writer (:func:`.walk_rows.walked_rows_kernel`), the rows of ``part``."""
        placement, factor, model = solver.placement, solver.factor, solver.model
        dummy_int = placement.dummy_is_free_rigid
        blocks = factor.block_capacity if factor.eliminates else 0
        # The topology part's lanes from the rows a world's topology holds at once, a block of them a lane; the
        # rest's, whose contact rows make its warps uneven, from every row the topology can fix, a row a lane.
        if part == walk_rows.WRITE_TOPOLOGY:
            rows = -(-placement.dense_internal_rows_at_once // walk_rows.WRITER_BLOCK_ROWS)
        else:
            rows = placement.dense_internal_max_rows
        lanes = walk_rows.writer_lanes(rows, placement.dense_max_constraints, model.device)
        law = (bind_dense_rows(placement), contact_input, patch_row_law(solver))
        law += (self.joint_seeds(solver, state_in), self.bias(solver.options, dt))
        inputs = [
            law[0],
            placement.row_support(),
            placement.constraint_count,
            lanes,
            *law[1:3],
            walk_rows.packed_sink(solver),
            factor.blocks() if blocks else Preelim(),
            self.mimic_slot if blocks and self.mimic_slot is not None else dummy_int,
            self.link_slot if blocks and self.link_slot is not None else dummy_int,
            *law[3:],
            placement.dense_phase_bounds,
            int(part),
            self.row_seeds(solver, dt, *law),
            placement.world_acts,
        ]
        wp.launch(
            walk_rows.writer_kernel(solver, part),
            dim=(solver.forest.world_count, lanes),
            inputs=inputs,
            block_dim=walk_rows.WRITERS["packed"].block,
            device=model.device,
        )

    def row_seeds(self, solver, dt: float, *law) -> walk_rows.RowSeeds:
        """The one writer's EMIT sink (:class:`.walk_rows.RowSeeds`) for a step of ``dt``, which the tiers that walk
        their rows read too, with the ``law``'s inputs (rows, contacts, patch law, joint seeds, bias)."""
        seeds = solver.placement.row_seeds
        seeds.dt, seeds.threshold = dt, self.effective_restitution_velocity_threshold
        if seeds.tiers:
            seeds.rows, seeds.data, seeds.law, seeds.joint, seeds.bias = law
        return seeds

    def write_support(self, solver, state_in, contact_input: ContactInput, dt: float) -> None:
        """Case K1's writer in two launches: the step's units listed, with the contact rows' records
        (:func:`.walk_rows.support_units_kernel`), then the walkers over the list into the support sink, a lane a unit
        (:func:`.walk_rows.support_rows_kernel`), each row of a slipping tree released by the predictor's border."""
        placement, model, sweep = solver.placement, solver.model, solver.sweep
        dense_rows, law = bind_dense_rows(placement), patch_row_law(solver)
        sink = sweep.support_sink(solver.v_hat)
        units = sweep.support_units
        units.start()
        wp.launch(
            sweep.support_units_kernel,
            dim=(solver.forest.world_count, sweep.support_lanes),
            inputs=[dense_rows, placement.unit_source, placement.constraint_count, contact_input, law, units.list],
            device=model.device,
        )
        wp.launch(
            sweep.support_rows,
            dim=sweep.support_threads,
            block_dim=walk_rows.WRITERS["support"].block,
            inputs=[
                dense_rows,
                placement.unit_source,
                units.list,
                sweep.support_threads,
                contact_input,
                law,
                sink,
                self.joint_seeds(solver, state_in),
                self.bias(solver.options, dt),
            ],
            device=model.device,
        )

    def write_slot_rows(self, solver, contacts: Contacts | None, contact_input: ContactInput, dt: float) -> None:
        """The sparse diagonal's one writer (:func:`.walk_rows.slot_writer`): every dense row walked into the slots sink
        with its diagonal, incident velocity and rhs; then the independent chains' marks in the slots they hold (the
        sparse contact marker), which the claim reads after the patches link."""
        placement = solver.placement
        # A world's lanes: a block, at most its rows.
        lanes = min(placement.dense_max_constraints, walk_rows.WRITERS["slots"].block)
        wp.launch(
            walk_rows.slot_writer(solver),
            dim=(solver.forest.world_count, lanes),
            inputs=[
                bind_dense_rows(placement),
                placement.row_support(),
                walk_rows.bind_slot_rows(solver),
                walk_rows.packed_sink(solver),
                placement.constraint_count,
                lanes,
                contact_input,
                patch_row_law(solver),
                solver.v_hat,
                self.bias(solver.options, dt),
            ],
            device=solver.model.device,
        )
        marker = solver.sweep.sparse_contact_marker
        if marker is not None and _holds_contact_rows(contacts):
            pool = placement.contact_pool
            threads = min(pool.launch_rows(), _CONTACT_BUILD_THREAD_CAP)
            pool.launch(
                marker,
                threads,
                inputs=[placement.dense_first, contact_input, threads],
                outputs=[placement.sparse_diagonal_row_dof],
                extent=contacts.rigid_contact_count,
                clamp=True,
            )

    def _build_friction_patches(self, solver, state_in, contacts):
        """Construct the shared contact law independently of the dynamics owner."""
        if contacts is not None:
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

    def link_patches(self, solver, contacts):
        """Link the patches' shared normal loads once every contact row's records are written, before any consumer reads
        them: after the row pass, or after the one writer where it writes them (the stage list's order)."""
        if solver.sweep.patch_frame_built(contacts):
            self.friction_patches.link(
                solver.placement.contact_world,
                solver.placement.contact_slot,
                solver.placement.contact_path,
                solver.placement.contact_slots_needed,
                solver.sweep.patch_routes(),
            )

    def record(self, solver) -> dict:
        """The row laws' route records: ``routes.contact_rows``, ``routes.velocity_limits`` and ``routes.friction``."""
        producer = solver._plan.route.producer
        # The route's producer writes every contact row (the row pass, write_contacts, takes it from the plan).
        sizes = solver.forest.size_groups if producer != "none" else ()
        if not self.options.enable_joint_velocity_limits:
            joint, reason = "off", "enable_joint_velocity_limits is False."
        elif solver.sweep.sparse_velocity_limits:
            joint, reason = (
                "sparse_factor_rows",
                "Native sparse factors solve velocity-limit rows in factor coordinates.",
            )
        else:
            joint, reason = "rows", "Dedicated velocity-limit rows."
        limits = bool(self.has_rigid_body_velocity_limits)
        model = self.model
        prescale = self.options.enable_joint_velocity_limits and model.joint_count > 0 and model.articulation_count > 0
        return {
            "contact_rows": {
                "producer": producer,
                # Every producer, every size: the builder method, its kernels in launch order and the law they write,
                # for the dense contact rows built outside the sweep (_row_builder names the rows built elsewhere).
                "builder_by_dof": {str(int(size)): self._row_builder(solver, producer, size) for size in sizes},
            },
            "velocity_limits": {
                "joint": joint,
                "reason": reason,
                # Lanes per articulation of the pre-solve PhysX velocity prescale; 0 when it launches no kernel.
                "prescale_lanes": int(self.velocity_prescale_lanes) if prescale else 0,
                "activation_fraction": json_value(self.options.velocity_limit_activation_fraction),
                "rigid_body_limits": limits,
                # Mirrors Placement.allocate_free_body_rows and the row allocation of write_contacts (both modes). At an
                # infinite activation fraction the allocator still launches but reserves no row.
                "free_body_rows": bool(limits and self.forest.free_rigid_body_count > 0),
            },
            "friction": {
                "friction_anchor_beta": json_value(self.friction_anchor_beta),
                "anchor_beta_source": self.friction_anchor_beta_source,
                "torsion": bool(self.spin_rows),
                "compliance": bool(self.options.contact_compliance),
                "reason": _BETA_REASONS[self.friction_anchor_beta_source],
                "pair_sort": self.friction_patches.record(),
            },
        }

    @staticmethod
    def _row_builder(solver, method, size) -> dict:
        """Name the kernels that build one DOF size's dense contact rows outside the sweep, in launch order (a launch
        shared by several sizes under each). Not named: the free bodies' rows (walk_rows.write_free_body_rows,
        routes.free_rigid_bodies)."""
        marker = ()
        if size == solver.spec.classes.diagonal_size:
            # The sparse diagonal's marker encodes schedule links into the slots its writer just wrote.
            marker = () if solver.sweep.sparse_contact_marker is None else (solver.sweep.sparse_contact_marker,)
        if method == "walk":
            # The allocator writes the contact rows' unit sources; the one writer writes every contact row's records and
            # walks every row (walk_rows): the general placement's into its packed rows, the support placement's (K1)
            # into its support sink, the sparse diagonal's (K2) into its slots sink.
            if solver._plan.route.support == "factor":
                writers = (solver.sweep.support_units_kernel, solver.sweep.support_rows)
            elif solver._plan.route.support == "diagonal":
                writers = (walk_rows.slot_writer(solver),)
            else:
                # The general placement's rows in two parts: the topology's walked in blocks, and the rest.
                writers = (walk_rows.writer_kernel(solver, walk_rows.WRITE_TOPOLOGY), walk_rows.writer_kernel(solver))
            kernels = (*writers, *marker)
        else:
            raise ValueError(f"No contact-row builder kernels are known for the method {method!r}.")
        entry = {
            "method": method,
            "kernels": [kernel if isinstance(kernel, str) else kernel_key(kernel) for kernel in kernels],
        }
        return entry
