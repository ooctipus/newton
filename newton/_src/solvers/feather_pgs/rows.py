# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The row laws' tables: what every row writer reads besides the step's state, built once per model.

:meth:`RowLaws.build` resolves the laws from the options (:class:`.configuration.Options`) and the model:

- the contact law: the friction law (patch friction with ``friction_anchor_beta > 0``, else point friction), the
  regularization weight, the restitution threshold, the shape materials and contact torsion;
- the rigid bodies' speed limits;
- the joint-limit tables (limited sizes, each coordinate's limit coordinate) and the warp joint-limit row builders;
- the mimic and connect plans (the bilateral rows' tables);
- the velocity prescale's launch;
- :attr:`RowLaws.set`, the route table's laws axis (:class:`.route_table.Laws`), once the bilateral pre-elimination
  is decided.

The device law functions stay in :mod:`.row_laws`, :mod:`.row_build` and :mod:`.prefix_rows`; the row passes that
write the rows stay in the solver's step until they join one row pass.
"""

from __future__ import annotations

import os
import warnings
from functools import cache

import numpy as np
import warp as wp

from ...sim import Model
from ...sim.enums import BodyFlags, JointType
from . import host_index, prefix_rows
from .configuration import Options
from .dense_rows import RowSupport
from .device_preludes import lane_group
from .field_group import SolverArrays
from .forest_plan import ForestPlan
from .friction_patches import _FrictionPatchState as FrictionPatchState
from .kernel_template import KernelTemplate, params
from .route_table import Laws
from .row_laws import ROW_LAW
from .solver_fields import CONNECT_PLAN, CONNECT_SLOTS, MIMIC_PLAN, MIMIC_SLOTS, VELOCITY_LIMIT_SLOTS

__all__ = ["RowLaws", "joint_limit_warp_kernel", "velocity_prescale_launch"]

_CALLER = 4
"""Warnings raised in :meth:`RowLaws.build` name the solver constructor's caller: build runs in the constructor body,
which runs inside the keyword-only shim (deprecate_nonkeyword_arguments)."""

JOINT_LIMIT_WARPS_PER_BLOCK = 4


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


JOINT_LIMIT_WARP_PARAMS = params(
    ("articulation_count", int),
    ("articulation_dof_start", wp.array[int]),
    ("art_to_world", wp.array[int]),
    ("group_to_art", wp.array[int]),
    ("resolved_worlds", wp.array[int]),
    ("law", prefix_rows.PrefixLaw),
    ("support", RowSupport),
    ("J_group", wp.array3d[float]),
)
"""The warp joint-limit row builder's operands: the rows' placement and Jacobian rows (:func:`.dense_rows.jacobian_row`:
the packed rows, by world, or the size group's)."""


@cache
def joint_limit_warp_kernel(size: int, warps_per_block: int) -> wp.Kernel:
    """Build the general route's joint-limit rows, one warp per articulation (:func:`.prefix_rows.prefix_rows`)."""
    # A row's side: its articulation's coordinates in the packed rows (cleared, then the coefficient), or the size
    # group's row (cleared each step).
    sink = f"""            float* J_side = &J_group.data[((size_t)row_index * J_group.shape[1] + slot) * J_group.shape[2] + first];
            if (support.packed != 0) {{
                for (int e = 0; e < {size}; ++e) J_side[e] = 0.0f;
            }}
            J_side[d0] = w0;"""
    rows = prefix_rows.prefix_rows(
        ((prefix_rows.limits(tuple(range(size)), "first_dof + d", "law."),),),
        law="law.",
        sink=sink,
        ordered=False,
        rhs=False,
        side="art",
    )
    snippet = f"""    const int group = block * {warps_per_block} + SUB;
    if (group >= articulation_count) return;
    const int art = group_to_art.data[group];
    const int world = art_to_world.data[art];
    if (world >= 0 && world < resolved_worlds.shape[0] && resolved_worlds.data[world] != 0) return;
    const int first_dof = articulation_dof_start.data[art];
    // The rows' first index and the side's first entry (dense_rows.jacobian_row; one side).
    const int row_index = support.packed != 0 ? world : group;
    const int first = support.packed != 0 && support.dense != 0 ? support.offset.data[art] : 0;
{rows}#endif
"""
    return KernelTemplate(
        f"build_joint_limit_rows_warp_{size}_{warps_per_block}",
        JOINT_LIMIT_WARP_PARAMS,
        snippet,
        preludes=(lane_group(32), *ROW_LAW.preludes()),
        native_name="joint_limit_warp_native",
    ).kernel()


class RowLaws:
    """The row laws' tables of one model and its options (:meth:`build`).

    The tables keep the names they had on the solver, without the leading underscore, except the tables uploaded by
    field groups (:data:`.solver_fields.MIMIC_PLAN`, :data:`.solver_fields.CONNECT_PLAN`), which keep their fields'
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
        if laws.friction_anchors_enabled and (options.contact_shared_anchor or options.contact_friction_shared_anchor):
            warnings.warn(
                "Patch friction selects its own friction locations and carries tangential displacement. "
                "contact_shared_anchor still applies to normal rows; set friction_anchor_beta=0 "
                "to apply shared-anchor flags to point friction rows as well.",
                UserWarning,
                stacklevel=_CALLER,
            )
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
        # Mimic/connect plans must exist before capacity selection so the row census
        # counts their rows (see Placement.internal_rows_per_world).
        laws._mimic_plan(model)
        laws._connect_plan(model)
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
        """The joint-limit tables: the limited sizes, each coordinate's limit coordinate, the warp limit kernels."""
        if not model.articulation_count or not model.joint_count:
            self.joint_limit_sizes = frozenset()
            self.joint_limit_q_index = None
            self.joint_limit_q_index_host = np.zeros(model.joint_dof_count, dtype=np.int32)
            self.joint_limit_warp_kernels = {}
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
        self.joint_limit_warp_kernels = (
            {
                size: joint_limit_warp_kernel(size, warps_per_block=JOINT_LIMIT_WARPS_PER_BLOCK)
                for size in self.joint_limit_sizes
            }
            if model.device.is_cuda
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

    def _connect_plan(self, model) -> None:
        """Precompute the static per-closure lookup tables for the CONNECT row family.

        Loop-closing BALL joints are excluded from FK/ID by
        :meth:`.forest_plan.ForestPlan.build` and enforced here as three bilateral rows
        pinning the joint's parent/child anchors together. Their stored owner comes from
        the child body's tree articulation, independent of joint ordering. FIXED (weld)
        loop joints warn and are ignored for now. All buffers are allocated once
        (CUDA-graph-safe); ``joint_enabled`` is snapshotted at init and may be changed
        later through :meth:`set_loop_joint_enabled`.

        A closure's parent may live outside the child's articulation when it is
        prescribed: a kinematic body (:attr:`~newton.BodyFlags.KINEMATIC`) or the world
        (``parent == -1``). Such closures contribute no parent DOFs; the parent anchor
        velocity enters the row target instead (a compliant attachment to a driven hand,
        for example). Dynamic cross-articulation closures still warn and are ignored.
        """
        self.connect_count = 0
        self.connect_slot = None
        self.connect_joint_to_index: dict[int, int] = {}
        if not self.forest.has_loop_joints:
            self.connect_world_np = None
            return
        joint_type = model.joint_type.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_X_p = model.joint_X_p.numpy()
        joint_X_c = model.joint_X_c.numpy()
        joint_enabled_arr = getattr(model, "joint_enabled", None)
        joint_enabled = joint_enabled_arr.numpy() if joint_enabled_arr is not None else None
        articulation_world = self.forest.articulation_world
        loop_joint_articulation = self.forest.loop_joint_articulation
        body_articulation = (
            self.forest.body_to_articulation.numpy() if self.forest.body_to_articulation is not None else None
        )
        kinematic_bodies = (model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) != 0

        loops = np.flatnonzero(loop_joint_articulation >= 0)
        art = loop_joint_articulation[loops]
        ball = joint_type[loops] == int(JointType.BALL)
        parent = joint_parent[loops]
        has_parent = parent >= 0
        parent_art = np.full(loops.size, -1, dtype=np.int32)
        if body_articulation is not None:
            parent_art[has_parent] = body_articulation[parent[has_parent]]
        # A closure to another articulation is enforced only from a kinematic parent.
        cross = has_parent & (parent_art != art)
        dynamic_cross = np.zeros(loops.size, dtype=bool)
        dynamic_cross[cross] = ~kinematic_bodies[parent[cross]]
        for i in np.flatnonzero(~ball | dynamic_cross):
            j = loops[i]
            if not ball[i]:
                warnings.warn(
                    f"SolverFeatherPGS: loop joint {j} has unsupported type "
                    f"{int(joint_type[j])} (only BALL/connect closures are enforced); ignored.",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"SolverFeatherPGS: loop joint {j} closes across articulations ({int(parent_art[i])} vs "
                    f"{int(art[i])}) and its parent is not kinematic; ignored.",
                    stacklevel=2,
                )
        enforced = ball & ~dynamic_cross
        joints = loops[enforced]

        n = int(joints.size)
        if n == 0:
            self.connect_world_np = None
            self.connect_joint_to_index = {}
            return
        self.connect_joint_to_index = {int(j): index for index, j in enumerate(joints)}
        art = art[enforced]
        self.connect_world_np = articulation_world[art].astype(np.int32)
        self.connect_valid_np = (
            np.ones(n, dtype=np.int32) if joint_enabled is None else (joint_enabled[joints] != 0).astype(np.int32)
        )
        self.connect_parent_prescribed_np = (~has_parent | cross)[enforced].astype(np.int32)
        self.connect_enabled_np = self.connect_valid_np.copy()
        self.connect_anchor_p_np = joint_X_p[joints, :3].astype(np.float32)
        self.connect_anchor_c_np = joint_X_c[joints, :3].astype(np.float32)
        self._arrays.upload(
            CONNECT_PLAN,
            _connect_art=art.astype(np.int32),
            _connect_body_p=parent[enforced].astype(np.int32),
            _connect_body_c=joint_child[joints].astype(np.int32),
            _connect_anchor_p=self.connect_anchor_p_np,
            _connect_anchor_c=self.connect_anchor_c_np,
            _connect_parent_prescribed=self.connect_parent_prescribed_np,
            _connect_world=self.connect_world_np,
            _connect_valid=np.ones(n, dtype=np.int32),
            _connect_enabled=self.connect_enabled_np,
        )
        self._arrays.allocate(CONNECT_SLOTS, closure=n)
        self.connect_count = n

    def connect_index(self, joint: int) -> int:
        """The closure index of an enforced loop-closing BALL joint.

        Raises:
            ValueError: If ``joint`` is not one.
        """
        index = self.connect_joint_to_index.get(int(joint))
        if index is None:
            raise ValueError(
                f"SolverFeatherPGS: joint {joint} is not an enforced loop-closing BALL joint of this solver."
            )
        return index

    def set_loop_joint_enabled(self, joint: int, enabled: bool) -> None:
        """Enable or disable the connect rows of a loop-closing BALL joint (:meth:`.SolverFeatherPGS.set_loop_joint_enabled`)."""
        index = self.connect_index(joint)
        self.connect_enabled_np[index] = 1 if enabled else 0
        self._connect_enabled.assign(self.connect_enabled_np)

    def set_loop_joint_anchors(self, joint: int, parent_anchor, child_anchor) -> None:
        """Move the anchors of a loop-closing BALL joint (:meth:`.SolverFeatherPGS.set_loop_joint_anchors`)."""
        index = self.connect_index(joint)
        self.connect_anchor_p_np[index] = np.asarray(parent_anchor, dtype=np.float32)
        self.connect_anchor_c_np[index] = np.asarray(child_anchor, dtype=np.float32)
        self._connect_anchor_p.assign(self.connect_anchor_p_np)
        self._connect_anchor_c.assign(self.connect_anchor_c_np)

    def loop_joint_anchors(self, joint: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the current ``(parent_anchor, child_anchor)`` of a loop-closing BALL joint [m]."""
        index = self.connect_index(joint)
        return self.connect_anchor_p_np[index].copy(), self.connect_anchor_c_np[index].copy()

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
