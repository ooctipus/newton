# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Levenberg-Marquardt optimizer backend for inverse kinematics."""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import warp as wp

from ..enums import JointType
from ..model import Model
from .ik_common import IKJacobianType, IKSolveResult, compute_costs, eval_fk_batched, fk_accum, mean_cost
from .ik_objectives import IKObjective


@dataclass(slots=True)
class BatchCtx:
    joint_q: wp.array2d[wp.float32]
    residuals: wp.array2d[wp.float32]
    fk_body_q: wp.array2d[wp.transform]
    problem_idx: wp.array[wp.int32]

    # AUTODIFF and MIXED
    fk_body_qd: wp.array2d[wp.spatial_vector] | None = None
    dq_dof: wp.array2d[wp.float32] | None = None
    joint_q_proposed: wp.array2d[wp.float32] | None = None
    joint_qd: wp.array2d[wp.float32] | None = None

    # ANALYTIC and MIXED
    jacobian_out: wp.array3d[wp.float32] | None = None
    motion_subspace: wp.array2d[wp.spatial_vector] | None = None
    fk_X_local: wp.array2d[wp.transform] | None = None


def _tile_storage_bytes(num_values: int) -> int:
    """Return the 16-byte-aligned shared storage for float tile values."""
    return (num_values * 4 + 15) // 16 * 16


def _lm_tiled_solve_shared_memory_bytes(n_dofs: int, n_residuals: int) -> int:
    """Return exact shared memory used by the fused LM linear solve [byte]."""
    if n_dofs < 1 or n_residuals < 1:
        raise ValueError("LM dimensions must be positive")
    return (
        _tile_storage_bytes(n_residuals * n_dofs)
        + _tile_storage_bytes(n_residuals)
        + 2 * _tile_storage_bytes(n_dofs * n_dofs)
        + 5 * _tile_storage_bytes(n_dofs)
        + _tile_storage_bytes(1)
    )


def _lm_tiled_solve_fits(n_dofs: int, n_residuals: int, available_bytes: int | None) -> bool:
    """Return whether the fused LM solve fits the device shared-memory limit."""
    return available_bytes is None or _lm_tiled_solve_shared_memory_bytes(n_dofs, n_residuals) <= available_bytes


def _lm_global_workspace_bytes(n_batch: int, n_dofs: int, n_residuals: int, available_shared_bytes: int | None) -> int:
    """Return global workspace needed when the fused LM solve cannot launch [byte]."""
    if _lm_tiled_solve_fits(n_dofs, n_residuals, available_shared_bytes):
        return 0
    return 4 * (n_batch * (n_dofs * n_dofs + n_dofs) + n_dofs)


@wp.kernel
def _accept_reject(
    cost_curr: wp.array[wp.float32],
    cost_prop: wp.array[wp.float32],
    pred_red: wp.array[wp.float32],
    rho_min: float,
    accept: wp.array[wp.int32],
):
    row = wp.tid()
    rho = (cost_curr[row] - cost_prop[row]) / (pred_red[row] + 1.0e-8)
    accept[row] = wp.int32(1) if rho >= rho_min else wp.int32(0)


@wp.kernel
def _update_lm_state(
    joint_q_proposed: wp.array2d[wp.float32],
    residuals_proposed: wp.array2d[wp.float32],
    costs_proposed: wp.array[wp.float32],
    accept_flags: wp.array[wp.int32],
    n_coords: int,
    num_residuals: int,
    lambda_factor: float,
    lambda_min: float,
    lambda_max: float,
    joint_q_current: wp.array2d[wp.float32],
    residuals_current: wp.array2d[wp.float32],
    costs: wp.array[wp.float32],
    lambda_values: wp.array[wp.float32],
):
    row = wp.tid()

    if accept_flags[row] == 1:
        for i in range(n_coords):
            joint_q_current[row, i] = joint_q_proposed[row, i]
        for i in range(num_residuals):
            residuals_current[row, i] = residuals_proposed[row, i]
        costs[row] = costs_proposed[row]
        lambda_values[row] = lambda_values[row] / lambda_factor
    else:
        new_lambda = lambda_values[row] * lambda_factor
        lambda_values[row] = wp.clamp(new_lambda, lambda_min, lambda_max)


@wp.kernel
def _zero_fixed_dof_jacobian_columns(
    joint_dof_mask: wp.array[wp.bool],
    jacobian: wp.array3d[wp.float32],
):
    row, residual, dof = wp.tid()
    if not joint_dof_mask[dof]:
        jacobian[row, residual, dof] = 0.0


def _validate_joint_dof_mask(model: Model, joint_dof_mask: wp.array[wp.bool]) -> None:
    if joint_dof_mask.dtype != wp.bool:
        raise ValueError("joint_dof_mask must have dtype wp.bool")
    if joint_dof_mask.ndim != 1 or joint_dof_mask.shape[0] != model.joint_dof_count:
        raise ValueError("joint_dof_mask must have shape [joint_dof_count]")
    if joint_dof_mask.device != model.device:
        raise ValueError("joint_dof_mask must be on the model device")

    # The mask acts on twist-space DOFs, but the integrator couples a
    # quaternion-integrated joint's DOFs to its coordinates (rotation about the
    # joint origin translates the body), so a partial mask would not keep the
    # remaining coordinates fixed. Require all-or-nothing masks for such joints.
    mask = joint_dof_mask.numpy()
    joint_type = model.joint_type.numpy()
    qd_start = model.joint_qd_start.numpy()
    quaternion_joints = (JointType.BALL, JointType.FREE, JointType.DISTANCE)
    for j in range(len(joint_type)):
        if joint_type[j] not in quaternion_joints:
            continue
        joint_mask = mask[qd_start[j] : qd_start[j + 1]]
        if joint_mask.any() and not joint_mask.all():
            raise ValueError(
                f"joint_dof_mask partially masks joint {j} "
                f"({JointType(joint_type[j]).name}): quaternion-integrated joints "
                "must have all of their DOFs masked together"
            )


class IKOptimizerLM:
    """Levenberg-Marquardt optimizer for batched inverse kinematics.

    The optimizer solves a batch of independent IK problems that share a
    single articulation model and objective list. Jacobians can be evaluated
    with ``IKJacobianType.AUTODIFF``, ``IKJacobianType.ANALYTIC``, or
    ``IKJacobianType.MIXED``.

    Args:
        model: Shared articulation model.
        n_batch: Number of evaluation rows solved in parallel. This is
            typically ``n_problems * n_seeds`` after any sampling expansion.
        objectives: Ordered IK objectives applied to every batch row.
        lambda_initial: Initial LM damping factor for each batch row.
        jacobian_mode: Jacobian backend to use.
        lambda_factor: Factor used to increase or decrease the damping term
            after each trial step.
        lambda_min: Minimum allowed damping value.
        lambda_max: Maximum allowed damping value.
        rho_min: Minimum ratio of actual to predicted decrease required to
            accept a step.
        problem_idx: Optional mapping from batch rows to base problem indices
            for per-problem objective data.
        joint_dof_mask: Optional model-wide mask, shape ``[joint_dof_count]``,
            indexed in DOF (velocity) space per :attr:`Model.joint_qd_start` —
            a free joint has 6 entries. ``True`` entries are optimized;
            ``False`` entries receive an exactly-zero update. Quaternion-
            integrated joints (free/ball/distance) must be masked
            all-or-nothing, which the constructor enforces. The mask array must
            not be modified after construction.
    """

    TILE_N_DOFS = None
    TILE_N_RESIDUALS = None
    USE_TILED_SOLVE = None
    _cache: ClassVar[dict[tuple[int, int, str, bool], type]] = {}

    def __new__(
        cls,
        model: Model,
        n_batch: int,
        objectives: Sequence[IKObjective],
        *a: Any,
        **kw: Any,
    ) -> IKOptimizerLM:
        n_dofs = model.joint_dof_count
        n_residuals = sum(o.residual_dim() for o in objectives)
        arch = model.device.arch
        available_shared_bytes = model.device.max_shared_memory_per_block if model.device.is_cuda else None
        use_tiled_solve = _lm_tiled_solve_fits(n_dofs, n_residuals, available_shared_bytes)
        key = (n_dofs, n_residuals, arch, use_tiled_solve)

        spec_cls = cls._cache.get(key)
        if spec_cls is None:
            spec_cls = cls._build_specialized(key)
            cls._cache[key] = spec_cls

        return super().__new__(spec_cls)

    def __init__(
        self,
        model: Model,
        n_batch: int,
        objectives: Sequence[IKObjective],
        lambda_initial: float = 0.1,
        jacobian_mode: IKJacobianType = IKJacobianType.AUTODIFF,
        lambda_factor: float = 2.0,
        lambda_min: float = 1e-5,
        lambda_max: float = 1e10,
        rho_min: float = 1e-3,
        *,
        problem_idx: wp.array[wp.int32] | None = None,
        joint_dof_mask: wp.array[wp.bool] | None = None,
    ) -> None:
        self.model = model
        self.device = model.device
        self.n_batch = n_batch
        self.n_coords = model.joint_coord_count
        self.n_dofs = model.joint_dof_count
        self.n_residuals = sum(o.residual_dim() for o in objectives)

        self.objectives = objectives
        self.jacobian_mode = jacobian_mode
        self.has_analytic_objective = any(o.supports_analytic() for o in objectives)
        self.has_autodiff_objective = any(not o.supports_analytic() for o in objectives)

        self.lambda_initial = lambda_initial
        self.lambda_factor = lambda_factor
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.rho_min = rho_min
        if joint_dof_mask is not None:
            _validate_joint_dof_mask(model, joint_dof_mask)
        self.joint_dof_mask = joint_dof_mask

        if self.TILE_N_DOFS is not None:
            assert self.n_dofs == self.TILE_N_DOFS
        if self.TILE_N_RESIDUALS is not None:
            assert self.n_residuals == self.TILE_N_RESIDUALS
        if self.USE_TILED_SOLVE is None:
            raise RuntimeError("IKOptimizerLM must be instantiated through its specialized constructor")

        grad = jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED)

        self._alloc_solver_buffers(grad)
        self.problem_idx = problem_idx if problem_idx is not None else self.problem_idx_identity
        self.tape = wp.Tape() if grad else None

        self._build_residual_offsets()

        self._init_objectives()
        self._init_cuda_streams()

    def _init_objectives(self) -> None:
        """Allocate any per-objective buffers that must live on ``self.device``."""
        for obj, offset in zip(self.objectives, self.residual_offsets, strict=False):
            obj.set_batch_layout(self.n_residuals, offset, self.n_batch)
            obj.bind_device(self.device)
            if self.jacobian_mode == IKJacobianType.MIXED:
                mode = IKJacobianType.ANALYTIC if obj.supports_analytic() else IKJacobianType.AUTODIFF
            else:
                mode = self.jacobian_mode
            obj.init_buffers(model=self.model, jacobian_mode=mode)

    def _init_cuda_streams(self) -> None:
        """Allocate per-objective Warp streams and sync events."""
        self.objective_streams = []
        self.sync_events = []

        if self.device.is_cuda:
            for _ in range(len(self.objectives)):
                stream = wp.Stream(self.device)
                event = wp.Event(self.device)
                self.objective_streams.append(stream)
                self.sync_events.append(event)
        else:
            self.objective_streams = [None] * len(self.objectives)
            self.sync_events = [None] * len(self.objectives)

    def _parallel_for_objectives(self, fn: Callable[..., None], *extra: Any) -> None:
        """Run <fn(obj, offset, *extra)> across objectives on parallel CUDA streams."""
        if self.device.is_cuda:
            main = wp.get_stream(self.device)
            init_evt = main.record_event()
            for obj, offset, obj_stream, sync_event in zip(
                self.objectives, self.residual_offsets, self.objective_streams, self.sync_events, strict=False
            ):
                obj_stream.wait_event(init_evt)
                with wp.ScopedStream(obj_stream):
                    fn(obj, offset, *extra)
                obj_stream.record_event(sync_event)
            for sync_event in self.sync_events:
                main.wait_event(sync_event)
        else:
            for obj, offset in zip(self.objectives, self.residual_offsets, strict=False):
                fn(obj, offset, *extra)

    def _alloc_solver_buffers(self, grad: bool) -> None:
        device = self.device
        model = self.model

        self.qd_zero = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.float32, device=device)
        self.body_q = wp.zeros((self.n_batch, model.body_count), dtype=wp.transform, requires_grad=grad, device=device)
        self.body_qd = (
            wp.zeros((self.n_batch, model.body_count), dtype=wp.spatial_vector, device=device) if grad else None
        )

        self.residuals = wp.zeros((self.n_batch, self.n_residuals), dtype=wp.float32, requires_grad=grad, device=device)
        self.residuals_proposed = wp.zeros(
            (self.n_batch, self.n_residuals), dtype=wp.float32, requires_grad=grad, device=device
        )
        self.residuals_3d = wp.zeros((self.n_batch, self.n_residuals, 1), dtype=wp.float32, device=device)

        self.jacobian = wp.zeros((self.n_batch, self.n_residuals, self.n_dofs), dtype=wp.float32, device=device)
        self.dq_dof = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.float32, requires_grad=grad, device=device)

        self.joint_q_proposed = wp.zeros(
            (self.n_batch, self.n_coords), dtype=wp.float32, requires_grad=grad, device=device
        )

        self.costs = wp.zeros(self.n_batch, dtype=wp.float32, device=device)
        self.costs_proposed = wp.zeros(self.n_batch, dtype=wp.float32, device=device)
        self.lambda_values = wp.zeros(self.n_batch, dtype=wp.float32, device=device)
        self.accept_flags = wp.zeros(self.n_batch, dtype=wp.int32, device=device)
        self.pred_reduction = wp.zeros(self.n_batch, dtype=wp.float32, device=device)
        if self.USE_TILED_SOLVE:
            self._lm_normal_matrix = None
            self._lm_gradient = None
            self._lm_zero_diagonal = None
        else:
            self._lm_normal_matrix = wp.empty(self.n_batch * self.n_dofs * self.n_dofs, dtype=wp.float32, device=device)
            self._lm_gradient = wp.empty(self.n_batch * self.n_dofs, dtype=wp.float32, device=device)
            self._lm_zero_diagonal = wp.zeros(self.n_dofs, dtype=wp.float32, device=device)
        self._cost_sum = wp.zeros(1, dtype=wp.float32, device=device)
        self._cost_sum_host = (
            wp.zeros(1, dtype=wp.float32, device="cpu", pinned=True) if device.is_cuda else self._cost_sum
        )
        self._cost_sum_value = ctypes.c_float.from_address(self._cost_sum_host.ptr)

        self.problem_idx_identity = wp.array(np.arange(self.n_batch, dtype=np.int32), dtype=wp.int32, device=device)

        self.X_local = wp.zeros((self.n_batch, model.joint_count), dtype=wp.transform, device=device)
        self.joint_S_s = (
            wp.zeros((self.n_batch, self.n_dofs), dtype=wp.spatial_vector, device=device)
            if self.jacobian_mode != IKJacobianType.AUTODIFF and self.has_analytic_objective
            else None
        )

    def _build_residual_offsets(self) -> None:
        offsets: list[int] = []
        offset = 0
        for obj in self.objectives:
            offsets.append(offset)
            offset += obj.residual_dim()
        self.residual_offsets = offsets

    def _ctx_solver(
        self,
        joint_q: wp.array2d[wp.float32],
        *,
        residuals: wp.array2d[wp.float32] | None = None,
        jacobian: wp.array3d[wp.float32] | None = None,
    ) -> BatchCtx:
        batch = joint_q.shape[0]
        body_qd = getattr(self, "body_qd", None)
        motion_subspace = getattr(self, "joint_S_s", None)
        ctx = BatchCtx(
            joint_q=joint_q,
            residuals=residuals if residuals is not None else self.residuals[:batch],
            fk_body_q=self.body_q[:batch],
            problem_idx=self.problem_idx[:batch],
            fk_body_qd=None if body_qd is None else body_qd[:batch],
            dq_dof=self.dq_dof[:batch],
            joint_q_proposed=self.joint_q_proposed[:batch],
            joint_qd=self.qd_zero[:batch],
            jacobian_out=jacobian if jacobian is not None else self.jacobian[:batch],
            motion_subspace=None if motion_subspace is None else motion_subspace[:batch],
            fk_X_local=self.X_local[:batch],
        )
        self._validate_ctx_for_mode(ctx)
        return ctx

    def _validate_ctx_for_mode(self, ctx: BatchCtx) -> None:
        missing: list[str] = []

        for name in ("joint_q", "residuals", "fk_body_q", "problem_idx"):
            if getattr(ctx, name) is None:
                missing.append(name)

        mode = self.jacobian_mode
        if mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED):
            for name in ("fk_body_qd", "dq_dof", "joint_q_proposed", "joint_qd"):
                if getattr(ctx, name) is None:
                    missing.append(name)

        needs_analytic = mode == IKJacobianType.ANALYTIC or (
            mode == IKJacobianType.MIXED and self.has_analytic_objective
        )
        if needs_analytic:
            for name in ("jacobian_out", "motion_subspace"):
                if getattr(ctx, name) is None:
                    missing.append(name)
            if ctx.fk_X_local is None:
                missing.append("fk_X_local")

        if missing:
            raise RuntimeError(f"solver context missing: {', '.join(missing)}")

    def _for_objectives_residuals(self, ctx: BatchCtx) -> None:
        def _do(obj, offset, body_q_view, joint_q_view, model, output_residuals, problem_idx_array):
            obj.compute_residuals(
                body_q_view,
                joint_q_view,
                model,
                output_residuals,
                offset,
                problem_idx=problem_idx_array,
            )

        self._parallel_for_objectives(
            _do,
            ctx.fk_body_q,
            ctx.joint_q,
            self.model,
            ctx.residuals,
            ctx.problem_idx,
        )

    def _residuals_autodiff(self, ctx: BatchCtx) -> None:
        eval_fk_batched(
            self.model,
            ctx.joint_q,
            ctx.joint_qd,
            ctx.fk_body_q,
            ctx.fk_body_qd,
        )

        ctx.residuals.zero_()
        self._for_objectives_residuals(ctx)

    def _residuals_analytic(self, ctx: BatchCtx) -> None:
        self._fk_two_pass(
            self.model,
            ctx.joint_q,
            ctx.fk_body_q,
            ctx.fk_X_local,
            ctx.joint_q.shape[0],
        )

        ctx.residuals.zero_()
        self._for_objectives_residuals(ctx)

    def _jacobian_at(self, ctx: BatchCtx) -> wp.array3d[wp.float32]:
        mode = self.jacobian_mode

        if mode == IKJacobianType.AUTODIFF:
            self._jacobian_autodiff(ctx)
            self._apply_joint_dof_mask(ctx.jacobian_out)
            return ctx.jacobian_out

        if mode == IKJacobianType.ANALYTIC:
            self._jacobian_analytic(ctx, accumulate=False)
            self._apply_joint_dof_mask(ctx.jacobian_out)
            return ctx.jacobian_out

        # MIXED mode
        if self.has_autodiff_objective:
            self._jacobian_autodiff(ctx)
        else:
            ctx.jacobian_out.zero_()

        if self.has_analytic_objective:
            self._jacobian_analytic(ctx, accumulate=self.has_autodiff_objective)

        self._apply_joint_dof_mask(ctx.jacobian_out)
        return ctx.jacobian_out

    def _apply_joint_dof_mask(self, jacobian: wp.array3d[wp.float32]) -> None:
        if self.joint_dof_mask is None:
            return
        wp.launch(
            _zero_fixed_dof_jacobian_columns,
            dim=(jacobian.shape[0], self.n_residuals, self.n_dofs),
            inputs=[self.joint_dof_mask],
            outputs=[jacobian],
            device=self.device,
        )

    def _jacobian_autodiff(self, ctx: BatchCtx) -> None:
        if self.tape is None:
            raise RuntimeError("Autodiff Jacobian requested but tape is not initialized")

        ctx.jacobian_out.zero_()
        self.tape.reset()
        self.tape.gradients = {}
        ctx.dq_dof.zero_()

        with self.tape:
            self._integrate_dq(
                ctx.joint_q,
                dq_in=ctx.dq_dof,
                joint_q_out=ctx.joint_q_proposed,
                joint_qd_out=ctx.joint_qd,
            )

            res_ctx = BatchCtx(
                joint_q=ctx.joint_q_proposed,
                residuals=ctx.residuals,
                fk_body_q=ctx.fk_body_q,
                problem_idx=ctx.problem_idx,
                fk_body_qd=ctx.fk_body_qd,
                joint_qd=ctx.joint_qd,
            )
            self._residuals_autodiff(res_ctx)
            residuals_flat = ctx.residuals.flatten()

        self.tape.outputs = [residuals_flat]

        for obj, offset in zip(self.objectives, self.residual_offsets, strict=False):
            if self.jacobian_mode == IKJacobianType.MIXED and obj.supports_analytic():
                continue
            obj.compute_jacobian_autodiff(self.tape, self.model, ctx.jacobian_out, offset, ctx.dq_dof)
            self.tape.zero()

    def _jacobian_analytic(self, ctx: BatchCtx, *, accumulate: bool) -> None:
        if not accumulate:
            ctx.jacobian_out.zero_()

        self._compute_motion_subspace(
            joint_q_in=ctx.joint_q,
            body_q=ctx.fk_body_q,
            joint_S_s_out=ctx.motion_subspace,
        )

        def _emit(obj, off, body_q_view, joint_q_view, model, jac_view, motion_subspace_view):
            if obj.supports_analytic():
                obj.compute_jacobian_analytic(body_q_view, joint_q_view, model, jac_view, motion_subspace_view, off)
            elif not accumulate:
                raise ValueError(f"Objective {type(obj).__name__} does not support analytic Jacobian")

        self._parallel_for_objectives(
            _emit,
            ctx.fk_body_q,
            ctx.joint_q,
            self.model,
            ctx.jacobian_out,
            ctx.motion_subspace,
        )

    def _mean_cost(self, active_batch_count: int) -> float:
        """Return the normalized cost over the valid leading rows."""
        return mean_cost(
            self.costs,
            self.n_residuals,
            active_batch_count,
            self._cost_sum,
            self._cost_sum_host,
            self._cost_sum_value,
        )

    def solve(
        self,
        joint_q_in: wp.array2d[wp.float32],
        joint_q_out: wp.array2d[wp.float32],
        max_iterations: int = 10,
        step_size: float = 1.0,
        *,
        active_batch_count: int | None = None,
        convergence_tolerance: float | None = 1.0e-6,
        convergence_check_interval: int = 1,
        projection: Callable[[wp.array2d[wp.float32]], None] | None = None,
        projection_interval: int = 1,
    ) -> IKSolveResult:
        """Run one continuous LM solve with optional convergence and projection.

        Args:
            joint_q_in: Input joint coordinates [m or rad], shape
                [n_batch, joint_coord_count].
            joint_q_out: Optimized joint coordinates [m or rad], shape
                [n_batch, joint_coord_count]. It may alias ``joint_q_in``.
            max_iterations: Maximum number of LM iterations.
            step_size: Unitless scale applied to every LM update.
            active_batch_count: Valid leading batch rows included in cost
                reporting and convergence. Padded rows are not executed.
            convergence_tolerance: Absolute mean-cost change that terminates
                the solve. Set to ``None`` to run exactly ``max_iterations``.
            convergence_check_interval: Iterations between convergence checks.
            projection: Optional in-place projection of ``joint_q_out`` called
                without resetting damping or sampling state.
            projection_interval: Iterations between projection calls.

        Returns:
            Immutable solve summary.
        """
        if joint_q_in.shape != (self.n_batch, self.n_coords):
            raise ValueError("joint_q_in has incompatible shape")
        if joint_q_out.shape != (self.n_batch, self.n_coords):
            raise ValueError("joint_q_out has incompatible shape")
        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        active_batch_count = self.n_batch if active_batch_count is None else active_batch_count
        if active_batch_count < 1 or active_batch_count > self.n_batch:
            raise ValueError("active_batch_count must be within the optimizer batch")
        if convergence_tolerance is not None and convergence_tolerance < 0.0:
            raise ValueError("convergence_tolerance must be non-negative or None")
        if convergence_check_interval < 1:
            raise ValueError("convergence_check_interval must be positive")
        if projection_interval < 1:
            raise ValueError("projection_interval must be positive")

        joint_q_in_active = joint_q_in[:active_batch_count]
        joint_q = joint_q_out[:active_batch_count]
        if joint_q_in.ptr != joint_q_out.ptr:
            wp.copy(joint_q, joint_q_in_active)

        self.lambda_values[:active_batch_count].fill_(self.lambda_initial)
        self.compute_costs(joint_q)
        initial_mean_cost = self._mean_cost(active_batch_count)
        previous_mean_cost = initial_mean_cost
        final_mean_cost = initial_mean_cost
        converged = False
        iterations = 0

        for iteration in range(max_iterations):
            self._step(joint_q, step_size=step_size)
            iterations = iteration + 1

            if projection is not None and iterations % projection_interval == 0:
                projection(joint_q)
                self.compute_costs(joint_q)

            check_convergence = convergence_tolerance is not None and (
                iterations % convergence_check_interval == 0 or iterations == max_iterations
            )
            if check_convergence:
                final_mean_cost = self._mean_cost(active_batch_count)
                if abs(previous_mean_cost - final_mean_cost) <= convergence_tolerance:
                    converged = True
                    break
                previous_mean_cost = final_mean_cost

        if convergence_tolerance is None or iterations % convergence_check_interval != 0:
            final_mean_cost = self._mean_cost(active_batch_count)

        return IKSolveResult(
            iterations=iterations,
            converged=converged,
            initial_mean_cost=initial_mean_cost,
            final_mean_cost=final_mean_cost,
        )

    def step(
        self,
        joint_q_in: wp.array2d[wp.float32],
        joint_q_out: wp.array2d[wp.float32],
        iterations: int = 10,
        step_size: float = 1.0,
    ) -> None:
        """Run a fixed number of LM iterations on joint configurations.

        Args:
            joint_q_in: Input joint coordinates [m or rad], shape
                [n_batch, joint_coord_count].
            joint_q_out: Optimized joint coordinates [m or rad], shape
                [n_batch, joint_coord_count]. It may alias ``joint_q_in``.
            iterations: Number of LM iterations.
            step_size: Unitless scale applied to every LM update.
        """
        if joint_q_in.shape != (self.n_batch, self.n_coords):
            raise ValueError("joint_q_in has incompatible shape")
        if joint_q_out.shape != (self.n_batch, self.n_coords):
            raise ValueError("joint_q_out has incompatible shape")
        if iterations < 0:
            raise ValueError("iterations must be non-negative")
        if joint_q_in.ptr != joint_q_out.ptr:
            wp.copy(joint_q_out, joint_q_in)

        self.lambda_values.fill_(self.lambda_initial)
        self.compute_costs(joint_q_out)
        for _ in range(iterations):
            self._step(joint_q_out, step_size=step_size)

    def _compute_residuals(
        self,
        joint_q: wp.array2d[wp.float32],
        output_residuals: wp.array2d[wp.float32] | None = None,
    ) -> wp.array2d[wp.float32]:
        buffer = output_residuals or self.residuals
        ctx = self._ctx_solver(joint_q, residuals=buffer)

        if self.jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED):
            self._residuals_autodiff(ctx)
        else:
            self._residuals_analytic(ctx)

        return ctx.residuals

    def _compute_motion_subspace(
        self,
        *,
        joint_q_in: wp.array2d[wp.float32],
        body_q: wp.array2d[wp.transform],
        joint_S_s_out: wp.array2d[wp.spatial_vector],
    ) -> None:
        n_joints = self.model.joint_count
        batch = body_q.shape[0]
        wp.launch(
            self._compute_motion_subspace_2d,
            dim=[batch, n_joints],
            inputs=[
                self.model.joint_type,
                self.model.joint_parent,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                joint_q_in,
                self.model.joint_axis,
                self.model.joint_dof_dim,
                body_q,
                self.model.joint_X_p,
            ],
            outputs=[
                joint_S_s_out,
            ],
            device=self.device,
        )

    def _integrate_dq(
        self,
        joint_q: wp.array2d[wp.float32],
        *,
        dq_in: wp.array2d[wp.float32],
        joint_q_out: wp.array2d[wp.float32],
        joint_qd_out: wp.array2d[wp.float32],
        step_size: float = 1.0,
    ) -> None:
        batch = joint_q.shape[0]

        wp.launch(
            self._integrate_dq_dof,
            dim=[batch, self.model.joint_count],
            inputs=[
                self.model.joint_type,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_dof_dim,
                self.model.joint_X_c,
                self.model.body_com,
                joint_q,
                dq_in,
                joint_qd_out,
                step_size,
            ],
            outputs=[
                joint_q_out,
                joint_qd_out,
            ],
            device=self.device,
        )
        joint_qd_out.zero_()

    def _step(
        self,
        joint_q: wp.array2d[wp.float32],
        step_size: float = 1.0,
    ) -> None:
        """Execute one Levenberg-Marquardt iteration with adaptive damping."""

        batch = joint_q.shape[0]
        ctx_curr = self._ctx_solver(joint_q)
        self._jacobian_at(ctx_curr)

        residuals_3d = self.residuals_3d[:batch]
        wp.copy(residuals_3d.flatten(), ctx_curr.residuals.flatten())

        ctx_curr.dq_dof.zero_()
        self._solve_normal_equations(
            ctx_curr.jacobian_out,
            residuals_3d,
            self.lambda_values[:batch],
            ctx_curr.dq_dof,
            self.pred_reduction[:batch],
        )

        self._integrate_dq(
            joint_q,
            dq_in=ctx_curr.dq_dof,
            joint_q_out=ctx_curr.joint_q_proposed,
            joint_qd_out=ctx_curr.joint_qd,
            step_size=step_size,
        )

        ctx_prop = self._ctx_solver(ctx_curr.joint_q_proposed, residuals=self.residuals_proposed[:batch])
        if self.jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED):
            self._residuals_autodiff(ctx_prop)
        else:
            self._residuals_analytic(ctx_prop)

        wp.launch(
            compute_costs,
            dim=batch,
            inputs=[ctx_prop.residuals, self.n_residuals],
            outputs=[self.costs_proposed[:batch]],
            device=self.device,
        )

        wp.launch(
            _accept_reject,
            dim=batch,
            inputs=[
                self.costs[:batch],
                self.costs_proposed[:batch],
                self.pred_reduction[:batch],
                self.rho_min,
            ],
            outputs=[self.accept_flags[:batch]],
            device=self.device,
        )

        wp.launch(
            _update_lm_state,
            dim=batch,
            inputs=[
                ctx_curr.joint_q_proposed,
                ctx_prop.residuals,
                self.costs_proposed[:batch],
                self.accept_flags[:batch],
                self.n_coords,
                self.n_residuals,
                self.lambda_factor,
                self.lambda_min,
                self.lambda_max,
            ],
            outputs=[joint_q, ctx_curr.residuals, self.costs[:batch], self.lambda_values[:batch]],
            device=self.device,
        )

    def reset(self) -> None:
        """Clear LM damping and accept/reject state before a new solve."""
        self.lambda_values.zero_()
        self.accept_flags.zero_()

    def compute_costs(self, joint_q: wp.array2d[wp.float32]) -> wp.array[wp.float32]:
        """Evaluate squared residual costs for a batch of joint configurations.

        Args:
            joint_q: Joint coordinates to evaluate, shape [n_batch, joint_coord_count].

        Returns:
            Costs for each batch row, shape [n_batch].
        """
        batch = joint_q.shape[0]
        if batch < 1 or batch > self.n_batch:
            raise ValueError("joint_q batch must be within the optimizer capacity")
        residuals = self._compute_residuals(joint_q, self.residuals[:batch])
        costs = self.costs[:batch]
        wp.launch(
            compute_costs,
            dim=batch,
            inputs=[residuals, self.n_residuals],
            outputs=[costs],
            device=self.device,
        )
        return costs

    def _solve_normal_equations(
        self,
        jacobian: wp.array3d[wp.float32],
        residuals: wp.array3d[wp.float32],
        lambda_values: wp.array[wp.float32],
        dq_dof: wp.array2d[wp.float32],
        pred_reduction: wp.array[wp.float32],
    ) -> None:
        raise NotImplementedError("This method should be overridden by specialized solver")

    @classmethod
    def _build_specialized(cls, key: tuple[int, int, str, bool]) -> type[IKOptimizerLM]:
        """Build a specialized LM optimizer with a device-compatible linear solve."""
        C, R, _, use_tiled_solve = key

        def _template(
            jacobians: wp.array3d[wp.float32],  # (n_batch, n_residuals, n_dofs)
            residuals: wp.array3d[wp.float32],  # (n_batch, n_residuals, 1)
            lambda_values: wp.array[wp.float32],  # (n_batch)
            # outputs
            dq_dof: wp.array2d[wp.float32],  # (n_batch, n_dofs)
            pred_reduction_out: wp.array[wp.float32],  # (n_batch)
        ):
            row = wp.tid()

            RES = _Specialized.TILE_N_RESIDUALS
            DOF = _Specialized.TILE_N_DOFS
            J = wp.tile_load(jacobians[row], shape=(RES, DOF))
            r = wp.tile_load(residuals[row], shape=(RES, 1))
            lam = lambda_values[row]

            Jt = wp.tile_transpose(J)
            JtJ = wp.tile_zeros(shape=(DOF, DOF), dtype=wp.float32)
            wp.tile_matmul(Jt, J, JtJ)

            diag = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
            for i in range(DOF):
                diag[i] = lam
            A = wp.tile_diag_add(JtJ, diag)
            g = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
            tmp2d = wp.tile_zeros(shape=(DOF, 1), dtype=wp.float32)
            wp.tile_matmul(Jt, r, tmp2d)
            for i in range(DOF):
                g[i] = tmp2d[i, 0]

            delta = wp.tile_map(wp.neg, g)
            wp.tile_cholesky_inplace(A)
            wp.tile_cholesky_solve_inplace(A, delta)
            wp.tile_store(dq_dof[row], delta)
            lambda_delta = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
            for i in range(DOF):
                lambda_delta[i] = lam * delta[i]

            diff = wp.tile_map(wp.sub, lambda_delta, g)
            prod = wp.tile_map(wp.mul, delta, diff)
            red = wp.tile_sum(prod)[0]
            pred_reduction_out[row] = 0.5 * red

        _template.__name__ = f"_lm_solve_tiled_{C}_{R}"
        _template.__qualname__ = f"_lm_solve_tiled_{C}_{R}"
        _lm_solve_tiled = wp.kernel(enable_backward=False, module="unique")(_template)

        # late-import jcalc_* helpers to avoid circular import error
        from ...sim.articulation import (  # noqa: PLC0415
            jcalc_motion_subspace,
            write_free_distance_motion_subspace,
        )
        from ...solvers.featherstone.kernels import (  # noqa: PLC0415
            dense_cholesky,
            dense_subs,
            jcalc_integrate,
            jcalc_transform,
        )

        @wp.kernel(module="unique")
        def _lm_normal_equations_global(
            jacobians: wp.array3d[wp.float32],
            residuals: wp.array3d[wp.float32],
            lambda_values: wp.array[wp.float32],
            n_residuals: int,
            n_dofs: int,
            normal_matrix: wp.array[wp.float32],
            gradient: wp.array[wp.float32],
            delta: wp.array2d[wp.float32],
        ):
            row, i, j = wp.tid()
            if j <= i:
                value = float(0.0)
                for residual in range(n_residuals):
                    value += jacobians[row, residual, i] * jacobians[row, residual, j]
                if i == j:
                    value += lambda_values[row]
                normal_matrix[row * n_dofs * n_dofs + i * n_dofs + j] = value
            if j == 0:
                value = float(0.0)
                for residual in range(n_residuals):
                    value += jacobians[row, residual, i] * residuals[row, residual, 0]
                gradient[row * n_dofs + i] = value
                delta[row, i] = -value

        @wp.kernel(module="unique")
        def _lm_cholesky_solve_global(
            normal_matrix: wp.array[wp.float32],
            gradient: wp.array[wp.float32],
            zero_diagonal: wp.array[wp.float32],
            lambda_values: wp.array[wp.float32],
            n_dofs: int,
            delta: wp.array[wp.float32],
            pred_reduction_out: wp.array[wp.float32],
        ):
            row = wp.tid()
            matrix_start = row * n_dofs * n_dofs
            vector_start = row * n_dofs
            dense_cholesky(n_dofs, normal_matrix, zero_diagonal, matrix_start, 0, normal_matrix)
            dense_subs(n_dofs, matrix_start, vector_start, normal_matrix, delta, delta)
            predicted_reduction = float(0.0)
            for i in range(n_dofs):
                value = delta[vector_start + i]
                predicted_reduction += value * (lambda_values[row] * value - gradient[vector_start + i])
            pred_reduction_out[row] = 0.5 * predicted_reduction

        @wp.kernel
        def _integrate_dq_dof(
            # model-wide
            joint_type: wp.array[wp.int32],  # (n_joints)
            joint_parent: wp.array[wp.int32],  # (n_joints)
            joint_child: wp.array[wp.int32],  # (n_joints)
            joint_q_start: wp.array[wp.int32],  # (n_joints + 1)
            joint_qd_start: wp.array[wp.int32],  # (n_joints + 1)
            joint_dof_dim: wp.array2d[wp.int32],  # (n_joints, 2)  → (lin, ang)
            joint_X_c: wp.array[wp.transform],  # (n_joints)
            body_com: wp.array[wp.vec3],  # (n_bodies)
            # per-row
            joint_q_curr: wp.array2d[wp.float32],  # (n_batch, n_coords)
            joint_qd_curr: wp.array2d[wp.float32],  # (n_batch, n_dofs)  (typically all-zero)
            dq_dof: wp.array2d[wp.float32],  # (n_batch, n_dofs)  ← update direction (q̇)
            dt: float,  # step scale (usually 1.0)
            # outputs
            joint_q_out: wp.array2d[wp.float32],  # (n_batch, n_coords)
            joint_qd_out: wp.array2d[wp.float32],  # (n_batch, n_dofs)
        ):
            """
            Integrate the candidate update ``dq_dof`` (interpreted as a
            joint-space velocity times ``dt``) into a new configuration.

            q_out  = integrate(q_curr, dq_dof)

            One thread handles one joint of one batch row. All joint types
            supported by ``jcalc_integrate`` (revolute, prismatic, ball,
            free, D6, ...) work out of the box.
            """
            row, joint_idx = wp.tid()

            # Static joint metadata
            t = joint_type[joint_idx]
            parent = joint_parent[joint_idx]
            child = joint_child[joint_idx]
            coord_start = joint_q_start[joint_idx]
            dof_start = joint_qd_start[joint_idx]
            lin_axes = joint_dof_dim[joint_idx, 0]
            ang_axes = joint_dof_dim[joint_idx, 1]

            # Views into the current batch row
            q_row = joint_q_curr[row]
            qd_row = joint_qd_curr[row]  # typically zero
            delta_row = dq_dof[row]  # update vector

            q_out_row = joint_q_out[row]
            qd_out_row = joint_qd_out[row]

            # Treat `delta_row` as acceleration with dt=1:
            #   qd_new = 0 + delta           (qd ← delta)
            #   q_new  = q + qd_new * dt     (q ← q + delta)
            jcalc_integrate(
                parent,
                joint_X_c[joint_idx],
                body_com[child],
                t,
                q_row,
                qd_row,
                delta_row,  # passed as joint_qdd
                coord_start,
                dof_start,
                lin_axes,
                ang_axes,
                dt,
                q_out_row,
                qd_out_row,
            )

        @wp.kernel(module="unique")
        def _compute_motion_subspace_2d(
            joint_type: wp.array[wp.int32],  # (n_joints)
            joint_parent: wp.array[wp.int32],  # (n_joints)
            joint_q_start: wp.array[wp.int32],  # (n_joints + 1)
            joint_qd_start: wp.array[wp.int32],  # (n_joints + 1)
            joint_q: wp.array2d[wp.float32],  # (n_batch, n_coords)
            joint_axis: wp.array[wp.vec3],  # (n_joint_dof_count)
            joint_dof_dim: wp.array2d[wp.int32],  # (n_joints, 2)
            body_q: wp.array2d[wp.transform],  # (n_batch, n_bodies)
            joint_X_p: wp.array[wp.transform],  # (n_joints)
            # outputs
            joint_S_s: wp.array2d[wp.spatial_vector],  # (n_batch, n_joint_dof_count)
        ):
            row, joint_idx = wp.tid()

            type = joint_type[joint_idx]
            parent = joint_parent[joint_idx]
            q_start = joint_q_start[joint_idx]
            qd_start = joint_qd_start[joint_idx]

            X_pj = joint_X_p[joint_idx]
            X_wpj = X_pj
            if parent >= 0:
                X_wpj = body_q[row, parent] * X_pj

            lin_axis_count = joint_dof_dim[joint_idx, 0]
            ang_axis_count = joint_dof_dim[joint_idx, 1]

            joint_q_1d = joint_q[row]
            S_s_out = joint_S_s[row]

            if type == JointType.FREE or type == JointType.DISTANCE:
                # IK integration applies angular increments about the fixed parent anchor.
                write_free_distance_motion_subspace(
                    X_wpj,
                    wp.transform_get_translation(X_wpj),
                    qd_start,
                    S_s_out,
                )
            else:
                jcalc_motion_subspace(
                    type,
                    joint_axis,
                    joint_q_1d,
                    lin_axis_count,
                    ang_axis_count,
                    X_wpj,
                    wp.transform_identity(),
                    wp.vec3(),
                    q_start,
                    qd_start,
                    S_s_out,
                )

        @wp.kernel(module="unique")
        def _fk_local(
            joint_type: wp.array[wp.int32],  # (n_joints)
            joint_q: wp.array2d[wp.float32],  # (n_batch, n_coords)
            joint_q_start: wp.array[wp.int32],  # (n_joints + 1)
            joint_qd_start: wp.array[wp.int32],  # (n_joints + 1)
            joint_axis: wp.array[wp.vec3],  # (n_axes)
            joint_dof_dim: wp.array2d[wp.int32],  # (n_joints, 2)  → (lin, ang)
            joint_X_p: wp.array[wp.transform],  # (n_joints)
            joint_X_c: wp.array[wp.transform],  # (n_joints)
            # outputs
            X_local_out: wp.array2d[wp.transform],  # (n_batch, n_joints)
        ):
            row, local_joint_idx = wp.tid()

            t = joint_type[local_joint_idx]
            q_start = joint_q_start[local_joint_idx]
            axis_start = joint_qd_start[local_joint_idx]
            lin_axes = joint_dof_dim[local_joint_idx, 0]
            ang_axes = joint_dof_dim[local_joint_idx, 1]

            X_j = jcalc_transform(
                t,
                joint_axis,
                axis_start,
                lin_axes,
                ang_axes,
                joint_q[row],  # 1-D row slice
                q_start,
            )

            X_rel = joint_X_p[local_joint_idx] * X_j * wp.transform_inverse(joint_X_c[local_joint_idx])
            X_local_out[row, local_joint_idx] = X_rel

        def _fk_two_pass(model, joint_q, body_q, X_local, n_batch):
            """Compute forward kinematics using two-pass algorithm.

            Args:
                model: newton.Model instance
                joint_q: 2D array [n_batch, joint_coord_count]
                body_q: 2D array [n_batch, body_count] (output)
                X_local: 2D array [n_batch, joint_count] (workspace)
                n_batch: Number of rows to process
            """
            wp.launch(
                _fk_local,
                dim=[n_batch, model.joint_count],
                inputs=[
                    model.joint_type,
                    joint_q,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_axis,
                    model.joint_dof_dim,
                    model.joint_X_p,
                    model.joint_X_c,
                ],
                outputs=[
                    X_local,
                ],
                device=model.device,
            )

            wp.launch(
                fk_accum,
                dim=[n_batch, model.joint_count],
                inputs=[
                    model.joint_parent,
                    X_local,
                ],
                outputs=[
                    body_q,
                ],
                device=model.device,
            )

        class _Specialized(IKOptimizerLM):
            TILE_N_DOFS = wp.constant(C)
            TILE_N_RESIDUALS = wp.constant(R)
            TILE_THREADS = wp.constant(32)
            USE_TILED_SOLVE = use_tiled_solve

            def _solve_normal_equations(
                self,
                jac: wp.array3d[wp.float32],
                res: wp.array3d[wp.float32],
                lam: wp.array[wp.float32],
                dq: wp.array2d[wp.float32],
                pred: wp.array[wp.float32],
            ) -> None:
                if self.USE_TILED_SOLVE:
                    wp.launch_tiled(
                        _lm_solve_tiled,
                        dim=[jac.shape[0]],
                        inputs=[jac, res, lam, dq, pred],
                        block_dim=self.TILE_THREADS,
                        device=self.device,
                    )
                    return
                wp.launch(
                    _lm_normal_equations_global,
                    dim=[jac.shape[0], self.TILE_N_DOFS, self.TILE_N_DOFS],
                    inputs=[
                        jac,
                        res,
                        lam,
                        self.TILE_N_RESIDUALS,
                        self.TILE_N_DOFS,
                        self._lm_normal_matrix,
                        self._lm_gradient,
                        dq,
                    ],
                    device=self.device,
                )
                wp.launch(
                    _lm_cholesky_solve_global,
                    dim=jac.shape[0],
                    inputs=[
                        self._lm_normal_matrix,
                        self._lm_gradient,
                        self._lm_zero_diagonal,
                        lam,
                        self.TILE_N_DOFS,
                        dq.flatten(),
                        pred,
                    ],
                    device=self.device,
                )

        _Specialized.__name__ = f"IK_{C}x{R}"
        _Specialized._integrate_dq_dof = staticmethod(_integrate_dq_dof)
        _Specialized._compute_motion_subspace_2d = staticmethod(_compute_motion_subspace_2d)
        _Specialized._fk_two_pass = staticmethod(_fk_two_pass)
        return _Specialized
