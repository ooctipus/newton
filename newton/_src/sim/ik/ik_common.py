# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Common enums and utility kernels shared across IK components."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import Enum

import warp as wp


@dataclass(frozen=True, slots=True)
class IKMemoryEstimate:
    """Estimated persistent Warp-array memory required by one IK solver [byte].

    The estimate is exact for solver-owned Warp arrays and objective arrays
    declared through :meth:`~newton.ik.IKObjective.estimate_memory`. It excludes
    caller-live arrays, CUDA streams/events, allocator granularity, JIT modules,
    and transient kernel or autodiff runtime storage. Callers selecting a batch
    from currently free memory must retain a runtime safety reserve.

    Attributes:
        frontend_bytes: Sampling and selection buffers owned by
            :class:`~newton.ik.IKSolver` [byte].
        optimizer_bytes: Persistent optimizer buffers [byte].
        objective_bytes: Objective target and workspace buffers [byte].
    """

    frontend_bytes: int
    optimizer_bytes: int
    objective_bytes: int

    @property
    def total_bytes(self) -> int:
        """Total estimated persistent device memory [byte]."""
        return self.frontend_bytes + self.optimizer_bytes + self.objective_bytes


@dataclass(frozen=True, slots=True)
class IKSolveResult:
    """Summary of one continuous inverse-kinematics solve.

    Attributes:
        iterations: Number of completed optimizer iterations.
        converged: Whether the declared convergence tolerance was reached.
        initial_mean_cost: Mean weighted squared residual before
            optimization, ``sum(costs) / (batch * residual_count)``.
        final_mean_cost: Mean weighted squared residual after optimization.
    """

    iterations: int
    converged: bool
    initial_mean_cost: float
    final_mean_cost: float


class IKJacobianType(Enum):
    """
    Specifies the backend used for Jacobian computation in inverse kinematics.
    """

    AUTODIFF = "autodiff"
    """Use Warp's reverse-mode autodiff for every objective."""

    ANALYTIC = "analytic"
    """Use analytic Jacobians for objectives that support them."""

    MIXED = "mixed"
    """Use analytic Jacobians where available, otherwise use autodiff."""


def mean_cost(
    costs: wp.array[wp.float32],
    residual_count: int,
    active_count: int,
    output: wp.array[wp.float32],
    output_host: wp.array[wp.float32],
    output_value: ctypes.c_float,
) -> float:
    """Return the mean weighted squared residual with preallocated scalars."""
    if residual_count < 1:
        raise ValueError("residual_count must be positive")
    if active_count < 1 or active_count > len(costs):
        raise ValueError("active_count must be within the cost array")
    wp.utils.array_sum(costs, out=output, value_count=active_count)
    if output.ptr != output_host.ptr:
        wp.copy(output_host, output)
        wp.synchronize_stream(costs.device)
    return output_value.value / (active_count * residual_count)


@wp.kernel
def fk_accum(
    joint_parent: wp.array[wp.int32],
    X_local: wp.array2d[wp.transform],
    body_q: wp.array2d[wp.transform],
):
    problem_idx, local_joint_idx = wp.tid()
    Xw = X_local[problem_idx, local_joint_idx]
    parent = joint_parent[local_joint_idx]
    while parent >= 0:
        Xw = X_local[problem_idx, parent] * Xw
        parent = joint_parent[parent]
    body_q[problem_idx, local_joint_idx] = Xw


@wp.kernel
def compute_costs(
    residuals: wp.array2d[wp.float32],
    num_residuals: int,
    costs: wp.array[wp.float32],
):
    problem_idx = wp.tid()
    cost = float(0.0)
    for i in range(num_residuals):
        r = residuals[problem_idx, i]
        cost += r * r
    costs[problem_idx] = cost
