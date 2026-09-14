# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Offset-eight and original MF partition descriptors; no canonical factors."""

import warp as wp


@wp.struct
class KineticSolveData:
    selector: wp.array[int]
    v_out: wp.array[float]
    status: wp.array[int]
    iterations: int
    omega: float
    friction_start_iteration: int
    iteration_offset: int


@wp.struct
class KineticMFData:
    meta: wp.array2d[int]
    J_a: wp.array3d[float]
    J_b: wp.array3d[float]
    MiJt_a: wp.array3d[float]
    MiJt_b: wp.array3d[float]
    mu: wp.array2d[float]
