# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The contact-unit law's CUDA prelude in a kernel of its own module.

Its native body is CUDA-only (``device_preludes.cuda_native``), so a CPU build of
it fails. In a module of its own it is built only where a CUDA test launches it,
and the CPU kernels of ``test_feather_pgs_contact_unit`` build without it.
"""

import warp as wp

from newton._src.solvers.feather_pgs import contact_unit
from newton._src.solvers.feather_pgs.device_preludes import cuda_native

_NATIVE_BODY = r"""
    const float* p = problems.data + i * 16;
    const ContactUnitSolution s = ContactUnit::solve(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10],
                                                     p[11], static_cast<int>(p[12]), p[13], p[14], p[15]);
    float* o = out.data + i * 5;
    o[0] = s.normal;
    o[1] = s.friction0;
    o[2] = s.friction1;
    o[3] = s.alpha;
    o[4] = static_cast<float>(s.evaluations);
#endif
"""


@wp.func_native(cuda_native(_NATIVE_BODY, preludes=(contact_unit.CONTACT_UNIT,)))
def _native_solve(problems: wp.array2d[float], out: wp.array2d[float], i: int): ...


@wp.kernel
def solve_points_native(problems: wp.array2d[float], out: wp.array2d[float]):
    _native_solve(problems, out, wp.tid())
