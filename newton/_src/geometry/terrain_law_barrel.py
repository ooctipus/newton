# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The terrain law's kernels for the cylinder class when one of its partners is a barrel cylinder.

They are :mod:`~newton._src.geometry.terrain_law_cylinder`'s kernels with each core's barrel radius read (a barrel
cylinder's side radius, ``scale[2]``), so that the core functions' barrel paths run: Newton's barrel cylinder, caps of
radius ``r`` and a side of revolution of a circular arc through their rims. The cylinder module's own kernels set the
radius to a literal 0 and compile those paths out; the shadow takes these for a class whose table holds a barrel
(:attr:`~newton._src.geometry.terrain_contacts.TerrainPairTable.barrels`,
:data:`~newton._src.geometry.terrain_shadow.TERRAIN_LAW_BARREL_KERNELS`), for every pair of the class: a flat-sided
cylinder or cone gets the same rows from both. The module is compiled without floating-point contraction, so the CPU
decides every boundary as the GPU does.
"""

from __future__ import annotations

import warp as wp

from .terrain_contacts import TERRAIN_PARTNER_CYLINDER, TERRAIN_PARTNER_SCAN_RUN
from .terrain_law_cylinder import TERRAIN_CYLINDER_WALK_SLOTS, cylinder_core_functions, cylinder_core_save
from .terrain_law_solid import solid_kernels

# Every float32 operation rounds as on the CPU (see the module docstring).
wp.set_module_options({"fuse_fp": False})

_core, _core_load = cylinder_core_functions(True, __name__)

_KERNELS = solid_kernels(
    "barrel",
    TERRAIN_PARTNER_CYLINDER,
    TERRAIN_CYLINDER_WALK_SLOTS,
    TERRAIN_PARTNER_SCAN_RUN[TERRAIN_PARTNER_CYLINDER],
    _core,
    cylinder_core_save,
    _core_load,
    __name__,
)
terrain_barrel_rows = _KERNELS["rows"]
terrain_barrel_rows_scan = _KERNELS["scan"]
terrain_barrel_rows_faces = _KERNELS["faces"]
terrain_barrel_rows_edges = _KERNELS["edges"]
terrain_barrel_rows_thetas = _KERNELS["thetas"]
terrain_barrel_rows_creases = _KERNELS["creases"]
terrain_barrel_rows_vertices = _KERNELS["vertices"]
terrain_barrel_rows_border = _KERNELS["border"]
terrain_barrel_rows_close = _KERNELS["close"]
terrain_barrel_rows_recovery = _KERNELS["recovery"]
terrain_barrel_rows_small = _KERNELS["small"]
terrain_barrel_rows_finish = _KERNELS["finish"]
