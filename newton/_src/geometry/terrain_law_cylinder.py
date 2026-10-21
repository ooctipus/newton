# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The terrain law's kernels for flat-capped cylinders and cones (rules F, E, V, R and X of the law for solid cores).

The kernels are :func:`~newton._src.geometry.terrain_law_solid.solid_kernels`'s for the class, launched in the order of
:data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_LAUNCHES` (:mod:`~newton._src.geometry.terrain_shadow`):
:func:`terrain_cylinder_rows` (the whole-pair rejects, one thread per pair), the scan
:func:`terrain_cylinder_rows_scan`, the job kernels :func:`terrain_cylinder_rows_faces`,
:func:`terrain_cylinder_rows_edges`, :func:`terrain_cylinder_rows_thetas`, :func:`terrain_cylinder_rows_creases` and
:func:`terrain_cylinder_rows_vertices` (one thread per listed feature), the close :func:`terrain_cylinder_rows_close`
and rule R's :func:`terrain_cylinder_rows_recovery`, rule X's :func:`terrain_cylinder_rows_xitems`,
:func:`terrain_cylinder_rows_xvalleys`, :func:`terrain_cylinder_rows_xdescents`, :func:`terrain_cylinder_rows_exit` and
:func:`terrain_cylinder_rows_settle`, and last the pairs the others deferred: :func:`terrain_cylinder_rows_pooled`
finishes those with more distinct rows than the class's 16 walk slots from their pooled candidates,
:func:`terrain_cylinder_rows_full` walks those whose work found no room. The walk, the rules and the
finish are :mod:`~newton._src.geometry.terrain_law_solid`'s; the module is compiled without floating-point contraction,
so the CPU decides every boundary as the GPU does.
"""

from __future__ import annotations

import warp as wp

from .terrain_contacts import TERRAIN_PARTNER_CYLINDER, TERRAIN_PARTNER_WALK_SLOTS
from .terrain_law_cores import TerrainCylinderCore
from .terrain_law_solid import TerrainSolidData, TerrainSolidJobs, solid_kernels
from .types import GeoType

_CONE = wp.constant(int(GeoType.CONE))

# Every float32 operation rounds as on the CPU (see the module docstring).
wp.set_module_options({"fuse_fp": False})

TERRAIN_CYLINDER_WALK_SLOTS = TERRAIN_PARTNER_WALK_SLOTS[TERRAIN_PARTNER_CYLINDER]
"""Walk slots of a pair of the class: its distinct rows while it walks its window, and its rows when they fit: 16."""


@wp.func
def _core(local: wp.transform, size: wp.vec4, kind: int, partner: int, solids: TerrainSolidData):
    """The pair's cylinder or cone in the heightfield frame (centre, axis, radius, half height)."""
    core = TerrainCylinderCore()
    core.c = wp.transform_get_translation(local)
    core.a = wp.normalize(wp.quat_rotate(wp.transform_get_rotation(local), wp.vec3(0.0, 0.0, 1.0)))
    core.r = size[0]
    core.h = size[1]
    core.cone = int(kind == _CONE)
    return True, core


@wp.func
def _core_save(jobs: TerrainSolidJobs, i: int, core: TerrainCylinderCore):
    """Keep class pair ``i``'s cylinder or cone (window frame) for its jobs."""
    jobs.cylinder_cores[i] = core


@wp.func
def _core_load(jobs: TerrainSolidJobs, i: int, solids: TerrainSolidData) -> TerrainCylinderCore:
    """Class pair ``i``'s kept cylinder or cone."""
    return jobs.cylinder_cores[i]


_KERNELS = solid_kernels(
    "cylinder", TERRAIN_PARTNER_CYLINDER, TERRAIN_CYLINDER_WALK_SLOTS, _core, _core_save, _core_load, __name__
)
terrain_cylinder_rows = _KERNELS["rows"]
terrain_cylinder_rows_scan = _KERNELS["scan"]
terrain_cylinder_rows_faces = _KERNELS["faces"]
terrain_cylinder_rows_edges = _KERNELS["edges"]
terrain_cylinder_rows_thetas = _KERNELS["thetas"]
terrain_cylinder_rows_creases = _KERNELS["creases"]
terrain_cylinder_rows_vertices = _KERNELS["vertices"]
terrain_cylinder_rows_close = _KERNELS["close"]
terrain_cylinder_rows_recovery = _KERNELS["recovery"]
terrain_cylinder_rows_xitems = _KERNELS["xitems"]
terrain_cylinder_rows_xvalleys = _KERNELS["xvalleys"]
terrain_cylinder_rows_xdescents = _KERNELS["xdescents"]
terrain_cylinder_rows_exit = _KERNELS["exit"]
terrain_cylinder_rows_settle = _KERNELS["settle"]
terrain_cylinder_rows_pooled = _KERNELS["pooled"]
terrain_cylinder_rows_full = _KERNELS["full"]
