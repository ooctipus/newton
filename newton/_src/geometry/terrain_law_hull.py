# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The terrain law's kernels for convex hulls and ellipsoids (rules F, E, V, R and X of the law for solid cores).

The kernels are :func:`~newton._src.geometry.terrain_law_solid.solid_kernels`'s for the class, launched in the order of
:data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_LAUNCHES` (:mod:`~newton._src.geometry.terrain_shadow`):
:func:`terrain_hull_rows` (the whole-pair rejects, one thread per pair), the scan :func:`terrain_hull_rows_scan`, the
job kernels :func:`terrain_hull_rows_faces`, :func:`terrain_hull_rows_edges`, :func:`terrain_hull_rows_thetas`,
:func:`terrain_hull_rows_creases` and :func:`terrain_hull_rows_vertices` (one thread per listed feature), the close
:func:`terrain_hull_rows_close` and rule R's :func:`terrain_hull_rows_recovery`, rule X's
:func:`terrain_hull_rows_xitems`, :func:`terrain_hull_rows_xvalleys`, :func:`terrain_hull_rows_xdescents` and
:func:`terrain_hull_rows_exit`, and last :func:`terrain_hull_rows_full`, which walks the pairs the others deferred (more
distinct rows than the class's 24 walk slots, or no room for their work). The walk, the rules and the finish are
:mod:`~newton._src.geometry.terrain_law_solid`'s; the module is compiled without floating-point contraction, so the CPU
decides every boundary as the GPU does.
"""

from __future__ import annotations

import warp as wp

from .terrain_contacts import TERRAIN_PARTNER_HULL, TERRAIN_PARTNER_WALK_SLOTS
from .terrain_law_cores import TerrainHullCore, TerrainHullCoreRecord
from .terrain_law_solid import TerrainSolidData, TerrainSolidJobs, solid_kernels
from .types import GeoType

_ELLIPSOID = wp.constant(int(GeoType.ELLIPSOID))
_CONVEX_MESH = wp.constant(int(GeoType.CONVEX_MESH))

# Every float32 operation rounds as on the CPU (see the module docstring).
wp.set_module_options({"fuse_fp": False})

TERRAIN_HULL_WALK_SLOTS = TERRAIN_PARTNER_WALK_SLOTS[TERRAIN_PARTNER_HULL]
"""Walk slots of a pair of the class: its distinct rows while it walks its window, and its rows when they fit: 24."""


@wp.func
def _core(local: wp.transform, size: wp.vec4, kind: int, partner: int, solids: TerrainSolidData):
    """The pair's convex hull (its points and faces from the hull table, scaled) or ellipsoid (semi-axes) in the
    heightfield frame; ``False`` for a shape of the class with no core yet (a barrel cylinder)."""
    core = TerrainHullCore()
    core.c = wp.transform_get_translation(local)
    core.rot = wp.quat_to_matrix(wp.normalize(wp.transform_get_rotation(local)))
    core.s = wp.vec3(size[0], size[1], size[2])
    core.table = solids.hulls
    if kind == _ELLIPSOID:
        core.ellipsoid = 1
        core.radius = wp.max(core.s[0], wp.max(core.s[1], core.s[2]))
        return True, core
    if kind != _CONVEX_MESH:
        return False, core
    hull = solids.hulls.shape_hull[partner]
    if hull < 0:
        return False, core
    points = solids.hulls.hull_points[hull]
    faces = solids.hulls.hull_faces[hull]
    triangles = solids.hulls.hull_triangles[hull]
    core.point_begin = points[0]
    core.point_count = points[1]
    core.face_begin = faces[0]
    core.face_count = faces[1]
    core.triangle_begin = triangles[0]
    core.triangle_count = triangles[1]
    # the radius, and the point lowest along the heightfield's up axis (the support searches' start)
    up = wp.cw_mul(wp.vec3(core.rot[2, 0], core.rot[2, 1], core.rot[2, 2]), core.s)
    radius = float(0.0)
    core.start = core.point_begin
    low = wp.dot(solids.hulls.points[core.point_begin], up)
    for i in range(core.point_begin, core.point_begin + core.point_count):
        p = solids.hulls.points[i]
        radius = wp.max(radius, wp.length(wp.cw_mul(p, core.s)))
        value = wp.dot(p, up)
        if value < low:
            low = value
            core.start = i
    core.radius = radius
    return True, core


@wp.func
def _core_save(jobs: TerrainSolidJobs, i: int, core: TerrainHullCore):
    """Keep class pair ``i``'s hull or ellipsoid (window frame) for its jobs, without its table."""
    record = TerrainHullCoreRecord()
    record.c = core.c
    record.rot = core.rot
    record.s = core.s
    record.ellipsoid = core.ellipsoid
    record.point_begin = core.point_begin
    record.point_count = core.point_count
    record.face_begin = core.face_begin
    record.face_count = core.face_count
    record.triangle_begin = core.triangle_begin
    record.triangle_count = core.triangle_count
    record.radius = core.radius
    record.start = core.start
    jobs.hull_cores[i] = record


@wp.func
def _core_load(jobs: TerrainSolidJobs, i: int, solids: TerrainSolidData) -> TerrainHullCore:
    """Class pair ``i``'s kept hull or ellipsoid, its table the solid classes' hulls."""
    record = jobs.hull_cores[i]
    core = TerrainHullCore()
    core.c = record.c
    core.rot = record.rot
    core.s = record.s
    core.ellipsoid = record.ellipsoid
    core.point_begin = record.point_begin
    core.point_count = record.point_count
    core.face_begin = record.face_begin
    core.face_count = record.face_count
    core.triangle_begin = record.triangle_begin
    core.triangle_count = record.triangle_count
    core.radius = record.radius
    core.start = record.start
    core.table = solids.hulls
    return core


_KERNELS = solid_kernels("hull", TERRAIN_PARTNER_HULL, TERRAIN_HULL_WALK_SLOTS, _core, _core_save, _core_load, __name__)
terrain_hull_rows = _KERNELS["rows"]
terrain_hull_rows_scan = _KERNELS["scan"]
terrain_hull_rows_faces = _KERNELS["faces"]
terrain_hull_rows_edges = _KERNELS["edges"]
terrain_hull_rows_thetas = _KERNELS["thetas"]
terrain_hull_rows_creases = _KERNELS["creases"]
terrain_hull_rows_vertices = _KERNELS["vertices"]
terrain_hull_rows_close = _KERNELS["close"]
terrain_hull_rows_recovery = _KERNELS["recovery"]
terrain_hull_rows_xitems = _KERNELS["xitems"]
terrain_hull_rows_xvalleys = _KERNELS["xvalleys"]
terrain_hull_rows_xdescents = _KERNELS["xdescents"]
terrain_hull_rows_exit = _KERNELS["exit"]
terrain_hull_rows_full = _KERNELS["full"]
