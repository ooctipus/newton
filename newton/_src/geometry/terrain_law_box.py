# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The terrain law's kernels for boxes and finite planes (rules F, E, V and R of the law for solid cores).

The kernels are :func:`~newton._src.geometry.terrain_law_solid.solid_kernels`'s for the class, launched in the order of
:data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_LAUNCHES` (:mod:`~newton._src.geometry.terrain_shadow`):
:func:`terrain_box_rows` (the whole-pair rejects, one thread per pair), the scan :func:`terrain_box_rows_scan`, the job
kernels :func:`terrain_box_rows_faces`, :func:`terrain_box_rows_edges`, :func:`terrain_box_rows_thetas`,
:func:`terrain_box_rows_creases`, :func:`terrain_box_rows_vertices` and :func:`terrain_box_rows_border` (one thread per
listed feature; the last the heightfield border's walls and clipped faces), the close :func:`terrain_box_rows_close` and
rule R's :func:`terrain_box_rows_recovery`, the finish kernels :func:`terrain_box_rows_small` (one thread per pair with
a plane group of three) and :func:`terrain_box_rows_finish` (a block of lanes per pair the others deferred: more
distinct rows than the class's 16 walk slots, finished from its pooled candidates). The rules and the finish are
:mod:`~newton._src.geometry.terrain_law_solid`'s; the module is compiled without floating-point contraction, so the CPU
decides every boundary as the GPU does.
"""

from __future__ import annotations

import warp as wp

from .terrain_contacts import TERRAIN_PARTNER_BOX, TERRAIN_PARTNER_SCAN_RUN, TERRAIN_PARTNER_WALK_SLOTS
from .terrain_law_cores import TerrainBoxCore
from .terrain_law_solid import TerrainSolidData, TerrainSolidJobs, solid_kernels

# Every float32 operation rounds as on the CPU (see the module docstring).
wp.set_module_options({"fuse_fp": False})

TERRAIN_BOX_WALK_SLOTS = TERRAIN_PARTNER_WALK_SLOTS[TERRAIN_PARTNER_BOX]
"""Walk slots of a pair of the class: its distinct rows while it walks its window, and its rows when they fit: 16."""


@wp.func
def _core(local: wp.transform, size: wp.vec4, kind: int, partner: int, solids: TerrainSolidData):
    """The pair's box in the heightfield frame (centre, rotation, half extents): a box's, or a finite plane's rectangle
    in its local xy as a box of no thickness (the pipeline's shape data holds its half width and length, as Newton's
    support function and drawing take them)."""
    core = TerrainBoxCore()
    core.c = wp.transform_get_translation(local)
    core.rot = wp.quat_to_matrix(wp.normalize(wp.transform_get_rotation(local)))
    core.h = wp.vec3(size[0], size[1], size[2])
    return core


@wp.func
def _core_save(jobs: TerrainSolidJobs, i: int, core: TerrainBoxCore):
    """Keep class pair ``i``'s box (window frame) for its jobs."""
    jobs.box_cores[i] = core


@wp.func
def _core_load(jobs: TerrainSolidJobs, i: int, solids: TerrainSolidData) -> TerrainBoxCore:
    """Class pair ``i``'s kept box."""
    return jobs.box_cores[i]


_KERNELS = solid_kernels(
    "box",
    TERRAIN_PARTNER_BOX,
    TERRAIN_BOX_WALK_SLOTS,
    TERRAIN_PARTNER_SCAN_RUN[TERRAIN_PARTNER_BOX],
    _core,
    _core_save,
    _core_load,
    __name__,
)
terrain_box_rows = _KERNELS["rows"]
terrain_box_rows_scan = _KERNELS["scan"]
terrain_box_rows_faces = _KERNELS["faces"]
terrain_box_rows_edges = _KERNELS["edges"]
terrain_box_rows_thetas = _KERNELS["thetas"]
terrain_box_rows_creases = _KERNELS["creases"]
terrain_box_rows_vertices = _KERNELS["vertices"]
terrain_box_rows_border = _KERNELS["border"]
terrain_box_rows_close = _KERNELS["close"]
terrain_box_rows_recovery = _KERNELS["recovery"]
terrain_box_rows_small = _KERNELS["small"]
terrain_box_rows_finish = _KERNELS["finish"]
