# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The terrain law's kernels for flat-sided cylinders and cones (rules F, E, V and R of the law for solid cores), and
the class's core functions for barrel cylinders too.

The core functions branch on a barrel cylinder's side radius
(:class:`~newton._src.geometry.terrain_law_cores.TerrainCylinderCore`). These kernels set it to a literal 0, so that
the compiler drops every barrel path from them (a fifth of their registers, the scan's and theta's); a class with a
barrel cylinder among its partners takes :mod:`~newton._src.geometry.terrain_law_barrel`'s kernels instead, the same
code with the radius read (the shadow's choice,
:attr:`~newton._src.geometry.terrain_contacts.TerrainPairTable.barrels`). A flat-sided cylinder or cone gets the same
rows from both.

The kernels are :func:`~newton._src.geometry.terrain_law_solid.solid_kernels`'s for the class, launched in the order of
:data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_LAUNCHES` (:mod:`~newton._src.geometry.terrain_shadow`):
:func:`terrain_cylinder_rows` (the whole-pair rejects, one thread per pair), the scan
:func:`terrain_cylinder_rows_scan`, the job kernels :func:`terrain_cylinder_rows_faces`,
:func:`terrain_cylinder_rows_edges`, :func:`terrain_cylinder_rows_thetas`, :func:`terrain_cylinder_rows_creases`,
:func:`terrain_cylinder_rows_vertices` and :func:`terrain_cylinder_rows_border` (one thread per listed feature; the last
the heightfield border's walls and clipped faces), the close :func:`terrain_cylinder_rows_close` and rule R's
:func:`terrain_cylinder_rows_recovery`, the finish kernels :func:`terrain_cylinder_rows_small` (one thread per pair with
a plane group of three) and :func:`terrain_cylinder_rows_finish` (a block of lanes per pair the others deferred: more
distinct rows than the class's 16 walk slots, finished from its pooled candidates). The rules and the finish are
:mod:`~newton._src.geometry.terrain_law_solid`'s; the module is compiled without floating-point contraction, so the CPU
decides every boundary as the GPU does.
"""

from __future__ import annotations

import warp as wp

from .terrain_contacts import TERRAIN_PARTNER_CYLINDER, TERRAIN_PARTNER_SCAN_RUN, TERRAIN_PARTNER_WALK_SLOTS
from .terrain_law_cores import TerrainCylinderCore
from .terrain_law_solid import TerrainSolidData, TerrainSolidJobs, solid_kernels
from .types import GeoType

_CONE = wp.constant(int(GeoType.CONE))

# Every float32 operation rounds as on the CPU (see the module docstring).
wp.set_module_options({"fuse_fp": False})

TERRAIN_CYLINDER_WALK_SLOTS = TERRAIN_PARTNER_WALK_SLOTS[TERRAIN_PARTNER_CYLINDER]
"""Walk slots of a pair of the class: its distinct rows while it walks its window, and its rows when they fit: 16."""


@wp.func
def cylinder_core_save(jobs: TerrainSolidJobs, i: int, core: TerrainCylinderCore):
    """Keep class pair ``i``'s cylinder or cone (window frame) for its jobs."""
    jobs.cylinder_cores[i] = core


def cylinder_core_functions(barrels: bool, module: str):
    """Return the class's ``make_core`` and ``load_core``
    (:func:`~newton._src.geometry.terrain_law_solid.solid_kernels`) for the kernels of ``module``: with ``barrels``, a
    barrel cylinder's side radius read from ``scale[2]``; without, every core's radius a literal 0, which the compiler
    carries through the core functions, dropping their barrel paths. Save a core with :func:`cylinder_core_save`."""

    def make_core(local: wp.transform, size: wp.vec4, kind: int, partner: int, solids: TerrainSolidData):
        """The pair's cylinder (a barrel cylinder's side radius from ``scale[2]``) or cone in the heightfield frame
        (centre, axis, radius, half height)."""
        core = TerrainCylinderCore()
        core.c = wp.transform_get_translation(local)
        core.a = wp.normalize(wp.quat_rotate(wp.transform_get_rotation(local), wp.vec3(0.0, 0.0, 1.0)))
        core.r = size[0]
        core.h = size[1]
        core.cone = int(kind == _CONE)
        core.barrel = 0.0
        if wp.static(barrels):
            if kind != _CONE:
                core.barrel = size[2]
        return core

    def load_core(jobs: TerrainSolidJobs, i: int, solids: TerrainSolidData) -> TerrainCylinderCore:
        """Class pair ``i``'s kept cylinder or cone."""
        core = jobs.cylinder_cores[i]
        if wp.static(not barrels):
            core.barrel = 0.0
        return core

    functions = []
    for function in (make_core, load_core):
        key = f"_{function.__name__}_{'barrel' if barrels else 'cylinder'}"
        function.__name__ = key
        function.__qualname__ = key
        functions.append(wp.func(function, module=module))
    return tuple(functions)


_core, _core_load = cylinder_core_functions(False, __name__)


_KERNELS = solid_kernels(
    "cylinder",
    TERRAIN_PARTNER_CYLINDER,
    TERRAIN_CYLINDER_WALK_SLOTS,
    TERRAIN_PARTNER_SCAN_RUN[TERRAIN_PARTNER_CYLINDER],
    _core,
    cylinder_core_save,
    _core_load,
    __name__,
)
terrain_cylinder_rows = _KERNELS["rows"]
terrain_cylinder_rows_scan = _KERNELS["scan"]
terrain_cylinder_rows_faces = _KERNELS["faces"]
terrain_cylinder_rows_edges = _KERNELS["edges"]
terrain_cylinder_rows_thetas = _KERNELS["thetas"]
terrain_cylinder_rows_creases = _KERNELS["creases"]
terrain_cylinder_rows_vertices = _KERNELS["vertices"]
terrain_cylinder_rows_border = _KERNELS["border"]
terrain_cylinder_rows_close = _KERNELS["close"]
terrain_cylinder_rows_recovery = _KERNELS["recovery"]
terrain_cylinder_rows_small = _KERNELS["small"]
terrain_cylinder_rows_finish = _KERNELS["finish"]
