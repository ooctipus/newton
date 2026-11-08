# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Declared environment variables and the construction-time record of them.

Every ``NEWTON_*`` variable that a module under ``newton._src`` reads is declared
in :data:`NEWTON_FLAGS`, together with the ``NEWTON_*`` variables that Isaac Lab
reads and Newton does not. A solver with a private family declares it next to the
solver (``FEATHER_PGS_*`` in ``newton._src.solvers.feather_pgs.environment_flags``).
``newton/tests/test_environment_flags.py`` scans the sources and keeps the
declarations equal to the read sites, so a new variable must be declared here.
A record only classifies what the environment held; the read sites are unchanged.
:func:`retired_notice` turns a record's retired names into a warning, once per
name and process.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, get_args

FlagKind = Literal["policy", "experiment", "diagnostic", "reference", "retired"]
"""``policy`` turns off or tunes a default, ``experiment`` opts into a non-default
path, ``diagnostic`` checks, captures, prints or times, ``reference`` selects a
comparison implementation, and ``retired`` no longer selects anything."""

FlagOwner = Literal["newton", "isaac_lab"]
"""``isaac_lab`` marks a name that Isaac Lab reads and Newton never does."""

FLAG_KINDS = get_args(FlagKind)
FLAG_OWNERS = get_args(FlagOwner)


@dataclass(frozen=True)
class EnvironmentFlag:
    """One declared environment variable."""

    name: str
    """Variable name."""
    kind: FlagKind
    """Role of the variable; see :data:`FlagKind`."""
    readers: tuple[str, ...]
    """Modules that read it: paths below ``newton._src``, or Isaac Lab modules for owner ``isaac_lab``."""
    meaning: str
    """One line: what a value selects, with the default when unset is not simply off."""
    owner: FlagOwner = "newton"
    """Which code base reads the variable."""


def declare(*flags: EnvironmentFlag) -> dict[str, EnvironmentFlag]:
    """Index declarations by name in declaration order, rejecting duplicates and unknown kinds or owners."""
    table: dict[str, EnvironmentFlag] = {}
    for flag in flags:
        if flag.name in table:
            raise ValueError(f"{flag.name} is declared twice")
        if flag.kind not in FLAG_KINDS or flag.owner not in FLAG_OWNERS:
            raise ValueError(f"{flag.name} has kind {flag.kind!r} and owner {flag.owner!r}")
        table[flag.name] = flag
    return table


def record_environment(
    declared: Mapping[str, EnvironmentFlag],
    prefixes: tuple[str, ...],
    environ: Mapping[str, str] | None = None,
) -> dict:
    """Classify the variables of ``environ`` whose names start with one of ``prefixes``.

    Args:
        declared: Declarations indexed by name.
        prefixes: Name prefixes that belong to the caller.
        environ: Variables to classify; defaults to :data:`os.environ`.

    Returns:
        ``{"applied": {name: value}, "unknown": [name], "retired": [name], "isaac_lab": {name: value}}``
        sorted by name: declared variables Newton reads, undeclared names (recorded, not
        rejected), retired names whose value did not raise, and variables only Isaac Lab reads.
    """
    environ = os.environ if environ is None else environ
    applied: dict[str, str] = {}
    unknown: list[str] = []
    retired: list[str] = []
    isaac_lab: dict[str, str] = {}
    for name in sorted(environ):
        if not name.startswith(prefixes):
            continue
        flag = declared.get(name)
        if flag is None:
            unknown.append(name)
        elif flag.owner == "isaac_lab":
            isaac_lab[name] = environ[name]
        elif flag.kind == "retired":
            retired.append(name)
        else:
            applied[name] = environ[name]
    return {"applied": applied, "unknown": unknown, "retired": retired, "isaac_lab": isaac_lab}


_WARNED_RETIRED: set[str] = set()
"""Retired names that :func:`retired_notice` has reported in this process."""


def retired_notice(record: Mapping[str, Any]) -> str | None:
    """Return a warning that names the retired variables of ``record`` this process has not reported.

    Retired names select nothing, so construction proceeds and the record lists them; the
    caller emits the text as a :class:`UserWarning`. Each name is reported once per process.

    Args:
        record: A record returned by :func:`record_environment`.

    Returns:
        The warning text, or ``None`` when the record lists no retired name not yet reported.
    """
    names = [name for name in record["retired"] if name not in _WARNED_RETIRED]
    if not names:
        return None
    _WARNED_RETIRED.update(names)
    return (
        f"Ignoring retired environment variables that select nothing: {', '.join(names)}. Remove them from "
        "the environment; the construction record lists them under environment['retired']."
    )


_COLLIDE = ("sim.collide",)
_NARROW = ("geometry.narrow_phase",)
_ROWS = ("geometry.heightfield_primitive_contact",)
_LAB_MANAGER = ("isaaclab_newton.physics.newton_manager",)

NEWTON_PREFIXES = ("NEWTON_",)

NEWTON_FLAGS = declare(
    EnvironmentFlag(
        "NEWTON_BROAD_PHASE_IMMOBILE_FILTER",
        "policy",
        _COLLIDE,
        "=0 keeps pairs whose shapes are both static or welded to the world (default 1 drops them).",
    ),
    EnvironmentFlag(
        "NEWTON_BROAD_PHASE_TEMPLATES",
        "policy",
        _COLLIDE,
        "World-template explicit broad phase: 1 for any model-built template list, 0 off; unset admits large ones.",
    ),
    EnvironmentFlag(
        "NEWTON_CACHE_PATH",
        "policy",
        ("newton.config",),
        "Directory for downloaded assets (default: the user cache directory).",
    ),
    EnvironmentFlag(
        "NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD",
        "retired",
        _NARROW,
        "Retired adaptive terrain manifold; values other than 0 raise in NarrowPhase.",
    ),
    EnvironmentFlag(
        "NEWTON_HEIGHTFIELD_ANALYTIC_CONVEX",
        "policy",
        _ROWS,
        "=0 keeps convex hulls on the generic terrain query instead of the closed-form row (default 1).",
    ),
    EnvironmentFlag(
        "NEWTON_HEIGHTFIELD_ANALYTIC_PRIMITIVES",
        "policy",
        _ROWS,
        "=0 keeps typed terrain queries on GJK/MPR instead of the closed-form rows (default 1).",
    ),
    EnvironmentFlag(
        "NEWTON_HEIGHTFIELD_CELL_REJECT",
        "policy",
        _NARROW,
        "=0 keeps the unculled tiled heightfield midphase (default 1).",
    ),
    EnvironmentFlag(
        "NEWTON_HEIGHTFIELD_CYLINDER_SPLIT",
        "experiment",
        _NARROW,
        "=1 splits the typed cylinder terrain query into passes; inert while the closed-form cylinder row is on.",
    ),
    EnvironmentFlag(
        "NEWTON_HEIGHTFIELD_FINITE_QUERY",
        "experiment",
        _NARROW,
        "=1 selects the finite-box terrain query and manifold owner.",
    ),
    EnvironmentFlag(
        "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL",
        "experiment",
        _NARROW,
        "=1 rejects separated terrain triangles with enclosing boxes before the contact query.",
    ),
    EnvironmentFlag(
        "NEWTON_HEIGHTFIELD_LANE_MIDPHASE",
        "policy",
        _NARROW,
        "=0 keeps one thread per pair over every cell of the terrain midphase instead of lanes for long windows "
        "(default 1).",
    ),
    EnvironmentFlag(
        "NEWTON_HEIGHTFIELD_PAIR_CSR",
        "experiment",
        _NARROW,
        "=1 groups terrain witnesses per shape pair and publishes at most four per pair.",
    ),
    EnvironmentFlag(
        "NEWTON_HEIGHTFIELD_PAIR_CSR_SHELL",
        "experiment",
        _NARROW,
        "=1 also admits positive-clearance finite-query face witnesses under pair CSR.",
    ),
    EnvironmentFlag(
        "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY",
        "experiment",
        _NARROW,
        "=1 splits terrain queries into typed per-primitive kernels; convex_collision may force it.",
    ),
    EnvironmentFlag(
        "NEWTON_HEIGHTFIELD_ROW_GATE",
        "policy",
        _NARROW,
        "=0 scans the whole terrain triangle stream in every typed row, even rows no current pair reaches (default 1).",
    ),
    EnvironmentFlag(
        "NEWTON_KAMINO_DISABLE_TILE_TRANSPOSE_UPDATE",
        "policy",
        ("solvers.kamino._src.linalg.factorize._tile_builtins",),
        "1, true or True disables Warp's native tile transpose-update builtins in Kamino.",
    ),
    EnvironmentFlag(
        "NEWTON_NARROW_PHASE_COHERENT_CONVEX",
        "experiment",
        _COLLIDE,
        "Temporal convex rejection: 0 (default), reject_only or auto; convex_collision overrides it.",
    ),
    EnvironmentFlag(
        "NEWTON_NARROW_PHASE_COHERENT_STATS",
        "diagnostic",
        _COLLIDE,
        "=1 compiles diagnostic counters into the coherent convex filter.",
    ),
    EnvironmentFlag(
        "NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS",
        "policy",
        _NARROW,
        "=0 stops packing independent heightfield pairs across lanes (default 1).",
    ),
    EnvironmentFlag(
        "NEWTON_NARROW_PHASE_PAIR_SHAPE_PREP",
        "experiment",
        _COLLIDE,
        "=1 prepares geometry only for the shapes in the explicit pair list.",
    ),
    EnvironmentFlag(
        "NEWTON_PROFILE",
        "diagnostic",
        ("sensors.sensor_camera", "sensors.sensor_tiled_camera", "solvers.steel.solver_steel", "viewer.viewer_rtx"),
        "Values other than 0 enable renderer and SolverSteel profiling ranges.",
    ),
    EnvironmentFlag(
        "NEWTON_REDUCER_COMPACT_EXPORT",
        "experiment",
        _NARROW,
        "=1 selects compact reduced-contact export when contact_export is None.",
    ),
    EnvironmentFlag(
        "NEWTON_REDUCER_EXPORT_CLEANUP",
        "experiment",
        _NARROW,
        "=1 retires consumed reducer slots during export and resets only counters.",
    ),
    EnvironmentFlag(
        "NEWTON_SPHINX_BUILD",
        "policy",
        ("viewer.viewer",),
        "A non-empty value marks a documentation build, which the viewer treats as Sphinx.",
    ),
    EnvironmentFlag(
        "NEWTON_TERRAIN_LAW",
        "policy",
        _COLLIDE,
        "Who writes heightfield contacts. Unset: the terrain law, for the heightfield pairs of every partner class it "
        "has a kernel for (point: spheres and capsules; box; cylinder: cylinders and cones; hull: convex hulls and "
        "ellipsoids), which the terrain route then drops, where the pipeline's configuration is one it supports (not "
        "speculative contacts, NEWTON_TERRAIN_LAW_SHADOW or a prebuilt narrow phase's own contact writer, which keep "
        "the terrain route). A class list or all requests the law for those classes and raises on such a "
        "configuration; none keeps the terrain route (a temporary opt-out until that route is removed).",
    ),
    EnvironmentFlag(
        "NEWTON_TERRAIN_LAW_SHADOW",
        "diagnostic",
        _COLLIDE,
        "Partner classes whose terrain-law kernels run beside the terrain route, which then writes every heightfield "
        "contact, into a side buffer nothing reads (comparison mode): point (spheres and capsules), box, cylinder "
        "(cylinders and cones), hull (convex hulls and ellipsoids) or all. Unset runs none.",
    ),
    EnvironmentFlag(
        "NEWTON_TERRAIN_LAW_SHADOW_JOBS",
        "policy",
        _COLLIDE,
        "A positive factor on the terrain law's solid-class job lists and candidate pools, scene-level buffers sized "
        "from its pairs' own bounds (default 1); work past their room takes their fixed reserve (its pair's rows "
        "unchanged) and raises the narrow phase's terrain_jobs status, and a pair whose work finds no room there "
        "either keeps no row. Read whenever the terrain law runs (by default with heightfield pairs, or in shadow).",
    ),
    EnvironmentFlag(
        "NEWTON_TERRAIN_LAW_RULE_X",
        "retired",
        ("core.environment_flags",),
        "Retired terrain-law rule X (a solid core's deepest exit, which gave no row at the law's operating point); "
        "ignored, and the collision pipeline warns once per process.",
    ),
    EnvironmentFlag(
        "NEWTON_TERRAIN_LAW_SHADOW_ROWS",
        "policy",
        _COLLIDE,
        "Rows of the terrain law's scene-level spill region, which holds the rows of pairs with more rows than their "
        "walk slots (default: from its pairs' own bounds on their rows); read whenever the terrain law runs (by "
        "default with heightfield pairs, or in shadow).",
    ),
    EnvironmentFlag(
        "NEWTON_ASSET_DIR",
        "policy",
        ("isaaclab.utils.assets",),
        "Isaac Lab's Newton asset repository URL or checkout (default: the newton-assets repository).",
        owner="isaac_lab",
    ),
    EnvironmentFlag(
        "NEWTON_COLLISION_BOX_SAT",
        "experiment",
        _LAB_MANAGER,
        "=1 makes Isaac Lab construct the collision pipeline with box_box_sat=True.",
        owner="isaac_lab",
    ),
    EnvironmentFlag(
        "NEWTON_CONTACT_MATCHING",
        "experiment",
        _LAB_MANAGER,
        "A non-empty value is passed by Isaac Lab as the pipeline's contact_matching.",
        owner="isaac_lab",
    ),
    EnvironmentFlag(
        "NEWTON_NARROW_PHASE_THREADS_X",
        "experiment",
        _LAB_MANAGER,
        "Isaac Lab multiplies narrow_phase.total_num_threads by this value after construction (default 1).",
        owner="isaac_lab",
    ),
)


def record_newton_environment(environ: Mapping[str, str] | None = None) -> dict:
    """Record the ``NEWTON_*`` environment once, when a collision pipeline is constructed."""
    return record_environment(NEWTON_FLAGS, NEWTON_PREFIXES, environ)
