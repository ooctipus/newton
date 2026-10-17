# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Declared environment variables.

Every ``NEWTON_*`` variable that a module under ``newton._src`` reads is declared
in :data:`NEWTON_FLAGS`, together with the ``NEWTON_*`` variables that Isaac Lab
reads and Newton does not. A solver with a private family declares it next to the
solver (``FEATHER_PGS_*`` in ``newton._src.solvers.feather_pgs.environment_flags``).
``newton/tests/test_environment_flags.py`` scans the sources and keeps the
declarations equal to the read sites, so a new variable must be declared here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

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


_COLLIDE = ("sim.collide",)
_NARROW = ("geometry.narrow_phase",)
_ROWS = ("geometry.heightfield_primitive_contact",)
_LAB_MANAGER = ("isaaclab_newton.physics.newton_manager",)

NEWTON_PREFIXES = ("NEWTON_",)

NEWTON_FLAGS = declare(
    EnvironmentFlag(
        "NEWTON_CACHE_PATH",
        "policy",
        ("utils.download_assets",),
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
        "NEWTON_HEIGHTFIELD_CONVEX_CELLS",
        "experiment",
        _NARROW,
        "=1 requests the terrain convex-cells owner; requires the typed primitive query.",
    ),
    EnvironmentFlag(
        "NEWTON_HEIGHTFIELD_CONVEX_SPLIT",
        "policy",
        (*_NARROW, *_COLLIDE),
        "Staged convex terrain queries: 1 on, 0 off; unset lets the collision pipeline decide.",
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
        "NEWTON_HEIGHTFIELD_WELD_FLAT_SEAMS",
        "experiment",
        ("geometry.heightfield_features",),
        "=1 ignores interior triangulation seams on flat terrain.",
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
        "NEWTON_NARROW_PHASE_CONVEX_CELLS",
        "experiment",
        _COLLIDE,
        "=1 replaces rejection-route support scans with direction cells; convex_collision overrides it.",
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
        ("sensors.sensor_tiled_camera", "viewer.viewer_rtx"),
        "Values other than 0 enable renderer profiling output.",
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
