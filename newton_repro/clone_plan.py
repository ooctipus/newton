# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dataclasses shared by the Newton repro exporter, loader, and replicator."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SiteRequest:
    """Newton site injection request captured before physics replication.

    Attributes:
        label: Stable site label assigned by ``NewtonManager.cl_register_site``.
        body_pattern: Prototype-local body-label regex. ``None`` denotes a
            global world-origin site.
        xform: Site transform as ``(px, py, pz, qx, qy, qz, qw)``.
    """

    label: str
    body_pattern: str | None
    xform: tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class ClonePlan:
    """Replication plan captured from Isaac Lab's scene cloner.

    Attributes:
        sources: Source prim paths used for cloning (e.g.
            ``("/World/envs/env_0",)``).
        destinations: Destination prim path templates with ``{}`` placeholders
            (e.g. ``("/World/envs/env_{}",)``).
        clone_mask: Boolean source-to-environment mapping of shape
            ``(num_sources, num_envs)``, ``bool``; ``True`` at ``(row, col)``
            means source ``row`` is active in environment ``col``.
        env_origins: Per-environment world positions [m] of shape
            ``(num_envs, 3)``, ``float32``.
        env_spacing: Nominal grid spacing [m] from the Isaac Lab scene config.
        up_axis: Stage/model up axis, usually ``"Z"``.
        simplify_meshes: Whether Newton convex-hull mesh approximation was
            requested for prototype builders.
        site_requests: Pending Newton site requests captured before cloning.
    """

    sources: tuple[str, ...]
    destinations: tuple[str, ...]
    clone_mask: np.ndarray
    env_origins: np.ndarray
    env_spacing: float | None = None
    up_axis: str = "Z"
    simplify_meshes: bool = True
    site_requests: tuple[SiteRequest, ...] = field(default_factory=tuple)

    @property
    def num_envs(self) -> int:
        """Number of environments encoded by the clone mask."""
        return int(self.clone_mask.shape[1])
