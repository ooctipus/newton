# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cook shared, already low-polygon CAD meshes before scene replication."""

import tempfile
from pathlib import Path

import numpy as np
import warp as wp


def cook_sdf_meshes(requests: list, *, device=None) -> list[dict]:
    """Cook once per mesh; retain holes and thin shells, not convex hulls.

    Mesh construction stays CPU-only. Voxel size is chosen by each asset's
    smallest physical feature, independently of its display tessellation.
    Winding-number signs also handle intersecting closed rack wires as a union.
    Newton's content-addressed cache avoids recooking unchanged CAD.
    """
    device = wp.get_device(device)
    if requests and not device.is_cuda:
        raise ValueError("These CAD scenes require CUDA for SDF collision")
    cache = Path(tempfile.gettempdir()) / "newton_clutter_sdf_cache"
    seen, report = set(), []
    for mesh, voxel_size in requests:
        if id(mesh) in seen:
            continue
        seen.add(id(mesh))
        if not np.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError("SDF voxel size must be finite and positive")
        if not mesh.is_watertight:
            raise ValueError("Clutter CAD must be watertight before SDF conversion")
        if mesh.sdf is None:
            band = max(4 * voxel_size, 0.002)
            mesh.build_sdf(
                device=device,
                target_voxel_size=voxel_size,
                narrow_band_range=(-band, band),
                margin=max(3 * voxel_size, 0.001),
                sign_method="winding",
                cache_dir=cache,
            )
        report.append({"vertices": len(mesh.vertices), "triangles": mesh.indices.size // 3, "voxel_size_m": voxel_size})
    return report
