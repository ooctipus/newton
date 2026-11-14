# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The convex hull of a full-dimensional point cloud (Qhull), kept in the collision cache.

An imported collision mesh approximated by its convex hull runs Qhull on every import of the asset (about 0.3 s for a
humanoid's links); the hull is a pure function of the points and the vertex budget, so it is stored with the meshes'
other collision tables (:mod:`._collision_cache`) and a later import loads it.
"""

from __future__ import annotations

import numpy as np

from . import _collision_cache


def convex_hull_3d(vertices: np.ndarray, maxhullvert: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (``float32``) and outward-wound triangles (``int32``) of the hull of full-rank ``float64`` points.

    The vertices are the input points the hull uses, in input order. ``maxhullvert`` above zero caps the hull's
    vertex count (Qhull's ``TA`` option).
    """
    return _collision_cache.cached("convex_hull", (vertices, maxhullvert), lambda: _qhull(vertices, maxhullvert))


def _qhull(vertices: np.ndarray, maxhullvert: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import ConvexHull, QhullError

    qhull_options = "Qt"
    if maxhullvert > 0:
        # qhull "TA" actually means "number of vertices added after the initial simplex"
        # from mujoco's user_mesh.cc
        qhull_options += f" TA{maxhullvert - 4}"
    try:
        hull = ConvexHull(vertices, qhull_options=qhull_options)
    except QhullError:
        # Retry with joggled input as a last resort before giving up on 3D.
        # QJ perturbs coordinates in-place (~1e-11); it does not reorder or
        # drop points, so hull.simplices indices still refer to the same rows
        # of the original input array. We intentionally pull coordinates from
        # `vertices` rather than `hull.points` below so the returned verts
        # remain a strict subset of the caller's input.
        hull = ConvexHull(vertices, qhull_options=qhull_options + " QJ")
    # Index into the original (un-joggled) input so that verts is always a
    # subset of the caller's vertices, even through the QJ retry path.
    verts = vertices.astype(np.float32)
    faces = hull.simplices.astype(np.int32)

    # Qhull leaves simplex winding arbitrary; flip faces whose normal opposes their facet's outward normal.
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    inward = np.einsum("ij,ij->i", np.cross(b - a, c - a), hull.equations[:, :3]) < 0
    faces[inward] = faces[inward][:, [0, 2, 1]]

    # trim vertices to only those that are used in the faces
    unique_verts, faces_flat = np.unique(faces, return_inverse=True)
    verts = verts[unique_verts]
    faces = faces_flat.reshape(faces.shape).astype(np.int32)

    return verts, faces
