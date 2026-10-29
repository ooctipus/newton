# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Removed research selectors and the packed sparse-factor route's flags select nothing: unknown names."""

import os
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import environment_flags

RETIRED_FLAGS = (
    "FEATHER_PGS_SINGLE_FACTOR",
    "FEATHER_PGS_BODY_BASIS_ROWS",
    "FEATHER_PGS_SPARSE_PACKETS",
    "FEATHER_PGS_SPARSE_PAIRED_GS",
    "FEATHER_PGS_COUPLED_CONTACT",
    "FEATHER_PGS_COUPLED_JACOBI",
    "FEATHER_PGS_DENSE_ROW_REGISTERS",
)

# The packed sparse-factor route's selectors, which Isaac Lab's G1 recipes still set.
PACKED_ROUTE_FLAGS = (
    "FEATHER_PGS_G1_CHAIN_SCAN",
    "FEATHER_PGS_SPARSE_FACTOR",
    "FEATHER_PGS_SPARSE_LEVEL_UPDATE",
    "FEATHER_PGS_SPARSE_LIMIT_JACOBI",
    "FEATHER_PGS_SPARSE_METRIC_TANGENTS",
    "FEATHER_PGS_SPARSE_PARALLEL_LIMITS",
    "FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS",
)


def _humanoids(device, worlds=4):
    """A free-root humanoid tree per world: two six-joint legs, a waist and two five-joint arms."""
    scene = newton.ModelBuilder()
    cfg = scene.default_shape_cfg.copy()
    cfg.mu = 0.8
    stance = wp.transform(wp.vec3(0.0, 0.0, 0.8), wp.quat_identity())
    pelvis = scene.add_link(xform=stance, mass=4.0, inertia=wp.mat33(np.eye(3) * 0.05), lock_inertia=True)
    scene.add_shape_box(pelvis, hx=0.1, hy=0.12, hz=0.06, cfg=cfg)
    joints = [scene.add_joint_free(child=pelvis, parent_xform=stance)]

    def chain(parent, origin, axes, step, shape):
        for index, axis in enumerate(axes):
            body = scene.add_link(mass=0.8, inertia=wp.mat33(np.eye(3) * 0.004), lock_inertia=True)
            offset = origin if index == 0 else step
            joints.append(
                scene.add_joint_revolute(
                    parent=parent,
                    child=body,
                    axis=axis,
                    parent_xform=wp.transform(offset, wp.quat_identity()),
                    target_ke=60.0,
                    target_kd=4.0,
                    limit_lower=-1.2,
                    limit_upper=1.2,
                )
            )
            parent = body
        shape(parent)
        return parent

    x, y, z = newton.Axis.X, newton.Axis.Y, newton.Axis.Z
    for side in (1.0, -1.0):
        chain(
            pelvis,
            wp.vec3(0.0, 0.1 * side, -0.06),
            (z, x, y, y, y, x),
            wp.vec3(0.0, 0.0, -0.12),
            lambda foot: scene.add_shape_box(foot, hx=0.08, hy=0.04, hz=0.02, cfg=cfg),
        )
    torso = chain(
        pelvis,
        wp.vec3(0.0, 0.0, 0.08),
        (z,),
        wp.vec3(),
        lambda body: scene.add_shape_box(body, hx=0.1, hy=0.12, hz=0.15),
    )
    for side in (1.0, -1.0):
        chain(
            torso,
            wp.vec3(0.0, 0.16 * side, 0.12),
            (y, x, z, y, z),
            wp.vec3(0.0, 0.0, -0.1),
            lambda hand: scene.add_shape_sphere(hand, radius=0.03, cfg=cfg),
        )
    scene.add_articulation(joints)
    builder = newton.ModelBuilder()
    builder.replicate(scene, worlds, spacing=(2.0, 2.0, 0.0))
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.rigid_contact_max = 1024
    return model


def _selection(model, flags):
    """Construct under exactly ``flags`` and return the solver's path selection."""
    with patch.dict(os.environ):
        for name in [name for name in os.environ if name.startswith(environment_flags.PREFIXES)]:
            del os.environ[name]
        os.environ.update(flags)
        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_iterations=8,
            enable_joint_limits=True,
            friction_anchor_beta=0.0,
        )
    return solver.get_path_selection()


class TestFeatherPGSRetiredExperiments(unittest.TestCase):
    """Keep removed switches from selecting anything: the record lists them as unknown names."""

    def test_removed_switches_run_the_general_route_cpu(self):
        """Construct a recipe's removed switches on the general route, list them as unknown and warn about none."""
        model = _humanoids("cpu")
        plain = _selection(model, {})
        flags = dict.fromkeys(RETIRED_FLAGS + PACKED_ROUTE_FLAGS, "1")
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            recipe = _selection(model, flags)
        self.assertEqual([str(item.message) for item in seen if "FEATHER_PGS_" in str(item.message)], [])
        self.assertEqual(
            recipe["environment"], {"applied": {}, "unknown": sorted(flags), "retired": [], "isaac_lab": {}}
        )
        self.assertNotIn("research_sparse_factor", recipe["routes"])
        # Everything but the environment record matches a construction without the flags.
        del plain["environment"], recipe["environment"]
        self.assertEqual(recipe, plain)


if __name__ == "__main__":
    unittest.main()
