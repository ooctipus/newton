# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reject removed research selectors and ignore the retired packed sparse-factor route's flags."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.core import environment_flags as core_flags
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
            pgs_mode="matrix_free" if model.device.is_cuda else "split",
            pgs_iterations=8,
            enable_joint_limits=True,
            friction_anchor_beta=0.0,
        )
    return solver.get_path_selection()


class TestFeatherPGSRetiredExperiments(unittest.TestCase):
    """Keep an obsolete recipe from silently selecting a different solver."""

    def test_solver_rejects_removed_flags_before_model_access(self):
        """Reject each retired selector before allocating or reading model data."""
        for flag in RETIRED_FLAGS:
            flags = dict.fromkeys(RETIRED_FLAGS, "0") | {flag: "1"}
            with self.subTest(flag=flag), patch.dict(os.environ, flags):
                with self.assertRaisesRegex(ValueError, flag + ".*retired"):
                    newton.solvers.SolverFeatherPGS(None)

    def test_packed_route_flags_run_the_general_route_cpu(self):
        """Construct a recipe's packed-route flags on the general route, record them as retired and warn about them."""
        model = _humanoids("cpu")
        plain = _selection(model, {})
        # The solver warns once per process about retired names; report them afresh.
        with (
            patch.object(core_flags, "_WARNED_RETIRED", set()),
            self.assertWarnsRegex(
                UserWarning,
                rf"^Ignoring retired environment variables that select nothing: {', '.join(sorted(PACKED_ROUTE_FLAGS))}\.",
            ),
        ):
            recipe = _selection(model, dict.fromkeys(PACKED_ROUTE_FLAGS, "1"))
        self.assertEqual(
            recipe["environment"],
            {"applied": {}, "unknown": [], "retired": sorted(PACKED_ROUTE_FLAGS), "isaac_lab": {}},
        )
        self.assertNotIn("research_sparse_factor", recipe["routes"])
        # Everything but the environment record matches a construction without the flags.
        del plain["environment"], recipe["environment"]
        self.assertEqual(recipe, plain)


if __name__ == "__main__":
    unittest.main()
