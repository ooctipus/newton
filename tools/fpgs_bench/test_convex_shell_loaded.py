# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Small loaded hull-pair adaptation of the existing two-body impact oracle.

The scene construction/step follows test_feather_pgs_restitution's two-body
helper, with actual finite hulls and their cube inertia instead of spheres.
No sphere point-count or one-row spin assertion is reused. The constructor
retains Allegro's 12 serial / eligible 24 parallel limits; actual routing is
recorded rather than assuming this small pair is parallel-eligible.
"""

import inspect
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton

DEVICE = os.environ.get("FPGS_TEST_DEVICE", "cpu")
DT = 1.0 / 240.0  # Two Newton substeps per Allegro sim_dt=1/120.
HALF = 0.05
RECORDS = []


def _model(device, restitution, friction):
    world = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=friction, restitution=restitution, margin=0.0, gap=0.001)
    mesh = newton.Mesh.create_box(
        HALF, HALF, HALF, duplicate_vertices=False, compute_normals=False, compute_uvs=False, compute_inertia=False
    )
    for mass, z in ((1.0, 0.0), (4.0, 2.0 * HALF + 0.0005)):
        inertia = (2.0 / 3.0) * mass * HALF * HALF
        body = world.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()),
            mass=mass,
            inertia=wp.mat33(inertia, 0.0, 0.0, 0.0, inertia, 0.0, 0.0, 0.0, inertia),
            lock_inertia=True,
        )
        world.add_shape_convex_hull(body=body, mesh=mesh, cfg=cfg)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_world(world)
    return builder.finalize(device=device)


def _scene(device, enabled, restitution, friction, tangent):
    model = _model(device, restitution, friction)
    # Both arms instantiate the original split factory on this deliberately
    # tiny unit fixture. The actual live threshold is unchanged in production.
    with (
        patch.dict(
            os.environ,
            {
                "NEWTON_NARROW_PHASE_COHERENT_CONVEX": "reject_only",
                "NEWTON_NARROW_PHASE_CONVEX_BSP": "1",
                "NEWTON_NARROW_PHASE_COHERENT_SHELL": "1" if enabled else "0",
            },
        ),
        patch("newton._src.sim.collide._SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", 0),
    ):
        pipeline = newton.CollisionPipeline(
            model, broad_phase="explicit", rigid_contact_max=5, broad_phase_output_max=1
        )
    solver = newton.solvers.SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        pgs_iterations=12,
        pgs_velocity_iterations=0,
        mf_gs_parallel_rows=128,
        mf_gs_parallel_sweeps=24,
        mf_gs_parallel_matrix_free=True,
        grouped_dynamics=True,
        lazy_kinematics=True,
        dense_max_constraints=192,
        mf_max_constraints=64,
    )
    state_in, state_out = model.state(), model.state()
    qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    qd[0], qd[2], qd[8] = tangent, 2.0, -1.0
    state_in.joint_qd.assign(qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    return SimpleNamespace(
        model=model,
        pipeline=pipeline,
        solver=solver,
        states=(state_in, state_out),
        contacts=pipeline.contacts(),
        control=model.control(),
        enabled=enabled,
    )


def _impact(device, enabled, restitution, friction, tangent):
    scene = _scene(device, enabled, restitution, friction, tangent)
    initial = scene.states[0].body_qd.numpy().copy()
    initial_q = scene.states[0].joint_q.numpy().copy()
    # Warm query history without stepping, reading forces, or changing the
    # authored incoming pose/velocity. The last collide is CURRENT geometry.
    scene.pipeline.collide(scene.states[0], scene.contacts)
    scene.pipeline.collide(scene.states[0], scene.contacts)
    np.testing.assert_array_equal(scene.states[0].body_qd.numpy(), initial)
    np.testing.assert_array_equal(scene.states[0].joint_q.numpy(), initial_q)
    warm = 0
    if enabled:
        warm = int(np.count_nonzero(scene.pipeline.narrow_phase._coherent_cache.data.query_done.numpy() == 3))
        if warm != 1:
            raise AssertionError("Loaded fixture did not execute the new complete current shell owner")
    scene.states[0].clear_forces()
    scene.solver.step(scene.states[0], scene.states[1], scene.control, scene.contacts, DT)
    scene.solver.check_constraint_capacity()
    after = scene.states[1].body_qd.numpy().copy()
    masses = scene.model.body_mass.numpy().astype(float)
    inertia = scene.model.body_inertia.numpy().astype(float)
    momentum_before = (masses[:, None] * initial[:, :3]).sum(axis=0)
    momentum_after = (masses[:, None] * after[:, :3]).sum(axis=0)

    def energy(velocity):
        return float(
            0.5 * np.sum(masses[:, None] * velocity[:, :3] ** 2)
            + 0.5 * np.einsum("bi,bij,bj->", velocity[:, 3:], inertia, velocity[:, 3:])
        )

    count = int(scene.contacts.rigid_contact_count.numpy()[0])
    result = {
        "enabled": enabled,
        "restitution": restitution,
        "friction": friction,
        "tangent": tangent,
        "contacts": count,
        "warm_commits": warm,
        "velocity": after.tolist(),
        "momentum_defect": float(np.max(np.abs(momentum_after - momentum_before))),
        "energy_before": energy(initial),
        "energy_after": energy(after),
        "relative_normal_speed": float(after[1, 2] - after[0, 2]),
        "spin": float(np.linalg.norm(after[:, 3:], axis=1).max()),
        "paths": sorted({int(x) for x in scene.solver.contact_path.numpy()[:count] if x >= 0}),
        "serial_limit": scene.solver.pgs_iterations,
        "parallel_limit": scene.solver.mf_gs_parallel_sweeps,
        "parallel_rows": scene.solver.mf_gs_parallel_rows,
    }
    if not np.isfinite(after).all():
        raise AssertionError("Nonfinite loaded hull state")
    return result


class TestConvexShellLoaded(unittest.TestCase):
    def test_authored_hull_mass_and_inertia(self):
        """Use actual finite hulls with explicit nonzero original mass/inertia."""
        from newton._src.geometry.convex_bsp import _BspOwner  # noqa: PLC0415
        from newton._src.geometry.convex_shell import _ShellOwner  # noqa: PLC0415

        model = _model("cpu", 0.6, 0.5)
        np.testing.assert_allclose(model.body_mass.numpy(), [1.0, 4.0])
        self.assertTrue(np.all(model.shape_type.numpy() == int(newton.GeoType.CONVEX_MESH)))
        self.assertEqual(model.joint_dof_count, 12)
        self.assertEqual(model.world_count, 1)
        self.assertTrue(np.all(model.body_inertia.numpy()[:, (0, 1, 2), (0, 1, 2)] > 0.0))
        bsp = _BspOwner(model)
        shell = _ShellOwner(model, bsp)
        sources = set(map(int, model.shape_source_ptr.numpy()))
        self.assertTrue(sources)
        self.assertEqual(set(map(int, shell.data.mesh_ids.numpy())), sources)
        self.assertTrue(np.all(shell.data.point_count.numpy() == 8))
        self.assertIs(shell.data.valid, bsp.data.valid)
        # Both original matrix-free solve and the collision owner are
        # CUDA-only. Check the real constructor signature without executing
        # another solver mode or claiming a CPU dynamics qualification.
        solver_signature = inspect.signature(newton.solvers.SolverFeatherPGS)

        def checked_solver(model, **kwargs):
            solver_signature.bind(model, **kwargs)
            return SimpleNamespace(**kwargs)

        with (
            patch.object(newton, "CollisionPipeline", return_value=SimpleNamespace(contacts=lambda: None)),
            patch.object(newton.solvers, "SolverFeatherPGS", side_effect=checked_solver),
        ):
            scene = _scene("cpu", True, 0.6, 0.5, 0.4)
        self.assertEqual(scene.solver.pgs_iterations, 12)
        self.assertEqual(scene.solver.mf_gs_parallel_sweeps, 24)
        self.assertEqual(scene.solver.mf_gs_parallel_rows, 128)

    @unittest.skipUnless(
        DEVICE.startswith("cuda"), "The original matrix-free loaded solve requires the root CUDA lease"
    )
    def test_current_twelve_sweep_hull_impact(self):
        """Evaluate both original/current-shell loaded impacts before final physical assertions."""
        failures = []
        for restitution, friction, tangent in ((0.6, 0.5, 0.0), (0.0, 0.5, 0.4)):
            pair = [_impact(DEVICE, enabled, restitution, friction, tangent) for enabled in (False, True)]
            RECORDS.extend(pair)
            for record in pair:
                if record["momentum_defect"] > 2e-3:
                    failures.append(("momentum", record))
                if record["energy_after"] > record["energy_before"] * 1.005:
                    failures.append(("energy_injection", record))
                if not 1 <= record["contacts"] <= 5 or not record["paths"]:
                    failures.append(("missing_or_overflowed_contacts", record))
                if tangent == 0.0 and abs(record["relative_normal_speed"] - restitution * 3.0) > 5e-3:
                    failures.append(("normal_impact_law", record))
            # Full kinetic energy and spin explicitly expose changed patch
            # torque; never compare per-point impulses or assume count one.
            if pair[1]["energy_after"] > pair[0]["energy_after"] + 5e-3 * pair[0]["energy_before"]:
                failures.append(("changed_patch_energy", pair))
            if pair[1]["spin"] > pair[0]["spin"] + 0.05:
                failures.append(("changed_patch_spin", pair))
        print("LOADED_CURRENT_SHELL", RECORDS)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
