# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Nut/bolt tunneling, side by side
#
# Presses identical nuts onto identical fixed bolts, one pair per contact
# stiffness, so a config that holds and a config that sinks through the thread
# run in the same scene under the same load.
#
# Nothing rotates the nuts, so any descent is the solver letting steel pass
# through steel.
#
#   python example_nut_bolt_tunneling.py                  # interactive
#   python example_nut_bolt_tunneling.py --viewer usd --output-path out.usd
#   python example_nut_bolt_tunneling.py --assembly m4_tight --max-force 800
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from nut_bolt_tunneling import (
    ISAACGYM_ENVS_REPO_URL,
    ISAACGYM_NUT_BOLT_FOLDER,
    THREAD_PITCH,
    _load_mesh,
    _press,
)

# (label, ke, kd) -- kd = solver_hz and ke = (kd/2)^2 is the critically damped
# contact that a 3200 Hz solver can actually hold; the others bracket it.
VARIANTS = (
    ("soft (newton default)", 2500.0, 100.0),
    ("task ke=2.56e6", 2.56e6, 3200.0),
    ("stiff ke=1e7", 1.0e7, 1.0e4),
)
SPACING = 0.06  # [m] between pairs


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.fps = args.collide_hz
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0

        self.viewer = viewer
        self.args = args
        self.pitch = THREAD_PITCH[args.assembly.split("_")[0]]
        self.settle_frames = args.settle_frames

        asset_dir = newton.examples.download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)
        bolt_file = str(asset_dir / f"factory_bolt_{args.assembly}.obj")
        nut_file = str(asset_dir / f"factory_nut_{args.assembly}_subdiv_3x.obj")
        bolt_mesh, _, bolt_extent = _load_mesh(bolt_file, args.gap, args.sdf_resolution)
        nut_mesh, _, nut_extent = _load_mesh(nut_file, args.gap, args.sdf_resolution)

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = args.gap

        self.bolt_top_z = float(bolt_extent[2])
        self.nut_start_z = float(self.bolt_top_z + nut_extent[2] * 0.5 + args.clearance)
        self.labels, self.nut_bodies = [], []

        for i, (label, ke, kd) in enumerate(VARIANTS):
            x = (i - (len(VARIANTS) - 1) / 2.0) * SPACING
            cfg = newton.ModelBuilder.ShapeConfig(
                margin=0.0,
                mu=args.mu,  # keep this well above 0 while cone is pyramidal
                ke=ke,
                kd=kd,
                gap=args.gap,
                density=8000.0,
                mu_torsional=0.0,
                mu_rolling=0.0,
                is_hydroelastic=False,
            )
            builder.add_shape_mesh(
                -1,  # static: this measures nut-through-thread, not bolt shove
                xform=wp.transform(wp.vec3(x, 0.0, float(bolt_extent[2] / 2.0)), wp.quat_identity()),
                mesh=bolt_mesh,
                cfg=cfg,
                label=f"bolt_{i}",
            )
            body = builder.add_body(
                label=f"nut_{i}",
                xform=wp.transform(wp.vec3(x, 0.0, self.nut_start_z), wp.quat_identity()),
            )
            builder.add_shape_mesh(body, mesh=nut_mesh, cfg=cfg, label=f"nut_shape_{i}")
            self.labels.append(label)
            self.nut_bodies.append(body)

        self.model = builder.finalize()
        self.model.rigid_contact_max = 40000
        self.collision_pipeline = newton.CollisionPipeline(
            self.model, reduce_contacts=True, rigid_contact_max=40000, broad_phase="sap"
        )
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone=args.cone,
            njmax=8000,
            nconmax=8000,
            iterations=15,
            ls_iterations=100,
            impratio=1.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.collision_pipeline.contacts()
        self.peak = wp.zeros(1, dtype=wp.float32)
        self.rest_z = None

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.0, -0.22, 0.09), pitch=-12.0, yaw=90.0
        )
        print(f"pressing {args.assembly} at {args.max_force:.0f} N, pitch {self.pitch * 1000:.2f} mm")
        print(f"{'frame':>6} " + " ".join(f"{lbl:>24}" for lbl in self.labels))

    def _z(self) -> np.ndarray:
        q = self.state_0.body_q.numpy()
        return np.array([q[b][2] for b in self.nut_bodies])

    def step(self):
        driving = self.frame >= self.settle_frames
        self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if driving:
                for body in self.nut_bodies:
                    wp.launch(
                        _press,
                        dim=1,
                        inputs=[
                            self.state_0.body_q,
                            self.state_0.body_qd,
                            self.state_0.body_f,
                            self.peak,
                            body,
                            0.0,  # target: the bolt base
                            self.args.press_kp,
                            self.args.press_kd,
                            self.args.max_force,
                        ],
                    )
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        if self.frame == self.settle_frames - 1:
            self.rest_z = self._z().copy()
        if self.rest_z is not None and self.frame % 10 == 0:
            pen = (self.rest_z - self._z()) / self.pitch
            cells = " ".join(f"{p:>+24.2f}" for p in pen)
            print(f"{self.frame:>6} {cells}", flush=True)

        self.frame += 1
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        p = newton.examples.create_parser()
        p.add_argument("--assembly", type=str, default="m16_tight",
                       help="m4_tight / m8_tight / m12_tight / m16_tight (also _loose).")
        p.add_argument("--collide-hz", type=float, default=200.0, help="Collision (frame) rate.")
        p.add_argument("--substeps", type=int, default=16, help="Solver substeps per frame.")
        p.add_argument("--max-force", type=float, default=400.0, help="Arm effort ceiling [N].")
        p.add_argument("--press-kp", type=float, default=5.0e4)
        p.add_argument("--press-kd", type=float, default=5.0e2)
        p.add_argument("--mu", type=float, default=0.75)
        p.add_argument("--cone", type=str, default="pyramidal", choices=["pyramidal", "elliptic"])
        p.add_argument("--gap", type=float, default=0.005)
        p.add_argument("--clearance", type=float, default=0.001)
        p.add_argument("--settle-frames", type=int, default=10)
        p.add_argument("--sdf-resolution", type=int, default=512)
        return p


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
