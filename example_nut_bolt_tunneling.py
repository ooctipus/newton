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
    NIST_ASSET_DIR,
    THREAD_PITCH,
    _load_mesh,
    _load_usd_mesh,
    _press,
)

GRAVITY_Z = -9.81

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

        # Bare size (m16) = the task's own USD; m16_tight = the IsaacGymEnvs mesh.
        self.gap = args.gap if args.gap > 0 else 0.4 * self.pitch
        if "_" not in args.assembly:
            bolt_file = NIST_ASSET_DIR / f"bolt_{args.assembly}.usd"
            nut_file = NIST_ASSET_DIR / f"nut_{args.assembly}.usd"
            loader = _load_usd_mesh
        else:
            asset_dir = newton.examples.download_external_git_folder(
                ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER
            )
            bolt_file = str(asset_dir / f"factory_bolt_{args.assembly}.obj")
            nut_file = str(asset_dir / f"factory_nut_{args.assembly}_subdiv_3x.obj")
            loader = _load_mesh
        bolt_mesh, _, bolt_extent = loader(bolt_file, self.gap, args.sdf_resolution)
        nut_mesh, _, nut_extent = loader(nut_file, self.gap, args.sdf_resolution)

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = self.gap

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
                gap=self.gap,
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
        self.nut_mass = np.array([float(self.model.body_mass.numpy()[b]) for b in self.nut_bodies])
        self.prev_vz = np.zeros(len(self.nut_bodies))
        self.applied_f = np.zeros(len(self.nut_bodies))
        self.reaction_f = np.zeros(len(self.nut_bodies))
        self.peak_reaction = np.zeros(len(self.nut_bodies))
        # Arrow lengths: the effort ceiling maps to `arrow_span` metres so the
        # picture stays readable whatever load is commanded.
        self.force_scale = args.arrow_span / max(args.max_force, 1.0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.0, -0.22, 0.09), pitch=-12.0, yaw=90.0
        )
        src = "task USD" if "_" not in args.assembly else "IsaacGymEnvs mesh"
        print(f"pressing {args.assembly} ({src}) at {args.max_force:.0f} N, "
              f"pitch {self.pitch * 1000:.2f} mm, gap {self.gap * 1000:.2f} mm")
        print(f"{'frame':>6} " + " ".join(f"{lbl:>24}" for lbl in self.labels))

    def _z(self) -> np.ndarray:
        q = self.state_0.body_q.numpy()
        return np.array([q[b][2] for b in self.nut_bodies])

    def _vz(self) -> np.ndarray:
        qd = self.state_0.body_qd.numpy()
        return np.array([qd[b][2] for b in self.nut_bodies])

    def _xy(self) -> np.ndarray:
        q = self.state_0.body_q.numpy()
        return np.array([[q[b][0], q[b][1]] for b in self.nut_bodies])

    def step(self):
        driving = self.frame >= self.settle_frames
        self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            # Mouse picking in the GL viewer: grab a nut and yank it around
            # while the press keeps pushing, to probe the contact by hand.
            self.viewer.apply_forces(self.state_0)
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

        # Contacts carry no force, so take the reaction from the momentum
        # balance over the frame: m*dv/dt = F_drive + m*g + F_contact.
        vz = self._vz()
        accel = (vz - self.prev_vz) / self.frame_dt
        self.applied_f = np.full(len(self.nut_bodies), -self.args.max_force if driving else 0.0)
        self.reaction_f = self.nut_mass * accel - self.applied_f - self.nut_mass * GRAVITY_Z
        self.prev_vz = vz
        if driving:
            self.peak_reaction = np.maximum(self.peak_reaction, np.abs(self.reaction_f))

        if self.frame == self.settle_frames - 1:
            self.rest_z = self._z().copy()
        if self.rest_z is not None and self.frame % 10 == 0:
            pen = (self.rest_z - self._z()) / self.pitch
            cells = " ".join(f"{p:>+11.2f}p {r:>+10.0f}N" for p, r in zip(pen, self.reaction_f))
            print(f"{self.frame:>6} {cells}", flush=True)

        self.frame += 1
        self.sim_time += self.frame_dt

    def _arrows(self, values: np.ndarray, base_dz: float) -> tuple[wp.array, wp.array]:
        """One vertical arrow per nut, signed length proportional to `values` [N]."""
        xy, z = self._xy(), self._z()
        starts, ends = [], []
        for i in range(len(self.nut_bodies)):
            base = wp.vec3(float(xy[i][0]), float(xy[i][1]), float(z[i] + base_dz))
            tip = wp.vec3(base[0], base[1], float(base[2] + values[i] * self.force_scale))
            starts.append(base)
            ends.append(tip)
        return wp.array(starts, dtype=wp.vec3), wp.array(ends, dtype=wp.vec3)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)

        # Red, pointing down: what the arm is pushing with (identical per column).
        starts, ends = self._arrows(self.applied_f, 0.030)
        self.viewer.log_arrows("force/applied", starts, ends, (1.0, 0.15, 0.15))

        # Blue, proportional to descent rate. This is the arrow that separates
        # the columns: at steady state the reaction balances the applied load
        # whether the nut is held or creeping, so only motion tells them apart.
        starts, ends = self._arrows(self._vz() * self.args.vel_arrow_gain / self.force_scale, 0.024)
        self.viewer.log_arrows("force/velocity", starts, ends, (0.25, 0.5, 1.0))

        # Green while the thread is holding, red once the nut has sunk a pitch:
        # what the contact is actually pushing back with.
        pen = (self.rest_z - self._z()) / self.pitch if self.rest_z is not None else np.zeros(len(self.nut_bodies))
        colors = wp.array(
            [wp.vec3(1.0, 0.2, 0.2) if p > 1.0 else wp.vec3(0.2, 1.0, 0.3) for p in pen], dtype=wp.vec3
        )
        starts, ends = self._arrows(self.reaction_f, 0.018)
        self.viewer.log_arrows("force/reaction", starts, ends, colors)

        self.viewer.end_frame()

    def test_final(self):
        pen = (self.rest_z - self._z()) / self.pitch
        print("\nsummary")
        for lbl, p_, f_ in zip(self.labels, pen, self.peak_reaction):
            state = "TUNNELED" if p_ > 1.0 else "held"
            print(f"  {lbl:<24} {p_:>+7.2f} pitch   peak reaction {f_:>7.0f} N   {state}")

    @staticmethod
    def create_parser():
        p = newton.examples.create_parser()
        p.add_argument("--assembly", type=str, default="m16",
                       help="m4/m8/m12/m16 = task USD; m16_tight etc = IsaacGymEnvs mesh.")
        p.add_argument("--collide-hz", type=float, default=200.0, help="Collision (frame) rate.")
        p.add_argument("--substeps", type=int, default=16, help="Solver substeps per frame.")
        p.add_argument("--max-force", type=float, default=400.0, help="Arm effort ceiling [N].")
        p.add_argument("--press-kp", type=float, default=5.0e4)
        p.add_argument("--press-kd", type=float, default=5.0e2)
        p.add_argument("--mu", type=float, default=0.75)
        p.add_argument("--cone", type=str, default="pyramidal", choices=["pyramidal", "elliptic"])
        p.add_argument("--gap", type=float, default=0.0, help="0 = auto (0.4 x thread pitch).")
        p.add_argument("--clearance", type=float, default=0.001)
        p.add_argument("--settle-frames", type=int, default=10)
        p.add_argument("--sdf-resolution", type=int, default=512)
        p.add_argument("--arrow-span", type=float, default=0.03,
                       help="Metres an arrow spans at the effort ceiling [m].")
        p.add_argument("--vel-arrow-gain", type=float, default=0.25,
                       help="Metres of blue arrow per m/s of nut descent.")
        return p


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
