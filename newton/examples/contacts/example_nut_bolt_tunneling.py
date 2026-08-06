# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Nut/Bolt Tunneling
#
# Seats nuts partway down a fixed bolt and presses them axially, one pair per
# contact stiffness, so a setting that holds the thread and one that lets the
# nut sink through run in the same scene under the same load.
#
# Nothing rotates the nuts once seated, so any descent is the solver letting
# steel pass through steel.
#
# Command: python -m newton.examples nut_bolt_tunneling
#
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples

# Contact stiffness under test, one nut/bolt pair each.
#   ke/kd reach MuJoCo as geom_solref = (2/kd, (kd/2)/sqrt(ke)), and REFSAFE
#   clamps the time constant to >= 2*dt. At a 3200 Hz solver, kd=3200 sits
#   exactly at that limit -- the stiffest contact the rate can represent.
VARIANTS = (
    ("soft (newton default)", 2500.0, 100.0),
    ("kd = solver rate", 2.56e6, 3200.0),
    ("stiffer than the rate", 1.0e7, 1.0e4),
)
SPACING = 0.06  # [m] between pairs

THREAD_PITCH = {"m4": 0.0007, "m8": 0.00125, "m12": 0.00175, "m16": 0.002}

# Assembly profile per size, matching the nut-threading reset:
#   (fully-screwed seat z, bolt tip z, screw ratio, nut align offset z) [m]
# Position runs from the seat to the tip as the fraction goes 0 -> 1, and the
# nut is rotated about z by distance/(ratio/2pi) * fraction, putting it at the
# screw phase that height implies. Seating by height alone leaves the internal
# thread meeting the external one at an arbitrary phase, and the solver then has
# to either jam the nut or eject it.
ASSEMBLY_PROFILE = {
    "m16": (0.022, 0.035, 0.002, 0.010),
    "m12": (0.0218, 0.035, 0.00175, 0.013),
    "m8": (0.018, 0.026, 0.00125, 0.0093),
    "m4": (0.01318, 0.020, 0.0007, 0.0048),
}


def assembly_pose(size: str, fraction: float) -> tuple[float, float]:
    """Nut root height [m] and z rotation [rad] at ``fraction`` along the assembly.

    Args:
        size: Thread size key into :data:`ASSEMBLY_PROFILE`.
        fraction: 0 seats the nut fully screwed down, 1 leaves it at the bolt tip.

    Returns:
        Nut root height [m] and its rotation about the bolt axis [rad].
    """
    seat_z, tip_z, screw_ratio, align_z = ASSEMBLY_PROFILE[size]
    distance = tip_z - seat_z
    yaw = fraction * distance / (screw_ratio / (2.0 * math.pi))
    # The align offset and the rotation are both about z, so z is unaffected.
    return seat_z + fraction * distance - align_z, yaw


def load_collider(usd_path: str, gap: float, resolution: int, narrow_band: float) -> newton.Mesh:
    """Build an SDF mesh from the collision geometry authored in a USD asset.

    The mesh keeps its authored frame: the assembly profile is expressed in that
    frame, so re-centering on the bounding box would make it meaningless.

    Args:
        usd_path: Asset path.
        gap: Contact detection distance [m], used as the SDF margin.
        resolution: Maximum SDF resolution.
        narrow_band: SDF narrow-band half width [m].

    Returns:
        Mesh carrying the cooked SDF.
    """
    from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    stage = Usd.Stage.Open(usd_path)
    prim = next((p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and UsdPhysics.CollisionAPI(p)), None)
    if prim is None:
        raise ValueError(f"no collision mesh in {usd_path}")

    mesh = UsdGeom.Mesh(prim)
    xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    points = np.array([xform.Transform(p) for p in mesh.GetPointsAttr().Get()], dtype=np.float32)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

    triangles, at = [], 0
    for count in counts:  # fan-triangulate anything that is not already a triangle
        face = indices[at : at + count]
        triangles.extend([face[0], face[i], face[i + 1]] for i in range(1, count - 1))
        at += count

    out = newton.Mesh(points, np.asarray(triangles, dtype=np.int32).flatten())
    out.build_sdf(max_resolution=resolution, narrow_band_range=(-narrow_band, narrow_band), margin=gap)
    return out


@wp.kernel
def press_nut(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    peak: wp.array(dtype=wp.float32),
    nut: int,
    target_z: float,
    kp: float,
    kd: float,
    max_force: float,
):
    """Load the nut the way an arm does: a position source with an effort ceiling.

    A raw force on a 30 g free body is unusable -- 4 kN is a 37 m/s velocity jump
    in one substep, which blows the solver up before any contact resolves.
    """
    z = wp.transform_get_translation(body_q[nut])[2]
    vz = wp.spatial_top(body_qd[nut])[2]
    f = wp.clamp(kp * (target_z - z) - kd * vz, -max_force, max_force)
    wp.atomic_add(body_f, nut, wp.spatial_vector(wp.vec3(0.0, 0.0, f), wp.vec3(0.0)))
    wp.atomic_max(peak, 0, wp.abs(f))


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / args.collide_hz
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0

        self.pitch = THREAD_PITCH[args.assembly]
        self.gap = args.gap if args.gap > 0 else 0.4 * self.pitch
        # A +/-5 mm band is about seven pitches of an m4 thread; scale it.
        band = args.narrow_band if args.narrow_band > 0 else 2.5 * self.pitch
        nut_z, nut_yaw = assembly_pose(args.assembly, args.assembly_fraction)

        bolt_mesh = load_collider(
            newton.examples.get_asset(f"nist_bolt_{args.assembly}.usd"), self.gap, args.sdf_resolution, band
        )
        nut_mesh = load_collider(
            newton.examples.get_asset(f"nist_nut_{args.assembly}.usd"), self.gap, args.sdf_resolution, band
        )

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = self.gap
        self.labels, self.nut_bodies = [], []

        for i, (label, ke, kd) in enumerate(VARIANTS):
            x = (i - (len(VARIANTS) - 1) / 2.0) * SPACING
            shape_cfg = newton.ModelBuilder.ShapeConfig(
                margin=0.0,
                # Keep friction well above zero while the cone is pyramidal: mu -> 0
                # zeroes the pyramidal row invweight, efc_D floors near 1e15, and the
                # float32 Hessian degenerates to NaN on contact-rich states.
                mu=args.mu,
                ke=ke,
                kd=kd,
                gap=self.gap,
                density=args.density,
                mu_torsional=0.0,
                mu_rolling=0.0,
                is_hydroelastic=False,
            )
            # Bolt is static: this measures nut-through-thread, not bolt shove.
            builder.add_shape_mesh(
                -1,
                xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                mesh=bolt_mesh,
                cfg=shape_cfg,
                label=f"bolt_{i}",
            )
            body = builder.add_body(
                label=f"nut_{i}",
                xform=wp.transform(
                    wp.vec3(x, 0.0, nut_z),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), nut_yaw),
                ),
            )
            builder.add_shape_mesh(body, mesh=nut_mesh, cfg=shape_cfg, label=f"nut_shape_{i}")
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
        self.mass = np.array([float(self.model.body_mass.numpy()[b]) for b in self.nut_bodies])
        self.prev_vz = np.zeros(len(self.nut_bodies))
        self.applied = np.zeros(len(self.nut_bodies))
        self.reaction = np.zeros(len(self.nut_bodies))
        self.peak_reaction = np.zeros(len(self.nut_bodies))
        self.rest_z = None
        self.force_scale = args.arrow_span / max(args.max_force, 1.0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, -0.16, 0.05), pitch=-10.0, yaw=90.0)
        print(
            f"{args.assembly} at assembly fraction {args.assembly_fraction:.2f} "
            f"({nut_yaw / (2 * math.pi):.2f} turns on), pressing {args.max_force:.0f} N, "
            f"pitch {self.pitch * 1000:.2f} mm, gap {self.gap * 1000:.2f} mm"
        )
        print(f"{'frame':>6} " + " ".join(f"{lbl:>24}" for lbl in self.labels))

    def _z(self) -> np.ndarray:
        q = self.state_0.body_q.numpy()
        return np.array([q[b][2] for b in self.nut_bodies])

    def _vz(self) -> np.ndarray:
        qd = self.state_0.body_qd.numpy()
        return np.array([qd[b][2] for b in self.nut_bodies])

    def step(self):
        driving = self.frame >= self.args.settle_frames
        self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)  # mouse picking in the GL viewer
            if driving:
                for body in self.nut_bodies:
                    wp.launch(
                        press_nut,
                        dim=1,
                        inputs=[
                            self.state_0.body_q,
                            self.state_0.body_qd,
                            self.state_0.body_f,
                            self.peak,
                            body,
                            0.0,  # drive toward the bolt base
                            self.args.press_kp,
                            self.args.press_kd,
                            self.args.max_force,
                        ],
                    )
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # Contacts carry no force, so take the reaction from the frame's momentum
        # balance: m*dv/dt = F_drive + m*g + F_contact.
        vz = self._vz()
        self.applied = np.full(len(self.nut_bodies), -self.args.max_force if driving else 0.0)
        self.reaction = self.mass * (vz - self.prev_vz) / self.frame_dt - self.applied + self.mass * 9.81
        self.prev_vz = vz
        if driving:
            self.peak_reaction = np.maximum(self.peak_reaction, np.abs(self.reaction))

        if self.frame == self.args.settle_frames - 1:
            self.rest_z = self._z().copy()
        if self.rest_z is not None and self.frame % 10 == 0:
            pen = (self.rest_z - self._z()) / self.pitch
            cells = " ".join(f"{p:>+11.2f}p {r:>+10.0f}N" for p, r in zip(pen, self.reaction))
            print(f"{self.frame:>6} {cells}", flush=True)

        self.frame += 1
        self.sim_time += self.frame_dt

    def _arrows(self, values: np.ndarray, base_dz: float):
        z = self._z()
        q = self.state_0.body_q.numpy()
        starts, ends = [], []
        for i, body in enumerate(self.nut_bodies):
            base = wp.vec3(float(q[body][0]), float(q[body][1]), float(z[i] + base_dz))
            starts.append(base)
            ends.append(wp.vec3(base[0], base[1], float(base[2] + values[i] * self.force_scale)))
        return wp.array(starts, dtype=wp.vec3), wp.array(ends, dtype=wp.vec3)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)

        # Red: the load being applied. Same in every column.
        starts, ends = self._arrows(self.applied, 0.022)
        self.viewer.log_arrows("force/applied", starts, ends, (1.0, 0.15, 0.15))

        # Blue: descent rate. At steady state the reaction balances the load
        # whether the nut is held or creeping, so motion is what separates them.
        starts, ends = self._arrows(self._vz() * self.args.vel_arrow_gain / self.force_scale, 0.016)
        self.viewer.log_arrows("force/velocity", starts, ends, (0.25, 0.5, 1.0))

        # Green until the nut has sunk a pitch, then red: the contact reaction.
        pen = (self.rest_z - self._z()) / self.pitch if self.rest_z is not None else np.zeros(len(self.nut_bodies))
        colors = wp.array(
            [wp.vec3(1.0, 0.2, 0.2) if p > 1.0 else wp.vec3(0.2, 1.0, 0.3) for p in pen], dtype=wp.vec3
        )
        starts, ends = self._arrows(self.reaction, 0.010)
        self.viewer.log_arrows("force/reaction", starts, ends, colors)

        self.viewer.end_frame()

    def test_final(self):
        """An unrotated nut cannot legally descend; a pitch means it ate into the bolt."""
        pen = (self.rest_z - self._z()) / self.pitch
        print("\nsummary")
        for label, p, f in zip(self.labels, pen, self.peak_reaction):
            # A diverged run must never read as a pass: `nan > 1.0` is False, so
            # testing the threshold alone silently reports NaN as "held".
            if not np.isfinite(p):
                verdict, shown = "DIVERGED", "     nan"
            else:
                verdict, shown = ("TUNNELED" if p > 1.0 else "held"), f"{p:>+8.2f}"
            print(f"  {label:<24} {shown} pitch   peak reaction {f:>7.0f} N   {verdict}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--assembly", type=str, default="m16", choices=sorted(THREAD_PITCH))
        parser.add_argument("--assembly-fraction", type=float, default=0.5,
                            help="0 = fully screwed down, 1 = at the bolt tip.")
        parser.add_argument("--max-force", type=float, default=400.0, help="Arm effort ceiling [N].")
        parser.add_argument("--collide-hz", type=float, default=200.0, help="Collision (frame) rate [Hz].")
        parser.add_argument("--substeps", type=int, default=16, help="Solver substeps per frame.")
        parser.add_argument("--mu", type=float, default=0.75)
        parser.add_argument("--cone", type=str, default="pyramidal", choices=["pyramidal", "elliptic"])
        parser.add_argument("--gap", type=float, default=0.0, help="0 = auto (0.4 x thread pitch) [m].")
        parser.add_argument("--narrow-band", type=float, default=0.0, help="0 = auto (2.5 x pitch) [m].")
        parser.add_argument("--sdf-resolution", type=int, default=512)
        parser.add_argument("--density", type=float, default=8000.0)
        parser.add_argument("--settle-frames", type=int, default=10)
        parser.add_argument("--press-kp", type=float, default=5.0e4)
        parser.add_argument("--press-kd", type=float, default=5.0e2)
        parser.add_argument("--arrow-span", type=float, default=0.02)
        parser.add_argument("--vel-arrow-gain", type=float, default=0.25)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
