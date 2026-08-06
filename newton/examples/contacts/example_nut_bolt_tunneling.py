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
import time

import numpy as np
import warp as wp

import newton
import newton.examples

# One nut/bolt pair per (ke, kd) cell, laid out as a grid so the whole contact
# parameter space is screened in a single run.
#
#   ke/kd reach MuJoCo as geom_solref = (2/kd, (kd/2)/sqrt(ke)), and REFSAFE
#   clamps the time constant to >= 2*dt. A 3200 Hz solver therefore cannot
#   represent anything stiffer than kd = 3200; asking for more silently yields a
#   clamped contact. The grid makes that ceiling visible as a boundary rather
#   than something to be taken on faith.
KE_RANGE = (1.0e3, 1.0e8)
KD_RANGE = (5.0e1, 2.0e4)
SPACING = 0.05  # [m] between pairs; override with --spacing

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

# `fully_screwed_nut_offset` is well up the shaft -- 21.8 mm on a 32 mm m12 bolt --
# so fraction 0 seats the nut flush with the tip, not down at the head. The
# keypoint that means "bottom of the thread" is `full_thread`, and starting there
# sweeps the nut along the whole shaft instead of its top ~13 mm. m12 authors no
# `full_thread`, so its `head` is used. This departs from the task's own reset,
# which starts at `fully_screwed_nut_offset`; --profile-start selects.
FULL_THREAD = {"m16": 0.010, "m12": 0.0, "m8": 0.0084, "m4": 0.0044}


def assembly_pose(size: str, fraction: float, start: str = "fully_screwed") -> tuple[float, float]:
    """Nut root height [m] and z rotation [rad] at ``fraction`` along the assembly.

    Args:
        size: Thread size key into :data:`ASSEMBLY_PROFILE`.
        fraction: 0 seats the nut fully screwed down, 1 leaves it at the bolt tip.

    Returns:
        Nut root height [m] and its rotation about the bolt axis [rad].
    """
    seat_z, tip_z, screw_ratio, align_z = ASSEMBLY_PROFILE[size]
    if start == "full_thread":
        seat_z = FULL_THREAD[size]
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
def press_nuts(
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    nut_index: wp.array(dtype=wp.int32),
    force: float,
    damping: float,
):
    """Press the nut down at a fixed force, with velocity feedback.

    This was a PD toward the bolt base with an effort ceiling, but the position
    term only ever decided whether the ceiling was reached: above roughly 3e4
    N/m it saturated and the drive was exactly `force`, and below that it quietly
    delivered less, so the number on the command line was not the load applied.
    Damping stays -- it measurably changes which cells hold.
    """
    nut = nut_index[wp.tid()]
    vz = wp.spatial_top(body_qd[nut])[2]
    f = wp.clamp(-force - damping * vz, -force, 0.0)
    wp.atomic_add(body_f, nut, wp.spatial_vector(wp.vec3(0.0, 0.0, f), wp.vec3(0.0)))


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

        self.spacing = args.spacing if args.spacing > 0 else SPACING
        self.side = args.grid
        self.count = self.side * self.side
        # Log-spaced so each axis spans decades rather than crowding the top end.
        self.ke_axis = np.geomspace(*KE_RANGE, self.side)
        self.kd_axis = np.geomspace(*KD_RANGE, self.side)
        self.pitch = THREAD_PITCH[args.assembly]
        self.gap = args.gap if args.gap > 0 else 0.4 * self.pitch
        # A +/-5 mm band is about seven pitches of an m4 thread; scale it.
        band = args.narrow_band if args.narrow_band > 0 else 2.5 * self.pitch
        nut_z, nut_yaw = assembly_pose(args.assembly, args.assembly_fraction, args.profile_start)

        bolt_mesh = load_collider(
            newton.examples.get_asset(f"nist_bolt_{args.assembly}.usd"), self.gap, args.sdf_resolution, band
        )
        nut_mesh = load_collider(
            newton.examples.get_asset(f"nist_nut_{args.assembly}.usd"), self.gap, args.sdf_resolution, band
        )

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = self.gap
        self.nut_bodies, self.cell_ke, self.cell_kd = [], [], []

        # One world per cell. Sharing a single world is fatal here: the MuJoCo
        # solve couples every body through one linear system, so the first cell
        # whose contact diverges takes the whole grid to NaN with it -- which is
        # exactly the sweep's job to survive.
        for cell in range(self.count):
            row, col = divmod(cell, self.side)  # row -> ke, col -> kd
            ke, kd = float(self.ke_axis[row]), float(self.kd_axis[col])
            x = (col - (self.side - 1) / 2.0) * self.spacing
            y = (row - (self.side - 1) / 2.0) * self.spacing
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
            cell_builder = newton.ModelBuilder()
            cell_builder.default_shape_cfg.gap = self.gap
            # Bolt is static: this measures nut-through-thread, not bolt shove.
            cell_builder.add_shape_mesh(
                -1,
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                mesh=bolt_mesh,
                cfg=shape_cfg,
                label="bolt",
            )
            body = cell_builder.add_body(
                label="nut",
                xform=wp.transform(
                    wp.vec3(0.0, 0.0, nut_z),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), nut_yaw),
                ),
            )
            cell_builder.add_shape_mesh(body, mesh=nut_mesh, cfg=shape_cfg, label="nut_shape")

            builder.begin_world()
            builder.add_builder(
                cell_builder,
                xform=wp.transform(wp.vec3(x, y, 0.0), wp.quat_identity()),
                label_prefix=f"c{cell}_",
            )
            builder.end_world()
            self.nut_bodies.append(len(self.nut_bodies))  # one body per world, in order
            self.cell_ke.append(ke)
            self.cell_kd.append(kd)

        self.model = builder.finalize()
        # Headroom matters more than memory here: an overflowing world silently
        # drops constraints, and a cell that holds only because its contacts were
        # discarded is indistinguishable from one that genuinely held.
        per_world = args.per_world_contacts
        budget = per_world * self.count
        self.model.rigid_contact_max = budget
        self.collision_pipeline = newton.CollisionPipeline(
            self.model, reduce_contacts=True, rigid_contact_max=budget, broad_phase="sap"
        )
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone=args.cone,
            njmax=per_world,
            nconmax=per_world,
            iterations=15,
            ls_iterations=100,
            impratio=1.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.collision_pipeline.contacts()

        # joint_q is the source of truth for a free body; writing body_q is
        # overwritten by the next eval_fk.
        self.q0 = self.state_0.joint_q.numpy().copy()
        self.qd0 = self.state_0.joint_qd.numpy().copy()
        self.episode = 0
        self.episode_frame = 0
        self.fps = 0.0
        self._clock = time.perf_counter()
        self.nut_index = wp.array(np.asarray(self.nut_bodies, dtype=np.int32), dtype=wp.int32)
        self.marker_z = float(nut_z + 0.03)
        self.rest_z = None
        self.worst = np.zeros(self.count)  # deepest penetration seen, in pitches
        self._panel = None  # matplotlib figure, built lazily on first render

        self.viewer.set_model(self.model)
        # Each cell is its own world, and the viewer spreads worlds apart on its
        # own unless told not to. The grid position already encodes (ke, kd), so
        # any extra spacing would just detach the picture from the layout.
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        span = self.side * self.spacing
        self.viewer.set_camera(pos=wp.vec3(0.0, -0.75 * span, 0.55 * span), pitch=-35.0, yaw=90.0)
        # How much of the nut is actually on the bolt. The assembly range spans
        # roughly one nut height, so a fraction step moves the nut only a few mm
        # -- invisible on a grid this wide, but decisive for whether it holds.
        bz = np.asarray(bolt_mesh.vertices)[:, 2]
        nz = np.asarray(nut_mesh.vertices)[:, 2]
        overlap = min(nut_z + nz.max(), bz.max()) - max(nut_z + nz.min(), bz.min())
        print(
            f"{args.assembly}, {self.side}x{self.side} = {self.count} pairs, "
            f"assembly fraction {args.assembly_fraction:.2f} ({nut_yaw / (2 * math.pi):.2f} turns on), "
            f"pressing {args.max_force:.0f} N, pitch {self.pitch * 1000:.2f} mm"
        )
        print(
            f"nut seated at z={nut_z * 1000:.2f} mm with {overlap * 1000:.2f} mm of "
            f"{(nz.max() - nz.min()) * 1000:.2f} mm engaged (bolt tip {bz.max() * 1000:.2f} mm)"
        )
        print(f"ke {KE_RANGE[0]:.0e}..{KE_RANGE[1]:.0e} down rows, kd {KD_RANGE[0]:.0e}..{KD_RANGE[1]:.0e} across columns")

    def _z(self) -> np.ndarray:
        q = self.state_0.body_q.numpy()
        return np.array([q[b][2] for b in self.nut_bodies])

    def _vz(self) -> np.ndarray:
        qd = self.state_0.body_qd.numpy()
        return np.array([qd[b][2] for b in self.nut_bodies])

    def penetration(self) -> np.ndarray:
        """Descent past the seated height, in thread pitches, per cell."""
        if self.rest_z is None:
            return np.zeros(self.count)
        return (self.rest_z - self._z()) / self.pitch

    def reset(self):
        """Re-seat every nut and start a fresh episode.

        Restoring joint_q also rescues the diverged worlds: their state is NaN,
        and nothing short of overwriting it brings them back.
        """
        self.state_0.joint_q.assign(self.q0)
        self.state_0.joint_qd.assign(self.qd0)
        self.state_0.clear_forces()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.rest_z = None
        self.worst = np.zeros(self.count)
        self.episode += 1
        self.episode_frame = 0

    def step(self):
        if self.args.reset_interval > 0 and self.episode_frame >= self.args.reset_interval:
            held, gone, bad = self._tally(self.worst)
            print(f"  episode {self.episode}: held {held}  tunneled {gone}  diverged {bad}  -- reset", flush=True)
            self.reset()

        driving = self.episode_frame >= self.args.settle_frames
        self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)  # mouse picking in the GL viewer
            if driving:
                wp.launch(
                    press_nuts,
                    dim=self.count,
                    inputs=[
                        self.state_0.body_qd,
                        self.state_0.body_f,
                        self.nut_index,
                        self.args.max_force,
                        self.args.press_kd,
                    ],
                )
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        if self.episode_frame == self.args.settle_frames - 1:
            self.rest_z = self._z().copy()

        pen = self.penetration()
        # A diverged cell must not wash out the map: keep it as NaN and colour it
        # separately rather than letting it compare False against the threshold.
        self.worst = np.where(np.isnan(pen), np.nan, np.fmax(self.worst, pen))
        if driving and self.frame % 20 == 0:
            now = time.perf_counter()
            self.fps = 20.0 / max(now - self._clock, 1e-9)
            self._clock = now
            held, gone, bad = self._tally(self.worst)
            print(
                f"{self.frame:>6}  ep {self.episode}  held {held:>5}   tunneled {gone:>5}"
                f"   diverged {bad:>5}   {self.fps:6.1f} fps",
                flush=True,
            )

        self.frame += 1
        self.episode_frame += 1
        self.sim_time += self.frame_dt

    def _tally(self, pen: np.ndarray) -> tuple[int, int, int]:
        bad = int(np.isnan(pen).sum())
        gone = int(np.nansum(pen > 1.0))
        return self.count - gone - bad, gone, bad

    def _draw_panel(self) -> np.ndarray:
        """Render the sweep as a labelled heatmap, returned as an RGB image."""
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib.colors import ListedColormap, Normalize  # noqa: PLC0415

        if self._panel is None:
            fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=110)
            cmap = plt.get_cmap("RdYlGn_r").copy()
            cmap.set_bad("0.45")  # diverged cells
            image = ax.imshow(
                np.zeros((self.side, self.side)),
                origin="lower",
                aspect="auto",
                cmap=cmap,
                norm=Normalize(vmin=0.0, vmax=2.0),
                interpolation="nearest",
            )
            ticks = np.linspace(0, self.side - 1, min(6, self.side)).astype(int)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{self.kd_axis[t]:.0e}" for t in ticks], fontsize=7)
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{self.ke_axis[t]:.0e}" for t in ticks], fontsize=7)
            ax.set_xlabel("kd  [contact damping]", fontsize=8)
            ax.set_ylabel("ke  [contact stiffness]", fontsize=8)
            bar = fig.colorbar(image, ax=ax)
            bar.set_label("penetration [thread pitches]   grey = diverged", fontsize=7)
            bar.ax.tick_params(labelsize=7)
            self._panel = (fig, ax, image)

        fig, ax, image = self._panel
        grid = np.ma.masked_invalid(self.worst.reshape(self.side, self.side))
        image.set_data(grid)
        held, gone, bad = self._tally(self.worst)
        ax.set_title(
            f"{self.args.assembly}  frac {self.args.assembly_fraction:.2f}  "
            f"{self.args.max_force:.0f} N   |   held {held}  tunneled {gone}  diverged {bad}"
            f"   |   ep {self.episode}   {self.fps:.0f} fps",
            fontsize=8,
        )
        fig.tight_layout()
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # Redrawing matplotlib every frame would dominate the step time.
        if self.frame % self.args.panel_interval == 0:
            self.viewer.log_image("penetration map", self._draw_panel())
        self.viewer.end_frame()

    def report(self):
        """Print the map. Not `test_final`: the runner's NaN guard would abort on
        the diverged cells, and those cells are the result, not a failure."""
        pen = self.worst
        held, gone, bad = self._tally(pen)
        print(f"\n{self.count} cells:  held {held}   tunneled {gone}   diverged {bad}")
        print("\nrows = ke (top: {:.0e}, bottom: {:.0e}), cols = kd (left: {:.0e}, right: {:.0e})".format(
            KE_RANGE[1], KE_RANGE[0], KD_RANGE[0], KD_RANGE[1]))
        print("  . held    x tunneled    ? diverged\n")
        for row in range(self.side - 1, -1, -1):
            line = ""
            for col in range(self.side):
                v = pen[row * self.side + col]
                line += "?" if np.isnan(v) else ("x" if v > 1.0 else ".")
            print(f"  ke={self.ke_axis[row]:>8.1e}  {line}")
        print("\n  kd:        " + " ".join(f"{self.kd_axis[c]:.0e}" for c in (0, self.side // 2, self.side - 1)))

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
        parser.add_argument("--sdf-resolution", type=int, default=256)
        parser.add_argument("--density", type=float, default=8000.0)
        parser.add_argument("--settle-frames", type=int, default=10,
                            help="Frames before the press engages. The nut is seated exactly, "
                                 "so this only needs to be long enough to read a rest height; "
                                 "raising it eats into the press within a fixed frame budget.")
        parser.add_argument("--press-kd", type=float, default=5.0e2,
                            help="Velocity feedback on the press [N.s/m].")
        parser.add_argument("--grid", type=int, default=32,
                            help="Grid side; 32 gives 1024 nut/bolt pairs.")
        parser.add_argument("--profile-start", type=str, default="fully_screwed",
                            choices=["fully_screwed", "full_thread"],
                            help="fully_screwed matches the task reset (nut near the tip); "
                                 "full_thread starts at the bottom of the shaft.")
        parser.add_argument("--reset-interval", type=int, default=200,
                            help="Frames per episode before re-seating; 0 never resets.")
        parser.add_argument("--panel-interval", type=int, default=10,
                            help="Frames between heatmap panel redraws.")
        parser.add_argument("--spacing", type=float, default=0.0,
                            help="Metres between cells; 0 = default (0.05).")
        parser.add_argument("--per-world-contacts", type=int, default=1024,
                            help="njmax/nconmax per cell. Too low silently drops contacts.")
        # The module-load chatter buries the only lines worth reading.
        parser.set_defaults(quiet=True)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
    example.report()
