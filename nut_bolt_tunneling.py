# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Nut/bolt tunneling diagnosis
#
# Drives a nut into a fixed bolt with a controlled disturbance and reports
# whether the nut passes through the thread. Exists so contact parameters can
# be screened in seconds instead of by training a policy and discovering it
# learned to hammer the nut through.
#
# The bolt is a static shape: this measures nut-through-thread, not bolt
# displacement.
#
#   python nut_bolt_tunneling.py --assembly m16_tight --force 500
#
###########################################################################

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh
import warp as wp

import newton
import newton.examples

# The task's own assets. The IsaacGymEnvs meshes that newton's example downloads
# are the same lineage (the USD prims are still named ..._loose) but the shipped
# colliders are decimated differently -- the m16 nut collider is 650 points, not
# the subdiv_3x mesh -- and there is no "tight" variant in the task at all.
NIST_ASSET_DIR = Path(
    "/home/zhengyuz/Projects/IsaacLab.wt/octi-factory-newton-tunneling"
    "/source/isaaclab_assets/data/Assets/Props/NIST"
)

ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"
SDF_CACHE_DIR = Path(tempfile.gettempdir()) / "newton_sdf_cache"

# Nominal thread pitch [m] per size, used to scale "how far may the nut legally
# descend before we call it tunneling".
THREAD_PITCH = {"m4": 0.0007, "m8": 0.00125, "m12": 0.00175, "m16": 0.002, "m20": 0.0025}

# Straight out of the task's assembly_keypoints.py. Bolt offsets are measured from
# the head (z=0) up the shaft; the nut's own center_axis_middle is where the nut
# frame carries its mid-height. The NIST USDs are authored in this same frame --
# bolt_m16's collider spans 0.010..0.035, matching full_thread..bolt_tip_offset --
# so the meshes must NOT be re-centered on their bounding box or the keypoints
# stop meaning anything.
KEYPOINTS = {
    #        tip     first   second  third   fully_screwed   nut center_axis_middle
    "m16": (0.035, 0.034, 0.032, 0.030, 0.022, 0.0165),
    "m12": (0.035, None, 0.0285, None, 0.0218, 0.018),
    "m8": (0.026, None, 0.0242, None, 0.018, 0.0126),
    "m4": (0.020, None, 0.0189, None, 0.01318, 0.0064),
}
ENGAGE_NAMES = ("tip", "first_thread", "second_thread", "third_thread", "fully_screwed")


def engaged_nut_z(size: str, engage: str) -> float:
    """Nut-frame origin height that seats the nut at `engage` on the bolt."""
    tip, first, second, third, screwed, nut_mid = KEYPOINTS[size]
    target = dict(zip(ENGAGE_NAMES, (tip, first, second, third, screwed)))[engage]
    if target is None:
        # Only m16 authors first/third_thread. Step down from the nearest
        # authored keypoint by whole pitches -- an approximation, flagged as one,
        # because engage depth turns out to decide whether the test discriminates
        # at all and the smaller sizes cannot otherwise be exercised.
        pitch = THREAD_PITCH[size]
        if engage == "third_thread":
            target = second - pitch
        elif engage == "first_thread":
            target = second + pitch
        else:
            raise ValueError(f"{size} has no {engage} keypoint")
    return target - nut_mid


@dataclass
class Cfg:
    """One point in the contact-parameter space under test."""

    assembly: str = "m16"  # task USD; "m16_tight" etc. use the IsaacGymEnvs meshes
    # Collision runs once per frame; the solver runs `substeps` times inside it.
    collide_hz: float = 200.0
    substeps: int = 16
    ke: float = 2.56e6
    kd: float = 3200.0
    # 0.75 is the task's assembly-pair friction. Do not lower it while cone is
    # pyramidal: mu->0 zeroes the pyramidal row invweight, efc_D is floored to
    # ~1e15, and the float32 Hessian degenerates to NaN on contact-rich states.
    mu: float = 0.75
    # 0 = auto: 0.4 x thread pitch. The detection distance has to be smaller
    # than the thread interference (1.15 mm radial on the task m16); the stock
    # example's 0.005 is four times that and reads as the baseline penetrating
    # twenty pitches.
    gap: float = 0.0
    margin: float = 0.0
    density: float = 8000.0
    is_hydroelastic: bool = False
    kh: float = 1e11
    sdf_resolution: int = 512
    impratio: float = 1.0
    cone: str = "pyramidal"
    solver_iterations: int = 15
    ls_iterations: int = 100
    # Disturbance
    drive: str = "speed"  # speed | press | hammer | force | impulse
    force: float = 500.0  # [N] downward, for drive=force
    impact_speed: float = 2.0  # [m/s] downward, for drive=impulse
    press_kp: float = 5.0e4  # [N/m] arm stiffness, for drive=press
    press_kd: float = 5.0e2  # [N.s/m] arm damping, for drive=press
    max_force: float = 1000.0  # [N] arm effort ceiling -- top of the observed Franka band
    press_speed: float = 0.05  # [m/s] prescribed descent, for drive=velocity
    clearance: float = 0.001  # [m] gap between nut base and bolt tip at t=0
    settle_frames: int = 10  # free-fall/settle before the drive engages
    frames: int = 120
    rest_z_ref: float = 0.0  # measured once per assembly; 0 = use start height
    engage: str = "fully_screwed"  # where the keypoints seat the nut on the bolt
    narrow_band: float = 0.0  # SDF band half-width; 0 = auto (2.5 x thread pitch)
    screw_frames: int = 0  # frames spent threading the nut on before the drive
    screw_force: float = 5.0  # [N] axial load while threading
    screw_torque: float = 0.05  # [N.m] about the bolt axis while threading

    @property
    def size(self) -> str:
        return self.assembly.split("_")[0]

    @property
    def solver_hz(self) -> float:
        return self.collide_hz * self.substeps


def _usd_collider_arrays(usd_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Triangles of the collision mesh authored in a NIST asset, in world space."""
    from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    stage = Usd.Stage.Open(str(usd_path))
    prim = next(
        (p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and UsdPhysics.CollisionAPI(p)),
        None,
    )
    if prim is None:
        raise ValueError(f"no collision mesh in {usd_path}")

    mesh = UsdGeom.Mesh(prim)
    xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pts = np.array([xform.Transform(p) for p in mesh.GetPointsAttr().Get()], dtype=np.float32)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

    tris, at = [], 0
    for c in counts:  # fan-triangulate anything that is not already a triangle
        face = idx[at : at + c]
        tris.extend([face[0], face[i], face[i + 1]] for i in range(1, c - 1))
        at += c
    return pts, np.asarray(tris, dtype=np.int32).flatten()


def _load_usd_mesh(usd_path: Path, gap: float, resolution: int, narrow_band: float = 0.005):
    """Same contract as :func:`_load_mesh`, but keeps the authored frame.

    The keypoints are expressed in this frame, so re-centering on the bounding
    box -- which is what the OBJ path does -- would make them unusable.
    """
    vertices, indices = _usd_collider_arrays(usd_path)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    center = np.zeros(3, dtype=np.float32)
    mesh = newton.Mesh(vertices, indices)
    mesh.build_sdf(
        max_resolution=resolution,
        narrow_band_range=(-narrow_band, narrow_band),
        margin=gap if gap else 0.05,
        cache_dir=SDF_CACHE_DIR,
    )
    return mesh, center, hi - lo


def _load_mesh(path: str, gap: float, resolution: int) -> tuple[newton.Mesh, np.ndarray, np.ndarray]:
    """Load a mesh centered on its bbox, returning the mesh, its center, and extents."""
    data = trimesh.load(path, force="mesh")
    vertices = np.array(data.vertices, dtype=np.float32)
    indices = np.array(data.faces.flatten(), dtype=np.int32)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    center = (lo + hi) / 2.0
    vertices = vertices - center
    mesh = newton.Mesh(vertices, indices)
    mesh.build_sdf(
        max_resolution=resolution,
        narrow_band_range=(-0.005, 0.005),
        margin=gap if gap else 0.05,
        cache_dir=SDF_CACHE_DIR,
    )
    return mesh, center, hi - lo


@wp.kernel
def _push_down(body_f: wp.array(dtype=wp.spatial_vector), nut: int, force: float):
    """Constant downward force on the nut, on top of gravity."""
    wp.atomic_add(body_f, nut, wp.spatial_vector(wp.vec3(0.0, 0.0, -force), wp.vec3(0.0)))


@wp.kernel
def _drive_speed(
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    peak: wp.array(dtype=wp.float32),
    nut: int,
    speed: float,
    kd: float,
    max_force: float,
):
    """Velocity source with a bounded force -- an impedance-controlled arm.

    Tunneling is a speed x timestep phenomenon, and a force source on a 30 g
    free body is unusable: 4 kN is a 37 m/s velocity jump in one substep, which
    blows the solver up before any contact is resolved. Commanding descent speed
    with a force ceiling keeps the load inside what an arm can deliver.
    """
    vz = wp.spatial_top(body_qd[nut])[2]
    f = wp.clamp(kd * (-speed - vz), -max_force, max_force)
    wp.atomic_add(body_f, nut, wp.spatial_vector(wp.vec3(0.0, 0.0, f), wp.vec3(0.0)))
    wp.atomic_max(peak, 0, wp.abs(f))


@wp.kernel
def _screw_on(
    body_f: wp.array(dtype=wp.spatial_vector),
    nut: int,
    force: float,
    torque: float,
):
    """Thread the nut onto the bolt so the hammer lands on an engaged joint.

    A nut parked on the chamfer is blocked by a thick tip; an engaged nut is
    held by a thin helical flank, which is the geometry that actually loses to
    a softened contact. Screwing it on is self-aligning -- placing it at an
    engaged height by hand would need the thread phase to match or the nut is
    born interpenetrating and the solver pins it.
    """
    wp.atomic_add(
        body_f, nut, wp.spatial_vector(wp.vec3(0.0, 0.0, -force), wp.vec3(0.0, 0.0, torque))
    )


@wp.kernel
def _press(
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
    """Impedance drive, the way an arm actually loads the nut.

    The gripper is a stiff position source, not a force source: the load on the
    thread is whatever the tracking error produces, capped by the arm's effort
    limit. Driving a 30 g free body with a raw 500 N step instead just launches
    it and blows up the solver.
    """
    z = wp.transform_get_translation(body_q[nut])[2]
    vz = wp.spatial_top(body_qd[nut])[2]
    f = kp * (target_z - z) - kd * vz
    f = wp.clamp(f, -max_force, max_force)
    wp.atomic_add(body_f, nut, wp.spatial_vector(wp.vec3(0.0, 0.0, f), wp.vec3(0.0)))
    wp.atomic_max(peak, 0, wp.abs(f))


class Rig:
    """A fixed bolt with a nut dropped/pressed onto its thread."""

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.gap = cfg.gap if cfg.gap > 0 else 0.4 * THREAD_PITCH[cfg.size]
        self.frame_dt = 1.0 / cfg.collide_hz
        self.sim_dt = self.frame_dt / cfg.substeps

        use_task_assets = "_" not in cfg.assembly
        if use_task_assets:
            bolt_file = NIST_ASSET_DIR / f"bolt_{cfg.assembly}.usd"
            nut_file = NIST_ASSET_DIR / f"nut_{cfg.assembly}.usd"
        else:
            asset_dir = newton.examples.download_external_git_folder(
                ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER
            )
            bolt_file = str(asset_dir / f"factory_bolt_{cfg.assembly}.obj")
            nut_file = str(asset_dir / f"factory_nut_{cfg.assembly}_subdiv_3x.obj")

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            margin=cfg.margin,
            mu=cfg.mu,
            ke=cfg.ke,
            kd=cfg.kd,
            kh=cfg.kh,
            gap=self.gap,
            density=cfg.density,
            mu_torsional=0.0,
            mu_rolling=0.0,
            is_hydroelastic=cfg.is_hydroelastic,
        )

        if use_task_assets:
            # A +/-5 mm band is ~7 pitches of an m4 thread; scale it to the feature.
            band = cfg.narrow_band if cfg.narrow_band > 0 else 2.5 * THREAD_PITCH[cfg.size]
            self.narrow_band = band
            bolt_mesh, bolt_center, bolt_extent = _load_usd_mesh(
                bolt_file, self.gap, cfg.sdf_resolution, band
            )
            nut_mesh, nut_center, nut_extent = _load_usd_mesh(
                nut_file, self.gap, cfg.sdf_resolution, band
            )
        else:
            self.narrow_band = 0.005
            bolt_mesh, bolt_center, bolt_extent = _load_mesh(bolt_file, self.gap, cfg.sdf_resolution)
            nut_mesh, nut_center, nut_extent = _load_mesh(nut_file, self.gap, cfg.sdf_resolution)
        self.bolt_extent, self.nut_extent = bolt_extent, nut_extent

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = self.gap

        # Task assets keep their authored frame, so the bolt goes in at identity
        # and the keypoints address it directly. The OBJ path still centers.
        if use_task_assets:
            bolt_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
            self.bolt_base_z = float(KEYPOINTS[cfg.size][4])  # fully-screwed seat
            self.bolt_top_z = float(KEYPOINTS[cfg.size][0])  # bolt tip
        else:
            self.bolt_base_z = 0.0
            bolt_xform = wp.transform(
                wp.vec3(0.0, 0.0, float(bolt_extent[2] / 2.0)), wp.quat_identity()
            )
            self.bolt_top_z = float(bolt_extent[2])
        builder.add_shape_mesh(-1, xform=bolt_xform, mesh=bolt_mesh, cfg=shape_cfg, label="bolt")

        # Task assets: seat the nut where the task's own keypoints put it, which
        # is what the reset does. Perching it above the tip instead tests the
        # chamfer -- a thick barrier -- rather than the thread flank that
        # actually loses to a softened contact.
        if use_task_assets:
            self.nut_start_z = float(engaged_nut_z(cfg.size, cfg.engage))
        else:
            self.nut_start_z = float(self.bolt_top_z + nut_extent[2] * 0.5 + cfg.clearance)
        nut_body = builder.add_body(
            label="nut",
            xform=wp.transform(wp.vec3(0.0, 0.0, self.nut_start_z), wp.quat_identity()),
        )
        builder.add_shape_mesh(nut_body, mesh=nut_mesh, cfg=shape_cfg, label="nut_shape")
        self.nut_body = nut_body

        self.model = builder.finalize()
        self.model.rigid_contact_max = 40000

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            rigid_contact_max=40000,
            broad_phase="sap",
        )
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone=cfg.cone,
            njmax=8000,
            nconmax=8000,
            iterations=cfg.solver_iterations,
            ls_iterations=cfg.ls_iterations,
            impratio=cfg.impratio,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.collision_pipeline.contacts()

        self.nut_mass = float(self.model.body_mass.numpy()[nut_body])
        self.peak_force = wp.zeros(1, dtype=wp.float32)
        # Geometric rest: where a nut lowered onto the thread sits when the
        # contact is stiff and well resolved. Every arm is measured against this
        # same reference, so a soft arm settling deeper already reads as
        # penetration rather than redefining its own zero.
        self.rest_z_ref = cfg.rest_z_ref if cfg.rest_z_ref > 0 else self.nut_start_z
        # Where the drive is trying to take the nut: all the way down the shank.
        self.target_z = float(self.bolt_base_z)

        # drive=impulse is applied in run() once the nut has settled onto the bolt

    def run(self) -> dict:
        cfg = self.cfg
        z_min = self.nut_start_z
        z_trace = []
        nan_frame = -1
        t0 = time.time()
        screw_end = cfg.settle_frames + cfg.screw_frames
        for frame in range(cfg.frames):
            screwing = cfg.settle_frames <= frame < screw_end
            driving = frame >= screw_end
            self.collision_pipeline.collide(self.state_0, self.contacts)
            for _ in range(cfg.substeps):
                self.state_0.clear_forces()
                if screwing:
                    wp.launch(
                        _screw_on,
                        dim=1,
                        inputs=[self.state_0.body_f, self.nut_body, cfg.screw_force, cfg.screw_torque],
                    )
                elif driving and cfg.drive == "force":
                    wp.launch(_push_down, dim=1, inputs=[self.state_0.body_f, self.nut_body, cfg.force])
                elif driving and cfg.drive == "speed":
                    wp.launch(
                        _drive_speed,
                        dim=1,
                        inputs=[
                            self.state_0.body_qd,
                            self.state_0.body_f,
                            self.peak_force,
                            self.nut_body,
                            cfg.press_speed,
                            cfg.press_kd,
                            cfg.max_force,
                        ],
                    )
                elif driving and cfg.drive in ("press", "hammer"):
                    wp.launch(
                        _press,
                        dim=1,
                        inputs=[
                            self.state_0.body_q,
                            self.state_0.body_qd,
                            self.state_0.body_f,
                            self.peak_force,
                            self.nut_body,
                            self.target_z,
                            cfg.press_kp,
                            cfg.press_kd,
                            cfg.max_force,
                        ],
                    )
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
                if driving and cfg.drive == "velocity":
                    qd = self.state_0.joint_qd.numpy()
                    qd[2] = -cfg.press_speed
                    self.state_0.joint_qd.assign(qd)
            if cfg.drive in ("impulse", "hammer") and frame == screw_end:
                qd = self.state_0.joint_qd.numpy()
                qd[2] = -cfg.impact_speed  # free joint: [wx,wy,wz? / vx,vy,vz] -- z linear slot
                self.state_0.joint_qd.assign(qd)
            z = float(self.state_0.body_q.numpy()[self.nut_body][2])
            z_trace.append(z)
            z_min = min(z_min, z)
            if not math.isfinite(z):
                nan_frame = frame
                break

        # An unrotated nut cannot legally descend: the thread crest blocks it.
        # Sinking a whole pitch without turning means the solver let it eat into
        # solid material. Requiring it to traverse the entire bolt instead is far
        # too strict -- the task's reward already reads a pitch of false descent
        # as progress, which is what the policy learned to farm.
        pitch = THREAD_PITCH.get(cfg.size, 0.002)
        penetration = self.rest_z_ref - z_min
        tunneled = bool(penetration > pitch)
        return {
            **asdict(cfg),
            "solver_hz": cfg.solver_hz,
            "nut_mass_kg": self.nut_mass,
            "nut_start_z": self.nut_start_z,
            "bolt_base_z": self.bolt_base_z,
            "bolt_top_z": self.bolt_top_z,
            "nut_z_min": float(z_min),
            "nut_z_end": z_trace[-1] if z_trace else float("nan"),
            "penetration_m": float(penetration),
            "penetration_pitches": float(penetration / pitch),
            "rest_z_ref": float(self.rest_z_ref),
            "gap_used": float(self.gap),
            "descent_m": float(self.nut_start_z - z_min),
            "descent_pitches": float((self.nut_start_z - z_min) / pitch),
            "tunneled": tunneled,
            "nan_frame": nan_frame,
            "blew_up": nan_frame >= 0,
            "settled_before_drive": nan_frame < 0 or nan_frame >= cfg.settle_frames,
            "peak_drive_force_N": float(self.peak_force.numpy()[0]),
            "rest_z": self.bolt_top_z + float(self.nut_extent[2]) * 0.5,
            "wall_s": round(time.time() - t0, 2),
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for f, v in asdict(Cfg()).items():
        if isinstance(v, bool):
            p.add_argument(f"--{f.replace('_', '-')}", action="store_true", default=v)
        else:
            p.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    p.add_argument("--json", type=str, default=None, help="Write the result dict here.")
    args = p.parse_args()

    cfg = Cfg(**{k: getattr(args, k) for k in asdict(Cfg())})
    result = Rig(cfg).run()
    print(json.dumps(result, indent=1, sort_keys=True))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
