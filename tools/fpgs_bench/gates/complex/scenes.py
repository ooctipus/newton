# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene adapters for the FeatherPGS patch-vs-point friction study.

``build(case, law, world_count, ...)`` constructs one scene at N worlds with the friction law
injected into the FeatherPGS constructor and nothing else changed. The returned :class:`Scene`
gives the runner a capturable ``simulate()`` (one frame, all substeps), a host-side
``before_frame(t)`` (that frame's control, launched outside the timed region) and the bodies and
time windows the physics metrics use. ``perturb_initial_poses`` then shifts the metric objects'
initial poses by a small seeded amount per world (identically in both arms), so that the worlds
are trials rather than replicas.

* ``hardware``, ``coins``, ``dishes``, ``cube``: the stock
  ``newton.examples.robot.example_robot_clutter.Example`` with its own defaults. Only
  ``newton.solvers.SolverFeatherPGS`` is swapped, while the example constructs its solver, for a
  subclass that sets ``friction_anchor_beta=0.0`` (point) or leaves it out (patch: Newton's
  default 0.2).
* ``pyramid``: ``example_pyramid``'s box stacks rebuilt from its constants (that example hard-codes
  XPBD), driven by FeatherPGS with the clutter example's solver settings, centred on the origin.
  Optional tilted gravity turns it into an incline creep test.
"""

import contextlib
import functools
import inspect
import math
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from unittest import mock

import numpy as np

LAWS = ("patch", "point")
# What a correctly constructed arm reports (point: explicit 0.0; patch: argument omitted).
EXPECTED_LAW = {
    "patch": {"friction_anchor_beta": 0.2, "anchors_enabled": True, "beta_source": "default", "requested": None},
    "point": {"friction_anchor_beta": 0.0, "anchors_enabled": False, "beta_source": "argument", "requested": 0.0},
}
CLUTTER = ("hardware", "coins", "dishes", "cube")
SCENES = (*CLUTTER, "pyramid")
# Container interiors in the container body frame (half extents [m]; the body origin is the floor's bottom plane).
CONTAINER_HALF_EXTENTS = {"hardware": (0.116, 0.076), "coins": (0.196, 0.146), "dishes": (0.175, 0.140)}
# Top surface of the container floor in the container body frame [m]: _clutter_container.container_mesh
# floor=0.004 (hardware bin) / 0.005 (coin tray); the dish rack's tray spans z 0-0.003.
FLOOR_TOP = {"hardware": 0.004, "coins": 0.005, "dishes": 0.003}
# Quiet window (offset after settle, length) [s]. It starts at the example's settle time, so residual
# settling of the dropped pile is not mixed into drift and creep. The robot holds still until settle and
# then descends toward the handle, which it reaches about 0.22 periods (2.2 s) after settle.
REST_AFTER_SETTLE = {"hardware": (0.0, 1.0), "coins": (0.0, 1.5), "dishes": (0.0, 1.5)}
# The group whose drift the overview shows next to the pooled one (rest.<group> in the results).
FOCUS_GROUP = {"hardware": "objects", "coins": "tower", "dishes": "dishes", "cube": "objects", "pyramid": "top"}


@dataclass
class Scene:
    case: str
    kind: str
    model: Any
    pipeline: Any
    contacts: Any
    solver: Any
    fps: float
    substeps: int
    world_count: int
    bodies_per_world: int
    simulate: Callable[[], None]
    states: Callable[[], tuple]  # (latest state, the last substep's collided input state)
    before_frame: Callable[[float], None]
    groups: dict  # metric object groups: name -> local body indices
    windows: dict  # name -> (t0, t1) [s]
    default_seconds: float
    support_body: int | None = None  # local container body; None measures in the world frame
    tcp_body: int | None = None
    shake_axis: int | None = None
    container_half_extents: tuple | None = None
    floor_top: float | None = None
    focus_group: str = "objects"
    special: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    law_record: dict = field(default_factory=dict)
    end_report: Callable[[], dict] | None = None
    after_perturb: Callable[[], None] | None = None  # re-run construction steps that read the initial state


def plain(value):
    """JSON-safe copy of simple values; everything else becomes its repr."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
        return plain(value.item())
    return repr(value)


AUDIT = False
"""``run_complex.py --audit``: every solver takes ``pgs_audit=True`` (a Newton that has it), which steers no route."""


@contextlib.contextmanager
def inject_law(law: str, record: dict):
    """Swap ``newton.solvers.SolverFeatherPGS`` for a subclass that changes only ``friction_anchor_beta``.

    point passes ``friction_anchor_beta=0.0``; patch omits the argument so Newton's default applies.
    Records the constructor kwargs, its wall time and every warning it raised.
    """
    if law not in LAWS:
        raise ValueError(f"law must be one of {LAWS}")
    import newton  # noqa: PLC0415

    base = newton.solvers.SolverFeatherPGS

    class LawSolver(base):
        def __init__(self, model, *args, **kwargs):
            record["caller_friction_anchor_beta"] = plain(kwargs.pop("friction_anchor_beta", "omitted"))
            if law == "point":
                kwargs["friction_anchor_beta"] = 0.0
            if AUDIT:
                kwargs["pgs_audit"] = True
            record["kwargs"] = {name: plain(value) for name, value in kwargs.items()}
            record["positional_args"] = len(args)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                start = time.perf_counter()
                super().__init__(model, *args, **kwargs)
                record["init_s"] = time.perf_counter() - start
            record["warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught]

    with mock.patch.object(newton.solvers, "SolverFeatherPGS", LawSolver):
        yield


class _StubViewer:
    """Accepts the viewer calls the examples make during construction; renders nothing."""

    picking_enabled = False

    def set_model(self, *args, **kwargs):
        pass

    def set_world_offsets(self, *args, **kwargs):
        pass

    def set_camera(self, *args, **kwargs):
        pass

    def apply_forces(self, *args, **kwargs):
        pass


def build_clutter(name, law, world_count, *, cpu_smoke=False, object_count=None, example_args=()) -> Scene:
    """The stock clutter example at ``world_count`` worlds; ``example_args`` go to its parser."""
    import warp as wp  # noqa: PLC0415

    from newton.examples.robot import example_robot_clutter as C  # noqa: PLC0415
    from newton.examples.robot._clutter_cube import build_cube  # noqa: PLC0415

    argv = ["--scene", name, "--world-count", str(world_count), "--no-cuda-graph"]
    if object_count is not None and name != "cube":
        argv += ["--object-count", str(object_count)]
    if cpu_smoke:
        argv += ["--pgs-mode", "split"]
    argv += list(example_args)
    args = C.create_parser().parse_args(argv)
    record = {}
    with contextlib.ExitStack() as stack:
        stack.enter_context(inject_law(law, record))
        if cpu_smoke:
            # SDF cooking needs CUDA: CPU smokes use the legacy BVH mesh path and the convex cube.
            stack.enter_context(mock.patch.object(C, "cook_sdf_meshes", lambda requests, device=None: []))
            scenes = dict(C.SCENES)
            scenes["cube"] = (functools.partial(build_cube, collision="convex"), 26)
            stack.enter_context(mock.patch.object(C, "SCENES", scenes))
        start = time.perf_counter()
        ex = C.Example(_StubViewer(), args)
        construct_s = time.perf_counter() - start

    info = ex.scene_info
    settle, period = float(ex.settle_seconds), float(args.period)
    objects = [int(b) for b in info["object_bodies"]]
    rate_scale = 0.0 if name == "cube" else 1.0  # Example.step zeroes the cube's velocity feed-forward

    def before_frame(sim_time):
        """Example.step's control, without its host syncs (check_capacity, --test traces)."""
        phase, rate = C.interaction_phase(
            sim_time, settle=ex.settle_seconds, period=args.period, times=ex.path_times, motion=args.motion
        )
        wp.launch(
            C._joint_targets,
            dim=(world_count, 9),
            inputs=[ex.path, phase, rate * rate_scale, ex.targets, ex.velocity_targets],
        )
        ex.sim_time = sim_time

    special, support, shake_axis, half_extents, floor_top = {}, None, None, None, None
    if name == "cube":
        lattice = [tuple(p) for p in info["metadata"]["lattice"]]
        # The robot holds still until settle; the cube is built assembled, so the second before settle is quiet.
        windows = {
            "settle": (0.0, max(settle - 1.0, 0.0)),
            "rest": (max(settle - 1.0, 0.0), settle),
            "turn": (settle + period * 2 / 7, settle + period * 3 / 7),
            "open": (settle + period * 3 / 7, settle + period * 4 / 7),
        }
        default_seconds = settle + period * 4 / 7
        groups = {"cubies": objects}
        special["top_center"] = objects[lattice.index((0, 0, 1))]
        special["lower_layer"] = [b for b, p in zip(objects, lattice, strict=True) if p[2] == -1]
    else:
        offset, length = REST_AFTER_SETTLE[name]
        rest = (settle + offset, settle + offset + length)
        windows = {
            "settle": (0.0, settle),
            "rest": rest,
            "strokes": (settle + 0.30 * period, settle + 0.66 * period),
            "release": (settle + 0.66 * period, settle + 0.76 * period),
        }
        default_seconds = rest[1] if name in ("hardware", "coins") else windows["strokes"][1]
        support = int(info["container_body"])
        shake_axis = {"x": 0, "y": 1}[ex.shake_axis]
        half_extents = CONTAINER_HALF_EXTENTS[name]
        floor_top = FLOOR_TOP[name]
        if name == "coins":
            groups = {"bed": objects[: info["bed_count"]], "tower": objects[info["bed_count"] :]}
        elif name == "dishes":
            groups = {"dishes": objects[: info["dish_count"]], "cutlery": objects[info["dish_count"] :]}
        else:
            groups = {"parts": objects}
    groups = {key: members for key, members in groups.items() if members}
    focus = FOCUS_GROUP[name] if FOCUS_GROUP[name] in groups or FOCUS_GROUP[name] == "objects" else "objects"

    def end_report():
        out = {"collision_buffers": ex.pipeline.narrow_phase.buffer_capacity_status()}
        if ex.body_pair_reduction:
            out["body_pair_reduction"] = plain(ex.pipeline.body_pair_reduction_stats())
        return out

    def after_perturb():
        # Example.__init__ primes the body-pair reduction with one collide of the initial state. dt is the time to the
        # next collide (one substep); a speculative pipeline requires it, the others ignore it.
        if ex.body_pair_reduction:
            ex.pipeline.collide(ex.state_0, ex.contacts, dt=ex.sim_dt)

    return Scene(
        case=name,
        kind="clutter",
        model=ex.model,
        pipeline=ex.pipeline,
        contacts=ex.contacts,
        solver=ex.solver,
        fps=1.0 / ex.frame_dt,
        substeps=ex.sim_substeps,
        world_count=world_count,
        bodies_per_world=ex.bodies_per_world,
        simulate=ex.simulate,
        states=lambda: (ex.state_0, ex.state_1),
        before_frame=before_frame,
        groups=groups,
        windows=windows,
        default_seconds=default_seconds,
        support_body=support,
        tcp_body=int(ex.tcp),
        shake_axis=shake_axis,
        container_half_extents=half_extents,
        floor_top=floor_top,
        focus_group=focus,
        special=special,
        info={
            "example_args": plain(vars(args)),
            "objects_per_world": ex.object_count,
            "shapes_per_world": ex.model.shape_count // world_count,
            "settle_seconds": settle,
            "period_seconds": period,
            "shake_axis": ex.shake_axis,
            "grasp_tilt_deg": ex.grasp_tilt,
            "body_pair_reduction": bool(ex.body_pair_reduction),
            "iterations": args.iterations,
            "capacities": {
                "contacts_per_world": ex.contact_capacity,
                "triangle_pairs_per_world": ex.triangle_pair_capacity,
                "mf_rows_per_world": ex.mf_rows,
                "dense_rows_per_world": args.dense_rows or (2304 if name == "cube" else 256),
                "rigid_contact_max": int(ex.contacts.rigid_contact_max),
            },
            "sdf_meshes": plain(ex.sdf_report),
            "construct_s": construct_s,
        },
        law_record=record,
        end_report=end_report,
        after_perturb=after_perturb,
    )


class _Stepper:
    """Ping-pong states and one frame of (clear_forces, collide, step) substeps, as the examples do."""

    def __init__(self, model, pipeline, contacts, solver, substeps, frame_dt):
        import newton  # noqa: PLC0415

        self.pipeline, self.contacts, self.solver = pipeline, contacts, solver
        self.substeps, self.dt = substeps, frame_dt / substeps
        self.state_0, self.state_1 = model.state(), model.state()
        self.control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_0)

    def simulate(self):
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0


def build_pyramid(
    law, world_count, *, rows=20, pyramids=1, tilt_deg=0.0, iterations=24, contacts_per_box=16, cpu_smoke=False
) -> Scene:
    """``example_pyramid`` stacks (test mode: no wrecking ball) with FeatherPGS; gravity tilted about +y.

    The stacks are centred on the origin. The example's Y_STACK = 15 m offset (room for its wrecking-ball
    ramp) would leave float32 positions a 0.95 um step along y and hide sub-micrometre drift.
    """
    import warp as wp  # noqa: PLC0415

    import newton  # noqa: PLC0415
    from newton.examples.contacts import example_pyramid as P  # noqa: PLC0415

    newton.use_coord_layout_targets = True
    start = time.perf_counter()
    tilt = math.radians(tilt_deg)
    gravity = (9.81 * math.sin(tilt), 0.0, -9.81 * math.cos(tilt))
    # replicate() copies the source builder's gravity into every world and FeatherPGS reads
    # gravity[0], so the tilt goes on the per-world builder (and the top level, for good measure).
    world = newton.ModelBuilder(gravity=gravity)
    world.add_shape_plane(xform=wp.transform_identity(), width=0.0, length=0.0)
    top, levels = [], []
    for pyramid in range(pyramids):
        y = (pyramid - (pyramids - 1) / 2) * P.PYRAMID_SPACING
        for level in range(rows):
            count = rows - level
            width = (count - 1) * P.CUBE_SPACING
            for i in range(count):
                position = wp.vec3(-width / 2 + i * P.CUBE_SPACING, y, level * P.CUBE_SPACING + P.CUBE_HALF)
                body = world.add_body(xform=wp.transform(p=position, q=wp.quat_identity()))
                world.add_shape_box(body, hx=P.CUBE_HALF, hy=P.CUBE_HALF, hz=P.CUBE_HALF)
                levels.append(level)
                if level == rows - 1:
                    top.append(body)
    boxes = world.body_count
    scene = newton.ModelBuilder(gravity=gravity)
    scene.replicate(world, world_count)
    model = scene.finalize()
    contacts_per_world = contacts_per_box * boxes + 64
    pipeline = newton.CollisionPipeline(model, broad_phase="sap", rigid_contact_max=contacts_per_world * world_count)
    contacts = pipeline.contacts()
    record = {}
    options = {
        "pgs_iterations": iterations,
        "mf_max_constraints": min(3 * contacts_per_world, 65535),
        "row_watermark": True,
    }
    # Trees before E5's C6 take the formulation as an option; later trees choose it from the device.
    if "pgs_mode" in inspect.signature(newton.solvers.SolverFeatherPGS.__init__).parameters:
        options["pgs_mode"] = "split" if cpu_smoke else "matrix_free"
    with inject_law(law, record):
        solver = newton.solvers.SolverFeatherPGS(model, **options)
    fps, substeps = 100.0, 10  # example_pyramid's cadence
    stepper = _Stepper(model, pipeline, contacts, solver, substeps, 1.0 / fps)
    construct_s = time.perf_counter() - start
    case = f"pyramid-r{rows}-p{pyramids}" + (f"-t{tilt_deg:g}" if tilt_deg else "")
    return Scene(
        case=case,
        kind="pyramid",
        model=model,
        pipeline=pipeline,
        contacts=contacts,
        solver=solver,
        fps=fps,
        substeps=substeps,
        world_count=world_count,
        bodies_per_world=boxes,
        simulate=stepper.simulate,
        states=lambda: (stepper.state_0, stepper.state_1),
        before_frame=lambda sim_time: None,
        groups={"boxes": list(range(boxes)), "top": top},
        windows={"settle": (0.0, 1.0), "rest": (1.0, 3.0)},
        default_seconds=3.0,
        focus_group="top",
        # Boxes are built with 0.04 m gaps between levels; an upright stack settles with the gaps closed,
        # level L's centre at z = L * 2 * CUBE_HALF + CUBE_HALF (survival is judged against that pose).
        special={"top": top, "levels": levels, "settled_z": [lv * 2 * P.CUBE_HALF + P.CUBE_HALF for lv in levels]},
        info={
            "rows": rows,
            "pyramids": pyramids,
            "tilt_deg": tilt_deg,
            "gravity": list(gravity),
            "model_gravity_world0": plain(model.gravity.numpy()[0].tolist()),
            "objects_per_world": boxes,
            "shapes_per_world": model.shape_count // world_count,
            "iterations": iterations,
            "stack_y_m": [(p - (pyramids - 1) / 2) * P.PYRAMID_SPACING for p in range(pyramids)],
            "capacities": {
                "contacts_per_world": contacts_per_world,
                "mf_rows_per_world": min(3 * contacts_per_world, 65535),
                "dense_rows_per_world": 32,
                "rigid_contact_max": int(contacts.rigid_contact_max),
            },
            "construct_s": construct_s,
        },
        law_record=record,
        end_report=lambda: {"collision_buffers": pipeline.narrow_phase.buffer_capacity_status()},
    )


def build(case, law, world_count, *, cpu_smoke=False, object_count=None, example_args=(), pyramid=None) -> Scene:
    if case in CLUTTER:
        return build_clutter(
            case, law, world_count, cpu_smoke=cpu_smoke, object_count=object_count, example_args=example_args
        )
    if case == "pyramid":
        return build_pyramid(law, world_count, cpu_smoke=cpu_smoke, **(pyramid or {}))
    raise ValueError(f"unknown scene {case!r}; choose from {SCENES}")


def _quat_mul(a, b):
    av, aw = np.asarray(a[:3]), a[3]
    bv, bw = np.asarray(b[:3]), b[3]
    v = aw * bv + bw * av + np.cross(av, bv)
    return np.array([*v, aw * bw - float(np.dot(av, bv))])


def perturb_initial_poses(scene: Scene, *, seed: int, xy_mm: float, lift_mm: float, yaw_deg: float) -> dict:
    """Shift each metric object's initial pose in worlds 1..N-1 by a small seeded amount.

    Every world otherwise starts as an exact replica of world 0 (one builder, ``replicate()``), so
    per-world outcomes would be one sample counted N times. Per (world, object): x and y uniform in
    +-``xy_mm``, a lift uniform in [0, ``lift_mm``] (never into the support below) and a yaw uniform
    in +-``yaw_deg`` about world z. World 0 keeps the example's exact poses. The draws depend only on
    the seed, the world and the object, so both arms of a pair start from identical states.

    Only free bodies (free joint to the world, identity joint frames: joint q == body pose) can be
    moved; articulated objects such as the cube's cubies stay replicas, which the record says.
    """
    record = {"seed": seed, "xy_mm": xy_mm, "lift_mm": lift_mm, "yaw_deg": yaw_deg, "world0": "unperturbed"}
    if scene.world_count < 2 or not (xy_mm or lift_mm or yaw_deg):
        return {**record, "bodies": 0, "reason": "disabled or a single world"}
    import warp as wp  # noqa: PLC0415

    import newton  # noqa: PLC0415

    model = scene.model
    joint_type, child = model.joint_type.numpy(), model.joint_child.numpy()
    parent, q_start = model.joint_parent.numpy(), model.joint_q_start.numpy()
    free = int(newton.JointType.FREE)
    joint_of = {
        int(c): j for j, (t, c, p) in enumerate(zip(joint_type, child, parent, strict=True)) if t == free and p == -1
    }
    local = sorted({b for members in scene.groups.values() for b in members})
    movable = [b for b in local if b in joint_of]
    if not movable:
        return {**record, "bodies": 0, "reason": "the metric objects are not free bodies; worlds stay replicas"}
    state_0, state_1 = scene.states()
    joint_q = np.array(state_0.joint_q.numpy(), dtype=np.float64)
    body_q = np.array(state_0.body_q.numpy(), dtype=np.float64)
    rng = np.random.default_rng(seed)
    xy, lift, yaw_max = xy_mm * 1e-3, lift_mm * 1e-3, math.radians(yaw_deg)
    moved, expected = [], {}
    for w in range(scene.world_count):
        draws = rng.uniform(-1.0, 1.0, size=(len(movable), 4))  # drawn for world 0 too: world w's draw is fixed
        if w == 0:
            continue
        for b, (dx, dy, dz, dyaw) in zip(movable, draws, strict=True):
            body = w * scene.bodies_per_world + b
            j = joint_of.get(body)
            if j is None:
                raise RuntimeError(f"body {body} is not a free body like its world-0 replica {b}")
            s = int(q_start[j])
            pose = joint_q[s : s + 7]
            if not (
                np.allclose(pose[:3], body_q[body, :3], atol=1e-6)
                and abs(np.dot(pose[3:], body_q[body, 3:])) > 1 - 1e-6
            ):
                raise RuntimeError(f"free joint {j} coordinates are not body {body}'s pose; cannot perturb safely")
            pose[:3] += (dx * xy, dy * xy, 0.5 * (dz + 1.0) * lift)
            half = 0.5 * dyaw * yaw_max
            quat = _quat_mul((0.0, 0.0, math.sin(half), math.cos(half)), pose[3:])
            pose[3:] = quat / np.linalg.norm(quat)
            expected[body] = pose.copy()
            moved.append(body)
    state_0.joint_q.assign(joint_q.astype(np.float32))
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    wp.copy(state_1.joint_q, state_0.joint_q)
    wp.copy(state_1.body_q, state_0.body_q)
    after = np.array(state_0.body_q.numpy(), dtype=np.float64)
    error = max(float(np.abs(after[b, :3] - expected[b][:3]).max()) for b in moved)
    if error > 1e-5:
        raise RuntimeError(f"forward kinematics did not reproduce the perturbed poses (error {error:.2e} m)")
    if scene.after_perturb is not None:
        scene.after_perturb()
    shift = np.linalg.norm(after[moved, :3] - body_q[moved, :3], axis=1) * 1e3
    return {
        **record,
        "bodies": len(moved),
        "objects_per_world": len(movable),
        "worlds": scene.world_count - 1,
        "fk_error_m": error,
        "shift_mm_max": float(shift.max()),
        "shift_mm_median": float(np.median(shift)),
    }
