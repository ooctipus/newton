"""Stage attribution of the held-tree finish and predict kernels on Kuka- and Franka-class scenes."""

import os
import sys
import time

os.environ.setdefault("FEATHER_PGS_STAGE_CLOCKS", "1")

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS

WORLDS = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
DT = 1.0 / 240.0


def T(x, y, z):
    return wp.transform(wp.vec3(x, y, z), wp.quat_identity())


def arm(scene, cfg, *, links, hand):
    """A fixed-base serial arm with a palm and the given finger layout."""
    base = scene.add_link(mass=2.0, inertia=wp.mat33(np.eye(3) * 0.1), lock_inertia=True)
    joints = [scene.add_joint_fixed(parent=-1, child=base, parent_xform=T(0.0, 0.0, 0.25 + 0.17 * links))]
    parent = base
    for link in range(links):
        body = scene.add_link(mass=1.5 - 0.1 * link, inertia=wp.mat33(np.eye(3) * 0.03), lock_inertia=True)
        joints.append(
            scene.add_joint_revolute(
                parent=parent,
                child=body,
                axis=newton.Axis.Y if link % 2 else newton.Axis.Z,
                parent_xform=T(0.0, 0.0, -0.15),
                target_ke=40.0,
                target_kd=3.0,
                limit_lower=-2.5,
                limit_upper=2.5,
            )
        )
        if link % 2:
            scene.add_shape_capsule(body, radius=0.03, half_height=0.06, cfg=cfg)
        parent = body
    palm = scene.add_link(mass=0.4, inertia=wp.mat33(np.eye(3) * 0.002), lock_inertia=True)
    joints.append(scene.add_joint_fixed(parent=parent, child=palm, parent_xform=T(0.0, 0.0, -0.1)))
    scene.add_shape_box(palm, hx=0.04, hy=0.04, hz=0.01, cfg=cfg)
    fingers, segments, prismatic, mimic = hand
    finger_joints = []
    for finger in range(fingers):
        p = palm
        for segment in range(segments):
            body = scene.add_link(mass=0.05, inertia=wp.mat33(np.eye(3) * 2e-5), lock_inertia=True)
            x = 0.03 * finger - 0.015 * (fingers - 1)
            if prismatic:
                joint = scene.add_joint_prismatic(
                    parent=p,
                    child=body,
                    axis=newton.Axis.X,
                    parent_xform=T(x, 0.0, -0.03),
                    target_ke=200.0,
                    target_kd=10.0,
                    limit_lower=0.0,
                    limit_upper=0.04,
                )
            else:
                joint = scene.add_joint_revolute(
                    parent=p,
                    child=body,
                    axis=newton.Axis.X,
                    parent_xform=T(x if segment == 0 else 0.0, 0.0, -0.03),
                    target_ke=2.0,
                    target_kd=0.1,
                    limit_lower=-0.3,
                    limit_upper=1.6,
                )
            joints.append(joint)
            finger_joints.append(joint)
            p = body
        scene.add_shape_capsule(p, radius=0.008, half_height=0.012, cfg=cfg)
    scene.add_articulation(joints)
    if mimic:
        scene.add_constraint_mimic(joint0=finger_joints[1], joint1=finger_joints[0], coef0=0.0, coef1=1.0)


def build(kind):
    scene = newton.ModelBuilder()
    cfg = scene.default_shape_cfg.copy()
    cfg.mu = 0.6
    if kind == "kuka":
        arm(scene, cfg, links=7, hand=(4, 4, False, False))
    else:
        arm(scene, cfg, links=7, hand=(2, 1, True, True))
    box = scene.add_body(xform=T(0.25, 0.0, 0.05), mass=0.3)
    scene.add_shape_box(box, hx=0.04, hy=0.04, hz=0.04, cfg=cfg)
    tray = scene.add_body(xform=T(-0.4, 0.0, 0.03), is_kinematic=True)
    scene.add_shape_box(tray, hx=0.08, hy=0.08, hz=0.03, cfg=cfg)
    builder = newton.ModelBuilder()
    builder.replicate(scene, WORLDS, spacing=(3.0, 3.0, 0.0))
    builder.add_ground_plane()
    model = builder.finalize(device="cuda:0")
    model.rigid_contact_max = 64 * WORLDS
    return model


FINISH = ("update_qdd", "root transport", "integrate joints", "relative", "poses", "motion", "bodies", "collect")
COLLECT = ("reduce", "bias projection", "geometric")
PREDICT = ("forces", "reduce+assemble", "held solve", "publish", "transport+twists")

for kind in ("kuka", "franka"):
    model = build(kind)
    solver = SolverFeatherPGS(
        model,
        pgs_iterations=8,
        enable_joint_limits=True,
        dense_max_constraints=192,
        friction_anchor_beta=0.0,
    )
    owner = solver.factor.held
    clocks = getattr(solver, "_stage_clocks", None)
    print(
        f"\n== {kind}: dofs {model.joint_dof_count // WORLDS} bodies {model.body_count // WORLDS} owner {type(owner).__name__}",
        flush=True,
    )
    if clocks is None:
        print("no stage clocks (owner not installed or instrumentation off); stepping for the profiler only")
    groups = list(getattr(owner, "groups", []))
    print(
        "groups:",
        [
            (
                getattr(g.layout, "bodies", None),
                getattr(g.layout, "dofs", None),
                getattr(g.layout, "lanes", None),
                getattr(g.layout, "roots", None),
            )
            for g in groups
        ],
    )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    states = [model.state() for _ in range(2)]
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    control = model.control()
    control.joint_target_q.assign(np.zeros_like(model.joint_target_q.numpy()))
    for _ in range(12):
        pipeline.collide(states[0], contacts)
        states[0].clear_forces()
        solver.step(states[0], states[1], control, contacts, DT)
        states.reverse()
    wp.synchronize_device("cuda:0")
    # Time one more step's launches for an absolute scale.
    t0 = time.perf_counter()
    for _ in range(5):
        pipeline.collide(states[0], contacts)
        states[0].clear_forces()
        solver.step(states[0], states[1], control, contacts, DT)
        states.reverse()
    wp.synchronize_device("cuda:0")
    print(f"step wall {1e3 * (time.perf_counter() - t0) / 5:.3f} ms per step ({WORLDS} worlds)")
    if clocks is None:
        continue
    for step in range(3):
        pipeline.collide(states[0], contacts)
        states[0].clear_forces()
        solver.step(states[0], states[1], control, contacts, DT)
        states.reverse()
        wp.synchronize_device("cuda:0")
        c = clocks.numpy().astype(np.float64)
        print(
            f" step +{step}: finish {np.median(c[:, :8].sum(axis=1)):,.0f}  collect [reduce {np.median(c[:, 13]):,.0f} bias {np.median(c[:, 14]):,.0f} geometric {np.median(c[:, 15]):,.0f}]  predict {np.median(c[:, 8:13].sum(axis=1)):,.0f}"
        )
    fin, pre, col = c[:, :8], c[:, 8:13], c[:, 13:16]
    body0, free, motion = c[:, 16:19], c[:, 19:22], c[:, 22:25]
    for name, block, labels in (
        ("finish", fin, FINISH),
        ("collect sub-stages", col, COLLECT),
        ("body terms, lane of body 0", body0, ("math", "publish", "store")),
        ("body terms, free body lane", free, ("math", "publish", "store+finalize")),
        ("joint motion, lane of body 0", motion, ("anchor", "jcalc_motion", "axis/motion stores")),
        ("predict", pre, PREDICT),
    ):
        med = np.median(block, axis=0)
        total = med.sum()
        print(f"{name}: total {total:,.0f} cycles (median over groups)")
        for label, value, p90 in zip(labels, med, np.percentile(block, 90, axis=0), strict=True):
            print(f"   {label:16} {value:9,.0f}  {100 * value / max(total, 1):5.1f}%   p90 {p90:9,.0f}")
    print(
        "contacts",
        int(contacts.rigid_contact_count.numpy()[0]),
        "resolved worlds",
        int(solver._resolved_worlds.numpy().sum()) if solver._resolved_worlds.shape[0] else "n/a",
    )
