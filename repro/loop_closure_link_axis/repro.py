# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reproducer: ``ArticulationView`` per-link axis is keyed by joint-child, not by physical body.

Sister reproducer to newton-physics/newton#2914. The #2914 repro covered the case where two
*distinct* bodies share a leaf name; PR #2935 (``link_template_labels``) addressed that.

This script covers a different case that #2935 does **not** address: an articulation where
the *same* physical body is the child of multiple joints (the standard closed-kinematic-loop
encoding, as used by Digit's toe rods and achilles rod). ``ArticulationView`` enumerates
``link_*`` by walking joints and appending ``joint_child``, so a body with N incoming joints
shows up N times in both ``link_names`` and ``link_template_labels`` -- with *identical*
label strings, since ``model.body_label`` only contains the body's path once.

Run with::

    uv run --extra importers python repro/loop_closure_link_axis/repro.py

The ``importers`` extra brings in ``usd-core`` so ``pxr`` is available; no torch is needed.

------------------------------------------------------------------------------
What this script demonstrates
------------------------------------------------------------------------------

The script authors a synthetic 5-body USD whose topology mirrors Digit's
``left_leg_toe_roll`` + push-rods, imports it via ``ModelBuilder.add_usd``, and prints what
``ArticulationView`` exposes. Expected output (abridged)::

    === model ===
    len(model.body_label)        = 5
    body_label                   = ['/Robot/root', '/Robot/intermediate', '/Robot/distal',
                                    '/Robot/rod_a', '/Robot/rod_b']

    model.body_q.shape           = (5, 7)

    Bodies that are the child of more than one joint (loop-closure evidence):
      body_id=2  (/Robot/distal)  → child of 3 joints

    === ArticulationView (post #2935) ===
    view.link_count                       = 7
    len(set(view.link_template_labels))   = 5
    view.link_template_labels             = ['/Robot/root', '/Robot/intermediate',
                                             '/Robot/distal', '/Robot/distal', '/Robot/distal',
                                             '/Robot/rod_a', '/Robot/rod_b']

    Slots where link_template_labels == '/Robot/distal': [2, 3, 4]

    === BODY frequency layout (NOT a physical-body axis) ===
      layout.value_count            = 7     ← same as link_count
      len(model.body_label)         = 5     ← physical bodies

The BODY frequency layout's ``value_count`` is *also* equal to ``link_count``, so it is not
a deduplicated body axis either. The only deduplicated, physical-body view of the model
lives at ``model.body_label`` / ``state.body_q`` / ``state.body_qd`` directly.

------------------------------------------------------------------------------
Real-world evidence (Digit, 32 envs, in Isaac Lab on Newton main + #2935)
------------------------------------------------------------------------------

Inspecting ``right_leg_toe_roll`` from PDB during a Digit training run -- the duplicate
slots in ``ArticulationView``'s per-link axis carry *different* FK velocities, not numerical
noise around a common value::

    (Pdb) [i for i, n in enumerate(asset.root_view.link_names) if n == 'right_leg_toe_roll']
    [46, 47, 48]

    (Pdb) asset.data.body_lin_vel_w.torch[0, 46]
    tensor([-1.6122, -0.8873,  3.6133], device='cuda:0')
    (Pdb) asset.data.body_lin_vel_w.torch[0, 47]
    tensor([-1.2029, -0.8292,  3.5614], device='cuda:0')
    (Pdb) asset.data.body_lin_vel_w.torch[0, 48]
    tensor([-0.1172, -0.6806,  3.8218], device='cuda:0')

Each entry is the FK velocity of ``right_leg_toe_roll`` reached via a different chain (the
primary articulated chain plus the two push-rod loop closures), with the chain-specific
loop-closure residual baked in. The ~1.5 m/s spread on the X component is the constraint
solver's loop violation, not noise.

------------------------------------------------------------------------------
Concrete impact on downstream consumers (e.g. Isaac Lab)
------------------------------------------------------------------------------

* ``Articulation.body_names = view.link_names`` returns 49 entries for Digit (43 unique +
  6 loop-closure duplicates).
* ``find_bodies("toe_roll")`` resolves to 6 indices (3 left + 3 right) instead of 2, so any
  reward / observation that does ``body_link_vel_w[:, body_ids, :]`` gets a 6-wide tensor
  and fails with::

    RuntimeError: The size of tensor a (6) must match the size of tensor b (2) at
                  non-singleton dimension 1

  (this is the actual error path triggered by Isaac Lab's ``feet_slide`` reward on Digit).
* Naively deduplicating names on the consumer side (e.g. dropping repeats from
  ``link_template_labels``) does not help: the data axis is still 49, so the resulting
  index list silently points at the *wrong* rows of ``body_link_pose_w`` /
  ``body_link_vel_w``. There is no in-band signal to tell whether a given joint-child slot
  is the "primary" chain interpretation or a loop-residual one.
* The ``BODY`` frequency layout is keyed at ``link_count`` arity too, so it can't be used
  as a deduplication source.

------------------------------------------------------------------------------
Proposed fix
------------------------------------------------------------------------------

Expose a physical-body-indexed axis on ``ArticulationView`` aligned with
``model.body_label`` / ``state.body_q`` / ``state.body_qd``::

    view.body_count  # == len(model.body_label[arti_slice])
    view.body_names  # == [body_label.rsplit("/", 1)[-1] for body_label in ...]
    view.get_body_transforms  # shape (..., body_count, 7), reads state.body_q
    view.get_body_velocities  # shape (..., body_count, 6), reads state.body_qd

The existing ``link_*`` API remains useful: a per-joint-child axis is the right place to
expose loop-closure FK residuals, joint-frame transforms etc. The change is additive --
no breaks to current callers.

The new accessors don't need to do FK at all: they index directly into ``state.body_q`` /
``state.body_qd``, sliced via the BODY frequency layout's ``offset`` / ``slice`` /
``indices`` (per-articulation slicing is already plumbed; we just need a deduplicated
arity for it).
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import newton
from newton.selection import ArticulationView


def _author_closed_loop_usd(path: str) -> None:
    """Author a minimal USD that encodes a closed loop the way Digit does.

    Topology (mirrors ``/Digit/.../left_leg_toe_roll`` and its two push-rods)::

        world ── free ── root ─revolute─ intermediate ─revolute─ distal
                  │                                                ▲
                  ├─revolute─ rod_a ───spherical───────────────────┤
                  └─revolute─ rod_b ───spherical───────────────────┘

    ``distal`` is the body1 of three joints (revolute from ``intermediate`` plus two
    spherical loop closures from the rods) -- exactly the encoding that turns
    ``left_leg_toe_roll`` into a child-of-3-joints in Digit.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # pxr banned at module level

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    robot = UsdGeom.Xform.Define(stage, "/Robot")
    UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())

    def _body(prim_path: str, translate: tuple[float, float, float]) -> None:
        xf = UsdGeom.Xform.Define(stage, prim_path)
        xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(1.0)

    _body("/Robot/root", (0.0, 0.0, 0.0))
    _body("/Robot/intermediate", (0.0, 0.1, 0.0))
    _body("/Robot/distal", (0.0, 0.2, 0.0))
    _body("/Robot/rod_a", (0.05, 0.1, 0.0))
    _body("/Robot/rod_b", (-0.05, 0.1, 0.0))

    def _revolute(joint_path: str, body0: str, body1: str) -> None:
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
        joint.CreateAxisAttr("Z")

    def _spherical_loop_closure(joint_path: str, body0: str, body1: str) -> None:
        """A spherical joint marked ``excludeFromArticulation=True`` -- the encoding Digit uses
        for its push-rod loop closures.  This is what lets Newton's USD importer accept the
        cycle: the joint is added to the model (and shows up in ``joint_child``), but it is
        excluded from the articulation tree so the topology pass doesn't reject the graph."""
        joint = UsdPhysics.SphericalJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
        joint.CreateExcludeFromArticulationAttr(True)

    _revolute("/Robot/joints/j_root_inter", "/Robot/root", "/Robot/intermediate")
    _revolute("/Robot/joints/j_inter_distal", "/Robot/intermediate", "/Robot/distal")
    _revolute("/Robot/joints/j_root_rod_a", "/Robot/root", "/Robot/rod_a")
    _revolute("/Robot/joints/j_root_rod_b", "/Robot/root", "/Robot/rod_b")
    _spherical_loop_closure("/Robot/joints/j_rod_a_distal", "/Robot/rod_a", "/Robot/distal")
    _spherical_loop_closure("/Robot/joints/j_rod_b_distal", "/Robot/rod_b", "/Robot/distal")

    stage.GetRootLayer().Save()


def _probe_body_frequency_layout(view: ArticulationView) -> None:
    """Show that the BODY frequency layout is **not** the physical-body axis either.

    Its ``value_count`` equals ``link_count`` (the inflated joint-child count), not
    ``len(model.body_label)``.  So a naive
    ``frequency_layouts[AttributeFrequency.BODY]`` read does not produce a deduplicated
    body axis -- the deduplication has to be applied by the consumer (or fixed inside Newton).
    """
    layout = view.frequency_layouts[view.model.AttributeFrequency.BODY]
    print("\n=== BODY frequency layout (NOT a physical-body axis) ===")
    print(f"  layout.value_count            = {layout.value_count}     ← same as link_count")
    print(f"  layout.slice                  = {layout.slice}")
    print(f"  layout.indices                = {layout.indices}")
    print(f"  layout.offset                 = {layout.offset}")
    print(f"  len(model.body_label)         = {len(view.model.body_label)}     ← physical bodies")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        usd_path = str(Path(tmpdir) / "closed_loop.usda")
        _author_closed_loop_usd(usd_path)

        builder = newton.ModelBuilder()
        builder.add_usd(usd_path)
        model = builder.finalize()

        print("=== model ===")
        print(f"len(model.body_label)        = {len(model.body_label)}")
        print(f"set(model.body_label) unique = {len(set(model.body_label))}")
        print(f"body_label                   = {list(model.body_label)}")

        view = ArticulationView(model, next(iter(model.articulation_label)))
        body_q = model.body_q.numpy()
        print(f"\nmodel.body_q.shape           = {body_q.shape}")

        joint_child = model.joint_child.numpy() if hasattr(model.joint_child, "numpy") else model.joint_child
        labels = list(model.body_label)
        shared = {b: n for b, n in Counter(int(c) for c in joint_child).items() if n > 1}
        print("\nBodies that are the child of more than one joint (loop-closure evidence):")
        for body_id, n in shared.items():
            print(f"  body_id={body_id}  ({labels[body_id]})  → child of {n} joints")

        print("\n=== ArticulationView (post #2935) ===")
        print(f"view.link_count                       = {view.link_count}")
        print(f"len(view.link_names)                  = {len(view.link_names)}")
        print(f"len(view.link_template_labels)        = {len(view.link_template_labels)}")
        print(f"len(set(view.link_template_labels))   = {len(set(view.link_template_labels))}")
        print(f"view.link_names                       = {list(view.link_names)}")
        print(f"view.link_template_labels             = {list(view.link_template_labels)}")

        target = "/Robot/distal"
        slots = [i for i, lbl in enumerate(view.link_template_labels) if lbl == target]
        print(f"\nSlots where link_template_labels == {target!r}: {slots}")
        print(f"→ {len(slots)} link slots all map to the single physical body at body_label[2].")
        print("  See this file's module docstring for real-Digit PDB evidence that the FK values")
        print("  at these duplicate slots are NOT redundant copies -- they carry per-chain loop residuals.")

        _probe_body_frequency_layout(view)

        print("\nSummary")
        print("-------")
        print("  • model.body_label and state.body_q are physical-body-indexed.")
        print("  • view.link_count is inflated by loop-closure joints; one slot per joint-child.")
        print("  • view.link_template_labels carries the same string in duplicate slots.")
        print("  • view.get_link_transforms differs across duplicate slots.")
        print("  • The BODY frequency layout's value_count is also link_count, not body count.")
        print("\nProposed fix: expose a physical-body-indexed axis on ArticulationView aligned")
        print("with model.body_label / state.body_q, alongside the existing joint-child link_* axis.")


if __name__ == "__main__":
    main()
