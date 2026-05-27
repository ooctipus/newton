# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reproducer: ``ArticulationView`` per-link axis is keyed by joint-child, not by physical body.

Loads Agility Digit's ``digit_v4.usd`` (bundled under ``asset/``), imports it via
``newton.ModelBuilder.add_usd``, builds an ``ArticulationView``, and empirically
demonstrates:

1. ``view.link_count == 49`` while there are only 43 unique physical bodies in the
   articulation -- ``link_*`` over-counts by 6 = the number of loop-closure joints
   (achilles + 4 toe push-rods, both legs).
2. ``view.link_names`` and ``view.link_template_labels`` both have 49 entries with several
   identical strings (e.g. three copies of ``right_leg_toe_roll``). Identical *full paths*
   in ``link_template_labels`` rule out the alternate hypothesis that this is a leaf-name
   collision between distinct bodies (which is what #2935 addressed).
3. The duplicate link slots resolve to the *same* physical body via the sorted
   ``arti_link_ids`` mapping that ``ArticulationView.__init__`` builds.
4. The BODY ``FrequencyLayout`` exposes the inflated 49-wide axis even though
   ``model.body_q`` is 43-wide, so any view accessor that gathers through that layout
   (e.g. ``get_link_transforms``, ``get_link_velocities``) reads past the end of the
   underlying state buffer once a state is allocated to the model's body count.

Sister reproducer to newton-physics/newton#2914 / PR #2935. #2935 plumbed
``link_template_labels`` to expose full paths so distinct bodies with colliding leaf
names are disambiguated -- but that does *not* address the case shown here, where the
*same* physical body appears multiple times along the per-link axis because
``ArticulationView`` walks ``joint_child`` without deduplicating.

Independent observation in Isaac Lab on Digit (32 envs) confirms the consequence: when a
solver state is allocated at the inflated ``link_count`` arity, FK at rest pose populates
the duplicate slots with per-chain results that disagree by the loop-closure residual --
``body_lin_vel_w[:, [duplicate_slots]]`` returns three different values for what the user
asked for as a single body.

Run with::

    uv run --extra importers python repro/loop_closure_link_axis/repro.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import newton
from newton.selection import ArticulationView


DIGIT_USD = Path(__file__).resolve().parent / "asset" / "digit_v4.usd"


def _show_link_axis(view: ArticulationView, model: newton.Model) -> tuple[list[str], list[str]]:
    print("\n=== ArticulationView per-link axis ===")
    print(f"  view.link_count                       = {view.link_count}")
    print(f"  len(view.link_names)                  = {len(view.link_names)}")
    print(f"  len(set(view.link_names))             = {len(set(view.link_names))}")
    print(f"  len(view.link_template_labels)        = {len(view.link_template_labels)}")
    print(f"  len(set(view.link_template_labels))   = {len(set(view.link_template_labels))}")
    print(f"  len(model.body_label)                 = {len(model.body_label)}")
    print(f"  len(set(model.body_label))            = {len(set(model.body_label))}")
    return list(view.link_names), list(view.link_template_labels)


def _show_link_template_label_dups(template_labels: list[str]) -> dict[str, list[int]]:
    """Group joint-child slot indices by full path. Duplicate full paths are the smoking
    gun: identical full paths cannot be a leaf-name collision between distinct bodies."""
    by_label: dict[str, list[int]] = {}
    for i, lbl in enumerate(template_labels):
        by_label.setdefault(lbl, []).append(i)
    dups = {lbl: idxs for lbl, idxs in by_label.items() if len(idxs) > 1}
    print(f"\n=== {len(dups)} link_template_labels appear at multiple per-link slots ===")
    for lbl, idxs in sorted(dups.items()):
        print(f"  slots {idxs}  ({len(idxs)}x)  →  {lbl!r}")
    return dups


def _reconstruct_arti_link_ids(model: newton.Model) -> list[int]:
    """Reproduce ``ArticulationView``'s per-link → body-id mapping for a single-articulation
    model. The view builds ``arti_link_ids`` by walking each joint in the articulation's
    joint range, appending ``joint_child[joint_id]`` (with duplicates), then ``sorted()``.
    The i-th link slot's body_id is ``arti_link_ids[i]``."""
    joint_child = model.joint_child.numpy()
    arti_start = model.articulation_start.numpy()
    arti_joint_begin = int(arti_start[0])
    arti_joint_end = int(arti_start[1]) if len(arti_start) > 1 else len(joint_child)
    return sorted(int(joint_child[j]) for j in range(arti_joint_begin, arti_joint_end))


def _show_joint_child_for_dups(model: newton.Model, dups: dict[str, list[int]]) -> None:
    """For each duplicated label, show that all duplicate slots map to a single physical
    body via ``arti_link_ids``."""
    print("\n=== Mapping duplicate link slots back to physical bodies ===")
    arti_link_ids = _reconstruct_arti_link_ids(model)
    labels = list(model.body_label)
    for lbl, idxs in sorted(dups.items()):
        body_ids = [arti_link_ids[i] for i in idxs]
        print(f"  {lbl!r}")
        print(f"    link slots               = {idxs}")
        print(f"    arti_link_ids[slots]     = {body_ids}  ({labels[body_ids[0]]!r})")
        print(f"    distinct physical bodies = {len(set(body_ids))}")


def _show_body_data_layout(view: ArticulationView, model: newton.Model) -> None:
    """Show the smoking-gun layout divergence: the BODY frequency layout's ``value_count``
    is the inflated joint-child count, while ``model.body_q`` is body-count. Any view
    accessor that goes through the BODY layout (``get_link_transforms``,
    ``get_link_velocities``) ends up slicing past the end of the underlying tensor."""
    print("\n=== Body data layout divergence ===")
    print(f"  model.body_q.shape            = {tuple(model.body_q.shape)}     ← physical bodies")
    layout = view.frequency_layouts[view.model.AttributeFrequency.BODY]
    print(
        f"  BODY frequency layout         value_count={layout.value_count}, slice={layout.slice},"
        f" offset={layout.offset}, indices={'<set>' if layout.indices is not None else None}"
    )
    print(
        "  → ArticulationView exposes the per-link axis at link_count arity, but the"
        " underlying model.body_q is at the physical-body arity. They diverge whenever"
        " loop-closure joints land in the articulation joint range."
    )


def main() -> None:
    print(f"Loading {DIGIT_USD}")

    builder = newton.ModelBuilder()
    builder.add_usd(str(DIGIT_USD))
    model = builder.finalize()

    print(f"\n=== model ===")
    print(f"  len(model.body_label)             = {len(model.body_label)}")
    print(f"  len(set(model.body_label))        = {len(set(model.body_label))}")
    print(f"  len(model.joint_child)            = {len(model.joint_child)}")

    body_id_counts = Counter(int(c) for c in model.joint_child.numpy())
    multi_parent_bodies = {b: n for b, n in body_id_counts.items() if n > 1}
    print(f"  bodies that are children of >1 joint  = {len(multi_parent_bodies)}")
    if multi_parent_bodies:
        labels = list(model.body_label)
        for body_id, n in sorted(multi_parent_bodies.items()):
            print(f"    body_id={body_id:>3}  ({labels[body_id]!r})  → child of {n} joints")

    arti_pattern = next(iter(model.articulation_label))
    view = ArticulationView(model, arti_pattern)

    _, template_labels = _show_link_axis(view, model)
    dups = _show_link_template_label_dups(template_labels)
    if not dups:
        print("\nNo duplicate link_template_labels in this articulation -- nothing to demonstrate.")
        return
    _show_joint_child_for_dups(model, dups)
    _show_body_data_layout(view, model)

    print("\nSummary")
    print("-------")
    print("  • link_template_labels has identical *full paths* at duplicate link slots,")
    print("    refuting the leaf-name-collision hypothesis -- those slots all map to the")
    print("    SAME physical body via the sorted arti_link_ids mapping.")
    print("  • ArticulationView's walk over joint_child does not deduplicate, so loop-")
    print("    closure joints (Digit's achilles + toe push-rods) inflate link_count past")
    print("    the actual physical body count.")
    print("  • The BODY FrequencyLayout is built at the inflated 49 arity, but model.body_q")
    print("    is 43-wide, so accessors that gather through the layout slice past the end")
    print("    of the underlying state buffer.")
    print("\nProposed fix: in ArticulationView.__init__, dedup arti_link_ids while preserving")
    print("model order. For tree articulations this is a no-op; for closed-loop articulations")
    print("it makes link_count match the number of unique physical bodies and aligns the BODY")
    print("frequency layout with model.body_q.")


if __name__ == "__main__":
    main()
