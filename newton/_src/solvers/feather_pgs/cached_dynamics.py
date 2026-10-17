# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Select the cached-dynamics owner that predicts and publishes over the existing operators."""

import os


def install(solver) -> None:
    """Install an admitted cache without changing its factor or row policy."""
    solver._tree_kinetic_state = None
    choice = solver.cached_dynamics
    if choice == "off":
        solver._cached_dynamics_selection_reason = "Explicit cached_dynamics='off' retains ordinary dynamics."
        return
    if solver._explicit_dense_kernels:
        solver._cached_dynamics_selection_reason = "An explicit dense kernel choice retains ordinary dynamics."
        return
    if solver.model.device.is_cpu:
        solver._cached_dynamics_selection_reason = "CPU uses ordinary dynamics; cached prediction requires CUDA."
        return

    disabled = []

    def enabled(option: str, *, default: bool = True) -> bool:
        admitted = choice == "on" or os.environ.get(f"FEATHER_PGS_{option}", "1" if default else "0") == "1"
        if not admitted:
            disabled.append(f"FEATHER_PGS_{option}")
        return admitted

    if enabled("TREE_KINETIC_STATE"):
        from .tree_kinetic_sparse import create_owner  # noqa: PLC0415

        solver._tree_kinetic_state = create_owner(solver)

    # The held-tree owner serves dense-route scenes: per-size groups or, where
    # the layout admits it, one cooperative group per world (the forest).
    if solver._tree_kinetic_state is None and enabled("MIXED_TREE_KINETIC_STATE"):
        from .mixed_tree_kinetic import create_owner  # noqa: PLC0415

        solver._tree_kinetic_state = create_owner(solver)
        if solver._tree_kinetic_state is not None:
            # The owner predicts kinematics and joint torques from its cached
            # state; the fused world-dynamics kernel (K1) that would produce
            # them on the general route is superseded, and every later stage
            # takes its ordinary branch.
            solver._fused_k1 = False

    if solver._tree_kinetic_state is not None:
        source = "Explicit cached_dynamics='on'" if choice == "on" else "Automatic cache selection"
        solver._cached_dynamics_selection_reason = (
            f"{source}: tree cache admitted for this topology, representation and state lifecycle."
        )
    else:
        solver._cached_dynamics_selection_reason = (
            "No enabled cache admitted this topology, representation and state lifecycle; ordinary dynamics retained."
        )
    if disabled:
        solver._cached_dynamics_selection_reason += f" Disabled cache options: {', '.join(disabled)}."
    if choice == "on" and solver._tree_kinetic_state is None:
        raise ValueError(
            "cached_dynamics='on' has no admitted cache for this topology, representation and solver policy"
        )
