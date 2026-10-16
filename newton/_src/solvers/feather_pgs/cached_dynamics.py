# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Select paired cached prediction and publication over existing operators."""

import os


def reserve_native_patch_factors(solver) -> bool:
    """Reserve factors for an explicitly requested native patch cache."""
    if not solver._friction_anchors_enabled or solver.sparse_mass_matrix != "auto" or solver.cached_dynamics != "on":
        return False
    from .kinetic_live_owner import plan_before_factor_storage  # noqa: PLC0415

    solver._kinetic_world_plan = plan_before_factor_storage(solver)
    return solver._kinetic_world_plan is not None


def install(solver, *, factor_warps: int) -> None:
    """Install an admitted cache without changing its factor or row policy."""
    solver._kinetic_world = None
    solver._tree_kinetic_state = None
    solver._g1_kinetic_state = None
    solver._franka_kinetic_state = None
    choice = solver.cached_dynamics
    if choice == "off":
        solver._cached_dynamics_selection_reason = "Explicit cached_dynamics='off' retains ordinary dynamics."
        return
    if solver._explicit_dense_kernels:
        solver._cached_dynamics_selection_reason = "An explicit dense kernel choice retains ordinary dynamics."
        return
    if solver.model.device.is_cpu:
        solver._cached_dynamics_selection_reason = "CPU uses ordinary dynamics; paired caches require CUDA."
        return

    disabled = []

    def enabled(option: str, *, default: bool = True) -> bool:
        admitted = choice == "on" or os.environ.get(f"FEATHER_PGS_{option}", "1" if default else "0") == "1"
        if not admitted:
            disabled.append(f"FEATHER_PGS_{option}")
        return admitted

    if (
        solver._sparse_mass_matrix_size is None
        # Native patch rows are physically qualified but slower than sparse
        # rows in the measured Kuka workload. Keep them explicitly selected.
        and (not solver._friction_anchors_enabled or choice == "on")
        and not solver._contact_torsion_enabled
        and not solver.contact_compliance
        and enabled("KUKA_KINETIC_WORLD")
    ):
        from .kinetic_live_owner import create_owner  # noqa: PLC0415

        solver._kinetic_world = create_owner(solver)

    if solver._sparse_factor is not None and enabled("G1_KINETIC_STATE"):
        solver._sparse_factor.install_kinetic_state()
        solver._tree_kinetic_state = solver._sparse_factor.kinetic_state
        # Retain the original packed-consumer marker used by saved captures.
        solver._g1_kinetic_state = solver._tree_kinetic_state

    if solver._tree_kinetic_state is None and enabled("TREE_KINETIC_STATE"):
        from .tree_kinetic_sparse import create_owner  # noqa: PLC0415

        solver._tree_kinetic_state = create_owner(solver)

    # Held Cholesky dynamics publishes complete body state for the ordinary
    # patch-friction row pipeline. Its cache does not own contact rows or history.
    if (
        solver._sparse_mass_matrix_size is None
        and not solver._contact_torsion_enabled
        and not solver.contact_compliance
        and enabled("FRANKA_KINETIC_STATE")
    ):
        from .franka_kinetic_factor import get_kernel  # noqa: PLC0415
        from .franka_kinetic_state import create_owner  # noqa: PLC0415

        solver._franka_kinetic_state = create_owner(solver)
        if solver._franka_kinetic_state is not None:
            solver._franka_kinetic_factor = get_kernel(
                str(solver.model.device.arch),
                warps_per_block=factor_warps,
                size=solver._franka_kinetic_state.layout.primary_dofs,
            )

    # The generic held-tree owner serves dense-route scenes no other owner
    # takes (measured: Franka cabinet -4.8%). Where the Franka held owner is
    # admitted it stays first: its per-world forest is still faster on its
    # scene (4.74 vs 5.42 ms) until the generic owner has that layout.
    if (
        solver._tree_kinetic_state is None
        and solver._franka_kinetic_state is None
        and enabled("MIXED_TREE_KINETIC_STATE")
    ):
        from .mixed_tree_kinetic import create_owner  # noqa: PLC0415

        solver._tree_kinetic_state = create_owner(solver)

    owners = [
        name
        for name, owner in (
            ("tree", solver._tree_kinetic_state),
            ("held Cholesky", solver._franka_kinetic_state),
            ("kinetic", solver._kinetic_world),
        )
        if owner is not None
    ]
    if owners:
        source = "Explicit cached_dynamics='on'" if choice == "on" else "Automatic cache selection"
        solver._cached_dynamics_selection_reason = (
            f"{source}: {', '.join(owners)} cache admitted for this topology, representation and state lifecycle."
        )
    else:
        solver._cached_dynamics_selection_reason = (
            "No enabled cache admitted this topology, representation and state lifecycle; ordinary dynamics retained."
        )
    if disabled:
        solver._cached_dynamics_selection_reason += f" Disabled cache options: {', '.join(disabled)}."
    if choice == "on" and not owners:
        raise ValueError(
            "cached_dynamics='on' has no admitted cache for this topology, representation and solver policy"
        )
