# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Select paired cached prediction and publication over existing operators."""

import os


def install(solver, *, factor_warps: int) -> None:
    """Install an admitted cache without changing its factor or row policy."""
    solver._kinetic_world = None
    solver._tree_kinetic_state = None
    solver._g1_kinetic_state = None
    solver._franka_kinetic_state = None
    choice = solver.cached_dynamics
    if choice == "off" or solver._explicit_dense_kernels or solver.model.device.is_cpu:
        return

    def enabled(option: str, *, default: bool = True) -> bool:
        return choice == "on" or os.environ.get(f"FEATHER_PGS_{option}", "1" if default else "0") == "1"

    if solver._sparse_mass_matrix_size is None and solver._point_contact_paths and enabled("KUKA_KINETIC_WORLD"):
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

    if solver._sparse_mass_matrix_size is None and solver._point_contact_paths and enabled("FRANKA_KINETIC_STATE"):
        from .franka_kinetic_factor import get_kernel  # noqa: PLC0415
        from .franka_kinetic_state import create_owner  # noqa: PLC0415

        solver._franka_kinetic_state = create_owner(solver)
        if solver._franka_kinetic_state is not None:
            solver._franka_kinetic_factor = get_kernel(
                str(solver.model.device.arch),
                warps_per_block=factor_warps,
                size=solver._franka_kinetic_state.layout.primary_dofs,
            )

    # Mixed held factors have physical coverage but no established performance
    # benefit. Retain explicit admission until their cost ranking justifies auto.
    if solver._tree_kinetic_state is None and enabled("MIXED_TREE_KINETIC_STATE", default=False):
        from .mixed_tree_kinetic import create_owner  # noqa: PLC0415

        solver._tree_kinetic_state = create_owner(solver)

    if choice == "on" and not any(
        owner is not None for owner in (solver._tree_kinetic_state, solver._franka_kinetic_state, solver._kinetic_world)
    ):
        raise ValueError(
            "cached_dynamics='on' has no admitted cache for this topology, representation and solver policy"
        )
