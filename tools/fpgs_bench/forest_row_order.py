# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Give the coupled-tree graph fixture a shared semantic contact-row order."""

from typing import Any

import warp as wp


@wp.kernel
def _order(
    counts: wp.array[int],
    kinds: wp.array2d[int],
    dofs: wp.array3d[int],
    factors: wp.array3d[float],
    contact_count: wp.array[int],
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_width: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    keys: wp.array2d[wp.int64],
    order: wp.array2d[int],
    inverse: wp.array2d[int],
):
    world = wp.tid()
    count = counts[world]
    for row in range(count):
        # A sparse inverse-factor column begins at its nonzero diagonal.
        # Its first support index and sign identify each scalar bound.
        keys[world, row] = wp.int64(kinds[world, row]) * wp.int64(1099511627776) + wp.int64(
            dofs[world, row, 0] * 2 + int(factors[world, row, 0] > 0.0)
        )
        order[world, row] = row
    for contact in range(contact_count[0]):
        slot = contact_slot[contact]
        if contact_world[contact] == world and slot >= 0:
            key = (
                wp.int64(1125899906842624)
                + wp.int64(shape0[contact] + 1) * wp.int64(16777216)
                + wp.int64(shape1[contact] + 1) * wp.int64(4)
            )
            for component in range(contact_width[contact]):
                keys[world, slot + component] = key + wp.int64(component)
    for row in range(1, count):
        item = order[world, row]
        position = row
        while position > 0:
            previous = order[world, position - 1]
            if keys[world, previous] <= keys[world, item]:
                break
            order[world, position] = previous
            position -= 1
        order[world, position] = item
    for row in range(count):
        inverse[world, order[world, row]] = row


@wp.kernel
def _gather2(source: wp.array2d[Any], order: wp.array2d[int], counts: wp.array[int], target: wp.array2d[Any]):
    world, row = wp.tid()
    target[world, row] = source[world, order[world, row]] if row < counts[world] else source[world, row]


@wp.kernel
def _gather3(source: wp.array3d[Any], order: wp.array2d[int], counts: wp.array[int], target: wp.array3d[Any]):
    world, row, column = wp.tid()
    target[world, row, column] = (
        source[world, order[world, row], column] if row < counts[world] else source[world, row, column]
    )


@wp.kernel
def _parents(counts: wp.array[int], inverse: wp.array2d[int], parents: wp.array2d[int]):
    world, row = wp.tid()
    if row < counts[world] and parents[world, row] >= 0:
        parents[world, row] = inverse[world, parents[world, row]]


@wp.kernel
def _slots(count: wp.array[int], worlds: wp.array[int], inverse: wp.array2d[int], slots: wp.array[int]):
    contact = wp.tid()
    if contact < count[0] and worlds[contact] >= 0 and slots[contact] >= 0:
        slots[contact] = inverse[worlds[contact], slots[contact]]


class RowOrder:
    """Retain test-only permutation scratch through eager and graph execution."""

    def __init__(self, case):
        self.solver, self.contacts = case[1], case[6]
        solver = self.solver
        self.order = wp.empty_like(solver.placement.row_type)
        self.inverse = wp.empty_like(solver.placement.row_type)
        self.keys = wp.empty(solver.placement.row_type.shape, dtype=wp.int64, device=solver.model.device)
        names = (
            "rhs",
            "diag",
            "impulses",
            "row_type",
            "row_parent",
            "row_mu",
            "row_beta",
            "row_cfm",
            "phi",
            "target_velocity",
            "row_restitution",
            "sparse_row_dof",
            "sparse_row_factor",
            "sparse_row_incident",
        )
        self.arrays = [
            (getattr(solver.placement, name), wp.empty_like(getattr(solver.placement, name))) for name in names
        ]
        self.solve = solver._launch_matrix_free_gs_solve

    def __call__(self, *args, **kwargs):
        solver, contacts = self.solver, self.contacts
        device = solver.model.device
        wp.launch(
            _order,
            dim=solver.forest.world_count,
            inputs=[
                solver.placement.constraint_count,
                solver.placement.row_type,
                solver.placement.sparse_row_dof,
                solver.placement.sparse_row_factor,
                contacts.rigid_contact_count,
                solver.placement.contact_world,
                solver.placement.contact_slot,
                solver.placement.contact_slots_needed,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                self.keys,
                self.order,
                self.inverse,
            ],
            device=device,
        )
        for source, target in self.arrays:
            wp.launch(
                _gather2 if source.ndim == 2 else _gather3,
                dim=source.shape,
                inputs=[source, self.order, solver.placement.constraint_count, target],
                device=device,
            )
            wp.copy(source, target)
        wp.launch(
            _parents,
            dim=solver.placement.row_parent.shape,
            inputs=[solver.placement.constraint_count, self.inverse, solver.placement.row_parent],
            device=device,
        )
        wp.launch(
            _slots,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                solver.placement.contact_world,
                self.inverse,
                solver.placement.contact_slot,
            ],
            device=device,
        )
        return self.solve(*args, **kwargs)
