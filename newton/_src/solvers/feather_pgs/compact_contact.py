# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The compact boundary: the sparse diagonal's contact rows written whole where the contacts are read.

A thread per contact row, in the contacts' collision order, writes the row by the one writer's unit walker on the slots
sink (:func:`produce_contacts`: its records by the contact law, the row walked as its own unit, its slots and header):
the rows the sparse diagonal's writer (:func:`.walk_rows.write_slot_rows`) writes from their unit sources, without
the unit sources' pass.
Allocation, admission, scheduling and the solve stay the original pipeline's. Persistent patch friction writes its
tangent rows with the patch row law; the patch link, ``_FrictionPatchState.link``, runs after this producer and alone
links the patch rings and divides the friction coefficient, in one ``link_patch_routes`` launch over every route.
"""

import warp as wp

from .contact_input import ContactInput
from .dense_rows import UNIT_CONTACT, DenseRows, unit_source
from .friction_patches import PatchRowLaw, patch_row_law
from .row_build import write_contact_row
from .walk_rows import (
    JointSeeds,
    PackedSink,
    SlotRows,
    bind_slot_rows,
    read_slot_unit,
    slot_header,
    slot_sink,
    walk_slot_row,
)


def supported(solver) -> bool:
    """Require the inherited complete sparse contract and its dense-six triple or unit mode.

    The sparse-diagonal owner exists only with its fused wide-articulation limits, whose admission already requires
    augmented drives, no velocity-limit, mimic or link rows, no debug rows and no pre-elimination: only the dense
    six position axes keep rows, at most lower + upper for each axis.
    """
    return bool(
        solver.model.device.is_cuda
        and solver.routes.case == "K2"
        and solver.placement.sparse_diagonal_dense_size == 6
        and solver.placement.sparse_diagonal_response_size > 6
        and (solver.placement.sparse_diagonal_contact_triples or solver.placement.sparse_diagonal_contact_units)
        and solver.placement.dense_max_constraints >= 12
        and 0 <= solver.placement.dense_internal_max_rows <= 12
        and solver._compact_diagonal_mass_size == solver.placement.sparse_diagonal_response_size
    )


@wp.kernel(enable_backward=False)
def produce_contacts(
    data: ContactInput,
    rows: DenseRows,
    slots: SlotRows,
    sink: PackedSink,
    law: PatchRowLaw,
    dt: float,
    contact_speculative_scale: float,
    contact_w: float,
    restitution_velocity_threshold: float,
    weighted: int,
    material: int,
):
    """Publish each contact's rows in their original collision order, one thread per row (3 per contact, 4 under
    contact torsion): its records by the contact law (:func:`.row_build.write_contact_row`), then the row walked as
    its own unit by the one writer's unit walker with the star (:func:`.walk_rows.packed_walker`), its slots and its
    header (:func:`.walk_rows.slot_header`).

    Tangent rows act at the friction location and follow the patch row law (``patch_tangent_row``): under
    persistent patches at the patch friction location with the anchor's tangential phi, ``law.beta`` and the
    undivided coefficient; under point friction (``law.patches.enabled`` 0, beta 0, a zero phi view) at the
    contact's friction points.
    """
    contact, row = wp.tid()
    if contact >= data.count[0]:
        return
    slot = data.slot[contact]
    if data.path[contact] != 0 or slot < 0 or row >= data.slots_needed[contact]:
        return
    world = data.world[contact]
    body_a, body_b, direction, row_point = write_contact_row(data, law, contact, row, False)
    unit = read_slot_unit(
        rows, data, law, JointSeeds(), data.first[world], slot + row, unit_source(UNIT_CONTACT, 4 * contact + row),
        body_a, body_b, direction, row_point,
    )  # fmt: skip
    unit.group = slots.dense_group[world]
    walk_slot_row(sink, unit)
    slot_header(
        rows, slots, data, data.incident, unit, dt, contact_speculative_scale, contact_w,
        restitution_velocity_threshold, weighted, material,
    )  # fmt: skip


def launch_contacts(solver, data: ContactInput, dt: float) -> None:
    """Produce the contact rows from the step's bound contact input; Warp capture retains these launch descriptors."""
    from .dense_rows import bind as bind_dense_rows  # noqa: PLC0415

    # A thread per (contact, row); in a capture whose launches bind, each replay runs the live count. The rows
    # take their own axis: the contact axis is captured at the pool's bound, which a scalar thread index times 3
    # would overflow.
    pool = solver.placement.contact_pool
    laws = solver.laws
    pool.launch(
        produce_contacts,
        (pool.launch_rows(), 4 if laws.contact_torsion_enabled else 3),
        inputs=[
            data,
            bind_dense_rows(solver.placement),
            bind_slot_rows(solver),
            slot_sink(solver),
            patch_row_law(solver),
            dt,
            solver.options.contact_speculative_scale,
            laws.contact_w,
            laws.effective_restitution_velocity_threshold,
            int(laws.set.weighted),
            int(solver.options.contact_compliance),
        ],
        extent=data.count,
    )
