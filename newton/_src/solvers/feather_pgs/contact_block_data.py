# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental raw-order coupled geometry and contact-local response products."""

from functools import cache

import warp as wp

from . import compact_contact as compact
from . import private_contact_islands as private
from .fused_contact_solve import BiasData, SolveData


@wp.struct
class BlockContactData:
    raw_tag: wp.array[int]
    gram: wp.array2d[float]


def allocate_block_data(max_contacts: int, device) -> BlockContactData:
    """Allocate exactly the existing raw capacity, without a positive-size floor."""
    if max_contacts < 0:
        raise ValueError("Contact block capacity must be nonnegative")
    data = BlockContactData()
    data.raw_tag = wp.zeros(max_contacts, dtype=int, device=device)
    data.gram = wp.empty((max_contacts, 4), dtype=float, device=device)
    return data


@wp.kernel
def route_contacts(data: compact.ContactBoundaryData, row_contact: wp.array2d[int], block: BlockContactData):
    """Reset current raw ownership while retaining the original inverse row map."""
    contact = wp.tid()
    block.raw_tag[contact] = 0
    if contact >= wp.min(data.count[0], data.slot.shape[0]):
        return
    slot = data.slot[contact]
    if data.path[contact] == 0 and slot >= 0:
        world = data.world[contact]
        for component in range(data.slots_needed[contact]):
            if slot + component < row_contact.shape[1]:
                row_contact[world, slot + component] = contact


_OWNER_PUBLICATION = "        routing.owner.data[world] = 1;"
_TAG_PUBLICATION = """        // Publish tags only in the final, completely admitted lane-zero branch.
        for (int i = 0; i < (s_residual_count - prefix) / 3; ++i)
            block.raw_tag.data[s_residual_contacts[i]] = 1;
"""


def partition_source() -> str:
    """Insert only final tag publication into the original bounded partition."""
    original = private._PARTITION
    if original.count(_OWNER_PUBLICATION) != 1:
        raise RuntimeError("Private island admission publication source changed")
    source = original.replace(_OWNER_PUBLICATION, _TAG_PUBLICATION + _OWNER_PUBLICATION)
    if source.replace(_TAG_PUBLICATION, "") != original:
        raise RuntimeError("Contact block partition modified original admission")
    return source


@cache
def get_partition_kernel(device_arch: str):
    """Retain one warp/world and the original CPU lane-zero control."""
    del device_arch
    source = partition_source()

    @wp.func_native(source)
    def native(
        world: int,
        lane: int,
        data: compact.ContactBoundaryData,
        counts: wp.array[int],
        bounds: wp.array2d[int],
        routing: private.IslandRouting,
        block: BlockContactData,
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def partition(
        data: compact.ContactBoundaryData,
        counts: wp.array[int],
        bounds: wp.array2d[int],
        routing: private.IslandRouting,
        block: BlockContactData,
    ):
        world, lane = wp.tid()
        native(world, lane, data, counts, bounds, routing, block)

    return partition


@wp.func
def response_product(jacobian: compact.ContactRow, response: compact.ContactRow):
    """Form J_i dot Y_j in physical coordinates, without a symmetry assumption."""
    value = float(0.0)
    for dof in range(6):
        value += jacobian.J[dof] * response.Y[dof]
    if jacobian.dof0 >= 0:
        if jacobian.dof0 == response.dof0:
            value += jacobian.j0 * response.y0
        if jacobian.dof0 == response.dof1:
            value += jacobian.j0 * response.y1
    if jacobian.dof1 >= 0:
        if jacobian.dof1 == response.dof0:
            value += jacobian.j1 * response.y0
        if jacobian.dof1 == response.dof1:
            value += jacobian.j1 * response.y1
    return value


@wp.func
def publish_row(
    data: compact.ContactBoundaryData,
    solve: SolveData,
    bias: BiasData,
    geometry: compact.ContactGeometry,
    component: int,
    packet: compact.ContactRow,
):
    """Publish current coefficients, including scalar-only dense zeros, and RHS."""
    compact.publish_contact_row(data, geometry, component, packet)
    row = geometry.slot + component
    if not geometry.has_dense:
        # Admitted coupled rows bypass the fallback dense clear. A reserved
        # scalar/key-key contact must not inherit last step's dense response.
        for dof in range(6):
            data.J[geometry.group, row, dof] = 0.0
            data.Y[geometry.group, row, dof] = 0.0
    solve.rhs_bias[geometry.world, row] = private.contact_rhs(solve, bias, geometry.world, packet)


@wp.kernel
def produce_contacts(
    data: compact.ContactBoundaryData,
    solve: SolveData,
    bias: BiasData,
    routing: private.IslandRouting,
    block: BlockContactData,
):
    """Prepare only current admitted coupled triples with raw-contact workers."""
    contact = wp.tid()
    if contact >= wp.min(data.count[0], wp.min(data.slot.shape[0], data.point0.shape[0])):
        return
    if block.raw_tag[contact] != 1:
        return
    world = data.world[contact]
    if routing.owner[world] != 1 or data.path[contact] != 0 or data.slot[contact] < 0:
        return
    geometry = compact.prepare_contact_geometry(data, contact)
    normal = compact.evaluate_contact_row(data, geometry, 0)
    publish_row(data, solve, bias, geometry, 0, normal)
    tangent1 = compact.evaluate_contact_row(data, geometry, 1)
    publish_row(data, solve, bias, geometry, 1, tangent1)
    block.gram[contact, 0] = response_product(tangent1, normal)
    tangent2 = compact.evaluate_contact_row(data, geometry, 2)
    publish_row(data, solve, bias, geometry, 2, tangent2)
    block.gram[contact, 1] = response_product(tangent2, normal)
    block.gram[contact, 2] = response_product(tangent2, tangent1)
    block.gram[contact, 3] = response_product(tangent2, tangent2)
