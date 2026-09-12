# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental private scalar islands and bounded coupled contact ownership."""

import re
from functools import cache

import numpy as np
import warp as wp

from . import compact_contact as compact
from .fused_contact_solve import (
    BiasData,
    FusedContactSolve,
    SolveData,
    _native,
    get_kernels,
    get_schedule_kernel,
    mark_fallback,
    produce_fallback,
    route_contacts,
)

GLOBAL_ROWS = 704
KEYS = 108
CONTACTS_PER_KEY = 4
RESIDUAL_ROWS = 160
RESIDUAL_CONTACTS = RESIDUAL_ROWS // 3
RESIDUAL_KEYS = 16


@wp.struct
class IslandRouting:
    owner: wp.array[int]
    fallback_counts: wp.array[int]
    fallback_bounds: wp.array2d[int]
    row_contact: wp.array2d[int]
    key_counts: wp.array2d[int]
    key_contact_ids: wp.array3d[int]
    key_reserved: wp.array2d[int]
    key_coordinates: wp.array2d[int]
    residual_row_count: wp.array[int]
    residual_row_ids: wp.array2d[int]
    residual_contact_ids: wp.array2d[int]
    residual_key_count: wp.array[int]
    residual_key_coordinates: wp.array2d[int]


def allocate_routing(worlds: int, key_coordinates: np.ndarray, device) -> IslandRouting:
    """Allocate stable captured metadata; no private canonical coefficient array."""
    coordinates = np.asarray(key_coordinates, dtype=np.int32)
    if coordinates.shape != (worlds, KEYS):
        raise ValueError("Private islands require exactly 108 key coordinates per world")
    if not np.all(coordinates == coordinates[:, :1] + np.arange(KEYS)):
        raise ValueError("Private islands require contiguous key response coordinates")
    if not np.all(np.isin(coordinates[:, 0], (0, 6))):
        raise ValueError("Private islands require the complete 108+6 coordinate partition")
    routing = IslandRouting()
    routing.owner = wp.zeros(worlds, dtype=int, device=device)
    routing.fallback_counts = wp.zeros(worlds, dtype=int, device=device)
    routing.fallback_bounds = wp.zeros((worlds, 2), dtype=int, device=device)
    routing.row_contact = wp.empty((worlds, GLOBAL_ROWS), dtype=int, device=device)
    routing.key_counts = wp.zeros((worlds, KEYS), dtype=int, device=device)
    routing.key_contact_ids = wp.empty((worlds, KEYS, CONTACTS_PER_KEY), dtype=int, device=device)
    routing.key_reserved = wp.zeros((worlds, KEYS), dtype=int, device=device)
    routing.key_coordinates = wp.array(coordinates, dtype=int, device=device)
    routing.residual_row_count = wp.zeros(worlds, dtype=int, device=device)
    routing.residual_row_ids = wp.empty((worlds, RESIDUAL_ROWS), dtype=int, device=device)
    routing.residual_contact_ids = wp.empty((worlds, RESIDUAL_CONTACTS), dtype=int, device=device)
    routing.residual_key_count = wp.zeros(worlds, dtype=int, device=device)
    routing.residual_key_coordinates = wp.empty((worlds, RESIDUAL_KEYS), dtype=int, device=device)
    return routing


_PARTITION = r"""
#if defined(__CUDA_ARCH__)
    constexpr int WORKERS = 32;
    #define ISLAND_SHARED __shared__
    #define ISLAND_SYNC() __syncwarp(0xffffffffu)
    #define ISLAND_ADD(pointer) atomicAdd(pointer, 1)
    #define ISLAND_INVALID() atomicExch(&s_valid, 0)
#else
    if (lane != 0) return;
    constexpr int WORKERS = 1;
    #define ISLAND_SHARED
    #define ISLAND_SYNC() ((void)0)
    #define ISLAND_ADD(pointer) ((*(pointer))++)
    #define ISLAND_INVALID() (s_valid = 0)
#endif
    ISLAND_SHARED int s_reserved[108], s_counts[108];
    ISLAND_SHARED int s_contact[235], s_coordinate[235], s_candidate[235];
    ISLAND_SHARED int s_residual_rows[160], s_residual_contacts[53];
    ISLAND_SHARED int s_valid, s_residual_count, s_reserved_count;
    const int count = counts.data[world];
    const int prefix = bounds.data[world * 2 + 1];
    const int raw_count = data.count.data[0];
    const int key_offset = routing.key_coordinates.data[world * 108];
    if (lane == 0) {
        routing.owner.data[world] = 0;
        routing.fallback_counts.data[world] = count < 0 ? 0 : (count > 704 ? 704 : count);
        routing.fallback_bounds.data[world * 2] = bounds.data[world * 2];
        routing.fallback_bounds.data[world * 2 + 1] = prefix;
        routing.residual_row_count.data[world] = 0;
        routing.residual_key_count.data[world] = 0;
        s_valid = count >= prefix && count <= 704 && prefix >= 0 && prefix <= 12
            && (count - prefix) % 3 == 0 && raw_count >= 0
            && raw_count <= data.slot.shape[0] && raw_count <= data.point0.shape[0];
    }
    for (int key = lane; key < 108; key += WORKERS) {
        s_reserved[key] = 0;
        s_counts[key] = 0;
        routing.key_counts.data[world * 108 + key] = 0;
        routing.key_reserved.data[world * 108 + key] = 0;
        for (int i = 0; i < 4; ++i)
            routing.key_contact_ids.data[(world * 108 + key) * 4 + i] = -1;
    }
    ISLAND_SYNC();
    if (!s_valid) return;
    const int contacts = (count - prefix) / 3;
    for (int i = lane; i < contacts; i += WORKERS) {
        const int row = prefix + i * 3;
        const int contact = routing.row_contact.data[world * 704 + row];
        s_contact[i] = contact;
        s_coordinate[i] = -1;
        s_candidate[i] = 0;
        if (contact < 0 || contact >= raw_count) { ISLAND_INVALID(); continue; }
        if (data.world.data[contact] != world || data.slot.data[contact] != row
            || data.path.data[contact] != 0 || data.slots_needed.data[contact] != 3) {
            ISLAND_INVALID(); continue;
        }
        int coordinates[2] = {-1, -1};
        bool other = false;
        for (int endpoint = 0; endpoint < 2; ++endpoint) {
            const int art = endpoint == 0 ? data.art0.data[contact] : data.art1.data[contact];
            const int shape = endpoint == 0 ? data.shape0.data[contact] : data.shape1.data[contact];
            if (art < -1 || art >= data.response_dofs.shape[0]
                || shape < -1 || shape >= data.shape_body.shape[0]) {
                ISLAND_INVALID(); continue;
            }
            const int body = shape >= 0 ? data.shape_body.data[shape] : -1;
            if (body < -1 || body >= data.body_single_dof.shape[0]) {
                ISLAND_INVALID(); continue;
            }
            const int response = art >= 0 ? data.response_dofs.data[art] : 0;
            if (response != 0 && response != 6 && response != 108) ISLAND_INVALID();
            other = other || (response > 0 && response != 108);
            if (art >= 0 && body >= 0 && response == 108) {
                const int dof = data.body_single_dof.data[body];
                if (dof >= 0) {
                    const int local = dof - data.dof_start.data[art];
                    if (local >= 0 && local < 108)
                        coordinates[endpoint] = data.dof_offset.data[art] + local;
                }
            }
        }
        if (coordinates[0] >= 0 && coordinates[0] == coordinates[1]) coordinates[1] = -1;
        const bool scalar = !other && ((coordinates[0] >= 0) != (coordinates[1] >= 0));
        const int coordinate = coordinates[0] >= 0 ? coordinates[0] : coordinates[1];
        s_candidate[i] = scalar ? 1 : 0;
        s_coordinate[i] = coordinate;
        for (int endpoint = 0; endpoint < 2; ++endpoint) {
            if (coordinates[endpoint] >= 0) {
                const int key = coordinates[endpoint] - key_offset;
                if (key < 0 || key >= 108) { ISLAND_INVALID(); continue; }
                if (!scalar) {
#if defined(__CUDA_ARCH__)
                    atomicExch(&s_reserved[key], 1);
#else
                    s_reserved[key] = 1;
#endif
                }
            }
        }
    }
    ISLAND_SYNC();
    if (!s_valid) return;
    for (int i = lane; i < contacts; i += WORKERS) {
        const int key = s_coordinate[i] - key_offset;
        const bool independent = s_candidate[i] != 0 && key >= 0 && !s_reserved[key];
        s_candidate[i] = independent ? 1 : 0;
        if (independent) {
            const int index = ISLAND_ADD(&s_counts[key]);
            if (index < 4) routing.key_contact_ids.data[(world * 108 + key) * 4 + index] = s_contact[i];
            else ISLAND_INVALID();
        }
    }
    ISLAND_SYNC();
    // Only bounded integer compaction is serial; raw topology reads are warp-parallel.
    if (lane == 0) {
        s_reserved_count = 0;
        for (int key = 0; key < 108; ++key) if (s_reserved[key]) {
            if (s_reserved_count < 16)
                routing.residual_key_coordinates.data[world * 16 + s_reserved_count] = key_offset + key;
            ++s_reserved_count;
        }
        if (s_reserved_count > 16) s_valid = 0;
        int row_count = prefix;
        for (int row = 0; row < prefix; ++row) s_residual_rows[row] = row;
        int residual_contacts = 0;
        for (int i = 0; i < contacts; ++i) if (!s_candidate[i]) {
            if (residual_contacts < 53) s_residual_contacts[residual_contacts] = s_contact[i];
            ++residual_contacts;
            for (int component = 0; component < 3; ++component) {
                if (row_count < 160) s_residual_rows[row_count] = prefix + i * 3 + component;
                ++row_count;
            }
        }
        s_residual_count = row_count;
        if (row_count > 160) s_valid = 0;
    }
    ISLAND_SYNC();
    for (int key = lane; key < 108; key += WORKERS) {
        const int index = world * 108 + key;
        routing.key_counts.data[index] = s_counts[key];
        routing.key_reserved.data[index] = s_reserved[key];
        if (s_counts[key] <= 4) {
            // Canonical slot order is deterministic even though bucket reservations are atomic.
            for (int i = 1; i < s_counts[key]; ++i) {
                const int contact = routing.key_contact_ids.data[index * 4 + i];
                int j = i;
                while (j > 0 && data.slot.data[routing.key_contact_ids.data[index * 4 + j - 1]]
                    > data.slot.data[contact]) {
                    routing.key_contact_ids.data[index * 4 + j] = routing.key_contact_ids.data[index * 4 + j - 1];
                    --j;
                }
                routing.key_contact_ids.data[index * 4 + j] = contact;
            }
        }
    }
    ISLAND_SYNC();
    if (!s_valid) return;
    for (int row = lane; row < s_residual_count; row += WORKERS)
        routing.residual_row_ids.data[world * 160 + row] = s_residual_rows[row];
    for (int i = lane; i < (s_residual_count - prefix) / 3; i += WORKERS)
        routing.residual_contact_ids.data[world * 53 + i] = s_residual_contacts[i];
    ISLAND_SYNC();
    if (lane == 0) {
        routing.residual_row_count.data[world] = s_residual_count;
        routing.residual_key_count.data[world] = s_reserved_count;
        routing.fallback_counts.data[world] = 0;
        routing.fallback_bounds.data[world * 2] = 0;
        routing.fallback_bounds.data[world * 2 + 1] = 0;
        routing.owner.data[world] = 1;
    }
#undef ISLAND_SHARED
#undef ISLAND_SYNC
#undef ISLAND_ADD
#undef ISLAND_INVALID
"""


@cache
def get_partition_kernel(device_arch: str):
    """Use one warp/world; the CPU branch executes the same bounded logic on lane zero."""
    del device_arch

    @wp.func_native(_PARTITION)
    def native(
        world: int,
        lane: int,
        data: compact.ContactBoundaryData,
        counts: wp.array[int],
        bounds: wp.array2d[int],
        routing: IslandRouting,
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def partition(
        data: compact.ContactBoundaryData, counts: wp.array[int], bounds: wp.array2d[int], routing: IslandRouting
    ):
        world, lane = wp.tid()
        native(world, lane, data, counts, bounds, routing)

    return partition


@wp.func
def contact_rhs(solve: SolveData, bias: BiasData, world: int, packet: compact.ContactRow):
    """Use original bias and predictor-frozen restitution for either private owner."""
    rhs = -packet.target
    inv_dt = 1.0 / bias.dt
    if packet.kind == 0:
        scale = bias.contact_speculative_scale
        if packet.phi <= 0.0:
            scale = packet.beta
        rhs += scale * packet.phi * inv_dt
    elif packet.kind == 3:
        scale = bias.joint_limit_speculative_scale
        if packet.phi < 0.0:
            scale = packet.beta
        rhs += scale * packet.phi * inv_dt
    if packet.kind == 0 and packet.restitution > 0.0:
        incident = float(0.0)
        for dof in range(6):
            coordinate = solve.dense_offsets[world] + dof
            global_dof = solve.world_dof_indices[world, coordinate]
            if global_dof >= 0:
                incident += packet.J[dof] * bias.incident[global_dof]
        if packet.dof0 >= 0:
            global_dof = solve.world_dof_indices[world, packet.dof0]
            if global_dof >= 0:
                incident += packet.j0 * bias.incident[global_dof]
        if packet.dof1 >= 0:
            global_dof = solve.world_dof_indices[world, packet.dof1]
            if global_dof >= 0:
                incident += packet.j1 * bias.incident[global_dof]
        incident -= packet.target
        if incident < -bias.restitution_threshold:
            if packet.phi <= 1.0e-6 or packet.phi + bias.dt * incident <= 1.0e-6:
                rhs = -packet.target + packet.restitution * incident
    return rhs


def scalar_source(device_arch: str) -> str:
    """Reuse the exact original independent-contact law with private scalar operands."""
    from .solver_feather_pgs import _get_pgs_solve_sparse_diagonal_kernel  # noqa: PLC0415

    original = _native(
        _get_pgs_solve_sparse_diagonal_kernel(704, 114, 6, device_arch, contact_triples=True)
    ).native_snippet
    begin = original.index("                const int normal_slot =")
    end = original.index("                normal = next_normal;\n            }\n        }", begin)
    body = original[begin:end]
    body, count = re.subn(
        r"const int normal_slot =.*?const float normal_denom = s_diag\[normal\];",
        "const float normal_j = values[0];\nconst float normal_y = values[3];\nconst float normal_denom = values[6];",
        body,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError("Original independent normal coefficient source changed")
    body, count = re.subn(
        r"const int tangent1_sparse_row =.*?const float tangent2_y = sparse_row_jy.data\[tangent2_jy \+ 1\];",
        "const float tangent1_j = values[1];\n"
        "const float tangent1_y = values[4];\n"
        "const float tangent2_j = values[2];\n"
        "const float tangent2_y = values[5];",
        body,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError("Original independent tangent coefficient source changed")
    for row, component in (("normal", 0), ("tangent1", 1), ("tangent2", 2)):
        for name, offset in (("s_rhs", 9), ("s_diag", 6), ("s_lambda", 13)):
            body = body.replace(f"{name}[{row}]", f"values[{offset + component}]")
        body = body.replace(f"s_mu[{row}]", "values[12]")
    body = body.replace("s_v[scalar_coord]", "velocity").replace("normal = next_normal;", "")
    if any(token in body for token in ("sparse_row", "s_lambda[", "s_v[", "next_normal", "s_diag[", "s_rhs[")):
        raise RuntimeError("Unmapped original scalar-contact operand")
    return (
        r"""
#if !defined(__CUDA_ARCH__)
    #define __fmul_rn(a, b) ((a) * (b))
    #define __fadd_rn(a, b) ((a) + (b))
    #define fmaxf(a, b) wp::max(a, b)
    #define sqrtf(a) wp::sqrt(a)
#endif
    const int key_index = world * 108 + key;
    const int physical = routing.key_coordinates.data[key_index];
    const int limit_index = world * 114 + physical;
    const int global_dof = solve.world_dof_indices.data[limit_index];
    const int count = routing.key_counts.data[key_index];
    const int active = solve.fused_limit_active.data[limit_index];
    float lower_lambda = 0.0f, upper_lambda = 0.0f;
    // This is a current output owner even when last step's active key is empty now.
    solve.fused_limit_lower_lambda.data[limit_index] = 0.0f;
    solve.fused_limit_upper_lambda.data[limit_index] = 0.0f;
    if (count == 0 && active == 0) return;
    float velocity = global_dof >= 0 ? solve.v_out.data[global_dof] : 0.0f;
    const float response = global_dof >= 0 ? solve.diagonal_inverse_mass.data[global_dof] : 0.0f;
    const float limit_denom = response + solve.fused_limit_cfm;
    const float lower_rhs = solve.fused_limit_lower_rhs.data[limit_index];
    const float upper_rhs = solve.fused_limit_upper_rhs.data[limit_index];
    const float omega = solve.omega;
    const int friction_start_iteration = solve.friction_start_iteration;
    float* packet = reinterpret_cast<float*>(address);
    for (int iteration = 0; iteration < solve.iterations; ++iteration) {
        const int global_iteration = solve.iteration_offset + iteration;
        int iteration_changed = 0;
        if (response > 0.0f) {
            if ((active & 1) != 0) {
                const float residual = velocity + lower_rhs;
                const float old_lambda = lower_lambda;
                lower_lambda = fmaxf(0.0f, old_lambda - omega * residual / limit_denom);
                const float delta = lower_lambda - old_lambda;
                velocity += response * delta;
                if (delta != 0.0f) iteration_changed = 1;
            }
            if ((active & 2) != 0) {
                const float residual = -velocity + upper_rhs;
                const float old_lambda = upper_lambda;
                upper_lambda = fmaxf(0.0f, old_lambda - omega * residual / limit_denom);
                const float delta = upper_lambda - old_lambda;
                velocity -= response * delta;
                if (delta != 0.0f) iteration_changed = 1;
            }
        }
        for (int contact = 0; contact < count; ++contact) {
            float* values = packet + contact * 16;
"""
        + body
        + r"""
        }
        if (global_iteration >= friction_start_iteration && iteration_changed == 0) break;
    }
    for (int contact = 0; contact < count; ++contact) {
        const int source = routing.key_contact_ids.data[key_index * 4 + contact];
        const int row = data.slot.data[source];
        for (int component = 0; component < 3; ++component)
            solve.world_impulses.data[world * 704 + row + component] = packet[contact * 16 + 13 + component];
    }
    if (global_dof >= 0) solve.v_out.data[global_dof] = velocity;
    solve.fused_limit_lower_lambda.data[limit_index] = lower_lambda;
    solve.fused_limit_upper_lambda.data[limit_index] = upper_lambda;
#if !defined(__CUDA_ARCH__)
    #undef __fmul_rn
    #undef __fadd_rn
    #undef fmaxf
    #undef sqrtf
#endif
"""
    )


_SCALAR_STORAGE = r"""
#if defined(__CUDA_ARCH__)
    // A 65-float lane stride separates equal-field accesses across all 32 banks.
    __shared__ float packet[32 * 65];
    return reinterpret_cast<uint64_t>(packet + threadIdx.x * 65);
#else
    // ARM CPU JIT does not support ELF TLS; this test-only allocation is released below.
    return reinterpret_cast<uint64_t>(malloc(64 * sizeof(float)));
#endif
"""

_SCALAR_RELEASE = r"""
#if !defined(__CUDA_ARCH__)
    free(reinterpret_cast<void*>(address));
#endif
"""

_SCALAR_STORE = r"""
    float* values = reinterpret_cast<float*>(address) + contact * 16;
    const bool first = packet.dof0 >= 0;
    values[component] = first ? packet.j0 : packet.j1;
    values[3 + component] = first ? packet.y0 : packet.y1;
    values[6 + component] = packet.diag;
    values[9 + component] = rhs;
    values[12] = packet.mu;
    values[13 + component] = 0.0f;
"""


@cache
def get_scalar_kernel(device_arch: str):
    """Assign one lane to each isolated key, retaining at most four private triples."""

    @wp.func_native(_SCALAR_STORAGE)
    def storage() -> wp.uint64: ...

    @wp.func_native(_SCALAR_STORE)
    def store_row(address: wp.uint64, contact: int, component: int, packet: compact.ContactRow, rhs: float): ...

    @wp.func_native(_SCALAR_RELEASE)
    def release(address: wp.uint64): ...

    @wp.func_native(scalar_source(device_arch))
    def solve_native(
        address: wp.uint64,
        world: int,
        key: int,
        data: compact.ContactBoundaryData,
        solve: SolveData,
        routing: IslandRouting,
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def scalar(data: compact.ContactBoundaryData, solve: SolveData, bias: BiasData, routing: IslandRouting):
        world, key = wp.tid()
        if routing.owner[world] != 0 and routing.key_reserved[world, key] == 0:
            count = routing.key_counts[world, key]
            address = wp.uint64(0)
            if count > 0:
                address = storage()
                for contact in range(count):
                    source = routing.key_contact_ids[world, key, contact]
                    geometry = compact.prepare_contact_geometry(data, source)
                    # Partition excluded every dense endpoint before any private publication.
                    geometry.has_dense = False
                    for component in range(3):
                        packet = compact.evaluate_contact_row(data, geometry, component)
                        rhs = contact_rhs(solve, bias, world, packet)
                        compact.publish_row_metadata(data, world, geometry.slot + component, packet)
                        solve.rhs_bias[world, geometry.slot + component] = rhs
                        store_row(address, contact, component, packet, rhs)
            solve_native(address, world, key, data, solve, routing)
            release(address)

    return scalar


class PrivateContactIslands(FusedContactSolve):
    """Replace complete admitted contact ownership; retain the original whole-world fallback."""

    def __init__(self, solver):
        from .compact_island_residual import get_kernel  # noqa: PLC0415 - avoid descriptor import cycle

        device = solver.model.device
        offsets = solver._sparse_diagonal_dense_offsets.numpy()
        key_offsets = np.where(offsets == 0, 6, 0).astype(np.int32)
        if offsets.shape != (solver.world_count,) or not np.all(np.isin(offsets, (0, 108))):
            raise ValueError("Private contact islands require the complete contiguous 108+6 response")
        coordinates = key_offsets[:, None] + np.arange(KEYS, dtype=np.int32)
        self.routing = allocate_routing(solver.world_count, coordinates, device)
        self.owner = self.routing.owner
        self.fallback_counts = self.routing.fallback_counts
        self.fallback_bounds = self.routing.fallback_bounds
        self.row_contact = self.routing.row_contact
        architecture = str(device.arch)
        self.partition_kernel = get_partition_kernel(architecture)
        self.scalar_kernel = get_scalar_kernel(architecture)
        self.residual_kernel = get_kernel(architecture)
        self.fallback_kernel = get_kernels(architecture)[1]
        self.schedule_kernel = get_schedule_kernel(architecture)
        self.data = None
        self.bias = None

    def prepare(self, solver, state_in, state_aug, contacts, dt):
        """Publish current complete partition before any accepted coefficient or solve write."""
        data = compact.bind_contact_data(solver, state_in, state_aug, contacts)
        self.data = data
        bias = BiasData()
        bias.dt = dt
        bias.contact_speculative_scale = solver.contact_speculative_scale
        bias.joint_limit_speculative_scale = 1.0
        bias.restitution_threshold = solver._effective_restitution_velocity_threshold
        bias.incident = solver.v_hat
        self.bias = bias
        device = solver.model.device
        wp.launch(route_contacts, dim=contacts.rigid_contact_max, inputs=[data, self.row_contact], device=device)
        wp.launch_tiled(
            self.partition_kernel,
            dim=[solver.world_count],
            inputs=[data, solver.slot_counter, solver.dense_phase_bounds, self.routing],
            block_dim=32,
            device=device,
        )
        wp.launch(
            compact.clear_contact_response,
            dim=(solver.world_count, 256),
            inputs=[self.fallback_counts, self.fallback_bounds, data.dense_group, data.J, data.Y],
            block_dim=256,
            device=device,
        )
        wp.launch(produce_fallback, dim=contacts.rigid_contact_max, inputs=[data, self.owner], device=device)
        wp.launch(mark_fallback, dim=contacts.rigid_contact_max, inputs=[data, self.owner], device=device)

    def launch_solve(self, solver, dense_rhs, iterations, omega, friction_start_iteration, iteration_offset):
        """Charge both disjoint private owners and the complete original fallback launch."""
        solve = SolveData()
        fields = {
            "world_constraint_count": "constraint_count",
            "world_dof_indices": "world_dof_indices",
            "world_diag": "diag",
            "world_impulses": "impulses",
            "world_row_type": "row_type",
            "world_row_parent": "row_parent",
            "world_row_mu": "row_mu",
            "dense_phase_bounds": "dense_phase_bounds",
            "sparse_contact_group_count": "_sparse_contact_group_count",
            "sparse_contact_group_heads": "_sparse_contact_group_heads",
            "sparse_contact_serial_count": "_sparse_contact_serial_count",
            "sparse_contact_serial_normals": "_sparse_contact_serial_normals",
            "dense_offsets": "_sparse_diagonal_dense_offsets",
            "dense_groups": "_sparse_diagonal_dense_groups",
            "sparse_row_dof": "_sparse_diagonal_row_dof",
            "sparse_row_jy": "_sparse_diagonal_row_jy",
            "fused_limit_active": "_fused_diagonal_limit_active",
            "fused_limit_lower_rhs": "_fused_diagonal_limit_lower_rhs",
            "fused_limit_upper_rhs": "_fused_diagonal_limit_upper_rhs",
            "diagonal_inverse_mass": "_diagonal_inverse_mass",
            "fused_limit_lower_lambda": "_fused_diagonal_limit_lower_lambda",
            "fused_limit_upper_lambda": "_fused_diagonal_limit_upper_lambda",
            "v_out": "v_out",
        }
        for target, source in fields.items():
            setattr(solve, target, getattr(solver, source))
        solve.rhs_bias = dense_rhs
        solve.dense_J = solver.J_by_size[6]
        solve.dense_Y = solver.Y_by_size[6]
        solve.fused_limit_cfm = solver.pgs_cfm
        solve.iterations = iterations
        solve.omega = omega
        solve.friction_start_iteration = friction_start_iteration
        solve.iteration_offset = iteration_offset
        device = solver.model.device
        wp.launch(
            self.scalar_kernel,
            dim=(solver.world_count, KEYS),
            inputs=[self.data, solve, self.bias, self.routing],
            block_dim=32,
            device=device,
        )
        wp.launch_tiled(
            self.residual_kernel,
            dim=[solver.world_count],
            inputs=[self.data, solve, self.bias, self.routing],
            block_dim=32,
            device=device,
        )
        wp.launch_tiled(
            self.fallback_kernel,
            dim=[solver.world_count],
            inputs=[solve, self.owner],
            block_dim=32,
            device=device,
        )
