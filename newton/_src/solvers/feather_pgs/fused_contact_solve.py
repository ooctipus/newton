# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental bounded contact preparation, schedule and solve ownership.

The public row capacity stays 704. Worlds outside the private 384-row and
32-dense-contact panel use the complete existing E2 path on the same call.
"""

import re
from functools import cache

import warp as wp

from . import compact_contact as compact

SHARED_ROWS = 384
DENSE_CONTACTS = 32
GLOBAL_ROWS = 704


@wp.struct
class SolveData:
    world_constraint_count: wp.array[int]
    world_dof_indices: wp.array2d[int]
    rhs_bias: wp.array2d[float]
    world_diag: wp.array2d[float]
    world_impulses: wp.array2d[float]
    world_row_type: wp.array2d[int]
    world_row_parent: wp.array2d[int]
    world_row_mu: wp.array2d[float]
    dense_phase_bounds: wp.array2d[int]
    sparse_contact_group_count: wp.array[int]
    sparse_contact_group_heads: wp.array2d[int]
    sparse_contact_serial_count: wp.array[int]
    sparse_contact_serial_normals: wp.array2d[int]
    dense_offsets: wp.array[int]
    dense_groups: wp.array[int]
    dense_J: wp.array3d[float]
    dense_Y: wp.array3d[float]
    sparse_row_dof: wp.array3d[int]
    sparse_row_jy: wp.array3d[float]
    fused_limit_active: wp.array2d[int]
    fused_limit_lower_rhs: wp.array2d[float]
    fused_limit_upper_rhs: wp.array2d[float]
    diagonal_inverse_mass: wp.array[float]
    fused_limit_cfm: float
    iterations: int
    omega: float
    friction_start_iteration: int
    iteration_offset: int
    fused_limit_lower_lambda: wp.array2d[float]
    fused_limit_upper_lambda: wp.array2d[float]
    v_out: wp.array[float]


@wp.struct
class BiasData:
    dt: float
    contact_speculative_scale: float
    joint_limit_speculative_scale: float
    restitution_threshold: float
    incident: wp.array[float]


@wp.kernel
def route_contacts(data: compact.ContactBoundaryData, row_contact: wp.array2d[int]):
    """Invert only successful current contact reservations without sorting them."""
    contact = wp.tid()
    if contact >= wp.min(data.count[0], data.slot.shape[0]):
        return
    slot = data.slot[contact]
    if data.path[contact] == 0 and slot >= 0:
        world = data.world[contact]
        for component in range(data.slots_needed[contact]):
            if slot + component < row_contact.shape[1]:
                row_contact[world, slot + component] = contact


@wp.kernel
def classify_worlds(
    data: compact.ContactBoundaryData,
    counts: wp.array[int],
    bounds: wp.array2d[int],
    row_contact: wp.array2d[int],
    owner: wp.array[int],
    fallback_counts: wp.array[int],
    fallback_bounds: wp.array2d[int],
):
    """Publish one current owner using exact row and dense-panel bounds."""
    world = wp.tid()
    count = counts[world]
    prefix = bounds[world, 1]
    valid = count >= prefix and prefix >= 0 and prefix <= 12 and count <= SHARED_ROWS
    valid = valid and (count - prefix) % 3 == 0
    dense_count = int(0)
    if valid:
        for row in range(prefix, count, 3):
            contact = row_contact[world, row]
            if contact < 0 or contact >= wp.min(data.count[0], data.slot.shape[0]):
                valid = False
            else:
                if data.world[contact] != world or data.slot[contact] != row:
                    valid = False
                if data.path[contact] != 0 or data.slots_needed[contact] != 3:
                    valid = False
                dense = False
                a = data.art0[contact]
                b = data.art1[contact]
                if a >= 0:
                    dense = data.response_dofs[a] == 6
                if b >= 0:
                    dense = dense or data.response_dofs[b] == 6
                if dense:
                    dense_count += 1
        valid = valid and dense_count <= DENSE_CONTACTS
    owner[world] = int(valid)
    fallback_counts[world] = wp.min(wp.max(count, 0), GLOBAL_ROWS)
    fallback_bounds[world, 0] = bounds[world, 0]
    fallback_bounds[world, 1] = prefix
    if valid:
        fallback_counts[world] = 0
        fallback_bounds[world, 0] = 0
        fallback_bounds[world, 1] = 0


@wp.kernel
def produce_fallback(data: compact.ContactBoundaryData, owner: wp.array[int]):
    """Retain the original E2 contact law only for current fallback worlds."""
    contact = wp.tid()
    if contact < wp.min(data.count[0], data.slot.shape[0]):
        if data.path[contact] == 0 and data.slot[contact] >= 0:
            if owner[data.world[contact]] == 0:
                compact.produce_contact(data, contact)


@wp.kernel
def mark_fallback(data: compact.ContactBoundaryData, owner: wp.array[int]):
    """Preserve the original independent-candidate sentinel without touching owned rows."""
    contact = wp.tid()
    if contact >= wp.min(data.count[0], data.slot.shape[0]):
        return
    slot = data.slot[contact]
    if data.path[contact] != 0 or slot < 0 or data.slots_needed[contact] != 3:
        return
    world = data.world[contact]
    if owner[world] != 0:
        return
    other = False
    a = data.art0[contact]
    b = data.art1[contact]
    if a >= 0:
        n = data.response_dofs[a]
        other = n > 0 and n != data.sparse_size
    if b >= 0:
        n = data.response_dofs[b]
        other = other or (n > 0 and n != data.sparse_size)
    if not other:
        d0 = data.sparse_dof[world, slot, 0]
        d1 = data.sparse_dof[world, slot, 1]
        if d0 >= 0 and d1 < 0:
            data.sparse_dof[world, slot, 1] = -2
        elif d1 >= 0 and d0 < 0:
            data.sparse_dof[world, slot, 0] = -2


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
""")
def block_barrier():
    """Order cooperative publication before native shared staging."""


def supported(solver) -> bool:
    """Admit only the inherited complete cold sparse-six position solve."""
    return bool(
        solver._compact_contact_boundary
        and compact.supported(solver)
        and solver.dense_max_constraints == GLOBAL_ROWS
        and solver.max_world_dofs == 114
        and solver._sparse_diagonal_response_size == 108
        and solver.pgs_mode == "matrix_free"
        and solver.pgs_schedule == "interleaved"
        and solver.pgs_velocity_iterations == 0
        and solver.pgs_iterations > 0
        and solver.friction_mode == "current"
        and not solver.pgs_warmstart
        and not solver._mf_warmstart_enabled
        and not solver._regularization_enabled
        and solver._contact_w == 1.0
        and not solver._has_free_rigid_bodies
        and not solver.model.requires_grad
    )


def _native(kernel):
    """Retrieve the single native function from the established solver factory."""
    candidates = [cell.cell_contents for cell in kernel.func.__closure__ or ()]
    candidates = [value for value in candidates if getattr(value, "native_snippet", None)]
    if len(candidates) != 1:
        raise RuntimeError("The original sparse native factory changed its closure")
    return candidates[0]


def _field_names(source):
    """Rebind the original native ABI to one value-owned Warp descriptor."""
    for name in sorted(SolveData.vars, key=len, reverse=True):
        source = re.sub(rf"\b{re.escape(name)}\b", f"solve.{name}", source)
    return source


def _access(source, name, transform):
    """Replace balanced array accesses while preserving their original index arithmetic."""
    needle = name + ".data["
    output = ""
    while needle in source:
        start = source.index(needle)
        end = start + len(needle)
        depth = 1
        cursor = end
        while depth:
            depth += (source[cursor] == "[") - (source[cursor] == "]")
            cursor += 1
        output += source[:start] + transform(source[end : cursor - 1])
        source = source[cursor:]
    return output + source


def sources(device_arch):
    """Reuse the original schedule and GS law, replacing only owned storage access."""
    from .solver_feather_pgs import (  # noqa: PLC0415
        _get_build_independent_sparse_contact_groups_kernel,
        _get_pgs_solve_sparse_diagonal_kernel,
    )

    original = _native(
        _get_pgs_solve_sparse_diagonal_kernel(
            704, 114, 6, device_arch, contact_triples=True, speculative_contact_batches=True
        )
    ).native_snippet
    schedule = _native(
        _get_build_independent_sparse_contact_groups_kernel(704, 114, device_arch, build_serial_contacts=True)
    ).native_snippet
    # Use the exact original schedule ABI names before shared-array substitution.
    for old, new in {
        "group_heads": "sparse_contact_group_heads",
        "group_count": "sparse_contact_group_count",
        "serial_count_out": "sparse_contact_serial_count",
        "serial_normals": "sparse_contact_serial_normals",
    }.items():
        schedule = re.sub(rf"\b{old}\b", new, schedule)
    fused = original
    begin = fused.index("    for (int row = lane; row < row_count; row += 32) {")
    end = fused.index("    for (int coord = lane; coord < 114; coord += 32) {", begin)
    fused = fused[:begin] + fused[end:]
    fused = re.sub(r"    __shared__ [^;]+;\n", "", fused)
    transformations = {
        "sparse_row_dof": lambda i: f"s_dof[({i}) - world * 704 * 2]",
        "sparse_row_jy": lambda i: f"s_jy[({i}) - world * 704 * 4]",
        "sparse_contact_group_heads": lambda i: f"s_heads[({i}) - world * 114]",
        "sparse_contact_group_count": lambda i: "s_group_count",
        "sparse_contact_serial_count": lambda i: "s_serial_count",
        "sparse_contact_serial_normals": lambda i: f"s_serial[({i}) - world * 235]",
        "dense_J": lambda i: f"s_J[s_panel[(({i}) / 6) - dense_group * 704] * 6 + (({i}) % 6)]",
        "dense_Y": lambda i: f"s_Y[s_panel[(({i}) / 6) - dense_group * 704] * 6 + (({i}) % 6)]",
    }
    for name, transform in transformations.items():
        fused = _access(fused, name, transform)
        schedule = _access(schedule, name, transform)
    for text in (fused, schedule):
        if "__syncthreads" in text:
            raise RuntimeError("Original warp-only law gained a CTA barrier")
    schedule = schedule.replace("#if defined(__CUDA_ARCH__)", "").replace("#endif", "")
    fused = fused.replace("#if defined(__CUDA_ARCH__)", "").replace("#endif", "")
    return original, _field_names(schedule), _field_names(fused)


@cache
def get_schedule_kernel(device_arch):
    """Keep the original fallback schedule entirely outside admitted worlds."""
    from .solver_feather_pgs import _get_build_independent_sparse_contact_groups_kernel  # noqa: PLC0415

    snippet = _native(
        _get_build_independent_sparse_contact_groups_kernel(704, 114, device_arch, build_serial_contacts=True)
    ).native_snippet
    for old, new in {
        "group_heads": "sparse_contact_group_heads",
        "group_count": "sparse_contact_group_count",
        "serial_count_out": "sparse_contact_serial_count",
        "serial_normals": "sparse_contact_serial_normals",
    }.items():
        snippet = re.sub(rf"\b{old}\b", new, snippet)

    @wp.func_native(_field_names(snippet))
    def native(world: int, lane: int, solve: SolveData): ...

    @wp.kernel(module="unique", enable_backward=False)
    def schedule(solve: SolveData, owner: wp.array[int]):
        world, lane = wp.tid()
        if owner[world] == 0:
            native(world, lane, solve)

    return schedule


_PREPARE = r"""
#if defined(__CUDA_ARCH__)
    const int thread = threadIdx.x;
    const int count = solve.world_constraint_count.data[world];
    const int prefix = solve.dense_phase_bounds.data[world * 2 + 1];
    const int group = solve.dense_groups.data[world];
    __shared__ float s_v[114], s_lambda[384], s_rhs[384], s_diag[384], s_mu[384];
    __shared__ int s_meta[384], s_dof[768], s_panel[384];
    __shared__ float s_jy[1536], s_J[654], s_Y[654];
    __shared__ int s_heads[114], s_serial[128], s_group_count, s_serial_count, s_panel_count;
    if (thread == 0) s_panel_count = prefix + 1;
    if (thread < 6) { s_J[thread] = 0.0f; s_Y[thread] = 0.0f; }
    for (int row = thread; row < count; row += 128) s_panel[row] = row < prefix ? row + 1 : 0;
    __syncthreads();
    for (int normal = prefix + thread * 3; normal < count; normal += 384) {
        const int contact = row_contact.data[world * 704 + normal];
        const int a = data.art0.data[contact], b = data.art1.data[contact];
        const bool dense = (a >= 0 && data.response_dofs.data[a] == 6)
            || (b >= 0 && data.response_dofs.data[b] == 6);
        if (dense) {
            const int base = atomicAdd(&s_panel_count, 3);
            for (int c = 0; c < 3; ++c) s_panel[normal + c] = base + c;
        }
    }
    __syncthreads();
    for (int row = thread; row < count; row += 128) {
        const int index = world * 704 + row;
        const int kind = data.row_type.data[index];
        const int parent = data.row_parent.data[index];
        s_lambda[row] = 0.0f;
        s_diag[row] = data.diag.data[index];
        s_meta[row] = (kind & 7) | ((parent + 1) << 3);
        s_mu[row] = data.row_mu.data[index];
        for (int c = 0; c < 2; ++c) s_dof[row * 2 + c] = data.sparse_dof.data[index * 2 + c];
        for (int c = 0; c < 4; ++c) s_jy[row * 4 + c] = data.sparse_jy.data[index * 4 + c];
        if (s_panel[row] > 0) for (int c = 0; c < 6; ++c) {
            s_J[s_panel[row] * 6 + c] = data.J.data[(group * 704 + row) * 6 + c];
            s_Y[s_panel[row] * 6 + c] = data.Y.data[(group * 704 + row) * 6 + c];
        }
        const float phi = data.phi.data[index], target = data.target.data[index];
        float rhs = -target;
        const float inv_dt = 1.0f / bias.dt;
        if (kind == 0) rhs += (phi <= 0.0f ? data.row_beta.data[index]
            : bias.contact_speculative_scale) * phi * inv_dt;
        else if (kind == 3) rhs += (phi < 0.0f ? data.row_beta.data[index]
            : bias.joint_limit_speculative_scale) * phi * inv_dt;
        if (kind == 0 && data.restitution.data[index] > 0.0f) {
            float incident = 0.0f;
            for (int c = 0; c < 6; ++c) {
                const int coord = solve.dense_offsets.data[world] + c;
                const int global = solve.world_dof_indices.data[world * 114 + coord];
                if (global >= 0) incident += data.J.data[(group * 704 + row) * 6 + c] * bias.incident.data[global];
            }
            for (int c = 0; c < 2; ++c) {
                const int coord = s_dof[row * 2 + c];
                if (coord >= 0) {
                    const int global = solve.world_dof_indices.data[world * 114 + coord];
                    if (global >= 0) incident += s_jy[row * 4 + c * 2] * bias.incident.data[global];
                }
            }
            incident -= target;
            if (incident < -bias.restitution_threshold
                && (phi <= 1.0e-6f || phi + bias.dt * incident <= 1.0e-6f))
                rhs = -target + data.restitution.data[index] * incident;
        }
        s_rhs[row] = rhs;
        solve.rhs_bias.data[index] = rhs;
        if (row >= prefix && (row - prefix) % 3 == 0) {
            const int contact = row_contact.data[index];
            const int a = data.art0.data[contact], b = data.art1.data[contact];
            const bool other = (a >= 0 && data.response_dofs.data[a] > 0 && data.response_dofs.data[a] != 108)
                || (b >= 0 && data.response_dofs.data[b] > 0 && data.response_dofs.data[b] != 108);
            if (!other) {
                const int d0 = s_dof[row * 2], d1 = s_dof[row * 2 + 1];
                if (d0 >= 0 && d1 < 0) s_dof[row * 2 + 1] = -2;
                else if (d1 >= 0 && d0 < 0) s_dof[row * 2] = -2;
            }
        }
    }
    __syncthreads();
    // Geometry is complete for every row before only warp0 enters the original law.
    if (thread >= 32) return;
"""

_PUBLISH = r"""
    for (int row = thread; row < count; row += 32) {
        const int index = world * 704 + row;
        data.sparse_dof.data[index * 2] = s_dof[row * 2];
        data.sparse_dof.data[index * 2 + 1] = s_dof[row * 2 + 1];
    }
    for (int i = thread; i < 114; i += 32)
        solve.sparse_contact_group_heads.data[world * 114 + i] = s_heads[i];
    for (int i = thread; i < s_serial_count; i += 32)
        solve.sparse_contact_serial_normals.data[world * 235 + i] = s_serial[i];
    if (thread == 0) {
        solve.sparse_contact_group_count.data[world] = s_group_count;
        solve.sparse_contact_serial_count.data[world] = s_serial_count;
    }
#endif
"""


@cache
def get_kernels(device_arch):
    """Compile the original fallback and the storage-remapped fused owner separately."""
    original, schedule, sweep = sources(device_arch)

    @wp.func_native(_field_names(original))
    def fallback_native(world: int, solve: SolveData): ...

    @wp.kernel(module="unique", enable_backward=False)
    def fallback(solve: SolveData, owner: wp.array[int]):
        world, _lane = wp.tid()
        if owner[world] == 0:
            fallback_native(world, solve)

    @wp.func_native(_PREPARE + "{ const int lane = thread;\n" + schedule + "}\n{\n" + sweep + "}\n" + _PUBLISH)
    def fused_native(
        world: int,
        data: compact.ContactBoundaryData,
        solve: SolveData,
        bias: BiasData,
        row_contact: wp.array2d[int],
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def fused(
        data: compact.ContactBoundaryData,
        solve: SolveData,
        bias: BiasData,
        row_contact: wp.array2d[int],
        owner: wp.array[int],
    ):
        world, lane = wp.tid()
        if owner[world] != 0:
            start = solve.dense_phase_bounds[world, 1]
            end = solve.world_constraint_count[world]
            group = data.dense_group[world]
            for coefficient in range(start * 6 + lane, end * 6, 128):
                row = coefficient // 6
                dof = coefficient % 6
                data.J[group, row, dof] = 0.0
                data.Y[group, row, dof] = 0.0
            block_barrier()
            for row in range(start + lane * 3, end, 384):
                compact.produce_contact(data, row_contact[world, row])
            block_barrier()
            fused_native(world, data, solve, bias, row_contact)

    return fused, fallback


class FusedContactSolve:
    """Own per-call routing and captured input bindings; never cache physical geometry."""

    def __init__(self, solver):
        device = solver.model.device
        self.owner = wp.zeros(solver.world_count, dtype=int, device=device)
        self.row_contact = wp.empty((solver.world_count, GLOBAL_ROWS), dtype=int, device=device)
        self.fallback_counts = wp.zeros(solver.world_count, dtype=int, device=device)
        self.fallback_bounds = wp.zeros((solver.world_count, 2), dtype=int, device=device)
        self.fused_kernel, self.fallback_kernel = get_kernels(str(device.arch))
        self.schedule_kernel = get_schedule_kernel(str(device.arch))
        self.data = None
        self.bias = None

    def prepare(self, solver, state_in, state_aug, contacts, dt):
        """Bind original current inputs and publish complete current fallback ownership."""
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
        wp.launch(
            classify_worlds,
            dim=solver.world_count,
            inputs=[
                data,
                solver.slot_counter,
                solver.dense_phase_bounds,
                self.row_contact,
                self.owner,
                self.fallback_counts,
                self.fallback_bounds,
            ],
            device=device,
        )
        wp.launch(
            compact.clear_contact_response,
            dim=(solver.world_count, 256),
            inputs=[
                self.fallback_counts,
                self.fallback_bounds,
                data.dense_group,
                data.J,
                data.Y,
            ],
            block_dim=256,
            device=device,
        )
        wp.launch(produce_fallback, dim=contacts.rigid_contact_max, inputs=[data, self.owner], device=device)
        wp.launch(mark_fallback, dim=contacts.rigid_contact_max, inputs=[data, self.owner], device=device)

    def schedule(self, solver):
        """Build only fallback schedules after the original final count publication."""
        solve = SolveData()
        solve.world_constraint_count = solver.constraint_count
        solve.dense_phase_bounds = solver.dense_phase_bounds
        solve.sparse_row_dof = solver._sparse_diagonal_row_dof
        solve.sparse_contact_group_count = solver._sparse_contact_group_count
        solve.sparse_contact_group_heads = solver._sparse_contact_group_heads
        solve.sparse_contact_serial_count = solver._sparse_contact_serial_count
        solve.sparse_contact_serial_normals = solver._sparse_contact_serial_normals
        wp.launch_tiled(
            self.schedule_kernel,
            dim=[solver.world_count],
            inputs=[solve, self.owner],
            block_dim=32,
            device=solver.model.device,
        )

    def launch_solve(self, solver, dense_rhs, iterations, omega, friction_start_iteration, iteration_offset):
        """Run the complete owner and exact masked original fallback, charging both launches."""
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
        wp.launch_tiled(
            self.fused_kernel,
            dim=[solver.world_count],
            inputs=[
                self.data,
                solve,
                self.bias,
                self.row_contact,
                self.owner,
            ],
            block_dim=128,
            device=solver.model.device,
        )
        wp.launch_tiled(
            self.fallback_kernel,
            dim=[solver.world_count],
            inputs=[solve, self.owner],
            block_dim=32,
            device=solver.model.device,
        )
