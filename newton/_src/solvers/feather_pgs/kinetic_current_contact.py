# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Default-off current raw geometry and complete ZERO ownership.

The raw pass replaces CSR, not collision output. Geometry is rebuilt on every
current substep. Only admitted dense contacts receive tangent/material staging;
held response construction remains after ZERO and authoritative allocation.
"""

import functools
from types import SimpleNamespace

import warp as wp

from . import (
    allocation_count_source,
    contact_allocation_source,
    contact_geometry_source,
    contact_normal,
    kernels,
    kinetic_allocate,
    kinetic_rows,
    kinetic_rows_triplet,
    kinetic_zero,
)
from . import kuka_joint_world as light
from . import simple_world as simple
from .kinetic_rows_types import ArmMap, DenseRowOutput, PrefixInput, RawRowInput, RowSettings, RowState
from .kinetic_source import checked_definitions
from .kinetic_types import CurrentKineticCache, HeldKineticOperator, KineticPlan


@wp.struct
class ContactGeometry:
    bodies: wp.array2d[int]
    values: wp.array2d[float]
    ready: wp.array[int]
    rejected: wp.array[int]
    prefix_count: wp.array[int]


def allocate(worlds, capacity, device):
    """One solver-owned scratch bank, overwritten before every consumption."""
    value = ContactGeometry()
    value.bodies = wp.empty((capacity, 2), dtype=int, device=device)
    value.values = wp.empty((capacity, 20), dtype=float, device=device)
    value.ready = wp.empty(worlds, dtype=int, device=device)
    value.rejected = wp.empty(worlds, dtype=int, device=device)
    value.prefix_count = wp.empty(worlds, dtype=int, device=device)
    return value


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31,stride=32;
#else
    const int lane=0,stride=1;
#endif
    int count=0;
    for(int code=lane;code<46;code+=stride) {
        const int dof=plan.dof_ids.data[world*35+code/2], qi=prefix.q_index.data[dof];
        if(qi>=0) {
            const float q=prefix.q.data[qi],bound=(code&1)?prefix.upper.data[dof]:prefix.lower.data[dof];
            const bool active=isfinite(bound)&&((code&1)?q>=bound-settings.activation_gap:q<=bound+settings.activation_gap);
            if(active)++count;
        }
    }
#if defined(__CUDA_ARCH__)
    for(int offset=16;offset>0;offset/=2)count+=__shfl_down_sync(0xffffffffu,count,offset);
#endif
    if(lane==0) {
        const int ready=reinterpret_cast<float*>(address)[512]!=0.0f?1:0;
        cache.ready.data[world]=ready;
        cache.rejected.data[world]=ready?0:1;
        cache.prefix_count.data[world]=count;
        if(world==0){state.raw_invalid.data[0]=0;state.active_count.data[0]=0;}
    }
""")
def _prepare_output(
    address: wp.uint64,
    world: int,
    plan: KineticPlan,
    prefix: PrefixInput,
    settings: RowSettings,
    state: RowState,
    cache: ContactGeometry,
): ...


@functools.cache
def get_prepare_kernel(arch):
    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_contact_prepare(
        plan: KineticPlan,
        zero_plan: light.JointWorldPlan,
        data: simple._SimpleWorldInput,
        current: CurrentKineticCache,
        state: RowState,
        prefix: PrefixInput,
        settings: RowSettings,
        cache: ContactGeometry,
    ):
        world, logical = wp.tid()
        lanes = light._lane()
        if lanes[1] == 1 and logical != 0:
            return
        address = light._storage()
        kinetic_zero._load(address, world, zero_plan, data, current, state)
        light._limits(address, world, data)
        _prepare_output(address, world, plan, prefix, settings, state, cache)
        light._release(address)

    return kinetic_contact_prepare


@wp.func
def _endpoint(shape: int, raw: RawRowInput, worlds: int):
    body, art, world = int(-1), int(-1), int(-1)
    if shape < -1 or shape >= raw.shape_body.shape[0]:
        return wp.vec3i(-1, -1, -2)
    if shape >= 0:
        body = raw.shape_body[shape]
        if body < -1 or body >= raw.body_to_articulation.shape[0] or body >= raw.body_q.shape[0]:
            return wp.vec3i(-1, -1, -2)
        if body >= 0:
            art = raw.body_to_articulation[body]
            if art < -1 or art >= raw.art_to_world.shape[0]:
                return wp.vec3i(-1, -1, -2)
            if art >= 0:
                world = raw.art_to_world[art]
                if world < 0 or world >= worlds:
                    return wp.vec3i(-1, -1, -2)
    return wp.vec3i(body, art, world)


@wp.func
def _geometry(c: int, raw: RawRowInput, data: simple._SimpleWorldInput, cache: ContactGeometry):
    """Retain BOTH original rounding laws using the same rotated local points."""
    ba, bb = cache.bodies[c, 0], cache.bodies[c, 1]
    aa, ab = raw.art_a[c], raw.art_b[c]
    n = -raw.normal[c]
    ta, tb = wp.vec3(0.0), wp.vec3(0.0)
    ra = wp.vec3(raw.point0[c])
    rb = wp.vec3(raw.point1[c])
    if ba >= 0:
        pose = raw.body_q[ba]
        ta = wp.transform_get_translation(pose)
        ra = wp.quat_rotate(wp.transform_get_rotation(pose), ra)
    if bb >= 0:
        pose = raw.body_q[bb]
        tb = wp.transform_get_translation(pose)
        rb = wp.quat_rotate(wp.transform_get_rotation(pose), rb)
    xa, xb = ra - raw.margin0[c] * n, rb + raw.margin1[c] * n
    if ba >= 0:
        xa = (ra + ta) - raw.margin0[c] * n
    if bb >= 0:
        xb = (rb + tb) + raw.margin1[c] * n
    oa, ob = ra - raw.margin0[c] * n, rb + raw.margin1[c] * n
    separation = (ta - tb) + (oa - ob)
    la, lb = oa, ob
    if aa >= 0:
        la += ta - data.articulation_origin[aa]
    if ab >= 0:
        lb += tb - data.articulation_origin[ab]
    if data.shared_anchor != 0:
        la -= 0.5 * separation
        lb += 0.5 * separation
    for k in range(3):
        cache.values[c, k] = n[k]
        cache.values[c, 3 + k] = xa[k]
        cache.values[c, 6 + k] = xb[k]
    cache.values[c, 15] = wp.dot(n, xa - xb)
    return separation, la, lb


@functools.cache
def _normal():
    """Original numerical checker; substitute only current scratch ownership."""
    checked_definitions(
        "simple_world.py",
        (
            "_SimpleWorldInput",
            "_SimpleRawContacts",
            "_normal_functions",
            "_check_normal",
            "_endpoint_speed",
            "_finite3",
            "_finite6",
            "_passes",
        ),
    )
    namespace = dict(simple.__dict__)
    namespace.update(globals())
    return contact_normal.bind("current", namespace)[1]


@functools.cache
def get_raw_kernel(arch):
    normal = _normal()

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_contact_geometry(
        threads: int,
        raw: RawRowInput,
        data: simple._SimpleWorldInput,
        state: RowState,
        cache: ContactGeometry,
    ):
        tid = wp.tid()
        count = raw.count[0]
        if count < 0 or count > cache.bodies.shape[0]:
            if tid == 0:
                wp.atomic_max(state.raw_invalid, 0, 1)
            return
        for c in range(tid, count, threads):
            a = _endpoint(raw.shape0[c], raw, cache.ready.shape[0])
            b = _endpoint(raw.shape1[c], raw, cache.ready.shape[0])
            world = wp.max(a[2], b[2])
            if a[2] < -1 or b[2] < -1 or (a[2] >= 0 and b[2] >= 0 and a[2] != b[2]) or world < 0:
                wp.atomic_max(state.raw_invalid, 0, 1)
                raw.world[c] = -1
                continue
            cache.bodies[c, 0] = a[0]
            cache.bodies[c, 1] = b[0]
            raw.art_a[c] = a[1]
            raw.art_b[c] = b[1]
            raw.world[c] = world
            separation, lever_a, lever_b = _geometry(c, raw, data, cache)
            if cache.ready[world] != 0:
                good = normal(c, raw, data, cache, state, separation, lever_a, lever_b)
                if not good:
                    wp.atomic_max(cache.rejected, world, 1)

    return kinetic_contact_geometry


@functools.cache
def get_finalize_kernel(arch):
    namespace = dict(kinetic_allocate.__dict__)
    namespace.update(globals())
    return allocation_count_source.bind(namespace, cached_geometry=True)


@functools.cache
def get_allocate_kernel(arch):
    """Bind original routing and monotone reservations to current cached geometry."""
    checked_definitions(
        "kernels.py", ("_contact_allocator", "_allocate_world_contact_slot", "allocate_world_contact_slots")
    )
    namespace = dict(kernels.__dict__)
    namespace.update(globals())
    return contact_allocation_source.bind(namespace, cached_geometry=True)[1]


def _stage_source():
    return contact_geometry_source.get_stage_source()


@wp.func_native(_stage_source())
def _stage_row(c: int, count: int, cache: ContactGeometry, raw: RawRowInput, settings: RowSettings): ...


def _contact_source():
    return (
        contact_geometry_source.get_prelude(kinetic_rows._COMMON, cached_geometry=True) + kinetic_rows_triplet._TRIPLET
    )


@wp.func_native(_contact_source())
def _contact(
    address: wp.uint64,
    worker: int,
    plan: KineticPlan,
    current: CurrentKineticCache,
    held: HeldKineticOperator,
    raw: RawRowInput,
    state: RowState,
    settings: RowSettings,
    arm: ArmMap,
    out: DenseRowOutput,
    cache: ContactGeometry,
): ...


@functools.cache
def get_contact_kernel(arch):
    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_contact_cached_triplet(
        plan: KineticPlan,
        current: CurrentKineticCache,
        held: HeldKineticOperator,
        raw: RawRowInput,
        state: RowState,
        settings: RowSettings,
        arm: ArmMap,
        out: DenseRowOutput,
        cache: ContactGeometry,
    ):
        worker, logical = wp.tid()
        lanes = kinetic_rows._lane()
        if lanes[1] == 1 and logical != 0:
            return
        address = kinetic_rows_triplet._storage()
        _contact(address, worker, plan, current, held, raw, state, settings, arm, out, cache)
        kinetic_rows._release(address)

    return kinetic_contact_cached_triplet


def install(binding, call):
    """Replace actual closure owners; original fallbacks remain separate."""
    cache, rows, zero, solve = binding.current_contact, call.rows, call.zero, call.solve
    allocation = solve.allocation
    arch, device, worlds = str(binding.device.arch), binding.device, binding.worlds
    prepare = get_prepare_kernel(arch)
    raw_kernel, finalize, allocate_kernel = get_raw_kernel(arch), get_finalize_kernel(arch), get_allocate_kernel(arch)
    prepare_args = [
        rows.plan,
        zero.arguments[0],
        zero.data,
        rows.current,
        rows.state,
        rows.prefix,
        rows.settings,
        cache,
    ]
    raw_args = [allocation.workers, rows.raw, zero.data, rows.state, cache]
    finalize_args = [rows.plan, rows.prefix, rows.raw, rows.state, rows.settings, allocation.data, cache]
    allocate_args = [*allocation.contact_args, cache, rows.raw, rows.settings]
    if rows.kernels[2] is not kinetic_rows_triplet.get_contact_kernel(arch) or len(rows.arguments[2]) != 8:
        raise ValueError("Current-contact requires the unchanged actual triplet owner")
    rows.kernels[2] = get_contact_kernel(arch)
    rows.arguments[2].append(cache)

    def zero_launch():
        wp.launch_tiled(prepare, dim=[worlds], inputs=prepare_args, block_dim=32, device=device)
        wp.launch(raw_kernel, dim=allocation.workers, inputs=raw_args, device=device)
        wp.launch(finalize, dim=worlds, inputs=finalize_args, device=device)

    def allocate_launch():
        wp.launch(allocate_kernel, dim=allocation.workers, inputs=allocate_args, device=device)
        wp.copy(allocation.mf.contact_end, allocation.data.mf_counter)
        wp.launch(
            kernels.allocate_rigid_velocity_limit_slots,
            dim=allocation.mf.free_bodies.size,
            inputs=allocation.limit_args,
            device=device,
        )
        for counter, cap, family, first_rejected, count in (
            (allocation.data.dense_counter, 192, 0, allocation.data.first_rejected_dense, rows.state.dense_count),
            (allocation.data.mf_counter, 64, 1, allocation.data.first_rejected_mf, rows.state.mf_count),
            (
                allocation.data.propagation_counter,
                192,
                2,
                allocation.data.first_rejected_propagation,
                allocation.data.propagation_count,
            ),
        ):
            wp.launch(
                kernels.finalize_constraint_counts_with_status,
                dim=worlds,
                inputs=[counter, cap, family, first_rejected, count, rows.state.capacity_status],
                device=device,
            )

    call.current_contact = SimpleNamespace(
        cache=cache,
        kernels=(prepare, raw_kernel, finalize, allocate_kernel, rows.kernels[2]),
        arguments=(prepare_args, raw_args, finalize_args, allocate_args, rows.arguments[2]),
    )
    zero.launch = zero_launch
    solve.allocate = allocation.launch = allocate_launch
