# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Accepted original ZERO admission over produced predictor/current geometry.

Retains original raw CSR and original light limit/normal law. No forcing, held
factor action, endpoint contraction, generalized integration or row work occurs.
"""

import functools
import hashlib
import linecache
from types import SimpleNamespace

import numpy as np
import warp as wp

from . import kuka_joint_world as light
from . import raw_world_contacts as raw_world
from . import simple_world as simple
from .kinetic_rows_types import RowState
from .kinetic_source import checked_definitions
from .kinetic_types import CurrentKineticCache


def load_source():
    """Retain original body/free-speed validation, delete only twist production."""
    checked_definitions(
        "kuka_joint_world.py",
        (
            "JointWorldPlan",
            "_TWISTS",
            "_LIMITS",
            "_STORAGE",
            "_limits",
            "_ready",
            "_storage",
            "_release",
            "_lane",
            "_body_twist",
            "_body_valid",
            "normal_source",
            "operation_source",
        ),
    )
    tail = light._TWISTS[light._TWISTS.index("    for(int b=lane;b<32;b+=stride) {") :]
    prescribed = "for(int k=0;k<6;++k) s[96+b*6+k]=data.body_v_s.data[body][k];"
    free = """            if(b>=30) {
                const int first=data.articulation_dof_start.data[art];
                for(int k=0;k<6;++k) {
                    float value=0.0f;
                    for(int d=0;d<6;++d) value+=data.joint_S_s.data[first+d][k]*data.v_hat.data[first+d];
                    s[96+b*6+k]=value;
                }
            }
"""
    if tail.count(prescribed) != 1 or tail.count(free) != 1:
        raise ValueError("Original twist producer/validation seams changed")
    tail = tail.replace(prescribed, "/* prescribed twist already produced */").replace(free, "")
    prefix = r"""
    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31,stride=32;
#else
    const int lane=0,stride=1;
#endif
    if(lane==0) s[513]=(state.predictor_status.data[world]!=0||current.valid.data[world]==0
        ||current.generation.data[world]!=state.state_generation.data[world])?1.0f:0.0f;
    for(int b=lane;b<32;b+=stride) {
        const int body=plan.body_ids.data[world*32+b];
        for(int k=0;k<6;++k) s[96+b*6+k]=state.endpoint_twists.data[body][k];
    }
"""
    return prefix + tail


@wp.func_native(load_source())
def _load(
    address: wp.uint64,
    world: int,
    plan: light.JointWorldPlan,
    data: simple._SimpleWorldInput,
    current: CurrentKineticCache,
    state: RowState,
): ...


_PUBLISH = r"""
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31;
    good=__all_sync(0xffffffffu,good);
#else
    const int lane=0;
#endif
    const bool selected=good&&reinterpret_cast<float*>(address)[512]!=0.0f&&invalid==0;
    if(lane==0) {
        state.resolved.data[world]=selected?1:0;
        if(!selected) {
#if defined(__CUDA_ARCH__)
            const int index=atomicAdd(state.active_count.data,1);
#else
            const int index=state.active_count.data[0]++;
#endif
            state.active_worlds.data[index]=world;
        }
    }
"""


@wp.func_native(_PUBLISH)
def _publish(address: wp.uint64, world: int, good: bool, invalid: int, state: RowState): ...


@functools.cache
def get_kernel(arch):
    """One warp/world, same current ZERO numerical law and complete raw CSR."""
    source = light.normal_source()
    filename = f"<kinetic-zero-normal-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = dict(simple.__dict__)
    namespace.update(light.__dict__)
    exec(compile(source, filename, "exec"), namespace)
    check_normal = namespace["_light_check_normal"]

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_zero(
        plan: light.JointWorldPlan,
        data: simple._SimpleWorldInput,
        raw: simple._SimpleRawContacts,
        buckets: raw_world.RawWorldContacts,
        current: CurrentKineticCache,
        state: RowState,
    ):
        world, logical = wp.tid()
        lanes = light._lane()
        lane, stride = lanes[0], lanes[1]
        if stride == 1 and logical != 0:
            return
        address = light._storage()
        _load(address, world, plan, data, current, state)
        light._limits(address, world, data)
        good = bool(True)
        if light._ready(address) and buckets.invalid[0] == 0:
            for index in range(buckets.offsets[world] + lane, buckets.offsets[world + 1], stride):
                good = check_normal(buckets.ids[index], raw, data, address, plan) and good
        _publish(address, world, good, buckets.invalid[0], state)
        light._release(address)

    return kinetic_zero


def bind_zero(rows, mf, *, buckets=None):
    """Bind current outputs; caller decides when current raw CSR must rebuild."""
    checked_definitions(
        "kuka_joint_world.py",
        (
            "JointWorldPlan",
            "_TWISTS",
            "_LIMITS",
            "_STORAGE",
            "_limits",
            "_ready",
            "_storage",
            "_release",
            "_lane",
            "_body_twist",
            "_body_valid",
            "normal_source",
            "operation_source",
        ),
    )
    checked_definitions(
        "simple_world.py",
        (
            "_SimpleWorldInput",
            "_SimpleRawContacts",
            "_check_normal",
            "_endpoint_speed",
            "_finite3",
            "_finite6",
            "_passes",
        ),
    )
    checked_definitions(
        "raw_world_contacts.py",
        (
            "RawWorldContacts",
            "RawWorldContactBuckets",
            "_endpoint_world",
            "_contact_world",
            "count_raw_world_contacts",
            "scatter_raw_world_contacts",
            "validate_raw_world_contacts",
        ),
    )
    a, device, worlds = rows.bundle["host"]["snapshot"], rows.device, rows.worlds
    plan = light.JointWorldPlan()
    for name in light.JointWorldPlan.vars:
        setattr(plan, name, getattr(rows.plan, name))
    arts = rows.raw.body_to_articulation.numpy()[rows.bundle["host"]["plan"]["body_ids"][:, [0, 30, 31]]]
    if not np.array_equal(arts, np.arange(worlds * 3).reshape(worlds, 3)):
        raise ValueError("Private origin alias requires checked actual world-major3 art order")
    data = simple._SimpleWorldInput()
    data.world_dof_indices = mf.world_dofs
    data.world_dof_count = wp.full(worlds, 29, dtype=int, device=device)
    data.limit_q_index = rows.prefix.q_index
    data.lower, data.upper, data.q = rows.prefix.lower, rows.prefix.upper, rows.prefix.q
    data.v_hat = rows.state.v_hat
    for name in (
        "body_to_articulation",
        "art_to_world",
        "prescribed_articulation",
        "is_free_rigid",
    ):
        setattr(data, name, getattr(rows.raw, name))
    data.articulation_dof_start = mf.art_dof_start
    data.articulation_response_dof_count = rows.raw.response_count
    data.body_flags = rows.bundle["inputs"].body_flags
    data.body_has_response_dofs = wp.array(a["solver_body_has_response_dofs"], dtype=int, device=device)
    data.body_response_dof_mask = rows.raw.body_response_mask
    data.body_q = rows.raw.body_q
    # These old canonical producers are deliberately unavailable: the source
    # adapter must never contract axes or read prescribed old twist storage.
    data.body_v_s = wp.empty(0, dtype=wp.spatial_vector, device=device)
    data.joint_S_s = wp.empty(0, dtype=wp.spatial_vector, device=device)
    data.articulation_origin = rows.current.origin.reshape((worlds * 3,))
    data.max_linear_velocity, data.max_angular_velocity = mf.max_linear, mf.max_angular
    data.max_depenetration_velocity = mf.max_depenetration
    s = rows.settings
    data.dt, data.beta, data.speculative_scale = (
        s.dt,
        s.beta,
        s.contact_speculative_scale,
    )
    data.activation_gap, data.restitution_threshold = (
        s.activation_gap,
        s.restitution_velocity_threshold,
    )
    data.shared_anchor = s.shared_anchor
    data.absolute_margin, data.relative_margin = 1e-5, 2.0**-20
    raw = simple._SimpleRawContacts()
    for name in simple._SimpleRawContacts.vars:
        setattr(raw, name, getattr(rows.raw, name))
    if buckets is None:
        buckets = raw_world.RawWorldContactBuckets(worlds, s.raw_capacity, device=device)
    if buckets.world_count != worlds or buckets.contact_capacity != s.raw_capacity or buckets.device != device:
        raise ValueError("Current retained raw CSR capacity/device mismatch")
    rows.state.raw_invalid = buckets.data.invalid
    kernel = get_kernel(str(device.arch))
    arguments = [plan, data, raw, buckets.data, rows.current, rows.state]

    def build_buckets():
        """Original complete current raw incidence; reuse only matching raw generation."""
        buckets.build(
            raw.count,
            raw.shape0,
            raw.shape1,
            raw.shape_body,
            rows.raw.body_to_articulation,
            rows.raw.art_to_world,
        )

    def classify():
        """Use current predictor; publish resolved and active exactly once."""
        rows.state.active_count.zero_()
        wp.launch_tiled(kernel, dim=[worlds], block_dim=32, inputs=arguments, device=device)

    def launch():
        """Conservative complete call charges original current CSR plus ZERO."""
        build_buckets()
        classify()

    return SimpleNamespace(
        plan=plan,
        data=data,
        raw=raw,
        buckets=buckets,
        kernel=kernel,
        arguments=arguments,
        build_buckets=build_buckets,
        classify=classify,
        launch=launch,
        rows=rows,
        mf=mf,
        metadata={
            "original_zero_law": True,
            "original_csr": True,
            "old_factor_tau_twist_producers": False,
            "clamp_added": False,
        },
    )
