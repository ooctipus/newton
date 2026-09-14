# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Current allocation/count adapter; original contact and free-limit kernels.

Only scalar-prefix COUNT is new; original46 candidate law/order is retained and
the row owner later emits current Z exactly once. No saved routes/counts, grouped
J, added raw queue or collision screen is consumed by this adapter.
"""

import inspect
import json
import re
from types import SimpleNamespace

import numpy as np
import warp as wp

from . import kernels as original
from . import solver_feather_pgs as solver
from .kinetic_rows_types import PrefixInput, RawRowInput, RowSettings, RowState
from .kinetic_source import checked_definitions
from .kinetic_types import KineticPlan


@wp.struct
class AllocationData:
    dense_counter: wp.array[int]
    mf_counter: wp.array[int]
    propagation_counter: wp.array[int]
    propagation_count: wp.array[int]
    dense_contact_flag: wp.array[int]
    dropped_dense: wp.array[int]
    dropped_mf: wp.array[int]
    dropped_propagation: wp.array[int]
    prefix_count: wp.array[int]


@wp.kernel
def initialize_count(
    plan: KineticPlan,
    prefix: PrefixInput,
    raw: RawRowInput,
    state: RowState,
    settings: RowSettings,
    allocation: AllocationData,
):
    """Charge current scalar-limit scan and initialize every allocation output."""
    world = wp.tid()
    count = int(0)
    if state.resolved[world] == 0:
        for code in range(46):
            dof = plan.dof_ids[world, code // 2]
            qi = prefix.q_index[dof]
            if qi >= 0:
                q = prefix.q[qi]
                bound = prefix.lower[dof]
                active = wp.isfinite(bound) and q <= bound + settings.activation_gap
                if (code & 1) != 0:
                    bound = prefix.upper[dof]
                    active = wp.isfinite(bound) and q >= bound - settings.activation_gap
                if active:
                    count += 1
    allocation.dense_counter[world] = count
    allocation.prefix_count[world] = count
    allocation.mf_counter[world] = 0
    allocation.propagation_counter[world] = 0
    allocation.propagation_count[world] = 0
    allocation.dense_contact_flag[world] = 0
    allocation.dropped_dense[world] = 0
    allocation.dropped_mf[world] = 0
    allocation.dropped_propagation[world] = 0
    state.dense_count[world] = 0
    state.mf_count[world] = 0
    if world == 0:
        for family in range(4):
            state.capacity_status[family] = 0
        if raw.count[0] < 0 or raw.count[0] > raw.shape0.shape[0]:
            state.raw_invalid[0] = 1
        if state.raw_invalid[0] != 0:
            state.capacity_status[3] = 1


def bind_allocation(rows, mf):
    """Bind fixed current services, preserving authoritative active/resolved inputs."""
    checked_definitions(
        "kernels.py",
        (
            "_allocate_world_contact_slot",
            "allocate_world_contact_slots",
            "allocate_rigid_velocity_limit_slots",
            "finalize_constraint_counts_with_status",
        ),
    )
    # Captures omit model.shape_transform. It is currently an unused formal in
    # the exact pinned allocator; fail if that reader ever becomes real.
    text = inspect.getsource(original._allocate_world_contact_slot.func)
    if len(re.findall(r"\bshape_transform\b", text)) != 1:
        raise ValueError("Need current model shape transforms for changed allocator")
    a = rows.bundle["host"]["snapshot"]
    meta = json.loads(a["metadata_json"].item())
    if not (
        meta["enable_joint_limits"]
        and not meta["enable_joint_velocity_limits"]
        and meta["_mimic_count"] == 0
        and meta["_connect_count"] == 0
        and meta["drive_mode"] == "augmented"
        and not meta["_preelim_active"]
    ):
        raise ValueError("Current count owner only covers admitted original prefix")
    s, raw, state = rows.settings, rows.raw, rows.state
    if (s.dense_capacity, s.mf_capacity) != (192, 64):
        raise ValueError("Changed row capacities")
    if raw.shape0.shape != (s.raw_capacity,) or not 0 < s.raw_capacity <= 4_000_000:
        raise ValueError("Changed raw allocation capacity")
    if rows.metadata["full_capacity"] and s.raw_capacity != 4_000_000:
        raise ValueError("Full-capacity mode requires exact decimal 4000000")
    device, worlds = rows.device, rows.worlds
    allocation = AllocationData()
    for name in AllocationData.vars:
        setattr(allocation, name, wp.zeros(worlds, dtype=int, device=device))
    body_has_response = wp.array(a["solver_body_has_response_dofs"], dtype=int, device=device)
    root_dof_start = wp.array(a["post3_solver_articulation_root_dof_start"], dtype=int, device=device)
    shape_transform_unused = wp.zeros(1, dtype=wp.transform, device=device)
    workers = min(s.raw_capacity, solver._CONTACT_BUILD_THREAD_CAP * solver._CONTACT_THREADS_X)
    values = {
        "contact_count": raw.count,
        "total_num_threads": workers,
        "contact_shape0": raw.shape0,
        "contact_shape1": raw.shape1,
        "contact_point0": raw.point0,
        "contact_point1": raw.point1,
        "contact_normal": raw.normal,
        "contact_thickness0": raw.margin0,
        "contact_thickness1": raw.margin1,
        "body_q": raw.body_q,
        "shape_transform": shape_transform_unused,
        "shape_body": raw.shape_body,
        "body_to_articulation": raw.body_to_articulation,
        "art_to_world": raw.art_to_world,
        "articulation_response_dof_count": raw.response_count,
        "body_flags": rows.bundle["inputs"].body_flags,
        "body_has_response_dofs": body_has_response,
        "is_free_rigid": raw.is_free_rigid,
        "has_free_rigid": int(mf.free_bodies.size > 0),
        "propagation_articulated_contacts": 0,
        "propagation_same_articulation": 0,
        "propagation_free_free": 0,
        "contact_gap_gate": meta["contact_gap_gate"],
        "same_articulation_contact_gap_gate": meta["same_articulation_contact_gap_gate"],
        "articulation_pair_contact_gap_gate": meta["articulation_pair_contact_gap_gate"],
        "max_constraints": 192,
        "mf_max_constraints": 64,
        "propagation_max_constraints": 192,
        "enable_friction": s.enable_friction,
        "contact_friction_gap_threshold": s.friction_gap_threshold,
        "contact_friction_anchor_limit": s.friction_anchor_limit,
        "contact_friction_articulation_pairs_only": s.friction_articulation_pairs_only,
        "row_capacity_telemetry": 0,
        "resolved_worlds": state.resolved,
        "contact_world": raw.world,
        "contact_slot": raw.slot,
        "contact_art_a": raw.art_a,
        "contact_art_b": raw.art_b,
        "world_slot_counter": allocation.dense_counter,
        "contact_path": raw.path,
        "mf_slot_counter": allocation.mf_counter,
        "propagation_slot_counter": allocation.propagation_counter,
        "dense_contact_world_flag": allocation.dense_contact_flag,
        "contact_slots_needed": raw.slots_needed,
        "dense_dropped_contact_rows": allocation.dropped_dense,
        "mf_dropped_contact_rows": allocation.dropped_mf,
        "propagation_dropped_contact_rows": allocation.dropped_propagation,
        "capacity_status": state.capacity_status,
    }
    parameters = inspect.signature(original.allocate_world_contact_slots.func).parameters
    if set(parameters) != set(values):
        raise ValueError("Original contact allocator signature changed")
    contact_args = [values[name] for name in parameters]
    limit_args = [
        mf.free_bodies,
        raw.body_to_articulation,
        raw.art_to_world,
        raw.is_free_rigid,
        rows.bundle["inputs"].body_flags,
        mf.max_linear,
        mf.max_angular,
        root_dof_start,
        state.v_hat,
        float(meta["velocity_limit_activation_fraction"]),
        64,
        state.resolved,
        mf.limit_slot,
        mf.limit_sign,
        allocation.mf_counter,
    ]

    def launch():
        """Count, route, snapshot contact extent, allocate free limits, finalize3."""
        wp.launch(
            initialize_count,
            dim=worlds,
            inputs=[rows.plan, rows.prefix, raw, state, s, allocation],
            device=device,
        )
        wp.launch(
            original.allocate_world_contact_slots,
            dim=workers,
            inputs=contact_args,
            device=device,
        )
        wp.copy(mf.contact_end, allocation.mf_counter)
        wp.launch(
            original.allocate_rigid_velocity_limit_slots,
            dim=mf.free_bodies.size,
            inputs=limit_args,
            device=device,
        )
        for counter, cap, family, count in (
            (allocation.dense_counter, 192, 0, state.dense_count),
            (allocation.mf_counter, 64, 1, state.mf_count),
            (allocation.propagation_counter, 192, 2, allocation.propagation_count),
        ):
            wp.launch(
                original.finalize_constraint_counts_with_status,
                dim=worlds,
                inputs=[counter, cap, family, count, state.capacity_status],
                device=device,
            )

    def check():
        """Untimed no-drop admission; never promote a clamped or unreadable frame."""
        if np.any(state.capacity_status.numpy()) or np.any(state.raw_invalid.numpy()):
            raise ValueError("Original allocation capacity/raw status rejects this frame")
        if np.any(allocation.propagation_count.numpy()):
            raise ValueError("Unbound propagation family")
        return {
            "dense_rows": int(state.dense_count.numpy().sum()),
            "mf_rows": int(state.mf_count.numpy().sum()),
            "prefix_rows": int(allocation.prefix_count.numpy().sum()),
            "recorded_routes_consumed": False,
            "raw_capacity": s.raw_capacity,
            "full_capacity": bool(rows.metadata["full_capacity"]),
        }

    return SimpleNamespace(
        data=allocation,
        launch=launch,
        check=check,
        contact_args=contact_args,
        limit_args=limit_args,
        workers=workers,
        rows=rows,
        mf=mf,
        source_pin="checked-function-AST",
        metadata={
            "new_count_scan_charged": True,
            "discarded_metadata_not_materialized": True,
            "original_allocator_unchanged": True,
            "propagation_enabled": False,
            "impulses_cleared_by_row_and_mf_owners": True,
        },
    )
