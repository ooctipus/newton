# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Explicit complete-cohort prediction/row/solve ownership, default off."""

import math
import os

import numpy as np
import warp as wp

from . import g1_kinetic_owner
from .g1_kinetic_state import ForceInput
from .parallel_world_rows import WorldInput, get_kernel, get_producer_kernel
from .raw_world_contacts import RawWorldContactBuckets, RawWorldContacts, _endpoint_world
from .sparse_packet_rows import PacketInput


@wp.func
def _route(
    c: int,
    shape0: wp.array[int],
    shape1: wp.array[int],
    shape_body: wp.array[int],
    body_art: wp.array[int],
    art_world: wp.array[int],
    worlds: int,
):
    a = _endpoint_world(shape0[c], shape_body, body_art, art_world, worlds)
    b = _endpoint_world(shape1[c], shape_body, body_art, art_world, worlds)
    if a < -1 or b < -1:
        return int(-2)
    if (a >= 0 and b >= 0 and a != b) or (a < 0 and b < 0):
        return int(-1)
    return wp.max(a, b)


@wp.kernel(enable_backward=False)
def _count(
    threads: int,
    raw_count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    shape_body: wp.array[int],
    body_art: wp.array[int],
    art_world: wp.array[int],
    counts: wp.array[int],
    data: RawWorldContacts,
    slot: wp.array[int],
    path: wp.array[int],
    needed: wp.array[int],
):
    tid = wp.tid()
    n = raw_count[0]
    worlds = counts.shape[0] - 1
    if n < 0 or n > data.ids.shape[0]:
        if tid == 0:
            wp.atomic_max(data.invalid, 0, 1)
        for c in range(tid, data.ids.shape[0], threads):
            slot[c] = -1
            path[c] = -1
            needed[c] = 0
        return
    for c in range(tid, n, threads):
        world = _route(c, shape0, shape1, shape_body, body_art, art_world, worlds)
        if world == -2:
            wp.atomic_max(data.invalid, 0, 1)
        elif world == -1:
            wp.atomic_add(counts, worlds, 1)
            slot[c] = -1
            path[c] = -1
            needed[c] = 0
        else:
            wp.atomic_add(counts, world, 1)


@wp.kernel(enable_backward=False)
def _scatter(
    threads: int,
    raw_count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    shape_body: wp.array[int],
    body_art: wp.array[int],
    art_world: wp.array[int],
    counts: wp.array[int],
    data: RawWorldContacts,
):
    if data.invalid[0] != 0:
        return
    worlds = counts.shape[0] - 1
    for c in range(wp.tid(), raw_count[0], threads):
        world = _route(c, shape0, shape1, shape_body, body_art, art_world, worlds)
        if world == -1:
            wp.atomic_add(counts, worlds, 1)
        elif world >= 0:
            at = data.offsets[world] + wp.atomic_add(counts, world, 1)
            if at >= data.offsets[world] and at < data.offsets[world + 1] and at < data.ids.shape[0]:
                data.ids[at] = c
            else:
                wp.atomic_max(data.invalid, 0, 1)


@wp.kernel(enable_backward=False)
def _validate(raw_count: wp.array[int], counts: wp.array[int], data: RawWorldContacts):
    world = wp.tid()
    begin, end = data.offsets[world], data.offsets[world + 1]
    if begin < 0 or end < begin or end > data.ids.shape[0] or end - begin != counts[world]:
        wp.atomic_max(data.invalid, 0, 1)
    if world == 0 and (begin != 0 or data.offsets[counts.shape[0] - 1] + counts[counts.shape[0] - 1] != raw_count[0]):
        wp.atomic_max(data.invalid, 0, 1)


class ParallelWorldBuckets(RawWorldContactBuckets):
    """Keep exact raw IDs, distinguishing intentionally ignored contacts from errors."""

    def build(self, count, shape0, shape1, shape_body, body_to_articulation, art_to_world, *, slot, path, needed):
        if count.shape != (1,) or shape0.shape != (self.contact_capacity,) or shape1.shape != shape0.shape:
            raise ValueError("Raw bucket descriptors must match the exact allocated contact capacity")
        arrays = (count, shape0, shape1, shape_body, body_to_articulation, art_to_world, slot, path, needed)
        if any(a.dtype != wp.int32 or a.ndim != 1 or a.device != self.device for a in arrays):
            raise ValueError("Raw bucket inputs must be same-device one-dimensional int32 arrays")
        if any(a.shape[0] < self.contact_capacity for a in (slot, path, needed)):
            raise ValueError("Canonical raw mappings must cover the complete contact allocation")
        self.counts.zero_()
        self.data.invalid.zero_()
        threads = max(1, min(self.contact_capacity, 65536))
        args = [threads, count, shape0, shape1, shape_body, body_to_articulation, art_to_world, self.counts, self.data]
        wp.launch(_count, dim=threads, inputs=[*args, slot, path, needed], device=self.device)
        wp.utils.array_scan(self.counts, self.data.offsets, inclusive=False)
        self.counts.zero_()
        wp.launch(_scatter, dim=threads, inputs=args, device=self.device)
        wp.launch(_validate, dim=self.world_count, inputs=[count, self.counts, self.data], device=self.device)


def supported(owner, *, split=False):
    """Reject unowned consumers before any original producer is bypassed."""
    s = owner.solver
    return bool(
        g1_kinetic_owner.supported(s)
        and owner.kinetic_state is not None
        and s.dense_max_constraints == 100
        and not owner.body_basis_rows
        and not owner.paired_gs
        and not s._has_prescribed_response
        and (not s._has_root_free or not s._has_rigid_body_velocity_limits or s.rigid_velocity_limit_slot is not None)
        and (
            not split
            or (
                owner.metric_tangents
                and owner.data.Z.shape == (s.world_count, 100, 18)
                and owner.data.incident.shape == (s.world_count, 100)
            )
        )
    )


def create(owner):
    """Return no owner for unsupported configurations without changing old storage."""
    split = os.environ.get("FEATHER_PGS_PARALLEL_WORLD_SPLIT") == "1"
    if not supported(owner, split=split):
        return None
    return ParallelWorld(owner, split=split)


class ParallelWorld:
    """Own only prediction through final velocity; current/final state stays separate."""

    def __init__(self, owner, *, split=False):
        self.sparse, self.solver, self.kinetic = owner, owner.solver, owner.kinetic_state
        self.split = split
        self.kernel = (get_producer_kernel if split else get_kernel)(self.kinetic.chain_scan)
        self.solve_kernel = owner.kernels.solve
        self.status = wp.zeros(2, dtype=int, device=self.solver.model.device)
        self.active = False
        self.buckets = ParallelWorldBuckets(
            self.solver.world_count, self.solver._max_contacts_alloc, device=self.solver.model.device
        )
        self.contacts = None
        self.force = None
        self.input = None
        self.rows = None

    def prepare(self, state_in, state_aug, control, stage3_qd, contacts, dt):
        self.active = False
        if not supported(self.sparse, split=self.split) or state_in.requires_grad or state_aug is not self.solver:
            return False
        if contacts is None or contacts.requires_grad:
            return False
        if contacts.rigid_contact_max > self.solver._max_contacts_alloc:
            return False
        if contacts.rigid_contact_shape0.device != self.solver.model.device:
            return False
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Parallel world requires a positive finite step duration")
        self.kinetic._validate_call(state_in, state_aug, dt)
        if stage3_qd.ptr != state_in.joint_qd.ptr:
            return False
        capacity = contacts.rigid_contact_max
        if self.buckets is None or self.buckets.contact_capacity != capacity:
            if self.solver.model.device.is_cuda and wp.get_stream(self.solver.model.device).is_capturing:
                return False
            self.buckets = ParallelWorldBuckets(self.solver.world_count, capacity, device=self.solver.model.device)
        s, model = self.solver, self.solver.model
        f = ForceInput()
        f.joint_q, f.joint_qd, f.predictor_qd = state_in.joint_q, state_in.joint_qd, stage3_qd
        f.joint_f, f.body_f, f.body_flags = control.joint_f, state_in.body_f, model.body_flags
        f.stiffness, f.reference, f.damping = (
            s._passive_spring_stiffness,
            s._passive_spring_ref,
            s._passive_joint_damping,
        )
        f.u0, f.S, f.qdd, f.vhat, f.dt = state_aug.joint_tau, state_aug.joint_S_s, state_aug.joint_qdd, s.v_hat, dt
        x = PacketInput()
        for target, source in (
            ("shape0", "shape0"),
            ("shape1", "shape1"),
            ("point0", "point0"),
            ("point1", "point1"),
            ("normal", "normal"),
            ("thickness0", "margin0"),
            ("thickness1", "margin1"),
        ):
            setattr(x, target, getattr(contacts, "rigid_contact_" + source))
        x.shape_body, x.body_q, x.screw, x.origin = (
            model.shape_body,
            state_in.body_q,
            state_aug.joint_S_s,
            s.articulation_origin,
        )
        x.art_a, x.art_b, x.cfm, x.phi = s.contact_art_a, s.contact_art_b, s.row_cfm, s.phi
        x.target, x.restitution = s.target_velocity, s.row_restitution
        x.dt, x.threshold = dt, s._effective_restitution_velocity_threshold
        x.shared_anchor, x.friction_anchor = int(s.contact_shared_anchor), int(s.contact_friction_shared_anchor)
        r = WorldInput()
        r.count = contacts.rigid_contact_count
        r.body_art, r.response_dofs, r.body_response, r.body_flags = (
            s.body_to_articulation,
            s.articulation_response_dof_count,
            s.body_has_response_dofs,
            model.body_flags,
        )
        r.prescribed, r.body_v = s._prescribed_articulation, state_aug.body_v_s
        r.material_mu, r.material_restitution = s.shape_material_mu, s.shape_material_restitution
        r.limit_q, r.lower, r.upper = s._joint_limit_q_index, model.joint_limit_lower, model.joint_limit_upper
        for target, source in (
            ("contact_world", "contact_world"),
            ("contact_slot", "contact_slot"),
            ("contact_path", "contact_path"),
            ("contact_needed", "contact_slots_needed"),
            ("counter", "slot_counter"),
            ("counts", "constraint_count"),
            ("phase", "dense_phase_bounds"),
            ("dense_flag", "dense_contact_world_flag"),
            ("capacity_status", "_constraint_capacity_status"),
            ("row_type", "row_type"),
            ("parent", "row_parent"),
            ("mu", "row_mu"),
            ("beta_rows", "row_beta"),
            ("rhs", "rhs"),
            ("diagonal", "diag"),
            ("row_w", "row_w"),
            ("impulses", "impulses"),
        ):
            setattr(r, target, getattr(s, source))
        r.dropped = s._row_dropped_dense if s._row_watermark else s._dummy_mf_slot_counter
        r.status = self.status
        r.limits, r.friction = int(s.enable_joint_limits), int(s.enable_contact_friction)
        r.anchor_limit, r.pairs_only, r.telemetry = (
            s.contact_friction_anchor_limit,
            int(s.contact_friction_articulation_pairs_only),
            int(s._row_watermark),
        )
        r.gap, r.same_gap, r.pair_gap = (
            s.contact_gap_gate,
            s.same_articulation_contact_gap_gate,
            s.articulation_pair_contact_gap_gate,
        )
        r.friction_gap, r.friction_scale, r.limit_gap = (
            s.contact_friction_gap_threshold,
            s.contact_friction_scale,
            s.joint_limit_activation_gap,
        )
        r.beta, r.cfm, r.speculative_scale, r.contact_w = (
            s.pgs_beta,
            s.pgs_cfm,
            s.contact_speculative_scale,
            s._contact_w,
        )
        self.force, self.input, self.rows, self.contacts = f, x, r, contacts
        self.active = True
        return True

    def build_rows(self):
        """Route after collision completion without constructing duplicate row metadata."""
        r, x, s = self.rows, self.input, self.solver
        self.buckets.build(
            r.count,
            x.shape0,
            x.shape1,
            x.shape_body,
            r.body_art,
            s.art_to_world,
            slot=r.contact_slot,
            path=r.contact_path,
            needed=r.contact_needed,
        )

    def solve(self, iterations, omega, friction_start):
        """Publish every canonical row/impulse and final generalized velocity once."""
        if not self.active:
            raise RuntimeError("Parallel world solve requires an admitted current preparation")
        wp.launch_tiled(
            self.kernel,
            dim=[self.solver.world_count],
            inputs=[
                self.sparse.plan,
                self.sparse.data,
                self.kinetic.plan,
                self.kinetic.data,
                self.force,
                self.input,
                self.rows,
                self.buckets.data,
                iterations,
                omega,
                friction_start,
                self.solver.v_out,
            ],
            block_dim=128,
            device=self.solver.model.device,
        )
        if self.split:
            s = self.solver
            wp.launch_tiled(
                self.solve_kernel,
                dim=[s.world_count],
                inputs=[
                    self.sparse.plan,
                    self.sparse.data,
                    s.constraint_count,
                    s.rhs,
                    s.diag,
                    s.impulses,
                    s.row_type,
                    s.row_parent,
                    s.row_mu,
                    iterations,
                    omega,
                    friction_start,
                    s.v_hat,
                    s.v_out,
                ],
                block_dim=32,
                device=s.model.device,
            )

    def check(self):
        if self.status.numpy()[0] != 0 or (self.buckets is not None and np.any(self.buckets.data.invalid.numpy())):
            raise RuntimeError("Parallel world raw mapping or physical representation guard failed")
