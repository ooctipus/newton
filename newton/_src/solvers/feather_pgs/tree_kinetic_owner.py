# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Admission and live bindings for complete tree current/next kinetic state."""

import hashlib
import math
import os

import numpy as np
import warp as wp

from . import world_scan_owner
from .tree_kinetic_plan import cooperative_block_dim, make_plan
from .world_scan_publication import PublicationData

_PLAN_FIELDS = (
    *world_scan_owner.PLAN_FIELDS,
    "joint_axis",
    "joint_X_p",
    "joint_X_c",
)


def _fingerprint(array):
    """Keep immutable-plan evidence without retaining full host model copies."""
    if array is None:
        return None
    values = array.numpy()
    return values.shape, values.dtype.str, hashlib.sha256(values.tobytes()).digest()


@wp.kernel
def _invalidate(
    world_mask: wp.array[wp.bool],
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    current_valid: wp.array[int],
    geometry_valid: wp.array[int],
):
    group = wp.tid()
    art = group_to_art[group]
    if not world_mask or world_mask[art_to_world[art]]:
        current_valid[art] = 0
        geometry_valid[group] = 0


class TreeKineticState:
    """Own one current cache with device state-bank tags and a separate held W."""

    def __init__(self, sparse_owner, *, admission=None):
        from . import tree_kinetic_state as native  # noqa: PLC0415

        self.sparse = sparse_owner
        self.solver = solver = sparse_owner.solver
        model, device, worlds = solver.model, solver.model.device, solver.world_count
        # Direct construction is also used by CPU physical representation tests.
        # Production admission remains CUDA-only through supported().
        self.admission = admission
        host = sparse_owner.host
        self.layout, parent, depth = make_plan(model, sparse_owner)
        bodies, dofs = self.layout.bodies, self.layout.dofs
        jump = np.empty((self.layout.rounds, bodies), np.int32)
        jump[0] = parent
        for level in range(1, self.layout.rounds):
            previous = jump[level - 1]
            jump[level] = np.where(previous >= 0, previous[np.maximum(previous, 0)], -1)
        children, offsets = [], [0]
        for body in range(bodies):
            children.extend(np.flatnonzero(parent == body)[::-1].tolist())
            offsets.append(len(children))
        if len(children) != bodies - 1:
            raise ValueError("Tree kinetic state requires every true parent edge")

        group_to_art = sparse_owner.plan.group_to_art.numpy()
        starts = model.articulation_start.numpy()
        q_starts = model.joint_q_start.numpy()
        free_roots = solver._free_root_joint_indices
        free_roots = free_roots.numpy() if free_roots is not None else []
        root_lookup = {int(joint): slot for slot, joint in enumerate(free_roots)}
        root_slot = np.empty(worlds, np.int32)
        q_index = np.full((worlds, dofs), -1, np.int32)
        for group, art in enumerate(group_to_art):
            start = int(starts[art])
            if self.layout.root_dofs and start not in root_lookup:
                raise ValueError("Tree kinetic state is missing a free-root transport slot")
            root_slot[group] = root_lookup.get(start, -1)
            for dof in range(self.layout.root_dofs, dofs):
                q_index[group, dof] = q_starts[start + host["dof_joint"][dof]]
        self.chain_scan_requested = (
            os.environ.get("FEATHER_PGS_TREE_CHAIN_SCAN", os.environ.get("FEATHER_PGS_G1_CHAIN_SCAN", "0")) == "1"
        )
        self.chain_scan = False
        chain_host = None
        if self.chain_scan_requested:
            from . import tree_chain_scan  # noqa: PLC0415

            try:
                chain_host = tree_chain_scan.make_plan(parent)
            except ValueError:
                # Unsupported packing preserves the complete original owner.
                pass
        self.block_dim = cooperative_block_dim(self.layout, chain_host)
        if chain_host is None:
            self.plan = native.KineticPlan()
        else:
            self.plan = tree_chain_scan.ChainPlan()
            self.plan.chain_meta = wp.array(chain_host["meta"].reshape(-1), dtype=int, device=device)
            for name in ("light_offsets", "light_children", "stage_width"):
                setattr(self.plan, name, wp.array(chain_host[name], dtype=int, device=device))
            self.plan.chain_levels = chain_host["levels"]
            self.plan.chain_width = int(chain_host["widths"].max())
            self.chain_scan = True
        for name, values in (
            ("parent", parent),
            ("depth", depth),
            ("jump", jump),
            ("children_offsets", np.asarray(offsets, np.int32)),
            ("children", np.asarray(children, np.int32)),
            ("q_index", q_index.reshape(-1)),
            ("root_slot", root_slot),
        ):
            setattr(self.plan, name, wp.array(values, dtype=int, device=device))
        self.data = native.KineticData()
        self.data.bias = wp.empty((worlds, dofs), dtype=float, device=device)
        self.data.com_offset = wp.empty((worlds, bodies), dtype=wp.vec3, device=device)
        self.data.geometric = wp.empty((worlds, self.layout.entries), dtype=float, device=device)
        self.data.current_valid = solver._fk_id_cache_valid
        self.data.geometry_valid = wp.zeros(worlds, dtype=int, device=device)
        self.data.generation = wp.zeros(worlds, dtype=wp.int64, device=device)
        self.data.geometry_generation = wp.full(worlds, -1, dtype=wp.int64, device=device)
        self.data.source_q = wp.zeros(worlds, dtype=wp.uint64, device=device)
        self.data.source_qd = wp.zeros(worlds, dtype=wp.uint64, device=device)
        self.data.status = wp.zeros(worlds, dtype=int, device=device)
        self.data.current_valid.zero_()
        self.bias = self.data.bias
        self.geometric = self.data.geometric
        self.geometry_valid = self.data.geometry_valid
        self.status = self.data.status
        body_velocity = getattr(sparse_owner, "requires_body_velocity", False)
        self.repair_kernel = native.get_state_kernel(
            False, self.chain_scan, layout=self.layout, body_velocity=body_velocity
        )
        self.finish_kernel = native.get_state_kernel(
            True, self.chain_scan, layout=self.layout, body_velocity=body_velocity
        )
        self.predictor_kernel = native.get_predictor_kernel(self.chain_scan, layout=self.layout)
        self._model_plan = {name: _fingerprint(getattr(model, name)) for name in _PLAN_FIELDS}
        self._mapping_plan = {
            "group_to_art": _fingerprint(sparse_owner.plan.group_to_art),
            "art_to_world": _fingerprint(sparse_owner.plan.art_to_world),
            "art_dof_start": _fingerprint(sparse_owner.plan.art_dof_start),
            "free_root_joint_indices": _fingerprint(solver._free_root_joint_indices),
        }

    def validate_model(self):
        """Reject changed baked ownership before notification invalidates any state."""
        make_plan(self.solver.model, self.sparse)
        for name, expected in self._model_plan.items():
            if _fingerprint(getattr(self.solver.model, name)) != expected:
                raise RuntimeError(f"Tree kinetic static {name} changed; reconstruct and recapture")
        for name, expected in self._mapping_plan.items():
            value = (
                self.solver._free_root_joint_indices
                if name == "free_root_joint_indices"
                else getattr(self.sparse.plan, name)
            )
            if _fingerprint(value) != expected:
                raise RuntimeError(f"Tree kinetic static {name} mapping changed; reconstruct and recapture")

    def invalidate(self, world_mask=None):
        """Invalidate only current geometry/bias, never silently rebuild held W."""
        if world_mask is not None and (
            world_mask.dtype != wp.bool
            or world_mask.shape != (self.solver.world_count,)
            or world_mask.device != self.solver.model.device
        ):
            raise ValueError("Tree kinetic reset requires the original per-world bool mask")
        wp.launch(
            _invalidate,
            dim=self.solver.world_count,
            inputs=[
                world_mask,
                self.sparse.plan.group_to_art,
                self.sparse.plan.art_to_world,
                self.data.current_valid,
                self.data.geometry_valid,
            ],
            device=self.solver.model.device,
        )

    def _validate_call(self, state, state_aug, dt):
        if state.requires_grad or state_aug is not self.solver:
            raise RuntimeError("Tree kinetic state excludes gradient/alternate augmented ownership")
        if self.admission is not None and not self.admission(self.solver):
            raise RuntimeError("Tree kinetic configuration changed; reconstruct and recapture")
        if dt is None or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Tree kinetic state requires a positive finite step duration")
        for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"):
            value = getattr(state, name)
            if value.device != self.solver.model.device or not value.is_contiguous:
                raise RuntimeError("Tree kinetic state requires contiguous same-device state arrays")

    def _publication(self, state_in, state_aug, state_out, dt):
        """Rebind current descriptors each call; no cached numeric model pointers."""
        solver, model = self.solver, self.solver.model
        data = PublicationData()
        for name in world_scan_owner.MODEL_FIELDS:
            setattr(data, name, getattr(model, name))
        for name, source in world_scan_owner.SOLVER_FIELDS.items():
            setattr(data, name, getattr(solver, source))
        data.joint_q, data.joint_qd = state_in.joint_q, state_in.joint_qd
        data.joint_q_new, data.joint_qd_new = state_out.joint_q, state_out.joint_qd
        data.joint_qdd = state_aug.joint_qdd
        data.body_q, data.body_qd = state_out.body_q, state_out.body_qd
        for name in world_scan_owner.CACHE_FIELDS:
            if name == "articulation_origin":
                value = solver.articulation_origin
            elif name == "body_inertia_terms":
                value = solver._body_inertia_terms
            else:
                value = getattr(state_aug, name)
            setattr(data, name, value)
        data.dt, data.angular_damping = dt, solver.angular_damping
        data.materialize_all_body_inertia = 0
        data.materialize_body_inertia_terms = 0
        return data

    def begin(self, state_in, state_aug, stage3_qd, dt, global_refresh):
        """Repair current caches or late-request geometry before any factor read."""
        self._validate_call(state_in, state_aug, dt)
        if stage3_qd.ptr != state_in.joint_qd.ptr:
            raise RuntimeError("Tree kinetic state excludes alternate prescaled predictor velocity")
        data = self._publication(state_in, state_aug, state_in, dt)
        wp.launch_tiled(
            self.repair_kernel,
            dim=[self.solver.world_count],
            inputs=[
                self.sparse.plan,
                self.plan,
                data,
                self.data,
                self.solver._mass_update_requested,
                int(global_refresh),
            ],
            block_dim=self.block_dim,
            device=self.solver.model.device,
        )

    def predict(self, state_in, state_aug, control, stage3_qd, dt):
        """Apply live force buckets through held W with original root transport."""
        from .tree_kinetic_state import ForceInput  # noqa: PLC0415

        self._validate_call(state_in, state_aug, dt)
        solver = self.solver
        force = ForceInput()
        force.joint_q, force.joint_qd = state_in.joint_q, state_in.joint_qd
        force.predictor_qd = stage3_qd
        force.joint_f, force.body_f = control.joint_f, state_in.body_f
        force.body_flags = solver.model.body_flags
        force.stiffness = solver._passive_spring_stiffness
        force.reference = solver._passive_spring_ref
        force.damping = solver._passive_joint_damping
        force.u0, force.S = state_aug.joint_tau, state_aug.joint_S_s
        force.qdd, force.vhat = state_aug.joint_qdd, solver.v_hat
        force.dt = dt
        wp.launch_tiled(
            self.predictor_kernel,
            dim=[solver.world_count],
            inputs=[self.sparse.plan, self.sparse.data, self.plan, self.data, force],
            block_dim=self.block_dim,
            device=solver.model.device,
        )

    def finish(self, state_in, state_aug, state_out, dt, next_refresh):
        """Integrate once, publish complete public state and construct next bias."""
        self._validate_call(state_in, state_aug, dt)
        self._validate_call(state_out, state_aug, dt)
        # Integration across threads requires separate generalized input/output
        # arrays. Reject overlap before any publication or source-state write.
        inputs = (state_in.joint_q, state_in.joint_qd)
        outputs = (state_out.joint_q, state_out.joint_qd)
        if any(a.ptr < b.ptr + b.capacity and b.ptr < a.ptr + a.capacity for a in inputs for b in outputs):
            raise RuntimeError("Tree kinetic integration requires non-overlapping generalized state banks")
        data = self._publication(state_in, state_aug, state_out, dt)
        wp.launch_tiled(
            self.finish_kernel,
            dim=[self.solver.world_count],
            inputs=[
                self.sparse.plan,
                self.plan,
                data,
                self.data,
                self.solver._mass_update_requested,
                int(next_refresh),
            ],
            block_dim=self.block_dim,
            device=self.solver.model.device,
        )
