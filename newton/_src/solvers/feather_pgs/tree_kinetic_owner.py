# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Admission and live bindings for complete tree current/next kinetic state."""

import hashlib
import math
import os

import numpy as np
import warp as wp

from . import cached_state_binding, world_scan_owner
from .tree_kinetic_plan import cooperative_block_dim, make_plan

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


class TreeKineticState:
    """Own one current cache with device state-bank tags and a separate held W."""

    def __init__(self, sparse_owner, *, admission=None, chain_scan_default=None, complete=True, response="sparse"):
        from . import tree_kinetic_state as native  # noqa: PLC0415

        self.sparse = sparse_owner
        self.solver = solver = sparse_owner.solver
        self.motion_velocity = bool(solver.enable_joint_velocity_limits)
        self._motion_binding = None
        model, device = solver.model, solver.model.device
        # Direct construction is also used by CPU physical representation tests.
        # Production admission remains CUDA-only through supported().
        self.admission = admission
        self.complete = complete
        host = sparse_owner.host
        self.layout, parent, depth = make_plan(model, sparse_owner, complete=complete)
        self.group_count = sparse_owner.plan.group_to_art.size
        self._cache_to_art = sparse_owner.plan.group_to_art.reshape((self.group_count, 1))
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
        root_slot = np.empty(self.group_count, np.int32)
        q_index = np.full((self.group_count, dofs), -1, np.int32)
        for group, art in enumerate(group_to_art):
            start = int(starts[art])
            if self.layout.root_dofs and start not in root_lookup:
                raise ValueError("Tree kinetic state is missing a free-root transport slot")
            root_slot[group] = root_lookup.get(start, -1)
            for dof in range(self.layout.root_dofs, dofs):
                q_index[group, dof] = q_starts[start + host["dof_joint"][dof]]
        if chain_scan_default is None:
            chain_scan_default = os.environ.get("FEATHER_PGS_G1_CHAIN_SCAN", "0") == "1"
        self.chain_scan_requested = os.environ.get("FEATHER_PGS_TREE_CHAIN_SCAN", str(int(chain_scan_default))) == "1"
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
        groups = self.group_count
        self.data.bias = wp.empty((groups, dofs), dtype=float, device=device)
        self.data.com_offset = wp.empty((groups, bodies), dtype=wp.vec3, device=device)
        self.data.geometric = wp.empty((groups, self.layout.entries), dtype=float, device=device)
        self.data.current_valid = solver._fk_id_cache_valid
        self.data.geometry_valid = wp.zeros(groups, dtype=int, device=device)
        self.data.generation = wp.zeros(groups, dtype=wp.int64, device=device)
        self.data.geometry_generation = wp.full(groups, -1, dtype=wp.int64, device=device)
        self.data.source_q = wp.zeros(groups, dtype=wp.uint64, device=device)
        self.data.source_qd = wp.zeros(groups, dtype=wp.uint64, device=device)
        self.data.status = wp.zeros(groups, dtype=int, device=device)
        self.data.current_valid.zero_()
        self.bias = self.data.bias
        self.geometric = self.data.geometric
        self.geometry_valid = self.data.geometry_valid
        self.status = self.data.status
        body_velocity = getattr(sparse_owner, "requires_body_velocity", False)
        self.repair_kernel = native.get_state_kernel(
            False,
            self.chain_scan,
            layout=self.layout,
            body_velocity=body_velocity,
            motion_velocity=self.motion_velocity,
        )
        self.finish_kernel = native.get_state_kernel(
            True, self.chain_scan, layout=self.layout, body_velocity=body_velocity
        )
        self.predictor_kernel = native.get_predictor_kernel(self.chain_scan, layout=self.layout, response=response)
        self._model_plan = {name: _fingerprint(getattr(model, name)) for name in _PLAN_FIELDS}
        self._plan_dimensions = self._dimensions()
        self._mapping_plan = {
            "group_to_art": _fingerprint(sparse_owner.plan.group_to_art),
            "art_to_world": _fingerprint(sparse_owner.plan.art_to_world),
            "art_dof_start": _fingerprint(sparse_owner.plan.art_dof_start),
            "free_root_joint_indices": _fingerprint(solver._free_root_joint_indices),
        }

    def _dimensions(self):
        model = self.solver.model
        return (
            self.solver.world_count,
            model.world_count,
            model.articulation_count,
            model.body_count,
            model.joint_count,
            model.joint_coord_count,
            model.joint_dof_count,
        )

    def validate_model(self):
        """Reject changed baked ownership before notification invalidates any state."""
        if self._dimensions() != self._plan_dimensions:
            raise RuntimeError("Tree kinetic dimensions changed; reconstruct and recapture")
        make_plan(self.solver.model, self.sparse, complete=self.complete)
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
        """Invalidate current state for every owned articulation in selected worlds."""
        cached_state_binding.invalidate(
            self.solver.world_count,
            world_mask,
            self._cache_to_art,
            self.sparse.plan.art_to_world,
            self.data.current_valid,
            self.data.geometry_valid,
        )

    def _validate_call(self, state, state_aug, dt):
        if state.requires_grad or state_aug is not self.solver:
            raise RuntimeError("Tree kinetic state excludes gradient/alternate augmented ownership")
        if self.admission is not None and not self.admission(self.solver):
            raise RuntimeError("Tree kinetic configuration changed; reconstruct and recapture")
        if bool(self.solver.enable_joint_velocity_limits) != self.motion_velocity:
            raise RuntimeError("Tree kinetic motion-velocity contract changed; reconstruct and recapture")
        if dt is None or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Tree kinetic state requires a positive finite step duration")
        for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"):
            value = getattr(state, name)
            if value.device != self.solver.model.device or not value.is_contiguous:
                raise RuntimeError("Tree kinetic state requires contiguous same-device state arrays")

    def _publication(self, state_in, state_aug, state_out, dt):
        """Bind live current and next-state publication descriptors."""
        return cached_state_binding.publication(self.solver, state_in, state_aug, state_out, dt)

    def begin(self, state_in, state_aug, stage3_qd, dt, global_refresh):
        """Repair current caches or late-request geometry before any factor read."""
        self._validate_call(state_in, state_aug, dt)
        binding = self._validate_motion(state_in, state_aug, stage3_qd)
        data = self._publication(state_in, state_aug, state_in, dt)
        arguments = [
            self.sparse.plan,
            self.plan,
            data,
            self.data,
            self.solver._mass_update_requested,
            int(global_refresh),
        ]
        if self.motion_velocity:
            arguments.append(stage3_qd)
        wp.launch_tiled(
            self.repair_kernel,
            dim=[self.group_count],
            inputs=arguments,
            block_dim=self.block_dim,
            device=self.solver.model.device,
        )
        self._motion_binding = binding

    def _validate_motion(self, state, state_aug, velocity):
        """Reject invalid working descriptors before changing current-cache validity."""
        raw = state.joint_qd
        if (
            not isinstance(velocity, wp.array)
            or velocity.device != raw.device
            or velocity.dtype != raw.dtype
            or velocity.shape != raw.shape
            or not velocity.is_contiguous
        ):
            raise RuntimeError("Tree kinetic motion velocity requires a matching contiguous state-vector descriptor")
        if not self.motion_velocity and velocity.ptr != raw.ptr:
            raise RuntimeError("Tree kinetic state excludes alternate prescaled predictor velocity")

        def overlaps(array):
            return velocity.ptr < array.ptr + array.capacity and array.ptr < velocity.ptr + velocity.capacity

        if (velocity.ptr != raw.ptr and overlaps(raw)) or any(
            overlaps(array) for array in (state.joint_q, state_aug.joint_qdd, self.solver.v_hat, self.solver.v_out)
        ):
            raise RuntimeError("Tree kinetic motion velocity overlaps state coordinates or response outputs")
        return state.joint_q.ptr, raw.ptr, velocity.ptr

    def predict(self, state_in, state_aug, control, stage3_qd, dt):
        """Apply live force buckets through held W with original root transport."""
        from .tree_kinetic_state import ForceInput  # noqa: PLC0415

        self._validate_call(state_in, state_aug, dt)
        binding = self._validate_motion(state_in, state_aug, stage3_qd)
        if self.motion_velocity and binding != self._motion_binding:
            raise RuntimeError("Tree kinetic predictor motion velocity differs from the current-cache binding")
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
            dim=[self.group_count],
            inputs=[self.sparse.plan, self.sparse.data, self.plan, self.data, force],
            block_dim=self.block_dim,
            device=solver.model.device,
        )

    def finish(self, state_in, state_aug, state_out, dt, next_refresh):
        """Integrate once, publish complete public state and construct next bias."""
        self._validate_call(state_in, state_aug, dt)
        self._validate_call(state_out, state_aug, dt)
        if not cached_state_binding.disjoint_states(state_in, state_out):
            raise RuntimeError("Tree kinetic integration requires non-overlapping generalized state banks")
        data = self._publication(state_in, state_aug, state_out, dt)
        wp.launch_tiled(
            self.finish_kernel,
            dim=[self.group_count],
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
