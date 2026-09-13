# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Live, preallocated descriptor bindings for the compact kinetic boundary.

The solver remains the owner of canonical row, response, MF, and raw-capacity
storage. This module owns only compact maps, statuses and queue metadata.
Binding does not launch kernels, read back dynamic values, or seed held state.
"""

import inspect
from types import SimpleNamespace

import numpy as np
import warp as wp

from . import kernels, raw_world_contacts, simple_world, world_scan_owner


def _struct(cls, **fields):
    result = cls()
    for name, value in fields.items():
        setattr(result, name, value)
    return result


def _arguments(kernel, values):
    parameters = inspect.signature(kernel.func).parameters
    if set(parameters) != set(values):
        raise ValueError(f"Changed original kernel arguments: {set(parameters) ^ set(values)}")
    return [values[name] for name in parameters]


def _launch(kernel, arguments, dim, device, block=32, tiled=True):
    def launch():
        if tiled:
            wp.launch_tiled(kernel, dim=[dim], inputs=arguments, block_dim=block, device=device)
        else:
            wp.launch(kernel, dim=dim, inputs=arguments, device=device)

    return launch


def _disjoint_states(state_in, state_out):
    """Reject cross-field byte overlap before any current or next-state write."""
    sources = [getattr(state_in, name) for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f")]
    outputs = [getattr(state_out, name) for name in ("joint_q", "joint_qd", "body_q", "body_qd")]
    if any(value is None or not value.is_contiguous for value in (*sources, *outputs)):
        return False
    return not any(
        a.ptr < b.ptr + b.capacity and b.ptr < a.ptr + a.capacity for a in sources for b in outputs if a.size and b.size
    )


class LiveBindings:
    """Allocate compact metadata once and bind actual current call operands."""

    def __init__(self, solver, plan):
        from . import kinetic_rows_types as rt  # noqa: PLC0415 -- optional experimental factories stay lazy
        from . import kinetic_types as kt  # noqa: PLC0415 -- optional experimental factories stay lazy

        self.solver, self.host_plan, self.plan = solver, plan, plan.data
        self.device, self.worlds = solver.model.device, plan.worlds
        device, worlds = self.device, self.worlds
        for name in ("L_by_size", "Linv_by_size"):
            value = getattr(solver, name).get(6)
            if value is None or value.shape != (worlds, 6, 6):
                raise ValueError("Kinetic owner requires existing complete original free lower/inverse storage")
        for name in ("rigid_body_max_linear_velocity", "rigid_body_max_angular_velocity"):
            value = getattr(solver, name)
            if value is None or value.shape != (solver.model.body_count,):
                raise ValueError("Kinetic ZERO requires original per-body velocity-limit inputs")
        self.endpoint_twists = wp.empty(solver.model.body_count, dtype=wp.spatial_vector, device=device)
        self.predictor_status = wp.zeros(worlds, dtype=int, device=device)
        self.external_nonzero = wp.zeros(worlds, dtype=int, device=device)
        self.active_worlds = wp.empty(worlds, dtype=int, device=device)
        self.active_count = wp.zeros(1, dtype=int, device=device)
        resolved = solver._resolved_simple_worlds
        self.resolved = resolved if resolved.size == worlds else wp.zeros(worlds, dtype=int, device=device)
        self.arm = rt.allocate_arm_map(worlds, device)
        self.row_valid = wp.empty((worlds, 192), dtype=int, device=device)
        self.row_status = wp.zeros(worlds, dtype=int, device=device)
        self.row_global_status = wp.zeros(1, dtype=int, device=device)
        self.secondary_nonzero = wp.zeros(worlds, dtype=int, device=device)
        self.selector = solver._local_solve_owner
        self.solve_status = wp.zeros(worlds, dtype=int, device=device)
        self.guard_error = wp.zeros(worlds, dtype=int, device=device)
        self.frame_status = wp.zeros(1, dtype=int, device=device)
        self.hybrid_eligible = wp.zeros(worlds, dtype=int, device=device)
        self.stream = wp.Stream(device) if device.is_cuda else None
        self.ready = wp.Event(device) if device.is_cuda else None
        self.done = wp.Event(device) if device.is_cuda else None
        self.pending = False
        self.identity = wp.array(np.arange(worlds, dtype=np.int32), dtype=int, device=device)
        self.world_count = wp.array([worlds], dtype=int, device=device)
        self.zero_worlds = wp.zeros(worlds, dtype=int, device=device)
        self.zero_deferred = wp.zeros((worlds, 29), dtype=int, device=device)
        self.empty_spatial = wp.empty(0, dtype=wp.spatial_vector, device=device)
        self.dummy_float = wp.zeros((1, 1), dtype=float, device=device)
        self.dummy_int = wp.zeros((1, 1), dtype=int, device=device)
        self.prefix_count = wp.zeros(worlds, dtype=int, device=device)
        self.propagation_counter = wp.zeros(worlds, dtype=int, device=device)
        self.propagation_count = wp.zeros(worlds, dtype=int, device=device)
        self.propagation_dropped = wp.zeros(worlds, dtype=int, device=device)
        self.limit_slot = solver.rigid_velocity_limit_slot
        self.limit_sign = solver.rigid_velocity_limit_sign
        if self.limit_slot is None:
            self.limit_slot = wp.full(12 * worlds, -1, dtype=int, device=device)
            self.limit_sign = wp.zeros(12 * worlds, dtype=float, device=device)
        self._calls = {}
        self.free_mass_mask = solver.mass_update_mask
        self.free_row_K = getattr(solver, "aug_row_K", None)
        if self.free_row_K is None:
            self.free_row_K = wp.zeros(1, dtype=float, device=device)
        self.free_dof_joint_offset = wp.zeros(6, dtype=int, device=device)
        self.free_lower_schedule = wp.array(
            np.asarray([r * 6 + c | (c + 1) << 8 | r << 16 for r in range(6) for c in range(r + 1)], np.int32),
            dtype=int,
            device=device,
        )
        self.free_cpu_H = solver.H_by_size[6] if device.is_cpu else None
        self.model_plan_values = {
            name: getattr(solver.model, name).numpy().copy() for name in world_scan_owner.PLAN_FIELDS
        }
        self.buckets = getattr(getattr(solver, "_joint_world", None), "buckets", None)
        if self.buckets is None:
            self.buckets = raw_world_contacts.RawWorldContactBuckets(worlds, solver._max_contacts_alloc, device=device)
        self.output = _struct(
            kt.KineticPredictorOutput,
            v_hat=solver.v_hat,
            joint_qdd=solver.joint_qdd,
            endpoint_twists=self.endpoint_twists,
            status=self.predictor_status,
            external_nonzero=self.external_nonzero,
        )

    def _publication(self, state_in, state_out, state_aug, dt, *, current):
        from .kinetic_state import PublicationData  # noqa: PLC0415 -- optional experimental factories stay lazy

        solver, model = self.solver, self.solver.model
        data = PublicationData()
        for name in world_scan_owner.MODEL_FIELDS:
            setattr(data, name, getattr(model, name))
        for name, source in world_scan_owner.SOLVER_FIELDS.items():
            setattr(data, name, getattr(solver, source))
        data.joint_q, data.joint_qd = state_in.joint_q, state_in.joint_qd
        data.joint_q_new, data.joint_qd_new = state_out.joint_q, state_out.joint_qd
        data.joint_qdd = state_aug.joint_qdd
        target = state_in if current else state_out
        data.body_q, data.body_qd = target.body_q, target.body_qd
        cache = solver._fk_id_cache
        for name in world_scan_owner.CACHE_FIELDS:
            if cache is not None:
                value = getattr(cache, name)
            elif name == "articulation_origin":
                value = solver.articulation_origin
            elif name == "body_inertia_terms":
                value = solver._body_inertia_terms
            else:
                value = getattr(state_aug, name)
            setattr(data, name, value)
        data.dt, data.angular_damping = float(dt), solver.angular_damping
        data.materialize_all_body_inertia = 0
        data.materialize_body_inertia_terms = 0
        return data

    def bind_step(self, state_in, state_out, control, contacts, dt, slots, *, state_aug=None):
        """Bind current arrays only; all producers execute through returned closures."""
        from . import kinetic_predictor, kinetic_state  # noqa: PLC0415 -- optional experimental factories stay lazy
        from . import kinetic_types as kt  # noqa: PLC0415 -- optional experimental factories stay lazy

        solver, model = self.solver, self.solver.model
        if state_in.requires_grad or state_out.requires_grad or not _disjoint_states(state_in, state_out):
            raise ValueError("Kinetic live state inputs and outputs must be distinct contiguous storage")
        if solver.enable_joint_velocity_limits:
            raise ValueError("Velocity prescale requires a distinct current-bias ownership path")
        if contacts.rigid_contact_max != solver._max_contacts_alloc:
            raise ValueError("Live contacts capacity changed; reconstruct bindings outside capture")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Physical timestep must be finite and positive")
        if control.joint_target_q.shape != (model.joint_dof_count,):
            raise ValueError("Kinetic drive law requires the admitted original DOF-layout targets")
        state_aug = solver if state_aug is None else state_aug
        current = self._publication(state_in, state_out, state_aug, dt, current=True)
        following = self._publication(state_in, state_out, state_aug, dt, current=False)
        sources = [
            state_in.joint_q,
            state_in.joint_qd,
            state_in.body_q,
            state_in.body_qd,
            state_in.body_f,
            control.joint_f,
            control.joint_target_q,
            control.joint_target_qd,
        ]
        destinations = [
            solver.v_hat,
            solver.v_out,
            state_aug.joint_qdd,
            state_out.joint_q,
            state_out.joint_qd,
            state_out.body_q,
            state_out.body_qd,
            *(getattr(following, name) for name in world_scan_owner.CACHE_FIELDS),
            following.fk_id_cache_valid,
        ]
        if any(value is None or not value.is_contiguous for value in (*sources, *destinations)):
            raise ValueError("Kinetic live inputs and public/cache outputs must be contiguous")
        if any(
            a.ptr < b.ptr + b.capacity and b.ptr < a.ptr + a.capacity
            for a in sources
            for b in destinations
            if a.size and b.size
        ):
            raise ValueError("Kinetic public/cache outputs overlap current source storage")
        if any(
            a.ptr < b.ptr + b.capacity and b.ptr < a.ptr + a.capacity
            for i, a in enumerate(destinations)
            for b in destinations[i + 1 :]
            if a.size and b.size
        ):
            raise ValueError("Kinetic output ownership overlaps")
        inputs = _struct(
            kt.CurrentForceInput,
            joint_q=state_in.joint_q,
            joint_qd=state_in.joint_qd,
            predictor_qd=state_in.joint_qd,
            joint_f=control.joint_f,
            body_f=state_in.body_f,
            body_flags=model.body_flags,
            joint_spring_stiffness=solver._passive_spring_stiffness,
            joint_spring_ref=solver._passive_spring_ref,
            joint_damping=solver._passive_joint_damping,
            joint_target_ke=model.joint_target_ke,
            joint_target_kd=model.joint_target_kd,
            joint_target_pos=control.joint_target_q,
            joint_target_vel=control.joint_target_qd,
            joint_effort_limit=model.joint_effort_limit,
            drive_row_by_dof=self.host_plan.drive_row_by_dof,
            drive_q_index_by_dof=self.host_plan.drive_q_index_by_dof,
            kinematic_dof=solver._kinematic_dof_mask,
            kinematic_joint=solver._kinematic_joint_mask,
            joint_qd_start=model.joint_qd_start,
            joint_q_start=model.joint_q_start,
            free_root_joints=solver._free_root_joint_indices,
            prescribed_body_v_s=current.body_v_s,
            state_generation=slots.state_generation,
            expected_held_generation=slots.expected_held_generation,
            dt=float(dt),
        )
        refresh = _struct(
            kt.RefreshInput,
            requested=slots.requested,
            state_generation=slots.state_generation,
            held_generation=slots.expected_held_generation,
            R=model.joint_armature,
            joint_target_ke=model.joint_target_ke,
            joint_target_kd=model.joint_target_kd,
            drive_row_by_dof=self.host_plan.drive_row_by_dof,
            dt=float(dt),
            status=slots.refresh_status,
        )
        output = _struct(
            kt.KineticPredictorOutput, **{name: getattr(self.output, name) for name in kt.KineticPredictorOutput.vars}
        )
        output.joint_qdd = state_aug.joint_qdd
        device, worlds, plan = self.device, self.worlds, self.plan
        arch = str(device.arch)
        construct_args = [plan, current, slots.schedule, slots.current, slots.geometric]
        refresh_args = [plan, slots.geometric, slots.held, refresh]
        predict_args = [plan, slots.current, slots.held, inputs, output]
        finish_args = [plan, following, slots.next_schedule, slots.next_current, slots.next_geometric]
        call = SimpleNamespace(
            current_publication=current,
            next_publication=following,
            inputs=inputs,
            output=output,
            refresh=refresh,
            slots=slots,
            plan=plan,
            device=device,
            worlds=worlds,
            contacts=contacts,
            construct_launch=_launch(kinetic_state.get_construct_kernel(arch), construct_args, worlds, device),
            refresh_launch=_launch(kinetic_predictor.get_refresh_kernel(arch), refresh_args, worlds, device),
            predict_launch=_launch(kinetic_predictor.get_predictor_kernel(arch), predict_args, worlds, device),
            finish_launch=_launch(kinetic_state.get_finish_kernel(arch), finish_args, worlds, device),
            kernel_arguments={
                "construct": construct_args,
                "refresh": refresh_args,
                "predict": predict_args,
                "finish": finish_args,
            },
        )
        call.rows = self._rows(call, state_in, contacts, dt)
        from .kinetic_live_services import bind_mf  # noqa: PLC0415 -- optional experimental factories stay lazy

        call.services = bind_mf(
            solver,
            call.rows,
            current,
            selector=self.selector,
            identity=self.identity,
            world_count=self.world_count,
            zero_worlds=self.zero_worlds,
            zero_deferred=self.zero_deferred,
            dummy_float=self.dummy_float,
            limit_slot=self.limit_slot,
            limit_sign=self.limit_sign,
        )
        call.zero = self._zero(call.rows, call.services)
        call.solve = self._solve(call.rows, call.services, slots.refresh_status)
        from . import kinetic_guard  # noqa: PLC0415 -- optional experimental factories stay lazy

        finish_args = [*finish_args, call.solve.guard]
        call.finish_launch = _launch(kinetic_guard.get_finish_kernel(arch), finish_args, worlds, device)
        call.kernel_arguments["finish"] = finish_args
        call.free = self._free(call, current)
        call.force = self._force(call.rows, call.services, call.solve.guard, contacts, dt)
        return call

    def validate_notification(self, flags):
        """Numeric model updates retain aliases; structural changes require recapture."""
        from ...sim import ModelFlags  # noqa: PLC0415 -- optional experimental factories stay lazy

        numeric = int(
            ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.BODY_INERTIAL_PROPERTIES
            | ModelFlags.SHAPE_PROPERTIES
            | ModelFlags.MODEL_PROPERTIES
        )
        if int(flags) != 0 and int(flags) & ~numeric == 0:
            if int(flags) & int(ModelFlags.MODEL_PROPERTIES):
                gravity = self.solver.model.gravity.numpy()
                if not len(gravity) or not np.array_equal(gravity, np.broadcast_to(gravity[0], gravity.shape)):
                    raise ValueError("Kinetic owner requires uniform gravity")
            return
        for name, expected in self.model_plan_values.items():
            if not np.array_equal(getattr(self.solver.model, name).numpy(), expected):
                raise ValueError("Kinetic topology changed; reconstruct the solver and recapture")

    def _rows(self, call, state_in, contacts, dt):
        from . import kinetic_rows, kinetic_rows_triplet  # noqa: PLC0415 -- optional experimental factories stay lazy
        from . import kinetic_rows_types as rt  # noqa: PLC0415 -- optional experimental factories stay lazy

        solver, model, device, worlds = self.solver, self.solver.model, self.device, self.worlds
        raw = rt.RawRowInput()
        raw.count = contacts.rigid_contact_count
        for name in ("point0", "point1", "normal", "margin0", "margin1", "shape0", "shape1"):
            setattr(raw, name, getattr(contacts, "rigid_contact_" + name))
        for name in ("world", "slot", "path", "art_a", "art_b", "slots_needed"):
            setattr(raw, name, getattr(solver, "contact_" + name))
        for name, source in {
            "body_to_articulation": "body_to_articulation",
            "art_to_world": "art_to_world",
            "response_count": "articulation_response_dof_count",
            "body_response_mask": "body_response_dof_mask",
            "prescribed_articulation": "_prescribed_articulation",
            "is_free_rigid": "is_free_rigid",
            "shape_mu": "shape_material_mu",
            "shape_restitution": "shape_material_restitution",
        }.items():
            setattr(raw, name, getattr(solver, source))
        raw.shape_body, raw.body_q = model.shape_body, state_in.body_q
        state = _struct(
            rt.RowState,
            active_worlds=self.active_worlds,
            active_count=self.active_count,
            resolved=self.resolved,
            predictor_status=call.output.status,
            state_generation=call.inputs.state_generation,
            held_generation=call.inputs.expected_held_generation,
            v_hat=call.output.v_hat,
            endpoint_twists=call.output.endpoint_twists,
            dense_count=solver.constraint_count,
            mf_count=solver.mf_constraint_count,
            primary_offset=self.host_plan.primary_offset,
            secondary_offset=self.host_plan.secondary_offset,
            raw_invalid=self.buckets.data.invalid,
            capacity_status=solver._constraint_capacity_status,
        )
        prefix = _struct(
            rt.PrefixInput,
            q=state_in.joint_q,
            q_index=solver._joint_limit_q_index,
            lower=model.joint_limit_lower,
            upper=model.joint_limit_upper,
        )
        settings = rt.RowSettings()
        for name, source in {
            "beta": "pgs_beta",
            "cfm": "pgs_cfm",
            "contact_speculative_scale": "contact_speculative_scale",
            "activation_gap": "joint_limit_activation_gap",
            "restitution_velocity_threshold": "_effective_restitution_velocity_threshold",
            "friction_scale": "contact_friction_scale",
            "friction_gap_threshold": "contact_friction_gap_threshold",
            "shared_anchor": "contact_shared_anchor",
            "friction_shared_anchor": "contact_friction_shared_anchor",
            "friction_anchor_limit": "contact_friction_anchor_limit",
            "friction_articulation_pairs_only": "contact_friction_articulation_pairs_only",
            "enable_friction": "enable_contact_friction",
        }.items():
            value = getattr(solver, source)
            setattr(settings, name, int(value) if isinstance(value, bool) else value)
        settings.dt, settings.bias_scale, settings.joint_limit_speculative_scale = float(dt), 1.0, 1.0
        settings.contact_w = solver._contact_w
        settings.raw_capacity, settings.dense_capacity, settings.mf_capacity = contacts.rigid_contact_max, 192, 64
        settings.workers = min(worlds, 4096)
        out = rt.DenseRowOutput()
        mapping = {
            "response": "Y_world",
            "physical_J": "J_world",
            "r0": "rhs_unbiased",
            "phase_bounds": "dense_phase_bounds",
        }
        private = {
            "valid": self.row_valid,
            "status": self.row_status,
            "global_status": self.row_global_status,
            "secondary_nonzero": self.secondary_nonzero,
        }
        for name in rt.DenseRowOutput.vars:
            setattr(out, name, private[name] if name in private else getattr(solver, mapping.get(name, name)))
        arch = str(device.arch)
        row_kernels = [
            kinetic_rows.get_arm_kernel(arch),
            kinetic_rows.get_prefix_kernel(arch),
            kinetic_rows_triplet.get_contact_kernel(arch),
            kinetic_rows.get_validate_kernel(arch),
        ]
        arguments = [
            [call.plan, call.slots.current, call.slots.held, state, self.arm, out],
            [call.plan, call.slots.held, prefix, state, settings, out],
            [call.plan, call.slots.current, call.slots.held, raw, state, settings, self.arm, out],
            [state, settings, out],
        ]

        def launch():
            out.global_status.zero_()
            for kernel, args, dim in zip(
                row_kernels, arguments, (worlds, worlds, settings.workers, worlds), strict=True
            ):
                wp.launch_tiled(kernel, dim=[dim], inputs=args, block_dim=32, device=device)

        return SimpleNamespace(
            plan=call.plan,
            current=call.slots.current,
            held=call.slots.held,
            raw=raw,
            state=state,
            prefix=prefix,
            settings=settings,
            arm=self.arm,
            out=out,
            device=device,
            worlds=worlds,
            launch=launch,
            arguments=arguments,
            kernels=row_kernels,
            inputs=call.inputs,
            publication=call.current_publication,
        )

    def _allocation(self, rows, mf):
        """Use current original raw routing and fixed-capacity counter finalizers."""
        from . import (  # noqa: PLC0415 -- optional experimental factories stay lazy
            solver_feather_pgs as original_solver,
        )
        from .kinetic_allocate import (  # noqa: PLC0415 -- optional experimental factories stay lazy
            AllocationData,
            initialize_count,
        )

        solver, s, raw, state = self.solver, rows.settings, rows.raw, rows.state
        device, worlds = self.device, self.worlds
        allocation = _struct(
            AllocationData,
            dense_counter=solver.slot_counter,
            mf_counter=solver.mf_slot_counter,
            propagation_counter=self.propagation_counter,
            propagation_count=self.propagation_count,
            dense_contact_flag=solver.dense_contact_world_flag,
            dropped_dense=solver._row_dropped_dense if solver._row_dropped_dense is not None else self.zero_worlds,
            dropped_mf=solver._row_dropped_mf if solver._row_dropped_mf is not None else self.zero_worlds,
            dropped_propagation=self.propagation_dropped,
            prefix_count=self.prefix_count,
        )
        body_has_response = solver.body_has_response_dofs
        root_dof_start = solver.articulation_root_dof_start
        shape_transform_unused = solver.model.shape_transform
        workers = min(s.raw_capacity, original_solver._CONTACT_BUILD_THREAD_CAP * original_solver._CONTACT_THREADS_X)
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
            "body_flags": self.solver.model.body_flags,
            "body_has_response_dofs": body_has_response,
            "is_free_rigid": raw.is_free_rigid,
            "has_free_rigid": int(mf.free_bodies.size > 0),
            "propagation_articulated_contacts": 0,
            "propagation_same_articulation": 0,
            "propagation_free_free": 0,
            "contact_gap_gate": self.solver.contact_gap_gate,
            "same_articulation_contact_gap_gate": self.solver.same_articulation_contact_gap_gate,
            "articulation_pair_contact_gap_gate": self.solver.articulation_pair_contact_gap_gate,
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
        parameters = inspect.signature(kernels.allocate_world_contact_slots.func).parameters
        if set(parameters) != set(values):
            raise ValueError("Original contact allocator signature changed")
        contact_args = [values[name] for name in parameters]
        limit_args = [
            mf.free_bodies,
            raw.body_to_articulation,
            raw.art_to_world,
            raw.is_free_rigid,
            self.solver.model.body_flags,
            mf.max_linear,
            mf.max_angular,
            root_dof_start,
            state.v_hat,
            self.solver.velocity_limit_activation_fraction,
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
                kernels.allocate_world_contact_slots,
                dim=workers,
                inputs=contact_args,
                device=device,
            )
            wp.copy(mf.contact_end, allocation.mf_counter)
            wp.launch(
                kernels.allocate_rigid_velocity_limit_slots,
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
                    kernels.finalize_constraint_counts_with_status,
                    dim=worlds,
                    inputs=[counter, cap, family, count, state.capacity_status],
                    device=device,
                )

        return SimpleNamespace(
            data=allocation,
            launch=launch,
            contact_args=contact_args,
            limit_args=limit_args,
            workers=workers,
            rows=rows,
            mf=mf,
        )

    def _zero(self, rows, mf):
        from . import kinetic_zero  # noqa: PLC0415 -- optional experimental factories stay lazy

        data = simple_world._SimpleWorldInput()
        data.world_dof_indices, data.world_dof_count = mf.world_dofs, self.solver.world_dof_count
        data.limit_q_index = rows.prefix.q_index
        data.lower, data.upper, data.q = rows.prefix.lower, rows.prefix.upper, rows.prefix.q
        data.v_hat = rows.state.v_hat
        for name in ("body_to_articulation", "art_to_world", "prescribed_articulation", "is_free_rigid"):
            setattr(data, name, getattr(rows.raw, name))
        data.articulation_dof_start = mf.art_dof_start
        data.articulation_response_dof_count = rows.raw.response_count
        data.body_flags, data.body_has_response_dofs = self.solver.model.body_flags, self.solver.body_has_response_dofs
        data.body_response_dof_mask = rows.raw.body_response_mask
        data.body_q = rows.raw.body_q
        data.body_v_s, data.joint_S_s = self.empty_spatial, self.empty_spatial
        data.articulation_origin = rows.current.origin.reshape((self.worlds * 3,))
        data.max_linear_velocity, data.max_angular_velocity = mf.max_linear, mf.max_angular
        data.max_depenetration_velocity = mf.max_depenetration
        s = rows.settings
        data.dt, data.beta, data.speculative_scale = s.dt, s.beta, s.contact_speculative_scale
        data.activation_gap, data.restitution_threshold = s.activation_gap, s.restitution_velocity_threshold
        data.shared_anchor = s.shared_anchor
        data.absolute_margin, data.relative_margin = 1e-5, 2.0**-20
        raw = _struct(
            simple_world._SimpleRawContacts,
            **{name: getattr(rows.raw, name) for name in simple_world._SimpleRawContacts.vars},
        )
        args = [self.host_plan.base, data, raw, self.buckets.data, rows.current, rows.state]
        kernel = kinetic_zero.get_kernel(str(self.device.arch))

        def build_buckets():
            self.buckets.build(
                raw.count, raw.shape0, raw.shape1, raw.shape_body, rows.raw.body_to_articulation, rows.raw.art_to_world
            )

        def classify():
            rows.state.active_count.zero_()
            wp.launch_tiled(kernel, dim=[self.worlds], block_dim=32, inputs=args, device=self.device)

        def launch():
            build_buckets()
            classify()

        return SimpleNamespace(
            build_buckets=build_buckets,
            classify=classify,
            launch=launch,
            arguments=args,
            data=data,
            raw=raw,
            buckets=self.buckets,
            kernel=kernel,
        )

    def _free(self, call, publication):
        from . import kinetic_free_refresh as free  # noqa: PLC0415 -- optional experimental factories stay lazy
        from . import solver_feather_pgs as original  # noqa: PLC0415 -- optional experimental factories stay lazy

        s, device, worlds = self.solver, self.device, self.worlds
        f, held = call.inputs, call.slots.held
        prepare = [
            call.plan.secondary_group,
            s.group_to_art[6],
            s.articulation_dof_start,
            call.slots.requested,
            f.drive_row_by_dof,
            f.joint_target_ke,
            f.joint_target_kd,
            f.dt,
            self.free_mass_mask,
            self.free_row_K,
        ]
        common = [
            s.model.articulation_start,
            s.articulation_dof_start,
            self.free_mass_mask,
            publication.joint_child,
            publication.joint_S_s,
            publication.body_I_s,
            s.group_to_art[6],
            s.R_by_size[6],
            self.free_dof_joint_offset,
        ]
        if device.is_cuda:
            factor = original._get_crba_cholesky_warp_kernel(6, str(device.arch), warps_per_block=4)
            inverse = original._get_inverse_cholesky_register_kernel(6, str(device.arch), lower_only=True)
            factor_args = [
                worlds,
                *common,
                self.free_lower_schedule,
                1,
                f.drive_row_by_dof,
                self.free_row_K,
                held.lower6,
            ]
            inverse_args = [held.lower6, s.group_to_art[6], self.free_mass_mask, held.inverse6]

            def launch_factor():
                wp.launch(factor, dim=worlds * 32, block_dim=128, inputs=factor_args, device=device)
                wp.launch(inverse, dim=worlds * 32, block_dim=256, inputs=inverse_args, device=device)
        else:
            factor = original._get_cholesky_kernel(6, str(device.arch), 64)
            assemble_args = [*common[:7], f.drive_row_by_dof, self.free_row_K, self.free_cpu_H]
            factor_args = [self.free_cpu_H, s.R_by_size[6], s.group_to_art[6], self.free_mass_mask, held.lower6]
            inverse_args = [held.lower6, s.group_to_art[6], self.free_mass_mask, held.inverse6]

            def launch_factor():
                wp.launch(free.assemble_free_cpu_reference, dim=(worlds, 6, 6), inputs=assemble_args, device=device)
                wp.launch_tiled(factor, dim=[worlds], block_dim=64, inputs=factor_args, device=device)
                wp.launch(free.inverse_lower_cpu_reference, dim=(worlds, 6), inputs=inverse_args, device=device)

        def launch():
            wp.launch(free.prepare_free_refresh, dim=worlds, inputs=prepare, device=device)
            launch_factor()

        return SimpleNamespace(
            launch=launch,
            arguments=[prepare, factor_args, inverse_args],
            lower=held.lower6,
            inverse=held.inverse6,
            R=s.R_by_size[6],
        )

    def _solve(self, rows, services, refresh_status):
        from . import kinetic_compact_coupled as compact  # noqa: PLC0415 -- optional experimental factories stay lazy
        from . import kinetic_guard as guards  # noqa: PLC0415 -- optional experimental factories stay lazy
        from . import kinetic_solve as native  # noqa: PLC0415 -- optional experimental factories stay lazy
        from .kinetic_solve_types import (  # noqa: PLC0415 -- optional experimental factories stay lazy
            KineticMFData,
            KineticSolveData,
        )

        device, worlds = self.device, self.worlds
        solve = _struct(
            KineticSolveData,
            selector=services.selector,
            v_out=services.v_out,
            status=self.solve_status,
            iterations=8,
            omega=1.0,
            friction_start_iteration=0,
            iteration_offset=0,
        )
        guard = _struct(
            guards.BoundaryGuard,
            refresh_status=refresh_status,
            predictor_status=rows.state.predictor_status,
            row_status=rows.out.status,
            solve_status=solve.status,
            resolved=rows.state.resolved,
            row_global_status=rows.out.global_status,
            raw_invalid=rows.state.raw_invalid,
            capacity_status=rows.state.capacity_status,
            error=self.guard_error,
            frame_status=self.frame_status,
        )
        mf_data = _struct(
            KineticMFData,
            **{name: getattr(services, "row_mu" if name == "mu" else name) for name in KineticMFData.vars},
        )
        hybrid = _struct(
            compact.HybridCoupledData,
            eligible=self.hybrid_eligible,
            mf_impulses=services.impulses,
            mf_contact_end=services.contact_end,
        )
        workers = min(worlds, 512)
        arguments = {
            "qualify": [rows.plan, rows.held, rows.state, rows.out, mf_data, solve, hybrid, workers],
            "materialize": [rows.plan, rows.held, rows.state, rows.out, solve, hybrid],
            "offset_eight": [rows.plan, rows.held, rows.state, rows.out, solve],
            "general": [
                *services.general_args,
                guard,
                rows.plan,
                rows.held,
                rows.state,
                rows.out,
                mf_data,
                solve,
                hybrid,
            ],
        }
        row_kernels = {
            "qualify": compact.get_qualify_kernel(str(device.arch)),
            "materialize": compact.get_materialize_kernel(str(device.arch)),
            "offset_eight": native.get_solve_kernel(str(device.arch)),
            "general": compact.get_general_kernel(services.general),
        }
        dimensions = {
            "qualify": (workers, 128),
            "materialize": (worlds, 32),
            "offset_eight": ((worlds + 1) // 2, 64),
            "general": (worlds, 32),
        }
        for name, kernel in row_kernels.items():
            if len(inspect.signature(kernel.func).parameters) != len(arguments[name]):
                raise ValueError("Changed complete solve signature: " + name)

        def dispatch(name):
            dim, block = dimensions[name]
            wp.launch_tiled(row_kernels[name], dim=[dim], block_dim=block, inputs=arguments[name], device=device)

        def join():
            if self.pending:
                wp.wait_event(self.done)
                self.pending = False

        def setup():
            join()
            guards.initialize_guard(guard, device)
            services.seed_velocity()

        def qualify():
            dispatch("qualify")

        def materialize():
            dispatch("materialize")
            guards.check_guard(guard, device)

        def solve_phase():
            if device.is_cuda:
                wp.record_event(self.ready)
                with wp.ScopedStream(self.stream):
                    wp.wait_event(self.ready)
                    try:
                        dispatch("offset_eight")
                    finally:
                        wp.record_event(self.done)
                        self.pending = True
                try:
                    dispatch("general")
                finally:
                    join()
            else:
                dispatch("general")
                dispatch("offset_eight")
            guards.check_guard(guard, device)

        allocation = self._allocation(rows, services)
        return SimpleNamespace(
            rows=rows,
            services=services,
            solve_descriptor=solve,
            mf_descriptor=mf_data,
            guard=guard,
            hybrid=hybrid,
            kernels=row_kernels,
            arguments=arguments,
            dimensions=dimensions,
            setup=setup,
            allocate=allocation.launch,
            build_rows=rows.launch,
            prepare_mf=services.prepare,
            qualify=qualify,
            materialize=materialize,
            solve=solve_phase,
            join=join,
            allocation=allocation,
            compact_coupled=True,
            stream=self.stream,
        )

    def _force(self, rows, mf, guard, contacts, dt):
        from . import kinetic_public_force as force  # noqa: PLC0415 -- optional experimental factories stay lazy

        r, state, out = rows.raw, rows.state, rows.out
        linear, spatial = contacts.rigid_contact_force, getattr(contacts, "force", None)
        if linear is None:
            return SimpleNamespace(launch=lambda: None, linear=None, spatial=spatial, arguments=[])
        values = {
            "contact_count": r.count,
            "contact_normal": r.normal,
            "contact_world": r.world,
            "contact_slot": r.slot,
            "contact_path": r.path,
            "world_impulses": out.impulses,
            "mf_impulses": mf.impulses,
            "propagation_impulses": self.dummy_float,
            "world_constraint_count": state.dense_count,
            "mf_constraint_count": state.mf_count,
            "propagation_constraint_count": self.propagation_count,
            "world_row_type": out.row_type,
            "world_row_parent": out.row_parent,
            "mf_row_type": mf.row_type,
            "mf_row_parent": mf.row_parent,
            "propagation_row_type": self.dummy_int,
            "propagation_row_parent": self.dummy_int,
            "enable_friction": rows.settings.enable_friction,
            "inv_dt": 1.0 / dt,
            "rigid_contact_force": linear,
        }
        first = force.get_kernel(force.NAMES[0])
        args = [*_arguments(getattr(kernels, force.NAMES[0]), values), guard.frame_status]
        second = force.get_kernel(force.NAMES[1]) if spatial is not None else None
        second_args = [r.count, linear, spatial, guard.frame_status]

        def launch():
            wp.launch(first, dim=rows.settings.raw_capacity, inputs=args, device=self.device)
            if second is not None:
                wp.launch(second, dim=rows.settings.raw_capacity, inputs=second_args, device=self.device)

        return SimpleNamespace(launch=launch, linear=linear, spatial=spatial, arguments=[args, second_args])
