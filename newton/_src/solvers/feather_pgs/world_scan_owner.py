# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private late all-world publication binding and original cache-epoch ownership."""

import numpy as np
import warp as wp

from ...sim import ModelFlags
from . import kuka_joint_owner, kuka_joint_world, simple_world, world_scan_publication

MODEL_FIELDS = (
    "joint_type",
    "joint_parent",
    "joint_child",
    "joint_q_start",
    "joint_qd_start",
    "joint_dof_dim",
    "joint_axis",
    "joint_X_p",
    "joint_X_c",
    "body_com",
    "body_mass",
    "body_inertia",
    "gravity",
)
SOLVER_FIELDS = {
    "kinematic_dof_mask": "_kinematic_dof_mask",
    "kinematic_joint_mask": "_kinematic_joint_mask",
    "free_root_joint_indices": "_free_root_joint_indices",
    "body_X_com": "body_X_com",
    "body_to_articulation": "body_to_articulation",
    "is_free_rigid": "is_free_rigid",
    "v_out": "v_out",
    "fk_id_cache_valid": "_fk_id_cache_valid",
}
CACHE_FIELDS = (
    "body_q_com",
    "joint_S_s",
    "articulation_origin",
    "body_v_s",
    "body_a_s",
    "body_f_s",
    "body_I_s",
    "body_inertia_terms",
)
PLAN_FIELDS = (
    "articulation_start",
    "joint_type",
    "joint_parent",
    "joint_child",
    "joint_q_start",
    "joint_qd_start",
    "joint_dof_dim",
    "body_flags",
)


def supported(solver):
    """Restrict publication to the direct Kuka recipe and its compatible light owner."""
    light = getattr(solver, "_joint_world", None)
    return (
        solver.model.device.is_cuda
        and not solver.model.particle_count
        and solver._fk_id_cache_enabled
        and (light is None or (isinstance(light, kuka_joint_owner.JointWorldOwner) and light.solver is solver))
        and simple_world._supported(solver)
    )


class WorldScanOwner:
    """Replace only the complete original late owner after all solve readers have joined."""

    def __init__(self, solver, *, host_plan):
        self.solver = solver
        self.joint_owner = getattr(solver, "_joint_world", None)
        self.plan = (
            self.joint_owner.plan if self.joint_owner is not None else host_plan.device_data(solver.model.device)
        )
        self.kernel = world_scan_publication.get_kernel(str(solver.model.device.arch))
        self.resolved_kernel = (
            world_scan_publication.get_resolved_kernel(str(solver.model.device.arch))
            if self.joint_owner is not None
            else None
        )
        self.model_plan_values = {name: getattr(solver.model, name).numpy().copy() for name in PLAN_FIELDS}

    def validate_notification(self, flags):
        """Keep numeric changes on current bindings without repeated topology readbacks."""
        numeric = int(
            ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.BODY_INERTIAL_PROPERTIES
            | ModelFlags.SHAPE_PROPERTIES
            | ModelFlags.MODEL_PROPERTIES
        )
        value = int(flags)
        if value != 0 and value & ~numeric == 0:
            return
        for name, expected in self.model_plan_values.items():
            if not np.array_equal(getattr(self.solver.model, name).numpy(), expected):
                raise RuntimeError(
                    "World publication static ownership changed; reconstruct the solver and recapture graphs"
                )

    def try_publish(self, state_in, state_aug, state_out, dt):
        """Bind live values and return false before any write for unsupported or aliased states."""
        solver, model = self.solver, self.solver.model
        if not supported(solver) or state_in.requires_grad or state_out.requires_grad:
            return False
        if getattr(solver, "_joint_world", None) is not self.joint_owner:
            return False
        light_active = getattr(solver, "_joint_world_active", False)
        if light_active and self.joint_owner is None:
            return False
        data = world_scan_publication.PublicationData()
        for name in MODEL_FIELDS:
            setattr(data, name, getattr(model, name))
        for name, source in SOLVER_FIELDS.items():
            setattr(data, name, getattr(solver, source))
        data.joint_q, data.joint_qd = state_in.joint_q, state_in.joint_qd
        data.joint_q_new, data.joint_qd_new = state_out.joint_q, state_out.joint_qd
        data.joint_qdd = state_aug.joint_qdd
        data.body_q, data.body_qd = state_out.body_q, state_out.body_qd
        cache = solver._fk_id_cache
        for name in CACHE_FIELDS:
            if cache is not None:
                value = getattr(cache, name)
            elif name == "articulation_origin":
                value = solver.articulation_origin
            elif name == "body_inertia_terms":
                value = solver._body_inertia_terms
            else:
                value = getattr(state_aug, name)
            setattr(data, name, value)
        sources = [getattr(state_in, name) for name in ("joint_q", "joint_qd", "body_q", "body_qd")]
        destinations = [
            data.v_out,
            data.joint_qdd,
            data.joint_q_new,
            data.joint_qd_new,
            data.body_q,
            data.body_qd,
            *(getattr(data, name) for name in CACHE_FIELDS),
            data.fk_id_cache_valid,
        ]
        if any(not value.is_contiguous for value in (*sources, *destinations)):
            return False
        source_ranges = [(value.ptr, value.ptr + value.capacity) for value in sources if value.size]
        destination_ranges = [(value.ptr, value.ptr + value.capacity) for value in destinations if value.size]
        if any(
            start < end_out and start_out < end
            for start, end in source_ranges
            for start_out, end_out in destination_ranges
        ):
            return False
        data.dt, data.angular_damping = dt, solver.angular_damping
        next_refresh = ((solver._step + 1) % solver.update_mass_matrix_interval) == 0
        parallel_next_refresh = next_refresh and solver._global_inertia_stream is not None
        data.materialize_all_body_inertia = int(next_refresh and not parallel_next_refresh)
        data.materialize_body_inertia_terms = int(parallel_next_refresh)
        inputs = [self.plan, data]
        kernel = self.kernel
        if light_active:
            # Only the current S3 light invocation has integrated resolved worlds.
            # The ordinary S4 classifier writes the same mask without integration.
            inputs.append(self.joint_owner.output.resolved)
            kernel = self.resolved_kernel
        wp.launch_tiled(
            kernel,
            dim=[solver.world_count],
            inputs=inputs,
            block_dim=32,
            device=model.device,
        )
        solver._fk_id_cache_source_state = state_out
        return True


def create_owner(solver):
    """Leave unsupported models on the complete original publication path."""
    if not supported(solver):
        return None
    try:
        light = getattr(solver, "_joint_world", None)
        host_plan = light.host_plan if light is not None else kuka_joint_world.bind_plan(solver)
        world_scan_publication.validate_scan_plan(host_plan)
    except ValueError:
        return None
    return WorldScanOwner(solver, host_plan=host_plan)
