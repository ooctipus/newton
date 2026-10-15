# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private lifecycle for the default-off joint-world Kuka experiment."""

import numpy as np
import warp as wp

from ...sim import ModelFlags
from . import kernels, kuka_joint_world, simple_world
from .raw_world_contacts import RawWorldContactBuckets

PUBLICATION_WIDTH = {
    "update_qdd_from_velocity": (35, "dof_ids"),
    "remove_free_root_transport_from_qdd": (2, "root_slots"),
    "integrate_generalized_joints": (32, "joint_ids"),
}


@wp.kernel(module="unique")
def active_joint_world_update_qdd_from_velocity(
    joint_qd: wp.array[float],
    kinematic_dof_mask: wp.array[int],
    inv_dt: float,
    v_new: wp.array[float],
    joint_qdd: wp.array[float],
    plan: kuka_joint_world.JointWorldPlan,
    active_worlds: wp.array[int],
    active_count: wp.array[int],
):
    linear = wp.tid()
    slot = linear // 35
    if slot >= active_count[0]:
        return
    world = active_worlds[slot]
    kernels._light_update_qdd_from_velocity(
        plan.dof_ids[world, linear % 35], joint_qd, kinematic_dof_mask, inv_dt, v_new, joint_qdd
    )


@wp.kernel(module="unique")
def active_joint_world_remove_free_root_transport_from_qdd(
    free_root_joint_indices: wp.array[int],
    joint_qd_start: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    joint_qd: wp.array[float],
    joint_qdd: wp.array[float],
    plan: kuka_joint_world.JointWorldPlan,
    active_worlds: wp.array[int],
    active_count: wp.array[int],
):
    linear = wp.tid()
    slot = linear // 2
    if slot >= active_count[0]:
        return
    world = active_worlds[slot]
    kernels._light_remove_free_root_transport_from_qdd(
        plan.root_slots[world, linear % 2],
        free_root_joint_indices,
        joint_qd_start,
        kinematic_joint_mask,
        joint_qd,
        joint_qdd,
    )


@wp.kernel(module="unique")
def active_joint_world_integrate_generalized_joints(
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    body_com: wp.array[wp.vec3],
    joint_X_c: wp.array[wp.transform],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_qdd: wp.array[float],
    dt: float,
    angular_damping: float,
    joint_q_new: wp.array[float],
    joint_qd_new: wp.array[float],
    plan: kuka_joint_world.JointWorldPlan,
    active_worlds: wp.array[int],
    active_count: wp.array[int],
):
    linear = wp.tid()
    slot = linear // 32
    if slot >= active_count[0]:
        return
    world = active_worlds[slot]
    kernels._light_integrate_generalized_joints(
        plan.joint_ids[world, linear % 32],
        joint_type,
        joint_parent,
        joint_child,
        joint_q_start,
        joint_qd_start,
        kinematic_joint_mask,
        joint_dof_dim,
        body_com,
        joint_X_c,
        joint_q,
        joint_qd,
        joint_qdd,
        dt,
        angular_damping,
        joint_q_new,
        joint_qd_new,
    )


_PUBLICATION_KERNELS = {
    "update_qdd_from_velocity": active_joint_world_update_qdd_from_velocity,
    "remove_free_root_transport_from_qdd": active_joint_world_remove_free_root_transport_from_qdd,
    "integrate_generalized_joints": active_joint_world_integrate_generalized_joints,
}


def get_publication_kernel(name: str):
    """Return the shared generalized law launched over a compact current-world prefix."""
    return _PUBLICATION_KERNELS[name]


def raw_inputs(solver, contacts):
    """Bind current raw descriptors without a per-step host readback."""
    raw = simple_world._SimpleRawContacts()
    raw.count = contacts.rigid_contact_count
    for name in ("shape0", "shape1", "point0", "point1", "normal", "margin0", "margin1"):
        setattr(raw, name, getattr(contacts, "rigid_contact_" + name))
    raw.shape_body = solver.model.shape_body
    raw.shape_mu, raw.shape_restitution = solver.shape_material_mu, solver.shape_material_restitution
    return raw


def simple_inputs(solver, state_in, state_aug, dt):
    """Preserve the original current-state classifier operands and margins."""
    data = simple_world._SimpleWorldInput()
    for name in (
        "world_dof_indices",
        "world_dof_count",
        "v_hat",
        "body_to_articulation",
        "art_to_world",
        "articulation_dof_start",
        "articulation_response_dof_count",
        "is_free_rigid",
        "body_has_response_dofs",
        "body_response_dof_mask",
        "articulation_origin",
    ):
        setattr(data, name, getattr(solver, name))
    data.limit_q_index = solver._joint_limit_q_index
    data.lower, data.upper, data.q = solver.model.joint_limit_lower, solver.model.joint_limit_upper, state_in.joint_q
    data.prescribed_articulation = solver._prescribed_articulation
    data.body_flags, data.body_q = solver.model.body_flags, state_in.body_q
    data.body_v_s, data.joint_S_s = state_aug.body_v_s, state_aug.joint_S_s
    data.max_linear_velocity = solver.rigid_body_max_linear_velocity
    data.max_angular_velocity = solver.rigid_body_max_angular_velocity
    data.max_depenetration_velocity = solver.rigid_body_max_depenetration_velocity
    data.dt, data.beta, data.speculative_scale = dt, solver.pgs_beta, solver.contact_speculative_scale
    data.activation_gap = solver.joint_limit_activation_gap
    data.restitution_threshold = solver._effective_restitution_velocity_threshold
    data.shared_anchor = int(solver.contact_shared_anchor)
    classifier = solver._simple_world_classifier
    data.absolute_margin, data.relative_margin = classifier.absolute_margin, classifier.relative_margin
    return data


def notification_plan_snapshot(flags):
    """Create one lazy host snapshot for this callback, never for numeric flags."""
    numeric = int(
        ModelFlags.JOINT_DOF_PROPERTIES
        | ModelFlags.BODY_INERTIAL_PROPERTIES
        | ModelFlags.SHAPE_PROPERTIES
        | ModelFlags.MODEL_PROPERTIES
    )
    value = int(flags)
    return None if value != 0 and value & ~numeric == 0 else {}


def read_notification_plan_field(model, name, snapshot):
    """Share a current read, while each owner retains its own admission check."""
    if snapshot is None:
        return getattr(model, name).numpy()
    if name not in snapshot:
        snapshot[name] = getattr(model, name).numpy()
    return snapshot[name]


class JointWorldOwner:
    """Overlap only raw-ID indexing; keep all public FK/cache publication late."""

    _MODEL_PLAN_FIELDS = (
        "articulation_start",
        "joint_type",
        "joint_parent",
        "joint_child",
        "joint_q_start",
        "joint_qd_start",
        "joint_dof_dim",
        "body_flags",
    )

    def __init__(self, solver, *, host_plan):
        self.solver = solver
        device = solver.model.device
        self.host_plan = host_plan
        self.plan = self.host_plan.device_data(device)
        self.buckets = RawWorldContactBuckets(solver.world_count, solver._max_contacts_alloc, device=device)
        self.stream = wp.Stream(device)
        self.ready, self.done = wp.Event(device), wp.Event(device)
        self.pending = False
        self.output = kuka_joint_world.LightOutput()
        self.output.resolved = solver._resolved_simple_worlds
        self.output.active_worlds = wp.empty(solver.world_count, dtype=int, device=device)
        self.output.active_count = wp.zeros(1, dtype=int, device=device)
        self.output.endpoint_twists = wp.empty(solver.model.body_count, dtype=wp.spatial_vector, device=device)
        self.output.predictor_fallback = wp.empty(solver.world_count, dtype=int, device=device)
        self.output.v_out = solver.v_out
        self.kernel = kuka_joint_world.get_kernel(str(device.arch))
        self.publication = {name: get_publication_kernel(name) for name in PUBLICATION_WIDTH}
        self.model_plan_values = {name: getattr(solver.model, name).numpy().copy() for name in self._MODEL_PLAN_FIELDS}
        from .kuka_active_response import ActiveKukaResponse  # noqa: PLC0415

        self.response = ActiveKukaResponse(solver, self.output.active_worlds, self.output.active_count)

    def validate_notification(self, flags, *, plan_snapshot=None):
        """Keep numeric reset notifications free of static-plan readbacks."""
        self.join_raw()
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
            if not np.array_equal(read_notification_plan_field(self.solver.model, name, plan_snapshot), expected):
                raise RuntimeError("Joint-world static ownership changed; reconstruct the solver and recapture graphs")

    def begin(self, state_in, state_out, contacts, collide_done_event):
        """Start complete ID indexing on a stream dependent on collision completion."""
        self.join_raw()
        solver = self.solver
        if (
            not simple_world._supported(solver)
            or contacts is None
            or contacts.rigid_contact_max != self.buckets.contact_capacity
            or getattr(state_in, "requires_grad", False)
            or getattr(state_out, "requires_grad", False)
        ):
            return False
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            source = getattr(state_in, name, None)
            if source is None:
                continue
            for destination in (state_out.joint_q, state_out.joint_qd):
                if (
                    not source.is_contiguous
                    or not destination.is_contiguous
                    or (
                        source.size
                        and destination.size
                        and source.ptr < destination.ptr + destination.capacity
                        and destination.ptr < source.ptr + source.capacity
                    )
                ):
                    return False
        raw = raw_inputs(solver, contacts)
        if any(
            getattr(raw, name).shape != (self.buckets.contact_capacity,)
            for name in ("shape0", "shape1", "point0", "point1", "normal", "margin0", "margin1")
        ):
            return False
        main = wp.get_stream(solver.model.device)
        main.record_event(self.ready)
        with wp.ScopedStream(self.stream, sync_enter=False):
            self.stream.wait_event(self.ready)
            if collide_done_event is not None:
                self.stream.wait_event(collide_done_event)
            try:
                self.buckets.build(
                    raw.count,
                    raw.shape0,
                    raw.shape1,
                    raw.shape_body,
                    solver.body_to_articulation,
                    solver.art_to_world,
                )
            finally:
                self.stream.record_event(self.done)
                self.pending = True
        return True

    def join_raw(self):
        """Join only a recorded current-call event, including exception recovery."""
        if self.pending:
            wp.get_stream(self.solver.model.device).wait_event(self.done)
            self.pending = False

    def predict_and_classify(self, state_in, state_aug, state_out, contacts, stage3_qd, dt):
        """Replace the predictor and complete ZERO screen after held factors are ready."""
        solver, model = self.solver, self.solver.model
        self.join_raw()
        self.output.active_count.zero_()
        predictor = kuka_joint_world.PredictorData()
        predictor.lower23, predictor.lower6 = solver.L_by_size[23], solver.L_by_size[6]
        predictor.inverse23, predictor.inverse6 = solver.Linv_by_size[23], solver.Linv_by_size[6]
        predictor.tau, predictor.qd, predictor.qdd = state_aug.joint_tau, stage3_qd, state_aug.joint_qdd
        predictor.kinematic_dof, predictor.kinematic_joint = solver._kinematic_dof_mask, solver._kinematic_joint_mask
        predictor.free_root_joints, predictor.joint_qd_start = solver._free_root_joint_indices, model.joint_qd_start
        predictor.body_inertia = model.body_inertia
        predictor.art_group_index = solver.art_group_idx
        predictor.requires_grad = model.requires_grad
        integration = kuka_joint_world.IntegrationData()
        for name in (
            "joint_type",
            "joint_parent",
            "joint_child",
            "joint_q_start",
            "joint_qd_start",
            "joint_dof_dim",
            "body_com",
            "joint_X_c",
        ):
            setattr(integration, name, getattr(model, name))
        integration.kinematic_joint_mask = solver._kinematic_joint_mask
        integration.joint_q, integration.joint_qd = state_in.joint_q, state_in.joint_qd
        integration.joint_q_new, integration.joint_qd_new = state_out.joint_q, state_out.joint_qd
        integration.angular_damping = solver.angular_damping
        wp.launch_tiled(
            self.kernel,
            dim=[solver.world_count],
            inputs=[
                self.plan,
                predictor,
                simple_inputs(solver, state_in, state_aug, dt),
                raw_inputs(solver, contacts),
                self.buckets.data,
                self.output,
                integration,
            ],
            block_dim=32,
            device=model.device,
        )

    def finish_active(self, state_in, state_aug, state_out, dt):
        """Finish the original generalized equations only for unresolved worlds."""
        solver, model = self.solver, self.solver.model
        extra = [self.plan, self.output.active_worlds, self.output.active_count]
        wp.launch(
            self.publication["update_qdd_from_velocity"],
            dim=solver.world_count * 35,
            inputs=[state_in.joint_qd, solver._kinematic_dof_mask, 1.0 / dt, solver.v_out, state_aug.joint_qdd, *extra],
            device=model.device,
        )
        wp.launch(
            self.publication["remove_free_root_transport_from_qdd"],
            dim=solver.world_count * 2,
            inputs=[
                solver._free_root_joint_indices,
                model.joint_qd_start,
                solver._kinematic_joint_mask,
                state_in.joint_qd,
                state_aug.joint_qdd,
                *extra,
            ],
            device=model.device,
        )
        wp.launch(
            self.publication["integrate_generalized_joints"],
            dim=solver.world_count * 32,
            inputs=[
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                solver._kinematic_joint_mask,
                model.joint_dof_dim,
                model.body_com,
                model.joint_X_c,
                state_in.joint_q,
                state_in.joint_qd,
                state_aug.joint_qdd,
                dt,
                solver.angular_damping,
                state_out.joint_q,
                state_out.joint_qd,
                *extra,
            ],
            device=model.device,
        )
        solver._stage7_update_kinematics(state_out, state_aug)
        solver.integrate_particles(model, state_in, state_out, dt)


def create_owner(solver):
    """Keep unsupported recipes on the complete existing implementation."""
    classifier = solver._simple_world_classifier
    if classifier is None or not classifier.enabled or not simple_world._supported(solver):
        return None
    if solver.model.particle_count or not solver._fk_id_cache_enabled or solver.dense_max_constraints != 192:
        return None
    if solver._has_rigid_body_velocity_limits and getattr(solver, "rigid_velocity_limit_slot", None) is None:
        return None
    try:
        host_plan = kuka_joint_world.bind_plan(solver)
    except ValueError:
        return None
    return JointWorldOwner(solver, host_plan=host_plan)
