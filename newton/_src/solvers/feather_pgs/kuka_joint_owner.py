# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private lifecycle for the default-off joint-world Kuka experiment."""

import ast
import functools
import hashlib
import inspect
import linecache
import textwrap

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


def publication_source(name: str) -> str:
    """Retain original generalized equations and change only active index ownership."""
    width, field = PUBLICATION_WIDTH[name]
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(kernels, name).func)))
    function = tree.body[0]
    function.name = f"active_joint_world_{name}"
    function.decorator_list = ast.parse('@wp.kernel(module="unique")\ndef kernel():\n    pass').body[0].decorator_list
    extras = (
        ast.parse(
            "def extra(plan: JointWorldPlan, active_worlds: wp.array[int], active_count: wp.array[int]):\n    pass"
        )
        .body[0]
        .args.args
    )
    function.args.args.extend(extras)

    class Index(ast.NodeTransformer):
        def visit_Call(self, node):
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "wp"
                and node.func.attr == "tid"
            ):
                return ast.copy_location(ast.Name(id="mapped_index", ctx=ast.Load()), node)
            return self.generic_visit(node)

    function = Index().visit(function)
    function.body = (
        ast.parse(
            f"linear = wp.tid()\nslot = linear // {width}\n"
            "if slot >= active_count[0]:\n    return\n"
            f"world = active_worlds[slot]\nmapped_index = plan.{field}[world, linear % {width}]\n"
        ).body
        + function.body
    )
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


@functools.cache
def get_publication_kernel(name: str):
    """Compile the original operation with a compact current-world prefix."""
    source = publication_source(name)
    filename = f"<active-joint-world-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(kernels.__dict__)
    namespace["JointWorldPlan"] = kuka_joint_world.JointWorldPlan
    exec(compile(source, filename, "exec"), namespace)
    return namespace[f"active_joint_world_{name}"]


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

    def validate_notification(self, flags):
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
            if not np.array_equal(getattr(self.solver.model, name).numpy(), expected):
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
