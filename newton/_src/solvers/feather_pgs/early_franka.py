# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Default-off contactless primary9 completion with disjoint publication."""

import ast
import functools
import hashlib
import inspect
import linecache
import textwrap

import numpy as np
import warp as wp

from . import kernels


@wp.struct
class CohortData:
    incidence: wp.array[int]
    invalid_raw: wp.array[int]
    early_owner: wp.array[int]
    early_count: wp.array[int]
    zero_mf: wp.array[int]
    late_owner: wp.array[int]
    art_mask: wp.array[int]
    early_arts: wp.array[int]
    late_arts: wp.array[int]
    counts: wp.array[int]
    art_world: wp.array[int]
    primary: wp.array[int]
    joint_art: wp.array[int]
    dof_art: wp.array[int]
    body_art: wp.array[int]
    art_start: wp.array[int]
    art_end: wp.array[int]
    dof_start: wp.array[int]
    joint_child: wp.array[int]
    max_joints: int


@wp.kernel
def mark_raw_incidence(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    shape_body: wp.array[int],
    workers: int,
    cohort: CohortData,
):
    """Exclude both raw endpoints, including contacts rejected by allocation."""
    raw_count = count[0]
    capacity = wp.min(shape0.shape[0], shape1.shape[0])
    if raw_count < 0 or raw_count > capacity:
        wp.atomic_max(cohort.invalid_raw, 0, 1)
    raw = wp.tid()
    while raw < wp.min(raw_count, capacity):
        for endpoint in range(2):
            shape = shape0[raw]
            if endpoint == 1:
                shape = shape1[raw]
            if shape >= 0:
                if shape >= shape_body.shape[0]:
                    wp.atomic_max(cohort.invalid_raw, 0, 1)
                else:
                    body = shape_body[shape]
                    if body >= 0:
                        if body >= cohort.body_art.shape[0]:
                            wp.atomic_max(cohort.invalid_raw, 0, 1)
                        else:
                            art = cohort.body_art[body]
                            if art >= cohort.art_world.shape[0]:
                                wp.atomic_max(cohort.invalid_raw, 0, 1)
                            elif art >= 0:
                                world = cohort.art_world[art]
                                if world >= 0 and world < cohort.incidence.shape[0]:
                                    if art == cohort.primary[world]:
                                        wp.atomic_max(cohort.incidence, world, 1)
                                else:
                                    wp.atomic_max(cohort.invalid_raw, 0, 1)
        raw += workers


@wp.kernel
def classify_early(
    slot_counter: wp.array[int],
    bounds: wp.array2d[int],
    mf_slot_counter: wp.array[int],
    cohort: CohortData,
):
    """Publish independent current solve views without touching late queue data."""
    world = wp.tid()
    rows = slot_counter[world]
    art = cohort.primary[world]
    selected = (
        art >= 0
        and rows > 0
        and rows <= 9
        and rows == bounds[world, 1]
        and mf_slot_counter[world] == 0
        and cohort.incidence[world] == 0
        and cohort.invalid_raw[0] == 0
    )
    cohort.early_owner[world] = int(selected)
    cohort.early_count[world] = wp.where(selected, rows, 0)
    if art >= 0:
        cohort.art_mask[art] = int(selected)


@wp.kernel
def compact_publication(cohort: CohortData):
    """Build only valid articulation IDs and count-guard all unused list tails."""
    art = wp.tid()
    if cohort.art_mask[art] != 0:
        slot = wp.atomic_add(cohort.counts, 0, 1)
        cohort.early_arts[slot] = art
    else:
        slot = wp.atomic_add(cohort.counts, 1, 1)
        cohort.late_arts[slot] = art


@wp.kernel
def prepare_late_owner(owner: wp.array[int], cohort: CohortData):
    """Retain canonical owners while removing early worlds from the late SINGLE view."""
    world = wp.tid()
    cohort.late_owner[world] = wp.where(cohort.early_owner[world] != 0, 0, owner[world])


def publication_source(name: str, early: bool) -> str:
    """Adapt only the original kernel's index selection, before any source access."""
    allowed = {
        "update_qdd_from_velocity": "dof",
        "integrate_generalized_joints": "joint",
        "eval_rigid_fk_kinematics": "art",
        "finalize_body_dynamics": "body",
        "prepare_world_impulses": "world",
    }
    kind = allowed[name]
    original = getattr(kernels, name)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original.func)))
    function = tree.body[0]
    function.name = f"{'early' if early else 'late'}_{name}"
    function.decorator_list = (
        ast.parse('@wp.kernel(module="unique")\ndef placeholder():\n    pass').body[0].decorator_list
    )
    function.args.args.append(ast.arg(arg="cohort", annotation=ast.Name(id="CohortData", ctx=ast.Load())))

    class Index(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "wp" and node.func.attr == "tid":
                    return ast.copy_location(ast.Name(id="mapped_index", ctx=ast.Load()), node)
            return self.generic_visit(node)

    function = Index().visit(function)
    if kind == "art":
        which = 0 if early else 1
        array = "early_arts" if early else "late_arts"
        prelude = f"""
slot = wp.tid()
if slot >= cohort.counts[{which}]:
    return
mapped_index = cohort.{array}[slot]
"""
    elif early:
        if kind == "world":
            raise ValueError("Prefix producer already initializes early impulses")
        width = "9" if kind == "dof" else "cohort.max_joints"
        prelude = f"""
linear = wp.tid()
slot = linear // {width}
if slot >= cohort.counts[0]:
    return
art = cohort.early_arts[slot]
offset = linear % {width}
"""
        if kind == "dof":
            prelude += "mapped_index = cohort.dof_start[art] + offset\n"
        else:
            prelude += "mapped_index = cohort.art_start[art] + offset\n"
            prelude += "if mapped_index >= cohort.art_end[art]:\n    return\n"
            if kind == "body":
                prelude += "mapped_index = cohort.joint_child[mapped_index]\nif mapped_index < 0:\n    return\n"
    else:
        prelude = "mapped_index = wp.tid()\n"
        if kind == "world":
            prelude += "if cohort.early_owner[mapped_index] != 0:\n    return\n"
        else:
            prelude += f"art = cohort.{kind}_art[mapped_index]\n"
            prelude += "if art >= 0 and cohort.art_mask[art] != 0:\n    return\n"
    function.body = ast.parse(prelude).body + function.body
    tree.body = [function]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


@functools.cache
def get_publication_kernel(name: str, early: bool):
    """Return the original equations with the explicitly selected index mapping."""
    source = publication_source(name, early)
    filename = f"<early-franka-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(kernels.__dict__)
    namespace.update(CohortData=CohortData)
    exec(compile(source, filename, "exec"), namespace)
    return namespace[f"{'early' if early else 'late'}_{name}"]


class EarlyFranka:
    """Own persistent cohort storage, stream events and original publication adapters."""

    def __init__(self, solver):
        self.solver = solver
        model = solver.model
        self.data = data = CohortData()
        world_count, art_count = solver.world_count, model.articulation_count
        for name in ("incidence", "early_owner", "early_count", "zero_mf", "late_owner", "early_arts"):
            setattr(data, name, wp.zeros(world_count, dtype=int, device=model.device))
        data.invalid_raw = wp.zeros(1, dtype=int, device=model.device)
        data.counts = wp.zeros(2, dtype=int, device=model.device)
        data.art_mask = wp.zeros(art_count, dtype=int, device=model.device)
        data.late_arts = wp.zeros(art_count, dtype=int, device=model.device)
        data.art_world, data.primary = solver.art_to_world, solver._local_primary_articulation
        data.body_art = solver.body_to_articulation
        data.art_start, data.art_end = model.articulation_start, solver.articulation_joint_end
        data.dof_start, data.joint_child = solver.articulation_dof_start, model.joint_child
        starts, ends = data.art_start.numpy(), data.art_end.numpy()
        dofs, children = data.dof_start.numpy(), data.joint_child.numpy()
        dof_ends = np.r_[dofs[1:], model.joint_dof_count]
        parents, body_art = model.joint_parent.numpy(), data.body_art.numpy()
        body_counts = np.bincount(body_art[body_art >= 0], minlength=art_count)
        joint_dims = model.joint_dof_dim.numpy().sum(axis=1)
        primary = data.primary.numpy()
        dof_art = np.full(model.joint_dof_count, -1, dtype=np.int32)
        for art in range(art_count):
            dof_art[dofs[art] : dof_ends[art]] = art
        for art in primary:
            local_children = children[starts[art] : ends[art]]
            if np.any(local_children < 0) or len(np.unique(local_children)) != len(local_children):
                raise ValueError("Early publication requires one distinct body per primary tree joint")
            if len(local_children) != body_counts[art] or np.any(body_art[local_children] != art):
                raise ValueError("Early publication must cover every primary body exactly once")
            if np.any(joint_dims[starts[art] : ends[art]] > 1):
                raise ValueError("Early publication requires fixed-base zero/one-DOF primary joints")
            local_parents = parents[starts[art] : ends[art]]
            if np.any(body_art[local_parents[local_parents >= 0]] != art):
                raise ValueError("Early publication requires articulation-local parent bodies")
            if dof_ends[art] - dofs[art] != 9:
                raise ValueError("Early publication requires exactly nine primary coordinates")
        data.max_joints = int(np.max(ends[primary] - starts[primary]))
        data.joint_art = model.joint_articulation
        data.dof_art = wp.array(dof_art, dtype=int, device=model.device)
        self.stream = wp.Stream(model.device)
        self.ready, self.done = wp.Event(model.device), wp.Event(model.device)
        self.active = False
        self.state_out = None
        # Force all factories before graph capture; no per-step compilation or allocation.
        self.kernels = {
            (name, early): get_publication_kernel(name, early)
            for name in (
                "update_qdd_from_velocity",
                "integrate_generalized_joints",
                "eval_rigid_fk_kinematics",
                "finalize_body_dynamics",
            )
            for early in (False, True)
        }
        self.clear_kernel = get_publication_kernel("prepare_world_impulses", False)

    def begin(self, state_in, state_out):
        """Reject aliased states before any work and reset the host launch lifecycle."""
        self.wait()
        names = ("joint_q", "joint_qd", "body_q", "body_qd")
        for source_name in names:
            source = getattr(state_in, source_name)
            for target_name in names:
                target = getattr(state_out, target_name)
                if source.size and target.size:
                    if source.ptr < target.ptr + target.capacity and target.ptr < source.ptr + source.capacity:
                        raise ValueError("Early Franka completion requires non-overlapping input/output states")
        self.active = False
        self.state_out = state_out

    def launch(self, state_in, state_aug, contacts, dt):
        """Fork only after current allocation, with private solve counts and frozen incident velocity."""
        solver, data = self.solver, self.data
        data.incidence.zero_()
        data.invalid_raw.zero_()
        data.counts.zero_()
        # Every primary entry is overwritten by classify_early; nonprimary entries
        # are immutable zero after construction.
        if (
            contacts is not None
            and getattr(contacts, "rigid_contact_count", None) is not None
            and contacts.rigid_contact_max > 0
        ):
            workers = min(contacts.rigid_contact_max, solver.world_count * 4)
            wp.launch(
                mark_raw_incidence,
                dim=workers,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    solver.model.shape_body,
                    workers,
                    data,
                ],
                device=solver.model.device,
            )
        wp.launch(
            classify_early,
            dim=solver.world_count,
            inputs=[
                solver.slot_counter,
                solver.dense_phase_bounds,
                solver.mf_slot_counter,
                data,
            ],
            device=solver.model.device,
        )
        wp.launch(compact_publication, dim=solver.model.articulation_count, inputs=[data], device=solver.model.device)
        wp.copy(solver.v_out, solver.v_hat)
        wp.get_stream(solver.model.device).record_event(self.ready)
        self.stream.wait_event(self.ready)
        self.active = True
        with wp.ScopedStream(self.stream, sync_enter=False, sync_exit=False):
            try:
                solver._launch_local_internal_solve(
                    solver.rhs,
                    solver.pgs_iterations,
                    solver.pgs_omega,
                    solver._contact_friction_start_iteration(solver.pgs_iterations),
                    0,
                    solve_pair=False,
                    solve_residual=False,
                    owner_view=data.early_owner,
                    count_view=data.early_count,
                    mf_count_view=data.zero_mf,
                )
                self.publish(state_in, state_aug, self.state_out, dt, early=True)
            finally:
                self.stream.record_event(self.done)

    def publish(self, state_in, state_aug, state_out, dt, *, early):
        """Run unchanged qdd/integration/FK/finalization equations for one disjoint partition."""
        solver, data, model = self.solver, self.data, self.solver.model

        def launch(name, dim, operands, block=256):
            wp.launch(
                self.kernels[name, early], dim=dim, inputs=[*operands, data], block_dim=block, device=model.device
            )

        launch(
            "update_qdd_from_velocity",
            solver.world_count * 9 if early else model.joint_dof_count,
            [state_in.joint_qd, solver._kinematic_dof_mask, 1.0 / dt, solver.v_out, state_aug.joint_qdd],
        )
        if not early:
            solver._remove_free_root_transport(state_in, state_aug)
        launch(
            "integrate_generalized_joints",
            solver.world_count * data.max_joints if early else model.joint_count,
            [
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
            ],
        )
        launch(
            "eval_rigid_fk_kinematics",
            solver.world_count if early else model.articulation_count,
            [
                model.articulation_start,
                solver.articulation_joint_end,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                state_out.joint_q,
                state_out.joint_qd,
                model.joint_X_p,
                model.joint_X_c,
                solver.body_X_com,
                model.joint_axis,
                model.joint_dof_dim,
                model.body_com,
                state_out.body_q,
                state_aug.body_q_com,
                solver.articulation_origin,
                state_aug.joint_S_s,
                state_aug.body_v_s,
                state_aug.body_a_s,
                solver._fk_id_cache_valid,
            ],
            16,
        )
        refresh = ((solver._step + 1) % solver.update_mass_matrix_interval) == 0
        compact = refresh and solver._global_inertia_stream is not None
        launch(
            "finalize_body_dynamics",
            solver.world_count * data.max_joints if early else model.body_count,
            [
                solver.body_to_articulation,
                state_out.body_q,
                state_aug.body_q_com,
                model.body_com,
                model.body_mass,
                model.body_inertia,
                solver.is_free_rigid,
                solver.articulation_origin,
                int(refresh and not compact),
                int(compact),
                model.gravity,
                state_aug.body_v_s,
                state_aug.body_a_s,
                state_aug.body_I_s,
                solver._body_inertia_terms,
                state_aug.body_f_s,
                state_out.body_qd,
            ],
            128,
        )

    def join(self, state_out):
        """Publish whole-state cache readiness only after both partitions complete."""
        wp.get_stream(self.solver.model.device).wait_event(self.done)
        self.solver._fk_id_cache_source_state = state_out
        self.active = False

    def wait(self):
        """Keep reset/notification and subsequent calls behind any prior early work."""
        if self.active:
            wp.get_stream(self.solver.model.device).wait_event(self.done)


def create_owner(solver):
    """Require the existing packet law and direct cache publication, otherwise stay late."""
    if (
        solver._row_packets is None
        or not solver._fk_id_cache_enabled
        or solver._fk_id_cache is not None
        or solver._fused_k1
        or solver.lazy_kinematics
        or solver._grouped_topology is not None
        or solver._prismatic_publication is not None
        or solver.model.particle_count
        or solver._propagation_contacts_enabled()
    ):
        return None
    try:
        return EarlyFranka(solver)
    except ValueError:
        # Unsupported topology retains the complete original late path.
        return None
