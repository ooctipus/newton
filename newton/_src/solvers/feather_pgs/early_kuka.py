# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Default-off scheduling of unchanged publication for current ZERO worlds."""

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
    resolved: wp.array[int]
    art_mask: wp.array[int]
    early_arts: wp.array[int]
    late_arts: wp.array[int]
    counts: wp.array[int]
    art_world: wp.array[int]
    joint_art: wp.array[int]
    dof_art: wp.array[int]
    body_art: wp.array[int]
    art_start: wp.array[int]
    art_end: wp.array[int]
    dof_start: wp.array[int]
    dof_end: wp.array[int]
    joint_child: wp.array[int]
    max_joints: int
    max_dofs: int


@wp.kernel
def compact_publication(cohort: CohortData):
    """Partition every articulation using only the actual current ZERO decision."""
    art = wp.tid()
    world = cohort.art_world[art]
    selected = int(0)
    if world >= 0 and world < cohort.resolved.shape[0]:
        selected = int(cohort.resolved[world] == 1)
    cohort.art_mask[art] = selected
    if selected != 0:
        slot = wp.atomic_add(cohort.counts, 0, 1)
        cohort.early_arts[slot] = art
    else:
        slot = wp.atomic_add(cohort.counts, 1, 1)
        cohort.late_arts[slot] = art


PUBLICATION = {
    "update_qdd_from_velocity": "dof",
    "remove_free_root_transport_from_qdd": "root",
    "integrate_generalized_joints": "joint",
    "eval_rigid_fk_kinematics": "art",
    "finalize_body_dynamics": "body",
}


def publication_source(name: str, early: bool) -> str:
    """Change only index ownership, keeping every original publication statement."""
    kind = PUBLICATION[name]
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(kernels, name).func)))
    function = tree.body[0]
    function.name = f"{'early' if early else 'late'}_kuka_{name}"
    function.decorator_list = ast.parse('@wp.kernel(module="unique")\ndef kernel():\n    pass').body[0].decorator_list
    function.args.args.append(ast.arg(arg="cohort", annotation=ast.Name(id="CohortData", ctx=ast.Load())))

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
    if kind == "art":
        which, array = (0, "early_arts") if early else (1, "late_arts")
        prelude = f"""
slot = wp.tid()
if slot >= cohort.counts[{which}]:
    return
mapped_index = cohort.{array}[slot]
"""
    elif kind == "root":
        comparison = "== 0" if early else "!= 0"
        prelude = f"""
mapped_index = wp.tid()
joint = free_root_joint_indices[mapped_index]
art = cohort.joint_art[joint]
selected = int(0)
if art >= 0:
    selected = cohort.art_mask[art]
if selected {comparison}:
    return
"""
    elif early:
        width = "cohort.max_dofs" if kind == "dof" else "cohort.max_joints"
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
            prelude += "if mapped_index >= cohort.dof_end[art]:\n    return\n"
        else:
            prelude += "mapped_index = cohort.art_start[art] + offset\n"
            prelude += "if mapped_index >= cohort.art_end[art]:\n    return\n"
            if kind == "body":
                prelude += "mapped_index = cohort.joint_child[mapped_index]\n"
    else:
        prelude = f"""
mapped_index = wp.tid()
art = cohort.{kind}_art[mapped_index]
if art >= 0 and cohort.art_mask[art] != 0:
    return
"""
    function.body = ast.parse(prelude).body + function.body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


@functools.cache
def get_publication_kernel(name: str, early: bool):
    """Compile unchanged equations with count-guarded current index ownership."""
    source = publication_source(name, early)
    filename = f"<early-kuka-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(kernels.__dict__)
    namespace.update(CohortData=CohortData)
    exec(compile(source, filename, "exec"), namespace)
    return namespace[f"{'early' if early else 'late'}_kuka_{name}"]


def allocate_cohort(solver):
    """Validate static ownership and allocate only constructor-owned bounded lists."""
    model, plan = solver.model, solver._model_plan
    data = CohortData()
    arts = model.articulation_count
    data.resolved = solver._resolved_simple_worlds
    data.art_world, data.body_art = solver.art_to_world, solver.body_to_articulation
    data.art_start, data.art_end = model.articulation_start, solver.articulation_joint_end
    data.dof_start, data.joint_child = solver.articulation_dof_start, model.joint_child
    starts, ends = data.art_start.numpy(), data.art_end.numpy()
    dofs, children = data.dof_start.numpy(), data.joint_child.numpy()
    dof_ends = dofs + np.asarray(plan.articulation_dof_count, dtype=np.int32)
    parents, body_art = model.joint_parent.numpy(), data.body_art.numpy()
    body_counts = np.bincount(body_art[body_art >= 0], minlength=arts)
    worlds = data.art_world.numpy()
    joint_art = np.full(model.joint_count, -1, np.int32)
    dof_art = np.full(model.joint_dof_count, -1, np.int32)
    if not np.all((worlds >= -1) & (worlds < solver.world_count)):
        raise ValueError("Early ZERO publication requires bounded articulation worlds")
    for art in range(arts):
        start, end = int(starts[art]), int(ends[art])
        first, last = int(dofs[art]), int(dof_ends[art])
        if not (0 <= start <= end <= starts[art + 1] <= model.joint_count):
            raise ValueError("Early ZERO publication requires disjoint valid articulation joints")
        if not (0 <= first <= last <= model.joint_dof_count) or np.any(dof_art[first:last] >= 0):
            raise ValueError("Early ZERO publication requires disjoint actual global DOFs")
        local_children = children[start:end]
        if np.any(local_children < 0) or len(np.unique(local_children)) != len(local_children):
            raise ValueError("Early ZERO publication requires one distinct body per joint")
        if not np.all(body_art[local_children] == art) or len(local_children) != body_counts[art]:
            raise ValueError("Early ZERO publication must cover each articulation body exactly once")
        local_parents = parents[start:end]
        if np.any(body_art[local_parents[local_parents >= 0]] != art):
            raise ValueError("Early ZERO publication requires articulation-local parent bodies")
        joint_art[start:end], dof_art[first:last] = art, art
    data.max_joints = max(1, int(np.max(ends - starts[:-1], initial=0)))
    data.max_dofs = max(1, int(np.max(dof_ends - dofs, initial=0)))
    for name in ("art_mask", "early_arts", "late_arts"):
        setattr(data, name, wp.zeros(arts, dtype=int, device=model.device))
    data.counts = wp.zeros(2, dtype=int, device=model.device)
    data.dof_end = wp.array(dof_ends, dtype=int, device=model.device)
    data.joint_art = wp.array(joint_art, dtype=int, device=model.device)
    data.dof_art = wp.array(dof_art, dtype=int, device=model.device)
    return data


class EarlyKuka:
    """Own one early stream; delay inertia writes until the original MF reader ends."""

    def __init__(self, solver):
        self.solver = solver
        self.data = allocate_cohort(solver)
        device = solver.model.device
        self.stream = wp.Stream(device)
        self.ready, self.reader_done, self.done = wp.Event(device), wp.Event(device), wp.Event(device)
        self.active, self.finalized = False, False
        self.current = None
        self.kernels = {
            (name, early): get_publication_kernel(name, early) for name in PUBLICATION for early in (False, True)
        }

    def begin(self, state_in, state_out):
        """Reject reentrancy and aliased outputs before classifier or cache writes."""
        if self.active:
            raise RuntimeError("Previous early ZERO publication did not join")
        fields = ("joint_q", "joint_qd", "body_q", "body_qd")
        for source_name in fields:
            source = getattr(state_in, source_name)
            for target_name in fields:
                target = getattr(state_out, target_name)
                if (
                    source.size
                    and target.size
                    and source.ptr < target.ptr + target.capacity
                    and target.ptr < source.ptr + source.capacity
                ):
                    raise ValueError("Early ZERO publication requires disjoint input/output states")
        self.current = None
        self.finalized = False

    def launch(self, state_in, state_aug, state_out, dt):
        """Fork original qdd/transport/integration/FK after the authoritative classifier."""
        solver = self.solver
        if solver._resolved_simple_worlds is not self.data.resolved:
            raise RuntimeError("Early ZERO decision owner changed")
        self.current = (state_in, state_aug, state_out, dt)
        self.data.counts.zero_()
        wp.launch(
            compact_publication, dim=solver.model.articulation_count, inputs=[self.data], device=solver.model.device
        )
        wp.copy(solver.v_out, solver.v_hat)
        wp.get_stream(solver.model.device).record_event(self.ready)
        self.stream.wait_event(self.ready)
        with wp.ScopedStream(self.stream, sync_enter=False, sync_exit=False):
            self.publish(*self.current, early=True, finalize=False)
        self.active = True

    def after_mf_inverse(self):
        """Queue finalization only after the original unmasked body-inertia reader."""
        if not self.active:
            return
        if self.finalized:
            raise RuntimeError("MF inverse completion was published twice")
        wp.get_stream(self.solver.model.device).record_event(self.reader_done)
        self.stream.wait_event(self.reader_done)
        with wp.ScopedStream(self.stream, sync_enter=False, sync_exit=False):
            self.publish(*self.current, early=True, finalize=True)
            self.stream.record_event(self.done)
        self.finalized = True

    def publish(self, state_in, state_aug, state_out, dt, *, early, finalize):
        """Run original equations on a disjoint current partition, without changing epochs."""
        solver, data, model = self.solver, self.data, self.solver.model
        arts = model.articulation_count

        def launch(name, dim, operands, block=256):
            wp.launch(
                self.kernels[name, early], dim=dim, inputs=[*operands, data], block_dim=block, device=model.device
            )

        if not finalize:
            launch(
                "update_qdd_from_velocity",
                arts * data.max_dofs if early else model.joint_dof_count,
                [state_in.joint_qd, solver._kinematic_dof_mask, 1.0 / dt, solver.v_out, state_aug.joint_qdd],
            )
            if solver._free_root_joint_count:
                launch(
                    "remove_free_root_transport_from_qdd",
                    solver._free_root_joint_count,
                    [
                        solver._free_root_joint_indices,
                        model.joint_qd_start,
                        solver._kinematic_joint_mask,
                        state_in.joint_qd,
                        state_aug.joint_qdd,
                    ],
                )
            launch(
                "integrate_generalized_joints",
                arts * data.max_joints if early else model.joint_count,
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
                arts,
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
        else:
            refresh = ((solver._step + 1) % solver.update_mass_matrix_interval) == 0
            compact = refresh and solver._global_inertia_stream is not None
            launch(
                "finalize_body_dynamics",
                arts * data.max_joints if early else model.body_count,
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

    def finish(self, state_in, state_aug, state_out, dt):
        """Publish the late complement and join before declaring whole-state cache readiness."""
        if not self.active or not self.finalized:
            raise RuntimeError("Early ZERO publication lacks its original MF reader event")
        self.publish(state_in, state_aug, state_out, dt, early=False, finalize=False)
        self.publish(state_in, state_aug, state_out, dt, early=False, finalize=True)
        wp.get_stream(self.solver.model.device).wait_event(self.done)
        self.solver._fk_id_cache_source_state = state_out
        # The main stream now owns completion. No stale capture/eager event wait
        # crosses into the next call; all within-step event records precede waits.
        self.active = False
        self.current = None

    def wait(self):
        """Reject a mid-step reset/notification instead of mutating live early inputs."""
        if self.active:
            raise RuntimeError("Reset/notification during active early ZERO publication")


def create_owner(solver):
    """Admit only the existing CUDA Kuka ZERO law with direct original cache publication."""
    classifier = solver._simple_world_classifier
    if (
        classifier is None
        or not classifier.enabled
        or solver._row_packets is not None
        or not solver._fk_id_cache_enabled
        or solver._fk_id_cache is not None
        or solver._fused_k1
        or solver.lazy_kinematics
        or solver._grouped_topology is not None
        or solver._prismatic_publication is not None
        or solver.model.particle_count
        or solver._propagation_contacts_enabled()
        or not solver._has_free_rigid_bodies
        or solver.rigid_velocity_limit_slot is None
    ):
        return None
    return EarlyKuka(solver)
