# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental opt-in complete Kuka owner with live state/cache lifetimes.

This module owns scheduling and transitions, not another dynamics algorithm.
Unsupported cold calls remain original. A change of contract after admission
requires eager held-factor demotion, or recapture before any private writes.
"""

import ast
import copy
import functools
import hashlib
import inspect
import linecache
import math
import textwrap
from types import SimpleNamespace

import numpy as np
import warp as wp

from ...sim import ModelFlags


@wp.kernel
def _prepare_epochs(
    clock: wp.array[wp.int64],
    incoming: wp.array[wp.int64],
    outgoing: wp.array[wp.int64],
    current_generation: wp.array[wp.int64],
    current_valid: wp.array[int],
    geometric_generation: wp.array[wp.int64],
    geometric_valid: wp.array[int],
    held_generation: wp.array[wp.int64],
    canonical_generation: wp.array[wp.int64],
    mass_request: wp.array[int],
    mass_update_mask: wp.array[int],
    global_refresh: int,
    next_refresh: int,
    requested: wp.array[int],
    repair: wp.array[int],
    expected_held: wp.array[wp.int64],
    next_geometry: wp.array[int],
    construct_status: wp.array[int],
    finish_status: wp.array[int],
):
    """Derive each graph replay's epochs from device state, not capture labels."""
    world = wp.tid()
    generation = incoming[world]
    if current_valid[world] == 0 or current_generation[world] != generation:
        # A reset or an original-path write authorizes a new current state,
        # even when its object and generalized coordinates retain their bytes.
        generation = wp.max(clock[world], generation) + wp.int64(1)
        incoming[world] = generation
    following = wp.max(clock[world], generation) + wp.int64(1)
    clock[world] = following
    outgoing[world] = following
    refresh = int(global_refresh != 0 or mass_request[0] != 0)
    requested[world] = refresh
    for articulation in range(3):
        mass_update_mask[world * 3 + articulation] = refresh
    next_geometry[world] = next_refresh
    expected_held[world] = held_generation[world]
    if refresh != 0:
        expected_held[world] = generation
    needs_current = (
        current_valid[world] == 0
        or current_generation[world] != generation
        or canonical_generation[world] != generation
    )
    needs_geometry = refresh != 0 and (geometric_valid[world] == 0 or geometric_generation[world] != generation)
    repair[world] = int(needs_current or needs_geometry)
    construct_status[world] = 0
    finish_status[world] = 0


@wp.kernel
def _invalidate_current(
    mask: wp.array[wp.bool],
    all_worlds: int,
    current_valid: wp.array[int],
    geometric_valid: wp.array[int],
    canonical_generation: wp.array[wp.int64],
):
    """Invalidate authored current state while leaving held mass untouched."""
    world = wp.tid()
    if all_worlds != 0 or mask[world]:
        current_valid[world] = 0
        geometric_valid[world] = 0
        canonical_generation[world] = wp.int64(-1)


@wp.kernel
def _publish_bank_generation(
    generation: wp.array[wp.int64],
    valid: wp.array[int],
    status: wp.array[int],
    canonical_generation: wp.array[wp.int64],
):
    """Tag only the canonical free bank actually published by this producer."""
    world = wp.tid()
    if valid[world] != 0 and status[world] == 0:
        canonical_generation[world] = generation[world]
    else:
        canonical_generation[world] = wp.int64(-1)


@wp.kernel
def _merge_construct_status(construct_status: wp.array[int], refresh_status: wp.array[int]):
    """Retain a cold/repair failure even when the refresh kernel takes reuse."""
    world = wp.tid()
    if construct_status[world] != 0:
        refresh_status[world] = refresh_status[world] | 256


@wp.kernel
def _finish_status(status: wp.array[int], frame_status: wp.array[int]):
    """Keep public force export behind the completed next-state status."""
    world = wp.tid()
    if status[world] != 0:
        wp.atomic_or(frame_status, 0, 256)


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    return (int)(threadIdx.x & 31);
#else
    return 0;
#endif
""")
def _lane() -> int: ...


@wp.func_native(r"""
    if (lane != 0) return;
    status.data[world] = 0;
    if (valid.data[world] == 0) { status.data[world] = 1; return; }
    if (converted.data[world] == generation.data[world]) return;
    const int group = groups.data[world];
    float* l = lower.data + group * 529;
    float* w = inverse.data + group * 529;
    const float* h = augmented.data + world * 180;
    for (int i=0; i<529; ++i) { l[i]=0.0f; w[i]=0.0f; }
    for (int row=0; row<23; ++row) for (int col=0; col<=row; ++col) {
        float value = 0.0f;
        if (row<7) value=h[40+row*(row+1)/2+col];
        else if (col<7) value=h[68+((row-7)/4)*28+col*4+(row-7)%4];
        else if ((row-7)/4==(col-7)/4) {
            const int r=(row-7)%4, c=(col-7)%4;
            value=h[((row-7)/4)*10+r*(r+1)/2+c];
        }
        for (int k=0; k<col; ++k) value-=l[row*23+k]*l[col*23+k];
        if (!wp::isfinite(value) || (row==col && !(value>0.0f))) {
            status.data[world]=2; return;
        }
        l[row*23+col] = row==col ? sqrtf(value) : value/l[col*23+col];
    }
    for (int col=0; col<23; ++col) for (int row=col; row<23; ++row) {
        float value = row==col ? 1.0f : 0.0f;
        for (int k=col; k<row; ++k) value-=l[row*23+k]*w[k*23+col];
        value/=l[row*23+row];
        if (!wp::isfinite(value)) { status.data[world]=3; return; }
        w[row*23+col]=value;
    }
    converted.data[world]=generation.data[world];
""")
def _demote(
    lane: int,
    world: int,
    augmented: wp.array2d[float],
    generation: wp.array[wp.int64],
    valid: wp.array[int],
    groups: wp.array[int],
    lower: wp.array3d[float],
    inverse: wp.array3d[float],
    converted: wp.array[wp.int64],
    status: wp.array[int],
): ...


@functools.cache
def get_demotion_kernel():
    """Materialize original-order factors only from retained augmented180."""

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_live_held_demotion(
        augmented: wp.array2d[float],
        generation: wp.array[wp.int64],
        valid: wp.array[int],
        groups: wp.array[int],
        lower: wp.array3d[float],
        inverse: wp.array3d[float],
        converted: wp.array[wp.int64],
        status: wp.array[int],
    ):
        world, logical_lane = wp.tid()
        lane = _lane()
        if logical_lane != lane:
            return
        _demote(lane, world, augmented, generation, valid, groups, lower, inverse, converted, status)

    return kinetic_live_held_demotion


@functools.cache
def get_repair_kernel(arch):
    """Add a world-uniform entry guard and recover the original constructor AST."""
    from . import kinetic_state  # noqa: PLC0415

    original_kernel = kinetic_state.get_construct_kernel(arch)
    function = ast.parse(textwrap.dedent(inspect.getsource(original_kernel.func))).body[0]
    function.decorator_list = []
    original_ast = ast.dump(function, include_attributes=False)
    original_name = function.name
    function.name += "_live_repair"
    function.args.args.append(ast.arg(arg="repair", annotation=ast.parse("wp.array[int]", mode="eval").body))
    candidates = [
        index
        for index, node in enumerate(function.body)
        if isinstance(node, ast.If)
        and any(isinstance(item, ast.Name) and item.id == "_admit" for item in ast.walk(node.test))
    ]
    if len(candidates) != 1:
        raise RuntimeError("Kinetic constructor admission seam changed")
    index = candidates[0]
    function.body.insert(index, ast.parse("if repair[world] == 0:\n    return\n").body[0])
    recovered = copy.deepcopy(function)
    del recovered.body[index]
    recovered.name = original_name
    recovered.args.args.pop()
    if ast.dump(recovered, include_attributes=False) != original_ast:
        raise RuntimeError("Repair guard altered original constructor arithmetic")
    tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    source = ast.unparse(tree) + "\n"
    filename = "<kinetic-live-repair-" + hashlib.sha256(source.encode()).hexdigest() + ">"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(original_kernel.func.__globals__)
    closure = inspect.getclosurevars(original_kernel.func)
    namespace.update(closure.globals)
    namespace.update(closure.nonlocals)
    exec(compile(source, filename, "exec"), namespace)  # Exact AST recovery is checked above.
    result = wp.kernel(namespace[function.name], module="unique", enable_backward=False)
    result.repair_original_ast = original_ast
    result.repair_recovered_ast = ast.dump(recovered, include_attributes=False)
    return result


def _disjoint_states(state_in, state_out):
    """Follow the existing publication owner's cross-array alias exclusion."""
    sources = [getattr(state_in, name) for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f")]
    outputs = [getattr(state_out, name) for name in ("joint_q", "joint_qd", "body_q", "body_qd")]
    if state_in is state_out or any(value is None or not value.is_contiguous for value in (*sources, *outputs)):
        return False
    return not any(
        source.ptr < target.ptr + target.capacity and target.ptr < source.ptr + source.capacity
        for source in sources
        for target in outputs
        if source.size and target.size
    )


def supported(solver):
    """Admit only the measured fixed recipe, without reading live values back."""
    from . import simple_world, solver_feather_pgs  # noqa: PLC0415

    return (
        simple_world._supported(solver)
        and not solver.model.particle_count
        and solver._fk_id_cache_enabled
        and solver.update_mass_matrix_interval == 2
        and solver._parallel_augmented_drive_topology
        and not solver_feather_pgs._FPGS_CAPTURE
        # Predictor endpoint twists must describe the same v_hat consumed by
        # ZERO. The admitted matrix-free slot recipe bypasses radial clamping.
        and (
            not solver._has_root_free
            or not solver._has_rigid_body_velocity_limits
            or getattr(solver, "rigid_velocity_limit_slot", None) is not None
        )
    )


class KineticWorldOwner:
    """Own one admitted physical call and state-associated device generations."""

    def __init__(self, solver, host_plan):
        from . import kinetic_live_bindings, kinetic_types  # noqa: PLC0415

        self.solver, self.host_plan = solver, host_plan
        self.device, self.worlds = solver.model.device, solver.world_count
        self.bindings = kinetic_live_bindings.LiveBindings(solver, host_plan)
        self.plan = self.bindings.plan
        self.held = kinetic_types.allocate_held(self.worlds, solver.L_by_size[6], solver.Linv_by_size[6], self.device)
        self.clock = wp.zeros(self.worlds, dtype=wp.int64, device=self.device)
        self.canonical_generation = wp.full(self.worlds, -1, dtype=wp.int64, device=self.device)
        self.converted_generation = wp.full(self.worlds, -1, dtype=wp.int64, device=self.device)
        self.demotion_status = wp.zeros(self.worlds, dtype=int, device=self.device)
        self.empty_mask = wp.empty(0, dtype=wp.bool, device=self.device)
        self.states, self.calls = {}, {}
        # Standard State ping-pong needs exactly two state banks and the two
        # directed call banks. Allocate these before capture: reset/notification
        # may access them before the first replay realizes graph allocations.
        self._state_pool = [self._allocate_state_slot() for _ in range(2)]
        self._call_pool = [self._allocate_call_slot() for _ in range(2)]
        self.ever_admitted = False
        self.last_private = False
        self.last_call = None
        self.last_output = None
        self.last_dt = None

    def _allocate_state_slot(self):
        """Own current and geometric storage outside graph capture only."""
        from . import kinetic_types  # noqa: PLC0415

        if self.device.is_capturing:
            raise RuntimeError("Prepare additional kinetic States eagerly and recapture")
        return SimpleNamespace(
            state=None,
            current=kinetic_types.allocate_current(self.worlds, self.device),
            geometric=kinetic_types.allocate_geometric(self.worlds, self.device),
            generation=wp.zeros(self.worlds, dtype=wp.int64, device=self.device),
        )

    def _allocate_call_slot(self):
        """Allocate one directed call's status vectors, never during capture."""
        from . import kinetic_types  # noqa: PLC0415

        if self.device.is_capturing:
            raise RuntimeError("Prepare additional kinetic call directions eagerly and recapture")

        def zeros(dtype=int):
            """Allocate only the original call-owned status or generation vector."""
            return wp.zeros(self.worlds, dtype=dtype, device=self.device)

        schedule, next_schedule = kinetic_types.KineticSchedule(), kinetic_types.KineticSchedule()
        schedule.geometry_requested, next_schedule.geometry_requested = zeros(), zeros()
        schedule.status, next_schedule.status = zeros(), zeros()
        return SimpleNamespace(
            held=self.held,
            expected_held_generation=zeros(wp.int64),
            requested=schedule.geometry_requested,
            repair_requested=zeros(),
            schedule=schedule,
            next_schedule=next_schedule,
            refresh_status=zeros(),
            canonical_generation=self.canonical_generation,
        )

    def _check_slot_capacity(self, state_in, state_out):
        """Reject unseen extra capture inputs before any private array writes."""
        missing = sum(id(state) not in self.states for state in (state_in, state_out))
        missing_call = (id(state_in), id(state_out)) not in self.calls
        if self.device.is_capturing and (missing > len(self._state_pool) or (missing_call and not self._call_pool)):
            raise RuntimeError("Prepare additional kinetic States/call directions eagerly and recapture")

    def _state_slot(self, state):
        """Bind actual State identity to owned storage without capture allocation."""
        key = id(state)
        if key not in self.states:
            slot = self._state_pool.pop() if self._state_pool else self._allocate_state_slot()
            slot.state = state
            self.states[key] = slot
        return self.states[key]

    def _slots(self, state_in, state_out):
        """Share only held/cache-bank ownership; keep each call's statuses distinct."""
        self._check_slot_capacity(state_in, state_out)
        current, future = self._state_slot(state_in), self._state_slot(state_out)
        key = (id(state_in), id(state_out))
        if key not in self.calls:
            slot = self._call_pool.pop() if self._call_pool else self._allocate_call_slot()
            slot.current, slot.geometric = current.current, current.geometric
            slot.next_current, slot.next_geometric = future.current, future.geometric
            slot.state_generation, slot.next_generation = current.generation, future.generation
            slot.schedule.generation, slot.next_schedule.generation = current.generation, future.generation
            self.calls[key] = slot
        return self.calls[key]

    def join(self):
        """Join private velocity writes before reset, notification, or fallback."""
        if self.last_call is not None:
            self.last_call.solve.join()

    def reset(self, state, world_mask=None):
        """Invalidate authored state and the current canonical bank, never held T."""
        self.join()
        if world_mask is not None and (world_mask.shape != (self.worlds,) or world_mask.dtype != wp.bool):
            raise ValueError("Kinetic reset requires the original per-world bool mask")
        # A reset can make any cached alternate state unsuitable for future
        # user-authored stepping. Invalidate the selected worlds in every slot.
        for slot in self.states.values():
            wp.launch(
                _invalidate_current,
                dim=self.worlds,
                inputs=[
                    self.empty_mask if world_mask is None else world_mask,
                    int(world_mask is None),
                    slot.current.valid,
                    slot.geometric.valid,
                    self.canonical_generation,
                ],
                device=self.device,
            )

    def notify_model_changed(self, flags):
        """Reuse the original notification contract and invalidate current caches."""
        self.join()
        self.bindings.validate_notification(flags)
        numeric = (
            ModelFlags.JOINT_PROPERTIES
            | ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.BODY_PROPERTIES
            | ModelFlags.BODY_INERTIAL_PROPERTIES
            | ModelFlags.MODEL_PROPERTIES
        )
        if int(flags) & int(numeric):
            self.reset(None)

    def _demote(self):
        """Restore canonical held factors before an eager original reuse step."""
        self.join()
        if self.device.is_capturing:
            raise RuntimeError("Kinetic owner contract changed during capture; demote eagerly and recapture")
        wp.launch_tiled(
            get_demotion_kernel(),
            dim=[self.worlds],
            block_dim=32,
            inputs=[
                self.held.augmented,
                self.held.generation,
                self.held.valid,
                self.plan.primary_group,
                self.solver.L_by_size[23],
                self.solver.Linv_by_size[23],
                self.converted_generation,
                self.demotion_status,
            ],
            device=self.device,
        )
        if np.any(self.demotion_status.numpy()):
            raise RuntimeError("Cannot demote an invalid held kinetic operator")
        self.solver._fk_id_cache_valid.zero_()
        self.solver._fk_id_cache_source_state = None
        self.reset(None)
        self.held.valid.zero_()
        self.last_private = False
        self.last_output = None

    def try_step(self, state_in, state_out, state_aug, control, contacts, dt, collide_done_event=None):
        """Replace one supported call before any original producer is launched."""
        solver = self.solver
        admissible = (
            supported(solver)
            and contacts is not None
            and contacts.rigid_contact_max == solver._max_contacts_alloc
            and control.joint_target_q is not None
            and control.joint_target_q.shape == (solver.model.joint_dof_count,)
            and math.isfinite(dt)
            and dt > 0.0
            and not state_in.requires_grad
            and not state_out.requires_grad
            and _disjoint_states(state_in, state_out)
        )
        if not admissible:
            if self.last_private:
                self._demote()
            return False
        self._check_slot_capacity(state_in, state_out)
        global_refresh = int(solver._step % 2 == 0 or solver._force_mass_update)
        if not self.last_private and self.ever_admitted and not global_refresh:
            return False
        if not self.ever_admitted and not global_refresh:
            return False
        if self.last_dt is not None and abs(self.last_dt - dt) > 1e-8:
            self.reset(None)
        self.last_dt = dt
        self.join()
        if not self.last_private:
            old = getattr(solver, "_joint_world", None)
            if old is not None:
                old.join_raw()
            for event in getattr(solver, "_memset_done_event", ()) or ():
                if event is not None:
                    wp.get_stream(self.device).wait_event(event)
        slots = self._slots(state_in, state_out)
        call = self.bindings.bind_step(state_in, state_out, control, contacts, dt, slots, state_aug=state_aug)
        self.last_call = call
        wp.launch(
            _prepare_epochs,
            dim=self.worlds,
            inputs=[
                self.clock,
                slots.state_generation,
                slots.next_generation,
                slots.current.generation,
                slots.current.valid,
                slots.geometric.generation,
                slots.geometric.valid,
                self.held.generation,
                self.canonical_generation,
                solver._mass_update_requested,
                solver.mass_update_mask,
                global_refresh,
                int((solver._step + 1) % 2 == 0),
                slots.requested,
                slots.repair_requested,
                slots.expected_held_generation,
                slots.next_schedule.geometry_requested,
                slots.schedule.status,
                slots.next_schedule.status,
            ],
            device=self.device,
        )
        wp.launch_tiled(
            get_repair_kernel(str(self.device.arch)),
            dim=[self.worlds],
            block_dim=32,
            inputs=[
                self.plan,
                call.current_publication,
                slots.schedule,
                slots.current,
                slots.geometric,
                slots.repair_requested,
            ],
            device=self.device,
        )
        wp.launch(
            _publish_bank_generation,
            dim=self.worlds,
            inputs=[slots.state_generation, slots.current.valid, slots.schedule.status, self.canonical_generation],
            device=self.device,
        )
        call.refresh_launch()
        call.free.launch()
        wp.launch(
            _merge_construct_status,
            dim=self.worlds,
            inputs=[slots.schedule.status, slots.refresh_status],
            device=self.device,
        )
        solver._mass_update_requested.zero_()
        solver._mass_update_global_flag = bool(global_refresh)
        solver._force_mass_update = False
        call.predict_launch()
        solver._clamp_rigid_velocity_limits(solver.v_hat)
        wp.copy(solver._debug_stage3_qd_work, state_in.joint_qd)
        wp.copy(solver._debug_stage3_joint_qdd, state_aug.joint_qdd)
        wp.copy(solver._debug_stage3_v_hat, solver.v_hat)
        if collide_done_event is not None:
            wp.get_stream(self.device).wait_event(collide_done_event)
        call.zero.launch()
        solve = call.solve
        try:
            solve.setup()
            solve.allocate()
            solve.build_rows()
            solve.prepare_mf()
            solve.qualify()
            solve.materialize()
            solve.solve()
        finally:
            solve.join()
        call.finish_launch()
        wp.launch(
            _publish_bank_generation,
            dim=self.worlds,
            inputs=[
                slots.next_generation,
                slots.next_current.valid,
                slots.next_schedule.status,
                self.canonical_generation,
            ],
            device=self.device,
        )
        wp.launch(
            _finish_status,
            dim=self.worlds,
            inputs=[slots.next_schedule.status, solve.guard.frame_status],
            device=self.device,
        )
        self.ever_admitted = self.last_private = True
        self.last_output = state_out
        solver._joint_world_active = False
        solver._fk_id_cache_source_state = state_out
        return True

    def update_contacts(self, contacts):
        """Export only when the real public API calls, never by a baked phase."""
        if not self.last_private:
            return False
        if contacts is not self.last_call.contacts:
            raise ValueError("Kinetic contact export must use the most recent solved Contacts")
        self.join()
        self.last_call.force.launch()
        return True


def create_owner(solver):
    """Keep unsupported constructors on the unchanged original solver."""
    if not supported(solver):
        return None
    from . import kinetic_live_plan  # noqa: PLC0415

    try:
        host_plan = kinetic_live_plan.build_plan(solver)
    except ValueError:
        return None
    return KineticWorldOwner(solver, host_plan)
