# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental streaming lifetime correction for the complete world-lane owner.

One fixed warp owns 32 worlds. Own-body terms and current screws are lane-private
shared fields, not another global dynamics representation. Primary public body
state is cooperatively scattered only in uniform finish kernels. Masked repair
and the two original free-root helpers retain direct canonical publication.
"""

import ast
import functools
import hashlib
import re

import warp as wp

from . import world_lane_state as original

KineticPlan = original.KineticPlan
KineticData = original.KineticData
PublicationData = original.PublicationData


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    __shared__ float values[257*32];
    return reinterpret_cast<uint64_t>(values);
#else
    return reinterpret_cast<uint64_t>(malloc(257*32*sizeof(float)));
#endif
""")
def _storage_refresh() -> wp.uint64: ...


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    __shared__ float values[127*32];
    return reinterpret_cast<uint64_t>(values);
#else
    return reinterpret_cast<uint64_t>(malloc(127*32*sizeof(float)));
#endif
""")
def _storage_held() -> wp.uint64: ...


@wp.func_native(r"""
#if !defined(__CUDA_ARCH__)
    free(reinterpret_cast<void*>(address));
#endif
""")
def _release(address: wp.uint64): ...


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31;
#else
    const int lane=0;
#endif
    // The explicit lifetime boundary must survive compiler store forwarding,
    // including masked repair, where no cross-lane barrier is appropriate.
    reinterpret_cast<volatile float*>(address)[field*32+lane]=value;
""")
def _store(address: wp.uint64, field: int, value: float): ...


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31;
#else
    const int lane=0;
#endif
    return reinterpret_cast<volatile float*>(address)[field*32+lane];
""")
def _load(address: wp.uint64, field: int) -> float: ...


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31;
    float* tile=reinterpret_cast<float*>(address)+offset*32;
    // AoS shared staging has conflict-free scalar writes (stride 13) and
    // contiguous cooperative reads. No adjacent-world body-ID assumption.
    for(int k=0;k<7;++k)tile[13*lane+k]=pose[k];
    for(int k=0;k<6;++k)tile[13*lane+7+k]=velocity[k];
    __syncwarp(0xffffffffu);
    const int base=world-lane;
    for(int item=lane;item<13*32;item+=32) {
        const int source=base+item/13, component=item%13;
        if(source<plan.body_ids.shape[1]) {
            const int body=plan.body_ids.data[local*plan.body_ids.shape[1]+source];
            if(component<7)
                reinterpret_cast<float*>(data.body_q.data)[7*body+component]=tile[item];
            else
                reinterpret_cast<float*>(data.body_qd.data)[6*body+component-7]=tile[item];
        }
    }
    __syncwarp(0xffffffffu);
#else
    if(world<plan.body_ids.shape[1]) {
        const int body=plan.body_ids.data[local*plan.body_ids.shape[1]+world];
        data.body_q.data[body]=pose;
        data.body_qd.data[body]=velocity;
    }
#endif
""")
def _publish(
    address: wp.uint64,
    offset: int,
    world: int,
    local: int,
    plan: KineticPlan,
    data: PublicationData,
    pose: wp.transform,
    velocity: wp.spatial_vector,
): ...


def _statements(source):
    return ast.parse(source).body


def _producer_source(schedule, *, finish, refresh, name):
    """Reorder the original typed algebra, preserving its per-body physical law."""
    parent, _, dofs, roots = schedule
    count, width = roots[1] - 1, 19 if refresh else 6
    axis_base = count * width
    output_base = axis_base + 6 * sum(d >= 0 for d in dofs)
    tree = ast.parse(original._producer_source(schedule, finish=finish, refresh=refresh, name=name))
    function = tree.body[0]
    function.args.args.extend(
        [
            ast.arg(arg="address", annotation=ast.parse("wp.uint64", mode="eval").body),
            ast.arg(arg="lane_world", annotation=ast.Name(id="int", ctx=ast.Load())),
            ast.arg(arg="active", annotation=ast.Name(id="bool", ctx=ast.Load())),
        ]
    )
    result = []
    tail = []
    free = []
    in_free = False
    in_tail = False
    for node in function.body:
        source = ast.unparse(node)
        if source.startswith(f"joint{roots[1]} ="):
            in_free = True
        if source.startswith("if not good:"):
            in_tail = True
        if in_tail:
            tail.append(node)
            continue
        if in_free:
            # Original structural-zero stores occur after independent free roots.
            if not source.startswith("cache.geometric["):
                free.append(node)
            continue
        if re.match(r"(?:projection|bias|column|h)\d+ =", source):
            continue
        if source.startswith("cache.bias[") or source.startswith("cache.geometric["):
            continue
        if source.startswith("good = good and wp.isfinite("):
            continue
        if re.match(r"(?:wrench|mass|moment|inertia)\d+ = \w+\d+ \+", source):
            continue
        if finish and source.startswith("data.body_q["):
            continue
        public = re.match(r"public_v(\d+) = _public_velocity", source)
        if public and finish:
            b = int(public[1])
            result.extend(
                _statements(f"""
radius{b} = wp.transform_point(pose{b}, data.body_com[body{b}]) - origin0
public_v{b} = wp.spatial_vector(wp.spatial_top(velocity{b}) + wp.cross(wp.spatial_bottom(velocity{b}), radius{b}), wp.spatial_bottom(velocity{b}))
if active:
    cache.com_offset[{b}, world] = radius{b}
_publish(address, {output_base}, lane_world, {b}, plan, data, public_pose{b}, public_v{b})
""")
            )
            continue
        # Padding participates in every finish collective, but never publishes
        # canonical/private data or touches the original free-root helpers.
        if finish and isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
            result.append(ast.If(test=ast.Name(id="active", ctx=ast.Load()), body=[node], orelse=[]))
        elif finish and isinstance(node, ast.If):

            class GuardWrites(ast.NodeTransformer):
                def visit_Assign(self, value):
                    if isinstance(value.targets[0], ast.Subscript):
                        return ast.If(test=ast.Name(id="active", ctx=ast.Load()), body=[value], orelse=[])
                    return value

            result.append(GuardWrites().visit(node))
        else:
            result.append(node)
        own = re.search(r"wrench(\d+).* = _(?:held|refresh)_terms", source)
        if own:
            b = int(own[1])
            offset = (b - 1) * width
            values = [f"wrench{b}[{k}]" for k in range(6)]
            if refresh:
                values += [f"mass{b}"] + [f"moment{b}[{k}]" for k in range(3)]
                values += [f"inertia{b}[{i}, {j}]" for i in range(3) for j in range(3)]
            result.extend(_statements("\n".join(f"_store(address, {offset + k}, {v})" for k, v in enumerate(values))))
        axis = re.match(r"axis(\d+) =", source)
        if axis:
            b = int(axis[1])
            result.extend(
                _statements(
                    "\n".join(f"_store(address, {axis_base + 6 * dofs[b] + k}, axis{b}[{k}])" for k in range(6))
                )
            )
    # Reverse dependencies hold only completed child suffixes, not every
    # ancestor's own wrench/moments or all descendant spatial columns.
    reverse = []

    def emit(source):
        reverse.extend(_statements(source))

    for b in reversed(range(1, roots[1])):
        offset = (b - 1) * width
        emit(f"sumw{b} = wp.spatial_vector(" + ", ".join(f"_load(address, {offset + k})" for k in range(6)) + ")")
        if refresh:
            emit(f"summ{b} = _load(address, {offset + 6})")
            emit(f"sump{b} = wp.vec3(" + ", ".join(f"_load(address, {offset + 7 + k})" for k in range(3)) + ")")
            emit(f"sumi{b} = wp.mat33(" + ", ".join(f"_load(address, {offset + 10 + k})" for k in range(9)) + ")")
        for child in (c for c, p in enumerate(parent) if p == b):
            emit(f"sumw{b} = sumw{b} + sumw{child}")
            if refresh:
                emit(
                    f"summ{b} = summ{b} + summ{child}\nsump{b} = sump{b} + sump{child}\nsumi{b} = sumi{b} + sumi{child}"
                )
        d = dofs[b]
        if d < 0:
            continue
        emit(
            f"projection{b} = wp.spatial_vector("
            + ", ".join(f"_load(address, {axis_base + 6 * d + k})" for k in range(6))
            + ")"
        )
        emit(
            f"bias{b} = wp.dot(projection{b}, sumw{b})\ncache.bias[{d}, world] = bias{b}\ngood = good and wp.isfinite(bias{b})"
        )
        if refresh:
            emit(f"column = _moment_action(summ{b}, sump{b}, sumi{b}, projection{b})")
            a = b
            while a >= 0:
                ad = dofs[a]
                if ad >= 0:
                    emit(
                        "ancestor = wp.spatial_vector("
                        + ", ".join(f"_load(address, {axis_base + 6 * ad + k})" for k in range(6))
                        + ")"
                    )
                    row, col = max(d, ad), min(d, ad)
                    emit(
                        f"value = wp.dot(ancestor, column)\ncache.geometric[{row * (row + 1) // 2 + col}, world] = value\ngood = good and wp.isfinite(value)"
                    )
                a = parent[a]
    if refresh:
        present = set()
        for b, d in enumerate(dofs):
            if d < 0:
                continue
            a = b
            while a >= 0:
                if dofs[a] >= 0:
                    present.add((max(d, dofs[a]), min(d, dofs[a])))
                a = parent[a]
        for row in range(sum(d >= 0 for d in dofs)):
            for col in range(row + 1):
                if (row, col) not in present:
                    emit(f"cache.geometric[{row * (row + 1) // 2 + col}, world] = 0.0")
    # No collective occurs in reverse, independent free-body publication or
    # status stamping, so padding can safely leave all of these untouched.
    if tail and isinstance(tail[-1], ast.Return):
        tail.pop()
    result.append(ast.If(test=ast.Name(id="active", ctx=ast.Load()), body=reverse + free + tail, orelse=[]))
    result.extend(_statements("return int(good)"))
    function.body = result
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


@functools.cache
def get_state_kernel(schedule, *, finish=False, refresh=False):
    """Compile the same state ABI with one shared allocation and fixed block32."""
    tag = hashlib.sha256(repr(schedule).encode()).hexdigest()[:10]
    extra = dict(globals())
    for current in (False, True) if not finish else (refresh,):
        name = f"stream_produce_{tag}_{int(finish)}_{int(current)}"
        source = _producer_source(schedule, finish=finish, refresh=current, name=name)
        extra["produce_refresh" if current else "produce_held"] = original._compile(source, name, extra)
    mode = "repair" if not finish else ("finish_refresh" if refresh else "finish_held")
    name = f"world_lane_{mode}_stream13_h45_{tag}"
    prefix = f"""@wp.kernel(module="unique", enable_backward=False)
def {name}(plan: KineticPlan, data: PublicationData, cache: KineticData, requests: wp.array[int], global_refresh: int):
    lane_world = wp.tid()
    world = wp.min(lane_world, plan.body_ids.shape[1] - 1)
    active = lane_world < plan.body_ids.shape[1]
"""
    if finish:
        body = f"""    address = {"_storage_refresh" if refresh else "_storage_held"}()
    {"produce_refresh" if refresh else "produce_held"}(world, plan, data, cache, address, lane_world, active)
    _release(address)
"""
    else:
        body = """    if not active:
        return
    refresh_now = global_refresh != 0 or requests[0] != 0
    valid = cache.current_valid[plan.arts[0, world]] & cache.current_valid[plan.arts[1, world]] & cache.current_valid[plan.arts[2, world]]
    if valid != 0 and original.retained._same_source(cache, world, data.joint_q, data.joint_qd) != 0:
        if not refresh_now or (cache.geometry_valid[world] != 0 and cache.geometry_generation[world] == cache.generation[world]):
            return
    address = _storage_refresh()
    if refresh_now:
        produce_refresh(world, plan, data, cache, address, lane_world, True)
    else:
        produce_held(world, plan, data, cache, address, lane_world, True)
    _release(address)
"""
    return original._compile(prefix + body, name, extra)


class StreamingWorldLaneState(original.WorldLaneState):
    """Keep the full original owner with streaming repair/finish and block32."""

    world_lane_streaming = True
    block_dim = 32

    def __init__(self, solver):
        super().__init__(solver)
        self.repair_kernel = get_state_kernel(self.schedule)
        self.finish_held_kernel = get_state_kernel(self.schedule, finish=True, refresh=False)
        self.finish_refresh_kernel = get_state_kernel(self.schedule, finish=True, refresh=True)
        self.finish_kernel = self.finish_refresh_kernel

    def begin(self, state_in, state_aug, dt, global_refresh):
        self._validate_call(state_in, state_aug, dt)
        wp.launch(
            self.repair_kernel,
            dim=self.solver.world_count,
            inputs=[
                self.plan,
                self._publication(state_in, state_aug, state_in, dt),
                self.data,
                self.solver._mass_update_requested,
                int(global_refresh),
            ],
            block_dim=32,
            device=self.solver.model.device,
        )

    def predict(self, state_in, state_aug, control, stage3_qd, dt):
        self._validate_call(state_in, state_aug, dt)
        if stage3_qd.ptr != state_in.joint_qd.ptr:
            raise RuntimeError("World-lane excludes alternate prescaled predictor velocity")
        force, factor = self._predictor_inputs(state_in, state_aug, control, stage3_qd, dt)
        wp.launch(
            self.predictor_kernel,
            dim=self.solver.world_count,
            inputs=[self.plan, self.data, force, factor],
            block_dim=32,
            device=self.solver.model.device,
        )

    def finish(self, state_in, state_aug, state_out, dt, next_refresh):
        self._validate_call(state_in, state_aug, dt)
        self._validate_call(state_out, state_aug, dt)
        if any(
            a.ptr < b.ptr + b.capacity and b.ptr < a.ptr + a.capacity
            for a in (state_in.joint_q, state_in.joint_qd)
            for b in (state_out.joint_q, state_out.joint_qd)
        ):
            raise RuntimeError("World-lane integration requires disjoint generalized state banks")
        wp.launch(
            self.finish_refresh_kernel if next_refresh else self.finish_held_kernel,
            dim=(self.solver.world_count + 31) // 32 * 32,
            inputs=[
                self.plan,
                self._publication(state_in, state_aug, state_out, dt),
                self.data,
                self.solver._mass_update_requested,
                int(next_refresh),
            ],
            block_dim=32,
            device=self.solver.model.device,
        )
