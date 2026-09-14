# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental current-motion maps and typed Allegro kinetic rows."""

import ast
import hashlib
import inspect
import linecache
import textwrap
from functools import cache

import numpy as np
import warp as wp

from ...sim import JointType, ModelFlags
from .franka_contact_packet import (
    ContactInput,
    _broadcast,
    _join,
    _lane,
    _stride,
    bind,
    contact_points,
    endpoint_projection,
    friction_multiplier,
)
from .kernels import (
    contact_restitution_fires,
    contact_tangent_basis,
    mixed_contact_restitution,
    prescribed_relative_contact_target,
)


@wp.struct
class KineticRows:
    coefficients: wp.array3d[float]
    encoding: wp.array2d[int]


@wp.struct
class MapInput:
    groups: wp.array2d[int]
    starts: wp.array2d[int]
    body_map: wp.array[int]
    motion: wp.array[wp.spatial_vector]
    L16: wp.array3d[float]
    L6: wp.array3d[float]
    inverse: wp.array2d[float]
    maps: wp.array2d[float]
    incident: wp.array[float]
    kinetic_incident: wp.array2d[float]


@cache
def get_map_kernel():
    """Share four-chain prefix products while reading the original held factor."""
    source = r"""
    const int ga = data.groups.data[world * 2], gb = data.groups.data[world * 2 + 1];
    const int sa = data.starts.data[world * 2], sb = data.starts.data[world * 2 + 1];
    const float* La = data.L16.data + ga * 256;
    const float* Lb = data.L6.data + gb * 36;
    float* W = data.inverse.data + world * 100;
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x;
#else
    for (int lane = 0; lane < 32; ++lane) {
#endif
    if (lane < 22) {
        const int n = lane < 16 ? 4 : 6;
        const int col = lane < 16 ? lane % 4 : lane - 16;
        const int offset = lane < 16 ? (lane / 4) * 4 : 0;
        const int stride = lane < 16 ? 16 : 6;
        const float* L = lane < 16 ? La + offset * 16 + offset : Lb;
        float* out = lane < 16 ? W + (lane / 4) * 16 : W + 64;
        for (int row = 0; row < n; ++row) {
            float v = row == col ? 1.0f : 0.0f;
            for (int k = 0; k < row; ++k) v -= L[row * stride + k] * out[k * n + col];
            out[row * n + col] = v / L[row * stride + row];
        }
        float uv = 0.0f;
        if (lane < 16) for (int row = lane; row < 16; ++row) uv += La[row * 16 + lane] * data.incident.data[sa + row];
        else for (int row = lane - 16; row < 6; ++row) uv += Lb[row * 6 + lane - 16] * data.incident.data[sb + row];
        data.kinetic_incident.data[world * 22 + lane] = uv;
    }
#if defined(__CUDA_ARCH__)
    __syncwarp();
#else
    }
    for (int lane = 0; lane < 32; ++lane) {
#endif
    if (lane < 24) {
        const int finger = lane / 6, axis = lane % 6;
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int j = 0; j < 4; ++j) {
            const float S = data.motion.data[sa + finger * 4 + j][axis];
            for (int row = 0; row < 4; ++row) {
                if (row >= j) v[row] += W[finger * 16 + row * 4 + j] * S;
                data.maps.data[world * 420 + (finger * 4 + j) * 24 + row * 6 + axis] = v[row];
            }
        }
    } else if (lane < 30) {
        const int axis = lane - 24;
        for (int row = 0; row < 6; ++row) {
            float v = 0.0f;
            for (int j = 0; j <= row; ++j) v += W[64 + row * 6 + j] * data.motion.data[sb + j][axis];
            data.maps.data[world * 420 + 384 + row * 6 + axis] = v;
        }
    }
#if !defined(__CUDA_ARCH__)
    }
#endif
"""

    @wp.func_native(source)
    def native(world: int, logical_lane: int, data: MapInput): ...

    @wp.kernel(module="unique", enable_backward=False)
    def allegro_kinetic_motion_maps(data: MapInput):
        world, lane = wp.tid()
        native(world, lane, data)

    return allegro_kinetic_motion_maps


@wp.struct
class PrefixInput:
    starts: wp.array2d[int]
    q_index: wp.array[int]
    lower: wp.array[float]
    upper: wp.array[float]
    velocity_limit: wp.array[float]
    q: wp.array[float]
    vhat: wp.array[float]
    inverse: wp.array2d[float]
    counts: wp.array[int]
    bounds: wp.array2d[int]
    velocity_slots: wp.array[int]
    velocity_signs: wp.array[float]
    row_dof: wp.array2d[int]
    row_sign: wp.array2d[float]
    kind: wp.array2d[int]
    parent: wp.array2d[int]
    mu: wp.array2d[float]
    beta: wp.array2d[float]
    cfm: wp.array2d[float]
    phi: wp.array2d[float]
    target: wp.array2d[float]
    restitution: wp.array2d[float]
    rhs: wp.array2d[float]
    gap: float
    activation: float
    pgs_beta: float
    pgs_cfm: float
    dt: float


@cache
def get_prefix_kernel():
    """Reserve stable position and signed velocity prefixes without dense e_i rows."""
    source = r"""
    const int first = p.starts.data[world * 2];
    const int second = p.starts.data[world * 2 + 1];
    const int M = p.kind.shape[1];
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x;
    for (int k = lane; k < 44; k += 32) {
#else
    for (int k = 0; k < 44; ++k) {
#endif
        const int dof = k / 2 < 16 ? first + k / 2 : second + k / 2 - 16;
        p.velocity_slots.data[dof * 2 + k % 2] = -1;
        p.velocity_signs.data[dof * 2 + k % 2] = 0.0f;
    }
    int base = 0;
    for (int phase = 0; phase < 2; ++phase) {
#if !defined(__CUDA_ARCH__)
        int phase_count = 0;
        for (int lane = 0; lane < 32; ++lane) {
#endif
        const int d = lane / 2, side = lane & 1, dof = first + d;
        const float sign = side == 0 ? 1.0f : -1.0f;
        bool active = false;
        float phi = 0.0f, target = 0.0f;
        if (phase == 0) {
            const int qi = p.q_index.data[dof];
            if (qi >= 0) {
                const float bound = side == 0 ? p.lower.data[dof] : p.upper.data[dof];
                const float q = p.q.data[qi];
                phi = side == 0 ? q - bound : bound - q;
                active = isfinite(bound) && (side == 0 ? q <= bound + p.gap : q >= bound - p.gap);
            }
        } else {
            const float limit = p.velocity_limit.data[dof];
            active = !(limit <= 0.0f) && !(p.activation > 0.0f && fabsf(p.vhat.data[dof]) < p.activation * limit);
            target = -limit;
        }
#if defined(__CUDA_ARCH__)
        const unsigned mask = __ballot_sync(0xffffffffu, active);
        const int phase_count = __popc(mask);
        const int rank = __popc(mask & (lane == 0 ? 0u : ((1u << lane) - 1u)));
#else
        const int rank = phase_count;
        phase_count += active;
#endif
        const int slot = base + rank;
        if (active && slot < M) {
            const int at = world * M + slot;
            p.row_dof.data[at] = d; p.row_sign.data[at] = sign;
            p.kind.data[at] = phase == 0 ? 3 : 4;
            p.parent.data[at] = -1; p.mu.data[at] = 0.0f;
            p.beta.data[at] = phase == 0 ? p.pgs_beta : 0.0f;
            p.cfm.data[at] = p.pgs_cfm;
            p.phi.data[at] = phi; p.target.data[at] = target;
            p.restitution.data[at] = 0.0f;
            p.rhs.data[at] = phase == 1 ? -target : ((phi < 0.0f ? p.pgs_beta : 1.0f) * phi / p.dt);
            const int offset = (d / 4) * 4;
            z.encoding.data[at] = offset | (4 << 5);
            for (int k = 0; k < 4; ++k) z.coefficients.data[at * 10 + k] = sign * p.inverse.data[world * 100 + (d / 4) * 16 + k * 4 + d % 4];
            if (phase == 1) {
                p.velocity_slots.data[dof * 2 + side] = slot;
                p.velocity_signs.data[dof * 2 + side] = sign;
            }
        }
#if !defined(__CUDA_ARCH__)
        }
#endif
        base += phase_count;
#if defined(__CUDA_ARCH__)
        if (lane == 0)
#endif
        p.bounds.data[world * 2 + phase] = base;
    }
#if defined(__CUDA_ARCH__)
    if (lane == 0)
#endif
    p.counts.data[world] = base;
"""

    @wp.func_native(source)
    def native(world: int, logical_lane: int, p: PrefixInput, z: KineticRows): ...

    @wp.kernel(module="unique", enable_backward=False)
    def allegro_kinetic_limit_prefix(p: PrefixInput, z: KineticRows):
        world, lane = wp.tid()
        native(world, lane, p, z)

    return allegro_kinetic_limit_prefix


@wp.func
def _map_id(maps: MapInput, body: int, art: int):
    result = int(-1)
    if body >= 0 and art >= 0:
        result = maps.body_map[body]
    return result


@wp.func
def _map_offset(index: int):
    return 16 if index == 16 else (index // 4) * 4


@wp.func
def _map_length(index: int):
    return 6 if index == 16 else 4


@wp.func
def _project_map(maps: MapInput, world: int, index: int, coordinate: int, lever: wp.vec3, direction: wp.vec3):
    value = float(0.0)
    if index >= 0:
        start = 384 if index == 16 else index * 24
        local = coordinate - _map_offset(index)
        if local >= 0 and local < _map_length(index):
            torque = wp.cross(lever, direction)
            for axis in range(3):
                value += maps.maps[world, start + local * 6 + axis] * direction[axis]
                value += maps.maps[world, start + local * 6 + axis + 3] * torque[axis]
    return value


@wp.kernel(enable_backward=False)
def produce_contacts(workers: int, data: ContactInput, maps: MapInput, z: KineticRows):
    """Share current geometry and emit compact Z with original contact metadata."""
    worker, _logical_lane = wp.tid()
    lane, stride = _lane(), _stride()
    total = wp.min(data.count[0], data.point0.shape[0])
    for contact in range(worker, total, workers):
        if data.path[contact] != 0 or data.slot[contact] < 0:
            continue
        world, slot = data.world[contact], data.slot[contact]
        art0, art1 = data.art0[contact], data.art1[contact]
        body0, body1 = int(-1), int(-1)
        if data.shape0[contact] >= 0:
            body0 = data.shape_body[data.shape0[contact]]
        if data.shape1[contact] >= 0:
            body1 = data.shape_body[data.shape1[contact]]
        normal = -data.normal[contact]
        point0, point1, t0, t1 = wp.vec3(), wp.vec3(), wp.vec3(), wp.vec3()
        if lane == 0:
            _body0, _body1, point0, point1 = contact_points(data, contact)
            t0, t1 = contact_tangent_basis(normal)
        # Warp-coherent geometry is read once by lane zero at the physical seam.
        point0, point1 = _broadcast(point0), _broadcast(point1)
        t0, t1 = _broadcast(t0), _broadcast(t1)
        anchor = 0.5 * (point0 + point1)
        id0, id1 = _map_id(maps, body0, art0), _map_id(maps, body1, art1)
        offset0, offset1, length0, length1 = int(0), int(0), int(0), int(0)
        if id0 >= 0:
            offset0, length0 = _map_offset(id0), _map_length(id0)
        if id1 >= 0:
            if length0 == 0:
                offset0, length0 = _map_offset(id1), _map_length(id1)
            elif _map_offset(id1) != offset0:
                offset1, length1 = _map_offset(id1), _map_length(id1)
        code = offset0 | (length0 << 5) | (offset1 << 8) | (length1 << 13)
        for item in range(lane, data.slots_needed[contact] * 10, stride):
            row, k = item // 10, item % 10
            if k < length0 + length1:
                coordinate = offset0 + k if k < length0 else offset1 + k - length0
                direction = normal if row == 0 else (t0 if row == 1 else t1)
                p0, p1 = point0, point1
                if data.shared_anchor != 0 or (row > 0 and data.friction_shared_anchor != 0):
                    p0, p1 = anchor, anchor
                value = float(0.0)
                if art0 >= 0:
                    value += _project_map(maps, world, id0, coordinate, p0 - data.origin[art0], direction)
                if art1 >= 0:
                    value -= _project_map(maps, world, id1, coordinate, p1 - data.origin[art1], direction)
                z.coefficients[world, slot + row, k] = value
        _join()
        if lane == 0:
            mu, materials = float(0.0), int(0)
            if data.shape0[contact] >= 0:
                mu += data.material_mu[data.shape0[contact]]
                materials += 1
            if data.shape1[contact] >= 0:
                mu += data.material_mu[data.shape1[contact]]
                materials += 1
            if materials > 0:
                mu /= float(materials)
            rest = mixed_contact_restitution(data.shape0[contact], data.shape1[contact], data.material_restitution)
            friction_mu = mu * friction_multiplier(data, contact, total, art0, art1)
            separation = wp.dot(normal, point0 - point1)
            for row in range(data.slots_needed[contact]):
                r = slot + row
                direction = normal if row == 0 else (t0 if row == 1 else t1)
                p0, p1 = point0, point1
                if data.shared_anchor != 0 or (row > 0 and data.friction_shared_anchor != 0):
                    p0, p1 = anchor, anchor
                target = prescribed_relative_contact_target(
                    body0, art0, body1, art1, p0, p1, direction, data.prescribed, data.origin, data.body_v
                )
                rhs = -target
                phi, beta, restitution = float(0.0), float(0.0), float(0.0)
                kind = int(2)
                parent = int(slot)
                row_mu = float(friction_mu)
                if row == 0:
                    phi = separation
                    beta = data.beta
                    restitution = rest
                    kind = 0
                    parent = -1
                    row_mu = mu
                    rhs += (data.bias_scale * beta if phi <= 0.0 else data.speculative_scale) * phi / data.dt
                    if restitution > 0.0:
                        relative = -target
                        for k in range(length0 + length1):
                            coordinate = offset0 + k if k < length0 else offset1 + k - length0
                            relative += z.coefficients[world, r, k] * maps.kinetic_incident[world, coordinate]
                        if contact_restitution_fires(phi, relative, data.dt, data.restitution_threshold):
                            rhs = -target + restitution * relative
                z.encoding[world, r] = code
                data.row_type[world, r] = kind
                data.row_parent[world, r] = parent
                data.row_mu[world, r] = row_mu
                data.row_beta[world, r] = beta
                data.row_cfm[world, r] = data.cfm
                data.phi[world, r] = phi
                data.target[world, r] = target
                data.restitution[world, r] = restitution
                data.rhs[world, r] = rhs
                data.diag[world, r] = data.cfm
        _join()


@wp.kernel(enable_backward=False)
def materialize_prefix(
    p: PrefixInput,
    counts: wp.array[int],
    mf_counts: wp.array[int],
    maps: MapInput,
    J16: wp.array3d[float],
    J6: wp.array3d[float],
):
    """Clear all current fallback rows and materialize its exact coordinate prefix."""
    world, lane = wp.tid()
    if counts[world] <= 128 and mf_counts[world] == 0:
        return
    ga, gb = maps.groups[world, 0], maps.groups[world, 1]
    for item in range(lane, counts[world] * 22, _stride()):
        row, d = item // 22, item % 22
        value = float(0.0)
        if row < p.bounds[world, 1] and p.row_dof[world, row] == d:
            value = p.row_sign[world, row]
        if d < 16:
            J16[ga, row, d] = value
        else:
            J6[gb, row, d - 16] = value


@wp.kernel(enable_backward=False)
def materialize_contacts(
    workers: int,
    data: ContactInput,
    counts: wp.array[int],
    mf_counts: wp.array[int],
    maps: MapInput,
    J16: wp.array3d[float],
    J6: wp.array3d[float],
):
    """Publish exact current endpoint rows only for the original 12-step fallback."""
    worker, _logical_lane = wp.tid()
    lane, stride = _lane(), _stride()
    total = wp.min(data.count[0], data.point0.shape[0])
    for contact in range(worker, total, workers):
        if data.path[contact] != 0 or data.slot[contact] < 0:
            continue
        world = data.world[contact]
        if counts[world] <= 128 and mf_counts[world] == 0:
            continue
        body0, body1, point0, point1 = contact_points(data, contact)
        normal = -data.normal[contact]
        t0, t1 = contact_tangent_basis(normal)
        anchor = 0.5 * (point0 + point1)
        for item in range(lane, data.slots_needed[contact] * 22, stride):
            row, d = item // 22, item % 22
            size = 16 if d < 16 else 6
            local = d if d < 16 else d - 16
            direction = normal if row == 0 else (t0 if row == 1 else t1)
            p0, p1 = point0, point1
            if data.shared_anchor != 0 or (row > 0 and data.friction_shared_anchor != 0):
                p0, p1 = anchor, anchor
            value = endpoint_projection(data, body0, data.art0[contact], size, local, p0, direction)
            value -= endpoint_projection(data, body1, data.art1[contact], size, local, p1, direction)
            r = data.slot[contact] + row
            if d < 16:
                J16[maps.groups[world, 0], r, d] = value
            else:
                J6[maps.groups[world, 1], r, d - 16] = value


def get_ink_stage(na, nb, oa, ob, dofs, capacity):
    """Replace only the input stage; retain the parallel law and inverse adjoint."""
    if (na, nb, oa, ob, dofs) != (16, 6, 0, 16, 22):
        raise ValueError("Allegro kinetic rows require the validated 16+6 layout")
    return f"""
    const int ink_ga = ink_meta.data[world * 4], ink_gb = ink_meta.data[world * 4 + 2];
    constexpr int ink_oa = 0, ink_ob = 16;
    for (int e = lane; e < 256; e += NT) s_L[e] = ink_L_a.data[(size_t)ink_ga * 256 + e];
    for (int e = lane; e < 36; e += NT) s_L[256 + e] = ink_L_b.data[(size_t)ink_gb * 36 + e];
    SYNC();
    if (lane < 16) s_Dinv[lane] = 1.0f / s_L[lane * 16 + lane];
    if (lane < 6) s_Dinv[16 + lane] = 1.0f / s_L[256 + lane * 6 + lane];
    if (lane < 22) {{
        float uv = 0.0f;
        if (lane < 16) for (int r = lane; r < 16; ++r) uv += s_L[r * 16 + lane] * s_v[r];
        else for (int r = lane - 16; r < 6; ++r) uv += s_L[256 + r * 6 + lane - 16] * s_v[16 + r];
        s_dv[lane] = uv;
    }}
    SYNC();
    if (lane < n_rows) {{
        const int i = lane;
        const int code = kinetic_rows.encoding.data[off_dense + i];
        const int o0 = code & 31, n0 = (code >> 5) & 7;
        const int o1 = (code >> 8) & 31, n1 = (code >> 13) & 7;
        float bjv = 0.0f;
        #pragma unroll
        for (int d = 0; d < 22; ++d) {{
            const int k = d >= o0 && d < o0 + n0 ? d - o0
                        : d >= o1 && d < o1 + n1 ? n0 + d - o1 : -1;
            Jr[d] = k >= 0 ? kinetic_rows.coefficients.data[((size_t)world * {capacity} + i) * 10 + k] : 0.0f;
            s_Yt[d * YS + i] = Jr[d]; bjv += Jr[d] * s_dv[d];
        }}
        s_rhs[i] += bjv;
    }}
"""


@cache
def get_parallel_factory(original):
    """Extend only the opted-in native ABI; keep the original factory untouched."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    factory = tree.body[0]
    factory.name = "kinetic_parallel_factory"
    factory.decorator_list = []
    signatures, calls = 0, 0
    for node in ast.walk(factory):
        if isinstance(node, ast.FunctionDef) and node is not factory:
            names = [arg.arg for arg in node.args.args]
            if "world_row_cfm" in names:
                node.args.args.insert(
                    names.index("world_row_cfm") + 1,
                    ast.arg(arg="kinetic_rows", annotation=ast.Name(id="KineticRows", ctx=ast.Load())),
                )
                signatures += 1
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "pgs_solve_parallel_native"
        ):
            names = [arg.id if isinstance(arg, ast.Name) else None for arg in node.args]
            node.args.insert(names.index("world_row_cfm") + 1, ast.Name(id="kinetic_rows", ctx=ast.Load()))
            calls += 1
    if (signatures, calls) != (2, 1):
        raise RuntimeError("Original parallel native signature seam changed")
    insertion = next(
        i
        for i, node in enumerate(factory.body)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "snippet" for t in node.targets)
    )
    factory.body[insertion:insertion] = ast.parse(
        "if register_whitening or WR or ZG or INK_CHECK or not INK or not MF or HD or EXACT or LEAN or BF or STOP:\n"
        "    raise ValueError('Kinetic rows require the ordinary 16+6 matrix-free parallel law')\n"
        "ink_stage = get_ink_stage(NA, NB, OA, OB, D, M_D)"
    ).body
    final_return = next(i for i, node in enumerate(factory.body) if isinstance(node, ast.Return))
    factory.body[final_return:final_return] = ast.parse("kernel._fpgs_kinetic_rows = True").body
    for node in ast.walk(factory):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace("pgs_solve_parallel_", "pgs_solve_kinetic_parallel_")
    namespace = dict(original.__globals__)
    namespace.update(KineticRows=KineticRows, get_ink_stage=get_ink_stage)
    ast.fix_missing_locations(tree)
    generated = ast.unparse(tree) + "\n"
    filename = f"<allegro-kinetic-parallel-{hashlib.sha256(generated.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(generated), None, generated.splitlines(keepends=True), filename)
    exec(compile(generated, filename, "exec"), namespace)
    return namespace[factory.name]


def topology_plan(solver):
    """Validate the four independent scalar chains in every actual world."""
    if sorted(solver.size_groups) != [6, 16] or solver.max_world_dofs != 22:
        raise ValueError("Kinetic rows require one 16-coordinate hand and one free-six body per world")
    worlds = solver.art_to_world.numpy()
    groups = np.full((solver.world_count, 2), -1, dtype=np.int32)
    starts = np.empty_like(groups)
    art_start = solver.articulation_dof_start.numpy()
    for column, size in enumerate((16, 6)):
        arts = solver.group_to_art[size].numpy()
        if len(arts) != solver.world_count or not np.array_equal(np.sort(worlds[arts]), np.arange(solver.world_count)):
            raise ValueError("Kinetic rows require complete one-articulation-per-size world coverage")
        groups[worlds[arts], column] = np.arange(len(arts), dtype=np.int32)
        starts[worlds[arts], column] = art_start[arts]
    masks = solver.body_response_dof_mask.numpy()
    body_art = solver.body_to_articulation.numpy()
    sizes = solver.articulation_response_dof_count.numpy()
    body_map = np.full(len(masks), -1, dtype=np.int32)
    for finger in range(4):
        for depth in range(4):
            mask = ((1 << (depth + 1)) - 1) << (finger * 4)
            body_map[(body_art >= 0) & (sizes[np.maximum(body_art, 0)] == 16) & (masks == mask)] = finger * 4 + depth
    body_map[(body_art >= 0) & (sizes[np.maximum(body_art, 0)] == 6) & (masks == 63)] = 16
    if np.any((masks != 0) & (body_map < 0)):
        raise ValueError("Kinetic rows reject a non-chain current response mask")
    # A fixed articulation root is what makes the four finger blocks independent.
    model = solver.model
    types, joint_parent = model.joint_type.numpy(), model.joint_parent.numpy()
    first_joint = model.articulation_start.numpy()
    primary = solver.group_to_art[16].numpy()
    if np.any(types[first_joint[primary]] != 3) or np.any(joint_parent[first_joint[primary]] != -1):
        raise ValueError("The hand must have a fixed root")
    joint_dofs = np.diff(model.joint_qd_start.numpy())
    for art in primary:
        selected = slice(first_joint[art], first_joint[art + 1])
        moving_types = types[selected][joint_dofs[selected] > 0]
        if not np.isin(moving_types, (JointType.PRISMATIC, JointType.REVOLUTE, JointType.D6)).all():
            raise ValueError("The hand requires original velocity-limit-supported moving joint types")
    secondary = solver.group_to_art[6].numpy()
    if np.any(types[first_joint[secondary]] != 4) or np.any(first_joint[secondary + 1] - first_joint[secondary] != 1):
        raise ValueError("The secondary articulation must be one free body")
    return groups, starts, body_map


def _signature(solver):
    names = (
        "joint_type",
        "joint_parent",
        "joint_child",
        "joint_qd_start",
        "joint_q_start",
        "joint_dof_dim",
        "articulation_start",
        "body_flags",
        "body_world",
    )
    return tuple(getattr(solver.model, name).numpy().tobytes() for name in names)


class AllegroKineticRows:
    """Own the current typed prefix, compact rows, and complete legacy fallback."""

    def __init__(self, solver):
        self.solver = solver
        groups, starts, body_map = topology_plan(solver)
        dev, worlds, capacity = solver.model.device, solver.world_count, solver.dense_max_constraints
        self.data, self.maps, self.prefix = KineticRows(), MapInput(), PrefixInput()
        self.data.coefficients = wp.empty((worlds, capacity, 10), dtype=float, device=dev)
        self.data.encoding = wp.empty((worlds, capacity), dtype=int, device=dev)
        self.maps.groups = wp.array(groups, dtype=int, device=dev)
        self.maps.starts = wp.array(starts, dtype=int, device=dev)
        self.maps.body_map = wp.array(body_map, dtype=int, device=dev)
        self.maps.inverse = wp.empty((worlds, 100), dtype=float, device=dev)
        self.maps.maps = wp.empty((worlds, 420), dtype=float, device=dev)
        self.maps.kinetic_incident = wp.empty((worlds, 22), dtype=float, device=dev)
        self.prefix.row_dof = wp.empty((worlds, capacity), dtype=int, device=dev)
        self.prefix.row_sign = wp.empty((worlds, capacity), dtype=float, device=dev)
        self.contact_input = None
        self._signature = _signature(solver)

    def validate_notification(self, flags=ModelFlags.ALL):
        """Reject static-plan changes before retiring any canonical producer."""
        numeric = (
            ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.BODY_INERTIAL_PROPERTIES
            | ModelFlags.SHAPE_PROPERTIES
            | ModelFlags.MODEL_PROPERTIES
        )
        if flags and not (int(flags) & ~int(numeric)):
            return
        if _signature(self.solver) != self._signature:
            raise RuntimeError("Allegro kinetic topology changed; reconstruct the solver before notification")
        topology_plan(self.solver)

    def begin_rows(self, state_in, state_aug, dt):
        """Read current motion and predictor with the current held-factor buffers."""
        s, m, p = self.solver, self.maps, self.prefix
        m.motion, m.L16, m.L6, m.incident = state_aug.joint_S_s, s.L_by_size[16], s.L_by_size[6], s.v_hat
        p.starts, p.inverse, p.q, p.vhat = m.starts, m.inverse, state_in.joint_q, s.v_hat
        for name, source in (
            ("q_index", "_joint_limit_q_index"),
            ("counts", "slot_counter"),
            ("bounds", "dense_phase_bounds"),
            ("velocity_slots", "velocity_limit_slot"),
            ("velocity_signs", "velocity_limit_sign"),
            ("kind", "row_type"),
            ("parent", "row_parent"),
            ("mu", "row_mu"),
            ("beta", "row_beta"),
            ("cfm", "row_cfm"),
            ("phi", "phi"),
            ("target", "target_velocity"),
            ("restitution", "row_restitution"),
            ("rhs", "rhs"),
        ):
            setattr(p, name, getattr(s, source))
        p.lower, p.upper, p.velocity_limit = (
            s.model.joint_limit_lower,
            s.model.joint_limit_upper,
            s.model.joint_velocity_limit,
        )
        p.gap, p.activation, p.pgs_beta, p.pgs_cfm, p.dt = (
            s.joint_limit_activation_gap,
            s.velocity_limit_activation_fraction,
            s.pgs_beta,
            s.pgs_cfm,
            dt,
        )
        self.contact_input = None
        wp.launch_tiled(get_map_kernel(), dim=[s.world_count], inputs=[m], block_dim=32, device=s.model.device)
        wp.launch_tiled(
            get_prefix_kernel(), dim=[s.world_count], inputs=[p, self.data], block_dim=32, device=s.model.device
        )

    def produce_contacts(self, state_in, state_aug, contacts, dt):
        """Defer current contact construction until final fallback counts are known."""
        self.contact_input = bind(self.solver, state_in, state_aug, contacts, dt, 1.0)

    def finish_rows(self):
        """Publish current direct rows and fully materialize only fallback worlds."""
        s = self.solver
        wp.launch_tiled(
            materialize_prefix,
            dim=[s.world_count],
            inputs=[self.prefix, s.constraint_count, s.mf_constraint_count, self.maps, s.J_by_size[16], s.J_by_size[6]],
            block_dim=32,
            device=s.model.device,
        )
        if self.contact_input is not None:
            workers = min(self.contact_input.point0.shape[0], 16384)
            wp.launch_tiled(
                produce_contacts,
                dim=[workers],
                inputs=[workers, self.contact_input, self.maps, self.data],
                block_dim=32,
                device=s.model.device,
            )
            wp.launch_tiled(
                materialize_contacts,
                dim=[workers],
                inputs=[
                    workers,
                    self.contact_input,
                    s.constraint_count,
                    s.mf_constraint_count,
                    self.maps,
                    s.J_by_size[16],
                    s.J_by_size[6],
                ],
                block_dim=32,
                device=s.model.device,
            )


def create_owner(solver):
    """Admit one bounded complete numerical law; leave unsupported solvers ordinary."""
    from . import solver_feather_pgs as source  # noqa: PLC0415

    supported = (
        solver.model.device.is_cuda
        and not solver.model.requires_grad
        and solver.pgs_mode == "matrix_free"
        and solver.drive_mode == "augmented"
        and solver.friction_mode == "current"
        and solver.pgs_schedule == "interleaved"
        and solver.enable_joint_limits
        and solver.enable_joint_velocity_limits
        and solver.mf_gs_parallel_rows == 128
        and solver.mf_gs_parallel_matrix_free
        and solver.mf_gs_response_block_rows == 0
        and solver.dense_max_constraints >= 128
        and solver.mf_gs_parallel_sweeps == 24
        and solver.pgs_iterations == 12
        and solver.pgs_velocity_iterations == 0
        and not solver.pgs_warmstart
        and solver.articulated_contact_response == "immediate"
        and solver._hinv_jt_writes_world
        and not solver.fuse_joint_velocity_limits
        and not solver._regularization_enabled
        and not solver._preelim_active
        and not solver._paired_factor_coordinates
        and not solver._has_loop_joints
        and not solver._fused_diagonal_joint_limits
        and not source._COMPACT_CONTACT_BOUNDARY
        and not solver._sparse_diagonal_contact_solve
        and not solver._mimic_count
        and not solver._connect_count
        and solver._contact_w == 1.0
        and not solver._debug_buffers_enabled
        and source._INK_ON
        and not source._WR_ON
        and not source._INK_CHECK
        and not source._CHECK_ROWS
        and not source._CHECK_ROWS_FUSED
        and not source._GROUPED_CHECK
        and not source._MF_EXACT_ROWSUM
        and not source._SHADOW_LEAN
        and not source._FPGS_CAPTURE
    )
    if not supported:
        return None
    try:
        return AllegroKineticRows(solver)
    except ValueError:
        return None
