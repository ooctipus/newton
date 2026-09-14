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
from .franka_contact_packet import ContactInput, bind


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


@wp.struct
class KineticRows:
    rowkeys: wp.array2d[int]
    maps: MapInput
    prefix: PrefixInput
    contact: ContactInput


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
            z.rowkeys.data[at] = ~(d * 2 + side);
            p.kind.data[at] = phase == 0 ? 3 : 4;
            p.parent.data[at] = -1; p.mu.data[at] = 0.0f;
            p.beta.data[at] = phase == 0 ? p.pgs_beta : 0.0f;
            p.cfm.data[at] = p.pgs_cfm;
            p.phi.data[at] = phi; p.target.data[at] = target;
            p.restitution.data[at] = 0.0f;
            p.rhs.data[at] = phase == 1 ? -target : ((phi < 0.0f ? p.pgs_beta : 1.0f) * phi / p.dt);
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


@wp.kernel(enable_backward=False)
def scatter_contact_keys(data: ContactInput, z: KineticRows):
    """Emit only the original allocated raw contact ID and row direction."""
    contact = wp.tid()
    if contact < wp.min(data.count[0], data.point0.shape[0]):
        if data.path[contact] == 0 and data.slot[contact] >= 0:
            for direction in range(data.slots_needed[contact]):
                z.rowkeys[data.world[contact], data.slot[contact] + direction] = contact * 4 + direction


def _row_source(*, kinetic):
    """Share original WR geometry and metadata between direct ingestion and fallback."""
    prefix = """
        const int token = ~key, d = token / 2;
        const float sign = (token & 1) == 0 ? 1.0f : -1.0f;
        #pragma unroll
        for (int q = 0; q < 22; ++q) {
            __PREFIX_VALUE__
        }
    """
    prefix = prefix.replace(
        "__PREFIX_VALUE__",
        "Jr[q] = q / 4 == d / 4 ? sign * z.maps.inverse.data[world * 100 + (d / 4) * 16 + (q % 4) * 4 + d % 4] : 0.0f;"
        if kinetic
        else "Jr[q] = q == d ? sign : 0.0f;",
    )
    projection = """
        #pragma unroll
        for (int q = 0; q < 22; ++q) {
            float value = 0.0f;
            for (int side = 0; side < 2; ++side) {
                const int body = side == 0 ? ba : bb, art = side == 0 ? aa : ab;
                if (body < 0 || art < 0) continue;
                const float sign = side == 0 ? 1.0f : -1.0f;
                const wp::vec3 point = side == 0 ? pa : pb;
                __PROJECT__
            }
            Jr[q] = value;
        }
    """
    projection = projection.replace(
        "__PROJECT__",
        """
                const int offset = side == 0 ? offset_a : offset_b;
                const int n = side == 0 ? size_a : size_b;
                if (q < offset || q >= offset + n) continue;
                const float* map = (side == 0 ? map_a : map_b) + (q - offset) * 6;
                const wp::vec3 torque = side == 0 ? torque_a : torque_b;
                float v = 0.0f;
                #pragma unroll
                for (int axis = 0; axis < 3; ++axis) {
                    v += map[axis] * direction[axis];
                    v += map[axis + 3] * torque[axis];
                }
                value += sign * v;
        """
        if kinetic
        else """
                const int n = c.response_dofs.data[art], local = q < 16 ? q : q - 16;
                if (n != (q < 16 ? 16 : 6) || (c.body_mask.data[body] & (1u << local)) == 0u) continue;
                const wp::spatial_vector S = c.motion.data[c.dof_start.data[art] + local];
                const wp::vec3 linear(S[0], S[1], S[2]), angular(S[3], S[4], S[5]);
                value += sign * wp::dot(direction, linear + wp::cross(angular, point - c.origin.data[art]));
        """,
    )
    if kinetic:
        projection = (
            """
        const int id_a = ba >= 0 && aa >= 0 ? z.maps.body_map.data[ba] : -1;
        const int id_b = bb >= 0 && ab >= 0 ? z.maps.body_map.data[bb] : -1;
        const int offset_a = id_a == 16 ? 16 : (id_a >= 0 ? (id_a / 4) * 4 : 0);
        const int offset_b = id_b == 16 ? 16 : (id_b >= 0 ? (id_b / 4) * 4 : 0);
        const int size_a = id_a < 0 ? 0 : (id_a == 16 ? 6 : 4);
        const int size_b = id_b < 0 ? 0 : (id_b == 16 ? 6 : 4);
        const int start_a = id_a == 16 ? 384 : (id_a >= 0 ? id_a * 24 : 0);
        const int start_b = id_b == 16 ? 384 : (id_b >= 0 ? id_b * 24 : 0);
        const float* map_a = z.maps.maps.data + world * 420 + start_a;
        const float* map_b = z.maps.maps.data + world * 420 + start_b;
        wp::vec3 torque_a(0.0f), torque_b(0.0f);
        if (size_a > 0) torque_a = wp::cross(pa - c.origin.data[aa], direction);
        if (size_b > 0) torque_b = wp::cross(pb - c.origin.data[ab], direction);
        """
            + projection
        )
    incident = (
        "relative += Jr[q] * z.maps.kinetic_incident.data[world * 22 + q];"
        if kinetic
        else "relative += Jr[q] * c.incident.data[z.maps.starts.data[world * 2 + (q < 16 ? 0 : 1)] + (q < 16 ? q : q - 16)];"
    )
    return (
        r"""
    const int at = world * z.rowkeys.shape[1] + row;
    const int key = z.rowkeys.data[at];
    if (key < 0) {
__PREFIX__
    } else {
        const auto& c = z.contact;
        const int contact = key >> 2, r = key & 3;
        const int sa = c.shape0.data[contact], sb = c.shape1.data[contact];
        const int ba = sa >= 0 ? c.shape_body.data[sa] : -1;
        const int bb = sb >= 0 ? c.shape_body.data[sb] : -1;
        const int aa = c.art0.data[contact], ab = c.art1.data[contact];
        const wp::vec3 normal = -c.normal.data[contact];
        wp::vec3 pa = ba >= 0 ? wp::transform_point(c.body_q.data[ba], c.point0.data[contact]) : c.point0.data[contact];
        wp::vec3 pb = bb >= 0 ? wp::transform_point(c.body_q.data[bb], c.point1.data[contact]) : c.point1.data[contact];
        pa = pa - c.margin0.data[contact] * normal;
        pb = pb + c.margin1.data[contact] * normal;
        const float separation = wp::dot(normal, pa - pb);
        wp::vec3 direction = normal;
        if (r > 0) {
            wp::vec3 t0 = wp::cross(normal, wp::vec3(1.0f, 0.0f, 0.0f));
            if (wp::length_sq(t0) < 1.0e-12f) t0 = wp::cross(normal, wp::vec3(0.0f, 1.0f, 0.0f));
            t0 = wp::normalize(t0);
            direction = r == 1 ? t0 : wp::normalize(wp::cross(normal, t0));
        }
        if (c.shared_anchor != 0 || (r > 0 && c.friction_shared_anchor != 0)) {
            const wp::vec3 anchor = 0.5f * (pa + pb); pa = anchor; pb = anchor;
        }
        float known = 0.0f;
        for (int side = 0; side < 2; ++side) {
            const int body = side == 0 ? ba : bb, art = side == 0 ? aa : ab;
            if (body >= 0 && art >= 0 && c.prescribed.data[art] != 0) {
                const wp::spatial_vector v = c.body_v.data[body];
                const wp::vec3 linear(v[0], v[1], v[2]), angular(v[3], v[4], v[5]);
                const wp::vec3 p = side == 0 ? pa : pb;
                const float value = wp::dot(direction, linear + wp::cross(angular, p - c.origin.data[art]));
                known += side == 0 ? value : -value;
            }
        }
        const float target = -known;
        float mu = 0.0f, restitution = 0.0f; int materials = 0;
        for (int side = 0; side < 2; ++side) {
            const int shape = side == 0 ? sa : sb;
            if (shape >= 0) {
                ++materials; mu += c.material_mu.data[shape];
                const float rest = c.material_restitution.data[shape];
                if (isfinite(rest)) restitution += rest < 0.0f ? 0.0f : (rest > 1.0f ? 1.0f : rest);
            }
        }
        if (materials) { mu /= static_cast<float>(materials); restitution /= static_cast<float>(materials); }
        float friction = mu * c.friction_scale;
        const bool filter = c.friction_pairs_only == 0 || (aa >= 0 && ab >= 0 && c.is_free.data[aa] == 0 && c.is_free.data[ab] == 0);
        if (r > 0 && filter && c.friction_anchor_limit > 0) {
            const int total = wp::min(c.count.data[0], c.point0.shape[0]);
            int rank = 0;
            for (int look = 1; look <= 8; ++look) {
                const int prev = contact - look;
                if (prev < 0 || prev >= total) break;
                if (c.shape0.data[prev] != sa || c.shape1.data[prev] != sb) break;
                ++rank;
            }
            const int next = contact + 1;
            if (rank > 0 || (next < total && c.shape0.data[next] == sa && c.shape1.data[next] == sb)) friction *= 0.5f;
        }
__PROJECTION__
        const float phi = r == 0 ? separation : 0.0f;
        float rhs = -target;
        if (r == 0) {
            rhs += (phi <= 0.0f ? c.bias_scale * c.beta : c.speculative_scale) * phi / c.dt;
            if (restitution > 0.0f) {
                float relative = -target;
                #pragma unroll
                for (int q = 0; q < 22; ++q) { __INCIDENT__ }
                if (relative < -c.restitution_threshold && (phi <= 1.0e-6f || phi + c.dt * relative <= 1.0e-6f))
                    rhs = -target + restitution * relative;
            }
        }
        c.row_type.data[at] = r == 0 ? 0 : 2;
        c.row_parent.data[at] = r == 0 ? -1 : c.slot.data[contact];
        c.row_mu.data[at] = r == 0 ? mu : friction;
        c.row_beta.data[at] = r == 0 ? c.beta : 0.0f;
        c.row_cfm.data[at] = c.cfm;
        c.phi.data[at] = phi; c.target.data[at] = target;
        c.restitution.data[at] = r == 0 ? restitution : 0.0f;
        c.rhs.data[at] = rhs; c.diag.data[at] = c.cfm;
        __PRIVATE_METADATA__
    }
""".replace("__PREFIX__", prefix)
        .replace("__PROJECTION__", projection)
        .replace("__INCIDENT__", incident)
        .replace(
            "__PRIVATE_METADATA__",
            "built_rhs = rhs; built_mu = r == 0 ? mu : friction; built_parent = r == 0 ? -1 : c.slot.data[contact];"
            if kinetic
            else "",
        )
    )


@cache
def get_fallback_kernel():
    """Materialize current physical rows in one pass over fallback worlds only."""
    source = (
        """
    if (counts.data[world] <= 128 && mf_counts.data[world] == 0) return;
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x, stride = blockDim.x;
#else
    const int lane = 0, stride = 1;
#endif
    const int ga = z.maps.groups.data[world * 2], gb = z.maps.groups.data[world * 2 + 1];
    for (int row = lane; row < counts.data[world]; row += stride) {
        float Jr[22];
"""
        + _row_source(kinetic=False)
        + """
        #pragma unroll
        for (int q = 0; q < 22; ++q) {
            if (q < 16) J16.data[(ga * J16.shape[1] + row) * 16 + q] = Jr[q];
            else J6.data[(gb * J6.shape[1] + row) * 6 + q - 16] = Jr[q];
        }
    }
"""
    )

    @wp.func_native(source)
    def native(
        world: int,
        logical_lane: int,
        z: KineticRows,
        counts: wp.array[int],
        mf_counts: wp.array[int],
        J16: wp.array3d[float],
        J6: wp.array3d[float],
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def allegro_kinetic_keyed_fallback(
        z: KineticRows, counts: wp.array[int], mf_counts: wp.array[int], J16: wp.array3d[float], J6: wp.array3d[float]
    ):
        world, lane = wp.tid()
        native(world, lane, z, counts, mf_counts, J16, J6)

    return allegro_kinetic_keyed_fallback


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
        const int row = i;
        const auto& z = kinetic_rows;
        float built_rhs = 0.0f, built_mu = 0.0f;
        int built_parent = -1;
        {{
        {_row_source(kinetic=True)}
        }}
        if (z.rowkeys.data[off_dense + i] >= 0) {{
            const int direction = z.rowkeys.data[off_dense + i] & 3;
            s_rhs[i] = built_rhs;
            s_parent[i] = built_parent;
            s_mu[i] = built_mu;
            const bool admit = i >= dense_lo && row_phase != 2 && row_phase != 3 && row_phase != 5;
            s_kind[i] = admit ? (direction == 0 ? 0 : 1) : -1;
        }}
        float bjv = 0.0f;
        #pragma unroll
        for (int d = 0; d < 22; ++d) {{
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
            node.value = node.value.replace("pgs_solve_parallel_", "pgs_solve_kinetic_keyed_parallel_")
            node.value = node.value.replace(
                "const int row_type = world_row_type.data[off_dense + i] & ",
                "const int row_type = (kinetic_rows.rowkeys.data[off_dense + i] >= 0 ? ((kinetic_rows.rowkeys.data[off_dense + i] & 3) == 0 ? 0 : 2) : world_row_type.data[off_dense + i]) & ",
            )
            # Contact metadata is produced by the row lane below, not an earlier global pass.
            for field, source in (("s_rhs", "rhs_bias"), ("s_parent", "world_row_parent"), ("s_mu", "world_row_mu")):
                node.value = node.value.replace(
                    f"{field}[i] = {source}.data[off_dense + i];",
                    f"{field}[i] = kinetic_rows.rowkeys.data[off_dense + i] < 0 ? {source}.data[off_dense + i] : 0;",
                )
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
        self.keyed_rows = True
        groups, starts, body_map = topology_plan(solver)
        dev, worlds, capacity = solver.model.device, solver.world_count, solver.dense_max_constraints
        self.data, self.maps, self.prefix = KineticRows(), MapInput(), PrefixInput()
        self.data.rowkeys = wp.empty((worlds, capacity), dtype=int, device=dev)
        self.maps.groups = wp.array(groups, dtype=int, device=dev)
        self.maps.starts = wp.array(starts, dtype=int, device=dev)
        self.maps.body_map = wp.array(body_map, dtype=int, device=dev)
        self.maps.inverse = wp.empty((worlds, 100), dtype=float, device=dev)
        self.maps.maps = wp.empty((worlds, 420), dtype=float, device=dev)
        self.maps.kinetic_incident = wp.empty((worlds, 22), dtype=float, device=dev)
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
        self.data.maps, self.data.prefix = m, p
        wp.launch_tiled(get_map_kernel(), dim=[s.world_count], inputs=[m], block_dim=32, device=s.model.device)
        wp.launch_tiled(
            get_prefix_kernel(), dim=[s.world_count], inputs=[p, self.data], block_dim=32, device=s.model.device
        )

    def produce_contacts(self, state_in, state_aug, contacts, dt):
        """Defer current contact construction until final fallback counts are known."""
        self.contact_input = bind(self.solver, state_in, state_aug, contacts, dt, 1.0)
        self.data.contact = self.contact_input

    def finish_rows(self):
        """Publish current direct rows and fully materialize only fallback worlds."""
        s = self.solver
        if self.contact_input is not None:
            wp.launch(
                scatter_contact_keys,
                dim=self.contact_input.point0.shape[0],
                inputs=[self.contact_input, self.data],
                device=s.model.device,
            )
        wp.launch_tiled(
            get_fallback_kernel(),
            dim=[s.world_count],
            inputs=[self.data, s.constraint_count, s.mf_constraint_count, s.J_by_size[16], s.J_by_size[6]],
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
