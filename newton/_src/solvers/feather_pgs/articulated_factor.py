# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental held articulated factors in a common articulation-origin frame.

This private owner supports fixed-base trees with zero or one DOF per joint.
It deliberately retains the existing current FK/inverse-dynamics producer.
All bodies of one factor must use the same spatial origin and coordinate axes.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np
import warp as wp

from ...sim.enums import BodyFlags


@wp.struct
class ArticulatedFactorData:
    art_ids: wp.array[int]
    body_ids: wp.array2d[int]
    parents: wp.array2d[int]
    dof_slots: wp.array2d[int]
    global_dofs: wp.array2d[int]
    counts: wp.array[int]
    S: wp.array3d[float]
    U: wp.array3d[float]
    invD: wp.array2d[float]
    valid: wp.array[int]


@wp.struct
class ArticulatedCurrentData:
    articulation_start: wp.array[int]
    joint_q_start: wp.array[int]
    joint_q: wp.array[float]
    joint_qd: wp.array[float]
    joint_f: wp.array[float]
    spring_stiffness: wp.array[float]
    spring_ref: wp.array[float]
    damping: wp.array[float]
    joint_S: wp.array[wp.spatial_vector]
    body_fb_s: wp.array[wp.spatial_vector]
    body_f_ext: wp.array[wp.spatial_vector]
    body_flags: wp.array[int]
    body_q: wp.array[wp.transform]
    body_com: wp.array[wp.vec3]
    articulation_origin: wp.array[wp.vec3]
    joint_tau: wp.array[float]
    joint_qdd: wp.array[float]


@dataclass
class ArticulatedFactorPlan:
    """Hold immutable tree ownership and the single selected native mapping."""

    data: ArticulatedFactorData
    max_links: int
    dofs: int


def allocate_factor_data(art_ids, body_ids, parents, dof_slots, global_dofs, counts, device):
    """Validate parent order and allocate stable factor buffers before capture."""
    arrays = [np.asarray(x, dtype=np.int32) for x in (art_ids, body_ids, parents, dof_slots, global_dofs, counts)]
    arts, bodies, parent, slots, globals_, count = arrays
    if bodies.ndim != 2 or arts.shape != (len(bodies),) or count.shape != arts.shape:
        raise ValueError("Articulated factor ownership shapes differ")
    if any(x.shape != bodies.shape for x in (parent, slots, globals_)):
        raise ValueError("Articulated factor tree shapes differ")
    if np.any(arts < 0) or len(np.unique(arts)) != len(arts):
        raise ValueError("Articulated factors require unique nonnegative articulation owners")
    width = bodies.shape[1]
    for a, length in enumerate(count):
        if not 0 < length <= width:
            raise ValueError("Articulated factor link count exceeds its fixed allocation")
        ids = bodies[a, :length]
        if np.any(ids < 0) or len(np.unique(ids)) != length:
            raise ValueError("Each factor must own distinct body links")
        if parent[a, 0] != -1 or slots[a, 0] != -1:
            raise ValueError("Articulated factor root must be fixed")
        if np.any(parent[a, 1:length] < 0) or np.any(parent[a, 1:length] >= np.arange(1, length)):
            raise ValueError("Articulated factor parents must precede their children")
        moving = slots[a, :length] >= 0
        active_slots = slots[a, :length][moving]
        if not np.array_equal(np.sort(active_slots), np.arange(len(active_slots))):
            raise ValueError("Articulated factor DOFs must cover a contiguous unique range")
        if np.any(globals_[a, :length][moving] < 0) or np.any(globals_[a, :length][~moving] != -1):
            raise ValueError("Fixed links have no generalized DOF")
    data = ArticulatedFactorData()
    for name, values in zip(
        ("art_ids", "body_ids", "parents", "dof_slots", "global_dofs", "counts"), arrays, strict=True
    ):
        setattr(data, name, wp.array(values, dtype=int, device=device))
    data.S = wp.zeros((*bodies.shape, 6), dtype=float, device=device)
    data.U = wp.zeros_like(data.S)
    data.invD = wp.zeros(bodies.shape, dtype=float, device=device)
    data.valid = wp.zeros(len(arts), dtype=int, device=device)
    return data


def build_factor_plan(solver, size: int) -> ArticulatedFactorPlan | None:
    """Admit the complete original size group, without task-name assumptions."""
    model = solver.model
    arts = solver.group_to_art[size].numpy()
    starts = model.articulation_start.numpy()
    ends = solver.articulation_joint_end.numpy()
    child = model.joint_child.numpy()
    parent = model.joint_parent.numpy()
    dimensions = model.joint_dof_dim.numpy()
    qd_start = model.joint_qd_start.numpy()
    dof_start = solver.articulation_dof_start.numpy()
    lengths = ends[arts] - starts[arts]
    if len(arts) == 0 or np.any(lengths <= 0):
        return None
    width = int(lengths.max())
    bodies = np.full((len(arts), width), -1, dtype=np.int32)
    parents = np.full_like(bodies, -1)
    slots = np.full_like(bodies, -1)
    globals_ = np.full_like(bodies, -1)
    for group, art in enumerate(arts):
        body_to_link = {}
        for local, joint in enumerate(range(int(starts[art]), int(ends[art]))):
            body = int(child[joint])
            n = int(dimensions[joint].sum())
            if body < 0 or body in body_to_link or n not in (0, 1):
                return None
            if local == 0:
                if parent[joint] >= 0 or n != 0:
                    return None
            elif int(parent[joint]) not in body_to_link:
                return None
            bodies[group, local] = body
            parents[group, local] = body_to_link.get(int(parent[joint]), -1)
            if n:
                globals_[group, local] = int(qd_start[joint])
                slots[group, local] = int(qd_start[joint] - dof_start[art])
            body_to_link[body] = local
        if not np.array_equal(np.sort(slots[group][slots[group] >= 0]), np.arange(size)):
            return None
    data = allocate_factor_data(arts, bodies, parents, slots, globals_, lengths, model.device)
    return ArticulatedFactorPlan(data, width, size)


_EXECUTION = r"""
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x & 31;
    constexpr int WORKERS = 32;
    #define FACTOR_SHARED __shared__
    #define FACTOR_SYNC() __syncwarp(0xffffffffu)
#else
    const int lane = 0;
    constexpr int WORKERS = 1;
    #define FACTOR_SHARED
    #define FACTOR_SYNC() ((void)0)
#endif
"""

_CLEANUP = "\n#undef FACTOR_SHARED\n#undef FACTOR_SYNC\n"


def factor_source(max_links: int, dofs: int, *, live_inputs: bool = False) -> str:
    """Generate one cooperative common-frame tree elimination, no edge congruences."""
    if not 0 < dofs < max_links <= 128:
        raise ValueError("Expected a bounded fixed-root zero/one-DOF tree")
    source = (
        _EXECUTION
        + f"\nconstexpr int LINKS = {max_links};\n"
        + r"""
    const int count = factor.counts.data[art];
    const int owner = factor.art_ids.data[art];
    if (mass_mask.data[owner] == 0) return;
    FACTOR_SHARED float inertia[LINKS * 36];
    FACTOR_SHARED int good;
    if (lane == 0) { good = 1; factor.valid.data[art] = 0; }
    for (int item = lane; item < count * 36; item += WORKERS) {
        const int link = item / 36, element = item % 36;
        const int body = factor.body_ids.data[art * LINKS + link];
        inertia[item] = body_I.data[body].data[element / 6][element % 6];
    }
    for (int item = lane; item < count * 6; item += WORKERS) {
        const int link = item / 6, component = item % 6;
        const int dof = factor.global_dofs.data[art * LINKS + link];
        factor.S.data[art * LINKS * 6 + item] = dof >= 0 ? joint_S.data[dof][component] : 0.0f;
        factor.U.data[art * LINKS * 6 + item] = 0.0f;
    }
    FACTOR_SYNC();
    for (int link = count - 1; link >= 0; --link) {
        const int index = art * LINKS + link;
        const int dof = factor.global_dofs.data[index];
        const int parent = factor.parents.data[index];
        if (dof >= 0) {
            for (int component = lane; component < 6; component += WORKERS) {
                float value = 0.0f;
                for (int j = 0; j < 6; ++j)
                    value += inertia[link * 36 + component * 6 + j] * factor.S.data[index * 6 + j];
                factor.U.data[index * 6 + component] = value;
            }
            FACTOR_SYNC();
            if (lane == 0) {
                float d = diagonal.data[dof];
                for (int j = 0; j < 6; ++j)
                    d += factor.S.data[index * 6 + j] * factor.U.data[index * 6 + j];
                if (!(d > 0.0f) || !isfinite(d)) { good = 0; factor.invD.data[index] = 0.0f; }
                else factor.invD.data[index] = 1.0f / d;
            }
        } else if (lane == 0) factor.invD.data[index] = 0.0f;
        FACTOR_SYNC();
        if (parent >= 0) {
            for (int element = lane; element < 36; element += WORKERS) {
                float reduced = inertia[link * 36 + element];
                if (dof >= 0)
                    reduced -= factor.U.data[index * 6 + element / 6] * factor.invD.data[index]
                        * factor.U.data[index * 6 + element % 6];
                inertia[parent * 36 + element] += reduced;
            }
        }
        FACTOR_SYNC();
    }
    if (lane == 0) factor.valid.data[art] = good;
    FACTOR_SYNC();
"""
        + _CLEANUP
    )
    if live_inputs:
        source = source.replace(
            "inertia[item] = body_I.data[body].data[element / 6][element % 6];",
            r"""
        float value = 0.0f;
        const int r = element / 6, c = element % 6;
        if (compact_source == 0) value = body_I.data[body].data[r][c];
        else if (r < 3 && c < 3) {
            if (r == c) value = body_mass.data[body];
        } else if (r >= 3 && c >= 3)
            value = inertia_terms.data[body * 12 + 3 + (r - 3) * 3 + c - 3];
        else {
            const int cr = r % 3, cc = c % 3;
            float cross = 0.0f;
            if (cr == 0 && cc == 1) cross = -inertia_terms.data[body * 12 + 2];
            if (cr == 0 && cc == 2) cross =  inertia_terms.data[body * 12 + 1];
            if (cr == 1 && cc == 0) cross =  inertia_terms.data[body * 12 + 2];
            if (cr == 1 && cc == 2) cross = -inertia_terms.data[body * 12];
            if (cr == 2 && cc == 0) cross = -inertia_terms.data[body * 12 + 1];
            if (cr == 2 && cc == 1) cross =  inertia_terms.data[body * 12];
            value = (r < 3 ? -body_mass.data[body] : body_mass.data[body]) * cross;
        }
        inertia[item] = value;
""",
        ).replace(
            "float d = diagonal.data[dof];",
            f"""float d = grouped_R.data[art * {dofs} + factor.dof_slots.data[index]];
                const int drive_row = drive_row_by_dof.data[dof];
                if (drive_row >= 0) {{
                    const float K = drive_K.data[drive_row];
                    if (K > 0.0f) d += K;
                }}""",
        )
    return source


@cache
def get_factor_kernel(max_links: int, dofs: int):
    """Factor full inertias; launch_tiled dim=[A], block_dim=32 is required."""

    @wp.func_native(factor_source(max_links, dofs))
    def factor_native(
        factor: ArticulatedFactorData,
        art: int,
        body_I: wp.array[wp.spatial_matrix],
        joint_S: wp.array[wp.spatial_vector],
        diagonal: wp.array[float],
        mass_mask: wp.array[int],
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def factor_kernel(
        factor: ArticulatedFactorData,
        body_I: wp.array[wp.spatial_matrix],
        joint_S: wp.array[wp.spatial_vector],
        diagonal: wp.array[float],
        mass_mask: wp.array[int],
    ):
        art, _lane = wp.tid()
        factor_native(factor, art, body_I, joint_S, diagonal, mass_mask)

    return factor_kernel


@cache
def get_live_factor_kernel(max_links: int, dofs: int):
    """Consume the original current full/compact inertia and drive owners.

    Launch with ``launch_tiled(dim=[A], block_dim=32)``. A zero original
    articulation mass mask preserves the entire held factor without reading
    either inertia representation. Global compact refresh and requested full
    refresh are selected by the caller's existing producer epoch.
    """

    @wp.func_native(factor_source(max_links, dofs, live_inputs=True))
    def factor_native(
        factor: ArticulatedFactorData,
        art: int,
        body_I: wp.array[wp.spatial_matrix],
        joint_S: wp.array[wp.spatial_vector],
        grouped_R: wp.array2d[float],
        drive_row_by_dof: wp.array[int],
        drive_K: wp.array[float],
        mass_mask: wp.array[int],
        inertia_terms: wp.array2d[float],
        body_mass: wp.array[float],
        compact_source: int,
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def factor_kernel(
        factor: ArticulatedFactorData,
        body_I: wp.array[wp.spatial_matrix],
        joint_S: wp.array[wp.spatial_vector],
        grouped_R: wp.array2d[float],
        drive_row_by_dof: wp.array[int],
        drive_K: wp.array[float],
        mass_mask: wp.array[int],
        inertia_terms: wp.array2d[float],
        body_mass: wp.array[float],
        compact_source: int,
    ):
        art, _lane = wp.tid()
        factor_native(
            factor,
            art,
            body_I,
            joint_S,
            grouped_R,
            drive_row_by_dof,
            drive_K,
            mass_mask,
            inertia_terms,
            body_mass,
            compact_source,
        )

    return factor_kernel


def apply_source(max_links: int, dofs: int, *, predictor: bool = False) -> str:
    """Apply the held factor to one generalized RHS with one warp per owner."""
    if not 0 < dofs < max_links <= 128:
        raise ValueError("Expected a bounded fixed-root zero/one-DOF tree")
    load = (
        "rhs.data[factor.global_dofs.data[index]]"
        if predictor
        else f"rhs.data[(art * rhs.shape[1] + row) * {dofs} + slot]"
    )
    store = (
        "result.data[factor.global_dofs.data[index]]"
        if predictor
        else f"result.data[(art * result.shape[1] + row) * {dofs} + slot]"
    )
    return (
        _EXECUTION
        + f"\nconstexpr int LINKS = {max_links};\n"
        + r"""
    const int count = factor.counts.data[art];
    FACTOR_SHARED float force[LINKS * 6], acceleration[LINKS * 6], reduced_rhs[LINKS];
    if (factor.valid.data[art] == 0) {
        union { unsigned int bits; float value; } invalid;
        invalid.bits = 0x7fc00000u;
        for (int link = lane; link < count; link += WORKERS) {
            const int index = art * LINKS + link;
            const int slot = factor.dof_slots.data[index];
            if (slot >= 0) __STORE__ = invalid.value;
        }
        FACTOR_SYNC();
        return;
    }
    for (int i = lane; i < count * 6; i += WORKERS) {
        force[i] = 0.0f;
        acceleration[i] = 0.0f;
    }
    FACTOR_SYNC();
    for (int link = count - 1; link >= 0; --link) {
        const int index = art * LINKS + link;
        const int slot = factor.dof_slots.data[index];
        const int parent = factor.parents.data[index];
        if (lane == 0) {
            float u = slot >= 0 ? __LOAD__ : 0.0f;
            if (slot >= 0) for (int j = 0; j < 6; ++j)
                u -= factor.S.data[index * 6 + j] * force[link * 6 + j];
            reduced_rhs[link] = u;
        }
        FACTOR_SYNC();
        if (parent >= 0) {
            for (int j = lane; j < 6; j += WORKERS) {
                float p = force[link * 6 + j];
                if (slot >= 0) p += factor.U.data[index * 6 + j] * factor.invD.data[index] * reduced_rhs[link];
                force[parent * 6 + j] += p;
            }
        }
        FACTOR_SYNC();
    }
    for (int link = 0; link < count; ++link) {
        const int index = art * LINKS + link;
        const int slot = factor.dof_slots.data[index];
        const int parent = factor.parents.data[index];
        if (lane == 0) {
            float u = reduced_rhs[link];
            if (slot >= 0 && parent >= 0) for (int j = 0; j < 6; ++j)
                u -= factor.U.data[index * 6 + j] * acceleration[parent * 6 + j];
            reduced_rhs[link] = slot >= 0 ? u * factor.invD.data[index] : 0.0f;
            if (slot >= 0) __STORE__ = reduced_rhs[link];
        }
        FACTOR_SYNC();
        for (int j = lane; j < 6; j += WORKERS) {
            float a = parent >= 0 ? acceleration[parent * 6 + j] : 0.0f;
            if (slot >= 0) a += factor.S.data[index * 6 + j] * reduced_rhs[link];
            acceleration[link * 6 + j] = a;
        }
        FACTOR_SYNC();
    }
""".replace("__LOAD__", load).replace("__STORE__", store)
        + _CLEANUP
    )


@cache
def get_apply_function(max_links: int, dofs: int):
    """Return an all-32-lane function ending with a warp synchronization.

    Exactly one articulation/RHS owns the caller CTA. The result is written in
    original size-group joint coordinates; callers may bridge it after return.
    """

    @wp.func_native(apply_source(max_links, dofs))
    def apply_native(
        factor: ArticulatedFactorData,
        art: int,
        row: int,
        rhs: wp.array3d[float],
        result: wp.array3d[float],
    ): ...

    return apply_native


@cache
def get_apply_kernel(max_links: int, dofs: int):
    """Apply active rows with launch_tiled dim=[A,R], block_dim=32."""
    apply_native = get_apply_function(max_links, dofs)

    @wp.kernel(module="unique", enable_backward=False)
    def apply_kernel(
        factor: ArticulatedFactorData,
        rhs: wp.array3d[float],
        result: wp.array3d[float],
        row_counts: wp.array[int],
    ):
        art, row, _lane = wp.tid()
        if row < row_counts[art]:
            apply_native(factor, art, row, rhs, result)

    return apply_kernel


@cache
def get_predictor_kernel(max_links: int, dofs: int):
    """Apply global current tau with launch_tiled dim=[A], block_dim=32."""

    @wp.func_native(apply_source(max_links, dofs, predictor=True))
    def predictor_native(factor: ArticulatedFactorData, art: int, rhs: wp.array[float], result: wp.array[float]): ...

    @wp.kernel(module="unique", enable_backward=False)
    def predictor(factor: ArticulatedFactorData, rhs: wp.array[float], result: wp.array[float]):
        art, _lane = wp.tid()
        predictor_native(factor, art, rhs, result)

    return predictor


def fused_dynamics_source(max_links: int, dofs: int) -> str:
    """Fuse current inverse dynamics and held response into one reverse tree pass."""
    if not 0 < dofs < max_links <= 128:
        raise ValueError("Expected a bounded fixed-root zero/one-DOF tree")
    return (
        _EXECUTION
        + f"\nconstexpr int LINKS = {max_links}, DOFS = {dofs}, KINEMATIC = {int(BodyFlags.KINEMATIC)};\n"
        + r"""
#if !defined(__CUDA_ARCH__)
    // CPU tile lanes execute serially; the native CPU body is already the
    // complete tree and must not add the current torque bucket repeatedly.
    if (logical_lane != 0) return;
#endif
    const int owner = factor.art_ids.data[art];
    const int count = factor.counts.data[art];
    const bool refresh = mass_mask.data[owner] != 0;
    const int joint_start = current.articulation_start.data[owner];
    FACTOR_SHARED float inertia[LINKS * 36];
    FACTOR_SHARED float force[LINKS * 6], response_force[LINKS * 6];
    FACTOR_SHARED float motion[LINKS * 6], U[LINKS * 6];
    FACTOR_SHARED float inverse[LINKS], reduced_rhs[LINKS];
    FACTOR_SHARED int good;
    if (lane == 0) good = refresh ? 1 : factor.valid.data[art];

    // Current external/bias preparation is independent across links. These
    // wrenches retain the CURRENT origin; the response wrench stays HELD.
    for (int link = lane; link < count; link += WORKERS) {
        const int body = factor.body_ids.data[art * LINKS + link];
        wp::spatial_vector external = wp::spatial_vector();
        if ((current.body_flags.data[body] & KINEMATIC) == 0)
            external = current.body_f_ext.data[body];
        const wp::vec3 f(external[0], external[1], external[2]);
        const wp::vec3 torque(external[3], external[4], external[5]);
        const wp::vec3 com = wp::transform_point(current.body_q.data[body], current.body_com.data[body]);
        const wp::vec3 at_origin = torque + wp::cross(com - current.articulation_origin.data[owner], f);
        for (int j = 0; j < 6; ++j)
            force[link * 6 + j] = current.body_fb_s.data[body][j] - (j < 3 ? f[j] : at_origin[j - 3]);
        inverse[link] = refresh ? 0.0f : factor.invD.data[art * LINKS + link];
    }
    for (int item = lane; item < count * 6; item += WORKERS) {
        const int link = item / 6, component = item % 6;
        const int dof = factor.global_dofs.data[art * LINKS + link];
        motion[item] = refresh ? (dof >= 0 ? current.joint_S.data[dof][component] : 0.0f)
                               : factor.S.data[art * LINKS * 6 + item];
        U[item] = refresh ? 0.0f : factor.U.data[art * LINKS * 6 + item];
        response_force[item] = 0.0f;
    }
    if (refresh) for (int item = lane; item < count * 36; item += WORKERS) {
        const int link = item / 36, element = item % 36;
        const int body = factor.body_ids.data[art * LINKS + link];
        const int r = element / 6, c = element % 6;
        float value = 0.0f;
        if (compact_source == 0) value = body_I.data[body].data[r][c];
        else if (r < 3 && c < 3) {
            if (r == c) value = body_mass.data[body];
        } else if (r >= 3 && c >= 3)
            value = inertia_terms.data[body * 12 + 3 + (r - 3) * 3 + c - 3];
        else {
            const int cr = r % 3, cc = c % 3;
            float cross = 0.0f;
            if (cr == 0 && cc == 1) cross = -inertia_terms.data[body * 12 + 2];
            if (cr == 0 && cc == 2) cross =  inertia_terms.data[body * 12 + 1];
            if (cr == 1 && cc == 0) cross =  inertia_terms.data[body * 12 + 2];
            if (cr == 1 && cc == 2) cross = -inertia_terms.data[body * 12];
            if (cr == 2 && cc == 0) cross = -inertia_terms.data[body * 12 + 1];
            if (cr == 2 && cc == 1) cross =  inertia_terms.data[body * 12];
            value = (r < 3 ? -body_mass.data[body] : body_mass.data[body]) * cross;
        }
        inertia[item] = value;
    }
    FACTOR_SYNC();
    for (int link = count - 1; link >= 0; --link) {
        const int index = art * LINKS + link;
        const int dof = factor.global_dofs.data[index];
        const int slot = factor.dof_slots.data[index];
        const int parent = factor.parents.data[index];
        if (refresh && dof >= 0) {
            for (int component = lane; component < 6; component += WORKERS) {
                float value = 0.0f;
                for (int j = 0; j < 6; ++j)
                    value += inertia[link * 36 + component * 6 + j] * motion[link * 6 + j];
                U[link * 6 + component] = value;
            }
            FACTOR_SYNC();
            float pivot = 0.0f;
#if defined(__CUDA_ARCH__)
            if (lane < 6) pivot = motion[link * 6 + lane] * U[link * 6 + lane];
            pivot += __shfl_down_sync(0xffffffffu, pivot, 4, 8);
            pivot += __shfl_down_sync(0xffffffffu, pivot, 2, 8);
            pivot += __shfl_down_sync(0xffffffffu, pivot, 1, 8);
#else
            for (int j = 0; j < 6; ++j) pivot += motion[link * 6 + j] * U[link * 6 + j];
#endif
            if (lane == 0) {
                float d = grouped_R.data[art * DOFS + slot];
                const int drive_row = drive_row_by_dof.data[dof];
                if (drive_row >= 0) {
                    const float K = drive_K.data[drive_row];
                    if (K > 0.0f) d += K;
                }
                d += pivot;
                if (!(d > 0.0f) || !isfinite(d)) { good = 0; inverse[link] = 0.0f; }
                else inverse[link] = 1.0f / d;
            }
        }
        FACTOR_SYNC();

        // Two independent eight-lane projections shorten the single-lane
        // chain, without identifying current and held axes on reuse calls.
        float current_dot = 0.0f, held_dot = 0.0f;
        if (dof >= 0) {
#if defined(__CUDA_ARCH__)
            float value = 0.0f;
            if (lane < 6) value = current.joint_S.data[dof][lane] * force[link * 6 + lane];
            if (lane >= 8 && lane < 14) value = motion[link * 6 + lane - 8] * response_force[link * 6 + lane - 8];
            value += __shfl_down_sync(0xffffffffu, value, 4, 8);
            value += __shfl_down_sync(0xffffffffu, value, 2, 8);
            value += __shfl_down_sync(0xffffffffu, value, 1, 8);
            current_dot = __shfl_sync(0xffffffffu, value, 0);
            held_dot = __shfl_sync(0xffffffffu, value, 8);
#else
            for (int j = 0; j < 6; ++j) {
                current_dot += current.joint_S.data[dof][j] * force[link * 6 + j];
                held_dot += motion[link * 6 + j] * response_force[link * 6 + j];
            }
#endif
        }
        if (lane == 0) {
            float u = 0.0f;
            if (dof >= 0) {
                const int q = current.joint_q_start.data[joint_start + link];
                float passive = current.spring_stiffness.data[dof]
                    * (current.spring_ref.data[dof] - current.joint_q.data[q]);
                passive -= current.damping.data[dof] * current.joint_qd.data[dof];
                float tau = -current_dot + current.joint_f.data[dof] + passive;
                tau += current.joint_tau.data[dof];
                current.joint_tau.data[dof] = tau;
                u = tau - held_dot;
            }
            reduced_rhs[link] = u;
        }
        FACTOR_SYNC();
        if (parent >= 0) {
            for (int j = lane; j < 6; j += WORKERS) {
                force[parent * 6 + j] += force[link * 6 + j];
                float p = response_force[link * 6 + j];
                if (dof >= 0) p += U[link * 6 + j] * inverse[link] * reduced_rhs[link];
                response_force[parent * 6 + j] += p;
            }
            if (refresh) for (int element = lane; element < 36; element += WORKERS) {
                float value = inertia[link * 36 + element];
                if (dof >= 0) value -= U[link * 6 + element / 6] * inverse[link] * U[link * 6 + element % 6];
                inertia[parent * 36 + element] += value;
            }
        }
        FACTOR_SYNC();
    }
    if (refresh) {
        for (int item = lane; item < count * 6; item += WORKERS) {
            factor.S.data[art * LINKS * 6 + item] = motion[item];
            factor.U.data[art * LINKS * 6 + item] = U[item];
        }
        for (int link = lane; link < count; link += WORKERS)
            factor.invD.data[art * LINKS + link] = inverse[link];
        if (lane == 0) factor.valid.data[art] = good;
    }
    FACTOR_SYNC();
    if (good == 0) {
        union { unsigned int bits; float value; } invalid;
        invalid.bits = 0x7fc00000u;
        for (int link = lane; link < count; link += WORKERS) {
            const int dof = factor.global_dofs.data[art * LINKS + link];
            if (dof >= 0) current.joint_qdd.data[dof] = invalid.value;
        }
        FACTOR_SYNC();
        return;
    }
    // Current subtree wrenches have no remaining readers. Reuse that complete
    // private lifetime for held accelerations, parent-first, not global scratch.
    for (int link = 0; link < count; ++link) {
        const int index = art * LINKS + link;
        const int dof = factor.global_dofs.data[index];
        const int parent = factor.parents.data[index];
        float projection = 0.0f;
        if (dof >= 0 && parent >= 0) {
#if defined(__CUDA_ARCH__)
            if (lane < 6) projection = U[link * 6 + lane] * force[parent * 6 + lane];
            projection += __shfl_down_sync(0xffffffffu, projection, 4, 8);
            projection += __shfl_down_sync(0xffffffffu, projection, 2, 8);
            projection += __shfl_down_sync(0xffffffffu, projection, 1, 8);
#else
            for (int j = 0; j < 6; ++j) projection += U[link * 6 + j] * force[parent * 6 + j];
#endif
        }
        if (lane == 0) {
            const float qdd = dof >= 0 ? (reduced_rhs[link] - projection) * inverse[link] : 0.0f;
            reduced_rhs[link] = qdd;
            if (dof >= 0) current.joint_qdd.data[dof] = qdd;
        }
        FACTOR_SYNC();
        for (int j = lane; j < 6; j += WORKERS) {
            float value = parent >= 0 ? force[parent * 6 + j] : 0.0f;
            if (dof >= 0) value += motion[link * 6 + j] * reduced_rhs[link];
            force[link * 6 + j] = value;
        }
        FACTOR_SYNC();
    }
"""
        + _CLEANUP
    )


@cache
def get_fused_dynamics_kernel(max_links: int, dofs: int):
    """Refresh held factors and compute current tau/qdd in one owned tree pass.

    Launch tiled ``[groups]`` with block_dim=32 after current FK and original
    drive preparation. ``current.joint_tau`` must hold this call's original
    explicit-drive bucket on entry. The selected old RNEA, factor and predictor
    must not also execute. No selected ``body_ft_s`` storage is read or written.
    """

    @wp.func_native(fused_dynamics_source(max_links, dofs))
    def fused_native(
        factor: ArticulatedFactorData,
        art: int,
        logical_lane: int,
        current: ArticulatedCurrentData,
        body_I: wp.array[wp.spatial_matrix],
        grouped_R: wp.array2d[float],
        drive_row_by_dof: wp.array[int],
        drive_K: wp.array[float],
        mass_mask: wp.array[int],
        inertia_terms: wp.array2d[float],
        body_mass: wp.array[float],
        compact_source: int,
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def fused_dynamics(
        factor: ArticulatedFactorData,
        current: ArticulatedCurrentData,
        body_I: wp.array[wp.spatial_matrix],
        grouped_R: wp.array2d[float],
        drive_row_by_dof: wp.array[int],
        drive_K: wp.array[float],
        mass_mask: wp.array[int],
        inertia_terms: wp.array2d[float],
        body_mass: wp.array[float],
        compact_source: int,
    ):
        art, _lane = wp.tid()
        fused_native(
            factor,
            art,
            _lane,
            current,
            body_I,
            grouped_R,
            drive_row_by_dof,
            drive_K,
            mass_mask,
            inertia_terms,
            body_mass,
            compact_source,
        )

    return fused_dynamics
