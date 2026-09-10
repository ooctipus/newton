# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Level-synchronous articulation dynamics with a lane group per articulation.

The serial FeatherPGS kernels visit an articulation's joints in one thread. At 16k
articulations that is 512 warps for the whole GPU and every warp is latency-bound on its
own chain. Here a group of ``lanes_per_articulation`` lanes owns one articulation and
sweeps the joint tree level by level (root to leaves for kinematics, leaves to root for the
backward passes), so several articulations share a warp and the joints of one level run in
parallel. Per-joint math is the existing ``compute_link_transform`` /
``compute_link_kinematics`` / ``jcalc_tau`` functions, and children are accumulated onto
their parent in the same order as the serial kernels (descending joint index), so results
match the serial kernels to the last bit.
"""

from __future__ import annotations

import os

import numpy as np
import warp as wp

from ...math.spatial import transform_twist
from .kernels import (
    BodyFlags,
    JointType,
    compute_link_kinematics,
    compute_link_transform,
    jcalc_tau,
    jcalc_transform,
    spatial_cross,
)


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    __syncwarp();
#endif
"""
)
def warp_sync():
    """Warp-level barrier with memory ordering (no-op on CPU)."""
    ...


class GroupedTopology:
    """Per-articulation level structure of the joint tree, flattened for the kernels."""

    def __init__(self, model, articulation_joint_end: np.ndarray, device, body_X_com=None):
        self._body_X_com = body_X_com
        art_start = model.articulation_start.numpy().astype(np.int64)
        joint_parent = model.joint_parent.numpy().astype(np.int64)
        joint_child = model.joint_child.numpy().astype(np.int64)
        art_count = int(model.articulation_count)
        joint_end = np.asarray(articulation_joint_end, dtype=np.int64)
        body_to_joint = {}
        depth = np.zeros(joint_parent.shape[0], dtype=np.int64)
        level_local: list[int] = []
        offsets = []
        children_offsets: list[int] = []
        children: list[int] = []
        max_levels = 0
        max_width = 1
        for art in range(art_count):
            start, end = int(art_start[art]), int(joint_end[art])
            body_to_joint.clear()
            for j in range(start, end):
                body_to_joint[int(joint_child[j])] = j
            levels: dict[int, list[int]] = {}
            for j in range(start, end):
                p = int(joint_parent[j])
                pj = body_to_joint.get(p, -1) if p >= 0 else -1
                d = 0 if pj < 0 or pj < start else int(depth[pj]) + 1
                depth[j] = d
                levels.setdefault(d, []).append(j - start)
            n_levels = (max(levels) + 1) if levels else 0
            max_levels = max(max_levels, n_levels)
            row = [len(level_local)]
            for lv in range(n_levels):
                js = levels.get(lv, [])
                max_width = max(max_width, len(js))
                level_local.extend(js)
                row.append(len(level_local))
            offsets.append(row)
            # Children of each joint in descending joint order (the serial accumulation order).
            for j in range(start, end):
                children_offsets.append(len(children))
                body = int(joint_child[j])
                kids = [c for c in range(end - 1, start - 1, -1) if int(joint_parent[c]) == body]
                children.extend(kids)
        children_offsets.append(len(children))
        # Articulation templates: articulations whose per-joint constants are bit-identical share
        # one constant table, so a warp of articulations reads each constant once (broadcast).
        self.template_count = 0
        try:
            self._build_templates(model, art_start, joint_end, joint_child, depth, device)
        except Exception as exc:  # diagnostics only; template kernels stay disabled
            self.template_count = -1
            self.template_error = repr(exc)
        lanes_env = os.environ.get("FEATHER_PGS_GROUP_LANES")
        print(
            f"[grouped-topology] articulations {art_count}, templates {self.template_count}, max levels {max_levels}, "
            f"max level width {max_width}",
            flush=True,
        )
        self.max_levels = int(max_levels)
        self.max_width = int(max_width)
        self.max_joints = int(max(int(joint_end[a] - art_start[a]) for a in range(art_count)) if art_count else 0)
        lanes = 1
        while lanes < self.max_width:
            lanes *= 2
        lanes = min(32, lanes * 2)  # twice the widest level measured best (AnymalD: 8)
        if lanes_env:
            lanes = int(lanes_env)
        self.lanes_per_articulation = int(min(32, max(1, lanes)))
        self.articulations_per_block = 32 // self.lanes_per_articulation
        self.blocks = (art_count + self.articulations_per_block - 1) // self.articulations_per_block
        off = np.zeros((art_count, self.max_levels + 1), dtype=np.int32)
        for art, row in enumerate(offsets):
            off[art, : len(row)] = row
            off[art, len(row) :] = row[-1]
        self.art_level_offsets = wp.array(off, dtype=wp.int32, device=device)
        self.level_joint_local = wp.array(np.asarray(level_local or [0], dtype=np.int32), dtype=wp.int32, device=device)
        self.children_offsets = wp.array(np.asarray(children_offsets, dtype=np.int32), dtype=wp.int32, device=device)
        self.children = wp.array(np.asarray(children or [0], dtype=np.int32), dtype=wp.int32, device=device)

    def _build_templates(self, model, art_start, joint_end, joint_child, depth, device) -> None:
        jt = model.joint_type.numpy()
        jd = model.joint_dof_dim.numpy()
        xp = model.joint_X_p.numpy()
        xc = model.joint_X_c.numpy()
        ax = model.joint_axis.numpy()
        qds = model.joint_qd_start.numpy().astype(np.int64)
        qs = model.joint_q_start.numpy().astype(np.int64)
        jp = model.joint_parent.numpy().astype(np.int64)
        art_count = int(model.articulation_count)
        n_dofs_total = int(ax.shape[0])
        templates: dict[bytes, int] = {}
        self.template_of_art = np.zeros(art_count, dtype=np.int32)
        art_body_base = np.zeros(art_count, dtype=np.int32)
        art_q_base = np.zeros(art_count, dtype=np.int32)
        art_qd_base = np.zeros(art_count, dtype=np.int32)
        tpl_rows = []  # per template: dict of flat arrays
        for art in range(art_count):
            start, end = int(art_start[art]), int(joint_end[art])
            if end <= start:
                raise ValueError("empty articulation")
            bodies = joint_child[start:end]
            body_base = int(bodies.min())
            q_base = int(qs[start])
            qd_base = int(qds[start])
            d_end = int(qds[end]) if end < qds.shape[0] else n_dofs_total
            body_to_local = {int(b): k for k, b in enumerate(bodies)}
            parent_local = np.array(
                [body_to_local.get(int(jp[j]), -1) if jp[j] >= 0 else -1 for j in range(start, end)], dtype=np.int32
            )
            # Root joints carry the articulation's world placement in X_p: per articulation, read from the model at
            # run time; the template stores identity so identical robots at different positions share a template.
            xp_tpl = np.array(xp[start:end], copy=True)
            for k_root in np.flatnonzero(parent_local < 0):
                xp_tpl[k_root] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=xp_tpl.dtype)
            key = b"".join(
                [
                    jt[start:end].tobytes(),
                    jd[start:end].tobytes(),
                    xp_tpl.tobytes(),
                    xc[start:end].tobytes(),
                    (qs[start:end] - q_base).astype(np.int32).tobytes(),
                    (qds[start:end] - qd_base).astype(np.int32).tobytes(),
                    ax[qd_base:d_end].tobytes(),
                    (bodies - body_base).astype(np.int32).tobytes(),
                    parent_local.tobytes(),
                    depth[start:end].astype(np.int32).tobytes(),
                ]
            )
            t = templates.get(key)
            if t is None:
                t = len(templates)
                templates[key] = t
                tpl_rows.append(
                    {
                        "n": end - start,
                        "type": jt[start:end].astype(np.int32),
                        "lin": jd[start:end, 0].astype(np.int32),
                        "ang": jd[start:end, 1].astype(np.int32),
                        "q_off": (qs[start:end] - q_base).astype(np.int32),
                        "qd_off": (qds[start:end] - qd_base).astype(np.int32),
                        "parent_local": parent_local,
                        "child_off": (bodies - body_base).astype(np.int32),
                        "X_p": xp_tpl,
                        "X_c": xc[start:end],
                        "axis": ax[qd_base:d_end],
                        "depth": depth[start:end].astype(np.int32),
                    }
                )
            self.template_of_art[art] = t
            art_body_base[art] = body_base
            art_q_base[art] = q_base
            art_qd_base[art] = qd_base
        self.template_count = len(templates)
        # Flatten templates.
        joint_offsets = [0]
        dof_offsets = [0]
        for row in tpl_rows:
            joint_offsets.append(joint_offsets[-1] + int(row["n"]))
            dof_offsets.append(dof_offsets[-1] + int(row["axis"].shape[0]))
        cat = lambda k: np.concatenate([row[k] for row in tpl_rows])  # noqa: E731
        self.tpl_joint_offset = wp.array(np.asarray(joint_offsets, dtype=np.int32), dtype=wp.int32, device=device)
        self.tpl_dof_offset = wp.array(np.asarray(dof_offsets, dtype=np.int32), dtype=wp.int32, device=device)
        self.tpl_type_host = cat("type")
        self.tpl_type = wp.array(self.tpl_type_host, dtype=wp.int32, device=device)
        self.tpl_lin = wp.array(cat("lin"), dtype=wp.int32, device=device)
        self.tpl_ang = wp.array(cat("ang"), dtype=wp.int32, device=device)
        self.tpl_q_off = wp.array(cat("q_off"), dtype=wp.int32, device=device)
        self.tpl_qd_off = wp.array(cat("qd_off"), dtype=wp.int32, device=device)
        self.tpl_parent_local = wp.array(cat("parent_local"), dtype=wp.int32, device=device)
        self.tpl_child_off = wp.array(cat("child_off"), dtype=wp.int32, device=device)
        self.tpl_X_p = wp.array(cat("X_p"), dtype=wp.transform, device=device)
        self.tpl_X_c = wp.array(cat("X_c"), dtype=wp.transform, device=device)
        self.tpl_axis = wp.array(cat("axis"), dtype=wp.vec3, device=device)
        # Per-template level lists (local joint indices) and children (local joint indices, descending).
        lvl_off = np.zeros((len(tpl_rows), self.max_levels_hint + 1), dtype=np.int32) if False else None
        max_levels = max(int(row["depth"].max()) + 1 for row in tpl_rows)
        lvl_off = np.zeros((len(tpl_rows), max_levels + 1), dtype=np.int32)
        lvl_joints: list[int] = []
        ch_off: list[int] = []
        ch: list[int] = []
        for t, row in enumerate(tpl_rows):
            d = row["depth"]
            lvl_off[t, 0] = len(lvl_joints)
            for lv in range(max_levels):
                lvl_joints.extend(int(k) for k in np.nonzero(d == lv)[0])
                lvl_off[t, lv + 1] = len(lvl_joints)
            pl = row["parent_local"]
            for k in range(int(row["n"])):
                ch_off.append(len(ch))
                ch.extend(int(c) for c in range(int(row["n"]) - 1, -1, -1) if int(pl[c]) == k)
        ch_off.append(len(ch))
        self.tpl_max_levels = int(max_levels)
        self.tpl_level_offsets = wp.array(lvl_off, dtype=wp.int32, device=device)
        self.tpl_level_joints = wp.array(np.asarray(lvl_joints or [0], dtype=np.int32), dtype=wp.int32, device=device)
        self.tpl_children_offsets = wp.array(np.asarray(ch_off, dtype=np.int32), dtype=wp.int32, device=device)
        self.tpl_children = wp.array(np.asarray(ch or [0], dtype=np.int32), dtype=wp.int32, device=device)
        self.art_template = wp.array(self.template_of_art, dtype=wp.int32, device=device)
        self.art_body_base = wp.array(art_body_base, dtype=wp.int32, device=device)
        self.art_q_base = wp.array(art_q_base, dtype=wp.int32, device=device)
        self.art_qd_base = wp.array(art_qd_base, dtype=wp.int32, device=device)

    def _body_x_com_rows(self, model, bodies):
        x = self._body_X_com if self._body_X_com is not None else getattr(model, "body_X_com", None)
        if x is None:
            raise ValueError("body_X_com unavailable")
        return x.numpy()[bodies]

    def _body_x_com_bytes(self, model, bodies) -> bytes:
        return self._body_x_com_rows(model, bodies).tobytes()


@wp.kernel(enable_backward=False)
def grouped_fk_kinematics(
    lanes_per_articulation: int,
    articulation_count: int,
    max_levels: int,
    art_level_offsets: wp.array2d[int],
    level_joint_local: wp.array[int],
    articulation_start: wp.array[int],
    articulation_joint_end: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_X_com: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_com: wp.array[wp.vec3],
    # outputs
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    articulation_origin: wp.array[wp.vec3],
    joint_S_s: wp.array[wp.spatial_vector],
    body_v_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
    fk_id_cache_valid: wp.array[int],
):
    """Poses, motion subspaces, velocities and accelerations, level by level."""
    block, lane = wp.tid()
    group = lane // lanes_per_articulation
    local = lane - group * lanes_per_articulation
    art = block * (32 // lanes_per_articulation) + group
    valid = art < articulation_count
    start = int(0)
    if valid:
        start = articulation_start[art]
    for level in range(max_levels):
        if valid:
            lo = art_level_offsets[art, level]
            hi = art_level_offsets[art, level + 1]
            k = lo + local
            while k < hi:
                j = start + level_joint_local[k]
                compute_link_transform(
                    j,
                    joint_type,
                    joint_parent,
                    joint_child,
                    joint_q_start,
                    joint_qd_start,
                    joint_q,
                    joint_X_p,
                    joint_X_c,
                    body_X_com,
                    joint_axis,
                    joint_dof_dim,
                    body_q,
                    body_q_com,
                )
                k += lanes_per_articulation
        warp_sync()
    origin = wp.vec3()
    if valid:
        if start < articulation_start[art + 1]:
            root_body = joint_child[start]
            if root_body >= 0:
                origin = wp.transform_point(body_q[root_body], body_com[root_body])
        if local == 0:
            articulation_origin[art] = origin
    for level in range(max_levels):
        if valid:
            lo = art_level_offsets[art, level]
            hi = art_level_offsets[art, level + 1]
            k = lo + local
            while k < hi:
                j = start + level_joint_local[k]
                parent = joint_parent[j]
                child = joint_child[j]
                parent_v_s = wp.spatial_vector()
                parent_a_s = wp.spatial_vector()
                if parent >= 0:
                    parent_v_s = body_v_s[parent]
                    parent_a_s = body_a_s[parent]
                compute_link_kinematics(
                    j,
                    parent,
                    child,
                    parent_v_s,
                    parent_a_s,
                    origin,
                    joint_type,
                    joint_qd_start,
                    joint_qd,
                    joint_axis,
                    joint_dof_dim,
                    body_q,
                    joint_X_p,
                    joint_S_s,
                    body_v_s,
                    body_a_s,
                )
                k += lanes_per_articulation
        warp_sync()
    if valid and local == 0:
        fk_id_cache_valid[art] = 1


@wp.kernel(enable_backward=False)
def grouped_tau(
    lanes_per_articulation: int,
    articulation_count: int,
    max_levels: int,
    art_level_offsets: wp.array2d[int],
    level_joint_local: wp.array[int],
    children_offsets: wp.array[int],
    children: wp.array[int],
    articulation_start: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_f: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_spring_stiffness: wp.array[float],
    joint_spring_ref: wp.array[float],
    joint_damping: wp.array[float],
    joint_S_s: wp.array[wp.spatial_vector],
    body_fb_s: wp.array[wp.spatial_vector],
    body_f_ext: wp.array[wp.spatial_vector],
    body_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    articulation_origin: wp.array[wp.vec3],
    add_existing_tau: int,
    # outputs
    body_ft_s: wp.array[wp.spatial_vector],
    body_fs_scratch: wp.array[wp.spatial_vector],
    tau: wp.array[float],
):
    """Inverse dynamics backward pass, leaves to root, one lane group per articulation.

    ``body_ft_s`` holds the same values the serial kernel leaves behind: the sum of the
    children's net wrenches accumulated in descending joint order onto the pre-existing entry.
    """
    block, lane = wp.tid()
    group = lane // lanes_per_articulation
    local = lane - group * lanes_per_articulation
    art = block * (32 // lanes_per_articulation) + group
    valid = art < articulation_count
    start = int(0)
    origin = wp.vec3()
    if valid:
        start = articulation_start[art]
        origin = articulation_origin[art]
    for step in range(max_levels):
        level = max_levels - 1 - step
        if valid:
            lo = art_level_offsets[art, level]
            hi = art_level_offsets[art, level + 1]
            k = lo + local
            while k < hi:
                i = start + level_joint_local[k]
                child = joint_child[i]
                # Children's net wrenches, descending joint order (matches the serial accumulation).
                f_t_s = body_ft_s[child]
                c_lo = children_offsets[i]
                c_hi = children_offsets[i + 1]
                for c in range(c_lo, c_hi):
                    f_t_s = f_t_s + body_fs_scratch[joint_child[children[c]]]
                body_ft_s[child] = f_t_s
                f_ext_com = wp.spatial_vector()
                if (body_flags[child] & BodyFlags.KINEMATIC) == 0:
                    f_ext_com = body_f_ext[child]
                f_ext_f = wp.spatial_bottom(f_ext_com)
                f_ext_t = wp.spatial_top(f_ext_com)
                com_world = wp.transform_point(body_q[child], body_com[child])
                com_rel = com_world - origin
                tau_origin = f_ext_f + wp.cross(com_rel, f_ext_t)
                f_ext_origin = wp.spatial_vector(f_ext_t, tau_origin)
                f_s = body_fb_s[child] + f_t_s - f_ext_origin
                jcalc_tau(
                    joint_type[i],
                    joint_S_s,
                    joint_f,
                    joint_q,
                    joint_qd,
                    joint_spring_stiffness,
                    joint_spring_ref,
                    joint_damping,
                    joint_q_start[i],
                    joint_qd_start[i],
                    joint_dof_dim[i, 0],
                    joint_dof_dim[i, 1],
                    f_s,
                    add_existing_tau,
                    tau,
                )
                body_fs_scratch[child] = f_s
                k += lanes_per_articulation
        warp_sync()


@wp.kernel(enable_backward=False)
def grouped_composite_inertia(
    lanes_per_articulation: int,
    articulation_count: int,
    max_levels: int,
    art_level_offsets: wp.array2d[int],
    level_joint_local: wp.array[int],
    children_offsets: wp.array[int],
    children: wp.array[int],
    articulation_start: wp.array[int],
    mass_update_mask: wp.array[int],
    joint_child: wp.array[int],
    body_I_s: wp.array[wp.spatial_matrix],
    # outputs
    body_I_c: wp.array[wp.spatial_matrix],
):
    """Composite rigid-body inertias, leaves to root, children added in descending joint order."""
    block, lane = wp.tid()
    group = lane // lanes_per_articulation
    local = lane - group * lanes_per_articulation
    art = block * (32 // lanes_per_articulation) + group
    valid = art < articulation_count
    if valid:
        if mass_update_mask[art] == 0:
            valid = False
    start = int(0)
    if valid:
        start = articulation_start[art]
    for step in range(max_levels):
        level = max_levels - 1 - step
        if valid:
            lo = art_level_offsets[art, level]
            hi = art_level_offsets[art, level + 1]
            k = lo + local
            while k < hi:
                i = start + level_joint_local[k]
                body = joint_child[i]
                acc = body_I_s[body]
                c_lo = children_offsets[i]
                c_hi = children_offsets[i + 1]
                for c in range(c_lo, c_hi):
                    acc += body_I_c[joint_child[children[c]]]
                body_I_c[body] = acc
                k += lanes_per_articulation
        warp_sync()


def get_grouped_hinv_jt_kernel(
    n_dofs: int,
    max_constraints: int,
    *,
    write_world: bool,
    write_group: bool,
    compute_diag: bool,
) -> wp.Kernel:
    """Response rows ``Y = H^-1 J^T`` with one warp per articulation and one lane per row.

    The Cholesky factor lives in shared memory; each lane solves its own row (forward then
    backward substitution) with the row's Jacobian and response in registers. Rows past the
    world's constraint count are not touched. Optionally the diagonal ``J_i . Y_i`` and the
    world-layout copies of ``J`` and ``Y`` are written, matching the tiled kernel's outputs.
    """
    N = int(n_dofs)
    M = int(max_constraints)
    world_store = (
        """
            const int dof_offset = articulation_world_dof_offset.data[art];
            const size_t wbase = (size_t)world * world_rows * world_dofs + (size_t)row * world_dofs + dof_offset;
            #pragma unroll
            for (int d = 0; d < N_; ++d) {
                J_world.data[wbase + d] = j[d];
                Y_world.data[wbase + d] = y[d];
            }"""
        if write_world
        else ""
    )
    group_store = (
        """
            #pragma unroll
            for (int d = 0; d < N_; ++d) Y_group.data[gbase + d] = y[d];"""
        if write_group
        else ""
    )
    diag_store = (
        """
            float acc = 0.0f;
            #pragma unroll
            for (int d = 0; d < N_; ++d) acc += j[d] * y[d];
            diag_group.data[(size_t)idx * M_ + row] = acc;"""
        if compute_diag
        else ""
    )
    snippet = f"""
#if defined(__CUDA_ARCH__)
    constexpr int N_ = {N};
    constexpr int M_ = {M};
    constexpr int WPB = 4;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int idx = block * warps_per_block + warp;
    __shared__ float Ls[WPB * N_ * N_];
    __shared__ float Ds[WPB * N_];
    float* L = Ls + warp * N_ * N_;
    float* Dinv = Ds + warp * N_;
    if (idx >= n_arts) return;
    const int art = group_to_art.data[idx];
    const int world = art_to_world.data[art];
    const int n_rows = world_constraint_count.data[world];
    if (n_rows <= 0) return;
    // Worlds the parallel sweep kernel owns (dense rows within its budget, no matrix-free rows) form the
    // response in-kernel; their world-layout J/Y are never read.
    if (skip_rows_le > 0 && n_rows <= skip_rows_le && mf_constraint_count.data[world] == 0) return;
    for (int e = lane; e < N_ * N_; e += 32) L[e] = L_group.data[(size_t)idx * N_ * N_ + e];
    __syncwarp();
    if (lane < N_) Dinv[lane] = 1.0f / L[lane * N_ + lane];
    __syncwarp();
    const int world_rows = Y_world.shape[1];
    const int world_dofs = Y_world.shape[2];
    // Match the tiled kernel: every row of a touched 32-row chunk is written (inactive rows
    // carry zero Jacobians and get zero responses).
    const int n_pad = ((n_rows + 31) / 32) * 32;
    for (int row = lane; row < n_pad && row < M_; row += 32) {{
        const size_t gbase = ((size_t)idx * M_ + row) * N_;
        float j[N_], y[N_];
        #pragma unroll
        for (int d = 0; d < N_; ++d) {{ j[d] = J_group.data[gbase + d]; y[d] = 0.0f; }}
        // Forward: L z = j, then backward: L^T y = z. Both loops fully unrolled at compile time so every
        // y[k] index is a constant (registers) and only the N(N-1)/2 real multiply-adds are emitted per pass;
        // the previous predicated form emitted N^2 per step (N^3 per row).
        #pragma unroll
        for (int i = 0; i < N_; ++i) {{
            float v = j[i];
            #pragma unroll
            for (int k = 0; k < i; ++k) v -= L[i * N_ + k] * y[k];
            y[i] = v * Dinv[i];
        }}
        #pragma unroll
        for (int i = N_ - 1; i >= 0; --i) {{
            float v = y[i];
            #pragma unroll
            for (int k = i + 1; k < N_; ++k) v -= L[k * N_ + i] * y[k];
            y[i] = v * Dinv[i];
        }}{group_store}{diag_store}{world_store}
    }}
#endif
"""

    @wp.func_native(snippet)
    def grouped_hinv_jt_native(
        block: int,
        warps_per_block: int,
        n_arts: int,
        L_group: wp.array3d[float],
        J_group: wp.array3d[float],
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        world_constraint_count: wp.array[int],
        Y_group: wp.array3d[float],
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
        diag_group: wp.array2d[float],
        mf_constraint_count: wp.array[int],
        skip_rows_le: int,
    ): ...

    def grouped_hinv_jt_template(
        warps_per_block: int,
        n_arts: int,
        L_group: wp.array3d[float],
        J_group: wp.array3d[float],
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        world_constraint_count: wp.array[int],
        Y_group: wp.array3d[float],
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
        diag_group: wp.array2d[float],
        mf_constraint_count: wp.array[int],
        skip_rows_le: int,
    ):
        block, _lane = wp.tid()
        grouped_hinv_jt_native(
            block,
            warps_per_block,
            n_arts,
            L_group,
            J_group,
            group_to_art,
            art_to_world,
            articulation_world_dof_offset,
            world_constraint_count,
            Y_group,
            J_world,
            Y_world,
            diag_group,
            mf_constraint_count,
            skip_rows_le,
        )

    suffix = (
        ("_world" if write_world else "") + ("_nogroup" if not write_group else "") + ("_diag" if compute_diag else "")
    )
    name = f"grouped_hinv_jt_{N}_{M}{suffix}"
    grouped_hinv_jt_template.__name__ = name
    grouped_hinv_jt_template.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(grouped_hinv_jt_template)


@wp.func
def jcalc_motion_t(
    type: int,
    axis_array: wp.array[wp.vec3],
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    X_sc: wp.transform,
    joint_qd: wp.array[float],
    qd_start: int,
    # outputs
    joint_S_s: wp.array[wp.spatial_vector],
):
    """``jcalc_motion`` with the axis table indexed separately from the state arrays."""
    if type == JointType.PRISMATIC:
        axis = axis_array[axis_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s
    if type == JointType.REVOLUTE:
        axis = axis_array[axis_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s
    if type == JointType.D6:
        v_j_s = wp.spatial_vector()
        if lin_axis_count > 0:
            axis = axis_array[axis_start + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 0]
            joint_S_s[qd_start + 0] = S_s
        if lin_axis_count > 1:
            axis = axis_array[axis_start + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 1]
            joint_S_s[qd_start + 1] = S_s
        if lin_axis_count > 2:
            axis = axis_array[axis_start + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 2]
            joint_S_s[qd_start + 2] = S_s
        if ang_axis_count > 0:
            axis = axis_array[axis_start + lin_axis_count + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 0]
            joint_S_s[qd_start + lin_axis_count + 0] = S_s
        if ang_axis_count > 1:
            axis = axis_array[axis_start + lin_axis_count + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 1]
            joint_S_s[qd_start + lin_axis_count + 1] = S_s
        if ang_axis_count > 2:
            axis = axis_array[axis_start + lin_axis_count + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 2]
            joint_S_s[qd_start + lin_axis_count + 2] = S_s
        return v_j_s
    if type == JointType.BALL:
        S_0 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        S_1 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        S_2 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1
        joint_S_s[qd_start + 2] = S_2
        return S_0 * joint_qd[qd_start + 0] + S_1 * joint_qd[qd_start + 1] + S_2 * joint_qd[qd_start + 2]
    if type == JointType.FIXED:
        return wp.spatial_vector()
    if type == JointType.FREE or type == JointType.DISTANCE:
        q_sc = wp.transform_get_rotation(X_sc)
        v_local = wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2])
        w_local = wp.vec3(joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5])
        v_j_s = wp.spatial_vector(wp.quat_rotate(q_sc, v_local), wp.quat_rotate(q_sc, w_local))
        ex = wp.quat_rotate(q_sc, wp.vec3(1.0, 0.0, 0.0))
        ey = wp.quat_rotate(q_sc, wp.vec3(0.0, 1.0, 0.0))
        ez = wp.quat_rotate(q_sc, wp.vec3(0.0, 0.0, 1.0))
        joint_S_s[qd_start + 0] = wp.spatial_vector(ex, wp.vec3())
        joint_S_s[qd_start + 1] = wp.spatial_vector(ey, wp.vec3())
        joint_S_s[qd_start + 2] = wp.spatial_vector(ez, wp.vec3())
        joint_S_s[qd_start + 3] = wp.spatial_vector(wp.vec3(), ex)
        joint_S_s[qd_start + 4] = wp.spatial_vector(wp.vec3(), ey)
        joint_S_s[qd_start + 5] = wp.spatial_vector(wp.vec3(), ez)
        return v_j_s
    return wp.spatial_vector()


@wp.kernel(enable_backward=False)
def template_fk_kinematics(
    lanes_per_articulation: int,
    articulation_count: int,
    max_levels: int,
    art_template: wp.array[int],
    art_body_base: wp.array[int],
    art_q_base: wp.array[int],
    art_qd_base: wp.array[int],
    tpl_joint_offset: wp.array[int],
    tpl_dof_offset: wp.array[int],
    tpl_level_offsets: wp.array2d[int],
    tpl_level_joints: wp.array[int],
    tpl_type: wp.array[int],
    tpl_lin: wp.array[int],
    tpl_ang: wp.array[int],
    tpl_q_off: wp.array[int],
    tpl_qd_off: wp.array[int],
    tpl_parent_local: wp.array[int],
    tpl_child_off: wp.array[int],
    tpl_X_p: wp.array[wp.transform],
    tpl_X_c: wp.array[wp.transform],
    tpl_axis: wp.array[wp.vec3],
    joint_X_p: wp.array[wp.transform],
    body_X_com: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    articulation_start: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    # outputs
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    articulation_origin: wp.array[wp.vec3],
    joint_S_s: wp.array[wp.spatial_vector],
    body_v_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
    fk_id_cache_valid: wp.array[int],
):
    """Forward kinematics with per-template constants (same math as ``eval_rigid_fk_kinematics``)."""
    block, lane = wp.tid()
    group = lane // lanes_per_articulation
    local = lane - group * lanes_per_articulation
    art = block * (32 // lanes_per_articulation) + group
    valid = art < articulation_count
    t = int(0)
    body_base = int(0)
    q_base = int(0)
    qd_base = int(0)
    tj = int(0)
    td = int(0)
    if valid:
        t = art_template[art]
        body_base = art_body_base[art]
        q_base = art_q_base[art]
        qd_base = art_qd_base[art]
        tj = tpl_joint_offset[t]
        td = tpl_dof_offset[t]
    # Pass 1: poses.
    for level in range(max_levels):
        if valid:
            lo = tpl_level_offsets[t, level]
            hi = tpl_level_offsets[t, level + 1]
            k = lo + local
            while k < hi:
                jl = tpl_level_joints[k]
                tjl = tj + jl
                pl = tpl_parent_local[tjl]
                X_pj = tpl_X_p[tjl]
                if pl < 0:
                    X_pj = joint_X_p[articulation_start[art] + jl]
                X_cj = tpl_X_c[tjl]
                X_wpj = X_pj
                if pl >= 0:
                    X_wpj = body_q[body_base + tpl_child_off[tj + pl]] * X_wpj
                X_j = jcalc_transform(
                    tpl_type[tjl],
                    tpl_axis,
                    td + tpl_qd_off[tjl],
                    tpl_lin[tjl],
                    tpl_ang[tjl],
                    joint_q,
                    q_base + tpl_q_off[tjl],
                )
                X_wcj = X_wpj * X_j
                X_wc = X_wcj * wp.transform_inverse(X_cj)
                child = body_base + tpl_child_off[tjl]
                X_sm = X_wc * body_X_com[child]
                body_q[child] = X_wc
                body_q_com[child] = X_sm
                k += lanes_per_articulation
        warp_sync()
    origin = wp.vec3()
    if valid:
        start = articulation_start[art]
        if start < articulation_start[art + 1]:
            root_body = body_base + tpl_child_off[tj]
            if root_body >= 0:
                origin = wp.transform_point(body_q[root_body], body_com[root_body])
        if local == 0:
            articulation_origin[art] = origin
    # Pass 2: motion subspaces, velocities, accelerations.
    for level in range(max_levels):
        if valid:
            lo = tpl_level_offsets[t, level]
            hi = tpl_level_offsets[t, level + 1]
            k = lo + local
            while k < hi:
                jl = tpl_level_joints[k]
                tjl = tj + jl
                pl = tpl_parent_local[tjl]
                child = body_base + tpl_child_off[tjl]
                parent_v_s = wp.spatial_vector()
                parent_a_s = wp.spatial_vector()
                X_wpj = tpl_X_p[tjl]
                if pl < 0:
                    X_wpj = joint_X_p[articulation_start[art] + jl]
                if pl >= 0:
                    parent = body_base + tpl_child_off[tj + pl]
                    parent_v_s = body_v_s[parent]
                    parent_a_s = body_a_s[parent]
                    X_wpj = body_q[parent] * X_wpj
                X_wpj_local = wp.transform(
                    wp.transform_get_translation(X_wpj) - origin, wp.transform_get_rotation(X_wpj)
                )
                v_j_s = jcalc_motion_t(
                    tpl_type[tjl],
                    tpl_axis,
                    td + tpl_qd_off[tjl],
                    tpl_lin[tjl],
                    tpl_ang[tjl],
                    X_wpj_local,
                    joint_qd,
                    qd_base + tpl_qd_off[tjl],
                    joint_S_s,
                )
                v_s = parent_v_s + v_j_s
                a_s = parent_a_s + spatial_cross(v_s, v_j_s)
                body_v_s[child] = v_s
                body_a_s[child] = a_s
                k += lanes_per_articulation
        warp_sync()
    if valid and local == 0:
        fk_id_cache_valid[art] = 1


@wp.func
def assemble_compact_inertia(mass: float, terms: wp.array2d[float], body: int):
    """Spatial inertia from mass and the 12 compact terms (same element formula as the warp kernel)."""
    m = wp.spatial_matrix()
    for row in range(6):
        for col in range(6):
            value = float(0.0)
            if row < 3 and col < 3:
                if row == col:
                    value = mass
            elif row >= 3 and col >= 3:
                value = terms[body, 3 + (row - 3) * 3 + col - 3]
            else:
                cross_row = row
                if row >= 3:
                    cross_row = row - 3
                cross_col = col
                if col >= 3:
                    cross_col = col - 3
                cross = float(0.0)
                if cross_row == 0 and cross_col == 1:
                    cross = -terms[body, 2]
                if cross_row == 0 and cross_col == 2:
                    cross = terms[body, 1]
                if cross_row == 1 and cross_col == 0:
                    cross = terms[body, 2]
                if cross_row == 1 and cross_col == 2:
                    cross = -terms[body, 0]
                if cross_row == 2 and cross_col == 0:
                    cross = -terms[body, 1]
                if cross_row == 2 and cross_col == 1:
                    cross = terms[body, 0]
                sign = mass
                if row < 3:
                    sign = -mass
                value = sign * cross
            m[row, col] = value
    return m


@wp.kernel(enable_backward=False)
def grouped_tau_mass(
    lanes_per_articulation: int,
    n_arts: int,
    max_levels: int,
    do_mass: int,
    n_dofs: int,
    schedule_count: int,
    fused_augmented_drive: int,
    art_level_offsets: wp.array2d[int],
    level_joint_local: wp.array[int],
    children_offsets: wp.array[int],
    children: wp.array[int],
    group_to_art: wp.array[int],
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_f: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_spring_stiffness: wp.array[float],
    joint_spring_ref: wp.array[float],
    joint_damping: wp.array[float],
    joint_S_s: wp.array[wp.spatial_vector],
    body_fb_s: wp.array[wp.spatial_vector],
    body_f_ext: wp.array[wp.spatial_vector],
    body_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    articulation_origin: wp.array[wp.vec3],
    add_existing_tau: int,
    body_mass: wp.array[float],
    body_inertia_terms: wp.array2d[float],
    R_group: wp.array2d[float],
    dof_joint_offset: wp.array[int],
    source_dof: wp.array2d[int],
    drive_row_by_dof: wp.array[int],
    row_K: wp.array[float],
    # outputs
    body_ft_s: wp.array[wp.spatial_vector],
    body_fs_scratch: wp.array[wp.spatial_vector],
    tau: wp.array[float],
    body_I_c: wp.array[wp.spatial_matrix],
    L_group: wp.array3d[float],
):
    """Inverse dynamics and (on mass-update steps) composite inertia, CRBA and Cholesky in one launch.

    One lane group per articulation of a size group. The backward pass over tree levels
    computes joint torques exactly like ``grouped_tau`` and, when ``do_mass`` is set,
    accumulates composite inertias from the compact terms in the same order as the warp
    composite kernel. The mass matrix is then assembled from the source-DOF table and factored
    in place in ``L_group`` (lower triangle, zero upper).
    """
    block, lane = wp.tid()
    group = lane // lanes_per_articulation
    local = lane - group * lanes_per_articulation
    gidx = block * (32 // lanes_per_articulation) + group
    valid = gidx < n_arts
    art = int(0)
    start = int(0)
    origin = wp.vec3()
    if valid:
        art = group_to_art[gidx]
        start = articulation_start[art]
        origin = articulation_origin[art]
    for step in range(max_levels):
        level = max_levels - 1 - step
        if valid:
            lo = art_level_offsets[art, level]
            hi = art_level_offsets[art, level + 1]
            k = lo + local
            while k < hi:
                i = start + level_joint_local[k]
                child = joint_child[i]
                c_lo = children_offsets[i]
                c_hi = children_offsets[i + 1]
                f_t_s = body_ft_s[child]
                for c in range(c_lo, c_hi):
                    f_t_s = f_t_s + body_fs_scratch[joint_child[children[c]]]
                body_ft_s[child] = f_t_s
                f_ext_com = wp.spatial_vector()
                if (body_flags[child] & BodyFlags.KINEMATIC) == 0:
                    f_ext_com = body_f_ext[child]
                f_ext_f = wp.spatial_bottom(f_ext_com)
                f_ext_t = wp.spatial_top(f_ext_com)
                com_world = wp.transform_point(body_q[child], body_com[child])
                com_rel = com_world - origin
                tau_origin = f_ext_f + wp.cross(com_rel, f_ext_t)
                f_ext_origin = wp.spatial_vector(f_ext_t, tau_origin)
                f_s = body_fb_s[child] + f_t_s - f_ext_origin
                jcalc_tau(
                    joint_type[i],
                    joint_S_s,
                    joint_f,
                    joint_q,
                    joint_qd,
                    joint_spring_stiffness,
                    joint_spring_ref,
                    joint_damping,
                    joint_q_start[i],
                    joint_qd_start[i],
                    joint_dof_dim[i, 0],
                    joint_dof_dim[i, 1],
                    f_s,
                    add_existing_tau,
                    tau,
                )
                body_fs_scratch[child] = f_s
                if do_mass != 0:
                    acc = assemble_compact_inertia(body_mass[child], body_inertia_terms, child)
                    for c in range(c_lo, c_hi):
                        acc += body_I_c[joint_child[children[c]]]
                    body_I_c[child] = acc
                k += lanes_per_articulation
        warp_sync()
    if do_mass == 0:
        return
    # CRBA: H[row][col] = S_proj . (I_c[body(source dof)] S_source), plus armature and drive stiffness.
    if valid:
        dof_start = articulation_dof_start[art]
        e = int(0)
        for row in range(n_dofs):
            for col in range(row + 1):
                if e % lanes_per_articulation == local:
                    source = source_dof[row, col]
                    value = float(0.0)
                    if source >= 0:
                        projection = col
                        if source == col:
                            projection = row
                        joint = start + dof_joint_offset[source]
                        force = body_I_c[joint_child[joint]] * joint_S_s[dof_start + source]
                        value = wp.dot(joint_S_s[dof_start + projection], force)
                    if row == col:
                        value += R_group[gidx, row]
                        if fused_augmented_drive != 0:
                            drive_row = drive_row_by_dof[dof_start + row]
                            if drive_row >= 0:
                                K = row_K[drive_row]
                                if K > 0.0:
                                    value += K
                    L_group[gidx, row, col] = value
                    if col < row:
                        L_group[gidx, col, row] = 0.0
                e += 1
    warp_sync()
    if valid and local == 0:
        for col in range(n_dofs):
            diagonal = L_group[gidx, col, col]
            for k in range(col):
                v = L_group[gidx, col, k]
                diagonal -= v * v
            diagonal = wp.sqrt(diagonal)
            L_group[gidx, col, col] = diagonal
            inverse_diagonal = 1.0 / diagonal
            for row in range(col + 1, n_dofs):
                v = L_group[gidx, row, col]
                for k in range(col):
                    v -= L_group[gidx, row, k] * L_group[gidx, col, k]
                L_group[gidx, row, col] = v * inverse_diagonal


def get_grouped_mass_kernel(n_dofs: int, max_joints: int) -> wp.Kernel:
    """Composite inertia (compact terms), CRBA and Cholesky for one articulation per warp.

    Composite inertias are accumulated child-into-parent in descending joint order with the
    36 matrix elements spread over the lanes, exactly like the warp composite kernel. The mass
    matrix is assembled from the source-DOF table in shared memory and factored in place by
    lane 0; the lower triangle is stored to ``L_group`` with a zero upper triangle.
    """
    N = int(n_dofs)
    MJ = int(max_joints)
    snippet = f"""
#if defined(__CUDA_ARCH__)
    constexpr int N_ = {N};
    constexpr int MJ_ = {MJ};
    constexpr int WPB = 4;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int gidx = block * warps_per_block + warp;
    __shared__ float Ic_s[WPB * MJ_ * 36];
    __shared__ float F_s[WPB * N_ * 6];
    __shared__ float H_s[WPB * N_ * N_];
    float* Ic = Ic_s + warp * MJ_ * 36;
    float* F = F_s + warp * N_ * 6;
    float* H = H_s + warp * N_ * N_;
    if (gidx >= n_arts) return;
    const int art = group_to_art.data[gidx];
    const int start = articulation_start.data[art];
    const int end = articulation_joint_end.data[art];
    const int count = end - start;
    // Assemble link inertias from mass and the 12 compact terms.
    for (int j = 0; j < count; ++j) {{
        const int body = joint_child.data[start + j];
        const float mass = body_mass.data[body];
        const float* terms = &body_inertia_terms.data[body * 12];
        for (int element = lane; element < 36; element += 32) {{
            const int row = element / 6;
            const int col = element - row * 6;
            float value = 0.0f;
            if (row < 3 && col < 3) {{
                if (row == col) value = mass;
            }} else if (row >= 3 && col >= 3) {{
                value = terms[3 + (row - 3) * 3 + col - 3];
            }} else {{
                const int cross_row = row < 3 ? row : row - 3;
                const int cross_col = col < 3 ? col : col - 3;
                float cross = 0.0f;
                if (cross_row == 0 && cross_col == 1) cross = -terms[2];
                if (cross_row == 0 && cross_col == 2) cross = terms[1];
                if (cross_row == 1 && cross_col == 0) cross = terms[2];
                if (cross_row == 1 && cross_col == 2) cross = -terms[0];
                if (cross_row == 2 && cross_col == 0) cross = -terms[1];
                if (cross_row == 2 && cross_col == 1) cross = terms[0];
                value = (row < 3 ? -mass : mass) * cross;
            }}
            Ic[j * 36 + element] = value;
        }}
    }}
    __syncwarp();
    // Composite: child into parent, descending joint order.
    for (int j = count - 1; j >= 0; --j) {{
        const int parent_joint = joint_ancestor.data[start + j];
        if (parent_joint >= start) {{
            const int pj = parent_joint - start;
            for (int element = lane; element < 36; element += 32) Ic[pj * 36 + element] += Ic[j * 36 + element];
        }}
        __syncwarp();
    }}
    // Publish body_I_c for downstream consumers.
    for (int j = 0; j < count; ++j) {{
        float* dst = reinterpret_cast<float*>(&body_I_c.data[joint_child.data[start + j]]);
        for (int element = lane; element < 36; element += 32) dst[element] = Ic[j * 36 + element];
    }}
    // Force components F[d] = I_c[body(d)] S[d].
    const int dof_start = articulation_dof_start.data[art];
    if (lane < N_) {{
        const int j = dof_joint_offset.data[lane];
        const auto S = joint_S_s.data[dof_start + lane];
        for (int r = 0; r < 6; ++r) {{
            float acc = 0.0f;
            for (int c = 0; c < 6; ++c) acc += Ic[j * 36 + r * 6 + c] * S.c[c];
            F[lane * 6 + r] = acc;
        }}
    }}
    __syncwarp();
    // H lower triangle from the source-DOF table.
    for (int e = lane; e < N_ * (N_ + 1) / 2; e += 32) {{
        // e -> (row, col) with col <= row
        int row = (int)((sqrtf(8.0f * (float)e + 1.0f) - 1.0f) * 0.5f);
        while (row * (row + 1) / 2 > e) --row;
        while ((row + 1) * (row + 2) / 2 <= e) ++row;
        const int col = e - row * (row + 1) / 2;
        const int source = source_dof.data[row * N_ + col];
        float value = 0.0f;
        if (source >= 0) {{
            const int projection = (source == col) ? row : col;
            const auto motion = joint_S_s.data[dof_start + projection];
            for (int c = 0; c < 6; ++c) value += motion.c[c] * F[source * 6 + c];
        }}
        if (row == col) {{
            value += R_group.data[gidx * N_ + row];
            if (fused_augmented_drive != 0) {{
                const int drive_row = drive_row_by_dof.data[dof_start + row];
                if (drive_row >= 0) {{
                    const float K = row_K.data[drive_row];
                    if (K > 0.0f) value += K;
                }}
            }}
        }}
        H[row * N_ + col] = value;
    }}
    __syncwarp();
    // Column-by-column Cholesky: every lane recomputes the diagonal (same op order as the
    // serial kernel), then lane ``row`` finishes its element of the column.
    for (int col = 0; col < N_; ++col) {{
        float diagonal = H[col * N_ + col];
        for (int k = 0; k < col; ++k) {{
            const float v = H[col * N_ + k];
            diagonal -= v * v;
        }}
        diagonal = sqrtf(diagonal);
        const float inverse_diagonal = 1.0f / diagonal;
        __syncwarp();
        if (lane == 0) H[col * N_ + col] = diagonal;
        for (int row = col + 1 + lane; row < N_; row += 32) {{
            float v = H[row * N_ + col];
            for (int k = 0; k < col; ++k) v -= H[row * N_ + k] * H[col * N_ + k];
            H[row * N_ + col] = v * inverse_diagonal;
        }}
        __syncwarp();
    }}
    for (int e = lane; e < N_ * N_; e += 32) {{
        const int row = e / N_;
        const int col = e - row * N_;
        L_group.data[(size_t)gidx * N_ * N_ + e] = (col <= row) ? H[e] : 0.0f;
    }}
#endif
"""

    @wp.func_native(snippet)
    def grouped_mass_native(
        block: int,
        warps_per_block: int,
        n_arts: int,
        group_to_art: wp.array[int],
        articulation_start: wp.array[int],
        articulation_joint_end: wp.array[int],
        joint_ancestor: wp.array[int],
        joint_child: wp.array[int],
        body_mass: wp.array[float],
        body_inertia_terms: wp.array2d[float],
        articulation_dof_start: wp.array[int],
        joint_S_s: wp.array[wp.spatial_vector],
        R_group: wp.array2d[float],
        dof_joint_offset: wp.array[int],
        source_dof: wp.array2d[int],
        fused_augmented_drive: int,
        drive_row_by_dof: wp.array[int],
        row_K: wp.array[float],
        body_I_c: wp.array[wp.spatial_matrix],
        L_group: wp.array3d[float],
    ): ...

    def grouped_mass_template(
        warps_per_block: int,
        n_arts: int,
        group_to_art: wp.array[int],
        articulation_start: wp.array[int],
        articulation_joint_end: wp.array[int],
        joint_ancestor: wp.array[int],
        joint_child: wp.array[int],
        body_mass: wp.array[float],
        body_inertia_terms: wp.array2d[float],
        articulation_dof_start: wp.array[int],
        joint_S_s: wp.array[wp.spatial_vector],
        R_group: wp.array2d[float],
        dof_joint_offset: wp.array[int],
        source_dof: wp.array2d[int],
        fused_augmented_drive: int,
        drive_row_by_dof: wp.array[int],
        row_K: wp.array[float],
        body_I_c: wp.array[wp.spatial_matrix],
        L_group: wp.array3d[float],
    ):
        block, _lane = wp.tid()
        grouped_mass_native(
            block,
            warps_per_block,
            n_arts,
            group_to_art,
            articulation_start,
            articulation_joint_end,
            joint_ancestor,
            joint_child,
            body_mass,
            body_inertia_terms,
            articulation_dof_start,
            joint_S_s,
            R_group,
            dof_joint_offset,
            source_dof,
            fused_augmented_drive,
            drive_row_by_dof,
            row_K,
            body_I_c,
            L_group,
        )

    name = f"grouped_mass_{N}_j{MJ}"
    grouped_mass_template.__name__ = name
    grouped_mass_template.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(grouped_mass_template)
