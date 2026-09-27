# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Retain the original free6 CRBA/factor/inverse service in the kinetic chain.

CUDA launches the exact 50dfa warp4 CRBA and register lower-inverse factories.
The extra preparation launch maps the caller's already-resolved world refresh
mask and prepares current enabled K. No new cadence, free-body algebra, diagonal
inertia assumption, primary cache write, or current MF spatial inverse is added.
CPU explicitly assembles the original S.I.S law then calls original tiled
Cholesky and a forward-inverse reference. Original native CUDA has empty CPU
bodies, and original tiled CRBA lane-scatter is not a CPU-complete assembly.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from . import solver_feather_pgs as original
from .kinetic_source import checked_definitions

vec6 = wp.types.vector(length=6, dtype=wp.float32)


@wp.kernel
def prepare_free_refresh(
    secondary_group: wp.array[int],
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    requested: wp.array[int],
    drive_row_by_dof: wp.array[int],
    ke: wp.array[float],
    kd: wp.array[float],
    dt: float,
    mass_mask: wp.array[int],
    row_K: wp.array[float],
):
    """Map the existing schedule and retain original enabled-drive K law."""
    world = wp.tid()
    art = group_to_art[secondary_group[world]]
    refresh = int(requested[world] != 0)
    mass_mask[art] = refresh
    if refresh != 0:
        start = articulation_dof_start[art]
        for local in range(6):
            dof = start + local
            row = drive_row_by_dof[dof]
            if row >= 0:
                value = ke[dof] * dt * dt + kd[dof] * dt
                if value <= 0.0:
                    value = 0.0
                row_K[row] = value


@wp.kernel
def assemble_free_cpu_reference(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    mass_mask: wp.array[int],
    joint_child: wp.array[int],
    screws: wp.array[wp.spatial_vector],
    inertia: wp.array[wp.spatial_matrix],
    group_to_art: wp.array[int],
    drive_row_by_dof: wp.array[int],
    row_K: wp.array[float],
    H: wp.array3d[float],
):
    """CPU-only complete original lower source/projection schedule."""
    group, row, col = wp.tid()
    art = group_to_art[group]
    if mass_mask[art] == 0:
        return
    start = articulation_dof_start[art]
    source = wp.min(row, col)
    projection = wp.max(row, col)
    body = joint_child[articulation_start[art]]
    value = wp.dot(screws[start + projection], inertia[body] * screws[start + source])
    if row == col:
        drive = drive_row_by_dof[start + row]
        if drive >= 0 and row_K[drive] > 0.0:
            value += row_K[drive]
    H[group, row, col] = value


@wp.kernel
def inverse_lower_cpu_reference(
    lower: wp.array3d[float],
    group_to_art: wp.array[int],
    mass_mask: wp.array[int],
    inverse: wp.array3d[float],
):
    """CPU-only literal forward solve from original register-inverse source."""
    group, column = wp.tid()
    if mass_mask[group_to_art[group]] == 0:
        return
    values = vec6(0.0)
    for row in range(6):
        value = float(row == column)
        for k in range(row):
            value -= lower[group, row, k] * values[k]
        diagonal = lower[group, row, row]
        values[row] = 0.0
        if diagonal != 0.0:
            values[row] = value / diagonal
    for row in range(6):
        inverse[group, row, column] = values[row]


@dataclass
class FreeRefresh:
    """Fixed-argument graph-callable original service; allocation is untimed."""

    bundle: dict
    publication: object
    requested: object
    R_group: object
    articulation_start: object
    articulation_dof_start: object
    group_to_art: object
    mass_mask: object
    row_K: object
    dof_joint_offset: object
    lower_schedule: object
    cpu_H: object

    def launch(self):
        """Prepare then run original factor/inverse even on all-reuse frames."""
        b, p = self.bundle, self.publication
        f, held = b["inputs"], b["held"]
        device = wp.get_device(b["launch"]["device"])
        groups = self.group_to_art.shape[0]
        wp.launch(
            prepare_free_refresh,
            dim=self.requested.shape[0],
            inputs=[
                b["plan"].secondary_group,
                self.group_to_art,
                self.articulation_dof_start,
                self.requested,
                f.drive_row_by_dof,
                f.joint_target_ke,
                f.joint_target_kd,
                f.dt,
                self.mass_mask,
                self.row_K,
            ],
            device=device,
        )
        common = [
            self.articulation_start,
            self.articulation_dof_start,
            self.mass_mask,
            p.joint_child,
            p.joint_S_s,
            p.body_I_s,
            self.group_to_art,
            self.R_group,
            self.dof_joint_offset,
        ]
        if device.is_cuda:
            wp.launch(
                original._get_crba_cholesky_warp_kernel(6, str(device.arch), warps_per_block=4),
                dim=groups * 32,
                block_dim=128,
                inputs=[
                    groups,
                    *common,
                    self.lower_schedule,
                    1,
                    f.drive_row_by_dof,
                    self.row_K,
                ],
                outputs=[held.lower6],
                device=device,
            )
            wp.launch(
                original._get_inverse_cholesky_register_kernel(6, str(device.arch), lower_only=True),
                dim=groups * 32,
                block_dim=256,
                inputs=[held.lower6, self.group_to_art, self.mass_mask],
                outputs=[held.inverse6],
                device=device,
            )
        else:
            wp.launch(
                assemble_free_cpu_reference,
                dim=(groups, 6, 6),
                inputs=[
                    self.articulation_start,
                    self.articulation_dof_start,
                    self.mass_mask,
                    p.joint_child,
                    p.joint_S_s,
                    p.body_I_s,
                    self.group_to_art,
                    f.drive_row_by_dof,
                    self.row_K,
                    self.cpu_H,
                ],
                device=device,
            )
            wp.launch_tiled(
                original._get_cholesky_kernel(6, str(device.arch), 64),
                dim=[groups],
                block_dim=64,
                inputs=[
                    self.cpu_H,
                    self.R_group,
                    self.group_to_art,
                    self.mass_mask,
                    held.lower6,
                ],
                device=device,
            )
            wp.launch(
                inverse_lower_cpu_reference,
                dim=(groups, 6),
                inputs=[held.lower6, self.group_to_art, self.mass_mask, held.inverse6],
                device=device,
            )


def bind_free_refresh(bundle, publication, requested, *, R_group=None):
    """Bind exact original group order/current source; no saved state factor input.

    The supplied world mask must already include interval/global/device requests.
    R_group defaults to the fixture's actual grouped R6, not primary RefreshInput.R
    (that descriptor deliberately has zero free entries). Current callers may
    provide a live grouped R6 array; it is read on every requested refresh.
    """
    checked_definitions(
        "solver_feather_pgs.py",
        ("_get_crba_cholesky_warp_kernel", "_get_inverse_cholesky_register_kernel", "_get_cholesky_kernel"),
    )
    host, plan = bundle["host"], bundle["host"]["plan"]
    snapshot = host["snapshot"]
    device = wp.get_device(bundle["launch"]["device"])
    worlds = host["worlds"]
    groups = np.asarray(snapshot["group_to_art_6"], np.int32)
    starts = np.asarray(snapshot["solve_articulation_dof_start"], np.int32)
    art_start = np.asarray(snapshot["full_model_articulation_start"], np.int32)
    selected = groups[plan["secondary_group"]]
    if len(groups) != worlds or len(np.unique(selected)) != worlds or set(selected.tolist()) != set(groups.tolist()):
        raise ValueError("Free articulation mapping is not exhaustive/disjoint")
    children = publication.joint_child.numpy()
    if not (
        np.array_equal(children[art_start[selected]], plan["body_ids"][:, 30])
        and np.array_equal(starts[selected, None] + np.arange(6), plan["dof_ids"][:, 23:29])
    ):
        raise ValueError("Free6 single-joint original schedule mismatch")
    if requested.dtype != wp.int32 or requested.shape != (worlds,) or requested.device != device:
        raise ValueError("Expected same-device world requested mask")
    for name in ("lower6", "inverse6"):
        a = getattr(bundle["held"], name)
        if a.shape != (len(groups), 6, 6) or a.device != device:
            raise ValueError("Original grouped factor shape/device mismatch")
    row_map = bundle["inputs"].drive_row_by_dof.numpy()
    free_rows = row_map[plan["dof_ids"][:, 23:29]].ravel()
    enabled = free_rows[free_rows >= 0]
    if len(np.unique(enabled)) != len(enabled):
        raise ValueError("Free drive rows must have unique original owners")
    row_capacity = max(1, int(row_map.max()) + 1)
    if R_group is None:
        R_group = wp.array(snapshot["post3_R_6"].copy(), dtype=float, device=device)
    if R_group.shape != (len(groups), 6) or R_group.dtype != wp.float32 or R_group.device != device:
        raise ValueError("Expected actual current grouped R6")
    schedule = np.asarray(
        [r * 6 + c | (c + 1) << 8 | r << 16 for r in range(6) for c in range(r + 1)],
        np.int32,
    )
    return FreeRefresh(
        bundle,
        publication,
        requested,
        R_group,
        wp.array(art_start, dtype=int, device=device),
        wp.array(starts, dtype=int, device=device),
        wp.array(groups, dtype=int, device=device),
        wp.zeros(len(starts), dtype=int, device=device),
        wp.zeros(row_capacity, dtype=float, device=device),
        wp.zeros(6, dtype=int, device=device),
        wp.array(schedule, dtype=int, device=device),
        wp.zeros((len(groups), 6, 6), dtype=float, device=device) if not device.is_cuda else None,
    )
