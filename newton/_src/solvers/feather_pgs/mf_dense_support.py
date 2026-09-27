# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepare exact 32-DOF tile support for dense constraint rows."""

import warp as wp


@wp.kernel(enable_backward=False)
def _mark_support(
    counts: wp.array[int],
    J_world: wp.array3d[float],
    Y_world: wp.array3d[float],
    masks: wp.array3d[wp.uint32],
):
    world, row, dof = wp.tid()
    if row < counts[world]:
        # Y-only response entries must remain visible even when J is zero.
        # Exact comparison also retains NaNs conservatively.
        if J_world[world, row, dof] != 0.0 or Y_world[world, row, dof] != 0.0:
            tile = dof // 32
            bit = wp.uint32(1) << wp.uint32(tile % 32)
            wp.atomic_or(masks, world, row, tile // 32, bit)


@wp.kernel(enable_backward=False)
def _merge_world_support(masks: wp.array3d[wp.uint32], world_masks: wp.array2d[wp.uint32]):
    world, row, word = wp.tid()
    bits = masks[world, row, word]
    if bits != wp.uint32(0):
        wp.atomic_or(world_masks, world, word, bits)


class DenseRowSupport:
    """Build exact operator support once for later repeated tile-restricted sweeps.

    Preparation reads O(mD) operator entries for m active rows and D world DOFs.
    A consumer can then visit only marked 32-DOF tiles instead of all D entries
    each sweep. A tile is retained whenever either J or Y has a nonzero entry;
    there is no magnitude threshold, row truncation or approximate fallback.
    Optional ``stage_velocities`` also merges row support into a world union,
    allowing consumers to stage only velocities that any row can read or write.
    Rebuild support after toggling it on; when off, union preparation is omitted.
    This helper prepares masks only and does not change the solver dispatch.
    """

    def __init__(self, world_count, max_rows, max_world_dofs, device, *, stage_velocities=False):
        if world_count < 0 or max_rows <= 0 or max_world_dofs <= 0:
            raise ValueError("world_count must be nonnegative and row/DOF capacities must be positive")
        self.world_count = int(world_count)
        self.max_rows = int(max_rows)
        self.max_world_dofs = int(max_world_dofs)
        self.device = wp.get_device(device)
        self.tile_count = (self.max_world_dofs + 31) // 32
        self.mask_word_count = (self.tile_count + 31) // 32
        self.stage_velocities = bool(stage_velocities)
        self.masks = wp.zeros(
            (self.world_count, self.max_rows, self.mask_word_count), dtype=wp.uint32, device=self.device
        )
        self.world_masks = wp.zeros((self.world_count, self.mask_word_count), dtype=wp.uint32, device=self.device)

    def prepare(self, counts, J_world, Y_world):
        """Rebuild support from current active rows, clearing all inactive-row masks."""
        self.masks.zero_()
        if self.stage_velocities:
            self.world_masks.zero_()
        if not self.world_count:
            return
        wp.launch(
            _mark_support,
            dim=(self.world_count, self.max_rows, self.max_world_dofs),
            inputs=[counts, J_world, Y_world],
            outputs=[self.masks],
            device=self.device,
        )
        if self.stage_velocities:
            wp.launch(
                _merge_world_support,
                dim=self.masks.shape,
                inputs=[self.masks],
                outputs=[self.world_masks],
                device=self.device,
            )
