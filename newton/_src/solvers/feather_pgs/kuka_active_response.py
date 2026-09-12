# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compact active dispatch for the unchanged Kuka 23+6 paired response law."""

import hashlib
import importlib
import inspect
from functools import cache

import warp as wp

_FACTORY_SHA256 = "458cfb455427e6f1da5d30325e290626450dfaf51f47758543b4999fb482f259"
_COUNT_LINE = "    int n_constraints = world_constraint_count.data[world];"
_EMPTY_GUARD = "\n    if (n_constraints <= 0) return;"

# The serial CPU control has the same held-factor action and publication law;
# the CUDA branch below is recovered from the original factory, not retyped.
_CPU_RESPONSE = """
#else
    const int secondary_group = secondary_group_by_primary.data[primary_group];
    const int primary_art = primary_group_to_art.data[primary_group];
    const int secondary_art = secondary_group_to_art.data[secondary_group];
    const int world = art_to_world.data[primary_art];
    const int m = wp::min(world_constraint_count.data[world], 192);
    const int po = articulation_world_dof_offset.data[primary_art];
    const int so = articulation_world_dof_offset.data[secondary_art];
    const bool factor_world = world_mf_constraint_count.data[world] == 0;
    for (int row = 0; row < m; ++row) {
        float j[29], z[29], y[29];
        for (int d = 0; d < 29; ++d) {
            j[d] = d < 23 ? primary_J.data[(primary_group * 192 + row) * 23 + d]
                : secondary_J.data[(secondary_group * 192 + row) * 6 + d - 23];
            z[d] = 0.0f;
            y[d] = 0.0f;
        }
        for (int d = 0; d < 23; ++d)
            for (int k = 0; k <= d; ++k)
                z[d] += primary_Linv.data[primary_group * 529 + d * 23 + k] * j[k];
        for (int d = 0; d < 6; ++d)
            for (int k = 0; k <= d; ++k)
                z[23 + d] += secondary_Linv.data[secondary_group * 36 + d * 6 + k] * j[23 + k];
        if (!factor_world) {
            for (int d = 0; d < 23; ++d)
                for (int k = d; k < 23; ++k)
                    y[d] += primary_Linv.data[primary_group * 529 + k * 23 + d] * z[k];
            for (int d = 0; d < 6; ++d)
                for (int k = d; k < 6; ++k)
                    y[23 + d] += secondary_Linv.data[secondary_group * 36 + k * 6 + d] * z[23 + k];
        }
        float diagonal = 0.0f;
        for (int d = 0; d < 29; ++d) {
            const int coord = d < 23 ? po + d : so + d - 23;
            const int index = (world * 192 + row) * 29 + coord;
            J_world.data[index] = j[d];
            Y_world.data[index] = factor_world ? z[d] : y[d];
            diagonal += factor_world ? z[d] * z[d] : j[d] * y[d];
        }
        world_diag.data[world * 192 + row] = diagonal + world_row_cfm.data[world * 192 + row];
    }
"""


def response_sources(device_arch, warps_per_block=4):
    """Return original and adapted snippets, rejecting an unreviewed source seam."""
    # The solver imports this owner; resolve its factory after construction.
    _get_paired_hinv_jt_kernel = importlib.import_module(f"{__package__}.solver_feather_pgs")._get_paired_hinv_jt_kernel
    source = inspect.getsource(_get_paired_hinv_jt_kernel)
    if hashlib.sha256(source.encode()).hexdigest() != _FACTORY_SHA256:
        raise RuntimeError("The original paired response factory changed; review the active adapter")
    original = _get_paired_hinv_jt_kernel(23, 6, 192, 29, device_arch, warps_per_block, factor_coordinates=True)
    native = [cell.cell_contents for cell in original.func.__closure__ if hasattr(cell.cell_contents, "native_snippet")]
    if len(native) != 1:
        raise RuntimeError("Expected exactly one original paired response native function")
    snippet = native[0].native_snippet
    if snippet.count(_COUNT_LINE) != 1 or snippet.count("\n#endif") != 1:
        raise RuntimeError("The paired response count or native branch seam changed")
    adapted = snippet.replace(_COUNT_LINE, _COUNT_LINE + _EMPTY_GUARD)
    return snippet, adapted.replace("\n#endif", _CPU_RESPONSE + "\n#endif")


@cache
def get_response_kernel(device_arch, warps_per_block=4):
    """Build a fixed-capacity active-list launch with original row-parallel math.

    Arguments are the original 17 paired-response arguments followed by the
    current compact active-world list, its count, and the static world-to-primary
    group map. The light owner must validate the complete list before dispatch.
    No count or capacity is changed here. Mixed MF worlds retain physical Y.
    """
    _, snippet = response_sources(device_arch, warps_per_block)
    cpu = str(device_arch) == "cpu"

    @wp.func_native(snippet)
    def active_response_native(
        primary_group: int,
        primary_Hinv: wp.array3d[float],
        primary_Linv: wp.array3d[float],
        primary_J: wp.array3d[float],
        primary_group_to_art: wp.array[int],
        secondary_Hinv: wp.array3d[float],
        secondary_Linv: wp.array3d[float],
        secondary_J: wp.array3d[float],
        secondary_group_to_art: wp.array[int],
        secondary_group_by_primary: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        world_constraint_count: wp.array[int],
        world_mf_constraint_count: wp.array[int],
        world_row_cfm: wp.array2d[float],
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
        world_diag: wp.array2d[float],
    ): ...

    def active_response_template(
        primary_Hinv: wp.array3d[float],
        primary_Linv: wp.array3d[float],
        primary_J: wp.array3d[float],
        primary_group_to_art: wp.array[int],
        secondary_Hinv: wp.array3d[float],
        secondary_Linv: wp.array3d[float],
        secondary_J: wp.array3d[float],
        secondary_group_to_art: wp.array[int],
        secondary_group_by_primary: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        world_constraint_count: wp.array[int],
        world_mf_constraint_count: wp.array[int],
        world_row_cfm: wp.array2d[float],
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
        world_diag: wp.array2d[float],
        active_worlds: wp.array[int],
        active_count: wp.array[int],
        primary_group_by_world: wp.array[int],
    ):
        active_index, lane = wp.tid()
        if wp.static(cpu):
            if lane != 0:
                return
        if active_index >= active_count[0]:
            return
        world = active_worlds[active_index]
        primary_group = primary_group_by_world[world]
        active_response_native(
            primary_group,
            primary_Hinv,
            primary_Linv,
            primary_J,
            primary_group_to_art,
            secondary_Hinv,
            secondary_Linv,
            secondary_J,
            secondary_group_to_art,
            secondary_group_by_primary,
            art_to_world,
            articulation_world_dof_offset,
            world_constraint_count,
            world_mf_constraint_count,
            world_row_cfm,
            J_world,
            Y_world,
            world_diag,
        )

    name = f"kuka_active_response_23_6_192_bd{32 * warps_per_block}"
    active_response_template.__name__ = name
    active_response_template.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(active_response_template)


class ActiveKukaResponse:
    """Bind canonical buffers to the light owner's already-validated active list."""

    def __init__(self, solver, active_worlds, active_count, *, warps_per_block=4):
        if (
            solver._paired_response_primary_size != 23
            or solver._paired_response_secondary_size != 6
            or not solver._paired_factor_coordinates
            or solver.dense_max_constraints != 192
            or solver.max_world_dofs != 29
            or active_worlds.shape != (solver.world_count,)
            or active_count.shape != (1,)
        ):
            raise ValueError("Active response requires the original Kuka 23+6 factor-coordinate layout")
        self.solver = solver
        self.active_worlds = active_worlds
        self.active_count = active_count
        self.block_dim = 32 * warps_per_block
        arch = str(solver.model.device.arch) if solver.model.device.is_cuda else "cpu"
        self.kernel = get_response_kernel(arch, warps_per_block)

    def inputs(self):
        """Read current held factors and canonical owners without allocating or copying."""
        solver = self.solver
        return [
            solver.Hinv_by_size[23],
            solver.Linv_by_size[23],
            solver.J_by_size[23],
            solver.group_to_art[23],
            solver.Hinv_by_size[6],
            solver.Linv_by_size[6],
            solver.J_by_size[6],
            solver.group_to_art[6],
            solver._paired_response_secondary_groups,
            solver.art_to_world,
            solver.articulation_world_dof_offset,
            solver.constraint_count,
            solver.mf_constraint_count,
            solver.row_cfm,
            solver.J_world,
            solver.Y_world,
            solver.diag,
            self.active_worlds,
            self.active_count,
            solver._paired_factor_primary_groups_by_world,
        ]

    def launch(self):
        """Launch before original bias/restitution and mixed-world qualification."""
        wp.launch_tiled(
            self.kernel,
            dim=[self.solver.world_count],
            inputs=self.inputs(),
            block_dim=self.block_dim,
            device=self.solver.model.device,
        )
