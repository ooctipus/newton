# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dimensions and phase lifetimes for a scalar tree with two free roots."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ForestLayout:
    """Bind physical ownership and nonoverlapping compact scratch intervals."""

    primary_bodies: int = 11
    primary_dofs: int = 9

    def __post_init__(self):
        if not (0 < self.primary_dofs <= 9 and self.primary_dofs < self.primary_bodies <= 14):
            raise ValueError("Scalar forest requires at most nine primary coordinates and sixteen bodies")

    @property
    def bodies(self):
        return self.primary_bodies + 2

    @property
    def physical_dofs(self):
        return self.primary_dofs + 12

    @property
    def response_dofs(self):
        return self.primary_dofs + 6

    @property
    def actions(self):
        return 19 * self.bodies

    @property
    def axes(self):
        # Scan slots and inertia-axis actions have separate lifetimes. Axes
        # survive both phases; they must follow either phase's complete extent.
        return max(19 * 16, self.actions + 6 * self.primary_dofs)

    @property
    def origins(self):
        return self.axes + 6 * self.primary_dofs

    @property
    def shifts(self):
        return self.origins + 9

    @property
    def state_floats(self):
        return self.shifts + 9

    @property
    def torque(self):
        return 6 * self.bodies

    @property
    def acceleration(self):
        return self.torque + self.response_dofs

    @property
    def force_floats(self):
        return max(120, self.acceleration + self.response_dofs)


DEFAULT_LAYOUT = ForestLayout()


def from_solver(solver):
    """Derive candidate dimensions; the topology planner proves complete ownership."""
    sizes = list(solver.size_groups)
    if sizes == [6]:
        primary = 6
    elif len(sizes) == 2 and 6 in sizes:
        primary = next(size for size in sizes if size != 6)
    else:
        raise ValueError("Scalar forest requires a primary and a six-coordinate free factor group")
    worlds = int(solver.world_count)
    if worlds <= 0 or solver.model.body_count % worlds:
        raise ValueError("Scalar forest requires complete homogeneous worlds")
    return ForestLayout(solver.model.body_count // worlds - 2, primary)
