# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bind regular-cylinder support to the shared terrain query schedule."""

from .heightfield_split import create_heightfield_split_kernels
from .support_function import SupportMapDataProvider, support_map_heightfield_cylinder
from .types import GeoType


def create_heightfield_cylinder_split_kernels():
    """Create regular-cylinder MPR, queued GJK and contact publication kernels."""
    return create_heightfield_split_kernels(
        int(GeoType.CYLINDER), support_map_heightfield_cylinder, SupportMapDataProvider, precomputed_center=True
    )
