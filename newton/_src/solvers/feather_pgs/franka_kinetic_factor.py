# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bind held geometric mass to the shared small Cholesky factory."""

from functools import cache

from . import small_cholesky


@cache
def get_kernel(device_arch: str, *, warps_per_block: int = 4, size: int = 9, geometric_index=()):
    """Bind admitted physical geometry to the retained Cholesky arithmetic."""
    del device_arch
    return small_cholesky.get_kernel(
        size, warps_per_block=warps_per_block, geometric=True, geometric_index=geometric_index
    )
