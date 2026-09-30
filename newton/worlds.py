# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU world identities and explicit lifecycle transactions.

.. experimental::
    This entire module is experimental and may change without deprecation.

The directory owns identities and membership only. Domains own physical state,
initialize admitted requests, and acknowledge row copies before publication.
Use :class:`newton.solvers.MuJoCoWorlds` for the concrete native physics runtime.
"""

from ._src.sim.worlds import (
    WorldCommands,
    WorldCompaction,
    WorldDirectory,
    WorldDirectoryData,
    WorldOperation,
    WorldPhase,
    WorldResults,
    WorldStatus,
    WorldTransaction,
    create_world_commands,
    create_world_results,
)

__all__ = [
    "WorldCommands",
    "WorldCompaction",
    "WorldDirectory",
    "WorldDirectoryData",
    "WorldOperation",
    "WorldPhase",
    "WorldResults",
    "WorldStatus",
    "WorldTransaction",
    "create_world_commands",
    "create_world_results",
]
