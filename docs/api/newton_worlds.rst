.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.worlds
=============

GPU world identities and explicit lifecycle transactions.

.. experimental::
    This entire module is experimental and may change without deprecation.

The directory owns identities and membership only. Domains own physical state,
initialize admitted requests, and acknowledge row copies before publication.
Use :class:`newton.solvers.MuJoCoWorlds` for the concrete native physics runtime.

.. py:module:: newton.worlds
.. currentmodule:: newton.worlds

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   WorldBatch
   WorldCommands
   WorldCompaction
   WorldDirectory
   WorldDirectoryData
   WorldOperation
   WorldPhase
   WorldResults
   WorldStatus
   WorldTransaction

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   create_world_commands
   create_world_results
   world_handle_at
   world_location
