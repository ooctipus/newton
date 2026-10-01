.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.worlds
=============

Compose growable GPU populations from explicit lifetime, storage and graph owners.

.. experimental::
    This entire module is experimental and may change without deprecation.

The directory owns identities and membership only. Domains own physical state,
initialize admitted requests, and acknowledge row copies before publication.
Field storage owns typed layouts; memory backing owns bytes and mappings; graph
updates own bindings to borrowed count sources. These mechanisms do not import a
solver or select a domain's initialization, relocation or failure policy.
Use :class:`newton.solvers.MuJoCoWorlds` for the concrete native physics runtime.

Import these contracts here rather than from ``newton._src``. Importing this
module does not initialize CUDA or a physics engine. CUDA storage and graph
mechanisms require Python 3.11 or newer; the directory also supports CPU execution.

Reserved capacity, certified ready prefixes and protected prefixes are separate
contracts. Counts used by recorded prefix operations must remain within each
operand's ready prefix. Mechanical maintenance accepts raw CUDA stream handles;
the concrete domain joins all consumers before remapping or retiring storage.
``MuJoCoWorlds`` provides its own Warp-stream interface and converts at this boundary.

Enable counts, launch extents and integer parameters have independent declared
sources. Every registered consumer and recorded updater must bind before graph
instantiation. Domain owners select overflow and advancement policies.
See :doc:`/design/dynamic_worlds` for complete operation sequences and admitted scope.

.. py:module:: newton.worlds
.. currentmodule:: newton.worlds

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   DeviceGraphUpdates
   FieldSpec
   FieldStorage
   FieldTransfer
   FieldView
   GraphKernelBinding
   KernelParameterBinding
   MemoryBacking
   VirtualReservation
   WorldBatchResult
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

   capture_parallel
   create_world_commands
   create_world_results
   world_handle_at
   world_location
