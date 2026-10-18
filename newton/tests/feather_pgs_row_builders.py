# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the contact-row builders a FeatherPGS route record names against the kernels a step launches.

``get_path_selection()["routes"]["contact_rows"]["builder_by_dof"]`` names, per DOF
size, the kernels that build the contact rows. :func:`assert_row_builders_named`
requires those names to be exactly the contact-row builders among the kernels one
recorded step launched. This module holds shared test helpers; it defines no tests.
"""

import contextlib
from unittest import mock

import warp as wp

# Every kernel a contact-row producer launches to write dense contact rows: the general builder's
# methods, the shared-metadata producers, the compact contact boundary, local row packets and keyed
# rows. The free-body matrix-free and propagation rows follow the allocation path, not a producer.
ROW_BUILDER_KERNELS = frozenset(
    (
        "populate_world_J_for_size",
        "contact_row_prelude",
        "contact_row_prelude_patch",
        "populate_world_J_masked",
        "populate_world_J_masked_patch",
        "populate_world_J_native",
        "build_contact_rows",
        "prepare_world_contact_rows",
        "populate_world_J_for_compact_size",
        "populate_sparse_diagonal_contact_response",
        "clear_contact_response",
        "produce_contacts",
        "produce_patch_contacts",
        "_build_contact_order",
        "_produce_ordered_contacts",
        "_produce_ordered_patch_contacts",
        "materialize_fallback",
        "materialize_patch_fallback",
        "scatter_contact_keys",
        "scatter_patch_contact_keys",
    )
)
# Kernels their factories name: the sparse response, the sparse-diagonal schedule marker and the
# keyed-row fallbacks.
_GENERATED_PREFIXES = ("populate_sparse_contact_response_", "mark_independent_sparse_contact_candidates")
_GENERATED_SUFFIXES = ("keyed_rows_fallback", "keyed_rows_patch_fallback")


def row_builder_kernels(keys) -> set:
    """Return the contact-row builder kernels among launched kernel keys."""
    return {
        key
        for key in keys
        if key is not None
        and (key in ROW_BUILDER_KERNELS or key.startswith(_GENERATED_PREFIXES) or key.endswith(_GENERATED_SUFFIXES))
    }


@contextlib.contextmanager
def launched_kernels():
    """Record the key of every kernel ``wp.launch`` and ``wp.launch_tiled`` launch inside the block."""
    keys = []

    def recording(launch):
        def record(kernel, *args, **kwargs):
            keys.append(getattr(kernel, "key", None))
            return launch(kernel, *args, **kwargs)

        return record

    with mock.patch.multiple(wp, launch=recording(wp.launch), launch_tiled=recording(wp.launch_tiled)):
        yield keys


def assert_row_builders_named(test, contact_rows, keys):
    """Require ``builder_by_dof`` to name exactly the contact-row builders among ``keys``.

    Args:
        test: The running test case.
        contact_rows: ``get_path_selection()["routes"]["contact_rows"]``, taken under the switches
            the recorded step ran with.
        keys: The kernel keys of at least one whole step.
    """
    builders = contact_rows["builder_by_dof"]
    named = {kernel for entry in builders.values() for kernel in entry["kernels"]}
    test.assertTrue(named, builders)
    test.assertEqual(row_builder_kernels(keys), named, builders)
    # Every producer writes the solver's friction law for every size.
    test.assertEqual({entry["law"] for entry in builders.values()}, {contact_rows["law"]}, builders)
