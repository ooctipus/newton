# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the contact-row builders a FeatherPGS route record names against the kernels a step launches.

``get_path_selection()["routes"]["contact_rows"]["builder_by_dof"]`` names, per DOF
size, the kernels that build the dense contact rows outside the sweep, in launch
order. :func:`assert_row_builders_named` requires those names to be exactly the
contact-row builders among the kernels one recorded step launched, in that order
and with one launch per step. It defines no tests.
"""

import collections
import contextlib
from unittest import mock

import warp as wp

# Every kernel a contact-row producer launches to write dense contact rows outside the sweep: the one writer, the
# compact contact boundary. The free-body matrix-free rows follow the allocation path, not a producer.
# Add every new producer kernel here: a launched builder missing from this list is invisible to the check below.
ROW_BUILDER_KERNELS = frozenset(
    (
        "write_walked_rows",
        "write_slot_rows",
        "produce_contacts",
    )
)
# Kernels their factories name: the support placement's writer (its units and its walkers), the sparse-diagonal
# schedule marker and the one writer's variants (its release and block phases).
_GENERATED_PREFIXES = (
    "support_rows_",
    "support_units_",
    "mark_independent_sparse_contact_candidates",
    "write_walked_rows_",
)


def is_row_builder(key) -> bool:
    """Whether a launched kernel key is a contact-row builder."""
    return key is not None and (key in ROW_BUILDER_KERNELS or key.startswith(_GENERATED_PREFIXES))


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
    """Require the record to name exactly the contact-row builders among ``keys``, in order and count.

    ``builder_by_dof`` names every launched builder and only those; each entry's kernels launch in the order the entry lists them
    (their launches, with repeats removed, start with the list); each builder launches once a step.

    Args:
        test: The running test case.
        contact_rows: ``get_path_selection()["routes"]["contact_rows"]``, taken under the switches
            the recorded step ran with.
        keys: The kernel keys of one whole step.
    """
    groups = [contact_rows["builder_by_dof"]]
    entries = [(size, entry) for group in groups for size, entry in group.items()]
    launched = [key for key in keys if is_row_builder(key)]
    named = {kernel for _, entry in entries for kernel in entry["kernels"]}
    test.assertTrue(named, groups)
    test.assertEqual(set(launched), named, groups)
    test.assertEqual(collections.Counter(launched), collections.Counter(named), groups)
    for size, entry in entries:
        own = [key for key in launched if key in entry["kernels"]]
        own = [key for index, key in enumerate(own) if index == 0 or own[index - 1] != key]
        test.assertEqual(own[: len(entry["kernels"])], entry["kernels"], (size, entry, launched))
