# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check explicit writer bindings and handled-triangle query composition."""

import unittest

import warp as wp

from newton._src.geometry.contact_reduction_global import create_mesh_triangle_contacts_to_reducer_kernel
from newton._src.geometry.heightfield_finite import create_query_kernel
from newton._src.geometry.heightfield_pair_csr import PairCSRData, _write_pair
from newton._src.geometry.narrow_phase import create_narrow_phase_process_mesh_triangle_contacts_kernel
from newton._src.sim.collide import write_contact


@wp.func
def bind_pair(data: PairCSRData, index: int) -> PairCSRData:
    local = data
    local.pair_index = data.triangle_pair[index]
    return local


class TestHeightfieldQueryFactories(unittest.TestCase):
    def test_explicit_writer_binding(self):
        """Compose typed pair-local writers without rewriting query source."""
        options = {"writer_binding": bind_pair, "writer_data_type": PairCSRData}
        finite = create_query_kernel(_write_pair, **options)
        generic = create_mesh_triangle_contacts_to_reducer_kernel(writer_func=_write_pair, skip_marked=True, **options)
        self.assertIs(finite, create_query_kernel(_write_pair, **options))
        self.assertIsNot(finite, create_query_kernel(_write_pair))
        self.assertIsNot(generic, create_mesh_triangle_contacts_to_reducer_kernel())
        self.assertIs(finite.func.__annotations__["writer_data"], PairCSRData)

    def test_explicit_marker_policy(self):
        """Keep direct and reduced fallback selection in their query owners."""
        for factory, args in (
            (create_narrow_phase_process_mesh_triangle_contacts_kernel, (write_contact,)),
            (create_mesh_triangle_contacts_to_reducer_kernel, ()),
        ):
            with self.subTest(factory=factory.__name__):
                ordinary = factory(*args)
                marked = factory(*args, skip_marked=True)
                self.assertNotEqual(ordinary.key, marked.key)
                self.assertIn("finite_fallback", marked.key)


if __name__ == "__main__":
    unittest.main()
