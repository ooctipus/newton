# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reject far raw candidates while retaining the final writer's admitted contacts."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.contact_data import ContactData
from newton._src.geometry.contact_reduction_global import (
    GlobalContactReducer,
    GlobalContactReducerData,
    decode_oct,
    encode_oct,
    write_contact_to_reducer,
)
from newton._src.sim.collide import ContactWriterData, write_contact


@wp.kernel(enable_backward=False)
def _buffer(contacts: wp.array[ContactData], reducer: GlobalContactReducerData):
    write_contact_to_reducer(contacts[wp.tid()], reducer, -1)


@wp.kernel(enable_backward=False)
def _compare_writer(contacts: wp.array[ContactData], reducer: GlobalContactReducerData, writer: ContactWriterData):
    contact = contacts[wp.tid()]
    write_contact_to_reducer(contact, reducer, -1)
    # The final exporter sees the packed normal, including its existing rounding.
    contact.contact_normal_a_to_b = decode_oct(encode_oct(contact.contact_normal_a_to_b))
    write_contact(contact, writer, -1)


def _contact(index, position, normal, depth, gap, margins=(0.0, 0.0), radii=(0.0, 0.0)):
    contact = ContactData()
    contact.contact_point_center = wp.vec3(*position)
    contact.contact_normal_a_to_b = wp.vec3(*normal)
    contact.contact_distance = depth
    contact.gap_sum = gap
    contact.margin_a, contact.margin_b = margins
    contact.radius_eff_a, contact.radius_eff_b = radii
    contact.shape_a, contact.shape_b = 2 * index, 2 * index + 1
    contact.sort_sub_key = index
    return contact


def _writer(contacts, device):
    count = len(contacts)
    writer = ContactWriterData()
    writer.contact_max = count
    writer.contact_count = wp.zeros(1, dtype=int, device=device)
    writer.shape_body = wp.full(2 * count, -1, dtype=int, device=device)
    writer.shape_gap = wp.array(np.repeat([c.gap_sum / 2 for c in contacts], 2), dtype=float, device=device)
    for name in ("out_shape0", "out_shape1", "out_tids"):
        setattr(writer, name, wp.empty(count, dtype=int, device=device))
    for name in ("out_point0", "out_point1", "out_offset0", "out_offset1", "out_normal"):
        setattr(writer, name, wp.empty(count, dtype=wp.vec3, device=device))
    for name in ("out_margin0", "out_margin1"):
        setattr(writer, name, wp.empty(count, dtype=float, device=device))
    return writer


class TestContactReducerGapScreen(unittest.TestCase):
    def check_far_capacity(self, device):
        contacts = [_contact(i, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.5, 0.01) for i in range(64)]
        reducer = GlobalContactReducer(8, device=device)
        values = wp.array(contacts, dtype=ContactData, device=device)
        wp.launch(_buffer, dim=len(contacts), inputs=[values, reducer.get_data_struct()], device=device)
        self.assertEqual(int(reducer.contact_count.numpy()[0]), 0)
        self.assertEqual(int(reducer.buffer_overflows.numpy()[0]), 0)
        reducer.clear()
        wp.launch(_buffer, dim=len(contacts), inputs=[values, reducer.get_data_struct()], device=device)
        self.assertEqual(int(reducer.contact_count.numpy()[0]), 0)

    def check_writer_coverage(self, device):
        rng = np.random.default_rng(91023)
        contacts = []
        for position_scale in (0.0, 1.0, 100.0, 10000.0):
            for gap in (0.0, 0.002, 0.2):
                for offset in (-0.1, -1e-5, 0.0, 1e-5, 0.1):
                    for radii in ((0.0, 0.0), (0.1, 0.04)):
                        normal = rng.normal(size=3)
                        normal /= np.linalg.norm(normal)
                        margins = (0.003, 0.007)
                        contacts.append(
                            _contact(
                                len(contacts),
                                rng.uniform(-1, 1, 3) * position_scale,
                                normal,
                                gap + sum(margins) + offset,
                                gap,
                                margins,
                                radii,
                            )
                        )
        # Extended search gaps, penetration and degenerate normals retain their inputs.
        contacts.append(_contact(len(contacts), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.15, 0.2))
        contacts.append(_contact(len(contacts), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), -0.1, 0.01))
        reducer = GlobalContactReducer(len(contacts), device=device)
        writer = _writer(contacts, device)
        values = wp.array(contacts, dtype=ContactData, device=device)
        wp.launch(_compare_writer, dim=len(contacts), inputs=[values, reducer.get_data_struct(), writer], device=device)
        count = int(reducer.contact_count.numpy()[0])
        retained = set(reducer.contact_fingerprints.numpy()[1 : count + 1].tolist())
        accepted_count = int(writer.contact_count.numpy()[0])
        accepted = set((writer.out_shape0.numpy()[:accepted_count] // 2).tolist())
        self.assertGreater(accepted_count, 50)
        self.assertTrue(accepted <= retained, sorted(accepted - retained))
        self.assertIn(len(contacts) - 2, retained)
        self.assertIn(len(contacts) - 1, retained)
        self.assertLess(count, len(contacts))
        self.assertEqual(int(reducer.buffer_overflows.numpy()[0]), 0)

    def test_far_capacity_cpu(self):
        """Avoid raw-buffer allocation for far candidates on CPU."""
        self.check_far_capacity("cpu")

    def test_writer_coverage_cpu(self):
        """Retain contacts admitted by the final writer on CPU."""
        self.check_writer_coverage("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_far_capacity_cuda(self):
        """Avoid raw-buffer allocation for far candidates on CUDA."""
        self.check_far_capacity("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_writer_coverage_cuda(self):
        """Retain contacts admitted by the final writer on CUDA."""
        self.check_writer_coverage("cuda:0")


if __name__ == "__main__":
    unittest.main()
