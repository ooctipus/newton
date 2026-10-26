# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The device idioms of FeatherPGS: lane groups, patch rings and the slot reservation contract.

The lane group's reductions replace shuffle-down ladders whose lane 0 the
sweeps read; a butterfly from the widest offset down gives every lane that
value bit for bit, which the CUDA tests below check against the ladder itself.
"""

import unittest
from functools import cache

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.device_preludes import (
    RING,
    cuda_native,
    dual_native,
    lane_group,
    reserve,
    reserved_count,
)

_UNBOUNDED = 2**31 - 1


@wp.kernel(enable_backward=False)
def _reserve_in_order(
    sizes: wp.array[int],
    capacity: int,
    counter: wp.array[int],
    first_rejected: wp.array[int],
    slots: wp.array[int],
    count: wp.array[int],
    status: wp.array[int],
):
    # One thread reserves every request in order, so the slots are those of a serial allocator.
    for request in range(sizes.shape[0]):
        slots[request] = reserve(counter, 0, sizes[request], capacity, first_rejected)
    count[0] = reserved_count(counter, 0, capacity, first_rejected, status, 1)


def _reserved(sizes, capacity):
    device = "cpu"
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    first_rejected = wp.full(1, _UNBOUNDED, dtype=wp.int32, device=device)
    slots = wp.empty(len(sizes), dtype=wp.int32, device=device)
    count = wp.empty(1, dtype=wp.int32, device=device)
    status = wp.zeros(3, dtype=wp.int32, device=device)
    wp.launch(
        _reserve_in_order,
        dim=1,
        inputs=[wp.array(sizes, dtype=wp.int32, device=device), capacity],
        outputs=[counter, first_rejected, slots, count, status],
        device=device,
    )
    return (
        slots.numpy().tolist(),
        int(counter.numpy()[0]),
        int(first_rejected.numpy()[0]),
        int(count.numpy()[0]),
        status.numpy().tolist(),
    )


class TestSlotReservations(unittest.TestCase):
    def test_reservations_that_fit_take_consecutive_slots(self):
        """Give each reservation that fits its consecutive slots and leave the status clear."""
        slots, counter, first_rejected, count, status = _reserved([3, 1, 2], 6)
        self.assertEqual(slots, [0, 3, 4])
        self.assertEqual((counter, first_rejected, count, status), (6, _UNBOUNDED, 6, [0, 0, 0]))

    def test_a_reservation_fits_whole_or_not_at_all(self):
        """Reject a reservation that crosses capacity whole and truncate the count at its first slot.

        A triple at slot 4 of 6 would leave two slots counted but never written; the first
        rejected slot excludes them, and every later reservation is rejected too.
        """
        slots, counter, first_rejected, count, status = _reserved([3, 1, 3, 1], 6)
        self.assertEqual(slots, [0, 3, -1, -1])
        self.assertEqual((counter, first_rejected, count), (8, 4, 4))
        self.assertEqual(status, [0, 1, 0])

    def test_one_slot_rejections_clamp_at_capacity(self):
        """Count every slot below capacity when one-slot reservations overflow, and raise the family's status."""
        slots, counter, first_rejected, count, status = _reserved([1] * 5, 3)
        self.assertEqual(slots, [0, 1, 2, -1, -1])
        self.assertEqual((counter, first_rejected, count), (5, 3, 3))
        self.assertEqual(status, [0, 1, 0])


@cache
def _ring_kernel():
    """Pool the values around each row's patch ring with the native walks, as the sweeps do."""
    body = """
    const auto parent = [&](int r) { return parents.data[r]; };
    const auto value = [&](int r) { return values.data[r]; };
    loads.data[row] = RING_LOAD(values.data[row], row, hops, parent, value);
    positive.data[row] = RING_ANY(row, hops, parent, [&](int r) { return values.data[r] > 0.0f; });
"""

    @wp.func_native(dual_native(body, preludes=(RING,)))
    def ring(
        row: int,
        hops: int,
        parents: wp.array[int],
        values: wp.array[float],
        loads: wp.array[float],
        positive: wp.array[int],
    ): ...

    @wp.kernel(enable_backward=False, module="unique")
    def ring_walks(
        hops: int, parents: wp.array[int], values: wp.array[float], loads: wp.array[float], positive: wp.array[int]
    ):
        row = wp.tid()
        ring(row, hops, parents, values, loads, positive)

    return ring_walks


class TestPatchRings(unittest.TestCase):
    def test_ring_walks_pool_in_ring_order_and_stop_at_the_hop_guard(self):
        """Pool a ring from each member in ring order, keep point contacts alone and bound a broken ring."""
        for device in ("cpu", *([wp.get_cuda_device()] if wp.is_cuda_available() else [])):
            with self.subTest(device=str(device)):
                # Rows 0 -> 2 -> 4 -> 0 form a ring, row 1 is a point contact, rows 3 and 5 a broken chain 3 -> 5 -> 5.
                parents = wp.array([2, -1, 4, 5, 0, 5], dtype=wp.int32, device=device)
                values = wp.array([1.0, 2.0, 4.0, 8.0, 16.0, 0.0], dtype=wp.float32, device=device)
                loads = wp.zeros(6, dtype=wp.float32, device=device)
                positive = wp.zeros(6, dtype=wp.int32, device=device)
                wp.launch(_ring_kernel(), dim=6, inputs=[2, parents, values], outputs=[loads, positive], device=device)
                # The broken chain stops after its two hops: 8 + 0 + 0.
                np.testing.assert_array_equal(loads.numpy(), [21.0, 2.0, 21.0, 8.0, 21.0, 0.0])
                np.testing.assert_array_equal(positive.numpy(), [1, 0, 1, 0, 1, 0])


_GROUP_SOURCE = """
    const int item = tid;
    const float x = floats.data[item];
    const int k = ints.data[item];
    // The ladder the sweeps read on the group's lane 0.
    float ladder = x;
    for (int offset = WIDTH >> 1; offset > 0; offset >>= 1)
        ladder += __shfl_down_sync(MASK, ladder, offset, WIDTH);
    float sum = x, low = x, high = x, segment = x;
    int index = item;
    float arg = x;
    SUM(sum);
    MINIMUM(low);
    MAXIMUM(high);
    ARGMAX(arg, index);
    SEGMENT_SUM(SEGMENT_WIDTH, segment);
    const unsigned votes = BALLOT(k > 0);
    out_float.data[item * 6 + 0] = READ(ladder, 0);
    out_float.data[item * 6 + 1] = sum;
    out_float.data[item * 6 + 2] = low;
    out_float.data[item * 6 + 3] = high;
    out_float.data[item * 6 + 4] = arg;
    out_float.data[item * 6 + 5] = segment;
    out_int.data[item * 6 + 0] = SCAN(k);
    out_int.data[item * 6 + 1] = (int)votes;
    out_int.data[item * 6 + 2] = BELOW(votes);
    out_int.data[item * 6 + 3] = COUNT(votes);
    out_int.data[item * 6 + 4] = FIRST(votes);
    out_int.data[item * 6 + 5] = index;
    const int first = RESERVE(&counter.data[0], COUNT(votes));
    if (k > 0) slots.data[item] = first + BELOW(votes);
#endif
"""


@cache
def _group_kernel(lanes, segment):
    snippet = cuda_native(_GROUP_SOURCE.replace("SEGMENT_WIDTH", str(segment)), preludes=(lane_group(lanes),))

    @wp.func_native(snippet)
    def group(
        tid: int,
        floats: wp.array[float],
        ints: wp.array[int],
        out_float: wp.array[float],
        out_int: wp.array[int],
        counter: wp.array[int],
        slots: wp.array[int],
    ): ...

    def kernel(
        floats: wp.array[float],
        ints: wp.array[int],
        out_float: wp.array[float],
        out_int: wp.array[int],
        counter: wp.array[int],
        slots: wp.array[int],
    ):
        group(wp.tid(), floats, ints, out_float, out_int, counter, slots)

    kernel.__name__ = kernel.__qualname__ = f"lane_group_probe_{lanes}_{segment}"
    return wp.kernel(enable_backward=False, module="unique")(kernel)


@unittest.skipUnless(wp.is_cuda_available(), "lane groups exchange values on CUDA warps")
class TestLaneGroupsOnCuda(unittest.TestCase):
    def test_lane_group_operations(self):
        """Check every group operation of warp groups of 4 to 32 lanes against host references.

        ``SUM`` must equal the shuffle-down ladder's lane 0 bit for bit on every lane of the group.
        """
        device = wp.get_cuda_device()
        rng = np.random.default_rng(7)
        count = 256
        for lanes in (4, 8, 16, 32):
            segment = max(lanes // 2, 1)
            with self.subTest(lanes=lanes):
                floats = (rng.normal(size=count) * 10.0 ** rng.integers(-3, 4, count)).astype(np.float32)
                floats[::7] = floats[1::7][: len(floats[::7])]  # ties for ARGMAX
                ints = rng.integers(-2, 3, count).astype(np.int32)
                out_float = wp.zeros(count * 6, dtype=wp.float32, device=device)
                out_int = wp.zeros(count * 6, dtype=wp.int32, device=device)
                counter = wp.zeros(1, dtype=wp.int32, device=device)
                slots = wp.full(count, -1, dtype=wp.int32, device=device)
                wp.launch(
                    _group_kernel(lanes, segment),
                    dim=count,
                    inputs=[wp.array(floats, device=device), wp.array(ints, device=device)],
                    outputs=[out_float, out_int, counter, slots],
                    block_dim=64,
                    device=device,
                )
                result = out_float.numpy().reshape(count, 6)
                integers = out_int.numpy().reshape(count, 6)
                groups = floats.reshape(-1, lanes)
                # SUM is the ladder's lane 0 value on every lane, bit for bit.
                np.testing.assert_array_equal(result[:, 1].view(np.int32), result[:, 0].view(np.int32))
                np.testing.assert_allclose(
                    result[:, 1].reshape(-1, lanes)[:, 0], groups.astype(np.float64).sum(axis=1), rtol=1e-5, atol=1e-3
                )
                np.testing.assert_array_equal(result[:, 2], np.repeat(groups.min(axis=1), lanes))
                np.testing.assert_array_equal(result[:, 3], np.repeat(groups.max(axis=1), lanes))
                np.testing.assert_array_equal(result[:, 4], np.repeat(groups.max(axis=1), lanes))
                # ARGMAX breaks a tie toward the larger index.
                expected_index = [
                    lanes * g + max(i for i in range(lanes) if group[i] == group.max())
                    for g, group in enumerate(groups)
                ]
                np.testing.assert_array_equal(integers[:, 5], np.repeat(expected_index, lanes))
                segments = floats.reshape(-1, segment).astype(np.float64).sum(axis=1)
                np.testing.assert_allclose(result[:, 5], np.repeat(segments, segment), rtol=1e-5, atol=1e-3)
                votes = (ints > 0).reshape(-1, lanes)
                bits = (votes * (1 << np.arange(lanes))).sum(axis=1)
                np.testing.assert_array_equal(integers[:, 1], np.repeat(bits, lanes).astype(np.int32))
                below = np.concatenate([np.concatenate([[0], np.cumsum(v)[:-1]]) for v in votes])
                np.testing.assert_array_equal(integers[:, 2], below)
                np.testing.assert_array_equal(integers[:, 3], np.repeat(votes.sum(axis=1), lanes))
                first = [int(np.argmax(v)) if v.any() else -1 for v in votes]
                np.testing.assert_array_equal(integers[:, 4], np.repeat(first, lanes))
                scan = np.concatenate([np.concatenate([[0], np.cumsum(g)[:-1]]) for g in ints.reshape(-1, lanes)])
                np.testing.assert_array_equal(integers[:, 0], scan)
                # RESERVE gives each group one block of slots, taken by its voting lanes in lane order.
                taken = slots.numpy()
                self.assertEqual(int(counter.numpy()[0]), int(votes.sum()))
                self.assertEqual(sorted(taken[taken >= 0].tolist()), list(range(int(votes.sum()))))
                for g, v in enumerate(votes):
                    group_slots = taken[g * lanes : (g + 1) * lanes][v]
                    if len(group_slots):
                        np.testing.assert_array_equal(group_slots, group_slots[0] + np.arange(len(group_slots)))


if __name__ == "__main__":
    unittest.main()
