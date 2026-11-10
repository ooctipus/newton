# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The device idioms of FeatherPGS: lane groups and their further words, patch rings and the slot reservation contract.

The lane group's reductions replace shuffle-down ladders whose lane 0 the
sweeps read; a butterfly from the widest offset down gives every lane that
value bit for bit, which the CUDA tests below check against the ladder itself.
"""

import unittest
from functools import cache

import numpy as np
import warp as wp

from newton._src.core import memory as toolkit
from newton._src.core.memory import Operand, Place
from newton._src.solvers.feather_pgs.device_preludes import (
    HELD,
    LANE_WORDS,
    RING,
    SHARED_STORAGE,
    DevicePrelude,
    cuda_native,
    dual_native,
    lane_all,
    lane_any,
    lane_group,
    lane_index,
    lane_maximum,
    lane_minimum,
    lane_read,
    lane_scan,
    lane_stride,
    lane_sum,
    lane_sync,
    memory_words,
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


# The further words on one thread a probe: lane words, held arrays and the memory toolkit's words (every global place
# and an operand's own load), on a warp or block group.
_WORDS_SOURCE = """    SHARED float scratch_value[32];
    SHARED int scratch_index[32];
    TOUCH(values.data + tid);
    FETCH(values.data, 64, LANE, LANES);
    const float x = LDG(values.data + tid);
    const int key = LOAD_keys(keys.data + tid);
    float total = x, top = x, best = x;
    int at = tid;
    BLOCK_SUM(scratch_value, total);
    BLOCK_MAXIMUM(scratch_value, top);
    BLOCK_ARGMAX(scratch_value, scratch_index, best, at);
    // Seven items held over the group's lanes: item i is 2 i, its row (10 i, 10 i + 1).
    constexpr int words = (7 + WIDTH - 1) / WIDTH;
    float held[words];
    float rows[words][2];
    HELD_LOAD(held, 7, [&](int k, int item, bool valid) {
        held[k] = valid ? 2.0f * item : 0.0f;
        rows[k][0] = valid ? 10.0f * item : 0.0f;
        rows[k][1] = valid ? 10.0f * item + 1.0f : 0.0f;
    });
    float row[2];
    HELD_ROW(rows, 5, row);
    float* const out = floats.data + 8 * tid;
    ST(out + 0, total);
    ST(out + 1, top);
    ST(out + 2, best);
    ST(out + 3, HELD_PICK(held, 6));
    ST(out + 4, row[0]);
    ST(out + 5, row[1]);
    ST(out + 6, READ_DOWN(x, 1));
    STCG(out + 7, LDCG(values.data + tid));
    int* const words_out = ints.data + 4 * tid;
    words_out[0] = at;
    words_out[1] = WARP_LANE;
    words_out[2] = (int)MATCH(key);
    words_out[3] = ADD(counter.data[0], 1);
"""


@cache
def _words_kernel(lanes):
    named = DevicePrelude("memory", *toolkit.words("LDG", "ST", "ADD", "TOUCH", "FETCH", "LDCG", "STCG"))
    preludes = (lane_group(lanes), LANE_WORDS, HELD, named, memory_words(Operand("keys")), SHARED_STORAGE)

    @wp.func_native(dual_native(_WORDS_SOURCE, preludes=preludes))
    def words(
        tid: int,
        values: wp.array[float],
        keys: wp.array[int],
        floats: wp.array[float],
        ints: wp.array[int],
        counter: wp.array[int],
    ): ...

    def kernel(
        values: wp.array[float],
        keys: wp.array[int],
        floats: wp.array[float],
        ints: wp.array[int],
        counter: wp.array[int],
    ):
        words(wp.tid(), values, keys, floats, ints, counter)

    kernel.__name__ = kernel.__qualname__ = f"lane_words_probe_{lanes}"
    return wp.kernel(enable_backward=False, module="unique")(kernel)


def _run_words(lanes, device, count=256):
    rng = np.random.default_rng(11)
    values = rng.normal(size=count).astype(np.float32)
    keys = (np.arange(count) % 3).astype(np.int32)
    floats = wp.zeros(count * 8, dtype=wp.float32, device=device)
    ints = wp.zeros(count * 4, dtype=wp.int32, device=device)
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _words_kernel(lanes),
        dim=count,
        inputs=[wp.array(values, device=device), wp.array(keys, device=device)],
        outputs=[floats, ints, counter],
        block_dim=max(lanes, 64),
        device=device,
    )
    return values, keys, floats.numpy().reshape(count, 8), ints.numpy().reshape(count, 4), int(counter.numpy()[0])


class TestLaneWords(unittest.TestCase):
    def test_words_on_the_cpu_are_the_one_thread_s(self):
        """On the CPU one thread is the group: reductions and reads are its own value, held arrays all its own."""
        values, _, floats, ints, total = _run_words(8, "cpu")
        np.testing.assert_array_equal(floats[:, 0], values)
        np.testing.assert_array_equal(floats[:, 1], values)
        np.testing.assert_array_equal(floats[:, 2], values)
        np.testing.assert_array_equal(floats[:, 3:6], np.tile([12.0, 50.0, 51.0], (len(values), 1)))
        np.testing.assert_array_equal(floats[:, 6], values)
        np.testing.assert_array_equal(floats[:, 7], values)
        np.testing.assert_array_equal(ints[:, 0], np.arange(len(values)))
        np.testing.assert_array_equal(ints[:, 1:3], np.tile([0, 1], (len(values), 1)))
        self.assertEqual(sorted(ints[:, 3]), list(range(len(values))))
        self.assertEqual(total, len(values))

    @unittest.skipUnless(wp.is_cuda_available(), "lane words exchange values on CUDA warps")
    def test_words_on_cuda(self):
        """Warp groups of 8 and 32 lanes and block groups of 64 and 128 against host references."""
        for lanes in (8, 32, 64, 128):
            with self.subTest(lanes=lanes):
                values, keys, floats, ints, total = _run_words(lanes, wp.get_cuda_device())
                groups = values.reshape(-1, lanes)
                width = min(lanes, 32)
                np.testing.assert_allclose(floats[:, 0], np.repeat(groups.sum(axis=1), lanes), rtol=1e-5, atol=1e-5)
                np.testing.assert_array_equal(floats[:, 1], np.repeat(groups.max(axis=1), lanes))
                np.testing.assert_array_equal(floats[:, 2], np.repeat(groups.max(axis=1), lanes))
                base = (np.arange(len(values)) // lanes) * lanes
                np.testing.assert_array_equal(ints[:, 0], base + np.repeat(groups.argmax(axis=1), lanes))
                np.testing.assert_array_equal(floats[:, 3:6], np.tile([12.0, 50.0, 51.0], (len(values), 1)))
                lane = np.arange(len(values)) % width
                down = np.where(lane + 1 < width, np.roll(values, -1), values)
                np.testing.assert_array_equal(floats[:, 6], down)
                np.testing.assert_array_equal(floats[:, 7], values)
                np.testing.assert_array_equal(ints[:, 1], np.arange(len(values)) % 32)
                segment = (np.arange(len(values)) // width) * width
                expected = [
                    sum(1 << j for j in range(width) if keys[segment[i] + j] == keys[i]) for i in range(len(values))
                ]
                np.testing.assert_array_equal(ints[:, 2].astype(np.uint32), np.array(expected, dtype=np.uint32))
                self.assertEqual(sorted(ints[:, 3]), list(range(len(values))))
                self.assertEqual(total, len(values))


class TestMemoryToolkit(unittest.TestCase):
    def test_read_rule_places_operands(self):
        """LDG where no thread writes an operand, LD and ST where one does, a measured place over the rule's."""
        lambdas = {name: toolkit.words(name)[0].split(" = ", 1)[1] for name in ("LDG", "LD", "ST", "LDCG", "STCG")}
        rows, sums, keys = Operand("rows"), Operand("sums", writes=True), Operand("keys", measured=Place.GLOBAL)
        cuda, cpu = toolkit.read(rows, sums, keys)
        for name, word in (("LOAD_rows", "LDG"), ("LOAD_sums", "LD"), ("STORE_sums", "ST"), ("LOAD_keys", "LD")):
            self.assertEqual(cuda.count(f"const auto {name} = {lambdas[word]}"), 1, name)
            self.assertEqual(cpu.count(f"const auto {name} ="), 1, name)
        self.assertNotIn("STORE_rows", cuda)
        self.assertNotIn("STORE_keys", cpu)
        cache = toolkit.read(Operand("cache", writes=True, measured=Place.PERSISTING))[0]
        self.assertIn(f"const auto LOAD_cache = {lambdas['LDCG']}", cache)
        self.assertIn(f"const auto STORE_cache = {lambdas['STCG']}", cache)
        for place in (Place.SHARED, Place.READ_ONLY):
            with self.assertRaisesRegex(ValueError, "global place"):
                Operand("stage", writes=True, measured=place)

    def test_words_are_declared_once_in_one_order(self):
        """The named words alone, each once, in the toolkit's order, on both devices."""
        cuda, cpu = toolkit.words("STCG", "LDCG", "FETCH", "TOUCH", "ADD", "ST", "LD", "LDG")
        names = ("LDG", "LD", "ST", "ADD", "TOUCH", "FETCH", "LDCG", "STCG")
        for text in (cuda, cpu):
            starts = [text.index(f"const auto {name} =") for name in names]
            self.assertEqual(starts, sorted(starts))
            self.assertEqual(sum(line.startswith("    const auto ") for line in text.splitlines()), len(names))

    def test_constant_table_is_one_text(self):
        """A constant table: one declaration in CONSTANT storage, CUDA's constant memory or the CPU's constant data."""
        self.assertEqual(toolkit.table("PARENT", (-1, 0, True)), "static CONSTANT int PARENT[3] = {-1, 0, 1};")
        self.assertEqual(toolkit.table("NONE", ()), "static CONSTANT int NONE[1] = {};")
        self.assertEqual(toolkit.CONSTANT_STORAGE, ("    #define CONSTANT __constant__", "    #define CONSTANT const"))


@cache
def _lane_functions_kernel(lanes):
    """A Python kernel over the lane group's Warp functions: what a megakernel composes."""

    def kernel(values: wp.array[float], floats: wp.array2d[float], ints: wp.array2d[int]):
        tid = wp.tid()
        x = values[tid]
        lane_sync(lanes)
        floats[tid, 0] = lane_sum(x, lanes)
        floats[tid, 1] = lane_minimum(x, lanes)
        floats[tid, 2] = lane_maximum(x, lanes)
        floats[tid, 3] = lane_read(x, 1, lanes)
        ints[tid, 0] = lane_sum(tid, lanes)
        ints[tid, 1] = lane_scan(1, lanes)
        ints[tid, 2] = lane_read(tid, 0, lanes)
        ints[tid, 3] = lane_index(lanes) + 100 * lane_stride(lanes)
        ints[tid, 4] = wp.where(lane_any(x > 0.0, lanes), 1, 0) + 2 * wp.where(lane_all(x > -10.0, lanes), 1, 0)

    kernel.__name__ = kernel.__qualname__ = f"lane_functions_probe_{lanes}"
    return wp.kernel(enable_backward=False, module="unique")(kernel)


def _run_lane_functions(lanes, device, count=64):
    values = np.random.default_rng(5).normal(size=count).astype(np.float32)
    floats = wp.zeros((count, 4), dtype=float, device=device)
    ints = wp.zeros((count, 5), dtype=int, device=device)
    wp.launch(
        _lane_functions_kernel(lanes),
        dim=count,
        inputs=[wp.array(values, device=device)],
        outputs=[floats, ints],
        block_dim=64,
        device=device,
    )
    return values, floats.numpy(), ints.numpy()


class TestLaneFunctions(unittest.TestCase):
    def test_on_the_cpu_one_thread_is_the_group(self):
        values, floats, ints = _run_lane_functions(8, "cpu")
        for column in range(4):
            np.testing.assert_array_equal(floats[:, column], values)
        tid = np.arange(len(values))
        np.testing.assert_array_equal(ints[:, 0], tid)
        np.testing.assert_array_equal(ints[:, 1], 0)
        np.testing.assert_array_equal(ints[:, 2], tid)
        np.testing.assert_array_equal(ints[:, 3], 100)
        np.testing.assert_array_equal(ints[:, 4], (values > 0) + 2)

    @unittest.skipUnless(wp.is_cuda_available(), "lane functions exchange values on CUDA warps")
    def test_on_cuda_against_host_references(self):
        for lanes in (8, 32):
            with self.subTest(lanes=lanes):
                values, floats, ints = _run_lane_functions(lanes, wp.get_cuda_device())
                groups = values.reshape(-1, lanes)
                tid = np.arange(len(values))
                np.testing.assert_allclose(floats[:, 0], np.repeat(groups.sum(axis=1), lanes), rtol=1e-5, atol=1e-5)
                np.testing.assert_array_equal(floats[:, 1], np.repeat(groups.min(axis=1), lanes))
                np.testing.assert_array_equal(floats[:, 2], np.repeat(groups.max(axis=1), lanes))
                np.testing.assert_array_equal(floats[:, 3], np.repeat(groups[:, 1], lanes))
                np.testing.assert_array_equal(ints[:, 0], np.repeat(tid.reshape(-1, lanes).sum(axis=1), lanes))
                np.testing.assert_array_equal(ints[:, 1], tid % lanes)
                np.testing.assert_array_equal(ints[:, 2], tid - tid % lanes)
                np.testing.assert_array_equal(ints[:, 3], tid % lanes + 100 * lanes)
                any_ = np.repeat((groups > 0).any(axis=1), lanes)
                np.testing.assert_array_equal(ints[:, 4], any_ + 2)


if __name__ == "__main__":
    unittest.main()
