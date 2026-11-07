# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for ordered reservation: outputs in input order from a block scan and a decoupled look-back."""

import unittest

import numpy as np
import warp as wp

from newton._src.core.ordered_reserve import (
    OrderedTiles,
    ordered_also,
    ordered_next,
    ordered_scan,
    ordered_tile,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

BLOCK = 64
ITEMS = 2
TILE = BLOCK * ITEMS


@wp.func
def _item_count(i: int, salt: int) -> int:
    """0 to 3 outputs per input, varying with the input and the launch."""
    h = ((i + salt) * 1103515245 + 12345) >> 7
    return h & 3


@wp.func
def _append(first: int, begin: int, per_lane: int, n: int, salt: int, out: wp.array[int]):
    for k in range(per_lane):
        i = begin + k
        if i < n:
            c = _item_count(i, salt)
            for j in range(c):
                out[first + j] = i * 4 + j
            first += c


@wp.kernel(enable_backward=False)
def _append_fixed(
    n: wp.array[int],
    salt: int,
    site: wp.array[wp.uint64],
    tiles: int,
    total: wp.array[int],
    out: wp.array[int],
):
    block, lane = wp.tid()
    claim = ordered_tile(site, block, tiles, total)
    per_lane = TILE // wp.block_dim()
    begin = claim[0] * TILE + lane * per_lane
    count = int(0)
    for k in range(per_lane):
        if begin + k < n[0]:
            count += _item_count(begin + k, salt)
    last = int(0)
    if claim[0] == tiles - 1:
        last = 1
    first = ordered_scan(site, claim, count, last, total)
    _append(first, begin, per_lane, n[0], salt, out)


@wp.kernel(enable_backward=False)
def _append_dynamic(
    n: wp.array[int],
    salt: int,
    site: wp.array[wp.uint64],
    blocks: int,
    total: wp.array[int],
    out: wp.array[int],
):
    block, lane = wp.tid()
    per_lane = TILE // wp.block_dim()
    live_tiles = (n[0] + TILE - 1) // TILE
    claim = ordered_next(site, block, live_tiles, blocks, total)
    while claim[0] >= 0:
        begin = claim[0] * TILE + lane * per_lane
        count = int(0)
        for k in range(per_lane):
            if begin + k < n[0]:
                count += _item_count(begin + k, salt)
        last = int(0)
        if claim[0] == live_tiles - 1:
            last = 1
        first = ordered_scan(site, claim, count, last, total)
        _append(first, begin, per_lane, n[0], salt, out)
        claim = ordered_next(site, block, live_tiles, blocks, total)


def _expected(n: int, salt: int) -> np.ndarray:
    i = np.arange(n, dtype=np.int64)
    h = (((i + salt) * 1103515245 + 12345) & 0xFFFFFFFF).astype(np.int64)
    h = np.where(h >= 2**31, h - 2**32, h)
    counts = (h >> 7) & 3
    return np.repeat(4 * i, counts) + np.concatenate([np.arange(c) for c in counts]) if n else np.zeros(0, np.int64)


def _launch(kernel, grid, device, live, salt, site, size, total, out):
    """Launch an appending kernel over ``live`` inputs with ``grid`` blocks (``size``: tiles or blocks)."""
    wp.launch_tiled(
        kernel,
        dim=[grid],
        inputs=[live, salt, site.site, size, total, out],
        block_dim=BLOCK,
        device=device,
    )


def _check(test, n, salt, total, out, *launch):
    total.zero_()
    _launch(*launch)
    want = _expected(n, salt)
    got_n = int(total.numpy()[0])
    test.assertEqual(got_n, want.shape[0])
    np.testing.assert_array_equal(out.numpy()[:got_n], want)


def test_fixed_tiles_write_in_input_order(test, device):
    """One tile per block: every launch writes its outputs in input order, launch after launch."""
    for n in (0, 1, TILE - 1, TILE, 3 * TILE + 5, 40 * TILE + 17):
        tiles = max(1, -(-n // TILE))
        site = OrderedTiles(tiles, device)
        live = wp.array([n], dtype=int, device=device)
        total = wp.zeros(1, dtype=int, device=device)
        out = wp.zeros(max(3 * n, 1), dtype=int, device=device)
        for salt in range(4):  # epochs advance over the same states
            with test.subTest(n=n, salt=salt):
                _check(test, n, salt, total, out, _append_fixed, tiles, device, live, salt, site, tiles, total, out)


def test_dynamic_tiles_follow_a_device_count(test, device):
    """A fixed grid strides over a device-side count of tiles; counts change between launches."""
    capacity = 60 * TILE
    site = OrderedTiles(-(-capacity // TILE), device)
    blocks = 7
    total = wp.zeros(1, dtype=int, device=device)
    out = wp.zeros(3 * capacity, dtype=int, device=device)
    live = wp.zeros(1, dtype=int, device=device)
    for salt, n in enumerate((5 * TILE + 3, 0, 1, capacity, 2 * TILE, 37 * TILE + 101)):
        with test.subTest(n=n):
            live.fill_(n)
            _check(test, n, salt, total, out, _append_dynamic, blocks, device, live, salt, site, blocks, total, out)


def test_captured_replays_repeat_bitwise(test, device):
    """A captured launch replays with the count of each replay and writes the same order every time."""
    capacity = 30 * TILE
    site = OrderedTiles(-(-capacity // TILE), device)
    blocks = 5
    total = wp.zeros(1, dtype=int, device=device)
    out = wp.zeros(3 * capacity, dtype=int, device=device)
    live = wp.zeros(1, dtype=int, device=device)
    live.fill_(capacity)
    with wp.ScopedCapture(device=device) as capture:
        _launch(_append_dynamic, blocks, device, live, 3, site, blocks, total, out)
    first = None
    for n in (capacity, 11 * TILE + 7, capacity, 0, capacity):
        live.fill_(n)
        total.zero_()
        wp.capture_launch(capture.graph)
        got = out.numpy()[: int(total.numpy()[0])]
        np.testing.assert_array_equal(got, _expected(n, 3))
        if n == capacity:
            if first is None:
                first = got.copy()
            np.testing.assert_array_equal(got, first)


def test_fixed_and_dynamic_launches_share_a_site(test, device):
    """Fixed and dynamic launches of one, several and no tiles alternate on one site, each in input order."""
    capacity = 24 * TILE
    site = OrderedTiles(-(-capacity // TILE), device)
    blocks = 3
    total = wp.zeros(1, dtype=int, device=device)
    out = wp.zeros(3 * capacity, dtype=int, device=device)
    live = wp.zeros(1, dtype=int, device=device)
    epochs = []
    for salt, (n, dynamic) in enumerate(
        ((9 * TILE + 1, False), (TILE, True), (0, True), (5, False), (capacity, True), (2 * TILE + 3, False))
    ):
        with test.subTest(n=n, dynamic=dynamic):
            live.fill_(n)
            if dynamic:
                launch = (_append_dynamic, blocks, device, live, salt, site, blocks, total, out)
            else:
                tiles = max(1, -(-n // TILE))
                launch = (_append_fixed, tiles, device, live, salt, site, tiles, total, out)
            _check(test, n, salt, total, out, *launch)
            words = site.site.numpy()
            test.assertEqual(int(words[0]), 0)  # the dynamic counter is back at 0 after every launch
            epochs.append(int(words[1]))
    # Launches of several tiles open a new epoch; launches of one tile or none keep it.
    test.assertEqual(epochs, [2, 2, 2, 2, 3, 4])


def test_writers_in_sequence_fill_one_list(test, device):
    """Launches append after the slots already counted: writers in sequence fill one list in segments."""
    site_a = OrderedTiles(8, device)
    site_b = OrderedTiles(8, device)
    total = wp.zeros(1, dtype=int, device=device)
    out = wp.zeros(3 * 16 * TILE + 17, dtype=int, device=device)
    for n_a, n_b in ((5 * TILE + 2, 3), (0, 7 * TILE), (TILE, TILE + 1)):
        with test.subTest(n_a=n_a, n_b=n_b):
            total.fill_(17)  # slots an earlier writer took
            live_a = wp.array([n_a], dtype=int, device=device)
            live_b = wp.array([n_b], dtype=int, device=device)
            tiles = max(1, -(-n_a // TILE))
            _launch(_append_fixed, tiles, device, live_a, 1, site_a, tiles, total, out)
            _launch(_append_dynamic, 3, device, live_b, 2, site_b, 3, total, out)
            want = np.concatenate([_expected(n_a, 1), _expected(n_b, 2)])
            got_n = int(total.numpy()[0])
            test.assertEqual(got_n, 17 + want.shape[0])
            np.testing.assert_array_equal(out.numpy()[17:got_n], want)


@wp.kernel(enable_backward=False)
def _append_two_lists(
    n: wp.array[int],
    site: wp.array[wp.uint64],
    other: wp.array[wp.uint64],
    blocks: int,
    total: wp.array[int],
    out: wp.array[int],
    other_total: wp.array[int],
    other_out: wp.array[int],
):
    """Inputs append to two lists per tile: count(i) items to ``out``, one item to ``other`` when count(i) > 1."""
    block, lane = wp.tid()
    per_lane = TILE // wp.block_dim()
    live_tiles = (n[0] + TILE - 1) // TILE
    claim = ordered_next(site, block, live_tiles, blocks, total)
    while claim[0] >= 0:
        begin = claim[0] * TILE + lane * per_lane
        count = int(0)
        picks = int(0)
        for k in range(per_lane):
            if begin + k < n[0]:
                c = _item_count(begin + k, 5)
                count += c
                if c > 1:
                    picks += 1
        last = int(0)
        if claim[0] == live_tiles - 1:
            last = 1
        first = ordered_scan(site, claim, count, last, total)
        pick = ordered_scan(other, ordered_also(claim), picks, last, other_total)
        _append(first, begin, per_lane, n[0], 5, out)
        for k in range(per_lane):
            i = begin + k
            if i < n[0]:
                if _item_count(i, 5) > 1:
                    other_out[pick] = i
                    pick += 1
        claim = ordered_next(site, block, live_tiles, blocks, total)


def test_one_claim_appends_to_two_lists(test, device):
    """A tile claimed from one site appends to a second site's list too, each list in input order."""
    capacity = 20 * TILE
    site = OrderedTiles(-(-capacity // TILE), device)
    other = OrderedTiles(-(-capacity // TILE), device)
    total = wp.zeros(1, dtype=int, device=device)
    other_total = wp.zeros(1, dtype=int, device=device)
    out = wp.zeros(3 * capacity, dtype=int, device=device)
    other_out = wp.zeros(capacity, dtype=int, device=device)
    live = wp.zeros(1, dtype=int, device=device)
    for n in (capacity, 3, 0, 7 * TILE + 5, TILE):
        with test.subTest(n=n):
            live.fill_(n)
            total.zero_()
            other_total.fill_(4)  # the second list already holds 4 items
            wp.launch_tiled(
                _append_two_lists,
                dim=[4],
                inputs=[live, site.site, other.site, 4, total, out, other_total, other_out],
                block_dim=BLOCK,
                device=device,
            )
            want = _expected(n, 5)
            test.assertEqual(int(total.numpy()[0]), want.shape[0])
            np.testing.assert_array_equal(out.numpy()[: want.shape[0]], want)
            i = np.arange(n)
            counts = np.bincount(want // 4, minlength=n)[:n] if n else np.zeros(0, int)
            picked = i[counts > 1]
            test.assertEqual(int(other_total.numpy()[0]), 4 + picked.shape[0])
            np.testing.assert_array_equal(other_out.numpy()[4 : 4 + picked.shape[0]], picked)


def test_epochs_wrap_past_zero(test, device):
    """The epoch after 2**31 - 1 is 1: zeroed states carry epoch 0, which no launch uses."""
    n = 6 * TILE + 9
    site = OrderedTiles(-(-n // TILE), device)
    words = site.site.numpy()
    words[1] = 2**31 - 1
    site.site.assign(words)
    tiles = -(-n // TILE)
    live = wp.array([n], dtype=int, device=device)
    total = wp.zeros(1, dtype=int, device=device)
    out = wp.zeros(3 * n, dtype=int, device=device)
    for salt in range(3):
        with test.subTest(salt=salt):
            _check(test, n, salt, total, out, _append_fixed, tiles, device, live, salt, site, tiles, total, out)
            test.assertEqual(int(site.site.numpy()[1]), salt + 1)


class TestOrderedReserve(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(
    TestOrderedReserve, "test_fixed_tiles_write_in_input_order", test_fixed_tiles_write_in_input_order, devices=devices
)
add_function_test(
    TestOrderedReserve,
    "test_dynamic_tiles_follow_a_device_count",
    test_dynamic_tiles_follow_a_device_count,
    devices=devices,
)
add_function_test(
    TestOrderedReserve,
    "test_fixed_and_dynamic_launches_share_a_site",
    test_fixed_and_dynamic_launches_share_a_site,
    devices=devices,
)
add_function_test(
    TestOrderedReserve,
    "test_writers_in_sequence_fill_one_list",
    test_writers_in_sequence_fill_one_list,
    devices=devices,
)
add_function_test(
    TestOrderedReserve,
    "test_one_claim_appends_to_two_lists",
    test_one_claim_appends_to_two_lists,
    devices=devices,
)
add_function_test(TestOrderedReserve, "test_epochs_wrap_past_zero", test_epochs_wrap_past_zero, devices=devices)
add_function_test(
    TestOrderedReserve,
    "test_captured_replays_repeat_bitwise",
    test_captured_replays_repeat_bitwise,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
