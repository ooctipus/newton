# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prove frozen plan arrays unchanged on the device with one small readback.

Cache owners bake topology (parents, axes, anchors, body membership, group
maps) into their kernels and must refuse to run after it changes. Reading
every plan array back to the host to fingerprint it on each model-change
notification costs several milliseconds per notification at 16k worlds, and
Isaac Lab notifies on every reset. This helper keeps an exact device copy of
each array's words, compares them with one kernel launch per array and reads
back a single mismatch mask. Arrays that cannot be viewed as device words
(host arrays, non-contiguous views, ``None``) keep a host fingerprint.
"""

import hashlib

import warp as wp

_SLOTS = 16


@wp.struct
class ProofWords:
    """Up to sixteen word views compared in one launch (slot i sets bit i on mismatch)."""

    count: int
    sizes: wp.array[int]
    c0: wp.array[wp.uint32]
    c1: wp.array[wp.uint32]
    c2: wp.array[wp.uint32]
    c3: wp.array[wp.uint32]
    c4: wp.array[wp.uint32]
    c5: wp.array[wp.uint32]
    c6: wp.array[wp.uint32]
    c7: wp.array[wp.uint32]
    c8: wp.array[wp.uint32]
    c9: wp.array[wp.uint32]
    c10: wp.array[wp.uint32]
    c11: wp.array[wp.uint32]
    c12: wp.array[wp.uint32]
    c13: wp.array[wp.uint32]
    c14: wp.array[wp.uint32]
    c15: wp.array[wp.uint32]
    f0: wp.array[wp.uint32]
    f1: wp.array[wp.uint32]
    f2: wp.array[wp.uint32]
    f3: wp.array[wp.uint32]
    f4: wp.array[wp.uint32]
    f5: wp.array[wp.uint32]
    f6: wp.array[wp.uint32]
    f7: wp.array[wp.uint32]
    f8: wp.array[wp.uint32]
    f9: wp.array[wp.uint32]
    f10: wp.array[wp.uint32]
    f11: wp.array[wp.uint32]
    f12: wp.array[wp.uint32]
    f13: wp.array[wp.uint32]
    f14: wp.array[wp.uint32]
    f15: wp.array[wp.uint32]


@wp.func
def _slot_mismatch(current: wp.array[wp.uint32], frozen: wp.array[wp.uint32], size: int, index: int) -> bool:
    return index < size and current[index] != frozen[index]


@wp.kernel(module="unique", enable_backward=False)
def _compare_words(words: ProofWords, mismatch: wp.array[int]):
    """One thread per word of the largest array; every slot is checked at that index."""
    index = wp.tid()
    bits = int(0)
    if _slot_mismatch(words.c0, words.f0, words.sizes[0], index):
        bits |= 1 << 0
    if words.count > 1 and _slot_mismatch(words.c1, words.f1, words.sizes[1], index):
        bits |= 1 << 1
    if words.count > 2 and _slot_mismatch(words.c2, words.f2, words.sizes[2], index):
        bits |= 1 << 2
    if words.count > 3 and _slot_mismatch(words.c3, words.f3, words.sizes[3], index):
        bits |= 1 << 3
    if words.count > 4 and _slot_mismatch(words.c4, words.f4, words.sizes[4], index):
        bits |= 1 << 4
    if words.count > 5 and _slot_mismatch(words.c5, words.f5, words.sizes[5], index):
        bits |= 1 << 5
    if words.count > 6 and _slot_mismatch(words.c6, words.f6, words.sizes[6], index):
        bits |= 1 << 6
    if words.count > 7 and _slot_mismatch(words.c7, words.f7, words.sizes[7], index):
        bits |= 1 << 7
    if words.count > 8 and _slot_mismatch(words.c8, words.f8, words.sizes[8], index):
        bits |= 1 << 8
    if words.count > 9 and _slot_mismatch(words.c9, words.f9, words.sizes[9], index):
        bits |= 1 << 9
    if words.count > 10 and _slot_mismatch(words.c10, words.f10, words.sizes[10], index):
        bits |= 1 << 10
    if words.count > 11 and _slot_mismatch(words.c11, words.f11, words.sizes[11], index):
        bits |= 1 << 11
    if words.count > 12 and _slot_mismatch(words.c12, words.f12, words.sizes[12], index):
        bits |= 1 << 12
    if words.count > 13 and _slot_mismatch(words.c13, words.f13, words.sizes[13], index):
        bits |= 1 << 13
    if words.count > 14 and _slot_mismatch(words.c14, words.f14, words.sizes[14], index):
        bits |= 1 << 14
    if words.count > 15 and _slot_mismatch(words.c15, words.f15, words.sizes[15], index):
        bits |= 1 << 15
    if bits != 0:
        wp.atomic_or(mismatch, 0, bits)


def fingerprint(array):
    """Host fingerprint of an array's shape, dtype and bytes (``None`` stays ``None``)."""
    if array is None:
        return None
    values = array.numpy()
    return values.shape, values.dtype.str, hashlib.sha256(values.tobytes()).digest()


class DeviceProof:
    """Frozen evidence for a named set of arrays, validated on the device when possible."""

    def __init__(self, entries, device):
        """``entries``: iterable of ``(label, array)``; ``device``: the owner's device."""
        self.device = device
        self.labels = []
        self._frozen = {}
        self._host = {}
        self._status = None
        words = []
        for label, array in entries:
            self.labels.append(label)
            view = self._words(array, device)
            if view is None or len(words) >= _SLOTS:
                self._host[label] = fingerprint(array)
                continue
            words.append((label, view))
        if words:
            self._status = wp.zeros(1, dtype=int, device=device)
            for label, view in words:
                self._frozen[label] = (wp.clone(view), view.shape[0])
            self._sizes = wp.array([size for _, size in self._frozen.values()], dtype=int, device=device)
            self._largest = max(size for _, size in self._frozen.values())

    @staticmethod
    def _words(array, device):
        if array is None or not isinstance(array, wp.array) or not device.is_cuda:
            return None
        if array.device != device or not array.is_contiguous or array.size == 0:
            return None
        try:
            return array.view(wp.uint32).flatten()
        except (TypeError, RuntimeError):
            return None

    def validate(self, current, what):
        """Raise ``RuntimeError`` naming the first changed array; ``current`` maps label to array."""
        launched = []
        for label in self.labels:
            array = current(label)
            if label in self._frozen:
                frozen, size = self._frozen[label]
                view = self._words(array, self.device)
                if view is None or view.shape[0] != size:
                    raise RuntimeError(f"{what} {label} changed; reconstruct and recapture")
                launched.append((label, view, frozen))
            elif fingerprint(array) != self._host[label]:
                raise RuntimeError(f"{what} {label} changed; reconstruct and recapture")
        if not launched:
            return
        # One launch for every array on the caller's stream, then one readback:
        # a notification arrives on most Isaac Lab steps, and twelve separate
        # compare launches per notification showed up as +1% on a 2.5 ms scene.
        words = ProofWords()
        words.count = len(launched)
        words.sizes = self._sizes
        for slot in range(_SLOTS):
            _, view, frozen = launched[min(slot, len(launched) - 1)]
            setattr(words, f"c{slot}", view)
            setattr(words, f"f{slot}", frozen)
        self._status.zero_()
        wp.launch(_compare_words, dim=self._largest, inputs=[words, self._status], device=self.device)
        mismatch = int(self._status.numpy()[0])
        for bit, (label, _, _) in enumerate(launched):
            if mismatch & (1 << bit):
                raise RuntimeError(f"{what} {label} changed; reconstruct and recapture")
