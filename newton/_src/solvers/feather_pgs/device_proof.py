# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prove frozen plan arrays unchanged on the device with one small readback.

Cache owners bake topology (parents, axes, body membership, group maps) into
their kernels and must refuse to run after it changes. Reading every plan
array back to the host to fingerprint it on each model-change notification
costs several milliseconds per notification at 16k worlds, and Isaac Lab
notifies on every reset. This helper keeps a position-sensitive checksum of
each array's words (two 32-bit sums, no device copy), recomputes them with one
kernel launch over every array and reads back one small vector. Arrays that
cannot be viewed as device words (host arrays, non-contiguous views, ``None``)
keep a host fingerprint.

An owner may also watch arrays its kernels read live, such as joint frames: a
changed watched array does not raise, the validation reports it, and its new
checksums replace the stored ones, so the owner can invalidate its caches and
keep them across later notifications that change nothing.
"""

import hashlib

import warp as wp

_SLOTS = 16


@wp.struct
class ProofWords:
    """Up to sixteen word views summed in one launch (slot i writes sums 2i and 2i+1)."""

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


@wp.func
def _mix(word: wp.uint32, index: int) -> wp.uint32:
    """A word's position-sensitive contribution (a multiplicative hash of word and index)."""
    value = word ^ (wp.uint32(index) * wp.uint32(0x9E3779B9))
    value = (value ^ (value >> wp.uint32(16))) * wp.uint32(0x85EBCA6B)
    value = (value ^ (value >> wp.uint32(13))) * wp.uint32(0xC2B2AE35)
    return value ^ (value >> wp.uint32(16))


@wp.func
def _slot_sums(
    current: wp.array[wp.uint32], size: int, thread: int, threads: int, slot: int, sums: wp.array[wp.uint32]
):
    """Sum this thread's strided words of one array, then add once to each of its two sums."""
    if thread >= size:
        return
    first = wp.uint32(0)
    second = wp.uint32(0)
    index = thread
    while index < size:
        value = _mix(current[index], index)
        first += value
        second += _mix(value, index + 1)
        index += threads
    wp.atomic_add(sums, 2 * slot, first)
    wp.atomic_add(sums, 2 * slot + 1, second)


# Words summed per thread before its one atomic per sum: the position-sensitive
# word hashes commute, so the stride only bounds atomic traffic.
_WORDS_PER_THREAD = 32


@wp.kernel(module="unique", enable_backward=False)
def _checksum_words(words: ProofWords, threads: int, sums: wp.array[wp.uint32]):
    """Each thread sums a stride of every array; two atomics per array per thread."""
    index = wp.tid()
    _slot_sums(words.c0, words.sizes[0], index, threads, 0, sums)
    if words.count > 1:
        _slot_sums(words.c1, words.sizes[1], index, threads, 1, sums)
    if words.count > 2:
        _slot_sums(words.c2, words.sizes[2], index, threads, 2, sums)
    if words.count > 3:
        _slot_sums(words.c3, words.sizes[3], index, threads, 3, sums)
    if words.count > 4:
        _slot_sums(words.c4, words.sizes[4], index, threads, 4, sums)
    if words.count > 5:
        _slot_sums(words.c5, words.sizes[5], index, threads, 5, sums)
    if words.count > 6:
        _slot_sums(words.c6, words.sizes[6], index, threads, 6, sums)
    if words.count > 7:
        _slot_sums(words.c7, words.sizes[7], index, threads, 7, sums)
    if words.count > 8:
        _slot_sums(words.c8, words.sizes[8], index, threads, 8, sums)
    if words.count > 9:
        _slot_sums(words.c9, words.sizes[9], index, threads, 9, sums)
    if words.count > 10:
        _slot_sums(words.c10, words.sizes[10], index, threads, 10, sums)
    if words.count > 11:
        _slot_sums(words.c11, words.sizes[11], index, threads, 11, sums)
    if words.count > 12:
        _slot_sums(words.c12, words.sizes[12], index, threads, 12, sums)
    if words.count > 13:
        _slot_sums(words.c13, words.sizes[13], index, threads, 13, sums)
    if words.count > 14:
        _slot_sums(words.c14, words.sizes[14], index, threads, 14, sums)
    if words.count > 15:
        _slot_sums(words.c15, words.sizes[15], index, threads, 15, sums)


def fingerprint(array):
    """Host fingerprint of an array's shape, dtype and bytes (``None`` stays ``None``)."""
    if array is None:
        return None
    values = array.numpy()
    return values.shape, values.dtype.str, hashlib.sha256(values.tobytes()).digest()


class DeviceProof:
    """Frozen and watched evidence for a named set of arrays, validated on the device when possible."""

    def __init__(self, entries, device, *, watched=()):
        """``entries``: iterable of ``(label, array)``; ``device``: the owner's device.

        ``watched`` names the entries whose change :meth:`validate` reports and
        stores instead of raising; every other entry is frozen.
        """
        self.device = device
        self.watched = frozenset(watched)
        self.labels = []
        self._sizes_by_label = {}
        self._host = {}
        self._sums = None
        self._frozen = None
        views = []
        for label, array in entries:
            self.labels.append(label)
            view = self._words(array, device)
            if view is None or len(views) >= _SLOTS:
                self._host[label] = fingerprint(array)
                continue
            self._sizes_by_label[label] = view.shape[0]
            views.append(view)
        if views:
            self._sums = wp.zeros(2 * _SLOTS, dtype=wp.uint32, device=device)
            self._sizes = wp.array([view.shape[0] for view in views], dtype=int, device=device)
            self._threads = -(-max(view.shape[0] for view in views) // _WORDS_PER_THREAD)
            self._frozen = self._checksums(views)

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

    def _checksums(self, views):
        """One launch over every view on the caller's stream, then one readback."""
        words = ProofWords()
        words.count = len(views)
        words.sizes = self._sizes
        for slot in range(_SLOTS):
            setattr(words, f"c{slot}", views[min(slot, len(views) - 1)])
        self._sums.zero_()
        wp.launch(_checksum_words, dim=self._threads, inputs=[words, self._threads, self._sums], device=self.device)
        return self._sums.numpy()[: 2 * len(views)].tolist()

    def validate(self, current, what):
        """Raise ``RuntimeError`` naming the first changed frozen array; ``current`` maps label to array.

        A watched array whose content changed is reported instead, and its new
        checksums (or host fingerprint) replace the stored ones, so the next
        validation compares with its content now. A watched array whose size
        changed raises as a frozen one does.

        Returns:
            The labels of the changed watched arrays, in entry order; empty when
            none changed.
        """
        views, moved, fingerprints = [], [], {}
        for label in self.labels:
            array = current(label)
            if label in self._sizes_by_label:
                view = self._words(array, self.device)
                if view is None or view.shape[0] != self._sizes_by_label[label]:
                    raise RuntimeError(f"{what} {label} changed; reconstruct and recapture")
                views.append((label, view))
                continue
            value, frozen = fingerprint(array), self._host[label]
            if value == frozen:
                continue
            # A host fingerprint leads with the array's shape and dtype.
            if label not in self.watched or value is None or frozen is None or value[:2] != frozen[:2]:
                raise RuntimeError(f"{what} {label} changed; reconstruct and recapture")
            fingerprints[label] = value
        stored = {}
        if views:
            # A notification arrives on most Isaac Lab steps: one launch and one
            # small readback per notification, and no frozen device copies.
            sums = self._checksums([view for _, view in views])
            for slot, (label, _) in enumerate(views):
                window = slice(2 * slot, 2 * slot + 2)
                if sums[window] == self._frozen[window]:
                    continue
                if label not in self.watched:
                    raise RuntimeError(f"{what} {label} changed; reconstruct and recapture")
                stored[label] = (window, sums[window])
        # Every frozen array is proven unchanged: store the watched arrays' new content.
        for label in self.labels:
            if label in fingerprints:
                self._host[label] = fingerprints[label]
            elif label in stored:
                window, values = stored[label]
                self._frozen[window] = values
            else:
                continue
            moved.append(label)
        return moved
