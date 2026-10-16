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


@wp.kernel(module="unique", enable_backward=False)
def _compare_words(current: wp.array[wp.uint32], frozen: wp.array[wp.uint32], mismatch: wp.array[int], bit: int):
    index = wp.tid()
    if current[index] != frozen[index]:
        wp.atomic_or(mismatch, 0, bit)


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
            if view is None or len(words) >= 31:
                self._host[label] = fingerprint(array)
                continue
            words.append((label, view))
        if words:
            self._status = wp.zeros(1, dtype=int, device=device)
            for label, view in words:
                self._frozen[label] = (wp.clone(view), view.shape[0])

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
        # Order the comparison after the caller's stream like array.numpy() does.
        with wp.ScopedStream(self.device.null_stream):
            self._status.zero_()
            for bit, (_, view, frozen) in enumerate(launched):
                wp.launch(
                    _compare_words, dim=frozen.size, inputs=[view, frozen, self._status, 1 << bit], device=self.device
                )
            mismatch = int(self._status.numpy()[0])
        for bit, (label, _, _) in enumerate(launched):
            if mismatch & (1 << bit):
                raise RuntimeError(f"{what} {label} changed; reconstruct and recapture")
