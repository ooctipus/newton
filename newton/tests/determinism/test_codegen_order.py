# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A module's generated source must not depend on the order Warp discovers its kernels.

Warp keys a module's compiled binary by a hash over its kernels' sorted hashes. When the
builder emits the kernels in discovery order instead, which follows object addresses and so
changes from process to process, one cache key holds sources that differ only in function
order. The compiler then schedules some kernels differently and fuses different products into
FMAs: FeatherPGS's patch friction build stored displacements that differed in the last bits,
and independent compiles of one tree diverged within two frames of the coins clutter scene.
"""

import unittest

import warp as wp
from warp._src.context import ModuleBuilder, ModuleHasher

from newton._src.solvers.feather_pgs import friction_patches


def _cuda_source(module, kernels):
    """Generate ``module``'s CUDA source with its kernels discovered in the order of ``kernels``."""
    options = module.resolve_options(wp.config) | {"output_arch": None}
    builder = ModuleBuilder(module, options, hasher=ModuleHasher(kernels, options))
    return builder.codegen("cuda")


class TestCodegenOrder(unittest.TestCase):
    def test_module_source_does_not_depend_on_kernel_discovery_order(self):
        module = wp.get_module(friction_patches.__name__)
        module.hash_module()
        kernels = sorted(module._get_live_kernels(), key=lambda kernel: kernel.key)
        self.assertGreater(len(kernels), 1, "the module must hold several kernels to have an order")
        forward = _cuda_source(module, kernels).splitlines()
        backward = _cuda_source(module, list(reversed(kernels))).splitlines()
        first = next((i for i, (a, b) in enumerate(zip(forward, backward, strict=False)) if a != b), None)
        self.assertTrue(
            first is None and len(forward) == len(backward),
            f"the source depends on the kernels' discovery order from line {first}: {forward[first]!r} against "
            f"{backward[first]!r}"
            if first is not None
            else "the sources differ in length",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
