# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One source per FeatherPGS law: the families render, compile and agree with their Warp functions.

The laws (``contact_laws``, ``row_laws``, ``projection_laws``) are written once in
Warp's language; ``law_source`` renders each as a CUDA function for native
sweeps. The tests check the rendering mechanism, the constants the laws share
with the code they replace, and the agreement of each law's two renderings.
"""

import os
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import contact_laws, kernels, law_source, projection_laws, row_laws
from newton._src.solvers.feather_pgs.device_preludes import cuda_native
from newton.tests.feather_pgs_law_harness import (
    FAMILIES,
    LAWS,
    TWIN_AGREEING,
    TWIN_SAMPLES,
    TWIN_TOLERANCES,
    exec_kernel,
    launch,
    mismatches,
    register_device_tests,
    twin_inputs,
    twin_kernels,
)

# ---------------------------------------------------------------------------------------------------------------------
# Mechanism
# ---------------------------------------------------------------------------------------------------------------------


class TestLawSource(unittest.TestCase):
    """The families render once per law, compile for CUDA and keep the constants of the code they replace."""

    def test_families_render_every_cuda_law(self):
        """Render one struct per family with a static function per law that has a CUDA form."""
        for family in FAMILIES:
            source = family.prelude().source
            self.assertEqual(source.count(f"struct {family.name} {{"), 1)
            self.assertIs(family.prelude(), family.prelude())
            for name in family.laws:
                sig = law_source.law_signature(LAWS.get(name) or getattr(contact_laws, name))
                rendered = f" {name}(" in source
                self.assertEqual(rendered, sig.cuda, name)
        self.assertFalse(law_source.law_signature(contact_laws.contact_points_oracle).cuda)
        self.assertEqual(
            [prelude.name for prelude in law_source.law_preludes(row_laws.ROW_LAW, row_laws.ROW_LAW)], ["RowLaw"]
        )

    def test_renderings_keep_the_python_statements(self):
        """Render one C++ statement per Python statement, in order, with Warp's rounding of float constants."""
        source = projection_laws.PROJECTION_LAW.prelude().source
        for line in (
            "auto scale = wp::max(wp::max(a, d), 1e-20f);",
            "auto largest = 0.5f * (a + d + wp::sqrt((a - d) * (a - d) + 4.0f * c * c));",
            "return wp::add(old, wp::mul(omega, wp::sub(result, old)));",
        ):
            self.assertIn(line, source)
        for value in (1.0e-6, 2.0e-7, 1.0e-20, 0.1, 1.0 / 3.0, 1.0e20, 6.0e-8):
            text = law_source._float_literal(value)
            self.assertEqual(np.float32(float(text[:-1])), np.float32(value), text)

    def test_subset_is_enforced(self):
        """Refuse what the CUDA rendering cannot keep equal to the Warp function, naming the law."""
        family = law_source.LawFamily("ScratchLaw")

        @family.law
        def chained(a: float, b: float) -> bool:
            return 0.0 < a < b

        @family.law
        def late_read(a: float) -> float:
            if a > 0.0:
                b = a
            return b

        for law in (chained, late_read):
            with self.assertRaises(law_source.LawSourceError) as raised:
                law_source.render_cuda(law)
            self.assertIn(law_source.law_signature(law).name, str(raised.exception))

    def test_constants_are_the_replaced_codes(self):
        """Keep the row kinds of the kernels' row types."""
        for name in ("CONTACT", "FRICTION", "JOINT_LIMIT", "JOINT_VELOCITY_LIMIT", "MIMIC", "CONNECT"):
            self.assertEqual(getattr(row_laws, f"ROW_{name}"), getattr(kernels, f"PGS_CONSTRAINT_TYPE_{name}"))

    def test_cuda_renderings_compile(self):
        """Compile the three families' CUDA renderings through NVRTC, without a device (sm_80)."""
        import tempfile  # noqa: PLC0415

        body = "    return 0;\n#endif\n"
        namespace = {
            "wp": wp,
            "snippet": cuda_native(body, preludes=law_source.law_preludes(*FAMILIES)),
        }
        source = (
            "@wp.func_native(snippet)\ndef families() -> int: ...\n\n"
            "def compile_families(out: wp.array[int]):\n    out[0] = families()\n"
        )
        kernel = exec_kernel("compile_families", source, namespace)
        with tempfile.TemporaryDirectory() as directory:
            try:
                wp.compile_aot_module(kernel.module, arch=80, module_dir=directory)
            except Exception as error:
                if "NVRTC" in str(error) and "not" in str(error):
                    self.skipTest(f"NVRTC unavailable: {error}")
                raise
            self.assertTrue(any(name.endswith((".cubin", ".ptx")) for name in os.listdir(directory)))


# ---------------------------------------------------------------------------------------------------------------------
# Twins: the Warp function and the CUDA rendering of each law agree
# ---------------------------------------------------------------------------------------------------------------------


def test_twins_agree(test, device):
    """Return equal results from each law's Warp function and CUDA rendering, to an iterative law's tolerance.

    Every other law's results are bitwise equal up to the sign of a zero: Warp negates by subtracting from zero.
    An iterative law's samples agree within its tolerance but for rare ones whose solve takes the other branch of
    a convergence or sticking test (``TWIN_TOLERANCES``).
    """
    rng = np.random.default_rng(1234)
    for name, law in LAWS.items():
        sig = law_source.law_signature(law)
        if not sig.cuda:
            continue
        with test.subTest(law=name):
            warp_kernel, native_kernel = twin_kernels(law)
            inputs = twin_inputs(name, rng, TWIN_SAMPLES, device)
            warp_results = launch(warp_kernel, TWIN_SAMPLES, inputs, sig, device)
            native_results = launch(native_kernel, TWIN_SAMPLES, inputs, sig, device)
            for k, (a, b) in enumerate(zip(warp_results, native_results, strict=True)):
                if name in TWIN_TOLERANCES and a.dtype.kind == "f":
                    key, bound = TWIN_TOLERANCES[name]
                    a64 = a.reshape(TWIN_SAMPLES, -1).astype(np.float64)
                    b64 = b.reshape(TWIN_SAMPLES, -1).astype(np.float64)
                    np.testing.assert_array_equal(np.isnan(a64), np.isnan(b64))
                    if key is None:
                        scale = np.maximum(np.nan_to_num(np.abs(a64)).max(axis=1, keepdims=True), 1.0e-3)
                    else:
                        scale = np.maximum(np.asarray(inputs[key], dtype=np.float64), 1.0e-6)[:, None]
                    gap = (np.nan_to_num(np.abs(a64 - b64)) / scale).max(axis=1)
                    test.assertLessEqual(float(np.quantile(gap, TWIN_AGREEING)), bound, f"result {k}")
                    test.assertLess(float(gap.max()), 1.0e3 * bound, f"result {k}")
                    continue
                bad = np.flatnonzero(mismatches(a, b, signed_zeros=False))
                test.assertEqual(bad.size, 0, f"result {k}: {bad.size} of {TWIN_SAMPLES} differ, first at {bad[:3]}")


class TestFeatherPGSLawTwins(unittest.TestCase):
    """The Warp function and the CUDA rendering of each law, on every test device."""


register_device_tests(TestFeatherPGSLawTwins, globals())


if __name__ == "__main__":
    unittest.main(verbosity=2)
