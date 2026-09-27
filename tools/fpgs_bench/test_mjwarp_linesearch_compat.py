# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test the optional workaround without Isaac Lab or a GPU launch."""

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mjwarp_linesearch_compat as compat


def _available():
    """Skip version-specific kernel tests when the optional backend is absent."""
    return (
        importlib.util.find_spec("mujoco_warp") is not None
        and compat.importlib.metadata.version("mujoco-warp") == "3.12.0"
    )


def _fixture(*, budget=2, adjacent=False):
    """Construct one complete scalar line-search owner with no collision inputs."""
    import numpy as np  # noqa: PLC0415 - optional CPU kernel test dependency
    import warp as wp  # noqa: PLC0415 - optional CPU kernel test dependency
    from mujoco_warp._src import types

    def array(value, dtype=float):
        host = np.asarray(
            value,
            dtype=np.float32 if dtype is float else (bool if dtype is bool else np.int32),
        )
        return wp.array(host, dtype=dtype, device="cpu")

    model = SimpleNamespace(
        nv=1,
        is_sparse=False,
        opt=SimpleNamespace(
            ls_iterations=budget,
            cone=types.ConeType.PYRAMIDAL,
            solver=types.SolverType.NEWTON,
            tolerance=array([1e-8]),
            ls_tolerance=array([0.01]),
            impratio_invsqrt=array([1]),
            warn_overflow=True,
        ),
        stat=SimpleNamespace(meaninertia=array([1])),
        block_dim=SimpleNamespace(linesearch_iterative=32),
    )
    data = SimpleNamespace(
        nworld=1,
        njmax=1,
        ne=array([0], int),
        nf=array([1], int),
        nefc=array([1], int),
        qfrc_smooth=array([[100.1 if adjacent else 2]]),
        nacon=array([0], int),
        qacc=array([[0]]),
        overflow=array([0], int),
        contact=SimpleNamespace(
            friction=wp.zeros(1, dtype=types.vec5, device="cpu"),
            dim=array([3], int),
            efc_address=array([[0] * 6], int),
        ),
        efc=SimpleNamespace(
            type=array([[1]], int),
            id=array([[0]], int),
            J_rownnz=array([[1]], int),
            J_rowadr=array([[0]], int),
            J_colind=array([[[0]]], int),
            J=array([[[1]]]),
            D=array([[2 if adjacent else 1]]),
            frictionloss=array([[0.1 if adjacent else 0.5]]),
            Ma=array([[0]]),
        ),
    )
    context = SimpleNamespace(
        search_unchanged=array([False], bool),
        Jaref=array([[100 if adjacent else -2]]),
        search=array([[1]]),
        search_dot=array([1]),
        mv=array([[100 if adjacent else 1]]),
        jv=array([[1]]),
        quad=wp.zeros((1, 1), dtype=wp.vec3, device="cpu"),
        done=array([False], bool),
        improvement=array([0]),
        alpha=array([0]),
        ls_exhausted=array([False], bool),
    )
    return model, data, context


def _convex_fixture(sign, *, collapsed=False):
    """Construct the physical Newton search for a mirrored strictly convex ray."""
    import numpy as np  # noqa: PLC0415 - optional CPU kernel test dependency
    import warp as wp  # noqa: PLC0415 - optional CPU kernel test dependency

    def array(value, dtype=float):
        host = np.asarray(value, dtype=np.float32 if dtype is float else np.int32)
        return wp.array(host, dtype=dtype, device="cpu")

    model, data, context = _fixture(budget=15)
    if collapsed:
        position, diagonal, jacobian = [-0.1, -1, 2], [99, 9, 9], [sign, sign, -sign]
        search = np.float32(sign * (22.9 / 109))
    else:
        position, diagonal, jacobian = [-0.1, 1], [9, 9], [sign, -sign]
        search = np.float32(sign * 0.49)
    count = len(position)
    data.njmax = count
    data.ne = array([0], int)
    data.nf = array([0], int)
    data.nefc = array([count], int)
    data.qfrc_smooth = array([[4 * sign]])
    data.efc.type = array([[6] * count], int)
    data.efc.id = array([[0] * count], int)
    data.efc.J_rownnz = array([[1] * count], int)
    data.efc.J_rowadr = array([list(range(count))], int)
    data.efc.J_colind = array([[[0]] * count], int)
    data.efc.J = array([[[value] for value in jacobian]])
    data.efc.D = array([diagonal])
    data.efc.frictionloss = array([[0] * count])
    context.Jaref = array([position])
    context.jv = array([[value * search for value in jacobian]])
    context.search = array([[search]])
    context.search_dot = array([search * search])
    context.mv = array([[search]])
    context.quad = wp.zeros((1, count), dtype=wp.vec3, device="cpu")
    return model, data, context


@unittest.skipUnless(_available(), "Require optional reviewed MJWarp 3.12.0")
class CompatTests(unittest.TestCase):
    def setUp(self):
        """Isolate process-local bindings and cache entries in each unit test."""
        from mujoco_warp._src import solver, warp_util

        self.solver, self.cache = solver, warp_util._KERNEL_CACHE
        self.original = solver._linesearch_iterative
        self.original_factory = solver._linesearch_iterative_kernel
        self.previous_cache = dict(self.cache)
        self.previous_install = compat._INSTALLED
        self.cache.clear()
        compat._INSTALLED = None

    def tearDown(self):
        """Restore every binding and cache entry owned by this test."""
        self.solver._linesearch_iterative = self.original
        self.solver._linesearch_iterative_kernel = self.original_factory
        self.cache.clear()
        self.cache.update(self.previous_cache)
        compat._INSTALLED = self.previous_install

    def test_source_and_version_fail_closed(self):
        """Reject unknown bytes and versions before replacing any runtime binding."""
        data = Path(self.solver.__file__).read_bytes()
        with self.assertRaisesRegex(RuntimeError, "reviewed"):
            compat._transform(data + b"\n# unreviewed edit\n")
        with patch.object(compat.importlib.metadata, "version", return_value="3.13.0"):
            with self.assertRaisesRegex(RuntimeError, "3.12.0"):
                compat.install()
        self.assertIs(self.solver._linesearch_iterative, self.original)

    def test_reviewed_generated_source(self):
        """Match the factory independently compared with the paired-GPU diagnostic AST."""
        source = compat._transform(Path(self.solver.__file__).read_bytes())
        self.assertEqual(
            hashlib.sha256(source.encode()).hexdigest(),
            "485f85ccb4af6b8d0e4ec45be56e2b6c7e3300612e203d5e59c5dea98b934d0d",
        )
        self.assertNotIn("trace_", source)

    def test_dependency_fail_closed(self):
        """Reject an altered original helper dependency without modifying it."""
        with patch.dict(compat._DEPENDENCIES, {"math.py": "0" * 64}):
            with self.assertRaisesRegex(RuntimeError, "dependency"):
                compat.install()
        self.assertIs(self.solver._linesearch_iterative, self.original)

    def test_pre_jit_guard(self):
        """Reject installation after the original line-search factory was cached."""
        self.cache[(1, hash("_linesearch_iterative_kernel"))] = object()
        with self.assertRaisesRegex(RuntimeError, "BEFORE"):
            compat.install()
        self.assertIs(self.solver._linesearch_iterative, self.original)

    def test_install_identity_and_public_metadata(self):
        """Keep installation idempotent, source files intact, and metadata JSON-safe."""
        path = Path(self.solver.__file__)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = compat.install()
        self.assertEqual(metadata, compat.install())
        self.assertEqual(json.loads(json.dumps(metadata)), metadata)
        self.assertEqual(metadata["source_sha256"], before)
        self.assertFalse(metadata["installed_package_modified"])
        self.assertIs(self.solver._linesearch_iterative_kernel, self.original_factory)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        self.solver._linesearch_iterative = lambda *args: None
        with self.assertRaisesRegex(RuntimeError, "Another owner"):
            compat.install()

    def test_exact_zero_improves_original_same_budget(self):
        """Accept the actual exact-zero root that the original two-slot search loses."""
        compat.install()
        first = _fixture()
        second = _fixture()
        self.original(*first, False)
        self.solver._linesearch_iterative(*second, False)
        self.assertEqual(float(first[2].alpha.numpy()[0]), 2.5)
        self.assertEqual(float(second[2].alpha.numpy()[0]), 2.0)
        self.assertEqual(int(second[1].overflow.numpy()[0]), 0)

    def test_adjacent_converges_and_true_budget_failure_still_warns(self):
        """Resolve an unattainable scalar tolerance but retain true exhaustion."""
        compat.install()
        original = _fixture(budget=15, adjacent=True)
        modified = _fixture(budget=15, adjacent=True)
        self.original(*original, False)
        self.solver._linesearch_iterative(*modified, False)
        self.assertEqual(int(original[1].overflow.numpy()[0]), 1024)
        self.assertEqual(int(modified[1].overflow.numpy()[0]), 0)
        self.assertEqual(float(modified[2].alpha.numpy()[0]), 1.0)
        failed = _fixture(budget=1)
        self.solver._linesearch_iterative(*failed, False)
        self.assertEqual(int(failed[1].overflow.numpy()[0]), 1024)

    def test_unsupported_models_use_original_factory(self):
        """Leave CG and unknown cones on the complete original driver."""
        from mujoco_warp._src import types

        compat.install()
        for field, value in (
            ("solver", types.SolverType.CG),
            ("cone", -1),
        ):
            model, data, context = _fixture()
            setattr(model.opt, field, value)
            self.assertFalse(compat.supports_model(model))
            with patch.object(self.solver, "_linesearch_iterative_kernel", wraps=self.original_factory) as original:
                self.solver._linesearch_iterative(model, data, context, False)
                original.assert_called_once()
        self.assertFalse(compat.supports_model(SimpleNamespace()))

    def test_ellipse_source_isolated_from_pyramid(self):
        """Keep both cone factories immutable and preserve the reviewed pyramid source."""
        from mujoco_warp._src import types

        data = Path(self.solver.__file__).read_bytes()
        original_source = compat._transform(data)
        ellipse_source = compat._ellipse_transform(data)
        self.assertEqual(
            ellipse_source.removeprefix(compat._ELLIPTIC_HELPERS).replace(
                "def _newton_elliptic_linesearch_iterative_kernel(", "def _newton_linesearch_iterative_kernel("
            ),
            original_source,
        )
        metadata = compat.install()
        self.assertEqual(
            metadata["generated_elliptic_factory_sha256"], hashlib.sha256(ellipse_source.encode()).hexdigest()
        )
        self.assertEqual(
            metadata["generated_factory_sha256"], "485f85ccb4af6b8d0e4ec45be56e2b6c7e3300612e203d5e59c5dea98b934d0d"
        )
        self.assertTrue(metadata["friction_delta"])
        self.assertFalse(metadata["general_trajectory_equivalence_validated"])
        for cone in (types.ConeType.PYRAMIDAL, types.ConeType.ELLIPTIC):
            model, data, context = _fixture()
            model.opt.cone = cone
            self.assertTrue(compat.supports_model(model))
            self.solver._linesearch_iterative(model, data, context, False)
        for name in ("_newton_linesearch_iterative_kernel", "_newton_elliptic_linesearch_iterative_kernel"):
            self.assertTrue(any(key[-1] == hash(name) for key in self.cache))
        self.assertFalse(any(key[-1] == hash("_linesearch_iterative_kernel") for key in self.cache))
        self.assertIsNot(compat._INSTALLED[2], compat._INSTALLED[3])

    def test_missing_sign_convex_physical_newton(self):
        """Reject the original false no-progress answer on both mirrored objectives."""
        compat.install()
        for sign in (1, -1):
            with self.subTest(sign=sign):
                original, modified = _convex_fixture(sign), _convex_fixture(sign)
                self.original(*original, False)
                self.solver._linesearch_iterative(*modified, False)
                old_x = float(original[1].qacc.numpy()[0, 0])
                new_x = float(modified[1].qacc.numpy()[0, 0])
                self.assertAlmostEqual(old_x, sign * 0.49, places=5)
                self.assertAlmostEqual(new_x, sign * 1.3, places=6)
                self.assertEqual(int(original[1].overflow.numpy()[0]), 0)
                self.assertEqual(int(modified[1].overflow.numpy()[0]), 0)

    def test_collapsed_anchor_convex_physical_newton(self):
        """Acquire from equal same-sign anchors after two curvature drops."""
        compat.install()
        for sign in (1, -1):
            with self.subTest(sign=sign):
                original = _convex_fixture(sign, collapsed=True)
                modified = _convex_fixture(sign, collapsed=True)
                self.original(*original, False)
                self.solver._linesearch_iterative(*modified, False)
                old_x = float(original[1].qacc.numpy()[0, 0])
                new_x = float(modified[1].qacc.numpy()[0, 0])
                self.assertAlmostEqual(old_x, sign * 1.3, places=5)
                self.assertAlmostEqual(new_x, sign * 2.2, places=6)
                self.assertEqual(int(original[1].overflow.numpy()[0]), 0)
                self.assertEqual(int(modified[1].overflow.numpy()[0]), 0)

    def test_acquisition_order_and_finite_guards(self):
        """Equal anchors need strict new separation; reversed and nonfinite fail closed."""
        import numpy as np  # noqa: PLC0415 - optional CPU kernel test dependency
        import warp as wp  # noqa: PLC0415 - optional CPU kernel test dependency

        compat.install()
        module = compat._INSTALLED[2]
        high, low = module._acquire_high, module._acquire_low

        @wp.kernel(module="unique", enable_backward=False)
        def guards(points: wp.array2d[float], result: wp.array2d[int]):
            i = wp.tid()
            lo = wp.vec3(-1.0, points[i, 0], 1.0)
            hi = wp.vec3(-1.0, points[i, 1], 1.0)
            trial = wp.vec3(points[i, 6], points[i, 2], 1.0)
            result[i, 0] = int(high(lo, hi, points[i, 3], points[i, 4], trial, points[i, 5]))
            result[i, 1] = int(low(lo, hi, points[i, 3], points[i, 4], trial, points[i, 5]))

        values = np.asarray(
            [
                [-2, -1, 1, 1, 1, 2, 3],
                [1, 2, -1, 1, 1, 0, 3],
                [-2, -1, 1, 1, 1, 1, 3],
                [1, 2, -1, 1, 1, 1, 3],
                [-2, -1, 1, 2, 1, 3, 3],
                [-2, -1, 1, 1, 1, np.nan, 3],
                [-2, -1, -1, 1, 1, 2, 3],
                [1, 2, 1, 1, 1, 0, 3],
                [-2, -1, 1, 1, 1, 2, np.nan],
            ],
            dtype=np.float32,
        )
        output = wp.zeros((len(values), 2), dtype=int, device="cpu")
        wp.launch(
            guards,
            len(values),
            inputs=[wp.array(values, dtype=float, device="cpu")],
            outputs=[output],
            device="cpu",
        )
        np.testing.assert_array_equal(output.numpy(), [[1, 0], [0, 1], *([[0, 0]] * 7)])

    def test_unique_cache_and_cache_clear(self):
        """Separate original kernels and respect the upstream cache invalidation."""
        compat.install()
        first = _fixture()
        self.solver._linesearch_iterative(*first, False)
        self.assertTrue(any(key[-1] == hash("_newton_linesearch_iterative_kernel") for key in self.cache))
        self.assertFalse(any(key[-1] == hash("_linesearch_iterative_kernel") for key in self.cache))
        keys = set(self.cache)
        self.cache.clear()
        self.assertEqual(len(self.cache), 0)
        self.solver._linesearch_iterative(*_fixture(), False)
        # Warp may reuse its unique-module Kernel under identical settings;
        # upstream cache repopulation, not object novelty, is the contract.
        self.assertEqual(set(self.cache), keys)
        self.solver._linesearch_iterative(*_fixture(budget=3), False)
        self.assertGreater(len(self.cache), len(keys))


if __name__ == "__main__":
    unittest.main()
