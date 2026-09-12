# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the installed ellipse-private Huber reformulation on CPU."""

import importlib.util
import unittest

import mjwarp_linesearch_compat as compat


def _available():
    return (
        importlib.util.find_spec("mujoco_warp") is not None
        and compat.importlib.metadata.version("mujoco-warp") == "3.12.0"
    )


def _kernel(module):
    import warp as wp  # noqa: PLC0415 - optional backend test
    from mujoco_warp._src import solver, types

    point = module._compute_efc_eval_pt_elliptic
    three = module._compute_efc_eval_pt_3alphas_elliptic

    @wp.kernel(enable_backward=False)
    def evaluate(
        rows: wp.array2d[float],
        alphas: wp.array[float],
        ne: int,
        nf: int,
        original: wp.array2d[wp.vec3],
        changed: wp.array2d[wp.vec3],
    ):
        i = wp.tid()
        x0, jv = rows[i, 0], rows[i, 1]
        alpha = alphas[i]
        diagonal, frictionloss = rows[:, 2], rows[:, 3]
        quad = wp.vec3(1.0, 2.0, 3.0)
        friction = types.vec5(0.5, 0.5, 0.1, 0.1, 0.1)
        original[i, 0] = solver._compute_efc_eval_pt_elliptic(
            i, alpha, ne, nf, 1.0, 7, diagonal, frictionloss, x0, jv, quad, friction, i, quad, quad
        )
        changed[i, 0] = point(i, alpha, ne, nf, 1.0, 7, diagonal, frictionloss, x0, jv, quad, friction, i, quad, quad)
        a, b, c = solver._compute_efc_eval_pt_3alphas_elliptic(
            i, 0.0, alpha, 2.0 * alpha, ne, nf, 1.0, 7, diagonal, frictionloss, x0, jv, quad, friction, i, quad, quad
        )
        original[i, 1] = a
        original[i, 2] = b
        original[i, 3] = c
        a, b, c = three(
            i, 0.0, alpha, 2.0 * alpha, ne, nf, 1.0, 7, diagonal, frictionloss, x0, jv, quad, friction, i, quad, quad
        )
        changed[i, 1] = a
        changed[i, 2] = b
        changed[i, 3] = c

    return evaluate


def _points(rows, alphas=1.0, *, ne=0, nf=None):
    import numpy as np  # noqa: PLC0415 - optional backend test
    import warp as wp  # noqa: PLC0415 - optional backend test

    compat.install()
    # This fallback makes the old implementation a cost-cancellation negative control.
    module = compat._INSTALLED[3] if len(compat._INSTALLED) > 3 else compat._INSTALLED[2]
    rows = np.asarray(rows, dtype=np.float32)
    alphas = np.broadcast_to(np.asarray(alphas, dtype=np.float32), (len(rows),)).copy()
    r = wp.array(rows, dtype=float, device="cpu")
    a = wp.array(alphas, dtype=float, device="cpu")
    old = wp.zeros((len(rows), 4), dtype=wp.vec3, device="cpu")
    new = wp.zeros_like(old)
    wp.launch(
        _kernel(module), len(rows), inputs=[r, a, ne, len(rows) if nf is None else nf], outputs=[old, new], device="cpu"
    )
    np.testing.assert_array_equal(r.numpy(), rows)
    np.testing.assert_array_equal(a.numpy(), alphas)
    return old.numpy(), new.numpy()


def _cost64(rows, alpha):
    import numpy as np  # noqa: PLC0415 - optional backend test

    x0, direction, diagonal, loss = np.asarray(rows, dtype=np.float32).astype(np.float64).T
    displacement = np.asarray(alpha, dtype=np.float32).astype(np.float64) * direction
    x1, kink = x0 + displacement, loss / diagonal

    def cost(x):
        return np.where(np.abs(x) < kink, 0.5 * diagonal * x * x, loss * (np.abs(x) - 0.5 * kink))

    return cost(x1) - cost(x0)


@unittest.skipUnless(_available(), "Require optional reviewed MJWarp 3.12.0")
class HuberTests(unittest.TestCase):
    def setUp(self):
        """Isolate the process-local public installer and original factory cache."""
        from mujoco_warp._src import solver, warp_util

        self.solver, self.cache = solver, warp_util._KERNEL_CACHE
        self.original = solver._linesearch_iterative
        self.previous_cache, self.previous_install = dict(self.cache), compat._INSTALLED
        self.cache.clear()
        compat._INSTALLED = None

    def tearDown(self):
        """Restore each test's installed dispatcher and cache owner."""
        self.solver._linesearch_iterative = self.original
        self.cache.clear()
        self.cache.update(self.previous_cache)
        compat._INSTALLED = self.previous_install

    def test_sub_ulp_displacement_cost_survives(self):
        """Retain real Huber cost changes lost by subtracting absolute FP32 costs."""
        import numpy as np  # noqa: PLC0415 - optional backend test

        rows = np.array([[100.0, 1.0, 2.0, 1.0], [-100.0, 1.0, 2.0, 1.0]], np.float32)
        alpha = np.float32(2.0**-30)
        old, new = _points(rows, alpha)
        np.testing.assert_array_equal(old[:, 0, 0], 0.0)
        np.testing.assert_array_equal(new[:, 0, 0], [alpha, -alpha])
        np.testing.assert_array_equal(new[:, :, 1:], old[:, :, 1:])
        # A small positive smooth term exposes the wrong total shifted-cost sign.
        self.assertGreater(old[1, 0, 0] + 0.5 * alpha, 0.0)
        self.assertLess(new[1, 0, 0] + 0.5 * alpha, 0.0)

    def test_nine_zones_kinks_and_zero_loss(self):
        """Match the independent Huber law across all region transitions and kinks."""
        import numpy as np  # noqa: PLC0415 - optional backend test

        values = [-3.0, -2.0, -1.75, 0.0, 1.75, 2.0, 3.0]
        rows = np.array([[x, y - x, 4.0, 8.0] for x in values for y in values], np.float32)
        rows = np.vstack((rows, [[3.0, -10.0, 4.0, 0.0]]))
        old, new = _points(rows)
        for column, alpha in enumerate((1.0, 0.0, 1.0, 2.0)):
            np.testing.assert_allclose(new[:, column, 0], _cost64(rows, alpha), rtol=1e-6, atol=1e-5)
        np.testing.assert_array_equal(new[:, :, 1:], old[:, :, 1:])

    def test_invalid_domains_keep_original(self):
        """Keep original invalid costs instead of hiding them with a finite reformulation."""
        import numpy as np  # noqa: PLC0415 - optional backend test

        rows = [
            [np.nan, 1, 2, 1],
            [np.inf, 1, 2, 1],
            [-np.inf, 1, 2, 1],
            [1, np.inf, 2, 1],
            [1, 1, 0, 1],
            [1, 1, -1, 1],
            [1, 1, 2, -1],
            [1, 1, np.inf, 1],
            [1, 1, 2, np.nan],
        ]
        old, new = _points(rows)
        np.testing.assert_array_equal(new, old)
        self.assertTrue(np.isnan(new[0, 0, 0]))
        old, new = _points([[1, 1, 2, 1]], np.inf)
        np.testing.assert_array_equal(new, old)

    def test_nonfriction_rows_and_original_globals_unchanged(self):
        """Preserve equality, ellipse and original module ownership beside friction loss."""
        import numpy as np  # noqa: PLC0415 - optional backend test

        names = (
            "_compute_efc_eval_pt_elliptic",
            "_compute_efc_eval_pt_3alphas_elliptic",
            "_compute_efc_eval_pt_alpha_zero_elliptic",
            "_eval_elliptic_reference",
            "_eval_elliptic_shifted",
        )
        original = {name: getattr(self.solver, name) for name in names}
        old, new = _points([[1, 1, 2, 1]] * 4, ne=1, nf=1)
        np.testing.assert_array_equal(new[[0, 2, 3]], old[[0, 2, 3]])
        np.testing.assert_array_equal(new[:, :, 1:], old[:, :, 1:])
        for name, function in original.items():
            self.assertIs(getattr(self.solver, name), function)
        pyramid, ellipse = compat._INSTALLED[2:4]
        self.assertIsNot(pyramid, ellipse)
        for name in names:
            if name not in ("_compute_efc_eval_pt_elliptic", "_compute_efc_eval_pt_3alphas_elliptic"):
                self.assertIs(getattr(ellipse, name), original[name])
            self.assertIs(getattr(pyramid, name), original[name])


if __name__ == "__main__":
    unittest.main()
