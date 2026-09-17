# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Source/ABI controls and the unchanged saved-input native replay adapter."""

import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import krylov_chord
from newton._src.solvers.feather_pgs import solver_feather_pgs as original
from tools.fpgs_bench import coupled_saved_probe


class TestKrylovChordNative(unittest.TestCase):
    def test_original_abi_and_full_current_owner_mapping(self):
        """Keep the actual staged tier owners and complete original fallback."""
        with patch.object(original, "_REGISTER_WHITENING", True):
            factory = krylov_chord.get_parallel_factory(original._get_pgs_solve_parallel_kernel)
            for arch in (120, 103):
                for tier in (32, 48):
                    args = {
                        "rows": tier,
                        "min_rows": 0 if tier == 32 else 32,
                        "sweeps": 24,
                        "matrix_free": True,
                        "inkernel_response": (18, 0, 0, 0),
                        "exact_row_sums": True,
                        "world_rows": True,
                    }
                    old = original._get_pgs_solve_parallel_kernel(72, 32, 18, arch, **args)
                    new = factory(72, 32, 18, arch, **args)
                    self.assertTrue(new._fpgs_krylov_chord)
                    self.assertIn("_kc24", new.key)
                    self.assertEqual([a.label for a in old.adj.args], [a.label for a in new.adj.args])
                    self.assertEqual(old._fpgs_block_dim, new._fpgs_block_dim)
                    source = new._fpgs_krylov_chord_native
                    self.assertIn("kc_solve_warp", source)
                    self.assertIn("24 - kc_consumed", source)
                    self.assertIn("kc_projected - s_x", source)
                    self.assertNotIn("ad_factor_dirty", source)

    def test_unsupported_factory_is_the_original(self):
        """Preserve unsupported budgets and configurations without truncation."""
        factory = krylov_chord.get_parallel_factory(original._get_pgs_solve_parallel_kernel)
        base = {
            "rows": 32,
            "sweeps": 24,
            "matrix_free": True,
            "inkernel_response": (18, 0, 0, 0),
            "exact_row_sums": True,
            "world_rows": True,
        }
        for changed in ({"sweeps": 8}, {"has_drive_rows": True}, {"nesterov": False}, {"exact_row_sums": False}):
            self.assertFalse(getattr(factory(72, 32, 18, 120, **(base | changed)), "_fpgs_krylov_chord", False))


@unittest.skipUnless(wp.is_cuda_available(), "Root owns the CUDA lease")
class TestKrylovChordCUDA(unittest.TestCase):
    def test_original_fallback_incoming_and_delayed_friction(self):
        """Actual staged 32/64-thread guards preserve the original continuation."""
        path = coupled_saved_probe.HELPER
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), coupled_saved_probe.HELPER_SHA)
        spec = importlib.util.spec_from_file_location("krylov_existing_physics", path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        rows = helper.reference("rows")
        flags = dict.fromkeys(("_INK_CHECK", "_WR_CHECK", "_WR_WARM", "_SHADOW_LEAN"), False)
        flags["_REGISTER_WHITENING"] = True
        with patch.multiple(original, **flags):
            krylov_chord.get_parallel_factory.cache_clear()
            for tier in (32, 48):
                for mode in ("incoming", "delayed"):
                    with self.subTest(tier=tier, mode=mode):
                        arrays, _saved, scalar, owned, _pins = rows.load(0, 0, tier)
                        # One real world, unchanged canonical geometry and operator.
                        world = int(owned[0])
                        count = int(arrays["world_constraint_count"][world])
                        arrays["world_constraint_count"][:] = 0
                        arrays["world_constraint_count"][world] = count
                        if mode == "incoming":
                            arrays["world_impulses"][world, 0] = np.float32(0.125)
                        else:
                            scalar["friction_start_iteration"] = 1
                        old, _old_graph, old_lease = coupled_saved_probe.bind(
                            arrays, scalar, tier, False, "cuda:0", krylov_chord
                        )
                        new, _new_graph, new_lease = coupled_saved_probe.bind(
                            arrays, scalar, tier, True, "cuda:0", krylov_chord
                        )
                        for name in coupled_saved_probe.WRITTEN:
                            np.testing.assert_allclose(new[name], old[name], rtol=3e-5, atol=3e-5, err_msg=name)
                        self.assertTrue(old_lease and new_lease)


def saved_probe():
    """Adapt only the existing candidate import and frozen CPU translation."""
    path = Path(__file__).with_name("coupled_saved_probe.py")
    source = path.read_text()
    assert (
        hashlib.sha256(source.encode()).hexdigest()
        == "a8e85830b8cfbad2b7595b5b1c24c25347c5777261adf442f1e21174e3998599"
    )
    old = '"spectral_residual"),'
    assert source.count(old) == 1
    source = source.replace(old, '"spectral_residual", "krylov_chord"),')
    seam = "    candidate_module.get_parallel_factory.cache_clear()"
    assert source.count(seam) == 1
    source = source.replace(
        seam,
        """    if args.candidate == "krylov_chord":
        cpu_control = importlib.import_module("tools.fpgs_bench.krylov_chord_control")
        assert hashlib.sha256(Path(cpu_control.__file__).read_bytes()).hexdigest() == (
            "05d97ac7c04e13b835235ceaf1cf2e9d2c97ff2d7191d58852f85f9aff5b269b"
        )
"""
        + seam,
    )
    exec(compile(source, str(path) + ":krylov", "exec"), {"__name__": "__main__"})


if __name__ == "__main__":
    if "--gpu-source" in sys.argv:
        saved_probe()
    else:
        unittest.main()
