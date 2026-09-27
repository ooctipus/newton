# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Actual Python dispatch controls; native fork/join controls run separately."""

import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from newton._src.solvers.feather_pgs import solver_feather_pgs as module


class Stub:
    """Leave unsupported optional paths absent and bind only this dispatch ABI."""

    def __getattr__(self, name):
        return None


class TestPairedOverlap(unittest.TestCase):
    def test_stream_admission_and_unsupported_fallback(self):
        for enabled, change, expected in (
            (False, {}, False),
            (True, {}, True),
            (True, {"use_parallel_streams": False}, False),
            (True, {"_paired_factor_solve_kernel": None}, False),
            (True, {"_paired_response_primary_size": 9}, False),
            (True, {"_mf_warmstart_enabled": True}, False),
            (True, {"_mimic_count": 1}, False),
            (True, {"_connect_count": 1}, False),
            (True, {"_pgs_solve_mf_gs_incremental_kernels": (object(),)}, False),
            (True, {"_rb_build_kernel": object()}, False),
        ):
            with self.subTest(enabled=enabled, change=change):
                obj = Stub()
                obj.use_parallel_streams = True
                obj.size_groups = []
                obj._paired_factor_solve_kernel = object()
                obj._paired_response_primary_size, obj._paired_response_secondary_size = 23, 6
                obj.max_world_dofs = 29
                obj._mimic_count = obj._connect_count = 0
                obj.__dict__.update(change)
                model = SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False)
                with (
                    patch.dict(os.environ, {"FEATHER_PGS_PAIRED_GENERAL_OVERLAP": str(int(enabled))}),
                    patch.object(module.wp, "Stream", return_value=object()),
                    patch.object(module.wp, "Event", return_value=object()),
                ):
                    module.SolverFeatherPGS._init_size_group_streams(obj, model)
                self.assertEqual(obj._paired_general_solve_stream is not None, expected)
                self.assertEqual(obj._paired_general_ready is not None, expected)
                self.assertEqual(obj._paired_general_done is not None, expected)

    def run_dispatch(self, overlap, components):
        trace = []

        class Stream:
            def __init__(self, name):
                self.name = name

            def record_event(self, event=None):
                trace.append((self.name, "record", event))
                return event

            def wait_event(self, event):
                trace.append((self.name, "wait", event))

        caller, paired = Stream("caller"), Stream("paired")
        obj = Stub()
        obj.model = SimpleNamespace(device="mock-device")
        obj._sync_timed = lambda _: nullcontext()
        obj._paired_factor_solve_kernel = "paired-kernel"
        obj._pgs_solve_mf_gs_kernel = "general-kernel"
        obj._pgs_solve_mf_gs_incremental_kernels = ()
        obj._paired_response_primary_size = 23
        obj._paired_response_secondary_size = 6
        obj.L_by_size = obj.Linv_by_size = {23: "L23", 6: "L6"}
        obj._paired_factor_primary_groups_by_world = "primary-groups"
        obj._paired_factor_secondary_groups_by_world = "secondary-groups"
        obj._paired_factor_primary_offsets_by_world = "primary-offsets"
        obj._paired_factor_secondary_offsets_by_world = "secondary-offsets"
        obj._paired_general_solve_stream = paired if overlap else None
        obj._paired_general_ready = "ready"
        obj._paired_general_done = "done"
        obj.world_count = 5
        obj._regularization_enabled = False
        obj._independent_components = (
            SimpleNamespace(prepare=lambda *args: trace.append("prepare"), selector="selector") if components else None
        )

        def launch(kernel, **kwargs):
            trace.append((kernel, "paired" if kwargs.get("stream") is paired else "caller"))

        with (
            patch.object(module.wp, "launch_tiled", side_effect=launch),
            patch.object(module.wp, "get_stream", return_value=caller),
            patch.object(module, "_FPGS_CAPTURE", False),
            patch.object(module, "_INK_CHECK", False),
            patch.object(module, "_WR_CHECK", False),
        ):
            module.SolverFeatherPGS._launch_matrix_free_gs_solve(
                obj,
                dense_rhs="rhs",
                mf_meta="mf-meta",
                iterations=8,
                omega=1.0,
                friction_start_iteration=0,
                row_phase_override=0,
            )
        return trace

    def test_default_order_is_unchanged(self):
        for components in (False, True):
            expected = (["prepare"] if components else []) + [
                ("paired-kernel", "caller"),
                ("general-kernel", "caller"),
            ]
            self.assertEqual(self.run_dispatch(False, components), expected)

    def test_fork_after_prepare_and_join_before_return(self):
        for components in (False, True):
            expected = (["prepare"] if components else []) + [
                ("caller", "record", "ready"),
                ("paired", "wait", "ready"),
                ("general-kernel", "caller"),
                ("paired-kernel", "paired"),
                ("paired", "record", "done"),
                ("caller", "wait", "done"),
            ]
            self.assertEqual(self.run_dispatch(True, components), expected)


if __name__ == "__main__":
    unittest.main()
