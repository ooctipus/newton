# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reuse corrected CSR physical controls with the proven finite-shell query."""

import ast
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from tools.fpgs_bench import test_heightfield_pair_csr as csr_tests


class TestHeightfieldShellPairCSR(csr_tests.TestHeightfieldPairCSR):
    loaded_feature_env = "NEWTON_HEIGHTFIELD_PAIR_CSR_SHELL"
    loaded_feature_attribute = "_heightfield_pair_csr_shell"
    loaded_report_label = "SHELL_PAIR_CSR_QUALIFICATION"

    def setUp(self):
        """Enable only the separately keyed query option in inherited controls."""
        context = patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_PAIR_CSR_SHELL": "1"})
        context.start()
        self.addCleanup(context.stop)

    def test_explicit_factory_modes(self):
        """Keep the cached query choice independent of process environment."""
        from newton._src.geometry.heightfield_pair_csr import get_query_kernels  # noqa: PLC0415

        with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_PAIR_CSR_SHELL": "1"}):
            original = get_query_kernels(False)
        with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_PAIR_CSR_SHELL": "0"}):
            shell = get_query_kernels(True)
        self.assertIs(original, get_query_kernels(False))
        self.assertIs(shell, get_query_kernels(True))
        self.assertEqual(original[1].key, "heightfield_pair_csr_finite_contacts")
        self.assertEqual(shell[1].key, "heightfield_pair_csr_shell_contacts")
        self.assertIsNot(original[1], shell[1])

    def test_actual_binding_cpu(self):
        """Require CSR admission before enabling the shell query factory."""
        from newton._src.geometry.heightfield_pair_csr import get_query_kernels  # noqa: PLC0415

        q = csr_tests.adaptive_tests.qualification_module()
        case = next(case for case in q.CASES if case.name == "support")
        for csr, shell in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(csr=csr, shell=shell):
                with patch.dict(
                    os.environ,
                    {
                        "NEWTON_HEIGHTFIELD_PAIR_CSR": str(int(csr)),
                        "NEWTON_HEIGHTFIELD_PAIR_CSR_SHELL": str(int(shell)),
                        "NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD": "0",
                        "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
                    },
                ):
                    scene = q.build_scene(case, True, "cpu", make_solver=False)
                narrow = scene.pipeline.narrow_phase
                self.assertIs(narrow._heightfield_pair_csr_shell, csr and shell)
                self.assertIs(narrow._heightfield_pair_csr, csr)
                if csr:
                    owner = narrow._pair_csr
                    self.assertIs(owner.shell_support, shell)
                    self.assertEqual((owner.midphase, owner.finite, owner.generic), get_query_kernels(shell))
                scene.pipeline.collide(scene.states[0], scene.contacts)
                narrow.check_buffer_capacity()
                self.assertGreater(int(scene.contacts.rigid_contact_count.numpy()[0]), 0)

    def _handled_shell_faces(self, device):
        q = csr_tests.adaptive_tests.qualification_module()
        for name in ("support", "rebound"):
            case = next(case for case in q.CASES if case.name == name)
            streams, handled = [], []
            for enabled in (False, True):
                with patch.dict(
                    os.environ,
                    {
                        "NEWTON_HEIGHTFIELD_PAIR_CSR": "1",
                        "NEWTON_HEIGHTFIELD_PAIR_CSR_SHELL": str(int(enabled)),
                        "NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD": "0",
                        "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
                    },
                ):
                    scene = q.build_scene(case, True, device, make_solver=False)
                scene.pipeline.collide(scene.states[0], scene.contacts)
                narrow = scene.pipeline.narrow_phase
                narrow.check_buffer_capacity()
                count = int(narrow.triangle_pairs_count.numpy()[0])
                triples = narrow.triangle_pairs.numpy()[:count].copy()
                mask = triples[:, 2] < 0
                handled.append(int(np.count_nonzero(mask)))
                triples[mask, 2] = ~triples[mask, 2]
                tags = narrow._pair_csr.triangle_pair.numpy()[:count]
                tagged = np.column_stack((triples, tags))
                streams.append(tagged[np.lexsort(tagged.T[::-1])])
                self.assertEqual(int(scene.contacts.rigid_contact_count.numpy()[0]), 4)
            np.testing.assert_array_equal(streams[0], streams[1])
            self.assertGreater(handled[1], handled[0], (name, handled))

    def test_handled_shell_faces_cpu(self):
        """Retire positive-face generic queries without changing the tagged stream."""
        self._handled_shell_faces("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Root owns CUDA execution")
    def test_handled_shell_faces_cuda(self):
        """Check native handled markers and exact pair tags on current geometry."""
        self._handled_shell_faces("cuda:0")

    def test_observer_actual_pipeline_cpu(self):
        """Check the real pipeline and reject wrong factories, markers and status."""
        from newton._src.geometry.heightfield_pair_csr import get_query_kernels  # noqa: PLC0415

        wrapper = Path(__file__).with_name("chain_capture_20260916") / "checked_sparse.py"
        syntax = ast.parse(wrapper.read_text())
        literal = next(
            ast.literal_eval(node.value)
            for node in syntax.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "collision_replacement" for target in node.targets)
        )
        namespace = {"os": os}
        exec(compile("def observe(narrow, result):\n" + literal, str(wrapper), "exec"), namespace)
        observe = namespace["observe"]
        q = csr_tests.adaptive_tests.qualification_module()
        case = next(case for case in q.CASES if case.name == "support")
        for enabled in (False, True):
            with patch.dict(
                os.environ,
                {
                    "NEWTON_HEIGHTFIELD_PAIR_CSR": "1",
                    "NEWTON_HEIGHTFIELD_PAIR_CSR_SHELL": str(int(enabled)),
                    "NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD": "0",
                    "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
                },
            ):
                scene = q.build_scene(case, True, "cpu", make_solver=False)
                narrow = scene.pipeline.narrow_phase
                scene.pipeline.collide(scene.states[0], scene.contacts)
                self.assertTrue(observe(narrow, {})["check_pass"])
                with patch.object(narrow._pair_csr, "finite", get_query_kernels(not enabled)[1]):
                    with self.assertRaisesRegex(RuntimeError, "exact production factories"):
                        observe(narrow, {})
                with patch.object(narrow, "_heightfield_pair_csr_shell", not enabled):
                    with self.assertRaisesRegex(RuntimeError, "actual pair-CSR dispatch"):
                        observe(narrow, {})
                with patch.object(narrow._pair_csr, "shell_support", not enabled):
                    with self.assertRaisesRegex(RuntimeError, "shell owner marker"):
                        observe(narrow, {})
                with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_PAIR_CSR_SHELL": ""}):
                    with self.assertRaisesRegex(RuntimeError, "explicit pair-CSR shell flag"):
                        observe(narrow, {})
                narrow._pair_csr.status.fill_(1)
                with self.assertRaisesRegex(RuntimeError, "sticky routing/capacity"):
                    observe(narrow, {})

    @unittest.skipUnless(wp.is_cuda_available(), "Root owns CUDA execution")
    def test_loaded_cuda(self):
        """Compare the unchanged nine loaded cases with CSR enabled in both arms."""
        with patch.dict(
            os.environ,
            {
                "NEWTON_HEIGHTFIELD_PAIR_CSR": "1",
                "NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD": "0",
                "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
            },
        ):
            csr_tests.adaptive_tests.TestAdaptiveManifold.test_loaded_cuda(self)


if __name__ == "__main__":
    unittest.main()
