# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The free-body and factor-row builds of the response operators on their own: the free-body row word the sweeps
decode, the paired components' qualification of a world, and (with ``ptxas`` on ``PATH``) the native builds'
offline CUDA assembly without spills.
"""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import field_group
from newton._src.solvers.feather_pgs.paired_components import PairedComponentData
from newton._src.solvers.feather_pgs.response_operator import FactorRows, FreeBody6, SparseFactor

DEVICE = "cpu"


class TestFreeBody6Build(unittest.TestCase):
    """The free-body rows' build."""

    def test_meta_word_is_the_consumed_word(self):
        """The packed word's fields are the ones :meth:`.FreeBody6.consume` decodes."""
        word = FreeBody6().pack("w", "a", "b", "inv", "rhs", "kind", "parent", "")
        self.assertIn("w.x = (a << 16) | (b & 0xFFFF);", word)
        self.assertIn("w.y = __float_as_int(inv);", word)
        consume = FreeBody6().consume("i", "n", "")
        self.assertIn("int dof_a = packed_dofs >> 16;", consume)
        self.assertIn("int dof_b = (packed_dofs << 16) >> 16;", consume)
        self.assertIn("float mf_diag = __int_as_float(meta.y);", consume)
        with self.assertRaises(ValueError):
            FreeBody6.build().meta(70000, "sm_120")


def _lower_factors(rng, groups, size):
    factor = np.tril(rng.normal(size=(groups, size, size))) * 0.3
    for group in range(groups):
        factor[group][np.diag_indices(size)] = rng.uniform(0.5, 2.0, size)
    factor[0, 2, 2] = 0.0  # a zero pivot: the solves' zero convention
    return factor.astype(np.float32)


def _components_data(P, rng, *, qualified=True):
    """Paired component inputs of two worlds: the first qualifies (valid rows on their own components), the second
    has a dense row on the free body's coordinates."""
    M, F, W = 6, 3, P + 6
    J = np.zeros((2, M, W), np.float32)
    Y = np.zeros((2, M, W), np.float32)
    J[:, :, :P] = rng.normal(size=(2, M, P))
    Y[:, :, :P] = rng.normal(size=(2, M, P))
    J[1, 1, P + 2] = 1.0  # couples the components: the world falls back
    row_type = np.array([[0, 2, 2, 3, 0, 3]] * 2, np.int32)
    row_parent = np.array([[-1, 0, 0, -1, -1, -1]] * 2, np.int32)
    meta = np.zeros((2, F * 4), np.int32)
    for row, (kind, parent) in enumerate(((0, 0xFFFF), (2, 0), (2, 0))):
        meta[:, row * 4] = (P << 16) | 0xFFFF
        meta[:, row * 4 + 1] = np.float32(0.5).view(np.int32)
        meta[:, row * 4 + 2] = np.float32(0.1).view(np.int32)
        meta[:, row * 4 + 3] = np.array(kind | (parent << 16), np.uint32).view(np.int32)
    L = _lower_factors(rng, 2, P)
    if not qualified:
        L[0, 0, 0] = np.inf
    return field_group.upload(
        PairedComponentData,
        DEVICE,
        world_count=2,
        counts=np.array([M, M], np.int32),
        mf_counts=np.array([3, 3], np.int32),
        primary_offset=np.zeros(2, np.int32),
        secondary_offset=np.full(2, P, np.int32),
        primary_group=np.array([0, 1], np.int32),
        selector=np.full(2, 7, np.int32),
        J=J,
        Y=Y,
        L=L,
        rhs=rng.normal(size=(2, M)).astype(np.float32),
        diag=rng.uniform(0.1, 1.0, (2, M)).astype(np.float32),
        mu=np.full((2, M), 0.5, np.float32),
        row_type=row_type,
        row_parent=row_parent,
        mf_meta=meta,
        mf_J_a=rng.normal(size=(2, F, 6)).astype(np.float32),
        mf_J_b=rng.normal(size=(2, F, 6)).astype(np.float32),
        mf_MiJt_a=rng.normal(size=(2, F, 6)).astype(np.float32),
        mf_MiJt_b=rng.normal(size=(2, F, 6)).astype(np.float32),
        mf_mu=np.full((2, F), 0.5, np.float32),
    )


class TestFactorRowsBuild(unittest.TestCase):
    """Paired factor rows: the held form of qualified worlds."""

    def test_components(self):
        """A world whose rows stay on their own components qualifies (selector -1); a coupled one falls back (3)."""
        for P in (5, 22):
            for qualified in (True, False):
                with self.subTest(primary=P, qualified=qualified):
                    data = _components_data(P, np.random.default_rng(37), qualified=qualified)
                    wp.launch(FactorRows.build().components(P), dim=(2, 128), inputs=[2, data], device=DEVICE)
                    self.assertEqual(list(data.selector.numpy()), [-1 if qualified else 3, 3])


def _assemble(kernel, arch: int = 120) -> dict:
    """Assemble a kernel's module offline (Warp's PTX, then ``ptxas``) and return its entry's resources."""
    with tempfile.TemporaryDirectory() as directory:
        options = kernel.module.resolve_options(wp.config, block_dim=128)
        kernel.module._compile(None, directory, output_arch=arch, use_ptx=True, options=options)
        ptx = sorted(Path(directory).rglob("*.ptx"))[0]
        source = Path(directory) / "module.ptx"
        source.write_bytes(ptx.read_bytes().rstrip(b"\0"))
        result = subprocess.run(
            ["ptxas", f"-arch=sm_{arch}", "-v", str(source), "-o", str(Path(directory) / "module.cubin")],
            check=True,
            capture_output=True,
            text=True,
        )
    name = f"{kernel.get_mangled_name()}_cuda_kernel_forward"
    entry = result.stderr[result.stderr.index(name) :]
    return {
        "registers": int(re.search(r"Used (\d+) registers", entry).group(1)),
        "spills": int(re.search(r"(\d+) bytes spill stores", entry).group(1)),
    }


@unittest.skipUnless(shutil.which("ptxas"), "ptxas is not on PATH")
class TestNativeBuildsAssemble(unittest.TestCase):
    """The native builds assemble for CUDA without spills."""

    def test_native_builds_assemble_without_spills(self):
        builds = {
            "free-body words": lambda: FreeBody6.build().meta(64, "sm_120"),
            "limit rows": lambda: SparseFactor.build().limit_rows,
            "mimic rows": lambda: SparseFactor.build().mimic_rows,
            "factor": lambda: SparseFactor.build().factor(12, 30),
            "contact rows": lambda: SparseFactor.build().contact_rows(
                12, prepare_rows=True, use_support_table=True, contact_records=True
            ),
            "paired components": lambda: FactorRows.build().components(22),
        }
        for name, build in builds.items():
            with self.subTest(build=name):
                self.assertEqual(_assemble(build())["spills"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
