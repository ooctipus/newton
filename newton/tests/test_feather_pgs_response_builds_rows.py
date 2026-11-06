# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The free-body row build of the response operators on its own: the free-body row word the sweeps decode, and (with
``ptxas`` on ``PATH``) the native build's offline CUDA assembly without spills.
"""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import response_builds_rows
from newton._src.solvers.feather_pgs.factor.walk import WALK_COORDINATE
from newton._src.solvers.feather_pgs.row_metadata import pack_mf_header, unpack_mf_header
from newton.tests.feather_pgs_law_harness import unpacked_rows, walk_inverse

DEVICE = "cpu"


class TestFreeBodyRowsBuild(unittest.TestCase):
    """The free-body rows' build."""

    def test_meta_word_is_the_consumed_word(self):
        """The packed word's fields are the ones the sweeps' row loop decodes."""
        word = pack_mf_header("w", "a", "b", "inv", "rhs", "kind", "parent", "")
        self.assertIn("w.x = (a << 16) | (b & 0xFFFF);", word)
        self.assertIn("w.y = __float_as_int(inv);", word)
        consume = unpack_mf_header("w", "")
        self.assertIn("int dof_a = packed_dofs >> 16;", consume)
        self.assertIn("int dof_b = (packed_dofs << 16) >> 16;", consume)
        self.assertIn("float mf_diag = __int_as_float(w.y);", consume)
        with self.assertRaises(ValueError):
            response_builds_rows.row_record_kernel(8 + 70000, 8, 8, 70000, 32, "sm_120")

    def test_rows_see_the_dynamics_mass_matrix(self):
        """Whiten a free box's contact rows by the factor of its dynamics' mass matrix, armature included.

        A 1 kg, 0.2 m box rests tilted on the ground with a free-joint armature of 0.05 kg m^2, several times its
        principal inertia (1/150 kg m^2): every row's diagonal is ``J H^-1 J^T + cfm`` with ``H`` the box's spatial
        inertia at its center of mass plus the armature, and the rows hold ``Z = L^-1 J^T`` with ``L L^T = H``.
        """
        import newton  # noqa: PLC0415

        armature, mass, half = 0.05, 1.0, 0.1
        builder = newton.ModelBuilder()
        rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.3)
        principal = 2.0 * mass * half * half / 3.0
        body = builder.add_body(
            # Its lowest edge 5 mm into the ground: the rows exist (a world whose contacts separate needs none).
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.12), rotation),
            mass=mass,
            inertia=wp.mat33(principal, 0.0, 0.0, 0.0, principal, 0.0, 0.0, 0.0, principal),
        )
        builder.add_shape_box(body, hx=half, hy=half, hz=half, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        builder.joint_armature[-6:] = [armature] * 6
        builder.add_ground_plane()
        model = builder.finalize(device=DEVICE)
        solver = newton.solvers.SolverFeatherPGS(model)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state, out = model.state(), model.state()
        pipeline.collide(state, contacts)
        solver.step(state, out, model.control(), contacts, 1.0 / 240.0)
        # A free body is a one-node tree: its contact rows are the line's, packed at the world's coordinates.
        rows = int(solver.placement.constraint_count.numpy()[0])
        self.assertGreater(rows, 0, "the box must touch the ground")
        # H at the center of mass: m on the linear block, the world-frame inertia on the angular one, plus armature.
        basis = np.array(wp.quat_to_matrix(rotation), dtype=np.float64).reshape(3, 3)
        inertia = basis @ (model.body_inertia.numpy()[body].astype(np.float64)) @ basis.T
        H = np.zeros((6, 6))
        H[:3, :3] = mass * np.eye(3)
        H[3:, 3:] = inertia
        H += armature * np.eye(6)
        L = np.linalg.inv(walk_inverse(solver, 0))  # the walk: H = L L^T
        np.testing.assert_allclose(L @ L.T, H, rtol=1e-5, atol=1e-6)
        # The box is each row's one body (the ground has none): the row's Jacobian over the world's six coordinates.
        J = unpacked_rows(solver)[0][0, :rows]
        expected = np.einsum("ri,ij,rj->r", J, np.linalg.inv(H), J) + solver.options.pgs_cfm
        np.testing.assert_allclose(solver.placement.diag.numpy()[0, :rows], expected, rtol=1e-5)


def _walk_records(rng, trees, size):
    """The walk's records (:data:`.factor.walk.WALK_COORDINATE` floats a coordinate) of ``trees`` trees of ``size``
    coordinates, one after another: finite and bounded, the qualification's read of each primary tree's factor."""
    return (rng.normal(size=(trees * size, WALK_COORDINATE)) * 0.3).astype(np.float32)


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
            "row records": lambda: response_builds_rows.row_record_kernel(8 + 64, 0, 8, 64, 32, "sm_120"),
        }
        for name, build in builds.items():
            with self.subTest(build=name):
                self.assertEqual(_assemble(build())["spills"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
