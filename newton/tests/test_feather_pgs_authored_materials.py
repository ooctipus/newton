# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Authored contact materials (:attr:`~newton.ShapeFlags.CONTACT_MATERIAL`) under FeatherPGS.

A contact of a shape that authors its stiffness and damping takes MuJoCo's soft contact at rest
(:func:`~newton._src.solvers.feather_pgs.row_laws.authored_contact_law`): the law is checked against MJWarp's own
constraint parameters and against the numbers MJWarp produced on the NIST assembly scene.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.row_laws import authored_contact_law, contact_material_rhs

SUBSTEP = 0.0125 / 10.0
"""NIST's substep, 1.25 ms, where MuJoCo's time-constant floor ``2 dt`` (2.5 ms) binds its stiff pairs."""

PART, GROUND, FINGER = (1.6e5, 800.0), (2.5e3, 100.0), (1.0e6, 2000.0)
"""NIST's authored materials (ke [N/m], kd [N s/m]): parts and sockets, the ground (Newton's default), the fingers."""


@wp.kernel
def _authored(
    materials: wp.array[wp.vec4],
    phi: wp.array[float],
    dt: float,
    kappa: wp.array[float],
    beta: wp.array[float],
    rho: wp.array[float],
):
    i = wp.tid()
    m = materials[i]
    k, b = authored_contact_law(m[0], m[1], m[2], m[3], phi[i], dt)
    _rhs, r = contact_material_rhs(phi[i], 0.0, k, b, dt)
    kappa[i] = k
    beta[i] = b
    rho[i] = r


def authored(pairs, phis, dt=SUBSTEP, device="cpu"):
    """The law's ``(kappa, beta, rho)`` for each ``((ke_a, kd_a), (ke_b, kd_b))`` pair at gap ``phi``."""
    materials = wp.array([wp.vec4(*a, *b) for a, b in pairs], dtype=wp.vec4, device=device)
    phi = wp.array(np.asarray(phis, dtype=np.float32), dtype=float, device=device)
    out = [wp.zeros(len(pairs), dtype=float, device=device) for _ in range(3)]
    wp.launch(_authored, dim=len(pairs), inputs=[materials, phi, dt], outputs=out, device=device)
    return tuple(a.numpy().astype(np.float64) for a in out)


def mujoco_reference(a, b, phi, dt):
    """MuJoCo's resting spring and dashpot per unit effective mass, written independently in float64: Newton's
    ``convert_solref(ke, kd, 1, 1)`` per geom, the mean of two geoms' solref, refsafe, ``_contact_kbimp`` on the
    default solimp, and the resting force ``(k imp pos + b v) / R`` with ``R = (1 - imp) / imp * w``."""

    def solref(ke, kd):
        return (2.0 / kd, kd / 2.0 / np.sqrt(ke)) if ke > 0.0 and kd > 0.0 else (0.02, 1.0)

    tc, dr = 0.5 * (np.asarray(solref(*a)) + np.asarray(solref(*b)))
    tc = max(tc, 2.0 * dt)
    dmin, dmax, width, mid = 0.9, 0.95, 0.001, 0.5
    k = 1.0 / (dmax * dmax * tc * tc * dr * dr)
    damp = 2.0 / (dmax * tc)
    x = min(abs(phi) / width, 1.0)
    y = 2.0 * x * x if x <= mid else 1.0 - 2.0 * (1.0 - x) ** 2
    imp = dmin + y * (dmax - dmin)
    return k * imp * imp / (1.0 - imp), damp * imp / (1.0 - imp)


class TestAuthoredContactLaw(unittest.TestCase):
    def test_law_matches_mujoco_reference(self):
        """The law is MuJoCo's pair rule and resting row, over NIST's materials, depths and the refsafe floor."""
        materials = (PART, GROUND, FINGER, (3.0e4, 50.0), (0.0, 100.0))
        pairs, phis, expected = [], [], []
        for i, a in enumerate(materials):
            for b in materials[i:]:
                for phi in (0.0, -2.0e-4, -5.0e-4, -8.0e-4, -3.0e-3, 4.0e-4):
                    pairs.append((a, b))
                    phis.append(phi)
                    expected.append(mujoco_reference(a, b, phi, SUBSTEP))
        kappa, beta, _rho = authored(pairs, phis)
        expected = np.asarray(expected)
        np.testing.assert_allclose(kappa, expected[:, 0], rtol=2e-5)
        np.testing.assert_allclose(beta, expected[:, 1], rtol=2e-5)

    def test_law_matches_mjwarp_on_nist(self):
        """The resting stiffness and damping MJWarp's NIST rows held (a stage-1 dump of the demo, 8 worlds, the trained
        policy, steps 60 to 1,000): ``K = kappa / (g w)`` [N/m] and ``C = beta / (g w)`` [N s/m] at MJWarp's own
        ``body_invweight0`` sum ``w`` and pyramid factor ``g = mu^2 (1 + mu^2) / 2``, the smallest at ``imp = 0.9``
        and the largest at ``imp = 0.95`` (deeper than 1 mm)."""
        # (pair, gap, w [1/kg], mu, K, C) from MJWarp's rows.
        rows = (
            ((PART, GROUND), 0.0, 200.00003, 1.0, 354.5752402416547, 8.42115151729455),
            ((PART, GROUND), -2.0e-3, 20.0, 1.0, 7901.233364361962, 177.77774304520327),
            ((PART, PART), 0.0, 200.00204, 0.75, 16338.608668265257, 86.23153852977404),
            ((PART, PART), -2.0e-3, 20.00017547607422, 0.75, 364085.62394359394, None),
        )
        kappa, beta, _rho = authored([row[0] for row in rows], [row[1] for row in rows])
        for (_pair, _phi, w, mu, stiffness, damping), k, b in zip(rows, kappa, beta, strict=True):
            g = mu * mu * (1.0 + mu * mu) / 2.0
            self.assertAlmostEqual(k / (g * w) / stiffness, 1.0, delta=2e-4)
            if damping is not None:
                self.assertAlmostEqual(b / (g * w) / damping, 1.0, delta=2e-4)

    def test_relative_compliance_on_nist(self):
        """The row's relative compliance ``rho = 1 / (h (h kappa + beta))`` at NIST's substep: 0.04 a seated part (the
        refsafe floor, deeper than 1 mm), 0.45 a part touching the ground."""
        _kappa, _beta, rho = authored([(PART, PART), (PART, GROUND)], [-2.0e-3, 0.0])
        self.assertAlmostEqual(rho[0], 0.04, delta=5e-4)
        self.assertAlmostEqual(rho[1], 0.451, delta=2e-3)


if __name__ == "__main__":
    unittest.main()
