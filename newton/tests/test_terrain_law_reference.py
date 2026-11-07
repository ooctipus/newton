# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the float64 terrain law (the oracle of the float32 kernels) against brute-force geometry."""

import unittest

import numpy as np

from newton.tests import terrain_law_reference as ref


def _closest_on_triangles(p, a, b, c):
    """Closest points of p on triangles (a, b, c), vectorized over the triangles (Ericson 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        den = np.where(va + vb + vc == 0.0, 1.0, va + vb + vc)
        out = a + ab * (vb / den)[:, None] + ac * (vc / den)[:, None]
        m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        t = (d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1.0, (d4 - d3) + (d5 - d6))
        out = np.where(m[:, None], b + t[:, None] * (c - b), out)
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        t = d2 / np.where(d2 - d6 == 0, 1.0, d2 - d6)
        out = np.where(m[:, None], a + t[:, None] * ac, out)
        out = np.where(((d6 >= 0) & (d5 <= d6))[:, None], c, out)
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        t = d1 / np.where(d1 - d3 == 0, 1.0, d1 - d3)
        out = np.where(m[:, None], a + t[:, None] * ab, out)
        out = np.where(((d3 >= 0) & (d4 <= d3))[:, None], b, out)
        out = np.where(((d1 <= 0) & (d2 <= 0))[:, None], a, out)
    return out


def _triangles(hf, lo, hi):
    box = hf.cells_in_box(lo, hi)
    tris = np.array(hf.window_features(box)[0])
    verts = np.array([hf.tri_verts(t) for t in tris])
    return tris, verts


def _local_minima(hf, p, reach):
    """Exact local minima of the distance from p to the surface: a triangle's closest point that is the closest point
    of every triangle containing it."""
    tris, v = _triangles(hf, p[:2] - reach - 0.05, p[:2] + reach + 0.05)
    q = _closest_on_triangles(np.broadcast_to(p, v[:, 0].shape), v[:, 0], v[:, 1], v[:, 2])
    d = np.linalg.norm(q - p, axis=1)
    out = []
    used = np.zeros(len(tris), bool)
    for i in np.argsort(d):
        if used[i] or d[i] > reach:
            continue
        same = np.linalg.norm(q - q[i], axis=1) <= 1.0e-9
        used |= same
        cl = _closest_on_triangles(np.broadcast_to(q[i], v[:, 0].shape), v[:, 0], v[:, 1], v[:, 2])
        contains = np.linalg.norm(cl - q[i], axis=1) <= 1.0e-9
        if np.all(same[contains]):
            out.append((q[i], d[i]))
    return out


def _rough(rng, n=33, half=1.6):
    Z = np.round(rng.uniform(0.0, 0.08, size=(n, n)) / 0.005) * 0.005
    e = ((Z - Z.min()) / (Z.max() - Z.min())).astype(np.float32)
    return ref.ReferenceHeightfield(ref.decode_heights(e, n, n, float(Z.min()), float(Z.max())), half, half)


class TestTerrainLawReference(unittest.TestCase):
    def test_point_rows_are_the_exact_local_minima(self):
        """Emit one sphere row per exact local minimum of the centre's surface distance within reach."""
        rng = np.random.default_rng(12)
        hf = _rough(rng)
        checked = 0
        for _ in range(300):
            x, y = rng.uniform(-1.2, 1.2, size=2)
            if rng.uniform() < 0.4:
                x = hf.vx(round((x + hf.hx) / hf.dx))
            if rng.uniform() < 0.4:
                y = hf.vy(round((y + hf.hy) / hf.dy))
            radius = 0.03
            p = np.array([x, y, hf.height(x, y) + radius + rng.uniform(0.002, 0.02)])
            reach = radius + 0.04
            rows = ref.pair_rows(hf, ref.Sphere(p, radius), reach)
            minima = _local_minima(hf, p, reach)
            self.assertEqual(len(rows), len(minima), msg=f"{p}: {rows} vs {minima}")
            # A cluster's plane lies within 1 um of its triangles.
            for q, d in minima:
                self.assertTrue(
                    any(np.linalg.norm(r.q - q) <= 2.0 * ref.FLAT_EPS and abs(r.d - d) <= ref.FLAT_EPS for r in rows),
                    msg=f"{p}: {q}",
                )
            checked += len(minima)
        self.assertGreater(checked, 300)

    def test_point_rows_point_up_on_steep_terrain(self):
        """Give spheres over steep terrain no row whose normal points down (none at a valley edge from outside).

        Every normal of a heightfield's rows lies in a feature's cone spanned by its faces' upward normals, or, for a
        core inside the solid, points from it to its closest surface point across a valley or a face. A sphere
        outside the solid but below a valley's line, beside the valley over lower ground, lies beyond both of its
        faces; the valley's edge is no support there (its normal would leave the faces' cone and push the sphere down).
        """
        rng = np.random.default_rng(5)
        Z = np.round(rng.uniform(0.0, 0.25, (33, 33)) / 0.01) * 0.01
        e = ((Z - Z.min()) / (Z.max() - Z.min())).astype(np.float32)
        hf = ref.ReferenceHeightfield(ref.decode_heights(e, 33, 33, float(Z.min()), float(Z.max())), 1.6, 1.6)
        rows = 0
        for _ in range(1000):
            x, y = rng.uniform(-1.3, 1.3, 2)
            p = np.array([x, y, hf.height(x, y) + rng.uniform(-0.05, 0.08)])
            for row in ref.pair_rows(hf, ref.Sphere(p, 0.03), 0.07):
                self.assertGreater(row.n[2], 0.0, msg=f"{p}: {row}")
                rows += 1
        self.assertGreater(rows, 1000)

    def test_solid_rows_on_a_plane_are_the_plane_manifold(self):
        """On a flat heightfield, emit exactly the analytic plane manifold of every core."""
        rng = np.random.default_rng(13)
        hf = ref.ReferenceHeightfield(np.zeros((33, 33)), 1.6, 1.6)
        verts = np.array([(x, y, z) for x in (-0.05, 0.05) for y in (-0.04, 0.04) for z in (-0.02, 0.02)])
        verts = np.concatenate([verts, [[0.0, 0.0, -0.03], [0.06, 0.0, 0.0]]])
        shapes = (
            ("box", {"half": (0.08, 0.06, 0.04)}, None),
            ("cylinder", {"radius": 0.05, "half_height": 0.04}, None),
            ("capsule", {"radius": 0.03, "half_height": 0.08}, None),
            ("hull", {}, verts),
        )
        for kind, dims, mesh in shapes:
            for _ in range(60):
                q = rng.normal(size=4)
                q /= np.linalg.norm(q)
                core = ref.make_core(kind, dims, np.array([*rng.uniform(-1.0, 1.0, size=2), 0.0, *q]), verts=mesh)
                lo, _hi = core.aabb()
                core = ref.make_core(
                    kind,
                    dims,
                    np.array([*core.aabb()[0][:2] * 0 + core.c[:2], -lo[2] + rng.uniform(-0.004, 0.02), *q]),
                    verts=mesh,
                )
                reach = core.r_eff + 0.04
                rows = ref.pair_rows(hf, core, reach)
                expected = core.plane_manifold(np.array([0.0, 0.0, 1.0]), np.zeros(3), reach)
                got = sorted(tuple(np.round(r.k, 9)) for r in rows)
                want = sorted(tuple(np.round(k, 9)) for k, _h in expected)
                self.assertEqual(got, want, msg=kind)

    def test_solid_rows_reach_every_buried_point(self):
        """Never report a box or cylinder farther from the rough surface than any of its boundary points."""
        rng = np.random.default_rng(14)
        hf = _rough(rng)
        checked = 0
        for kind, dims in (
            ("box", {"half": (0.08, 0.06, 0.04)}),
            ("cylinder", {"radius": 0.02, "half_height": 0.0625}),
        ):
            for _ in range(80):
                q = rng.normal(size=4)
                q /= np.linalg.norm(q)
                pose = np.array([*rng.uniform(-1.0, 1.0, size=2), 0.0, *q])
                core = ref.make_core(kind, dims, pose)
                lo, hi = core.aabb()
                top = max(hf.height(x, y) for x in np.linspace(lo[0], hi[0], 7) for y in np.linspace(lo[1], hi[1], 7))
                pose[2] = top - lo[2] + rng.uniform(-0.02, 0.0)
                core = ref.make_core(kind, dims, pose)
                rows = ref.pair_rows(hf, core, 0.04)
                # Signed surface distances of boundary samples: the separation (or deepest point) is no farther.
                samples = self._boundary(core, kind)
                _tris, v = _triangles(hf, samples[:, :2].min(0) - 0.1, samples[:, :2].max(0) + 0.1)
                nearest = np.inf
                for x in samples:
                    q = _closest_on_triangles(np.broadcast_to(x, v[:, 0].shape), v[:, 0], v[:, 1], v[:, 2])
                    dist = float(np.linalg.norm(q - x, axis=1).min())
                    nearest = min(nearest, -dist if x[2] < hf.height(x[0], x[1]) else dist)
                if nearest > 0.04 - 1.0e-4:
                    continue
                checked += 1
                self.assertTrue(rows, msg=f"{kind} {pose}")
                self.assertLessEqual(min(r.d for r in rows), nearest + 2.0 * ref.FLAT_EPS, msg=f"{kind} {pose}")
                for r in rows:
                    self.assertAlmostEqual(float(r.n @ (r.k - r.q)), r.d, delta=1.0e-9)
        self.assertGreater(checked, 80)

    @staticmethod
    def _boundary(core, kind):
        if kind == "box":
            g = np.linspace(-1.0, 1.0, 9)
            pts = []
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    for u in g:
                        for v in g:
                            loc = np.zeros(3)
                            loc[axis] = sign
                            loc[(axis + 1) % 3] = u
                            loc[(axis + 2) % 3] = v
                            pts.append(core.c + core.R @ (loc * core.half))
            return np.array(pts)
        ax = core.axis
        ref_dir = np.array([1.0, 0.0, 0.0]) if abs(ax[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = ref_dir - ax * (ax @ ref_dir)
        u /= np.linalg.norm(u)
        w = np.cross(ax, u)
        th = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        ring = np.outer(np.cos(th), u) + np.outer(np.sin(th), w)
        pts = [core.c + z * ax + core.r * ring for z in np.linspace(-core.h, core.h, 9)]
        pts += [core.c + s * core.h * ax + rho * ring for s in (-1.0, 1.0) for rho in (0.0, 0.5 * core.r)]
        return np.concatenate(pts)

    def test_cluster_planes_hold_their_triangles(self):
        """Keep every vertex of a cluster of several triangles within 1 um of the cluster plane."""
        rng = np.random.default_rng(15)
        Z = np.round(rng.uniform(0.0, 0.05, size=(41, 41)) / 0.01) * 0.01
        Z[5:20, 5:30] = 0.2 + 1.0e-7 * rng.integers(-2, 3, size=(15, 25))
        Z[25:40, :] = np.tan(np.radians(12.0)) * 0.1 * np.arange(41)[None, :] + 1.0
        e = ((Z - Z.min()) / (Z.max() - Z.min())).astype(np.float32)
        hf = ref.ReferenceHeightfield(ref.decode_heights(e, 41, 41, float(Z.min()), float(Z.max())), 2.0, 2.0)
        self.assertGreater(len(hf.records), 5)
        for root in hf.records:
            n, a = hf.plane(root)
            for t in np.flatnonzero(hf.cluster == root):
                for v in hf.tri_verts(t):
                    self.assertLessEqual(abs(float(n @ (v - a))), ref.FLAT_EPS * (1.0 + 1.0e-6))
        # The flat patch and the slope are one cluster each.
        self.assertEqual(
            len({hf.cluster[hf.tri_id(r, c, s)] for r in range(6, 18) for c in range(6, 28) for s in (0, 1)}), 1
        )
        self.assertEqual(
            len({hf.cluster[hf.tri_id(r, c, s)] for r in range(26, 39) for c in range(1, 39) for s in (0, 1)}), 1
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
