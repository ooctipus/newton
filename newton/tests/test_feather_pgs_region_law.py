# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The solver's region law against its float64 reference.

The region visit (``region_laws.region_visit``) and the closed bodies' sweeps
(``region_sweep.sweep_closed_bodies``) run in float32 on synthetic free
bodies: random inertias, regions in random frames at random arms, one to
thirteen candidates a region, friction units of no, two or three rows, rigid,
compliant and mixed candidates. The reference
(:mod:`newton.tests.feather_pgs_region_reference`) solves the same regions as
whitened rows in float64 with the same sweeps; the two must leave every
candidate's normal speed the same to float32's rounding of the solve.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import region_laws, region_sweep
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_region_reference as ref

RIGID, COMPLIANT, MIXED = 0, 1, 2


def tangent_basis(n):
    """The tangent pair of ``contact_laws.contact_tangent_basis``."""
    t0 = np.cross(n, [1.0, 0.0, 0.0])
    if t0 @ t0 < 1.0e-12:
        t0 = np.cross(n, [0.0, 1.0, 0.0])
    t0 /= np.linalg.norm(t0)
    t1 = np.cross(n, t0)
    return t0, t1 / np.linalg.norm(t1)


@wp.kernel(enable_backward=False)
def _region_reach(
    cand: wp.array[wp.vec4], spans: wp.array[wp.vec2i], arms: wp.array[wp.vec3], reach: wp.array[wp.vec2]
):
    """Each region's idle reach (:func:`.region_laws.idle_reach`), as the listing's fill forms it (no second body)."""
    g = wp.tid()
    span = spans[g]
    reach[g] = region_laws.idle_reach(cand, span[0], span[1], arms[g], wp.vec3(0.0), False)


@wp.kernel(enable_backward=False)
def _region_terms(
    body_inertia: wp.array2d[wp.mat33],
    region_body: wp.array[int],
    frames: wp.array[wp.mat33],
    arms: wp.array[wp.vec3],
    laws: wp.array[wp.vec2],
    kinds: wp.array[int],
    cfm: float,
    wnn: wp.array[wp.mat33],
    wnf: wp.array[wp.mat33],
    wff: wp.array[wp.mat33],
    ball_values: wp.array[wp.vec3],
    ball_vectors: wp.array[wp.mat33],
    ginv: wp.array[wp.mat33],
    stick: wp.array[wp.mat33],
    stick_factor: wp.array[wp.mat33],
    free_factor: wp.array[wp.mat33],
):
    """Each region's step terms (:func:`.region_sweep.region_terms`), as the listing's fill forms them."""
    g = wp.tid()
    frame = frames[g]
    law = laws[g]
    Wnn, C, G, values, vectors, Ginv, K, Lk, Lw = region_sweep.region_terms(
        body_inertia,
        region_body[g],
        -1,
        arms[g],
        wp.vec3(0.0),
        frame[0],
        frame[1],
        frame[2],
        law[0],
        law[1],
        kinds[g] & region_sweep.FRICTION_ROWS,
        cfm,
    )
    ginv[g] = Ginv
    stick[g] = K
    stick_factor[g] = Lk
    free_factor[g] = Lw
    wnn[g] = Wnn
    wnf[g] = C
    wff[g] = G
    ball_values[g] = values
    ball_vectors[g] = vectors


class Bodies:
    """Synthetic free bodies with their regions: the kernel's stream and the reference's regions."""

    def __init__(self, count: int, compliance: int, seed: int, light: bool = False):
        """``light``: parts of 5 to 50 g with gyration radii of 2 to 6 mm and their regions' candidates within a
        patch of 0.1 to 0.6 mm (a screw or a nut on a hardware pile), where the region's tilt response is 1e4 to
        1e6 times its normal one."""
        rng = np.random.default_rng(seed)
        self.count = count
        self.spans, self.frames, self.arms, self.cands, self.cand_spans = [], [], [], [], []
        self.laws, self.bfs, self.kinds, self.inertia, self.bases, self.regions = [], [], [], [], [], []
        for _ in range(count):
            mass = rng.uniform(0.005, 0.05) if light else rng.uniform(0.02, 0.5)
            q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
            gyration = (4.0e-6, 3.6e-5) if light else (1.0e-4, 9.0e-4)
            inertia = q @ np.diag(mass * rng.uniform(*gyration, size=3)) @ q.T
            A = np.zeros((6, 6))
            A[:3, :3] = np.eye(3) / mass
            A[3:, 3:] = np.linalg.inv(inertia)
            basis = np.linalg.cholesky(A)
            self.inertia.append([A[:3, :3], A[:3, 3:], A[3:, 3:]])
            self.bases.append(basis)
            first = len(self.frames)
            regions = []
            predicted = np.concatenate([rng.normal(size=3) * 0.02 + [0.0, 0.0, -0.0125], rng.normal(size=3) * 0.2])
            if light:
                predicted = np.concatenate([rng.normal(size=3) * 0.3 + [0.0, 0.0, -1.0], rng.normal(size=3) * 30.0])
            for _ in range(rng.integers(1, 9)):
                n = rng.normal(size=3)
                n[2] = abs(n[2]) + 0.5
                n /= np.linalg.norm(n)
                t0, t1 = tangent_basis(n)
                arm = rng.normal(size=3) * (0.004 if light else 0.02)
                size = int(rng.choice([2, 3, 4, 4, 5, 6] if light else [1, 1, 2, 3, 4, 4, 6, 9, 13]))
                patch = rng.uniform(1.0e-4, 6.0e-4) if light else 0.01
                points = rng.uniform(-patch, patch, size=(size, 2)) * (size > 1)
                a = np.ones((size, 3))
                a[:, 1] = [np.cross(x * t0 + y * t1, n) @ t0 for x, y in points]
                a[:, 2] = [np.cross(x * t0 + y * t1, n) @ t1 for x, y in points]
                normal = np.stack([np.r_[n, np.cross(arm, n)], np.r_[0, 0, 0, t0], np.r_[0, 0, 0, t1]])
                friction = np.stack([np.r_[t0, np.cross(arm, t0)], np.r_[t1, np.cross(arm, t1)], np.r_[0, 0, 0, n]])
                gap = rng.uniform(-2.0e-4, 1.0e-5, size=size) if light else rng.uniform(-2.0e-5, 6.0e-5, size=size)
                rhs = (a @ normal) @ predicted + np.where(gap < 0.0, 0.05 * gap / 0.00125, gap / 0.00125)
                eps = np.zeros(size)
                if compliance == COMPLIANT or (compliance == MIXED and rng.random() < 0.5):
                    share = 1.0 if compliance == COMPLIANT else 0.6
                    eps = np.where(rng.random(size) < share, rng.uniform(0.5, 9.0, size=size), 0.0)
                unit_rows = int(rng.choice([0, 2, 3, 3])) if size > 1 else int(rng.choice([0, 2, 2]))
                mu, mu_t = rng.uniform(0.2, 1.0), (rng.uniform(0.002, 0.01) if unit_rows == 3 else 0.0)
                bf = np.zeros(3)
                bf[:unit_rows] = (friction @ predicted)[:unit_rows] + rng.normal(size=unit_rows) * 1.0e-3
                unit = None
                if unit_rows:
                    coefficients = np.array([mu, mu, mu_t])[:unit_rows]
                    unit = ref.FrictionUnit((friction @ basis)[:unit_rows], bf[:unit_rows].copy(), coefficients)
                regions.append(ref.Region((a @ normal) @ basis, rhs.copy(), eps.copy(), unit))
                self.frames.append(np.stack([n, t0, t1]))
                self.arms.append(arm)
                start = len(self.cands)
                self.cands += [[a[i, 1], a[i, 2], rhs[i], eps[i]] for i in range(size)]
                self.cand_spans.append([start, len(self.cands)])
                self.laws.append([mu, mu_t])
                self.bfs.append(bf)
                self.kinds.append(unit_rows | (int(region_sweep.KIND_SPRINGS) if (eps > 0.0).any() else 0))
            self.spans.append([first, len(self.frames)])
            self.regions.append(regions)

    def sweep(self, sweeps: int, device) -> tuple[np.ndarray, np.ndarray]:
        """The kernel's sweeps from rest: each body's change in the reference's coordinates, and the candidates'
        impulses."""

        def array(values, dtype):
            kind = np.int32 if dtype in (int, wp.vec2i, wp.vec3i) else np.float32
            return wp.array(np.ascontiguousarray(np.asarray(values, dtype=kind)), dtype=dtype, device=device)

        bodies, regions, candidates = self.count, len(self.frames), len(self.cands)
        impulses = wp.zeros(candidates + 3 * regions, dtype=float, device=device)
        change = wp.zeros(bodies, dtype=wp.spatial_vector, device=device)
        inertia = wp.array(np.asarray(self.inertia, dtype=np.float32), dtype=wp.mat33, device=device)
        frames, arms, laws = array(self.frames, wp.mat33), array(self.arms, wp.vec3), array(self.laws, wp.vec2)
        region_body = np.concatenate([np.full(last - first, b) for b, (first, last) in enumerate(self.spans)])
        terms = [
            wp.zeros(regions, dtype=dtype, device=device)
            for dtype in (wp.mat33, wp.mat33, wp.mat33, wp.vec3, wp.mat33) + (wp.mat33,) * 4
        ]
        wp.launch(
            _region_terms,
            dim=regions,
            inputs=[inertia, array(region_body, int), frames, arms, laws, array(self.kinds, int), 0.0],
            outputs=terms,
            device=device,
        )
        cands, cand_spans = array(self.cands, wp.vec4), array(self.cand_spans, wp.vec2i)
        reach = wp.zeros(regions, dtype=wp.vec2, device=device)
        wp.launch(_region_reach, dim=regions, inputs=[cands, cand_spans, arms], outputs=[reach], device=device)
        wp.launch(
            region_sweep.sweep_closed_bodies,
            dim=bodies,
            inputs=[
                bodies,
                array([0, 0, bodies, 0], int),
                array(np.arange(bodies), int),
                array(self.spans, wp.vec2i),
                inertia,
                wp.array(np.asarray(self.bases, dtype=np.float32), dtype=wp.spatial_vector, device=device),
                frames,
                arms,
                cand_spans,
                laws,
                array(self.bfs, wp.vec3),
                array(self.kinds, int),
                array(candidates + 3 * np.arange(regions), int),
                *terms,
                cands,
                wp.zeros(candidates, dtype=float, device=device),
                array(np.arange(candidates), int),
                sweeps,
                0.0,
            ],
            outputs=[
                wp.zeros(regions, dtype=wp.vec3, device=device),
                wp.zeros(regions, dtype=wp.vec3, device=device),
                array(-np.ones((regions, 3)), wp.vec3i),
                wp.zeros(regions, dtype=wp.vec3, device=device),
                array(np.full(regions, int(region_sweep.REGION_IDLE)), int),
                reach,
                impulses,
                change,
            ],
            device=device,
        )
        return change.numpy().reshape(bodies, 6).astype(np.float64), impulses.numpy().astype(np.float64)


class TestRegionLaw(unittest.TestCase):
    """The solver's region visit returns the reference's solve."""

    SWEEPS = 8

    def check(self, compliance: int, bodies: int, median: float, largest: float, light: bool = False):
        problem = Bodies(bodies, compliance, seed=compliance + (7 if light else 0), light=light)
        change, impulses = problem.sweep(self.SWEEPS, wp.get_device("cpu"))
        errors, loads = [], []
        for b, regions in enumerate(problem.regions):
            velocity, states, _log = ref.solve(regions, 6, self.SWEEPS)
            rows = np.vstack([region.rows for region in regions])
            errors.append(float(np.abs(rows @ (change[b] - velocity)).max()))
            spans = problem.cand_spans[problem.spans[b][0] : problem.spans[b][1]]
            for (first, last), state in zip(spans, states, strict=True):
                loads.append(abs(impulses[first:last].sum() - state.lam.sum()) / max(state.lam.sum(), 1.0e-3))
        self.assertLess(float(np.median(errors)), median)
        self.assertLess(max(errors), largest)
        # A region's load is unique where its candidates' impulses are not.
        self.assertLess(float(np.percentile(loads, 99)), 1.0e-2)

    def test_rigid_candidates(self):
        """A dual active set of at most three rows: 3e-9 m/s at the median of a body's candidates."""
        self.check(RIGID, 200, 2.0e-8, 2.0e-5)

    def test_compliant_candidates(self):
        """Newton on the region's three coordinates, whole steps."""
        self.check(COMPLIANT, 200, 2.0e-7, 2.0e-4)

    def test_mixed_candidates(self):
        """The rigid set solved in the closing springs' metric."""
        self.check(MIXED, 120, 5.0e-8, 2.0e-5)

    def test_light_parts_with_small_patches(self):
        """The rank rule and the set's multipliers from the set's rows orthonormal in the block's metric: candidates
        a fraction of a millimetre apart on a part whose tilt response is 1e4 to 1e6 times its normal one keep the
        reference's solve. In the normal equations (``A K A^T`` formed and inverted) the rule's measure is a
        difference of nearly equal Gram entries, and float32 read 5e-4 m/s off the reference here (on hardware's
        pile, a near-dependent third row entered and threw a 13 g part to 200 m/s)."""
        self.check(RIGID, 300, 2.0e-6, 1.0e-4, light=True)

    def test_the_laws_render_for_native_sweeps(self):
        """Every law of the family has a native form (the general sweep's region phase calls them)."""
        source = region_laws.REGION_LAW.prelude().source
        for law in ("region_visit", "region_block", "frame_twist", "frame_wrench", "frame_response"):
            self.assertIn(f" {law}(", source)


class TestRegionStream(unittest.TestCase):
    """The stream's bodies and their marks through the solver."""

    def test_a_held_frame_keeps_the_listing_a_new_one_would_make(self):
        """List a contact frame's regions once and hold them through its later steps: a held step's listing (its
        bodies' places and marks, each region's candidates by contact, the order and its levels) is the one a new
        listing of that step's rows makes.

        A stack of three boxes and a tilted pair of boxes on the ground, colliding once every four steps: the steps
        between hold the frame (the patch builder carries its regions, every candidate is admitted again and no
        mark changes).
        """
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for k in range(3):
            body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.05 + 0.0999 * k), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        tilt = wp.quat_from_axis_angle(wp.vec3(1.0, 0.3, 0.0), 0.2)
        for k in range(2):
            body = builder.add_body(xform=wp.transform((0.4, 0.1 * k, 0.04 + 0.075 * k), tilt))
            builder.add_shape_box(body, hx=0.04, hy=0.04, hz=0.04)
        model = builder.finalize(device="cpu")
        solver = SolverFeatherPGS(model, pgs_iterations=8)
        regions = solver.sweep.regions
        self.assertIsNotNone(regions)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state, out, control = model.state(), model.state(), model.control()
        dt = 1.0 / 480.0
        names = (
            "cursor", "body_open", "body_regions", "body_cand", "world_regions", "world_levels", "lanes", "reg_cand",
            "reg_kind", "reg_order", "reg_level", "cand_contact", "level_end", "level_region",
        )  # fmt: skip

        def listing():
            count = int(regions.cursor.numpy()[0])
            candidates = int(regions.reg_cand.numpy()[:count, 1].max()) if count else 0
            sizes = {"reg_cand": count, "reg_kind": count, "reg_order": count, "reg_level": count}
            sizes.update({"cand_contact": candidates, "level_end": count, "level_region": count})
            return {name: getattr(regions, name).numpy()[: sizes.get(name, None)].copy() for name in names}

        held = 0
        for step in range(12):
            if step % 4 == 0:
                pipeline.collide(state, contacts)
            solver.step(state, out, control, contacts, dt)
            state, out = out, state
            if regions.relist.numpy()[0] != 0:
                continue
            held += 1
            kept = listing()
            # The same step's rows listed anew.
            regions.relist.zero_()
            regions.list(solver, out, contacts, dt)
            self.assertEqual(int(regions.relist.numpy()[0]), 1)
            made = listing()
            for name in names:
                np.testing.assert_array_equal(kept[name], made[name], err_msg=f"step {step}: {name}")
        self.assertGreaterEqual(held, 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
