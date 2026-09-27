# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent controls for current-light and late-publication composition."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import world_scan_publication as scan
from tools.fpgs_bench import test_kuka_joint_world as saved
from tools.fpgs_bench import test_world_scan_publication as publication

GENERALIZED = ("v_new", "joint_qdd", "joint_q_new", "joint_qd_new")


def masks(bundle, resolved):
    """Map current world ownership onto all35 physical DOFs and all37 coordinates."""
    worlds = len(resolved)
    dof_world = np.empty(bundle.dimensions[0], np.int32)
    dof_world[bundle.plan.dof_ids.ravel()] = np.repeat(np.arange(worlds), 35)
    joint_world = np.empty(bundle.dimensions[2], np.int32)
    joint_world[bundle.plan.joint_ids.ravel()] = np.repeat(np.arange(worlds), 32)
    coordinate_world = np.repeat(joint_world, np.diff(bundle.values["joint_q_start"].numpy()))
    return resolved[dof_world].astype(bool), resolved[coordinate_world].astype(bool)


def damping(bundle):
    """Use the actual live damping scalar in both original and composed descriptors."""
    bundle.values["angular_damping"] = 0.05
    bundle.data.angular_damping = 0.05


def prepare(snapshot, device, resolved, *, refresh=(0, 0), changed=False):
    """Seed already-owned generalized outputs from the original law, then poison old inputs."""
    candidate = publication.bind(snapshot, device, refresh=refresh, changed=changed)
    reference = publication.bind(snapshot, device, refresh=refresh, changed=changed)
    damping(candidate)
    damping(reference)
    publication.original(reference)
    dofs, coordinates = masks(candidate, resolved)
    for name in GENERALIZED:
        selected = coordinates if name == "joint_q_new" else dofs
        value = candidate.values[name].numpy()
        value[selected] = reference.values[name].numpy()[selected]
        candidate.values[name].assign(value)
    # Only the skipped generalized phase reads these old-state fields. The
    # complete late body publication must consume current q_new/qd_new instead.
    for name, selected in (("joint_q", coordinates), ("joint_qd", dofs)):
        value = candidate.values[name].numpy()
        value[selected] = np.nan
        candidate.values[name].assign(value)
    candidate.resolved = wp.array(resolved, dtype=int, device=device)
    return candidate, reference


def launch(bundle):
    """Launch the actual current-mask variant with unchanged complete late body work."""
    wp.launch_tiled(
        scan.get_resolved_kernel(str(wp.get_device(bundle.device).arch)),
        dim=[len(bundle.plan.body_ids)],
        inputs=[bundle.device_plan, bundle.data, bundle.resolved],
        block_dim=32,
        device=bundle.device,
    )


def seeds(bundle):
    """Save exact selected outputs, read-only fields and held epoch values before publication."""
    return {
        name: value.numpy().copy()
        for name, value in bundle.values.items()
        if hasattr(value, "numpy")
        and (name not in publication.OUTPUTS or name in (*GENERALIZED, "body_I_s", "body_inertia_terms"))
    }


def check(test, candidate, reference, before):
    """Reuse all15 independent physical checks and enforce exact current output ownership."""
    publication.check_outputs(test, candidate, reference)
    dofs, coordinates = masks(candidate, candidate.resolved.numpy())
    for name in GENERALIZED:
        selected = coordinates if name == "joint_q_new" else dofs
        np.testing.assert_array_equal(candidate.values[name].numpy()[selected], before[name][selected], err_msg=name)
    for name in before.keys() - publication.OUTPUTS.keys():
        np.testing.assert_array_equal(candidate.values[name].numpy(), before[name], err_msg=name)
    if not candidate.data.materialize_all_body_inertia:
        arts = candidate.values["body_to_articulation"].numpy()
        nonfree = candidate.values["is_free_rigid"].numpy()[arts] == 0
        np.testing.assert_array_equal(candidate.values["body_I_s"].numpy()[nonfree], before["body_I_s"][nonfree])
    if not candidate.data.materialize_body_inertia_terms:
        np.testing.assert_array_equal(candidate.values["body_inertia_terms"].numpy(), before["body_inertia_terms"])
    prescribed = candidate.values["kinematic_dof_mask"].numpy() != 0
    np.testing.assert_array_equal(candidate.values["joint_qdd"].numpy()[prescribed], 0)
    np.testing.assert_array_equal(candidate.values["fk_id_cache_valid"].numpy(), 1)


def actual_light(snapshot, device, *, refresh=(0, 0), fallback=None):
    """Bind the real light outputs directly into late publication; retain active solved velocity."""
    candidate = publication.bind(snapshot, device, refresh=refresh)
    reference = publication.bind(snapshot, device, refresh=refresh)
    damping(candidate)
    damping(reference)
    early = saved.bind(snapshot, device)
    # These are the exact same owners used by the production handoff. Light
    # publishes only selected generalized state, leaving unresolved v_out at
    # the saved original post-solve value for this component control.
    for name in early.integration._cls.vars:
        setattr(early.integration, name, candidate.values[name])
    early.output.v_out = candidate.values["v_new"]
    early.predictor.qdd = candidate.values["joint_qdd"]
    early.data.q = candidate.values["joint_q"]
    early.data.dt = candidate.data.dt
    if fallback == "csr":
        early.buckets.data.invalid.fill_(1)
        early.buckets.data.offsets.fill_(-1000)
    elif fallback == "inverse":
        inverse = early.predictor.inverse23.numpy()
        inverse[early.plan.primary_group[0], 0, 0] = np.nan
        early.predictor.inverse23.assign(inverse)
    saved.launch(early)
    candidate.resolved = early.output.resolved
    reference.values["v_new"].assign(candidate.values["v_new"].numpy())
    publication.original(reference)
    return candidate, reference, early


class TestWorldScanComposition(unittest.TestCase):
    def test_current_resolved_api(self):
        """Require a separate current-resolved variant without replacing standalone publication."""
        self.assertTrue(callable(scan.get_resolved_kernel))
        self.assertTrue(callable(scan.get_kernel))

    def test_saved_all_none_mixed_and_next_epochs(self):
        """Match all15 physical outputs for every current mask and all held-cache publication modes."""
        for index, snapshot in enumerate(saved.snapshots()):
            for mode in ("none", "mixed", "all"):
                resolved = np.zeros(512, np.int32)
                if mode == "all":
                    resolved[:] = 1
                elif mode == "mixed":
                    resolved[::3] = 1
                with self.subTest(snapshot=index, mode=mode):
                    refresh = ((0, 0), (0, 1), (1, 0))[index % 3]
                    candidate, reference = prepare(snapshot, "cpu", resolved, refresh=refresh, changed=index == 3)
                    before = seeds(candidate)
                    launch(candidate)
                    check(self, candidate, reference, before)

    def test_poison_control_detects_generalized_rerun(self):
        """Demonstrate that rerunning the original generalized phase corrupts poisoned selected state."""
        iterator = saved.snapshots()
        candidate, _ = prepare(next(iterator), "cpu", np.ones(512, np.int32))
        publication.launch(candidate)
        self.assertFalse(np.isfinite(candidate.values["joint_q_new"].numpy()).all())

    def check_light(self, snapshot, device, refresh=(0, 0), fallback=None):
        """Check current light decisions, actual generalized seeds and complete late publication."""
        candidate, reference, early = actual_light(snapshot, device, refresh=refresh, fallback=fallback)
        resolved = candidate.resolved.numpy()
        if fallback == "csr":
            np.testing.assert_array_equal(resolved, 0)
            np.testing.assert_array_equal(early.output.predictor_fallback.numpy(), 1)
        else:
            self.assertGreater(np.count_nonzero(resolved), 0)
            self.assertLess(np.count_nonzero(resolved), len(resolved))
            original = saved.original_selection(early)
            self.assertTrue(np.all(original.resolved.numpy()[resolved != 0] == 1))
            if fallback == "inverse":
                self.assertEqual(resolved[0], 0)
                self.assertEqual(early.output.predictor_fallback.numpy()[0], 1)
        active = early.output.active_worlds.numpy()[: early.output.active_count.numpy()[0]]
        np.testing.assert_array_equal(np.sort(active), np.flatnonzero(resolved == 0))
        before = seeds(candidate)
        launch(candidate)
        check(self, candidate, reference, before)

    def test_actual_light_all_four_and_invalid_csr(self):
        """Consume actual light output from both devices and both epochs, including invalid-CSR fallback."""
        for index, snapshot in enumerate(saved.snapshots()):
            with self.subTest(snapshot=index):
                self.check_light(snapshot, "cpu", refresh=(0, index % 2))
                if index == 0:
                    self.check_light(snapshot, "cpu", fallback="csr")
                    self.check_light(snapshot, "cpu", fallback="inverse")

    def check_transitions(self, device, graph):
        """Reset two persistent buffers and current masks without changing bound graph scalars."""
        iterator = saved.snapshots()
        snapshot = next(iterator)
        buffers = []
        for epoch in (0, 1):
            candidate, _ = prepare(snapshot, device, np.zeros(512, np.int32), refresh=(0, epoch))
            launch(candidate)
            captured = None
            if graph:
                with wp.ScopedCapture(device=device) as captured:
                    launch(candidate)
            buffers.append((candidate, captured))
        for index, mode in enumerate(("all", "mixed", "none", "mixed")):
            for epoch, (candidate, captured) in enumerate(buffers):
                resolved = np.zeros(512, np.int32)
                if mode == "all":
                    resolved[:] = 1
                elif mode == "mixed":
                    resolved[(index + epoch) % 2 :: 2] = 1
                fresh, reference = prepare(snapshot, device, resolved, refresh=(0, epoch), changed=index % 2 == 1)
                for name, value in candidate.values.items():
                    if hasattr(value, "numpy"):
                        wp.copy(value, fresh.values[name])
                wp.copy(candidate.resolved, fresh.resolved)
                before = seeds(candidate)
                if graph:
                    wp.capture_launch(captured.graph)
                else:
                    launch(candidate)
                check(self, candidate, reference, before)

    def test_current_mask_and_numeric_reset_transitions(self):
        """Replace stale selected bytes through both current buffers and both next-refresh epochs."""
        self.check_transitions("cpu", False)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA requires the root GPU lease")
    def test_cuda_actual_light_all_four(self):
        """Check actual native light-to-publication ownership and physical state on all four inputs."""
        for index, snapshot in enumerate(saved.snapshots()):
            with self.subTest(snapshot=index):
                self.check_light(snapshot, "cuda:0", refresh=(0, index % 2))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA requires the root GPU lease")
    def test_cuda_two_graph_buffers_current_mask_transitions(self):
        """Replay two fixed-pointer buffers through selected, active and numeric-reset transitions."""
        self.check_transitions("cuda:0", True)


if __name__ == "__main__":
    unittest.main()
