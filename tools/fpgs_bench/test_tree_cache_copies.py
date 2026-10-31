# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reuse exact copied state without hiding changed coordinates or reset epochs."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.held_tree import HeldTree
from tools.fpgs_bench import test_mixed_tree_steps as mixed_steps
from tools.fpgs_bench.public_state_oracle import public_state
from tools.fpgs_bench.test_mixed_tree_cache import DT, make_solver, model_fixture


def generations(owner):
    """Read construction counts only outside captured/timed device work."""
    return [group.data.generation.numpy().copy() for group in owner.groups]


def copy_state(model, state):
    result = model.state()
    result.assign(state)
    return result


class TestTreeCacheCopies(unittest.TestCase):
    def check_copied_state(self, device):
        model = model_fixture(device, reordered=True)
        solver = make_solver(model, permuted=True)
        owner = HeldTree(solver)
        state, output = model.state(), model.state()
        state.joint_qd.assign(np.linspace(-0.2, 0.3, model.joint_dof_count, dtype=np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        owner.begin(state, solver.factor, state.joint_qd, DT, True)
        solver.v_out.assign(state.joint_qd)
        owner.finish(state, solver.factor, output, DT, True)
        bank = copy_state(model, output)
        before = generations(owner)
        geometry = [group.geometric.numpy().copy() for group in owner.groups]
        # The caches were published into ``output``; its exact copy keeps them.
        owner.begin(bank, solver.factor, bank.joint_qd, DT, True, state_out=output)
        for group, previous, mass in zip(owner.groups, before, geometry, strict=True):
            np.testing.assert_array_equal(group.data.generation.numpy(), previous)
            np.testing.assert_array_equal(group.geometric.numpy(), mass)
        public_state(self, model, bank)

        # Generalized coordinates alone do not prove that public body state was copied.
        for field_name in ("body_q", "body_qd"):
            partial = copy_state(model, bank)
            target = owner.groups[0]
            art = target.sparse.plan.group_to_art.numpy()[0]
            joint = model.articulation_start.numpy()[art]
            body = model.joint_child.numpy()[joint]
            field = getattr(partial, field_name)
            values = field.numpy().copy()
            values[body, 0] += np.float32(0.03)
            field.assign(values)
            before = generations(owner)
            owner.begin(partial, solver.factor, partial.joint_qd, DT, True, state_out=bank)
            for group, previous in zip(owner.groups, before, strict=True):
                expected = previous.copy()
                if group is target:
                    expected[0] += 1
                np.testing.assert_array_equal(group.data.generation.numpy(), expected)
            public_state(self, model, partial)
            bank = partial

        # A changed value in a different bank repairs only its owning articulation.
        for changed_velocity in (False, True):
            changed = copy_state(model, bank)
            target = owner.groups[int(changed_velocity)]
            group_index = 1
            art = target.sparse.plan.group_to_art.numpy()[group_index]
            field = changed.joint_qd if changed_velocity else changed.joint_q
            values = field.numpy().copy()
            index = (
                solver.forest.articulation_dof_start.numpy()[art]
                if changed_velocity
                else target.plan.q_index.numpy().reshape(target.group_count, -1)[group_index, 0]
            )
            values[index] += np.float32(0.025)
            field.assign(values)
            before = generations(owner)
            owner.begin(changed, solver.factor, changed.joint_qd, DT, True, state_out=bank)
            for group, previous in zip(owner.groups, before, strict=True):
                expected = previous.copy()
                if group is target:
                    expected[group_index] += 1
                np.testing.assert_array_equal(group.data.generation.numpy(), expected)
            public_state(self, model, changed)
            bank = changed

        mask = wp.array([False, True, False], dtype=wp.bool, device=device)
        owner.invalidate(mask)
        before = generations(owner)
        owner.begin(bank, solver.factor, bank.joint_qd, DT, True)
        for group, previous in zip(owner.groups, before, strict=True):
            arts = group.sparse.plan.group_to_art.numpy()
            worlds = solver.forest.art_to_world.numpy()[arts]
            np.testing.assert_array_equal(group.data.generation.numpy(), previous + (worlds == 1))

        # Publication may omit next mass terms; an exact copy cannot manufacture them.
        solver.v_out.assign(bank.joint_qd)
        owner.finish(bank, solver.factor, output, DT, False)
        bank = copy_state(model, output)
        target = owner.groups[0]
        requests = np.zeros(model.articulation_count, np.int32)
        requests[target.sparse.plan.group_to_art.numpy()[0]] = 1
        solver.factor.mass_update_requested.assign(requests)
        before = generations(owner)
        owner.begin(bank, solver.factor, bank.joint_qd, DT, False, state_out=output)
        for group, previous in zip(owner.groups, before, strict=True):
            expected = previous.copy()
            if group is target:
                expected[0] += 1
            np.testing.assert_array_equal(group.data.generation.numpy(), expected)
            np.testing.assert_array_equal(group.status.numpy(), 0)
        if model.device.is_cuda:
            wp.synchronize()  # Complete auxiliary streams before fixture arrays are released.

    def test_exact_copy_and_changed_values_cpu(self):
        """Reuse copied coordinates while preserving per-articulation repairs."""
        self.check_copied_state("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_exact_copy_and_changed_values_cuda(self):
        """Exercise cooperative copy checks and exact per-group cache epochs."""
        self.check_copied_state("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_complete_graph_copyback(self):
        """Capture the caller's output-to-input copy after every complete step."""
        cases = [mixed_steps.make_case(enabled, True) for enabled in (False, True)]
        graphs, masks = [], []
        for case in cases:
            model, solver, state, output, control, contacts = case
            for _ in range(4):
                solver.step(state, output, control, contacts, DT)
                state.assign(output)
            mask = wp.array([False] * 3, dtype=wp.bool, device=model.device)
            wp.synchronize()
            with wp.ScopedCapture(device=model.device) as capture:
                solver.reset(state, world_mask=mask)
                for _ in range(4):
                    solver.step(state, output, control, contacts, DT)
                    state.assign(output)
            graphs.append(capture.graph)
            masks.append(mask)
        checker = mixed_steps.TestMixedTreeSteps()
        owner = cases[1][1].factor.held
        for selected in ([False] * 3, [True, False, False], [False, True, True], [False] * 3):
            before = generations(owner)
            for mask, graph in zip(masks, graphs, strict=True):
                mask.assign(np.array(selected))
                wp.capture_launch(graph)
            checker.assert_pair(cases)
            for group, previous in zip(owner.groups, before, strict=True):
                arts = group.sparse.plan.group_to_art.numpy()
                reset = np.array(selected)[cases[1][1].forest.art_to_world.numpy()[arts]]
                np.testing.assert_array_equal(group.data.generation.numpy(), previous + 4 + reset)
        wp.synchronize()  # Keep both graph fixtures alive through all asynchronous work.


if __name__ == "__main__":
    unittest.main(verbosity=2)
