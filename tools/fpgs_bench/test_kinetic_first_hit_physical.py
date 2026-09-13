# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independently check first-hit physical response and current-row lifetimes."""

import importlib
import inspect
import os
import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import kinetic_first_hit as native

FIXTURES = "/tmp/fpgs-kuka-demand-state-Lv4cAaUh"
PACKAGE = "newton._src.solvers.feather_pgs."
DEVICE = os.environ.get("FPGS_TEST_DEVICE", "cpu")


@contextmanager
def references():
    """Reuse existing saved-input binders, but execute this checkout's natives.

    The original physical-eight and original-current-geometry oracle are not
    modified. Saved J/Y/RHS/solved state remain control-only, never inputs.
    """
    names = (
        "types",
        "predictor",
        "state",
        "rows_types",
        "rows",
        "rows_triplet",
        "solve_types",
        "solve",
        "guard",
        "allocate",
        "services",
        "free_refresh",
        "zero",
        "hybrid",
        "compact_coupled",
        "public_force",
    )
    aliases = {"kinetic_" + name: importlib.import_module(PACKAGE + "kinetic_" + name) for name in names}
    old_path = sys.path[:]
    old_aliases = {name: sys.modules.get(name) for name in aliases}
    try:
        sys.path.insert(0, FIXTURES)
        sys.modules.update(aliases)
        yield importlib.import_module("test_kinetic_current_solve")
    finally:
        sys.path[:] = old_path
        for name, previous in old_aliases.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def prepare(owner):
    """Build current allocation, packets and MF before any response is demanded."""
    owner.setup()
    owner.allocate()
    owner.build_rows()
    owner.prepare_mf()
    owner.qualify()
    owner.materialize()


def build_case(reference, gpu=0, phase=0, device="cpu"):
    """Use actual native current state and held response, not saved coefficients."""
    rows = reference.build_rows(gpu, phase, device)
    rows.kernels = inspect.getclosurevars(rows.launch).nonlocals["kernels"]
    owner = reference.bind_current_solve(rows)
    call = SimpleNamespace(rows=rows, solve=owner)
    if not native.install(call):
        raise AssertionError("Actual original recipe was not admitted")
    prepare(owner)
    return call


def packet_problems(reference, owner):
    """Independently project endpoint wrenches, then apply FP64 held T^T T.

    Also return the untouched original-current-geometry/physical-H oracle.
    Matrix-free uses its current physical inertia, distinct from held free6.
    """
    original = reference.physical_problem(owner, original_geometry=True)
    problems = []
    rows = owner.rows
    plan, held, state, current, output = map(
        reference.host_arrays, (rows.plan, rows.held, rows.state, rows.current, rows.out)
    )
    for original_problem in original:
        p = dict(original_problem)
        w, m = p["world"], len(p["j"])
        t = reference.coordinate_matrix(plan, held, state, w)
        if state["mf_count"][w] == 0:
            j = np.zeros((m, 29), np.float64)
            prefix = int(output["phase_bounds"][w, 1])
            for r in range(m):
                packet = output["physical_J"][w, r]
                if r < prefix:
                    j[r, int(packet[0])] = packet[1]
                    continue
                for endpoint in range(2):
                    body, mask = int(packet[12 + endpoint]), int(packet[14 + endpoint])
                    wrench = packet[6 * endpoint : 6 * endpoint + 6].astype(np.float64)
                    if 0 <= body < 30:
                        for d in range(23):
                            if mask & (1 << d):
                                j[r, p["po"] + d] += current["axes"][w, d].astype(np.float64) @ wrench
                    elif body == 30:
                        for d in range(6):
                            if mask & (1 << d):
                                j[r, p["so"] + d] += current["free_axes"][w, d].astype(np.float64) @ wrench
            z = j @ t.T
            y = z @ t
            diag = np.einsum("ij,ij->i", z, z) + output["row_cfm"][w, :m]
        elif p["selector"] > 0:
            j = output["physical_J"][w, :m].astype(np.float64)
            y = output["response"][w, :m].astype(np.float64)
            diag = output["diag"][w, :m].astype(np.float64)
        else:
            z = output["response"][w, :m].astype(np.float64)
            j, y = np.linalg.solve(t, z.T).T, z @ t
            diag = output["diag"][w, :m].astype(np.float64)
        p.update(j=j, y=y, diag=diag, rhs=output["r0"][w, :m].astype(np.float64) - j @ p["vhat"])
        problems.append(p)
    return problems, original


def check_status(test, owner):
    """Retain existing current-error guards without requiring untouched Y finite."""
    owner.join()
    owner.allocation.check()
    for value in (
        owner.rows.out.status,
        owner.rows.out.global_status,
        owner.solve_descriptor.status,
        owner.guard.frame_status,
    ):
        np.testing.assert_array_equal(value.numpy(), 0)
    test.assertTrue(np.isfinite(owner.solve_descriptor.v_out.numpy()).all())


def check_solution(test, reference, owner, problems, *, original_geometry=False):
    """Run the existing action, momentum, cone and eight-sweep physical gates."""
    with mock.patch.object(reference, "physical_problem", return_value=problems):
        return reference.check_solve(
            test, owner, complete=owner.rows.device.is_cuda, original_geometry=original_geometry
        )


def check_cache(test, owner, problems, before):
    """Only demanded current MF0 rows may replace packet-ready storage with Y."""
    rows = owner.rows
    after = rows.out.valid.numpy()
    response = rows.out.response.numpy()
    mf = rows.state.mf_count.numpy()
    used = np.arange(192)[None, :] < rows.state.dense_count.numpy()[:, None]
    np.testing.assert_array_equal(before[used], 1)
    test.assertTrue(np.isin(after[used], (1, 2)).all())
    np.testing.assert_array_equal(after[used & (mf[:, None] > 0)], 1)
    maximum = 0.0
    count = 0
    for p in problems:
        w, m = p["world"], len(p["j"])
        if mf[w] != 0:
            continue
        for r in np.flatnonzero(after[w, :m] == 2):
            actual, expected = response[w, r], p["y"][r]
            maximum = max(maximum, float(np.max(np.abs(actual - expected) / (1 + np.abs(expected)))))
            count += 1
    test.assertLessEqual(maximum, 128 * np.finfo(np.float32).eps / 2)
    print(
        "FIRST_HIT_CACHE",
        {
            "demanded_rows": count,
            "action_scaled": maximum,
            "packet_only_rows": int(np.count_nonzero(used & (after == 1) & (mf[:, None] == 0))),
        },
        flush=True,
    )


def check_body_delta(test, reference, owner, problems):
    """A serial FP64 physical body-delta twist gives exactly each packet J dv."""
    rows = owner.rows
    plan, current, output = map(reference.host_arrays, (rows.plan, rows.current, rows.out))
    rng = np.random.default_rng(5317)
    maximum = 0.0
    for p in problems:
        w = p["world"]
        if len(p["jm"]) != 0:
            continue
        dv = rng.normal(size=29)
        motion = np.zeros((32, 6), np.float64)
        for b in range(30):
            d, parent = plan["body_local_dof"][w, b], plan["body_parent"][w, b]
            if d >= 0:
                motion[b] = current["axes"][w, d] * dv[p["po"] + d]
            if parent >= 0:
                motion[b] += motion[parent]
        motion[30] = dv[p["so"] : p["so"] + 6] @ current["free_axes"][w].astype(np.float64)
        for r in range(int(output["phase_bounds"][w, 1]), len(p["j"])):
            packet = output["physical_J"][w, r]
            value = 0.0
            for endpoint in range(2):
                b = int(packet[12 + endpoint])
                if b >= 0:
                    value += packet[endpoint * 6 : endpoint * 6 + 6].astype(np.float64) @ motion[b]
            expected = p["j"][r] @ dv
            maximum = max(maximum, abs(value - expected) / (1 + abs(expected)))
    test.assertLessEqual(maximum, 1e-12)


def exercise_actual(test, gpu, phase, device):
    """Actual current rows, both physical laws and the full demand lifetime."""
    with references() as reference:
        call = build_case(reference, gpu, phase, device)
        owner, rows = call.solve, call.rows
        try:
            problems, original = packet_problems(reference, owner)
            check_body_delta(test, reference, owner, problems)
            before = rows.out.valid.numpy().copy()
            owner.solve()
            check_status(test, owner)
            check_solution(test, reference, owner, problems)
            check_solution(test, reference, owner, original, original_geometry=True)
            check_cache(test, owner, problems, before)
        finally:
            owner.join()
            rows.bundle["host"]["snapshot"].close()


def physical_eight_warm(reference, problem, initial):
    """Reuse exactly the existing eight recurrence with its initial lambda input."""
    source = inspect.getsource(reference.physical_eight)
    header = "def physical_eight(p):"
    seed = 'lam, lm = np.zeros(len(p["j"])), np.zeros(len(p["jm"]))'
    if source.count(header) != 1 or source.count(seed) != 1:
        raise AssertionError("Original physical-eight warm seed seam changed")
    source = source.replace(header, "def physical_eight_warm(p, initial):")
    source = source.replace(seed, 'lam, lm = initial.copy(), np.zeros(len(p["jm"]))')
    namespace = {"np": np}
    exec(compile(source, "<unchanged physical-eight except initial lambda>", "exec"), namespace)
    return namespace["physical_eight_warm"](problem, initial)


def check_warm(test, reference, owner, p, initial):
    """Compare physical velocity and net delta-lambda action after sibling writes."""
    expected, _expected_lam, _ = physical_eight_warm(reference, p, initial)
    actual = owner.solve_descriptor.v_out.numpy()[p["dofs"]].astype(np.float64)
    final = owner.rows.out.impulses.numpy()[p["world"], : len(initial)].astype(np.float64)
    velocity = reference.scaled_max(actual, expected)
    net = p["y"].T @ (final - initial)
    action = reference.scaled_max(actual - p["vhat"], net)
    test.assertLessEqual(velocity, reference.VELOCITY_BOUND)
    test.assertLessEqual(action, reference.ACTION_BOUND)
    for n in np.flatnonzero(p["types"] == 0):
        radius = max(float(p["mu"][n + 1]) * final[n], 0.0)
        violation = max(np.hypot(final[n + 1], final[n + 2]) - radius, 0.0) / (1 + radius)
        test.assertLessEqual(violation, reference.ACTION_BOUND)
    print("FIRST_HIT_WARM", {"velocity_scaled": velocity, "net_action_scaled": action}, flush=True)
    return final


def exercise_lifetime(test, device="cpu"):
    """Current prefix moves a separating contact into demand; stale Y is poison.

    This is a labeled bounded row-law control on actual current geometry. The
    input residuals are intentionally loaded, not a claim about natural census.
    """
    gpu = 0 if device == "cpu" or "RTX" in wp.get_device(device).name else 1
    with references() as reference:
        call = build_case(reference, gpu, 0, device)
        owner, rows = call.solve, call.rows
        try:
            problems, _ = packet_problems(reference, owner)
            p = next(p for p in problems if len(p["jm"]) == 0 and len(p["j"]) >= 6)
            w, po = p["world"], p["po"]
            normals = np.flatnonzero(p["types"] == 0)
            # Pick a true nonzero normal/prefix coupling, not a prescribed or
            # structurally absent response. A current bound supplies its sign.
            inverse = np.linalg.inv(p["h"])
            coupling = p["j"][normals] @ inverse[:, po : po + 23]
            rr, d = np.unravel_index(np.abs(coupling).argmax(), coupling.shape)
            test.assertGreater(abs(coupling[rr, d]), 1e-8)
            sign = -1 if coupling[rr, d] > 0 else 1
            global_dof = int(rows.plan.dof_ids.numpy()[w, d])
            qi = int(rows.prefix.q_index.numpy()[global_dof])
            q = rows.prefix.q.numpy()[qi]
            bounds = getattr(rows.prefix, "upper" if sign < 0 else "lower")
            changed = bounds.numpy().copy()
            changed[global_dof] = q - 0.01 if sign < 0 else q + 0.01
            bounds.assign(changed)
            # An odd, sparse active queue must not touch the other worlds.
            active = np.array([w], np.int32)
            queue = rows.state.active_worlds.numpy().copy()
            queue[0] = w
            rows.state.active_worlds.assign(queue)
            rows.state.active_count.assign(np.array([1], np.int32))
            resolved = np.ones(rows.worlds, np.int32)
            resolved[active] = 0
            rows.state.resolved.assign(resolved)
            held = rows.held.T.numpy().copy()
            rows.out.valid.fill_(2)
            rows.out.response.fill_(float("nan"))
            prepare(owner)
            np.testing.assert_array_equal(rows.held.T.numpy(), held)
            m = int(rows.state.dense_count.numpy()[w])
            first = int(rows.out.phase_bounds.numpy()[w, 1])
            test.assertGreater(first, 0)
            np.testing.assert_array_equal(rows.out.valid.numpy()[w, :m], 1)
            test.assertTrue(np.isnan(rows.out.response.numpy()[w, first:m]).all())
            packet = rows.out.physical_J.numpy()[w]
            prefix_row = next(r for r in range(first) if int(packet[r, 0]) == po + d and packet[r, 1] == sign)
            problems, _ = packet_problems(reference, owner)
            p = problems[w]
            normals = np.flatnonzero(p["types"] == 0)
            delta = 0.02 / p["diag"][prefix_row]
            incident_change = p["j"][normals] @ (p["y"][prefix_row] * delta)
            selected = int(normals[np.argmin(incident_change)])
            test.assertLess(float(np.min(incident_change)), 0.0)
            r0 = rows.out.r0.numpy().copy()
            r0[w, :m] = 0
            r0[w, :first] = 100.0
            r0[w, normals] = 100.0
            r0[w, prefix_row] = -0.02
            r0[w, selected] = -0.5 * float(np.min(incident_change))
            rows.out.r0.assign(r0)
            p["rhs"] = r0[w, :m].astype(np.float64) - p["j"] @ p["vhat"]
            before_output = owner.solve_descriptor.v_out.numpy().copy()
            owner.solve()
            check_status(test, owner)
            test.assertEqual(int(rows.out.valid.numpy()[w, selected]), 2)
            test.assertGreater(rows.out.impulses.numpy()[w, selected], 0)
            check_warm(test, reference, owner, p, np.zeros(m))
            unused = np.flatnonzero(rows.out.valid.numpy()[w, first:m] == 1) + first
            test.assertGreater(len(unused), 0)
            test.assertTrue(np.isnan(rows.out.response.numpy()[w, unused]).all())
            outside = np.ones(before_output.shape, bool)
            outside[p["dofs"]] = False
            np.testing.assert_array_equal(owner.solve_descriptor.v_out.numpy()[outside], before_output[outside])

            # Warm tangent impulses are deliberately nonzero, so clearing a
            # zero parent must still undo both old physical tangent responses.
            prepare(owner)
            problems, _ = packet_problems(reference, owner)
            p = problems[w]
            m = len(p["j"])
            n = int(np.flatnonzero(p["types"] == 0)[0])
            initial = np.zeros(m, np.float64)
            initial[n : n + 3] = (0.02, 0.04, -0.03)
            impulses = rows.out.impulses.numpy().copy()
            impulses[w, :m] = initial
            rows.out.impulses.assign(impulses)
            initial = impulses[w, :m].astype(np.float64)
            r0 = rows.out.r0.numpy().copy()
            r0[w, :m] = 100.0
            r0[w, n : n + 3] = (-0.01, 0.4, -0.3)
            rows.out.r0.assign(r0)
            p["rhs"] = r0[w, :m].astype(np.float64) - p["j"] @ p["vhat"]
            owner.solve()
            check_status(test, owner)
            final = check_warm(test, reference, owner, p, initial)
            test.assertGreater(abs(final[n + 1] - initial[n + 1]), 1e-5)
            test.assertGreater(abs(final[n + 2] - initial[n + 2]), 1e-5)
            np.testing.assert_array_equal(rows.out.valid.numpy()[w, n : n + 3], 2)
        finally:
            owner.join()
            rows.bundle["host"]["snapshot"].close()


class TestFirstHitPhysical(unittest.TestCase):
    """Compare native first-hit state with independent physical coordinates."""

    def test_native_api_exists(self):
        """Require the candidate API before running physical fixture controls."""
        native = importlib.import_module("newton._src.solvers.feather_pgs.kinetic_first_hit")
        for name in ("get_contact_kernel", "get_prefix_kernel", "get_solve_kernel", "install"):
            self.assertTrue(callable(getattr(native, name)))

    def test_actual_four_cpu_current_and_held(self):
        """Check both GPUs' refresh/reuse fixtures; CPU generic remains excluded."""
        for gpu in range(2):
            for phase in range(2):
                with self.subTest(gpu=gpu, phase=phase):
                    exercise_actual(self, gpu, phase, "cpu")

    def test_prefix_demand_warm_sibling_and_current_cache(self):
        """Test late first hit, changed prefix, warm siblings and odd active count."""
        exercise_lifetime(self)

    def test_live_cold_and_current_force_held_reuse(self):
        """Retain actual cold publication and current-force laws after installation."""
        from tools.fpgs_bench import test_kinetic_live_bindings as live  # noqa: PLC0415

        original = live.bound_call

        def installed(*args, **kwargs):
            result = original(*args, **kwargs)
            call = result[-1]
            self.assertTrue(native.install(call))
            self.assertTrue(call.first_hit)
            return result

        with mock.patch.object(live, "bound_call", side_effect=installed):
            live.TestKineticLiveBindings.test_cold_current_refresh_predict_and_next_state(self)
            live.TestKineticLiveBindings.test_current_force_and_control_reuse_held_operator(self)
            live.TestKineticLiveBindings.test_world_gravity_changes_current_force_not_held_mass(self)

    @unittest.skipUnless(DEVICE.startswith("cuda"), "Root owns CUDA physical leases")
    def test_cuda_actual_current_and_held(self):
        """Compare each actual held phase on this card, including original MF."""
        gpu = 0 if "RTX" in wp.get_device(DEVICE).name else 1
        for phase in range(2):
            exercise_actual(self, gpu, phase, DEVICE)

    @unittest.skipUnless(DEVICE.startswith("cuda"), "Root owns CUDA physical leases")
    def test_cuda_prefix_demand_warm_sibling_and_current_cache(self):
        """Exercise the actual warp delta-twist/cache and sibling update seams."""
        exercise_lifetime(self, DEVICE)


if __name__ == "__main__":
    unittest.main()
