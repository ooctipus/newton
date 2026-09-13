# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Observe live graph ownership only outside the original timing windows."""

from contextlib import contextmanager

import numpy as np

from .kinetic_live_probe import RETIRED, Observer, check_publication


def profile_arguments(worlds, output, mode):
    """Keep the actual recipe; discovery timing has only forty measured steps."""
    if mode not in ("graph", "performance") or worlds != (512 if mode == "graph" else 16384):
        raise ValueError("Require 512 graph diagnosis or 16K discovery timing")
    return [
        "--task",
        "Isaac-Lift-KukaAllegro",
        "--physics",
        "feather_pgs",
        "--device",
        "cuda:0",
        "--num-envs",
        str(worlds),
        "--seed",
        "0",
        "--warmup-steps",
        "2" if mode == "graph" else "200",
        "--steps",
        "3" if mode == "graph" else "40",
        "--repeats",
        "1",
        "--profile-steps",
        "2" if mode == "graph" else "40",
        "--output",
        str(output),
    ]


class Boundaries:
    """Remove all per-call observers before timing; inspect two existing fences."""

    def __init__(self, solver_type, report, mode):
        self.cls, self.report, self.mode = solver_type, report, mode
        self.originals, self.replacements = {}, {}
        self.graph = self.clock = None
        report.update(graph_boundaries=[], pre_timing_calls=[], pre_timing_retired_calls=[])

    def install(self):
        """Count Python eager/capture calls, never device replay calls."""
        original = self.cls.step

        def step(solver, *args, **kwargs):
            capturing = solver.model.device.is_capturing
            cold = not bool(getattr(getattr(solver, "_kinetic_world", None), "ever_admitted", False))
            result = original(solver, *args, **kwargs)
            owner = getattr(solver, "_kinetic_world", None)
            self.report["pre_timing_calls"].append(
                {
                    "capture": capturing,
                    "cold": cold,
                    "private": owner is not None and owner.last_private,
                }
            )
            return result

        self.originals["step"], self.replacements["step"] = original, step
        self.cls.step = step
        for name in RETIRED:
            old = getattr(self.cls, name)

            def tracked(solver, *args, _name=name, _old=old, **kwargs):
                self.report["pre_timing_retired_calls"].append(_name)
                return _old(solver, *args, **kwargs)

            self.originals[name], self.replacements[name] = old, tracked
            setattr(self.cls, name, tracked)

    def restore(self):
        """Restore before the first timed step and also on failed construction."""
        for name, replacement in tuple(self.replacements.items()):
            if getattr(self.cls, name) is not replacement:
                raise ValueError("Graph observation hook ownership changed: " + name)
            setattr(self.cls, name, self.originals[name])
            del self.replacements[name]

    def check(self, manager):
        """Inspect actual current state after existing synchronization only."""
        import newton  # noqa: PLC0415 - runtime selected before this boundary

        self.restore()
        solver = manager._solver
        Observer(self.cls, {}, worlds=512 if self.mode == "graph" else 16384, perturb=False)._admit(
            solver, manager._solver_dt
        )
        owner = getattr(solver, "_kinetic_world", None)
        graph = manager._graph
        record = {"index": len(self.report["graph_boundaries"]), "passed": False}
        self.report["graph_boundaries"].append(record)
        if owner is None or not owner.last_private or graph is None:
            raise ValueError("Actual private graph was not established")
        owner.join()
        if self.report["pre_timing_retired_calls"] or not self.report["pre_timing_calls"]:
            raise ValueError("Original stages ran during private graph establishment")
        if not all(item["private"] for item in self.report["pre_timing_calls"]):
            raise ValueError("Original fallback entered the graph establishment")
        if not any(item["capture"] for item in self.report["pre_timing_calls"]):
            raise ValueError("No actual graph capture call was observed")
        for slots in owner.calls.values():
            for array in (slots.schedule.status, slots.next_schedule.status, slots.refresh_status):
                if np.any(array.numpy()):
                    raise ValueError("Live graph producer status failed")
        if np.any(owner.last_call.solve.guard.frame_status.numpy()):
            raise ValueError("Live graph solve/publication status failed")
        if np.any(owner.held.valid.numpy() != 1):
            raise ValueError("Live graph held operator is invalid")
        clock = owner.clock.numpy()
        if self.clock is not None:
            if graph is not self.graph or np.any(clock <= self.clock):
                raise ValueError("Graph changed or device epochs did not advance")
            record["epoch_delta_min"] = int(np.min(clock - self.clock))
        self.graph, self.clock = graph, clock.copy()
        state = manager._state_0
        arrays = (state.joint_q, state.joint_qd, state.body_q, state.body_qd, solver.v_hat, solver.v_out)
        if not all(np.isfinite(array.numpy()).all() for array in arrays):
            raise ValueError("Nonfinite actual graph state")
        contacts = manager.get_contacts()
        count = int(contacts.rigid_contact_count.numpy()[0])
        if count and not np.isfinite(contacts.rigid_contact_force[:count].numpy()).all():
            raise ValueError("Nonfinite actual graph public contact force")
        if self.mode == "graph":
            reference = solver.model.state()
            newton.eval_fk(solver.model, state.joint_q, state.joint_qd, reference)
            record["publication"] = check_publication(
                state.body_q.numpy(),
                state.body_qd.numpy(),
                reference.body_q.numpy(),
                reference.body_qd.numpy(),
            )
        record.update(
            passed=True,
            current_calls=len(owner.calls),
            state_slots=len(owner.states),
            clock_min=int(clock.min()),
            clock_max=int(clock.max()),
            dense_max=int(solver.constraint_count.numpy().max()),
            mf_max=int(solver.mf_constraint_count.numpy().max()),
            raw_count=count,
            step_host=int(solver._step),
            scope="Boundary status/finite/epoch checks; no per-replay counters or own-order contact-eight claim",
        )


@contextmanager
def measuring(solver_type, report, checked_capture, mode):
    """Compose with existing checks without extra physics calls or graph edits."""
    observer = Boundaries(solver_type, report, mode)
    original = checked_capture.install_boundary_check

    def install(harness, checks, save, get_manager, compat=None):
        original(harness, checks, save, get_manager, compat)
        checked = harness._model_meta

        def metadata(physics):
            result = checked(physics)
            observer.check(get_manager())
            return result

        harness._model_meta = metadata

    checked_capture.install_boundary_check = install
    try:
        observer.install()
        yield observer
    finally:
        observer.restore()
        if checked_capture.install_boundary_check is not install:
            raise ValueError("Graph boundary installer ownership changed")
        checked_capture.install_boundary_check = original
