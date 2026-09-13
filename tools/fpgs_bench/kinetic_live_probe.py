# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Observe actual eager Kuka steps without snapshots or performance claims.

This first gate checks live ownership, cache cadence and public FK. It is not
an original-eight contact-solve or long-run convergence acceptance gate.
"""

import functools
from contextlib import contextmanager

import numpy as np

BOUND = 3.0e-5
RETIRED = (
    "_stage1_prepare_augmented_drives",
    "_stage1_fk_id",
    "_stage1_crba",
    "_stage3_compute_v_hat",
    "_stage4_build_rows",
    "_stage7_update_kinematics",
)


def transition(step):
    """Select explicit diagnostic perturbations, never a performance workload."""
    return {1: "force_target", 3: "subset_reset", 5: "odd_refresh"}.get(step)


def check_held(before, after, refreshed):
    """Require exact held storage retention only on actual unrequested reuse."""
    if not refreshed:
        for name, old in before.items():
            if not np.array_equal(old, after[name]):
                raise ValueError("Unrequested held operator change: " + name)


def check_publication(q, velocity, reference_q, reference_velocity):
    """Compare public poses and COM velocities with independent original FK."""
    if not all(np.isfinite(a).all() for a in (q, velocity, reference_q, reference_velocity)):
        raise ValueError("Nonfinite public state or FK reference")
    position = float(np.max(np.abs(q[:, :3] - reference_q[:, :3])))
    quaternion = float(
        np.max(
            np.minimum(
                np.linalg.norm(q[:, 3:] - reference_q[:, 3:], axis=1),
                np.linalg.norm(q[:, 3:] + reference_q[:, 3:], axis=1),
            )
        )
    )
    speed = float(np.max(np.abs(velocity - reference_velocity) / (1 + np.abs(reference_velocity))))
    result = {"position_abs": position, "quaternion_sign_equivalent": quaternion, "velocity_scaled": speed}
    if max(result.values()) > BOUND:
        raise ValueError("Public state differs from original FK: " + repr(result))
    return result


class Observer:
    """Own diagnostic hooks around actual Solver.step and retained public APIs."""

    def __init__(self, solver_type, report, *, worlds, perturb):
        self.cls, self.report = solver_type, report
        self.worlds, self.perturb = worlds, perturb
        self.pending, self.solver, self.reference_state = None, None, None
        self.originals, self.replacements = {}, {}
        self.report.update(calls=[], force_exports=0, resets=0, perturbations=[])

    def _replace(self, name, replacement):
        """Record exact hook ownership for deterministic restoration."""
        self.originals[name] = getattr(self.cls, name)
        self.replacements[name] = replacement
        setattr(self.cls, name, replacement)

    def _admit(self, solver, dt):
        """Reject an original-path fallback masquerading as a private live test."""
        expected = {
            "world_count": self.worlds,
            "max_world_dofs": 29,
            "pgs_iterations": 8,
            "pgs_velocity_iterations": 0,
            "update_mass_matrix_interval": 2,
            "dense_max_constraints": 192,
            "mf_max_constraints": 64,
            "propagation_max_constraints": 192,
            "pgs_mode": "matrix_free",
            "pgs_schedule": "interleaved",
            "pgs_warmstart": False,
        }
        bad = {
            name: getattr(solver, name, None)
            for name, value in expected.items()
            if getattr(solver, name, None) != value
        }
        if bad or dt != 1 / 240 or solver._kinetic_world is None or solver.model.device.is_capturing:
            raise ValueError("Require the admitted actual eager recipe: " + repr(bad))
        if self.solver is not None and self.solver is not solver:
            raise ValueError("A second solver entered the same probe")
        self.solver = solver
        return solver._kinetic_world

    def _perturb(self, solver, state, control, step):
        """Exercise real authored inputs and reset/request APIs outside timing."""
        import warp as wp  # noqa: PLC0415 - initialize only inside the owned runtime

        event = transition(step) if self.perturb else None
        restores = []
        owner = solver._kinetic_world
        if event == "force_target":
            dof = int(owner.plan.dof_ids.numpy()[0, 0])
            body = int(owner.plan.body_ids.numpy()[0, 30])
            for array, index, delta in (
                (control.joint_f, dof, 0.1),
                (control.joint_target_q, dof, 0.01),
                (state.body_f, (body, 0), 0.25),
            ):
                old = array.numpy().copy()
                changed = old.copy()
                changed[index] += delta
                array.assign(changed)
                restores.append((array, old))
        elif event == "subset_reset":
            qindex = int(owner.plan.q_index.numpy()[0, 0])
            q = state.joint_q.numpy().copy()
            q[qindex] += 0.001
            state.joint_q.assign(q)
            mask = np.zeros(self.worlds, dtype=bool)
            mask[: min(2, self.worlds)] = True
            solver.reset(state, wp.array(mask, dtype=wp.bool, device=solver.model.device))
        elif event == "odd_refresh":
            if step % 2 != 1:
                raise ValueError("Odd-request control moved to an ordinary refresh")
            solver._mass_update_requested.fill_(1)
        if event is not None:
            self.report["perturbations"].append({"step": step, "event": event})
        return restores, event

    @staticmethod
    def _held(owner):
        """Read actual current held arrays for untimed cadence checks."""
        return {
            name: getattr(owner.held, name).numpy().copy()
            for name in ("T", "augmented", "lower6", "inverse6", "generation")
        }

    def install(self):
        """Wrap one real call, never substitute a snapshot-driven physics loop."""
        original_step = self.cls.step

        @functools.wraps(original_step)
        def step(solver, state_in, state_out, control, contacts, dt, collide_done_event=None):
            import newton  # noqa: PLC0415 - preserve pre-import experiment flags

            if self.pending is not None:
                raise ValueError("Nested live probe step")
            owner = self._admit(solver, dt)
            index = int(solver._step)
            if not self.report["calls"] and index != 0:
                raise ValueError("The cold step was not observed")
            if control is None:
                raise ValueError("Actual Lab callback did not supply control")
            restores, event = self._perturb(solver, state_in, control, index)
            requested = bool(index % 2 == 0 or solver._force_mass_update or solver._mass_update_requested.numpy()[0])
            before = self._held(owner)
            record = {"step": index, "requested_refresh": requested, "event": event, "retired_calls": []}
            self.pending = record
            try:
                result = original_step(solver, state_in, state_out, control, contacts, dt, collide_done_event)
            finally:
                self.pending = None
                for array, old in restores:
                    array.assign(old)
            if solver._step != index + 1 or not owner.last_private or owner.last_output is not state_out:
                raise ValueError("Actual step did not complete exactly one private call")
            if record["retired_calls"]:
                raise ValueError("Retired pipeline still ran: " + repr(record["retired_calls"]))
            owner.join()
            if np.any(owner.last_call.solve.guard.frame_status.numpy()):
                raise ValueError("Private frame status failed")
            if np.any(owner.held.valid.numpy() != 1):
                raise ValueError("Private step failed to initialize held data")
            after = self._held(owner)
            check_held(before, after, requested)
            slots = owner.calls[(id(state_in), id(state_out))]
            requested_actual = slots.requested.numpy()
            if np.any(requested_actual != int(requested)):
                raise ValueError("Primary/free refresh request disagrees with original cadence")
            if requested and not np.array_equal(after["generation"], slots.state_generation.numpy()):
                raise ValueError("Held data has the wrong actual refresh generation")
            if np.any(slots.next_current.valid.numpy() != 1):
                raise ValueError("Next current state was not published")
            if not np.array_equal(slots.next_current.generation.numpy(), slots.next_generation.numpy()):
                raise ValueError("Next-state generation was not advanced")
            if not np.array_equal(owner.canonical_generation.numpy(), slots.next_generation.numpy()):
                raise ValueError("The current free canonical bank is stale")
            if event == "subset_reset" and np.any(slots.repair_requested.numpy()[:2] != 1):
                raise ValueError("Reset worlds did not repair current data")
            solver.check_constraint_capacity()
            if self.reference_state is None:
                self.reference_state = solver.model.state()
            newton.eval_fk(solver.model, state_out.joint_q, state_out.joint_qd, self.reference_state)
            record["publication"] = check_publication(
                state_out.body_q.numpy(),
                state_out.body_qd.numpy(),
                self.reference_state.body_q.numpy(),
                self.reference_state.body_qd.numpy(),
            )
            if not all(
                np.isfinite(array.numpy()).all()
                for array in (state_out.joint_q, state_out.joint_qd, solver.v_hat, solver.v_out)
            ):
                raise ValueError("Nonfinite actual dynamics output")
            record.update(
                dense_max=int(solver.constraint_count.numpy().max()),
                mf_max=int(solver.mf_constraint_count.numpy().max()),
                repaired_worlds=int(np.count_nonzero(slots.repair_requested.numpy())),
                generation_min=int(slots.next_generation.numpy().min()),
            )
            self.report["calls"].append(record)
            print("KINETIC_LIVE_CALL", record, flush=True)
            return result

        self._replace("step", step)
        for name in RETIRED:
            old = getattr(self.cls, name)

            def tracked(solver, *args, _name=name, _old=old, **kwargs):
                if self.pending is not None:
                    self.pending["retired_calls"].append(_name)
                return _old(solver, *args, **kwargs)

            self._replace(name, tracked)
        old_force, old_reset = self.cls.update_contacts, self.cls.reset

        @functools.wraps(old_force)
        def export(solver, contacts):
            result = old_force(solver, contacts)
            if solver is self.solver:
                count = int(contacts.rigid_contact_count.numpy()[0])
                if count and not np.isfinite(contacts.rigid_contact_force[:count].numpy()).all():
                    raise ValueError("Nonfinite exported contact forces")
                self.report["force_exports"] += 1
            return result

        @functools.wraps(old_reset)
        def reset(solver, *args, **kwargs):
            result = old_reset(solver, *args, **kwargs)
            self.report["resets"] += 1
            return result

        self._replace("update_contacts", export)
        self._replace("reset", reset)

    def close(self):
        """Restore owned hooks without overwriting a foreign replacement."""
        changed = []
        for name, replacement in self.replacements.items():
            if getattr(self.cls, name) is replacement:
                setattr(self.cls, name, self.originals[name])
            else:
                changed.append(name)
        if changed:
            raise ValueError("Live probe hook ownership changed: " + repr(changed))


@contextmanager
def observing(solver_type, report, *, worlds, perturb=False):
    """Restore observation hooks even when live construction or physics fails."""
    observer = Observer(solver_type, report, worlds=worlds, perturb=perturb)
    try:
        observer.install()
        yield observer
    finally:
        observer.close()
