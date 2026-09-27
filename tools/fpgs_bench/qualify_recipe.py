# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Qualify an execution recipe on a warmed live Lab model; never record timing.

By default the candidate enables grouped/lazy execution. An explicit candidate
PGS mode changes only that mode and preserves the original grouped/lazy settings.
Replay holds the loaded control fixed and shares reference-generated contacts.
It tests execution-path closeness, publication and reset invalidation, not
independent collision trajectories, projection convergence or RL equivalence.
Use the timing recipe's common environment: FEATHER_PGS_GROUP_LANES=16,
FEATHER_PGS_ROWS_MASKED=1, NEWTON_NARROW_PHASE_THREADS_X=4. Contact diagnostics
do not replace the original strict qualification gate.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import checked_capture as checked
import numpy as np


class QualificationComplete(Exception):
    """Unwind the original environment context before any timing window."""


def errors(reference, candidate, *, atol: float, rtol: float = 1.0e-4) -> dict:
    """Measure finite, shape-matched arrays without demanding bit identity."""
    left, right = np.asarray(reference), np.asarray(candidate)
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    if left.shape != right.shape or not finite:
        return {"pass": False, "finite": finite, "reference_shape": left.shape, "candidate_shape": right.shape}
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {
        "pass": bool(np.all(delta <= atol + rtol * np.abs(left))),
        "finite": True,
        "max_abs": float(delta.max(initial=0)),
        "rms": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "atol": atol,
        "rtol": rtol,
    }


def compare_states(reference, candidate) -> dict:
    """Check generalized state and the published state consumed by body sensors."""
    result = {
        name: errors(getattr(reference, name).numpy(), getattr(candidate, name).numpy(), atol=atol)
        for name, atol in (("joint_q", 1.0e-4), ("joint_qd", 1.0e-3), ("body_qd", 1.0e-3))
    }
    left, right = reference.body_q.numpy(), candidate.body_q.numpy()
    result["body_position"] = errors(left[:, :3], right[:, :3], atol=1.0e-4, rtol=0.0)
    # Quaternion signs represent the same rotation; do not reject q versus -q.
    aligned = right[:, 3:] * np.where(np.sum(left[:, 3:] * right[:, 3:], axis=1, keepdims=True) < 0, -1, 1)
    result["body_quaternion"] = errors(left[:, 3:], aligned, atol=1.0e-4)
    return result


def contact_diagnostics(reference, candidate, shape0, shape1, shape_world, world_weight, dt) -> dict:
    """Measure vector force error and net impulse relative to one gravity increment.

    Match Newton's sensor writer: +force on shape0, -force on shape1.
    shape_world uses -1 for global/static/kinematic endpoints. Zero weight has
    no gravity-based normalization and is explicitly not admitted by this gate.
    The 0.1% gravity-increment budget is a declared physical diagnostic, not a
    floating-point error bound; net translation does not bound contact torques.
    """
    left, right = np.asarray(reference, float), np.asarray(candidate, float)
    delta = right - left
    magnitude = np.linalg.norm(delta, axis=1)
    budget = 0.01 + 0.001 * np.linalg.norm(left, axis=1)
    point = {
        "pass": bool(np.all(magnitude <= budget)),
        "atol_N": 0.01,
        "rtol": 0.001,
        "max_vector_error_N": float(magnitude.max(initial=0)),
        "max_budget_fraction": float((magnitude / budget).max(initial=0)),
    }
    impulse = np.zeros((len(world_weight), 3), dtype=float)
    for shape, sign in ((shape0, 1), (shape1, -1)):
        ids = np.asarray(shape, dtype=int)
        if np.any((ids < 0) | (ids >= len(shape_world))):
            raise ValueError("Active contact references an invalid shape")
        worlds = shape_world[ids]
        keep = worlds >= 0
        np.add.at(impulse, worlds[keep], sign * delta[keep] * dt)
    scale = np.asarray(world_weight, float) * dt
    defined = scale > 0
    norm = np.linalg.norm(impulse, axis=1)
    normalized = np.divide(norm, scale, out=np.zeros_like(norm), where=defined)
    world = {
        "pass": bool(defined.all() and np.all(normalized <= 0.001)),
        "budget": 0.001,
        "impulse_error_Ns": impulse.tolist(),
        "weight_step_Ns": scale.tolist(),
        "normalized_error": [float(value) if valid else None for value, valid in zip(normalized, defined, strict=True)],
        "undefined_zero_weight_worlds": np.flatnonzero(~defined).tolist(),
        "scope": "Last solver substep only; not the sum over all substeps. Internal same-world impulses cancel.",
    }
    return {
        "point_vector": point,
        "world_net_impulse": world,
        "physical_contact_gate_pass": point["pass"] and world["pass"],
    }


def qualify(
    manager,
    solver_cfg,
    report: dict,
    save,
    *,
    steps: int,
    require_contacts: bool,
    candidate_pgs_mode: str | None = None,
) -> None:
    """Replay unchanged substep/collision budgets on independent solver histories."""
    if candidate_pgs_mode not in (None, "split", "matrix_free"):
        raise ValueError("Candidate PGS mode must be split or matrix_free")
    # Runtime imports must follow the explicit checkout selection in main().
    wp = importlib.import_module("warp")
    SolverFeatherPGS = importlib.import_module("newton.solvers").SolverFeatherPGS

    model, pipeline = manager._model, manager._collision_pipeline
    kwargs = manager._filter_solver_kwargs(SolverFeatherPGS, solver_cfg)
    candidate_kwargs = (
        {**kwargs, "grouped_dynamics": True, "lazy_kinematics": True}
        if candidate_pgs_mode is None
        else {**kwargs, "pgs_mode": candidate_pgs_mode}
    )
    # Infinite configuration sentinels are valid; never sanitize numerical results.
    report["solver_kwargs"] = {
        name: {
            key: str(value) if isinstance(value, (float, np.floating)) and np.isinf(value) else value
            for key, value in settings.items()
        }
        for name, settings in (("reference", kwargs), ("candidate", candidate_kwargs))
    }
    report["budgets"] = {
        "solver_dt": manager._solver_dt,
        "substeps": manager._num_substeps,
        "collision_decimation": manager._collision_decimation,
    }
    manager._solver.publish_kinematics(manager._state_0)
    initial = model.state()
    initial.assign(manager._state_0)
    solvers = [SolverFeatherPGS(model, **settings) for settings in (kwargs, candidate_kwargs)]
    report["paths"] = [
        {
            "grouped_dynamics": solver.grouped_dynamics,
            "lazy_kinematics": solver.lazy_kinematics,
            "grouped_active": solver._grouped_topology is not None,
            "fused_k1": solver._fused_k1,
            "pgs_mode": solver.pgs_mode,
        }
        for solver in solvers
    ]
    report["recipe_changes"] = {
        key: {"reference": report["paths"][0][key], "candidate": report["paths"][1][key]}
        for key in ("grouped_dynamics", "lazy_kinematics", "pgs_mode")
        if report["paths"][0][key] != report["paths"][1][key]
    }
    if candidate_pgs_mode is not None and report["paths"][1]["pgs_mode"] != candidate_pgs_mode:
        raise RuntimeError("Constructed candidate PGS mode differs from the requested recipe")
    report["grouped_execution_admitted"] = report["paths"][1]["grouped_active"]
    report["lazy_execution_admitted"] = report["paths"][1]["fused_k1"]
    states = [[model.state(), model.state()] for _ in solvers]
    loaded_final = [model.state(), model.state()]
    contacts = pipeline.contacts()
    body_world = model.body_world.numpy().astype(int)
    dynamic = (model.body_flags.numpy().astype(int) & int(importlib.import_module("newton").BodyFlags.DYNAMIC)) != 0
    local = (body_world >= 0) & dynamic
    world_mass = np.bincount(body_world[local], weights=model.body_mass.numpy()[local], minlength=model.world_count)
    world_weight = world_mass * np.linalg.norm(model.gravity.numpy()[: model.world_count], axis=1)
    shape_body = model.shape_body.numpy().astype(int)
    shape_world = np.full(len(shape_body), -1, dtype=int)
    attached = np.flatnonzero(shape_body >= 0)
    shape_world[attached] = np.where(dynamic[shape_body[attached]], body_world[shape_body[attached]], -1)
    report["contact_normalization"] = {
        "world_dynamic_mass_kg": world_mass.tolist(),
        "world_weight_N": world_weight.tolist(),
        "gravity_budget": 0.001,
    }
    collision_manager = SimpleNamespace(_collision_pipeline=pipeline, _solver=solvers[0])
    report["observations"], report["contact_counts"] = [], []
    save()
    for phase in ("loaded", "reset_replay"):
        for solver, pair in zip(solvers, states, strict=True):
            for state in pair:
                state.assign(initial)
            if phase == "reset_replay":
                solver.reset(pair[0])
                reset_errors = compare_states(initial, pair[0])
                if not all(value["pass"] for value in reset_errors.values()):
                    raise RuntimeError("Solver reset modified the authored state")
        for step in range(steps):
            pipeline.collide(states[0][0], contacts)
            for substep in range(manager._num_substeps):
                report["contact_counts"].append(int(contacts.rigid_contact_count.numpy()[0]))
                for solver, pair in zip(solvers, states, strict=True):
                    # Retain the same loaded force/control inputs on both branches.
                    wp.copy(pair[0].body_f, initial.body_f)
                    solver.step(pair[0], pair[1], manager._control, contacts, manager._solver_dt)
                    pair.reverse()
                mid = (
                    manager._collision_decimation > 0
                    and (substep + 1) % manager._collision_decimation == 0
                    and substep + 1 < manager._num_substeps
                )
                if mid or substep + 1 == manager._num_substeps:
                    for solver, pair in zip(solvers, states, strict=True):
                        solver.publish_kinematics(pair[0])
                if mid:
                    pipeline.collide(states[0][0], contacts)
            observation = {
                "phase": phase,
                "step": step,
                "state": compare_states(states[0][0], states[1][0]),
                "solvers": [{}, {}],
            }
            report["observations"].append(observation)
            forces = []
            for solver, entry in zip(solvers, observation["solvers"], strict=True):
                checked.check_solver("feather_pgs", solver, entry)
                solver.update_contacts(contacts)
                forces.append(contacts.rigid_contact_force.numpy()[: report["contact_counts"][-1]].copy())
            observation["state"]["contact_force"] = errors(forces[0], forces[1], atol=1.0e-2, rtol=1.0e-3)
            if not observation["state"]["contact_force"]["finite"]:
                raise RuntimeError("Nonfinite contact force")
            count = report["contact_counts"][-1]
            observation["contact_diagnostics"] = contact_diagnostics(
                *forces,
                contacts.rigid_contact_shape0.numpy()[:count],
                contacts.rigid_contact_shape1.numpy()[:count],
                shape_world,
                world_weight,
                manager._solver_dt,
            )
            checked.check_collision("feather_pgs", collision_manager, observation)
            save()
            if not all(value["pass"] for name, value in observation["state"].items() if name != "contact_force"):
                raise RuntimeError(f"Execution recipe diverged beyond declared tolerances: {phase}, step {step}")
        if phase == "loaded":
            for saved, pair in zip(loaded_final, states, strict=True):
                saved.assign(pair[0])
        else:
            report["reset_replay_errors"] = [
                compare_states(saved, pair[0]) for saved, pair in zip(loaded_final, states, strict=True)
            ]
            if not all(value["pass"] for case in report["reset_replay_errors"] for value in case.values()):
                raise RuntimeError("Same-state reset/replay differs beyond declared tolerances")
    report["loaded_contacts_observed"] = max(report["contact_counts"], default=0) > 0
    if require_contacts and not report["loaded_contacts_observed"]:
        raise RuntimeError("Requested loaded-contact coverage but no contacts were observed")
    report["physical_contact_gate_pass"] = all(
        item["contact_diagnostics"]["physical_contact_gate_pass"] for item in report["observations"]
    )
    report["strict_contact_gate_pass"] = all(item["state"]["contact_force"]["pass"] for item in report["observations"])
    if not report["strict_contact_gate_pass"]:
        raise RuntimeError(
            "Original strict contact-force gate failed after complete loaded/reset replay; see physical diagnostics"
        )
    report["qualified"] = True


def main(argv: list[str] | None = None) -> int:
    """Reuse the source-pinned Lab constructor/warmup and stop at its first boundary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab", type=Path, required=True)
    parser.add_argument("--newton", type=Path, required=True)
    parser.add_argument("--expected-run-sha256", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--replay-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--solver-attr", action="append", default=[])
    parser.add_argument(
        "--candidate-pgs-mode",
        choices=("split", "matrix_free"),
        help="Change only the candidate PGS mode; otherwise compare grouped/lazy execution.",
    )
    parser.add_argument("--require-contacts", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    lab, newton, output = (path.resolve() for path in (args.isaaclab, args.newton, args.output))
    if min(args.num_envs, args.warmup_steps, args.replay_steps) < 1:
        parser.error("World, warmup and replay counts must be positive")
    if output.exists() or any(
        output.is_relative_to(root) for root in (lab, newton, Path(__file__).resolve().parents[2])
    ):
        parser.error("Use a fresh output outside all source checkouts")
    if any(
        value.partition("=")[0] not in {"dense_max_constraints", "mf_max_constraints", "propagation_max_constraints"}
        for value in args.solver_attr
    ):
        parser.error("Only explicit solver capacities may override the original task recipe")
    report = {
        "qualified": False,
        "timing_valid": False,
        "task": args.task,
        "num_envs": args.num_envs,
        "scope": __doc__,
        "warmup_steps": args.warmup_steps,
        "replay_steps": args.replay_steps,
        "seed": args.seed,
        "overrides": args.override,
        "solver_attr": args.solver_attr,
        "candidate_pgs_mode": args.candidate_pgs_mode,
        "environment": {key: value for key, value in os.environ.items() if key.startswith(("FEATHER_PGS_", "NEWTON_"))},
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n")

    previous_argv = sys.argv
    try:
        path = lab / checked.HARNESS
        if checked.sha(path) != args.expected_run_sha256:
            raise RuntimeError("Lab harness differs from the selected source pin")
        report["run_profiled_sha256"] = checked.sha(path)
        report["qualifier_sha256"] = checked.sha(Path(__file__))
        report["commits"] = {
            name: subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            for name, root in (("isaaclab", lab), ("newton", newton))
        }
        selected = checked.select_imports(lab, newton)
        harness = ModuleType("_recipe_lab_warmup")
        harness.__file__ = str(path)
        exec(compile(path.read_bytes(), str(path), "exec"), harness.__dict__)
        PhysicsManager = importlib.import_module("isaaclab.physics").PhysicsManager
        NewtonManager = importlib.import_module("isaaclab_newton.physics").NewtonManager

        for name in checked.REQUIRED_PACKAGES:
            importlib.import_module(name)
        report["import_provenance"] = checked.import_provenance(selected, required=checked.REQUIRED_PACKAGES)
        checked.install_collision_storage_overrides(
            harness, importlib.import_module("newton").CollisionPipeline, report, save
        )

        def boundary(physics):
            if physics != "feather_pgs":
                raise RuntimeError("Qualification supports only FPGS execution recipes")
            qualify(
                NewtonManager,
                PhysicsManager._cfg.solver_cfg,
                report,
                save,
                steps=args.replay_steps,
                require_contacts=args.require_contacts,
                candidate_pgs_mode=args.candidate_pgs_mode,
            )
            raise QualificationComplete()

        harness._model_meta = boundary
        sys.argv = [
            str(path),
            "--physics",
            "feather_pgs",
            "--task",
            args.task,
            "--device",
            args.device,
            "--num-envs",
            str(args.num_envs),
            "--warmup-steps",
            str(args.warmup_steps),
            "--seed",
            str(args.seed),
            "--output",
            str(output),
        ]
        for name, values in (("--override", args.override), ("--solver-attr", args.solver_attr)):
            for value in values:
                sys.argv.extend((name, value))
        save()
        try:
            harness.main()
        except QualificationComplete:
            pass
        else:
            raise RuntimeError("Lab warmup did not reach the qualification boundary")
        return 0
    except BaseException as error:
        report.update(qualified=False, error=repr(error))
        raise
    finally:
        sys.argv = previous_argv
        save()


if __name__ == "__main__":
    raise SystemExit(main())
