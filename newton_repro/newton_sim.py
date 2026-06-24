# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin standalone Newton simulation manager."""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Mapping

import numpy as np
import torch
import warp as wp
from newton import CollisionPipeline, Contacts, ModelBuilder, eval_fk
from newton.geometry import HydroelasticSDF
from newton.solvers import SolverMuJoCo

# Solver-cfg keys that ``NewtonMJWarpManager._build_solver`` drops before forwarding
# to ``SolverMuJoCo.__init__`` (metadata and the deprecated ``ls_parallel`` field).
_SOLVER_IGNORED_KWARGS: frozenset[str] = frozenset({"class_type", "solver_type", "ls_parallel"})


def _to_kwargs(cfg_dict: Mapping | None, target_cls: type, ignore: frozenset[str] = frozenset()) -> dict:
    """Return only keyword arguments accepted by ``target_cls.__init__`` (minus *ignore*)."""
    valid = set(inspect.signature(target_cls.__init__).parameters) - {"self", "model"} - ignore
    return {key: value for key, value in dict(cfg_dict or {}).items() if key in valid}


def _collision_kwargs(cfg_dict: Mapping | None) -> dict:
    kwargs = _to_kwargs(cfg_dict, CollisionPipeline)
    hydro_cfg = kwargs.get("sdf_hydroelastic_config")
    if isinstance(hydro_cfg, dict):
        kwargs["sdf_hydroelastic_config"] = HydroelasticSDF.Config(**hydro_cfg)
    return kwargs


class NewtonSim:
    """Standalone MuJoCo-Warp Newton simulation wrapper."""

    def __init__(
        self,
        builder: ModelBuilder,
        solver_kwargs: Mapping | None,
        collision_kwargs: Mapping | None,
        physics_dt: float,
        num_substeps: int = 1,
        use_mujoco_contacts: bool = True,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        device: str = "cuda:0",
        num_envs: int | None = None,
        collision_decimation: int = 0,
        joint_overrides: Mapping[str, object] | None = None,
    ) -> None:
        self.device = device
        self.physics_dt = float(physics_dt)
        self.num_substeps = int(num_substeps)
        self.solver_dt = self.physics_dt / self.num_substeps
        self.use_mujoco_contacts = bool(use_mujoco_contacts)
        self.collision_decimation = int(collision_decimation)
        self.pending_notify_flags: set[int] = set()
        self.graph = None

        self.model = builder.finalize(device=device)
        self.model.set_gravity(gravity)
        self.model.num_envs = num_envs

        # Apply per-DOF joint property overrides (implicit-actuator gains, effort/limit) BEFORE the
        # solver is constructed. ``SolverMuJoCo`` bakes force limits and gains into its compiled
        # model at construction; applying them afterwards (via ``notify_model_changed``) does not
        # reliably re-derive the MuJoCo force ranges, leaving the PD effectively unclamped (huge
        # targets -> unbounded torque -> NaN). Setting them here makes the solver compile correctly.
        self._apply_joint_overrides(joint_overrides)

        # Request per-contact ``force`` before allocating the contact buffer so contact-force
        # sensors (e.g. fingertip forces) can read it. Must precede contact allocation below.
        request_attrs = getattr(self.model, "request_contact_attributes", None)
        if request_attrs is not None:
            with contextlib.suppress(Exception):
                request_attrs("force")

        self.state = self.model.state()
        self.control = self.model.control()
        eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state, None)

        solver_args = _to_kwargs(solver_kwargs, SolverMuJoCo, ignore=_SOLVER_IGNORED_KWARGS)
        solver_args["use_mujoco_contacts"] = self.use_mujoco_contacts
        self.solver = SolverMuJoCo(self.model, **solver_args)

        self.collision_pipeline = None
        if self.use_mujoco_contacts:
            self.contacts = Contacts(
                rigid_contact_max=self.solver.get_max_contact_count(),
                soft_contact_max=0,
                device=device,
                requested_attributes=self.model.get_requested_contact_attributes(),
            )
        else:
            self.collision_pipeline = CollisionPipeline(self.model, **_collision_kwargs(collision_kwargs))
            self.contacts = self.collision_pipeline.contacts()

    def _apply_joint_overrides(self, joint_overrides: Mapping[str, object] | None) -> None:
        """Write per-DOF joint property arrays onto the model before solver construction.

        Each value is a flat array aligned with the model's DOF order (e.g. captured
        ``joint_target_ke`` / ``joint_target_kd`` / ``joint_armature`` / ``joint_effort_limit``).
        Shorter arrays fill the leading DOFs; longer arrays are truncated.
        """
        if not joint_overrides:
            return
        for attr, values in joint_overrides.items():
            target = getattr(self.model, attr, None)
            if target is None or values is None:
                continue
            with contextlib.suppress(Exception):
                src = np.asarray(values, dtype=np.float32).reshape(-1)
                dst = wp.to_torch(target)
                count = min(int(dst.shape[0]), int(src.shape[0]))
                dst[:count] = torch.as_tensor(src[:count], dtype=dst.dtype, device=dst.device)

    def notify_model_changed(self, flag: int) -> None:
        """Queue a solver model-change notification flag for the next step."""
        self.pending_notify_flags.add(int(flag))

    def update_contacts(self) -> None:
        """Populate the contact buffer's forces from the solver (for contact-force sensors).

        Mirrors ``NewtonManager._update_sensors``' ``solver.update_contacts`` call so contact
        sensors can read per-contact forces on the Newton-collision-pipeline path (where the
        per-step :meth:`_simulate` does not otherwise refresh them).
        """
        if self.contacts is not None:
            with wp.ScopedDevice(self.device):
                self.solver.update_contacts(self.contacts, self.state)

    def _simulate(self) -> None:
        if self.pending_notify_flags:
            for flag in sorted(self.pending_notify_flags):
                self.solver.notify_model_changed(flag)
            self.pending_notify_flags.clear()

        contacts = None if self.use_mujoco_contacts else self.contacts
        if self.collision_pipeline is not None:
            eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state, None)
            self.collision_pipeline.collide(self.state, self.contacts)

        # Mirror NewtonManager._run_solver_substeps: optionally re-collide every
        # ``collision_decimation`` substeps, skipping the final substep whose
        # contact set would only feed the next tick's top-of-loop collide().
        collide_mid_loop = (
            self.collision_decimation > 0 and self.collision_pipeline is not None and contacts is not None
        )
        for i in range(self.num_substeps):
            self.solver.step(self.state, self.state, self.control, contacts, self.solver_dt)
            self.state.clear_forces()
            if collide_mid_loop and (i + 1) % self.collision_decimation == 0 and i + 1 < self.num_substeps:
                self.collision_pipeline.collide(self.state, self.contacts)

        if self.use_mujoco_contacts:
            self.solver.update_contacts(self.contacts, self.state)

    def step(self) -> None:
        """Step the Newton simulation by one physics dt."""
        with wp.ScopedDevice(self.device):
            if self.graph is None:
                self._simulate()
            else:
                wp.capture_launch(self.graph)

    def capture_graph(self) -> None:
        """Capture one physics step into a Warp CUDA graph."""
        if "cuda" not in self.device:
            self.graph = None
            return
        with wp.ScopedDevice(self.device):
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph
