# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin standalone Newton simulation manager."""

from __future__ import annotations

import inspect
from collections.abc import Mapping

import warp as wp
from newton import CollisionPipeline, Contacts, ModelBuilder, eval_fk
from newton.geometry import HydroelasticSDF
from newton.solvers import SolverMuJoCo


def _to_kwargs(cfg_dict: Mapping | None, target_cls: type) -> dict:
    """Return only keyword arguments accepted by ``target_cls.__init__``."""
    valid = set(inspect.signature(target_cls.__init__).parameters) - {"self", "model"}
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
    ) -> None:
        self.device = device
        self.physics_dt = float(physics_dt)
        self.num_substeps = int(num_substeps)
        self.solver_dt = self.physics_dt / self.num_substeps
        self.use_mujoco_contacts = bool(use_mujoco_contacts)
        self.pending_notify_flags: set[int] = set()
        self.graph = None

        self.model = builder.finalize(device=device)
        self.model.set_gravity(gravity)
        self.model.num_envs = num_envs

        self.state = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state, None)

        solver_args = _to_kwargs(solver_kwargs, SolverMuJoCo)
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

    def notify_model_changed(self, flag: int) -> None:
        """Queue a solver model-change notification flag for the next step."""
        self.pending_notify_flags.add(int(flag))

    def forward(self) -> None:
        """Update body transforms from joint coordinates without stepping physics."""
        eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state, None)

    def _simulate(self) -> None:
        if self.pending_notify_flags:
            for flag in sorted(self.pending_notify_flags):
                self.solver.notify_model_changed(flag)
            self.pending_notify_flags.clear()

        if self.collision_pipeline is not None:
            eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state, None)
            self.collision_pipeline.collide(self.state, self.contacts)

        for _ in range(self.num_substeps):
            contacts = None if self.use_mujoco_contacts else self.contacts
            self.solver.step(self.state, self.state, self.control, contacts, self.solver_dt)
            self.state.clear_forces()

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
