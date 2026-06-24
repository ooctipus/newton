# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read Newton repro bundles and build standalone Newton simulations."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import yaml
from capture.exporter import (
    CLONE_PLAN_FILENAME,
    ENV_CFG_FILENAME,
    ENV_ORIGINS_FILENAME,
    EXTRAS_DIRNAME,
    SIM_CFG_FILENAME,
    STAGE_FILENAME,
)
from clone_plan import ClonePlan, SiteRequest
from newton_sim import NewtonSim
from replicate import build_and_label

from pxr import Usd


@dataclass(frozen=True)
class Bundle:
    """Loaded bundle metadata and canonical file paths."""

    bundle_dir: str
    stage_path: str
    extras_dir: str
    clone_plan: ClonePlan
    env_cfg: dict
    sim_cfg: dict


def _artifact_path(bundle_dir: str, filename: str) -> str:
    """Return the canonical artifact path, falling back to legacy root layout."""
    extras_path = os.path.join(bundle_dir, EXTRAS_DIRNAME, filename)
    if os.path.exists(extras_path):
        return extras_path
    root_path = os.path.join(bundle_dir, filename)
    if os.path.exists(root_path):
        return root_path
    return extras_path


def _read_clone_plan(bundle_dir: str) -> ClonePlan:
    with open(_artifact_path(bundle_dir, CLONE_PLAN_FILENAME)) as f:
        plan_json = json.load(f)
    site_requests = tuple(
        SiteRequest(
            label=str(item["label"]),
            body_pattern=item.get("body_pattern"),
            xform=tuple(float(v) for v in item["xform"]),
            per_world=bool(item.get("per_world", False)),
        )
        for item in plan_json.get("site_requests", [])
    )
    return ClonePlan(
        sources=tuple(plan_json["sources"]),
        destinations=tuple(plan_json["destinations"]),
        clone_mask=np.asarray(plan_json["clone_mask"], dtype=np.bool_),
        env_origins=np.load(_artifact_path(bundle_dir, ENV_ORIGINS_FILENAME)),
        env_spacing=plan_json.get("env_spacing"),
        up_axis=str(plan_json.get("up_axis", "Z")),
        simplify_meshes=bool(plan_json.get("simplify_meshes", True)),
        site_requests=site_requests,
    )


def load_bundle(bundle_dir: str) -> Bundle:
    """Load a bundle directory written by :func:`exporter.export`."""
    bundle_dir = os.path.abspath(os.path.expanduser(bundle_dir))
    with open(_artifact_path(bundle_dir, ENV_CFG_FILENAME)) as f:
        env_cfg = yaml.unsafe_load(f) or {}
    sim_cfg_path = _artifact_path(bundle_dir, SIM_CFG_FILENAME)
    sim_cfg: dict = {}
    if os.path.exists(sim_cfg_path):
        with open(sim_cfg_path) as f:
            sim_cfg = yaml.safe_load(f) or {}
    return Bundle(
        bundle_dir=bundle_dir,
        stage_path=_artifact_path(bundle_dir, STAGE_FILENAME),
        extras_dir=os.path.join(bundle_dir, EXTRAS_DIRNAME),
        clone_plan=_read_clone_plan(bundle_dir),
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
    )


def build_newton_from_bundle(
    bundle: Bundle, num_envs: int | None = None, device: str = "cuda:0"
) -> tuple[NewtonSim, torch.Tensor]:
    """Build :class:`NewtonSim` from a loaded bundle."""
    plan = bundle.clone_plan
    if num_envs is not None and num_envs != plan.num_envs:
        if not 1 <= num_envs <= plan.num_envs:
            raise ValueError(f"num_envs must be in [1, {plan.num_envs}] (captured size), got {num_envs}.")
        plan = ClonePlan(
            sources=plan.sources,
            destinations=plan.destinations,
            clone_mask=plan.clone_mask[:, :num_envs].copy(),
            env_origins=plan.env_origins[:num_envs].copy(),
            env_spacing=plan.env_spacing,
            up_axis=plan.up_axis,
            simplify_meshes=plan.simplify_meshes,
            site_requests=plan.site_requests,
        )

    stage = Usd.Stage.Open(bundle.stage_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {bundle.stage_path}")

    builder, _ = build_and_label(
        stage=stage,
        sources=plan.sources,
        destinations=plan.destinations,
        env_ids=np.arange(plan.num_envs, dtype=np.int64),
        mapping=plan.clone_mask,
        positions=plan.env_origins,
        up_axis=plan.up_axis,
        simplify_meshes=plan.simplify_meshes,
        default_shape_cfg=bundle.sim_cfg.get("default_shape_cfg", {}),
        site_requests=plan.site_requests,
    )
    sim = NewtonSim(
        builder=builder,
        solver_kwargs=bundle.sim_cfg.get("solver_kwargs", {}),
        collision_kwargs=bundle.sim_cfg.get("collision_kwargs", {}) or {},
        physics_dt=float(bundle.sim_cfg["physics_dt"]),
        num_substeps=int(bundle.sim_cfg.get("num_substeps", 1)),
        use_mujoco_contacts=bool(bundle.sim_cfg.get("use_mujoco_contacts", True)),
        gravity=tuple(float(v) for v in bundle.sim_cfg.get("gravity", (0.0, 0.0, -9.81))),
        device=device,
        num_envs=plan.num_envs,
        collision_decimation=int(bundle.sim_cfg.get("collision_decimation", 0)),
        joint_overrides=_load_joint_overrides(bundle.extras_dir, plan.num_envs),
    )
    env_origins = torch.as_tensor(plan.env_origins, dtype=torch.float32, device=device)
    return sim, env_origins


def load_mdp_class(bundle_dir: str):
    """Import the bundle's sibling ``mdp.py`` by path and return its top-level ``MDP`` class.

    Each task bundle ships an ``mdp.py`` defining an ``MDP`` class that implements the repro
    contract (``act`` / ``apply_actuator`` / ``forward`` / ``reset_done``). Loading by path keeps
    bundles self-contained (no package install); the caller must have the ``newton_repro`` root on
    ``sys.path`` so the MDP's ``from envs... import`` statements resolve.
    """
    mdp_path = os.path.join(bundle_dir, "mdp.py")
    if not os.path.exists(mdp_path):
        raise FileNotFoundError(
            f"Bundle has no mdp.py at {mdp_path}. Write one with a top-level ``MDP`` class "
            "(see scripts/newton_repro/tasks/position_anymal_c/mdp.py for an example)."
        )
    spec = importlib.util.spec_from_file_location("newton_repro_bundle_mdp", mdp_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load MDP module from {mdp_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    mdp_cls = getattr(module, "MDP", None)
    if mdp_cls is None:
        raise RuntimeError(f"{mdp_path} must define a top-level MDP symbol")
    return mdp_cls


def _load_joint_overrides(extras_dir: str, num_envs: int) -> dict[str, np.ndarray]:
    """Build per-DOF joint property arrays from ``actuator_state.npz`` for *num_envs* worlds.

    The captured arrays cover the full captured model; the first world's per-DOF block is tiled
    to *num_envs* so the solver is built with the live implicit-actuator gains and effort limits
    regardless of how many environments the replay uses. Returns ``{}`` when no capture exists.
    """
    path = os.path.join(extras_dir, "actuator_state.npz")
    if not os.path.exists(path):
        return {}
    overrides: dict[str, np.ndarray] = {}
    with np.load(path) as data:
        dof_start = data["joint_dof_world_start"] if "joint_dof_world_start" in data.files else None
        jd_per = int(dof_start[1] - dof_start[0]) if dof_start is not None and len(dof_start) > 1 else None
        for attr in ("joint_target_ke", "joint_target_kd", "joint_armature", "joint_effort_limit", "joint_friction"):
            if attr not in data.files:
                continue
            arr = np.asarray(data[attr], dtype=np.float32).reshape(-1)
            block = arr[:jd_per] if jd_per is not None else arr
            overrides[attr] = np.tile(block, num_envs)
    return overrides
