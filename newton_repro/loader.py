# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read Newton repro bundles and build standalone Newton simulations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import yaml
from pxr import Usd

from clone_plan import ClonePlan, SiteRequest
from capture.exporter import (
    CLONE_PLAN_FILENAME,
    ENV_CFG_FILENAME,
    ENV_ORIGINS_FILENAME,
    EXTRAS_DIRNAME,
    SIM_CFG_FILENAME,
    STAGE_FILENAME,
)
from newton_sim import NewtonSim
from replicate import build_and_label


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
            raise ValueError(
                f"num_envs must be in [1, {plan.num_envs}] (captured size), got {num_envs}."
            )
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
    )
    env_origins = torch.as_tensor(plan.env_origins, dtype=torch.float32, device=device)
    return sim, env_origins
