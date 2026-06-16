# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read Newton repro bundles and build standalone Newton simulations."""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import yaml
from capture.exporter import (
    CLONE_PLAN_FILENAME,
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
    sim_cfg: dict


def _artifact_path(bundle_dir: str, filename: str) -> str:
    """Path to a bundle artifact under ``extras/``."""
    return os.path.join(bundle_dir, EXTRAS_DIRNAME, filename)


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
    with open(_artifact_path(bundle_dir, SIM_CFG_FILENAME)) as f:
        sim_cfg = yaml.safe_load(f) or {}
    return Bundle(
        bundle_dir=bundle_dir,
        stage_path=_artifact_path(bundle_dir, STAGE_FILENAME),
        extras_dir=os.path.join(bundle_dir, EXTRAS_DIRNAME),
        clone_plan=_read_clone_plan(bundle_dir),
        sim_cfg=sim_cfg,
    )


def apply_overrides(cfg: dict, overrides) -> None:
    """Apply ``dotted.path=value`` overrides to a nested dict in place (value parsed as a Python literal).

    e.g. ``num_substeps=16`` or ``collision.rigid_contact_max=8000000``. Raises if the path
    does not already exist, so a typo'd key fails loudly instead of being silently ignored.
    """
    for item in overrides or ():
        path, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must be KEY=VALUE, got {item!r}")
        *parents, leaf = path.split(".")
        node = cfg
        for k in parents:
            node = node[k]
        if leaf not in node:
            raise KeyError(f"override path {path!r} not found in sim_cfg")
        try:
            node[leaf] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            node[leaf] = raw


def _walk_cfg(cfg: dict, depth: int = 0):
    """Yield ``(label, value, is_header)`` rows for a nested dict; dict keys become indented headers."""
    for key, value in cfg.items():
        label = "  " * (depth + 1) + key
        if isinstance(value, dict):
            yield f"{label}:", None, True
            yield from _walk_cfg(value, depth + 1)
        else:
            yield label, value, False


def print_sim_cfg(cfg: dict) -> None:
    """Print sim_cfg grouped by namespace -- the dotted paths ``--set`` accepts and their current values."""
    rows = list(_walk_cfg(cfg))
    width = max((len(label) for label, _v, is_header in rows if not is_header), default=0)
    print("\n=== sim_cfg (override any leaf with --set <dotted.path>=<value>) ===")
    for label, value, is_header in rows:
        print(label if is_header else f"{label:<{width}}  {value}")
    print()


def build_newton_from_bundle(
    bundle: Bundle, num_envs: int | None = None, device: str = "cuda:0"
) -> tuple[NewtonSim, torch.Tensor]:
    """Build :class:`NewtonSim` from a loaded bundle (override sim_cfg first via :func:`apply_overrides`)."""
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

    # Captured live post-reset model state (friction / inertia / joint actuator props).
    # Applied for any num_envs <= captured: NewtonSim takes the env-major prefix (globals
    # then per-env entries are front-loaded, so the first N envs are a contiguous slice).
    model_state_path = os.path.join(bundle.extras_dir, "model_state.npz")
    model_state = None
    if os.path.exists(model_state_path):
        with np.load(model_state_path) as data:
            model_state = {k: data[k] for k in data.files}

    builder, _ = build_and_label(
        stage=stage,
        sources=plan.sources,
        destinations=plan.destinations,
        env_ids=np.arange(plan.num_envs, dtype=np.int64),
        mapping=plan.clone_mask,
        positions=plan.env_origins,
        up_axis=plan.up_axis,
        default_shape_cfg=bundle.sim_cfg.get("shape", {}),
        site_requests=plan.site_requests,
    )
    sim = NewtonSim(
        builder=builder,
        solver_kwargs=bundle.sim_cfg.get("solver", {}),
        collision_kwargs=bundle.sim_cfg.get("collision", {}) or {},
        physics_dt=float(bundle.sim_cfg["physics_dt"]),
        num_substeps=int(bundle.sim_cfg.get("num_substeps", 1)),
        use_mujoco_contacts=bool(bundle.sim_cfg.get("use_mujoco_contacts", True)),
        gravity=tuple(float(v) for v in bundle.sim_cfg.get("gravity", (0.0, 0.0, -9.81))),
        device=device,
        num_envs=plan.num_envs,
        model_state=model_state,
    )
    env_origins = torch.as_tensor(plan.env_origins, dtype=torch.float32, device=device)
    return sim, env_origins
