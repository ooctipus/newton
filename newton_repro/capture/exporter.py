# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Write a Newton repro bundle directory."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

import numpy as np
import yaml
from clone_plan import ClonePlan

from pxr import Usd

STAGE_FILENAME: str = "stage.usd"
CLONE_PLAN_FILENAME: str = "clone_plan.json"
ENV_ORIGINS_FILENAME: str = "env_origins.npy"
ENV_CFG_FILENAME: str = "env_cfg.yaml"
SIM_CFG_FILENAME: str = "sim_cfg.yaml"
EXTRAS_DIRNAME: str = "extras"


def _write_clone_plan(out_path: str, clone_plan: ClonePlan) -> None:
    payload = {
        "sources": list(clone_plan.sources),
        "destinations": list(clone_plan.destinations),
        "clone_mask": clone_plan.clone_mask.tolist(),
        "num_envs": clone_plan.num_envs,
        "env_spacing": clone_plan.env_spacing,
        "up_axis": clone_plan.up_axis,
        "simplify_meshes": clone_plan.simplify_meshes,
        "site_requests": [
            {"label": site.label, "body_pattern": site.body_pattern, "xform": list(site.xform)}
            for site in clone_plan.site_requests
        ],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_yaml(out_path: str, payload: Mapping, *, safe: bool) -> None:
    dump = yaml.safe_dump if safe else yaml.dump
    with open(out_path, "w") as f:
        dump(dict(payload), f, default_flow_style=False, sort_keys=False)


def export(
    out_dir: str,
    stage: Usd.Stage,
    clone_plan: ClonePlan,
    env_cfg: Mapping,
    sim_cfg: Mapping | None = None,
) -> None:
    """Write a Newton repro bundle into *out_dir*.

    Existing canonical data files under ``extras/`` are overwritten. Sibling
    Python files, such as hand-written MDP files, are left intact.

    Args:
        out_dir: Destination directory. Created if missing.
        stage: USD stage exported verbatim via :meth:`pxr.Usd.Stage.Export`.
        clone_plan: Replication plan captured at the cloner hook.
        env_cfg: Plain-Python ``class_to_dict(env_cfg)`` dump dumped to
            ``env_cfg.yaml``. The caller is responsible for converting Isaac
            Lab configclass instances and for ensuring the dict contains only
            plain Python types (otherwise the yaml will carry Python class
            tags and the load side will need those modules importable).
        sim_cfg: Optional flat standalone-runtime config dumped to
            ``extras/sim_cfg.yaml`` (consumed by :mod:`loader`/:mod:`repro`).
    """
    os.makedirs(out_dir, exist_ok=True)
    extras_dir = os.path.join(out_dir, EXTRAS_DIRNAME)
    os.makedirs(extras_dir, exist_ok=True)

    stage.Export(os.path.join(extras_dir, STAGE_FILENAME))
    _write_clone_plan(os.path.join(extras_dir, CLONE_PLAN_FILENAME), clone_plan)
    np.save(os.path.join(extras_dir, ENV_ORIGINS_FILENAME), clone_plan.env_origins)
    _write_yaml(os.path.join(extras_dir, ENV_CFG_FILENAME), env_cfg, safe=False)
    if sim_cfg is not None:
        _write_yaml(os.path.join(extras_dir, SIM_CFG_FILENAME), sim_cfg, safe=True)
