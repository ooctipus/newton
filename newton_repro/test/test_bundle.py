# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the bundle export/load round-trip.

Exercises :mod:`exporter` and :mod:`loader` against synthetic in-memory
inputs (a couple of USD Xforms, hand-rolled clone-plan arrays, a plain dict
env_cfg). They do not require Isaac Lab.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pytest
import yaml

from pxr import Usd, UsdGeom

# Allow ``import clone_plan`` etc. without polluting sys.path globally.
_REPRO_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
if _REPRO_DIR not in sys.path:
    sys.path.insert(0, _REPRO_DIR)

from capture.exporter import (  # noqa: E402
    CLONE_PLAN_FILENAME,
    ENV_CFG_FILENAME,
    ENV_ORIGINS_FILENAME,
    EXTRAS_DIRNAME,
    SIM_CFG_FILENAME,
    STAGE_FILENAME,
    export,
)
from clone_plan import ClonePlan, SiteRequest  # noqa: E402
from loader import load_bundle  # noqa: E402

_CANONICAL_FILENAMES = (STAGE_FILENAME, CLONE_PLAN_FILENAME, ENV_ORIGINS_FILENAME, ENV_CFG_FILENAME)


def _make_stage(prim_paths: tuple[str, ...] = ("/World/envs/env_0",)) -> Usd.Stage:
    """Build a tiny in-memory USD stage with a few Xforms."""
    stage = Usd.Stage.CreateInMemory()
    for path in prim_paths:
        UsdGeom.Xform.Define(stage, path)
    return stage


def _make_clone_plan(num_sources: int = 1, num_envs: int = 4) -> ClonePlan:
    sources = tuple(f"/World/envs/env_{i}" for i in range(num_sources))
    destinations = tuple("/World/envs/env_{}" for _ in range(num_sources))
    clone_mask = np.ones((num_sources, num_envs), dtype=np.bool_)
    env_origins = np.stack(
        [
            np.arange(num_envs, dtype=np.float32),
            np.zeros(num_envs, dtype=np.float32),
            np.zeros(num_envs, dtype=np.float32),
        ],
        axis=1,
    )
    return ClonePlan(sources, destinations, clone_mask, env_origins)


def _make_env_cfg() -> dict:
    return {
        "sim": {
            "dt": 0.005,
            "physics": {
                "num_substeps": 1,
                "solver_cfg": {"njmax": 100, "nconmax": 200, "use_mujoco_contacts": True, "cone": "elliptic"},
                "collision_cfg": None,
            },
        },
        "decimation": 4,
        "episode_length_s": 20.0,
        "scene": {"num_envs": 4, "env_spacing": 2.5},
        "observations": {
            "policy": {
                "base_lin_vel": "isaaclab.envs.mdp:base_lin_vel",
                "base_ang_vel": "isaaclab.envs.mdp:base_ang_vel",
            }
        },
    }


# ----------------------------------------------------------------------------
# ClonePlan
# ----------------------------------------------------------------------------


def test_clone_plan_num_envs_matches_mask() -> None:
    plan = _make_clone_plan(num_sources=1, num_envs=7)
    assert plan.num_envs == 7


# ----------------------------------------------------------------------------
# Round-trip
# ----------------------------------------------------------------------------


def test_export_load_roundtrip(tmp_path: pathlib.Path) -> None:
    plan = _make_clone_plan(num_sources=1, num_envs=4)
    env_cfg = _make_env_cfg()
    stage = _make_stage(("/World/envs/env_0", "/World/envs/env_1"))

    export(str(tmp_path), stage, plan, env_cfg)

    # All canonical data files are present under extras/.
    artifact_dir = tmp_path / EXTRAS_DIRNAME
    for name in _CANONICAL_FILENAMES:
        assert (artifact_dir / name).is_file(), f"missing {name}"

    loaded = load_bundle(str(tmp_path))
    assert loaded.clone_plan.sources == plan.sources
    assert loaded.clone_plan.destinations == plan.destinations
    np.testing.assert_array_equal(loaded.clone_plan.clone_mask, plan.clone_mask)
    np.testing.assert_array_equal(loaded.clone_plan.env_origins, plan.env_origins)
    assert loaded.env_cfg == env_cfg

    # Stage path is the caller's responsibility but verify the on-disk content.
    reopened = Usd.Stage.Open(str(artifact_dir / STAGE_FILENAME))
    assert reopened is not None
    assert reopened.GetPrimAtPath("/World/envs/env_0").IsValid()
    assert reopened.GetPrimAtPath("/World/envs/env_1").IsValid()


def test_export_load_sim_cfg(tmp_path: pathlib.Path) -> None:
    plan = _make_clone_plan(num_sources=1, num_envs=2)
    stage = _make_stage()
    sim_cfg = {
        "physics_dt": 0.005,
        "decimation": 8,
        "episode_length_s": 10.0,
        "num_substeps": 1,
        "gravity": [0.0, 0.0, -9.81],
        "use_mujoco_contacts": False,
        "solver_kwargs": {"njmax": 200},
        "collision_kwargs": {"broad_phase": "explicit"},
        "default_shape_cfg": {"margin": 0.01, "gap": 0.01},
    }

    export(str(tmp_path), stage, plan, {"sim": {"dt": 0.005}}, sim_cfg)

    assert (tmp_path / EXTRAS_DIRNAME / SIM_CFG_FILENAME).is_file()
    bundle = load_bundle(str(tmp_path))
    assert bundle.sim_cfg == sim_cfg


def test_export_overwrites_existing_files(tmp_path: pathlib.Path) -> None:
    plan = _make_clone_plan()
    stage = _make_stage()
    export(str(tmp_path), stage, plan, {"sim": {"dt": 0.005}})
    export(str(tmp_path), stage, plan, {"sim": {"dt": 0.01}})
    assert load_bundle(str(tmp_path)).env_cfg == {"sim": {"dt": 0.01}}


def test_export_preserves_sibling_files(tmp_path: pathlib.Path) -> None:
    """Sibling payloads (e.g. policy.pt) must not be touched by export."""
    extras_dir = tmp_path / EXTRAS_DIRNAME
    extras_dir.mkdir()
    sibling = extras_dir / "policy.pt"
    sibling.write_bytes(b"fake policy bytes")

    plan = _make_clone_plan()
    stage = _make_stage()
    export(str(tmp_path), stage, plan, {"sim": {"dt": 0.005}})

    assert sibling.is_file()
    assert sibling.read_bytes() == b"fake policy bytes"


def test_export_creates_missing_dir(tmp_path: pathlib.Path) -> None:
    out_dir = tmp_path / "nested" / "subdir"
    plan = _make_clone_plan()
    stage = _make_stage()
    export(str(out_dir), stage, plan, {"sim": {"dt": 0.005}})
    assert out_dir.is_dir()


# ----------------------------------------------------------------------------
# Load failure modes
# ----------------------------------------------------------------------------


def test_load_raises_on_missing_file(tmp_path: pathlib.Path) -> None:
    plan = _make_clone_plan()
    stage = _make_stage()
    export(str(tmp_path), stage, plan, {"sim": {"dt": 0.005}})

    os.unlink(tmp_path / EXTRAS_DIRNAME / ENV_ORIGINS_FILENAME)
    with pytest.raises(FileNotFoundError, match=ENV_ORIGINS_FILENAME):
        load_bundle(str(tmp_path))


# ----------------------------------------------------------------------------
# On-disk format
# ----------------------------------------------------------------------------


def test_clone_plan_json_format(tmp_path: pathlib.Path) -> None:
    """clone_plan.json contains the documented replay fields."""
    plan = ClonePlan(
        sources=("/World/envs/env_a", "/World/envs/env_b"),
        destinations=("/World/envs/env_a_{}", "/World/envs/env_b_{}"),
        clone_mask=np.array([[True, False, True], [False, True, True]], dtype=np.bool_),
        env_origins=np.zeros((3, 3), dtype=np.float32),
        env_spacing=2.5,
        up_axis="Z",
        simplify_meshes=False,
        site_requests=(SiteRequest("ft_0", "/World/envs/env_0/Robot/base", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),),
    )
    stage = _make_stage()
    export(str(tmp_path), stage, plan, {"sim": {"dt": 0.005}})

    with open(tmp_path / EXTRAS_DIRNAME / CLONE_PLAN_FILENAME) as f:
        data = json.load(f)
    assert set(data.keys()) == {
        "sources",
        "destinations",
        "clone_mask",
        "num_envs",
        "env_spacing",
        "up_axis",
        "simplify_meshes",
        "site_requests",
    }
    assert data["sources"] == list(plan.sources)
    assert data["destinations"] == list(plan.destinations)
    assert data["clone_mask"] == [[True, False, True], [False, True, True]]
    assert data["num_envs"] == 3
    assert data["env_spacing"] == 2.5
    assert data["up_axis"] == "Z"
    assert data["simplify_meshes"] is False
    assert data["site_requests"] == [
        {
            "label": "ft_0",
            "body_pattern": "/World/envs/env_0/Robot/base",
            "xform": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "per_world": False,
        }
    ]

    loaded = load_bundle(str(tmp_path)).clone_plan
    assert loaded.env_spacing == plan.env_spacing
    assert loaded.up_axis == plan.up_axis
    assert loaded.simplify_meshes == plan.simplify_meshes
    assert loaded.site_requests == plan.site_requests


def test_per_world_site_request_roundtrip(tmp_path: pathlib.Path) -> None:
    """A ``per_world`` (bodyless, world-frame) site request round-trips through the bundle."""
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_origins=np.zeros((2, 3), dtype=np.float32),
        site_requests=(SiteRequest("ft_0", None, (0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0), per_world=True),),
    )
    stage = _make_stage()
    export(str(tmp_path), stage, plan, {"sim": {"dt": 0.005}})

    with open(tmp_path / EXTRAS_DIRNAME / CLONE_PLAN_FILENAME) as f:
        data = json.load(f)
    assert data["site_requests"] == [
        {"label": "ft_0", "body_pattern": None, "xform": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], "per_world": True}
    ]

    loaded = load_bundle(str(tmp_path)).clone_plan
    assert loaded.site_requests == plan.site_requests
    assert loaded.site_requests[0].per_world is True


def test_env_cfg_yaml_format(tmp_path: pathlib.Path) -> None:
    """A plain-dict env_cfg round-trips cleanly (no Python class tags)."""
    plan = _make_clone_plan()
    stage = _make_stage()
    env_cfg = _make_env_cfg()
    export(str(tmp_path), stage, plan, env_cfg)

    text = (tmp_path / EXTRAS_DIRNAME / ENV_CFG_FILENAME).read_text()
    assert "!!python/" not in text
    assert "\nsim:\n" in "\n" + text
    assert yaml.safe_load(text) == env_cfg


def test_env_origins_npy_preserves_caller_dtype(tmp_path: pathlib.Path) -> None:
    """env_origins.npy is stored as-is; export does not coerce dtype."""
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.ones((1, 4), dtype=np.bool_),
        env_origins=np.zeros((4, 3), dtype=np.float64),
    )
    stage = _make_stage()
    export(str(tmp_path), stage, plan, {"sim": {"dt": 0.005}})
    arr = np.load(tmp_path / EXTRAS_DIRNAME / ENV_ORIGINS_FILENAME)
    assert arr.dtype == np.float64
    assert arr.shape == (4, 3)
