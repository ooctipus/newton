# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the bundle export/load round-trip.

Exercises :mod:`exporter` and :mod:`loader` against synthetic in-memory
inputs (a couple of USD Xforms, hand-rolled clone-plan arrays, a plain dict
sim_cfg). They do not require Isaac Lab.
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
    ENV_ORIGINS_FILENAME,
    EXTRAS_DIRNAME,
    SIM_CFG_FILENAME,
    STAGE_FILENAME,
    export,
)
from clone_plan import ClonePlan, SiteRequest  # noqa: E402
from loader import load_bundle  # noqa: E402

_CANONICAL_FILENAMES = (STAGE_FILENAME, CLONE_PLAN_FILENAME, ENV_ORIGINS_FILENAME, SIM_CFG_FILENAME)


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


def _make_sim_cfg() -> dict:
    """The single runtime config the loader consumes (flat, plain-dict)."""
    return {
        "physics_dt": 0.005,
        "decimation": 4,
        "episode_length_s": 20.0,
        "num_substeps": 1,
        "gravity": [0.0, 0.0, -9.81],
        "use_mujoco_contacts": True,
        "solver": {"njmax": 100, "nconmax": 200, "cone": "elliptic"},
        "collision": {},
        "shape": {"margin": 0.01, "gap": 0.01},
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
    sim_cfg = _make_sim_cfg()
    stage = _make_stage(("/World/envs/env_0", "/World/envs/env_1"))

    export(str(tmp_path), stage, plan, sim_cfg)

    # All canonical data files are present under extras/.
    artifact_dir = tmp_path / EXTRAS_DIRNAME
    for name in _CANONICAL_FILENAMES:
        assert (artifact_dir / name).is_file(), f"missing {name}"

    loaded = load_bundle(str(tmp_path))
    assert loaded.clone_plan.sources == plan.sources
    assert loaded.clone_plan.destinations == plan.destinations
    np.testing.assert_array_equal(loaded.clone_plan.clone_mask, plan.clone_mask)
    np.testing.assert_array_equal(loaded.clone_plan.env_origins, plan.env_origins)
    assert loaded.sim_cfg == sim_cfg

    # Stage path is the caller's responsibility but verify the on-disk content.
    reopened = Usd.Stage.Open(str(artifact_dir / STAGE_FILENAME))
    assert reopened is not None
    assert reopened.GetPrimAtPath("/World/envs/env_0").IsValid()
    assert reopened.GetPrimAtPath("/World/envs/env_1").IsValid()


def test_export_overwrites_existing_files(tmp_path: pathlib.Path) -> None:
    plan = _make_clone_plan()
    stage = _make_stage()
    export(str(tmp_path), stage, plan, {"physics_dt": 0.005})
    export(str(tmp_path), stage, plan, {"physics_dt": 0.01})
    assert load_bundle(str(tmp_path)).sim_cfg == {"physics_dt": 0.01}


def test_export_preserves_sibling_files(tmp_path: pathlib.Path) -> None:
    """Sibling payloads (e.g. policy.pt) must not be touched by export."""
    extras_dir = tmp_path / EXTRAS_DIRNAME
    extras_dir.mkdir()
    sibling = extras_dir / "policy.pt"
    sibling.write_bytes(b"fake policy bytes")

    plan = _make_clone_plan()
    stage = _make_stage()
    export(str(tmp_path), stage, plan, _make_sim_cfg())

    assert sibling.is_file()
    assert sibling.read_bytes() == b"fake policy bytes"


def test_export_creates_missing_dir(tmp_path: pathlib.Path) -> None:
    out_dir = tmp_path / "nested" / "subdir"
    plan = _make_clone_plan()
    stage = _make_stage()
    export(str(out_dir), stage, plan, _make_sim_cfg())
    assert out_dir.is_dir()


# ----------------------------------------------------------------------------
# Load failure modes
# ----------------------------------------------------------------------------


def test_load_raises_on_missing_env_origins(tmp_path: pathlib.Path) -> None:
    plan = _make_clone_plan()
    stage = _make_stage()
    export(str(tmp_path), stage, plan, _make_sim_cfg())

    os.unlink(tmp_path / EXTRAS_DIRNAME / ENV_ORIGINS_FILENAME)
    with pytest.raises(FileNotFoundError, match=ENV_ORIGINS_FILENAME):
        load_bundle(str(tmp_path))


def test_load_raises_on_missing_sim_cfg(tmp_path: pathlib.Path) -> None:
    """sim_cfg is the single required config -- its absence must fail loudly."""
    plan = _make_clone_plan()
    stage = _make_stage()
    export(str(tmp_path), stage, plan, _make_sim_cfg())

    os.unlink(tmp_path / EXTRAS_DIRNAME / SIM_CFG_FILENAME)
    with pytest.raises(FileNotFoundError, match=SIM_CFG_FILENAME):
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
    export(str(tmp_path), stage, plan, _make_sim_cfg())

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
        {"label": "ft_0", "body_pattern": "/World/envs/env_0/Robot/base", "xform": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
    ]

    loaded = load_bundle(str(tmp_path)).clone_plan
    assert loaded.env_spacing == plan.env_spacing
    assert loaded.up_axis == plan.up_axis
    assert loaded.simplify_meshes == plan.simplify_meshes
    assert loaded.site_requests == plan.site_requests


def test_sim_cfg_yaml_format(tmp_path: pathlib.Path) -> None:
    """sim_cfg round-trips as plain YAML (no Python class tags)."""
    plan = _make_clone_plan()
    stage = _make_stage()
    sim_cfg = _make_sim_cfg()
    export(str(tmp_path), stage, plan, sim_cfg)

    text = (tmp_path / EXTRAS_DIRNAME / SIM_CFG_FILENAME).read_text()
    assert "!!python/" not in text
    assert yaml.safe_load(text) == sim_cfg


def test_env_origins_npy_preserves_caller_dtype(tmp_path: pathlib.Path) -> None:
    """env_origins.npy is stored as-is; export does not coerce dtype."""
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.ones((1, 4), dtype=np.bool_),
        env_origins=np.zeros((4, 3), dtype=np.float64),
    )
    stage = _make_stage()
    export(str(tmp_path), stage, plan, _make_sim_cfg())
    arr = np.load(tmp_path / EXTRAS_DIRNAME / ENV_ORIGINS_FILENAME)
    assert arr.dtype == np.float64
    assert arr.shape == (4, 3)
