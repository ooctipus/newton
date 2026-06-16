# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sanity tests for :mod:`parity` Newton model/state comparison helpers.

These tests do not exercise the export/load round-trip end-to-end (that is
done by ``capture/capture.py`` verification against a live Isaac Lab task). They instead
guard the comparison logic in :mod:`parity` against regressions:

* Two identical Newton builders finalize to models that compare equal.
* A single perturbed body mass produces a clear mismatch report.
"""

from __future__ import annotations

import pathlib
import sys

import newton
import pytest

_TEST_DIR = str(pathlib.Path(__file__).resolve().parent)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from parity import assert_model_state_equal, finalize_model_state  # noqa: E402

_SIM_CFG = {"gravity": (0.0, 0.0, -9.81)}


def _build_minimal(mass: float = 1.5, radius: float = 0.1) -> newton.ModelBuilder:
    """Builder with a single free-floating body + sphere shape."""
    b = newton.ModelBuilder()
    body = b.add_body(mass=mass, label="b0")
    b.add_joint_free(child=body)
    b.add_shape_sphere(body=body, radius=radius)
    return b


def test_identical_builders_pass_parity() -> None:
    model_a, state_a = finalize_model_state(_build_minimal(), _SIM_CFG, device="cpu", num_envs=1)
    model_b, state_b = finalize_model_state(_build_minimal(), _SIM_CFG, device="cpu", num_envs=1)
    assert_model_state_equal(model_a, state_a, model_b, state_b)


def test_perturbed_body_mass_fails_parity() -> None:
    model_a, state_a = finalize_model_state(_build_minimal(mass=1.5), _SIM_CFG, device="cpu", num_envs=1)
    model_b, state_b = finalize_model_state(_build_minimal(mass=2.5), _SIM_CFG, device="cpu", num_envs=1)
    with pytest.raises(AssertionError, match="body_mass"):
        assert_model_state_equal(model_a, state_a, model_b, state_b)


def test_perturbed_shape_radius_fails_parity() -> None:
    model_a, state_a = finalize_model_state(_build_minimal(radius=0.1), _SIM_CFG, device="cpu", num_envs=1)
    model_b, state_b = finalize_model_state(_build_minimal(radius=0.2), _SIM_CFG, device="cpu", num_envs=1)
    with pytest.raises(AssertionError, match=r"Newton model/state parity failed"):
        assert_model_state_equal(model_a, state_a, model_b, state_b)
