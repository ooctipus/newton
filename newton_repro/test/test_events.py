# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`envs.mdp.events`.

These exercise the Newton-only ports of Isaac Lab randomization terms on a
synthetic in-process model. They do not require Isaac Lab.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import newton
import numpy as np
import pytest
import warp as wp
from newton.solvers import SolverNotifyFlags

_REPRO_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_REPRO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPRO_DIR))

# The events module lives at ``envs/mdp/events.py``. Because ``envs/`` is not a
# Python package on the toolkit's import surface (it's a directory of bundles),
# load events.py by absolute path to keep the test directory's sys.path clean.
_EVENTS_PATH = _REPRO_DIR / "envs" / "mdp" / "events.py"
_spec = importlib.util.spec_from_file_location("newton_repro_envs_mdp_events", _EVENTS_PATH)
assert _spec is not None and _spec.loader is not None
events = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(events)

from newton_sim import NewtonSim  # noqa: E402


def _build_sim(masses: tuple[float, ...] = (1.5, 0.5, 2.0)) -> NewtonSim:
    """Build a three-body sim each with one sphere shape."""
    b = newton.ModelBuilder()
    bodies = []
    for i, mass in enumerate(masses):
        body = b.add_body(mass=mass, label=f"body_{i}")
        b.add_joint_free(child=body)
        b.add_shape_sphere(body=body, radius=0.1)
        bodies.append(body)
    return NewtonSim(
        builder=b,
        solver_kwargs={},
        collision_kwargs={},
        physics_dt=0.005,
        device="cpu",
        num_envs=1,
    )


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def test_find_body_indices_matches_regex() -> None:
    sim = _build_sim()
    base = events.find_body_indices(sim.model, r"body_\d")
    np.testing.assert_array_equal(base, np.array([0, 1, 2], dtype=np.int64))
    just_one = events.find_body_indices(sim.model, r"body_1")
    np.testing.assert_array_equal(just_one, np.array([1], dtype=np.int64))
    none = events.find_body_indices(sim.model, r"missing")
    assert none.shape == (0,)


def test_shape_indices_for_bodies() -> None:
    sim = _build_sim()
    body_idx = events.find_body_indices(sim.model, r"body_1")
    shape_idx = events.shape_indices_for_bodies(sim.model, body_idx)
    # body_1 owns exactly one sphere shape.
    assert shape_idx.shape == (1,)


# ----------------------------------------------------------------------------
# Material randomization
# ----------------------------------------------------------------------------


def test_randomize_rigid_body_material_writes_into_range() -> None:
    sim = _build_sim()
    events.randomize_rigid_body_material(
        sim,
        static_friction_range=(0.4, 1.5),
        restitution_range=(0.0, 0.1),
        seed=42,
    )
    mu = wp.to_torch(sim.model.shape_material_mu).cpu().numpy()
    rest = wp.to_torch(sim.model.shape_material_restitution).cpu().numpy()
    assert np.all(mu >= 0.4) and np.all(mu <= 1.5)
    assert np.all(rest >= 0.0) and np.all(rest <= 0.1)
    assert int(SolverNotifyFlags.SHAPE_PROPERTIES) in sim.pending_notify_flags


def test_randomize_rigid_body_material_targets_shape_subset() -> None:
    sim = _build_sim()
    # snapshot the original (all default) values
    mu0 = wp.to_torch(sim.model.shape_material_mu).cpu().numpy().copy()

    target_body = events.find_body_indices(sim.model, r"body_1")
    target_shapes = events.shape_indices_for_bodies(sim.model, target_body)

    events.randomize_rigid_body_material(
        sim,
        static_friction_range=(2.0, 2.0),  # degenerate range -> exact value
        restitution_range=(0.0, 0.0),
        shape_indices=target_shapes,
        seed=0,
    )
    mu = wp.to_torch(sim.model.shape_material_mu).cpu().numpy()
    # targeted shape got exactly 2.0; other shapes are unchanged from default.
    assert mu[target_shapes[0]] == pytest.approx(2.0)
    other = [i for i in range(mu.shape[0]) if i not in set(target_shapes.tolist())]
    np.testing.assert_array_equal(mu[other], mu0[other])


def test_randomize_rigid_body_material_seed_is_deterministic() -> None:
    sim_a = _build_sim()
    sim_b = _build_sim()
    for sim in (sim_a, sim_b):
        events.randomize_rigid_body_material(
            sim, static_friction_range=(0.4, 1.5), restitution_range=(0.0, 0.1), seed=12345
        )
    mu_a = wp.to_torch(sim_a.model.shape_material_mu).cpu().numpy()
    mu_b = wp.to_torch(sim_b.model.shape_material_mu).cpu().numpy()
    np.testing.assert_array_equal(mu_a, mu_b)


# ----------------------------------------------------------------------------
# Mass randomization
# ----------------------------------------------------------------------------


def test_randomize_rigid_body_mass_add_targets_subset() -> None:
    sim = _build_sim(masses=(1.5, 0.5, 2.0))
    mass0 = wp.to_torch(sim.model.body_mass).cpu().numpy().copy()
    inv0 = wp.to_torch(sim.model.body_inv_mass).cpu().numpy().copy()

    target = events.find_body_indices(sim.model, r"body_1")
    events.randomize_rigid_body_mass(
        sim,
        body_indices=target,
        mass_distribution_params=(0.0, 0.0),  # degenerate -> add 0
        operation="add",
        recompute_inertia=False,
        seed=0,
    )
    mass = wp.to_torch(sim.model.body_mass).cpu().numpy()
    inv = wp.to_torch(sim.model.body_inv_mass).cpu().numpy()
    # Nothing should have changed numerically.
    np.testing.assert_allclose(mass, mass0)
    np.testing.assert_allclose(inv, inv0, rtol=1e-5)
    assert int(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES) in sim.pending_notify_flags


def test_randomize_rigid_body_mass_scale_changes_inertia() -> None:
    sim = _build_sim(masses=(1.5, 0.5, 2.0))
    mass0 = wp.to_torch(sim.model.body_mass).cpu().numpy().copy()
    inertia0 = wp.to_torch(sim.model.body_inertia).cpu().numpy().copy()

    target = events.find_body_indices(sim.model, r"body_1")
    events.randomize_rigid_body_mass(
        sim,
        body_indices=target,
        mass_distribution_params=(2.0, 2.0),  # degenerate -> scale by exactly 2
        operation="scale",
        recompute_inertia=True,
        seed=0,
    )
    mass = wp.to_torch(sim.model.body_mass).cpu().numpy()
    inertia = wp.to_torch(sim.model.body_inertia).cpu().numpy()
    # Targeted body's mass doubled, others unchanged.
    np.testing.assert_allclose(mass[target], mass0[target] * 2.0)
    untouched = [i for i in range(mass.shape[0]) if i not in target.tolist()]
    np.testing.assert_allclose(mass[untouched], mass0[untouched])
    # Targeted body's inertia tensor doubled.
    np.testing.assert_allclose(inertia[target], inertia0[target] * 2.0, rtol=1e-5)
    np.testing.assert_allclose(inertia[untouched], inertia0[untouched], rtol=1e-5)


def test_randomize_rigid_body_mass_clamps_min_mass() -> None:
    sim = _build_sim(masses=(1.5, 0.5, 2.0))
    target = events.find_body_indices(sim.model, r"body_1")
    # Use ``operation="abs"`` with a value below min_mass so clamping is
    # unambiguous regardless of how Newton's finalize folds shape mass into
    # the input body mass.
    events.randomize_rigid_body_mass(
        sim,
        body_indices=target,
        mass_distribution_params=(-10.0, -10.0),  # degenerate -> set to -10.0
        operation="abs",
        min_mass=0.01,
        recompute_inertia=False,
        seed=0,
    )
    mass = wp.to_torch(sim.model.body_mass).cpu().numpy()
    assert mass[int(target[0])] == pytest.approx(0.01)


def test_randomize_rigid_body_mass_rejects_bad_min_mass() -> None:
    sim = _build_sim()
    target = events.find_body_indices(sim.model, r"body_0")
    with pytest.raises(ValueError, match="min_mass"):
        events.randomize_rigid_body_mass(
            sim,
            body_indices=target,
            mass_distribution_params=(0.0, 0.0),
            min_mass=0.0,
        )
