# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-only ports of Isaac Lab ``mdp.events`` randomization terms.

Each function mirrors the Newton behavior of its Isaac Lab counterpart but
takes a :class:`NewtonSim` instead of a ``ManagerBasedEnv``. Body/shape
selection is by index (or via the small helpers at the bottom of this file)
because the standalone toolkit deliberately has no concept of
``SceneEntityCfg``.

The functions write into the live :class:`newton.Model` arrays and notify
the solver of the relevant property changes. Call them once at MDP
``__init__`` time to emulate Isaac Lab's ``mode="startup"`` events; call
them inside ``MDP.reset`` to emulate ``mode="reset"`` events.

Currently implemented:

* :func:`randomize_rigid_body_material`
* :func:`randomize_rigid_body_mass`

Helpers:

* :func:`find_body_indices` -- regex over ``model.body_label``.
* :func:`shape_indices_for_bodies` -- map body indices to their shape indices.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
import warp as wp
from newton.solvers import SolverNotifyFlags


def find_body_indices(model, pattern: str) -> np.ndarray:
    """Return indices of model bodies whose label fully matches *pattern*.

    Mirrors the role of Isaac Lab's ``SceneEntityCfg(body_names=...)`` resolution
    against ``Articulation.data.body_names``.

    Args:
        model: A finalized :class:`newton.Model`.
        pattern: Python regex applied with :func:`re.fullmatch`.

    Returns:
        ``(N,)`` ``int64`` array of body indices (possibly empty).
    """
    labels = list(model.body_label)
    return np.asarray(
        [i for i, label in enumerate(labels) if re.fullmatch(pattern, label)],
        dtype=np.int64,
    )


def shape_indices_for_bodies(model, body_indices: Sequence[int] | np.ndarray) -> np.ndarray:
    """Return shape indices belonging to any of *body_indices*.

    Args:
        model: A finalized :class:`newton.Model`.
        body_indices: Sequence of body indices to gather shapes for.

    Returns:
        ``(M,)`` ``int64`` array of shape indices (possibly empty).
    """
    shape_body = wp.to_torch(model.shape_body).cpu().numpy()
    body_set = {int(i) for i in body_indices}
    return np.asarray([i for i, b in enumerate(shape_body) if int(b) in body_set], dtype=np.int64)


def _apply_op(
    base: np.ndarray,
    samples: np.ndarray,
    operation: Literal["add", "scale", "abs"],
) -> np.ndarray:
    if operation == "add":
        return base + samples
    if operation == "scale":
        return base * samples
    if operation == "abs":
        return samples
    raise ValueError(f"Unsupported operation {operation!r}; expected one of 'add', 'scale', 'abs'.")


def _sample(
    rng: np.random.Generator,
    distribution: Literal["uniform", "log_uniform", "gaussian"],
    params: tuple[float, float],
    size: int,
) -> np.ndarray:
    lo, hi = float(params[0]), float(params[1])
    if distribution == "uniform":
        return rng.uniform(lo, hi, size=size)
    if distribution == "log_uniform":
        return np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))
    if distribution == "gaussian":
        return rng.normal(lo, hi, size=size)
    raise ValueError(f"Unsupported distribution {distribution!r}; expected 'uniform', 'log_uniform', or 'gaussian'.")


def randomize_rigid_body_material(
    sim,
    static_friction_range: tuple[float, float] = (1.0, 1.0),
    restitution_range: tuple[float, float] = (0.0, 0.0),
    shape_indices: Sequence[int] | np.ndarray | None = None,
    *,
    seed: int | None = None,
) -> None:
    """Sample per-shape friction and restitution into the Newton model.

    Mirrors the Newton path of
    :class:`isaaclab.envs.mdp.events.randomize_rigid_body_material`.
    Newton uses a single friction coefficient, so ``dynamic_friction_range``
    and ``num_buckets`` from the Isaac Lab signature are intentionally
    omitted -- they have no effect on Newton.

    Args:
        sim: :class:`NewtonSim`. The function writes into
            ``sim.model.shape_material_mu`` / ``sim.model.shape_material_restitution``
            and queues a :attr:`SolverNotifyFlags.SHAPE_PROPERTIES` notification.
        static_friction_range: ``(lo, hi)`` for uniform per-shape friction
            samples [unitless].
        restitution_range: ``(lo, hi)`` for uniform per-shape restitution
            samples [unitless].
        shape_indices: Optional subset of shapes to randomize. ``None`` means
            every shape in the model. See :func:`shape_indices_for_bodies` to
            derive indices from body labels.
        seed: Optional RNG seed (forwarded to :func:`numpy.random.default_rng`).
    """
    rng = np.random.default_rng(seed)
    mu = wp.to_torch(sim.model.shape_material_mu)
    rest = wp.to_torch(sim.model.shape_material_restitution)

    n = mu.shape[0] if shape_indices is None else int(np.asarray(shape_indices).shape[0])
    mu_samples = torch.from_numpy(rng.uniform(*static_friction_range, size=n).astype(np.float32)).to(mu.device)
    rest_samples = torch.from_numpy(rng.uniform(*restitution_range, size=n).astype(np.float32)).to(rest.device)

    if shape_indices is None:
        mu[:] = mu_samples
        rest[:] = rest_samples
    else:
        idx = torch.as_tensor(np.asarray(shape_indices, dtype=np.int64), device=mu.device)
        mu[idx] = mu_samples
        rest[idx] = rest_samples

    sim.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)


def randomize_rigid_body_mass(
    sim,
    body_indices: Sequence[int] | np.ndarray,
    mass_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "add",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    recompute_inertia: bool = True,
    min_mass: float = 1e-6,
    *,
    seed: int | None = None,
) -> None:
    """Sample per-body mass perturbations into the Newton model.

    Mirrors :class:`isaaclab.envs.mdp.events.randomize_rigid_body_mass`. The
    randomization is computed relative to the *current* values on the model
    (i.e. defaults at startup), so repeated calls compound. If you want
    reset-time re-randomization from defaults, snapshot ``body_mass`` /
    ``body_inertia`` after finalize and restore before calling.

    Args:
        sim: :class:`NewtonSim`. Writes into ``sim.model.body_mass``,
            ``sim.model.body_inv_mass``, and (if *recompute_inertia*)
            ``sim.model.body_inertia`` / ``sim.model.body_inv_inertia``.
            Queues a :attr:`SolverNotifyFlags.BODY_INERTIAL_PROPERTIES`
            notification.
        body_indices: Bodies to randomize. Use
            :func:`find_body_indices(model, ".*base.*")` to mirror Isaac Lab's
            body-name regex.
        mass_distribution_params: ``(lo, hi)`` (uniform/log_uniform) or
            ``(mean, std)`` (gaussian) [kg].
        operation: ``"add"`` adds the sample, ``"scale"`` multiplies,
            ``"abs"`` replaces.
        distribution: Sampling distribution.
        recompute_inertia: When ``True``, scale each affected body's inertia
            tensor by the mass ratio (preserves uniform-density assumption).
        min_mass: Lower bound applied via :func:`numpy.maximum` [kg]. Must be
            positive.
        seed: Optional RNG seed forwarded to :func:`numpy.random.default_rng`.
    """
    if min_mass <= 0:
        raise ValueError(f"min_mass must be positive, got {min_mass}")

    rng = np.random.default_rng(seed)
    idx = np.asarray(body_indices, dtype=np.int64)
    if idx.size == 0:
        return

    body_mass_t = wp.to_torch(sim.model.body_mass)
    body_inv_mass_t = wp.to_torch(sim.model.body_inv_mass)
    current_mass = body_mass_t.cpu().numpy().astype(np.float64)
    target_mass = current_mass.copy()

    samples = _sample(rng, distribution, mass_distribution_params, size=idx.shape[0])
    target_mass[idx] = _apply_op(current_mass[idx], samples, operation)
    target_mass = np.maximum(target_mass, min_mass)

    body_mass_t[:] = torch.from_numpy(target_mass.astype(np.float32)).to(body_mass_t.device)
    body_inv_mass_t[:] = torch.from_numpy((1.0 / target_mass).astype(np.float32)).to(body_inv_mass_t.device)

    if recompute_inertia:
        body_inertia_t = wp.to_torch(sim.model.body_inertia)
        body_inv_inertia_t = wp.to_torch(sim.model.body_inv_inertia)
        ratios = (target_mass[idx] / current_mass[idx]).astype(np.float32)
        inv_ratios = 1.0 / np.where(ratios == 0, 1.0, ratios)
        idx_t = torch.as_tensor(idx, device=body_inertia_t.device)
        ratios_t = torch.from_numpy(ratios).to(body_inertia_t.device)
        inv_ratios_t = torch.from_numpy(inv_ratios).to(body_inv_inertia_t.device)
        body_inertia_t[idx_t] *= ratios_t[:, None, None]
        body_inv_inertia_t[idx_t] *= inv_ratios_t[:, None, None]

    sim.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)
