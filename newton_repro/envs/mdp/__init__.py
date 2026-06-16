# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone, Isaac Lab-free MDP helpers shared by per-bundle ``mdp.py`` files.

This package mirrors the shape of :mod:`isaaclab.envs.mdp` (``events``,
eventually ``observations``, ``rewards``, ``terminations``) but contains only
the subset of behavior that operates directly on a Newton :class:`Model` /
:class:`NewtonSim` -- no Isaac Lab managers, no scene entities, no
``SceneEntityCfg``. Per-bundle MDP files import from here when they need
faithful reproductions of common Isaac Lab randomization/observation logic.
"""
