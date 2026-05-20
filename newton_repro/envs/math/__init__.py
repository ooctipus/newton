# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone math utilities for the Newton repro toolkit.

Two parallel surfaces, neither importing Isaac Lab:

* :mod:`envs.math.torch` -- ``torch.Tensor`` helpers copied verbatim from
  :mod:`isaaclab.utils.math`. Drop-in for per-bundle MDP scripts.
* :mod:`envs.math.warp` -- ``@wp.func`` counterparts callable from other
  Warp kernels (matches the patterns used by Newton's own utility-math
  modules: ``wp.set_module_options({"enable_backward": False})``, typed
  ``quatf`` / ``vec3f`` / ``mat33f`` signatures, ``Float`` generics for
  scalar helpers, ``wp.array[Any]`` + ``wp.overload`` for launcher kernels).

Random-input parity between the two surfaces is enforced by
:mod:`test.test_math`.
"""
