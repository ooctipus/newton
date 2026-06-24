# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load and run an exported rsl_rl TorchScript policy with no rsl_rl/Isaac Lab dependency.

The bundle's ``extras/policy.pt`` is produced by ``runner.export_policy_to_jit`` (see
:mod:`capture.export_policy`). For a CNN actor its scripted ``forward`` has the signature
``forward(obs_1d: Tensor, obs_2d: list[Tensor]) -> Tensor``: it normalizes the concatenated 1-D
observation groups, runs the per-image CNN(s), concatenates the latents, and returns the
deterministic action mean. This wrapper just loads that module and feeds the obs tensors.
"""

from __future__ import annotations

import contextlib

import torch


class JitPolicy:
    """Thin wrapper around an exported TorchScript actor.

    Args:
        path: Path to the exported ``policy.pt``.
        device: Torch device to load and run on.
    """

    def __init__(self, path: str, device: str) -> None:
        self.device = device
        self.module = torch.jit.load(path, map_location=device).eval()

    @torch.inference_mode()
    def act(self, obs_1d: torch.Tensor, images: list[torch.Tensor]) -> torch.Tensor:
        """Return the deterministic action ``(num_envs, action_dim)``.

        Args:
            obs_1d: Concatenated 1-D observation groups (e.g. ``policy`` + ``proprio``), pre-clip,
                shape ``(num_envs, dim_1d)``. The baked-in empirical normalizer is applied inside.
            images: List of 2-D observation tensors, e.g. ``[depth(num_envs, 1, H, W)]``.
        """
        return self.module(obs_1d.to(self.device), [img.to(self.device) for img in images])

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Reset any recurrent state (no-op for feed-forward CNN/MLP actors)."""
        reset_fn = getattr(self.module, "reset", None)
        if reset_fn is not None:
            with contextlib.suppress(Exception):
                reset_fn()
