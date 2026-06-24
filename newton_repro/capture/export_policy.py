# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a trained rsl_rl policy into a Newton repro bundle as ``extras/policy.pt``.

Mirrors the load + JIT-export half of ``scripts/reinforcement_learning/rsl_rl/play_rsl_rl.py``
(builds the env, wraps it for rsl_rl, constructs the runner, resolves a checkpoint -- including
W&B runs -- loads it, and calls ``runner.export_policy_to_jit``) but stops after the export
instead of running the play loop. The exported TorchScript module bundles the empirical
normalizer + actor so the dependency-free replay only needs ``torch.jit.load``.

Example::

    ./isaaclab.sh -p scripts/newton_repro/capture/export_policy.py \\
        --task=Isaac-Lift-KukaAllegro-Camera \\
        --output_dir=scripts/newton_repro/tasks/lift_kuka_allegro_camera \\
        --wandb_run_id=wdm0zlu4 --log_project_name=isaaclab_benchmarks \\
        --num_envs=2 --headless --enable_cameras \\
        presets=newton_mjwarp,single_camera,newton_renderer,depth64
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import logging
import os
import pathlib
import sys

_CAPTURE_DIR = str(pathlib.Path(__file__).resolve().parent)
_REPRO_DIR = str(pathlib.Path(__file__).resolve().parents[1])
for _path in (_CAPTURE_DIR, _REPRO_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from capture.exporter import EXTRAS_DIRNAME  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402
from isaaclab_tasks.utils.preset_cli import setup_preset_cli  # noqa: E402

logger = logging.getLogger("newton_repro.export_policy")

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an rsl_rl policy into a Newton repro bundle.")
    parser.add_argument("--task", type=str, required=True, help="Isaac Lab task name.")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point key.")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Bundle directory (policy.pt written to extras/)."
    )
    parser.add_argument("--num_envs", type=int, default=2, help="Env count for building the runner.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Local checkpoint path (overrides W&B).")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="W&B run id to download the checkpoint from.")
    parser.add_argument("--log_project_name", type=str, default=None, help="W&B project name.")
    parser.add_argument("--wandb_username", type=str, default=None, help="W&B entity/username (else WANDB_USERNAME).")
    parser.add_argument(
        "--wandb_checkpoint_iteration", type=int, default=-1, help="W&B checkpoint iteration (-1 = latest)."
    )
    add_launcher_args(parser)
    return parser


def _resolve_checkpoint(args_cli: argparse.Namespace, agent_cfg) -> str:
    if args_cli.checkpoint:
        from isaaclab.utils.assets import retrieve_file_path

        return retrieve_file_path(args_cli.checkpoint)
    if args_cli.wandb_run_id:
        from isaaclab.utils.wandb import get_model_checkpoint

        project = args_cli.log_project_name or getattr(agent_cfg, "wandb_project", None)
        return get_model_checkpoint(
            run_id=args_cli.wandb_run_id,
            project=project,
            checkpoint=args_cli.wandb_checkpoint_iteration,
            wandb_username=args_cli.wandb_username,
        )
    raise SystemExit("Provide either --checkpoint or --wandb_run_id.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s %(message)s")
    parser = _build_parser()
    args_cli, remaining = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + remaining

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    if args_cli.num_envs is not None and hasattr(env_cfg, "scene"):
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None and hasattr(env_cfg, "sim"):
        env_cfg.sim.device = args_cli.device
    # No stepping happens here, and eager CUDA-graph capture can trip Newton's lazy MuJoCo
    # contact-mapping allocation, so disable the graph for this build-and-export pass.
    with contextlib.suppress(Exception):
        env_cfg.sim.physics.use_cuda_graph = False
    env_cfg.seed = getattr(agent_cfg, "seed", None)

    out_dir = os.path.abspath(os.path.expanduser(args_cli.output_dir))
    extras_dir = os.path.join(out_dir, EXTRAS_DIRNAME)
    os.makedirs(extras_dir, exist_ok=True)

    import gymnasium as gym
    from packaging import version
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    installed_version = metadata.version("rsl-rl-lib")

    with launch_simulation(env_cfg, args_cli):
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        checkpoint_path = _resolve_checkpoint(args_cli, agent_cfg)
        logger.info("Loading checkpoint: %s", checkpoint_path)

        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
        runner_cls = OnPolicyRunner if agent_cfg.class_name == "OnPolicyRunner" else DistillationRunner
        runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=str(args_cli.device or env.unwrapped.device))
        runner.load(checkpoint_path)

        if version.parse(installed_version) < version.parse("4.0.0"):
            raise SystemExit(f"rsl-rl-lib >= 4.0.0 required for grouped-obs JIT export; found {installed_version}.")
        runner.export_policy_to_jit(path=extras_dir, filename="policy.pt")
        logger.info("Exported policy to %s", os.path.join(extras_dir, "policy.pt"))
        with contextlib.suppress(Exception):
            env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
