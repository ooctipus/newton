# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Probe the LIVE Isaac Lab env + policy for a few steps to get ground-truth action/obs behavior.

Mirrors the load half of ``play_rsl_rl.py`` (build env, wrap, build runner, resolve the W&B
checkpoint, get the inference policy) then steps a fixed number of times logging the policy
action magnitude and per-group observation ranges. Used to compare against the standalone repro.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import sys

import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_task_config
from isaaclab_tasks.utils.preset_cli import setup_preset_cli

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _dump_actuator_properties(robot) -> None:
    """Print the live articulation's effective per-joint actuator properties (env 0).

    These are the values Isaac Lab's actuator models have written onto the model — the ground
    truth the standalone must reproduce. Compared against the captured ``actuator_state.npz``.
    """
    names = list(robot.joint_names)

    def _row(attr: str):
        with contextlib.suppress(Exception):
            return getattr(robot.data, attr).torch[0].tolist()
        return None

    ke = _row("joint_stiffness")
    kd = _row("joint_damping")
    arm = _row("joint_armature")
    eff = _row("joint_effort_limits")
    print("[live_probe] === live actuator properties (env 0) ===")
    for i, name in enumerate(names):
        def _g(row, i=i):
            return round(row[i], 4) if row is not None and i < len(row) else None

        print(f"[live_probe]   {i:2d} {name:<24} ke={_g(ke)} kd={_g(kd)} armature={_g(arm)} effort={_g(eff)}")


def _dump_hand_body_state(robot) -> None:
    """Dump the live hand-tip body state (quat + velocities, env 0) the policy actually observes.

    ``body_state_b`` uses ``body_quat_w`` / ``body_lin_vel_w`` / ``body_ang_vel_w`` for
    ``["palm_link", ".*_tip"]``. Comparing these against the standalone's Newton ``state.body_q`` /
    ``body_qd`` pins down the frame/velocity convention the hand-tips observation needs.
    """
    body_ids, body_names = robot.find_bodies(["palm_link", ".*_biotac_tip"])
    jp = robot.data.joint_pos.torch[0]
    print(f"[live_probe] === hand body state (env 0); joint_pos[:7]={[round(v, 4) for v in jp[:7].tolist()]} ===")
    for bid, nm in zip(body_ids, body_names):
        q = robot.data.body_quat_w.torch[0, bid]
        lv = robot.data.body_lin_vel_w.torch[0, bid]
        av = robot.data.body_ang_vel_w.torch[0, bid]
        print(
            f"[live_probe]   {nm:<20} quat_w={[round(v, 4) for v in q.tolist()]} "
            f"lin_v={[round(v, 4) for v in lv.tolist()]} ang_v={[round(v, 4) for v in av.tolist()]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe live env + policy behavior.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--log_project_name", type=str, default=None)
    parser.add_argument("--wandb_username", type=str, default=None)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--fixed_action", type=float, default=None, help="Apply this constant action instead of policy.")
    parser.add_argument("--adr_difficulty", type=int, default=None, help="Pin ADR curriculum difficulty (10 = max/eval).")
    add_launcher_args(parser)
    args_cli, remaining = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + remaining

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.adr_difficulty is not None:
        d = args_cli.adr_difficulty
        env_cfg.curriculum.adr.params.update(init_difficulty=d, min_difficulty=d, max_difficulty=d)
        print(f"[live_probe] pinned ADR difficulty to {d}")
    with contextlib.suppress(Exception):
        env_cfg.sim.physics.use_cuda_graph = False
    env_cfg.seed = getattr(agent_cfg, "seed", None)

    import gymnasium as gym
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.utils.wandb import get_model_checkpoint
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    installed_version = metadata.version("rsl-rl-lib")

    with launch_simulation(env_cfg, args_cli):
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        project = args_cli.log_project_name or getattr(agent_cfg, "wandb_project", None)
        ckpt = get_model_checkpoint(
            run_id=args_cli.wandb_run_id, project=project, checkpoint=-1, wandb_username=args_cli.wandb_username
        )
        print(f"[live_probe] checkpoint: {ckpt}")

        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=str(args_cli.device or env.unwrapped.device))
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        robot = env.unwrapped.scene["robot"]
        _dump_actuator_properties(robot)
        _dump_hand_body_state(robot)

        obs = env.get_observations()
        num_actions = env.num_actions
        for step in range(args_cli.steps):
            with torch.inference_mode():
                actions = policy(obs)
            if args_cli.fixed_action is not None:
                actions = torch.full((env.num_envs, num_actions), args_cli.fixed_action, device=env.unwrapped.device)
            # Absolute (not relative-to-default) arm joint state straight from the articulation,
            # so the trajectory can be compared against the standalone without obs-noise/offsets.
            jpos = robot.data.joint_pos.torch[0]
            jvel = robot.data.joint_vel.torch[0]
            print(
                f"[live_probe] step {step:2d}: |action|mean={actions.abs().mean().item():.4f} "
                f"action_max={actions.abs().max().item():.3f} "
                f"arm_qpos[0,:4]={[round(v, 4) for v in jpos[:4].tolist()]} "
                f"arm_qvel_max={jvel.abs().max().item():.3f}"
            )
            obs, _, dones, _ = env.step(actions)
            with contextlib.suppress(Exception):
                policy.reset(dones)
        with contextlib.suppress(Exception):
            env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
