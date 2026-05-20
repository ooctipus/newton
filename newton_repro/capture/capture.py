# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture an Isaac Lab task into a portable Newton repro bundle."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import pathlib
import shutil
import sys
import traceback
from collections.abc import Mapping

import numpy as np

_CAPTURE_DIR = str(pathlib.Path(__file__).resolve().parent)
_REPRO_DIR = str(pathlib.Path(__file__).resolve().parents[1])
for _path in (_CAPTURE_DIR, _REPRO_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from clone_plan import ClonePlan, SiteRequest  # noqa: E402
from capture.exporter import EXTRAS_DIRNAME, export as export_bundle  # noqa: E402
from loader import load_bundle  # noqa: E402
from replicate import build_and_label  # noqa: E402
from test.parity import assert_model_state_equal, finalize_model_state  # noqa: E402

from isaaclab.utils import class_to_dict  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import (  # noqa: E402
    add_launcher_args,
    fold_preset_tokens,
    launch_simulation,
    resolve_task_config,
)

logger = logging.getLogger("newton_repro.capture")

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture an Isaac Lab task into a Newton repro bundle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", type=str, required=True, help="Isaac Lab task name.")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point key.")
    parser.add_argument("--num_envs", type=int, default=None, help="Override env_cfg.scene.num_envs.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric before launch.")
    parser.add_argument("--output_dir", type=str, required=True, help="Destination directory for the bundle.")
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Optional policy copied into the bundle as extras/policy.pt for task MDP use.",
    )
    parser.add_argument("--mdp", type=str, default=None, help="Optional mdp.py copied into the bundle.")
    parser.add_argument("--agent_cfg", action="store_true", default=False, help="Also dump agent_cfg.yaml.")
    parser.add_argument(
        "--capture_reset_state",
        action="store_true",
        default=False,
        help="Let env construction finish, reset once, and dump reset/command/observation extras.",
    )
    parser.add_argument(
        "--no_verify_model",
        action="store_true",
        default=False,
        help="Skip live-vs-exported Newton model/state parity verification.",
    )
    parser.add_argument(
        "--verify_device",
        type=str,
        default="cpu",
        help="Device used to finalize parity models. Defaults to cpu.",
    )
    add_launcher_args(parser)
    return parser


def _plain(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items() if key != "class_type"}
    if isinstance(value, type):
        return f"{value.__module__}.{value.__name__}"
    return value


def _extract_sim_cfg(env_cfg) -> dict:
    sim_cfg = env_cfg.sim
    physics_cfg = sim_cfg.physics
    solver_cfg = physics_cfg.solver_cfg
    collision_cfg = getattr(physics_cfg, "collision_cfg", None)
    default_shape_cfg = getattr(physics_cfg, "default_shape_cfg", None)
    return {
        "physics_dt": float(sim_cfg.dt),
        "decimation": int(getattr(env_cfg, "decimation", 1)),
        "episode_length_s": float(getattr(env_cfg, "episode_length_s", 0.0)),
        "num_substeps": int(getattr(physics_cfg, "num_substeps", 1)),
        "gravity": [float(v) for v in getattr(sim_cfg, "gravity", (0.0, 0.0, -9.81))],
        "use_mujoco_contacts": bool(getattr(solver_cfg, "use_mujoco_contacts", True)),
        "solver_kwargs": _plain(class_to_dict(solver_cfg)),
        "collision_kwargs": _plain(class_to_dict(collision_cfg)) if collision_cfg is not None else {},
        "default_shape_cfg": _plain(class_to_dict(default_shape_cfg)) if default_shape_cfg is not None else {},
    }


class _CaptureDone(Exception):
    """Sentinel raised inside the patched cloner to short-circuit env creation."""


def _capture_site_requests() -> tuple[SiteRequest, ...]:
    from isaaclab_newton.physics import NewtonManager

    return tuple(
        SiteRequest(
            label=str(label),
            body_pattern=body_pattern,
            xform=tuple(float(v) for v in tuple(xform)),
        )
        for (body_pattern, _xform_key), (label, xform) in NewtonManager._cl_pending_sites.items()
    )


def _verify_exported_bundle(out_dir: str, live_builder, sim_cfg: dict, num_envs: int, device: str) -> None:
    """Rebuild the model from the exported bundle and assert it matches the live model."""
    from pxr import Usd

    bundle = load_bundle(out_dir)
    stage = Usd.Stage.Open(bundle.stage_path)
    if stage is None:
        raise RuntimeError(f"Failed to reopen exported stage: {bundle.stage_path}")
    plan = bundle.clone_plan
    rebuilt_builder, _ = build_and_label(
        stage=stage,
        sources=plan.sources,
        destinations=plan.destinations,
        env_ids=list(range(plan.num_envs)),
        mapping=plan.clone_mask,
        positions=plan.env_origins,
        up_axis=plan.up_axis,
        simplify_meshes=plan.simplify_meshes,
        default_shape_cfg=sim_cfg.get("default_shape_cfg", {}),
        site_requests=plan.site_requests,
    )
    live_model, live_state = finalize_model_state(live_builder, sim_cfg, device=device, num_envs=num_envs)
    rebuilt_model, rebuilt_state = finalize_model_state(rebuilt_builder, sim_cfg, device=device, num_envs=num_envs)
    assert_model_state_equal(live_model, live_state, rebuilt_model, rebuilt_state)


def _make_capture_hook(env_cfg, args_cli: argparse.Namespace, agent_cfg, original_replicate):
    captured = {"done": False}

    def _hook(
        stage,
        sources,
        destinations,
        env_ids,
        mapping,
        positions=None,
        quaternions=None,
        device="cpu",
        up_axis="Z",
        simplify_meshes=True,
    ):
        if positions is None:
            raise RuntimeError("newton_physics_replicate was called without positions; cannot capture env origins.")

        site_requests = _capture_site_requests()
        clone_plan = ClonePlan(
            sources=tuple(sources),
            destinations=tuple(destinations),
            clone_mask=mapping.detach().cpu().numpy(),
            env_origins=positions.detach().cpu().numpy(),
            env_spacing=float(getattr(env_cfg.scene, "env_spacing", 0.0)),
            up_axis=str(up_axis),
            simplify_meshes=bool(simplify_meshes),
            site_requests=site_requests,
        )
        sim_cfg = _extract_sim_cfg(env_cfg)
        export_bundle(args_cli.output_dir, stage, clone_plan, class_to_dict(env_cfg), sim_cfg)
        extras_dir = os.path.join(args_cli.output_dir, EXTRAS_DIRNAME)

        if args_cli.policy:
            shutil.copy(args_cli.policy, os.path.join(extras_dir, "policy.pt"))
        if args_cli.mdp:
            shutil.copy(args_cli.mdp, os.path.join(args_cli.output_dir, "mdp.py"))
        if args_cli.agent_cfg and agent_cfg is not None:
            from isaaclab.utils.io import dump_yaml

            dump_yaml(os.path.join(extras_dir, "agent_cfg.yaml"), agent_cfg)

        live_builder = None
        live_stage_info = None
        if not args_cli.no_verify_model or args_cli.capture_reset_state:
            live_builder, live_stage_info = original_replicate(
                stage,
                sources,
                destinations,
                env_ids,
                mapping,
                positions=positions,
                quaternions=quaternions,
                device=device,
                up_axis=up_axis,
                simplify_meshes=simplify_meshes,
            )

        if not args_cli.no_verify_model:
            _verify_exported_bundle(
                args_cli.output_dir,
                live_builder=live_builder,
                sim_cfg=sim_cfg,
                num_envs=clone_plan.num_envs,
                device=args_cli.verify_device,
            )

        captured["done"] = True
        logger.info(
            "Captured bundle: %s sources=%d num_envs=%d sites=%d",
            args_cli.output_dir,
            len(sources),
            clone_plan.num_envs,
            len(site_requests),
        )
        if args_cli.capture_reset_state:
            return live_builder, live_stage_info
        raise _CaptureDone()

    _hook.captured = captured
    return _hook


def _torch_to_numpy(value):
    import torch
    import warp as wp

    if hasattr(value, "torch"):
        value = value.torch
    elif hasattr(value, "numpy") and value.__class__.__module__.startswith("warp"):
        value = wp.to_torch(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _write_reset_state(out_dir: str, env) -> None:
    import torch
    import warp as wp
    from isaaclab.utils.assets import read_file

    extras_dir = os.path.join(out_dir, EXTRAS_DIRNAME)
    os.makedirs(extras_dir, exist_ok=True)

    obs_dict, _ = env.reset()
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    env_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device, dtype=torch.long)

    root_state = wp.to_torch(robot.data.root_state_w)[env_ids]
    joint_pos = wp.to_torch(robot.data.joint_pos)[env_ids]
    joint_vel = wp.to_torch(robot.data.joint_vel)[env_ids]
    reset_state = torch.cat((root_state, joint_pos, joint_vel), dim=-1)
    np.save(os.path.join(extras_dir, "reset_state.npy"), _torch_to_numpy(reset_state).astype(np.float32))

    np.savez(
        os.path.join(extras_dir, "robot_state.npz"),
        root_state=_torch_to_numpy(root_state).astype(np.float32),
        joint_pos=_torch_to_numpy(joint_pos).astype(np.float32),
        joint_vel=_torch_to_numpy(joint_vel).astype(np.float32),
        default_root_pose=_torch_to_numpy(robot.data.default_root_pose).astype(np.float32),
        default_root_vel=_torch_to_numpy(robot.data.default_root_vel).astype(np.float32),
        default_joint_pos=_torch_to_numpy(robot.data.default_joint_pos).astype(np.float32),
        default_joint_vel=_torch_to_numpy(robot.data.default_joint_vel).astype(np.float32),
    )

    command = unwrapped.command_manager.get_term("goal_point")
    command_payload = {
        "cmd_buf": _torch_to_numpy(command.cmd_buf).astype(np.float32),
        "cmd_mask": _torch_to_numpy(command.cmd_mask).astype(np.bool_),
        "cmd_indices": _torch_to_numpy(command.cmd_indices).astype(np.int64),
        "terrain_env_origins": _torch_to_numpy(unwrapped.scene.terrain.env_origins).astype(np.float32),
        "episode_length": _torch_to_numpy(unwrapped.episode_length_buf).astype(np.int64),
    }
    if hasattr(command, "cmd_ids"):
        command_payload["cmd_ids"] = _torch_to_numpy(command.cmd_ids).astype(np.int64)
    np.savez(os.path.join(extras_dir, "command_state.npz"), **command_payload)

    obs_payload = {key: _torch_to_numpy(value).astype(np.float32) for key, value in dict(obs_dict).items()}
    if obs_payload:
        np.savez(os.path.join(extras_dir, "initial_observations.npz"), **obs_payload)

    robot_cfg = getattr(unwrapped.cfg.scene, "robot", None)
    for actuator_cfg in (getattr(robot_cfg, "actuators", {}) or {}).values():
        network_file = getattr(actuator_cfg, "network_file", None)
        if not network_file:
            continue
        dst = os.path.join(extras_dir, os.path.basename(str(network_file)))
        if not os.path.exists(dst):
            with open(dst, "wb") as f:
                f.write(read_file(network_file).read())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s %(message)s")
    parser = _build_parser()
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    if args_cli.num_envs is not None and hasattr(env_cfg, "scene"):
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None and hasattr(env_cfg, "sim"):
        env_cfg.sim.device = args_cli.device
    if args_cli.disable_fabric and hasattr(env_cfg, "sim"):
        env_cfg.sim.use_fabric = False
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else getattr(agent_cfg, "seed", None)

    out_dir = os.path.abspath(os.path.expanduser(args_cli.output_dir))
    args_cli.output_dir = out_dir

    if args_cli.policy and not os.path.isfile(args_cli.policy):
        parser.error(f"--policy path does not exist: {args_cli.policy}")
    if args_cli.mdp and not os.path.isfile(args_cli.mdp):
        parser.error(f"--mdp path does not exist: {args_cli.mdp}")

    import gymnasium as gym

    from isaaclab_newton import cloner as cloner_pkg
    from isaaclab_newton.cloner import newton_replicate as cloner_impl

    with launch_simulation(env_cfg, args_cli):
        original = cloner_impl.newton_physics_replicate
        hook = _make_capture_hook(env_cfg, args_cli, agent_cfg, original)
        cloner_pkg.newton_physics_replicate = hook
        cloner_impl.newton_physics_replicate = hook
        env = None
        try:
            env = gym.make(args_cli.task, cfg=env_cfg)
            if args_cli.capture_reset_state:
                if not hook.captured["done"]:
                    logger.error("Cloner hook did not fire for task %r. Is the Newton preset active?", args_cli.task)
                    return 2
                _write_reset_state(out_dir, env)
                logger.info("Capture complete -- bundle at %s", out_dir)
                return 0
        except _CaptureDone:
            logger.info("Capture complete -- bundle at %s", out_dir)
            return 0
        except Exception:
            logger.error("Capture failed before reaching the cloner hook:\n%s", traceback.format_exc())
            return 1
        finally:
            if env is not None:
                with contextlib.suppress(Exception):
                    env.close()

        logger.error("Cloner hook did not fire for task %r. Is the Newton preset active?", args_cli.task)
        return 2


if __name__ == "__main__":
    sys.exit(main())
