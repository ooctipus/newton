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

from capture.exporter import EXTRAS_DIRNAME  # noqa: E402
from capture.exporter import export as export_bundle
from clone_plan import ClonePlan, SiteRequest  # noqa: E402
from loader import load_bundle  # noqa: E402
from replicate import build_and_label  # noqa: E402
from test.parity import assert_model_state_equal, finalize_model_state  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab.utils import class_to_dict  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402
from isaaclab_tasks.utils.preset_cli import setup_preset_cli  # noqa: E402

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
        "collision_decimation": int(getattr(physics_cfg, "collision_decimation", 0)),
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
            per_world=bool(per_world),
        )
        for (body_pattern, per_world, _xform_key), (label, xform) in NewtonManager._cl_pending_sites.items()
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
    """Build a replacement for ``NewtonReplicateContext.replicate`` that exports a bundle.

    The live cloner dispatches replication through
    :meth:`isaaclab_newton.cloner.replicate.NewtonReplicateContext.replicate` (drained from
    :data:`isaaclab.cloner.REPLICATION_QUEUE`), so we patch that method. The bound context
    (*self*) carries the stage, up-axis, and mesh-simplification flag; the merged clone mapping
    (sources/destinations/env_ids/mask/positions) comes from ``self._merged_mapping()``.
    """
    captured = {"done": False}

    def _hook(self):
        sources, destinations, env_ids, mapping, positions, quaternions = self._merged_mapping()
        if positions is None:
            raise RuntimeError("Newton replication ran without positions; cannot capture env origins.")

        # Capture pending site requests before the original build consumes/clears them.
        site_requests = _capture_site_requests()
        clone_plan = ClonePlan(
            sources=tuple(sources),
            destinations=tuple(destinations),
            clone_mask=mapping.detach().cpu().numpy(),
            env_origins=positions.detach().cpu().numpy(),
            env_spacing=float(getattr(env_cfg.scene, "env_spacing", 0.0)),
            up_axis=str(self.up_axis),
            simplify_meshes=bool(self.simplify_meshes),
            site_requests=site_requests,
        )
        sim_cfg = _extract_sim_cfg(env_cfg)
        export_bundle(args_cli.output_dir, self.stage, clone_plan, class_to_dict(env_cfg), sim_cfg)
        extras_dir = os.path.join(args_cli.output_dir, EXTRAS_DIRNAME)

        if args_cli.policy:
            shutil.copy(args_cli.policy, os.path.join(extras_dir, "policy.pt"))
        if args_cli.mdp:
            shutil.copy(args_cli.mdp, os.path.join(args_cli.output_dir, "mdp.py"))
        if args_cli.agent_cfg and agent_cfg is not None:
            from isaaclab.utils.io import dump_yaml

            dump_yaml(os.path.join(extras_dir, "agent_cfg.yaml"), agent_cfg)

        # Run the real replication when we need the live model (parity check) or when
        # full env construction must continue (reset-state capture).
        live_result = None
        if not args_cli.no_verify_model or args_cli.capture_reset_state:
            live_result = original_replicate(self)

        if not args_cli.no_verify_model and live_result is not None:
            _verify_exported_bundle(
                args_cli.output_dir,
                live_builder=live_result[0],
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
            return live_result
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


def _dump_robot_state(extras_dir: str, unwrapped, robot_name: str = "robot") -> None:
    """Dump the reset articulation state + per-joint defaults/limits for *robot_name*.

    Writes ``robot_state.npz`` (with ``joint_names`` for name-based standalone mapping)
    and the legacy ``reset_state.npy`` (``[root_state(13), joint_pos, joint_vel]`` per env).
    """
    import torch
    import warp as wp

    robot = unwrapped.scene[robot_name]
    env_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device, dtype=torch.long)
    root_state = wp.to_torch(robot.data.root_state_w)[env_ids]
    joint_pos = wp.to_torch(robot.data.joint_pos)[env_ids]
    joint_vel = wp.to_torch(robot.data.joint_vel)[env_ids]

    np.save(
        os.path.join(extras_dir, "reset_state.npy"),
        _torch_to_numpy(torch.cat((root_state, joint_pos, joint_vel), dim=-1)).astype(np.float32),
    )

    payload = {
        "root_state": _torch_to_numpy(root_state).astype(np.float32),
        "joint_pos": _torch_to_numpy(joint_pos).astype(np.float32),
        "joint_vel": _torch_to_numpy(joint_vel).astype(np.float32),
        "default_root_pose": _torch_to_numpy(robot.data.default_root_pose).astype(np.float32),
        "default_root_vel": _torch_to_numpy(robot.data.default_root_vel).astype(np.float32),
        "default_joint_pos": _torch_to_numpy(robot.data.default_joint_pos).astype(np.float32),
        "default_joint_vel": _torch_to_numpy(robot.data.default_joint_vel).astype(np.float32),
    }
    for attr in ("joint_pos_limits", "soft_joint_pos_limits", "joint_vel_limits", "joint_names"):
        value = getattr(robot.data, attr, None) if attr != "joint_names" else getattr(robot, attr, None)
        if value is None:
            continue
        if attr == "joint_names":
            payload["joint_names"] = np.asarray(list(value), dtype=object)
        else:
            payload[attr] = _torch_to_numpy(value).astype(np.float32)
    np.savez(os.path.join(extras_dir, "robot_state.npz"), **payload)


def _dump_object_states(extras_dir: str, unwrapped) -> None:
    """Dump root state + defaults for every non-articulation rigid asset (e.g. ``object``, ``table``).

    Backend-agnostic: identified by duck typing (has ``data.root_state_w`` but no
    ``data.joint_pos``) rather than ``isinstance`` checks, since Newton-backed assets are not
    instances of ``isaaclab.assets.RigidObject``.
    """
    import torch
    import warp as wp

    env_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device, dtype=torch.long)
    payload: dict[str, np.ndarray] = {}
    for name in list(unwrapped.scene.keys()):
        with contextlib.suppress(Exception):
            asset = unwrapped.scene[name]
            data = getattr(asset, "data", None)
            if data is None or not hasattr(data, "root_state_w") or hasattr(data, "joint_pos"):
                continue
            payload[f"{name}__root_state"] = _torch_to_numpy(wp.to_torch(data.root_state_w)[env_ids]).astype(np.float32)
            payload[f"{name}__default_root_pose"] = _torch_to_numpy(data.default_root_pose).astype(np.float32)
            payload[f"{name}__default_root_vel"] = _torch_to_numpy(data.default_root_vel).astype(np.float32)
    if payload:
        np.savez(os.path.join(extras_dir, "object_state.npz"), **payload)


def _dump_command_states(extras_dir: str, unwrapped) -> None:
    """Dump the resolved command buffer of every active command term, keyed by term name."""
    command_manager = getattr(unwrapped, "command_manager", None)
    if command_manager is None:
        return
    payload: dict[str, np.ndarray] = {}
    for name in list(getattr(command_manager, "active_terms", []) or []):
        with contextlib.suppress(Exception):
            payload[name] = _torch_to_numpy(command_manager.get_command(name)).astype(np.float32)
    payload["episode_length"] = _torch_to_numpy(unwrapped.episode_length_buf).astype(np.int64)
    if payload:
        np.savez(os.path.join(extras_dir, "command_state.npz"), **payload)


def _dump_camera_states(extras_dir: str, unwrapped) -> None:
    """Dump static world pose + intrinsics for every Camera sensor (base camera is fixed per env)."""
    from isaaclab.sensors import Camera

    payload: dict[str, np.ndarray] = {}
    for name, sensor in unwrapped.scene.sensors.items():
        if not isinstance(sensor, Camera):
            continue
        data = sensor.data
        payload[f"{name}__pos_w"] = _torch_to_numpy(data.pos_w).astype(np.float32)
        payload[f"{name}__quat_w_world"] = _torch_to_numpy(data.quat_w_world).astype(np.float32)
        payload[f"{name}__intrinsic_matrices"] = _torch_to_numpy(data.intrinsic_matrices).astype(np.float32)
        payload[f"{name}__width"] = np.asarray(int(sensor.cfg.width), dtype=np.int64)
        payload[f"{name}__height"] = np.asarray(int(sensor.cfg.height), dtype=np.int64)
        payload[f"{name}__data_types"] = np.asarray(list(sensor.cfg.data_types), dtype=object)
        clip = getattr(getattr(sensor.cfg, "spawn", None), "clipping_range", (0.0, 1.0e6))
        payload[f"{name}__clipping_range"] = np.asarray(clip, dtype=np.float32)
    if payload:
        np.savez(os.path.join(extras_dir, "camera_state.npz"), **payload)


def _dump_model_actuators(extras_dir: str) -> None:
    """Dump the finalized Newton model's per-DOF joint gains (implicit-actuator PD, armature, limits).

    Isaac Lab writes actuator stiffness/damping onto the Newton model *after* finalize, so these
    arrays carry the live PD gains the standalone must replay. Stored for the whole model; the
    standalone slices the first env's ``jd_per`` entries.
    """
    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager.get_model()
    if model is None:
        return
    payload: dict[str, np.ndarray] = {}
    for attr in (
        "joint_target_ke",
        "joint_target_kd",
        "joint_armature",
        "joint_effort_limit",
        "joint_friction",
        "joint_dof_mode",
        "joint_dof_world_start",
        "joint_coord_world_start",
    ):
        array = getattr(model, attr, None)
        if array is None:
            continue
        with contextlib.suppress(Exception):
            payload[attr] = np.asarray(array.numpy())
    if payload:
        np.savez(os.path.join(extras_dir, "actuator_state.npz"), **payload)


def _dump_actuator_networks(extras_dir: str, unwrapped) -> None:
    """Copy any actuator-network checkpoints referenced by the robot config (e.g. LSTM nets)."""
    from isaaclab.utils.assets import read_file

    robot_cfg = getattr(unwrapped.cfg.scene, "robot", None)
    for actuator_cfg in (getattr(robot_cfg, "actuators", {}) or {}).values():
        network_file = getattr(actuator_cfg, "network_file", None)
        if not network_file:
            continue
        dst = os.path.join(extras_dir, os.path.basename(str(network_file)))
        if not os.path.exists(dst):
            with open(dst, "wb") as f:
                f.write(read_file(network_file).read())


def _write_reset_state(out_dir: str, env) -> None:
    """Reset once and dump a task-agnostic snapshot (robot, objects, commands, cameras, gains, obs).

    Each section is independent and failure-tolerant so a task missing one piece (e.g. no
    cameras, no command manager) still produces a usable bundle.
    """
    extras_dir = os.path.join(out_dir, EXTRAS_DIRNAME)
    os.makedirs(extras_dir, exist_ok=True)

    obs_dict, _ = env.reset()
    unwrapped = env.unwrapped

    for section in (
        lambda: _dump_robot_state(extras_dir, unwrapped),
        lambda: _dump_object_states(extras_dir, unwrapped),
        lambda: _dump_command_states(extras_dir, unwrapped),
        lambda: _dump_camera_states(extras_dir, unwrapped),
        lambda: _dump_model_actuators(extras_dir),
        lambda: _dump_actuator_networks(extras_dir, unwrapped),
    ):
        try:
            section()
        except Exception:
            logger.warning("Reset-state dump section failed (continuing):\n%s", traceback.format_exc())

    obs_payload = {key: _torch_to_numpy(value).astype(np.float32) for key, value in dict(obs_dict).items()}
    if obs_payload:
        np.savez(os.path.join(extras_dir, "initial_observations.npz"), **obs_payload)


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
    if args_cli.disable_fabric and hasattr(env_cfg, "sim"):
        env_cfg.sim.use_fabric = False
    if args_cli.capture_reset_state:
        # A one-shot reset capture does not benefit from CUDA-graph stepping, and eager graph
        # capture can trip Newton's lazy MuJoCo contact-mapping allocation (allocations are
        # forbidden mid-capture). Disable the graph so the first simulate runs eagerly.
        with contextlib.suppress(Exception):
            env_cfg.sim.physics.use_cuda_graph = False
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else getattr(agent_cfg, "seed", None)

    out_dir = os.path.abspath(os.path.expanduser(args_cli.output_dir))
    args_cli.output_dir = out_dir

    if args_cli.policy and not os.path.isfile(args_cli.policy):
        parser.error(f"--policy path does not exist: {args_cli.policy}")
    if args_cli.mdp and not os.path.isfile(args_cli.mdp):
        parser.error(f"--mdp path does not exist: {args_cli.mdp}")

    import gymnasium as gym

    with launch_simulation(env_cfg, args_cli):
        # Imported after the app launches (Omniverse import-ordering requirement). The live
        # cloner drains REPLICATION_QUEUE and calls NewtonReplicateContext.replicate(), so we
        # patch that method to intercept the merged clone mapping and export the bundle.
        from isaaclab_newton.cloner.replicate import NewtonReplicateContext

        original = NewtonReplicateContext.replicate
        hook = _make_capture_hook(env_cfg, args_cli, agent_cfg, original)
        NewtonReplicateContext.replicate = hook
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
            NewtonReplicateContext.replicate = original
            if env is not None:
                with contextlib.suppress(Exception):
                    env.close()

        logger.error("Cloner hook did not fire for task %r. Is the Newton preset active?", args_cli.task)
        return 2


if __name__ == "__main__":
    sys.exit(main())
