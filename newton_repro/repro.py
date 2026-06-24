# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a captured Newton repro bundle without importing Isaac Lab."""

from __future__ import annotations

import argparse
import pathlib
import sys

import newton.viewer as newton_viewer

_REPRO_DIR = str(pathlib.Path(__file__).resolve().parent)
_TASKS_DIR = pathlib.Path(_REPRO_DIR) / "tasks"
if _REPRO_DIR not in sys.path:
    sys.path.insert(0, _REPRO_DIR)

from loader import build_newton_from_bundle, load_bundle, load_mdp_class  # noqa: E402


def _resolve_bundle_dir(env: str) -> str:
    """Resolve ``--env`` as either a bundle path or a task name."""
    env_path = pathlib.Path(env).expanduser()
    if env_path.exists():
        return str(env_path.resolve())

    if not env_path.is_absolute():
        task_path = _TASKS_DIR / env_path
        if task_path.exists():
            return str(task_path.resolve())

    task_names = sorted(path.name for path in _TASKS_DIR.iterdir() if path.is_dir() and not path.name.startswith("__"))
    available = ", ".join(task_names) if task_names else "none"
    raise FileNotFoundError(
        f"--env must be a bundle path or a task name under {_TASKS_DIR}; got {env!r}. Available tasks: {available}."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a Newton repro bundle without Isaac Lab.")
    parser.add_argument("--env", required=True, help="Bundle directory or task name under scripts/newton_repro/tasks.")
    parser.add_argument("--num_envs", type=int, default=None, help="Replay at most this many captured environments.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Newton device.")
    parser.add_argument(
        "--headless", action="store_true", default=False, help="Use ViewerNull instead of opening ViewerGL."
    )
    parser.add_argument("--record", type=str, default=None, help="Optional Newton ViewerFile output path.")
    parser.add_argument("--steps", type=int, default=1000, help="MDP control steps to run.")
    parser.add_argument(
        "--capture_graph", action="store_true", default=False, help="Capture Newton step with Warp CUDA graph."
    )
    args = parser.parse_args()

    env_dir = _resolve_bundle_dir(args.env)
    bundle = load_bundle(env_dir)
    sim, env_origins = build_newton_from_bundle(bundle, num_envs=args.num_envs, device=args.device)
    decimation = int(bundle.sim_cfg.get("decimation", 1))
    step_dt = sim.physics_dt * decimation

    mdp_cls = load_mdp_class(bundle.bundle_dir)
    mdp = mdp_cls(
        sim=sim,
        env_origins=env_origins,
        num_envs=env_origins.shape[0],
        physics_dt=sim.physics_dt,
        decimation=decimation,
        episode_length_s=float(bundle.sim_cfg.get("episode_length_s", 0.0)),
        device=args.device,
        extras_dir=bundle.extras_dir,
    )

    if args.capture_graph:
        sim.capture_graph()

    if args.record:
        viewer = newton_viewer.ViewerFile(args.record)
    elif args.headless:
        viewer = newton_viewer.ViewerNull(num_frames=args.steps)
    else:
        viewer = newton_viewer.ViewerGL(headless=False)
    viewer.set_model(sim.model)
    viewer.set_world_offsets((0.0, 0.0, 0.0))
    viewer.set_visible_worlds(range(env_origins.shape[0]))

    for step_idx in range(args.steps):
        if not viewer.is_running():
            break
        viewer.begin_frame(step_idx * step_dt)
        if viewer.should_step():
            mdp.act()
            for _ in range(decimation):
                mdp.apply_actuator()
                sim.step()
            mdp.forward()
            mdp.reset_done()
        viewer.log_state(sim.state)
        if sim.contacts is not None:
            viewer.log_contacts(sim.contacts, sim.state)
        mdp.log_visuals(viewer)
        viewer.end_frame()

    viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
