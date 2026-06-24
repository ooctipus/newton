# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dependency-free validation: standalone obs/depth vs captured ground truth + policy rollout."""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import torch
import warp as wp

_REPRO = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, _REPRO)

from loader import build_newton_from_bundle, load_bundle, load_mdp_class  # noqa: E402

BUNDLE = str(pathlib.Path(__file__).resolve().parent)
DEVICE = os.environ.get("REPRO_DEVICE", "cuda:1")


def main() -> int:
    bundle = load_bundle(BUNDLE)
    sim, env_origins = build_newton_from_bundle(bundle, num_envs=2, device=DEVICE)
    decimation = int(bundle.sim_cfg.get("decimation", 1))
    mdp = load_mdp_class(BUNDLE)(
        sim=sim,
        env_origins=env_origins,
        num_envs=env_origins.shape[0],
        physics_dt=sim.physics_dt,
        decimation=decimation,
        episode_length_s=float(bundle.sim_cfg.get("episode_length_s", 0.0)),
        device=DEVICE,
        extras_dir=bundle.extras_dir,
    )

    obs_1d, depth = mdp.get_observations()
    cap = np.load(os.path.join(bundle.extras_dir, "initial_observations.npz"))
    cap_1d = torch.as_tensor(np.concatenate([cap["policy"], cap["proprio"]], axis=-1), device=DEVICE)
    cap_img = torch.as_tensor(cap["base_image"], device=DEVICE)

    def report(name, a, b):
        err = (a - b).abs()
        print(f"{name:28s} shape={tuple(a.shape)} max_abs={err.max().item():.4e} mean_abs={err.mean().item():.4e}")

    # Diagnostics: captured joint order + reset fidelity.
    with np.load(os.path.join(bundle.extras_dir, "robot_state.npz"), allow_pickle=True) as rs:
        cap_joint_pos = torch.as_tensor(rs["joint_pos"], device=DEVICE)
    sa_joint_pos = mdp.entities.joint_pos(wp.to_torch(sim.state.joint_q))
    default = mdp._default_joint_pos
    print("default_joint_pos[0,:7] :", [round(v, 3) for v in default[0, :7].tolist()])
    print("abs joint_pos[0,:7]     :", [round(v, 3) for v in sa_joint_pos[0, :7].tolist()])
    print("(abs - default)[0,:7]   :", [round(v, 3) for v in (sa_joint_pos - default)[0, :7].tolist()])
    print("captured initial_obs jp[0,:7]:", [round(v, 3) for v in cap_1d[0, 46:53].tolist()])
    report("standalone vs captured robot_state.joint_pos", sa_joint_pos, cap_joint_pos)
    report("captured robot_state.joint_pos vs initial_obs.joint_pos (absolute)", cap_joint_pos, cap_1d[:, 46:69])
    from envs.mdp import observations as _obs

    body_q = wp.to_torch(sim.state.body_q)
    _, rq = _obs.body_pose_w(body_q, mdp.entities.robot_root_body_idx)
    _, oq = _obs.body_pose_w(body_q, mdp.entities.object_body_idx)
    print("robot_root_quat_w[0] (xyzw):", rq[0].tolist(), " object_quat_w[0]:", oq[0].tolist())

    print("=== obs parity (standalone vs captured initial_observations) ===")
    report("obs_1d (policy+proprio)", obs_1d, cap_1d)
    # Per-segment breakdown of the 1D obs (policy group, then proprio = contact, jpos, jvel, tips).
    segs = [("object_quat_b", 0, 4), ("target_pose", 4, 11), ("last_action", 11, 34),
            ("contact", 34, 46), ("joint_pos", 46, 69), ("joint_vel", 69, 92), ("hand_tips", 92, 157)]  # fmt: skip
    for nm, lo, hi in segs:
        report(f"  {nm}", obs_1d[:, lo:hi], cap_1d[:, lo:hi])
    report("base_image (depth)", depth, cap_img)
    raw = wp.to_torch(mdp.camera.render(sim.state))
    print(
        f"raw depth: min={raw.min().item():.4f} max={raw.max().item():.4f} mean={raw.mean().item():.4f} "
        f"frac<=0={(raw <= 0).float().mean().item():.3f}"
    )
    print(f"std norm depth: min={depth.min().item():.4f} max={depth.max().item():.4f} mean={depth.mean().item():.4f}")
    print(f"cap depth:      min={cap_img.min().item():.4f} max={cap_img.max().item():.4f} mean={cap_img.mean().item():.4f}")
    print("cam pos_w[0]:", mdp._camera_state["base_camera__pos_w"][0].tolist())

    print("=== policy action: captured obs vs standalone obs ===")
    a_cap = mdp.policy.act(cap_1d, [cap_img])
    print(f"  on CAPTURED obs: action[0,:6]={[round(v, 3) for v in a_cap[0, :6].tolist()]} |mean|={a_cap.abs().mean().item():.4f}")
    o1d, dimg = mdp.get_observations()
    a0 = mdp.policy.act(o1d, [dimg])
    print(f"  on STANDALONE obs: action[0,:6]={[round(v, 3) for v in a0[0, :6].tolist()]} |mean|={a0.abs().mean().item():.4f}")
    # Feed standalone 1D obs but captured image (and vice versa) to localize the discrepancy.
    a_mix1 = mdp.policy.act(cap_1d, [dimg])
    a_mix2 = mdp.policy.act(o1d, [cap_img])
    print(f"  cap_1d + std_img: |mean|={a_mix1.abs().mean().item():.4f}   std_1d + cap_img: |mean|={a_mix2.abs().mean().item():.4f}")

    print("=== repro.py-mirror rollout (forward + reset_done, like the real entrypoint) ===")
    from envs.mdp.terminations import abnormal_robot_state, object_out_of_bound

    n_abnormal = 0
    n_oob = 0
    nan_seen = False
    for step in range(60):
        mdp.act()
        for _ in range(decimation):
            mdp.apply_actuator()
            sim.step()
        # Break the termination into its two causes for diagnosis.
        jv = mdp.entities.joint_vel(wp.to_torch(sim.state.joint_qd))
        bq = wp.to_torch(sim.state.body_q)
        opos, _ = _obs.body_pose_w(bq, mdp.entities.object_body_idx)
        ab = abnormal_robot_state(jv, mdp._joint_vel_limits)
        oob = object_out_of_bound(opos, mdp.env_origins)
        n_abnormal += int(ab.sum().item())
        n_oob += int(oob.sum().item())
        mdp.forward()
        mdp.reset_done()
        if torch.isnan(jv).any():
            nan_seen = True
        if step % 5 == 0 or nan_seen:
            rel = (opos - mdp.env_origins)[0]
            print(
                f"step {step:2d}: |act|={mdp.last_action.abs().mean().item():.2f} "
                f"|jvel|max={jv.abs().max().item():.2f} abnormal={int(ab.sum())} oob={int(oob.sum())} "
                f"obj_rel[0]={[round(v, 2) for v in rel.tolist()]}"
            )
        if nan_seen:
            break
    print(f"ROLLOUT SUMMARY: 60 steps, abnormal_terms={n_abnormal}, oob_terms={n_oob}, nan_seen={nan_seen}")

    print("=== reset path (procedural) ===")
    mdp.reset(torch.arange(mdp.num_envs, device=DEVICE), to_captured=False)
    body_q = wp.to_torch(sim.state.body_q)
    opos, _ = _obs.body_pose_w(body_q, mdp.entities.object_body_idx)
    rel = opos - mdp.env_origins
    print(f"post-reset object rel-to-env pos[0]={[round(v, 3) for v in rel[0].tolist()]}")
    o1d, dimg = mdp.get_observations()
    print(f"post-reset obs nan={torch.isnan(o1d).any().item()} action|mean|={mdp.policy.act(o1d, [dimg]).abs().mean().item():.4f}")
    print("OK: standalone rollout completed without error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
