# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reproduce MuJoCo Warp solver NaN with ANYmal-D.

Uses IsaacLab's NewtonManager directly to build the model from
a cloner-exported stage, ensuring 100% identical setup.

Usage:
    # First export from IsaacLab:
    NEWTON_SAVE_CLONER_STAGE=nan_debug/cloner_export_512 \
        python scripts/reinforcement_learning/rsl_rl/play.py \
        --task=Isaac-Velocity-Rough-Anymal-D-v0 --num_envs=512 presets=newton

    # Then replay:
    python -m newton.examples robot_nan_repro \
        --stage nan_debug/cloner_export_512/cloner_stage.usd \
        --info nan_debug/cloner_export_512/cloner_info.json \
        --policy logs/rsl_rl/.../exported/policy.pt
"""

import json

import numpy as np
import torch
import warp as wp

import newton
import newton.examples
from newton import ModelBuilder
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
from newton.solvers import SolverMuJoCo
from pxr import Usd

NUM_DOFS = 12
ACTION_SCALE = 0.5
LSTM_URL = "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/IsaacLab/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt"

# ANYmal-D default standing pose (from IsaacLab ArticulationCfg)
ANYMAL_DEFAULT_JOINTS = [0.03, 0.4, -0.8, -0.03, 0.4, -0.8,
                          0.03, -0.4, 0.8, -0.03, -0.4, 0.8]


def _quat_rotate_inverse(q, v):
    q_w = q[..., 3]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def _build_from_export(stage, info):
    """Exact replica of IsaacLab's _build_newton_builder_from_mapping."""
    import inspect

    sources = info["sources"]
    positions = torch.tensor(info["positions"])
    quaternions = torch.tensor(info["quaternions"])
    up_axis = info["up_axis"]
    simplify_meshes = info["simplify_meshes"]
    num_envs = info["num_envs"]
    resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]

    builder = ModelBuilder(up_axis=up_axis)
    builder.add_usd(stage, ignore_paths=info["ignore_paths"], schema_resolvers=resolvers)

    env0_pos = positions[0]
    protos = {}
    for src_path in sources:
        p = ModelBuilder(up_axis=up_axis)
        SolverMuJoCo.register_custom_attributes(p)
        p.add_usd(stage, root_path=src_path, load_visual_shapes=True,
                   skip_mesh_approximation=True, schema_resolvers=resolvers)
        if simplify_meshes:
            p.approximate_meshes("convex_hull", keep_visual_shapes=True)
        protos[src_path] = p

    for col in range(num_envs):
        builder.begin_world()
        delta_pos = (positions[col] - env0_pos).tolist()
        builder.add_builder(protos[sources[0]],
                             xform=wp.transform(delta_pos, quaternions[col].tolist()))
        builder.end_world()

    # Solver + collision from export
    solver_cfg = dict(info.get("solver_cfg", {}))
    solver_cfg.pop("solver_type", None)
    valid_args = set(inspect.signature(SolverMuJoCo.__init__).parameters.keys()) - {"self", "model"}
    solver_cfg = {k: v for k, v in solver_cfg.items() if k in valid_args}

    collision_cfg = dict(info.get("collision_cfg", {}))
    collision_cfg.pop("sdf_hydroelastic_config", None)

    physics_dt = info.get("physics_dt", 0.005)
    num_substeps = info.get("num_substeps", 1)
    sim_dt = physics_dt / num_substeps

    return builder, num_envs, solver_cfg, collision_cfg, sim_dt


class Example:
    def __init__(self, viewer, args):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.decimation = 4
        self.viewer = viewer
        self.device = wp.get_device()
        self.torch_device = "cuda" if self.device.is_cuda else "cpu"
        self.frame_count = 0

        with open(args.info) as f:
            info = json.load(f)
        stage = Usd.Stage.Open(args.stage)

        builder, self.num_envs, solver_cfg, collision_cfg, self.sim_dt = _build_from_export(stage, info)

        self.model = builder.finalize()
        print(f"[INFO] Model: {self.model.body_count} bodies, {self.model.shape_count} shapes, "
              f"{getattr(self.model, 'world_count', self.num_envs)} worlds")
        print(f"[INFO] Solver: {solver_cfg}")
        print(f"[INFO] Collision: {collision_cfg}")
        print(f"[INFO] sim_dt: {self.sim_dt}")

        self.solver = SolverMuJoCo(self.model, **solver_cfg)
        self.collision_pipeline = newton.CollisionPipeline(self.model, **collision_cfg)
        self.state_0 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

        jc_starts = self.model.joint_coord_world_start.numpy()
        jd_starts = self.model.joint_dof_world_start.numpy()
        self._jc_per = int(jc_starts[1]) - int(jc_starts[0])
        self._jd_per = int(jd_starts[1]) - int(jd_starts[0])

        # Set default standing pose for all envs
        jq_np = self.state_0.joint_q.numpy()
        for w in range(self.num_envs):
            jc0 = int(jc_starts[w])
            for j, val in enumerate(ANYMAL_DEFAULT_JOINTS):
                jq_np[jc0 + 7 + j] = val
        self.state_0.joint_q.assign(wp.array(jq_np, dtype=wp.float32))
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self._has_policy = False
        if args.policy:
            self._load_policy(args.policy)

        self.viewer.set_model(self.model)

    def _load_policy(self, policy_path):
        print(f"[INFO] Loading policy: {policy_path}")
        self._policy = torch.jit.load(policy_path, map_location=self.torch_device).eval()
        self._lstm = torch.hub.load_state_dict_from_url(
            LSTM_URL, map_location=self.torch_device, check_hash=False,
            file_name="anydrive_3_lstm_jit.pt")
        self._lstm_h = torch.zeros(2, self.num_envs * NUM_DOFS, 8, device=self.torch_device)
        self._lstm_c = torch.zeros(2, self.num_envs * NUM_DOFS, 8, device=self.torch_device)

        self._joint_pos_initial = torch.tensor(ANYMAL_DEFAULT_JOINTS,
                                                device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        self._act = torch.zeros(self.num_envs, NUM_DOFS, device=self.torch_device)
        self._gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=self.torch_device)
        self._commands = torch.zeros(self.num_envs, 3, device=self.torch_device)
        self._commands[:, 0] = 1.0
        self._targets = self._joint_pos_initial.expand(self.num_envs, -1).clone()
        self._has_policy = True
        print("[INFO] Policy + LSTM loaded")

    def _apply_policy(self):
        if self.frame_count % 100 == 0:
            self._commands[:, 0].uniform_(-1.5, 1.5)
            self._commands[:, 1].uniform_(-1.0, 1.0)
            self._commands[:, 2].uniform_(-1.0, 1.0)

        jq = wp.to_torch(self.state_0.joint_q).reshape(self.num_envs, self._jc_per)
        jqd = wp.to_torch(self.state_0.joint_qd).reshape(self.num_envs, self._jd_per)
        with torch.no_grad():
            vel_b = _quat_rotate_inverse(jq[:, 3:7], jqd[:, :3])
            a_vel_b = _quat_rotate_inverse(jq[:, 3:7], jqd[:, 3:6])
            grav = _quat_rotate_inverse(jq[:, 3:7], self._gravity_vec.expand(self.num_envs, -1))
            obs = torch.cat([vel_b, a_vel_b, grav, self._commands,
                             jq[:, 7:] - self._joint_pos_initial, jqd[:, 6:], self._act], dim=1)
            self._act = self._policy(obs)
            self._targets = self._joint_pos_initial + ACTION_SCALE * self._act

    def _apply_lstm_torques(self):
        if not self._has_policy:
            return
        jq = wp.to_torch(self.state_0.joint_q).reshape(self.num_envs, self._jc_per)
        jqd = wp.to_torch(self.state_0.joint_qd).reshape(self.num_envs, self._jd_per)
        with torch.no_grad():
            sea_in = torch.stack([(self._targets - jq[:, 7:]).flatten(),
                                   jqd[:, 6:].flatten()], dim=-1).unsqueeze(1)
            torques, (self._lstm_h[:], self._lstm_c[:]) = self._lstm(
                sea_in, (self._lstm_h, self._lstm_c))
            torques = torques.reshape(self.num_envs, NUM_DOFS).clamp(-80.0, 80.0)
            jf = wp.to_torch(self.control.joint_f).reshape(self.num_envs, self._jd_per)
            jf[:, :6] = 0.0
            jf[:, 6:] = torques

    def simulate(self):
        # Match IsaacLab's _simulate_physics_only ordering
        self._apply_lstm_torques()
        self.collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.sim_dt)
        self.state_0.clear_forces()

    def step(self):
        if self._has_policy:
            self._apply_policy()
        for _ in range(self.decimation):
            self.simulate()
        self.sim_time += self.frame_dt
        self.frame_count += 1

        jq = wp.to_torch(self.state_0.joint_q).reshape(self.num_envs, self._jc_per)
        nan_mask = torch.isnan(jq).any(dim=1)
        if nan_mask.any():
            print(f"[NaN] frame={self.frame_count} worlds={torch.where(nan_mask)[0].tolist()}")

        # Reset fallen
        if self._has_policy:
            fallen = (jq[:, 6].abs() < 0.7) | (jq[:, 2] < -10.0) | nan_mask
            if fallen.any():
                jq[fallen, 2] = 0.62
                jq[fallen, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.torch_device)
                jq[fallen, 7:] = self._joint_pos_initial
                jqd = wp.to_torch(self.state_0.joint_qd).reshape(self.num_envs, self._jd_per)
                jqd[fallen] = 0.0
                self._act[fallen] = 0.0
                fallen_idx = torch.where(fallen)[0]
                offsets = (fallen_idx.unsqueeze(1) * NUM_DOFS +
                           torch.arange(NUM_DOFS, device=self.torch_device).unsqueeze(0)).flatten()
                self._lstm_h[:, offsets] = 0.0
                self._lstm_c[:, offsets] = 0.0
                newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        if self.frame_count % 100 == 0:
            z = wp.to_torch(self.state_0.joint_q).reshape(self.num_envs, self._jc_per)[:, 2]
            print(f"[step] frame={self.frame_count} z=[{z.min():.2f},{z.median():.2f},{z.max():.2f}]", flush=True)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        pass

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--stage", type=str, required=True)
        parser.add_argument("--info", type=str, required=True)
        parser.add_argument("--policy", type=str, default=None)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
