# Newton Repro

Capture an Isaac Lab RL task into a portable **bundle**, then replay it in **pure
Newton + MuJoCo-Warp + Torch** — no Isaac Lab, no Omniverse Kit.

Use it to inspect a trained policy's physics behaviour, profile the Newton solver,
and hand a self-contained repro to the Newton team.

## Two phases

| Phase | Needs Kit? | Entry point |
|-------|-----------|-------------|
| **Capture** — freeze a task's model + reset state into a bundle | yes (Isaac Lab) | `capture/capture.py` |
| **Replay** — rebuild the Newton model from a bundle and step it | **no** (Kit-free) | `repro.py` |

The replay path (`repro.py`, `loader.py`, `newton_sim.py`, `replicate.py`, `profiling.py`)
imports **only** `newton` / `warp` / `torch` / `pxr` — never `isaaclab` or `omni`. Keep it that way.

## Replay

```bash
python scripts/newton_repro/repro.py --env factory_nut_thread --num_envs 4 --steps 400
```

`--env` accepts a task name under `tasks/` or a bundle directory path.

| Flag | Purpose |
|------|---------|
| `--num_envs N` | Replay the first N captured envs (any N ≤ captured; `model_state` is sliced to the env-major prefix). |
| `--device` | Newton device (default `cuda:0`). |
| `--steps` | Control steps to run. |
| `--policy NAME` | Task policy variant; factory has `tap` / `hammer` (`extras/policy_<name>.pt`). |
| `--headless` | Use `ViewerNull` (no window). `--record PATH` writes a Newton `ViewerFile` instead. |
| `--set path=value` | Override any `sim_cfg` leaf by dotted path (repeatable). The full config + paths are printed at startup. e.g. `--set num_substeps=8 --set collision.rigid_contact_max=8000000`. |
| `--profile [--topk K]` | Re-exec under Nsight Systems (graphed, full speed) and print the top-K GPU kernels by total time, labelled with their Warp module. |
| `--capture_graph` | Capture the physics step into a Warp CUDA graph (on by default). |

`sim_cfg` namespaces: top-level scalars (`physics_dt`, `decimation`, `num_substeps`,
`use_mujoco_contacts`), `solver.*` (MuJoCo-Warp: `njmax`, `nconmax`, `iterations`, `cone`, …),
`collision.*` (`rigid_contact_max`, `max_triangle_pairs`, …), `shape.*` (`ke`, `kd`).

## Bundle format (`tasks/<task>/extras/`)

| File | Contents |
|------|----------|
| `stage.usd` | Captured USD stage (env 0 + source prims), replayed verbatim. |
| `clone_plan.json` | Sources / destinations / clone mask / env spacing / sites. |
| `env_origins.npy` | `(num_envs, 3)` world positions of each env. |
| `sim_cfg.yaml` | The single runtime config the loader consumes (see namespaces above). |
| `model_state.npz` | Live post-reset model arrays (friction, inertia, joint PD gains) the EventManager/actuator setup write *after* the cloner hook; applied **before** the solver compiles. |
| `reset_state.npy` | Per-env robot root + dof reset state. |
| `initial_observations.npz`, `robot_state.npz`, `command_state.npz` | Captured reset observations / asset states (task-specific). |
| `policy_*.pt` | TorchScript policy variants (normalizer baked in). |

`mdp.py` lives in `tasks/<task>/` (not `extras/`) and holds the replay logic.

## Task MDP contract

Each `tasks/<task>/mdp.py` defines a top-level `MDP` class that `repro.py` drives:

```python
class MDP:
    def __init__(self, sim, env_origins, num_envs, physics_dt, decimation,
                 episode_length_s, device, extras_dir, policy=...): ...
    def act(self): ...               # run the policy, store control targets
    def apply_actuator(self): ...    # write per-substep control (called `decimation`× / step)
    def forward(self): ...           # advance counters; return (terminated, truncated)
    def reset_done(self): ...        # reset envs whose episode ended
    def reset(self, env_ids): ...    # restore the captured reset state for env_ids
    def log_visuals(self, viewer): ...  # optional debug geometry
```

Per control step `repro.py` runs:
`act()` → `decimation`×(`apply_actuator()`; `sim.step()`) → `forward()` → `reset_done()`.

`--policy` is passed to `__init__` only for MDPs that declare a `policy` parameter.

## Capture (Kit-side)

```bash
env_isaaclab/bin/python scripts/newton_repro/capture/capture.py \
    --task Isaac-Factory-v0 --num_envs 4 presets=nut_thread_m16,franka,newton_mjwarp,eval \
    --output_dir scripts/newton_repro/tasks/factory_nut_thread \
    --policy <policy.pt> --mdp scripts/newton_repro/tasks/factory_nut_thread/mdp.py \
    --capture_reset_state --no_simplify_meshes --headless
```

- `--capture_reset_state` resets once and dumps the reset/observation extras.
- `--no_simplify_meshes` preserves per-shape SDF approximation (required for SDF assets like the
  factory nut/bolt — convex-hull simplification would break the rebuild).
- `--no_verify_model` skips the live-vs-rebuilt parity check (use for very large `--num_envs`).

`capture/exporter.py` writes the bundle; everything under `extras/` is overwritten, hand-written
siblings (e.g. `mdp.py`) are left intact.

## Layout

```
repro.py          replay CLI / loop
loader.py         load a bundle -> NewtonSim (+ apply_overrides, print_sim_cfg)
newton_sim.py     Newton model + SolverMuJoCo/CollisionPipeline wrapper + CUDA graph
replicate.py      rebuild the cloned Newton model from the USD stage + clone plan
clone_plan.py     ClonePlan / SiteRequest dataclasses
profiling.py      FpsMeter + the --profile nsys top-k kernel report
benchmark.py      num_envs sweep (throughput / memory / top kernels) -> PDF scaling report
capture/          capture.py (Kit-side) + exporter.py (writes the bundle)
envs/             Kit-free reimplementations (math, events, sensors) used by task MDPs
tasks/<task>/     per-task mdp.py + extras/ bundle (+ optional kernels.py / networks.py)
test/             pytest (bundle round-trip, math parity, import contract, events)
```

## Tests

```bash
./isaaclab.sh -p -m pytest scripts/newton_repro/test/<file>.py
```
