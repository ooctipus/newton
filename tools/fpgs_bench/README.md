# Compare FPGS and MJWarp with the existing Isaac Lab harness

This driver compares the handoff task recipes without editing Isaac Lab or tuning physics parameters. It loads `scripts/benchmarks/fpgs_profile/compare_gpus.py` from an explicit Lab checkout and uses that checkout's unchanged capture script, analyzer, source-digest helper, and result aggregation. Existing Lab recipes are preserved; Newton-only aliases are merged into an isolated copy, with conflicting definitions rejected. The driver and its CPU tests require only Python's standard library.

Prepare the selected Lab `.venv` beforehand with compatible Newton, Warp, MuJoCo, and MuJoCo Warp dependencies. `uv`, Nsight Systems (`nsys`), `nvidia-smi`, Bash, and dedicated idle GPUs must be available. Lab's graph-aware handoff harness must support `FPGS_NSYS_TRACE_MODE=graph` and strict direct-graph analysis. The driver uses `UV_NO_SYNC=1`; it does not install or synchronize dependencies. A CPU preflight queries the actual Lab uv interpreter for backend versions, verifies `sys.prefix` is the selected Lab `.venv`, and verifies each selected `newton.__file__`, with CUDA hidden. Inherited uv project/directory/isolated-runtime overrides are removed.

Example, from a Newton checkout containing this directory:

```sh
uv run --no-project --python /path/to/isaaclab/.venv/bin/python \
  python tools/fpgs_bench/compare_backends.py \
  --isaaclab /path/to/isaaclab \
  --fpgs /path/to/fpgs-newton \
  --mjwarp /path/to/mjwarp-newton \
  --gpus 0 1 --task ant --task humanoid \
  --repeats 3 --num-envs 16384 --warmup-steps 200 \
  --steps 40 --profile-steps 40 \
  --fpgs-env 0:FEATHER_PGS_DENSE_ROW_BUDGETS=1 \
  --fpgs-env 0:FEATHER_PGS_DENSE_ROW_REGISTERS=1 \
  --fpgs-env 1:FEATHER_PGS_DENSE_ROW_BUDGETS=1 \
  --fpgs-env 1:FEATHER_PGS_DENSE_ROW_REGISTERS=1 \
  --output-dir /path/outside/all/checkouts/backend-comparison
```

The example explicitly enables the current cached dense-row implementation on both GPUs; use a Newton revision containing those flags. Omit those four overrides to measure the selected revision's defaults. The two Newton paths may be the same checkout when comparing backends from one revision. Task names come from the selected Lab helper (currently `ant`, `humanoid`, `franka`, `anymald`, `allegro`, `g1`, `kuka`, and `cartpole`). Repeated `--task` options preserve order and remove duplicates. Both arms use seed zero. FPGS receives the existing FPGS recipe's solver attributes and environment settings; MJWarp receives its own existing backend defaults, with no FPGS attributes or solver flags inherited.

Explicit per-device optimization flags can be added only on the FPGS side, for example `--fpgs-env 0:FEATHER_PGS_DENSE_ROW_BUDGETS=1`. The accepted prefixes are `FEATHER_PGS_` and `NEWTON_NARROW_PHASE_`. There are no generic solver/physics-attribute overrides; the driver does not judge whether a manually selected feature flag preserves numerical semantics. All inherited `FEATHER_`, `NEWTON_`, and `FPGS_PROBE_` flags are discarded before each child launch.

The Newton-only alias `--task keyboard-so101` selects `IsaacContrib-Keyboard-SO101`, the SO101 robot typing a procedural keyboard, not keyboard teleoperation. It adds no solver attributes or task-specific flags on either backend. This is a heavier scene with a 108-DOF keyboard plus the 6-DOF robot; its environment default is 4,096 worlds, while this driver's global default remains 16,384. Start with an explicit `--num-envs 32` smoke test, then try `--num-envs 4096` before considering 16K. Do not reuse Anymal/Allegro parallel-row or in-kernel-response flags. The task builds an 8,192-snapshot IK reset buffer on first reset and can perform additional IK/forward work during later resets, so wall timings can have substantial reset overhead. Headless benchmarking needs no human input; the SO101 USD and dependencies must be available. FPGS and MJWarp retain their different task-preset solver settings, so their timing ratio is not solver-accuracy parity.

Each backend batch starts one process per selected GPU concurrently, waits for the full batch, and then runs the other backend. Backend order reverses on alternate rounds (A/B, B/A, A/B for three rounds); no two profilers share one selected GPU. Existing output directories and all output locations inside source checkouts are rejected. Recorded process groups receive TERM even if their leader has already exited, then KILL if group members survive the bounded grace period; direct children are reaped. This also runs after spawn failures, interrupts, or SIGTERM. Cleanup signals are recorded in the manifest. SIGKILL or machine failure cannot be handled by a Python process; inspect remaining processes before resuming after either event.

`manifest.json` records commands, explicit flags, GPU UUIDs, hostname, driver and Lab runtime package versions, exact source commits and dirty-tree digests (including untracked files), and driver/helper/harness hashes. It checks every guarded source before and after batches. `summary.json` retains graph medians/ranges/spreads, separate auxiliary-graph time, and unprofiled wall medians/ranges. `ratios.json` is written only after every required result passes the unchanged Lab analyzer/result checks and every round completes. Check `manifest.json` has `status: complete`; failed manifests and raw captures remain available for diagnosis.

Ratios compare physics-graph time, not equivalent solver accuracy or identical trajectories. Contact counts are diagnostic, not parity proof. Auxiliary sensor graphs are excluded from physics time and reported separately. Unprofiled environment-step wall time includes resets, events, and host work and is not reinforcement-learning training throughput. Cross-device simulations may follow different trajectories; per-device backend timings are not a same-input solver parity test.

Run the portable CPU tests without Nsight Systems, a GPU, or Isaac Lab:

```sh
uv run --no-project python -m unittest discover -s tools/fpgs_bench -p test_compare_backends.py -v
```

The tests mock all external process and GPU operations. They cover recipe/environment isolation, alternating paired starts, source and import guards, strict result-error propagation, incomplete results, output protection, and child cleanup failures. GPU measurement and combined-physics correctness remain separate validation gates.
