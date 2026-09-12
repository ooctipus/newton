# Compare FPGS and MJWarp with the existing Isaac Lab harness

This driver compares the handoff task recipes without editing Isaac Lab or tuning physics parameters. It loads `scripts/benchmarks/fpgs_profile/compare_gpus.py` from an explicit Lab checkout and uses that checkout's unchanged capture script, analyzer, source-digest helper, and result aggregation. Existing Lab recipes are preserved; Newton-only aliases are merged into an isolated copy, with conflicting definitions rejected. The driver and its CPU tests require only Python's standard library.

Prepare the selected Lab `.venv` beforehand with compatible Newton, Warp, MuJoCo, and MuJoCo Warp dependencies. `uv`, Nsight Systems (`nsys`), `nvidia-smi`, Bash, and dedicated idle GPUs must be available. Lab's graph-aware handoff harness must support `FPGS_NSYS_TRACE_MODE=graph` and strict direct-graph analysis. The driver uses `UV_NO_SYNC=1`; it does not install or synchronize dependencies. A CPU preflight queries the actual Lab uv interpreter for backend versions, verifies `sys.prefix` is the selected Lab `.venv`, and verifies each selected `newton.__file__`, with CUDA hidden. Inherited uv project/directory/isolated-runtime overrides are removed.

Example after buffer calibration, from a Newton checkout containing this directory:

```sh
uv run --no-project --python /path/to/isaaclab/.venv/bin/python \
  python tools/fpgs_bench/compare_backends.py \
  --isaaclab /path/to/isaaclab \
  --fpgs /path/to/fpgs-newton \
  --mjwarp /path/to/mjwarp-newton \
  --gpus 0 1 --task ant --task humanoid \
  --repeats 3 --num-envs 16384 --warmup-steps 200 \
  --steps 40 --profile-steps 40 \
  --check-overflow \
  --fpgs-env 0:FEATHER_PGS_DENSE_ROW_BUDGETS=1 \
  --fpgs-env 0:FEATHER_PGS_DENSE_ROW_REGISTERS=1 \
  --fpgs-env 1:FEATHER_PGS_DENSE_ROW_BUDGETS=1 \
  --fpgs-env 1:FEATHER_PGS_DENSE_ROW_REGISTERS=1 \
  --output-dir /path/outside/all/checkouts/backend-comparison
```

The example explicitly enables the current cached dense-row implementation on both GPUs; use a Newton revision containing those flags and the FPGS sticky capacity-check API. Omit those four feature overrides to measure the selected revision's defaults. Add the separately calibrated `--capacity` settings described below; the example does not supply universal safe capacities. The two Newton paths may be the same checkout when comparing backends from one revision. Task names come from the selected Lab helper (currently `ant`, `humanoid`, `franka`, `anymald`, `allegro`, `g1`, `kuka`, and `cartpole`). Repeated `--task` options preserve order and remove duplicates. Both arms use seed zero. FPGS receives the existing FPGS recipe's solver attributes and environment settings; MJWarp receives its own existing backend defaults, with no FPGS attributes or solver flags inherited.

Explicit per-device optimization flags can be added only on the FPGS side, for example `--fpgs-env 0:FEATHER_PGS_DENSE_ROW_BUDGETS=1`. The accepted prefixes are `FEATHER_PGS_` and `NEWTON_NARROW_PHASE_`. There are no generic solver/physics-attribute overrides; the driver does not judge whether a manually selected feature flag preserves numerical semantics. All inherited `FEATHER_`, `NEWTON_`, and `FPGS_PROBE_` flags are discarded before each child launch.

Calibrated buffer sizes can be passed explicitly with repeated
`--capacity TASK:BACKEND:FIELD=VALUE` options. Each setting applies to both GPUs
for that task/backend and is recorded in the manifest. FPGS accepts
`dense_max_constraints`, `mf_max_constraints`, and `propagation_max_constraints`;
MJWarp accepts `njmax` and `nconmax`. Both accept Newton collision settings
`rigid_contact_max` and `max_triangle_pairs`. Sizes must be positive integers.
This does not admit timestep, substep, iteration, tolerance, or contact-law changes.
The additional Newton `broad_phase_output_max` capacity requires
`--check-overflow`. The unchanged Lab configuration rejects that new key,
including Hydra `+` overrides, so the checked entrypoint instead passes this
single allowlisted option directly to `Newton.CollisionPipeline` construction.
It does not change Lab files or configuration objects. The original constructor
validates supported internal explicit pipelines; conflicting values, unsupported
owners and Newton versions without this option fail. The full input pair list
is still traversed. The report records its length, requested/effective output
capacity and effective narrow-phase split/sparse flags and launch-grid size.
Shrinking output capacity can change dispatch as well as allocation, so measure
the resulting pipeline rather than assuming storage-only behavior. Construction
interception is restored on exit and adds no per-step callback or kernel.

Calibrate before benchmarking: measure unclamped demand through warmup,
contact-rich states and held-out resets, then verify zero overflow with a
documented reserve. MJWarp `njmax` is per-world; its `nconmax` sizes a pooled
contact buffer of `nconmax * num_worlds`, not a hard per-world contact cap.
Newton `rigid_contact_max` is global. Do not reuse a global size at a different
world count without calibration, and check actual allocated capacities because
constructors can enlarge requested values. FPGS row storage and generated
kernels must be constructed together; never change live capacity attributes.
Detailed `row_watermark=True` diagnostics add device work and are not timing
samples. Internal Newton collision queues reset demand counters each call:
calibrate their unclamped high-water marks and overflows with a full-run
observer, including warmup and held-out contact/reset states, before timing.
Two boundary snapshots cannot establish those transient queues' safety.

Use `--check-overflow` for the resulting timing comparison. Newton's
`nsys_checked.sh` calls the selected Lab harness unchanged through
`checked_capture.py`, wrapping only its existing post-warmup and post-profile
metadata reads. At both boundaries it executes
`SolverFeatherPGS.check_constraint_capacity()` for FPGS, or reads every MJWarp
sticky warning/overflow bit. When the manager has a Newton collision pipeline,
it also executes that actual owner's `NarrowPhase.check_buffer_capacity()`
after saving all 13 sticky verifier flags. This includes broad-phase, query,
GJK/manifold, mesh/triangle and contact buffers, plus global-reducer hash load
and insertion failures; a hash-load warning need not imply lost contacts.
No flags are cleared or suppressed, no additional
physics kernels are launched, and no checks run inside the timing windows.
FPGS checks dense, matrix-free and propagation row capacity plus incoming
contact-prefix storage; the separate narrow-phase checker covers its transient
queues. Checked mode requires narrow verification to remain enabled and rejects
particle/soft-contact, hydroelastic, body-pair reduction and post-narrow
sort/matching owners: their additional storage is not covered by this verifier.
A missing Newton pipeline is marked not applicable only for actual internal
MJWarp collision; an external-collision backend missing its pipeline fails.
MJWarp rejects every nonzero bit, including solver/line-search
iteration exhaustion and unknown future bits, not just storage overflow.
An older FPGS/narrow-phase revision without the required checker, disabled MJ warnings, or MJ
sleeping/resettable warning history is explicitly unsupported in checked mode.

Each capture writes `capture_checks.json`, retaining per-boundary flags and
realized capacities, and each accepted run embeds it in `manifest.json`.
The parent requires both checks to pass, with matching backend, output paths
and pinned Lab/wrapper hashes, before accepting timing or writing final ratios.
A warning-affected run remains failed even if its states are finite; changing
iteration budgets or suppressing warnings is not a clean comparison.
This checks available sticky flags within the admitted owners, not numerical
convergence or universal coverage of every Newton collision feature. Full-run
high-water calibration still determines appropriate sizes; boolean flags do
not estimate demand. Legacy behavior remains the default when `--check-overflow` is omitted:
the original Lab shell is used, and `status: complete` alone then establishes
neither capacity safety nor warning-free execution.

The Newton-only alias `--task keyboard-so101` selects `IsaacContrib-Keyboard-SO101`, the SO101 robot typing a procedural keyboard, not keyboard teleoperation. It adds no solver attributes or task-specific flags on either backend. This is a heavier scene with a 108-DOF keyboard plus the 6-DOF robot; its environment default is 4,096 worlds, while this driver's global default remains 16,384. Start with an explicit `--num-envs 32` smoke test, then try `--num-envs 4096` before considering 16K. Do not reuse Anymal/Allegro parallel-row or in-kernel-response flags. The task builds an 8,192-snapshot IK reset buffer on first reset and can perform additional IK/forward work during later resets, so wall timings can have substantial reset overhead. Headless benchmarking needs no human input; the SO101 USD and dependencies must be available. FPGS and MJWarp retain their different task-preset solver settings, so their timing ratio is not solver-accuracy parity.

Each backend batch starts one process per selected GPU concurrently, waits for the full batch, and then runs the other backend. Backend order reverses on alternate rounds (A/B, B/A, A/B for three rounds); no two profilers share one selected GPU. Existing output directories and all output locations inside source checkouts are rejected. Recorded process groups receive TERM even if their leader has already exited, then KILL if group members survive the bounded grace period; direct children are reaped. This also runs after spawn failures, interrupts, or SIGTERM. Cleanup signals are recorded in the manifest. SIGKILL or machine failure cannot be handled by a Python process; inspect remaining processes before resuming after either event.

`manifest.json` records commands, explicit flags, GPU UUIDs, hostname, driver and Lab runtime package versions, exact source commits and dirty-tree digests (including untracked files), and driver/helper/harness hashes. Checked mode additionally pins both Newton wrapper files and passes explicit Lab/Newton/source-hash environment values; inherited copies are removed. It checks every guarded source before and after batches. `summary.json` retains graph medians/ranges/spreads, separate auxiliary-graph time, and unprofiled wall medians/ranges. `ratios.json` is written only after every required result passes the unchanged Lab analyzer/result checks, the optional overflow checks, and every round completes. Check `manifest.json` has `status: complete`; failed manifests and raw captures remain available for diagnosis.

Ratios compare physics-graph time, not equivalent solver accuracy or identical trajectories. Contact counts are diagnostic, not parity proof. Auxiliary sensor graphs are excluded from physics time and reported separately. Unprofiled environment-step wall time includes resets, events, and host work and is not reinforcement-learning training throughput. Cross-device simulations may follow different trajectories; per-device backend timings are not a same-input solver parity test.

Run the portable CPU tests without Nsight Systems, a GPU, or Isaac Lab:

```sh
uv run --no-project python -m unittest discover -s tools/fpgs_bench -p test_compare_backends.py -v
uv run --no-project python -m unittest discover -s tools/fpgs_bench -p test_checked_capture.py -v
```

The tests mock all external process and GPU operations. They cover recipe/environment isolation, alternating paired starts, source and import guards, strict result-error propagation, checked helper routing, incomplete/failed or misbound overflow reports, output protection, and child cleanup failures. The checked-capture tests also run the shell against inert command stubs and execute a pinned fake harness to verify actual boundary hooks and unchanged argument forwarding. GPU measurement and combined-physics correctness remain separate validation gates.
