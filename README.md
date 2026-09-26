# Sparse FPGS benchmark evidence

This supporting evidence bundle is separate from the solver implementation PR. [results.md](results.md) contains the complete 20 matched task recipes; [results.json](results.json) retains exact sampling, unrounded measurements, matched configurations, checks, storage dimensions, and artifact hashes. The retained helpers are byte-for-byte copies of the files used by the captures. The adapters have only portable path/GPU admission changes; original capture-driver hashes and portable-adapter hashes are recorded separately.

## Source and scope

- Clean Newton baseline: `5238407d320823e71a5a4623c2c83c293d7b00fd`.
- Isaac Lab: `53ee6b44c2334341305dbdf385a3916c6b140799`, publicly reachable at `ooctipus/IsaacLab` branch `ooctipus/fpgs-contact-reset-20260913`, plus the included `isaaclab.patch`.
- Retained helpers: exact bytes from local Newton commit `9233cffd6a4729c71a864e2ec2621c6d6d6de604`. `compare_backends.py` was last changed at `020c213ed0b5c3c19f3d824104ed72c32a0cc27b`. At preparation time (September 26, 2026), both commits returned GitHub HTTP 422 (not publicly reachable), which is why their exact helper files are included. Existing public branch `ooctipus/fpgs-structural-progress-20260917` at `301b6688ea74a7372a007f03989664890f0d2540` has older, nonidentical Python helpers; it must not be silently substituted.
- The three Isaac Lab profiling dependencies (`compare_gpus.py`, `run_profiled.py`, `analyze_nsys.py`) are clean, unchanged files at the Lab pin above. The optional MJWarp line-search helper is not needed for this FPGS-only comparison.
- RTX PRO 6000 Blackwell Max-Q, GPU 0; 16,384 environments, seed 0, 200 warmup steps, a 40-step environment-loop timing window synchronized at its boundaries, then 40 separate main-physics-graph profiling steps. No per-step synchronization is added by the timing harness. The 20-task table is one discovery round, not repeated performance qualification. No video or visualizer. These are random-action environment FPS, not training FPS.

The primary comparison is inclusive environment-loop FPS (`16384 / loop_seconds`). Main-physics-graph times exclude eager environment work and, on rough-terrain tasks, the separately captured height-scanner sensor graph. Native task physics, friction features, solver budgets, and matched capacities are retained.

ANYmal-D flat and G1 rough additionally have three independent captures per revision: the initial discovery capture plus two balanced rounds. Round 2 runs main ANYmal, main G1, PR G1, PR ANYmal; round 3 reverses the revision order. Use the same entry point with `--task anymald` or `--task g1`, one fresh output directory per capture. Exact round summaries, checks, order, artifact hashes, and observed ranges are in `results.json`; these ranges are not confidence intervals.

## Files

- `benchmark.py`, `capture_adapter.py`, `launch_capture.py`: three thin adapters. Portable copies replace local paths with explicit `--isaaclab`, `--newton`, `--capacity-file`, and `--gpu-uuid` inputs and the bundled `helpers` directory. They enforce pinned source/driver hashes and select the existing checked capture implementation; they do not replace the environment or physics loop. The clean-main commit cannot be admitted as a dirty candidate. No capture driver was edited.
- `helpers/`: exact retained comparison/checking/launch files, including their original copyright notices.
- `capacities.json`: compact historical 4K FPGS capacity inputs, with the accepted SO101 Keyboard row capacity updated to 736. The original input hash and 704-to-736 adjustment are retained explicitly. Global collision storage is scaled with the world count; per-world row storage is not. Drawer is excluded; the twentieth task is the existing KukaAllegro reorientation task, using KukaAllegro's capacity recipe.
- `isaaclab.patch`: the pre-existing shared Lab changes and their untracked regression test/changelog. Neither Lab nor task physics was edited by this benchmark task.
- `provenance.json`: source and content hashes, plus validation results. Original Newton and Isaac Lab license texts are included alongside the retained source notices.

The first five baseline captures used an earlier thin-adapter revision, before the later CLI controls and sparse-storage observations were added. Their original adapter hashes are recorded separately. The retained timing/capture helpers and effective workload were unchanged. The original five captures do not contain the later storage observations; the direct clean-main storage examples therefore cite `repeat2-main`, paired with `candidate-checkpoint`, at the same clean commits and matched settings.

SO101 Keyboard's historical 704-row configuration failed the native overflow check (`705` requested rows in one world, dropped rows flagged); no FPS is accepted from that capture. Both matched 736-row retries passed. The default portable capacity file uses this accepted value, so a full 20-task invocation does not repeat the known failure. Other tasks retain their original capacities.

## Dirty Lab provenance

The matched Lab delta changes visual-ground-plane extent and restores the Keyboard task's zero joint-position reset seed. The latter is a task initialization change, not merely a visualization edit, and is applied identically to both arms. The additional regression test and changelog are included in the patch. Original capture source-delta SHA256: `cfd75b639b9f38b3b509853e8009a915c754fb4c4757c40d5264590ca5bca15a`.

That source digest also includes the untracked `.venv` symlink text pointing to `/home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910/.venv`. The environment itself is not bundled, and a portable checkout with a different environment path will have a different source digest. Runtime package versions and import paths are recorded separately in each local capture manifest.

External robot assets are not bundled. They are resolved from configured cached USD assets; the Lab commit alone does not pin every asset. In particular, this run's Franka root selects `Colliders="gripper_only"`; an older failed capture used a different root asset variant.

Inspected Franka asset hashes (not an exhaustive dependency audit): `franka_panda.usda` = `eaf27723ba1934d792c6876679eb75491bdff944ad108d4e59734fa07dc65809`; `payloads/base_physx_capsules_compact.usda` = `7c636f38f1eb47f34c89cf37126d4c1a9d569dcc99cfe9051e068855af3a43a9`. The latter was unchanged but was not the selected collider variant in this run. Both were found under the cached `Assets/Isaac/6.1/Isaac/IsaacLab/Robots/FrankaEmika/` asset root. These captures are comparisons under current matched assets, not numerical reproductions of the historical Franka workload.

## Checks and limits

Accepted captures passed exact source/import admission, stable driver hashes, finite-state boundaries, public constraint checks, retained sticky row-warning checks, and full-log warning rejection. Collision boundary snapshots and disabled row watermarks are not a lifetime no-drop certificate or convergence proof. Matching seeds does not establish identical trajectories. Sensor/host/graph scopes must not be summed unless their nonoverlap has been verified.

This compact bundle excludes environment dumps, credentials, asset blobs, full repository history, Nsight binaries, and large captures. Raw capture locators and SHA256 hashes are retained for local audit; the raw artifacts themselves are not distributed here.

## Reproduction entry point

Use existing local checkouts at the pins above, with `isaaclab.patch` applied to Lab, and the candidate at `d5c2177d2507ed12e87d6dc628373639a8e5231b`. The original environment was Python 3.12.14, Warp 1.17.0, Torch 2.11.0+cu130, MuJoCo/MuJoCo-Warp 3.12.0, Nsight Systems 2025.6.3.541, and NVIDIA driver 610.43.03. Reuse a compatible prepared Lab environment; the driver does not install or synchronize dependencies.

The single entry point is `benchmark.py`. For example, after setting `lab`, `newton`, `revision`, `rtx_uuid`, and `output` to explicit local values:

```bash
uv run --no-sync --project "$lab" python benchmark.py \
  --isaaclab "$lab" --newton "$newton" --commit "$revision" \
  --gpu-uuid "$rtx_uuid" --num-envs 16384 --warmup-steps 200 \
  --steps 40 --profile-steps 40 \
  --solver-attr "_kernel_overrides={'sparse_mass_matrix': True}" \
  --output "$output"
```

This runs all 20 tasks. Select `False` for clean-main controls and `True` for the candidate. The explicit hook is applied to the solver class before construction and its active representation is recorded. It does not force ineligible configurations onto the sparse path. `--plan-only` validates source/configuration without creating environments or GPU contexts. GPU 0 is the only supported device in this bounded reproduction driver; the UUID must match your explicit input and that GPU must be idle. Use fresh output directories.

For a subset, use repeated `--task` options, for example `--task anymald --task g1`. SO101 Keyboard automatically receives 736 rows from `capacities.json`; do not apply a global 736-row override to unrelated tasks. The actual captures used unchanged live drivers and an explicit per-task Keyboard override; the portable capacity file encodes the same effective configuration.

The separate Ant/Humanoid controls add only `--solver-attr "pgs_mode='matrix_free'"` on both revisions, with `--task ant` or `--task humanoid`. Ant retains 64 dense rows, 512 MF rows, one substep and mass refresh every step; Humanoid retains 128 dense rows, 512 MF rows, two substeps and mass refresh every two steps. Both keep environment decimation 2, eight GS iterations, current friction, augmented drives, interleaved scheduling, native joint limits and all other recorded features. Their matrix-free main → matrix-free PR ratios isolate the code change; split → matrix-free comparisons do not.
