# FPGS recipe selection: discovery and repeated evidence

Snapshot for PR preparation, 2026-09-19. The three-round `qualified_repeat3/` campaign and the fresh one-pass `terrain_mj_timing/` campaign are complete. Discovery results are retained separately below; repeated FPGS gains and the newly measured MJWarp comparison have their own tables and limitations. Drawer is excluded throughout.

## Measurement scope

FPGS-variant ratios are baseline FPGS time / candidate FPGS time: above 1 means faster. The separately labeled backend table instead uses MJWarp time / FPGS time. Both cards use 4,096 environments, seed 0, 200 warmup steps, 40 measured steps and 40 physics-profile steps. Physics graph time and unprofiled environment wall time are separate measurements; wall time is not end-to-end RL training. Each FPGS variant pair retains the same timestep, substeps, decimation, iteration allowances, topology and capacity overrides. Isaac Lab is unchanged.

## Execution-recipe discovery: all 13 tasks

Source: [execution_discovery/survey.json](/home/octi/Projects/fpgs-recipes-evidence-20260919/execution_discovery/survey.json), with per-task commands, actual recipe/path metadata, checks and source guards in `execution_discovery/<task>/manifest.json`. Both arms use Newton `4e42ebe2`; the candidate explicitly enables grouped dynamics and lazy kinematics, except G1 flat as described below.

| Task | RTX PRO6000 physics | GB300 physics | RTX PRO6000 wall | GB300 wall |
| --- | ---: | ---: | ---: | ---: |
| Ant | 1.07× | 1.02× | 0.87× | 1.04× |
| ANYmal-D rough | 1.02× | 1.01× | 1.01× | 1.01× |
| Cartpole | 0.92× | 0.89× | 0.95× | 0.97× |
| Cassie flat | 1.09× | 1.13× | 1.02× | 1.04× |
| Cassie rough | 1.04× | 1.03× | 1.00× | 1.00× |
| Franka reach | 1.16× | 1.10× | 1.13× | 1.07× |
| G1 flat* | 1.14× | 1.39× | 1.05× | 1.11× |
| Go2 flat | 1.11× | 1.14× | 1.06× | 1.07× |
| Go2 rough | 1.04× | 1.04× | 0.98× | 1.09× |
| H1 flat | 1.04× | 1.12× | 1.05× | 0.99× |
| H1 rough | 1.02× | 1.05× | 1.00× | 1.03× |
| Humanoid | 1.04× | 1.15× | 1.04× | 1.14× |
| UR10 reach | 1.10× | 1.08× | 1.01× | 0.96× |

*G1 flat is a separate grouped-plus-sparse-factor candidate, with lazy kinematics still false. The requested sparse factor became active on both cards; the requested kinetic-state owner did not. The existing row capacity stays 56, rather than being inflated to admit a different owner. This is not evidence for a kinetic-state gain or a grouped/lazy-only gain.

## Serial matrix-free discovery: three completed first-pass results

These are separate from the 13-task grouped/lazy discovery above. Both arms use Newton `4e42ebe2`; the only requested change is `pgs_mode=split` → `pgs_mode=matrix_free`. Actual solver metadata confirms that grouped dynamics and lazy kinematics remain false in both arms. Each result is one capture per arm/card, not repeated timing and not a comparison with MJWarp.

| Task / evidence | RTX physics ms, before → after | RTX physics gain | GB300 physics ms, before → after | GB300 physics gain | RTX wall gain | GB300 wall gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Ant](/home/octi/Projects/fpgs-recipes-evidence-20260919/serial_mf_discovery/ant/manifest.json) | 0.794 → 0.656 | 1.21× | 0.771 → 0.786 | 0.98× | 0.95× | 1.03× |
| [Cartpole](/home/octi/Projects/fpgs-recipes-evidence-20260919/serial_mf_discovery/cartpole/manifest.json) | 0.211 → 0.189 | 1.12× | 0.232 → 0.201 | 1.15× | 0.98× | 1.02× |
| [Humanoid](/home/octi/Projects/fpgs-recipes-evidence-20260919/serial_mf_discovery/humanoid/manifest.json) | 5.718 → 2.784 | 2.05× | 3.710 → 2.943 | 1.26× | 1.24× | 1.05× |

The numerical budgets are unchanged: Ant retains 8 PGS iterations / 1 substep, Cartpole 4 / 1, and Humanoid 8 / 2. All retain simulation dt 1/120 s and decimation 2; Humanoid's solver dt is 1/240 s. Actual metadata shows `mf_gs_parallel_rows=0` and `mf_gs_parallel_matrix_free=False` in both arms. The stored `mf_gs_parallel_sweeps=24` setting is inactive: this is serial matrix-free solving with the original 8/4/8 iteration allowances, not 24 additional sweeps.

Capacities also remain unchanged between arms: Ant raw/dense/matrix-free/propagation = `16896/64/512/64`; Cartpole `4096/8/64/8`; Humanoid `24832/128/512/128`. All 12 captures exited successfully, all 24 untimed boundary checks passed, recorded pre/post states were finite, and final source/idle guards passed. These are available sticky-capacity checks, not exhaustive contact-demand or convergence proof. All three manifests retain `physical_quality_accepted=false`, `performance_accepted=false` and `repeated_timing_evidence=false`.

Humanoid is the substantial candidate: it saves 2.934 ms RTX / 0.767 ms GB300 physics, but subsequent loaded replay does not pass (details below), so repeated timing and promotion are deferred. Ant is slower on GB300 physics and RTX wall time; Cartpole saves only 0.022/0.031 ms physics with approximately neutral wall time. Neither supports a substantial broadly useful recipe promotion. The mode change avoids dense Delassus construction/storage in favor of serial matrix-free row/DOF response products; it does not establish that `split` is always worse or that both modes have equivalent convergence at these budgets.

## Shared terrain path: four first-pass results

Each discovery pair uses the same Newton `4e42ebe2` and existing FPGS recipe. The only additions are `NEWTON_HEIGHTFIELD_CELL_REJECT=1` and `NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS=1`; grouped/lazy settings and capacities are unchanged. All four manifests completed with passing checks and final source/idle guards. Savings are milliseconds of physics per environment step, not wall-time savings.

| Task / evidence | RTX PRO6000 speedup | RTX saved ms | GB300 speedup | GB300 saved ms |
| --- | ---: | ---: | ---: | ---: |
| [ANYmal-D rough](/home/octi/Projects/fpgs-recipes-evidence-20260919/terrain_anyrough_discovery/manifest.json) | 1.51× | 10.15 | 1.62× | 26.45 |
| [Cassie rough](/home/octi/Projects/fpgs-recipes-evidence-20260919/terrain_cassierough_discovery/manifest.json) | 2.10× | 6.29 | 2.98× | 16.43 |
| [Go2 rough](/home/octi/Projects/fpgs-recipes-evidence-20260919/terrain_go2rough_discovery/manifest.json) | 1.80× | 8.58 | 2.10× | 16.17 |
| [H1 rough](/home/octi/Projects/fpgs-recipes-evidence-20260919/terrain_h1rough_discovery/manifest.json) | 1.73× | 5.52 | 2.14× | 14.51 |

These are separate experiments from the execution table; their ratios must not be multiplied into a claimed combined gain.

## Completed three-round FPGS results

Each time below is the median of three capture-level mean physics-graph times, in milliseconds per environment step. The medians were independently recomputed from the three `capture_analysis.json` files per arm/card. Baseline is Newton `4e42ebe2`; candidate is `da52d13e`. The four rough-task candidates change only the capability-selected terrain defaults. G1 flat separately enables grouped dynamics plus the sparse factor, retaining lazy kinematics false and capacity 56; it does not activate the kinetic-state owner.

| Task / repeated evidence | RTX physics ms, before → after | RTX physics gain | GB300 physics ms, before → after | GB300 physics gain | RTX wall gain | GB300 wall gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [ANYmal-D rough](/home/octi/Projects/fpgs-recipes-evidence-20260919/qualified_repeat3/anymald-rough/manifest.json) | 29.99 → 19.74 | 1.52× | 69.00 → 42.43 | 1.63× | 1.44× | 1.54× |
| [Cassie rough](/home/octi/Projects/fpgs-recipes-evidence-20260919/qualified_repeat3/cassie-rough/manifest.json) | 11.99 → 5.72 | 2.10× | 24.70 → 8.32 | 2.97× | 1.37× | 1.78× |
| [Go2 rough](/home/octi/Projects/fpgs-recipes-evidence-20260919/qualified_repeat3/go2-rough/manifest.json) | 19.34 → 10.71 | 1.81× | 30.68 → 14.67 | 2.09× | 1.53× | 1.75× |
| [H1 rough](/home/octi/Projects/fpgs-recipes-evidence-20260919/qualified_repeat3/h1-rough/manifest.json) | 13.09 → 7.56 | 1.73× | 27.44 → 12.75 | 2.15× | 1.22× | 1.63× |
| [G1 flat](/home/octi/Projects/fpgs-recipes-evidence-20260919/qualified_repeat3/g1-flat/manifest.json) | 8.05 → 7.06 | 1.14× | 9.39 → 6.73 | 1.39× | 1.09× | 1.18× |

All five manifests are complete: 60 captures exited successfully, all 120 untimed boundary checks passed, numerical budgets remained unchanged, and all final source/idle guards passed. The timing harness retains `physical_quality_accepted=false` and `performance_accepted=false`: repeated timings are not automatic physical acceptance. The bounded geometry and capacity-56 physical tests are documented below; neither those tests nor this campaign establish full contact-law or training equivalence.

## Fresh MJWarp comparison with the same terrain improvement

Source: [terrain_mj_timing/survey.json](/home/octi/Projects/fpgs-recipes-evidence-20260919/terrain_mj_timing/survey.json) and its four task manifests. MJWarp uses the existing Newton `4e42ebe2` implementation with `NEWTON_HEIGHTFIELD_CELL_REJECT=1`, `NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS=1`, and `NEWTON_NARROW_PHASE_THREADS_X=4`; the FPGS candidate receives the same conservative terrain path through its new defaults. Both use external Newton collision. This avoids crediting FPGS alone for a shared collision improvement.

**Sampling is asymmetric:** the FPGS column is the three-round candidate median above; the MJWarp column is one fresh capture-level mean over 40 profiled environment steps, not three repeated runs. Ratios are MJWarp / FPGS physics time; below 1 means FPGS is slower. These are not whole-environment or training-throughput ratios.

| Task | RTX FPGS ms | RTX MJWarp ms | RTX FPGS speedup | GB300 FPGS ms | GB300 MJWarp ms | GB300 FPGS speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ANYmal-D rough | 19.74 | 16.95 | 0.86× | 42.43 | 36.15 | 0.85× |
| Cassie rough | 5.72 | 9.32 | 1.63× | 8.32 | 11.97 | 1.44× |
| Go2 rough | 10.71 | 10.14 | 0.95× | 14.67 | 13.69 | 0.93× |
| H1 rough | 7.56 | 11.16 | 1.48× | 12.75 | 15.62 | 1.23× |

The shared collision work therefore gives substantial FPGS-over-FPGS gains, but does not produce a 4× MJWarp lead here. ANYmal-D rough and Go2 rough remain behind MJWarp in these captures. No fresh G1-flat/MJWarp ratio is inferred from this four-task campaign.

The eight MJWarp captures are complete with passing source/import guards, final source/idle guards, finite recorded states, and all 16 untimed boundary checks passing. All MJWarp warning masks and nonzero-world counts are zero; collision capacity and external-contact conversion checks passed with no unsupported owners. Logs contain no MJWarp line-search or capacity/overflow warnings. The source-checked `newton-bracket-elliptic-huber-v3` compatibility fix was active in all eight, without suppressing warning bits or changing iteration budgets or gradient tolerances.

Other warnings remain and must not be described as absent: MuJoCo/MJWarp 3.12 versus the declared 3.11 dependency requirement, shape-color deprecation warnings, COM-pose warnings for ANYmal/Go2, an ANYmal RNN warning, and Cassie's `Inertia validation corrected 8192 bodies` warning. In particular, Cassie's inertia correction is a physical-comparison caveat, not a capacity failure.

Effective MJWarp capacities match across cards: ANYmal-D `njmax=112, naconmax=135168, njmax_nnz=2016`; Cassie `88, 49152, 1584`; Go2 `96, 86016, 1728`; H1 `96, 61440, 2400`. These retain the existing compact plan rather than inflating buffers; backend-specific capacities are not forced equal. Both backends retain simulation dt 0.005 s, two solver substeps of 0.0025 s and decimation 4. Their inherited solver budgets remain FPGS 8 iterations versus MJWarp Newton 100 iterations / 50 line-search iterations; this is not a claim of equal numerical algorithms or established convergence/contact-law parity. The MJWarp reservations derive from prior demand evidence, not a new exhaustive demand sweep.

## Common default versus conditional selection

The conservative terrain path is a source-capability default, not task-name dispatch: it is admitted only for scenes with heightfields and no triangle-mesh shapes. Explicit environment-variable opt-outs remain respected; mixed mesh/heightfield scenes retain the original path. In the 18 measurable, non-Drawer survey tasks, source inspection identifies **four newly affected rough tasks**, **G1 rough already opted in**, and **13 tasks without the relevant heightfield path**. This describes eligibility, not a measured no-regression result for all 18. The finite-query path remains a separate opt-in; primitive-on-heightfield workloads such as capsule/cylinder robots use the conservative default.

Grouped/lazy execution is not a universal default:

- Cartpole loses physics performance on both cards; grouping overhead can outweigh useful work on tiny articulations. Several wall-time results also do not improve.
- Grouping/lazy can displace specialized simple-world paths; grouping excludes the prismatic linear-state owner used by Keyboard, and lazy kinematics excludes the tuned G1 kinetic-state owner. Existing ANYmal-D flat and Allegro recipes already enable both, so applying them again is not a new optimization.
- Lazy kinematics requires correct publication before collision, sensors or other public body-state reads. Recipe selection must preserve those hooks and specialized-owner admission, not merely change two booleans.
- Lab explicitly supplies `grouped_dynamics=False` and `lazy_kinematics=False` unless its configuration is overridden. Changing Newton constructor defaults alone does not change that behavior; overriding an explicit false would violate the caller's choice. Keep qualified explicit recipes or capability-safe internal selection, without Lab edits or task-name dispatch.

Source references: candidate `newton/_src/geometry/narrow_phase.py:2466`; `newton/_src/solvers/feather_pgs/{simple_world.py:485,prismatic_linear_state.py:26,g1_kinetic_owner.py:44}`; Lab `source/isaaclab_newton/isaaclab_newton/physics/{feather_pgs_manager_cfg.py:131,newton_manager.py:2179,feather_pgs_manager.py:40}`. Constructor docstrings document the selection tradeoffs, including block versus matrix-free row-solve costs, without promising universal wins.

## Qualification status and remaining gate

The serial matrix-free structural screen is complete, with the separate first-pass table above. Humanoid's subsequent loaded replay **did not pass**; it remains an unqualified structural candidate, not an accepted recipe. Ant and Cartpole are not promoted from their small, inconsistent whole-environment changes. No extra parallel sweeps, relaxed numerical gates or Lab/default-alias changes were introduced.

Humanoid evidence: `physical_humanoid_mode_gpu{0,1}.json`, tool `8994ee32`, runtime `4e42ebe2`, 64 worlds, 128 rows, 512 pooled contacts, eight iterations and two substeps. RTX stops at loaded step 0 with joint-velocity max/RMS difference 0.1230/0.00546; GB300 stops at step 1 with 0.03375/0.00128. Maximum body-position differences are 95.4/26.2 micrometres, but force differences also exceed the declared budget. Last-substep net-contact-impulse differences reach 9.79%/0.858% of one gravity impulse. These are differences from split, not independently measured physical error. State remains finite and capacity checks pass; reset coverage is not reached. Repeated timing and promotion are deferred, not the matrix-free architecture rejected.

Source diagnosis at `4e42ebe2`: this recipe uses the same unilateral limit law and per-tangent disk projection in both modes. Split accumulates `rhs + C lambda` using a mirrored lower triangle; matrix-free accumulates `bias + J v` and immediately applies row/sibling velocity updates. Exact-arithmetic equivalence requires matching operators and row order. Atomic contact-slot allocation means shared geometry alone does not prove the same GS order; finite-precision accumulation also differs. Neither is yet established as the cause. Both modes refresh mass on even substeps and reuse it on odd ones; there is no identified refresh-schedule mismatch. Next compare per-substep rows/operators and fixed-budget complementarity/residuals against an independently converged diagnostic solve. Keep performance runs at the original eight iterations. Do not change acceptance tolerances merely to admit the faster mode.

- Default geometry/contact suite: **15 tests passed in each card run**, including matched primitive/terrain states, pose withdrawal/regrowth, explicit opt-out and mixed-mesh fallback. Evidence: [RTX log](/home/octi/Projects/fpgs-recipes-evidence-20260919/default_geometry_gpu0.log), [GB300 log](/home/octi/Projects/fpgs-recipes-evidence-20260919/default_geometry_gpu1.log). The suite includes CPU and CUDA checks.
- G1 sparse factor at row capacity 56: the complete-owner two-step/graph CUDA test passed on both cards; logs show the capacity-56 kernels executing. Evidence: [RTX log](/home/octi/Projects/fpgs-recipes-evidence-20260919/sparse56_gpu0.log), [GB300 log](/home/octi/Projects/fpgs-recipes-evidence-20260919/sparse56_gpu1.log). This qualifies that bounded test, not every G1 trajectory or the unadmitted kinetic owner.
- Go2 flat and Franka reach grouped/lazy strict replay **did not pass** on either card. Evidence: `physical_go2flat_gpu{0,1}.json` and `physical_frankareach02_gpu{0,1}.json`. Their loaded replays exceeded declared tolerances; these are not qualified recipe gains, and the mismatch alone does not establish a physics/contact bug.
- `qualified_repeat3/` has completed three rounds for all four terrain tasks and G1 flat; repeated timing results and bounded qualification scope are recorded above. The fresh [terrain_mj_timing/survey.json](/home/octi/Projects/fpgs-recipes-evidence-20260919/terrain_mj_timing/survey.json) comparison is also complete with the same shared collision improvement, preserving the existing capacities and numerical budgets from [TERRAIN_MJ_PLAN.json](/home/octi/Projects/fpgs-recipes-evidence-20260919/TERRAIN_MJ_PLAN.json). Its one-pass MJWarp sampling and unresolved physical-equivalence caveats remain explicit; the plan's inherited demand fields are not fresh measurements.

## Exact pins and paths

All evidence above is rooted at `/home/octi/Projects/fpgs-recipes-evidence-20260919/`; all worktrees below are under `/home/octi/Projects/IsaacLab.wt/`.

| Role | Worktree | Exact commit |
| --- | --- | --- |
| Baseline and both discovery runtime arms | `newton-fpgs-structural-resume-20260919` | `4e42ebe2d9edeb630814da51cb3d4a2c41e99fa6` |
| Shared collision default implementation | `newton-heightfield-default-20260919` | `5e55dc1bd5337de6405d10a080c5da71a8a7890f` |
| Frozen repeated-run candidate, including selection docs | `newton-fpgs-recipe-docs-20260919` | `da52d13e98802a996582282050ea9ffb2aae2b58` |
| Execution discovery and repeated-run tools | `newton-fpgs-recipes-20260919` | `9233cffd6a4729c71a864e2ec2621c6d6d6de604` |
| Capacity-56 sparse-factor test source | `newton-fpgs-cap56-tests-20260919` | `536f9c628857edfe7f073bcf48e78c8402d17d87` |
| Terrain first-pass discovery tools | `newton-fpgs-survey20-20260919` | `81eb595912ce5c706415d2b214d923b54602474b` |
| Unchanged Isaac Lab | `contact-reset-20260913` | `53ee6b44c2334341305dbdf385a3916c6b140799` |

The exact tool pin above was verified with `git rev-parse 9233cffd`: `9233cffd6a4729c71a864e2ec2621c6d6d6de604`. Candidate `da52d13e` has the same executable behavior as `5e55dc1b`; its intervening commits add docstrings/changelog only. The fresh MJWarp run uses the `4e42ebe2` runtime and `81eb5959` tools with the explicitly enabled shared terrain flags. The original survey artifacts at `/home/octi/Projects/fpgs-survey20-evidence-20260919/` are retained unchanged; this note does not replace their older measurements with discovery estimates.

## Exact opt-in G1-flat reproduction

This reproduces the settings in [the three-round G1-flat manifest](/home/octi/Projects/fpgs-recipes-evidence-20260919/qualified_repeat3/g1-flat/manifest.json), writing a new output directory. Use the pinned worktrees above and idle GPUs. The runner launches each arm on both cards together, alternates arm order across three rounds, and enables capacity checks. It supplies the common flags `FEATHER_PGS_GROUP_LANES=16`, `FEATHER_PGS_ROWS_MASKED=1` and `NEWTON_NARROW_PHASE_THREADS_X=4` to both FPGS arms; inherited feature flags are cleared.

```bash
uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python \
  python /home/octi/Projects/IsaacLab.wt/newton-fpgs-recipes-20260919/tools/fpgs_bench/compare_variants.py \
  --isaaclab /home/octi/Projects/IsaacLab.wt/contact-reset-20260913 \
  --baseline /home/octi/Projects/IsaacLab.wt/newton-fpgs-structural-resume-20260919 \
  --candidate /home/octi/Projects/IsaacLab.wt/newton-fpgs-recipe-docs-20260919 \
  --output-dir /home/octi/Projects/fpgs-recipes-evidence-20260919/g1-flat-reproduce01 \
  --task g1-flat \
  --candidate-env FEATHER_PGS_SPARSE_FACTOR=1 \
  --candidate-env FEATHER_PGS_SPARSE_LEVEL_UPDATE=1 \
  --candidate-solver-attr grouped_dynamics=True \
  --capacity g1-flat:fpgs:rigid_contact_max=49152 \
  --capacity g1-flat:fpgs:dense_max_constraints=56 \
  --gpus 0 1 --seed 0 --rounds 3 --num-envs 4096 \
  --warmup-steps 200 --steps 40 --profile-steps 40 --trace-mode graph
```

Only the two sparse flags and grouped-dynamics opt-in distinguish this recipe from the unchanged task configuration. Lazy kinematics remains false; the actual repeated-run metadata records serial `matrix_free` with parallel rows zero. Capacity 56 admits the sparse factor but not the kinetic-state owner; do not inflate it to 100 or add kinetic/lazy/parallel flags and call that the same recipe. This command changes neither Lab source nor default task aliases. The bounded capacity-56 qualification and the limits on physical-equivalence claims remain as recorded above.

## Integration and independent correctness fix

This branch integrates the measured shared-terrain default and concise constructor
selection guidance. It leaves solver-mode and grouped/lazy defaults unchanged;
the explicit G1-flat command above preserves baseline aliases and Lab sources.
Timing tables refer to their recorded pins, not to a new capture of this combined
tip. Supporting qualifier/test changes belong with these changes, not standalone
performance PRs.

A separate fused-dynamics bug used velocity-prescaled motion state for explicit
passive damping. With damping 0.2 and input velocity 10 rad/s capped to 2 rad/s,
the old fused torque was -0.4 N·m rather than the analytic -2 N·m. The fix passes
original velocity separately for damping, keeping motion prescaling unchanged.
The regression fails on both GPUs before the fix; all four spring tests pass on
both afterward, using distinct fresh generated-kernel caches. Evidence:
`damping_before02_gpu{0,1}.log` and `damping_after_gpu{0,1}.log` in the evidence
root. Initial `damping_before_gpu*` attempts stopped on an unsupported one-lane
test fixture; the corrected regression explicitly uses 16 lanes. This is not a
performance claim or an explanation of the unqualified Franka replay mismatch.
