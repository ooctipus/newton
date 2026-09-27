# FeatherPGS vs MJWarp: handoff report (2026-09-11)

Everything below is measured on GPU 1 of this machine (RTX 5090, 170 SMs) at 16,384 environments in CUDA-graph mode,
unless marked as an estimate. Nothing is committed. All changes live behind opt-in flags in two worktrees:

- Newton: `~/Projects/newton.wt/fpgs-northstar` (branch `zhengyuz/fpgs-northstar-20260908` off `5631a64a`), used via
  `PYTHONPATH=/home/zhengyuz/Projects/newton.wt/fpgs-northstar`.
- Isaac Lab: `~/Projects/IsaacLab.wt/fpgs-official-lab` (harness `scripts/benchmarks/fpgs_profile/`, manager hooks).

The full design document with every measurement, including the negative ones, is
`fpgs-results-20260906/fpgs_solver_shape_and_spirit_20260909.md` (sections 1-19). Read sections 2, 12.7, 17, 18, 19.

## 1. Principle, north star, spirit

**Principle.** FeatherPGS keeps the solve fixed and small: factor the joint-space inertia once per substep, form the
contact response once, and run a fixed number of projected sweeps on a small per-world block. No line search, no
convergence test, no refactorisation, no data-dependent iteration count. Work per world is fixed at model build. The
designed-in arithmetic advantage over MJWarp's Newton-with-line-search is 10 to 25x per world on the flagship tasks
(design doc 1.1); the mission is to execute that design at or above MJWarp's efficiency.

**North star.** One block (or one lane group) per world per substep that takes joint state, factors and contacts in
and puts velocities out, with the world's state resident on-chip for the whole substep and at most four launches per
substep (contact bucketing, world dynamics, sweep, integrate). Collision stays Newton's shared pipeline.

**Spirit (the nine rules, design doc section 2).** One formulation for every world, specialised only by size class
at build time. Work proportional to what is active, never to capacity. The batch of worlds is the SIMT axis;
intra-world parallelism only where a lane cannot hold the world. Specialise at build time, never dispatch at run
time in the hot loop. State stays on-chip within a stage; only compact per-world blobs cross stages. At most four
launches per substep. Everything topological decided once at model build. Performance work never changes the math
(same rows, order, projections, iteration count, parameters; only floating-point order may move). Every change is
measured the same way: physics-graph microseconds per substep by stage, paired A/B on GPU 1, identical contact
statistics and finiteness, Nsight instruction budgets.

**Honesty rule learned this round.** "Physics identical" must be checked at the per-substep velocity level against
the real kernels (see 5.1), not only through contact statistics, which were insensitive to 2% velocity differences.

## 2. Standing table (same-day MJWarp, `profile_artifacts_20260910/`, clear ON)

| Task | FPGS best (us / env step) | FPGS 09-09 legacy | MJWarp | speedup |
|---|---:|---:|---:|---:|
| Velocity-Flat-AnymalD | 8,990 (world rows on) | 15,678 | 35,094 | 3.90x |
| Reorient-Cube-Allegro | 17,605 | 39,713 | 59,806 | 3.40x |
| Lift-KukaAllegro (legacy path) | 17,634 | 18,602 | 72,902 | 4.13x |
| Lift-Franka (legacy) | 5,854 | 5,734 | 29,984 | 5.12x |
| Velocity-Rough-G1 (grouped dynamics) | 38,931 | 42,769 | 50,126 | 1.29x |
| Cartpole (legacy) | 248 | 244 | 541 | 2.18x |
| Ant (legacy, dense) | 2,072 | 2,069 | 2,216 | 1.07x |
| Humanoid (legacy, dense) | 7,920 | 8,230 | 8,511 | 1.07x |

All states finite; contact counts match the legacy path. Run-to-run spread is ~1% on clean runs; one Allegro outlier
of 8% occurred when another process touched GPU 1 (use `scratchpad/insitu/clean_run.sh`, which waits and retries).
The 16.6k Allegro numbers seen mid-day carried a stale-Jacobian defect (skipped clear) and are void.

Per-substep budget today, AnymalD (us): sweep tiers 261 + 186, contact rows in-kernel (inside those), K1 dynamics
102, CRBA + Cholesky 71, remaining row kernels 108, collision 119, trisolve 40, FK 32, integrate 20, response
leftovers 23. Allegro (us): sweep tiers 352 + 106 + 31 + 30, collision ~1,200 (GJK 617, MPR 360, manifold 140), rows
~340, K1 130, CRBA 95.

## 3. What was built this round (all opt-in; recommended per-task flags at the end)

1. Augmented-state allocation moved to `__init__` (fixes graph-vs-eager divergence; design doc 11).
2. Parallel projection sweep kernel `_get_pgs_solve_parallel_kernel` (scaled Jacobi, Nesterov, adaptive restart,
   per-contact projection, tiers 32/64/96/128 rows; matrix-free variant; exact-row-sum step scale option). Doc 13, 16, 17.17.
3. Response solve folded into the sweep kernel in whitened coordinates (`FEATHER_PGS_INK=1`): Z = L^-1 J^T, A = Z^T Z,
   one backward substitution at the end; response and diagonal stages skipped for owned worlds. Doc 17.2-17.4.
4. Kernel-level fixes found with Nsight: register blow-up (218 -> 96), shared-memory footprint (16.5 -> 9.7 KB),
   column sums once per block, 22-way bank conflict in the d-major block (stride padded to AM+4). Doc 17.8-17.9.
5. Per-tier world lists (`FEATHER_PGS_TIER_BLOCKS=16384`, Allegro), per-row contact bias, three-thread prelude,
   one thread per used DOF in the J pass, velocity-limit rows one thread per DOF. Doc 17.6, 17.7, 17.12, 17.16.
6. Newton narrow phase: bounding-sphere reject before GJK (`prepare_convex_pair`); narrow-phase thread scale
   (`NEWTON_NARROW_PHASE_THREADS_X=4`). Doc 17.11.
7. World kernel steps 1-3 (`FEATHER_PGS_WORLD_ROWS=1`): per-world contact lists, contact rows (geometry, friction
   decision, metadata, bias, restitution, J) built inside the sweep kernel, producers skipped for owned worlds. Doc 18.
8. Ground-truth tooling: `scratchpad/formulation/truth.py` (real kernels at 24 and 400 iterations on captured
   substeps), consecutive-substep capture (`FEATHER_PGS_CAPTURE_COUNT`), numpy models validated to 2e-7 against
   the kernels, check modes (`FEATHER_PGS_INK_CHECK`, `FEATHER_PGS_WORLD_ROWS_CHECK`, `FEATHER_PGS_CHECK_GROUPED`).

Recommended flags. Common: `FEATHER_PGS_GROUP_LANES=16 FEATHER_PGS_ROWS_MASKED=1 NEWTON_NARROW_PHASE_THREADS_X=4`.
AnymalD: `FEATHER_PGS_INK=1 FEATHER_PGS_MF_EXACT_ROWSUM=1 FEATHER_PGS_WORLD_ROWS=1` with
`grouped_dynamics=True mf_gs_parallel_rows=48 mf_gs_parallel_matrix_free=True lazy_kinematics=True`.
Allegro: `FEATHER_PGS_INK=1 FEATHER_PGS_TIER_BLOCKS=16384` with `mf_gs_parallel_rows=128 mf_gs_parallel_matrix_free=True`
(world rows neutral there; exact row sums slower there). Never set `FEATHER_PGS_SKIP_J_CLEAR` (wrong physics).

## 4. What was measured and closed (do not redo without a new idea)

| direction | measurement | result |
|---|---|---|
| Exact per-world solve (staggered Newton, velocity space) | 6 consecutive captured substeps, warm and cold, vs kernel fixed points | ~26 inner iterations either way, ~500 sweep-equivalents vs 24 today. 20x too expensive. Doc 19.2 |
| Warm start | Newton contact matching + FPGS built-in: 5.6 ms radix sort per step. In-kernel per-world table (`FEATHER_PGS_WR_WARM=1`): statistics equal at 24 sweeps | bookkeeping ~35 us per tier launch = the 8 sweeps it saves; 16 warm sweeps shift contacts 3%. Break-even. Doc 19.3 |
| Finger-block structure on hands | premise exact (block-diagonal factor; rows touch 4 of 16 hand DOFs); bound via shadow twin with quarter-range reductions | 3% of the sweep kernel. Not built. Doc 19.4, 19.6 |
| Small-world packing | occupancy limiter of the 32-row tier | register-limited (86 regs), not block-limited. No gain |
| Collision as a per-world pass | Nsight on Newton GJK: 107 regs, 1.65 GB spill traffic per launch, 1/3 lanes active | headroom 2-3x, inside Newton's GJK (register-resident simplex), not in pairing or cadence. Doc 19.5 |
| Fused one-pass row builder | metadata identical, J within 4e-6 | slower than two passes (latency chain per row). Doc 17.16 |
| Two-phase in-kernel row build | geometry once per contact into shared | slower: the two chains serialise. Doc 18.2 |
| Sweep barriers (shuffles on the single-warp tier) | statistics identical | no change: tier is instruction-bound, not barrier-bound |
| Lean sweep bound (no restart/vote/friction) | shadow twin on true state | sweep phase 234 -> 127 us per substep at best; floor ~1 ms per step. Doc 19 (8-9x discussion) |

## 5. Facts a successor must know

5.1 **The two solvers converge to different answers.** Legacy row Gauss-Seidel and the parallel projection differ by
1.8% median (5.5% p90) in per-world velocity at convergence, because each projects the friction pair with its own
per-row step metric; neither is the maximal-dissipation Coulomb solution. Legacy GS is converged at 24 iterations
(1e-5 from 400); the parallel kernel is 0.4% (AnymalD) / 0.05% (Allegro) from its own fixed point at 24 sweeps.
Contact statistics do not see any of this. Use `truth.py` for every solver-side claim. Doc 19.1.

5.2 **Skipping the Jacobian clear is wrong** whenever sparse row producers exist (joint/velocity limits) or worlds
have two articulations. Ant went non-finite; Allegro statistics shifted. Doc 17.14.

5.3 **Never edit Newton kernel sources while a run is in flight** (Warp extracts source by line number at JIT).
Never allocate Warp arrays inside a CUDA graph capture (`wp.zeros` becomes a memset node). Never call `.numpy()`
lazily in `step()` (Isaac Lab's first step is inside capture). Never launch or allocate per contact capacity
(`rigid_contact_max` is 11.2M on AnymalD); grid-stride with `_CONTACT_BUILD_THREAD_CAP`.

5.4 **The GPU-side floor of the current design.** With the dynamics fold (~0.3-0.5 ms per step) AnymalD lands near
8.5 ms (4.1x); Allegro's ceiling under the same collision cadence is set by Newton's GJK (5.3 ms of 17.6). 8-9x is
not reachable on this road with fixed physics parameters (substeps, collision cadence, contact set, sweeps, tolerance).

5.5 **Measurement hygiene.** GPU 1 must be exclusive; `clean_run.sh` detects foreign compute processes and repeats.
The nsys stage classifier puts MJWarp's convex narrow phase (`ccd_kernel`, 13.7 ms per step on Allegro) under
"other"; MJWarp's collision is 2.5x more expensive than Newton's on the hand. The RL policy LSTM (1.4 ms per step)
runs eagerly outside the physics graph and is not in any number above.

## 6. Open directions that could still change the ceiling (untested, in order)

1. **Dual active-set solve per world** on the Delassus block (30-60 rows), warm-started: exact answer in 10-20
   sweep-equivalents if the active set changes 1-2 times per substep. Not tested (the Newton test was a different
   formulation). Test first on the captured operators with `truth.py` as reference.
2. **Anderson / conjugate-residual acceleration** of the projected iteration (halving iteration counts is common).
3. **Incremental reuse of the whitened rows across substeps** (contacts persist 100%; the rows + Z phase is 92 us per
   substep on the 48-row tier, more than the sweeps).
4. **Correct block (3x3 per contact) preconditioning** with a proper cone projection in the block metric.
5. **Register-resident GJK in Newton** (spills and divergence measured; 2-3x on 5.3 ms of Allegro).
6. **Dynamics fold** into the world kernel (FK, ID, CRBA, Cholesky, trisolve, integrate): launch tails and factor
   round trips only, ~0.3-0.5 ms per step per task.

## 7. Where things are

- Design doc: `fpgs-results-20260906/fpgs_solver_shape_and_spirit_20260909.md` (sections 11-19 are this round).
- Round-close artifacts and table: `fpgs-results-20260906/profile_artifacts_20260910/` (`remeasure.sh`, `make_table.py`).
- Scratch (session-local, may vanish): `/tmp/claude-1776732611/.../scratchpad/` with `insitu/` (batch scripts,
  `clean_run.sh`, nsys analyses), `formulation/` (`truth.py`, `warm_par_study.py`, `warm_study.py`, `sap_proto.py`),
  `cap/{anymald,allegro}_seq/` (6 consecutive substeps with kernel ground truth `_truth.npz`).
- Memory notes: `~/.claude/projects/-home-zhengyuz-Projects-IsaacLab-wt-fpgs-official-lab-recovered-20260830/memory/`.
- Harness: `scripts/benchmarks/fpgs_profile/{run_profiled.py,nsys_run.sh,ncu_run.sh,analyze_*.py}` in the Isaac Lab
  worktree (`--trace-stats` prints contacts, broad-phase pairs, GJK items, z range, finiteness per step).
