# FeatherPGS: the right shape and the right spirit (revision 2)

Date: 2026-09-09. Revision 2 replaces the first draft of the same day. The first draft synthesised two days of
measurements in one pass; this revision adds the analysis that draft asserted rather than derived: where the 10x
against MJWarp has to come from stage by stage (Section 1), an instruction-level anatomy of the sweep kernel
(Section 3), a microbenchmark of the sweep mappings with bitwise-identical results (Section 3), and the resulting
changes to the shape, the targets and the milestone order (Sections 4 to 7). Every number was measured on GPU 1 of
this machine (RTX 5090, 170 SMs, 100 KB shared per SM) at 16,384 worlds unless marked as an estimate. Artifacts are
under `profile_artifacts_20260908/` (`nsys/`, `ncu/`, `ncu_src/`, `k2/`, `inc/`, `ab/`).

What changed against the first draft:

- The decisive stage is K1 (dynamics, rows, response), not the sweep. FPGS's own dynamics stage today already
  exceeds the 10x budget on AnymalD, Lift, Ant and Humanoid. The first draft ordered work the other way round.
- The lane-per-world sweep is now measured, not estimated: 6.6 us for 16 rows and 8 sweeps at 16k worlds, against
  72 us for the same math in today's warp-plus-shared-memory shape and 226 us for today's real kernel. The result
  depends on packing the symmetric response block; the unpacked version is 5x slower because of a second wave.
- The 17-to-32-row and 33-to-64-row classes have measured mappings too (lane-per-row with the response row in
  registers: 64 us at 32 rows; two rows per lane: 278 us at 58 rows and 12 sweeps against 3,697 us today on Allegro).
- The 10x is split into three honest numbers per task: solver-only, whole physics graph, and end-to-end. Collision
  bounds the whole-graph number on rough terrain and dexterous hands for both engines, and the Isaac Lab host floor
  bounds end-to-end FPS at roughly 4 to 7x whatever the solver does.
- A contact-bucketing stage (K0), a world-interleaved blob layout, a structure-aware response for the hand class,
  and a pre-M0 root-cause task for the `matrix_free` divergence were missing and are added.

## 1. Where the 10x has to come from

### 1.1 What is and is not different about the two solvers

Both engines are reduced-coordinate: MuJoCo is a generalized-coordinate engine, so the joint-space inertia, the
factorisation and the Jacobian sizes are the same order for both. Reduced coordinates are not FeatherPGS's advantage
over MJWarp. The advantage is the solve:

- MJWarp runs Newton on the constraint problem. Each iteration updates the Hessian (`JTDAJ`, `JTCJ`), refactors an
  n x n Cholesky, and runs a line search with several cost evaluations. Isaac Lab configures `iterations=100`,
  `ls_iterations=50` (15 on Lift), `tolerance=1e-6`; the loop stops on a global `_solve_done` reduction. Measured
  Newton iterations per substep: Lift 4.1, Ant 6.5, Humanoid 7.5, G1 8.6, Allegro 16.1, AnymalD 17.6. Each iteration
  is 11 kernel launches; a substep is 155 to 264 graph nodes.
- FeatherPGS factors H once, forms the response `Y = H^-1 J^T` once, and runs a fixed number of projected
  Gauss-Seidel sweeps (8, 12 on Allegro) on a small response block. No line search, no convergence test, no
  refactorisation, no data-dependent iteration count. Work per world is fixed at model build, so a substep can be a
  handful of launches with no device-side control flow.

Arithmetic per world on AnymalD (n = 18, about 11 active rows plus limits): MJWarp about 17.6 x (1.9k Cholesky +
7.8k JTDAJ + line-search evaluations) which is over 250k flops; FeatherPGS about 7.8k for Y, 2.2k for A = J Y and
1k for eight sweeps, about 11k flops. The designed-in arithmetic advantage is 10 to 25x depending on the task, and
the structural advantage is roughly 60 launches against 4. The mission is to execute that design at MJWarp's
efficiency or better; today FeatherPGS executes it at about the same one-sixth issue rate through a shape that
spends most of its instructions on bookkeeping.

### 1.2 Stage split per substep, both engines (us, graph-busy, corrected classification)

| Task | Engine | Total | Dynamics | Rows | Response | Solve / sweep | Collision | Nodes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Lift-Franka | FPGS | 637 | 255 | 122 | 34 | 140 | 137 | 61 |
| | MJWarp | 2,844 | 547 | | | 2,131 | 151 | 155 |
| Ant (pre-fix trace) | FPGS | 1,672 | 248 | 940 | 463 | 12 | 32 | 36 |
| | MJWarp | 1,000 | 442 | | | 524 | 23 | 160 |
| Humanoid (pre-fix trace) | FPGS | 2,797 | 487 | 1,208 | 1,064 | 25 | 53 | 34 |
| | MJWarp | 1,977 | 936 | | | 970 | 57 | 182 |
| Cartpole | FPGS | 89 | 59 | 14 | 10 | 2 | 10 | 33 |
| | MJWarp | 222 | 132 | | | 78 | 8 | 89 |
| Flat-AnymalD | FPGS | 1,116 | 371 | 138 | 175 | 314 | 81 | 37 |
| | MJWarp | 4,356 | 670 | | | 3,522 | 151 | 264 |
| Rough-G1 | FPGS | 4,337 | 1,512 | 213 | 627 | 428 | 1,529 | 40 |
| | MJWarp | 6,193 | 1,995 | | | 2,279 | 1,887 | 235 |
| Reorient-Allegro | FPGS | 6,409 | 565 | 914 | 502 | 3,697 | 767 | 53 |
| | MJWarp | 7,254 | 880 | | | 4,565 | 1,795 | 259 |
| Lift-KukaAllegro | FPGS | 2,244 | 785 | 567 | 279 | 380 | 248 | 55 |
| | MJWarp | 9,198 | 1,517 | | | 7,352 | 290 | 229 |

Ant and Humanoid after the dense fixes in the worktree are at parity with MJWarp (Ant 1,035 vs 1,000; Humanoid
2,050 vs 1,977 per substep, from the whole-graph times). MJWarp's "dynamics" includes its own FK/CRB/factor kernels
and the Newton-to-MuJoCo state conversion; its collision on G1 and Allegro is the same Newton pipeline FPGS uses
(mesh-triangle and CCD kernels), which is why both engines pay 0.8 to 1.9 ms there.

### 1.3 Conclusions that reorder the plan

1. **MJWarp's dynamics is not a cap on the ratio.** It is 15 to 47% of MJWarp's substep and runs far from its own
   floor, so it stays in the denominator. The 10x budget for FPGS is simply MJWarp/10: AnymalD 436 us, Lift 284,
   Ant 100, Humanoid 198, Cartpole 22, G1 619, Allegro 725, Kuka 920.
2. **FPGS's own dynamics, rows and response already exceed those budgets.** AnymalD spends 684 us in those three
   stages against a 436 us budget for the whole substep; Lift 411 against 284; Ant and Humanoid 248 and 487 against
   100 and 198. Even a zero-cost sweep does not reach 10x anywhere except KukaAllegro. The fused world-dynamics
   kernel (K1) is where the mission is decided; the sweep is second.
3. **Collision bounds the whole-graph number on two tasks.** On G1 the shared collision cost (1,529 us) alone is
   2.5x the 10x budget; on Allegro it is about the budget. Whole-graph 10x there needs collision work in Newton
   (shared with MJWarp) and is outside this solver plan. Solver-only 10x is still reachable on both.
4. **End-to-end FPS is host-bound.** Isaac Lab's manager step costs 8.4 ms per env step at any environment count on
   Lift and 4 to 5 ms on AnymalD. MJWarp AnymalD at 16k envs is a 34.9 ms graph; a 1 ms FPGS graph gives a 5 to 6 ms
   step, so the FPS ratio saturates at roughly 6x. Solver progress must be reported as physics-graph us per substep
   by stage; FPS ratios belong to the host-floor track (P1).

## 2. The spirit: nine rules

1. **One formulation for every world.** Every world is a block-diagonal joint-space inertia (articulation H blocks
   plus 6x6 free-body blocks), rows with Jacobians over the world's DOFs, a response block `A = J H^-1 J^T`, and a
   Gauss-Seidel sweep in a fixed row order. No `dense` / `split` / `matrix_free` modes, no local / pair / residual /
   general queues, no propagation variants. Specialisation is by size class only, generated at model build.
   Precondition: the `matrix_free` divergence on Ant/Humanoid (bodies at z = 25 m, contacts down 30x) must be
   root-caused first, because it says the two current formulations disagree and the unified one must follow the
   right one.
2. **Work is proportional to what is active, never to capacity.** Loops run to active counts; buffers are sized to
   class bounds but never streamed to their bound. The two dense fixes were this rule applied twice.
3. **The 32 lanes are 32 worlds unless the footprint forbids it.** The batch is the SIMT axis. The sweep is scalar
   per row and sequential per world: one lane per world (measured 6.6 us against 72 us for the same math with a
   warp per world). Intra-world parallelism (threads over bodies, rows, DOF columns) is used only where the
   per-world state does not fit a lane's registers and shared slice, which is the dynamics and response stage.
   Never a serial loop in one thread per world at low occupancy.
4. **Specialise at build time; never dispatch at run time inside the hot loop.** The instruction anatomy in
   Section 3 shows a runtime-dispatched general loop costs 115 warp-instructions per row update for 12 of
   arithmetic. Row kinds, phase bounds, drive/limit slots, friction pairing, class membership and loop bounds are
   compile-time or per-launch uniforms; the per-row kind switch is the only remaining branch and it is uniform.
5. **State stays on-chip inside a stage; only compact per-world blobs cross stages, laid out for the consumer.**
   The K2 blob is world-interleaved (k-major, world-minor) for the lane-per-world class so 32 worlds load with one
   coalesced instruction per element, and the response block is stored as a packed symmetric lower triangle. This
   layout decision is worth 5x on its own (34 to 6.6 us in the microbenchmark).
6. **At most four launches per substep**, as a budget: K0 contact bucketing, K1 world dynamics, K2 sweep, K3
   integrate. Dependency latency at about 4 us per launch is 130 to 230 us of today's substep.
7. **Everything topological is decided once at model build.** Row slot layout, phase bounds, class membership,
   mimic/connect coupling, static contact-pair filters. No per-substep classify, compact, snapshot or gate kernels.
8. **Performance work never changes the math.** Same rows, same order, same projections, same iteration count,
   omega, cfm, friction gating, restitution, drive model. Only floating-point operation order may differ, and for
   the sweep not even that: the microbenchmark shows all three mappings bitwise-identical to a sequential reference
   when each element's update order is preserved, so the K2 oracle can be exact rather than tolerance-based.
   Colouring, velocity caps and sweep schedules are correctness studies with their own acceptance.
9. **Every change is measured the same way.** Physics-graph us per substep per stage from the harness, paired A/B
   on GPU 1, identical `contacts_active`, row counts, `body_z` range and finiteness, ncu instruction budget per
   stage, then checkpoint success before release.

## 3. Measured anatomy of the sweep

### 3.1 Where today's incremental kernel spends its instructions

Nsight Compute SASS counters on `pgs_solve_mf_gs_incremental_192_32_18_rows32` (AnymalD, one launch,
`ncu_src/inc_src.ncu-rep`):

| Quantity | Value |
|---|---:|
| Warp instructions per world per launch | 12,492 |
| Row updates per world per launch (about 11 active rows x 8 sweeps) | 88 |
| Warp instructions per row update | about 115 |
| FFMA + FADD + FMUL share of instructions | 10.3% |
| Integer and address arithmetic (IADD3, IMAD, LEA, MOV) | 33.8% |
| Control (BRA, BSSY, BSYNC, ISETP) | 23.8% |
| Shared loads and stores (LDS, STS) | 11.9% |
| Occupancy limit | 15 blocks per SM from 5.7 KB shared; 28% achieved |
| Cycles per issued instruction per warp | 6.85 |

The kernel is neither memory-bound (L1 hit 94%, DRAM 6%) nor arithmetic-bound. It is a general loop with runtime
row lists, per-row kind decode, lane-0 stores, two `__syncwarp` per update and address arithmetic for every shared
access, and only 12 of its 115 instructions per update do the mathematics. The legacy kernel it replaced issued
more (14.6k per world) with 42% L1 hit because it re-read J and Y from global on every sweep, which is why staging
those in shared did not help: the instruction count, not the bytes, was the limit.

### 3.2 Microbenchmark of the sweep mappings

`k2/k2_bench.cu` runs the same projected Gauss-Seidel (fixed row order, unilateral projection, synthetic SPD
response blocks) at 16,384 worlds in four thread mappings and checks every result against a CPU reference (max
difference 1.2e-7 to 1.8e-7 for all variants; all GPU variants are bitwise identical to each other). Times exclude
the reset memset. `k2/k2_bench_gpu1.txt` is the raw output.

| Rows m | Sweeps | Today's shape: warp per world, A in shared, syncwarp | Lane per row, A row in registers, one shuffle per update | Lane per world, packed symmetric A in shared (17 KB per 32 worlds) | Lane per world, full-square A (32 KB) |
|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 39.5 | 16.9 | 6.6 | 33.3 |
| 12 | 8 | 55.9 | 21.0 | 6.6 | 33.3 |
| 16 | 8 | 72.2 | 25.1 | 6.6 | 34.1 |
| 24 | 8 | 122.9 | 49.1 | | |
| 32 | 8 | 155.9 | 63.6 | | |
| 32 | 12 | 225.4 | 81.8 | | |
| 48 | 12 | 1,368 | 237 (two rows per lane) | | |
| 58 | 12 | 1,607 | 278 (two rows per lane) | | |
| 64 | 12 | 1,696 | 302 (two rows per lane) | | |

Readings:

- **Lane per world is flat in m at this scale.** 6.6 us for 8, 12 or 16 rows: the kernel is at the launch and
  latency floor, 512 blocks of 32 worlds in one wave at 5 blocks per SM. Against today's real AnymalD kernel
  (226 us, of which the response build and bookkeeping are the majority) it is 34x; against the same math in
  today's shape it is 11x.
- **Packing the response block is not optional.** Full-square A needs 32 KB per block, 3 blocks per SM, and the
  512 blocks spill into a second wave: 34 us. Packed lower triangle needs 17 KB, 5 blocks per SM, one wave: 6.6 us.
  The kernel uses 209 registers per thread, which is fine at 32 threads per block.
- **Lane per row with the response row in registers is the 17-to-32 class**: 2.5x over today's shape at 24 to 32
  rows, 96 registers, no shared memory, no barriers, one shuffle per update. It is 3.5x slower than lane-per-world
  at 16 rows, which is the cost of spending 32 lanes on one world.
- **Two rows per lane is the 33-to-64 class**: 278 us at 58 rows and 12 sweeps (Allegro's mean and iteration
  count), 153 registers, against 3,697 us for today's real kernel on Allegro and 1,607 us for the same math in
  today's shape. This is 13x on the Allegro sweep without changing the math, and it replaces the first draft's
  claim that only colouring could move this class.
- Rows in a real world are not all unilateral: friction rows read the parent normal's impulse and scale a sibling,
  drive rows use a different update, velocity-limit rows run in a separate pass. In the lane-per-world mapping
  these are per-lane scalar branches over a small enumerated kind set; the cost is the divergence within a warp of
  32 worlds, bounded because all worlds in a class share the same phase filters. In the lane-per-row mappings
  they are uniform branches on the row being updated. Neither changes the instruction count materially.

## 4. The shape: four kernels per substep

```
collision (Newton pipeline, unchanged)  ->  contacts (global list)
K0  world_contact_bucket          contact -> world slot; per-world counts; class routing input
K1  world_dynamics<class>         block per world (or lane per world for the tree passes, see 4.1)
    FK -> ID/bias -> CRBA H -> Cholesky L (or cached L) -> rows -> Y = H^-1 J^T -> A = J Y (packed), rhs, diag
    out: K2 blob in the consumer's layout {A packed, rhs, inv_diag, kind/parent/mu/w, lambda_in, m}, Y, per-class world lists
K2  world_sweep                   one launch; block-uniform class: lane per world (m <= 16) |
                                  lane per row, A row in registers (m <= 32) | two rows per lane (m <= 64)
    N sweeps; friction gating; drive and velocity-limit projections; out: lambda, applied impulse per row
K3  world_integrate               v = v_hat + Y * applied; integrate q, poses, twists; sensors; store L on rebuild steps
```

Four launches per substep for any mix of classes: K2 is one launch whose grid covers the upper bound of blocks and
whose blocks read their class from K1's per-class world lists and exit early when past the list end; class is
uniform per block so there is no divergence between mappings. K0 exists because the collision pipeline emits a
global contact list; today the same work is spread over `allocate_world_contact_slots`, `populate_world_J` and
their clears (about 200 us on AnymalD in six launches). It is one scatter with per-world atomics (estimate 10 to
20 us) until the collision pipeline can emit per-world buckets.

### 4.1 K1: world dynamics, and the one decision still to be measured

The work per world (AnymalD, 17 bodies, 18 DOF, about 24 rows) is about 40k flops: FK and ID over 17 bodies, CRBA,
an 18 x 18 Cholesky (2k), Y for 24 rows (7.8k), A = J Y (10k). Over 16k worlds that is 0.65 GFLOP: about 7 us at
the GPU's issue rate if perfectly SIMT, and about 5 us of coalesced body-state traffic. Today the same work is
684 us in about 20 launches. Existing data points for the fused kernel's efficiency: FPGS's own block-per-world
tile kernels run at 62 us (`crba_cholesky_18`) and 148 us (`hinv_jt_tiled`, capacity-chunked) for one stage each;
MJWarp's body-parallel kernels run at 27 to 43 us each. A K1 at 60 to 100 us therefore needs each sub-stage at
about 10 us, which is 3 to 6x better than the existing Warp tile kernels. That is plausible only with hand-written
CUDA (`wp.func_native`, as the incremental kernel already does) and the right thread mapping, and it is the one
estimate in this document that is not yet backed by a measurement.

Two mappings are candidates and the first milestone measures both on AnymalD before anything else is built:

- **Block per world, threads over bodies and rows.** FK/ID/CRBA in level-synchronous order over the tree (depth 4
  for quadrupeds, about 8 for G1), threads over rows for the row build and the two triangular solves per row,
  threads over row pairs for A. Shared footprint for class S is 6 to 8 KB (H and L, J and Y for 32 rows, packed A,
  body poses and twists), 12 blocks per SM. Weakness: a quadruped has 17 bodies and 4 levels, so the tree passes
  keep half the lanes idle and the block spends its time in barrier-separated short phases.
- **Lane per world for the tree passes, warp per world for rows and response.** FK/ID/CRBA as one scalar program
  per lane with body state in world-interleaved local memory (L1-resident, coalesced, the same trick that makes the
  packed lane-per-world sweep 6.6 us); then the 32 lanes of a warp switch to 32 rows of one world for the solves
  and A. Weakness: 17 bodies x (7 + 6 + 10) floats of live state per lane is 400 floats, so it lives in L1 not
  registers, and the warp's 32 worlds must be visited in turn for the row phase (32 x the row work per warp, which
  is fine because the row phase is small).
- A branch-parallel hybrid (4 lanes per quadruped, one per leg; the base couples them) is the Featherstone-native
  mapping and is worth a look if both of the above disappoint.

`update_mass_matrix_interval` is preserved by storing L per world and loading it on skip steps. Row build keeps the
current row-builder semantics as device functions over the slot layout [drive][limit][vel-limit][contact] with the
same phase bounds, so K2's phase logic is unchanged. Response is for active rows only: two triangular solves per
row against L in shared, then A for active pairs, `diag_i = A_ii + cfm_i`, no generic tile solves.

### 4.2 K2: the sweep, three measured mappings

Per Section 3.2. K1 chooses the mapping per world from its active row count and appends the world to the class
list; the K2 blob layout follows the class (world-interleaved and packed for lane-per-world; row-major per world
for the two register mappings). Row kinds are decoded once by K1 into a small enumeration. Expected K2 cost at
16k worlds: 7 us where all worlds are in the 16-row class (AnymalD's median world has 11 active rows), 25 to 65 us
for the 32-row class, 240 to 300 us for the 64-row class at 12 sweeps. Rows beyond 64 fall to a legacy path behind
a flag until the structure-aware response below shrinks them.

### 4.3 The dexterous-hand class: structure-aware response

Allegro's response block is 58 x 58 on average because every finger couples to every other through the cube.
`A = J_h H_h^-1 J_h^T + J_c M_c^-1 J_c^T`: the first term is block-diagonal per finger (the palm is fixed, so
fingers do not couple through the hand), the second is rank 6. Storing A as four finger blocks plus a 58 x 6 factor
`U = J_c M_c^-1/2` is the same matrix in a different factorisation (about 7 KB instead of 20 KB, and a row update
costs one finger-block column plus a 6-term dot product instead of 58 FMAs). It is exact up to floating-point
order and is the FPGS-native way to make the hand class cheap: Featherstone sparsity becomes response sparsity. It
is a class-L optimisation to prototype after the plain two-rows-per-lane mapping lands, and it is separate from
colouring, which changes the math and stays a correctness study.

### 4.4 K3: integrate

`v = v_hat + Y * applied` (Y is L2-resident from K1), semi-implicit integration of q and body poses, twist
publication, sensor inputs, L store on rebuild steps. 10 to 30 us. Folds into the next substep's K1 once graphs are
stable, giving three launches per substep.

### 4.5 Size classes

| Class | Bounds | K1 mapping | K2 mapping | Covers |
|---|---|---|---|---|
| S | DOF <= 18, rows <= 32, no free bodies | 32 threads (decided by E1) | lane per world (m <= 16), lane per row otherwise | Cartpole, Ant, quadrupeds |
| M | DOF <= 32, rows <= 64, free bodies allowed | 64 threads | lane per row / two rows per lane | Franka tasks with objects, Humanoid, Cassie, H1 |
| L | DOF <= 64, rows <= 128, free bodies allowed | 128 threads | two rows per lane, then structure-aware response | G1, KukaAllegro, Allegro, SO101 |
| XL | beyond L | today's multi-kernel path behind a flag until retired | | |

Classes are chosen from the model once; a world whose active row count exceeds its class bound is routed to the
next mapping at runtime, never dropped. Worlds are appended to class lists in K1, so the slowest class does not set
the launch time of the others.

## 5. What this buys, by task (us per substep, 16k worlds)

Three numbers per task: solver-only (everything but collision), whole physics graph, and the end-to-end bound.
K2 numbers are measured; K1 and K3 are the estimates from Section 4.1 and 4.4 with the stated uncertainty.

| Task | Solver-only today FPGS / MJWarp | Solver-only target | Ratio | Whole graph target (with collision) | Ratio vs MJWarp graph |
|---|---:|---:|---:|---:|---:|
| Lift-Franka | 500 / 2,693 | 100 to 150 | 18 to 27x | 240 to 290 | 10 to 12x |
| Ant | 1,003 / 977 | 80 to 120 | 8 to 12x | 110 to 150 | 7 to 9x |
| Humanoid | 1,997 / 1,920 | 150 to 200 | 10 to 13x | 200 to 250 | 8 to 10x |
| Cartpole | 79 / 214 | 20 to 30 | 7 to 11x | 30 to 40 | 6 to 7x |
| Flat-AnymalD | 1,035 / 4,205 | 120 to 180 | 23 to 35x | 200 to 260 | 17 to 22x |
| Rough-G1 | 2,808 / 4,306 | 310 to 440 | 10 to 14x | 1,850 to 1,970 | 3.1 to 3.3x |
| Reorient-Allegro | 5,642 / 5,459 | 510 to 610 | 9 to 11x | 1,280 to 1,380 | 5.3 to 5.7x |
| Lift-KukaAllegro | 1,996 / 8,908 | 400 to 500 | 18 to 22x | 650 to 750 | 12 to 14x |

Reading: solver-only 10x is within reach of this shape on every task, with Ant and Cartpole the hardest because
MJWarp's Newton loop is already cheap there (few iterations, small n). Whole-graph 10x is reachable on Lift,
AnymalD and Kuka, is bounded by shared collision cost on G1 and Allegro, and by MJWarp's small solve on Ant. The
end-to-end FPS ratio is bounded near 4 to 7x by the Isaac Lab host floor at 16k envs regardless of the solver;
raising it is the P1 track.

## 6. What to stop doing, what to keep

Stop:

- Treating the sweep as the bottleneck. It is 28% of the AnymalD substep and its mapping is now measured; the 61%
  in dynamics, rows and response is the mission.
- Micro-tuning legacy kernels beyond bugs of rule 2. The dense fixes and the incremental sweep were worth doing;
  further tuning of the 37-to-61-launch pipeline competes with the shape change for the same people.
- Adding modes or presets per task. Every mode is a semantics surface to validate and a shape the fused design
  cannot absorb.
- Staging memory to fix an instruction problem. Measured twice (305 to 485 us, 420 to 856 us) and explained by the
  SASS anatomy.
- Quoting end-to-end FPS ratios as solver results.

Keep:

- Row-builder semantics, projection code, friction gating, drive and velocity-limit models, as device functions.
- The `*_denseamat` local-owned kernels as the reference for response-block sweeps; the propagation-coloured
  kernels as the starting point of the colouring study.
- The `_FPGS_CAPTURE` dump facility as the fixed-input oracle for kernel-level A/B.
- The FeatherPGS test suite (162 cases pass on the worktree), the profiling harness, and `k2_bench.cu` as the
  reference for K2 mappings (extend it with friction and drive kinds before M0's K2 lands).

## 7. Milestones and gates

| # | Deliverable | Gate | Weeks |
|---|---|---|---:|
| Pre | Root cause of the `matrix_free` divergence on Ant/Humanoid; define which formulation is the truth. Tolerance for impulses and joint velocities from the legacy solver's run-to-run variance. Extend `k2_bench.cu` with friction, drive and velocity-limit kinds. | written finding; oracle tolerance fixed | 1 |
| E1 | K1 prototype on AnymalD: FK + ID + CRBA + Cholesky in one kernel, block-per-world and lane-per-world variants, no rows | us per substep and ncu instruction budget for both; pick the mapping | 1 |
| M0 | Class S: K0 + K1 + K2 (lane per world and lane per row) + K3 for Ant and AnymalD; legacy path selectable by flag | bitwise-identical K2 against the sequential oracle on captured inputs; row sets and impulse statistics identical; state statistics match over 300 steps; tests pass; AnymalD substep <= 436 us (10x), stretch <= 200 | 3 |
| M1 | Class M (Humanoid, Lift with objects): free-body DOFs in the same formulation; retires `split` vs `matrix_free` | as M0 plus checkpoint success on Lift and Humanoid | 3 |
| M2 | Class L (G1, Kuka, Allegro): two rows per lane, then structure-aware response; heterogeneous class lists in one K2 launch | as M0 on G1 and Kuka; four launches per substep on the heterogeneous benchmark | 3 |
| M3 | Retire legacy paths behind an XL flag; warm startup compile under 5 s per class | full suite, cluster success sweep | 2 |
| C1 | Colouring / block-projection study for the hand class | accepted only on checkpoint success | parallel |
| P1 | Isaac Lab host floor (manager fusion, sync removal) and collision `compute_shape_aabbs` / mesh pipeline | end-to-end FPS | parallel |

## 8. Risks

- **K1 efficiency is the unmeasured estimate.** If E1 lands at 150 to 200 us rather than 60 to 100, AnymalD still
  reaches 10x but Ant and Humanoid do not; the lane-per-world tree pass and the branch-parallel hybrid are the
  fallbacks, and the answer is known after week 2.
- **Wave quantisation.** At 16k worlds, one wave of 32-world blocks is 512 blocks; any footprint that drops below
  3 blocks per SM doubles K2's time (34 vs 6.6 us measured). Class S's blob must stay at or under 17 KB per block;
  world counts other than 16k change the arithmetic and must be checked.
- **Register pressure.** Measured 209 registers for the packed lane-per-world sweep at 16 rows and 153 for two rows
  per lane at 64; both fine at 32 threads per block but they leave no room for extra per-lane state. Friction and
  drive kinds must be added to the microbenchmark before the design is frozen.
- **Divergence within a warp of 32 worlds.** Row counts, kinds and friction gating differ between the worlds in a
  lane-per-world block; the bound is the slowest world in the block. Sorting the class list by active row count in
  K1's epilogue keeps blocks homogeneous.
- **Numerical drift acceptance.** K1's dot-product orders differ from the legacy kernels; K2 can be exact. The
  oracle tolerance from the pre-milestone bounds what is accepted.
- **Collision and host floor are outside this plan** and cap two of the three headline numbers. They must be
  reported alongside the solver numbers, not netted against them.

## 9. Addendum (2026-09-09, later): the response-block K2 built and measured in the real solver

The K2 mappings of Section 3.2 were implemented as a drop-in replacement of the matrix-free sweep
(`newton/_src/solvers/feather_pgs/response_block_sweep.py`, solver option `mf_gs_response_block_rows`, off by
default): a warp-per-world build kernel forms `A = J Y^T`, the initial residual and the row metadata for every world
with at most 64 dense-plus-matrix-free rows and appends it to a class list; a lane-per-world sweep (packed symmetric
block in shared memory, rows unrolled at codegen) handles worlds with at most 16 rows; warp-per-world sweeps with one
or two rows per lane handle 17 to 32 and 33 to 64 rows; the legacy kernel keeps the rest. All row semantics are
kept (contact, friction cone with sibling scaling, joint limits, velocity-limit pass, PhysX drives, phases, gating,
stationary exit). Correctness against the legacy kernel on captured inputs: AnymalD max velocity difference 1.7e-5
on values up to 11.9 and impulses 4.8e-6 on 21.5; Allegro (velocity-limit rows, 64-row class) 5.8e-5 on 20.7 and
3.2e-7 on 0.34. In-situ physics statistics match.

Measured in the real solver (GPU 1, 16k worlds, per env step of 8 substeps, Nsight Systems graph span):

| Task | Legacy sweep kernels | Response-block path (build + 3 sweeps + legacy remainder) | Graph span legacy / new |
|---|---:|---:|---:|
| Flat-AnymalD | 1,897 us | 2,108 us summed, overlapped on 3 streams | 8,683 / 8,204 us (5.5% better) |
| Reorient-Allegro | 17,335 us | 28,534 us | 39,608 / 50,013 us (26% worse) |

Why the microbenchmark multipliers (Section 3.2: 6.6 us lane sweep, 278 us for 58 rows) do not transfer:

1. **Real rows cost 10x the instructions of the synthetic unilateral loop.** Friction cones (parent read, square
   root, division, sibling scaling), admission by phase and kind, gating, and the stationary check bring a row update
   to about 90 to 110 warp instructions in every mapping, against about 10 in the microbenchmark. In the warp
   mappings that is the same order as the legacy kernel (about 140), so they cannot beat it by more than 1.3x.
2. **The lane mapping is bounded by world count, not by instruction efficiency.** 16k worlds are 512 warps, three
   per SM: each warp is latency-bound on its own sequential chain (about 12k instructions at 6 to 9 cycles each),
   so the lane sweep lands at 60 to 100 us however cheap each instruction is. The 6.6 us figure assumed a 2.5k-
   instruction warp. Sorting worlds by row count into buckets and removing per-lane branches were necessary but not
   sufficient.
3. **The response block is not free.** Building `A` costs `m^2 D` multiply-adds per world against `2 x iterations
   x m x D` for the legacy re-evaluation of `J v`: equal at 17 rows and 8 sweeps (AnymalD), twice the legacy
   arithmetic at 45 rows and 12 sweeps (Allegro). In-situ the build is 72 us on AnymalD (inputs L2-hot) and 334 us
   on Allegro. A K1 that already holds `J` and `Y` on-chip would absorb it; a separate build kernel does not.
4. **Divergence and reconvergence details matter more than memory.** Two of the four hangs and slowdowns met on the
   way were warp-level: the CUDA driver's lazy compile of a 64-row fully unrolled kernel, and warp-sync intrinsics
   with lanes that had left the block. Both are now understood (runtime row loops; explicit lane masks).

Consequences for the plan: the K2 targets in Sections 3.2 and 5 are withdrawn for the drop-in design. The sweep
can only be amortised across worlds (lane-per-world) when its per-row instruction count is small, which requires the
projection logic itself to be shared across worlds (rows grouped by kind at build time, friction pairs solved as
units), and the block build has to live in K1. The remaining plan order stands (K1 first), with the K2 estimate
replaced by the measurements above. The implementation stays in the worktree behind the option as the reference
for that work.

## 10. Addendum: first K1 conversions (2026-09-09, later)

`newton/_src/solvers/feather_pgs/grouped_dynamics.py` (solver option `grouped_dynamics`, Isaac Lab cfg field, off by
default; `FEATHER_PGS_CHECK_GROUPED=1` runs the serial kernels alongside and prints the maximum difference):

- **Template forward kinematics** (lane group per articulation, level-synchronous, per-template kinematic constants
  read once per warp, body mass properties kept live because Isaac Lab randomises them per world after solver init):
  bit-identical to `eval_rigid_fk_kinematics`; 83 us against 90 us at 8 lanes per articulation (107 us at 4 lanes,
  140 us at 1 lane). Nsight Compute: L2 throughput 82%, 53 cycles per instruction: the kernel is bound by the
  state traffic of body poses, twists and motion subspaces at 32-byte sector granularity, not by constants and not
  by instructions. Only a fused K1 that keeps these on-chip removes that.
- **Grouped inverse dynamics** (backward pass, children gathered onto their parent in descending joint order):
  bit-identical to `eval_rigid_tau_add`; 59 to 66 us against 100 us.
- **Grouped response solve** `Y = H^-1 J^T` (warp per articulation, lane per row, factor in shared memory,
  predicated unrolled register loops): 4e-6 against the tiled kernel; 72 us against 148 us on AnymalD (18 DOF);
  456 us against 417 us on Allegro (16 DOF, 45 to 58 rows), so it is not a win there.

In-situ result with all three on (GPU 1, 16k worlds, Nsight Systems graph span per env step, physics statistics
identical): AnymalD 8,683 -> 7,840 us (9.7% faster); Allegro 39,608 -> 39,904 us (neutral). Together with the
response-block sweep (Section 9) the day's kernel-level work moved AnymalD by about 10% and Allegro by nothing,
which is the measured statement of what per-kernel rewrites can do against a 684 us dynamics stage: the remaining
gap is structural (state traffic and launch count), and the fused K1 with on-chip state is the next piece of work.

## 11. Addendum: the fused K1 and a baseline correction (2026-09-09, later still)

### 11.1 The legacy graph-mode baseline was wrong physics

While validating the fused kernel, its in-situ runs showed 2.5x more contacts than the legacy runs (161k against
63k on AnymalD at 16k worlds after 240 steps). Bisection with identical step counts and per-step statistics
(`run_profiled.py --trace-stats`) showed:

| path | launch mode | contacts after step 0 / 11 (6+6 steps) | contacts after 240 steps |
|---|---|---|---|
| legacy | eager | 47,538 / 86,598 | 161,871 |
| legacy | CUDA graph | 46,917 / 52,669 | 63,016 |
| legacy, FK/ID cache off (`FEATHER_PGS_FK_ID_CACHE=0`) | CUDA graph | 47,536 / 86,596 | 161,685 |
| fused K1 | eager or graph | 47,538 / 86,599 | 161,517 to 161,669 |

The legacy solver behaves differently under graph replay than in eager mode, and the difference disappears when the
FK/ID cache is disabled. Eager, cache-off and K1 agree step by step, so the eager result is the reference: under
random actions the robots collapse (contacts grow), whereas the graph-mode legacy run keeps them standing. Not the
cause: parallel streams, the mass-update interval, `pgs_mode`, cache-flag zeroing, or a stream race (a 1 ms spin
kernel at step start changed nothing).

Root cause, found with a device-side compare of the cached FK/ID arrays against a fresh evaluation that runs inside
the graph (`FEATHER_PGS_DEBUG_CACHE_CMP=1`): at the first substep of every replay the cached bias force `body_f_s`
was exactly zero for all 278,528 bodies while poses, motion subspaces, twists and accelerations were exact. The
solver allocates its augmented state lazily on the first `step()`; Isaac Lab makes that first call inside the CUDA
graph capture, and a `wp.zeros()` issued during capture becomes a memset node. Every replay therefore re-zeroed
`body_f_s` (and `body_ft_s`, `joint_qdd`) between the two captured substeps, so the gravity and Coriolis bias
forces were dropped on every other substep. Fix: allocate the augmented state in the constructor. This is a general
rule for graph-captured solvers: no array allocation inside the capture region.

Consequence: every graph-mode FeatherPGS number quoted so far for AnymalD (8,683 us per env step legacy, 7,840 with
the grouped kernels) was measured in a regime with 2.5x fewer contact rows. In the correct regime the sweep alone is
514 + 318 us per substep. Corrected AnymalD baselines with the fix (16k worlds, graph mode, physics identical to
eager, Nsight Systems graph span per env step, 200+40 steps): legacy 15,664 us; grouped kernels without K1 14,872;
grouped kernels with the kinematics-only K1 15,681. The MJWarp comparison for AnymalD has to be redone against
these; the 3.9x figure in Section 1.3 is overstated by roughly 2x for this task. Other tasks are affected only if
they use the FK/ID cache under graph capture in the same way (all FeatherPGS runs in Isaac Lab do), so every row of
the Section 1 and Section 5 tables needs re-measurement.

### 11.2 Fused world-dynamics kernel (`fused_dynamics.py`, class S)

One launch per substep per size class, a lane group (8 lanes) per articulation, 8 articulations per block, all
intermediate state in shared memory (about 2.5 KB per articulation): Pass 1 poses level by level (`jcalc_transform`
port), Pass 2 motion subspaces, twists, accelerations, bias forces (`jcalc_motion` port), Pass 3 backward inverse
dynamics into `joint_tau` (`jcalc_tau` port) with the compact inertia terms for the legacy composite kernel. A
variant (`FEATHER_PGS_FUSED_K1_MASS=1`) also forms the composite inertia, the joint-space inertia and the Cholesky
factor in the same launch.

Exactness (`FEATHER_PGS_K1_REAL_CHECK=1`, legacy chain recomputed from the same inputs every substep): body poses
1.5e-5 on 160 m, motion subspaces 1.5e-5 on 1, twists 0, accelerations 0, bias forces 7.5e-4 on 332 N,
joint torques 7.5e-4 on 588 N m, factor 3.9e-5 on 7.6: float rounding, no structural difference. Physics statistics
match the eager reference step by step (Section 11.1).

Cost (Nsight Systems, 16k worlds, per launch): kinematics-only 156 us against the chain it replaces in the correct
regime (`eval_rigid_fk_id` 4 cached + `eval_rigid_tau_add` 98, or the grouped `grouped_tau` 56 + cached FK); with
the mass path 530 us (32 lanes) or 630 us (8 lanes) on mass substeps, against composite 94 + tiled CRBA/Cholesky
62. Nsight Compute on the mass variant: 6% theoretical occupancy (32 KB static shared per block of 4
articulations), 12% issue-active, stalls dominated by long scoreboard. The mass path is therefore opt-in and off;
the tiled CRBA/Cholesky at 11 to 62 us for 16k worlds is already near the traffic floor and does not belong in a
lane-group kernel.

Balance with the corrected baseline (per env step, 8 substeps, graph mode): the fused launch costs 1,331 us and
replaces `grouped_tau` 523 + the cached `eval_rigid_fk_id` 34, so K1 as it stands is a net loss of about 800 us per
env step (15,681 against 14,872). The kernel does the same work as the chain and still publishes the same state,
and it does it with 17% occupancy (shared memory bound at 20 KB per block on a 100 KB carveout), 10% issue
utilisation and long-scoreboard stalls on template and state loads. Lane count 8 -> 16 did not help (164 us); a
merged pose+motion sweep was slower (177 us). The win it enables is structural: with poses recomputed at substep start from
`joint_q`, the end-of-substep `eval_rigid_fk_kinematics` + `finalize_body_dynamics` (639 + 272 us per env step)
are needed only where an external consumer reads body state (collision every `collide_every` substeps, sensors at the
env step), and the row build and response block can consume the on-chip poses directly. That is the next checkpoint.

### 11.3 Re-measurement of every task with the fix (2026-09-09, GPU 1, 16k envs, graph mode)

Nsight Systems graph span per env step, 200 warm-up + 40 measured steps, 3 profiled; artifacts in
`profile_artifacts_20260909_fixed/` (`make_table.py` regenerates the table). "grouped" is `grouped_dynamics=True`
with the fused K1 off. MJWarp is unchanged code, re-run under the same harness settings.

| Task | FPGS legacy | FPGS grouped | MJWarp | FPGS speedup over MJWarp, legacy | FPGS speedup over MJWarp, grouped | contacts FPGS / MJWarp |
|---|---:|---:|---:|---:|---:|---|
| Flat-AnymalD | 15,678 | 14,872 | 35,068 | 2.24x | 2.36x | 161k / 107k |
| Cartpole | 244 | 244 | 542 | 2.22x | 2.22x | 0 / 0 |
| Ant | 2,069 | 2,050 | 2,240 | 1.08x | 1.09x | 52k / n.a. |
| Humanoid | 8,230 | 8,071 | 8,469 | 1.03x | 1.05x | 73k / n.a. |
| Lift-KukaAllegro | 18,602 | 18,954 | 68,897 | 3.70x | 3.63x | 241k / 238k |
| Reorient-Cube-Allegro | 39,713 | 39,887 | 58,970 | 1.48x | 1.48x | 217k / n.a. |
| Rough-G1 | 42,769 | 41,517 (tiled response solve) | 51,104 | 1.19x | 1.23x | 112k / 108k |
| Lift-Franka | 5,734 | 6,113 | 26,391 | 4.60x | 4.32x | 1.6k / 1.1k |

Tasks that stood still under the bug (hands, Ant, Humanoid, Cartpole, Lift-Franka at 5,734 against 5,096 before) are
within about ten percent of the earlier numbers. The two legged tasks moved: AnymalD 8,683 -> 15,678 (contacts 63k -> 161k) and Rough-G1 34,696 -> 42,769
(contacts 54k -> 112k); their ratios drop from 3.9x to 2.2x and from 1.4x to 1.2x. The Allegro reorient ratio rose
from 1.1x to 1.5x because the earlier MJWarp figure came from a different run configuration.

Note on G1: the grouped response solve (`grouped_hinv_jt`) is wrong above 32 DOF (G1 has 43; check mode shows
differences of 1e3 against the tiled kernel), which drove the state non-finite. It is now gated to <= 32 DOF and G1
falls back to the tiled kernel; the earlier 34,696 us G1 figure was a stale-cache run, not a grouped-kernel run.

### 11.4 K1 after staging the template tables in shared memory

With one template per block, the per-joint template records (parent, type, offsets, child, children range, `X_p`,
`X_c`, axes, level tables; about 2 KB per block) are copied to shared memory once per block instead of being
re-read from global memory per lane per pass: the fused launch drops from 166 to 110 us (bit-exact, real-mode check
unchanged). Skipping the two outputs nobody reads (`body_a_s`, `body_ft_s`) is folded in. In situ AnymalD is now
15,014 us per env step against 14,872 for the grouped chain without K1: K1 costs 883 us per step and replaces 557,
so it is within 330 us of break-even before any structural saving.

Follow-ups measured the same way (16k AnymalD, per launch): 16 lanes per articulation with the staged tables
109 us and `template_fk_kinematics` 80 -> 55 us as a side effect (the grouped chain without K1 also benefits: 14,482 us
per env step at 16 lanes); dropping the three unread publishes (`body_a_s`, `body_ft_s`, `body_f_s`) 102 us; hoisting
every global load above the stores of the same joint 97.5 us (all bit-exact in the real-mode check). In situ:
14,864 us per env step against 14,482 for the grouped chain without K1 at the same lane count, so K1 still costs
about 380 us per env step (780 us of fused launches replacing 540 us of `grouped_tau` plus cached FK).

### 11.5 Where the time goes in the correct regime (AnymalD, grouped chain, 16 lanes, 14,102 us busy per env step)

| Stage | us per env step | share | dominant kernels (us per launch, 8 launches) |
|---|---:|---:|---|
| sweep (s5) | 6,700 | 47% | full re-evaluation kernel for worlds above 32 rows 511; incremental kernel 318 |
| rows (s4) | 2,343 | 17% | populate_world_J 184, diag 48, clear rows 33, contact bias 30 |
| response Y = H^-1 J^T | 1,327 | 9% | grouped_hinv_jt 106 |
| collision + contact forces | 928 + ~400 | 9% | narrow phase 115 and 50 (4 launches), contact force kernels |
| dynamics (s1: FK, ID, CRBA, composite) | 1,636 | 12% | grouped_tau 64, template_fk 56, crba_cholesky 47, composite 96 (4), finalize 35 |
| trisolve, integrate, sensors, misc | ~1,170 | 8% | |

Two conclusions replace Section 1.3's ordering. First, stage 1 is 12% of the busy time, so a perfect K1 (zero cost,
end-of-substep publish removed) saves at most about 1,600 us of 14,100: the fused-dynamics work of Sections 10 to
11.4 is capped at roughly 11% on this task and cannot be the decisive stage. Second, the sweep is 47%, and 60% of
the sweep is the legacy full re-evaluation kernel that runs for worlds with more than 32 active rows, because the
incremental kernel is lane-per-row and capped at 32 rows. In the correct regime (robots on the ground) a large share
of worlds exceed 32 rows. The next work items are therefore: the row-count distribution in this regime (capture in
progress), an incremental sweep for 33 to 64 rows (two rows per lane, or the warp64 response-block path), and the
row build (`populate_world_J` 184 us).

### 11.6 The sweep in the correct regime

Row statistics of the captured sweep inputs (AnymalD, step 1608, robots on the ground): mean 30 active rows per
world, median 30, p90 37, maximum 48; 61% of worlds have at most 32 rows and 39% have 33 to 48. No matrix-free rows.

Measured per launch (8 launches per env step, grouped chain, 16 lanes):

| configuration | sweep kernels us per launch | env step us |
|---|---:|---:|
| incremental <=32 rows + legacy for the rest (today's default) | 318 + 511 | 14,482 |
| incremental two tiers (32; 48, two rows per lane) | 327 + 604 + 9 | 15,358 |
| incremental two tiers (32; 64) | 326 + 873 + 9 | 17,646 |
| legacy full re-evaluation kernel for every world | 690 | 13,500 |
| response-block sweep, rows=64 (Section 9) | 902 + 452 + 221 build + 90 | 17,577 |

The incremental-residual kernel, a win at 11 active rows (Section 3), is a loss at 30: its response-block build
re-reads J and Y from global memory per row pair (m^2 x 144 bytes, 130 KB per world at 30 rows, 1.3 GB per launch),
and the sequential row loop gains little from the incremental residual when the build costs as much as eight full
sweeps. Turning it off is the best measured AnymalD configuration so far: 13,500 us per env step, 2.60x MJWarp
(35,068). A rebuilt A-build that stages J once in shared memory and keeps each lane's Y rows in registers is being
measured next; if it does not beat 690 us the default row budget should become 0 for contact-heavy tasks.

Result of the staged build (J once in shared memory, each lane's Y rows in registers; bit-exact statistics): 323 us
at 32 rows (unchanged) and 680 us at 48 (worse, register pressure from two Y rows per lane). The build is not the
bottleneck; the sequential row loop is (8 iterations x 30 rows x about 100 instructions and two warp syncs per row,
57 SM-cycles per row update). The staging was reverted; the two-tier codegen stays (opt-in above 32 rows). The row
budget default is being decided per task: AnymalD is best with the incremental kernel off (13,500 us).

Per-task check of the row budget (graph span us per env step, grouped chain, 16 lanes): Lift-Franka 7,016 (off) vs
6,972 (32), G1 40,168 (off) vs 41,517 (32), AnymalD 13,500 (off) vs 14,482 (32). The Isaac Lab preset default
`mf_gs_incremental_rows` is set to 0; the kernel stays available. Lift-Franka also shows that the grouped kernels are
a loss on a fixed-base 9-DOF arm (5,734 legacy, 6,113 grouped at 8 lanes, 7,016 at 16 lanes): `grouped_dynamics`
should stay off there and on for the legged robots.

Best measured configurations after this work (MJWarp / FPGS, per env step): AnymalD 13,500 (2.60x), G1 40,168
(1.27x), Lift-Franka 5,734 (4.60x), Lift-KukaAllegro 18,602 (3.70x), Reorient-Allegro 39,713 (1.48x), Humanoid
8,071 (1.05x), Ant 2,050 (1.09x), Cartpole 244 (2.22x).

## 12. Decision memo: the shape of the contact solve (2026-09-09, evening)

### 12.1 Method

Rather than argue from a roofline again, the candidate iteration schemes were run offline on the real per-world
Delassus operators captured from the corrected-regime AnymalD run (`scratch/cap/anymald_fixed`, step 1608, 512
contact-only worlds sampled, 29.8 rows per world, 9.9 contacts). Script: `scratch/formulation/study.py`. Each scheme
runs from the same warm-start impulses; error is the median relative error of the resulting change in generalized
velocity, either against the scheme's own converged solution (convergence speed) or against the converged row
Gauss-Seidel solution (model agreement). Converged solutions are also checked for the properties a correct contact
solution must have: normal residual velocity >= 0, complementarity `lambda_n r_n = 0`, friction inside the cone, and
maximal dissipation (friction opposes the tangential residual velocity when sliding).

### 12.2 Results on AnymalD

| scheme | converges to | consistency of converged solution | sweeps to reach today's accuracy |
|---|---|---|---|
| row Gauss-Seidel (today: sequential rows, friction pair projected onto the disk) | reference | min r_n -2.6e-2, max lambda_n r_n 1.0e-2, cos(friction, -v_t) 0.88 when sliding | 8 (by definition, 3.1e-3) |
| contact-block GS / Jacobi, 3x3 block step `D^-1 r` then cone projection | 32% different velocity change, 9% more total normal impulse | min r_n -1.95, max lambda_n r_n 12: violates complementarity, wrong solution | 2 to 4 (to the wrong solution) |
| parallel scalar projection (all rows at once, `lambda_i -= w r_i / A_ii`, per-contact cone projection), w = 0.7 to 0.8 | 1.3% from row GS | min r_n -1.5e-5, max lambda_n r_n 6e-6, cos 0.98: more consistent than row GS | 16 to 24 (1.2e-2 to 1.8e-2 of its own fixed point; 32 gives 1.2e-3); w = 1.0 oscillates |
| the same with Nesterov momentum, w = 0.5 | 1.3% from row GS | same | 12 to 16 (2.0e-2 to 1.6e-2) |
| APGD (Nesterov on the QP, step 1/L) | 1.5% from row GS | min r_n -1.6e-3, cos 1.00: the most physical | 64+ (slow: 8e-4 at 64) |

Two conclusions. The 3x3 block metric that makes block schemes converge in two sweeps is a different and wrong
friction model on these operators: it is rejected. The parallel scalar-step projection converges to a solution that
is at least as physical as today's (better complementarity and dissipation) and within 1.3% of it in velocity, needs
about 3x the sweeps of row Gauss-Seidel, and is the GPU-native mapping: each sweep is one dense matrix-vector product
of the per-world response block (30 x 30, in shared memory) plus one independent cone projection per contact, no
sequential dependency, no per-row sync.

Cost model per world (warp instructions): today 8 sweeps x 30 rows x about 100 = 24,000. Parallel scheme: build A
once (m^2 D / 32 lanes, about 1,000 with loads) plus 20 sweeps x about 100 (30 FMA + 30 shared loads for the
matrix-vector product, 25 for the projection, sync and early-exit ballot) = 3,000. About 7x fewer instructions; at the
current issue efficiency the AnymalD sweep goes from 690 us to roughly 100 us per substep, which alone is 35% of the
env step. The kernel shape is one warp per world for m <= 32 with `A` stored transposed so the matrix-vector product
reads are conflict-free, two rows per lane up to 64, sixteen lanes per world (two worlds per warp) below 16 rows.

This agrees with the literature: Tonge et al. (SIGGRAPH 2012) show projected Jacobi with relaxation/mass splitting
matching PGS wall-clock while removing order dependence; Mazhar et al. (TOG 2015) show accelerated projected
gradient (Nesterov) beating Gauss-Seidel by one to two orders of magnitude at scale because it parallelises; MJWarp
itself abandons Gauss-Seidel for a Newton method on compacted fixed-size tiles. FeatherPGS's advantage is that its
per-world operator is tiny (m <= 64) and already available as `J` and `Y = H^-1 J^T`, so the whole solve fits in
one warp's shared memory: the Newton-style direct solve is also open to us later (an m x m Cholesky per iteration is
about 140 lane-FMAs), with the parallel projection as the robust first version.

### 12.3 What is being verified before committing

The same study on the two hand tasks (45 to 58 rows with an object) and on G1, where coupling between contacts is
stronger; results appended below when available. The decision rule: adopt the parallel scalar projection if it
converges (no oscillation at w = 0.7) to within 1.5% of row Gauss-Seidel on every captured task and needs at most
about 24 sweeps; otherwise fall back to Nesterov-APGD with the same kernel skeleton.

### 12.4 Results on all four captures (`scratch/formulation/study2.py`; drive rows frozen, limits solved)

Relaxed Jacobi with a fixed w diverges on the hand tasks and G1 (many contacts coupled through one body): a fixed
relaxation is not acceptable. Per-row step scaling `w_i = A_ii / sum_j |A_ij|` (the generalisation of Tonge's mass
splitting) and the Lipschitz step of APGD are both stable on every world of every task. Error is the relative error
of the generalized velocity change; "today" is row Gauss-Seidel at 8 sweeps against its own converged solution.

| task (rows, contacts per world) | today: median / p90 / max | scaled parallel, 24 sweeps | scaled + Nesterov, 24 sweeps | scaled + Nesterov, 48 sweeps | converged parallel vs converged row GS (median / p90) |
|---|---|---|---|---|---|
| AnymalD (30, 10) | 3.8e-3 / 2.0e-2 / 6.8e-2 | 4.4e-2 / 9.1e-2 / 0.16 | 8.0e-3 / 1.5e-2 / 0.12 | 1.3e-3 / 5.5e-3 / 0.14 | 1.3e-2 / 3.7e-2 |
| Reorient-Allegro (53, 13.5, 12 drive rows frozen) | 1.1e-2 / 0.21 / 0.84 | 0.14 / 1.1 / 3.3 | 3.3e-2 / 0.36 / 1.8 | 6.3e-3 / 0.12 / 2.2 | 2.2e-3 / 0.46 |
| Lift-KukaAllegro (35, 11) | 0 / 0.10 / 0.44 | 0 / 1.5e-2 / 0.88 | 0 / 1.7e-3 / 0.81 | 0 / 5.5e-5 / 0.44 | 0 / 2.6e-15 |
| Rough-G1 (28, 8) | 4e-7 / 7.7e-2 / 0.75 | 1.2e-2 / 0.23 / 0.85 | 2.9e-3 / 9.1e-2 / 0.70 | 3.2e-4 / 4.0e-2 / 0.54 | 6.2e-3 / 0.19 |

Reading: (1) today's solver is itself far from converged on the hardest tenth of hand and humanoid worlds (p90 0.1
to 0.2, max 0.4 to 0.8 after 8 sweeps); the accuracy bar is the median plus a bounded tail, not convergence.
(2) The scaled parallel projection with Nesterov momentum reaches today's median accuracy in 16 to 24 sweeps on the
legged tasks and 32 to 48 on the hands, and its tail at 48 sweeps is comparable to today's tail. (3) Its converged
solution differs from the row Gauss-Seidel solution by about 1% in the median (a different but equally valid friction
fixed point; complementarity holds to 1e-3 or better on all tasks), with a 10% tail on the hands and G1 where
neither method is well converged. (4) Plain APGD with the 1/L step is slower than the scaled scheme on every task
and is not needed.

### 12.5 Recommendation

Build the sweep as one warp per world over the per-world response block in shared memory, with the **scaled parallel
projection with Nesterov momentum** as the iteration: each sweep is one matrix-vector product (lane per row, `A`
stored transposed), one independent projection per contact (normal clamp, friction pair onto the disk), one ballot
for early exit on the residual. Per-row step scales and the Nesterov sequence are computed once per launch from `A`.
Warm start and the existing phase structure are unchanged. Expected cost per world: `A` build about 1,000 warp
instructions plus about 100 per sweep; at 24 sweeps that is 3,400 against 24,000 today (7x), and early exit lets
easy worlds stop at 8 while hard hand worlds run to 48 to 64 within the same launch. Expected sweep stage on AnymalD:
690 -> roughly 100 us per substep; hands roughly 3x.

What this is not: it is not a convergence upgrade. A solver that converges the hard worlds (hand with object,
humanoid pile) needs a Newton-type method on the same shared response block (the route MJWarp took); with m <= 64
the per-iteration factor is about 650 lane-FMAs, so it is affordable, but it is a multi-week build with real risk in
the nonsmooth friction handling. The parallel projection is the right first version because its kernel skeleton
(response block in shared memory, lane per row, per-world early exit) is exactly what the Newton path would reuse.

Validation plan before merging: the offline harness runs the new kernel on the four captures against row
Gauss-Seidel (median and tail), then in situ physics statistics per step on all eight tasks, then policy training
curves on AnymalD and Allegro (the only test that settles whether the 1% fixed-point difference matters).

### 12.6 Newton per world: prototype results (`scratch/formulation/sap_proto.py`, captures with the mass-matrix factor)

The capture now includes each articulation's Cholesky factor, so the prototype works with the true `H` (checked:
`Y = J H^-1` to 3e-6). Two Newton formulations were built and run on 48 to 64 worlds of AnymalD and Allegro:

1. **Convex compliant contact (SAP / MuJoCo family).** Primal Newton in `v` with the closed-form cone projection in
   the regularization metric and an exact line search. It converges (18 to 40 iterations) and satisfies maximal
   dissipation exactly, but its converged solution differs from the rigid Coulomb solution by 24% (median) on
   AnymalD and 2% on Allegro, with complementarity violations up to 3.2. This is not a bug: the convex cone
   projection assigns a normal impulse to a separating contact whenever it slides faster than `r_n / mu` (the
   "hovering while sliding" relaxation of the convex family), and the algebra shows it is independent of the
   friction regularization ratio. On collapsed robots most contacts slide, so the effect is large. It is the model
   MuJoCo uses, so MJWarp comparisons would stay apples to apples, but it is a different physics from today's.
2. **Staggered Newton for rigid Coulomb friction.** Outer loop fixes each contact's friction radius from the current
   normal impulse; the inner problem is then a convex QP solved exactly by Newton; the radii are iterated to a fixed
   point. Results (beta = 1e-3 compliance, cold start):

| task | outer iters (median / p90 / max) | total Newton iters | vs converged row GS | vs converged parallel projection | consistency |
|---|---|---|---|---|---|
| AnymalD | 6 / 8 / 8 | 28 / 34 / 40 | 2.7e-2 / 5.7e-2 / 0.15 | 9.0e-3 / 3.0e-2 / 6.5e-2 | min r_n -7e-3, max lambda r 3.8e-3, cos 1.000 |
| Allegro | 4 / 8 / 8 | 20 / 37 / 42 | 1.9e-5 / 1.4e-2 / 5.0e-2 | 6.2e-5 / 7.1e-3 / 1.7e-2 | min r_n -1.7e-5, max lambda r 1.5e-6, cos 1.000 |

It converges on every world of both tasks including the hand worlds where the fixed-point iterations leave a 10 to
50% tail, it recovers the rigid Coulomb model (agreement with the two rigid references at the 1% level, exact
maximal dissipation, complementarity at the compliance level), and it has no sliding artefact. Cost per world is
about 20 to 40 Newton iterations of `n^2 m + n^3/6` plus line-search evaluations; on Allegro that is roughly 18k warp
instructions against about 60k for today's 12 Gauss-Seidel sweeps, so it is both converged and about 3x cheaper
there; on AnymalD it is comparable to today, not cheaper. Warm-starting radii and velocity across substeps (which the
solver already does for impulses) and a cheaper line search should cut the iteration count substantially; the
prototype starts cold.

### 12.7 Decision

Build the per-world solver on the shared-memory response block with two stages in one kernel: the scaled parallel
projection with Nesterov momentum first (cheap, stable, converges most worlds in 16 to 24 sweeps), then the
staggered Newton polish for worlds whose residual has not met tolerance (hands, humanoid piles). Both stages use the
same data (`J`, `Y` or `H`, `A`), the same per-contact projection code and the same per-world early exit, so the
kernel skeleton is shared and the Newton stage can be added after the projection stage is in production. Rigid
Coulomb friction is kept as the contact model; the convex compliant model is documented as a deliberate non-choice
because of the sliding artefact on legged robots. The offline harness (`study2.py`, `sap_proto.py`) is the oracle for
both stages: every kernel version is checked on the four captures before it touches the solver.

## 13. Stage 1 kernel: scaled parallel projection in the solver (2026-09-09, night)

`_get_pgs_solve_parallel_kernel` (solver kwargs `mf_gs_parallel_rows`, `mf_gs_parallel_sweeps`,
`mf_gs_parallel_nesterov`, `mf_gs_parallel_tol`; Isaac Lab cfg fields of the same names; off by default). Same
signature and world hand-off as the incremental kernel (the legacy kernel skips the worlds it owns), two tiers
(<= 32 rows, one row per lane; 33 to 64 rows, two rows per lane). Warp per world: `J` staged in shared memory,
response block `A` built row-major with padded stride, residual reference folded to zero impulse, per-row step
scales, Nesterov, per-contact projection, tolerance early exit, one reconstruction of `v`.

Oracle (`scratch/formulation/oracle_parallel.py`, AnymalD capture, 16k worlds): kernel matches the numpy
implementation of the same algorithm to 3e-6 relative in every configuration; kernel vs legacy row Gauss-Seidel
differs by up to 16 to 22% on individual worlds (the model difference measured in Section 12). Cold-start timing per
launch: 32-row tier 53 us fixed + 3.7 us per sweep (137 us at 24 sweeps, 61% of worlds); 48-row tier 106 fixed +
7.3 per sweep (274 us at 24 sweeps, 39% of worlds); legacy kernel 682 us. Things that did not help: packed
symmetric `A` (index math and bank conflicts cost more than the halved build), `Y` rows in registers (register
pressure slowed the sweeps).

In situ (AnymalD, grouped chain, 16 lanes, graph mode): physics statistics within 0.2% of the legacy trace step by
step; graph span 11,541 us per env step at 24 sweeps and 11,217 at 16, against 13,500 for the best legacy
configuration (15 to 17% faster whole graph; 3.04x to 3.13x MJWarp). Sweep kernels 440 us per substep at 24 sweeps
(307 for the 33-to-48-row tier, 133 for the 32-row tier) against 690. The wide tier is the cost centre: two rows per
lane and 14 KB of shared memory per warp (7 warps per SM).

### 13.1 Wide tier as a two-warp block; build variants

Two rows per lane was the wide tier's problem (7 warps per SM, doubled per-thread sweep work). It now runs as a
64-thread block with one row per thread, block-level sync and a shared early-exit flag, and no `J` staging (shared
memory 12.5 KB per block). The response-block build has two variants selected at codegen by DOF count: pairs
spread over the threads with `J`, `Y` from L1 (better at D = 18: AnymalD wide tier 274 -> 186 us), or row per
thread with `J_i` in registers and `Y_j` broadcast (better at D = 43: G1 wide tier 476 -> 382 us in one run, 513 in a
repeat; noisy). Symmetric packed storage and `Y` rows in registers were tried and rejected (slower).

In situ (graph mode, 16k envs): AnymalD 10,898 us per env step at 24 sweeps (legacy best 13,500; 3.22x MJWarp), sweep
kernels 225 + 154 + 9 us per substep against 690. G1 is a loss (43,574 vs 40,168): with 43 DOF the m^2 D build and
the per-sweep matrix-vector product cost more than the row-serial kernel saves, and worlds above 64 rows still go to
the legacy kernel. Per-task setting for now: `mf_gs_parallel_rows=48` on AnymalD, 0 on G1; a matrix-free variant of
the parallel sweep (no `A`, per-sweep `Y^T` and `J` products, row-sum bound for the step scale) is the candidate for
high-DOF robots.

### 13.2 Accuracy in situ and the sweep budget

A capture taken while the parallel kernel was running (its own warm start, AnymalD step 1608) gives the in-situ
accuracy: the 24-sweep scaled Nesterov iteration is 7.9e-3 (median) / 1.5e-2 (p90) / 9.8e-2 (max) from its converged
solution, against today's 8-sweep row Gauss-Seidel at 3.8e-3 / 1.9e-2 / 0.149: a slightly worse median, a better
tail. Fewer sweeps are not free: at 16 and 12 sweeps the end-state contact count rises (158k -> 168k -> 182k), the
physics softens and the wide tier gets more rows, so the env step does not get faster (10,945 and 11,253 us against
10,898 at 24). On this capture the two tiers cost 172 + 160 us per launch against 665 for the legacy kernel (2.0x).
Nesterov without restart fails to settle on a few worlds (one converged reference shows an approach velocity of
0.17); an adaptive restart is the standard remedy and is the next kernel change.

### 13.3 Adaptive restart

Nesterov momentum with O'Donoghue-Candes adaptive restart (drop the momentum whenever `(y - x_new) . (x_new - x_old)
> 0`; one warp reduction per sweep, exact against the numpy reference to 4e-6). On the warm-started AnymalD capture,
error against the scheme's own converged solution (median / p90 / max):

| sweeps | Nesterov | Nesterov + restart | today (row GS, 8 sweeps) |
|---:|---|---|---|
| 16 | 1.1e-2 / 3.4e-2 / 0.11 | 9.9e-3 / 3.1e-2 / 9.0e-2 | 3.8e-3 / 1.9e-2 / 0.149 |
| 24 | 7.9e-3 / 1.5e-2 / 9.8e-2 | 2.4e-3 / 7.6e-3 / 2.7e-2 | |
| 32 | 3.2e-3 / 1.1e-2 / 6.1e-2 | 3.7e-4 / 2.8e-3 / 2.2e-2 | |
| 48 | 1.4e-3 / 4.4e-3 / 8.5e-2 | 1.5e-5 / 2.3e-4 / 2.0e-2 | |

With restart the 24-sweep solve is more accurate than today's solve on every statistic and the converged
solution satisfies complementarity (min r_n -2e-4 against -0.17 without restart). The reduction costs about 25% per
sweep in the oracle (32-row tier 160 -> 199 us at 24 sweeps); 24 sweeps with restart is the default.

## 14. Row build: one thread per (contact, dof) (2026-09-09, night)

Nsight Compute on the legacy `populate_world_J_for_size`: 65k threads (capped grid-stride loop over 161k contacts),
8% issue utilisation, 27 cycles of long-scoreboard stall per instruction: one thread per contact walking the kinematic
chain six times (normal and two friction rows, two bodies) and accumulating with read-modify-write stores into a
pre-cleared Jacobian. Pure latency, 184 us per substep on AnymalD plus 33 us for the clear.

`row_build.py::populate_world_J_masked` (env `FEATHER_PGS_ROWS_MASKED=1`, articulations up to 32 DOF): one thread
per (contact, local dof). Each thread recomputes the contact scalars (its 32 sibling threads read the same contact,
L1-resident), tests the per-body response-dof bitmask the solver already builds (`body_response_dof_mask`) instead
of walking the chain, and writes the three Jacobian entries of its dof directly with coalesced stores; lane 0 writes
the row metadata. Bit-identical to the legacy kernel in check mode (`FEATHER_PGS_CHECK_ROWS=1`: Jacobian and all
eight metadata arrays differ by exactly 0 over 8 substeps).

Measured iterations of the row build (AnymalD in situ, per substep): a single kernel with 32 threads per contact all
repeating the contact prelude ran at 1,927 us (the prelude is the expensive part, not the chain walk); a two-kernel
split with per-contact geometry scratch fell into the capacity trap twice (11.2M-entry contact buffers: 868 us of
early-exiting threads, then 1 GB of memsets captured into the graph from the scratch allocation); the current form,
a per-contact prelude writing one flag word (51 us, grid-stride over 262k threads) plus a per-(contact, dof) Jacobian
pass that recomputes the cheap geometry (107 us), is at 158 us against 184 legacy, 1 ulp from the legacy Jacobian.
Both kernels are latency bound on dependent gathers (contact -> shape -> body -> pose). The intended final form is one
native kernel per contact-warp: lane 0 does the prelude once and broadcasts by shuffle, all lanes write their dof,
no scratch at all; estimated 50 to 60 us. The clear of the active Jacobian rows (33 us) becomes unnecessary once the
contact rows are written in full.

### 14.1 State after the row build and sweep ILP (AnymalD, 16k envs, graph mode)

Sweep matrix-vector product with four independent accumulators and float4 loads of the extrapolated impulses:
narrow tier 199 -> 172 us cold (oracle), wide tier unchanged. Skipping the active-row Jacobian clear when the masked
builder writes the contact rows in full: physics identical step by step (contact counts differ by 0 to 5 of 47k to
63k). Env step: **10,923 us** (legacy best 13,500; 3.21x MJWarp; the day started at 15,678 in the corrected regime).
Per substep: wide-tier sweep 255, narrow-tier sweep 160, Jacobian pass 108, prelude 51, response solve ~100,
grouped inverse dynamics 63, template FK 55, diag 48, CRBA 47.

Native warp-per-contact builder (`row_build_native.py`, `FEATHER_PGS_ROWS_NATIVE=1`): 239 us per substep, slower than
both the two-kernel Warp version (158) and the legacy kernel (184), and one substep in eight disagreed with the
legacy rows in check mode (a row-type/slot mismatch not yet understood). Withdrawn; the two-kernel version stays.
The lesson from the four row-build iterations: the per-contact prelude (materials, friction anchors, prescribed
targets, 24 scattered metadata stores) costs more than the chain walks did, and it is bound by dependent gathers,
not by threads.

## 15. Lazy kinematics with the fused K1 (2026-09-09, night)

Solver kwarg `lazy_kinematics` (Isaac Lab cfg field of the same name): with the fused world-dynamics kernel active,
the end-of-substep `eval_rigid_fk_kinematics` + `finalize_body_dynamics` publish is skipped; the Isaac Lab manager
calls `solver.publish_kinematics(state_0)` before mid-loop collision and at the end of each decimation step, which is
where body poses and CoM velocities are read. K1 recomputes everything from `joint_q` at each substep start, so the
solver never uses the published state itself. Physics trace identical to the baseline (contact counts within 0 to 5
of 47k to 63k per step). AnymalD env step **10,626 us** (from 10,923; 3.30x MJWarp): FK publish 8 -> 4 launches per
step, `grouped_tau` 63 replaced by `fused_dynamics` 94 (K1 now pays for itself because the publish it removes is
larger than its own excess). Default off until validated on the other tasks.

## 16. Parallel sweep for the hands: velocity-limit and drive rows, 96-row tier (2026-09-09, night)

The parallel kernel now treats joint velocity-limit rows (type 4) as unilateral rows (the legacy kernel applies them
as a monotone one-sided impulse within a launch; the unilateral projection is the proper complementarity form and
differs from it only when a limit impulse would have to relax within the same launch), applies the PhysX drive
formula to drive rows (type 1) as their own fixed-point update, and adds a third tier (65 to 96 rows, three warps per
world). The host gates on velocity limits and drive rows are removed; the 32-row limit on grouped kernels does not
apply here. Oracle checks on the Allegro (up to 95 rows, velocity-limit rows present) and KukaAllegro captures and
in-situ traces against the legacy path follow.

Results: exact against the numpy reference on all three tiers of the Allegro capture (velocity-limit rows present);
in-situ Allegro physics within 0.5% of the legacy trace; env step 38,161 us against 39,713 (4%). The sweep stage is
at parity, not better: 64-row tier 1,070 us for 12.6k worlds, 96-row tier 424 for 2.9k, legacy 531 for the 3.8k worlds
above 96 rows, total 2,074 against 2,093. On the hands the response-block build dominates (m^2 D = 55^2 x 22 = 66k
multiply-adds per world, four times AnymalD), so forming `A` costs as much as the sweeps it accelerates. The
KukaAllegro preset uses paired factor coordinates and is outside the parallel kernel's eligibility. Next candidate for
the hands: a matrix-free parallel sweep (per sweep `Y^T (y - lambda0)` then `J_i . dv`, no `A`), which needs a cheaper
step scale than the exact row sums; its convergence with a row-sum bound is the next offline study.

### 16.1 Step-scale bound and the dense-mode tasks

Offline (`study2.py`): the row-sum bound `sum_d |J_id| sum_j |Y_jd|` gives step scales 0.6 to 0.8 of the exact ones,
converges on every world (no divergence at gain 1; gain 2 diverges), and needs about 1.5x the sweeps of the exact
scale for the same error (24 bound sweeps ~ 16 exact). It makes a matrix-free parallel sweep (no response block:
per sweep `Y^T (y - lambda0)` then `J_i . dv`) viable; for the hands, where the `A` build is 66k multiply-adds per
world, the estimate is 87k against 138k per world, about 1.5x on the sweep stage. Ant and Humanoid run the dense
tiled-row solver (`pgs_solve_tiled_row_64`) and are untouched by the parallel kernel (2,068 and 8,203 us, parity with
MJWarp as before).

### 16.2 Standing at the end of the night (16k envs, GPU 1, graph mode, us per env step)

| Task | FPGS best today (config) | MJWarp | FPGS speedup |
|---|---:|---:|---:|
| Flat-AnymalD | 10,626 (grouped, 16 lanes, parallel sweep 48 rows + restart, masked rows, no clear, K1 + lazy kinematics) | 35,068 | 3.30x (was 2.24x corrected, 3.9x under the bug) |
| Reorient-Cube-Allegro | 38,161 (parallel sweep 96 rows) | 58,970 | 1.55x (was 1.48x) |
| Rough-G1 | 40,168 (grouped, row budget 0) | 51,104 | 1.27x (was 1.19x) |
| Lift-Franka | 5,734 (legacy) | 26,391 | 4.60x |
| Lift-KukaAllegro | 18,602 (legacy; paired-factor preset, parallel kernel not eligible) | 68,897 | 3.70x |
| Humanoid | 8,071 (grouped) | 8,469 | 1.05x |
| Ant | 2,050 (grouped) | 2,240 | 1.09x |
| Cartpole | 244 | 542 | 2.22x |

### 16.3 Matrix-free parallel sweep (`mf_gs_parallel_matrix_free`)

Same kernel generator, codegen flag: `J` (row-major) and `Y` (dof-major, float4-reducible) staged in shared memory,
no response block; per sweep `dv = Y^T y` (one thread per dof) then `r_i = b_i + J_i . dv` (one thread per row);
step scale `A_ii / (sum_d |J_id| sum_j |Y_jd|)`. Exact against the numpy reference on all tiers. Cold oracle
timing at 24 sweeps, per launch: Allegro 64-row tier 943 -> 657 us, 96-row tier 459 -> 288; AnymalD 32-row tier
172 -> 188 (worse: the block build was cheap there), 48-row tier 186 -> 170. The variant is a per-task choice: on
for the hands, off for AnymalD. Because the bound scale needs about 1.5x the sweeps, the in-situ comparison runs it
at 36 sweeps.

In situ on Allegro (physics trace within 0.3% of legacy at every step): matrix-free, 96-row budget, 36 sweeps
36,187 us per env step; 24 sweeps 34,130 (1.73x MJWarp, from 1.48x). Per substep at 24 sweeps: 64-row tier 735,
96-row tier 234, 32-row tier 44, and 549 us of legacy kernel for the roughly 3.8k worlds above 96 rows, which is the
next tier to add (matrix-free only: the response block of a 128-row world would exceed shared memory).

With the 128-row matrix-free tier (four warps per world) Allegro runs entirely in the parallel kernel (legacy
remainder 8.5 us): **30,430 us per env step at 24 sweeps (1.94x MJWarp)**, 32,613 at 36 sweeps (1.81x). Physics
trace within 1.5% of legacy at 24 sweeps and within 0.3% at 36. Per substep at 24 sweeps: 64-row tier 732 (12.6k
worlds, 70% of the sweep), 96-row tier 238, 128-row tier 72, 32-row tier 43. In the matrix-free sweep the `Y^T y`
phase uses only D threads of the block (22 of 64); splitting the rows into chunks across the idle threads is the
next kernel change.

Chunking the `Y^T y` phase across all block threads: 64-row tier 657 -> 640 us cold, Allegro env step 30,144 (1.96x).
Small: the tiers are bound by the five block-level synchronisations per sweep and the shared-memory dependency
chain, not by arithmetic (about 10% issue efficiency at 7 blocks per SM).

### 16.4 Allegro anatomy at 30,144 us per env step (per substep)

Sweep 1,068 (28%), rows 919 (24%: two row-build launches per substep because hand-cube contacts belong to two size
groups, 110 + 86 each; velocity-limit row kernels 107; contact bias 61; diag 78), collision 707 (19%: GJK 806,
MPR 393, manifold 136 per collision call, 4 calls per step), response solve 481 (13%: tiled 16-DOF 401 + free-body
263), dynamics 380 (the FK/ID cache is disabled when velocity limits are enabled, so Allegro pays the full
`eval_rigid_fk_id` 192 + `eval_articulation_fk` 85 + `eval_rigid_tau_add` 102 every substep), CRBA 132. The fused K1
with lazy kinematics targets the 380; the duplicated row build and the velocity-limit row kernels are the next row
items; collision is now a first-order term on this task.

### 16.5 K1 on the hands

K1 no longer requires the FK/ID cache (which velocity limits disable) and writes the free bodies' spatial inertia
itself, so the Allegro hand (16 DOF) and its cube (free body, 6 DOF) both run through it. Real-mode check on Allegro:
poses 4e-6 on 48 m, twists 5e-5 on 17, spatial inertia exact, torques 1.4e-5 on 2.2. In situ with lazy kinematics:
**29,815 us per env step (1.98x MJWarp)**, full FK/ID 192 + 102 per substep replaced by 185 + 27 (two size groups;
template staging is off because the model has two templates, one per size group, which the next change fixes), the
per-substep FK publish 85 reduced to four per step. The grouped response solve is slower than the tiled kernel on
Allegro (434 vs 401) and gets its own switch.

With the tiled response solve kept on the hands (`FEATHER_PGS_GROUPED_HINV=0`) and per-size-group template staging:
Allegro **29,661 us per env step (1.99x MJWarp)**; the hand's fused kernel still runs without staged templates
(184 us) because the topology assigns more than one template to the 16-DOF group, which is being investigated.
AnymalD regression run 10,715 (noise band of 10,626).

### 16.6 Templates and world placement

The kinematic templates keyed every joint's parent transform `X_p`, including the root joint's, which for fixed-base
robots holds the environment's world offset: Allegro produced 16,385 templates for 32,768 articulations (one per
hand, one shared by all cubes), which disabled per-block template staging in K1 and bloated the template tables.
Root-joint parent transforms are now read per articulation from `model.joint_X_p` at run time by both the template
FK kernel and K1, and the template stores identity for them; the key no longer depends on placement. Verification
(template FK check mode, K1 real-mode check on Allegro and AnymalD) and the timing rerun are in progress.

Result: Allegro now has 2 templates (hand, cube); K1 real-mode check exact on both tasks (Allegro poses 4e-6 on 48 m,
torques 1.5e-5 on 2.2; AnymalD unchanged). The hand's fused kernel runs with staged templates: 185 -> 130 us. Allegro
**29,148 us per env step (2.02x MJWarp)**; AnymalD 10,867 (noise band). Allegro per substep now: sweep ~1,000, rows
~920 (two size-group launches over the same contacts), collision ~700, response solve 666 (tiled 401 + free-body
`hinv_jt_par_row` 265, which is disproportionate for a 6-DOF body), K1 159.

### 16.7 One row-build launch pair for two size groups

The masked row build now takes two articulation sizes and two Jacobian blocks, so a hand-cube contact is built once
and written into both groups' blocks (one prelude, one Jacobian pass per substep instead of two of each). Allegro
rows 220 + 172 -> 158 + 107 us per substep; env step **28,608 us (2.06x MJWarp)**, physics trace unchanged. Note for
check mode: the legacy builder accumulates into a cleared Jacobian, so `FEATHER_PGS_CHECK_ROWS` must run without
`FEATHER_PGS_SKIP_J_CLEAR` (a first check under the skip reported spurious differences).

Rows check with a valid reference (clear enabled): both Allegro groups' Jacobians within 4e-6, metadata exact.
Per-task settings so far: AnymalD keeps the grouped response solve (98 vs 193 us tiled); Allegro keeps the tiled
one (401 vs 434). Next target on Allegro: the response solve, 401 (tiled, 16 DOF, ~55 rows) + 265 (free-body cube,
one thread per (articulation, capacity row) with 70% of threads exiting) per substep, both far from hardware limits.

### 16.8 Response solve: the predicated loops were the problem

Nsight Compute on Allegro: `grouped_hinv_jt` moved 1.35 GB through L2 per launch and was throttled on the
shared-memory pipe. Its substitution loops kept the row in registers by predicating every `k` against a runtime `i`
(N reads of the factor per step, N^3 shared loads per row) instead of unrolling both loops at compile time (N(N-1)/2
multiply-adds per pass). The loops are now fully unrolled; check against the tiled kernel and timings follow. The
free-body kernel `hinv_jt_par_row` (one thread per articulation x capacity row, 3.1M threads, 83% occupancy but
latency-bound on gathers) is the other half on Allegro.

Result: exact against the tiled kernel (Allegro 1.2e-4 on 1.9e3, AnymalD 2e-6 on 15), but the gain is small: Allegro
434 -> 421 us, AnymalD 98 -> 91. The compiler had already folded most of the predication; the kernel moves about
13 KB per articulation (factor, Jacobian rows, world-layout copies of J and Y) at an effective 0.5 TB/s, so it is
latency-bound on the dependent substitution chain rather than instruction- or bandwidth-bound. Per-task choice stands
(grouped on AnymalD, tiled on the hands). Allegro **28,113 us per env step (2.10x MJWarp)** in this run.

### 16.9 Standing at the end of this round (16k envs, GPU 1, graph mode, us per env step)

| Task | FPGS best | MJWarp | speedup | configuration |
|---|---:|---:|---:|---|
| Flat-AnymalD | 10,626 to 10,867 (run to run) | 35,068 | 3.23x to 3.30x | grouped, 16 lanes, parallel sweep 48 rows + restart, masked rows, no clear, K1, lazy kinematics |
| Reorient-Cube-Allegro | 28,113 | 58,970 | 2.10x | matrix-free parallel sweep 128 rows, masked two-group rows, no clear, K1, lazy kinematics, tiled response solve |
| Rough-G1 | 40,168 | 51,104 | 1.27x | grouped, row budget 0 |
| Lift-Franka | 5,734 | 26,391 | 4.60x | legacy |
| Lift-KukaAllegro | 18,602 | 68,897 | 3.70x | legacy (paired-factor preset) |
| Humanoid | 8,071 | 8,469 | 1.05x | grouped (dense tiled-row solver) |
| Ant | 2,050 | 2,240 | 1.09x | grouped (dense tiled-row solver) |
| Cartpole | 244 | 542 | 2.22x | legacy |

Remaining first-order items: collision (shared pipeline, 19% on Allegro, 12% on AnymalD), the free-body response
solve on Allegro (265 us per substep for a 6-DOF body), the row diagonal (48 us on AnymalD), and for the legged
tasks the wide sweep tier (contact reduction is the lever there). The Newton polish stage stays planned behind these.

### 16.10 Free-body response solve

The cube's `hinv_jt_par_row` (one thread per articulation x capacity row, 3.1M threads, 70% exiting, 265 us per
substep) is replaced by routing small size groups to the grouped warp-per-articulation kernel
(`FEATHER_PGS_GROUPED_HINV_MAX`, default 32; on Allegro 8 keeps the 16-DOF hand on the tiled kernel and sends the
6-DOF cube to the grouped one).

## 17. Response solve folded into the sweep kernel, in whitened coordinates (2026-09-10)

### 17.1 What the response stage was costing
Allegro at 29.3 ms per env step spent 6.8 ms in stage 4 response: `hinv_jt` for the hand (420 us per substep) and
the cube (280-350), then three diagonal kernels (78). Both `hinv_jt` kernels are bandwidth-bound on the world-layout
copies they write (J and Y, 22 floats per row each) so that the sweep kernel can read them back one launch later.
The sweep kernel already stages every row of the world in shared memory; the round trip through HBM is pure shape debt.

### 17.2 The change (`FEATHER_PGS_INK=1`, `_get_pgs_solve_parallel_kernel(inkernel_response=(NA, NB, OA, OB))`)
The parallel sweep kernel gathers the group-layout Jacobian rows of the world's (at most two) articulations, stages
their Cholesky factors in shared memory (16x16 + 6x6 floats on Allegro) and forms the response itself. Three
iterations were needed to make it pay:

1. `Y_i = H^-1 J_i^T` per lane with `j[]`, `y[]` in registers and fully unrolled loops: the compiler hoisted every
   L load, 68 -> 218 registers, occupancy 17%, the sweep tier got slower than the stage it replaced (966 vs 719 us).
2. Solve in shared memory (`Y` as a d-major column, outer loops not unrolled, inner loops unrolled with predicates):
   72 registers, but Nsight Compute showed shared memory as the block limiter (16.5 KB -> 5 blocks per SM, 21%).
   Moving this lane's J row to registers and compiling the drive tables out when there are no PhysX drive rows took
   shared memory to ~9.5 KB: 38% occupancy, tier 738 us, Allegro 22.8 ms.
3. Whitened coordinates. `A = J H^-1 J^T = Z^T Z` with `Z = L^-1 J^T`, so the sweep needs neither J nor Y: one
   forward substitution per row (half the work), `r_i = b_i + Z_i . (Z^T y)`, the step scale bound
   `sum_j |A_ij| <= sum_d |Z_id| sum_j |Z_jd|`, and at the end one backward substitution `dv = L^-T (Z^T (x - lam0))`
   on a single D-vector per articulation. `J_i . v_in` is folded into `b_i` while J is at hand. The A-based variant
   (AnymalD) solves in the not-yet-live response block storage and publishes Z rows to the world buffer, because
   solving straight in global memory made every step of the chain a store-then-load round trip (rows32 tier
   161 -> 246 us).

Worlds the parallel kernel owns (dense rows within budget, no matrix-free rows) are skipped by the grouped `hinv_jt`
kernel and the three diagonal kernels (`skip_rows_le`, `mf_constraint_count == 0`), so the legacy kernel keeps its
inputs for the worlds it still solves. The grouped response kernel now also writes the group-layout Y for sizes the
plan routed to the per-row kernel (that omission was the "free-body NaN": the diagonal is formed from group J.Y).

### 17.3 Verification
`FEATHER_PGS_INK_CHECK=1` makes the kernel write its Y (matrix-free: `L^-T Z_i`) and diagonal into the world arrays
after the response stage ran; the launch site compares against snapshots. Allegro: Y within 6.1e-5 of 2.4e3 (one
float ulp), diagonal 1.5e-3 of 7.6e3; AnymalD diagonal 2.4e-6 of 6.3 (A-based, diagonal only since Z occupies the
buffer). Same-configuration repeat runs differ by 33-35 contacts at step 0 (collision atomics), the in-kernel path
differs from the staged path by 90-113 at steps 0-1: same order, chaotic amplification of ulp differences.

### 17.4 Results (16k envs, GPU 1, graph mode, us per env step)

| Task | before | after | MJWarp | speedup |
|---|---|---|---|---|
| Reorient-Cube-Allegro (matrix-free rows=128) | 28,113 | 22,458 | 58,970 | 2.63x |
| Velocity-Flat-AnymalD (A-based rows=48) | 10,626 | 10,357 | 35,068 | 3.39x |

Allegro stage totals now: sweep 8.3 ms, collision 5.6, rows 4.25, dynamics 1.5, CRBA 0.76, response 0.43. The
matrix-free variant on AnymalD is faster still (10,192) but its contact count is 1.7% below every A-based run,
i.e. less converged at 24 sweeps; not taken.

### 17.5 What this says about the shape
The sweep kernel is now one launch per tier that takes J rows and Cholesky factors in and puts velocities out. The
response solve, the diagonal, the J.v bias and the velocity reconstruction are all inside it. Occupancy, not
arithmetic, sets its speed (Nsight: 38% theoretical, stalls on shared-memory latency), so the next levers are
shared-memory footprint (Z rows are sparse on fixed-base hands: one finger plus the cube, 10 of 22 DOFs) and the
per-world row kernels that still run one thread per world.

### 17.6 Per-row contact bias and the collision accounting (2026-09-10)
`compute_world_contact_bias` ran one thread per world with a serial loop over the rows (62 us per launch on
Allegro with 64 blocks); one thread per (world, row) takes it to 14 us. Allegro 22,105 us per env step (2.67x),
AnymalD 10,087 (3.48x). A 4x wider slot-allocation grid changed nothing (atomics per world), left at 1x.

Collision, measured properly. The nsys stage attribution had put MJWarp's convex narrow phase (`ccd_kernel`, GJK/EPA,
two launches per collide) under "other": on Reorient-Cube-Allegro MJWarp spends 13.7 ms per env step there plus
0.5 ms broad phase and primitives, i.e. ~14 ms of its 59 ms is collision. Newton's pipeline for the same scene
(1 plane, 16,384 boxes, 294,912 convex-mesh hand links; 340k candidate pairs, 322k to GJK, 76-130k manifolds,
135-211k contacts) costs 5.6 ms per env step. So on the hands FPGS already carries a cheaper collision stage than
MJWarp and the "unchanged collision" comparison is the one in the tables above; the remaining collision lever on our
side is the 5.6 ms (GJK 0.8 ms and MPR 0.4 ms per pass, 4 passes), which is a Newton narrow-phase matter (occupancy:
128 registers, 4 blocks per SM heuristic), not a solver one.

### 17.7 Velocity-limit rows and narrow-phase threads (2026-09-10)
`populate_joint_velocity_limit_J_for_size` now runs one thread per (articulation, DOF, side) and reads eligibility
from the slot array (27 -> 9.4 us per launch). The slot allocator reserves one range per articulation with a single
atomic instead of two per DOF (order within an articulation unchanged; the velocity-limit rows are order-free in the
parallel kernel and sit in their own phase in the legacy one); it did not get faster (42 us), so its cost is the
per-articulation serial walk with 128 blocks, like `prescale_joint_velocity_limits` (41 us). Both remain
(0.67 ms per env step on Allegro). Scaling Newton's grid-stride narrow-phase thread count 4x
(`NEWTON_NARROW_PHASE_THREADS_X=4`, opt-in in NewtonManager) takes collision from 5.69 to 5.37 ms per env step.

Standing (16k envs, GPU 1, graph mode, us per env step): Allegro 21,642 (2.73x MJWarp 58,970); AnymalD 10,087
(3.48x of 35,068). Allegro stage totals: sweep 8.47, collision 5.36, rows 3.67, dynamics 1.52, CRBA 0.76,
response 0.43, integrate 0.31, FK 0.33.

### 17.8 The step-scale bound was 43% of the sweep kernel (2026-09-10)
Sweep-count timing of the 64-row tier (16/24/32 sweeps: 568/704/836 us) gave 17 us per sweep and ~300 us fixed.
The fixed part was the row-sum bound: every lane recomputed all D column sums `sum_j |Y_jd|` (22 x 64 shared loads
per row, as much as 20 sweeps). The column sums are now formed once per block in the same pass that forms
`Y^T lam0`. 64-row tier 704 -> 549 us, 96-row tier 227 -> 159. Allegro 19,660 us per env step (3.00x MJWarp);
sweep stage 8.47 -> 6.49 ms, now 33% of the step. Physics statistics unchanged (contacts 230,062).

Measurement hygiene: another session started training jobs on GPU 1 during this round (6-7 minute runs, repeated);
two runs were contaminated (whole-graph 32.0 and 22.3 ms with every stage inflated). `clean_run.sh` now waits for an
idle GPU 1, watches for foreign compute processes during the run and repeats if one appears.

### 17.9 Bank conflicts in the d-major response block (2026-09-10)
Nsight Compute on the 64-row tier after 17.8: 149M shared-memory wavefronts, stalls short_scoreboard 4.8 and
mio_throttle 2.5 per issue. The d-major block `s_Yt[d * AM + j]` with `AM = 64` puts every lane's column in the same
bank: in the `Y^T y` reduction lane d reads `s_Yt[d*64 + j]`, all D lanes hit bank `j mod 32`, a 22-way conflict on
the kernel's hottest load. Padding the stride to `AM + 4` (one float4) spreads the lanes over the banks.
64-row tier 549 -> 360 us, 96-row 159 -> 120. Allegro 17,780 us per env step (3.32x MJWarp); sweep stage 4.59 ms
(26% of the step; collision is now the largest stage at 5.36). Y and diagonal check unchanged (1 ulp). AnymalD
(A-based, row-major block with stride AM+1, already conflict-free) unchanged at 10,128.

### 17.10 Per-tier world lists (2026-09-10)
Each sweep tier launched all 16k worlds and let the blocks of the other tiers exit; a 16k-block launch that does
nothing still costs ~30-55 us (Nsight caught 48-row-tier launches at 55 us with 4.8M instructions). A 9 us classify
kernel now appends each dense-only world to its tier's list (`FEATHER_PGS_TIER_BLOCKS`), and each tier launches
against its list. Grid-striding with fewer blocks (2048/4096) serializes worlds per block and slows the big tier
(360 -> 456/376 us); launching the full world count against the list keeps one world per block: Allegro rows
32/96/128 tiers 37.6/120/40 -> 30/106/31 us, 64-row tier unchanged. Allegro 17,543 us per env step (3.36x MJWarp).
AnymalD gains nothing (10,349 vs 10,128, within noise plus the classify kernel); its two tiers stay at 300/191 us,
so the exiting blocks were not its fixed cost. Per-task: Allegro on, AnymalD off.

### 17.11 Bounding-sphere reject before GJK (2026-09-10, Newton narrow phase)
`prepare_convex_pair` had a bounding-sphere overlap test only for plane pairs; every other AABB-overlapping pair
went to GJK. A conservative reject (`|p_a - p_b| > r_a + r_b + gap + margins`, radii from `shape_collision_radius`)
now precedes it. Allegro: GJK items 321.7k -> 304.3k per pass (5% pruned; AABB overlap is already tight for the
fingers), manifolds identical within run noise (76,366 vs 76,361), contacts identical within noise. GJK kernel
745 -> 617 us per pass, collision 5.36 -> 4.78 ms per env step. Allegro 16,999 us per env step (3.47x MJWarp 58,970).

### 17.12 Jacobian pass: one thread per used DOF (2026-09-10)
`populate_world_J_masked` used 32 lanes per contact; 10 of them idle on 22-DOF worlds and 14 on 18-DOF worlds.
Lanes per contact are now the larger group size. Allegro 158 -> 108 us per launch, AnymalD 110 -> 82.
Allegro 16,582 us per env step (3.56x MJWarp), AnymalD 9,887 (3.55x). Bit-identical J (same arithmetic, other mapping).
Rows stage is now prelude 106 + J 108 + slots 33 + bias 14 + velocity limits ~85 us per substep on Allegro; MJWarp's
dense contact Jacobian for the same scene is 48 us per substep, so the row build is still ~7x heavier than the
reference: the prelude writes 8 metadata arrays x 3 rows per contact.

### 17.13 What the row build is bound by (2026-09-10)
Nsight on Allegro: `contact_row_prelude` (one thread per contact, 262k threads) issues 5% of the time with
long-scoreboard stalls of 68 per issue; `allocate_world_contact_slots` (65k threads) the same. All their threads are
resident at once, so the launch time is one contact's dependency chain: contact -> slot/articulation -> shapes ->
bodies -> transforms -> materials -> friction-anchor look-back (up to 8 dependent loads) -> three prescribed-target
evaluations -> 24 scattered 4-byte stores. Neither 4x nor 16x more threads changes it (106 us either way; the J pass
gets slower with idle threads). MJWarp forms the same contact Jacobians in 48 us per substep.

The shape that fixes this is one kernel, one thread per contact row (3 per contact): slot and geometry once, the row's
metadata as one record, the row's D Jacobian entries in a contiguous loop; no flags array, no second pass, no 22x
geometry recompute. Expected ~120-150 us per substep for the whole build instead of prelude 106 + J 108 + slots 33 +
bias 14. Not started; recorded as the next rows item.

Per-substep Allegro budget at the end of this round (us): sweep tiers 352+106+31+30, collision ~1200 (GJK 617,
MPR 360, manifold 140), rows 106+108+33+14+85, K1 130, CRBA 95, response leftovers 54, integrate 39, FK 42.

### 17.14 Correction: the skipped Jacobian clear was wrong (2026-09-10, round close)
The round-close re-measurement ran every task with the flags of the two best tasks, and Ant came out non-finite.
Bisecting on Ant (eager, 12 steps): masked row build with the clear = legacy to the last contact; masked build
without the clear diverges (z_min 0.21 vs 0.36 by step 11). Allegro is affected too (step-0 contacts 134,614 with
the clear vs 134,958 without, qd_max 13.9 vs 11.2); AnymalD is not (identical within noise). Two reasons:
joint-limit / velocity-limit / drive rows write a single Jacobian entry and rely on the cleared row, and the masked
contact builder writes only the involved articulation's block, so in a two-articulation world (hand + cube) a
cube-ground contact leaves the hand block of that row stale. `FEATHER_PGS_SKIP_J_CLEAR` is therefore retired from
every configuration; all 2026-09-10 Allegro numbers before this section carried that defect (kernel-level timings
stand; whole-graph numbers are re-measured below with the clear on). The clear itself (`clear_grouped_jacobian_active_rows`,
one warp per articulation over the active rows) moves ~250 MB per substep on Allegro; the shape that removes it is
"every row producer writes the complete row across all groups", which touches five producers and is deferred.

### 17.15 Standing at the round close (2026-09-10; 16k envs, GPU 1, graph mode, us per env step, clear ON)
Artifacts: `fpgs-results-20260906/profile_artifacts_20260910/` (`remeasure.sh`, `make_table.py`; MJWarp re-run the
same day; runs repeated automatically when a foreign job touched GPU 1).

| Task | FPGS best | FPGS 09-09 legacy | MJWarp (same day) | FPGS speedup |
|---|---:|---:|---:|---:|
| Velocity-Flat-AnymalD (grouped, A-based rows=48, INK, lazy, masked rows) | 10,255 | 15,678 | 35,094 | 3.42x |
| Reorient-Cube-Allegro (grouped, matrix-free rows=128, INK, tier lists, lazy, masked rows) | 17,605 (re-run; 19,375 outlier) | 39,713 | 59,806 | 3.40x |
| Lift-KukaAllegro (legacy) | 17,634 | 18,602 | 72,902 | 4.13x |
| Lift-Franka (legacy) | 5,854 | 5,734 | 29,984 | 5.12x |
| Velocity-Rough-G1 (grouped) | 38,931 | 42,769 | 50,126 | 1.29x |
| Cartpole (legacy) | 248 | 244 | 541 | 2.18x |
| Ant (legacy, dense) | 2,072 | 2,069 | 2,216 | 1.07x |
| Humanoid (legacy, dense) | 7,920 | 8,230 | 8,511 | 1.07x |

All states finite; contact counts match the legacy path (Ant 52,500 vs 52,338; AnymalD 161,578). The Allegro number
with the correct Jacobian is 19,375, not the 16.6-17.5k measured with the skipped clear: with stale rows the
velocity-limit rows were dead, so the correct physics has more active rows per world (the 96-row tier takes 164 us
instead of 105) and more contacts survive. Common flags for every FPGS run: `FEATHER_PGS_GROUP_LANES=16`,
`FEATHER_PGS_ROWS_MASKED=1`, `NEWTON_NARROW_PHASE_THREADS_X=4`; the bounding-sphere reject is on for all.

### 17.16 Row build: fused per-row kernel withdrawn, prelude split into three threads (2026-09-10)
The one-thread-per-row fused builder (`row_build_rows.py`, metadata identical, J within 4e-6) is slower than the
two passes (Allegro 257 us vs 106 + 108; AnymalD 159 vs 51 + 82): the per-row thread carries the prelude chain plus
a 22-iteration loop of dependent `joint_S_s` loads, whereas the two-pass J kernel spreads those loads over 22 threads.
Kept opt-in (`FEATHER_PGS_ROWS_FUSED=1`) as the record of the measurement. What did help is splitting the prelude
itself into three threads per contact, one per row (flags and normal row on thread 0, one friction row each on
threads 1 and 2): one prescribed-target evaluation and eight stores per chain instead of three and twenty-four.
Prelude 106 -> 85 us (Allegro), 51 -> 45 (AnymalD). Allegro 17,624 / 17,661 us per env step on two clean runs
(3.39x MJWarp 59,806); AnymalD 10,020 / 10,149 (3.47x). The 19,375 Allegro entry of 17.15 is therefore an outlier
of that run (its FK stage also read 606 us against 335 elsewhere); the table entry is re-run below.

Table 17.15 update: Reorient-Cube-Allegro re-run (clean, prelude split included) = 17,605 us per env step, 3.40x
MJWarp 59,806; contacts 215,350. All other rows of 17.15 stand.

### 17.17 Matrix-free sweep with the exact step scale (2026-09-10)
The A-based variant existed because the matrix-free row-sum *bound* needs ~1.5x the sweeps (AnymalD contacts
158k vs 161k at 24 sweeps). The bound is not what makes the matrix-free kernel cheap; the missing response block is.
`exact_row_sums` (`FEATHER_PGS_MF_EXACT_ROWSUM=1`) forms `sum_j |Z_i . Z_j|` on the fly once per row (n x D shared
broadcast loads per lane, the same arithmetic the A build did) and never stores A: shared memory drops from 12.7 KB to
~7 KB, the per-sweep cost is the matrix-free one (2 n D instead of n^2), and the iteration is the A-based one to
rounding. AnymalD: 48-row tier 300 -> 234 us, 32-row tier 191 -> 157; 9,159 us per env step (3.83x MJWarp 35,094),
contacts 161,527 (A-based semantics restored). One kernel variant now serves both tasks; the A-based response block
is no longer the recommended path.
Allegro with exact row sums: 64-row tier 352 -> 442 us, 96-row 106 -> 265 (n x D = 64 x 22 broadcast loads per lane
is the cost the column-sum fix removed); 19,790 us per env step vs 17,605 with the bound. Per task: AnymalD exact,
Allegro bound.

## 18. The world kernel: one block per world per substep (2026-09-10, night)

Constraint from the user: physics parameters are fixed (substeps, collision cadence, contact set, sweeps, tolerances).
Ceilings under that constraint: Allegro's collision alone is 5.3 of the 6.0 ms a 10x needs, so Allegro tops out near
6x with a perfect solver; AnymalD's collision is 0.95 of 9.2 ms, so 8-9x is reachable there if the ~45 solver launches
per substep collapse into one block per world with no HBM round trips between stages. That is the build.

Plan (each step measured, physics compared with the check machinery):
1. Per-world contact lists (`build_world_contact_lists`), so a block can find its contacts.
2. Contact rows built inside the sweep kernel (`world_rows`): geometry, friction decision, metadata, bias and
   restitution, and the J row into shared memory; whitened response and sweep as before. Existing producers keep
   running for the worlds the kernel does not own; the kernel ignores their contact-row J.
3. Remove the now-redundant producers for owned worlds (J pass, bias, restitution), then the metadata prelude once the
   warm-start gather and force reporting read what the kernel writes.
4. Fold the fused dynamics (K1 + CRBA + Cholesky) in front and integration behind.
5. Two-articulation worlds (Allegro).

### 18.1 Steps 1-3 on AnymalD (2026-09-10, night)
`FEATHER_PGS_WORLD_ROWS=1`: per-world contact lists (6 us), contact rows built inside the sweep kernel from the list
(geometry, friction decision, metadata, bias, restitution, J row into the lane's d-major column, then the in-place
whitened solve), producers skipped for owned worlds (prelude 45 -> 5 us, J pass 82 -> 38, bias 12; the kernel writes
row type/parent/mu for the force-reporting kernels). Check mode: J bit-identical, bias within 2e-6 relative; graph
trace identical to the two-pass path at every step. Sweep tiers 234/157 -> 279/177 us (the in-kernel build costs
~65 us per substep: one block-latency chain of ~10 dependent loads per row lane), rows stage 1,644 -> 946 us per env
step. AnymalD 9,065 us per env step (3.87x MJWarp 35,094). Per substep now: sweep 455, rows 118, K1 99, collision
119, CRBA 70, integrate 40, FK 34.

### 18.2 Phase anatomy of the world kernel and two dead ends (2026-09-10, night)
`FEATHER_PGS_WR_STOP=n` returns after phase n. AnymalD, us per launch (48-row tier / 32-row tier): staging 8/8,
contact rows + whitened solve 92/31, exact row sums 58/10, 24 sweeps + velocities 114/129. The 32-row tier is one
warp per world; its sweep costs 5.4 us each: about eight dependent phases (reduction, residual, clamp, store, friction
projection, restart dot, store, early-exit vote), ~700 cycles of latency per sweep with 27% achieved occupancy.
Dead ends: (a) a two-phase row build (geometry once per contact into shared memory, then rows) is slower
(290/210 vs 279/177) because the two chains run back to back: latency, not redundant arithmetic, is the cost;
(b) the exact row sums cost as much as 12 sweeps on the 48-row tier (n^2 D), which is exactly what the row-sum
bound saves and then pays back in extra sweeps. Standing: AnymalD 8,874 us per env step (3.95x MJWarp 35,094)
with world rows on; the J pass now launches one thread per used DOF (38 -> 24 us).

## 19. Toward the exact per-world solve (2026-09-11)

### 19.1 Ground truth first: what today's kernels converge to
Six consecutive substeps of AnymalD and Allegro were captured (`FEATHER_PGS_CAPTURE_COUNT`, with contact identity
per row) and the real kernels were re-run offline on the captured state (`formulation/truth.py`), 16,384 AnymalD
worlds, relative velocity difference per world (median / p90 / max):

| comparison | median | p90 | max |
|---|---:|---:|---:|
| legacy row GS, 24 vs 400 iterations | 1.2e-5 | 5.0e-4 | 0.11 |
| parallel projection, 24 vs 400 sweeps | 4.1e-3 | 1.6e-2 | 0.42 |
| parallel 400 vs legacy 400 (the two fixed points) | 1.8e-2 | 5.5e-2 | 0.32 |
| parallel 24 vs legacy 24 (what today's tasks see) | 2.1e-2 | 5.7e-2 | 0.28 |

Three consequences. (1) The legacy GS is converged at 24; the parallel kernel is not (0.4% typical, tens of percent
in the worst worlds). (2) The two methods do not converge to the same answer: each projects the friction pair with
its own per-row step metric (1/A_ii for GS, w_i/A_ii for the parallel kernel), and the fixed point of a disk
projection under an anisotropic metric has the friction force not antiparallel to the slip velocity. Neither is the
maximal-dissipation Coulomb solution; they differ from each other by ~2% median. (3) The contact statistics that
matched between legacy and parallel all day are therefore insensitive at the 2% velocity level; "physics identical"
was true at the statistics level, not at the per-substep velocity level. An exact per-world solve (staggered Newton,
isotropic friction disk) would be the maximal-dissipation answer and differ from both by the same few-percent class.
A numpy row-GS model written for this study projected friction once per sweep and disagreed with the kernel by 40%:
offline references are taken from the kernels themselves from here on.

### 19.2 The exact per-world solve is not the lever; warm start is (2026-09-11)
Staggered Newton (fixed friction radii per outer iteration, convex QP by Newton with exact line search) on the
captured AnymalD substeps: 5-6 outer, ~26 inner iterations, cold or warm (warm starting the radii and velocity does
not reduce the count), about 450-520 sweep-equivalents of arithmetic per world against 24 today. As formulated it is
20x too expensive; the exact per-world solve is withdrawn as step 1 of the plan.

What the ground truth suggests instead: the parallel kernel is 0.4% (median) from its own fixed point at 24 sweeps
on AnymalD, and contacts persist across substeps (100% matched by shape pair and point). Warm-starting the kernel
from the previous substep's impulses (numpy model of the kernel, validated to 2e-7 against it; reference = kernel at
400 sweeps; 192 worlds x 5 substep pairs):

| sweeps | cold error (median / p90) | warm error (median / p90) |
|---|---:|---:|
| 4 | 0.26 / 0.48 | 2.4e-2 / 0.10 |
| 8 | 0.13 / 0.27 | 1.0e-2 / 5.2e-2 |
| 12 | 5.6e-2 / 0.14 | 4.4e-3 / 2.4e-2 |
| 16 | 1.7e-2 / 5.6e-2 | 1.4e-3 / 1.1e-2 |
| 24 (today, cold) | 4.2e-3 / 1.9e-2 | 3.2e-4 / 2.6e-3 |

Warm-started 12 sweeps match today's cold 24 (4.4e-3 vs 4.2e-3 median); warm 24 is 13x closer to the fixed point.
MJWarp and PhysX both warm start their contact solvers; FeatherPGS has the option (`pgs_warmstart`, needs contact
matching in the collision pipeline) but the tasks run it cold. Halving the sweeps at equal accuracy is worth
~120 us per substep on AnymalD (~1 ms per step, 11%). The physics parameters are unchanged; the iterate sequence is.
Allegro (same study, velocity-limit rows admitted; model vs kernel 5e-8): cold 24 = 4.9e-4 / 1.7e-2 (median / p90);
warm 16 = 8.9e-4 / 2.8e-2; warm 24 = 2.7e-4 / 1.4e-2. About 1.5x fewer sweeps for equal accuracy on the hand, 2x on
the quadruped. Both numbers understate a real warm start, which compounds across substeps (the study warms from a
cold 24-sweep result).

### 19.3 Warm start in situ (2026-09-11)
FeatherPGS's existing warm start (`pgs_warmstart=True`, Newton `contact_matching="latest"`) costs a radix sort of the
contacts every substep (5.6 ms per env step) plus matching, apply and gather kernels: AnymalD 17.3 ms per step. Not
usable. A native version inside the world kernel (`FEATHER_PGS_WR_WARM=1`: per-world table of shape pair, local
point and three impulses per contact, written by the block at the end of the substep, matched at the start with a
same-position fast path, staged through shared memory): no sort, no extra launch, statistics equal to cold within
noise at 24 sweeps. But its bookkeeping costs ~35 us per tier launch, about the 8 sweeps it saves:

| AnymalD, us per env step | cold 24 | warm 24 | warm 16 | warm 12 |
|---|---:|---:|---:|---:|
| whole graph | 8,990 | 9,590 | 9,118 | 8,936 |
| contacts (statistics) | 161k | 162k | 166k | 177k |

Fewer sweeps also move the statistics (more converged answer, 3% more contacts at 16). Together with 19.2 this closes
the "fewer iterations" direction for this kernel: neither an exact solve nor a warm start pays on the quadruped, and
the per-substep fixed work of a block, not the sweep count, is what remains. Small-world packing (two worlds per
block) does not help either: the 32-row tier is register-limited (86 registers -> 23 warps per SM), not block-limited.
Remaining measured items: fold the dynamics launches into the world kernel (~1.5-2 ms per step on both tasks), and the
finger-block structure of the whitened rows on hands (~1 ms on Allegro).

### 19.4 Structure inside the hand: premise verified (2026-09-11)
Allegro capture: the hand's 16x16 mass matrix has zero mass outside the four 4x4 finger blocks (fixed palm), contact
Jacobian rows have a median of 2 nonzeros (p90 8) of 22, and the whitened rows Z = L^-1 J^T have 4 hand nonzeros
(max 8 for finger-finger contacts) of 16 plus the cube's 6. So a row touches at most 14 of 22 DOFs and typically 10.
What that buys in the kernel: the d-major reduction t = Z^T y (the sweep's largest shared-memory item) restricted per
finger to that finger's rows (rows permuted by finger inside the block), and the forward substitution restricted to
the finger's 4x4 block: an estimated 25% of the Allegro sweep phase (~1 ms per env step, 6%), Allegro only. Also the
dynamics fold's gain must be revised down: the CRBA + Cholesky warp kernel is a serial Cholesky on lane 0 (44 us per
launch) and the triangular solve is the same chain; folding them into the world block removes launch tails and the
factor round trip, not the chains. Expected ~0.3-0.5 ms per step, not 1.5-2.

### 19.5 Collision: what Newton's GJK kernel is bound by (2026-09-11)
Nsight on the Allegro narrow phase (322k convex-hull pairs per pass): GJK kernel 107 registers -> 4 blocks per SM
(24% occupancy achieved), 1.65 GB of local-memory loads per launch (spilled simplex/support arrays), 10.8 of 32 lanes
active per issued instruction (branch divergence); MPR kernel 122 registers, 7.9 of 32 lanes. Issue slots 32-38%
busy. The headroom is 2-3x and it lives in register pressure and divergence inside Newton's GJK/MPR (simplex and
support arrays indexed dynamically spill to local memory), not in the pairing or cadence. A per-world persistent
formulation would not change this; a register-resident GJK would. That is a Newton geometry task, recorded here as
the collision lever with its measured cause.

### 19.6 Hand structure: bound measured, not worth building (2026-09-11)
Shadow twin of each Allegro tier with the hand DOFs' reduction restricted to a quarter of the rows (the best the
finger ordering could achieve), timed on the true state: 64-row tier 305 -> 296 us, 96-row 161 -> 155, 128-row
31 -> 30, 32-row 29 -> 27. Three percent of the sweep kernel, ~0.1 ms per env step. The reduction is not where the
Allegro sweep time goes; the per-row work and its bookkeeping are (lean twin: 305 -> 220). Item withdrawn.

### 19.7 Where the four items landed
| item | premise | measured outcome |
|---|---|---|
| 1. exact per-world solve (staggered Newton) | contacts persist 100% | 26 inner iterations warm or cold, ~500 sweep-equivalents: 20x too expensive. Warm-starting the existing kernel instead: break-even (bookkeeping = 8 sweeps), statistics move with convergence. |
| 2. finger-block structure on hands | exact block-diagonal factor, rows touch 4 of 16 hand DOFs | 3% of the sweep kernel; not built. |
| 3. collision as a per-world pass | GJK is the largest Allegro stage | bottleneck is register spills (1.65 GB local traffic per launch) and divergence (1/3 lanes) inside Newton's GJK; a per-world pass does not change that, a register-resident GJK would. |
| 4. small-world packing | 60% of AnymalD worlds under 32 rows | tier is register-limited (86 registers, 23 warps/SM), not block-limited; no gain. |
Standing: AnymalD 8,990 us per env step (3.90x MJWarp) with the world-rows kernel; Allegro 17,605 (3.40x). What
remains measured-positive: the dynamics fold (~0.3-0.5 ms per step per task) and Newton's GJK (2-3x on 5.3 ms of
Allegro's 17.6, i.e. Allegro to ~14 ms / 4.3x).
