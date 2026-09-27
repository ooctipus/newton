# Many-body, multi-block PGS candidate

Status: opt-in experiment. Direct static-pair handling fixes a diagnosed support
convergence regression while retaining substantial measured solve gains. Hardware
still has a dynamic-contact convergence regression; do not promote universally.
The latest dense-tile coin runs also have unresolved full-cycle nonfinite failures.
Multi-block/tile/cache dispatch remains opt-in; scalar dense-impulse ownership
hardening is shared with the existing scalar-disk projector.

## Hypothesis and scope

Keep ordinary PGS, the current contact law, timestep, substeps and sweep budget.
Keep Isaac Lab and collision unchanged. Replace the free-rigid contact sweep's
single warp per world with conflict-free body-pair batches across multiple GPU
blocks. Retain the existing dense/articulated response and velocity-limit order.

The dishes N=1 profile attributes 97.3% of summed GPU kernel time to the serial
MF solve. Recent current-code dishes captures take about 297 ms per environment
frame on RTX and 287 ms on GB300. A 4x reduction of the dominant solve would
suggest roughly 200 ms removable per frame, before preparation, launches, host
overhead and changed convergence. This is a hypothesis, not a task-level claim.
Hardware and coins provide larger contact graphs; existing multi-task captures
remain the baseline for later non-regression qualification.

## Work replaced and added

- Replace whole-world serial free-body row traversal and warp reductions with
  body-local contact updates; keep each normal and its friction siblings owned
  together. Group repeated contacts on the same unordered body pair to avoid
  assigning a separate color to every contact point.
- Add GPU pair construction/sorting, conflict coloring and color barriers.
- Preserve every row: excess colors use an explicit serial tail, not truncation.
- Keep dense robot constraints before free contacts and velocity limits after
  contacts in each sweep. Different articulated links are not independent bodies.
- First implementation is opt-in and rejects unsupported schedules/features;
  it does not silently change established specialized paths.

## Decision gate

First checkpoint within 90 minutes: unit correctness plus smallest integrated
dishes measurement on RTX, then paired GB300. Count preparation and fallback
costs. Require a credible >=2x whole-physics route, not a small component gain.
If timing contradicts the hypothesis, inspect color occupancy, tail size and
launch/preparation cost, then make one diagnosis-driven corrective experiment.

Validation: conflict ownership, no lost/duplicated rows, reset/empty topology,
CUDA graph replay, impulse/velocity consistency, momentum and Coulomb feasibility;
then fixed-budget residuals and physical behavior on loaded contacts. Changed GS
ordering need not be bit-identical. Dishes first, then hardware/coins at N=1/2/4;
no broad benchmark until bounded admission succeeds. No TGS or added sweeps.

## First mapping: diagnosed loss, not a promoted optimization

RTX N=1 dishes, frames 4-12: unchanged serial physics 291.03 ms; initial
multi-block mapping 1,444.62 ms (4.96x longer). The admission guard stopped the
candidate after frame 12. Finite/capacity checks passed; this is not convergence
qualification. CPU tests (3) and GB300 synthetic CUDA tests (5), including graph
rebuild and dense/contact/velocity-limit ordering, pass.

At the last sample, 1,422 contact units form 59 body pairs: 32 colors, four tail
pairs. Color occupancies start 12/9/4/2, then mostly one pair. The dynamic rack
is a shared-body hub, so pair coloring alone does not expose broad parallelism.
Generated PTX also confirms repeated global DOF-map/velocity loads and velocity
writes per row component; the pair owner does not yet retain body velocities
in registers. Each color additionally launches capacity-sized work (8,064
threads/world), even though disjoint pairs cannot outnumber six-DOF bodies.

Next bounded corrective mapping: retain the two owned body velocities locally
through a pair, and bound the launch by the number of bodies rather than row
capacity. Profile and frozen-input comparisons must distinguish execution cost
from changed constraint trajectories before accepting the result.

## Mapping diagnosis and structural next step

Same short integrated dishes window, frames 4-12, milliseconds of physics per
environment frame (10 unchanged physics calls, 24 unchanged sweeps each):

| Route | RTX PRO6000 | GB300 |
| --- | ---: | ---: |
| Original | 291.03 | 296.64 |
| Scalar pair owner | 1,444.62 | 1,337.22 |
| Register-local pair owner, body-bounded launch | 666.26 | 641.53 |
| Warp-per-pair owner | 469.75 | not measured |

All are early admission windows, not full-cycle qualified timings. No candidate
above beats the original. Native CUDA compilation failures are not timings.
Nine CUDA checks now cover color conflict ownership, graph reset/rebuild,
friction feasibility, phase ordering, reversed/static/self endpoints, more than
64 bodies, 4,098 contact units and unsigned parent indices above 32,767.

The register-local version's RTX node-level trace places 96.4% of summed kernel
time in contact updates; grouping/coloring is 0.5%. Launch preparation is not
the dominant loss. An identical 2,676-row frozen solve of the first mapping takes
138.28 ms versus 27.61 ms serial, including preparation and captured dispatch.
Impulse/velocity consistency passes, but cone feasibility is not a convergence
certificate. The frozen comparison separates execution cost from trajectories.

### Pair-block mass splitting, not TGS

The dynamic rack is a high-degree graph hub: strict conflict-free edge coloring
must serialize its incident pairs. Further color-kernel tuning cannot remove
that dependency. The next opt-in experiment runs all pair blocks concurrently
from the same velocity, with a mathematically defined consensus update:

- For body b incident on n_b pair blocks, use n_b M_b^-1 locally, then sum each
  block's local velocity change divided by n_b. The result is exactly the physical
  M_b^-1 sum(J_b^T delta-lambda); real masses and stored impulses are unchanged.
- Preserve normal regularization g = D_old (1-w)/w, instead of scaling it with
  the artificial local diagonal. Same-body endpoints combine Jacobians and
  responses including cross terms. Dense articulated rows retain their existing
  serial phase before the free-body blocks.
- Frictionless splitting has a block-majorizer argument. That argument does not
  automatically prove the existing scalar Coulomb projection's convergence.
  Check physical normal/complementarity/dissipation defects at the same 24
  sweeps, and impulse changes as well as averaged velocity changes.

Cost gate: frozen solve below about 13.8 ms (2x over serial), including split and
consensus work, with comparable convergence. Then unchanged integrated dishes
and paired GPU timing; only after admission expand to hardware/coins and N=2/4.
If fixed-budget quality loses, diagnose the remaining coupling modes rather
than hiding the loss with additional iterations.

This direction is consistent with public PhysX's ordinary PGS multi-block and
slab-velocity averaging design, not a claim of binary equivalence or copied
implementation: [PGS orchestration](https://github.com/NVIDIA-Omniverse/PhysX/blob/da950a3537927784951853c66618036f332ca0ce/physx/source/gpusolver/src/PxgCudaSolverCore.cpp#L1219),
[velocity averaging](https://github.com/NVIDIA-Omniverse/PhysX/blob/da950a3537927784951853c66618036f332ca0ce/physx/source/gpusolver/src/CUDA/solverMultiBlock.cu#L790).

## Mass-split first measurement

CPU colored tests: 7 passed. CUDA: 13 passed, including four new mass-split tests
for unequal-mass star convergence at 24 sweeps, physical impulse balance, cone
feasibility/energy, self/static/reversed endpoints, regularization and graph
degree reset. These are targeted tests, not a full solver suite or scene proof.

Same-input frozen solve (each GPU captures its own serial-history state, so row
sets differ across cards; within each pair every input is restored identically):

| GPU | MF rows | Original solve | Mass split | Speedup |
| --- | ---: | ---: | ---: | ---: |
| RTX PRO6000 | 2,571 | 26.135 ms | 9.290 ms | 2.81x |
| GB300 | 2,664 | 26.613 ms | 8.727 ms | 3.05x |

Both retain 24 sweeps and include preparation/consensus. On RTX, natural defect
changes 0.01425 -> 0.01331, complementarity 0.005661 -> 0.002949, dissipation
defect 0.0009885 -> 0.0007357; worst normal violation worsens 0.18884 -> 0.20910
(10.7%). GB300's measured defects all improve slightly or substantially. Both
remain finite and preserve effective impulse/velocity consistency; neither is
fully converged. Last-sweep impulse changes are recorded, avoiding a false stop
based solely on cancellation in averaged velocities.

RTX integrated dishes N=1, unchanged scene, 36 frames (only 0.6 simulated seconds):

| Window | Original physics | Mass split | Speedup |
| --- | ---: | ---: | ---: |
| Frames 4-12 | 291.03 ms | 128.87 ms | 2.26x |
| Frames 13-24 | 370.45 ms | 153.75 ms | 2.41x |
| Frames 25-36 | 390.19 ms | 166.16 ms | 2.35x |

No capacity drops/nonfinite states. Contacts differ as trajectories evolve; the
separate frozen-input test is the execution/convergence control. These are not
PhysX/MJWarp comparisons, not full-training throughput and not full shake-cycle
validation. The default path is unchanged. Opt in with
`FEATHER_PGS_MF_MULTIBLOCK=1 FEATHER_PGS_MF_MULTIBLOCK_MODE=mass_split`.

Evidence is in the existing
`/home/octi/Projects/fpgs-recipes-evidence-20260919/benchmark` directory:
`multiblock-split-frozen-dishes-n1-rtx.json`,
`multiblock-split-frozen2-dishes-n1-gb300.json`, and
`multiblock-split-dishes-n1-rtx.json`. Cold-compilation admission stops are
preserved as incomplete records, not misreported as performance losses.

## Scaling and cross-scene admission

The dishes gain also survives N=2/4 on RTX. For the final short window (frames
25-36), physics ms per batched environment step is:

| GPU / environments | Original | Mass split | Speedup |
| --- | ---: | ---: | ---: |
| RTX / 1 | 390.19 | 166.16 | 2.35x |
| RTX / 2 | 419.46 | 175.55 | 2.39x |
| RTX / 4 | 410.26 | 178.03 | 2.30x |
| GB300 / 1 | 385.50 | 161.54 | 2.39x |

All finite/capacity checks pass; these remain 0.6-second discovery runs. In a
later dishes snapshot taken from the candidate's own trajectory (call 300;
5,868 MF rows), the identical-input isolated solve improves 58.81 -> 13.61 ms,
but the natural residual worsens 0.00955 -> 0.02781 (2.91x). Normal violation
changes 0.13059 -> 0.13496, while complementarity and dissipation improve.
This is a real convergence tradeoff, not grounds to call the candidate qualified.

Generality checks on GB300:

| Scene / matched short window | Earlier current FPGS | Mass split | Ratio |
| --- | ---: | ---: | ---: |
| Coins / frames 4-12 | 1,884.95 ms | 90.39 ms | 20.85x |
| Hardware / frames 4-7 | 1,259.51 ms | 160.50 ms | 7.85x |

Earlier baselines are from `admission-20260920/coins-fpgs-n1-gb300.json` and
`fpgs-main-check-20260920/hardware-current-n1-gb300.json`, respectively, not new
full-trajectory controls. No row drops occurred, but hardware already reaches
35,712/37,248 MF slots (95.9%); longer safety is not established. No capacities
or collision geometry were changed to obtain these numbers.

Coins' identical-input control (call 30, 13,491 MF rows, 24 sweeps) confirms the
execution gain: 183.58 -> 7.203 ms (25.49x). It also exposes a numerical cost:
natural residual 0.00349 -> 0.01008 (2.89x worse) and dissipation defect
3.38e-5 -> 4.60e-5 (1.36x worse), despite better normal/complementarity defects.
Cone feasibility and physical impulse balance are insufficient acceptance tests.

Next diagnostic: decompose the same natural residual into normal/tangent parts
before choosing a correction. Do not increase the 24-sweep allowance or tune
launch minutiae while convergence is unresolved. A simultaneous 2D disk update
has a better maximum-dissipation fixed-point argument than component-wise
rescaling, but should only be pursued for a demonstrated tangent limitation.

### Diagnosed normal bottleneck: static support under a high-degree body

The decomposition rejects the tangent hypothesis. Coins' normalized natural
normal residual is about 3x worse, while the tangent residual is about 3x better.
On RTX the worst candidate row connects local DOFs `[9, -1]`: tray to static
world, with 65 incident tray pairs. Its physical effective diagonal is 6.64,
impulse 5.94e-5 and remaining compressive velocity residual -0.02815. Equal
mass splitting makes this support correction unnecessarily conservative.

Bounded structural correction: separate all static-contact pairs from the
dynamic pair degree count. Canonical `(-1, body)` grouping makes them inherently
body-disjoint, irrespective of static-shape count. Solve dynamic pairs and
consensus, then solve the static pairs directly in one parallel GS phase.
Every row is still visited exactly once per sweep; 24 sweeps, friction law,
regularization, masses, timestep and collision are unchanged. This is a general
static/dynamic partition, not a hard-coded tray/rack path. Check the total defect
afterwards: improved static support may shift error to adjacent dynamic contacts.

### Direct-static correction: matched-input result

The diagnostic now runs three routes from identical input, eliminating trajectory
variation from the correction comparison. In the coin snapshot:

| GPU / route | Solve ms | Natural residual |
| --- | ---: | ---: |
| RTX / serial | 184.743 | 0.003429 |
| RTX / split all pairs | 7.523 | 0.009704 |
| RTX / direct static pairs | 9.199 | 0.001737 |
| GB300 / serial | 181.935 | 0.003118 |
| GB300 / split all pairs | 7.206 | 0.009892 |
| GB300 / direct static pairs | 8.266 | 0.002127 |

Thus the correction preserves 20.08x RTX / 22.01x GB300 isolated solve speedup,
and all reported aggregate normal/complementarity/dissipation defects improve
versus serial on these snapshots. The remaining worst natural row moves from
tray/static support to ordinary dynamic contacts. Physical impulse balance and
cone feasibility remain clear. This is not a claim of full-scene convergence.

On the direct-static candidate's own later dishes trajectory (call 300, 5,508 MF
rows), serial 53.101 ms -> corrected 13.275 ms, approximately 4.00x isolated
speedup. Natural 0.01063 -> 0.01034, normal 0.18192 -> 0.17585, complementarity
0.003140 -> 0.001231 and dissipation 0.0002769 -> 0.0001226. Unlike the earlier
all-pair history, this loaded state does not show a support-driven natural
regression. One snapshot cannot establish a trajectory-wide guarantee.

Targeted tests now pass 15 CUDA / 7 CPU cases. Added checks cover high-degree
static support at 24 sweeps, static-only/reversed endpoints, physical impulse
response and dynamic/static/empty graph transitions. Enable the correction with
`FEATHER_PGS_MF_MULTIBLOCK_DIRECT_STATIC=1` in addition to the two opt-in flags
above. No production defaults, Lab code, geometry, capacities or sweep budgets
have changed.

## Final checkpoint and next substantive correction

Full-cycle admission uses 2 seconds settling plus 10 seconds robot motion:

- GB300 coins: 720/720 frames (12 simulated seconds), 103.79 seconds wall.
  All finite-state and available capacity checks pass; no recorded dropped rows.
  Peak 14,517/15,984 MF rows, 4,851/5,328 contacts, 44/256 dense rows.
  Physics averages 97.83 ms in frames 4-12 and 124.68 ms over the whole cycle;
  no matching full-cycle serial baseline exists, so no full-cycle speedup claim.
- RTX dishes: 696/720 frames (11.6 simulated seconds), stopped by the unchanged
  150-second wall budget at 151.15 seconds. No recorded capacity/finite failures;
  peak 6,981/8,064 MF rows, 2,339/2,688 contacts and 107/256 dense rows.
  This is a near-complete cycle, not a completed cycle or convergence certificate.

The corrected dishes early windows are 127.08 / 141.95 / 143.16 ms for frames
4-12 / 13-24 / 25-36, respectively: 2.29x / 2.61x / 2.73x versus the fresh
original FPGS baseline above. These task-level ratios include collision and all
physics work, but remain short settling-window comparisons, not whole-cycle
speedups or comparisons against PhysX/MJWarp.

Files: `multiblock-static-cycle-coins-n1-gb300.json` and
`multiblock-static-cycle-dishes-n1-rtx.json` in the existing benchmark directory.
These runs advance beyond the earlier settling-only timings into robot contacts.
Neither the helper's finite/capacity checks nor a few frozen solves certify all
trajectory physics; all reports retain `qualified: false`.

Hardware exposes the remaining limitation. Its GB300 frozen call-40 state has
5,535 MF rows: serial 54.86 ms, all-pair splitting 9.64 ms, direct-static 10.84 ms
(5.06x execution speedup). Direct-static natural residual is 0.01002 versus
serial 0.00348; complementarity is 0.000520 versus 0.000187. Normal velocity and
dissipation improve slightly, but this is not a no-worse-quality result. The
worst remaining normal row joins dynamic offsets `[9, 15]` with incident pair
degrees 26 and 2. The static-support correction is no longer sufficient.
No earlier capacity checks reported loss and row count is below capacity;
exact-capture native flags were not serialized, so do not claim that stronger
capacity certificate. Evidence: `multiblock-static-frozen-hardware-n1-gb300.json`.

Next target: reduce the over-conservative high-degree **dynamic** coupling using
a general graph-aware block partition or hub correction, at the same 24 sweeps.
Test normal progress and total preparation/solve cost on the identical hardware
state; retain coins and dishes as controls. Do not change tangential projection
to treat a demonstrated normal-contact error. No TGS, scene-specific body names,
extra iterations, capacity reduction or collision changes are justified here.

## PhysX-target execution candidate (20:23 UTC)

Clean matched GB300 coin frames 4-12 remain 97.83 ms FPGS versus 42.35 ms
PhysX physics: 2.31x more reduction is needed. The next two changes remove
intermediate work without changing the mass-split mathematics:

1. Parallel preparation: replace the serial per-world pair grouping, unused
   greedy coloring and serial degree counting with sorted-key head flags,
   an inclusive prefix scan, parallel pair emission and degree atomics.
   Add two integer scan buffers; preserve exact stable pair/row order and
   direct-static degree semantics. Keep old preparation for a causal control;
   absent color diagnostics must be marked unavailable, not reported as colors.
2. Compact dense phases: avoid revisiting every free object's DOFs for robot
   constraints in every sweep. Retain canonical J/Y producers and the exact
   dense/contact/velocity-limit order. Any compact support must include nonzero
   response Y as well as Jacobian J; arbitrary articulation cross-response,
   friction sibling updates and rigid velocity limits must remain supported.
   Preparation cost and fallback work count against the saving.

The latest coins node trace identifies both costs, but both GPUs were occupied
by other users' jobs. `physx-target-coins-rtx.nsys-rep`,
`physx-target-profile-coins-n1-rtx.json` and
`physx-target-before-coins-n1-gb300.json` are **contended diagnostics, not timing
evidence**. No new speedup or regression is claimed. All our GPU jobs finished;
other users' jobs were left untouched. Continue CPU implementation/tests while
waiting for an idle benchmark window; do not qualify performance under contention.

Gate: exact preparation equivalence including empty/full/multiple worlds,
single/triplet rows, reversed/static/self endpoints and large parent indices;
same-input dense-phase response/order checks. Then clean GPU graph tests and
same-input timing with the previous route, followed by unchanged short integrated
coin/dish/hardware windows. No TGS, added sweeps, geometry, capacity or Lab changes.
Execution savings do not resolve hardware's separate convergence regression.
Its saved degrees 26/2 above are total graph degrees, including static pairs;
runtime direct-static splitting excludes those static pairs.

### CPU implementation checkpoint

- Parallel preparation is implemented and integrated behind
  `FEATHER_PGS_MF_MULTIBLOCK_PARALLEL_PREPARE=1` (requires mass-split mode).
  Existing preparation remains the default/control. Three new CPU tests compare
  active pair payloads and degrees exactly; scratch adds eight bytes per allocated
  MF row per world. Evidence helpers record the flag and absent color diagnostics.
- Limit-only phase 5 now stages/stores only the MF velocity-limit tail, not the
  contact prefix. It returns before state staging when both limit segments are
  empty, except when fused drive clamps may still act. Phase 0 is unchanged.
  This is execution work removal, not a different projection or iteration budget.
- `mf_dense_support.py` prepares exact per-row 32-DOF tile masks using the union
  of nonzero J and Y. It retains Y-only entries, tiny values and nonfinite entries
  conservatively. Coins require four mask words per dense row (4 KiB/world for
  256 rows). This building block is CPU-tested but **not integrated into the
  dense solve**: it provides no runtime gain yet. The next kernel change must
  remove full-D prefetch/register work as well as restrict dot/response traversal;
  sibling updates must use their own row's mask.

Combined CPU gate: 22 tests passed, 17 CUDA tests skipped because devices were
deliberately hidden. Covers grouping, storage/source generation, tile support
and packed metadata. Ruff and diff checks pass. No CUDA correctness, graph replay
or clean performance result exists for these changes yet; do not promote them
or extrapolate the contended trace into a claimed PhysX speedup.

Next executable gate requires an idle GPU: existing CUDA tests, then short
same-input old/new-preparation controls including preparation time. Follow with
matched integrated coins/dishes windows and the hardware residual control.
Hardware's fixed-budget convergence regression remains unresolved; cost-bounded
star clusters are a separate proposed numerical experiment, not implemented here.

### Idle-GPU window: parallel preparation measurement

Other users' jobs finished at approximately 20:34 UTC; all measurements below
were then taken without another compute job on the measured card. Explicit CUDA
tests cover parallel/legacy preparation equality and captured preparation plus
24 sweeps through loaded/empty resets. Existing projection, storage and metadata
checks also pass. These are targeted correctness checks, not trajectory qualification.

Identical-input coin solve, same mass splitting, direct static phase and 24 sweeps;
both controls include the updated tail-only limit staging:

| GPU | MF rows | Serial preparation | Parallel preparation | Speedup |
| --- | ---: | ---: | ---: | ---: |
| RTX PRO6000 | 13,488 | 7.348 ms | 4.989 ms | 1.47x |
| GB300 | 13,428 | 7.117 ms | 4.863 ms | 1.46x |

Physical defects agree to floating-point accumulation noise between preparation
variants: natural residual 0.00222156 on RTX and 0.00170051 on GB300. The much
larger ratios against original serial FPGS are separate algorithm comparisons,
not the incremental gain in this table.

Fresh integrated coins N=1, frames 4-12, physics milliseconds per environment
frame (ten unchanged 1/600-second physics calls):

| GPU | Current FPGS | PhysX | PhysX/FPGS throughput |
| --- | ---: | ---: | ---: |
| RTX PRO6000 | 61.587 | 37.354 | PhysX 1.65x faster |
| GB300 | 62.367 | 42.104 | PhysX 1.48x faster |

Whole-environment times are 65.767/45.576 ms RTX and 66.501/50.689 ms GB300
(FPGS/PhysX). FPGS records no finite/capacity failures over all 36 frames. This
is 0.6 seconds of settling, not a full vigorous cycle; PhysX's native capacity
diagnostics are not exposed by this helper. The loaded PhysX USD schema defaults
to TGS and the manager does not override solverType; this is schema/default
evidence, not a native runtime query. FPGS development remains non-TGS.

Files in the existing benchmark directory:
`physx-target-parallel-coins-n1-{rtx,gb300}.json`,
`physx-target-parallel-integrated-coins-n1-{rtx,gb300}.json`, and
`physx-target-reference-coins-n1-{rtx,gb300}.json`.

The clean node trace `physx-target-parallel-coins-rtx.nsys-rep` puts 42.2% of
summed kernel time in the old dense/limit kernel. Its strictly alternating
phase-6/phase-5 launches separate into 98.982 us versus 1.743 us on average;
phase 6 alone accounts for 41.5% of all kernel time. Dynamic pair solves account
for 20.3%, direct static pairs 19.1%. Pair emission is now 0.1%. Thus the next
structural target is specifically full-world dense traversal, not further
preparation tuning. Profile attribution is not an unprofiled timing prediction.

### Prepared contact coefficients candidate

In parallel with dense traversal, test removing repeated contact-operator work.
The current mass-split kernel recomputes a 12-component split-curvature product,
its warp reduction and regularized projection coefficients on every row of every
sweep. J, response, degrees and regularization are fixed during these 24 sweeps.
Prepare exact residual/old-impulse coefficients once, then reuse them while
retaining current velocities, impulses, contact order and sibling updates.
Add bounded per-row coefficient storage and a preparation launch; both count
against the saving. Keep uncached execution as the identical-input control.
No dropped contacts, patch-friction substitution, extra mass replicas or sweeps.

This targets the remaining pair kernels (39.4% of the clean coin profile before
dense traversal changes), not an unrelated small launch tweak. Gate: unequal and
same-endpoint response, degree changes/static partition, zero/tiny regularization
weights, delayed friction, fixed-budget residual and physical impulse consistency;
then total same-input solve cost. It is a hypothesis, not a claimed speedup.

### Dense-tile result and bounded staging correction

GB300 S2: identical-input solve 5.111 -> 3.680 ms (1.39x), integrated frames
4-12 physics 62.367 -> 50.449 ms (1.24x). Frozen physical defects agree to
rounding, and all 36 integrated frames have clear finite/capacity checks.
This is less than eliminating the entire old dense kernel, as expected: only
its operator traversal changed. The clean S2 profile still spends 6.949 ms/frame
in phase 6 (14.8% of summed kernel time), versus 0.059 ms/frame preparing masks.
Full-world mapped velocity load/store remains in every dense sweep. The profile
isolates the kernel, not the cost of those particular instructions.

Next corrective mapping: compute the union of exact row-support tiles once and
restrict dense-phase velocity staging/publication to that union. Keep all response
and sibling supports; retain full shared indexing, but never access uninitialized
unsupported entries. Add one mask per world and include its preparation cost.
Keep full staging as an explicit identical-input control. This removes more
unneeded world traversal, not a different projector or additional sweeps.

First prepared-coefficient RTX control: dense tiles without caching 3.580 ms,
with caching 3.222 ms (1.11x); both changes versus the same-input earlier
structure 5.327 ms give 1.65x. All reported residuals agree to rounding and
cache-miss diagnostics remain zero. Combined integrated RTX physics is
42.837 / 38.786 / 38.925 ms for frames 4-12 / 13-24 / 25-36. Fresh PhysX is
37.354 / 37.264 / 37.337 ms, so physics parity is not yet achieved.
No claim of broad-scene or full-cycle convergence follows from these results.

### Union staging: gain measured, full-cycle admission failed

GB300 identical-input, 13,482 MF rows: full velocity staging 3.344 ms ->
union staging 2.859 ms (1.17x). Against the same-input pre-tile/pre-cache
structure, 5.123 -> 2.859 ms (1.79x). All measured defects agree to rounding;
cache misses are zero. Targeted support/coefficient tests pass on both GPUs.

Short integrated coin windows initially reached approximately PhysX speed, but
must not be accepted as a full-scene win: union-staged runs become nonfinite
at frame 252/720 on GB300 and 288/720 on RTX, after robot/tray contact starts.
Capacities and cache-miss flags remain clear. Further isolation:

| Route | GPU | Full-cycle admission |
| --- | --- | --- |
| Parallel preparation only | RTX | 720/720, finite, capacities clear |
| Tiles + cache, full velocity staging | GB300 | 720/720, finite, capacities clear |
| Tiles + cache, full velocity staging | RTX | Nonfinite at frame 660 |
| Tiles, no cache, full velocity staging | RTX | Nonfinite at frame 348 |

Thus the newest union staging is not the only route exposing the failure, and
coefficient caching is not required for it. Do not attribute it to floating-point
trajectory variation without a cause. The code audit identifies an inherited
dense-friction shared-memory read/write hazard: all lanes read and then overwrite
the sibling impulse without an intervening ordering guarantee. End-of-row warp
synchronization does not protect that earlier read/write pair. The correction
under test gives scalar impulse mutation one lane owner and broadcasts its
results while retaining warp-parallel physical velocity responses. This is a
programming-model issue, not yet proof of the integrated failure's cause.
See NVIDIA's [warp synchronization guidance](https://developer.nvidia.com/blog/?p=9333).

These records remain in the existing benchmark directory under
`physx-target-{staged,fullstage,tiles,parallel}-cycle-coins-n1-*.json`.
The default multi-block/tile/cache paths are still opt-in. No Isaac Lab source,
geometry, timestep, substep, iteration or capacity setting changed for these runs.

### Checkpoint: structural cost result, not a reliable PhysX win

Fresh PhysX N=1 coins, all 720 frames (2 s settling + 10 s six-cycle motion),
weighted by frame count, synchronized milliseconds per environment frame:

| GPU / route | Physics | Whole environment | Admission |
| --- | ---: | ---: | --- |
| RTX PhysX | 40.064 | 48.222 | 720/720, finite |
| RTX FPGS parallel preparation only | 94.180 | 98.271 | 720/720, finite, capacities clear |
| GB300 PhysX | 45.245 | 53.906 | 720/720, finite |
| GB300 FPGS tiles + coefficients, full staging | 52.921 | 57.418 | 720/720, finite, capacities clear |

The GB300 complete FPGS capture is 2.36x faster than the previous direct-static
full-cycle capture (124.676 ms physics), but PhysX is still 1.17x faster. This is
one surviving trajectory, not acceptance of a configuration whose RTX run failed.
PhysX native capacity flags remain unavailable; its configured defaults were not
changed and no capacity overflow was reported in these reference logs.

Scalar ownership hardening did not resolve the full-cycle problem: the subsequent
RTX union-staged run failed at frame 492. The bounded pre-fix racecheck reported
zero hazards; source-level hardening must not be misrepresented as the demonstrated
failure cause. It covers both current and coulomb_newton's shared dense scalar
disk projection; bisection modes retain their existing projector.

The important same-input control now includes active robot contact, not just
settling/speculative rows: at frame 240 on RTX, the old structural solve takes
18.486 ms versus 2.629 ms with tiles, coefficients and union staging (7.03x).
All equivalent structural variants agree in dense natural residual (~0.002247),
MF natural residual (~0.012017), complementarity and dissipation to rounding.
Physical impulse/velocity consistency is about 3e-7 relative. These remain finite-
budget, unconverged results: final-sweep velocity change is ~0.474 (mixed generalized
velocity units), and the worst MF normal still closes at ~0.268 m/s. This snapshot
does not establish trajectory stability or a 7x whole-physics gain.

Existing replay helper now supports graph-backed history followed by a frozen
eager solve, avoiding thousands of eager history launches. The active snapshot is
`physx-target-owner-grasp-coins-n1-rtx.json`, reproduced with:

```text
--history multiblock --graph-history-steps 239 --after-calls 10
--min-dense-rows 20 --min-mf-rows 100 --repeats 2
```

Enable the experimental flags `FEATHER_PGS_MF_MULTIBLOCK=1`,
`FEATHER_PGS_MF_MULTIBLOCK_MODE=mass_split`, and the `..._DIRECT_STATIC`,
`..._PARALLEL_PREPARE`, `..._DENSE_TILES`, `..._CACHE_COEFFICIENTS`,
`..._STAGE_VELOCITIES` flags (each 1). Use the same coins bundle and Newton import
path recorded in the capture. Omitted opt-ins remain disabled.

A diagnostic `--probe-unsafe-solve --unsafe-speed 10000` follows the candidate
until its solve output becomes nonfinite or exceeds that generalized-speed norm,
then replays exactly that input. It never clamps velocities. The first such probe
completed 720 frames without a trigger; its eager synchronization/clone overhead
is not performance evidence, nor does one pass invalidate the graph-run failures.
The initial helper exited with a diagnostic no-capture exception despite the
admission report completing; the helper now reports that no trigger was observed.

Next decisive investigation is the failure frontier: first invalid operator/input
versus first unstable projection, and graph versus eager execution from the same
saved state. Record live diagonal positivity, actual dynamic pair degrees, phase
boundaries and impulse/response balance. Do not change timestep/iterations, cap
velocities, or accept a faster path solely because one trajectory survives.

Final targeted gate: 56 CPU/RTX tests pass across grouping, coefficients, dense
support, storage and metadata; the ten dense-support/ownership tests also pass
on GB300. Ruff and diff checks pass. No Lab source changes or default promotion.
Runtime source SHA256 pins:

```text
solver_feather_pgs.py 10943692f1630c26ec36674ee2effd675cc859b23101cf32fba1a7f7de2c3c48
mf_multiblock.py     9085f4a1a78e796b91d615dcbd4086a52b20cf7fdd04579b2c446332d885bca9
mf_dense_support.py  b12f242b8f28aaa41dbef5641a10f02947c496379ae20c8773ad0f1d6939978c
```

### September 21 idle-RTX recheck

At 04:28 UTC both cards were compute-idle. Runtime sources match the pins above;
no implementation or recipe change preceded this run. One fresh full 720-frame
coin cycle on RTX completed with finite states, no capacity failure or cache miss:

| Route | Physics ms/frame | Environment ms/frame |
| --- | ---: | ---: |
| Full experimental FPGS | 42.565 | 46.591 |
| Fresh PhysX reference | 40.002 | 48.024 |

FPGS/PhysX throughput is 0.940x for physics and 1.031x for the whole environment.
This surviving run does not erase the previously recorded nonfinite failures,
and neither ratio is training throughput. Captures:
`physx-target-resume-cycle-coins-n1-rtx-20260921.json` and
`physx-target-resume-reference-coins-n1-rtx-20260921.json`.

The next bounded diagnostic retains the first unsafe solve's exact input/output
inside the original CUDA graph, then reuses the existing frozen comparison. It
must not generate a new eager physics step before replay: that would change the
inputs being diagnosed. Added snapshot work is excluded from performance claims.

The first live-graph probe did not trigger on RTX (720 frames). Its GB300 latch
at graph call 2214 / environment step 222 produced an apparent response/impulse
defect of 2.736 versus about 1.2e-5 in isolated replay, but this comparison is
INVALIDATED by a diagnostic lifetime defect: snapshot arrays were allocated
inside capture while expected to persist across subsequent graph launches.
Installed Warp creates CUDA graph allocation nodes and instantiates graphs with
`cudaGraphInstantiateFlagAutoFreeOnLaunch` (`warp/native/warp.cu`, allocation code
around 801 and instantiation around 3652). Gated-off later copies do not preserve
data in reallocated storage. The implausible zero dense count is another reason
not to interpret this capture as a production solver fault.

`physx-target-graph-frontier-coins-n1-gb300-20260921.json` is retained only as a
record of the invalid diagnostic. Corrected instrumentation allocates persistent
buffers before capture, saves one raw snapshot on latch, and reports input/output
velocity norms and the largest balance-defect DOF. No solver-cause conclusion
follows from the invalid capture.

A clean-cache, uninstrumented RTX control still failed at frame 288/720 with
nonfinite robot/clutter state, despite clear capacity flags, zero dropped rows
and zero coefficient-cache misses. Runtime source hashes were unchanged. The
previously absent isolated `WARP_CACHE_PATH` compiled new kernels, so an old
generated-kernel cache is not necessary for the failure. Partial-run timings are
not a full-cycle performance result. Capture:
`physx-target-freshcache-coins-n1-rtx-20260921.json`.

Corrected probe smoke test (deliberately tiny trigger, not a physical failure)
retained graph call 1 across the remaining calls of an environment frame. It
preserved 5 dense rows including enabled mimic slot 0, and original/replayed
impulse-response defects matched at 7.1e-10 absolute. Snapshot storage is now
constructor-owned outside capture; pointer, shape and dtype bindings are checked.
Helper SHA256: `f2c8a46576f6362a09594bb261b36131fd1d7b47dc7ca1f4abdbed0f7b766e8b`.

Corrected real probes on both cards agree between live graph and isolated replay:

| GPU | Graph solve / environment frame | Peak input angular component | Free body | Active rows touching body |
| --- | ---: | ---: | ---: | ---: |
| RTX | 1615 / 162 | -10130.161 rad/s | 333 | 0 |
| GB300 | 2095 / 210 | 10095.217 rad/s | 210 | 0 |

Those peaks already enter the solve and leave it unchanged. All active operators
are finite, mimic slot 0 is present, and original/replayed impulse-response
defects are about 1.3e-5 absolute (roundoff). Raw input/output NPZs accompany
`physx-target-persistent-frontier-coins-n1-{rtx,gb300}-20260921.json`.
This points upstream of the captured contact solve, not to graph corruption.

A separate contact-free CPU reproduction uses one free coin, zero gravity,
torque, damping and contacts, initial spin (20,40,60) rad/s, and dt=1/600.
Its captured mass/inertia are physically consistent with the scene's cylinder.
Generic FPGS (multiblock disabled) grows rotational energy to 59.48x after 1 s,
15651.63x after 2 s and 8.13e6x after 3 s. An explicit body-Euler reference closely
matches; a small-step-appropriate RK4 reference remains within 5e-6 relative of
the initial energy over 3 s. The existing explicit gyroscopic integration is
therefore independently unstable, without any contact scheduling or buffer issue.

A bounded, default-off `FEATHER_PGS_FREE_BODY_IMPLICIT_GYRO=1` experiment replaces
only that predictor term for independent six-DOF free bodies. It retains the
effective factor (including armature/implicit drive terms), physical COM inertia,
external forces, timestep and contact budget. The first backward-Euler candidate
passed 720 frames on both cards (RTX 37.682 ms physics / 41.688 ms environment;
GB300 44.389 / 48.751) but lost about 22% rotational energy in the isolated 3 s
test. Its `physx-target-implicitgyro-...` captures are not accuracy qualification.

The current candidate uses midpoint gyroscopic torque instead:
`H_eff (omega_new - omega_hat) + dt [g((omega_old+omega_new)/2) - g(omega_old)] = 0`,
where `g(omega) = omega cross (I_physical omega)`, with three small Newton updates.
No velocity cap or extra damping is introduced. The actual full-solver float32
contact-free test now ends 3 s at energy ratio 1.00000268 and angular-momentum
magnitude ratio 1.00000158. Momentum **direction** still has about 1.98% error at
this timestep: orientation integration remains first-order, not fully midpoint.
The independent numerical experiment shows direction error improving under
timestep refinement; do not call this exact momentum conservation.

Five focused CPU tests cover the integrated solver, force/effective-inertia
retention, small inertia scale, frame rotation, kinematic/nonfree eligibility,
lower-triangle-only Cholesky use and timestep accuracy. Existing free-root and
momentum tests also pass (19 tests, one skipped; six momentum tests additionally
pass with the gyro opt-in). Changed runtime/test files pass the repository Ruff
checks; standalone evidence helpers pass the E4/E7/E9/F smoke check (not a claim
that they satisfy every Lab/ Newton style rule). No Lab source or default
promotion. Current source pins:

```text
solver_feather_pgs.py bfa4345833882ce28ce242e4b131e5227e344ffa52f662d22a072f3783c480f6
free_body_gyro.py     31fe41f304e56041ce27bc499fa335f2017bac32a253365436e40c8e1c88b0ee
```

### Midpoint candidate: complete coin cycle

Both cards completed all 720 frames, with finite robot/clutter state, zero dropped
rows, clear collision/constraint capacity flags and zero coefficient-cache misses.
Enable the midpoint gyro opt-in **in addition to** the seven structural candidate
flags above. No change to dt, 24 contact sweeps, decimation, geometry or capacities.

| GPU | FPGS physics ms | PhysX physics ms | Physics speedup | FPGS environment ms | PhysX environment ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX | 42.461 | 40.002 | 0.942x | 46.688 | 48.024 |
| GB300 | 44.739 | 45.245 | 1.011x | 49.142 | 53.906 |

Each midpoint timing is one complete cycle, not repeated performance or complete
contact-convergence qualification. It is not consistently faster than PhysX.
Whole-environment throughput ratios are 1.029x RTX / 1.097x GB300; these are not
training results. Do not interpret the timing difference from backward Euler as
isolated gyro-kernel overhead: trajectories and contact loads differ.

Final candidate captures are
`physx-target-midpointgyro-coins-n1-{rtx,gb300}-20260921.json`.
PhysX references are the fresh RTX
`physx-target-resume-reference-coins-n1-rtx-20260921.json` and saved GB300
`physx-target-reference-cycle-coins-n1-gb300.json`, both the same complete bundle.
Peak rows were dense 113 / MF 14238 on RTX and dense 47 / MF 15426 on GB300,
within unchanged 256 / 15984 capacities. GB300's contact peak was 5154 / 5328:
clear in this run, but not a guarantee for other seeds or batch sizes.

The new path remains default-off. The demonstrated advance is identifying and
stabilizing an inherited free-rotation failure that contaminated many-body
qualification; it is not a new claimed contact-solver speedup. Further contact
accuracy, repeated full-cycle runs and batch-size checks remain before promotion.
