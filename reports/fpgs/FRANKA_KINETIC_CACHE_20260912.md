# Franka kinetic-cache checkpoint: correct components, costly first mapping

Checkpoint at 17:37 UTC, 2026-09-12, after the bounded approximately60-minute
prototype. The 4x-MJWarp objective remains open for BOTH tasks. This report
does not change Newton runtime or enable the scratch Franka experiment.
The preceding repeated Kuka improvement remains in
[FOURX_PROGRESS2_20260912.md](FOURX_PROGRESS2_20260912.md): 2.135x/2.131x
the fixed corrected MJWarp reference on RTX/GB. Franka remains at its prior
accepted5.746087/5.195039ms, or1.864x/1.992x that reference.

Isaac Lab1d8feb82 remains untouched and clean. Newton source ancestry, original
handoff, eight GS sweeps, dt1/240, two physics substeps and environment
decimation4 are preserved. No capacity, parent pointer or task parameter
changed. This report continues the `ooctipus/newton` fork branch; no PR or
default-on promotion is implied. Large captures and prototype sources remain
local at the exact paths/pins below.

## What the experiment actually replaces

The candidate changes primary9 dynamics representation. One body-world
producer publishes current public poses/COM velocities and canonical S/V/A,
COM/origin/body-bias fields, contracts generalized bias every state, and builds
geometric9x9 mass only on its refresh epoch. It avoids primary full/compact
spatial-inertia and composite intermediates. Two free/prescribed bodies retain
the original finalizer. The consumer applies current controls, passive and
external forces, R/K, factorization/prediction, canonical mimic/limit prefix,
and an exact signed-bound-tested rank-one action. Rejected eligible worlds
run the unchanged eight-sweep native local9 solver.

This is not an arithmetic-reduction theorem: the geometric Gram has184
body/lower-entry contributions per world. The first implementation retains
canonical primary body-cache writes, rather than pricing unimplemented lazy
stores as savings. Contact20/40/general, free6 solves, full live allocation,
invalidation and final generalized integration are not replaced by this core.

## Actual numerical gate passed on both GPUs

Each physical GPU used its own two immutable512-world inputs, refresh1600
then reuse1601. Newly generated Hgeo/L and their epoch are shared into reuse;
current bias/forces are not held with mass. Public-output controls additionally
test fresh-current mass and mixed refresh on each input. Numerical convergence
and physical action, not cross-algebra bit identity, are the criteria.

- Chained held-H error is at most2.046e-6 RTX /1.655e-6 GB; held physical
  basis-action residual at most2.974e-6/2.569e-6. Separate fresh-current H
  checks reach2.161e-6 and action residual3.045e-6.
- Hard mimic velocity residual is at most2.236e-8, active signed-bound
  violation is zero, and momentum defect is at most1.460e-7.
- All original-fallback diagonal, impulse and velocity deltas are zero.
  Across four checkpoints,1,903 analytic and33 local9-fallback world cases
  are covered;112 other-owner cases are not privately solved here.
- Both full two-epoch graph replays preserve78 writable and118 readonly
  allocations. This is repeatability of the same path, not a bit-parity rule.
- Public body state, current S/V/A/force and free-body inertia pass the frozen
  controls. Primary full/compact inertia stays untouched. A1% wrong-mass
  negative control gives approximately.009902 physical-action residual.

An initial producer discrepancy was diagnosed as global-translation rounding,
not a Gram-law error. One root-relative private scan improves H accuracy
against independent FP64 geometry; global translations are restored for public
poses. CPU controls also cover changed q/qd/gravity, controls/external forces,
kinematic exclusion, stale epochs and nonpositive factors. These extra changed
cases are not mislabeled as CUDA coverage. No complete trajectory or MJWarp
physical-parity claim follows from the component tests.

## First16K cost result and causal attribution

The inputs are32 full-storage copies of each saved512 scene, with rebased
global ownership and independent arrays. They are not an actual new16K
trajectory or a modulo read from a small cached input. Fixed prefix capacity
stays192. Source, input, UUID, process-reaping and idle checks pass.

Forty retained event samples follow ten warmups. All restores precede the
start event. The timed graph runs producer, consumer and local9 once at each
of two epochs. Two epochs correspond to ONE simulation tick, not one
environment step: decimation4 times two substeps is EIGHT solver substeps per
environment step. The multiplier is therefore4. A prelaunch units error was
caught by independent review and now has a regression checking both original
profiles and24 FK/finalizer/qdd calls across three profiled environment steps.

| Core measurement, milliseconds | RTX | GB |
| --- | ---: | ---: |
| Median two-epoch event | .538655996 | .508063972 |
| Environment-equivalent serial core | **2.154623985** | **2.032255888** |

These are partial implementation costs, NOT whole-physics times or gains.
A separate six-sample node profile attributes the miss:

| Mean node cost per environment, ms | RTX | GB |
| --- | ---: | ---: |
| Current body/kinetic producer | .873065 | .708971 |
| Current consumer | 1.144253 | 1.146667 |
| Original local9 fallback | .126507 | .150741 |
| Inter-node gaps | .004651 | .003115 |
| Six-node span | 2.148475 | 2.009493 |

Consumer refresh/reuse costs are146.768/139.296us RTX and149.909/136.757us GB.
Most consumer cost remains on reuse; fallback and gaps are not the dominant
loss. This identifies the expensive owner, not an internal phase or occupancy
cause. Native producer/consumer resources are64/56 registers and3712/1140B
shared on both GPUs. Producer offline stack is72B despite node-local metadata
reporting zero; there are no offline spills.

RTX uses real CUPTI process/correlation joins. GB exports correlationId0 for
all96 graph nodes. The reader retains them all and proves sixteen serialized
same-process/same-graph/same-stream launch windows, each with the same six
ordered node IDs and a successful completion fence before the next launch.
This is explicit window attribution, not a fabricated correlation join.
Four CPU reader tests include failing-first actual GB records and malformed,
overlapping and unfenced negatives. No GPU rerun was needed for that repair.

## Live dependency and budget correction

The monolithic consumer is NOT ready to integrate. It produces the prefix
needed by the original contact/MF allocator but reads finalized MF count for
rank admission. Final MF count also includes velocity-limit rows from current
free6 predictor. That is a live dependency cycle; reading an old count is not
a fix. Raw-zero alone does not prove absence of free-body velocity-limit MF.
The four saved inputs do not exercise raw-zero/MF-positive states.

A correct split can run force/factor/predictor and prefix early, then original
contact/free-velocity-limit allocation after collision, then late rank/local9.
It must charge the additional launch, reads, ownership and synchronization.
Original S1-3 may overlap collision; the monolithic late consumer cannot claim
that overlap. Consequently the serial2.154624ms is a lower bound on this
partial service's serial cost, NOT a universal necessary wall-time bound
against the old2.140011ms exposed family. This qualifies the stronger wording
in the initial local card/cost audit. A different schedule needs actual overlap
measurement and complete retained-work accounting.

Under the card's fixed-exposed-owner model,20% allows.990811ms and10% allows
1.565411ms of replacement exposed time. The first serial core misses both.
Even deleting the entire traced consumer leaves1.004222ms RTX before omitted
services, so consumer-only work with other owners fixed cannot claim20%.
The target is still4x, not a redefinition to10%.

The one proposed causal successor caches the held inverse action, preserving
canonical L, and splits early and late consumers. Source currently repeats
triangular solves for prediction, mimic action and each sparse-row diagonal.
Removing those repetitions could materially reduce the measured consumer;
that is an unvalidated hypothesis, not a proven internal bottleneck. A new
cost card and physical-action checks must precede its one bounded native
retry. No tile/register/parameter sweep is authorized by this checkpoint.

## Reproduction and retained evidence

Prototype directory: `/tmp/fpgs-franka-kinetic-prototype-euL5lYJq`.
Use the existing paired owner `/tmp/fpgs-capacity-20260911-O7dqFC/run_pair.py`
with the unchanged Lab interpreter, a fresh output directory, CUDA hidden in
the parent and exact UUIDs in its children. No Lab edit is required.

- Numerical wrapper `/tmp/fpgs-franka-kinetic-correctness-EJKDViYD/run_correctness.py`,
  `7e29e3514118197e2b568bdd2aa36bd5401e419d020039c84bfd52eeff8473ec`;
  use `--newton /home/octi/Projects/newton-fpgs-composed-world-20260912` at50dfa28d.
  Numerical parent manifest `f645a8c36f4837dc02858a66a4f057e247fe4d2da8a5ee4a28f39bf5be3e696a`.
- Cost wrapper `run_cost.py`, `6f22deec198066fdc6f224c638480e343d1bc2564fa49e2a714f76e5e97d4243`;
  use accepted064 tree `--newton /home/octi/Projects/newton-fpgs-fourx-progress-20260912`
  and forward `--copies 32 --samples 40` after the owner's `--`.
  Cost parent manifest `43e2f35fdfd74844e471aa1faf22d0fc8765955e7ea8ed5a98c2abbc310d40dd`.
- Producer `991cecd0a87ba1384cf76120cbd37ac31efe3f4b284b9794692ad979129994f5`;
  consumer `f51979a118c2ea07743c62d2ca5d024412da5df516f3e42e3d2627f81ae30e86`.
- Node wrapper `/tmp/fpgs-franka-kinetic-nodes-jne3Tlsn/run_nodes.py`,
  `daa274d670a6c098e4d6a28b8b5a9904a8d1fc573167ec1b2eb146cb9bece72c`;
  same accepted064 tree and `--copies 32 --samples 6`.
  Node parent manifest `b8a919235d8c1012bac46ff1da0f46720cee97285f44d4fc2a3522a9af3ab3f2`.
- Full node, dependency and budget qualification:
  `/tmp/fpgs-franka-kinetic-nodes-jne3Tlsn/FINDINGS.md`,
  `af0da8d0613885a6ae10f65e9db29e71273b75fa36a6d35427cca4ff71ead6bc`.
  Evidence `5168d51499581ec03254391748317b221120753cc54e667fdeac379d4db414d6`.

The independent numeric/cost audits and exact retained free/generalized launch
recipe are linked by those local reports. All three paired parents completed
and were reaped. This checkpoint records a diagnosed first-mapping loss and
a separately costed next hypothesis, not a measured optimization gain.
