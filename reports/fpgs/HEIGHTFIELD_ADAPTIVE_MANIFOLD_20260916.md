# Adaptive four-witness heightfield manifold (default off)

## Frozen experiment policy

Base: qualified chain `ef9481bcc7b17fc9496f28a206ab6d0c72331507`.
Flag: `NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD=1` (default zero).
This is a new finite contact discretization, **not** an equivalent reduction of
the previous pointwise Coulomb system. No timestep, substep, solver iteration,
capacity, material, witness normal, or friction-allocation policy is changed.

Replace the existing final global-reducer export, not its geometry producer.
For supported ordinary heightfield/box or heightfield/convex pairs, gather the
old surviving hash winners after the original per-key roundoff suppression.
Apply the stock writer's admission before selection. Select up to four distinct
admissible IDs: deepest, farthest from it, maximum projected triangle area,
then maximum added two-triangle area. Break score ties by contact ID. Unlike
MJWarp's mixed-unit 0.001 cutoffs, do not terminate at zero score: coincident
points with different normals can represent real edge constraints. Every
selected ID retains its original geometry, depth, fingerprint, and material.
Unsupported pair types export the complete original surviving pool; custom
writers, speculative, deterministic and hydroelastic modes keep the old export kernel.
Deterministic mode is excluded because allocator-ID spread ties are not a
substitute for its schedule-independent fingerprint ordering.
The existing matching, optional sorting, and body-pair postreduction remain.

The observer-only coherent full-16K census in
`/tmp/fpgs-adaptive-witness-population-90c5BsvH/population.json` found 24,868
RTX heightfield/convex pairs and 78,284 normal contacts; 5,553 pairs exceed four.
A four cap could remove 14,666 normals (18.7%), before any qualification or
performance claim. Only about 300 normals are eligible for the earlier nearly
coplanar-only restriction. 14,919 pairs span multiple raw runs: unchanged
friction-anchor-limit-two behavior can therefore change tangential row counts
when this exporter changes contiguity. This is charged and tested, not called
an allocator bug fix.

## Complete cost / gates

The replacement charges read-only hash probes for all 36 normal/voxel/predictive
keys, old per-key suppression, shared candidate staging, four selection passes,
writer admission and output, plus complete unsupported-pair fallback. There is
no second original export, new global sort, global geometry array, or new query.
No speedup is inferred from contact counts. Root's minimum useful target is
one millisecond whole G1 RTX saving, standalone or combined with shell support.

Regression-first controls cover the missing feature, all-original fallback,
cross-key IDs, rejected winners, zero-area/different-normal witnesses, duplicate
pair routing, and reset/reuse. Loaded controls reuse the existing rebound,
support, tilted-foot, sliding, and border fixtures, with explicit production
friction configuration; yaw and step geometry exercise torsion and mixed
normals. CUDA and whole-task timing are root-owned.

## Result: standalone not promoted

The single paired 16,384-world whole-task screen passed its source, ownership,
capacity and idle guards, but did not meet the one-millisecond RTX milestone:

| Whole environment step | Original | Adaptive four | Saving | Speedup |
| --- | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | 15.032385825 ms | 14.816138250 ms | 0.216247575 ms | 1.014595x |
| GB300 | 19.827471325 ms | 19.789886175 ms | 0.037585150 ms | 1.001899x |

This is one unpromoted screen, not repeated performance acceptance. The runtime
was unchanged across initial qualification, the fixture correction, and timing.
Artifact: `/tmp/fpgs-g1-heightfield-adaptive-whole-paired16k-20260916-01`;
manifest SHA256 `635a68016f2247068ab375158adf66e7abcda8562969678fc0b2e54970c61dca`.

Strict node attribution found real downstream work reduction, largely offset
by the new exporter. Numbers below are profiled owner sums per environment
step, not standalone throughput predictions or hardware-counter measurements:

| Profiled ownership | RTX | GB300 |
| --- | ---: | ---: |
| Original export | 0.302094 ms | 0.323757 ms |
| Adaptive export | 0.979333 ms | 1.526946 ms |
| Added export cost | +0.677240 ms | +1.203189 ms |
| Row-family saving | 0.246474 ms | 0.231637 ms |
| Original GS saving | 0.601046 ms | 0.662541 ms |
| Combined row/GS saving | 0.847520 ms | 0.894178 ms |

The new exporter runs exactly four times per environment step; the old exporter
is absent. Actual launch is block32/grid24576, 64 registers, 128 bytes static
plus 1,616 bytes dynamic shared memory, and zero reported local memory. Hidden
SM120/103 compilation likewise reports no stack or spills. These facts identify
an expensive ownership boundary, not a proven stall mechanism. They do not
license a tile/grid sweep or infer a new whole gain by adding component savings.

Supplemental strict reader preserves the original parent analyzer rejection,
then proves all 48 physics roots and 12 auxiliary roots, source/correlation
guards, actual owner activation, and zero unproven nodes:

- `/tmp/fpgs-adaptive-manifold-strict-kolTndj0/read_adaptive.py`, SHA256
  `7b9c4be33c0a9005bed56b7b5927a8256f23889f21d97a99bc491b28288130c5`.
- `candidate_gpu0.json`, SHA256
  `bb1cd746acde0ce992ea9aca9a18134f0a422260fdbf23c314e5a4116eebcf30`.
- `candidate_gpu1.json`, SHA256
  `06b32221fc1aeda816f8c92cea906f8d3d20506a96137b19dea4e0cbdd428d8a`.

## Initial native failures and causal fixture correction

Initial logs remain at `/tmp/fpgs-adaptive-manifold-native-20260916-Q2OiYBL4/gpu{0,1}.log`.
Focused pool controls passed on both cards. The loaded suite did **not** pass:
RTX support failed in both arms; GB support failed in the candidate; sliding
failed in both arms on both cards; step support failed the original settling
gate and the candidate's sampled-force gate. No capacity, nonfinite or momentum
failure was observed. These records have not been erased or relabeled as passes.

Two fixture premises required correction, with no runtime/allocator change:

1. The original support gate averaged only five instantaneous forces, separated
   by roughly 20 steps. RTX original's last sample was 24.9073 N; candidate's
   step221 sample was 19.7961 N, while candidate final force was 22.07054 N versus
   22.0725 N weight. Such samples are not the time-average support impulse. The
   corrected observer accumulates every existing public-force call over the
   fixed final 80 steps, applies the **same 2% mean-weight tolerance**, and checks
   the full impulse against actual two-body momentum with the unchanged 2e-4
   tolerance. It retains all old sampled failures, force extrema, and peak spin.
   Original final speed/spin/penetration gates remain unchanged. Deliberately
   wrong mean and momentum data still fail the CPU oracle.
2. The inherited sliding formula `v0^2 / (2*mu*g)` assumes all normal load has
   full material friction. Production anchor-limit two does not implement that
   premise: `kernels.py` halves the selected contacts' coefficient, and each
   disk uses its **own parent's** normal impulse while other normals have no
   tangent rows. This existing documented approximation can under-represent
   full point-Coulomb or pooled-patch friction; it is not an adaptive-exporter
   bug fix. Neither simply renaming mu nor relaxing stopping tolerances is
   justified. The production case now checks its actual own-parent disks,
   nonnegative normals and momentum, retaining the old analytic failure as a
   diagnostic. A separate anchor-limit-zero sliding control retains the exact
   original stopping-distance/speed gate in both arms.

Step support was authored by raising half the heightfield after choosing the
flat-floor initial pose. Its initial penetration is 8.32 mm. It is explicitly
a penetration-recovery/mixed-normal stress, not initially equilibrated support.
Its old final-settling failure remains hard; no initial pose, duration or settle
threshold was changed. Candidate tail spin still reaches 0.23334 rad/s and flat
support reaches 0.11208 rad/s transiently, even though final settling and complete
interval support pass. A mean-force pass is not a claim of zero rocking.

## Corrected bounded physical scope

Corrected native logs:
`/tmp/fpgs-adaptive-manifold-loaded-tail-20260916-iws6t0uD/gpu{0,1}.log`.
Both unittest processes still exit **1**, solely because the **original**
`step_support` does not settle. All nine candidate records per card pass the
applicable bounded gates; this is not an all-suite PASS claim.

- Candidate full-80-step relative mean-weight error: flat support
  `0.0001606794726`; step support `0.00000928185213` (same on both cards).
- Candidate maximum cone excess `3.7252903e-9`, negative normal impulse zero,
  maximum per-step scaled momentum error `9.3138279e-7`.
- Full-friction analytic sliding passes both arms/cards: candidate distance
  `0.5071898103 m`, final speed `1.5735764e-6 m/s`, versus the authored
  `0.5096839959 m` analytical distance and unchanged 0.05 m/s / 0.05 m gates.
- Production-anchor sliding still travels beyond 1.27 m; passing its actual
  cone/momentum checks does not claim the full-mu analytical motion.
- Restitution/rebound, tilted support, finite border and yaw recovery remain
  part of the bounded control set. No independent high-iteration reference for
  each new manifold, long-horizon task qualification or universal support/wrench
  equivalence was established.

Corrected log SHA256s: RTX
`7af146782157688608116a11b5abe1ab3e8b61aad8343e2c8938ce065c64fd31`;
GB `2eaa520d990ceb25e2b2536a83c7e40a988826df68763da1edc24c772d879c8a`.

## Source and verification record

The missing-module regression failed before implementation (session16181,
`ModuleNotFoundError`). Final focused CPU run62754: five PASS, two CUDA skips;
the oracle controls reject wrong support averages/momentum and out-of-cone
impulses. Initial native pool controls pass both cards. Hidden compilation
report: `/tmp/fpgs-adaptive-manifold-offline-nWtkYD/offline01/report.json`, SHA256
`27ffe21ffa42d15ba4cbd3a224fe67e881366f2e56d1d33aaad40a54dfff466a`.
Full `uvx pre-commit run -a` and the six-file targeted pre-commit run pass;
measured runtime and observer SHA256s were rechecked afterward.

Measured source pins (unchanged by the fixture repair):

- `newton/_src/geometry/heightfield_manifold.py`:
  `68ec8bbeb33fe1b25c58b8d819b10cbf1f2d2b988a8ff703d2319904f2045581`.
- `newton/_src/geometry/narrow_phase.py`:
  `c1d8e37be9825c0ba24841b74948994bd59f5594e8663b33f24c79b4404a2950`.
- Corrected `tools/fpgs_bench/test_heightfield_adaptive_manifold.py`:
  `f4de6443ab4ff044008d949c5cd34b8ed1b6983fbe7c9315b24da042c56aa261`.
- Root-owned existing capture observer `chain_capture_20260916/checked_sparse.py`:
  `7188ce1002fd201187ad8d1195112d54abfdb30d433c2250f0093167fa7219a8`.
- Root-owned existing capture parent `chain_capture_20260916/run.py`:
  `eca46ac3777f670d5ef92780d6b4eb4995c773038da3d5846d08c3f884e690ec`.

The observer requires explicit adaptive flags on both arms and proves exact
selected exporter ownership while retaining the original source/capacity/root
checks. No new benchmark framework was introduced. Standalone remains default
off and unpromoted. A direct query-owned representation that eliminates the
hash round trip is a separate unimplemented structural question, not a claimed
gain of this prototype.
