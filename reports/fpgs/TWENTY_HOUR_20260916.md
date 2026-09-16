# Twenty-hour structural optimization continuation

Requested window: 2026-09-16 11:04 UTC through 2026-09-17 07:04 UTC.
Target: at least 4x corrected MJWarp whole-physics throughput on RTX PRO6000
for Franka, KukaAllegro lift, Allegro reorientation, ANYmal-D flat, G1 rough
and SO101 Keyboard; retain paired GB300 measurements. Environment wall time
is reported separately and is not end-to-end RL training throughput.

## Fixed references and scope

- Worktree/branch: `newton-fpgs-structural-twentyh-20260916` /
  `ooctipus/fpgs-structural-twentyh-20260916`, on the `ooctipus/newton` fork.
- Starting handoff: `57e1dcfaded9283dc9f62951093324b1da5acd6b`.
- Accepted runtime comparator: `ca0d427af809571bb5501f644c1a6e03990cd2a8`.
  The previous study's experimental flags remain disabled. Its zero accepted
  additional gain and diagnosed failures are not reclassified as progress.
- Isaac Lab: `53ee6b44c2334341305dbdf385a3916c6b140799`, unchanged.
- Preserve task dt, substeps, maximum iteration allowances and calibrated
  capacities. Preserve numerical convergence and physical behavior, not bit
  identity. No dropped rows, contacts or unsupported fallback work omitted
  from costs. No sudo, parent-pointer update or original-worktree edits.
- Main agent alone launches GPU work, with one job per device. Source trees
  remain frozen during guarded timing runs. Reuse existing capture drivers.

## First checkpoint and working plan

1. Recover valid matched comparisons and current accepted flags before
   choosing a task. Exclude warning-inflated MJWarp runs and label mixed-source
   ratios. Refresh only missing/stale comparisons with shared collision
   improvements enabled symmetrically. Do not benchmark failed prototypes as
   the baseline. Allegro and Keyboard previously cleared 4x in qualified
   comparisons; protect those gains.
2. Select a complete work-retirement boundary with a substantial whole-step
   budget, including classification, production, conversion, fallback,
   synchronization and publication. Record the pre-code cost hypothesis.
3. Build the smallest integrated candidate. Use bounded physical controls
   before an early native cost screen, and reserve extensive qualification
   for a measured positive candidate. First implementation checkpoint within
   90 minutes; do not silently extend past two hours.
4. If prediction and timing disagree, diagnose the cause and allow one
   cause-directed correction. Do not repeat tile/threshold grids or reopen
   closed numerical methods without materially new evidence.
5. A promotion requires balanced repeated whole-physics timing and relevant
   physical/reset/cross-task checks. Report measured gain, diagnosed loss and
   unimplemented hypotheses separately. A component gain is not the target.

Initial parallel review: corrected per-task gaps and profiles; collision
work/capacity scaling; and contact/force/solve/state representation lifetimes.
No runtime candidate is selected merely because the prior tree-scan sketch
reduces operation or synchronization counts.

## First funded structural experiment: packed tree traversal

Selected at 11:22 UTC; first native-ready checkpoint 12:15,
no silent extension beyond 12:45. This is a new topology-derived parallel
chain traversal, not the previous serial-chain/2-jet sketch or paired-world
GS remapping. Default-off `FEATHER_PGS_G1_CHAIN_SCAN=1` changes only the
already admitted G1 current/next dynamics owner's tree evaluation.

Pack twelve heavy chains into 60 of the existing 64 CTA lanes, with no chain
crossing a warp. Compute local chain prefixes through literal full-warp
shuffles and resolve two light-edge levels in shared memory. Keep pose and
the original common-frame motion/bias scans separate: no additional spatial
adjoint transforms. Use additive reverse suffixes for subtree wrenches and
moments in both state production and live external-force prediction. No
prefix subtraction, approximate inertia, extra global intermediate, or
change to the held factor/contact/GS/public-state contracts.

Pre-code accounting: pose and motion each perform 109 combines rather than
124; their internal logical shared accesses fall from 6536 to 2035 scalars.
Tree CTA joins fall from 27 to 9, keeping other joins (finish total 36 to18).
The countervailing cost is important: reverse sums perform 87 rather than43
adds per component, plus new shuffle instructions and mapping reads. Current
finish/repair/predict total about3.147 ms RTX; removing10% of complete physics
would require nearly halving that entire family. Counts alone do not establish
that outcome, and the full4x target requires additional owners even if it wins.

Reuse the original physical tests and native benchmark. First screen includes
the complete live owner, unchanged collision and retained factor/row/solver
work. If slower, attribute actual execution/resources and allow one correction
only when supported by that diagnosis. No tile/mask tuning grid.

Separately, a one-round fresh Franka comparison is running on accepted ca0d,
same source for both backends, original calibrated capacities and corrected
MJWarp, with `NEWTON_NARROW_PHASE_THREADS_X=4` shared symmetrically. It is a
baseline refresh, not an optimization result. Raw parent:
`/tmp/fpgs-franka-symmetric-current-backends-paired16k-20260916-01`.

## First matched baseline refresh and native checkpoint

The three one-round comparisons below use accepted ca0d on both backends,
fixed corrected Lab, 16K worlds, seed0, 200 warm steps and40 wall/physics
samples. Shared narrow-phase mapping is4 for both backends. All12 children
exit0, all24 boundary checks pass and MJ warning masks are0. These are new
baseline measurements, not gains from the present experiment or repeated
performance qualification. Environment wall time is separate, not RL training.

| Task | RTX FPGS / MJ ms | Ratio | GB FPGS / MJ ms | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Franka |5.118216 /10.584648|2.0680x|4.686182 /10.365698|2.2120x|
| KukaAllegro |10.520933 /27.460161|2.6101x|10.665790 /25.755104|2.4147x|
| ANYmal-D |9.243896 /36.874710|3.9891x|9.490690 /42.296333|4.4566x|

Their manifests are `/tmp/fpgs-{franka,kuka,anymald}-symmetric-current-backends-paired16k-20260916-01/manifest.json`.
SHA256 respectively:
`904f0c4db94c4dde2d7671f5a318bb6eed8fcacb5af327b056e2a751cda06c28`,
`a8d934312da1adf8a6bc2b0ba211877bd6c1630103339332f7ae05b724f59237`,
`620cd3ed1f20015b723c486e0f9f54e17834513a1bf8aac150e16725ec025546`.

Chain runtime froze at11:36, with9 CPU controls passing. All3 original native
G1 physical/lifecycle selectors then pass on each GPU, clean exit0:
41.736 s RTX /44.307 s GB including loading/compilation, not physics timing.
Actual graph keys include all three `_chain` factories. Existing16 saved
current/held states, anchored current-force response, five-world mixed resets,
mass epochs and graph contact sequence0/5/0/5 retain their original gates.
The inherited target-layout deprecation is the only warning. Logs:
`/tmp/fpgs-g1-chain-native-20260916-bRBEaurw/gpu{0,1}.log`.
Independent initial source review found no blocker; resource review continues
without delaying the early integrated whole-physics cost screen.

## First integrated screen and loss diagnosis

The first paired whole screen completes with all children and guards passing:
RTX15.597073 ->15.905649 ms (0.98060x), GB20.413217 ->20.954826 ms
(0.97415x). This is a loss, not a promoted optimization. Environment wall is
29.859002 ->30.723191 ms RTX and35.311652 ->34.932367 ms GB. Manifest:
`/tmp/fpgs-g1-chain-whole-paired16k-20260916-01/manifest.json`.

Source-matched CUDA-hidden compilation of original and chain repair, finish
and predictor passes all12 SM120/SM100 entries. Registers original->chain:
RTX80->72,91->81,48->40; GB95->72,95->82,40->40. Shared memory is unchanged
(8000/8000/1660 bytes), no spills. Thus register pressure or spilling does
not explain the timing loss. Static instruction/shared-load/barrier counts
are not measured dynamic traffic or stalls. AOT report:
`/tmp/fpgs-g1-chain-offline-haLDAvku/offline01/report.json`, SHA256
`e1f04e014f887f6c1ad34afca04b109477a11921d8ec0b5dd4c5e1aa6c05f563`.

The reverse reducer serializes its6/19 components through the shuffle chain;
PTX also reloads light-child metadata and stage widths for each component.
The original reducer exposes independent component operations. This is a
concrete possible countercost, in addition to the higher additive work count;
kernel attribution is required before funding a correction. A single
round-major vector reduction is the proposed causal correction, not a mapping
or register sweep. It is not yet implemented or measured at this checkpoint.

Both node-capture parents encounter the previously documented auxiliary-graph
analyzer rejection after producing complete captures. Their nonzero parent
exits are retained. Baseline and candidate are captured as first arms under
`/tmp/fpgs-g1-chain-node-paired16k-20260916-01` and
`/tmp/fpgs-g1-chain-node-candidate16k-20260916-01`. This diagnostic uses12
environment steps, not the old reader's3; the strict reader is mechanically
normalized to12 and requires48 physics/12 auxiliary roots. No timing or
physical gate is relaxed. Whole throughput above remains the valid separate
event-level measurement.

Two further read-only screens do not justify runtime experiments yet:
conservative terrain hierarchy cannot retire enough current query work, and
exact Franka mimic condensation leaves the full state family intact while
requiring projected responses across every loaded/fallback path. Neither is
counted as an optimization result. Historical pair-shape preparation is an
existing opt-in feature (partial public-AABB validity contract), not new work.
