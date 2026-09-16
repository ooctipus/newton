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

## One cause-directed correction: first positive whole screen

Strict12-step attribution isolates the initial loss to finish:
2.329925 ->2.644243 ms RTX,2.500325 ->3.086221 ms GB. Three-owner exclusive
cost rises3.132194 ->3.423382 and3.302338 ->3.834197 ms. Predictor slightly
improves; there is no added owner or mass-refresh cadence change. All four
strict audits pass, preserved under `/tmp/fpgs-g1-chain-strict-nDEKxWay`.

The single correction changes only the reverse reduction to fixed6/19-channel,
round-major vectors. Per-channel addition order, chains, full-warp masks,
state ownership and physical rules remain unchanged. It removes repeated
metadata walks and exposes independent channel operations. Runtime helper SHA
`1d45c38899df64991d666e7fe72993d9f8039d733320e3f323c71d9938d12073`.
CPU4 controls and targeted precommit pass; all6 offline CUDA entries compile
without spills. Finish registers91/92 and predictor56/54, shared unchanged.
The predictor's increased register footprint is charged, not omitted.

The original3 CUDA physical/lifecycle selectors pass on both GPUs, exit0,
7.406/8.131 seconds including loading; no changed physical tolerance.
Logs: `/tmp/fpgs-g1-chain-vector-native-20260916-32hIV2qh/gpu{0,1}.log`.

First whole screen, one round, all children/guards pass:

| Whole physics ms/env step | RTX | GB300 |
| --- | ---: | ---: |
| Accepted original |15.626156|20.496853|
| Chain vector reduction |15.022117|19.833487|
| Original/candidate |1.040210x|1.033447x|
| Time removed |0.604039|0.663367|

The actual environment wall baseline/candidate values are
30.397366/28.255761 ms RTX and34.169252/33.665474 ms GB; not RL training.
Evidence: `/tmp/fpgs-g1-chain-vector-whole-paired16k-20260916-01`.
This is a discovery gain, not repeated qualification or the4x target.
No further mapping, register or arithmetic micro-tuning is funded for this
candidate. Check repeatability and move to a larger work-retirement boundary.

### Three-round repeat and second funded boundary

Three alternating paired rounds complete with all12 children clean and all
24 boundaries passing. Median physics15.672744 ->15.091197 ms RTX (1.038535x),
20.452424 ->19.848266 ms GB (1.030439x). Candidate samples:
RTX[15.091197,15.086214,15.095232], GB[19.848266,19.842375,19.883883] ms.
Median environment wall29.609188 ->29.377521 ms RTX,34.177377 ->34.317887 GB.
This confirms a modest physics gain, not a substantial environment-wall gain.
Evidence: `/tmp/fpgs-g1-chain-vector-repeat-paired16k-20260916-01`.
The explicit G1 recipe may retain CHAIN_SCAN=1; unrelated task owners remain
unchanged. No further traversal tuning is planned.

Second boundary funded at12:14 UTC: model-compiled coordinate-to-current-state
production. First native-ready checkpoint13:04, no silent extension beyond
13:34. Reusable exact per-model FIXED/scalar joint descriptors replace repeated
local transform and motion-axis reconstruction. Joint-owned qdd conversion,
original free-root transport, integration and local pose formation carry scalar
coordinates in registers, publishing all original qdd/q/qd outputs. Keep the
old scalar integration arithmetic, all root COM/normalization/damping laws,
kinematic DOF v_out side effects and kinematic joint copy behavior. One join
before pose scan replaces the separate preparatory joins; tree traversal is
the unchanged now-measured vector-chain implementation.

Relative scalar quaternions use precomputed A*cos(q/2)+B*sin(q/2); prismatic
translation is constant-plus-linear. Motion axes come from the already computed
child pose and constant child-local screw, not a reconstructed parent anchor.
Use the live body's COM point once for physical and public state. The original
solver builds body_X_com=(body_com,identity rotation), so its repeated rotation
product is unnecessary; no approximate inertia or principal-axis change.

Descriptor constants are exactly deduplicated, with a per-joint index map
(approximately2.75 MiB for16K G1) and about80 bytes per unique descriptor;
worst-case unique count is bounded by the existing joint count. Charge the
added indexed reads, constructor cost and register liveness. G1's existing
static axis/anchor proof protects these constants; mass/COM/inertia remain
live. General/free cases preserve original laws. No new per-step launch or
world-state intermediate. Keep current bias/geometric434, held W, all contact,
solve, sensor and public-state contracts.

Source retirement includes redundant quaternion/point transformations and
qdd/qnext reloads, not just type dispatch. It does not establish that half the
state family or10% of whole physics will disappear. Reuse original CPU/native
physical controls plus focused general-child-frame/kinematic algebra tests,
then an early complete whole screen. Do not extend into a math/mapping sweep.

### Vector-chain attribution and coordinate candidate readiness

The vector candidate's separate strict node accounting passes all48 physics
and12 auxiliary roots across both cards. The legacy parent retains exit1 for
its known auxiliary-graph analyzer limitation; this is diagnostic evidence,
not a replacement throughput run. Artifacts:
`/tmp/fpgs-g1-chain-vector-node-candidate16k-20260916-01` and
`/tmp/fpgs-g1-chain-vector-strict-Os6BOg4H/candidate_gpu{0,1}.json`.

| Owner ms/env step | RTX | GB300 |
| --- | ---: | ---: |
| Complete finish |1.842625|1.930120|
| Predictor |0.560355|0.581146|
| Repair |0.181208|0.198781|
| Three-owner exclusive union |2.567020|2.688023|

These replace the older2.33/2.50ms finish estimates when costing further work.
Removing1ms now requires eliminating about54% of RTX finish time, not43%.
Finish still owes the original integration, physical state, bias/geometric
cache and public outputs; its whole duration is not disposable work.

Compiled-coordinate first source freeze12:27 UTC, before the13:04 checkpoint.
Focused CPU and original controls12/12 pass. The descriptor algebra admits
ordinary FP32 unit-normalization roundoff but falls back to original algebra
for nonrigid/nonfinite constants; it never normalizes authored input. Native
testing and complete timing are still pending. Hidden-CUDA AOT actual repair/
finish registers are64/76 onSM120 and64/82 onSM100, versus vector-chain72/91
and72/92. Shared8000B, stack72B, no spills. This is static resource evidence,
not achieved occupancy or speed. AOT report:
`/tmp/fpgs-g1-chain-offline-haLDAvku/offline03/report.json`, SHA256
`3f5a8335a0d3eaa91e4e91c5aec507117c59d7246d4ab25dd06eee4243a8c8b8`.

### Fresh ca0d backend refresh, continued

Same accepted ca0d source on both backends, corrected MJWarp and unchanged
Lab/budgets, one paired discovery round. Shared G1 terrain changes apply on
both sides; Allegro MJWarp keeps its native contact pipeline. These are fresh
baseline ratios, not new optimization gains.

| Task | RTX FPGS ms | RTX MJWarp ms | MJ/FPGS | GB FPGS ms | GB MJWarp ms | MJ/FPGS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G1 rough |15.610290|37.959959|2.431727x|20.440196|38.258522|1.871730x|
| Allegro reorient |15.146122|63.862091|4.216399x|16.531282|81.857731|4.951687x|

Artifacts `/tmp/fpgs-ca0d-refresh-g1-JZ51Sjhs-01` and
`/tmp/fpgs-ca0d-refresh-allegro-JZ51Sjhs-01`; both parents exit0. This G1 row
does not include the newer chain traversal. Do not present a cross-run ratio
using its MJ denominator and a different FPGS capture as a paired comparison.

Keyboard4096 refresh also completes exit0: FPGS/MJ physics6.960440/27.851542ms
RTX (4.001405x),6.054350/28.557772ms GB (4.716902x). Environment wall is
27.455639/31.150523ms RTX and24.369083/32.006444ms GB. Artifact:
`/tmp/fpgs-ca0d-refresh-keyboard-so101-JZ51Sjhs-01`. RTX is only just above4x
in this discovery round, not evidence of comfortable margin.

Compiled-coordinate native controls now pass3/3 on both devices, exit0:
41.372/45.778s including module loading/compilation. Existing complete graph,
reset/refresh, current-force, held-factor and physical saved-row controls
use their unchanged tolerances. Actual repair/finish keys end `_chain_compiled`;
predictor remains `_chain`. Logs:
`/tmp/fpgs-g1-compiled-native-20260916-KhBYwmTS/gpu{0,1}.log`.
The first whole screen compares qualified chain ef9481bc with this candidate,
CHAIN_SCAN=1 on both, COMPILED_COORDINATE_STATE=0/1, not against a losing
prototype. Source is frozen through completion of that screen.

### Close coordinate-only replacement, 12:44 UTC

First paired complete discovery passes source/capacity/finite/idle guards:

| Whole physics ms/env step | RTX | GB300 |
| --- | ---: | ---: |
| Qualified vector chain |15.099226|19.818439|
| Compiled coordinates |14.937630|19.838525|
| Baseline/candidate |1.010818x|0.998988x|

Environment wall27.878866 ->28.674384ms RTX and33.803177 ->34.436833ms GB.
Artifact `/tmp/fpgs-g1-compiled-coordinate-whole-paired16k-20260916-01`.
This is not a substantial gain and is not promoted into the retained recipe.

One node diagnosis explains the limited effect: finish1.842625 ->1.709921ms
RTX (1.077609x),1.930120 ->1.925291ms GB; repair0.181208 ->0.175995ms RTX,
0.198781 ->0.179859ms GB. Predictor, factor, rows, GS and collision remain
unchanged. The lower-register candidate really retires its three preparatory
joins and scalar rereads, but leaves15 executed full-finish joins, complete
pose/motion/subtree scans, current inertia/bias/publication, and on refresh
the19-channel moments and434 six-term projections. There is no discovered
duplicate producer, cadence error or missed retirement to correct.

At least1,135 FP32 public/cache values per world still have to be emitted,
plus434 geometric values on refresh. This is a source output count, not a
DRAM-traffic or memory-bound claim. Descriptor indirection and the shared
scalar-speed transfer are added work. Neither static register reductions
nor deleted arithmetic imply a large whole-step saving.

Node capture `/tmp/fpgs-g1-compiled-coordinate-node-candidate16k-20260916-01`
retains legacy parent exit1 for its auxiliary analyzer. The exact-key extension
of the existing strict reader passes48 physics/12 auxiliary roots on both:
`/tmp/fpgs-g1-compiled-strict-lb18MR06/read_compiled.py` and
`candidate_gpu{0,1}.json`. The reader adds only exact compiled-owner aliases
and requires actual compiled owner selection, retaining all old checks.

Close the candidate default-off, preserve source/tests/evidence, and do not
start a mapping/algebra sweep or port it to other tasks on this result.
The retained runtime choice remains vector chain with compiled coordinates0.
Franka and Kuka coordinate-only port screens also lack a substantial budget:
affected finish+repair families are approximately1.102/1.920ms; a10% whole
saving would require removing about46%/55% of those complete families while
their scans, inertial work and public outputs remain. No port is funded.
