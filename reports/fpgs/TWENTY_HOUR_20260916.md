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

### Branch-local nonlinear iteration: bounded CPU closure

The distinct branch-Jacobi study uses one original metric sweep per structural
branch per global pass, with foreign root response frozen, then merges physical
response deltas. Same maximum8 passes, all rows and original circular-friction
law. Branches come from non-root held-factor connectivity, not body names;
shared waist DOFs stay together. A second prescribed control scales only the
shared six-coordinate response by the number of nonempty branch blocks,
retaining original full physical impulse application. No parameter grid.

Plain branch-Jacobi passes the existing16 selected component gates, but the
hardest7410 cases load only one branch. Original/plain residual dots1927/1931,
global passes78/79 and sliding roots69/69: there is no total-work retirement.
An ideal parallel branch critical path of901 dots is not a2x throughput claim
when16384 worlds already supply GPU parallelism. Of69 sliding roots,67 remain
on the serial branch. The split control misses2 component gates; held-H error
improves in one of them despite component misses, so these are not all labeled
global convergence regressions.

A controlled actual-USD two-foot fixture loads both feet. Baseline/plain/split
scaled held-H velocity errors are1.092e-6/1.790e-5/1.365e-3 at8 passes;
complementarity4.198e-6/5.985e-5/6.091e-3. Independent original512/1024
reference velocities are stable. Plain has a modest real finite-budget change;
root splitting slows this fixture substantially. A6x6 root-only correction
cannot freely remove nonlinear branch coupling while preserving momentum:
consistent branch impulse sensitivities/solves would be additional work.

Close without native funding. Frozen CPU evidence, exact scripts and source
pins: `/tmp/fpgs-g1-branch-split-cpu-QQECL6nh/README.md`, SHA256
`93fd234ec87eafbf662431ce4d162f6d03003464dc4c6654fb2721089545262d`.
This is numerical/work evidence, not GPU performance. The initial invalid
zero-diagonal fixture was preserved and excluded; the corrected fixture uses
independent physical J H^-1 J^T and original CFM.

### Fund one complete parallel-construction owner

Start13:10 UTC; native-ready/first-integrated checkpoint14:40, no silent
extension beyond15:10 without a concrete implementation cause. Default-off
Newton-only candidate on qualified vector-chain semantics; coordinate-only
experiment remains disabled. Primary milestone is2ms RTX whole-physics
saving, not a standalone contact-kernel gain.

Replace the complete current prediction/row-preparation/solve boundary with
raw-world indexing followed by four contact-worker warps per world and the
original metric GS on warp0. Current-state repair, held-factor refresh and
complete final-state production stay separate and unchanged. Use the existing
exact-capacity raw-ID bucket implementation, not a new routing framework.
All contact admission, raw-ID neighbor friction rules, material mixing,
speculative bias/restitution, limits, physical response and sensor mapping
remain covered. Row-slot order may differ; physical convergence must qualify.
No dropped constraints, smaller capacity, extra solver passes or stale forces.

This is distinct from PACKETS/SMALL_STEP: both form all contact triplets
serially inside one warp. PACKETS retains separate allocation/metadata/key/bias
services; SMALL_STEP additionally splits the heavy fallback cohort. The new
owner must retire those old producers, not retain their full work alongside it.
Do not fuse final state or count its necessary mathematics as removed work.

Charge raw indexing: three clears, count, exclusive scan, scatter and validation,
two endpoint-to-world walks per raw contact, count/scatter atomics, and
1,310,732 bytes of exact-capacity IDs/world arrays plus existing scan workspace.
New geometry-based admission/material work is required, not free. Old raw IDs
must remain the keys for neighboring-contact decisions; bucket neighbors are
not raw neighbors.

The critical adverse cost is retained128-thread CTA registers and approximately
10KB shared storage throughout warp0 GS, with three idle worker warps, versus
old32-thread/1116B GS. This may reduce independent resident solve warps and
erase construction gains. No occupancy or memory-bound claim is established.
The current exposed prediction+rows+GS boundary is approximately7.046ms RTX;
new owner PLUS indexing must fit about5.046ms for the2ms milestone. One early
complete paired screen decides, followed by one causal diagnosis/correction
if justified—not a block-size, arena-size or register sweep.

Routing review before integration: original allocation deliberately skips
static/static and cross-world raw contacts. The parallel owner will preserve
their canonical raw IDs and mark their metadata unused, counting these skips
separately when validating bucket coverage. It will not turn one benign gap
into a whole-cohort O(worlds * raw_count) fallback. Invalid descriptors remain
errors. These are local candidate bucket kernels; existing source-pinned
bucket consumers remain unchanged.

### Fund one world-lane complete-state prototype, 13:25 UTC

Separate worktree `/home/octi/Projects/newton-fpgs-world-lane-20260916`, fork
branch `ooctipus/fpgs-world-lane-20260916`, based on120c37fc. The G1 candidate
and its source-frozen timing can proceed independently. CPU/AOT/resource
checkpoint14:40 UTC; complete integrated-ready target16:30. Default-off,
initially admitted to the existing Franka topology, never task-name dispatch.

The existing retained control already packs two worlds per warp using16
lanes/world and half-warp synchronization. The proposed complete owner uses
one world per thread, model-generated typed forward/reverse traversal,
field-major private caches and unchanged canonical public outputs. It must
replace current-state repair, force/factor/prediction and finish together.
Physical parameters remain live; notification/source-bank/held cadence and
next-call drive coefficients retain their original contracts.

This is not the closed coordinate-only experiment. The primary tree's scan
currently executes29 pose and29 motion compositions; a generated traversal
needs10 of each. Its factor9 is serial in one of16 lanes and prediction uses
two lanes. A lower-only geometric cache needs44 distinct nonzero entries
plus one structural zero, rather than evaluating79 nonzero full-matrix
entries. No native or throughput result exists yet.

Use one liveness-aware DFS/unwind layout, not a block-size/layout sweep.
Separate held/refresh finish at the original cadence avoids charging refresh
register allocation to held calls. Root-fixed terms cannot affect generalized
outputs; ancestor moment blocks die as six-float inertia-action columns are
formed and projected. Preliminary register estimates are110–150 held,
215–255+ refresh and80–120 factor/predict, not compiled measurements.
Refresh spills and strided canonical AoS publication are explicit risks.

The minimum core public q/qd/qdd, body pose/velocity and S output is about
1440 bytes/world/finish, excluding free-body auxiliary outputs. It cannot be
removed or treated as a free transpose. The approximately1.676ms original
state family needs roughly2.25x repair/finish and4x factor/predict to save1ms;
even perfect removal would not alone make Franka4x faster than MJWarp.
Judge the complete original-versus-candidate physics path first, then one
causal correction if justified. No source-only arithmetic saving is promoted
as measured progress.

Parallel-world first runnable freeze approximately13:40 UTC, ahead of the
14:40 checkpoint. CPU4 focused controls pass, including raw gaps/cross-world
skips, malformed descriptors, legal row permutation and missing-owner fallback.
The original G1 lifecycle test skips only its internal v_hat comparison when
the complete owner is actually active: v_hat is now private, while independent
prediction/momentum and public state/reset/graph checks remain. Candidate
global v_hat is poisoned in focused native controls to expose hidden consumers.

The first complete entry compiles at128/120 registers per thread onSM120/100,
10428B shared, zero stack/spills. Four128-thread resident-resource costs remain
through the warp0-only GS phase; this strengthens the residency risk, not a
measured slowdown claim. Retained sparse Z/support/incident buffers stay
allocated for unsupported host fallback, but candidate Z/incident production
and consumption are private. All old preparation producers must disappear
from actual candidate graph accounting; allocation reduction is not claimed.

Paired original-law native controls launched after UUID idle and source-pin
checks. Logs `/tmp/fpgs-g1-parallel-native-20260916-RJMp4PPD/gpu{0,1}.log`.
No whole-physics gain has been measured for this candidate yet.

An additional terrain idea was screened against exact prior art, not recoded:
general mixed-height flat-cell run/rectangle coalescing already has a CPU-only
study at `/tmp/fpgs-g1-live-flat-runs-p1kkj4qj/RESULTS.md`. Its old, pre-current-
geometric-cull cohort reduces606886 candidate triangles to338007 retained query
items,44.30%; only49.18% are in complete horizontal cells. This is not the
closed22.8% whole-foot gate. Current finite/fallback queries cost1.388055/
0.823625ms RTX and buffered reduction0.201310ms. Uniformly applying that old
count fraction gives only about1.069ms gross across all three, BEFORE grouping,
new queries, seam handling and mutation invalidation; this is not a timing
prediction. Active-key clear/export remain. No credible>=1ms net case is yet
established, so no native implementation or new census is funded.

First native run completes three full-production selectors on both devices,
but saved-row diagonal comparisons fail seven cases. CPU diagnosis establishes
a fixture input mismatch: begin() recomputes shared body poses and candidate
screws while the control retained captured screws. For current world0, the
independent physical diagonal discrepancy7.619e-5 relative matches observed
7.572e-5; this is not evidence of a changed row law. Restore the independent
captured pose/S/origin inputs for BOTH row producers after cache initialization.
No diagonal or physical tolerance changes; full-production recomputation tests
remain unchanged apart from the already-declared retired v_hat comparison.

A separate real raw-overflow bug is fixed: preserve public contacts-capacity
latching and canonical invalid raw-slot clearing, not only the private error.
The exact missing-latch regression was reproduced on RTX by removing only
that atomic latch in an isolated source-generated test process: one expected
failure at the public contacts assertion, exit1. Fixed runtime preserves the
sticky flag across reset. Valid-frame arithmetic is unchanged.

Corrected GB native4/4 passes; RTX passes all metadata/momentum/cone/lifecycle/
overflow controls but misses two finite-eight natural-residual comparisons in
world7410 (current0.000360334 vs original0.000217287; held0.000187893 vs
0.000096506). The same saved cases improve on GB, consistent with changed
atomic row ordering. Do not call this qualification or relax the tolerances:
independent original-H high-iteration diagnosis is pending. Logs and the
separate failing-latch control are preserved in
`/tmp/fpgs-g1-parallel-native-fixed-20260916-eL2lqGml`.

The first complete timing screen will proceed as a provisional cost diagnosis
while that numerical question is investigated. It cannot promote the candidate
or establish the4x goal. Final guard-corrected hidden AOT is
`/tmp/fpgs-parallel-world-aot-9d_2gcsc/offline04/report.json`, SHA256
`8b7e880739c5f18f3e59b71ae880d70701d51771b56b8426b80b6ebefb5d6f1d`;
resources remain128/120 registers,10428B shared, zero stack/spills.

### First integrated parallel-owner fault and numerical diagnosis, 14:07 UTC

The first paired16K whole screen is a FAILED integration run, not a timing:
`/tmp/fpgs-g1-parallel-world-whole-paired16k-20260916-01`. Both baseline
children exit0; both candidate children fail with CUDA illegal memory access.
Parent43895 is reaped exit1. No candidate throughput is reported.

Exact32-endpoint capture and independent CPU original-H references explain
the two earlier component misses. Artifact
`/tmp/fpgs-parallel-row-order-ySECtv/rtx_reference.json`, SHA256
`ec71bc55842b08f9cc9d87176971b49dadc22a2af3016ac0a43f5166e2a34355`.
All32 high-iteration references converge (maximum physical score9.83e-12);
the two world7410 orderings reach the same reference within1.04e-11 held-H
distance. Native scaled H error improves0.1015515 to0.1011291 current and
0.1526113 to0.1522775 held. Both eight-sweep endpoints remain substantially
underconverged; no universal convergence claim. Independently reconstructed
physical J rows are equal under permutation; original FP64 eight-sweep
recurrences reproduce each ordering's native endpoint within8.08e-6 H.
Replace unnecessary component-by-component monotonicity with the original-law
reproduction and common converged-reference accuracy gates, retaining absolute
momentum/cone checks and printing component diagnostics. Do not tune tolerances
to these misses or erase the original failing logs.

A first memcheck with zero warmup did not exercise physics: the observer
correctly rejected uninitialized private state. Reaped89369 exit1; its zero
memory errors are NOT clearance. Actual16K warmup2 memcheck13557 exits86 with
five out-of-bounds four-byte writes in the complete owner, preserved at
`/tmp/fpgs-parallel-live-memcheck-warm-20260916-AVtXypam/run.log`.
Root localization: row_w is intentionally a one-element dummy when contact
regularization is disabled, which is required by this candidate's admission.
The new native row producer writes it unconditionally at per-row offsets.
Remove these unused writes and add the production-shaped dummy regression;
do not allocate a full buffer to conceal the ownership error. Re-run the
actual integrated memory check before interpreting whole timings.

The isolated world-lane complete Franka owner is native-ready at14:01 UTC,
ahead of its checkpoint. Five CPU tests pass. Actual AOT reports SM120/100
repair255/255 registers with8/16B spill stores/loads, held finish167/168,
refresh255/255 and fused factor/predict152/148, all other spill counts zero.
No block-size or numerical sweep. RTX two native selectors pass, including
saved physical references and actual five-world loaded/reset/notification/
two-bank graph replay. Maximum physical H error1.02e-6; matched p16 body
velocity error5.22e-5. GB native check is pending; no speedup claim yet.

The exact two row_w stores are removed in parallel_world_rows.py SHA256
`98e1f79c7c88f7e56edd19bdb8e9d6c3659e3d17b6af545a7671a6b208e01834`.
No allocation, law or ABI change. CPU6 tests pass, including the source
regression that fails before this removal. Native production-shaped dummy
weight storage remains(1,1), with a poison-unchanged assertion. The frozen
reference-based helper passes all16 preserved RTX cases in0.915s; maximum
scaled H-error increase is4.9007e-8. This is not a new GPU run.

Corrected actual16K GB memcheck26152 exits0 with zero errors and both checked
capacity/private-owner boundaries passing:
`/tmp/fpgs-parallel-live-memcheck-rowwfix-20260916-30gpbWZx`.
The instrumented timings are not performance evidence.

First complete clean parallel-world cost screen, root95571 exit0:
`/tmp/fpgs-g1-parallel-world-whole-paired16k-20260916-02/manifest.json`.
All four children and eight boundaries pass; final source/idle guard passes.
RTX qualified chain15.028502 -> complete owner21.378284ms (0.702980x).
GB19.838911 ->26.804659ms (0.740129x). This is a substantial loss, not
accepted progress. Actual solve counters advance1600 to2240, matching the
640 expected measured substeps; routing sees about78K raw contacts at the
boundaries, not world_count times the raw allocation. Node attribution is
still required to establish retired calls and locate the extra6.35/6.97ms.

One possible causal correction is documented but not yet funded: preserve
four-warp parallel prediction/row construction, publish existing Z/incident/
v_hat and cold impulses, then launch the unchanged32-thread GS. It restores
the original solve residency at the price of one extra launch and up to
127.69MiB additional stores/substep at maximum row occupancy, plus cache-
dependent reads. No new buffers; old fallback storage already exists. It
must not reinstate old allocation/bias/row/predictor producers. Producer
error must veto the separate solve through its original status guard.
This is one boundary correction, not a block/register sweep. Actual timing
must justify it before implementation.

Franka world-lane GB native2/2 also passes. Its first paired16K whole screen
starts14:17 UTC against exact ca0d p16, unchanged flags/capacities/budgets,
with a minimal untimed owner-key/layout observer. No new performance claim
exists until that complete screen finishes.

### Causal cost diagnosis and one split correction, 14:28 UTC

The G1 node capture preserves the original auxiliary-graph analyzer failure;
the existing strict12-step reader separately verifies48 physics and12 auxiliary
roots, every process/correlation, both checked boundaries, and final source/idle
guards. No unproven nodes. Exact extension:
`/tmp/fpgs-g1-parallel-world-strict-jMoAcF1R/read_parallel.py`, SHA256
`031b5ff7536cb7f8db7bafac219794f2940907714642c12e476e614760afc612`.

RTX complete owner13.221249ms plus CSR0.175301; GB14.363626 plus0.188515.
The replaced family grows7.046171 ->13.396550ms RTX and7.566958 ->14.552141GB
against the existing qualified-chain node reference, independently explaining
the whole loss. Retained factor, collision and finish stay flat. All old
prediction/row/GS producers have zero calls. Memory activity decreases rather
than duplicating:32 memcpy nodes/90.18MB per environment step disappear.
Observed owner resources:126/120 registers,128 threads,10428B shared and zero
reported local memory. This does not measure achieved occupancy or counters.

Independent generated-source comparison establishes identical GS control flow
after private-storage substitutions: no extra sweeps,16-probe roots, row visits
or fallback rules. Legal contact reordering and different incident+rhs floating
association can change observed iteration work; actual root/probe counts are
not measured. The new CSR cost is small. The concrete adverse boundary keeps
four warps' resources resident throughout a warp0-only solve and serializes
each world's parallel contact completion before that solve.

Fund the one planned split correction with native-ready checkpoint15:20 UTC.
Keep the complete experiment reproducible as split=0; add nested default-off
FEATHER_PGS_PARALLEL_WORLD_SPLIT=1. Parallel construction publishes existing
interfaces, followed by the unchanged32-thread metric solver. Retain the
complete cost budget and numerical/capacity gates. No tuning grid. A missing
producer-factory CPU regression fails before implementation.

Franka world-lane whole01 is not a performance comparison: RTX ca0d baseline
aborts with malloc_consolidate during scene cloning before solver creation;
GB baseline passes and no candidate launches. Original failed artifact is
preserved. The unchanged retry whole02 completes with all source/idle/capacity
guards passing. RTX5.020306 ->5.166324ms (0.971737x); GB4.693110 ->4.834694ms
(0.970715x). This is a small loss, not accepted progress. A paired node capture
will identify whether forward/reverse work reduction was erased by factor,
publication, or execution costs before proposing any corrective implementation.

Franka paired node23346 exits0; the strict original interval reader verifies
all four captures (48 physics roots each, no auxiliary roots). Complete
state-family exclusive RTX1.671307 ->1.774082ms; GB1.400978 ->1.540940ms.
Factor/prediction saves about0.18ms, while publication adds about0.29ms.
All old factor/predictor calls retire and held/refresh cadence is correct.
Held and refresh publication cost0.592005/0.613865ms RTX despite167/255
registers, so refresh-register reduction alone is not a supported diagnosis.
Source exposes serial per-world arithmetic and world-strided canonical/model
AoS accesses, but no bandwidth/stall claim is established. The driver reports
RmProfilingAdminOnly=1; no sudo or permission change is attempted.

Artifact `/tmp/fpgs-franka-world-lane-strict-wX7i4mJ2/SUMMARY.md`, SHA256
`3abf71fc53701595fdbbafe85102534a3a7612de6368b9820d980af7e2910878`.
The original1ms saving milestone would now require publication to fall by
about91% while retaining the other measured work. No such corrective design
is supported. Preserve this experiment as unpromoted; do not extract the
0.18ms factor-only change as a claimed large whole-physics improvement.

The split G1 correction is native-ready14:34, ahead of15:20. Hidden AOT
reports producer128/120 registers,2440B shared, no stack/spills, versus the
unchanged32-thread GS72 registers/1116B shared. It releases producer resources
before solving rather than pretending its peak register count disappeared.
CPU7 tests pass; the untimed observer has3 positive and8 negative controls.
Paired native runs pass three production lifecycle/capacity selectors each,
but a newly added componentwise W*J check rejects8 saved cases on both cards
at4–8e-6 absolute coefficient differences. Original-control diagnosis is
pending; no runtime or tolerance change has been made in response.

### Parallel split closure: correct publication, remaining producer loss

The exact original-control probe resolves the new coefficient diagnostic:
`/tmp/fpgs-g1-split-coeff-control-Av2BjEPS/probe.py`, SHA256
`f4b5d4b6a3704aad011807c102010e23bd2878a2fafa721420f2919abde0d7c2`;
its `rtx.log` SHA256 is
`d84072607249cb59ac08d6ed033d5378dd5bb4b0b7df7c1e79374f11a487c9ff`.
Root process 38070 exits 0. Both original and candidate fail the same eight
FP64 componentwise coefficient comparisons, while published native Z is
bit-identical in all 16 cases after matching canonical raw-contact IDs.
All 16 unchanged momentum, cone, original-law eight-sweep reproduction and
independent converged held-H reference checks pass. This is a newly exposed
control-incompatible coefficient check, not evidence of a candidate row error;
the initial failing logs remain preserved.

The narrow test repair retains finite coefficients, support identities and
zero padding through `check_rows(coefficients=False)`. It compares canonical
Z and incident values to the unchanged original producer after raw-ID matching,
using the existing `rtol=3e-5, atol=3e-6`, without widening either tolerance.
All metadata, physical component diagnostics, momentum/cone and independent
eight-sweep/reference assertions remain; publication comparisons run after
those physical checks. Frozen test SHA256:
`af193338e0aa7b0a96e32791c5e7b32bced6e42628bb0cdd54998cd48e26c47a`.
CPU 7/7 and targeted precommit pass. Final native reruns 40542/63414 both exit 0,
with all four selectors passing on each card: saved current/held physical rows,
empty predictor publication, loaded reset/notification/graph lifecycle, and
sticky capacity plus production-shaped dummy-weight storage. Logs:
`/tmp/fpgs-g1-parallel-split-native-final-20260916-sJ6l4wuR/gpu{0,1}.log`.

The clean paired whole-physics screen 87023 exits 0, with source, idle,
activation and capacity guards passing and unchanged physics budgets:
`/tmp/fpgs-g1-parallel-split-whole-paired16k-20260916-01/manifest.json`.

| Device | Qualified chain baseline (ms) | Split candidate (ms) | Baseline/candidate |
| --- | ---: | ---: | ---: |
| RTX | 15.053081175 | 17.438470650 | 0.86321x |
| GB | 19.887524600 | 23.348452625 | 0.85177x |

The correction recovers much of the complete-owner loss but remains slower
than the qualified chain. It is not accepted or promoted.

Node process 62877 preserves the expected old auxiliary-graph analyzer exit 1:
`/tmp/fpgs-g1-parallel-split-node-candidate16k-20260916-01`. Both captures and
final source/idle checks complete. The supplemental strict reader retains all
48 physics/12 auxiliary roots, process/correlation checks, source pins and both
checked boundaries; it explicitly verifies the split producer and restored
original solver, each eight calls per environment step. Reader:
`/tmp/fpgs-g1-parallel-split-strict-R1HtzAhs/read_split.py`, SHA256
`abe95f3747e9f8a9d08b08f811b35e89828a6c0f5f28994a71f5d527abf456b5`.
Its `candidate_gpu0.json` / `candidate_gpu1.json` hashes are
`03137f5bb89803c579e1606b5f622d516e6eab6892cd5829b329becd004a647d` /
`e7bf34d2833e8a8fb361fe69bf24e7bd3e31a82a0d01d92e1f34e0311b93cd17`.

| Summed node time per environment step (ms) | RTX | GB |
| --- | ---: | ---: |
| New parallel predictor/row producer | 5.294352 | 6.506496 |
| Exact raw-contact CSR and CUB scan | 0.174584 | 0.188234 |
| Restored original 32-thread metric GS | 3.994346 | 4.411032 |
| All physics graph memory activity | 0.099728 | 0.091558 |

Against the existing qualified-chain strict reference, not a new paired
baseline node capture, old prediction plus all row work was 3.088404/3.144614 ms.
New producer plus CSR is 5.468936/6.694730 ms: an increase of 2.380532/3.550116 ms.
Original GS is essentially flat, changing only +0.036579/-0.011312 ms. Memory
activity decreases by 0.052736/0.055771 ms; retained factor, collision and finish
remain essentially flat. All old predictor/row producers and the complete
fused owner have zero calls. Thus remaining loss is localized to the producer,
not duplicated old work, CSR or a slower replacement solve.

Actual producer launches use 128 threads, 128/120 registers and 2440 B shared;
original GS remains 32 threads, 72 registers and 1116 B shared. Reported local
memory per thread is zero. These are launch resource facts, not measured
occupancy, bandwidth, stall or instruction counters. No further mapping or
parameter sweep is justified by this closure. Runtime and tests stay frozen;
the separate unpromoted world-lane closure was preserved by commit `cd9315da`.

## Next structural decisions, 15:25 UTC

Parallel-owner and split source/test/report closure is committed and pushed as
`f22128016220fb4a4424b09f257833bef893a652`; full and new-file precommit pass.
Both variants remain default-off. The accepted increment is still the earlier
qualified vector-chain result, not any of the subsequent losing experiments.

Read-only/CPU screens avoid repeating several apparently larger opportunities:

- Exact cold zero-impulse admission is sound, but prior current RTX censuses
  admitted only about14% of worlds and6.3--6.6% of contact-whitening work.
  Such worlds already take one cheap unchanged GS sweep. The old full16K
  impulse-only replay gives14.099% RTX/14.429% GB final-zero worlds; this is
  not a cold-admission proof or the same census. Prediction is required for
  admission, so it does not automatically retire the inverse/factor owner.
- Implicit signed-limit columns eliminate only a subset of the0.560ms RTX
  limit-prefix owner. They retain the same GS dot/update and add W/index
  gathers. Maintaining physical-coordinate residuals instead was already
  tried by PRESENT_PORTS and lazy-compliance variants, with complete losses.
- Prior body-basis formation already discovers actual touched bodies, not a
  fixed seven-body set. Global pair scheduling cannot claim deletion of that
  nonexistent fixed-basis work. The saved3.123 contacts/template/world is not
  an actual body-pair reuse census.
- CPU replay of the exact native friction-root recurrence uses157 probes for
  69 sliding transactions in the selected G1 controls, mostly two or three
  probes, not16 each. These are not measured GPU counters. A fixed
  eigenbasis squared-secular Halley policy uses172 and changes no row/sweep
  work. Its CPU correctness is not a performance result. Preserved analysis:
  `/tmp/fpgs-metric-secular-root-etJFmt3g/RESULTS.md`.
- Tensor Core packing/precision emulation is not a free switch in installed
  Warp1.17. Sparse support12 whitening has234 useful products per triple;
  a padded16x16x8 operation already has2048 before precision emulation and
  still owes geometry, grouping, incident, norms and publication. No native
  Tensor Core experiment is justified by peak-throughput arithmetic alone.

Fund one new **sparse supernodal factor/inverse producer**, default-off
`FEATHER_PGS_SPARSE_SUPERNODAL=1`. Derive twelve branch-preserving panels from
the admitted434-entry graph: six finger chains, two arm chains, torso, two
legs and root. Independent fingers/legs, then arms, torso and root give four
factor stages, followed by reverse-tree inverse composition. This changes
both factor and inverse dataflow, unlike the earlier scalar-level rewrite.
The complete algebra retains2242 factor and2242 inverse products,391 factor
divisions,43 square roots and434 inverse divisions; it adds587 separator
assembly additions. The proposed fixed four-warp CTA reduces whole-CTA joins
from47 to8, with about2.35KiB additional private separator scratch. No new
world-scaled matrix, response conversion or changed held-generation cadence.

The current complete refresh owner costs1.780ms RTX/1.505ms GB. Halving it
would be a useful structural increment, not4x across tasks; the prospective
whole-saving target is about1ms and is not established by operation counts.
Preserve all current H/R/K inputs, serial/parallel drive readiness, requested
refreshes, failure/status publication and canonical W434. First missing-API,
actual-H/action and lifecycle controls precede an early original-driver whole
screen. Native-ready checkpoint16:30; no silent extension beyond17:00 or
block-size grid. Main agent alone owns GPU runs.

In parallel, an isolated **heightfield shell-support reduction** candidate
starts fromf221 at `newton-heightfield-shell-support-20260916`, branch
`ooctipus/heightfield-shell-support-20260916`. The prior finite query discards
positive-shell analytical face manifolds and repeats the generic query because
ordinary spatial reduction can leave one-sided close support. The proposed
correction retains the analytical manifold only with a paired buffered
heightfield reducer: preserve the exact legacy inner-depth predicate, allow
outer spatial support only within unchanged margin+gap, and rank inner
contacts above outer ones. Keep original deepest/voxel competition, finite
borders, seam filters, capacities and all non-heightfield paths.

This is not the closed direct-fallback shortcut or fused original-manifold
experiment: those retained the old reducer and either removed only discarded
clipping or added the original ten-support polygon builder. Every buffered
heightfield witness sharing a bin, including generic fallback witnesses, must
use the same priority encoding. Direct/unreduced collision keeps its original
fallback. No extra result buffer, queue or launch is planned. Current entire
generic fallback0.824ms RTX/2.924ms GB is only an upper bound, not guaranteed
retirement; about1ms RTX is a stretch hypothesis including duplicate work.
Loaded rebound/support/sliding and mixed inner/outer controls must precede
timing; eventual MJWarp comparison must share the collision improvement.

## Structural screens and causal decisions, 16:10 UTC

The shell-support candidate in the isolated heightfield worktree completes its
original paired native checks. The first standalone invocation omitted the
historical grouped-lanes recipe and failed before any physical solve; those
logs remain preserved. Reusing the original lanes16/rows-masked/threads4
settings gives four passing selectors on each card. Rebound collection is
audited separately: reconstruct cleared J as Y*(L*L.T), retain denominator-only
CFM semantics, and compare independent contact geometry and exactly eight
original sweeps. All loaded gates pass; candidate rebound spin/residuals are
smaller in these repeats, not a universal trajectory or ordering guarantee.

One complete G1 shell screen passes all source/capacity/idle guards:
RTX15.051504525 ->14.913547525ms (1.00925x),
GB19.820232075 ->16.565692400ms (1.19646x).
This is a useful GB discovery, not an accepted RTX increment or a fresh MJWarp
ratio. Strict node attribution against the preserved qualified-chain reference
shows collision saving0.699232/3.851594ms, but rows plus GS add0.528779/0.488086ms.
Two checked endpoints contain about23% more contacts and12.5% more total rows;
these samples are not timed-average populations. There is no duplicated old
query/reducer. Full closure and exact pins are in the isolated worktree's
`reports/fpgs/HEIGHTFIELD_SHELL_SUPPORT_20260916.md`; the option remains off.

The register-front supernodal factor also passes four native selectors/card.
Actual whole captures give RTX15.054503250 ->14.814186625ms,
GB19.841636000 ->20.720324550ms. A second parent exact-key check rejected the
new refresh name after capture; the failed manifest is preserved. A narrow
parent extension then passes all four recorded-child checks and rejects three
negative controls. It does not rewrite the original outcome. Strict node
analysis retains all48 physics/12 auxiliary roots and finds complete refresh
1.780024 ->1.574609ms RTX, 1.504000 ->2.376645ms GB, with all downstream owners
essentially flat. Eight new calls replace all old refresh calls. The GB loss
is in the new factor, not contact count, solver iterations or retained work.

Source/compiler evidence qualifies the original47->8 CTA-join hypothesis:
the new critical path still has19 warp-local pivots versus15 old levels,
868 explicit shuffle broadcasts,96 registers versus40/39, and about17.3K
static SASS instructions versus980 looped instructions. Static counts are not
dynamic work or hardware-stall measurements. Preserve this first version,
then fund one shared-front correction: element-parallel bounded loops and
warp-private shared fronts, retaining the same panel algebra and eight CTA
joins. Charge additional shared traffic and warp synchronization. Prospective
complete-owner targets are <=0.980ms RTX and <=1.504ms GB, not predictions.
Native-ready checkpoint17:15, no silent extension beyond17:30 or mapping grid.
Full source/native/whole/node closure: `SUPERNODAL_20260916.md`.

Additional bounded screens prevent rediscovering earlier paths:

- Franka current state/prediction/factor is only1.671307ms. The closed
  world-lane owner already removes generic chain/weld work; its0.184506ms
  predictor/factor saving cannot support a new1ms state-only claim. The
  current owner already caches next-state bias/H and skips valid repair.
- Existing Lab has a deferred-publication hook, but calls arbitrary state
  force callbacks before the next solver invocation. Newton cannot infer
  that those callbacks do not read public poses/velocities. Even deleting
  an entire alternating Franka finish phase would save <0.475ms; mandatory
  private geometry/motion/bias and contact inputs make actual public-only
  savings substantially smaller. No Lab change or unsafe lazy flag is made.
- General coplanar-hull contact deletion is not exact Coulomb equivalence:
  moving normal load from a sticking center to sliding rim changes yaw
  friction. Even frictionless deletion requires actual row/bias domination;
  mixed penetration/speculation can violate it. Seven-to-four hull reduction
  in the zero-friction rebound fixture is not a full G1 cost or quality proof.
- The exact original metric-GS root and physical path-certificate screens
  remain CPU closures, not native gains. Certificate16 removes only about
  12% of row dots before added bounds/norm checks; no native funding.

A separate bounded CPU study now tests history-two Anderson acceleration of
the original ordered metric-GS map, not the earlier diagonal/spectral AA4 or
Nesterov method. Keep the native16-cap root and maximum eight GS sweeps,
project proposals onto original cones, rebuild the actual kinetic correction,
and charge every history/physical-residual/projection/rebuild operation.
Common converged held-H accuracy and physical invariants, not bit identity or
componentwise monotonicity, decide quality. One frozen formulation, no history
or coefficient grid; first formula checkpoint16:35, results17:00.

The accepted session increment remains the earlier vector-chain improvement.
Neither the shell nor first supernodal result is promoted as progress toward
the RTX4x-everywhere goal, which remains unmet.

## Shared-front correction and numerical screen closure, 16:33 UTC

The one shared-front supernodal correction completes five native tests on each
GPU, then an early whole-task comparison with all source/capacity/idle guards
passing. RTX15.071086600 ->14.804360150ms (1.018017x),
GB19.864856800 ->20.088351775ms (0.988874x). It misses the predeclared large-gain
targets and remains default-off. No further panel/mapping tuning is funded.

Strict node attribution retains48 physics and12 auxiliary roots per card,
preserving the inherited auxiliary-analyzer failure separately. Complete factor
time is1.565091ms RTX /1.768867ms GB, versus the original1.780024/1.504000ms
and first register-front1.574609/2.376645ms. The correction recovers about0.608ms
of the GB prototype loss while barely changing RTX. Actual resource records
show58/56 registers,7336 bytes shared and zero local memory. Reduced registers
and static code size are not proof of a hardware bottleneck or sufficient gain.
Other owners remain approximately flat; no old refresh is duplicated.

Artifacts: `/tmp/fpgs-supernodal-shared-native-20260916-x4R0c3LI`,
`/tmp/fpgs-g1-supernodal-shared-whole-paired16k-20260916-01`,
`/tmp/fpgs-g1-supernodal-shared-node-candidate16k-20260916-01`, and unchanged
strict-reader reuse at `/tmp/fpgs-supernodal-shared-strict-PKqnl8i1`.
Full frozen source, resource and measurement pins are in `SUPERNODAL_20260916.md`.

The single frozen original-GS AA(2) CPU screen reduces consumed sweeps79->48
and actual native-root probes157->111 across sixteen saved G1 cases. Charging
physical-stop scans, proposed-impulse reconstruction and merit checks instead
increases operator dot/action events2408->5340 before history arithmetic.
These events differ in support/latency and are not measured wall time, but
they do not establish the cheap complete replacement required for native work.
Fifteen cases meet unchanged physical stopping tolerances versus fourteen
original. RTX current8192 is farther from the common held-H reference while
still passing every physical stop criterion; this is an accuracy trade-off,
not a false declaration of an incorrect contact law. No more AA variants are
funded. Exact policy, per-case results and charges are preserved in
`/tmp/fpgs-g1-aa2-metric-gs-IPTQJpKv/RESULTS.md`.

Source studies also recover the already-measured Franka pair-shape preparation
option: historical RTX whole savings0.315868ms, not a new architecture or a
current timing claim. The existing flag replaces full shape preparation with
the immutable explicit-pair endpoint union. Current per-shape arithmetic stays
identical to the full producer. Prior rigid-body hierarchy screening found one
leaf per participating Franka body and no new pair reduction. Kuka's current
state family already contains compact next-state bias/mass, one-warp four-round
pose/motion scans and no duplicated primary canonical factor/tau producer.
Neither audit establishes another1ms deletion.

Next bounded numerical question: can three warm, element-parallel Cholesky
relaxation passes replace the sequential pivot schedule, retain the original
exact sparse-inverse consumer, and satisfy a conservative current-H error check?
This is not stale-H reuse or unchanged-W contact response. Seven saved one-step
pairs provide a CPU falsification screen only; they do not cover the next
scheduled refresh, resets or a whole-task population. Charge matrix assembly,
all passes, residual/norm checks, retained factor storage and full fallback.
No runtime implementation or speedup is implied by this question.

## Warm-factor and field-major screens closed, 17:16 UTC

The three-pass warm Jacobi Cholesky screen fails the existing current-H
operator and predictor backward checks in all seven retained one-step pairs,
despite positive pivots. Actual energy-operator defects are 1.98%--8.05%; the
exact inverse is not the failure. The unfinished factor dependency propagation
is. Three passes plus unchanged exact inversion and a residual check require
11,644 counted products versus 4,484 original, before norms and fallback.
This is not a measured runtime comparison. No native implementation is funded.
The initial rounded-LLT off-pattern initialization assertion is retained; the
corrected study uses independently reconstructed previous physical H rather
than truncating a factor. Exact formulation and outputs:
`/tmp/fpgs-g1-warm-jacobi3-A6251f/RESULTS.md`, SHA256
`3d3e5a4ca468faae2e07012c256bd7946e1b5090314c0093d7643d6529cbcc43`.

A distinct field-major solver screen replaces per-row warp reductions with
one independent world per GPU lane, transposed contact fields, private
shared-coordinate state, and a separately charged canonical-W decode.
Original finite-eight transactions, native tangent root, scalar fallback and
incoming-impulse semantics remain. There is no production hook or W transpose.
CPU and two native selectors per GPU pass, including sixteen physical saved
cases and thirty-five mapped worlds with graph/stale-state controls.

Actual full16K last-solved inputs reproduce the original solver exactly on
both cards. Valid complete-component medians (ms per call) are:

| GPU | Original GS+decode | Field-major GS+decode | Original/candidate |
| --- | ---: | ---: | ---: |
| RTX | 0.507535994 | 3.791040063 | 0.13387777x |
| GB | 0.557071984 | 3.533216000 | 0.15766712x |

Ten warmup and forty measured balanced AB/BA rounds restore cold inputs outside
every event. Both candidate kernels are charged; packing is explicitly excluded.
All16,384 worlds per card pass hard finite/cone/sign/momentum checks and native
velocity-reference tolerances. One GB complementarity diagnostic worsens,
without a hard-law failure. The loss therefore is not attributed to physics
rejection. Valid artifact:
`/tmp/fpgs-g1-field-major-component-paired16k-20260916-02`.
Run01's helper-hash edit during execution correctly fails its source guard and
is preserved separately. No accepted result relies on it.

The intended collective retirement occurred: no main-kernel spills, explicit
barriers or shuffles. But main work exposes only512 warps rather than16,384,
serializes up-to18-term support operations, reloads coefficients, moves impulse
and cross scratch into global memory, and incurs divergent row/root work.
These are source/resource facts, not hardware-counter causal percentages.
Even with free packing the component needs roughly9--11x improvement to reach
its declared saving target. No justified single correction supports that;
the path is closed without a subgroup grid or production rewrite. Detailed
source/native/resource/quality pins: `FIELD_MAJOR_GS_20260916.md`.

The next bounded structural question revisits finite contact representation,
not another execution mapping. MJWarp's adaptive four-witness selection is
distinct from the previously failed four fixed directions. A different
manifold can be a legitimate approximation; lack of bit identity or exact
old pointwise friction is not itself a rejection. It must still preserve
physical support, impact/friction behavior and convergence against its own
discretization. Current geometry and complete selection/export cost must
justify funding before implementation; normal-bin equality is not a plane
certificate, and voxel exports cannot silently resurrect suppressed points.
No contact is dropped from current production, no capacity or Lab code changes,
and no gain is claimed from this open question.

## Streaming closure and finite-manifold implementation, 18:20 UTC

The cause-directed Franka world-lane streaming correction is closed and
unpromoted. Its physical/lifecycle tests pass on both cards, and the intended
finish register reduction occurs (167/255 to80, no spills). Complete paired
physics nevertheless regresses RTX5.141021 to5.286214ms and
GB4.715610 to5.077819ms. Strict48-root attribution shows publication adds
.355465/.487562ms, outweighing factor/prediction savings of.188539/.171183ms.
No duplicate old producer or changed cadence explains the miss. Shared traffic,
22 publication joins and only512 world-lane warps are explicit costs; hardware
counters do not establish a dominant stall cause. No further mapping variant.
Source and full evidence are preserved in fork commit
`c56caee4ec13b226d19626621c1e881764e79571`, branch
`ooctipus/fpgs-world-lane-streaming-20260916`, including
`reports/fpgs/FRANKA_WORLD_LANE_STREAMING_20260916.md`.

The contact census required one coherent snapshot, not an untimed read of
overwritten next-state geometry. Private copies immediately after original
SparseFactor.build_rows identify the same retained solved rows byte-for-byte;
the unchanged private cold-eight replay exactly matches all active impulses
and public velocity on both cards. Copied geometry/material/route arrays are
diagnostic only and their188,984,988 bytes per build are not timing evidence.
Capture01 failed an overly strict eager-allocation guard before measurement;
capture02 uses Warp's supported graph-owned mempool allocation with persistent
user references. Failed01 and its original adapter remain unchanged.

Valid capture, both children exit0 and final source/idle guard pass:
`/tmp/fpgs-g1-raw-manifold-input-paired16k-20260916-02`.
Input NPZ hashes: RTX
`aa34c04517f34656ce2d2413c0d2f8113bff21ad04639bbac30e1474a73c888e`;
GB `cab846f8c85522e083de92f9160459802efe944773c8daf5865292020ba4b8a9`.
CPU census:
`/tmp/fpgs-adaptive-witness-population-90c5BsvH/{census.py,population.json}`.
An independent reader confirms geometry signs/charts and pair/run counts.

| One last-solved exported snapshot | RTX | GB |
| --- | ---: | ---: |
| Actual heightfield/convex shape pairs |24,868|24,840|
| Normal contacts |78,284|78,360|
| Pairs with more than four contacts |5,553|5,601|
| Normal contacts removed by unconditional cap4 |14,666|14,711|
| Pairs fragmented across raw runs |14,919|15,238|
| Contiguous raw pair runs |53,859|54,438|

These are OLD reducer-export contacts, not pre-reduction triangle candidates.
Only roughly300 contacts meet the descriptive parallel-normal/near-plane
cap4 subset, so that restrictive path cannot justify a structural gain.
General pair selection is a different finite manifold: the median maximum
normal separation in pairs above four is about50 degrees. Existing-width
removal proxies are not predicted candidate row counts or speedups. The
documented friction-anchor limit applies to raw contiguous runs; changed export
ordering can change it. Pair-level ownership alone does not guarantee raw
contiguity under the unchanged atomic contact writer.

Fund one default-off replacement-export prototype, based on clean qualified
chain ef9481, in `newton-heightfield-adaptive-manifold-20260916`, branch
`ooctipus/heightfield-adaptive-manifold-20260916`. Reuse existing pair lists and
global-reducer winners; replace final export rather than retain it and add a
second compaction. For admitted standard heightfield/box-or-convex pairs,
choose deepest, farthest, line-spread and triangle-edge-spread witnesses using
the local MJWarp strategy. Keep each witness's own geometry/normal/material,
original final-writer acceptance and deterministic ID ties. Retain up to four
distinct admissible IDs even at zero scores; do not import mixed-unit .001
cutoffs. Unsupported/speculative/custom writer paths retain the old exporter.
No changes to the friction allocator, Lab, capacities, substeps or iterations.

The bounded screen must charge all hash requests, old duplicate filtering,
selection passes and export. Native-ready checkpoint is approximately19:30;
the performance milestone is at least1ms whole G1 RTX saving, standalone or
combined with the existing shell path, not a one-percent export improvement.
Reuse loaded finite/shell fixtures for support/COP, rebound, tilt, border,
slide, yaw and step-straddling. Own-manifold reference convergence and hard
cone/momentum checks are necessary but do not alone certify cross-manifold
physical quality. Shared collision improvements also belong in MJWarp's arm.

Operational correction: completed-agent tasks sent as ordinary messages did
not restart work. New bounded assignments now always use followup_task, and
completion is checked before scheduling. This prevented-work interval is not
counted as implementation or validation progress.

### Root-free sequential solve reopened,18:42 UTC

Re-examining the frozen ANYmal spectral-GS control found that the prior
13/2048 pointwise velocity-reference regressions are finite-iteration
tradeoffs, not cone/sign/momentum failures. They remain documented, including
maximum excess scaled kinetic velocity error0.0030623. Across the complete
saved cohort, median/p99/max kinetic error changed from
0.006389918/0.03199576/0.05573845 to
0.00006906845/0.002584805/0.006777305. The biased normal residual is not raw
contact closing velocity; these quantities must not be conflated.

The user permits numerical changes and does not require every finite-budget
case to improve monotonically. Fund one unqualified native cost screen of
the same frozen algorithm: retire EX1 and Nesterov, normal-first ordered
updates with a shared tangent spectral denominator and circular projection,
at most the same24 sweeps, same physical stop, same staging/whitening/decode,
and complete original fallback. No root iterations, Schur preparation,
history, extra passes or mapping grid. The milestone is at least1ms whole
ANYmal RTX saving, not a replay-only arithmetic reduction.

Prototype tree: `newton-fpgs-anymal-spectral-gs-20260916`, base57e1dcfa,
flag `FEATHER_PGS_SPECTRAL_GS=1`, default off. Native runtime SHA256
`aedd42a5ef76041620c7333bf34029c06d12d1410aeed1deaf366befa8f69943`.
CPU controls and independent source review pass. Exact live72/32-capacity
AOT report `/tmp/fpgs-spectral-contact-offline-tVyCzVM2/offline02/report.json`
SHA256 `b7b7cc0d0b8d8732d2f8998c7fe02a6efff6e62b0e6149a3992ab63809425872`:
SM120 tier32/48 uses83/128 registers; SM10389/95, no stack/spills.
Sequential warp dependencies and the tier48 second-warp resource lifetime
remain cost risks. Neither AOT nor operation counts establish GPU speed.

The adaptive terrain native pool tests pass on both cards, but loaded
qualification currently fails support/sliding assertions in both old and
new paths. A cause-focused audit found sparse instantaneous-force sampling
and a full-friction analytic sliding assumption incompatible with the
explicit production two-anchor policy. Those are leads, not qualification.
The runtime remains frozen while an early, explicitly unqualified whole
physics cost screen runs. No candidate gain is promoted from these checks.

### Integrated outcomes and next structural direction,19:01 UTC

The ordered spectral prototype is closed, default-off and unpromoted, committed
and pushed as `2dfa26a753b380213a2e9dd4733df4d461bd4446` on the scoped fork
branch. Whole02 original/candidate physics: RTX9.219848/22.605269075ms,
GB9.501435475/24.7519868ms. All source/idle/capacity guards pass. Native saved
operators pass2048 hard checks, but the owner is4–5x slower; eight CPU/native
translation checks have scaled kinetic velocity error<=1.2112e-5.
The ordered per-contact dependencies, reductions and lane-zero projections
are not removed by lower scalar-product counts. Correcting the CPU proxy for
unconditional native updates gives72.008M versus84.268M products, only14.55%
less, not the earlier31.76%. No hardware stall claim is made.

Whole01 is preserved as an observer setup failure: root checked a constructor
compatibility argument `mf_warmstart` as if it were an instance attribute.
The actual stored field is `_mf_warmstart_enabled`. This was fixed before the
fresh02 capture; source-based actual-assignment controls were then added to
the independent review. No physics result came from01. Review dummy objects
that repeat an observer's assumed fields cannot establish real API validity.

Adaptive4 standalone whole: RTX15.032385825->14.81613825ms(1.014595x),
GB19.827471325->19.789886175ms(1.001899x), one unpromoted round. Strict12-step
node audit proves48 physics/12 auxiliary roots, zero unproven nodes, and
replacement export exactly4 calls with the old export absent. Export grows
RTX0.302094->0.979333ms andGB0.323757->1.526946ms, erasing much of the
row-plus-solve saving0.847520/0.894178ms. State/factor work is approximately
flat. Strict artifacts: `/tmp/fpgs-adaptive-manifold-strict-kolTndj0/`.
This is evidence of hash-pool gathering/selection cost, not measured stalls.

Corrected loaded controls use every force sample over the same fixed final80
steps, retain the original2% support and2e-4 momentum bounds, and preserve the
old five-sample diagnostics. A separate full-friction sliding case retains
the analytic stopping test; production-anchor cases check their actual law.
Both cards complete18 records. All9 candidate cases/card pass; the old
initially penetrating step-support case still fails settling, so the suite
correctly exits1 rather than hiding the baseline failure. Logs:
`/tmp/fpgs-adaptive-manifold-loaded-tail-20260916-iws6t0uD/gpu{0,1}.log`.
This bounded geometry/physical evidence is not yet an independent full-task
or own-manifold reference-convergence qualification.

Next funded studies target the demonstrated causes, not exporter tuning:
query-to-adaptive-manifold dataflow that retires global bin/hash round trips,
and concurrent root-free contact proposals that avoid both the sequential
dependency chain and the earlier coupled-Jacobi local roots. Exact adaptive
selection still requires retained candidates or repeated queries; four
output witnesses do not imply four-record streaming state. Each study must
account for complete producer/fallback/selection/solve work before promotion.

Adaptive standalone closure is committed/pushed as
`c59c10bcf732ec719ba5a353c7b1bf1fe7996ca5`; measured runtime bytes are
unchanged. The new isolated pair-CSR branch starts there, not from shell, so
there is no hidden shell-policy composition. It keeps separate packed query
kernels and replaces bin/hash bookkeeping with variable-cardinality pair
membership over the existing raw witness buffer. A fixed four-record query
pool is expressly not the design. Native-ready checkpoint20:30 UTC; root
retains exclusive GPU scheduling and the early complete-cost decision.

### Concurrent spectral and pair-CSR checkpoints,19:30 UTC

The concurrent spectral proposal preserves the earlier parallel delta-response
recurrence and permanent half-step latch, but removes local roots/Schur work
and the ordered contact loop. Same24-pass maximum and original staging,
whitening, decode, capacities and complete unsupported fallback. Default-off
`FEATHER_PGS_SPECTRAL_JACOBI`; isolated tree
`newton-fpgs-spectral-jacobi-20260916`, runtime SHA256
`8fd17fcbdb8ffce9b2df33d4ce226636ebf82047b5df7dc7f983e121a57b0065`.

CPU2048 saved cases: no hard failures;1982 closer to the fixed velocity
reference,66 genuine regressions. Scaled kinetic error median/p99/max changes
0.006389918/0.031995761/0.055738454 to
0.000118792/0.006623684/0.018130025. Maximum regression excess0.007910961;
these tails are not declared converged away. Counted operator+setup products
49.830876M versus84.268494M; physical merit/projection and synchronization are
additional work, not free. CPU artifact
`/tmp/fpgs-spectral-jacobi-cpu-etPRAs/`; four exact-capacity AOT builds pass,
zero spills, with86/91 registers on RTX and87/101 on GB.

Native saved-operator replay completes on both cards,2048 hard checks pass,
eight CPU/native translation controls have scaled kinetic velocity
error<=1.21146e-5. Despite lower counted products, original/candidate replay
is roughly0.048/0.095ms RTX tier32 and0.052/0.095ms tier48;
GB0.050/0.101 and0.054/0.101ms. Replay includes identical output restore and
is not extrapolated to whole-task speed. Artifact
`/tmp/fpgs-spectral-jacobi-native-20260916-AbNbPv7B/`.
This recovers much of the ordered prototype's loss, not parity or a gain.
One complete paired whole-physics screen is running; the cause review now
charges physical stopping/merit and operator synchronization rather than
assuming arithmetic retirement predicts time.

Pair-CSR runtime and focused tests are ready, earlier than the20:30 checkpoint.
The full raw pool is retained; count/scan/scatter supplies variable-length
pair buckets with no fixed per-pair cap. Selection uses all stock-writer-
accepted raw witnesses, so this is a different manifold from selecting old
hash survivors. Separate original finite/generic queries remain. Stock-writer,
unique-pair and supported-mode admission is checked; unsupported contacts
alone reach old reduction/export. Added integer storage is
4*(triangle_capacity+2*raw_capacity+3*pair_capacity+4) bytes,20.8125MiB at
the unchanged G1 capacities. Original fallback geometry/hash storage remains.

Seven CPU controls, native-code AOT and full/targeted pre-commit pass. A real
raw-overflow test exposed that the public status(clear=True) API must return
the CSR error before clearing it; the implementation now does so. Actual
CPU CollisionPipeline objects pass the literal observer in both flag arms;
six wrong-dispatch/ownership/status cases reject. This is dispatch evidence,
not physical quality. Native loaded controls and early whole cost follow.
No new candidate is promoted, and no Isaac Lab code was edited.

### Cause-directed corrections,19:53 UTC

Concurrent spectral whole-task screen completed with all source/idle/ownership
and capacity checks passing: RTX9.218097850->12.435956575ms (0.741246x),
GB9.483441025->14.137671900ms (0.670792x). Artifact
`/tmp/fpgs-anymal-spectral-jacobi-whole-paired16k-20260916-01`.
An output-only adaptation of the existing five-phase clock observer preserves
byte-identical physical outputs on both cards and strips back to production
source exactly. In the instrumented owner, merit/commit occupies52.35--54.57%
on RTX and53.44--55.74% on GB; proposals another roughly15%, operators11%,
preparation14--16%. Its changed register use means these fractions are not
uninstrumented task-time measurements. Artifact
`/tmp/fpgs-spectral-jacobi-phase-f78F1Iy2/`.

The old counted-products prediction omitted repeated invariant divisions and
square roots inside full physical merit. Across the frozen CPU schedules the
literal native map repeats6.451M invariant-denominator divisions and1.398M
invariant diagonal square roots. The original Nesterov owner instead caches
its step and has a cheap delta exit. One cause-directed correction is funded:
compute proposal reciprocals and physical-merit weights once, recycling the
same three shared caches, while retaining every merit term, stop, permanent
latch, physical operator and24-pass maximum. No removal of validation from
the solver algorithm and no mapping grid. It must save4.218ms relative to the
losing RTX candidate to reach original-minus1ms. That recovery is unproven.
Pre-correction source/evidence is committed/pushed as
`5f1548faccf787e2930c06cdb7f5e1e65284bf38`.

Pair-CSR whole02: RTX15.036678875->14.161686525ms (1.061786x),
GB19.870284175->18.936348000ms (1.049320x), source/idle/capacity checks pass.
The0.875/0.934ms savings are below the standalone1ms milestone and are NOT
accepted because loaded support fails. Whole01 stopped before measurement:
the new public CSR capacity flag was correctly exposed by runtime, but root
had not extended the pinned observer's exact permitted-key set. The wrapper
now adds exactly that one named flag only when explicitly requested; full
actual CPU pipeline checks pass both arms and reject sticky failure. Old
failure artifacts remain.

First four native routing/reset/overflow controls pass both cards. The first
loaded attempt failed before stepping because root omitted the fixture's
existing `FEATHER_PGS_GROUP_LANES=16`; it was NOT a CSR physics failure. The
correct repeated recipe also uses `FEATHER_PGS_ROWS_MASKED=1` and
`NEWTON_NARROW_PHASE_THREADS_X=4`; no runtime change was made for this setup.
Corrected logs `/tmp/fpgs-pair-csr-loaded-20260916-HBoLF6K6/gpu{0,1}.log`
complete18 cases/card and expose actual candidate failures: flat/tilted/yaw/
step support does not settle. Flat support final spin0.5844rad/s versus old
about1e-6; penetration0.675mm. Weight average and momentum/cone pass, showing
that these checks alone do not certify the manifold. No gate is relaxed.

The causal CPU geometry study finds that unrestricted spread over all raw
witnesses keeps one near-vertical support witness and three distant positive-
gap side witnesses. Those side witnesses are valid detection candidates, but
cannot replace the load-bearing footprint. Four outputs are not in themselves
a correct manifold. MJWarp's inspected heightfield producer uses zero-cutoff
queries of margin-inflated geometry, not Newton's full positive detection-gap
domain; there is no established identical pre-selection eligibility rule to
copy. An activation/nearest-depth priority correction is under study, with
operation-scaled roundoff handling, not guessed distance/angle thresholds.

Strict node audit `/tmp/fpgs-pair-csr-strict-UGnL9ijL/` proves48 physics and12
auxiliary roots with zero unproven nodes. Original full hash reduction is
absent; only fallback export remains. RTX hash clear falls0.659046->0.007328ms;
new count/scan/scatter/fallback reduction/export plus retained fallback export
cost0.558189ms. CSR exporter alone0.368163ms RTX/0.627026ms GB. Compared with
the qualified chain, rows get WORSE by0.175580/0.189590ms while GS saves
0.392467/0.464824ms. The raw manifold differs from old-survivor adaptive4;
its previously measured downstream savings cannot simply be carried over.

The previously CPU-screened ordered spectral policy is also reopened for one
G1-specific cost screen. Unlike ANYmal's parallel baseline, G1 already uses
ordered fused transactions, so it need not incur a new sequential dependency
chain. This retires the complete metric inverse/root/KKT proposal, retaining
the current physical operator, normal-first transaction, own-parent cone,
CFM denominators and eight sweeps. Isolated default-off branch
`ooctipus/fpgs-g1-spectral-tangents-20260916` starts at qualified chain ef9481,
without CSR or shell. Native transaction/current-held/graph/empty/saved16
controls pass on both cards; full native translation uses the frozen CPU map,
not a claim that every finite-budget case improves. AOT64 registers versus
original72, unchanged1116B shared, no spills. First whole screen is running.

Read-only collision-coherence audit did not fund duplicate work: the proposed
cached-axis/current-support rejection already exists for Allegro. Franka's
entire relevant generic owner is0.481ms and Kuka's GJK+MPR owners total
0.960673ms RTX before necessary cache/support overhead, below this study's
1ms structural threshold. Current Kuka uses explicit pairs; its historical
kernel name alone was not evidence of NXN admission. No new implementation.

G1 spectral whole screen completed19:54 with all guards passing:
RTX15.031834625->14.480169350ms (1.038098x),
GB19.837211200->18.904325600ms (1.049348x). Positive but below the1ms RTX
standalone milestone; default-off and not promoted. Preserve it as a possible
composition component, without funding instruction/mapping micro-tuning or a
new qualification campaign before a material combined whole-task gain exists.

## Closed checkpoints and corrective algorithms, 20:14 UTC

The cached spectral-Jacobi correction preserves every original physical merit
term, stop, permanent half latch and24-pass limit. Twelve CPU controls and
the native2048-case hard checks pass on both cards, including the previously
reported664 pointwise metric trade-offs (not newly relabeled failures or
universal improvement). Registers85/91, unchanged shared memory and no spills.
The complete paired screen passes all guards:

| Whole physics, ms/env step | RTX | GB300 |
| --- | ---: | ---: |
| Original production reference |9.214019375|9.488502400|
| Cached spectral Jacobi |10.193371200|11.207959225|
| Reference/candidate |0.903923x|0.846586x|

Artifact `/tmp/fpgs-anymal-spectral-jacobi-cached-whole-paired16k-20260916-01`.
The cause-directed correction recovers roughly2.24/2.93ms versus the separate
uncached screen, but still loses to production. It is closed default-off and
unpromoted at fork commit `72b3bb0302c885cf3db3c3ec9e62f306438aded5`.
Further instruction tuning is not funded. A distinct, fixed-policy CPU study
will test reusing the next unrelaxed contact proposal as a conservative natural-
residual bound for the permanent latch and early-stop gate. Full physical
checks remain required when that gate permits stopping and for final quality;
small natural residual alone does not bound absolute complementarity/MDP.
All final lookahead and midpoint replacement proposals must be charged. The
new workflow needs a further1.979352ms RTX whole saving from the cached version
to reach original-minus1ms; the uncached instrumented merit fraction is not a
prediction of this saving.

G1 spectral tangents are preserved at fork commit
`da4d74f816b59f43308140c67bbc10db17264dce`, and physically unqualified pair-CSR
at `eabc4f8c3d2aacae78e78a6004d02047b175f7f5`. Both include root's exact measured
observer extensions; all measured runtime bytes remain pinned in their reports.
Full and targeted pre-commit pass and all three checkpoint trees are clean.

One bounded pair-CSR correction is funded after independent arithmetic review.
Cache outward lower/upper intervals for the stock writer's reconstructed
separation during the existing count pass. Use the same second normalization
as the writer, not just unpack_contact's first normalization. The interval
covers represented witness arithmetic only, not geometric-query error or the
later body-local/world roundtrip. One additional bucket scan obtains the
minimum upper separation; the original four spread scans prioritize intervals
whose lower bound is at most max(0, minimum upper). Outside that cohort, fill
only remaining slots by nearest upper separation. This retains nearest
all-separated footprints without a guessed physical distance or normal angle.
Nonfinite bounds stick a failure. All raw candidates and original fallback
remain; capacities are unchanged. Added vec2 storage is14,155,784 bytes at the
calibrated raw capacity, plus count arithmetic and one extra scan. The original
loaded support/rebound/friction fixtures and an early whole screen decide;
neither a geometry-only pass nor an isolated exporter speed is promotion.

## Corrected manifold and combined G1 screen, 20:43 UTC

The priority correction is preserved clean/pushed at
`1e16a323876778aaeddf6c4e8420e3f155234bfc`. Six native arithmetic/geometry/
routing/fallback/reset/overflow controls pass on each GPU, and all nine
candidate loaded cases pass their existing physical gates. The unchanged
paired selector still exits1 because the OLD arm fails full-friction/step
controls on both cards and yaw support on RTX. Those failures remain visible.
Artifact `/tmp/fpgs-pair-csr-priority-native-20260916-1Tn0yyls`.

Independent endpoint inspection also checks the free rebound omitted by the
inherited failure-list gate: candidate normal speed1.80000019m/s matches
authored restitution0.6 and incoming speed3, energy ratio0.360000076, and
spin3.3587e-5rad/s. Flat support speed2.0426e-6m/s and spin1.9744e-5rad/s replace
the earlier unstable footprint. Full-friction travel0.5071853m agrees with
the existing0.509684m analytic control. Production selected-anchor sliding
leaves the finite terrain; sampled descent starts after the center crosses
the edge, not an observed interior fallthrough. This is bounded physical
coverage, not proof of all robot trajectories.

Corrected whole screen passes all guards:
RTX15.005645075->14.326358525ms (1.047415x),
GB19.825631200->19.141688100ms (1.035731x), in
`/tmp/fpgs-g1-heightfield-pair-csr-priority-whole-paired16k-20260916-01`.
The corrected selector retains0.679287/0.683943ms savings, below the standalone
1ms milestone. Root's actual CPU collision pipeline observer passes off/on,
rejects wrong interval ownership and sticky failure, and passes explicit
acknowledgement; the added vec2 bytes are included in its recorded allocation.

An isolated composition combines these unchanged runtime bytes with spectral
tangents da4d74f8. Only the two existing observer wrappers require conflict
resolution; both feature/factory/capacity checks remain. Composition commit
`9df743db` on `ooctipus/fpgs-g1-csr-spectral-20260916`; full pre-commit and12
executed CPU controls pass (10 CUDA controls explicitly skipped in this CPU
run; their separate native checkpoints are above). First complete paired
screen passes all guards: RTX15.040457050->13.729104500ms (1.095516x),
GB19.844910400->18.179141775ms (1.091631x). Artifact
`/tmp/fpgs-g1-csr-spectral-whole-paired16k-20260916-01`.
Three alternating repeats are running. The combined gain is measured, not
the sum of separate owner predictions; it is not yet promoted or a new MJWarp
ratio. A fresh shared-CSR MJWarp denominator must use the same collision path.

The separate spectral residual-reuse CPU study diagnoses a real pure-Q tail:
its conservative scaled bound falls while absolute normal error grows, so
the Q-only latch misses the original instability. One fixed correction uses
independent Q-increase OR normal/complementarity-increase latch streams,
without a deadband/grid or altered full physical stop. All2048 hard checks
pass;1974 cases are closer to the original held-H reference,74 are genuinely
farther (the previous66 plus8 new). Median/p99/max scaled H distance is
0.000120582/0.006974387/0.024961323 versus original
0.006389918/0.031995761/0.055738454. This is not universal improvement.
The corrected workflow removes92.47% of full-merit row visits in the frozen
CPU accounting, adds6.28% proposals and0.65% operator work, and preserves
all unsuccessful full-stop gates. Native implementation is funded once;
no speed claim or further policy grid follows from CPU counts.

## Repeated G1 composition and fresh fair denominator, 21:01 UTC

Three alternating paired repeats completed with unchanged budgets and all
guards passing, in
`/tmp/fpgs-g1-csr-spectral-repeat-paired16k-20260916-01`:

| Median whole physics, ms/env step | RTX | GB300 |
| --- | ---: | ---: |
| Qualified chain reference |15.086237125|19.841528600|
| Corrected CSR plus spectral tangents |13.766957450|18.188351025|
| Reference/candidate |1.095829429x|1.090892109x|
| Saving |1.319279675|1.653177575|

Candidate samples are13.768985975/13.758160875/13.766957450ms RTX and
18.141839575/18.188351025/18.216925650ms GB. Median wall throughput improves
1.036549259x/1.045772244x, not the same ratio as physics.

The twelve-step node capture preserves its inherited auxiliary-graph analyzer
rejection. The exact strict reader
`/tmp/fpgs-g1-csr-spectral-strict-eqJS9NBX/read_combined.py` separates all48
physics and12 auxiliary roots and retains correlation/source/unknown-owner
guards. New CSR owners occur4 times and spectral solve8 times per environment
step; old full hash reducer and old metric solve are absent. Independent replay
at `/tmp/fpgs-g1-csr-spectral-strict-review-qLL47J63` produces byte-identical
JSON hashes: RTX `da4c825b45df40c97a553834a2ad6d6bdeff704bd030b0a5b695d60764be9006`,
GB `7e83b6f9599336bc4631840a297d7921bf19a71cfb90073b9aa91e697a953d7d`.

| Current exclusive node family, ms/env step | RTX | GB300 |
| --- | ---: | ---: |
| Collision |3.043369500|7.790226500|
| Dynamics |2.635625333|2.376835500|
| Rows/response |2.702515833|2.708331917|
| GS |3.182044333|3.220647500|
| Publication |1.843514250|1.931879667|
| Force export |0.209936333|0.135122667|
| Memory |0.158304000|0.152407250|

Root spans13.895110/18.4446755ms include gaps; these diagnostic node timings
are not substituted for graph-mode medians. Row preparation increased versus
the qualified chain reference; all of that cost is charged.

Fresh fair comparison uses composition9df743db on BOTH backends, fixed Lab53ee,
shared corrected CSR1/adaptive0, identical original16K200/40/40 recipe and
calibrated capacities. Only FPGS enables its chain/spectral solver owners.
The external compatibility adapter
`/tmp/fpgs-g1-csr-spectral-fair-GAursBVt` reuses the original checked driver;
both actual collision owners, interval arrays and sticky status are verified.
Artifact `/tmp/fpgs-g1-csr-spectral-fair-paired16k-20260916-01` is complete,
all four runs exit0, both boundary checks pass, all states are finite,
physics budgets are unchanged and MJ warning masks are0 on both cards.

| Fresh whole physics, ms/env step | RTX | GB300 |
| --- | ---: | ---: |
| FPGS composition |13.711120450|18.148208975|
| Corrected MJWarp with shared CSR |37.277615275|37.823784600|
| MJWarp/FPGS |2.718786945x|2.084160737x|

This is one fresh backend pair, supported by the separate three-repeat
composition comparison; it is not4x or a claim of improved other tasks.
The prior same-session G1 ratios were2.431726696x/1.871729653x. Do not divide
new FPGS time by the old collision denominator. Runtime composition stays
source-frozen at9df743db; this ledger records results without invalidating the
replay source snapshot. General loaded robot trajectory coverage remains
bounded by the component controls, not a universal convergence proof.

The separate Q-or-N native probe passes all2048 hard checks and eight
translation controls (maximum scaled held-H error1.2105231e-5). There are675
pointwise diagnostics; nine native-only crossings exceed the normal threshold
by3.7e-8 to3.04e-6, not new hard failures. Tier32 replay is11-13% slower than
original and tier48 approximately flat, despite1.19-1.26x improvement over the
cached loser. Original-controlled whole timing is running; no speed claim.
Finite-shell query plus corrected CSR is also source-frozen for native/whole
testing, without old hash priority reduction or any new arrays/capacities.

## Shell-query composition, repeated/fair results, 21:38 UTC

The Q-or-N residual workflow is closed default-off at fork commit
`e7729d105b339e4ee6ac780b04af34fcc06feaa2`, clean and pushed. Whole physics
is9.225785400->10.124493750ms RTX and9.564807650->10.536172725ms GB,
reference/candidate0.911234243x/0.907806649x. Artifact
`/tmp/fpgs-anymal-spectral-residual-whole-paired16k-20260916-01` passes all
guards. The original recurrence has no full physical-merit scans to retire:
the92.47% reduction was relative to the cached loser. Lookahead, grouped
proposal, two reductions and persistent residual state remain additional work.
No third policy or instruction-tuning rescue is funded from this loss.

Finite-shell query plus corrected CSR is preserved at fork commit
`92cd3bc81000d82ff4eb1bfc010b3b5f9d0ef1c0`. Both native suites pass22 tests
and all18 loaded records per card, keeping CSR enabled on both loaded arms.
Candidate rebound vz1.8000001907m/s and energy ratio0.360000076294 match the
authored restitution; spin2.96623e-6rad/s. Full-friction travel0.50718540m
matches the existing0.509684m control. Production selected-anchor sliding
leaves the finite edge without observed interior fall-through. These remain
bounded fixtures, not universal trajectory equivalence. Native artifact
`/tmp/fpgs-shell-pair-csr-native-20260916-CxMjp4Ok`.

The first whole comparison against qualified chain passes all guards:
RTX15.044929850->13.782255325ms (1.091615958x),
GB19.807130175->15.882781250ms (1.247081973x), in
`/tmp/fpgs-g1-heightfield-shell-pair-csr-whole-paired16k-20260916-01`.
This1.262674525/3.924348925ms saving includes CSR membership and every changed
row/solve cost. The initial RTX environment wall result is slightly worse;
it is not relabeled an end-to-end improvement.

Root composes unchanged92cd runtime with9df spectral runtime at clean fork
commit `a61ea916ab55ee83109accf1a0360a02f7e83f7f`, branch
`ooctipus/fpgs-g1-shell-csr-spectral-20260916`, pushed. Cherry-pick is clean;
all three actual feature observers remain. Full pre-commit passes, and the
combined CPU suite executes16 passing controls with11 native selectors skipped.
Separate component native coverage is recorded above; no combined loaded-G1
physical-reference trajectory test is claimed.

First integrated whole screen passes all guards:
RTX15.045235725->13.175193875ms (1.141936572x),
GB19.847759200->14.887794550ms (1.333156441x), in
`/tmp/fpgs-g1-shell-csr-spectral-whole-paired16k-20260916-01`.
Three alternating repeats then reproduce the gain:

| Median whole physics, ms/env step | RTX | GB300 |
| --- | ---: | ---: |
| Qualified chain reference |15.087577775|19.844632975|
| Shell/CSR/spectral composition |13.205344050|14.907120800|
| Reference/candidate |1.142535758x|1.331218365x|
| Saving |1.882233725|4.937512175|

Candidate samples13.176627950/13.217565650/13.205344050ms RTX and
14.907120800/14.892250600/14.925237900ms GB. Median unprofiled environment
wall throughput improves1.074274917x/1.206008153x. Artifact
`/tmp/fpgs-g1-shell-csr-spectral-repeat-paired16k-20260916-01`.

Fresh fair backend comparison uses clean a61ea916 on both arms, fixed Lab53ee,
shared CSR1/shell1/adaptive0 and original calibrated16K200/40/40 recipe.
The reused adapter `/tmp/fpgs-g1-shell-csr-spectral-fair-iThUAefV` changes only
the selected composition, exact observer pin and shared shell flag, additionally
pinning all five prior adapter files. Original driver/checked shell are byte-
identical. All four runs exit0, capacity/construction checks pass, states remain
finite, budgets are unchanged, and both MJ warning masks remain0.

| Fresh whole physics, ms/env step | RTX | GB300 |
| --- | ---: | ---: |
| FPGS shell/CSR/spectral |13.220150900|14.941340800|
| Corrected MJWarp, same collision redesign |36.492822025|34.728007250|
| MJWarp/FPGS |2.760393758x|2.324289882x|

Artifact `/tmp/fpgs-g1-shell-csr-spectral-fair-paired16k-20260916-01`.
The shared collision change also helps MJWarp; dividing by an older denominator
would overstate the ratio. Other five task ratios are not newly remeasured or
claimed improved by this G1-specific dispatch. The4x-across-tasks goal is unmet.

Current twelve-step node attribution preserves the inherited auxiliary analyzer
rejection and proves48 physics/12 auxiliary roots with zero unproven owners.
Strict reader `/tmp/fpgs-g1-shell-csr-spectral-strict-R80XhYQP/read_shell_combined.py`
pins the prior complete reader, shell source/factory, actual requested/observed
flags and both interval allocation/status checks. It retains all original
source/metadata/unknown-owner guards. Output hashes are
RTX `a941e861d64a718890b31d375e874dbff9f4a5ed53c51dbc8aa24c2bd3db7bb0`,
GB `7881130690022e36426ab0c5b0abfeff3c8cad5d7350d44fd0170498182340ee`.

| Current exclusive node family, ms/env step | RTX | GB300 |
| --- | ---: | ---: |
| Collision |2.330609833|4.301366917|
| Dynamics |2.630962750|2.370376500|
| Rows/response |2.782452000|2.790419750|
| GS |3.221614750|3.250951500|
| Publication |1.842091583|1.932111833|

Diagnostic spans13.298025167/15.064496917ms include all remaining owners and
gaps. Shell query cuts collision versus the previous composition but adds row
and GS cost; the complete gain above includes both effects.

One new fixed-cohort stage-once representation now has four native test groups
passing on each GPU: original spectral transaction/current-held/graph-empty/
saved16 controls plus0/32/33/100-row and scalar-fallback boundaries. It stages
existing Z and immutable metadata only for actual counts<=32, preserves the
original>32 body and arithmetic association, and shares one3952-byte union.
AOT63/64 registers and4080 shared bytes, no spills, are charged on both paths.
Current eligible world share is approximately93%, not all worlds. The complete
paired cost screen is running against a61ea916; no gain is yet established.

Further source screens prevent repeated work. Exact horizontal-cell2-to-1 quad
merging would retire only21.7-22.0% of historical current-query calls; the
four-edge query and retained visits add work, with no credible1ms RTX budget.
ANY18 leaf-first L117 coordinates already have a complete earlier contiguous-
panel implementation, including rows, EX1, sparse recurrence and decode, that
lost9.295346->11.759834ms RTX. Neither is funded as a new implementation.
The distinct normal-only LCP elimination hypothesis gets one fixed-policy CPU
screen, with complete on-demand Gram-column costs, all-normal feasibility,
rank/work guards, full physical merit and literal remaining-budget fallback.
Its small observed normal working sets are not an admission oracle or a speed
claim; the original error is mostly tangential. No new converged reference.

## Staged-row closure and coupled-normal screen, 21:56 UTC

The staged-row whole screen passes every existing guard but misses the
structural-gain gate: RTX13.168334400->13.021441900ms (1.011280817x),
GB14.882789850->14.573015150ms (1.021256734x), in
`/tmp/fpgs-g1-staged-rows-whole-paired16k-20260916-01`.
RTX environment wall time is worse27.174752->28.218408ms; GB improves only
28.465651->28.074566ms. No promotion or mapping/tile tuning is funded.

One cause-directed node capture attributes exclusive GS3.221614750->
3.060125833ms RTX and3.250951500->2.890316833ms GB, with row and collision
families nearly unchanged. Thus removing repeated global row/metadata loads
does save solver time, but not the proposed approximately1ms whole budget.
Actual resources are63/64 registers and4080 shared bytes versus64/1116 in
the reference; no local spills. This is not hardware-counter evidence of a
cache/bandwidth bottleneck. Endpoint workload differs: RTX contacts89865->
89080/rows385818->383509, GB89429->89434/384219->384262. These are complete
capture costs, not an identical-input isolated kernel experiment.

Artifact `/tmp/fpgs-g1-staged-rows-node-candidate16k-20260916-01` retains the
inherited auxiliary analyzer rejection. Reader
`/tmp/fpgs-g1-staged-rows-strict-MZC6tvqu/read_staged.py` checks exact runtime
and observer pins, actual staged owner/flags,48 physics/12 auxiliary roots,
all process correlations, zero unproven owners and every original source,
status and capacity guard. Root's independent exact reader replay at
`/tmp/fpgs-staged-root-replay-q2nNeKBN` produces byte-identical hashes:
RTX `5c0be57df59c4449dd3e1d2edd2290c3837f358f175eb275b2a75d39d0380553`,
GB `6bf7aef39a2f171f1bf9eea5fdbc241e71b40faf5dcda9f3c9c5e83016df4c08`.
The first root invocation failed because the reader self-digest required a
physical copy at its output location; that failure is retained in tool output.
An exact patch-created copy then passed without weakening any assertion.

The fixed normal-LCP/tangent CPU screen passes all2048 existing hard checks,
with1981 closer held-H results and67 genuine held-H regressions. It still
requires45,474 passes (22.204 mean) and56.208M counted products, equivalent
to26.094 original action pairs before division/sqrt/reduction/branch cost.
Only613 cases reach the physical stopping gate. Stable normal working sets
do not eliminate normal/tangent coupling: the tangent action reintroduces
normal residual after the certified normal solve. No native implementation
is funded on this result. Full saved-input/reference guards pass in
`/tmp/fpgs-normal-block-cpu-oGrTia7y/all2048`; this does not establish generic
overflow admission or a new timing gain.

## Further structural falsification, 22:19 UTC

Staged-row closure is preserved at clean pushed fork commit
`24e5f298cd9523cb77777434ed4c2b589076fd7e`; all measured runtime/test/observer
hashes remain unchanged. The normal-block CPU helper, tests and report are
preserved at clean pushed fork commit
`2f70a1f7b41eed945806f7b73633babe198d6263`. Neither is promoted.

The unchanged simultaneous spectral proposal was transferred to the old16
pinned G1 current/held cases with its original G1 eight-pass cap. This is a
CPU falsification, not a new reference or current-population qualification.
At RTX7410 refresh, ordered->concurrent natural residual is0.000199078->
0.020362671, biased normal violation0->0.213406934, complementarity
0.001155671->0.050467060, and MDP0.000097995->0.024654545. Cone and momentum
remain sound; they do not excuse the much worse contact equation errors.
Artifacts `/tmp/fpgs-g1-concurrent-spectral-screen-FJbJr7zv/{CARD.md,check.py}`.
The first invocation lacked the concurrent helper on the G1 branch; a pinned
import of the existing helper corrected setup without changing its algorithm.

One cause diagnosis independently differentiates the same map, including
normal-dependent disk radius, and reproduces all16 eight-pass trajectories.
192 directional checks agree within4.006e-9 relative error. An observable
quotient excludes only invariant response-null impulse directions. Along
RTX7410's existing cold trajectory, half-relaxed observable eigenvalues reach
-1.70 to-1.76 while active normal counts alternate6/2/5/2; near its ordered-eight
endpoint the map is locally stable. This is off-target coupled active-set
overshoot, not a proved unstable final fixed point. GB8192 instead settles
onto a stable local map near0.735 but decays too slowly within eight passes.
No new damping/iteration grid or native implementation is funded. Full note:
`/tmp/fpgs-g1-concurrent-derivative-BvS9IN1y/CARD.md`; final diagnostic log SHA
`7f567f6e868fde96f269316e948a9ff6a32047286f51b20384f06629bf1c1f68`.

Two additional source budgets close without runtime work. Direct G1 ancestor
reconstruction replaces109 pose combines with226 and109 motion combines with
at least206, while removing only about five joins; bias/moments/projection
and publication remain. It cannot be credited with the whole1.84ms finish.
Card `/tmp/fpgs-g1-direct-ancestor-card-mG5KmX/CARD.md`.
G1 ancestor-prefix endpoint maps are a transfer of implemented Allegro
algebra, not new algebra. On the old16 cases, shared-root union reduces map
setup8424->7416 products; adding all emitted-direction endpoint contractions
gives31464 versus original31824, before lookup/incident/copy costs. There is
no new1ms case. The old Allegro report's native/lifecycle pass does not alone
establish its performance acceptance.

Temporal positive-distance warm GJK already existed in the old coherent
convex query. The corrected full-shell successor lost17.327->130.477ms RTX.
Penetrating MPR portal seeding is distinct, but its complete Kuka GJK+MPR
owner ceiling0.9607ms is below10% whole before any cache/revalidation work;
no AABB/broadphase/manifold/publication retirement follows. No implementation.

An independent final-state census uses the existing coherent full16K inputs,
not new captures: nonzero impulses passing the fixed physical gates occupy
7679 worlds/136048 rows RTX and7768/137771 GB (41.23%/41.65% of all rows).
Zero-impulse worlds are excluded from this possible early-stop opportunity.
This is own-rounded-Z FP64 diagnostic arithmetic, not native stopping or
independent J/H qualification; final convergence alone does not say when
those worlds could stop. Artifact
`/tmp/fpgs-g1-physical-stop-screen-6evKUP1W/{CARD.md,check.py}`. A same-trajectory
prefix/work screen is in progress, with check cost and original stationary
exits charged; there is no timing or native claim.

The next funded CPU hypothesis combines analytic coupled Jacobian actions
with feasible-chord globalization. It retains the existing small physical
active Gram, but removes Jacobian/LU construction, held-L reload and repeated
line-trial physical operators. A single chord response supplies affine
residuals for all eight line trials, each still paying its nonlinear full-map
merit check. Fixed max8 GMRES/CGS2, original24 committed outer allowance,
literal remaining-budget fallback and fresh final physical check are charged.
Card `/tmp/fpgs-anymal-krylov-chord-card-nJhiwwNd/CARD.md`, SHA
`79da7879550483df1553fd5ea8d1dc35a36aaa0db28c7a7207de2813604e1097`.
Root and independent review find no algebraic blocker for one frozen CPU
falsification. Actual inner/outer counts, physical tails and complete costs
must justify native work; no speedup or GPU authority follows from the card.

## Population-stop closure and coupled-prototype readiness, 22:34 UTC

The full16K prefix replay closes the G1 physical-stop route without native
work. It uses current spectral8 on the old coherent metric/terrain inputs,
not today's shell/CSR population. Zero-final-impulse worlds are excluded.
Compared with original exact-stationary/max8 exits, nonzero-world update
row dots fall1,643,273->1,396,789 RTX and1,648,436->1,398,740 GB:15.00%/15.15%.
Tangent proposals fall only7.20%/7.32%; lazy setup is effectively unchanged.
Even legal first-failing-row checks, omitting terminal/unchanged passes and
zero-disk tangent work, add644,378/644,968 dots. Complete net dots therefore
grow24.21%/23.98%, before projection, normalization, metadata and control.
No check-cadence grid or native implementation is funded.

Independent existing-helper checks on34 fixed worlds/card pass unchanged
tolerances; maximum impulse difference3.45e-7. The replay uses FP32 warp-tree
dots, not a claim of native FMA/bit identity or independent J/H qualification.
Earlier passing states need not remain passing after further allowed sweeps.
Full first-pass and stationary accounting is retained in
`/tmp/fpgs-g1-spectral-stop-population-6jOtK9kO/RESULTS.md`, SHA
`a37abc39024857988180fccc6e8007936e1b1cd14547509a06b91d350c6cff79`.
Replay SHA
`c347d9f252e97b785bc2a8e93eb9874ccdba226e327a1ebe7f232c6217a5f8bc`.
Root read the complete report. This is a work-count rejection, not a measured
GPU regression or a new gain.

The coupled Krylov/chord prototype passes10 CPU tests, including literal
original-reference translation, analytic moving-radius derivative, nonsymmetric
right-block solve, inactive RHS, feasible-chord counterexamples, held/current
publication, singular continuation, late23-correction budget and false-carried
stop. Root read the complete helper/tests/card; independent final review finds
no algebra or allowance blocker for the frozen first-eight screen.
Helper SHA
`88a2cc806b618deebc9c39a76f7fd2d987cdc0f9a3c2e70e4b906af8663efd5e`;
tests SHA
`301764615f82e07474f28e019690adb95c55fb699b7eb34d2efbe845da4b2d68`.
Review corrected omitted pre-CGS/control norm costs before cohort evaluation:
the actual q4 Krylov plus orthogonal reductions total28, not the card's24.
No numerical policy changed. The first-eight saved-workload screen is next;
no native authority or performance claim is established by these tests.

The frozen first-eight screen then passes all eight hard/pointwise checks;
all eight improve held-H reference distance and natural residual and stop
physically in3--6 corrections (34 total), with zero fallback. GMRES takes51
steps, q-squared95:22 directions useq1, sevenq2, fiveq3. All34 line trials
accept alpha1. Complete counted products148,052 equal17.13565 original action
pairs, versus338,904 original products including EX1. This misses the
optimistic10--12-pair planning gate; physical success is not a speed result.
Chord responses4.325, full Gram3.244, derivative1.693, Krylov Gram1.107 and
fresh checks1.000 action equivalents are substantial; local/preparation/merit
services make up the rest. No counterfactual triangular-Gram saving is counted.

Root authorizes one unchanged existing all2048 CPU run to resolve fallback
and cost tails, particularly historical world142: q1.5 and few outer
corrections support the central coupling hypothesis, while eight samples
cannot establish or reject the complete native case. This is an explicit
broader falsification despite missing the planning gate, not a gate silently
relaxed for promotion. No native work is authorized yet.
First-eight summary SHA
`ad73e3515fc77df41953eabd7cd9616c498412ba653d564c0a4d7ca56b2fd97b` at
`/tmp/fpgs-krylov-chord-cpu-622baUWd/first8/summary.json`.
The same card's pre-cohort clarifications make full-F forcing, exact inherited
norm Armijo and false-fresh-stop allowance semantics explicit; frozen SHA
`783517c6b3153af5564447c26a8cb0179ea2e1df4ec0421ec2a0d326e70f914a`.
Helper and numerical policy remain as reviewed above; actual counters retain
the correction from24 to28 reductions atq4 rather than the old card subtotal.

## Coupled all2048 result and causal globalization correction, 22:49 UTC

The unchanged broader CPU screen completes in about20s, with all2048 source,
input and existing-reference guards passing twice. Zero hard feasibility or
momentum failures;2017 pointwise checks pass,2032 results are closer in held-H,
12 are genuinely farther,2029 improve natural residual. Candidate corrections
7345 plus literal original fallback1454 total8799, versus24*2048 allowance.
1982 results pass the physical stop. There are67 fallback cases:58 exhausted
Newton-chord lines and nine true-linear forcing failures. Complete retained
traces, not a newly solved reference, are in
`/tmp/fpgs-krylov-chord-cpu-622baUWd/all2048/`; summary SHA
`e5b622fdb05fb02b1000ca68eebf95d84ff4b3b70bacd0eddf4694ccf7ab14fc`,
compressed trace SHA
`8d7fb438fdc7f1b5894082c7fed526c3fb333df5893f5921ddf2ba4c935eee35`.

Held-H scaled median/p99/max improve0.0063899/0.0319958/0.0557385 to
0.000002626/0.0102766/0.0302664. Not every component improves: the worst MDP
gap increases0.0794364->0.0897091, and RTX current48/world21 normal violation
increases0.00435972->0.0343440 after terminal fallback. These tails remain
explicit. All12 held-H losses occur in line-search fallback; the worst scaled
increase is0.00270612 (RTX held48/world232).

Counted products34.404153M equal15.97177 original action pairs versus84.268494M
original EX1+recurrence products. Krylov q=9876, q-squared18332,7412 true
linear checks,7971 line trials/626 rejections,2229 Gram builds/5183 reuses.
This is not a complete instruction or timing model: independent review finds
at least277462 additional retraction/slide products and72546 retraction roots
omitted from the frozen counters. Those corrections alone raise the proxy to
16.1006 pairs. Retained whitening/decode, additional scalar control, divisions,
square roots and reductions are not assigned free wall time. The original
product comparator also omits its scalar services.

One exact replay of all67 guarded states matches each complete stored work
record. All58 failed Newton directions have genuine descent slopes between
-1.000 and-0.920 relative to squared merit. Endpoint retraction changes24--95%
of their norm and makes50 chords ascend; the other eight need steps smaller
than the allowed1/128. One-sided differences confirm the analytic derivative.
The nine forcing failures exhaust the fixed eight Krylov steps, not a hidden
rank/nonfinite failure. Root read the complete causal replay source. Artifact
`/tmp/fpgs-krylov-chord-cause-HseE5K9Z/result.json`, SHA
`3754e203c3bf029ae01ac75a5cb4c2e028e6dcd605ffeb13b64644a3071a3e5f`.

The existing full-map projected endpoint supplies a merit-decreasing chord
within the original eight trial values at all67 frozen failure states. This
does not predict later trajectories. Root approves exactly one correction:
only after eight failed Newton-chord trials, try the old proven direction
pg-x with one additional physical response and up to eight same-Armijo trials.
Because both endpoints are feasible, every chord point is feasible; the old
per-trial cone retraction is redundant in exact arithmetic. This restores the
prior safeguarded direction without its repeated line-trial physical operators.
Keep forcing/rank/nonfinite/capacity guards, original24 committed corrections,
initial forcing norm, cache epochs, fresh check and remaining-budget fallback.
No native authority yet; freeze and test this one correction before judging
complete cost or transfer to G1's separate original eight-pass allowance.

## Corrected CPU checkpoint and bounded native cost authorization, 23:08 UTC

The single projected-direction correction is preserved at clean pushed fork
commit `c1b00ae5b4fe682481f574569f2307fa57b0020c`; initial CPU implementation
remains independently preserved at `31db487eae7bb093cd25733db07bc18ff98edcfe`.
Thirteen regression controls pass, including exhausted Newton trials followed
by an accepted projected chord, both directions failing after a prior commit,
zero allowance and nonfinite-trial routing. Full and owned-file pre-commit pass.
Corrected helper SHA
`05d97ac7c04e13b835235ceaf1cf2e9d2c97ff2d7191d58852f85f9aff5b269b`,
tests `74446257733338d7a423aa73e04c5c615a450345b691483a91086c48b92cff0c`.
Root and independent review read the complete changed source/card/tests.

Corrected all2048 has zero hard failures,2029 pointwise passes,2041 closer
held-H results and five genuine held-H regressions.2017 pass the fresh
physical stop. Scaled H median/p99/max are2.312e-6/0.00584668/0.0268277.
7625 candidate corrections plus607 original fallback passes total8232.
There are137 safeguard directions,398 retry trials/120 accepts, with169956
extra physical-response products.30 terminal fallbacks remain:17 line and
13 forcing. Actual Krylov q10961/q-squared23919;9405 total line trials include
1780 rejections. Counted products34.318395M equal15.93196 action pairs;
known omissions add292802 products, raising the lower bound to16.06789.
This remains a proxy, not a measured GPU gain.

The correction DOES NOT pass the old all-case pointwise component gate.
Of19 misses, nine follow line fallback, nine forcing fallback, and one
exhausts all24 candidate corrections. Worst scaled H loss is GB current48/
world168:0.0186851->0.0268277; RTX held48/world142 is0.0218130->0.0246959.
The other three H losses are about4.51--5.11e-5. RTX current48/world21 normal
violation increases0.00435972->0.0378981 after forcing fallback. RTX held32/
world224 exhausts24 with complementarity0.0103814->0.0243626. Aggregate and
MDP maxima improve, but do not hide these genuine finite-budget losses.
Artifact `/tmp/fpgs-krylov-chord-safeguard-HV6mVO8t/all2048/summary.json`, SHA
`c23294122313374c0873ddcce553af4e04a2a28692267a27dd77b4d396053ce5`;
compressed trace SHA
`815d93d4e78865c9c23bcd8f463a47898d29a21dc89c5f9601836673208b837a`.

Root explicitly funds ONE default-off native COST-FALSIFICATION prototype,
not promotion or full numerical qualification. The low coupled inner count,
many physical stops and roughly2.45x counted-product retirement justify an
early integrated measurement despite missing the optimistic planning proxy.
Further pure-CPU globalization tuning before that measurement risks repeating
the prior validation-first failure pattern. No third numerical policy is
funded. Native hard physical, implementation, budget and lifecycle controls
precede timing; known component/reference tradeoffs stay separately reported.
Only a substantial complete saving would fund deeper numerical qualification.
The first-integrated target is00:30 UTC; a miss must be explained, not hidden.

Native tree `newton-fpgs-krylov-chord-native-20260916` branches from c1b00ae5.
The linear fragment and outer owner are implemented independently; root adds
the opt-in constructor and original-protocol observer. Active<=18 GMRES can
remain within warp0 with two enclosing CTA handoffs; the old LU was already
warp-local, so warp-locality itself is not a new saving. Additional scratch
and resource pressure remain charged. Held L is never reused as a Jacobian
scratchpad. Original row staging, whitening and publication are retained.
The new observer explicitly labels the unresolved numerical qualification,
checks actual kernel identity and unchanged24/settings, and keeps all original
source/capacity guards. Three observer controls pass after a missing-module
regression; root-owned pre-commit passes. No GPU work or new timing yet.

## Native first cost and G1 transfer closure, 23:36 UTC

The corrected policy's unchanged G1 transfer is not funded for native work.
The initial 16-case port retained the borrowed ANY 64-row guard. Separately
raising only that CPU admission to G1's original 100-row capacity exposed the
four excluded cases, without changing runtime capacities, active18, or G1's
eight-correction allowance. All four hard checks pass, but RTX current world7410
held-H error worsens 0.00270058 to 0.0453928; the current/held RTX cases use all
eight corrections, 34/46 Krylov steps and 45/68 line trials. The held case does
improve H error 0.152670 to 0.0492656. Two GB cases stop after two corrections,
but computed stopping is not identical to every independent diagnostic gate.
This is a genuine quality/work limitation, not an unsupported-row success.
External results SHA `60d471fd71e01749914f3497aa309447e710bd9e4bae1b367f20a8b1d5b00128`
at `/tmp/fpgs-g1-krylov-chord-transfer-t231GMpM/RESULTS.md`.

Native outer SHA `bca38cf5a2b624533f59f478d564831f1655a976d6bf89efb21b1974b8b2e0de`
and linear SHA `d68534d5a8abe82901f63707d3d27a661296aec5344dcf75fa3b1debab964249`
pass root and independent source reviews. The first AOT compile exposed Warp's
int/float cast macros; static_cast fixes that exact failure. Explicit nonfinite
map propagation prevents CUDA fmaxf from masking an invalid line trial.
Full actual 72/32-capacity owners compile on both architectures: RTX tier32/48
use 96/128 registers and 9572/11912 shared bytes; GB uses 97/128 registers and
8272/11912 shared bytes. All four have zero stack/spill. No occupancy/stall
claim follows from those static resources. AOT report SHA
`c224ce8c1a2fa9a063699704cd5cd8f52acdc800db53129592f3391b9bebfaee`.

Three linear CUDA controls pass on RTX. GB passes the same three plus actual
original fallback with incoming impulse/delayed friction on both staged tiers.
The existing GB saved probe completes all 1024 cases with zero hard failures,
ten pointwise diagnostic records and four CPU translation H errors at most
1.342e-5. The ten diagnostic IDs are a subset of the frozen CPU's eleven GB
IDs, not evidence that all finite-budget components are qualified.

Complete restored-operator replay is substantially slower:

| GB saved partition | Original ms | Candidate ms | Original/candidate |
|---|---:|---:|---:|
| current32 | 0.050219 | 0.227130 | 0.2211x |
| current48 | 0.054536 | 0.565464 | 0.0964x |
| held32 | 0.050224 | 0.269453 | 0.1864x |
| held48 | 0.054528 | 0.356765 | 0.1528x |

Artifact `/tmp/fpgs-krylov-chord-native-probe-t9F4yYPG/gb-probe.log`.
These 512-world, restored-output component timings are NOT whole-physics
ratios. The worst frozen CPU current48 case already requires 15 corrections,
80 Krylov steps, sum(q squared)=490 and 97 line trials. Native iteration blowup
is not established. Serial reductions, control and long-tail completion are
credible causes despite lower counted products; one phase/counter diagnostic
is authorized to distinguish them before proposing a structural correction.

The first whole GB attempt stopped at its initial idle guard: an unrelated Kit
job occupied GB after preparation. No capture or whole timing was produced;
failed artifact `fpgs-anymal-krylov-chord-whole-gb16k-20260916-01` is preserved.
RTX also has unrelated simulator jobs. Root requested a reserved GPU and does
not kill them or weaken the idle check. The root-owned runner now accepts one
or two distinct GPUs, with the exact original sampling, source, budget,
capacity and idle guards unchanged. Four observer/CLI controls pass; scoped
pre-commit passes. This scheduling adaptation changes no Lab or solver code.

## Native whole-cost closure and next ownership screen, September 17, 00:40 UTC

The first complete RTX Krylov/chord comparison finished at 23:46, before its
00:30 checkpoint. All source, idle, capacity and unchanged-budget guards pass:
original physics 9.222293025 ms, candidate 27.762799075 ms, or 0.332181672x.
This is an 18.540506050 ms loss, not accepted progress. Environment wall time
also worsens, 17.281787150 to 34.908291101 ms. The GB whole run remains an
initial-idle-guard failure with no measurements; redundant whole reruns are
not required to establish this decisive RTX regression. Artifact
`/tmp/fpgs-anymal-krylov-chord-whole-rtx16k-20260916-01/manifest.json`, SHA
`23f76e015d80d0a24b072d0adb0c667f2ee874ea44584ca21754e1d8b6207f50`.

The one cause-directed diagnostic is complete. On 1024 RTX saved cases the
instrumented outputs are bit-identical to the uninstrumented candidate, and
all eight per-world work counters exactly match the frozen CPU records:
3809 committed corrections, 3823 linear calls, 5468 Krylov steps, 11792 sum of
Krylov-depth squares, 1107 Gram builds, 186 projected trials, 283 old passes,
and 4676 total line trials. Native iteration blowup is not the explanation.
The linear service alone executes 35842 dot/norm reductions, each with five
shuffle stages. Lower product counts did not remove their sequential control
and communication cost. Diagnostic instrumentation raises registers to
167--168; its within-world cycle fractions are NOT production wall-time shares.
Root read the complete diagnostic, count-join, report and native source.
Diagnostic result SHA
`4de29b0310b6f2c3563905ca90bd17d4b1c4932cc44afc66541365f25ca50107`
at `/tmp/fpgs-krylov-diagnostic-HKxyZtZT/rtx01/result.json`.

The default-off prototype, tests and cost rejection are committed and pushed
as `b189e905` on `ooctipus/fpgs-krylov-chord-native-20260916`. Full and owned
pre-commit pass; the CPU suite has 20 passes and four CUDA skips, with native
linear/fallback tests separately passing on both GPUs. No candidate runtime
has been promoted, and no accepted physics timing has regressed.

Additional bounded source screens close without implementation: field-major
consumer packing already lost even with a free host transpose; integer-redux
floating sums add scaling/conversion/control without a defensible latency
advantage; forward prefix scans and finish-to-begin geometry coherence are
already implemented; exact limb/root derivative elimination encounters
singular local blocks; static terrain coefficients retain the main geometric
work and must support live elevation mutation. No micro-optimization grid is
funded from these findings. In particular, the old supernodal report's 17.3K
figure counts static SASS instructions, NOT shared-memory bytes; its already
measured shared-front correction uses 7336 bytes and remains below the gate.

One inexpensive complete ownership screen is authorized: pack four independent
full 32-lane G1 worlds per 128-thread CTA in BOTH spectral solve and limit
prefix, preserving all numerical work, capacities, interfaces and publication.
Actual RTX attributes limit one-warp blocks to 24 resident warps; the current
64-register solve permits 32 with this grouping. The optimistic residency-
scaled solve-plus-prefix saving is approximately 1.03 ms, not a promised gain
or a hard timing bound. Although below the earlier 1.2 ms planning estimate,
it is close to the structural 1 ms gate and cheap enough for one early whole-
physics falsification. No mapping sweep or factor redesign is authorized.
New isolated branch: `ooctipus/fpgs-g1-fullwarp-group-20260917`, based on a61.

Unrelated Kit jobs continue cycling on the GPUs. Root uses only verified idle
windows, has requested a reserved device, and does not kill those jobs or
weaken measurement guards. The twenty-hour window remains active until
07:04 UTC; the four-times-across-tasks goal remains unmet.

## Full-warp closure and two distinct structural screens, 01:20 UTC

The four-world/full-warp grouping screen is complete and rejected, not
promoted. On RTX the complete guarded comparison is 13.144145525 ms original
to 13.961268150 ms candidate (0.941472177x); wall time also worsens. The strict
48-physics/12-auxiliary-root trace places the loss in GS: 3.22163875 to
4.048535 ms, while limit-prefix improves 0.5600134 to 0.520920 ms. All eight
solve/prefix launches use the intended 128-thread/4096-block ownership, old
keys are absent, capacity/source/idle guards pass, and registers do not grow.
On an older source-matched population, grouping consecutive worlds exposes
1.633x dot-work inflation from unequal completion lengths. This is a causal
proxy, not a current-population stall measurement, but its magnitude explains
why more reserved warps did not yield more useful throughput. A persistent
queue would add reset/atomics/guards to an already sub-1-ms solve-only ideal;
no queue or mapping grid is funded. Both cards pass four native correctness
selectors, including odd-tail, remapped worlds, capacity and graph cases.
Default-off branch `ooctipus/fpgs-g1-fullwarp-group-20260917` is pushed as
`932e15b3c259cfa631c9622e4916db8b0c79e590`, with full/owned pre-commit passing.
Whole artifact: `/tmp/fpgs-g1-fullwarp-group-whole-rtx16k-20260917-01`.
Strict reader: `/tmp/fpgs-g1-fullwarp-strict-3xS3kf6f/read_fullwarp.py`.

Franka's new compact-workspace screen preserves the original two-world
half-warp execution and all arithmetic, scans, current/held lifetimes and
public state. It overlays dead pose/motion slots with later wrench/moment
storage; padded-body slots hold final actions without overwriting live body
terms. Compiler shared memory falls 7296 to 3136 bytes, with repair/finish
registers unchanged at 64/80 and no spills. This is not the earlier rejected
scalar-world/lane streaming, which added joins and serialized worlds. CPU
physical/lifetime tests and four GB native selectors pass. The first GB whole
attempt stops after baseline and before candidate when an unrelated training
job takes the device; final idle guard is false. There is NO valid whole
comparison and no claimed speedup. The incomplete run is preserved at
`/tmp/fpgs-franka-compact-workspace-whole-gb16k-20260917-01`.

G1's second screen changes only the joint-limit prefix to one simultaneous
projected update, retaining ordered contact/friction work and the original
maximum eight outer passes. A finite negative quadratic-energy guard commits
the complete prefix transaction; rejection rolls back all scratch and runs
the original scalar prefix within the same pass. There is no contact-Jacobi
map, Gram/factor, line-search grid, new global array, extra launch or new
converged reference. Fixed four-lane row groups use existing support, followed
by deterministic node-owned W gathers; all added reductions/indices/fences
are charged. The saved 16 current/held cases pass independent physical
checks. A preselected 1268-case interacting/uniform CPU cohort has zero hard
finite/cone/momentum failures, but small finite-eight residual tradeoffs
remain explicitly recorded. Uniform samples remove approximately 34.4% of
ALL original dot-reduction stages and most serialized prefix fences, while
outer passes rise about 1%; these are source-path counts, not GPU timings.
The actual 3.2216 ms GS owner must reach at most 2.2216 ms to save 1 ms.
One early native cost screen is authorized on new a61-based branch
`ooctipus/fpgs-g1-limit-jacobi-20260917`. No further CPU policy sweep is funded.
Frozen CPU decision card:
`/tmp/fpgs-g1-limit-block-card-GowrgB7P/CARD.md`, SHA
`f2445ed73fbf3fbbecf6406bf33d5fa6813d7c30609339c7fd0efaa05a2a68ad`.

Both GPUs are occupied by unrelated jobs at this checkpoint. Root requested
an uninterrupted measurement window and continues implementation/review;
no sudo, job termination, Lab change or weakened idle guard is used.

## Two measured RTX gains and one causal correction, 02:15 UTC

Franka compact workspace passes all four native selectors on both cards.
RTX native correctness ran under external load and is not timing evidence.
Two complete idle/source/capacity-guarded RTX comparisons measure:

| Run | Original ms | Compact ms | Original / compact |
| --- | ---: | ---: | ---: |
| 01 | 5.108246725 | 4.797041475 | 1.064874413x |
| 03 | 5.049426675 | 4.864011150 | 1.038119881x |

These are 0.185--0.311 ms savings, below the intended approximately 0.5 ms
Franka structural milestone. Preserve the useful lifetime redesign without
a mapping grid; do not present the larger first result as a stable saving.
Run 02 ended with a baseline initialization segfault before physics reporting;
the candidate never ran, and retry 03 completed. The first GB whole run and
first RTX paired node run each stopped between arms when an unrelated job
acquired the selected GPU. Neither incomplete run supplies a comparison.
GB whole retry is currently running. The default-off nine-file checkpoint is
pushed on `ooctipus/fpgs-franka-compact-workspace-20260917` as
`8ec4db342c0c5596060e89ef4923412a4a0ef1c1`; full and owned pre-commit passed.
Whole artifacts use `/tmp/fpgs-franka-compact-workspace-whole-rtx16k-20260917-01`
and the same prefix ending `-03`. No fresh MJWarp denominator is claimed.

G1 limit-only Jacobi's initial W-column response mapping loses whole RTX:
13.190505650 to 13.370332925 ms. Strict node attribution finds GS growing
3.22163875 to 3.39099633 ms, accounting for almost the whole 0.179827275 ms
loss, while other families remain flat. This was not rejected merely because
the first implementation was slow. One cause-directed correction replaces
43-node index probes and irregular W gathers per changed row with contiguous
actual-Z scatter into a 43-value shared response. Prefix proposals still use
the same old state; the same numerical descent guard commits or rolls back
the entire update before the original ordered contact work. No iteration,
contact law, global buffer or launch is added. Scratch shrinks 688 to 516 B;
each changed-row scatter explicitly orders shared-node overlap.

Corrected native tests pass all three selectors on each card; GB native ran
under external load and provides correctness evidence only. Independent
source reviews pass. The integrated RTX correction measures 13.160521725 to
12.535854600 ms, 1.049830438x and 0.624667125 ms saved, with all source,
capacity and idle guards passing. This is a useful one-run gain below the
original approximately 1 ms screen, not a repeated or cross-card result.
The finite-eight numerical tradeoffs in the preselected CPU cohort remain
documented; neither those tests nor finite whole-run state prove arbitrary
long-horizon convergence. Initial loss is retained at `add5433f`; corrected
runtime SHA is `5a15a0d64f914466df6e847426f5b9e83173d33fab490ada4aaa04c6211e3ac0`.
Artifact: `/tmp/fpgs-g1-limit-jacobi-zscatter-whole-rtx16k-20260917-01`.

A bounded adaptive-held-mass study does not justify another prototype.
The cheap operator perturbation bound is 0.546--2.386 on seven adjacent saved
epochs, despite actual energy perturbations of 3.04--6.97%; it cannot certify
a small longer-hold defect. Halving scheduled refreshes saves at most 0.890 ms
of the current 1.781 ms factor owner before current-H/check/cache costs.
This is not evidence that the original interval-two integration is wrong.
Card: `/tmp/fpgs-heightfield-static-geometry-kF1HG0SL/ADAPTIVE_HELD_MASS.md`.
Source screens also find no credible standalone large saving from bundled
contact dots or a separate dense free-root publication kernel. No tuning
grids or additional proof framework are funded. The all-task 4x goal remains
unmet; Lab sources and physics allowances remain unchanged.
