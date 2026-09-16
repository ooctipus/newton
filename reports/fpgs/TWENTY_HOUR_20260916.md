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
