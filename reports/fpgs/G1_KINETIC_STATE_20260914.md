# G1 current/next compact kinetic state

Pre-code card, 2026-09-14 22:58 UTC. Base qualified `1a9efc33`;
default-off `FEATHER_PGS_G1_KINETIC_STATE=1`. First checkpoint 00:28 UTC;
no silent extension beyond 00:58. No Isaac Lab, timestep, substep, capacity,
contact law or eight-sweep changes. No GPU work before root's allocation.

## Complete boundary and prior art

Retain the 434-value held inverse whitener, its 15-level factor/inverse,
all sparse contact/limit preparation and the metric-tangent GS/decode.
Replace original generalized integration plus template FK/publication,
body-dynamics finalization, current cached body wrenches, grouped torque,
both composite-inertia routes and full-inertia masked repair with one
current/next compact state producer and current-force/W predictor.
The new refresh reads geometric434 plus original R/current positive K.
No full spatial inertia, composite inertia or body wrench is an output.

The closed G1 spatial experiment (`cd0ddcb1`, cadence fix `5c90b0c6`)
instead replaced W and every consumer with a 502-float articulated operator
while retaining public FK/publication. Its corrected whole loss is not a
measurement of this boundary. The present experiment does not reopen ABA,
implicit row transforms or a spatial mapping sweep.

## Math, lifetime and required outputs

Each body's moment is `(m, h=m*r, Q=R I_com R^T + m[(r.r)I-r r^T])`.
Keep all nine Q entries, including model-inertia asymmetry. Parent-edge sums
give exact subtree moments. For a screw `(l,w)`, inertia action is
`(m*l+w cross h, h cross l+Q*w)`; use the existing 434 source/projection map.
The actual 44-body tree has depth10 and43 edges: moment reduction takes559
scalar additions on refresh, bias-wrench reduction258 each state. Preserve
all six welds and free-root inertial/Coriolis terms.

Cache current bias43, current screws43x6, COM offsets44x3 and current origin.
Publish original public generalized q/qd and body pose/COM velocity.
Geometric434 is a separate current-state generation; held W changes only
when the original mass mask requests it. Current external COM wrenches,
passive force, control.joint_f and already-clamped augmented u0 are consumed
at prediction, never cached across controls. Preserve original free-root
predictor and integration transport. Current S remains the contact input.

Reject unsupported/changed ownership before retired-cache reads. Preserve
model notification, reset, source-bank and generation invalidation; a late
mass request repairs missing current geometry. Consume the original mass
request/force flag tail (the old spatial integration's diagnosed bug).
Existing sparse admission excludes prescribed bodies; the old row metadata
helper reads body_v_s only for prescribed articulations. No diagnostic mode
may silently read a retired canonical array.

## Conservative complete cost gate

Existing source-pinned RTX diagnostic: dynamics exclusive4.893914333ms plus
publication2.415289333ms =7.309203667ms. Reserve the ENTIRE current factor
node2.342925ms, original drive preparation0.154144ms and mask0.027829333ms.
Do not convert overlapping owner sums or operation counts to saved time.
For a2ms whole saving the complete replacement allowance is2.784305334ms.
Planning allocations: finish<=1.90, current force/predict<=0.60,
repair/epochs<=0.20ms, total2.70ms. These are falsifiable budgets, NOT timing
predictions. G1 prediction needs no Kuka endpoint-twist scan: sparse rows use
current S and v_hat directly. Cross-warp44-body state/reduction dependencies
and native resource growth are the principal risks.

Use existing current/held G1 fixtures and physical oracles, including changed
frames, COM, forces, controls, refresh/reset and actual graph lifecycle.
Regression-first admission, CPU/offline SM120/SM100, then early paired
physical and original whole owner. No new benchmark framework. Measure the
entire finish/predict/repair family and all retained stages. A loss receives
one causal diagnosis, not a block-size or parameter grid.

## Source/offline checkpoint (23:28 UTC)

Missing-owner regression failed before implementation (CPU session90834,
ModuleNotFoundError). Runtime ownership is split only for development: native
math/state in g1_kinetic_state.py, constructor/binding in g1_kinetic_owner.py,
original solver/sparse hooks and unchanged row/GS consumers. Two device bank
pointer tags perworld (256KiB at16K) prevent stale private-cache reuse between
different captured state-pair graphs. Status is sticky; constructor/source
invalidation does not erase a recorded native failure.

Original offline compiler reused at
`/tmp/fpgs-g1-kinetic-state-offline-S8ptYf/compile02.py`, adapter SHA256
`364e1aea0913fd903af3b6dff655409642ec98356f43021cc8355bdb51984cd3`.
The host owner import was explicitly stubbed while that file was being built;
only real native generators were compiled. This tests no admission/lifecycle.
All8 SM120/SM100 entries compiled; report/pins are in `offline02/report.json`.
Offline01 preserved the Warp finish array-alias tuple-assignment failure;
using original Kuka-style separate q/qd assignments fixed code generation
without changing equations.

| Kernel | SM120 registers | SM100 registers | ptxas shared | Stack / spills |
| --- | ---: | ---: | ---: | --- |
| repair44 |80|95|8000B|72B /0|
| finish44 |91|95|8000B|72B /0|
| current-force predict43 |48|40|1660B|0 /0|
| geometric sparse factor434 |40|39|2252B|72B /0|

These are native resource results, not achieved occupancy, local-memory-free
claims or timing. Actual owner CPU/current-held physical and integrated whole
cost remain required. No GPU work was performed by this agent.

## First complete checkpoint (2026-09-15 00:28 UTC)

Runtime `9a2d417f82116288eec9e9c3ed58975dd6271c7e`, test-only followups
through `bf3a3c8adfee7aa5ce3cc4b1c8ad5ba495711484`; qualified baseline
`1a9efc33efbc0f23e1e7676a5edded795f224c97`. The runtime did not change
during these checks. This is a measured discovery gain, **not promotion**.

| Per-environment-step physics, 16K | RTX PRO6000 | GB300 |
| --- | ---: | ---: |
| Qualified baseline |18.205599 ms|20.909588 ms|
| Compact current/next state |15.658029 ms|20.464849 ms|
| Baseline / candidate |1.1627x|1.0217x|
| Time removed |2.547571 ms|0.444739 ms|

The unchanged graph-level protocol used seed0, 200 warm steps, 40 wall steps
and 40 profiled steps, with two Newton substeps and eight-sweep allowance.
Both arms use the same sparse level/metric/parallel-limit and terrain-query
options; only `FEATHER_PGS_G1_KINETIC_STATE` changes. Lab stays at corrected
`53ee6b44c2334341305dbdf385a3916c6b140799`. Capacities remain dense100,
raw294912, broad49152 and triangle1769472. Wall times, not RL training, were
32.526046 ->29.785007 ms RTX and36.179844 ->34.203834 ms GB.

Whole evidence: `/tmp/fpgs-g1-kinetic-state-paired16k-20260915-02`.
All four child processes exit0 with no cleanup signals; all eight boundary
checks pass actual source/owner, finite state and native capacity/status
checks. Each arm has160 physics and40 auxiliary direct graphs. Independent
review rehashed all16 pinned helper/source files and found no CUDA error or
traceback in the four capture logs. One round does not estimate timing spread.

The preserved `...-01` discovery attempt stopped in the external observer,
which incorrectly required every current cache to be valid after Lab resets.
The actual owner intentionally invalidates reset worlds and repairs before
use. The observer now accepts Boolean validity, checks generation for valid
geometry and retains sticky native status checks. Whole02 has331--358
reset-invalidated worlds at boundaries with status0. This observer correction
does not change solver code or excuse a failed native read.

### Disjoint node diagnosis

Original strict node membership reader was reused. Each of four captures
passes12 physics/3 auxiliary roots and zero unproven correlations; candidate
has768 physics nodes versus1020 baseline. The original generic Lab analyzer
rejects auxiliary graphs in node mode, so both node parent manifests remain
failed. These independently resolved nodes are diagnostic evidence, not a
replacement for the successful whole-graph timing above.

Baseline: `/tmp/fpgs-g1-kinetic-state-nodes-paired16k-20260915-01`.
Candidate: `/tmp/fpgs-g1-kinetic-state-nodes-candidate-paired16k-20260915-01`
(first pair labeled baseline by that parent, but explicitly candidate source
and flag1; do not mistake the label for old code).

| Exclusive complete replaced boundary | RTX | GB300 |
| --- | ---: | ---: |
| Original current dynamics + publication |7.308948 ms|5.184853 ms|
| Candidate current dynamics + publication |5.054775 ms|4.917256 ms|
| Candidate finish / predict / repair sum |3.148854 ms|3.329983 ms|
| Retained factor, original -> geometric input |2.333621 ->1.781323 ms|1.765184 ->1.505024 ms|

RTX finish2.329569 +predict0.632704 +repair0.186581 ms exceeds the2.70 ms
planning allocation. The geometric-input factor recovers0.552298 ms against
the conservative full-old-factor reserve. No old torque, FK/finalizer,
composite-inertia, predictor or conversion producer remains. GB's old
boundary was already2.124095 ms cheaper, and its new family costs more;
the small GB saving is therefore not evidence of an unconsumed refresh flag
or hidden duplicate pipeline. Launch counts alone are not active-world
refresh counts: native cache/mass guards can early-return within a launch.

### Physical status and remaining gates

Five CPU tests pass. On both GPUs all three focused physical methods pass
their assertions: current/held H and force prediction,16 saved loaded-contact
states, complete five-world eager steps, masked reset/model notification,
source-bank replacement and graph contact sequence0/5/0/5. Saved physical
contact momentum defect is at most4.257338e-8 and cone violation7.45e-9.

An independent FP64 public pose/COM-velocity oracle replaced generic FP32 FK
as the public velocity reference. At saved root translations35--76 m, generic
and original serial FK lose accuracy by world-position subtraction. The
candidate satisfies the original component gate against FP64 on all16 saved
states. A separate near-zero angular component in complete stepping cancels
ancestor terms with condition~285.57; the final public velocity gate is a
coordinate-invariant, separate linear/angular three-vector error divided by
`1 + norm(reference)`, bounded by3e-6. Observed maximum1.412238199e-6.
This is an explicitly changed physical error metric, not bit-identity or an
unchanged elementwise tolerance claim. Pose and downstream physical gates
remain unchanged.

Standalone CUDA tests still intermittently crash or emit CUDA709 during
cyclic solver destruction, including explicit `gc.collect()` before shutdown.
All physical assertions can pass before this failure; that does **not** make
the test process clean. A destructor-order instrumentation control exited0
but detected no already-destroyed-stream use, so that cause is not established.
Do not blindly patch weak references: the original solver also stores itself
as its debug augmented state. Minimal original-versus-new lifecycle diagnosis
and a clean paired rerun remain required, as do repeated whole timing and
representative retained-path checks before adoption. The all-task4x target
remains unmet; these results concern G1 only.

### Teardown repair and clean physical rerun (00:48 UTC)

The isolated original/new eight-lifetime probe passes on both cards, but the
full unchanged three-method suite reproducibly exits139 on both. Faulthandler
locates the crash in `SolverFeatherPGS.__del__ -> wp.synchronize_stream`, during
explicit cyclic collection. The local Warp stream finalizer destroys its CUDA
handle without clearing the surviving Python wrapper's handle. The solver
therefore must not assume that all referenced stream wrappers remain usable
when cyclic finalization reaches it. Exact finalization order was not observed
in the earlier instrumented run; the full crash location is now established.

A causal control changes only the destructor to drain the device context,
leaving the entire physical suite untouched. Both processes exit0, all three
methods pass (5.133/5.235 s),40 solver owners drain, and no CUDA/error/exception
is logged. The actual Newton destructor now uses that context synchronization,
handling partial construction and unavailable shutdown callables. It changes
no timestep operation or native physics kernel; at destruction it may also
wait for unrelated work in the same device context.

The new CPU teardown regression fails on the original implementation and
passes after the repair. Combined with the five existing CPU methods, seven
tests pass. The actual edited source, without the control monkeypatch, passes
all three CUDA physical methods on both GPUs in3.888/3.948 s; both processes
exit0 without CUDA709 or crash. Logs and the preserved failing controls are in
`/tmp/fpgs-g1-gc-isolated-lNKjkZ`. This closes the observed teardown blocker;
repeated whole timing and retained-path qualification remain before adoption.

### Retained-path CUDA qualification

On report tip dcb71e5b, the six unchanged Allegro/Kuka integration selectors
listed in `FOURX_QUALIFIED_INTEGRATION_20260914.md` all pass on both cards:
16.978 s RTX and17.357 s GB, sessions4225/89148, clean exit0 with no skips.
They exercise retained current/held rows, parallel response, complete fallback,
reset, actual stepping and two-graph transitions; they do not replace G1's
separate physical tests or establish new whole-task timings. The only warning
is the inherited target-layout deprecation. This confirms that adding the
default-off owner and generic teardown repair retains those checked paths.

### Three-round paired confirmation, 01:04 UTC

The existing checked alternating parent completed all twelve children at
baseline `1a9efc33efbc0f23e1e7676a5edded795f224c97` and candidate
`58063eea6163f87b9310a9ef8ddc3e777fd2decc`. Both Newton trees were clean and
actual imports matched the pins. Unchanged Lab `53ee6b44` had only its known
untracked environment directory; no Lab source changed. All sixteen helper
hashes recheck, all children exit0 without cleanup signals, and final source
and idle guards pass.

The run retains16,384 worlds, seed0,200 warmup,40 synchronized wall and40
profile steps per child; sim dt0.005, decimation4, two Newton substeps at
dt0.0025 and maximum8 GS sweeps. Both arms retain metric sparse W434,
parallel limits, level factor and the same finite/weld/packed/cull collision
owners. Only the kinetic-state selector differs. Dense100, rigid294,912,
broad49,152 and triangle1,769,472 capacities are unchanged.

| Median of three complete-physics runs | RTX | GB300 |
| --- | ---: | ---: |
| Baseline |18.188704850 ms|20.916695625 ms|
| Kinetic current/next producer |15.657344400 ms|20.503289775 ms|
| Saving |2.531360450 ms|0.413405850 ms|
| Baseline / candidate |1.161672x|1.020163x|
| Baseline range |18.149767--18.245808 ms|20.887191--20.938246 ms|
| Candidate range |15.651164--15.712527 ms|20.488669--20.516612 ms|

This confirms13.917% lower RTX physics time and1.976% lower GB time.
Unprofiled whole-environment wall medians are32.453053 ->30.327333 ms RTX
and35.407030 ->35.022617 ms GB; these include host/reset work and are not
physics or RL-training throughput. Every capture independently validates
160 physics and40 auxiliary graph launches, totaling1,920 physics roots
and480 separately excluded auxiliary roots. All24 finite/capacity/owner
boundaries pass; no unsupported collision owner or sticky failure appears.
Post-reset invalid private caches are explicitly observed and legal: the
next native use repairs them. The twelve driver logs contain no warning,
deprecation, traceback or CUDA-error matches under this limited log scan.

Artifact: `/tmp/fpgs-g1-kinetic-state-repeat-paired16k-20260915-01`;
manifest SHA256
`26c9b353b807694e4415733b98c9d1873ef0afb593e9b2cb9f653cc42ea30fdd`.
This closes repeated timing, not a new convergence or cross-task qualification
claim. The focused physical evidence above and clean teardown remain separate
gates; representative retained-path qualification is still required before
adoption. No MJWarp backend was run in this batch. Any MJWarp normalization
must remain explicitly labeled as the earlier historical reference, not a
fresh simultaneous denominator or proof of the all-task4x target.
