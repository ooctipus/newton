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
