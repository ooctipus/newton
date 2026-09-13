# Kuka compact kinetic lifetime: first experiment

Implementation GO: 2026-09-13 00:08 UTC. First checkpoint: 01:38 UTC.
This is an unvalidated structural hypothesis, not a new performance result.
Accepted Newton remains50dfa28d; Isaac Lab remains unchanged.

## Complete cost and mechanism

Replace the primary current-force/predictor, dense row/action/eight-sweep,
and public-state/next-private-cache producers together. Current projected
bias23, compact screws/COM geometry, and a held180-coefficient inverse
whitener replace the canonical spatial-force, tau, factor and row bridges.
Current arbitrary external forces and controls remain current on every call.
The held operator remains distinct from current geometry and current MF inverse.

The conservative existing selected family has8.122447667ms RTX exclusive
busy time, not the sum of its overlapping kernels. The current repeated
12.862500350ms needs3.949476350ms net removal to reach the original-handoff
2x ceiling8.913024ms. The resulting complete replacement allowance is about
4.172972ms; the study uses the slightly conservative rounded4.172948ms.
New producers, materialization, fallback, retained-work dilation and added
gaps all count. The retained trace busy+gaps already costs4.732132ms, so this
scope alone cannot promise4x handoff, whose ceiling is4.456512ms.

The existing tau traversal is already O(n). The proposed saving removes its
global reverse-accumulator lifetime and repeated projection/consumption;
moving bias projection into the preceding finish does not make that work free.
For common-arm self-contact, combine actual relative anchor moments before
the held action, retaining every nonzero common-ancestor term. Actual captures
have shared anchors disabled. This is not the previously failed six-column
per-body response cache or a retry of the old serial sparse-front/row mapping.

## First causal implementation

1. Current arbitrary-force/passive/drive projection fused with held prediction
   and all35-DOF/current endpoint output. Retain original free6 service.
2. Current/next compact kinetic producer, eventually direct geometric180 and
   held refresh, with real first-output to second-input chaining.
3. Direct current dense packets and offset kinetic original-eight recurrence,
   retaining independent MF overlap and complete coupled fallback.
4. Public state, current force routes/counts and next-cache publication, then
   complete paired cost and live integration gates.

The initial512 fixture seed may derive private current data and held T from
original captured state. That is explicitly a producer-excluded component
diagnostic, never the complete replacement cost. No standalone factor timing
or percent-level mapping campaign is authorized by this experiment.

Independent checks cover arbitrary body/joint forces, current changed reuse
controls, effort limits/passive forces, large unrelated sibling wrenches,
all35 DOFs/free transport/prescribed twists, stale cache/held generations,
poison and original held physical action. Numerical/physical errors, not bit
identity, determine acceptance. Wrong finite operators are negative controls
for the independent oracle, not a demand for a costly runtime certificate.

Root owns all GPU runs; RTX and GB300 run paired with frozen imports. The
other three agents own predictor/operator, checked plan/state producer, and
independent tests respectively. Sources begin in the explicit scratch
`/tmp/fpgs-kuka-demand-state-Lv4cAaUh`; no accepted runtime is edited in place.

## Source studies

- `/tmp/fpgs-kuka-kinetic-lifetime-IvJRS2/CARD.md`, SHA256
  `8fc9cddc7b26ef28b93eb1fef3d0f2757873cf660bd9516c655743d687f10135`.
- Initial ABI SHA256 `c79ec5ce51a6ec50cd5dfcfa1d59067e27bedf9d4a33dc2089c703d8dc20a1b2`.
  Its force-half-swap prose is incorrect: literal original arithmetic uses
  `F=body_f[:3]`, `torque=body_f[3:]+cross(COM-origin,F)`. The implementation
  header and revised ABI must preserve this law; no actual half swap occurs.
- `/tmp/fpgs-kuka-state-owner-I5RRnj/MAP.md`, SHA256
  `ee6d9d2069980ec6744c743e5451075fa09deeabd2a2cd769607daf17fb6ff38`.

No gain is recorded at GO. If timing misses, diagnose one measured causal
boundary before a targeted correction; do not silently extend beyond two hours.

## First GPU physical checkpoint

The current-force/held-predictor primitive passes both actual CUDA test methods
on RTX and GB300. Both refresh/reuse saved512 states, changed arbitrary forces
and controls, original same-device force kernels, held free6 fallback, large
unrelated sibling wrenches, invalid-state isolation and two-graph replay are
covered. No tolerance or native arithmetic correction was needed for this GPU
batch. Ten CPU tests also pass; two GPU-only methods are skipped only in CPU runs.

Maximum observed candidate held-action backward error is1.172e-7; predictor
scaled error1.324e-6. The original same-device force oracle has small near-zero
component differences versus FP64, but its resulting physical action and
prediction pass the same gates. Component-bit proximity is not the criterion.

This still seeds private current coefficients and T from original current/held
data outside the tested primitive. It does not yet qualify live cache production,
contact solve/convergence, cold/reset behavior, or a new performance result.
The separately implemented current/next producer and native T refresh are the
next integrated gate. Direct current rows will reuse the existing Y allocation
as private Z, rather than introducing a second full response panel.

Parent `/tmp/fpgs-kuka-demand-predictor-physical-paired512-20260913-01`
is complete and reaped; both GPUs are idle and source guards pass. SHA256:

- Manifest: `8e0630f2f9a5e2cc922dc2b0989eb68db5246c3345b1da18186abccf383f523d`.
- RTX audit: `75ec7e9a0bf722883ad5c4a7b798f90ee8b84ea45e6520282f47ac9f2762d9ef`.
- GB audit: `2d415a299ffc2d6263750d1bdfd757c941d4623ae7f703d928418a41ddb3c18b`.
- Predictor native: `57b7b3bff720c248babd9148c835df1881f36e0f66c0c3ef09dc8e9d756adac6`.
- Types: `cb4077d182a021399fb15c747501f111981fd06d72dc0f89ba2e96d0c0a2bf85`.
- Independent tests: `8b291190096eb499f95af86d3c7e8ab616278b81811d280f9a11cb398ff84f00`.
- Explicit source/input inventory: scratch `pins_predictor01.json`.

## Generated-state and kinetic-core checkpoint, 00:50 UTC

The current/next state producer passes its actual CUDA physical and graph test
on both cards. The connected producer/refresh/predictor/finish chain then passes
two actual CUDA tests per card, including changed forcing on the newly produced
second state and two independently captured graph replays. These checks do not
qualify contact convergence or a live cache/reset protocol.

The first **kinetic-core-only lower-bound** timing uses actual historical16384
world archives, not repeated512 worlds. Each event contains two physical calls:
native primary refresh, current force/predictor and complete next-state producer
for each call; the second finish also produces next geometric coefficients.
Ten warmups precede40 measured graph replays, with every mutable seed restored
outside events. Event times are multiplied by4 for eight-call environment
equivalents. Both cards pass exact replay, all65 mutable/100 readonly array
ownership checks, physical-source pins and final process/idle guards.

| Hardware | Median core ms/env equivalent | Min–max |
| --- | ---: | ---: |
| RTX PRO6000 | 2.662528038 | 2.654336–2.678912 |
| GB300 | 2.322815895 | 2.314496–2.335104 |

**Not a whole-physics gain.** Contacts, allocation, dense rows/eight sweeps,
MF, retained free-factor refresh, callbacks and cold repair are excluded.
Initial warm current construction is setup, but both next-state producers
are charged. Free6 still uses the captured held operator. The captures are
historical full-state fixtures, not today's active-row capture distribution.
This experiment prices an architectural portion; it is not matched live A/B.

Against the planning allowance4.172948ms, the RTX core consumes2.662528ms,
leaving1.510420ms for the remaining replacement work and required accounting.
That is a tight **necessary budget**, not a prediction that2x will be achieved.
Accepted Kuka and Franka performance numbers remain unchanged.

Reproduction uses the root paired owner
`/tmp/fpgs-capacity-20260911-O7dqFC/run_pair.py` and scratch
`run_core_cost.py`, with `pins_core_cost01.json` and the completed chain physical
manifest supplied explicitly. All runs use clean accepted Newton50dfa28d and
the unchanged Lab environment. Completed/reaped artifact parents and SHA256:

- State physical: `/tmp/fpgs-kuka-demand-state-physical-paired512-20260913-01`,
  manifest `59c39eab60a9315cbf11fb3d8fc83a85f135c9bb519bc618e0b5f97086df9735`.
- Chain physical: `/tmp/fpgs-kuka-demand-chain-physical-paired512-20260913-01`,
  manifest `c07ffc5c283e9e3c2a8b60ad41cbd76944f745a3eba0d52fffaf21cd8a633496`.
- Core cost: `/tmp/fpgs-kuka-demand-core-cost-paired16k-20260913-01`,
  manifest `dcb292f58163681fdbcc28ce0a1e0c1381ceae74f79976d928445bc9f0a10dc6`.
- RTX cost audit: `c328b0e3e2bc1ff060ff7e4efec7424f9724a06607a4f2e874e7240ec63df8ee`.
- GB cost audit: `4c39163a1d731158162e7c8823acce676995d136c67e47dfc9b7fd87c4d9241f`.

Direct current contact rows and the offset-eight consumer are next. The first
row CPU smoke produces17741 dense rows with exact original prefix bounds and
no missing used rows; independent physical row checks are still pending at
this checkpoint. Saved raw allocation routes are explicitly external controls
until the retained allocator is connected. No saved J/Z/RHS feeds the candidate.

## Integrated implementation checkpoint, 01:38--01:48 UTC

At the 90-minute checkpoint the native two-call owner is assembled; no new
whole-task gain is accepted. It now connects original free6 refresh, current
force/prediction, the retained ZERO/CSR policy, current count/allocation, direct
rows, current MF inverse/preparation, qualification and positive-selector
materialization, original eight-sweep generic/private owners, mandatory joins,
guarded next-state publication and public contact forces. The second current
state comes from the first actual solved output, not a second archived state.
Original force export is once per two physical calls, not after both calls.

Actual current-eight tests pass on both GPUs, including the coupled-positive
GB world and independent MF cases, unchanged physical velocity/momentum/cone/
residual bounds, serial versus concurrent dispatch, poisoned outputs, two
graphs, failure-publication guards and public-force conversion. Maximum scaled
velocity difference against original current geometry/FP64 held H is5.373e-6,
below the unchanged3e-5 bound. This is finite-eight agreement, not proof that
all tasks have converged or that release parity is established.

The local row test still fails its added-r0 coefficient gate:6.545e-5 RTX and
7.559e-5 GB against7.629e-6. The failure is preserved. Both-card generated PTX
identifies different FMA grouping in point rotation; world-point rounding
then changes phi, amplified by phi/dt. Native arithmetic and tolerances are
unchanged. The complete-eight physical checks pass despite this local error;
bit-level coefficient matching is not substituted for physical acceptance.
Diagnosis: `/tmp/fpgs-kinetic-row-diagnostic-eGFPdS/ROW_ROUNDING_DIAGNOSIS.md`,
SHA256 `f11c133bfcdcfbf9f15f9eb22a317982d9bd09639b5c5c681a981cb39e7ea0d7`.

The connected two-call512 GPU gate then passes both cards. Its strengthened
second run independently reconstructs forces/bias on each actual current q/qd,
checks prediction and endpoint motion, original current eight-sweep physics,
original integration/publication, actual emitted-force conversion, repeated
graphs, and the complete readonly inventory. The largest predictor scaled
error is5.629e-7 and endpoint error1.172e-6. All current capacity/no-drop checks
pass. The complete inventory has230 mutable/225 readonly arrays, with exact
reshape aliases deduplicated; unused poisoned row storage is not consumed.

The event scope is now larger than the earlier selected-only budget: it also
charges retained MF, CSR, three count finalizers and public force conversion.
Recomputing the original disjoint Nsight union gives RTX selected-exclusive
9.771697ms, retained-outside2.862754667ms and gaps0.220128333ms. Thus its
conservative full-boundary screening allowance is5.822196650ms, **not**
4.172948ms. The same-window GB allowance is5.122562667ms. Free6 was already
selected and receives no second credit. This scope correction is not a gain.
Audited mapping: `/tmp/fpgs-kuka-complete-boundary-budget-MaJ7kfmz/NOTE.md`,
SHA256 `f2cde8056607d7217b6e9d14b8cb5c9330995b0d272919cb3182b7763c90f266`.

Completed/reaped physical parents:

- `/tmp/fpgs-kuka-demand-current-eight-physical-paired512-20260913-01`: four
  CUDA tests per card pass; original-current-eight and export/guard controls.
- `/tmp/fpgs-kuka-demand-boundary-physical-paired512-20260913-01`: three tests
  per card pass, including actual two-call solve/publication and graph replay.
- `/tmp/fpgs-kuka-demand-boundary-physical-paired512-20260913-02`: strengthened
  two-call gate passes both cards; manifest SHA256
  `d3b8bbd0884c640943a12518a3907411a06a63546816c9047b22b76b8b664e42`.

The first actual16K full-boundary cost run starts at01:48 UTC with a fresh
paired parent. It retains fixed historical collision/control inputs and
excludes collision generation, callbacks, cold repair and Lab/sensors. It is
an architectural cost screen, not a live-task measurement. Accepted Kuka and
Franka tables remain unchanged. A miss requires a causal whole-boundary trace
and at most one targeted corrective experiment, not a tile-size tuning grid.
