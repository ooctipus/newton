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
