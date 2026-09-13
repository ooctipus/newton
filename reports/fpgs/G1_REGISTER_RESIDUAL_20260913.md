# G1 register residual32/64 experiment

Conditional native GO18:29 UTC; first resource decision18:55, integrated
checkpoint19:44 (75 minutes). Base bd1cc095, runtime19099 sparse43 plus packed
original triangle query. No finite-query, factor, row producer, capacity or
solver-allowance changes.

Replace only current GS/setup/decode with separate32/64-row warp owners and
exact original >64 fallback. Stream existing sparse18 Z into shared43xN SoA,
form each owned Gram row into named register scalars, then maintain residual
and lambda with static ordered projections and warp delta broadcasts. Two
owned rows per lane at64 mean128 Gram scalars, not64. Stream coefficients and
node indices during formation instead of retaining36 extra registers. No
global Gram storage or queue. Integrate the exact cold unilateral stationary
test before Gram formation; no upstream W/response deletion is credited.

Preserve normal/limit projection, current-parent friction disk and sibling
update before current row, eight sweeps, early stationarity, applied-impulse
decode, all statuses/count guards/publication and >64 rows without truncation.
Charge all three filtered launches and empty CTAs. The full-cost hypothesis is
GS+setup near3ms versus current~5.7ms, potentially~10% of the25.56ms family;
this is not a measured saving or4x promise. Stop if the64 owner spills Gram
to local memory or its resource footprint makes that hypothesis untenable.

Reuse existing original-eight physical controls plus31/32/33/64/65 boundary,
stationary and loaded-friction cases. No new benchmark or certificate framework.

## Resource and CPU checkpoint18:43

The conditional gate passes: both intrinsic owners keep every Gram scalar in
registers with zero local stack/spill traffic under offline SM120 and SM103
compilation. Final SoA uses a33/65-column pitch: unpadded32/64 strides put
different support-node reads in the same bank during constant-column Gram
formation. This is the original padded design, not a worker/threshold sweep.

| Owner | Registers RTX / GB | Static shared |
| --- | ---: | ---: |
| Original sparse GS |40 /37|700B|
| Register32 |112 /112|6104B|
| Register64 |234 /232|11736B|

The64 owner has128 named Gram scalars per lane, not64. Resource cost and code
size remain substantial risks: initial cubins were~168KB/303KB for32/64 versus
~49KB original (container sizes, not exact instruction-text or cache evidence).
The complete comparison will charge three filtered launches and all formation.
Offline artifacts: `/tmp/fpgs-register-resources-final-39xd6bdd`.

CPU regression-first missing API was observed. Consolidated controls pass8CPU
tests with1root-owned CUDA skip:31/32/33/64 loaded and stationary, cross-bank
normal30/tangents31/32, nonzero synthetic initial impulses with delayed
friction, actual current geometry and held response, stale status, plus the
three inherited constructor/metadata controls. Original contact response and
>64 GS have no CPU native bodies: the current-geometry CPU control explicitly
binds independently computed J/W responses; only CUDA tests can claim actual
original producer and65-row fallback execution. CPU namespace/setup failures
were repaired without changing CUDA arithmetic or numerical thresholds.

Runtime activation is `FEATHER_PGS_REGISTER_RESIDUAL=1` with the existing sparse
factor flag. No new device allocations or queues; only the solve method is
replaced. Existing factor, current limit/contact producers, raw reservation,
packing, RHS/restitution, force/publication and notification lifetime remain.
No upstream ZERO/W-removal credit is taken: stationary detection only bypasses
this owner's Gram/GS/decode. The >64 native fallback adds one entry count guard
and otherwise retains its entire original recurrence.

No CUDA physical or complete performance result is claimed at this checkpoint.
