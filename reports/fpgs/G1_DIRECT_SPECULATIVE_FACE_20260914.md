# Skip finite patches selected for immediate fallback

Research-only, default off: `NEWTON_NARROW_PHASE_FINITE_FACE_DIRECT=1`.
Base: clean `4a0bf739778041d5f833abfd1d077703677eb12d`; Isaac Lab remains
`53ee6b44c2334341305dbdf385a3916c6b140799`, with unchanged task budgets.

This card records the hypothesis after the first CPU implementation, before
GPU qualification or timing; it is not a retrospective performance claim.

## Work removed and cost model

The finite cuboid/triangle query can establish that its supporting point
projects inside the actual triangle and has speculative positive clearance.
It currently continues through full projected-face clipping and patch
selection, only for `query_contacts` to discard that patch and invoke the
original manifold. Select that same fallback before constructing the discarded
patch. This does not remove the required original fallback or change its
contact writer, reducer, material parameters or finite-iteration support.

Existing G1 exclusive node attribution assigns the finite-query producer
2.177 ms on RTX and 7.451 ms on GB300. Those are upper bounds on removable
producer work, not predicted whole-step gains. The speculative-path coverage
is unknown. A useful result must save about 1 ms or more on RTX without a
material GB regression; smaller gains are not grounds for a tuning campaign.
Added work is a pair of scalar comparisons and, only within the candidate
interval, a conservative roundoff guard. No new buffers, launches, compaction,
representations or conversions are introduced.

## Correctness and early decision

- Require the original supporting-face branch, not an infinite-plane guess
  outside the finite triangle. Near either margin/threshold boundary retain
  the original floating-point path.
- Compare direct geometry against the unchanged pure geometry query with a
  host-side original admission check, including rotated cuboids, sloped
  triangles, penetration, finite borders and threshold-adjacent cases.
- Reuse existing current-terrain and loaded-contact physical tests on both
  GPUs, then an early paired whole-physics A/B with all original capacity and
  source checks. Keep full fallback/publication costs in the measurement.
- If the integrated result misses the cost model, inspect the finite/fallback
  node attribution once. Reopen only for a specific newly evidenced cause;
  no register or block-size sweep.

The private query API regression failed on the unchanged source before its
implementation. Initial eight CPU tests passed with the switch both off and
on; stronger independent rotated/sloped controls are being added before
freezing. No GPU timing or physical admission was claimed at that checkpoint.

## Completed screen, 02:33 UTC: below the funded saving

Frozen runtime `d3ccebffd9147ba55a92f7b9a4e512cefdccae6e` passes nine CPU
tests with the switch both off and on. Four explicit physical selectors per
GPU pass without skips: both new independent geometry controls, the saved
96-pair complete current geometry/lifecycle check, and the original reduced
loaded support/tilt/sliding/finite-border cases. Physical manifest
`/tmp/fpgs-g1-direct-face-physical-paired-20260914-01/manifest.json`, SHA256
`38bd4f9ef291fc524e5904e71d7f40bb3ec20c536b9f3883c179bc1f16a8370f`.

One complete16K paired A/B at200 warm/40 wall/40 physics steps:

| GPU | Original4a | Direct speculative fallback | Ratio | Saved |
|---|---:|---:|---:|---:|
| RTX |22.934269ms|22.764429ms|1.007461x|0.169841ms|
| GB300 |27.643392ms|27.194016ms|1.016525x|0.449375ms|

Both variants retain sparse factor1 and CELL/FINITE/WELD/PACK1; only DIRECT
changes0->1. All four children exit0, with actual activation, capacity,
finite-state and final source/idle guards passing. Whole manifest
`/tmp/fpgs-g1-direct-face-live-paired16k-20260914-01/manifest.json`, SHA256
`b653d2ea9afa68ec13957594c91ca4412647da214b7a0e116691a88681f8d629`.
Original whole owner/explicit compatibility command is in
`/tmp/fpgs-g1-direct-face-ready-CWoOZZC7/LAUNCH.md`.

The one bounded three-step node diagnosis shows the intended finite producer
falling2.176737->2.009486ms RTX and7.451126->6.960619ms GB. Retained fallback
is0.921536->0.930838ms RTX and3.364523->3.291819ms GB. The small whole saving
therefore is not a large producer win erased by another stage: the deleted
patch work accounts for only a small fraction of the complete finite query
cost. Most query work and the required original fallback remain. No exact
activation-frequency or hardware-counter claim is made.

Node manifest SHA256
`7681300ed5bb8497b53e9da04a55bc7665e5326b6d5f2b74c1b5a17871604d0a`,
under `/tmp/fpgs-g1-direct-face-nodes-paired16k-20260914-01`. The original
parent remains failed for its inherited auxiliary-graph analyzer refusal;
both simulations completed. Independent original source/capacity checks and
the existing strict reader pass, retaining12 physics and3 auxiliary roots,
all memory nodes and0 unproven nodes per capture. Only the two already
reviewed finite collision aliases are added to its owner set.

Decision: do not promote, compose, repeat-time or tune this sub-percent RTX
candidate. Preserve the implementation and causal result as a closed research
branch. The next structural work must remove substantially more than this
discarded-patch subset. No Isaac Lab, task-budget or parent-pointer changes.
