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
freezing. No GPU timing or physical admission is claimed yet.
