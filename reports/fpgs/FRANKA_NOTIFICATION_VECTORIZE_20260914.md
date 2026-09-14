# Franka current topology validation: vectorize all checks

## Pre-code decision (2026-09-14 12:17 UTC)

This is a large **environment-step** repair, not a claimed whole-physics gain.
The fixed, contact-correct Isaac Lab source remains 53ee6b44c. Newton starts
at 7df75c46bd468b64c17958190fc800cfeb62201e. No Lab source or notification
semantics will change.

Fresh source-checked captures in
`/tmp/fpgs-tenhour-franka-current-nodes-paired16k-20260914-01` reproduce
approximately 60--62 ms environment steps despite about 5--5.5 ms physics.
The current reset profile contains 46.626 ms in `reset_idx`. Corrected root
pose writes issue JOINT_PROPERTIES notifications. These require complete
row-packet topology validation; its existing Python loop checks 16,384
articulations and up to 32,768 mimic entries on each notification.

Replace only the per-articulation Python predicate with NumPy array checks.
Keep current topology-signature readbacks, all structural checks, disabled
mimic checks, invalidation, exceptions and downstream physics unchanged.
No cached admission result, reduced validation cadence, or changed flags.

Expected removable work is roughly 38 ms from the earlier CPU diagnosis;
the current complete A/B will decide the actual environment saving. GPU
physics should remain unchanged within timing noise. A useful result must
remove at least 20 ms of the current complete RTX environment step.

## Gates and timebox

- Regression first: Python line-event count in the exact admission predicate
  must remain bounded as worlds increase from 32 to 4,096. This tests removal
  of interpreted per-world work without a flaky time threshold.
- Independent scalar oracle across valid randomized world/group/list orders,
  zero/one/two mimics, and optional world arrays. Late wrong-world/DOF entries,
  a third mimic and malformed negative/out-of-range indices must reject.
- Existing current-prefix, notification and packet tests remain unchanged.
- Reuse the original fixed-Lab paired RTX/GB whole benchmark, identical task
  settings and capacities, rather than introducing a new timing harness.
- First CPU result by 12:35 UTC; frozen candidate by 12:45 UTC. GPU A/B runs
  immediately after the currently leased Allegro batch. Diagnose a miss once.

## CPU checkpoint, 12:24 UTC

Regression first confirmed: the old function executed 251 versus 28,735
Python line events for 32 versus 4,096 worlds (the bound was 351). It also
admitted malformed negative list indices / out-of-bounds slice endpoints, and
raised on too-short DOF/world arrays. The new checks reject these inputs before
gathering. This is conservative strengthening for malformed plans, not a change
to any valid supported topology.

The four new tests and all nine existing row-packet tests pass, with no skips.
The independent scalar oracle covers 192 valid/mutated configurations with and
without explicit mimic-world validation. Actual saved prefix captures, disabled
leading entries, sticky queue overflow and structural reconstruction checks
also pass. GPU kernel sources, notify flag semantics and invalidation are
unchanged. Complete environment A/B remains pending; no gain claimed yet.
