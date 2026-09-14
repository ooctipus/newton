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

## CPU checkpoint, approximately 12:21 UTC

Regression first confirmed: the old function executed 251 versus 28,735
Python line events for 32 versus 4,096 worlds (the bound was 351). It also
admitted malformed negative list indices / out-of-bounds slice endpoints, and
raised on too-short DOF/world arrays. The new checks reject these inputs before
gathering. This is conservative strengthening for malformed plans, not a change
to any valid supported topology.

The four new tests, all nine existing row-packet tests, and all five existing
notification tests pass, with no skips.
The independent scalar oracle covers 192 valid/mutated configurations with and
without explicit mimic-world validation. Actual saved prefix captures, disabled
leading entries, sticky queue overflow and structural reconstruction checks
also pass. GPU kernel sources, notify flag semantics and invalidation are
unchanged. Independent review found no changed valid topology semantics or
notification behavior. Full pre-commit passed before the frozen runtime
commit `99c796ca`.

## Complete paired results, 12:39 UTC

Three original paired A/B rounds preserve fixed Lab, 16,384 worlds, 200 warmup
steps, 40 synchronized environment steps and 40 graph physics steps. Both arms
retain SIMPLE_WORLD_ZERO and LOCAL_ROW_PACKETS, raw32768/broad7680,
dense192/MF64/prop192, original timestep/two substeps/eight maximum GS sweeps.
All original source, capacity, actual-owner and final idle guards pass.

| Two alternating repeat medians | Old wall | New wall | Wall ratio | Old/new physics |
| --- | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | 62.8812 ms | 28.9060 ms | 2.1754x | 5.5276 / 5.4836 ms |
| GB300 | 63.0507 ms | 28.9393 ms | 2.1787x | 5.0361 / 5.0121 ms |

The independent discovery round was 62.7230 -> 29.5323 ms RTX (2.1239x)
and 63.6908 -> 28.5926 ms GB300 (2.2275x). This comfortably exceeds the
20 ms RTX environment saving gate; physics is essentially unchanged. These
are environment steps, not full policy/learning throughput, and this result
does NOT count toward the 4x-over-MJWarp physics objective.

Artifacts:

- `/tmp/fpgs-franka-notification-validation-paired16k-20260914-01`, manifest
  SHA256 `ed3a7098d2b8c7207ce0a38a80f14e0de910ccd183e96fd57b2afd35215cf743`.
- `/tmp/fpgs-franka-notification-validation-paired16k-20260914-repeat02`,
  manifest SHA256
  `01994506d028ffeff980c8738b8d89daefbfe86dd147df5eda94ffd956fedbb0`.

Reproduction uses the existing
`/tmp/fpgs-kinetic-fixed-variants-TZRkIPYw/run.py`, baseline worktree
`newton-fpgs-g1-metric-tangent-20260914` at7df75c46 and this candidate at
99c796ca. Exact command, runtime imports, file pins, device UUIDs and every
capacity result are recorded in each original parent manifest. Runtime source
remains unchanged by this report-only update. A standalone CPU check on a
constructed full two-mimic16K topology reduced the predicate from median
43.1679 to4.2881 ms; that is diagnostic evidence, not the environment metric.
