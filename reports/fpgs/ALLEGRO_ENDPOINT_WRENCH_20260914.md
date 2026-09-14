# Allegro row-local endpoint wrench reuse

Pre-code card, 2026-09-14 22:28 UTC. Isolated branch from composed
`1a9efc33efbc0f23e1e7676a5edded795f224c97`; qualified Allegro runtime remains
7fca and original worktrees are preserved. One causal correction, no mapping
grid, new global producer, physics allowance or capacity change.

## Cause and complete cost gate

The matched keyed node captures expose 5.852930 ms RTX / 7.396244 ms GB in
the four parallel tiers, versus 4.889749 / 5.508978 before keyed ingestion.
RTX tiers32/64/96/128 cost 0.392277 / 3.137761 / 1.922785 / 0.401024 ms.
These include ingestion, scaling, all original sweeps and inverse publication;
the captures do not separately measure ingestion or actual executed sweeps.
The earlier retired setup is already credited, not funding for this correction.

The actual offline SM120 solve64 cubin repeats endpoint body-map lookup,
three articulation-origin loads, point-minus-origin and wrench cross product
inside every unrolled coordinate contraction. For example, endpoint A's q0
chain at SASS0x5b60--0x5db0 recurs for q1 at0x6080--0x62d0; endpoint B repeats
likewise. Source is `allegro_kinetic_rows._row_source`, q22/side projection.
This is emitted work, not an assumed source-level compiler omission.

Compute each current endpoint descriptor and direction-specific wrench once
per row, then reuse them for the 22 coordinate contractions. Body-map loads
drop from22 to1 per responsive endpoint; origin/wrench work drops from4 or6
to1 per connected endpoint. Keep coefficient loads, A-then-B addition and
axis accumulation order, all metadata, canonical exports and the full shared
panel. Added cost is row-local register lifetime only, charged in AOT and whole
timing. There is no shared exchange, barrier, global cache or admission pass.

Hypothesis: at least1 ms RTX whole saving, requiring parallel tiers at most
4.852930 ms if other owners remain flat (17.1% reduction). This is plausible
from the repeated dependent chains, not established by instruction counts.
The current owner and incomplete phase attribution do not guarantee the gain.
One original paired whole A/B against qualified1a9/7fca decides it; a miss
does not authorize register/block/tier tuning. No new four-times claim follows.

## Why this is not contact-leader geometry sharing

The old AnymalD WORLD_ROWS two-phase geometry-once/shared-then-row experiment
lost290/210 versus279/177 microseconds: its dependent chains ran serially
(`fpgs_solver_shape_and_spirit_20260909.md`, fixed Lab, section18.2). Current
Allegro one-row-per-thread SIMD also executes a contact-bearing warp's geometry
instructions with leader masking; duplicate lane work is not three times
issued work. Historical four512 fixtures have varying0--28-row prefixes and
370--383 cross-warp triplets among6640--6691 contacts. These are old fixtures,
not today's16K cohort timing. There is no free lane%3 alignment or broadcast
across warps. Do not reopen that shared two-phase mapping here.

## Lifetime, fallback and checks

Read CURRENT frames, anchors and origins on every substep after the existing
normal/tangent shared-anchor selection. HELD factor/current map cadence stays
unchanged. Invalid/static/nonresponsive endpoints remain skipped before any
indexed origin/map load. Both endpoints retain their original signs, including
same-articulation contacts. Prefix rows and `_row_source(kinetic=False)` stay
unchanged, as do >128/MF physical fallback and default-off unrelated tasks.

Extend only the existing CPU source-owner assertion regression-first; reuse
the eight CPU tests, four existing Allegro CUDA selectors and existing offline
compiler for all four tiers on SM120/SM100. Compare original native ABI and
fallback source; report registers/shared/stack/spills and verify emitted
descriptor/wrench chain removal before requesting GPU work. Root owns all GPU
qualification and timing; this worktree introduces no benchmark framework.

## Frozen CPU and offline readiness

The existing source-owner regression fails before the edit (one syntactic
body-map read inside q rather than two outside it), then all eight existing
CPU tests pass. A final post-format repeat passes8/8 in10.038 s. This covers
the unchanged fallback/current maps, constructor and original/keyed ABI binding;
CPU binding is not execution of the CUDA parallel algorithm. Root separately
reviews the complete runtime diff and finds no source blocker.

Final offline02 passes all16 records: map/prefix/key/fallback and all four
actual parallel entries on both SM120/SM100. Resources, rows32/64/96/128:

| Tier | RTX registers old -> new | GB registers old -> new | Shared bytes |
| --- | ---: | ---: | ---: |
| 32 | 87 -> 91 | 83 -> 86 | 5868 |
| 64 | 82 -> 108 | 83 -> 110 | 10024 |
| 96 | 98 -> 103 | 98 -> 96 | 14172 |
| 128 | 98 -> 103 | 98 -> 96 | 18232 |

All stacks/spills remain zero and shared storage is unchanged. The64-tier
register increase is real added cost, not evidence of a speedup; no resource
grid is authorized. Actual final SM120 solve64 SASS reduces body-map constant
pointer instructions44->2 and origin pointer instructions46->4 (the two
prescribed-target uses remain). Coordinate coefficients and all sweeps remain.

The first offline attempt compiles all records but its final source-pin guard
correctly rejects it: formatting changed Python source during compilation.
`offline01` is preserved and is NOT the frozen evidence. No numerical/native
algorithm correction follows that attempt; fresh offline02 uses frozen bytes.

Evidence: `/tmp/fpgs-allegro-endpoint-wrench-offline-IhJgpMP6/offline02/report.json`,
SHA256 `1b0d8362d7c0034a71e1f2cdb92ebc5a34bf7fccc15b072604a8e42ee775cf10`.
The reused offline adapter is `compile02.py` in that parent directory, SHA256
`f427d8d3e88b5cf9ac864a6d1413d05f9a4e4e1b3b0d205e82312b856384deb4`.
It reuses the qualified offline compiler with only source/output roots changed.

Frozen runtime SHA256:

- `allegro_kinetic_rows.py`:
  `156fbe7b3755e5b64f4e62efdbdcad0b30163e86ef5440b9a90169bdfb8e488c`.
- Unchanged `solver_feather_pgs.py`:
  `c8f118751a9133dfd4d7b85039a840f20a67e15e48600acea3a27c3ac514dc02`.
- Byte-identical `_row_source(kinetic=False)`:
  `2dbac7e5534cc519de4a2519b30347c6666a74597952c9bd24056af0f8ede99e`.

Reproduce CPU checks from this worktree. The same environment/interpreter
runs the retained `compile02.py`; it has the original fixed `offline02` output
literal, so a later offline reproduction must rebind only that literal in a
copy to a NEW directory. Preserve the original adapter and both attempts:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/home/octi/Projects/newton-fpgs-allegro-endpoint-wrench-20260914 \
  uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python \
  python -m unittest \
  tools.fpgs_bench.test_allegro_kinetic_rows.TestAllegroKineticRowsCPU \
  tools.fpgs_bench.test_allegro_kinetic_lifecycle.TestAllegroKineticLifecycleCPU
```

Use the same four Allegro CUDA selectors listed in the
[composed integration card](FOURX_QUALIFIED_INTEGRATION_20260914.md), with this
tree selected explicitly. They remain pending root execution, followed by
the existing corrected/current-capacity whole driver. Full pre-commit passes.
Status: concrete emitted-work deletion; GPU physical and performance outcomes
unmeasured. Original qualified source and measured gains remain unchanged.
