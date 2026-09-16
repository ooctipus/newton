# Complete sleeping-contact island owner

Continuation of the ten-hour structural window, 2026-09-15 23:44 through
2026-09-16 09:44 UTC. This candidate starts from `928deb6d`, the closed,
default-off scalar owner. Performance reference remains accepted `ca0d427a`,
not that slower prototype. Fixed Lab `53ee6b44` is unchanged.

## Why this boundary

Contact-free sleeping alone failed its whole-physics milestone after one
diagnosed correction. This experiment must instead retire complete dormant
contact production, row allocation, response building and solve visits while
preserving canonical contact geometry and reported force.

Historical 4K full-geometry connectivity (not positive-impulse-only edges)
finds 88,012 RTX scalar/static contacts, 384 scalar/arm contacts, and no
scalar/scalar contacts. Only 187 keys attach to arm islands. Of 278,463
allocated contact rows, 263,964 lie outside arm-containing islands. GB has
275,934 of 290,541 rows outside arm islands. These older snapshots are NOT
current population, sustained quiet admission, or measured savings. A small
current-run census reuses the existing checked-capture boundary.

The current RTX node capture attributes 0.881 ms to contact production,
0.271 ms scheduling, 0.127 ms allocation and 1.369 ms GS. Additional narrow
query work exists, but broadphase wake discovery remains required. These
diagnostic sums are NOT additive whole-physics savings. The complete owner
must save approximately 0.7 ms whole physics versus `ca0d427a` after all
island, wake, cache, fallback and public-state costs to justify qualification.

## Ownership and correctness

- Model-derived mechanical components; static ground never couples islands.
  Use linear-storage connected components, not a dense component adjacency
  matrix. Unsupported components make their connected island awake.
- Preserve all current raw contact records, including zero-force records.
  Cached asleep contacts and freshly generated awake contacts share the
  original canonical buffer and unchanged capacity/overflow law.
- Retain broadphase discovery and take complete wake closure over old leased
  membership plus current candidate edges before excluding any narrow pair.
- Preserve force publication. Cached geometry without cached solved force is
  not an admissible replacement. Control wake after collision must rebuild
  the entire affected island's rows before any solver work is skipped.
- Reuse only geometry sealed at the frozen pose/model generation. Quiet
  velocity alone does not certify an older collision packet. Reset, authored
  state, model changes, timestep changes and buffer switching revoke leases.
- Follow MJWarp's tree/island wake and incremental-collision reasoning, but
  do not copy its omission of dormant public contacts: Newton consumers need
  their geometry and forces. No Lab changes or physical-budget reductions.

First consumer capability remains complete independent scalar components;
connectivity/cache ownership is topology-general. Do not report unsupported
fallback on other tasks as evidence of speedup or general articulated sleep.

## Checkpoints

Funded 01:00 UTC. Current-population screen and native connectedness/cache
primitives first; integrated loaded-contact/native checkpoint by 02:00 UTC,
first paired whole-physics result or explicit implementation blocker by
02:30 UTC. A measured mapping mismatch receives one diagnosed correction,
not an open-ended tuning grid. Use existing tests/paired drivers; no new
benchmark framework, sudo, inflated maxima, dropped rows, or source-pointer
changes. Compare against the retained baseline on RTX and GB300 together.

## Integrated checkpoint

At 01:35 UTC, 12 integrated and inherited scalar-owner tests passed on both
GPUs. Actual loaded contacts slept with their geometry and force retained;
control changes after collision and newly arriving dynamic contacts restored
the affected rows. Captured masked resets, fresh output and in-place state
publication passed. A zero-gap tangency fixture initially jittered in both
reference and candidate; a 1 mm collision envelope stabilized the fixture
without changing the integration or solver budgets. This is not a benchmark
configuration change.

The first-capture contract is explicit: call
`solver.prepare_contact_sleep(contacts)` before the first collision/capture.
Binding during the first captured solve is too late, since its preceding
collision has already recorded a graph without the cache callbacks. The
benchmark's setup-only adapter calls this Newton API when the canonical
Contacts allocation is created. No Isaac Lab source is changed and no timed
Lab callback is replaced. Stream migration requires ordinary caller-provided
ordering/synchronization; simultaneous reuse is unsupported.

Two cold-capture/stream-migration regressions reproduced the original failures
on both GPUs before that integration correction. At 01:45 UTC the complete
33-test targeted suite passed on each GPU, including both corrected capture
tests, loaded physics, graph replay, topology, geometry and force primitives.
The first whole-physics timing is pending; this is not performance acceptance.
