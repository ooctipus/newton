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

## First complete measurement: not accepted

Candidate `7b003d0a`, reference `ca0d427a`, fixed Lab `53ee6b44`:
`/tmp/fpgs-contact-islands-paired4k-20260916-01/manifest.json`.
All four children exited 0; capacity, source and final idle guards passed.

| GPU | Reference physics | Candidate physics | Reference / candidate |
| --- | ---: | ---: | ---: |
| RTX PRO6000 | 6.862266 ms | 7.764994 ms | 0.883744x |
| GB300 | 6.112485 ms | 7.403039 ms | 0.825672x |

Whole-environment wall time likewise regressed: 28.446714 to 31.304290 ms RTX,
27.000021 to 29.156845 ms GB. This one discovery run is not repeat-qualified.
Neither this candidate nor the earlier scalar-only owner replaces the accepted
reference.

At the post-profile endpoints the candidate retained 80,582 / 80,912 cached
contacts (RTX / GB) and slept 430,190 / 430,439 components out of 446,464.
These are two sampled endpoints per GPU, not time-weighted admission. All
cache/island/force status checks passed. Actual dormant work was removed; lack
of admission does not explain the loss.

Matched five-step node diagnosis:
`/tmp/fpgs-contact-islands-nodes-paired4k-20260916-01/manifest.json`.
All guards passed. The following are kernel-duration sums, not additive
whole-physics savings or accepted timing:

| Stage | RTX reference -> candidate | GB reference -> candidate |
| --- | ---: | ---: |
| Contact producer | 0.8036 -> 0.4148 ms | 0.5542 -> 0.3911 ms |
| Contact schedule | 0.2723 -> 0.1448 ms | 0.2762 -> 0.1793 ms |
| Primitive narrowphase | 0.1809 -> 0.0495 ms | 0.1385 -> 0.0524 ms |
| GS | 1.0368 -> 1.1499 ms | 1.2434 -> 1.3106 ms |
| Added contact-sleep maintenance | 0 -> 1.9392 ms | 0 -> 1.8250 ms |

The identical GS kernel still launches eight times per environment step with
4096 world blocks, retains serial/coupled work and limit projections, and uses
unchanged registers/shared memory. Actual executed sweep counts and the precise
hardware cause of its small regression were not captured. Cached contacts
mostly removed already-parallel independent work, not the dominant convex or
serial solve work.

RTX added maintenance: topology construction 0.4454 ms, aggregation 0.1959,
input/wake 0.5191, masks 0.1215, grants/sealing 0.4154, geometry cache 0.1372,
force cache 0.0493 and status 0.0553. Reusing topology between substeps exposes
only about 0.2263 ms before replacement overhead. A topology-only tuning pass
is therefore NOT funded as a rescue. A substantially smaller complete owner,
not a launch/parameter grid, is required.

## One funded architectural correction: singleton/static leases

At 02:07 UTC, fund one complete-controller replacement, not repeated topology
caching alone. The first dynamics consumer already owns independent scalar
mechanical components. Restrict its contact leases to singleton components
touching only static geometry. Every dynamic/dynamic broadphase candidate must
wake both endpoints unconditionally and veto sleeping for the entire collision
generation. Coupled components stay awake; do not claim general articulated or
multi-component island sleeping. This capability is model-derived, with no
Keyboard-specific geometry, branch count or physical parameter.

Under that invariant no dormant lease contains a dynamic edge, so the full
parent/union/aggregation pipeline is unnecessary. Fold live input validation
and quiet/grant decisions into the existing scalar prepare/finish owners.
Capture eligible static-contact forces before granting; preserve global status
veto, exact input-pose geometry, total acceleration and unexpected-response
handling. Allocator/publication consumers select current sleeping endpoints
directly instead of repeated full-capacity mask production. Geometry incidence
belongs to the collision generation. Keep full broadphase discovery, static
pose/model invalidation, and fallback for unmanaged contact generations.

The measured gross retirement opportunity is about 1.3885 ms RTX for topology,
redundant input checks, separate grants and masks, before replacement costs.
This is a plausible but tight correction, not a promised net win or a 2x path.
Use the original 33 tests plus conservative-coupling regressions and the same
paired protocol. Target frozen integrated implementation by 02:45 UTC and
first paired whole result by 03:15 UTC. No microparameter grid follows if the
complete corrected owner still fails the approximately 0.7 ms RTX milestone.

At 02:24 UTC, the direct endpoint allocator/force consumers pass their three
CPU controls, and the fused mode-2 finish passes four CPU controls. A new test
fails its grant assertion against the exact HEAD `finish_components` body
loaded in an isolated module using the new descriptor; this is not a replay
of the old descriptor or the complete old controller. The test covers stale
pose seals, total solved acceleration at tiny timesteps, cache/contact status,
generation and coupling vetoes, canonical zero velocity and unexpected solved
responses. Runtime integration and native timing are still pending.

The first attempted native pre-replacement coupling checks encountered an ABI
mismatch during concurrent controller replacement (15 versus 19 force-capture
arguments). They are not evidence of a physical regression. The new broad-only
pair and three-component-chain tests will instead qualify the explicitly more
conservative singleton/static policy after the integrated controller is frozen.

The first integrated mode-2 native run exercised 36 tests on each GPU and
failed six loaded-sleep admission assertions per card; physical state/contact
comparisons and the two conservative-coupling controls passed. A one-fixture
RTX diagnosis found zero coupling/static/status vetoes and physically quiet
velocities around 9e-11, but counters reset to zero every collision tick.
Fusing the exact geometry-seal check into quiet-history accumulation was the
cause: the second substep moves the input coordinate by about 7e-16 relative
to the collision pose. The original owner accumulated physical quietness
independently and required an exact seal only when granting sleep. Restore
that distinction without relaxing geometry validity or physical thresholds,
then rerun the complete native suite before timing.

At 02:35 UTC, that corrected complete 36-test suite passes on both RTX and
GB300 (11.73 / 12.17 seconds). Loaded contacts now acquire leases, retain
geometry/forces, and restore ordinary rows on wake; broad-only dynamic pairs
and loaded three-component chains remain awake independent of pair order.
Cold capture, synchronized stream migration, masked resets and fresh/in-place
publication pass. Full repository pre-commit passes before pinning the
whole-physics measurement; no performance acceptance follows from these tests.

## Corrected complete measurement and closure

Runtime `348896b433b47d066d61045fa3c03aba0469c5d2`, reference `ca0d427a`,
unchanged Lab `53ee6b44`. The 40-step whole screen is at
`/tmp/fpgs-contact-islands-paired4k-20260916-02/manifest.json`, SHA256
`f9b380f6985203754bf864050240d57b82106994938116d9fe311f0d603b2fe5`.

| GPU | Reference physics | Candidate physics | Reference / candidate |
| --- | ---: | ---: | ---: |
| RTX PRO6000 | 6.842999 ms | 6.376398 ms | 1.073176x |
| GB300 | 5.955107 ms | 6.531289 ms | 0.911781x |

Whole-environment wall time is 28.694467 -> 32.295702 ms RTX and
26.784003 -> 27.230645 ms GB. It does not improve. This is one discovery
round, not repeat-qualified performance or a new MJWarp ratio.

Independent artifact review verifies 26 recorded hashes, all four child exits,
all eight capacity/finite boundaries, unchanged budgets and final source/idle
guards. Setup binding ran exactly once per candidate at raw capacity 147456.
Dense 704 and broad 57344 capacities remain unchanged. Post-profile candidate
sleep counts are 428610 RTX / 429546 GB of 442368 eligible components;
dormant-contact counts are 80462 / 85228 and cached counts 80462 / 85172.
Invalid-contact flags remain false. These are endpoint samples, not
time-weighted admission or full physical qualification.

The matched short node diagnosis is
`/tmp/fpgs-contact-islands-nodes-paired4k-20260916-02/manifest.json`.
All four children, eight capacity boundaries and final source/idle guards pass.
The following are graph-only summed kernel times, NOT additive critical-path
savings:

| Owner | RTX reference -> candidate | GB reference -> candidate |
| --- | ---: | ---: |
| Explicit sleep/cache/held-force maintenance | 0 -> 0.514 ms | 0 -> 0.468 ms |
| Fused scalar prepare + finish | 0 -> 0.965 ms | 0 -> 0.654 ms |
| Contact producer | 0.814 -> 0.508 ms | 0.559 -> 0.378 ms |
| Contact schedule | 0.316 -> 0.180 ms | 0.342 -> 0.218 ms |
| Other row/response kernels | 0.544 -> 0.510 ms | 0.614 -> 0.577 ms |
| Main GS | 1.013 -> 1.488 ms | 1.490 -> 1.110 ms |
| Collision | 1.219 -> 1.132 ms | 1.243 -> 1.163 ms |

The baseline uses its original scalar dynamics/publication owners, not a free
zero-cost replacement for prepare/finish. Compared with mode 1, explicit
maintenance falls 1.979 -> 0.514 ms RTX and 1.861 -> 0.468 ms GB; these totals
include the old `_mark_contacts` pass omitted from the earlier maintenance-only
figures. Some checks moved into prepare/finish, whose combined time grows
0.777 -> 0.965 ms RTX and 0.517 -> 0.654 ms GB. The funded replacement really
removed its targeted duplicated work.

The remaining opportunity is limited: retained broadphase and convex
GJK/MPR/manifold work dominate the narrowphase savings, while the original GS
kernel still launches eight times with unchanged resources and retained coupled
work. Its opposite short-window time changes across cards are measured, but
the exact cause is not established. Five-step node spans also invert the
40-step whole-screen ordering (RTX 6.618 -> 6.746 ms; GB 6.227 -> 5.993 ms).
They are workload-sensitive diagnostic windows, not replacement whole results.

Close this candidate for the current large-gain target. The approximately
0.7 ms RTX milestone was not achieved, GB regresses in the whole screen, and
the result is not consistently beneficial. No repeat/qualification campaign,
mapping grid, default enablement or dependency-pointer promotion follows.
Retain the default-off implementation and causal evidence on the fork; accepted
`ca0d427a` and all previously accepted backend ratios remain unchanged.

Next funded work is a separate rank-aware active-contact CPU falsification
from retained `ca0d427a`, documented in `ACTIVE_WRENCH_20260916.md` on
`ooctipus/fpgs-active-wrench-20260916`. That is a new complete response/solve
boundary, not another sleeping-controller micro-optimization.
