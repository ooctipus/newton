# Complete current-packet / endpoint-residual experiment

Pre-code decision: 2026-09-15 12:27 UTC. First checkpoint 13:57 UTC;
no silent extension beyond 14:27. Default-off Newton-only prototype,
`FEATHER_PGS_KUKA_ENDPOINT_RESIDUAL=1`. Branch from `dfc11128`, which
contains retained `ca0d427a` and the closed default-off lazy-response study.
Performance reference remains retained `ca0d427a`, never a losing prototype.
No Isaac Lab, timestep, substep, eight-sweep allowance or capacity changes.

## Explicit reopening: what already failed, what changes

The endpoint-residual idea itself is NOT new. First-hit `e96aaab3` already
used physical delta velocity, dirty-coalesced endpoint scans and first-use
responses. Its whole RTX time regressed 11.299988 to 12.386436 ms; offset
cost grew 0.992790 to 2.442478 ms while its producer saved only 0.042100 ms.
It retained 48 packet floats/contact plus three rows of metadata production,
reloaded axes at each refresh, used shared endpoint twists and reduction
gathers, and performed up to 87 whole-warp shuffles per first action.
Diagnosis: `/tmp/fpgs-kuka-first-hit-node-audit-zavQZq6b/FINDINGS.md`.

The later lazy-response `eb9c0ec8` retained full J production and also missed:
10.508200 to 10.645301 ms RTX. Its sparse/shared first-use response consumer
cost 1.090220 ms in the node diagnosis, near the retained offset cost;
late positive materialization added 0.207115 ms. Neither version is promoted.

This prototype must replace BOTH boundaries:

- Fold current r0/material metadata and row-to-contact/prefix identity into
  existing allocation. Reuse the existing 20-scalar ContactGeometry packet;
  no new 48-float packet or all-normal J producer remains on the lazy path.
- Keep physical delta velocity. Cache six current-axis scalars and six
  endpoint delta-twist scalars per body lane. Rebuild twists with four linear
  parent-doubling rounds only before a contact residual after committed
  velocity changes. Coalesce intervening prefix/sibling changes.
- Each endpoint's own lane projects its twist at the current correct anchor;
  two scalar gathers form the residual. Prefixes use one coordinate gather.
  Do not gather twelve endpoint components or retain the old J-dot residual.
- Construct J and held R = T-transpose times T times J only on first needed
  response, using the newer sparse/shared action. Cache R/diagonal, not J.
- Preserve the original raw-striped eager producer and complete numerical
  hybrid/general fallback for unqualified or positive coupled MF worlds.
  Price any earlier MF preparation and admission; do not rebuild fallback
  serially inside a late materializer. Preserve original concurrent joins.

## Complete budget and falsification

Strict ca0d RTX diagnostic whole time is 10.636625 ms. Complete affected
rows/response + joined GS + qualification/materialization + MF services is
3.668279 ms. A 10% whole reduction requires this entire family <=2.604617 ms.
Reserve unchanged MF services 0.450102 ms, leaving <=2.154515 ms:

| Replacement planning ceiling | ms |
| --- | ---: |
| New complete GS owner | 1.30 |
| Added allocation/r0/identity work | 0.20 |
| Retained eager stripe/materialization | 0.30 |
| Admission/remaining producer services | 0.21 |
| Total after reserved MF services | 2.01 |

These are falsifiable engineering allowances, NOT measured predictions.
The first integrated paired 16K screen decides; no native-only speed claim.
All affected owners and unchanged allocation cost must be counted, including
fallback, status/epoch work and publication. A loss gets one immediate node
diagnosis; no register/block grid or unfunded consumer-only correction.

Saved-scope FP64 replay supports, but does not prove, the mapping: historical
351 active MF0 cases require 2,494 deferred refreshes / 31,766 contact visits.
Contact shuffle accounting falls 158,830 to 133,364 before implicit-prefix
savings. The selected current 11 hard cases instead require 230 / 992:
accounting grows 4,960 to 8,424. These are not today's 16K GPU populations,
hardware instruction measurements or an occupancy claim. Tail cost is real.

## Required physical and lifecycle checks

Reuse existing current-contact/live-owner fixtures and original numerical
oracles. Regression first; then focused CPU/native checks on both GPUs before
the early whole screen. Preserve every canonical visit, eight sweeps, original
stationary exit, cone and sibling transaction order. Compare residuals, final
velocity/impulses, momentum and friction feasibility, not bit identity.

Include current/held changes, both offset orders, common-arm/self contacts,
static/prescribed/free endpoints, one/three-row layouts, late activation,
MF-independent and positive coupled fallback, empty/regrow/reset and captured
readiness reset. Do not add all-body or per-coefficient certificates for unused
work: retain current admission, executed-residual/first-response finite checks
and final public-state checks. Reset status before allocation; never erase a
current allocator error or a newly produced row afterward. Original public
predictor endpoint twists remain immutable during the solve.

Root exclusively owns GPU jobs. Reuse the existing paired driver and strict
node reader; no new benchmark framework. Full staged pre-commit and exact
source pins precede measurement. Repeat/loaded qualification is funded only
after an integrated gain. No all-task 4x or new MJWarp ratio is claimed.

## Implementation checkpoint, 12:49 UTC

The four producer/admission factories and the complete solve source compile
on CPU. GPU compilation and actual numerical checks are still pending; this
is not a performance or correctness result.

Actual source makes the pre-code shuffle count incomplete: each refresh also
gathers its own coordinate once and six free-body coordinates, in addition to
the 28 parent-scan shuffles. Thus the new counted subset is
`2 * contact_visits + 35 * refreshes`: 150,822 historical and 10,034 for the
selected current hard cohort. The old five-reduction count also omitted the
result broadcast(s), so the original partial counts must not be treated as a
like-for-like hardware instruction prediction. The complete timing budget,
early whole-screen decision and hard-cohort warning remain unchanged.

Source review also requires retaining the common-arm response cache for eager
fallback, preserving the prefix slot reservation across early initialization,
and keeping the representation tag separate from the numerical selector.

At 12:53 UTC the three focused CPU controls and both direct physical CPU
checks pass, including actual partial-arm ancestry, late activation,
shared-anchor cancellation, both coordinate orders, held changes, empty/regrow
readiness, momentum and friction cones. Actual CUDA replay
and integrated timing are not yet run. Review corrected the first-use common
arm predicate to require the original full shared-arm support, and requires
the route array to be persistent binding-owned storage, not a per-substep
allocation.

The early reset does not clear every capacity slot. Current-row coverage comes
from unchanged contiguous successful allocator reservations: each immediately
emits its complete packet or sticks an error. The original prefix predicates
likewise emit every reserved prefix or stick an error; overflow is already a
sticky capacity failure. Thus a valid frame has no unproduced counted row.
The unchanged row validator is supplemental to this producer-coverage proof.

## First native failure and diagnosis, 12:58 UTC

Full staged pre-commit passes. CPU3 passes; the main native physical/held/
captured-readiness test passes on both GPUs. The shared-anchor case fails on
both at normalized impulse difference 8.0757537e-6 versus the unchanged
7.6293945e-6 gate. No whole-task timing was run on this version.

A diagnostic RTX replay (impulse comparisons logged, NOT counted as a pass)
locates the failure at the body13/fixed-child14 row: original impulse
13.9474868774 versus 13.9460897446, difference 0.00139713 in the first epoch;
after held change the difference reaches 0.00244522. A later zero-reference
tangent gets 0.00139460. Velocity, momentum, cone and immutability checks pass.
Independent FP32 emulation identifies one-ULP twist differences caused by
different addition grouping in the four-round parent scan, despite identical
responsive ancestry and an exactly zero original relative J. Positive CFM
amplifies that residual noise into reported impulse.

The bounded correction is to sample the same canonical moving ancestor's
twist for fixed primary endpoints, derived from the existing admitted topology.
Real contact anchors, r0, first-use J/R, iteration law and tolerance stay
unchanged. This is a numerical correction, not a performance claim or gate
relaxation. The same two native tests must pass before the early whole screen.

## Focused gate passed, 13:05 UTC

After canonical fixed-ancestor sampling, the unchanged three CPU tests pass
(2.097 s), and both native tests pass on RTX (7.352 s) and GB300 (7.290 s).
This includes actual CUDA captured replay, empty/regrow, held changes, delayed
friction, partial ancestry, late activation, shared-anchor cancellation,
momentum/cone checks and predictor immutability. Full staged pre-commit passes.
These are focused gates, not long loaded-contact qualification.

Runtime SHA256: `f71363bdeb15d5d51f52d189e1e82a0e8c5a9b88117f2d4dbd9c448d058ab019`.
Binding SHA256: `c73affabc5d1c4e0de644e2ff620ca7637151ac877a5c1ebaf75c54208e58411`.
Test SHA256: `740d270b48adb73f00de365c4aa24e27cc47a0b22821a965bd3d2962b08755b2`.
The early screen uses retained `ca0d427a`, fixed Lab `53ee6b44`, fixed bench
`961b7e2f`, 16,384 worlds, seed0, 200 warmup steps and 40 measured/profile
steps, two substeps/eight solver visits and unchanged 311296/442368 raw/broad
capacities. Both variants keep the closed lazy-response flag off; only the
candidate enables endpoint residuals. No new MJWarp ratio is implied.

## Closed integrated result, 13:21 UTC

Measured runtime `8dcbf9597680e44394471ecfbc7b4145678b349d` is **not promoted**.
It removes substantial row-production work, but does not produce a useful
complete speedup. This version remains default-off; retained `ca0d427a` is
unchanged. No repeated timing or long qualification is funded for this version.

| Paired discovery, ms/environment step | RTX retained | RTX candidate | GB retained | GB candidate |
| --- | ---: | ---: | ---: | ---: |
| Complete physics | 10.462228 | 10.491465 | 10.648697 | 10.601913 |
| Environment wall, not training | 34.525367 | 35.009569 | 37.406492 | 36.126657 |
| Physics retained/candidate | | 0.997213x | | 1.004413x |

All capacity, finite-current-state, source and final-idle guards pass. This
single discovery is effectively flat, not a claim of a sub-percent win. RTX
end-of-run dense rows are 139093 retained versus 141082 candidate; trajectories
are not bit-identical, but there is no capacity-drop explanation for the time.

The immediate paired three-step node diagnosis uses the same source and caps.
Every input has 12 physics roots, 1728 nodes, zero auxiliary/unproven nodes and
passing source/process-correlation guards. The preserved strict interval
reader is reused with name-only aliases. Crucially, the affected family now
includes the **entire allocator** because production work moved into it.

| Strict node interval union, ms | RTX retained | RTX candidate | GB retained | GB candidate |
| --- | ---: | ---: | ---: | ---: |
| Current allocator | 0.207425 | 0.411702 | 0.213909 | 0.344384 |
| Row/response producers | 1.743841 | 0.507297 | 2.186068 | 0.641350 |
| Joined ordered/general solve | 1.403361 | 2.281357 | 1.041365 | 2.134431 |
| MF services | 0.448555 | 0.458539 | 0.499855 | 0.494686 |
| Qualification/materialization | 0.331830 | 0.328406 | 0.410034 | 0.481680 |
| Complete affected union | 4.135012 | 3.987300 | 4.351232 | 4.095476 |
| Diagnostic whole span | 10.897874 | 10.762840 | 10.853483 | 10.645587 |

Rows retire 1.236544/1.544717 ms RTX/GB, but the allocator adds
0.204277/0.130474 ms and joined solve adds 0.877996/1.093066 ms. Only
0.147712/0.255756 ms survives in the complete affected union. GB phase unions
overlap slightly; complete exclusive affected time is 4.348554 to 4.093438 ms.
These three-step diagnostic spans do not supersede whole-screen throughput.

The new offset entry alone costs 2.239426/2.084650 ms versus retained
1.361516/0.992704 ms. Its resource table reports 119/116 registers per thread,
versus 60/56 retained, and 12080 B shared versus 5760 B, with zero reported
local memory. The shared footprint contains both eager and endpoint row
arenas plus readiness scratch. This is declared resource evidence, **not**
achieved occupancy, stall counters or proof that shared memory is the limiter.

Why the complete prediction missed: r0/public metadata work was moved, not
deleted; every contact visit still decodes its packet, loads direction and
anchors and evaluates endpoint projection arithmetic. Dirty refreshes add
the 35-shuffle body scan; first-use J, T and T-transpose actions still cost
time. Reducing a dot product's shuffle count alone did not price these costs.
The producer retirement is real, but the cheaper complete consumer assumed
by the 1.30 ms planning ceiling was not achieved. No register/tile grid or
isolated shared-arena tuning follows this failed screen.

Any further solver redesign must justify roughly 0.94 ms additional RTX
joined-solve removal with current other costs (including its own new block
production/storage) to reach a 10% whole-time target in the paired diagnosis.
A read-only prior-art/accounting study of triplet-local residual reuse is
separate, not authorization for another prototype or a claimed speedup.

Artifacts:

- Whole: `/tmp/fpgs-kuka-endpoint-residual-paired16k-20260915-01`, manifest
  SHA256 `1d82db625fbfd796e991c3f4d34937460208563e47d358a18ced1ea75e8f2f2f`.
- Nodes: `/tmp/fpgs-kuka-endpoint-residual-nodes-paired16k-20260915-01`, manifest
  SHA256 `e85b1e12a568f48391fcfd90a1c6997528f71eb5429ca5112c3b61680a583ec5`.
- Strict outputs: `/tmp/fpgs-kuka-endpoint-node-audit-2lE7fAC3/{baseline,candidate}_gpu{0,1}.json`.
  The reader is `/tmp/fpgs-kuka-lazy-node-audit-AQ9Y0jsn/audit.py`, with only
  endpoint owner names appended to the existing categories and
  `current_allocator` added to `affected_fork`.

### Triplet-local residual follow-up: not funded

Read-only prior-art and exact original FP64 replay accounting found an
implementable four-coefficient cache that would preserve ordered normal,
tangent and sibling transactions. It is not the earlier G1 sticking-block
solver, which changed the numerical law. It still needs four 29-term dot
products per first-used friction block, coefficient loads, normal geometry,
all physical velocity updates and the original cone work.

Current 11 selected controls: 230 refreshes = 86 normal + 72 tangent1 + 72
tangent2; ideal triplet reuse removes 144 (62.61%), with nine first-use blocks.
Historical 351 active MF0 controls: 2494 = 1456 normal + 519 tangent1 + 519
tangent2; ideal removal is 1038 (41.62%), with 65 first-use blocks. The
31766 effective residual visits and 3365 nonzero updates remain. Instrumented
FP64 outputs are byte-identical to the existing uninstrumented oracle.

Removing about 41.2% of the entire current joined solve would therefore
require almost all of it to be removable scan cost at the historical average,
before new block work. The selected hard tail is not population evidence.
This does not fund a substantial-gain prototype; no runtime/test changes or
GPU tuning were made for it. Do not reopen it as an untested new 4x direction
without a materially different complete budget and exposure evidence.
