# Kuka current-contact ownership experiment — 2026-09-15

Status: complete default-off prototype, CPU/offline ready; no GPU measurement
or promotion. Enable only with `FEATHER_PGS_KUKA_CURRENT_CONTACT=1` alongside the
already-qualified Kuka kinetic owner. The established flag prefix is accepted by
the unchanged benchmark driver; no driver whitelist change is required.

Base: `1a9efc33efbc0f23e1e7676a5edded795f224c97`, branch
`ooctipus/fpgs-kuka-current-contact-20260915`. Original worktrees are preserved.
Start: 2026-09-14 23:58 UTC; first complete CPU/offline checkpoint by 01:29 UTC.

## Falsifiable complete boundary

The source-matched current-capacity RTX diagnostic assigns 2.978533 ms to
raw CSR (0.308832), ZERO (0.571403), prefix count initialization (0.195680),
raw contact allocation (0.224853), and contact-triplet response (1.678444).
The entire replacement family must cost at most 1.978533 ms to save 1 ms.
This is a funding budget, not an expected or measured speedup. The remaining
allocation/finalization work and 0.283189 ms of all memory activity are not
credited as retired work. Reference:
`/tmp/fpgs-kuka-current-node-audit-20260914-Rvnhvi37/candidate_gpu0.json`.

One integrated producer will classify complete raw incidence and publish
current geometry/endpoint descriptors without preparing held responses for
contacts in correctly ZERO worlds. Subsequent original allocation policy and
triplet contraction consume that descriptor. CSR and repeated geometry/count
producers must be replaced, not retained alongside the new path.

## Non-negotiable contract

- Preserve both arithmetic laws: ZERO's endpoint-relative separation/phi and
  allocation/row construction's world-anchor separation/phi. They are not
  interchangeable in floating point.
- Preserve original raw-index plus/minus friction rank, all normal/tangent and
  restitution laws, authoritative allocation slots, capacities and overflow
  latches. Do not substitute compacted neighbors for raw neighbors.
- Reproduce dynamic, prescribed, free/MF and complete fallback ownership;
  correctly ZERO worlds perform no response construction. Keep full original
  row order, limits, force-export routing, held/current cadence and budgets.
- Produce from the actual current state on every substep, even when collision
  raw storage is reused. Cold/reset/state alias/empty/grow paths may not observe
  an earlier state/substep packet. Default-off retains original source/ABI.
- No timestep, iteration, geometry, capacity, tolerance or Lab changes.

Highest-risk seam: replace ZERO's complete raw-contact proof and conservative
invalid-incidence behavior while keeping its relative-geometry rounding law
distinct from the allocator's authoritative world-anchor row admission.

## Prior art and qualification

The old packet-boundary card (`/tmp/fpgs-kuka-packet-boundary-zdIvIRN1/CARD.md`)
closed a standalone response-packet proposal at its then-current budget; it did
not measure this combined geometry/ZERO owner. Accepted triplet RHS batching,
support masking and common-arm elimination are already included in the base
and are not claimed again. Avoid global response packets and two-phase shared
anchor broadcasts that previously added serial dependency chains.

Use existing kinetic geometry/lifecycle fixtures first (failing regression
before implementation), then CPU controls and actual-entry offline compilation.
Root alone launches GPU physical smoke and the original complete whole/node
protocol. No new benchmark framework or census. One diagnosed corrective mapping
is permitted after a first loss, not a variant grid.

## Implemented boundary and ownership

`kinetic_current_contact.py` owns five entries: world readiness plus an
independent 46-candidate prefix count; one complete raw incidence/dual-geometry
normal check; ZERO/counter publication; original allocation policy consuming
current descriptors; and the original triplet contraction consuming 20 staged
scalars. Two lazy seams in `kinetic_live_bindings.py` allocate scratch before
capture and replace the actual bound launch closures. The original default
native modules and ABIs are unchanged.

On this path `RawWorldContactBuckets.build`, old `kinetic_zero`, and
`initialize_count` are not launched. Dense tangent/material staging happens only
after successful original allocation, with original raw-index rank and next
neighbor tests. No response is formed for a correctly ZERO world. Original
free/MF preparation (including its geometry), free velocity limits, three count
finalizers, prefix row production, mixed/general solves and public publication
remain; none of that retained work is claimed as retired. Original row slots
are still assigned by the original raw-worker atomic allocation policy, with
normal/tangent order and all 46 ordered prefix candidates preserved.

There is one solver scratch bank, not one bank per state. Its capacity cost is
88 bytes per raw slot plus 12 bytes per world: **27,590,656 bytes** at the current
311,296 raw / 16,384 world recipe, or 352,000,000 bytes plus world storage at the
original 4M maximum. Old fallback CSR storage remains allocated but unproduced
on this path. This capacity and packet traffic are charged to the experiment.
The existing joined solve and same-stream producer sequence protect lifetime;
every current substep overwrites the active raw prefix, even if collision
generation is unchanged. There is no collision-generation-only reuse test.

## CPU and offline checkpoint

The first new regression failed with `ModuleNotFoundError` before the module
was implemented. Existing simple-world numerical tests are reused against the
new raw checker. A real live Kuka topology fixture covers self dense, free MF,
arm/free mixed rows, prescribed motion, active scalar prefix, shared anchors,
raw-neighbor friction filtering, ZERO omission and changed current geometry.
Raw invalidity/overflow, empty/shrink/regrow and current-generation mismatch
are tested. CPU comparisons use original current row producers as the oracle;
the normal-law tests preserve original geometric/restitution/margin controls.
The final CPU batch passes 44 executed tests plus one explicit CUDA skip
(45 selected). Repeating the 22 existing binding/lifecycle tests with the new
flag enabled also passes, including state banks, masked reset, notification,
held demotion and public force routing. The old native/source oracle is retained.

All five actual CUDA entries compile offline for both SM120 and SM100:

| Entry | RTX registers | GB registers | Shared bytes |
| --- | ---: | ---: | ---: |
| World prepare/count | 38 | 32 | 2304 |
| Raw geometry/ZERO | 64 | 64 | 1024 |
| ZERO/count publication | 40 | 32 | 1024 |
| Original-policy allocation/staging | 40 | 32 | 1024 |
| Cached contact triplet | 80 | 80 | 1280 |

All ten compiled entries have zero stack and zero spills. These are resource
facts, not achieved occupancy or speedup. The original triplet also used 80
registers. The retained offline compiler is mechanically rebound at
`/tmp/fpgs-kuka-current-contact-offline-EdTFpXg9/compile.py` (SHA-256
`033777b4d376cb85fcf6ac944aedded03f5467c8611716b11aafd915efd3c561`).
Its actual-entry/source-guard report is `offline01/report.json` (SHA-256
`7dde54d3da72eb17eeebb5e2f07be6ac65977a7b4550dcca5c652d4832228642`).
Native source SHA-256 is
`17ad7f0827da875ae4ba77b0eeb4653c0c323077c4feaf1260860ac3bbd81196`;
binding source SHA-256 is
`8c75cb5c64b97665db9bdc3d619d40b20beff3fe318acf3d1f00ef8e0cb74bb5`.

## Root-owned next check

The standard unit selector below executes the actual new CUDA boundary on
existing CPU-generated current/held fixture operands, compares rows by raw ID
against original current producers, and replays two graphs through
empty/regrow prefixes. It is a boundary smoke test, **not** full live dynamics,
eight-sweep or whole-physics qualification. Follow with the existing live
physical helper and original whole timing recipe; do not add a benchmark
framework. Use an idle, UUID-isolated device selected by root:

```bash
CUDA_VISIBLE_DEVICES=<root-selected-UUID> FPGS_TEST_DEVICE=cuda:0 PYTHONPATH=. \
  uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python \
  python -m unittest \
  tools.fpgs_bench.test_kinetic_current_contact.TestKineticCurrentContact.test_cuda_current_contact_boundary
```

CPU checks use the same interpreter with `CUDA_VISIBLE_DEVICES=''`, selecting
`tools.fpgs_bench.test_kinetic_current_contact`, `test_kinetic_live_bindings`,
`test_kinetic_live_owner` and `test_kinetic_native_port`. Repeat the two lifecycle
modules with `FEATHER_PGS_KUKA_CURRENT_CONTACT=1`. The first complete CPU/offline
checkpoint was reached around 00:26 UTC, well before the 01:29 deadline.

## Completed physical checks and repeated timing, 01:12 UTC

Runtime is clean `c1c678a8738b306a32ed6c241b7cb004c93804aa`; the baseline is
qualified `1a9efc33efbc0f23e1e7676a5edded795f224c97`. The new boundary CUDA
selector passes on both cards in 4.315/4.483 s, including original-current
normal/tangent/MF physical rows and two graphs with counts 0/8/1/8.
Logs: `/tmp/fpgs-kuka-current-contact-cuda-20260915-6QaLN8zd`.

The existing eager 4K loaded-trajectory qualifier completes 1608 Newton calls
on both cards, sampling refresh1600 and held1601. Actual new-owner identity,
current geometry, original eight-sweep physical criteria and original public
finish pass. Momentum backward defects are at most 1.746e-7 RTX /3.061e-7 GB,
cone error 4.374e-8 /2.304e-8, and contact-normal excess 4.905e-7 /5.132e-7.
The trajectory contains ZERO, MF0 and independent worlds but no coupled cohort;
its `contact_eight_accepted=false` and `missing_cohorts=["coupled"]` are retained,
not relabeled as complete natural-cohort coverage. Evidence:
`/tmp/fpgs-kuka-current-contact-live-physical-paired4k-20260915-01`.

The unchanged loaded coupled/type-4 fixture separately passes against the exact
c1 production module aliases on both UUIDs: one test, no skips, 1.325/1.349 s,
sessions64985/32054, both clean exit0. Selector:
`test_kinetic_compact_physical.TestCompactPhysical.test_cuda_loaded_capacity_and_vlim`
in `/tmp/fpgs-kuka-compact-coupled-JaVsIZBa`. It retains loaded192/64 capacities,
swapped offsets, current MF distinct from held free inertia and active stateless
limits. This is compositional coverage of the retained mixed solver; it does
not itself exercise the new raw cache or establish coupled trajectory coverage.

The original paired benchmark is unchanged: fixed corrected Lab
`53ee6b44c2334341305dbdf385a3916c6b140799`, 16K worlds, seed0, 200 warm steps,
40 wall steps, 40 profiled steps, original eight-sweep/two-substep allowance.
Both arms retain raw311296/broad442368/dense192/MF64/propagation192 capacities
and SIMPLE_WORLD_ZERO, LOCAL_ROW_PACKETS, INDEPENDENT_COMPONENTS,
PAIRED_GENERAL_OVERLAP, KUKA_JOINT_WORLD, WORLD_SCAN_PUBLICATION and
KUKA_KINETIC_WORLD. Only KUKA_CURRENT_CONTACT changes from0 to1.

| Three alternating paired rounds, median ms/env step | RTX | GB300 |
| --- | ---: | ---: |
| Baseline physics |10.999563|11.276758|
| Current-contact physics |10.480220|10.689198|
| Physics baseline/candidate |1.049555x|1.054968x|
| Baseline environment wall |35.900195|38.259504|
| Current-contact environment wall |35.257890|37.119032|
| Wall baseline/candidate |1.018217x|1.030725x|

These are environment stepping, not full RL training measurements. Candidate
physics samples are [10.475700,10.535112,10.480220] RTX and
[10.738321,10.689198,10.685025] GB. All12 child processes exit0 without cleanup
signals; all24 finite/capacity boundaries pass. All16 helper/source hashes match;
the final source/idle guard passes. The unchanged Lab has only its existing
untracked venv symlink. Full recipe, sources and artifacts:
`/tmp/fpgs-kuka-current-contact-repeat-paired16k-20260915-01/manifest.json`,
SHA256 `7760ca66391b3375b00f22c733dc87e2e080c0320d8035cf10ba48fd833f8557`.
The prior one-round discovery is separately retained at
`/tmp/fpgs-kuka-current-contact-paired16k-20260915-01` (RTX10.991662->10.468107,
GB11.338841->10.640728 ms); do not pool it with the three repeats.

### Complete node diagnosis and next structural opportunity

Original node analyzer and strict membership audit pass all four captures,
12 physics roots each, zero unproven correlations. Physics nodes fall from
1920 to1728. The complete replaced family is2.977986->2.543053 ms RTX and
3.404967->2.861500 ms GB. All five old producers disappear; all five new entries
execute eight times per environment step. The retained triplet contraction
still costs1.476172/1.860778 ms, or58%/65% of the new family. Thus the original
1 ms whole-saving plan was missed because expensive contraction remains, not
because old producers were left running or a register-spill regression.
The measured approximately5% whole gain remains useful, but it is not4x.

Evidence: `/tmp/fpgs-kuka-current-contact-nodes-paired16k-20260915-01` and
`/tmp/fpgs-kuka-current-contact-node-audit-zU6BTmOA`. Using the previous corrected
MJWarp denominator only gives approximately2.62x RTX /2.42x GB; this experiment
does not measure MJWarp anew. A separate next experiment will test an early
MF0 response/solve fork, including retained mixed work and all join/guard costs.
