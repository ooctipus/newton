# Kuka cooperative response: one causal correction

## Pre-code contract

Authorized 2026-09-15 near 02:01 UTC; first source/offline/cost-ready checkpoint
03:30 UTC. This isolated tree starts at
`867e6b77c7186c31858a688638076578337f23ed`, preserving the complete first
[private-response experiment](KUKA_PRIVATE_RESPONSE_20260915.md). No Lab,
physics budget, public capacity, row order, numerical tolerance or default-off
admission changes. Root owns all GPU runs. No mapping grid or new harness.

The original default-off flags select this successor only in the new tree.
The comparison baseline for an incremental gain remains qualified
current-contact (`c1c678a`, report successor `629dedb0`), not the slow private
prototype. The prototype is a causal diagnostic control, not an accepted
performance baseline.

## Measured mismatch and conditional budget

The first private owner passed both direct native and actual two-bank CUDA
selectors on both cards. Its initial graph screen had test-only source drift
and is labeled provisional in the predecessor card. The subsequent clean
three-step node pair at 867e passes all original guards:
`/tmp/fpgs-kuka-private-response-nodes-paired16k-20260915-01`.

The strict complete affected union is RTX 3.564513 -> 7.128285 ms and GB300
4.095402 -> 5.439198 ms. RTX exclusive contribution is 7.065714 ms with
0.062571 ms memory overlap. Slot-map allocator deltas are 0.005323 ms RTX and
0.003093 ms GB. The private kernel alone is 6.153778 / 4.7505 ms; retained
positive-MF offset costs 0.880502 / 0.617355 ms and is serialized after it on
the same stream. Native resources in the actual captures are 99 registers,
27,984 shared bytes and zero local memory on both cards.

Two concrete causes motivate ONE combined correction:

1. Contacts that were raw-striped across 4096 workers are serialized per world
   in a one-warp owner with its full panel resident throughout construction and
   eight sweeps. Four warps will construct independent contacts cooperatively.
2. The independent positive-MF offset owner unnecessarily waits for the private
   MF0 owner. Move positive offset then general to the already-required main
   stream; retain one final join with the private stream.

The second change loses the original positive offset/general overlap and must
pay their sum. The 15 existing main-chain owners, including both sequential
solves, sum to **2.521943 ms RTX / 2.582270 ms GB**. Present arm-to-general
spans total 3.962372 / 3.357237 ms, so their queue/resource stalls are not free.
No achieved occupancy or measured private phase fraction is inferred.

The complete replacement allowance remains **2.595454 ms RTX**, including
map delta, exposed memory, guards, gaps, masked fallback and both streams.
It leaves only 0.073511 ms above the observed main-kernel sum before those
charges. Removing the false edge changes the private target from about
1.715 ms to approximately 2.5 ms. Under an ideal four-way-contact model with
unchanged leader work, contact construction must account for about 79% of
the current 6.154 ms private time. That fraction is a condition, not evidence.
The correction may recover the loss without achieving a substantial whole
gain. Measure the complete max(private, sequential main) region and unchanged
whole task, not just a faster private kernel.

## Ownership and physical invariants

Use exactly one 128-thread CTA per world. Warp zero performs original cache
initialization and prefix rows; all four warps rendezvous. Contact slots are
striped as prefix_count + warp_id + 4*k, skipping tangent-map entries -1.
Never assume three-row alignment: normal-only contacts and arbitrary prefix
counts retain their original allocations. Every normal owns its original
one or three rows. Retain one Z192x29 panel and one held/current cache, with
four disjoint 288-float contact scratch areas and per-warp error ownership.

A second block-wide rendezvous completes all response/metadata/error writes
before warp zero merges errors, validates and runs the unchanged eight-sweep
law. Only warp zero enters that recurrence, keeping its original world_slot
zero. Every early return before a block barrier must be block-uniform. No
concurrent warp may overwrite another warp's scratch or plain error state.

The early private ready/done event pair remains. Main-stream offset then
general consume completed main rows/MF/qualification; a finally-join waits
for private completion before original full guards and public finish. Do not
overwrite the private done event with a second same-stream solve. Existing
reset/notification, two-state banks, current/held cadence, capacity failure,
force output and positive-MF full fallback remain in scope.

## Minimum verification and stop rule

Reuse existing CPU regressions, both CUDA selectors and actual-entry offline
compilation. Direct helper, production launch and offline entry must all use
the same 128-thread mapping. Add only bounded controls for cooperative normal
1/3 slot coverage, error retirement and the changed event ownership; do not
weaken existing invariants. Original eight-source recovery stays exact.

Freeze after CPU/offline/pre-commit readiness, then root runs physical smoke
and the original integrated whole/node protocol promptly. No promised gain,
promotion or further mapping retry follows from source/resource counts.

## First source/offline checkpoint

The combined correction changes only `kinetic_private_response.py`, its
existing test module and this card. Runtime flags, binding/descriptor ABI,
allocator, complementary rows, positive fallback and original eight-sweep
source remain unchanged. Actual launch and direct boundary use `BLOCK_DIM=128`.
The regression-first mapping test failed before implementation because the
original owner had no cooperative block binding (session 88164).

The arena retains all original panel/cache offsets through float 6276. Three
additional 288-float scratch regions occupy [6276,7140); four int error words
occupy [7140,7144). Only each contact warp publishes its own errors. The old
error-dependent early skip is omitted within the contact loop: invalid
contacts still report failure and never publish a solved/public state, while
other disjoint contacts may finish before the joined validation fails.
Warp zero initializes all error words even on rejected begin, and merges them
only after all contact writers rendezvous. Prefix/eight use their original
warp-local arithmetic and fences.

The added bounded fixture has an odd prefix and twelve actual contacts with
mixed one/three-row allocations spanning all four warp residues. It compares
the private solution with original row/eight kernels on the SAME completed
allocation, avoiding a false equal-atomic-order assumption. It poisons one
normal map owned by warp two, checks failure/no velocity write, restores the
map and checks error retirement. Both existing CUDA selectors are retained;
the direct selector also runs this mixed/error fixture on the actual device.
CPU selection passes 45 tests with three explicit root-only CUDA skips
(48 selected, 3.062 seconds, session 93376). No tolerance was changed.

All nine actual entry families compile successfully for SM120 and SM100,
with original entry/source guards and no GPU context. Private compilation
uses the actual 128-thread binding: **96 registers, 31,840 shared bytes,
zero stack and zero spills on both architectures**. Each emitted PTX contains
exactly two block `bar.sync 0` sites (lines 1084 and 2921); the second precedes
the four shared-error reads and leader-warp recurrence. The two sites reuse
one hardware barrier. The extra warps do not duplicate the response panel.
These resource observations are not an occupancy or performance claim.

The preserved original compiler is rebound only for the new source root and
private block128:
`/tmp/fpgs-kuka-cooperative-response-offline-BzpXK8oB/compile.py`, SHA256
`296b695d5aadeb29e713f57b67fc868f7e131235cb4bfd5e7e7cf3f16b9de51c`.
Report `offline01/report.json` in that directory has SHA256
`a017016a6ad44919b736ce338631471779dcbf8898b49307c4634403b35015fb`.
Frozen native SHA256 is
`8b7a7202d392b6d076c7925cb2e8f09dcd6bc49df086b3df5488cd447b898ef8`.

The first full pre-commit pass succeeds; root independently reviewed the
layout, block-uniform returns/barriers and changed stream join without a
source blocker. Actual CUDA qualification and integrated cost remain pending;
root can run the same two selectors and original current-contact comparison
without a new benchmark wrapper.
