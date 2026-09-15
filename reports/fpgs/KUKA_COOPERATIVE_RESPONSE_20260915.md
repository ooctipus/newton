# Kuka cooperative response: one causal correction

Status: **diagnosed loss; the one corrective experiment is closed**. Physical
controls pass, but clean whole time is slower than accepted current-contact
on both cards. No promotion or further mapping grid is justified here.

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

## Frozen physical and whole result

Numerical source/test tip is
`e5d682c8c8702012bbcbb6f9bacc3f87588b8c44`. Both existing CUDA selectors pass:
RTX two tests in 4.204 s (session 68868), GB300 two in 4.386 s (48192), both
reaped zero. They cover original/native numerical law, mixed one/three-row
slots, nonleader-warp errors and repair, plus actual two-bank streams/graphs,
positive-MF transitions, held/current cadence and failure withholding public
finish. The only warning is the inherited target-layout deprecation. The
selector names and nine-flag isolated-device recipe are in the predecessor
card; tolerances and task allowances are unchanged.

The original paired whole parent 27686 reaped zero with all four children
zero at this clean frozen tip. This is ONE discovery round, not repeated
promotion evidence. All values below are ms per batched environment step;
wall is synchronized environment time, not RL training throughput.

| Card | Accepted physics | Cooperative physics | Candidate minus baseline | Accepted wall | Cooperative wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | 10.411883 | 11.834728 | +1.422845 | 35.638270 | 36.318580 |
| GB300 | 10.678865 | 11.665263 | +0.986398 | 36.380676 | 37.412236 |

The baseline is clean accepted current-contact
`629dedb057cc34cb05139b98d6dece5609ae306e`, never the slower first private
prototype. Runtime imports, source and final idle guards pass. Both arms use
the eight accepted flags including CURRENT_CONTACT=1; PRIVATE_RESPONSE is
0 only in baseline and 1 only in candidate. Fixed Lab remains
`53ee6b44c2334341305dbdf385a3916c6b140799` (only untracked `.venv` recorded),
with actual selected Newton and fixed backend imports verified. Budgets stay
16,384 environments, seed0, 200 warmup/40 wall/40 graph steps, raw311296,
broad442368, dense192/MF64/propagation192, two substeps, solver dt1/240,
simulation dt1/120, decimation4 and original eight sweeps. The timing driver's
physical-acceptance field remains false: the separate bounded CUDA tests are
not silently relabeled a full production physical qualification.

Whole artifacts:
`/tmp/fpgs-kuka-cooperative-response-paired16k-20260915-01`, manifest SHA256
`f4b8de86833886f4d4cf174668a90121cf9e7394db0348e024ac245de3af15c3`.

## Final strict attribution

Node parent 37051 reaped zero, with all four children and final source/idle
guards passing at the SAME source. Output:
`/tmp/fpgs-kuka-cooperative-response-nodes-paired16k-20260915-01`, manifest
SHA256 `8d7e8923871a03e14b99922de548c868c25caed86dd5c721a441fb9d5d529f69`.
The existing ce09 strict interval reader, unchanged apart from reviewed
owner/family aliases, produced `strict_cooperative_response_node_audit.json`
inside each `round_01_{baseline,candidate}/round_01_kuka_fpgs_gpu{0,1}`.
Every audit has 12 physics roots, 1728 baseline or 1752 candidate nodes,
zero unproven nodes/auxiliary roots and successful source/process/correlation
checks. These three-step diagnostic spans are kept separate from graph whole
timings above.

| Strict interval, ms/environment step | RTX baseline | RTX candidate | GB baseline | GB candidate |
| --- | ---: | ---: | ---: | ---: |
| Whole physics root spans | 10.538243 | 12.058912 | 10.702230 | 11.490696 |
| Complete affected fork union | 3.595766 | 5.092613 | 3.967386 | 4.763133 |
| Affected fork exclusive busy | 3.595766 | 5.046533 | 3.964324 | 4.721917 |
| State/held union, retained | 3.204715 | 3.213205 | 2.824085 | 2.822292 |
| Current geometry union, retained | 0.859392 | 0.871767 | 0.791443 | 0.789449 |
| Allocator including candidate slot map | 0.208192 | 0.216203 | 0.212693 | 0.214955 |
| Collision union, retained | 1.832513 | 1.869367 | 2.065333 | 2.088596 |

The complete affected family is rows/response + MF services + qualification/
materialization + positive offset/general + private response. It excludes
current geometry and allocator, which are charged separately rather than
forgotten. Candidate family sums are 5.707505 / 5.492076 ms, but the measured
unions above, not these sums, determine overlap accounting. RTX complete
disjoint accounting is affected-exclusive5.046533 + prefix0.103829 +
collision1.869367 + allocator0.216203 + geometry0.871767 + sensors0.282251 +
memory-exclusive0.212949 + state/held3.213205 + overlap-once0.046080 +
gap0.196728 = 12.058912 ms. No large memset or hidden retired-owner saving is
inserted into this budget.

| Candidate owner sums, ms/environment step | RTX | GB300 |
| --- | ---: | ---: |
| Complete private response/eight | 3.146638 | 2.952277 |
| Masked retained rows/response | 0.625889 | 0.736975 |
| Retained MF services | 0.455851 | 0.503692 |
| Qualification/materialization/guards | 0.371702 | 0.365736 |
| Positive-MF offset | 0.682219 | 0.631284 |
| Positive-MF general | 0.425206 | 0.302112 |

Actual private resources match the cooperative binding: eight calls per
environment step, grid16384/block128, 96 registers, 31,840 shared bytes and
zero local memory on both cards. Positive offset retains block64, 5760 shared
bytes and 60/56 registers; general retains block32, 4340 shared bytes and
100 registers. There is no measured occupancy counter or isolated private
contact-versus-sweep phase timing in these captures.

### Why the conditional budget failed

The first private RTX owner cost 6.153778 ms; this correction costs 3.146638
ms, about 1.956x diagnostic recovery. It is NOT a task speedup versus accepted
current-contact. The private owner alone exceeds the entire pre-code
2.595454 ms replacement allowance by 0.551184 ms, before added allocator
work, exposed retained main work and gaps. The ideal four-way construction
premise requiring roughly 79% contact time did not yield its required result.
The unchanged full panel and leader-only prefix/eight lifetime remain real
source constraints; their individual time shares are not measured here.

Removing the explicit private-to-offset dependency also did not achieve the
ideal max(private,main) overlap. RTX retained main affected owners sum to
2.560867 ms; only 0.614892 ms of their summed intervals overlap private within
the affected union. Positive solves together are sum=union=exclusive
1.107425 ms: their actual GPU intervals do not overlap private or another
category in this capture. Thus independent host streams do not make the
sequential positive work free. The source still pays its row/MF readiness
chain and lost original positive offset/general overlap. These measured
intervals establish the failed overlap premise, not a claim that register
counts caused achieved occupancy or identify a unique hardware stall reason.

The complete affected union is higher by 1.496847 ms RTX / 0.795747 ms GB;
allocator deltas are only 0.008010 / 0.002262 ms. Retained state and collision
remain close to their controls. The loss is concentrated in the complete
replacement boundary rather than an unrelated publication/collision change.

Close the one corrective experiment as a diagnosed loss. Preserve both first
private and cooperative sources and their distinct whole/node evidence. No
promotion, repeat against the slower prototype, further tile/warp grid or new
MJWarp ratio follows. Reopening needs a genuinely different work-removal
hypothesis and evidence, not another mapping of this same full panel.

## Exact existing-protocol reproduction

Retain the source pins above in separate clean checkouts and original tool
tree `newton-fpgs-structural-bench-20260913` at
`961b7e2f751bcd1d8b03368e7956b54c81414897`. The mandatory existing adapter is
`/tmp/fpgs-kinetic-fixed-variants-TZRkIPYw/run.py`, SHA256
`c062388f23f82c524690f272fb2d14141ed4126db641de6dcd86f8ade5bfe416`.
It preserves original `compare_variants.py` SHA256
`48406c079588d3088fc25c1cdce2bc2659aaec28539013e5811c9c55378445b5`,
`compare_backends.py` SHA256
`79818fce1deab05b9409d386d7a240265cf2f7e5b7f019109f3ab99c9f52e403`,
and `checked_capture.py` SHA256
`42b289bd0082d194180d5981651a108d11a66058ff4e67649ad7ddb408c9bea6`.
The current repository copies are not interchangeable with these pins.
Preserve the fixed import startup identified in each manifest; no direct
driver command may omit that explicit Lab package selection.

The tested one-round graph settings are reproduced below; choose a NEW output
directory and let the original driver enforce both UUID idle/source guards.
This is documentation, not a requested additional GPU run.

```bash
kuka_variant_args=()
for kuka_flag in SIMPLE_WORLD_ZERO LOCAL_ROW_PACKETS INDEPENDENT_COMPONENTS \
  PAIRED_GENERAL_OVERLAP KUKA_JOINT_WORLD WORLD_SCAN_PUBLICATION \
  KUKA_KINETIC_WORLD KUKA_CURRENT_CONTACT; do
  kuka_variant_args+=(--baseline-env "FEATHER_PGS_${kuka_flag}=1")
  kuka_variant_args+=(--candidate-env "FEATHER_PGS_${kuka_flag}=1")
done
uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python \
  python /tmp/fpgs-kinetic-fixed-variants-TZRkIPYw/run.py \
  --isaaclab /home/octi/Projects/IsaacLab.wt/contact-reset-20260913 \
  --baseline /home/octi/Projects/newton-fpgs-kuka-current-contact-20260915 \
  --candidate /home/octi/Projects/newton-fpgs-kuka-cooperative-response-20260915 \
  --task kuka --gpus 0 1 --rounds 1 --seed 0 --num-envs 16384 \
  --warmup-steps 200 --steps 40 --profile-steps 40 --trace-mode graph \
  --capacity kuka:fpgs:rigid_contact_max=311296 \
  --capacity kuka:fpgs:broad_phase_output_max=442368 \
  "${kuka_variant_args[@]}" \
  --baseline-env FEATHER_PGS_KUKA_PRIVATE_RESPONSE=0 \
  --candidate-env FEATHER_PGS_KUKA_PRIVATE_RESPONSE=1 \
  --output-dir /tmp/fpgs-kuka-cooperative-response-reproduction-new
```

For the closed node protocol only, use `--profile-steps 3 --trace-mode node`
and another new output directory. The strict reader/source aliases described
above are unchanged. Closing documentation introduces no runner, runtime,
test, native-factory, Lab or task-budget change.
