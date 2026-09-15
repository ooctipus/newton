# Franka kinetic paired16: one complete-owner correction, pre-code

Status: proposal only; no runtime edit, new worktree, or GPU execution authorized
by this card. Root must read and approve before implementation.

## Frozen cause and comparison

Read the complete latest report at
`/home/octi/Projects/newton-fpgs-franka-kinetic-state-20260915/reports/fpgs/FRANKA_KINETIC_STATE_20260915.md`,
report HEAD `edf8b25200afbc53fe974ec137f0ab470e3bc339`.
If funded, create one isolated branch/worktree from host-fixed native parent
`edd49d0b407a7e0720e0f028e067de90861fb363`; preserve the old tree and all failures.
Accepted whole-performance comparator remains qualified `1a9efc33`, not the
unsuccessful kinetic prototype. Reuse its exact original runner and capacities.

Post-host-fix whole02 measured RTX 5.457961 -> 5.466557 ms and GB
5.025921 -> 5.383826 ms. Wall remained 28.784147 -> 44.972803 ms RTX and
27.884037 -> 44.296596 ms GB. No promotion. The remaining approximately 16 ms
host-proof cost is separate and remains charged; do not optimize hashes in
this experiment or claim improvement over the broken 1.69-second prototype.

Matched nodes are at
`/tmp/fpgs-franka-kinetic-state-nodes-paired16k-20260915-01`.
The original `ce09a496` strict reader confirms 12 physics roots per capture,
three environment steps, and 24 calls to each of the three new owners. These
are three diagnostic steps, not three repeated throughput benchmarks.

| Complete state/factor/drive/mask family (ms) | RTX old | RTX new | GB old | GB new |
| --- | ---: | ---: | ---: | ---: |
| Summed kernels | 2.513538 | 2.055565 | 2.089395 | 2.141363 |
| Union | 2.105090 | 2.024397 | 1.695736 | 2.113129 |
| Exclusive busy | 2.026978 | 2.019715 | 1.635162 | 2.105043 |

New RTX finish/predict/repair sums are 1.047489 / 0.538603 / 0.216000 ms.
GB values are 1.177771 / 0.534379 / 0.191637 ms. No repeated full-repair or
H-request bug was found: one expensive first repair per environment step,
then short current/held guard calls. All retired global producers are absent.
Old tau/factor/composite overlap was 0.271510 ms RTX / 0.260445 ms GB.
**Packing does not restore this overlap, and receives zero credit for it.**

## One fixed mapping across all three owners

Use one 32-thread CTA for exactly two independent worlds. Hardware lane
`h` selects subgroup `g=h>>4`, local lane `l=h&15`, and world `2*CTA+g`.
The one mapping applies to repair, finish, and current-force/held-factor
prediction. Keep the three launches separate, eight calls each per environment
step; no phase fusion, new producer, persistent queue, or block-size sweep.
Keep the existing default-off `FEATHER_PGS_FRANKA_KINETIC_STATE` admission and
factory ABI. Add a distinct `_p16` suffix to these three kernel names for the
existing untimed actual-owner checks; factor9 and its name stay unchanged.

Each subgroup owns its own original 896-float state scratch, or 120-float
predictor scratch. Retain the existing offsets and global KineticPlan/cache
layout; this is not a scratch-relayout or cache-lifetime experiment. CUDA grid
size becomes ceil(worlds/2) CTAs, block32. The CPU reference path continues
one complete world per logical item and its existing arithmetic.

Preserve all 13 physical bodies, 21 generalized coordinates, primary9/free6/
prescribed6 ownership, and the exact parent-jump composition order. Pose and
motion scans still perform four rounds; only padding lanes13..15 participate.
Use subgroup masks `0x0000ffff`/`0xffff0000` and shuffle width16. Replace every
full-warp sync/vote used by these owners with its corresponding subgroup
operation; no block barrier may depend on another subgroup's cache admission.
Out-of-range odd tails and current-valid early returns must be uniform within
their own 16 lanes and independent of the other subgroup. A valid subgroup
must not access the absent world's plan, memory, or source tags.

Keep the same body/subtree, per-row and per-factor arithmetic order. In
particular, subtree reductions remain body10 down to1; L9/L6 solves remain
the original forward/back substitutions. They now occupy four lanes per
hardware warp across two worlds instead of two lanes for one world. No
precomputed inverse, new factor arithmetic, coefficient truncation, changed
PGS projection, iteration, tolerance, timestep, or contact law is allowed.

## Complete cost allowance and adverse work

All three owners total 1.802093 ms RTX. Ideal 2x throughput, not a prediction,
would yield 0.9010465 ms and save 0.9010465 ms. The requested conservative
owner reduction is at least 0.750000 ms: combined time <=1.052093 ms, or
1.713x faster. Thus the mapping has only 0.1510465 ms headroom above ideal
halving (about 16.8% of that ideal replacement cost).

Explicitly charge:

- 21-coordinate update/predict output requires two local-lane rounds, not one.
- Refresh collection of 19 independent components requires two rounds, not
  one; bias-only collection still has six components.
- H81 output takes up to six iterations per lane instead of three. H81 values,
  complete nine-entry rotated inertias and factor9 input are unchanged.
- State payload doubles from 3584 to 7168 shared bytes per CTA; including the
  old 128-byte generated overhead projects about 7296 bytes, not an AOT fact.
  Predictor payload doubles from480 to960 bytes (about1088 with old overhead).
  Compile and charge actual registers, local stack, shared bytes and barriers.
  Old finish uses80/90 registers and72 stack bytes; no occupancy claim follows
  from packing, and larger shared storage can reduce resident CTAs.
- All physical memory traffic, four pose/motion rounds, required full free
  I36 and prescribed V6 services, source/status checks and live external-force
  reduction remain. The split intrinsic/external reductions remain separate.
- Retained factors, drives, row formation, PGS, sensor/publication consumers,
  clears and all overlap remain charged. Do not assume the observed RTX mixed
  packet growth or remaining host regression disappears.

Finish plus repair alone could save only 0.631745 ms even under ideal halving;
that does not support the 0.75 ms margin. Predictor must be included. Its
additional ideal-halving opportunity is 0.269302 ms. The previous paired
publication trial's approximately0.2 ms gain is not proof of this result:
this proposal also packs the measured0.539 ms live-force/held-L owner and
the compact bias/H producer, whose caches did not belong to that old boundary.
It remains a packing correction, not a globally new algorithm.

## Verdict and bounded qualification if funded

**Plausible enough for one causal experiment, but high risk.** No source bound
makes <=1.052093 ms impossible: most body work uses only13 lanes today, and
serial predictor solves use only two. There is also no measured evidence that
the extra rounds/shared-memory cost fits the narrow0.151 ms overhead allowance.
Neither 2x owner scaling nor >=0.55 ms whole saving is promised.

Reuse the original CPU and three native physical selectors, independent FP64
H/bias/current-force/public-velocity references, original8-sweep loaded solve,
held/current factor epochs, reset/model notification/request consumption and
graph/source-bank checks. Add only the necessary existing-fixture odd-world
and mixed-validity controls (e.g. five worlds: cached/invalid subgroups and an
absent paired tail); do not weaken the physical oracle. Verify all13 outputs,
free/prescribed services and both state banks. Confirm unchanged factor source.

After source/CPU/AOT review and freeze, root owns paired native runs, then one
original whole A/B against qualified1a9 at16K/200warm/40wall/40physics, followed
by source-matched nodes only if needed for the decision. Success requires an
actual >=0.55 ms RTX whole-physics saving with all original guards and no
unexplained GB regression. Host-cost repair is still needed before any final
promotion. If the combined mapping misses, close it; no tile/register grid.

Current reviewed module SHA256:
`e776ea36c9bc03fbfdebce4e7ea917efd6e4b352ef1a5653b57f05adc9a78f65`.
Retained factor SHA256:
`eda0a22bb7328cef4f691970a8551fb5bc98d209c51977dbca60123ee91c9b8e`.
Existing physical test SHA256:
`25931d1c3b1c434bfd6bd8315b62a517b50c17b8f99ea05f443b3bbec6d06c4d`.

## Funding record

Root approved this single correction at 02:18 UTC on 2026-09-15, with a
03:45 UTC checkpoint. Implementation is isolated on branch
`ooctipus/fpgs-franka-kinetic-paired16-20260915` from `edd49d0b`.
The original pre-code card remains unchanged at its external path.

## Implementation and CPU/offline checkpoint

All three owners now map two independent CUDA worlds to 16-lane subgroups of
one 32-thread CTA. Each subgroup has its own storage slice, masked scans,
barriers and finite-value votes. The incomplete final pair returns before any
world array access. The original serial CPU owner and numerical operations are
unchanged. Factor source, solver hooks, iteration budget and host notification
proof are unchanged.

The missing paired-owner key regression failed before implementation. The
seven owner CPU controls and three retained-factor controls now pass (10 tests,
1.665 seconds). Existing saved/current/held and physical tolerances are retained.
The actual native lifecycle fixture now has five worlds, mixed repair validity
in both pair orders, an absent upper tail, graph empty/regrow, and an upper
subgroup failure-vote isolation check. These CUDA controls have not yet run.

Offline compilation passed all eight SM120/SM100 entries without GPU use:

| Owner | Registers SM120/SM100 | Shared bytes/CTA | Stack bytes | Spills / CTA barriers |
| --- | --- | --- | --- | --- |
| Repair | 64 / 64 | 7296 | 72 | 0 / 0 |
| Finish | 80 / 80 | 7296 | 72 | 0 / 0 |
| Predictor | 42 / 40 | 1088 | 0 | 0 / 0 |
| Retained factor9 | 40 / 40 | 1808 | 0 | 0 / 0 |

The report is `/tmp/fpgs-franka-kinetic-paired16-offline-M62eLv/offline01/report.json`,
SHA256 `a860cc987b0ca1daf0b5920f98655e7d2165ea4d853d18a9fb04bfb5ed4250b1`.
Runtime SHA256 at this checkpoint is
`544266cd56498da2525d17503e17edf6bd01a9dbf438553ca4d6676d25b88ce9`.
These are compilation/resource facts, not occupancy or throughput evidence.
Paired native physical checks and the original whole A/B remain root-owned and
required before any performance or qualification claim.

## Completed native physical qualification

Frozen runtime/test commit: `91f505f5dd6d0479a1d1845036f0598a72139703`.
Root ran the unchanged three-selector CUDA class on each original UUID. All
three passed on both cards, with no skip, CUDA error or process cleanup error:
RTX session31448 exited0 in6.798 seconds; GB session19083 exited0 in7.040 seconds.
The only warning was the inherited target-coordinate-layout deprecation.

The five-world test covered both half-warp validity orders, the absent paired
tail, isolated upper-half failure votes, actual eight-sweep loaded dispatch,
masked/global refresh request consumption, reset, notifications and graph
empty/regrow. The two saved selectors retained all four actual current/held
payloads, current external/control/passive forces, nontrivial anchors and the
independent FP64 matrix/bias/public-state and held momentum oracles.

Observed normalized maxima were identical across the two actual cards:

| Gate | Maximum error |
| --- | --- |
| Current H9 | 1.339639e-6 |
| Positive current bias | 2.084246e-6 |
| Held momentum9 | 1.730141e-7 |
| Held/current predictor velocity | 8.721723e-8 |
| Public COM/angular velocity | 7.119308e-7 |
| Actual refreshed factor9 | 4.158073e-7 |
| Five-world matched8 body velocity | 1.323715e-4 |

The last value is the inherited secondary7e-4 trajectory gate, not a replacement
for the unchanged3e-6 physical gates. No tolerance or iteration change was made.
The complete original root tool outputs, including commands/source pin/exit,
were persisted (not rerun) at `/tmp/fpgs-franka-paired16-native-fPqY6YLm/gpu{0,1}.log`.
SHA256: RTX `44c3abe2aa62ad05d07b57676ab69242d29187c7dd72dab7f1277db84998f088`;
GB `0d595b45a9d0803a5fb4f399b9221cf3cd71c28ec42cedba8e3b34e4430e3206`.

## Completed whole result: below the predeclared milestone

Root whole parent90977 exited0. One matched round used accepted1a9 baseline,
paired16 candidate91f505f5, fixed Lab53ee,16K worlds,200 warmup,40 unprofiled wall
steps and40 physics-profile steps, seed0. Both arms retained local row packets,
simple-world zero, group16, masked rows and collision thread4. Only the Franka
kinetic owner flag differed. Raw32768/broad7680/dense192/MF64/prop192 matched;
the explicit full broad input212992 was retained. Actual model checks retained
13 bodies/21 DOFs per world, two solver substeps at1/240, env decimation4 and
matrix-free8 sweeps. All four returns were0 with no cleanup signal; both untimed
row/collision checks, source pins, budget equality and final idle guards passed.

| Card | Accepted physics ms | Paired16 physics ms | Saved ms | Ratio | Wall ms before → after |
| --- | --- | --- | --- | --- | --- |
| RTX | 5.453217175 | 5.032844825 | 0.420372350 | 1.083526x | 28.026434 → 44.762043 |
| GB | 5.060163875 | 4.729253575 | 0.330910300 | 1.069971x | 28.648983 → 44.353130 |

RTX misses the pre-code0.55 ms whole milestone by0.129627650 ms. This is one
timing round, not repeated qualification. The host regression remains material:
RTX event.apply9.0205→25.6469 ms and reset_idx12.4862→29.2126 ms (nested timers,
not additive). A separately authorized host repair cannot retroactively turn
this GPU milestone into a pass.

Artifact: `/tmp/fpgs-franka-kinetic-paired16-whole-paired16k-20260915-01/manifest.json`,
SHA256 `3916f890d9af5b2420923c2e2f6b36a1d896790dc1d1439442d9ce64ec3580c2`.
The original generic checked observer does not serialize new owner keys;
actual owner presence/absence is established by the native lifecycle selector
and the complete source-matched node capture below, not a fabricated observer
field. Capacity flags alone do not prove physical equivalence.

## Strict matched-node closure

Root node parent97441 exited0. The matched current captures use the same source,
flags,16K/200warm/40wall, but three profiled env steps. All four metadata, source,
capacity, budget and process/idle guards passed. Original ce09 reader and e76
interval-union implementation were reused unchanged, with an external explicit
Franka owner-name binding (including only the new `_p16` aliases). Each capture
has12 physics roots, zero auxiliary roots and zero unproven graph nodes. All
kernel, memset and memcpy nodes remain charged:1860→1584 nodes per three steps,
400→324 kernels and220→204 memory nodes per env step. Each of the three new
owners executes eight times per env step (24 observed calls, not24 independent
timing repeats).

The earlier edd49 captures were re-read with the same binding/reader. Owner
times below are sums per env step across those diagnostic captures; they are
not independently stackable whole gains:

| Owner | RTX prior edd49 ms | RTX paired16 ms | GB prior edd49 ms | GB paired16 ms |
| --- | --- | --- | --- | --- |
| Repair | 0.216000333 | 0.184064000 | 0.191637000 | 0.126666667 |
| Current force/held-L predictor | 0.538603333 | 0.360640000 | 0.534379000 | 0.322304000 |
| Finish/publication | 1.047489000 | 0.918347333 | 1.177770667 | 0.756981333 |
| Three-owner sum | 1.802093000 | 1.463051333 | 1.903786667 | 1.205952000 |

The RTX combined reduction is0.339041667 ms, not the planned0.75 ms. The
1.463051333 ms replacement exceeds the1.052093 ms target by0.410958333 ms.
Finish achieves only about1.14x, predictor1.49x and repair1.17x. The complete
repair remains first in each env step:~103–105 us RTX/~62–68 us GB, followed by
seven~9–15/~7–11 us calls. There is no extra complete repair or duplicate old
publication/force/composite producer in the candidate. Held factor6 and factor9
remain eight calls each: together0.160033 ms RTX/0.157482 ms GB, stable against
the earlier kinetic owner's0.162006/0.157641 ms. Free-body full inertia,
prescribed motion and all retained MF services remain charged.

The measured current complete state/factor/drive/mask family is:

| Metric | RTX accepted → paired16 ms | GB accepted → paired16 ms |
| --- | --- | --- |
| Sum | 2.531264 → 1.713719 | 2.078101 → 1.443935 |
| Interval union | 2.127563 → 1.680322 | 1.692186 → 1.413194 |
| Exclusive busy | 2.049643 → 1.675650 | 1.631633 → 1.405087 |
| Internal overlap (sum minus union) | 0.403701 → 0.033397 | 0.385915 → 0.030741 |

The exposed state-family savings are therefore0.373993 ms RTX/0.226545 ms GB.
Packing did not recover the original tau/mass overlap and receives no such
credit. Prior edd49 exclusive state costs were2.019715/2.105043 ms; the mapping
improved those costs, but that is not an accepted-baseline whole comparison.

The complete disjoint RTX node-span accounting (candidate minus matched
baseline, ms) is state−0.373993, solve+0.117046, rows+0.077461,
MF services+0.007637, collision+0.015947, sensors−0.001909,
allocation−0.000096, memory−0.005675, cross-family overlap counted once−0.073579
and graph work gaps−0.026752. These sum to−0.263913 ms, exactly the diagnostic
span5.542476→5.278563. GB diagnostic span is5.151680→4.914570. These three-step
node spans explain ownership; the40-step whole result remains timing authority.

Retained solve/response changes must not be misidentified as new-owner cost.
RTX MF packet40 grows0.525568→0.642283 ms, while its general fallback grows
0.021728→0.109973 ms (the sums overlap; joined solve exclusive grows only
0.117046 ms). General response grows0.036843→0.084181 ms and general prefix
materialization0.027211→0.042507 ms. Current routing in
`franka_row_packets.py:337` selects from current dense/MF rows and body ownership;
the same source is retained. The candidate fallback has ten~27–30 us calls and
fourteen~2–4 us calls, versus all~2–3 us for baseline. Different physical
trajectories/active fallback work are consistent with these measurements; the
boundary counts do not prove a per-call routing distribution, so no stronger
cause or matching-work timing claim is made. All such costs remain in the whole
result. GB joined solve exclusive grows0.593658→0.674935 ms.

### Resource premise: static limitation, not an occupancy measurement

Actual node resources confirm8192 CTAs×32 threads for each paired owner,
versus16384×32 previously. State storage is3712→7296 bytes/CTA; predictor
608→1088. RTX repair/finish/predictor registers are64/80/42; GB64/80/40, with
zero recorded local-memory bytes. All full13-body outputs and H entries remain.

Root separately queried driver static limits (cuDeviceGetAttribute IDs
10/39/81/82/106): RTX has warp32,1536 threads/SM,102400 shared bytes/SM,
65536 registers/SM and24 CTAs/SM; GB has warp32,2048 threads/SM,233472 shared
bytes/SM,65536 registers/SM and32 CTAs/SM. Ignoring allocation granularity,
the RTX state shared-memory-only limit falls from floor(102400/3712)=27 to
floor(102400/7296)=14 CTAs. Combined with the24-CTA limit, the corresponding
nominal worlds-in-resident-CTAs upper limit changes24→28, not24→48.
GB's shared-only limit62→32 does not impose the same loss relative to its
32-CTA limit; registers and allocation granularity still apply. This weakens
the ideal2x world-packing premise and is consistent with differing finish
scaling. It is not achieved occupancy, a throughput bound, or a measured stall
cause. No third mapping or resource sweep is authorized by this diagnosis.

### Reproducible evidence pins and closure

Current node manifest:
`/tmp/fpgs-franka-kinetic-paired16-nodes-paired16k-20260915-01/manifest.json`,
SHA256 `1e4a3f91ec98beee4213ffd5d0bc172beb5eda814aa3e75a30e6419390dbea74`.
Prior node manifest:
`/tmp/fpgs-franka-kinetic-state-nodes-paired16k-20260915-01/manifest.json`,
SHA256 `9b0bcbb2c3873adf6ff10c829d8da4ca4df302bfa3bceeb96ff15ec319242b62`.

Read-only adapter: `/tmp/fpgs-franka-kinetic-paired16-node-audit-4UYrYfWa/audit.py`,
SHA256 `b59bf44e8e29cbd26bbb83f82cdac8c5adbccfb2ebc9d46e2203902fd227e940`.
It invokes original `/tmp/fpgs-kuka-live-node-audit-VoxZZR5Z/audit.py` main,
SHA256 `ce09a4964d8ffc1b529eec077f93b0a755dae7fc351768582166b756b8823bbe`,
which pins original e76 interval reader. CLI is unchanged: `audit.py CAPTURE_DIR
--output FRESH_JSON`. The adapter directory contains `current_baseline_gpu{0,1}.json`,
`current_candidate_gpu{0,1}.json`, and corresponding `prior_*` files; each original
reader output pins its SQLite/metadata/check inputs and includes all owners,
resources, call durations and disjoint interval accounting. Current JSON SHA256s
in baseline0/baseline1/candidate0/candidate1 order are:

- `f78a09581a43c86d3c8760ad91e3ce4bfae24af1ed4a226d9d07e1ca5ac89e3a`
- `31379e1ec18a794387b7dfda41f85b947a4c3d1d12ab73ce3c9374fd254c158c`
- `2eac360daa108c1fc4088a1b4df75722832527cd31a50b232426fa864f065c8b`
- `3625b614aefd6e39a64b3b1322bc6582ed06d57a9b90bed3c58584156ce04dac`

**Close this mapping correction below its predeclared milestone.** Physical
controls passed and the GPU result is positive, but neither the RTX owner
target nor the0.55 ms whole target passed, and wall time regressed. No promotion,
no silently reduced milestone and no further mapping grid. Any separately
funded host-only correction must preserve this record and be measured on its
own complete boundary. Runtime and tests remain exactly91f505f5; this closure
changes documentation only.
