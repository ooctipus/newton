# Franka paired16 workspace lifetime — 2026-09-17

## Status and complete hypothesis

Default-off, unpromoted experiment from `ca0d427af809571bb5501f644c1a6e03990cd2a8`
on `ooctipus/fpgs-franka-compact-workspace-20260917`. Four native controls pass
on each GPU. Two initial clean RTX whole comparisons save0.311205/0.185416 ms;
a later three-round AB/BA/AB comparison confirms a0.257110-ms median saving
(1.053551x), below the predeclared0.5-ms milestone. The positive structural
result is preserved without a tuning grid or promotion. GB whole is flat:
4.690438→4.688745 ms (1.000361x). Completed strict paired node traces attribute
most of the RTX reduction to publication/repair; neither achieved occupancy
nor a stall mechanism was measured. Fresh symmetric backend comparisons give
2.229781x RTX /2.236029x GB versus corrected MJWarp, not a4x or full-convergence
claim. Unprofiled environment-wall time is essentially unchanged.

`FEATHER_PGS_FRANKA_COMPACT_WORKSPACE=1` changes only the workspace helpers of
the existing admitted `FrankaKineticState` repair/finish owners. The paired16
mapping, five-argument ABI, four-round scan arithmetic, reverse subtree order,
factor/current/held cadence, predictor, all public state, free-body finalization,
forces, reset/notification guards and eight-pass constraint solver are unchanged.
The original factory is selected when the flag is off.

This is not another world-lane state owner. The previous world-lane and streaming
experiments serialized one world per thread; streaming also added 22 publication
joins and large lane-private shared fields. The accepted paired16 report
explicitly retained its original 896-float offsets and excluded scratch relayout.
Here two worlds still occupy one warp, and no new join, global intermediate,
copy or producer is added. No arithmetic, public write or shared access is
claimed removed: the experiment reduces the live shared allocation.

The current strict Franka RTX reference has approximately 0.916 ms publication
and 0.157 ms exclusive repair per environment step. Predictor/factors and the
remaining state family cannot be credited. A theoretical shared-resource
residency improvement from 13 to 24 one-warp blocks, if it translated linearly
to those owners, would save about 0.49 ms. That is a falsifiable optimistic
model near the 0.5-ms whole target, not achieved occupancy or a throughput
prediction. Bank addressing and the complete unchanged physical work remain
charged; this is one layout, not a tuning grid.

## Exact lifetime layout

Each world uses 376 floats, versus 896 previously:

| Region | Floats | Lifetime |
| --- | ---: | --- |
| Body slots `19*b`, bodies0..15 | 304 | Pose7/velocity6/acceleration6, then each primary body's wrench6/moments13 |
| Padded body slots13..15, offset247 | 54 of their57 | Inertia-axis action after scans/publication, disjoint from all13 physical bodies |
| Axes, offset304 | 54 | Through bias/mass projection |
| Origins, offset358 | 9 | Through each body's publication |
| Shifts, offset367 | 9 | Through each body's publication |

All cross-body pose reads finish before the body-term loop. That loop first
loads each body's pose/motion into values; `_primary_terms` consumes them and
only then overwrites its own slot. Free bodies11/12 finalize from their own
untouched slots before the existing half-warp fence. No pose/motion consumer
follows `_collect`; its final finite checks read canonical body arrays.

The old action scratch at0:54 must not survive this overlay: another lane may
still read subtree terms. Actions instead use the padded slots247:301. The
existing post-body, post-reduction and post-action half-warp fences suffice;
none is added or removed. CPU scans/padding explicitly become16 while the
immutable plan retains its32-wide indexing. Omitted padded bodies16..31 have
no physical IDs or parents. Arithmetic for every physical body is unchanged.

## Initial verification

The new-module regression failed with `ModuleNotFoundError` before runtime
implementation. The focused CPU suite then passed11 tests; its CUDA class was
skipped. It reuses all four source/payload-guarded saved current/held physical
epochs, independently assembled mass/bias, live-force action, original joint
integration, changed free-root anchors, notification/reset proof controls and
actual flag/factory identity. No physical tolerance changed.

The inherited native selectors remain the saved current/held services, original
publication/graph repair and five-world loaded/reset/masked-refresh/graph
controls, including odd paired tails and independent subgroup validity/votes.

First CUDA-hidden AOT compiled original and compact repair/finish on SM120/103:

| Owner | Original/compact shared bytes | Original/compact registers | Stack | Spills / CTA barriers |
| --- | ---: | ---: | ---: | ---: |
| Repair | 7296 /3136 | 64 /64 | 72 B unchanged | 0 /0 |
| Finish | 7296 /3136 | 80 /80 | 72 B unchanged | 0 /0 |

The intended shared reduction is realized without register/spill growth.
Resource facts do not establish achieved residency, stalls or a whole gain.
Final-pin AOT artifacts:
`/tmp/fpgs-franka-compact-workspace-offline-WfNsXEvw/offline02/report.json`,
SHA256 `061f37a346a5f99a745cf1ffb89b9aae435575fdc1d0707c814abab13182303b`.
The compiler minimally reuses the existing hidden AOT helper, not a new
benchmark framework; the initial identical resource result remains in
`offline01`.

## Native physical/lifecycle scope

Root ran the four inherited native controls on each card:4/4 pass, without
tolerance changes. GB ran in17.977 seconds; RTX in18.351 seconds. RTX native
execution intentionally co-ran with an external GPU job and is **correctness
evidence only**, not a timing sample. Native test duration is not a performance
comparison on either card.

Both logs report maximum normalized matched-eight body/joint velocity
differences1.323715e-4/1.197640e-4, independent current-H error1.339639e-6,
held primary momentum1.730141e-7, and current bias2.084246e-6. The controls cover
current/held factors and live forces, original integration/free-root anchors,
canonical public state, model notifications, five-world loaded solves, masked
repair/refresh, independent half-warps, odd tail and graph replay. This is the
existing bounded physical scope, not full-task convergence qualification.

- GB log: `/tmp/fpgs-franka-compact-workspace-native-PcgkXN/gb.log`, SHA256
  `78d8224180c5f258a4305c57af6ec3bb944602f5416c3eee7880c27c0caa78d0`.
- RTX co-run log:
  `/tmp/fpgs-franka-compact-workspace-native-PcgkXN/rtx-shared-correctness.log`,
  SHA256 `3dd2be32f0d4a0c18f559c9e686ca9eaf0c89e6e466e075360e370cc511db358`.

## Whole comparison and preserved incomplete attempts

The original fixed-import variant harness and checked boundary observer are
reused. Both arms retain16,384 worlds, seed0, warmup200, wall40/profile40,
graph execution, raw contacts32768, broadphase7680, original timestep/substeps
and maximum8 GS passes. Both explicitly enable the retained kinetic owner,
paired16 and masked rows; world-lane/streaming flags remain0. Only the new
compact-workspace flag differs0/1. Actual repair/finish factory identity,
unchanged predictor/cache/factor shapes, source/idle and capacity guards pass
for all three initial completed comparisons.

| RTX ms/environment step | Run01 original | Run01 compact | Retry03 original | Retry03 compact |
| --- | ---: | ---: | ---: | ---: |
| Whole physics |5.108246725|4.797041475|5.049426675|4.864011150|
| Environment wall |28.256386874|27.897828998|29.001154425|27.916883773|
| Physics baseline/candidate |1.064874413x| |1.038119881x| |

The first0.311205-ms saving did not repeat at that magnitude: retry03 saves
0.185416 ms. These are two clean single-round discoveries, not a balanced
multi-round performance promotion. Wall stepping is not full RL throughput.
These initial runs alone establish no phase attribution. The later strict
node attribution is reported separately below; no achieved occupancy or
memory-stall cause is inferred.

GB retry02 is flat: whole physics4.690438375→4.688745475 ms, a0.001692900-ms
difference (1.000361056x). Wall stepping worsens27.596603299→28.507825048 ms.
Both processes exit0; both boundary checks, actual flag0/1 owner identities,
capacity checks and final source/idle guards pass. Thus the RTX benefit is not
established as a two-card gain. Contact populations also differ between arms
(GB before/after1739/1802 versus1680/1787); no identical-workload or isolated
phase-cost claim is made from these whole-task runs.

Completed manifests:

- `/tmp/fpgs-franka-compact-workspace-whole-rtx16k-20260917-01/manifest.json`,
  SHA256 `e73489f4cbaa303e96b772ff6db896bcc1e089e23e26660661cb93c175b95cfe`.
- `/tmp/fpgs-franka-compact-workspace-whole-rtx16k-20260917-03/manifest.json`,
  SHA256 `0f191c0999b1c6312720e0d4c620f5368ff06435dbaf16303caffa04e18a20f9`.
- `/tmp/fpgs-franka-compact-workspace-whole-gb16k-20260917-02/manifest.json`,
  SHA256 `b9176ef4747733753726756e5ae14afded0cce4d3a882c8ee4fe80235819d327`.

Preserve three unrelated incomplete attempts without imputing candidate failure:

- GB01 completed only the baseline before an external process4192934 acquired
  the selected GPU. Parent failed its idle guard before launching candidate;
  final source guard passed. This attempt supplies no valid GB comparison.
  `/tmp/fpgs-franka-compact-workspace-whole-gb16k-20260917-01/manifest.json`,
  SHA256 `54aa1064eaaf39ac297b00bf124f7fba9a600c911884d5d9c27431fa9efb58d9`.
- RTX02 baseline exited139 during startup before capture reports; candidate
  never launched. Final source/idle guards passed. No runtime fix was made.
  `/tmp/fpgs-franka-compact-workspace-whole-rtx16k-20260917-02/manifest.json`,
  SHA256 `7eadc3b048dc7b75783fbf7db665f1476b41d15af34964aa6a534561127ce019`.
- RTX node01 completed only baseline. External process56285 appeared between
  arms; the parent idle guard stopped candidate launch. Final source guard
  passed and final idle guard failed. There is no valid paired node attribution.
  `/tmp/fpgs-franka-compact-workspace-node-rtx16k-20260917-01/manifest.json`,
  SHA256 `bada0d58386aa29ca1e870e203d33c402aa9907935ae62d6c575b7724589ea7b`.

## Fresh symmetric backend comparison

The unchanged compact tree `ee2a50d525e64b406e9411b65b0e945d556b7442`
was compared with corrected MJWarp using the same Newton tree on both arms,
fixed Lab `53ee6b44`, shared collision mapping,16,384 worlds and the original
200 warmup /40 wall /40 profiler steps. Calibrated capacities remain FPGS
raw32768/broad7680 and MJ njmax56/nconmax1/raw2048/broad7680. No timestep,
substep, solver-iteration, MJ line-search iteration or gradient-tolerance
budget changed. The MJ correction remains the pinned existing helper, not a
package edit or suppressed warning.

| Card | Compact FPGS physics graph, ms | MJWarp physics graph, ms | MJ/FPGS | Unprofiled environment wall FPGS/MJ, ms |
| --- | ---: | ---: | ---: | ---: |
| RTX |4.762999600|10.620445450|2.229780882x|28.344955476 /28.620338949|
| GB |4.643132000|10.382175850|2.236028579x|28.874020951 /29.187371823|

The physics numbers are whole-graph profiler-window measurements; the
environment-wall loop runs separately outside the profiler. Neither is full
RL training throughput. These fresh backend ratios are not an attribution of
the compact change versus the original FPGS owner, nor proof of identical
contact manifolds or task convergence.

Independent replay passed the original and extended capacity/warning checks
at both boundaries of all four processes, every24 driver/helper pin, complete
source snapshots and exact ratio recomputation. Both backends observed the
same actual collision mapping: constructor32768 threads, live131072 after
the fixed Lab THREADS_X=4 multiplication, block128, broad capacity7680,
split/sparse GJK disabled. FPGS observed the actual c376 repair/finish factories
and retained predictor; MJ had no compact owner. Original process/idle gates
are preserved; the inherited fair manifest does not add a new final-idle field.

The first GB fair attempt is preserved as an observer failure before timing,
not a physics failure: the added guard incorrectly expected constructor32768
at the post-warmup live boundary. Exact Lab source and a CPU reproduction
established the required multiplication; the repair requires exactly131072
and records actual values before validation. Newton/runtime bytes did not
change and the gate was not broadened to a range.

Artifacts and manifest SHA256:

- `/tmp/fpgs-franka-compact-fair-rtx16k-20260917-01/manifest.json`:
  `c88a4fc85ecaa813cf8bc06da9610ec80a798d6929b6a8e55daffb9dea51cde2`.
- `/tmp/fpgs-franka-compact-fair-gb16k-20260917-02/manifest.json`:
  `b9d81fe4c868021467821f91d3c71944383cdec5c992e72e7cea294caead46cb`.
- Preserved GB01 observer failure:
  `/tmp/fpgs-franka-compact-fair-gb16k-20260917-01/manifest.json`,
  `6c7bb0fcafb25386ec39e72deac458f316836b8fc3dbf7aa141a270132b1409d`.
- Minimal source-pinned external adapter and exact recipe:
  `/tmp/fpgs-franka-compact-fair-gnAUbbKZ/LAUNCH.md`,
  `c58d5001396005fa7fce4e42d4a9f84fb0804e18b2f8d505ed121027d61bd77d`.

## Strict paired node attribution

Completed RTX retry02 and GB01 each retain12 traced environment steps,
48 physics graph roots,0 auxiliary roots,0 unproven nodes, process/correlation,
root membership, source/idle, sticky capacity and actual owner guards. The
unchanged strict reader was independently replayed in
`/tmp/fpgs-franka-compact-replay-wJkvLsdZ`; all four outputs are byte-identical
to `/tmp/fpgs-franka-compact-strict-haL57HiU`.

| Exclusive busy ms/env step | RTX original | RTX compact | GB original | GB compact |
| --- | ---: | ---: | ---: | ---: |
| Publication |0.917649583|0.679012083|0.751590833|0.722463833|
| Repair |0.156829583|0.125437750|0.097842667|0.099872000|
| Joined solve |0.558873083|0.557497083|0.605970250|0.655087167|
| Collision |1.122970000|1.102260833|1.015855333|0.975988500|

The RTX state-family interval union is1.683285583→1.414616250 ms. Publication
and repair explain most of the observed RTX profile reduction; this is an
owner attribution, not a measured occupancy or stall diagnosis. Finish retains
80 registers and repair64, while static shared memory drops7296→3136 B for
both, with block32/grid8192 and8 calls each per environment step. Predictor,
factor shapes and public buffers remain unchanged.

Overlapping solve launches must not be summed: RTX original solve sum1.342757417
ms has union0.569830417 and exclusive0.558873083 ms. The table consistently
uses exclusive intervals; state-family union is separately labeled, not added
again. On GB, small publication savings coexist with increased solve time and
reduced collision time. The arms have differing endpoint contact populations,
so this does not establish an identical-workload isolated speedup.

The12-step node-trace graph spans are RTX5.173683833→4.879476583 ms and
GB4.889160000→4.862930417 ms. They are not substituted for40-step whole-graph
timing or the separate unprofiled wall measurements; the GB span difference
does not override the earlier essentially flat whole comparison.

Node manifests:

- RTX `/tmp/fpgs-franka-compact-workspace-node-rtx16k-20260917-02/manifest.json`:
  `24d88b1005950f2753b7206dd38f3abc639b885732a9bc0b88b34f20b2d46863`.
- GB `/tmp/fpgs-franka-compact-workspace-node-gb16k-20260917-01/manifest.json`:
  `e984944bca6e26774032571306c27d938166a4048d61faebe46a42770e22bff0`.

Strict reader SHA256:
`1d17ad0be8827ab8f703bd1e50d1c95cd5a3de488a0134543e468a39f76f10e4`.
Output SHA256, original then compact:

- RTX: `0e3468020feed8aac42a574a6544cb4764b717c745e61d56002888648c93ab62`,
  `9aaae1d29e4870703bdc1f8f55a13f7ea8dc92e0e83206c9f40c416fd1774981`.
- GB: `281c9b797027ccd4d6df4028df1934e29f5a7d3d039ca09ed4c8b56c1684338d`,
  `4139d8def6421c822c54f72c33b8b4f595d747c387421291d77126fc8fda03b9`.

## Balanced three-round outcome

The later RTX alternating AB/BA/AB run completed all six processes with exit0,
unchanged source/actual capacities/budgets, both checked boundaries per process
and final source/idle guards passing. Independent replay checked every recorded
artifact hash, re-read all six results/budget fingerprints and reproduced the
summary exactly. No runtime or observer bytes changed.

| RTX physics graph, ms/env step | Round1 AB | Round2 BA | Round3 AB | Median |
| --- | ---: | ---: | ---: | ---: |
| Original |5.101858825|5.049989300|5.058346075|5.058346075|
| Compact |4.830341350|4.783661125|4.801236475|4.801236475|

The ratio of medians is1.053550705x, saving0.257109600 ms. Unprofiled
environment-wall medians are28.674924126→28.602801674 ms (1.002521517x),
essentially flat. Thus repeated RTX evidence supports a modest physics-owner
benefit, not the0.5-ms milestone, a two-card win,4x acceleration or RL throughput
improvement. The default-off feature remains unpromoted and GB whole remains
flat. No further layout/mapping tuning is implied.

`/tmp/fpgs-franka-compact-workspace-repeat3-rtx16k-20260917-01/manifest.json`,
SHA256 `5fd660e70c152f88c01289e28dafa372e1860b807ee388c6fbbeb676fa2d7765`.

## Frozen runtime and observer pins

| File | SHA256 |
| --- | --- |
| `franka_compact_workspace.py` | `ad6e62f07e5e7f9af65b8468ab34374183f345afe395dd922899dc3aba763716` |
| `franka_kinetic_state.py` | `66b8b5e7d9d4784d3da7104d204627a2e7965b8468aed6e93eb1dc618cb7daf2` |
| `test_franka_compact_workspace.py` | `0a797cc966f17850b0f97000f7e91163e61b4b2bab0cd074c9c699e388473c28` |
| `compact_workspace_capture_20260917/run.py` | `29c421987a0cc086f1b5eafcf34c142c5b8b249c38362b213c47e01ba55556c5` |
| `compact_workspace_capture_20260917/checked_compact_workspace.py` | `2212ca222b16d0d29743b862541358275e8cc94bc460478bb3eb90f657f04623` |
| `compact_workspace_capture_20260917/nsys_checked.sh` | `152afd356e4d38c79b486017effd0305fe737be4579bee23fe268a33391d8ba7` |

Runtime/observer bytes remain the measured pins. Initial full and owned
targeted precommit passed. The nine-file checkpoint was committed and pushed
as `8ec4db342c0c5596060e89ef4923412a4a0ef1c1` on the isolated ooctipus branch;
report-only checkpoint `ee2a50d525e64b406e9411b65b0e945d556b7442` added GB02
and the incomplete node attempt. The present report-only update records the
later strict node, fair-backend and three-round evidence. No dependency pointer
or accepted handoff is changed. The external strict node reader reuses the
original complete Franka interval/correlation accounting and adds only exact
compact aliases/guards.
