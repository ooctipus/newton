# Pair-private terrain query and reduction

Candidate funded 2026-09-15 13:46 UTC; first checkpoint 15:16 UTC. This is
a CLOSED, unpromoted default-off Newton collision experiment based on retained
`ca0d427af809571bb5501f644c1a6e03990cd2a8`. Isaac Lab, task parameters,
substeps, iterations and calibrated capacities remain unchanged. No parent
dependency pointer change or accepted speedup is implied.

## Complete ownership hypothesis

Own a current heightfield/convex pair through conservative current culling,
local triangle compaction, the existing finite query and required generic
query, contact selection and the original public writer. Retain callback
geometry locally rather than reconstructing or requerying selected contacts.
Hoist invariant pair setup where the existing mathematics permits it.
Retire the global callback buffer, hash/probe/active append, global clear,
reduction and export traversal for admitted scenes. Do not claim hash savings
inside the query: the current query only buffers contacts; hashing is later.

The fast pair has an initially bounded local triangle/record store. Overflow
must route the entire pair before publication to a separate pair-private
streaming reducer, not the original global reducer. A single overflow must
not restore unconditional full-frame clear/reduce/export. Retain winner
geometry in the exceptional reducer too. Unsupported scene-level features
use the complete existing pipeline through constructor admission.

Initial local storage estimate: 64 triangle IDs (256 B), 64 callback records
(1792 B) and 245 uint64 score slots (1960 B), or 4008 B/pair before headers
and setup. Separate exceptional reducer storage is approximately 8680 B/CTA.
These are implementation estimates, not measured native resource counts.

Implementation mapping selected before coding: two independent 16-lane pairs
per 32-thread CUDA block, with isolated scratch and subgroup synchronization.
A lone 16-thread block would still consume a hardware warp and halve the
assumed useful lane fraction. CPU execution uses its existing single lane.
Admission is bound only after a one-time canonical pair-uniqueness check;
arbitrary duplicate explicit pairs retain the original global reducer.
Existing large backing allocations stay reserved but untouched initially:
the experiment removes their producer/consumer work, not their allocation.

## Budget and falsifiers

The accepted G1 strict node capture has RTX whole physics 15.661389 ms.
Complete terrain ownership is 3.350804 ms: finite query 1.373173, generic
query 0.811199, clear 0.658218, reduction 0.203232 and export 0.304981 ms.
The first 10% whole-time milestone requires complete replacement cost at
most 1.784665 ms, including setup, compaction, selection, publication,
routing and all exceptional work. This is a falsification budget, not a
prediction or a claim that 10% meets the user's 4x goal.

Removing the old clear/reduce/export saves at most 1.166431 ms. Query must
still save 0.399708 ms before added work; 0.20/0.30 ms of new work raises
the query requirement to 0.599708/0.699708 ms (27.5/32.0%). An illustrative
screen is query <=1.45 ms, selection/publication <=0.25 ms and routing/
exceptions <=0.084665 ms. The first whole paired benchmark decides whether
this premise survived actual execution; there is no standalone kernel gate.

Unlike the closed c41f direct-query experiment, this removes the complete
global reducer lifecycle, compacts locally after the accepted culls and
retains winner geometry. Nevertheless, it loses cross-pair query packing:
saved 96-pair welded fixtures imply only about 62% useful 16-lane query
slots, about 1.6x minimum query-entry inflation. The saved raw callback
maxima 42/47 do not bound the live 16K population. Increased registers,
finite/generic lifetime overlap and exceptional work can erase the saving.

## Minimal causal checks and measurement

Use regression-first focused tests, existing saved current geometry and
existing benchmark driver. Preserve oct-normal encode/decode before
selection, world-space support scores, scaled beta, shape-local voxel
competition, fingerprint ties, per-bin roundoff suppression, cross-bin
deduplication and the stock contact writer. Exercise current heights/poses,
empty/regrowing pairs, forced local overflow, generic fallback and captured
replays. Capacity counters must expose original demand; local overflow is
not permission to drop contacts. Compare physical geometry and stability,
not arbitrary bit-identity requirements.

Root owns paired RTX PRO6000/GB300 GPU execution, one job per device.
Start with one integrated 16K retained/candidate discovery using the fixed
Lab and settings, then immediately diagnose a theory/timing mismatch.
Allow one cause-specific correction, not a tile/register tuning grid.
Repeated timing and loaded-contact/cross-task qualification are funded only
after a useful integrated result. Shared collision improvements must also
be enabled for corrected MJWarp before reporting a new backend ratio.

Ownership evidence:
`/tmp/fpgs-g1-present-ports-nodes-paired16k-20260915-01/round_01_baseline/round_01_g1_fpgs_gpu0/strict_present_ports_node_audit.json`.
Prior-art report:
`/home/octi/Projects/newton-fpgs-terrain-direct-20260913/reports/fpgs/G1_TERRAIN_DIRECT_20260913.md`.

## First implementation checkpoint, 14:05 UTC

The complete default-off owner and two narrow integration seams are present.
Two 16-lane CUDA subgroups have separate aligned scratch; the exceptional
kernel keeps one serial winner set and never enters the global reducer.
Logical triangle demand is counted once in fast traversal; logical callback
demand is counted only for a successful fast pair or its complete replay.
Duplicate explicit pairs, deterministic mode, mixed meshes, predictive contact,
hydroelastic and unsupported/custom ownership retain the original pipeline.

Actual implementation storage is 4024 B/pair (8048 B/fast CTA) and 8840 B
for the separate exceptional CTA, before compiler-added storage. This is
declared scratch, not measured native resources. CPU-only scratch uses a
matched allocation/release; CUDA uses shared storage and no allocation.

Important partial hypothesis: this first complete draft still prepares finite
and generic query descriptors per triangle. It removes the global lifecycle
but does not yet implement the card's amortized query setup. Serial lane-zero
culling and the combined generic/finite register lifetime are charged risks,
not hidden behind the eliminated stages. No performance result is available.

Missing-module regression95256 failed before implementation. Four initial
CPU controls2193 pass (13.758 s), with seam welding disabled; this is a
bring-up result, not the current G1 qualification. Current WELD=1 CPU and
native GPU checks remain pending. Bring-up fixed Warp attribute assignments,
native integer casts, unsupported CPU thread-local storage and CPU scratch
release placement. Review also fixed odd-capacity subgroup alignment before
GPU execution. No numerical tolerance, Lab source or task budget changed.

## Native readiness, 14:08 UTC

Final WELD=1 CPU controls15789 pass 4/4 (1.903 s). Both native CUDA
selectors pass without skips: RTX62184 (43.231 s), GB85632 (44.933 s),
including initial module compilation. These are test durations, not timings.
All sessions exited zero and were reaped; both cards are compute-idle.
Tests cover three-pair odd tails, changed decoded heights and poses,
empty/regrowth, graph replay, complete exceptional publication and both
pinned 96-pair scenes. Original scoring envelopes compare support maxima
and normal/voxel depth within 2e-4 while permitting ties and output reorder.
The unchanged inherited joint-target-layout deprecation is the only warning.

Full staged `uvx pre-commit run -a` passes99528. Runtime source SHA256:

- `heightfield_pair_terrain.py`: `4353dc6186df116699b706b121ac04206c537f7a909c6055236f56e14b790311`.
- `narrow_phase.py`: `ddc22b1c00d2d3fb91d1e110908e8d60175b1b494cbeb554014a7beffa56a35a`.
- `sim/collide.py`: `3c6d9d8ad42855cd33d09b944d5437bcbd9afc4d9827778c599ed1c8040ae480`.
- Focused test: `7fe72619647d726228051bb0adfdb27d8bc3c2cfed764e2e24daa9e47542f6b5`.

Use the fixed Lab Python through `uv run --no-project --python` and set
`NEWTON_HEIGHTFIELD_WELD_FLAT_SEAMS=1` before importing the test module.
CPU selector: `tools.fpgs_bench.test_heightfield_pair_terrain.TestHeightfieldPairTerrainCPU`.
CUDA selector: `tools.fpgs_bench.test_heightfield_pair_terrain.TestHeightfieldPairTerrainCUDA`.
Select one GPU UUID per native process; root ran the two cards concurrently.

The original checked sparse benchmark is reused with only the new Boolean
flag and an untimed private owner/status observation added. Timing, capacity,
source/idle and existing G1 kinetic checks are unchanged. Prepared adapter:
`/tmp/fpgs-g1-pair-terrain-checked-lDHRWzAJ/run_live_sparse_checked.py`.
The first complete 16K physics comparison is next; no gain is claimed yet.

## First whole result and diagnosed launch mismatch

Runtime `9b230fdf9ab597293de706780df554177c0dd4db` fails the first screen:

| GPU | Retained physics ms | Pair-private physics ms | Retained/candidate |
| --- | ---: | ---: | ---: |
| RTX PRO6000 | 15.580928 | 40.319060 | 0.386441x |
| GB300 | 20.471399 | 71.914381 | 0.284664x |

Whole environment wall also regresses, 29.826997 -> 53.163606 ms RTX and
34.426962 -> 85.664914 ms GB. Parent28520 exits zero, all source/finite/
capacity and final-idle checks pass. No promotion or new MJWarp ratio.
Artifacts: `/tmp/fpgs-g1-pair-terrain-paired16k-20260915-01`.

The immediate node diagnosis preserves the original analyzer limitation:
baseline parent97775 and candidate-only continuation4186 exit one because
the inherited node analyzer rejects auxiliary graphs. Actual simulations
and boundary checks complete; final source/idle guards pass. The existing
auxiliary-aware strict reader accepts all four captures: 12 physics roots,
3 separate auxiliary roots, 768 baseline/720 candidate physics nodes and
zero unproven nodes. The failed original parent statuses are not relabeled.

Complete affected ownership includes geometric culling and active-count reset,
which the conservative first card had not credited. Baseline union is
3.658525 ms RTX / 8.292071 ms GB; candidate is 28.755518 / 59.631313 ms,
including the new four-byte callback-counter clear. All seven retired kernel
owners have zero candidate calls. New fast owner costs 28.627177/59.466502;
exceptional owner costs only 0.124384/0.161824; clear costs 0.003957/0.002987.
All four untimed candidate boundaries have zero overflow routes, which is
not a claim about every captured step. Timed exception cost nevertheless
bounds its contribution to the loss.

Native fast resources are 254/252 registers, 8176 B shared, grid384/block32.
The exceptional kernel has 255 registers, 8968 B shared, grid384/block32.
The old finite/generic kernels launch grid6144/block32, with registers
139/168 RTX and 133/168 GB; original packed culling uses grid1536/block128.
No achieved occupancy, bandwidth or hardware-stall attribution is claimed.

Source confirms additional costs: serial pair culling, querying with selection
inside callbacks, and visiting all 35 bins/245 slots even on empty pairs.
However, it also reveals a concrete implementation mismatch: new launch
sizing derives from `num_tile_blocks`, ignoring the calibrated expanded
`total_num_threads` domain. It therefore launches 16x fewer query blocks.
This is separate from register pressure and invalidates treating the first
loss as sufficient evidence against the ownership algorithm.

One cause-specific corrective trial is funded before editing: derive fast
pair groups from `total_num_threads / 16` on CUDA (single-lane CPU), preserving
the original calibrated query launch width. Keep every kernel equation,
record capacity, selection rule and other launch unchanged. No tile sweep
or setup-only polishing. Repeat focused tests and one whole paired screen
against retained ca0d; a gain over the losing prototype is not progress.
The >=10% whole saving remains about1.56 ms, not a reset target. With the
complete boundary credited, replacement must be about2.10 ms or less.

Strict full readouts and unchanged-reader reproduction:
`/tmp/fpgs-g1-pair-terrain-strict-Mj8ohc2j` (`REPRODUCE.sh`). Baseline capture:
`/tmp/fpgs-g1-pair-terrain-nodes-paired16k-20260915-01`; candidate continuation:
`/tmp/fpgs-g1-pair-terrain-nodes-candidate-paired16k-20260915-01`.

## Launch-only correction readiness, 14:35 UTC

Fast groups now derive from `total_num_threads / width`, with width16 on
CUDA and width1 on CPU. The live CUDA grid becomes6144/block32; kernel
mathematics, native module source, contact selection and exceptional launch
are unchanged. Corrected module SHA256:
`1fd344b03de537f5534bb25beb05e5766742616f1fa6467320c5d07c058ee753`.
All other runtime/test hashes above remain unchanged.

CPU18633 passes4/4 in1.935 s. Native RTX30778 passes2/2 in3.457 s and
GB67579 passes2/2 in3.629 s, without skips. These are cached-module test
durations, not performance evidence. All sessions exit zero and are reaped.
Full staged pre-commit21098 passes. The next whole paired screen retains
the original ca0d baseline, task settings, capacities and measurement law.

## Corrected whole result and consumer redesign, 14:52 UTC

The launch-only runtime `358e2efdabfde100966635e979c437b504ba7b95` still loses:

| GPU | Retained physics ms | Corrected pair owner ms | Retained/candidate |
| --- | ---: | ---: | ---: |
| RTX PRO6000 | 15.607482 | 23.863201 | 0.654040x |
| GB300 | 20.531565 | 59.539593 | 0.344839x |

Environment wall is28.921012 ->37.613297 ms RTX and34.206417 ->73.200989 ms
GB. Whole parent69745 exits zero with source/finite/capacity/final-idle guards
passing. Artifacts: `/tmp/fpgs-g1-pair-terrain-paired16k-20260915-02`.
This remains unpromoted; recovery relative to the first losing prototype
is not an optimization gain over retained ca0d.

Immediate candidate-only node parent21326 exits one at the same inherited
auxiliary-graph analyzer limitation. Capture/source/idle checks complete.
The unchanged strict reader accepts both captures:12 physics roots/720 nodes,
3 separate auxiliary roots, no unproven nodes. Complete affected union is
11.944897 ms RTX /47.281805 ms GB (GB exclusive47.281389). Fast owner alone
is11.817697/47.115501; exception0.123606/0.162741 and counter clear
0.003595/0.003563. All seven retired owners have zero calls. The actual fast
grid is6144/block32 with unchanged254/252 registers and8176 B shared.
Strict JSONs and frozen-reader reproduction:
`/tmp/fpgs-g1-pair-terrain-strict02-y1pApaGg`.

A single live16K fast-launch hardware-counter attempt per GPU is blocked by
`ERR_NVGPUCTRPERM`; sessions12783/58442 both exit one and are reaped. No
privilege/driver changes were attempted, and no counter evidence exists.
Commands/logs: `/tmp/fpgs-g1-pair-terrain-ncu-2c8P43ww`. Its instrumented
wall timings are NOT performance evidence. Source/pair settings were reused
from the pinned node manifest; this is a diagnostic, not another benchmark.

Static evidence reveals a distinct implementation mismatch. Exact cached
PTX contains eight generic-address `atom.max.u64` score sites. CPU ptxas13.2
assembly of that NVRTC12.9 PTX lowers each on both architectures to shared
64-bit CAS retry/control loops after pointer-space dispatch. The original
ordinary reducer's eight explicit global MAX sites lower to direct
`REDG.E.MAX.64` instructions (its separate hash CAS remains). This proves
different generated work, NOT that CAS dominates measured time. Fast SASS
has29,344 instructions versus old generic26,360 and finite7,224: an
order-of-magnitude code-size expansion is not supported either.

One complete fast-consumer redesign is now funded, not a launch/register
grid: callbacks reserve/store immutable records only. After all query lanes
finish and complete-pair overflow is ruled out, reuse dead triangle-ID words
for cached normal-bin/voxel tags and derive a35-bit active-bin mask. Seven
slot-owner lanes compute original packed-score maxima in registers and
write each winner once, without shared64 score atomics. Perform original
simultaneous ULP suppression in two subgroup comparison waves, then publish
active bins only with the original32-bit cross-bin deduplication. Fast clear
initializes four header words, not245 scores. Exceptional streaming replay,
query mathematics, budgets, capacities and public writer remain unchanged.

Charge all new work:7*B*n tag probes (worst15,680 forB35/n64), record
classification, repeated matching projection loads, subgroup mask reductions,
seven winner stores per active bin and remaining reservation/public atomics.
This removes score CAS, unconditional empty-slot scans and callback scoring
live state; it does not hoist pair setup or restore cross-pair query packing.
Neither the static lowering nor the proposed work count certifies a gain.
Focused existing controls then one whole paired retained/candidate screen
decide it. The complete replacement budget remains about2.10 ms RTX for
the first10% whole milestone; a win against23.86 ms alone does not count.

## Deferred-consumer readiness, 15:05 UTC

Runtime SHA256`3dfd1a018d74928570529e2d17ee33409a1449b1dd1a1e488d0de3a16c0ce970`;
test SHA256`263b976e221b1ab01ae8931eb10633a2d869385d24c227a9b5ea45d731e32774`.
Narrow/collision hooks are unchanged. The added record-callback API check
fails against exact358e source loaded in memory (98423); an earlier first
attempt raced implementation and passed58138, which is not a failed control.
CPU30417 passes4/4 in14.667 s including changed-module compilation; full
pre-commit90941 passes. Native RTX85461 passes2/2 in36.807 s and GB84291
passes2/2 in37.053 s, including compilation. All sessions exit zero and are
reaped. These durations are readiness checks, not performance measurements.

Existing fixtures cover empty/regrowth/current heights and poses, captured
replay, three-pair odd halves, complete forced overflow and both saved96
geometry scenes. The saved scenes include24 multi-normal pairs per card and
5/1 multi-voxel pairs. Highest saved voxel is82: high bits32--34/voxel99 and
an explicit nontransitive ULP winner-bank fixture remain source-reviewed,
not independently exercised. The unfinished helper was not added to delay
the first whole discovery. These gaps must close before promotion if useful.

Review corrected a predicate inversion before final qualification: spatial
selection uses original positive `depth < beta`, including original NaN
behavior. No numerical tolerance or query equation changed.

Fresh exact PTX and CPU-assembled SASS confirm zero shared64 score MAX/CAS
sites in the fast owner. Static resource estimates do NOT improve:254
registers on both architectures,8176 B shared,784 B stack and no assembler
register-spill stores/loads. The code is only slightly smaller (SM120
29,344 ->29,216 instructions). Exceptional normalized SASS instruction
streams are exactly identical on both architectures. These CPU assembler
facts are not achieved occupancy, driver-counter or timing evidence.

## Close the fused pair-owner experiment, 15:18 UTC

Final measured runtime is `a68ad0dc184284b8af9fcf677d339236688a91ba`.
The valid RTX discovery still loses:15.633675 ->24.555561 ms physics,
retained/candidate0.636665x. Environment wall29.317159 ->38.990825 ms,
0.751899x. Parent54026 exits zero; source, both actual finite/capacity
boundaries and selected-device final-idle checks pass. Artifact:
`/tmp/fpgs-g1-pair-terrain-rtx16k-20260915-03`.

The planned paired parent22163 stopped BEFORE launching simulations because
an unrelated user started a GB300 process3040296. No process was interrupted.
The original parser then rejected an explicit single-GPU invocation before
creating output. A21-line source-pinned continuation narrows the already
validated device list toRTX before enumeration, launching and final checks;
all actual selected-device checks and original timing/budget law remain
unchanged. Its manifest explicitly disclaims a simultaneous pair. Adapter:
`/tmp/fpgs-g1-pair-terrain-single-card-Eh9GUsil/run.py`, SHA256
`b627a3e210ba71d2a7bd65100b647ead9f7c5d4ff2040a08bd96c4dc5542d259`.

GB300 became idle and the symmetric GB-only continuation started18434.
Both simulations/source/capacity checks finished, but another unrelated
process3046952 appeared before the final idle check. Parent18434 exits one.
Therefore its20.476986 ->63.151481 ms raw physics values are INVALID comparison
evidence and are not included in a valid GPU speedup table. Preserve artifacts
`/tmp/fpgs-g1-pair-terrain-gb16k-20260915-03`; do not repeatedly compete with
external jobs. GB-only adapter SHA256 is
`5c16d94f7cbd467f560f81ca40925a3ce85ddb1f2749293a7f2a3dec33b03f0f`.
All root GPU/profiler sessions from this experiment are reaped; no external
process was signaled or modified.

Decision: do not promote and do not run another mapping/register/slot sweep.
The original global pipeline really was retired, capacities stayed intact,
and two diagnosed implementation issues were addressed:16x underlaunch and
shared64 winner-update lowering. Neither correction delivered a net gain.
The replacement selection also adds scans; its loss does NOT prove a measured
CAS stall fraction, memory-bound status or impossibility of pair-local
ownership. This particular fused query/consumer mapping remains much slower
than retained packed queries. No new MJWarp ratio, other-task benefit or
accepted optimization is claimed. Retained ca0d and the fixed Lab are unchanged.

Do not fund remaining high-mask/ULP fixture expansion or loaded-contact
qualification for this losing candidate. Preserve the stated coverage gaps.
The next distinct investigation measures complete-world row/solve bypass
eligibility across the full16K retained population, using only an untimed
last-solved-row census; it does not assume the biased historical16 cases are
representative. No new runtime for that proposal is funded here.

Exact local paired launch is preserved in
`reports/fpgs/REPRO_G1_PAIR_TERRAIN_20260915.sh`; invoke it with `bash` and a
fresh output directory. It depends on the source-pinned local observer/driver
paths recorded above. Large captures and original worktrees remain local and
unchanged. This script reproduces the closed candidate, NOT a recommended
production setting.
