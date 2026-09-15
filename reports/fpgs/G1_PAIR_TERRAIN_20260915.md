# Pair-private terrain query and reduction

Candidate funded 2026-09-15 13:46 UTC; first checkpoint 15:16 UTC. This is
an unmeasured, default-off Newton collision experiment based on retained
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
