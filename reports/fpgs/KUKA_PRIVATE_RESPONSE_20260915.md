# Kuka MF0 private response and early solve fork

## Pre-code contract

Experimental, default off; no performance result yet. Parent authorized one
complete integrated experiment on 2026-09-15, with a 90-minute source/AOT/cost
readiness checkpoint near 02:40 UTC. No Isaac Lab or task-budget changes.

This worktree starts at current-contact
`c1c678a8738b306a32ed6c241b7cb004c93804aa`, preserving that original tree and the
qualified `1a9efc33efbc0f23e1e7676a5edded795f224c97` ancestry. The independent
source study is `/tmp/fpgs-kuka-private-response-study-Rvs4DGgP/CARD.md`, SHA256
`c4a19d9c8188a4c9f82517edc007c429015a1f44219656ab9e1ce244f832d140`.

## Removed boundary and new ownership

After **all** original allocation finalizers, select only unresolved worlds
with finalized `mf_count == 0`, under the existing exact recipe/capacity and
current/held generation guards. This is not final-zero-impulse admission.
Launch their current-response producer and original eight-sweep PGS owner on
the existing private stream before the positive-MF current-response and
qualification chain. Preserve the original positive-MF path, its independent
component overlap, and the final all-world join, guard and publication.

For selected worlds, omit the original arm/prefix/contact response producers,
global physical-J and kinetic-Z writes, repeated solve-side Z reads, and the
late offset-solve visit. Build current contact and prefix response into a
private world panel, retaining fixed sparse T180, held inverse6, the current
axes/origins and the existing raw-keyed contact cache. Publish original
canonical row metadata, impulses, solve status and `v_out`, not public body
state early. Original row reservation/order, contact/friction triples,
stateless limits, friction scheduling, early exit and eight-sweep law remain.

An allocator-produced normal-slot to raw-contact map is new storage/work;
charge its complete capacity and traffic. A full int[16384,192] map is
12,582,912 bytes. The existing global J/Z allocation remains for positive-MF
fallback; no allocation saving is claimed. Every map/cache use is bounded by
the current substep's completed allocation and is overwritten before reuse.

## Complete cost model

Exact c1 diagnostic RTX ownership from the closed current-contact captures:

| Affected ownership | ms/environment step |
| --- | ---: |
| Cached contact triplet contraction | 1.476172 |
| Arm + prefix + validation | 0.301440 |
| Current MF services | 0.448534 |
| Qualification + bridge + guard/init | 0.340075 |
| Offset/general solve union, not sum | 1.076044 |
| Complete affected fork region | **3.642265** |

Current discovery whole is 10.4681065 ms. A 10% whole saving requires the
complete replacement fork region to cost at most **2.595454 ms**, including
masked retained work, private owner, qualification/bridge, fork/join effects,
and added map/allocator cost. The old allocator's 0.207137 ms is not free.
The exposed 0.448534 ms MF services and approximately 0.300597 ms
qualification/bridge may overlap with the early owner; this is an unmeasured
scheduling opportunity, not guaranteed removable time.

Do not reclaim current all-world finish (1.667179 ms), complete state-held
region (3.201218 ms), collision (1.871128 ms), current-contact staging or force
publication. The current-contact five-owner family already costs 2.543053 ms
and retired the old CSR/ZERO/allocator/triplet family; that earlier gain is not
this candidate's gain.

## Lifetime and physical hazards

- Final MF count includes rigid velocity-limit allocation. Do not fork on an
  earlier contact-only count, or infer positive-MF independence from endpoints.
- Keep exact primary/free offsets, current MF versus held dense response and
  prescribed/nonresponding behavior. Positive-MF worlds retain original
  qualification, compact-coupled and general fallbacks.
- Original arm work initializes row status, validity and cold impulses. Move
  this ownership for selected worlds; do not leave a duplicate producer.
- Intermediate guards read both per-world and global row status. They must
  not race unfinished private production. Keep the final all-world post-join
  guard and error-before-public-finish semantics.
- Current MF preparation writes MF-only arrays, not `v_out`; velocity seeding
  happens before the fork. Private MF0 work must not read current MF buffers.
- Public q/qd, pose/COM velocity, cache generations and force outputs remain
  behind the joined original finish. Do not publish before another world's
  global error can be observed.
- Publish every canonical row/impulse field consumed by force export. Keep
  original fallback for unsupported debug, gradient or warm-start modes.

## Prior failures are not erased

Existing compact-coupled and kinetic-offset owners are already private in
velocity/impulse space; this candidate cannot claim those savings again.
They still consume globally materialized Z and fork after the MF chain.

Old Kuka full world-panel work cleared 256x17 shared entries, built per-thread
raw29/z29, used dynamic sparse-L lookup and serial substitutions, and ran MF
inside the dense sweep. That mapping lost and its physical gate was not
passed. Fr private-prefix/full-boundary and private-island trials likewise
showed underfilled geometry and sequential/shared-panel costs. Those are
real risks, not evidence that this fixed-T/current-cache, early-MF0 boundary
has been measured. No warp-packing or block/tier grid is authorized.

The literal 192x29 float panel alone is 22,272 bytes/world. With T, axes and
metadata it may approach 27–28 KB, versus the current offset owner's 2,880 B.
Charge actual emitted resources; do not infer achieved occupancy from static
register/shared-memory counts.

## Validation and decision

Use existing row/operator/loaded-coupled and live current/held/reset fixtures,
with regression-first checks for producer retirement and original recurrence
recovery. Compile actual native entries for both SM120 and SM100 offline.
Root owns the original paired physical smoke and whole/node timing protocols;
no new benchmark framework or observer is part of this experiment.

First measure the smallest complete implementation. Preserve whole and node
results separately. If the first mapping loses, identify one concrete causal
gap before one corrective experiment; no blind mapping grid. Failure to reach
readiness at the checkpoint requires a concrete barrier, not a silent
extension. No measured gain or qualification is claimed by this pre-code card.

## First complete source/offline checkpoint

The actual new owner is `kinetic_private_response.py`; nine constructor/call
binding lines in `kinetic_live_bindings.py` select it only when BOTH
`FEATHER_PGS_KUKA_CURRENT_CONTACT=1` and
`FEATHER_PGS_KUKA_PRIVATE_RESPONSE=1`. Missing either flag leaves the existing
path. The only new global device storage is the solver-owned int[worlds,192]
slot map; current-contact storage is reused, not duplicated. Used normal and
tangent map entries are written during the original atomic reservation;
prefix entries are never consumed. Both directed state calls share this map
only after the preceding original joined solve.

The implementation uses one 32-thread warp per active MF0 world. It owns
6276 float scratch entries: triplet scratch288, Z5568, T180, held inverse36,
current motion174, origins9 and C-map21. The original one-world solve metadata
adds 2880 bytes. Prefix and cached-triplet arithmetic retain their original
expressions and metadata publication; physical-J stores are absent in MF0.
The original CUDA eight recurrence is recovered exactly after undoing only
local-Z addresses, the removed unused global alias and the two-to-one-world
shared declarations. Held/current coefficient reads use the completed local
cache. The original CPU recurrence is retained for diagnostic comparisons.

The original row-global clear moves before the fork. The private owner does
not read or write row-global status; original raw/capacity flags are complete
before admission. Masked row owners do not touch private per-world fields.
The unchanged qualifier resets selector/eligible and returns before MF0 row
reads; private code never reads those selectors. The intermediate check and
general owner skip MF0. The general wrapper preserves the old invalid-world
guard before indexing MF count. Materialization returns on selector0 before
row-status reads. Original all-world post-join check and guarded public finish
are retained. No public state is published on the early private stream.

Regression-first: four tests failed specifically because the owner was absent;
inactive flag combinations already passed. With the implementation, all four
CPU tests pass, including nonzero loaded prefix/triplet impulses, early output,
poisoned selected global J/Z remaining unchanged, positive-MF/ZERO complement,
late type-4 withdrawal, current-geometry/held-operator reuse, shrink/empty/regrow
and capacity-drop failure followed by repair. The copied original/native
boundary control also passes on CPU. These are not CUDA or whole-step claims.

Existing controls pass: 37 executed plus one explicit CUDA skip across the
live-binding, live-owner and current-contact modules; with both flags enabled,
the existing 22 live-binding/lifecycle controls pass without skips. The root
also composes the already-qualified device-draining destructor and its two CPU
regressions in this isolated tree. That cleanup occurs at destruction, not in
the measured native producer/solve boundary.

Fresh actual-entry offline compilation completed for all nine changed entry
families on SM120 and SM100. All have zero stack frames and zero spills:

| Entry | RTX / GB registers | Shared bytes |
| --- | ---: | ---: |
| Complete private response/eight | 102 / 99 | 27984 |
| Original allocator plus slot map | 40 / 40 | 1024 |
| Positive-MF arm | 72 / 63 | 128 |
| Positive-MF prefix | 40 / 40 | 896 |
| Positive-MF cached contact | 80 / 80 | 1280 |
| Positive-MF validation | 38 / 32 | 128 |
| Retained two-world offset | 60 / 56 | 5760 |
| Bounds-safe positive-MF general | 100 / 100 | 4340 |
| Intermediate positive check | 40 / 32 | 1024 |

The general wrapper's malformed-world branch retains a second static copy of
the guarded source. Its extra registers/code are included above; no achieved
occupancy is inferred. The first pre-correction AOT01 remains preserved and is
not the final source pin.

Offline report:
`/tmp/fpgs-kuka-private-response-offline-qg8xXcLu/offline02/report.json`, SHA256
`3e2b71646e0c383b1cc001df4e90911d8fe4ff9f1840c7c9b1680bae300fcf94`.
It reuses the original offline compiler through the explicit binding
`compile.py` and the output-only repeat `compile02.py`; no GPU context was
available. Final native SHA256:
`f067046755e18ef8f6dd311c006daf6d900a338799868fadc98a021dd5a1f52c`.
Final binding SHA256:
`3da6a76be0aa40acdaf8a0c6fc7b2d5642376c04a482fc1e4ebd8204817af42c`.

The existing-fixture CUDA selector is
`tools.fpgs_bench.test_kinetic_private_response.TestKineticPrivateResponse.test_cuda_private_response_boundary`.
It compares actual original CUDA rows/eight against the private CUDA owner,
checks its own atomic map, poisoned selected panels, two graphs with count
shrink/regrow and finalized positive-MF withdrawal. The second selector,
`TestKineticPrivateResponse.test_cuda_two_bank_stream_lifecycle`, constructs
actual CUDA models and two production solvers, then calls original
`solver.step` with the real stream/finish ownership. It checks two captured
directions, positive-MF to MF0 to positive-MF transitions while another world
retains MF work, public state versus current-contact, held-bank reuse and
device generations, and global failure withholding public finish. Constructor
admission and the actual MF0 cohort are assertions, not assumed evidence.
Only the existing model fixture gained a `device="cpu"` keyword; unchanged
callers retain CPU behavior. The test additions pass their CPU/default checks
with two explicit CUDA skips. Root owns both GPU execution and the first
integrated whole test; no GPU or performance outcome is claimed at this source
checkpoint.

For root's isolated per-UUID physical lease, set the seven accepted Kuka flags
plus both experimental flags before importing Newton, then run from this tree:

```sh
FPGS_TEST_DEVICE=cuda:0 uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python \
  python -m unittest \
  tools.fpgs_bench.test_kinetic_private_response.TestKineticPrivateResponse.test_cuda_private_response_boundary \
  tools.fpgs_bench.test_kinetic_private_response.TestKineticPrivateResponse.test_cuda_two_bank_stream_lifecycle -v
```

The exact flags are `FEATHER_PGS_SIMPLE_WORLD_ZERO`,
`FEATHER_PGS_INDEPENDENT_COMPONENTS`, `FEATHER_PGS_PAIRED_GENERAL_OVERLAP`,
`FEATHER_PGS_LOCAL_ROW_PACKETS`, `FEATHER_PGS_KUKA_JOINT_WORLD`,
`FEATHER_PGS_WORLD_SCAN_PUBLICATION`, `FEATHER_PGS_KUKA_KINETIC_WORLD`,
`FEATHER_PGS_KUKA_CURRENT_CONTACT` and `FEATHER_PGS_KUKA_PRIVATE_RESPONSE`,
all set to 1. Root supplies `CUDA_VISIBLE_DEVICES` with the single card UUID;
no GPU launch is performed by the implementing agent.

Final combined CPU run selects 46 tests: 43 pass and three root-only CUDA
selectors are explicitly skipped (2.798 seconds, process reaped cleanly).
The staged full `uvx pre-commit run -a` passes. Both original runtime and
regression-first failure artifacts remain untouched; no push is part of this
checkpoint.

The first incremental whole baseline must be the c1 current-contact runtime
(the report-only successor `629dedb0` is equivalent), with current-contact=1
in both arms and private-response=1 only in the candidate. The baseline is not
the older 1a9 runtime without current-contact. Reuse the original fixed-import
variants/capture driver, 16K raw311296/broad442368, 200 warmup and 40 wall/40
profile steps. No new driver or whole-time claim is introduced here.
