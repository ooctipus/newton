# Complete sparse supernodal refresh: bounded experiment

Funded 2026-09-16 around 15:25 UTC; native-ready checkpoint 16:30, no
unexplained extension beyond 17:00. This is an unqualified, default-off
factor algorithm experiment, not an accepted improvement or a new MJWarp
comparison. The source baseline is the qualified chain path at `f221`.

## Complete boundary and prior work

Replace only the complete held factor/inverse producer: current geometric
H434 plus original R and ready drive K to canonical W434. No global panel,
factor, response or downstream conversion is introduced. Current-state
production, held refresh/request masks, predictor, materialized Z, metric
eight-sweep solve, contact export and final public state remain unchanged.

The accepted geometric refresh costs about 1.780 ms RTX / 1.505 ms GB.
The falsifiable hypothesis is approximately halving that complete owner,
not counting isolated Cholesky savings. A matched whole-physics discovery
must show the saving without downstream growth. Original node costs are
diagnostic exclusive owner measurements, not a separate accepted timing.

Prior left-looking and right-looking level experiments are recorded in
`G1_LEVEL_UPDATE_20260914.md`. The accepted latter schedule visits 1,142
Schur destinations, executes 2,242 products and 45 factor-wide joins, but
retains the original per-column inverse's dependent traversal. This trial
changes both factor and inverse ownership. It is not spatial ABA, a block
size sweep, or constexpr expansion of the same level loop. The repository
also contains Kamino blocked LLT; that is not an existing implementation of
this packed FPGS inverse boundary.

## Exact panel graph and work

The original admitted 434-entry pattern is certified against the emitted
panel coverage at construction. Branch junctions split the natural chains:

| Panels | Count | Eliminated width b | Ancestor separator s |
| --- | ---: | ---: | ---: |
| Outer finger chains | 2 | 3 | 12 |
| Other finger chains | 4 | 2 | 12 |
| Arm chains | 2 | 5 | 7 |
| Torso | 1 | 1 | 6 |
| Leg chains | 2 | 6 | 6 |
| Free root | 1 | 6 | 0 |

Four fixed warp workers within the existing 128-thread CTA execute:
finger fans plus legs, then arms, then torso, then root. The two finger
workers each process their three independent panels; the other workers
process one leg each. Maximal consecutive fundamental supernodes would
merge through branch junctions and serialize opposite arm contributions;
that is deliberately not the graph used here.

Each front starts with original H only in its owned BB/SB columns. Its SS
corner starts at zero and receives only child updates. Factor C=chol(F_BB),
V=F_SB C^-T; pass the resulting SS update to the parent. No original H_SS
is duplicated or computed and subtracted back out. Form C^-1 and E=V C^-1
locally. After root publication, traverse ancestors first and publish
W_SB=-W_SS E into exactly the original packed slots. W_SS contains the
complete ancestor path, not sibling state.

Factor products are 157 local + 440 coupling + 1,645 separator = 2,242.
Inverse products are the same 157 + 440 + 1,645 = 2,242. Preserve 391
factor divisions, 43 square roots and 434 inverse divisions. Child update
assembly adds 587 scalar additions. The arithmetic is not claimed smaller;
the hypothesis is removal of repeated global topology gathers, CTA-level
pivot synchronization and dependent per-column inverse traversal.

Geometric source has eight whole-CTA joins: one after assembly, four after
forward stages, three after reverse stages. Original has 47, including
assembly and publication. Warp-local shuffles/row work remain and are
charged. Nongeometric physical-H assembly adds its original force join.

Shared storage is 434 inverse/factor slots plus 587 descendant-update
slots and status: 4,088 bytes before compiler overhead, versus original
1,740. Maximum front is 15 by 15 (120 lower entries). Its row ownership
holds at most fifteen scalar values per active lane; inverse composition
caches at most twelve ancestor values and three-to-six local values.
Compiler registers/spills and whole cost are decisive, not these estimates.

## Lifecycle, dispatch and gates

`FEATHER_PGS_SPARSE_SUPERNODAL=1` exposes `sparse.supernodal=True` and binds
`sparse_supernodal43_434`, or `g1_kinetic_sparse_supernodal43_434` after
kinetic installation. Flag off retains the old factory objects and source.
There is no additional world-scaled array or launch. The existing level
flag may remain enabled but does not select the active refresh when this
new opt-in is present. Only the already admitted exact SparsePlan is used;
the panel algorithm generalizes to tree separators, but no cross-task
native qualification is claimed.

Held mask-zero calls return before touching W/valid/status. R and drive K,
including the original serial compact-drive search and parallel readiness,
are read at the original refresh. Invalid/nonfinite factors latch status
and invalidate publication; this is not transactional rollback of old W.
All full-warp collectives execute for padded lanes. CTA joins are outside
warp-specific panel bodies, and parallel panels own disjoint columns.

The first missing-module regression failed before implementation. Focused
CPU checks execute both native CPU bodies against the actual G1 USD's
independently reconstructed H/action, including all finger and leg
contributions, geometric assembly, serial/parallel drive readiness, held
invalid-current input, failed refresh and flag-off identity. Existing full
native current/held action and continuing graph controls remain required;
the CPU body is not a substitute for CUDA execution. Root owns both GPUs
and the unchanged paired whole benchmark. No performance gain is claimed.

## First CPU and hidden-compiler checkpoint

Three focused CPU controls pass. Independent read-only reviews by both
other agents found no concrete panel-algebra, full-mask, shared-lifetime or
held-mask blocker. The first SM120/SM100 offline compiler run succeeds:

| Geometric owner, block128 | SM120 registers | SM100 registers | Shared | Stack | Spills |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original level owner | 40 | 39 | 2,252 B | 72 B | 0 |
| Supernodal owner | 96 | 96 | 4,600 B | 0 | 0 |

The nongeometric test factory uses96/95 registers,5,632 B shared and zero
stack/spills. These are static resources, not achieved occupancy or timing.
The higher register footprint is an explicit first-mapping cost; no register
or block-size search is authorized before the integrated result. Initial
offline artifacts: `/tmp/fpgs-supernodal-offline-sGFHXsrp/offline01/`.

## First native result and complete cost closure

Frozen first implementation pins:

- `sparse_supernodal.py`: `7176f84167a0bba9124e3d2b58ee7636b0fc65df41fc653c0679a6708cc99ef9`
- `sparse_factor.py`: `1ff4a3b329c81d4368963a5dfd67d238b407a07208d0e6bfcceb707f8ef2a24f`
- `test_sparse_supernodal.py`: `9498f169905e929679305dc287cb7c060d6ca11e07af19585b88247753faa999`
- Final source-matched offline report:
  `/tmp/fpgs-supernodal-offline-sGFHXsrp/offline02/report.json`,
  `ee9712448269adaff4f4ad36c26ae70999d072cd4db7f9938de34a95ff597b65`.

Three new CPU controls and eight original flag-off controls pass; targeted
and all-files precommit pass. Root ran the four native selectors on each
GPU: all pass, including actual physical H/action, geometric/nongeometric
assembly, serial/parallel drive readiness, held masks, failed refresh and
original dense-versus-candidate continuing-state graph controls. Logs:
`/tmp/fpgs-supernodal-native-20260916-HAZvGCRX/gpu{0,1}.log`.
No GPU execution was performed by the implementation agent.

The unchanged paired whole driver completed every actual capture and all
eight original capacity/owner boundaries. Original source and final idle
guards pass. The parent exits1 at a second exact-key check outside the
untimed observer, after all capture work: the new refresh key was not yet
included there. Root preserved that failed manifest, extended only the
exact-key allowlist, replayed checks on all four recorded children, and
verified three negative controls reject. This is not a rerun of timing or
a reason to relabel the original parent successful.

| Actual graph physics, ms/environment step | RTX | GB |
| --- | ---: | ---: |
| Qualified chain baseline | 15.054503250 | 19.841636000 |
| Register-front supernodal | 14.814186625 | 20.720324550 |
| Baseline / candidate | 1.016222 | 0.957593 |
| Baseline minus candidate | 0.240317 | -0.878689 |

Whole artifacts:
`/tmp/fpgs-g1-supernodal-whole-paired16k-20260916-01/`.
The mixed result is not promotion-qualified and does not approach the
prospective half-refresh hypothesis. There is no new MJWarp or other-task
measurement.

The subsequent twelve-step candidate node capture preserves the old
auxiliary-root analyzer rejection. Its strict extension accounts for48
physics and12 auxiliary roots, all source/capacity/finite/process-correlation
guards, and zero unproven graph nodes. Both cards execute exactly eight new
refresh launches per environment step and zero old refresh launches. The
per-world refresh-mask population is not instrumented; the original mask
owner/cadence is unchanged in source. All three state owners still execute
eight times, and no extra mass/row/solver producer appears.

| Node owner, ms/environment step | RTX old -> new | GB old -> new |
| --- | ---: | ---: |
| Complete refresh | 1.780024 -> 1.574609 | 1.504000 -> 2.376645 |
| Original GS | 3.957767 -> 3.943048 | 4.422344 -> 4.397013 |
| Contact producer | 1.251471 -> 1.247737 | 1.342003 -> 1.340533 |
| All rows, exclusive | 2.527910 -> 2.514818 | 2.554350 -> 2.534210 |
| Predictor | 0.560355 -> 0.556411 | 0.581146 -> 0.587467 |
| Repair | 0.181208 -> 0.181037 | 0.198781 -> 0.199363 |
| Finish | 1.842625 -> 1.838688 | 1.930120 -> 1.929389 |
| Collision, exclusive | 3.743738 -> 3.738117 | 8.378876 -> 8.374411 |

The old column uses the preserved qualified-chain node capture
`/tmp/fpgs-g1-chain-vector-strict-Os6BOg4H/`, not a fresh same-window
baseline. It diagnoses the owner change, not a second accepted throughput
comparison. The refresh delta is -0.205415ms RTX and +0.872645ms GB,
consistent with the whole result; no downstream growth explains the loss.
Actual new resources are grid16384/block128,96 registers,4600 B shared and
zero reported local memory on both GPUs.

Strict artifacts:
`/tmp/fpgs-supernodal-strict-Vplhu6e6/candidate_gpu0.json` SHA256
`e97facf7ede9147b5cc6915b4d0f7939a9c68b9ed2a9ac0fcd65bf55635b444c`;
`candidate_gpu1.json`
`ee6c643fa2097bfbe4a9feb0577c3b1a8a564f05a3801f63ab7802ff0f6f6e10`.
Reader SHA256 `618976dec17fbc26157bc1cb9682a9c9a6fd0e88505ed214038ea9c577d849cc`.
Raw capture:
`/tmp/fpgs-g1-supernodal-node-candidate16k-20260916-01/`.

## Source cause and one prospective correction

Eight whole-CTA joins do not mean eight local pivot rounds. The first
mapping's critical forward path has7 finger pivots,5 arm pivots,1 torso
pivot and6 root pivots:19 warp-local rounds, versus15 original levels.
Its source adds552 forward and316 inverse shuffle broadcasts per world.
Each lane retains an entire front row, then an ancestor row and local E
values; the measured96 registers are substantially above original40/39.

The first emitted native has17,384/17,296 static SASS instructions on
SM120/100 versus968/984 for the looped original. These are not dynamic-work
ratios. PTX has868 shuffle sites; SASS duplicates many sites into distinct
control paths, so its1,729 SHFL and458 BSSY/BSYNC sites are not claimed
runtime counts. No achieved occupancy, instruction-cache or stall counter
was collected. The observations identify added ownership/instruction costs,
not an isolated hardware bottleneck.

Root funds exactly one causal correction, after committing this recoverable
first version: retain the same twelve panels, four fixed workers, eight
CTA joins, exact factor/inverse algebra and canonical output. Replace
register-resident full rows with warp-private shared fronts and element-
parallel bounded loops. This targets the96-register liveness, replicated
column control and868 broadcasts. Added shared traffic and approximately
200--220 warp-local synchronization sites must be charged; there is no
new global matrix, task routing, block-size search or numerical method.

Prospective complete-owner falsification targets are RTX <=0.980ms to save
at least0.8ms against the qualified baseline, and GB <=1.504ms for no
regression. They are acceptance thresholds, not expected performance.
CPU actual-H/lifecycle checks and compact-loop/lower-register AOT evidence
precede root-owned paired native and early whole tests. Native-ready target
17:15 UTC, no unexplained extension beyond17:30. No correction runtime has
been written at this closure checkpoint.
