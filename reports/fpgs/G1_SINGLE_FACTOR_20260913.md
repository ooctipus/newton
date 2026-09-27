# G1 forward-only response experiment

GO 2026-09-13 11:33 UTC; first integrated checkpoint 13:03 UTC.
Default-off candidate, not an accepted improvement. The fixed 4x-MJ target and
representative baselines remain in [the window plan](STRUCTURAL_FOURX_20260913.md).

## Evidence and falsifiable budget

Fresh fixed-Lab G1 node captures are at
`/tmp/fpgs-g1-postfix-nodes-paired16k-20260913-01`. Both actual simulations and
capacity checks completed; the original node analyzer refused G1 auxiliary
sensor graphs. The failed parent manifest is preserved. Both parents were
reaped and source/GPU-idle guards independently passed. A process-correlated
reader separates 12 physics roots from 3 auxiliary roots, without rerunning
physics or accepting instrumented timings as whole-physics speedups.

Initial RTX attribution identifies approximately 7.992 ms in the 43-DOF
per-row H-inverse response and 1.339 ms in its separate diagonal production.
The original eight-sweep solver costs approximately 6.228 ms. These initial
node sums require exclusive/overlap confirmation; no ratio is claimed from
them. Saving 10% of the accepted 43.717024 ms requires at least 4.371702 ms
net removal. The provisional response/diagonal replacement allowance is
4.959298 ms, including decode and any added solve preparation or solve growth.
This single scope cannot deliver the 13.066073 ms 4x-MJ ceiling. Terrain
collision and remaining dynamics/solve need independent substantial cuts.
Shared collision changes must also remeasure the MJWarp denominator.

## Removed and added work

Keep the existing held Cholesky L and current physical Jacobian J. Replace
per-row `H^-1 J^T` with only `Z = L^-1 J^T`, storing Z in existing Y storage.
Produce `diag = ||Z||^2` in that pass. Do not construct an inverse, a new
maximum-sized response panel, or a second canonical Y.

Run the existing eight-sweep contact/joint-limit projection law in offset
kinetic coordinates. Once per solve, form `rhs + J v_hat` in shared row state;
start kinetic delta at zero and use Z for residual dots and velocity updates.
Then decode once per world, `v_out = v_hat + L^-T delta`. Preserve original
CFM, friction scheduling, stationary-sweep exit, current rows and public
impulses/state. Restitution and other pre-solve consumers still see physical J.

Admit only a homogeneous single full-response articulation per world, no MF
or propagation owner, augmented drives, current friction, cold start, original
interleaved positive sweep budget, no separate velocity passes, regularization,
pre-elimination, debug or competing local/parallel response owners. Unsupported
construction or per-call contracts use original production and solve. No
private state survives a call; reset and held-factor refresh retain their
original ownership. New admission must not change the task parameters.

## Early gates and stop conditions

1. Test forward action, kinetic diagonal and final decode against independent
   physical response on well-scaled and conditioned SPD operators, including
   43 DOFs, inactive/partial rows, permuted world/articulation maps and zero
   rows. Check actual original eight-sweep contact/limit behavior and graph
   replay; numerical/physical differences, not bit identity, govern acceptance.
2. Run a short actual G1 graph/lifecycle comparison on both cards, followed
   promptly by a 16K live A/B with existing capacities and unchanged Lab.
   No standalone response-only promotion or large capture-wrapper campaign.
3. At 90 minutes report integrated gain, diagnosed loss or actual blocker.
   If the first mapping misses its theoretical removal, inspect response,
   shared preparation, GS and decode costs. Permit one correction for the
   observed cause; no generic tile sweep or raised grouped-DOF limit.
4. Promotion requires repeated paired whole-physics gain, physical convergence,
   capacity/warning checks and unsupported-task/reset fallback checks. G1-only
   dispatch is not evidence of improvements in Franka, Kuka or other tasks.

## First integrated result and correction, 12:10 UTC

The original tiled candidate at `d69683f7` passed all four tests on each GPU
(including native-produced diagonal, original eight sweeps, independent
momentum action and graph replay); no CUDA tests skipped. Test parent:
`/tmp/fpgs-g1-single-factor-tests-paired-20260913-02`.

The first actual 16K live A/B completed with all four children, both metadata
capacity checks per child, unchanged budgets, actual fixed-backend imports and
final source/idle guards passing. One discovery round, not promotion:
`/tmp/fpgs-g1-single-factor-live-paired16k-20260913-01`.

| Whole physics | RTX ms | GB300 ms |
|---|---:|---:|
| Original 50dfa | 43.391887 | 72.405953 |
| First tiled forward-only | 44.880584 | 74.684839 |
| Original / candidate | 0.966830x | 0.969487x |

Candidate node diagnosis at
`/tmp/fpgs-g1-single-factor-candidate-nodes-paired16k-20260913-01`
proves all eight calls use the replacement response, GS and decode. No old
H-inverse response or diagonal kernels remain. Actual simulations and capacity
checks pass; the original analyzer again refuses auxiliary graph roots and the
failed parent is preserved. Independent final source guard passed, and GPU
compute-idle was observed before releasing the next job. The preserved reader
plus explicit candidate owner-name mapping is at
`/tmp/fpgs-g1-single-factor-node-owner-00CzJ8nG/audit.py`.
This node sample follows only one wall step versus the old attribution's 40;
it diagnoses gross owner costs, not an accepted matched node speedup.

| Owner, node sums | Original RTX / GB ms | Candidate RTX / GB ms |
|---|---:|---:|
| Row response | 7.992 / 9.606 | 8.193 / 9.908 |
| Separate diagonal | 1.339 / 0.495 | retired |
| GS | 6.228 / 7.566 | 7.729 / 8.669 |
| New velocity decode | absent | 1.137 / 1.278 |

The removed arithmetic did not reduce the forward owner's execution cost.
Generated CUDA confirms the candidate retains the same library forward TRSM,
shared physical-J and forward panels, and adds a squared-response tile for its
norm. Both measured row-response kernels still use 255 registers/thread and
23,920 dynamic shared bytes/block. This is a resource/source observation, not
a counter-backed assertion of the limiting pipeline or occupancy.

The first targeted correction replaces that materialization with one block
per world: share L once across all active rows, let each warp hold one row's
forward response in registers, and emit its norm directly. It keeps the same
offset-coordinate GS and physical decode so the diagnosed response cause is
isolated. No inverse, new persistent buffer, capacity change, iteration change,
or Lab edit. The original library prototype remains available as a diagnostic
control. Native tests and integrated timing are pending for this correction.

## Native correction: measurable but below the structural milestone

At `a4093d57`, all four physical/graph tests again passed on each GPU, zero
skips (`/tmp/fpgs-g1-single-factor-tests-paired-20260913-03`). The next clean
paired live A/B completed at
`/tmp/fpgs-g1-single-factor-live-paired16k-20260913-02`, with all original
capacity/import/budget/finite/source/idle checks passing:

| Whole physics | RTX ms | GB300 ms |
|---|---:|---:|
| Original 50dfa | 43.409484 | 72.285541 |
| Native forward-only | 41.665184 | 70.348522 |
| Original / candidate | 1.041865x | 1.027535x |

These are one-round discovery results, not accepted performance or proof of
actual-task convergence. They fall short of the 10% whole-physics milestone;
no new accepted MJWarp ratio is claimed. The baseline comparison is against
accepted FPGS, not against the slower first prototype.

The follow-up three-step node capture uses the matching 200 warm / 40 wall
schedule, unchanged capacity/budgets, and shows response 4.869476 / 5.522732 ms,
decode 1.143821 / 1.281173 ms, and GS 7.776190 / 8.748587 ms (RTX / GB). New
response uses 48 registers, 7,908 static shared bytes, one block/world; all
eight old response/diagonal calls are retired. This diagnoses why the targeted
correction helped and why most of its saving was consumed by offset-GS
initialization and physical decoding. The complete changed boundary's exclusive
cost is 13.787257 / 15.547852 ms versus original 15.556000 / 17.660330 ms.

Raw nodes: `/tmp/fpgs-g1-single-factor-candidate-nodes-paired16k-20260913-02`.
The original auxiliary-root analyzer failure is preserved; both simulations
and capacity checks pass. The same process-correlated audited reader with only
explicit native owner-name mapping is `audit_native.py` under the existing
`/tmp/fpgs-g1-single-factor-node-owner-00CzJ8nG` directory. Root independently
rechecked the full source guard and paired compute-idle after parent reap.

The candidate remains default-off and experimental. Do not spend this window
on tile sweeps to promote a few percent: any further work must remove the
remaining representation conversions or a complete substantially larger owner.
Terrain collision has a separately funded, physically tested candidate; shared
collision improvements still require a new MJWarp denominator. A host admission
regression additionally rejects partial-warp block sizes before native dispatch.
