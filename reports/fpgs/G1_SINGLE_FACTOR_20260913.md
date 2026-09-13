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
