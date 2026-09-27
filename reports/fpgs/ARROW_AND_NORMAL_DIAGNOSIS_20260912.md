# Complete-state and normal-row experiments: measured diagnosis

Checkpoint at 22:20 UTC. These are isolated experiments, not new runtime
promotions. The accepted whole-physics table in
[the progress report](FOURX_PROGRESS2_20260912.md) is unchanged. The 4x goal
remains unmet. Isaac Lab, physical substeps, maximum eight GS sweeps and
capacities remain unchanged; original worktrees and the handoff are preserved.

## Franka: composite mass helps, complete arrow state still loses

The new primary representation stores a 45-coefficient geometric arrow:
the lower 7x7 arm block, two seven-entry arm/finger couplings and three finger
entries. Thirteen subtree composite moments replace the next-mass Gram
assembly. Held response uses an arm inverse-factor action and a 2x2 Schur
complement, retaining actual coupling and asymmetric drive terms. The opposed
mimic pair has an analytic recurrence equivalent to the original eight ordered
pairs with denominator-only CFM, including both impulses. Binding or uncertain
limits retain the complete original eight-sweep fallback from the original
free velocity. No canonical H/L/G cache is read or published by the candidate.

The complete owner includes current forces, primary/free dynamics, constraints,
generalized integration, all public body state, and next geometry and bias.
Current raw production, demand/MF preparation, unresolved canonical repair,
cold/reset/odd-refresh handling and live synchronization are still omitted.
There is no authorization to fuse across the existing per-substep force
callbacks. The reuse call sees the same collision generation, not a fresh
collision evaluation or an unbounded stale cache.

Both GPUs pass the original independent actual-CUDA suite with fifteen
physical records each and no skipped tests. Action error is at most 1.27e-5
against the existing 3e-5 gate; next physical H error is below 2.33e-6 against
3e-6. Current momentum, residual, both mimic impulses, current/held transitions,
public state and repeated graph controls pass without widening a tolerance.
Original finite-eight tails remain: full20 bound violation reaches .001010837
RTX / .011226350 GB; complementarity reaches about .02712 / .02968. This is
not universal convergence or MJWarp physical parity.

The first matched complete-owner screen loses: cooperative 2.064512 to arrow
2.293888 ms RTX, and 2.261376 to 2.593152 ms GB. Rather than reject composite
assembly from that result, a second controlled four-arm experiment separates
the mass collector from the complete representation change.

Forty samples per arm after ten warmups use eight balanced orders, each five
times, and ten appearances of every arm in every position. Each event contains
two actual chained owners, scaled by four to the environment cadence. All
arrays are restored outside timing. These are materialized current snapshots,
not a new closed-loop whole-task rollout.

| Complete owner, ms/environment equivalent | RTX PRO6000 | GB300 |
| --- | ---: | ---: |
| Cooperative | 2.072576 | 2.273920 |
| Typed free | 2.171008 | 2.347776 |
| Composite arrow | 2.302080 | 2.601472 |
| Arrow with original Gram collector | 2.375744 | 2.691520 |

Composite assembly saves .073664 / .090048 ms relative to Gram in all forty
matched samples on each card. Both compile to 127 registers, 3712 bytes shared,
136 bytes stack and zero spills. Current/held/public outputs are exact; only
the next geometric floats differ, under their independent physical gate.
The Gram variant also passes all fifteen physical records per card. Thus
composite arithmetic is beneficial; it does not cause the overall regression.
The complete action/fallback representation and its compiled lifetime cost
more than the typed predecessor. Packing/storage changes prevent attributing
that difference to one instruction or claiming a register-only explanation.

All four arms select the same 15,552 RTX / 15,424 GB worlds each call. RTX
fallback counts are 320/352, with six or seven rows; GB has 192 in each call.
Every known fallback world is isolated from other fallback worlds in a
consecutive two-world grouping. Long-tail latency is a plausible cause, not
proven merely by these counts.

One fixed two-world/16-lane mapping is authorized for a bounded follow-up:
share instruction issue across independent complete worlds without changing
their math. This differs from the failed four-way serial ABA mapping, but can
reduce resident warps and leaves every known heavy fallback block intact.
There will be no lane sweep or composite-only micro-optimization promotion.
Even a 30% arrow-owner reduction leaves almost no room under the proposed
20%-whole-family allowance for the still-missing live services. Compare with
the best cooperative owner too, not just with the slower arrow baseline.

## Kuka: normal-first screening closes on measured work, not bit identity

The earlier rare output disagreements were traced to current normal J/Y/diag
reassociation in rejected worlds. Against independent FP64 current geometry,
four of six differing worlds have closer candidate coefficients; the other
two are only slightly worse. Own-coefficient original-eight, serial/overlap,
metadata, force conversion, MF and capacity controls pass. No wrong index,
sign or constraint law was located. Original output-acceptance failures remain
recorded, but demanding a bit repair is not the reason to close this mapping.

Forty balanced complete-family samples regress in every pair on both cards:
3.469184 to 3.780928 ms RTX; 3.637760 to 4.052480 ms GB. Fixed-source node
profiling then retains all kernels, graph copies, memset nodes and stream
joins, and accounts separately for restores outside the graph.

| Family, ms/environment equivalent | RTX original → candidate | GB original → candidate |
| --- | ---: | ---: |
| Current contact geometry/J/RHS/restitution | .800485 → .832754 | 1.047104 → 1.035958 |
| Added prefix/response/trial/commit | 0 → .721484 | 0 → .789139 |
| Original all-row / rejected-row response | .527961 → .279299 | .427770 → .256813 |
| Paired/general solver union | .853256 → .771609 | .881683 → .826093 |
| Complete busy union | 3.404289 → 3.708909 | 3.553721 → 3.991027 |
| Uncovered graph gaps | .056534 → .051085 | .126074 → .119024 |

The baseline already skips zero-force tangents and stops stationary sweeps.
Omitting roughly 80% of rows is therefore not an 80% solver-time opportunity.
The new normal path still constructs common geometry/metadata, while rejected
tangents repeat common work. Exposed prefix transactions cost more than the
response and solver work removed. Gaps improve and general8 remains hidden;
neither lost overlap nor increased launch gaps explains the regression.

Even deleting the entire new prefix transaction AND entire normal owner for
free cannot recover a 10% current whole-physics gain with the remaining work
unchanged. A normal-J-store or fused-predicate correction is strictly narrower.
Close that family rather than polishing it. Any reopening must replace a
substantial retained response/recurrence or another complete ownership boundary.

## Reproduction evidence

All large captures and experimental native sources remain local. Exact paired
commands, source hashes, UUIDs, array/cohort evidence and event samples are in
the following manifests. Children are reaped and final source/idle guards pass.

- Arrow physical: `/tmp/fpgs-franka-arrow-correctness-paired512-20260912-01/manifest.json`,
  `11ac7e43059c2055a8d818d46d95f7ceea8cdf922dddcec765cfc7aeb7020910`.
- Arrow matched: `/tmp/fpgs-franka-arrow-matched-paired16k-20260912-01/manifest.json`,
  `936ab25c72098b2e5e0db4f97eb7ab3f768924e6f228c0fea9af17f7eb36186b`.
- Gram physical: `/tmp/fpgs-franka-gram-correctness-paired512-20260912-01/manifest.json`,
  `1a2b2974ef2c9dde5b0b9f642864916e7f97aa5260e9ef9e18cff30283223c33`.
- Four-arm: `/tmp/fpgs-franka-arrow-ablation-paired16k-20260912-01/manifest.json`,
  `b73d80cd6a3987e39609a4a60ada8e90a203ed1ca119dec6e1e7998f8bdc06ee`.
- Kuka family cost: `/tmp/fpgs-kuka-normal-cost-paired16k-20260912-01/manifest.json`,
  `20a55a130682e962415507c9b433e967b5d97c2499999a402d9237056bae28f0`.
- Kuka node cause: `/tmp/fpgs-kuka-normal-node-audit-VJ7Q30cU/FINDINGS.md`,
  `3e3a7e277c2a3050f7a52c93126a78d3c7e0b7825be9d4feba6ddd2211052dc6`.
- Two-world decision: `/tmp/fpgs-franka-two-world-card-f0lBS8la/CARD.md`,
  `cd90097e` prefix. This is a proposal, not accepted runtime or performance.
