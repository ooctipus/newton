# ANYmal compact branch response: candidate card

2026-09-13 23:23 UTC. Implemented experimental path; CPU/offline checks pass.
CUDA physics, live admission, graphs and performance are not yet accepted.
First integrated checkpoint: 00:03 UTC (90 minutes after card creation).
Base bd1cc095, inherited runtime19099. Existing G1 paths are unchanged;
this is a separate default-off ANYmal constructor admission.

## Removed work and retained law

ANYmal has a free root6 and four independent leg3 chains. Reverse elimination
preserves a117-entry lower factor rather than the ordinary171-entry dense
triangle (324-entry storage). A terrain-contact row needs root6 plus its own
leg3, not18 response coordinates. Unlike the failed lazy active-panel method,
every current row is prepared cooperatively; no contact or tangent activation
is predicted, no projection is deferred, and no row-sum preconditioner changes.

The complete candidate must replace, not supplement:

- Dense H/L production with direct packed leaf-first L117 from the existing
  current composite inertia/screws and held-generation request. Do not build W
  or convert an already-produced canonical factor.
- Predictor actions with the two sparse triangular actions on this same held L.
- Full18 contact whitening with root6/leg3 whitening in the existing row owner.
- Full18 exact Gram dots with root6 plus the matching leg3 contribution.
- Full18 zero-filled Z storage and repeated reductions with root/leg panels;
  only CTA-local leg membership is needed, not a per-world global queue.
- Final response decode with the same packed L transpose solve and canonical
  velocity/impulse publication.

Retain the actual geometry, RHS/restitution, force and reset lifetimes, all
rows, normal-dependent friction disk, original full EX1 row sums, Nesterov
restart and maximum24 parallel sweeps. Retain the original8-sweep fallback
allowance. Preserve dt/substeps and all calibrated capacities. Unsupported
constructors, two-leg self-contact unions, and larger row counts require a
complete explicit route; retired canonical factors must never be read.

## Cost hypothesis and risks

Historical ANYmal complete parallel solve owns about4.1ms of roughly9.2ms
whole physics on RTX. Instrumented CTA phase evidence puts both preparation
and recurrence near45%; these are not exclusive wall fractions. Reducing
both is essential: merely reducing visits with unchanged preparation already
failed the10% screen, and serial lazy preparation made the later candidate
1.62x slower. Target at least0.92ms net whole saving; this requires about23%
of the old solve family before charging any added dispatch or factor cost.

This representation can remove arithmetic from both phases without serial
normal/tangent insertion. It does not establish a timing prediction: packing,
leg-index gathers, membership construction, smaller parallelism, register
lifetime and complete fallback may erase the savings. Preserve efficient
existing producers wherever they are outside the replaced boundary. No tile,
register-cap or block-size sweep is authorized by this card.

The inspected source is `_get_pgs_solve_parallel_kernel`: existing INK/WR
already avoids global contact J/Y and separately computes the exact row sum.
Those already-removed producers cannot be credited again. The existing G1
SparseFactor class is not directly reusable: its admission excludes this
in-kernel parallel owner and hardcodes43/434/18 topology throughout.

## Initial mathematical check and bounded acceptance

Reused `/tmp/fpgs-anymal-lazy-reference-zE5yUe/rows.py` and its pinned actual
512-world captures, step00/tier48, first8 owned worlds per card. Reconstructing
H from the saved ordinary factor introduces cross-leg roundoff6e-9 to2.3e-8.
Imposing the structural zeros implied by the model, then reversing the factor,
gives117 nonzeros and at most9 response coordinates per row. The normalized
Gram difference from the saved-factor response is2.1e-9 to8.4e-9. This is a
small CPU mathematical sample, not a production geometry/factor/rollout test;
production must assemble from true current inertia, not threshold H entries.

Numerical gates compare held/current action, residual/complementarity, cones,
public velocity and loaded/reset/fallback behavior at the original allowances.
No bit identity or independent per-operation proof is required. Reuse the
existing captured inputs/evaluators and live drivers. A source-coupled unit
test is acceptable; a new benchmark framework or census is not.

Implement only if source review confirms the full fallback and producer
replacement can be bounded. First complete paired timing must charge refresh,
preparation, recurrence, publication and fallback. A miss gets one causal
diagnosis and a correction justified by it, not indefinite mapping variants.
No promotion until repeated whole timing and physical qualification pass.

## Implemented boundary and exact ownership

The default-off `FEATHER_PGS_BRANCH_RESPONSE=1` constructor admits the actual
18-DOF/free-root/four-leg topology, including fixed foot descendants. It requires
the existing EX1/WR32/48 matrix-free recipe (24 parallel sweeps and ordinary
8-sweep fallback). Unsupported constructors retain the original path. After
admission, unsupported settings or notified topology changes fail before private
work; reconstruction and graph recapture are required. Numeric model changes
still flow through the original notification and mass-update request.

`branch_response.py` owns one `(worlds, 1, 117)` held L plus two integer vectors.
The original H/L arrays become one-element dummies; there is no old-factor
conversion, second inverse, extra response panel or per-world leg queue. Existing
J/Y arrays are retained because the actual >48-row fallback needs them. The
original row builder produces those fallback J rows; two packed solves produce
Y in the original buffers. Existing CFM assembly, generic solve, public state,
contact forces, body-velocity limits, reset and allocation stay in their owners.

Current composite inertia and screws assemble only structural H entries, adding
the authoritative effective armature and enabled augmented K. Factor refresh
obeys the original device mass mask; mask-zero returns before any held write.
Prediction uses the same held L and current generalized force. The original18
predictor is tiled (the old loop threshold is 12), so the replacement is one
cooperative warp/world with 18 active coordinate lanes and two shuffle solves.
It is not the abandoned one-active-lane-per-CTA draft.

The compact32/48 owner constructs every row, including both tangents, and stores
root6 plus up to two leg3 blocks (12 slots). Current endpoint masks select exact
leg unions; no activity prediction or approximate support threshold is used.
Two-leg self contacts are supported. Three-leg overflow is sticky and cannot be
overwritten by a fourth leg; admitted two-body rows cannot require that route.
Current physical projection writes directly into shared compact coordinates,
without an intermediate local J18. The original RHS/bias/current-velocity
accumulation order is retained. EX1 includes all rows and matching leg terms.

The recurrence keeps the original NCH/float4 chunk mapping: NT32 uses one chunk;
NT64 uses three chunks across 54 coordinate lanes. Root6 reductions keep that
mapping and association; leg reductions use CTA-local membership words and FFS
gathers inside each chunk. Those tags, ballots and gathers are added work, not
free operation-count credit. The full projection, Nesterov restart, stopping
test, friction siblings and iteration counts remain the original native body.
Final packed transpose action writes canonical DOF order.

## Checks and bounded next gate

The missing-module regression failed before implementation. Final CPU suite:
six tests pass and the three explicitly root-owned CUDA tests skip. The direct
factor/predictor/fallback CPU bodies run against the actual USD topology and an
independent body-energy operator, including changed numeric mass and exact held
reuse. Preserved current/held input cases from both cards and both row tiers
have maximum normalized Gram discrepancy `1.3102144020254568e-8`; this is a
reference check, not an integrated live claim. Both selected CUDA factories bind
the unchanged 93-argument saved ABI in a no-launch CPU smoke.

The three existing-module CUDA selectors are:

- `TestBranchResponse.test_cuda_actual_current_held`: original native versus
  compact24 over all owned worlds of eight current/held/tier input cases; public
  velocity, momentum, cones and residuals retain the original allowances.
- `TestBranchResponse.test_cuda_two_leg_self_contact_and_warm`: actual current
  endpoint geometry with two-leg support and nonzero normal/tangent warm state.
- `TestBranchResponse.test_cuda_live_constructor_fallback_and_graph`: actual
  two-world constructor activation and H/L retirement, original current row
  production for both <=48 and >48 contact counts, held reuse, two-call graph
  replay, masked reset and complete body/force publication. No supplied J enters
  that integrated test. This test is still unrun on CUDA.

Run through the unchanged strict `run_checks.py` (SHA `027da7a4…`) and paired
owner `run_pair.py` (SHA `155eba00…`), with `FPGS_TEST_DEVICE=cuda:0`; all three
selectors/card must run without skips. No new test runner is introduced.

The minimal existing benchmark boundary now records requested flag, actual
owner, L shape and valid/status counts, rejecting requested-but-inactive. This
is commit `961b7e2f` in structural-bench; 18 CPU tests and full pre-commit pass.
It performs no factor-panel readback and no timed instrumentation. The requested
owner must have all current factors valid and status zero after warm-up.

## Final offline resources, not timings

Retained output: `/tmp/fpgs-anymal-branch-final-offline-mfmfq1ii`.
These are actual sm120 and sm103 ptxas builds; every kernel has zero stack and
zero spills. The separate initial artifact retains the rejected local-J18 and
serial-predictor resource result; no performance conclusion is drawn from it.

| Kernel | sm120 registers | sm103 registers | Shared bytes |
| --- | ---: | ---: | ---: |
| Direct refresh L117 | 42 | 32 | 1032 |
| Cooperative predictor | 62 | 50 | 128 |
| Large-row response | 54 | 54 | 128 |
| Compact32 | 118 | 102 | 4072 |
| Compact48 | 150 | 130 | 5828 |

The coherent mapping removes the earlier hoisted-factor/local-J lifetime, not
an arithmetic or block-size sweep. Resource reduction does not establish a
whole gain. Actual complete physics/cost remains the next decision, targeting
at least 0.92 ms RTX net, with all retained production and fallback charged.

## First CUDA result and admission correction

Preserved first pair: `/tmp/fpgs-anymal-branch-physical-paired-20260913-01`.
Both cards passed the current/held and warm two-leg physical tests. Maximum
current/held scaled velocity error on RTX was `3.3128989584452733e-6`. The
actual constructor test failed at inactive owner before private work; this is
not a completed integrated qualification. Both child exits were 139 after the
test failure, an unresolved teardown failure, not explained merely by the
assertion. Parent failure and final source/idle success remain preserved.

A root-authorized RTX-only, unchanged-source constructor diagnostic evaluated
every admission predicate. The sole rejected condition was `_fused_k1=True`;
the competing mass flag was false and all topology/other predicates passed.
The original force-only K1 returns before mass/L access (`fused_dynamics.py`),
while the existing composite producer still supplies the current I/S refresh.
The correction therefore retains K1 without changing it and rejects only its
mass-producing variant plus debug checks. A focused host-admission regression
models CUDA-only metadata explicitly and rejects mass/debug variants. No
runtime arithmetic or physical tolerance changes accompany this correction.
The repeated three-selector pair uses the existing `--fpgs` original GROUP16,
ROWS_MASKED1/X4 setup for the small grouped constructor; it must also exit
normally before acceptance. No automatic complete timing follows a failure.

Pair02 (`/tmp/fpgs-anymal-branch-physical-paired-20260913-02`) passed both
saved-input controls and the actual constructor/eager publication, large-row
producer and held-reuse checks on both cards. Its first baseline graph capture
failed with CUDA905 (uncaptured stream dependency), followed by901; both child
exits were normally1, not139. The test omitted the original documented
`seed_double_buffer_events()` call inside capture. Actual unchanged
`FeatherPGSManager._prepare_cuda_graph_capture` already makes that call. The
test-only correction adds it for both original and candidate (a no-op for the
retired candidate memset owner). No runtime or physical comparison is changed.

Pair03 passed both graph replays and their full public-state comparisons, then
failed before reset execution because the test passed int32 instead of the
original cache invalidator's `wp.bool` world mask. Both children exited1 and
the parent source/idle guard passed. The test now uses the actual Lab/API bool
mask; runtime and physical tolerances remain unchanged. The failed03 record is
preserved at `/tmp/fpgs-anymal-branch-physical-paired-20260913-03`.
