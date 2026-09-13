# Kuka compact kinetic lifetime: first experiment

Implementation GO: 2026-09-13 00:08 UTC. First checkpoint: 01:38 UTC.
This is an unvalidated structural hypothesis, not a new performance result.
Accepted Newton remains50dfa28d; Isaac Lab remains unchanged.

## Complete cost and mechanism

Replace the primary current-force/predictor, dense row/action/eight-sweep,
and public-state/next-private-cache producers together. Current projected
bias23, compact screws/COM geometry, and a held180-coefficient inverse
whitener replace the canonical spatial-force, tau, factor and row bridges.
Current arbitrary external forces and controls remain current on every call.
The held operator remains distinct from current geometry and current MF inverse.

The conservative existing selected family has8.122447667ms RTX exclusive
busy time, not the sum of its overlapping kernels. The current repeated
12.862500350ms needs3.949476350ms net removal to reach the original-handoff
2x ceiling8.913024ms. The resulting complete replacement allowance is about
4.172972ms; the study uses the slightly conservative rounded4.172948ms.
New producers, materialization, fallback, retained-work dilation and added
gaps all count. The retained trace busy+gaps already costs4.732132ms, so this
scope alone cannot promise4x handoff, whose ceiling is4.456512ms.

The existing tau traversal is already O(n). The proposed saving removes its
global reverse-accumulator lifetime and repeated projection/consumption;
moving bias projection into the preceding finish does not make that work free.
For common-arm self-contact, combine actual relative anchor moments before
the held action, retaining every nonzero common-ancestor term. Actual captures
have shared anchors disabled. This is not the previously failed six-column
per-body response cache or a retry of the old serial sparse-front/row mapping.

## First causal implementation

1. Current arbitrary-force/passive/drive projection fused with held prediction
   and all35-DOF/current endpoint output. Retain original free6 service.
2. Current/next compact kinetic producer, eventually direct geometric180 and
   held refresh, with real first-output to second-input chaining.
3. Direct current dense packets and offset kinetic original-eight recurrence,
   retaining independent MF overlap and complete coupled fallback.
4. Public state, current force routes/counts and next-cache publication, then
   complete paired cost and live integration gates.

The initial512 fixture seed may derive private current data and held T from
original captured state. That is explicitly a producer-excluded component
diagnostic, never the complete replacement cost. No standalone factor timing
or percent-level mapping campaign is authorized by this experiment.

Independent checks cover arbitrary body/joint forces, current changed reuse
controls, effort limits/passive forces, large unrelated sibling wrenches,
all35 DOFs/free transport/prescribed twists, stale cache/held generations,
poison and original held physical action. Numerical/physical errors, not bit
identity, determine acceptance. Wrong finite operators are negative controls
for the independent oracle, not a demand for a costly runtime certificate.

Root owns all GPU runs; RTX and GB300 run paired with frozen imports. The
other three agents own predictor/operator, checked plan/state producer, and
independent tests respectively. Sources begin in the explicit scratch
`/tmp/fpgs-kuka-demand-state-Lv4cAaUh`; no accepted runtime is edited in place.

## Source studies

- `/tmp/fpgs-kuka-kinetic-lifetime-IvJRS2/CARD.md`, SHA256
  `8fc9cddc7b26ef28b93eb1fef3d0f2757873cf660bd9516c655743d687f10135`.
- Initial ABI SHA256 `c79ec5ce51a6ec50cd5dfcfa1d59067e27bedf9d4a33dc2089c703d8dc20a1b2`.
  Its force-half-swap prose is incorrect: literal original arithmetic uses
  `F=body_f[:3]`, `torque=body_f[3:]+cross(COM-origin,F)`. The implementation
  header and revised ABI must preserve this law; no actual half swap occurs.
- `/tmp/fpgs-kuka-state-owner-I5RRnj/MAP.md`, SHA256
  `ee6d9d2069980ec6744c743e5451075fa09deeabd2a2cd769607daf17fb6ff38`.

No gain is recorded at GO. If timing misses, diagnose one measured causal
boundary before a targeted correction; do not silently extend beyond two hours.

## First GPU physical checkpoint

The current-force/held-predictor primitive passes both actual CUDA test methods
on RTX and GB300. Both refresh/reuse saved512 states, changed arbitrary forces
and controls, original same-device force kernels, held free6 fallback, large
unrelated sibling wrenches, invalid-state isolation and two-graph replay are
covered. No tolerance or native arithmetic correction was needed for this GPU
batch. Ten CPU tests also pass; two GPU-only methods are skipped only in CPU runs.

Maximum observed candidate held-action backward error is1.172e-7; predictor
scaled error1.324e-6. The original same-device force oracle has small near-zero
component differences versus FP64, but its resulting physical action and
prediction pass the same gates. Component-bit proximity is not the criterion.

This still seeds private current coefficients and T from original current/held
data outside the tested primitive. It does not yet qualify live cache production,
contact solve/convergence, cold/reset behavior, or a new performance result.
The separately implemented current/next producer and native T refresh are the
next integrated gate. Direct current rows will reuse the existing Y allocation
as private Z, rather than introducing a second full response panel.

Parent `/tmp/fpgs-kuka-demand-predictor-physical-paired512-20260913-01`
is complete and reaped; both GPUs are idle and source guards pass. SHA256:

- Manifest: `8e0630f2f9a5e2cc922dc2b0989eb68db5246c3345b1da18186abccf383f523d`.
- RTX audit: `75ec7e9a0bf722883ad5c4a7b798f90ee8b84ea45e6520282f47ac9f2762d9ef`.
- GB audit: `2d415a299ffc2d6263750d1bdfd757c941d4623ae7f703d928418a41ddb3c18b`.
- Predictor native: `57b7b3bff720c248babd9148c835df1881f36e0f66c0c3ef09dc8e9d756adac6`.
- Types: `cb4077d182a021399fb15c747501f111981fd06d72dc0f89ba2e96d0c0a2bf85`.
- Independent tests: `8b291190096eb499f95af86d3c7e8ab616278b81811d280f9a11cb398ff84f00`.
- Explicit source/input inventory: scratch `pins_predictor01.json`.

## Generated-state and kinetic-core checkpoint, 00:50 UTC

The current/next state producer passes its actual CUDA physical and graph test
on both cards. The connected producer/refresh/predictor/finish chain then passes
two actual CUDA tests per card, including changed forcing on the newly produced
second state and two independently captured graph replays. These checks do not
qualify contact convergence or a live cache/reset protocol.

The first **kinetic-core-only lower-bound** timing uses actual historical16384
world archives, not repeated512 worlds. Each event contains two physical calls:
native primary refresh, current force/predictor and complete next-state producer
for each call; the second finish also produces next geometric coefficients.
Ten warmups precede40 measured graph replays, with every mutable seed restored
outside events. Event times are multiplied by4 for eight-call environment
equivalents. Both cards pass exact replay, all65 mutable/100 readonly array
ownership checks, physical-source pins and final process/idle guards.

| Hardware | Median core ms/env equivalent | Min–max |
| --- | ---: | ---: |
| RTX PRO6000 | 2.662528038 | 2.654336–2.678912 |
| GB300 | 2.322815895 | 2.314496–2.335104 |

**Not a whole-physics gain.** Contacts, allocation, dense rows/eight sweeps,
MF, retained free-factor refresh, callbacks and cold repair are excluded.
Initial warm current construction is setup, but both next-state producers
are charged. Free6 still uses the captured held operator. The captures are
historical full-state fixtures, not today's active-row capture distribution.
This experiment prices an architectural portion; it is not matched live A/B.

Against the planning allowance4.172948ms, the RTX core consumes2.662528ms,
leaving1.510420ms for the remaining replacement work and required accounting.
That is a tight **necessary budget**, not a prediction that2x will be achieved.
Accepted Kuka and Franka performance numbers remain unchanged.

Reproduction uses the root paired owner
`/tmp/fpgs-capacity-20260911-O7dqFC/run_pair.py` and scratch
`run_core_cost.py`, with `pins_core_cost01.json` and the completed chain physical
manifest supplied explicitly. All runs use clean accepted Newton50dfa28d and
the unchanged Lab environment. Completed/reaped artifact parents and SHA256:

- State physical: `/tmp/fpgs-kuka-demand-state-physical-paired512-20260913-01`,
  manifest `59c39eab60a9315cbf11fb3d8fc83a85f135c9bb519bc618e0b5f97086df9735`.
- Chain physical: `/tmp/fpgs-kuka-demand-chain-physical-paired512-20260913-01`,
  manifest `c07ffc5c283e9e3c2a8b60ad41cbd76944f745a3eba0d52fffaf21cd8a633496`.
- Core cost: `/tmp/fpgs-kuka-demand-core-cost-paired16k-20260913-01`,
  manifest `dcb292f58163681fdbcc28ce0a1e0c1381ceae74f79976d928445bc9f0a10dc6`.
- RTX cost audit: `c328b0e3e2bc1ff060ff7e4efec7424f9724a06607a4f2e874e7240ec63df8ee`.
- GB cost audit: `4c39163a1d731158162e7c8823acce676995d136c67e47dfc9b7fd87c4d9241f`.

Direct current contact rows and the offset-eight consumer are next. The first
row CPU smoke produces17741 dense rows with exact original prefix bounds and
no missing used rows; independent physical row checks are still pending at
this checkpoint. Saved raw allocation routes are explicitly external controls
until the retained allocator is connected. No saved J/Z/RHS feeds the candidate.

## Integrated implementation checkpoint, 01:38--01:48 UTC

At the 90-minute checkpoint the native two-call owner is assembled; no new
whole-task gain is accepted. It now connects original free6 refresh, current
force/prediction, the retained ZERO/CSR policy, current count/allocation, direct
rows, current MF inverse/preparation, qualification and positive-selector
materialization, original eight-sweep generic/private owners, mandatory joins,
guarded next-state publication and public contact forces. The second current
state comes from the first actual solved output, not a second archived state.
Original force export is once per two physical calls, not after both calls.

Actual current-eight tests pass on both GPUs, including the coupled-positive
GB world and independent MF cases, unchanged physical velocity/momentum/cone/
residual bounds, serial versus concurrent dispatch, poisoned outputs, two
graphs, failure-publication guards and public-force conversion. Maximum scaled
velocity difference against original current geometry/FP64 held H is5.373e-6,
below the unchanged3e-5 bound. This is finite-eight agreement, not proof that
all tasks have converged or that release parity is established.

The local row test still fails its added-r0 coefficient gate:6.545e-5 RTX and
7.559e-5 GB against7.629e-6. The failure is preserved. Both-card generated PTX
identifies different FMA grouping in point rotation; world-point rounding
then changes phi, amplified by phi/dt. Native arithmetic and tolerances are
unchanged. The complete-eight physical checks pass despite this local error;
bit-level coefficient matching is not substituted for physical acceptance.
Diagnosis: `/tmp/fpgs-kinetic-row-diagnostic-eGFPdS/ROW_ROUNDING_DIAGNOSIS.md`,
SHA256 `f11c133bfcdcfbf9f15f9eb22a317982d9bd09639b5c5c681a981cb39e7ea0d7`.

The connected two-call512 GPU gate then passes both cards. Its strengthened
second run independently reconstructs forces/bias on each actual current q/qd,
checks prediction and endpoint motion, original current eight-sweep physics,
original integration/publication, actual emitted-force conversion, repeated
graphs, and the complete readonly inventory. The largest predictor scaled
error is5.629e-7 and endpoint error1.172e-6. All current capacity/no-drop checks
pass. The complete inventory has230 mutable/225 readonly arrays, with exact
reshape aliases deduplicated; unused poisoned row storage is not consumed.

The event scope is now larger than the earlier selected-only budget: it also
charges retained MF, CSR, three count finalizers and public force conversion.
Recomputing the original disjoint Nsight union gives RTX selected-exclusive
9.771697ms, retained-outside2.862754667ms and gaps0.220128333ms. Thus its
conservative full-boundary screening allowance is5.822196650ms, **not**
4.172948ms. The same-window GB allowance is5.122562667ms. Free6 was already
selected and receives no second credit. This scope correction is not a gain.
Audited mapping: `/tmp/fpgs-kuka-complete-boundary-budget-MaJ7kfmz/NOTE.md`,
SHA256 `f2cde8056607d7217b6e9d14b8cb5c9330995b0d272919cb3182b7763c90f266`.

Completed/reaped physical parents:

- `/tmp/fpgs-kuka-demand-current-eight-physical-paired512-20260913-01`: four
  CUDA tests per card pass; original-current-eight and export/guard controls.
- `/tmp/fpgs-kuka-demand-boundary-physical-paired512-20260913-01`: three tests
  per card pass, including actual two-call solve/publication and graph replay.
- `/tmp/fpgs-kuka-demand-boundary-physical-paired512-20260913-02`: strengthened
  two-call gate passes both cards; manifest SHA256
  `d3b8bbd0884c640943a12518a3907411a06a63546816c9047b22b76b8b664e42`.

The first actual16K full-boundary cost run starts at01:48 UTC with a fresh
paired parent. It retains fixed historical collision/control inputs and
excludes collision generation, callbacks, cold repair and Lab/sensors. It is
an architectural cost screen, not a live-task measurement. Accepted Kuka and
Franka tables remain unchanged. A miss requires a causal whole-boundary trace
and at most one targeted corrective experiment, not a tile-size tuning grid.

## Full-cost checkpoint, 02:12 UTC

The complete diagnostic owner misses its structural cost screen. At roughly
two hours after implementation GO, the first successful40-sample paired run
is complete, reaped and source/idle checked. Accepted task timings are unchanged.

| Hardware | Complete owner median ms/env equivalent | Min--max | Screening ceiling |
| --- | ---: | ---: | ---: |
| RTX PRO6000 | 8.946559906 | 8.856448--9.028352 | 5.822196650 |
| GB300 | 9.924863815 | 9.851136--10.027520 | 5.122562667 |

These events contain the actual two-call producer/solve/publication chain,
including retained free6, MF, CSR, counts, guards, fork/join and public force.
Four times the two-call event gives the eight-call environment equivalent.
Fixed historical raw collision/control inputs remain a diagnostic limitation;
collision generation, callbacks, cold repair and Lab/sensors remain outside.
Neither a whole-environment result nor a fresh MJWarp comparison is claimed.
The miss is3.124363ms RTX and4.802301ms GB against the respective ceilings.

The earlier cost01 failed its cross-replay state-identity gate before producing
a timing result. A separate full16K eager/eager/graph diagnostic traced state
differences to contact-slot permutations from the unchanged atomic allocator:
initial current geometry, prediction, counts and raw routes matched. This was
not specific to graph replay. No native correction or tolerance widening was
made. The failed run and its original check remain preserved.

The successor gate compares every active world's candidate result with the
unmodified original CUDA eight sweeps for that same emitted contact order,
including current MF, net response, momentum, friction and zero/prescribed
ownership. Both physical runs and the cost run pass the unchanged numerical
bounds. Checks occur outside timed events. Across the cost replay checks,
maximum scaled velocity difference is below5.5e-7, versus the3e-5 bound.

This is implementation agreement, **not convergence acceptance**. Some of the
historical worst worlds retain substantial normal and joint-bound residuals
in both implementations. Selected64/128-sweep diagnostics do not consistently
remove them; those extra sweeps are not used in the benchmark. A bounded
parallel diagnosis will distinguish metric interpretation and inherited law
or conflicting constraints from new implementation harm, then check fresh
current captures before any live promotion.

The revised measurable question is where the complete graph spends its
exposed time after the cheap core is connected. Trace current row/action
production, retained MF/CSR preparation, guards, and the actual joined solve
critical leg, retiming the core in the same graph. The standalone core/full
difference is not additive causal attribution. RTX has no positive coupled
selectors in this run, so coupled fallback alone cannot explain its miss.
No candidate arithmetic changes or tile sweep are authorized by this miss.
Next checkpoint is03:00 UTC: a causal complete-path trace and one structural
correction hypothesis, or an explicit negative diagnosis if no sufficiently
large removable cost is identified.

Completed local evidence:

- Cost01 failure: `/tmp/fpgs-kuka-demand-boundary-cost-paired16k-20260913-01`.
- Order diagnostic: `/tmp/fpgs-kuka-demand-boundary-replay-diagnostic-paired16k-20260913-01`.
- Full-order physical gate: `/tmp/fpgs-kuka-demand-full-order-physical-paired16k-20260913-01`,
  manifest `6ed1ff6599b5571a6129c7df779176a9cdebcdf5478d366a9df0327f0d98613b`.
- Cost02: `/tmp/fpgs-kuka-demand-boundary-cost-paired16k-20260913-02`,
  manifest `71f10b8c7849a96a9ea2d3112d3f4208ec5c3734218bfe096492d10719678472`.
- RTX cost audit: `a6751659d07150f2948c81264ff1268725cb964ebd0f26c2b88fd380e5fa65c4`.
- GB cost audit: `d490466e754a181791754d5a81fa38883d8db8acb6858db4de940f8eeab98666`.

## Causal trace and corrective decisions, 02:21--02:48 UTC

The next checkpoint has a concrete cost diagnosis, not a new gain. Paired
Nsight graph-node capture and its CPU reader both pass, with the profiler
parents reaped and source/idle guards satisfied. Each three-environment
equivalent contains twelve restored two-call graphs. Restores and numerical
oracles are outside charged graph ranges; every kernel, copy and memset
inside those ranges is retained. Nonzero CUPTI correlations are checked;
missing IDs use explicit same-process completion-fenced intervals and repeated
node identities, not a fabricated correlation join.

| Complete-graph cost, ms/env equivalent | RTX | GB |
| --- | ---: | ---: |
| Graph span | 9.004254 | 9.934781 |
| Core force/prediction/refresh/next state, exclusive | 2.768914 | 2.337514 |
| Current dense-row family, exclusive | 2.566589 | 2.870132 |
| Contact-row kernel alone, exclusive | 2.291911 | 2.592906 |
| Retained CSR and ZERO, exclusive | 0.894345 | 0.775989 |
| Joined offset/general solve union | 1.042239 | 1.514100 |
| Coupled materialization, exclusive | 0.077749 | 0.633781 |
| Explicit current-error guards, exclusive | 0.036299 | 0.049333 |
| Uncovered graph span | 0.159167 | 0.191051 |

The contact row is a subset of its row family, not an additional additive
cost. On GB the general solver's1.476511ms duration overlaps the offset owner;
its exclusive contribution is0.612341ms. Materialization plus that exclusive
contribution exposes1.246122ms, not2.110292ms. No stall counters, bandwidth
utilization or hardware-ceiling claim is available. Contact registers are
90/86 with no reported local memory; few long coupled warp chains are a
source-supported inference rather than per-warp timing evidence.

The suspected large repeated state traversal was ruled out. ZERO reads the
predictor's endpoints; it does not rebuild them. Finish computes a different
q/qd epoch and bias acceleration. There are small dead stores and spills, but
no source-backed1ms deletion. That narrow route is closed without a new
state-fusion or register-tuning experiment. Diagnosis:
`/tmp/fpgs-kuka-state-repeat-review-YFavLtzO/NOTE.md`, SHA256
`48680579c0eda4189e7fa085e4baf020c1ae7a12a0aeccb80f5a790d58b9deb5`.

Two specific corrective experiments are authorized, each against its own
identified cause and each with the entire path charged:

1. **Contact-triplet work elimination**, GO02:36 UTC, checkpoint04:06 UTC.
   Fresh four-input counts show98.79--99.09% common-arm contacts and
   99.69--99.73% with no responsive free endpoint; no contact touches more
   than two fingers. The current kernel nevertheless evaluates all four
   finger blocks and free6 action. Exact endpoint-union support removes
   about42.5% of scalar held-action products. Batch normal/tangent0/tangent1
   and hoist repeated incident vectors to remove repeated dependency stages;
   retain all actual output rows/coefficient stores and all eight sweeps.
   Arm-lane dependency remains, so arithmetic fractions are not timing
   fractions. Hypothesis: at least1ms complete-path saving, not merely a
   component speedup. This alone cannot meet the full2x target. No global
   body-port cache, new screening or fused solve is added.
2. **Coupled primary-kinetic/free-physical representation**, GO02:44 UTC,
   checkpoint04:14 UTC. Keep primary23 as offset kinetic coordinates and
   free6 physical, deleting primary physical-Y construction. Dense free
   action still uses its held operator; MF uses its current-pose inverse.
   Preserve original dense/MF/rigid-limit and friction-sibling ordering.
   Extend the existing qualification visit and existing general dispatch;
   do not add a second certification scan or solver launch. Unsupported
   cases retain original fallback. The historical GB1.246122ms exposed
   opportunity is not a savings forecast. Franka is not a drop-in transfer
   and is not reopened without new measured structural evidence.

The corresponding pinned cards are:

- `/tmp/fpgs-kuka-triplet-card-pIKTct/CARD.md`,
  `e920cc315ca5418d5fc0e586544ba1d2b22103fe491d9a0c56216527441666bc`.
- `/tmp/fpgs-kuka-hybrid-coupled-card-AEKmC4H4/CARD.md`,
  `5b999ff65f5782630d3aa9f31c5f14f429c91d2b0ee0924defd720553092470e`;
  its admission/dispatch mechanics are superseded by `ABI_MINIMAL.md`,
  `07f239ef722df74b096a72f5ec2265f83d2d045d6c3dd347e793c2f19dc93047`.

### Fresh current inputs and numerical limits

A separate paired full-state capture of accepted50dfa completes with exactly
eight physical calls, original refresh/reuse, unchanged capacities, no row
drops and original Lab code. Both complete16384-world phase archives are
retained locally. Host snapshots perturb execution and supply **no timings**.
Coupled cases move between samples: fresh RTX has two positive worlds in
both phases, while fresh GB has none then one. This is a state-dependent
tail, not a GB-specific defect. Fresh matched prototype timings must precede
any claimed correction delta.

The historical large residuals are velocity deficits, not joint-position
errors or a CFM compliance allowance. Selected normal/bound inequality LPs
are feasible but require maximum primary joint rates of at least39.1/16.2
rad/s. This is not full friction-complementarity feasibility. Friction-off
and128-sweep diagnostics do not consistently cure the tails. Bounded note:
`/tmp/fpgs-kuka-residual-meaning-DEjZ9YJQ/FINDINGS.md`, SHA256
`0e373d5e74b0f861656652104e21f5a846c75265a4b9d8545de172c964da000c`.

Reading fresh accepted physical J, exact RHS and actual joined velocities
gives maximum dense-normal deficits0.02670/0.02086m/s RTX and
0.00532/0.01426m/s GB over the two phases. Joint-bound maxima are
0.00591/0.000455rad/s RTX and0.01640/0.00986rad/s GB; most active worlds
are near zero. The historical1rad/s tails do not occur in these snapshots.
This dense-row census is not a complete MF/friction/trajectory convergence
gate and does not qualify the candidate. No target or iteration was changed.

The explicit fresh construction adapter restores its source selector before
native launches. It preserves the actual original stage1-return predictor
input, current prescribed sources and first-solve-to-second-state aliases;
it does not substitute captured phase1 output. Five CPU binding/schema tests
pass. Root's full fresh GPU physical gate is running at this checkpoint;
no fresh candidate timing or corrective speedup has yet been accepted.

Evidence roots and SHA256:

- Nodes: `/tmp/fpgs-kuka-demand-boundary-nodes-paired16k-20260913-01`, manifest
  `17708a4da7bc136d102d2db4a028994f72316b59d36a6e7e64671d96163c0c95`.
- Node reader result: `/tmp/fpgs-kuka-boundary-ordered-nodes-b4EI8ujc/evidence01.json`,
  `e193acd034c84c78a33f406ae66b00c6cc2967c90fac5e79c3f6b875c47380bf`.
- Fresh captures: `/tmp/fpgs-kuka-full-current-paired16k-20260913-01`, manifest
  `1dfe6a2387ac7edcfc804f59abd4db146d7601e813561dd5b865eb9bfeaf5dbc`.
- Fresh dense census: `/tmp/fpgs-kuka-fresh-residual-UHiWKCgp/evidence01.json`,
  `281cc9f86785339398e2a115578adbdf3091da5490eb3c94ffc6c9d17497ba40`.
- Fresh adapter: `kinetic_fresh_fixture.py` in the kinetic scratch directory,
  `327ed4f7f496d99a8663da0855afdc867979bca773be430db6d5f1be84e7275e`.

All accepted task performance tables remain unchanged. The implementation
successors are experiments, not promoted Newton runtime changes.
