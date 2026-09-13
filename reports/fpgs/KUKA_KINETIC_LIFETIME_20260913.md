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

## Fresh matched contact correction, 03:03 UTC

The unchanged prototype passes its fresh complete-boundary GPU checks on both
cards: own current prediction, independent held action, every active world's
original same-order eight, and repeated two-call graphs. The first actual
solve remains the second call's state; this is not a captured phase1 replay.
The subsequent fresh baseline cost completes with40 samples per card.

The contact-triplet successor then passes both the independent historical512
current-geometry check and fresh16384-world complete path on both GPUs, with
two tests each and no skips/errors. Its matched cost also passes source,
readonly, per-order original-eight, and final GPU-idle guards.

| Complete replacement section, ms/env equivalent | RTX | GB |
| --- | ---: | ---: |
| Fresh unchanged prototype | 10.339200020 | 9.249216080 |
| Fresh contact-triplet successor | 9.552639961 | 8.540672302 |
| Time removed | 0.786560059 | 0.708543777 |
| Section speedup | 1.08x | 1.08x |
| Previously derived2x-handoff screening budget | 5.822196650 | 5.122562667 |

This is a measured structural section gain, **not whole-physics performance**
or convergence acceptance. Collision generation, callbacks, cold repair and
Lab/sensors remain outside. The at-least1ms correction hypothesis was not
reached; the42.5% scalar-action count was never a timing prediction. Keep
this source fixed rather than tune it for the remaining fractions. The
coupled representation correction is the next already-authorized experiment.
Fresh and historical timings are different states, not an A/B gain comparison.

The unchanged fresh timing range is10.191744--11.297664ms RTX and
9.183616--9.323008ms GB. The triplet range is9.413504--9.593728ms RTX and
8.454784--8.593920ms GB. This is one matched sequence, not yet repeated live
task evidence. No accepted Kuka or Franka number is replaced.

Evidence and frozen source pins:

- Fresh physical parent:
  `/tmp/fpgs-kuka-fresh-boundary-physical-paired16k-20260913-01`, manifest
  `e4b81f3aaae36f1ec9d5a7fdcb8d9690fa69aab57e11cef2d53398ed7404feb5`.
- Fresh baseline cost:
  `/tmp/fpgs-kuka-fresh-boundary-cost-paired16k-20260913-01`, manifest
  `1a0fc7e31c3671134b5b6f1e0cf58cf970bd8b68c2eea6bc11c6c671a4d6c890`.
- Triplet physical parent:
  `/tmp/fpgs-kuka-triplet-boundary-physical-paired16k-20260913-01`, manifest
  `542703b95aba06b875b63cecf96a123fc64107c66ba790d11fb317dd134116e5`.
- Triplet cost parent:
  `/tmp/fpgs-kuka-triplet-fresh-boundary-cost-paired16k-20260913-01`, manifest
  `396a39438c420c1b346a93e18d66c9aa1e3b7033141386bc8e3c89674c86e107`.
- Native `kinetic_rows_triplet.py` in the shared kinetic scratch directory:
  `8bbf458d280d7447afc983b25669c5dc09c6fbaae4788981b8fd6ff03e234fcc`.
- `run_triplet_fresh_boundary_cost.py`:
  `76287917225b2eb7b75b72e8ecec3ba9612859d34ba67530e904a0f84dc0cfad`.

The runner differs from the fresh baseline only in its description and
pre-setup contact installation. It retains10 warm samples then40 measured
samples, two physical calls scaled by four, and the same untimed physical
checks. All dt/substeps/eight sweeps/capacities remain unchanged.

## Coupled correction and causal dispatch check, 04:00 UTC

The coupled kinetic prototype passes four paired physical tests per card,
including loaded192/64 rows, both articulation offsets, independent current
geometry and actual fresh two-call state. These checks preserve the original
eight-sweep law; they are not long-run convergence or live-task acceptance.
Its forty-sample complete-section cost does not demonstrate a useful gain:

| Fresh complete section, ms/env equivalent | RTX | GB |
| --- | ---: | ---: |
| Original prototype | 10.339200020 | 9.249216080 |
| Coupled kinetic prototype | 10.204031944 | 9.355584145 |
| Original with existing active queue | 10.400703907 | 9.222720146 |
| Coupled with the same active queue | 10.195839882 | 9.270080090 |

The fixed queue probe reuses the existing active-count/IDs and unchanged
compiled general kernel, with512 workers. It adds no array, preparation pass
or new native mathematics. Both queued variants pass fresh physical checks,
forty cost samples, original-eight replay, source and final GPU-idle guards.
Applying the queue to both variants avoids crediting dispatch to the kinetic
representation. It does not recover a substantial saving; close this probe
without a worker-count sweep. All parents and children are reaped.

Fresh paired node traces explain where the original coupled gain went:

| Charged stage, ms/env | RTX original -> coupled | GB original -> coupled |
| --- | ---: | ---: |
| Materializer exclusive | 0.764447 -> 0.077461 | 0.320715 -> 0.073835 |
| Joined offset/general solve union | 1.684873 -> 2.113941 | 1.252384 -> 1.609173 |
| Qualification exclusive | 0.183104 -> 0.205301 | 0.209141 -> 0.214603 |

The primary conversion deletion is real. Its0.686986/0.246880ms saving is
substantially or completely offset by the larger solve. Mutually exclusive
native bodies reserve additive shared storage:13,184B versus4,340B, with
89 versus82 registers and no local spills. GB phase0 has no hybrid worlds,
yet the original-branch general duration rises48.677 to71.979us. This
supports investigating execution/resource overhead but does not establish
an occupancy or stall attribution. Hardware counters remain unavailable.

### Actual recurring-work census

A separate untimed diagnostic inserts only integer observations into both
native paths. Removing the marked observations recovers the original source
exactly. Its changed register count means its durations are not performance
evidence. Both variants pass the same current-order original-eight gate on
both GPUs, two tests per card and no skips/errors. The actual first solve
still supplies the second call's state. No cost runner imports the counters.

| Coupled world | Sweeps, original/candidate | Dense visits | Zero-normal, zero-tangent-pair visits |
| --- | ---: | ---: | ---: |
| RTX4930, each phase | 8 / 8 | 392 | 224 (57.14%) |
| RTX10325, each phase | 3 / 3 | 108 | 66 (61.11%) |
| GB8730, phase1 | 8 / 8 | 304 | 160 (52.63%) |

The independent-world sweep histograms also match in this census. There is
no observed sweep inflation here. These tangent observations occur after
the original friction/denominator gates and before residual/impulse work;
they identify actually visited work, not an apparent inactive-row fraction.
All coupled MF updates are zero in these particular inputs, which does not
authorize removing MF coupling, its current inverse, or rigid-limit laws.

This is new evidence for one compact-coupled recurrence experiment: retain
the existing offset/triplet dense loop, perform current MF and stateless
rigid limits inside each sweep, and share mutually exclusive scratch space.
The proposed existing-materializer bridge converts only the six secondary
columns, preserving primary Z and the original diagonal. Its bridge,
fallback, qualification, state publication and joins must all be charged.
With other costs fixed, a1ms RTX net saving needs joined solve at most
1.349661ms, versus2.113941ms in this coupled prototype. Avoidable visits
are not proportional wall-time savings. GB cannot offer1ms from this scope
even at the optimistic exposed floor. No compact successor is timed yet.

The contact-triplet1.08x above is **prototype-to-prototype**, not improvement
over accepted Newton. No additional whole-task gain is accepted in this
24-hour window at this checkpoint; all accepted task tables remain unchanged.
This coupled correction alone cannot close the full2x-handoff budget.

The compact correction is authorized at04:03 UTC, with native review targeted
at04:35 and an initial90-minute checkpoint at05:33. The actual gate uses a
fresh companion original control, not an assumed transfer of these timings.
Decision: `/tmp/fpgs-kuka-hybrid-cause-AA6yvh5h/CENSUS_DECISION.md`, SHA256
`b1036cff05115f084be3cc1e6c2cf2848629ebbbc09ed73671707f61233cc850`.

The separate typed-collision layout remains NO-GO for implementation: fresh
RTX replaceable CSR/count/MF scope is1.199508ms, leaving only0.199508ms
for every writer/replacement service to save1ms. Retaining current MF
inverse/action alone spends0.194325ms. A new W-by32 reservation is rejected;
existing storage can be reused only with all readers and fallback generation
handling redesigned. Source feasibility is not sufficient timing evidence.
Updated budget: `/tmp/fpgs-kuka-collision-layout-JRnnqo/FRESH_BUDGET.md`,
SHA256 `746ed2dc74d551fdd0713831357a2082a98ef588cc18722774092054fbbfbe60`.

Evidence roots and SHA256:

- Coupled physical:
  `/tmp/fpgs-kuka-hybrid-boundary-physical-paired16k-20260913-01`, manifest
  `05a9243b2a6e20b08c5060f01b9d7bbc179894db47b83beb49d06577e75b2396`.
- Coupled cost:
  `/tmp/fpgs-kuka-hybrid-fresh-boundary-cost-paired16k-20260913-01`, manifest
  `7ce7c31a75106e65b9ac288706b5a19a146268f4f45e1b42ecb1c0f1faa4bd4b`.
- Fresh baseline node reader:
  `/tmp/fpgs-kuka-fresh-baseline-nodes-VaV6hzef/evidence01.json`,
  `cba4ec1995f7272bad0c897561a2d77508b6a91565fe12c681f1b1fe579d5418`.
- Fresh coupled node reader:
  `/tmp/fpgs-kuka-fresh-hybrid-nodes-48JhGLsw/evidence01.json`,
  `07ea429e563ac909a649642ebd7f5362ad00b840de5cf32e3af40760995233e5`.
- Queued physical:
  `/tmp/fpgs-kuka-active-queue-physical-paired16k-20260913-01`, manifest
  `65f62330bf302a299f5c1e2f7f3a944eb16ed6aca784d3570fe8707f4bbb7efe`.
- Queued original/coupled cost parents:
  `/tmp/fpgs-kuka-queued-fresh-boundary-cost-paired16k-20260913-01`, manifest
  `99970df93f6e2f2adb97507e6947d27a4ac4d0f5ccc6d21d261fa48bc783c93e`;
  `/tmp/fpgs-kuka-queued-hybrid-fresh-boundary-cost-paired16k-20260913-01`,
  `7c4841b22522cddb344e2195518cf2147cc3abc564718829232eac2662c0a3c8`.
- Actual sweep census:
  `/tmp/fpgs-kuka-sweep-diagnostic-paired16k-20260913-01`, manifest
  `d5b1a6a2c4ec2431ffc97ec98843f1d5b9d14eacfab7fb3575c3f808652a462b`;
  RTX/GB driver logs respectively
  `f976643c915d582a98156f917df4c93c05fbe5f1547a49ce68c1638c50ebe235` /
  `4ddfc702096ed70deaa21f4da3a1665e50e663778fb5c04cfe25ad657d28a24d`.
- Source/PTX cause and conditional compact design:
  `/tmp/fpgs-kuka-hybrid-cause-AA6yvh5h/NOTE.md`,
  `45470c3528dfb80d06c0b78af1ba9858392b298203c4fb85b3d18c741ad5db81`.

## Compact recurrence clears the standalone gate, 04:25 UTC

The single compact correction completes its physical and cost gates before
the05:33 checkpoint. It preserves primary Z, converts only six free response
columns and replaces the eligible dense recurrence with the existing compact
contact-triple law, followed by current MF and stateless rigid limits inside
each sweep. Alternative native paths share one1053-float arena. Offline both
architectures use92 registers and4340B shared with no stack/spills; the bridge
uses40 registers and256B shared. These resource counts are not speed claims.

Four CUDA physical tests pass on each GPU with no skips/errors: loaded192/64
and both offsets, independent fresh current geometry, historical512 original
geometry and actual fresh16384-world two-call graphs. The fresh complete
coupled cohort remains2/2 RTX and0/1 GB. Ordinary per-order eight-sweep,
velocity/momentum/response/contact-law and public-state checks are unchanged.
Arbitrary nonzero initial warm-start impulses are not newly numerically tested;
the cold recipe is retained and the existing initial-impulse readers survive.

A new original companion control and the standalone compact candidate each
complete10 warm and40 measured samples, full section and final source/idle
guards. All parents and children are reaped.

| Fresh standalone complete section | RTX | GB |
| --- | ---: | ---: |
| Contemporaneous original prototype, ms/env | 10.396544456 | 9.253376007 |
| Compact correction, ms/env | 9.175871849 | 8.679872036 |
| Time removed, ms/env | 1.220672607 | 0.573503971 |
| Section speedup | 1.13x | 1.07x |

The primary at-least1ms RTX gate and no-GB-regression condition pass. This is
one paired prototype-section comparison, not a newly accepted live Newton
runtime, whole-task speedup or long-run convergence result. The fixed2x/4x
targets and accepted task table remain unchanged. No queue or contact-triplet
correction is included in these numbers. Their composition is the next separate
physical/cost measurement; do not add component savings.

Evidence and SHA256:

- Native `kinetic_compact_coupled.py` in
  `/tmp/fpgs-kuka-compact-coupled-JaVsIZBa`:
  `84a6bd0479dd68c47b59381ef0cc23c0b4b921c9b6dfda7c05a4148c943b88d9`.
- Ready/readiness scope: `READY.md` in that directory,
  `02423e4ab3b0430b548757ff996a2673b0ef90083f2f58b9184f3b974324cfbe`.
- Physical parent `/tmp/fpgs-kuka-compact-boundary-physical-paired16k-20260913-01`,
  manifest `a891159aa0bc9cda2f56b356569142cd3307366b8699f84d460f2cebb7f8d615`.
- Fresh original cost `/tmp/fpgs-kuka-fresh-boundary-cost-paired16k-20260913-02`,
  manifest `c0a70695ee37ee07188a4a33d5646133f4eab1c98bb72021754e333ed3a28093`.
- Compact cost `/tmp/fpgs-kuka-compact-fresh-boundary-cost-paired16k-20260913-01`,
  manifest `0dd370f7f965bc7fae8898dcefc9a0d3312d1f1d673970a30a318491eb5eaab0`.
- Complete runner `run_compact_fresh_boundary_cost.py`:
  `cade3b30baae7f8c8c51b1664859fcec3dbf06582d3f3ae6d4f4f541cc7157f6`.

The exact full-cost scope, first-output-to-second-input ownership and excluded
live collision/callback/cold-repair/Lab work are unchanged from the prior gates.

## Composition preserves the gain; compact trace closes the cause, 04:51 UTC

The compact recurrence and contact-triplet row producer are now measured
together, not credited by adding their separate savings. Composition changes
the existing row-kernel closure before setup; it adds no owner, array, queue or
launch. Two complete-boundary CUDA physical tests pass on each GPU with no
skips/errors, followed by the unchanged current-order original-eight checks,
10 warm and 40 measured samples, source guards and final idle checks. All
parents and children are reaped.

| Fresh Kuka prototype replacement section | RTX | GB |
| --- | ---: | ---: |
| Contemporary original control, ms/env | 10.396544456 | 9.253376007 |
| Contemporary triplet-only control, ms/env | 9.503615856 | 8.542719841 |
| Compact plus triplet, ms/env | 8.328000069 | 7.966592073 |
| Time removed versus original, ms/env | 2.068544388 | 1.286783934 |
| Time removed versus triplet alone, ms/env | 1.175615787 | 0.576127768 |
| Section speedup versus original | 1.25x | 1.16x |

These remain prototype-section results, normalized to eight physical calls per
environment step. Collision generation, inter-call callbacks, cold repair and
Isaac Lab execution are not included. They are not accepted whole-task times,
new MJWarp ratios, long-run convergence evidence or evidence of gains in
Franka/other tasks. In particular, do not add the new section time to an old
retained-work estimate and call that a measured live result. Accepted task
tables and the unmet additional 2x/4x targets are unchanged.

A separate compact-only causal trace passes all source and physical authority
checks. It covers 12 tagged two-call graphs (three environment equivalents),
with restoration and physical comparisons outside the measured graphs. Every
memory node and join is charged. Fork costs use the interval union rather than
adding overlapping kernel durations.

| Causal trace, ms/env | RTX | GB |
| --- | ---: | ---: |
| Original joined offset/general solve | 1.684873 | 1.252384 |
| Earlier coupled joined solve | 2.113941 | 1.609173 |
| Compact joined solve | 1.060191 | 0.928544 |
| Compact general-solve exclusive tail | 0.000000 | 0.036661 |
| Compact response materializer | 0.077440 | 0.073685 |
| Compact full graph span | 9.233372 | 8.668917 |

The compact correction therefore retains the conversion deletion while
removing the larger solve cost that defeated the earlier coupled prototype.
The RTX joined solve is below the predeclared 1.349661ms planning bound. This
is a measured causal explanation for the successful retry, not an assertion
about hardware occupancy or peak throughput. The trace is compact-only;
composition performance comes from its separate complete-section cost run.

Next is a bounded live-integration design: replace the old production path
before its producers run, preserve actual contact-force export cadence, and
handle reset/notification and held-mass refresh lifetimes. No production solver
change has been landed at this checkpoint. A separate first-hit contact census
failed its new synthetic CUDA control on both cards before collecting fresh
counts; diagnosis is pending. That diagnostic failure is not a failed compact
physical test and supplies no activity or performance evidence.

Evidence and SHA256:

- Composition adapter `kinetic_compact_triplet_boundary.py` in
  `/tmp/fpgs-kuka-compact-coupled-JaVsIZBa`:
  `525f7334d426af18bf7d80cbc8cae0f0e806a04fc304a4a8783e65bc2a80960c`.
- Composition runner `run_compact_triplet_fresh_boundary_cost.py`:
  `2e3abf1c4e30c29ee7918240db8e3fafa338abc87c9f7efea1d98e8b0d69e6d9`.
- Composition physical parent
  `/tmp/fpgs-kuka-compact-triplet-physical-paired16k-20260913-01`, manifest
  `a01981de948f4638dbeb903b27b4017288418f27701ebe8e0ad72babf8914b33`.
- Contemporary triplet-only cost parent
  `/tmp/fpgs-kuka-triplet-fresh-boundary-cost-paired16k-20260913-02`, manifest
  `184789c47808b2d3db9e91605a12d9301b7470c03e6b5488280bbdad58f00b9c`.
- Composition cost parent
  `/tmp/fpgs-kuka-compact-triplet-cost-paired16k-20260913-01`, manifest
  `335861498ac2251e51a2d01f08974b1d130d926dfdb2710e66a26903da9b8ed5`.
- Compact-only trace parent
  `/tmp/fpgs-kuka-fresh-compact-nodes-paired16k-20260913-01`, manifest
  `bfe926f55f01b0a8d7e7d6c46b0f5a4ffaf934c141b82c0b6860d748f7fcb0b8`.
- Compact-only node reader
  `/tmp/fpgs-kuka-fresh-compact-nodes-MfaDuxY8/evidence01.json`:
  `17ec10a8f2760032af255bbd3a26bee1396962188aee26a37a9b9b752dc5691f`.
