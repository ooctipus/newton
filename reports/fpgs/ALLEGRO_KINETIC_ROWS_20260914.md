# Allegro current-motion kinetic rows

Pre-code candidate, 2026-09-14 20:15 UTC. Base `92507273fc6a6fdc7a219d928a3d6cf942e473d0`.
First integrated-readiness checkpoint 21:40 UTC; no GPU launch without the main
agent's lease. Default-off `FEATHER_PGS_ALLEGRO_KINETIC_ROWS=1`; preserve the
measured direction-cell collision option and enable it equally in both arms.

## Complete boundary and budget

Replace generalized dense coordinate-prefix/contact J production and the
parallel solve's per-row forward triangular solve with current-body motion
maps whitened by the **held** factor. A contact's two current endpoint wrenches
contract these maps directly into Z. Typed position/velocity limits use signed
inverse-factor columns. The consumer takes Z, constructs the unchanged scaled
parallel iteration, and applies the original final inverse adjoint. No new J
producer remains on owned worlds.

The original process-correlated three-step node reader, with only explicit
owner regrouping, measures the seven retired producers plus complete solve at
7.657296667 ms RTX exclusive (7.936795333 ms union), 7.895162667 ms GB exclusive.
The seven producers alone expose 2.316216 / 2.207666 ms; these are not additive
to unrelated overlapping stages. Target: at least 2 ms RTX whole-physics saving,
so the complete new family including maps, rows, prefix, solve and fallback
must cost at most 5.657296667 ms. Counts and arithmetic are not timing evidence.

Inputs: `/tmp/fpgs-allegro-direction-cells-nodes-paired16k-20260914-01`, candidate
arm. Retired names: `allocate_joint_velocity_limit_slots`,
`populate_joint_velocity_limit_J_for_size`, `clear_grouped_jacobian_active_rows`,
`populate_world_J_masked`, `contact_row_prelude`, `compute_world_contact_bias`,
`apply_world_contact_restitution_matrix_free`; position-prefix work is replaced
too but not credited in that seven-owner budget.

## Algebra and storage

For current motion columns S_b and held H=L L^T, K_b=L^-1 S_b^T.
For current endpoint wrench w=(direction, (point-origin) cross direction),
Z=K_a w_a-K_b w_b. Same-articulation endpoints combine in the same kinetic
coordinates. Current geometry must never be projected through stale motion
columns. R and implicit drive augmentation stay in the original held L.

The actual hand has four independent four-coordinate chains. Sixteen unique
ancestor prefixes permit incremental outer-product map construction: 240
hand multiply-add terms plus a 126-term free-six map, rather than independent
six-RHS 16-coordinate solves for every body. Proposed storage is 384 hand-map
and 36 cube-map floats/world plus the small inverse-factor columns and current
L^T v_hat vector. A contact has at most two responsive endpoints; finger/cube
needs at most ten Z coefficients. Charge all storage, map construction, and
current-velocity conversion in the complete family.

## Preserved laws and fallback

Keep all sixteen velocity-limited coordinates, original activation fraction,
both signed limit rows, capacity/status counters, row ordering/phase bounds,
friction/material/anchor/restitution semantics, and public raw-contact impulse
ownership. Velocity prescaling is **retained**: it acts before FK and scales
the articulation's eligible scalar velocities together; post-predictor limit
rows cannot replace that law. Keep the existing 24 parallel iterations and
12-iteration matrix-free/GS fallback and original stopping/projection rule.

Worlds above the parallel row budget or with matrix-free rows materialize
canonical J before the original response/fallback. Their active J prefixes
must be fully cleared before materialization, including ping-pong reuse and
grow/shrink transitions. Canonical allocations can remain for this fallback;
no memory-saving claim is made. Warmstart, debug, deferred response, unsupported
row families/topologies and mutable-plan notifications require explicit
admission/reconstruction handling before private ownership.

## Prior art, tests, and decision

The old Allegro unit-pair sharing idea was a CPU-only ~11.8% duplicate-row
screen, not this complete boundary. Prior WORLD_ROWS still built general J
and ran per-row triangular recurrence. The Kuka six-body-column cache was a
CPU NO-GO at 1.70x arithmetic: it lacked this four-block prefix reuse and must
not be cited as a native failure. Franka packets supply a lifecycle precedent,
not proof that Allegro's 24-step parallel operator can be changed.

Regression first: constructor admission; current/held motion-map action
against independently formed current J and held L; signed position/velocity
rows; contact metadata and restitution; original parallel impulse/velocity
and physical residual checks; >128/MF fallback, reset/grow/shrink and graph.
Reuse the saved actual Allegro current-owner/row captures and existing solver
oracles, with no new benchmark or capture framework. Run CPU/precommit and
offline SM120/SM100 compilation before source freeze, then the original paired
physical and whole owners. One diagnosed correction is permitted only after
complete owner timing explains a mismatch; no layout or parameter sweep.

## CPU and native-source readiness

Eight CPU controls pass, including a real saved-model production constructor,
on the four original clean 512-world captures under
`/tmp/fpgs-ten-hour-20260911-gc5FsH/current_whole_owner_allegro512_gpu{0,1}`.
Their hashes are checked by `tools/fpgs_bench/test_allegro_kinetic_rows.py`.
Current physical-J reconstruction defects are 4.15e-8 to 4.40e-8, below the
2e-6 gate. Changed current motion with held L, exact signed fallback rows,
metadata and the untouched ordinary parallel ABI are covered. CPU native
binding checks do not claim execution of the CUDA solve.

The bias oracle uses the original kernel on the identical current produced
geometry. Comparing directly to historical CUDA phi made a CPU transform
rounding difference of at most 4.77e-7 metres appear as a 1.14e-4 velocity-bias
error after division by dt. That intermediate comparison was not retained as
a physical correctness gate; no tolerance was widened. GPU matched-input
velocity/impulse and independent momentum checks remain required.

Storage is explicit: 420 map + 100 inverse + 22 incident floats/world;
10 coefficients, one encoding word and two fallback-token words per dense
row; four group/start indices and 22 body-map indices/world. At 16,384 worlds
and dense capacity192 this adds 200,802,304 bytes (191.5 MiB), while canonical
J remains allocated for the complete fallback. No memory saving is claimed.

Offline SM120/SM100 compile passes in
`/tmp/fpgs-allegro-kinetic-rows-offline-ShKBY7/offline05`. The four solve tiers
use 94/99/96/96 registers and 5,868/10,024/14,172/18,232 bytes shared memory;
all have zero stack and spills. Map/prefix/contact producers use 48/44/72
registers and no stack/spills. An initial dynamic register-vector scatter
generated an 88-byte stack; replacing it with constant-index gathering before
measurement removed that unintended materialization. There was no block-size
or parameter sweep. The final host-admission source is pinned by offline05.

Admission excludes response-block mode and capacities below128 because those
change the available parallel tiers; the ordinary complete path is retained.
Moving hand joints must belong to the original PRISMATIC/REVOLUTE/D6
velocity-limit-supported set. The unsupported-type regression failed before
that host guard and passes after it. Topology changes require reconstruction
before any producer is retired.

## Initial paired native evidence

The three matched-input CUDA selectors passed on RTX and GB300 in 6.834 /
6.772 seconds. Root sessions80632/1374 both reaped0; source hashes matched
before/after. Evidence is retained in those tool transcripts, not a separate
log artifact. These cover all four saved captures, current/held row geometry,
the unchanged24-step parallel impulse/velocity law, independent physical
momentum action, exact fallback rows and composed graph/grow-shrink behavior.

The separate full production lifecycle selector passed eager current/held,
>128-row fallback and reset comparisons on both cards. Its first graph attempt
failed in the ORIGINAL solver capture at an uncaptured external-stream
dependency (CUDA905, then901); sessions5730/23793 reaped1 and are preserved.
The correction matches the existing Lab capture protocol: drain eager work
and call `seed_double_buffer_events()` inside each original/candidate capture.
It changes no native math, execution streams, or physics tolerance.

The corrected full lifecycle test passed on both cards: root sessions68607
(RTX) and65466 (GB300), both reaped0, 28.502/30.856 seconds including cache
loads (not simulation timing). This includes current/held factors and geometry,
poisoned retired J, >128-row exact fallback, reset, and both original/candidate
two-step graph replay through empty/regrow transitions. Final module b1a4e1f0,
solver89c5ec00 and lifecycle test1da40f32 hashes matched before/after. Evidence
remains the preserved root tool transcripts.

Status: CPU, native numerical controls and complete production lifecycle pass.
Ready for the unchanged original paired16K whole-step benchmark; no timing
claim, performance acceptance or promotion yet.
