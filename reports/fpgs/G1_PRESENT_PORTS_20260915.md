# G1 current body-plus-present-limit ports: pre-code card

Recorded September15, before runtime edits. This is an unimplemented
representation hypothesis, not a speedup or a reopened measured port loss.
The isolated branch is `ooctipus/fpgs-g1-present-ports-20260915`, based on
accepted `b1bad06ae0cd5b86ac07b2f2844ef62770e99075`. Original trees,
Isaac Lab, timestep, substeps, row capacities and eight-sweep allowance stay
unchanged. Root owns every GPU launch. Implementation funding requires root's
review of this concrete mapping; the funded source/AOT checkpoint is 90 minutes
from this card's commit, with no silent extension or mapping grid.

## Complete boundary and decision

Fresh accepted RTX node ownership from the existing strict audit is:

| Owner | Milliseconds per environment step |
| --- | ---: |
| `sparse_factor_contact_triplet18` | 1.246069333 |
| `sparse_factor_parallel_limit_prefix43` | 0.553664000 |
| `sparse_metric_tangent43_s18_c100` | 3.943349333 |
| Complete targeted boundary | 5.743082666 |

Source: `/tmp/fpgs-g1-zero-row-expiry-nodes-paired16k-20260915-01/round_01_baseline/round_01_g1_fpgs_gpu0/strict_zero_expiry_node_audit.json`.
These are sequential targeted owners, not sums of overlapping streams.
The milestone is at least 1.5 ms whole RTX improvement against accepted
b1bad: the replacement, including every added routing, metadata, setup,
decode, masked fallback and synchronization cost, must fit 4.243082666 ms.
Retained collision, current/held dynamics, all-body publication, raw contact
force export, allocator, ordinary row metadata and bias are not free savings.
Any increase in those retained families is charged to the replacement.

The first test is an integrated original-protocol whole comparison after
existing physical smoke, not an isolated port-kernel timing. A mismatch gets
one causal diagnosis before any corrective experiment; no tile/capacity grid.
The corrected expiry's small measured gain is separate and not this baseline.

## Exact representation and admission

Let P contain six current screw-map rows for each PRESENT responding contact
body and one unsigned coordinate selector for each PRESENT distinct limit
DOF. Body mapping uses current S and the original articulation origin, even
when the factor is held. With conventional lower L, F = P L^-T and
C = F F^T = P Hheld^-1 P^T. Native construction uses the equivalent existing
reverse-ordered packed W and its proved eighteen-coordinate support templates.

An individual contact row is a six-component wrench against its body's
port block; the lower/upper sign remains part of each original limit row.
Use h = P delta-v for residuals and accumulate port impulse pi from ACTUALLY
COMMITTED coefficient deltas. Do not decode from final lambda alone: incoming
lambda and delayed friction clearing do not initialize or update kinetic du.
At finish, recover du = F^T pi and apply the existing held-W decode once.
The original ordered scalar-normal/metric-disk transactions, scalar sibling
updates, denominator-only CFM, exact changed exit and maximum eight sweeps
remain unchanged. No new stopping tolerance or impulse-cone aggregation.

Proposed default-off flag: `FEATHER_PGS_SPARSE_PRESENT_PORTS=1`, requiring
the existing metric100 sparse owner. Fast admission is P count at most32
and one responding supported body per contact, with a static other endpoint.
Body6, body13 and body14 are each explicit supported possibilities; body14
is not inferred from the foot-only sample. Self/two-dynamic contacts,
unsupported descriptors and more than32 ports take the complete original
fallback. Up to55 ports are possible with all three bodies and all37 limits;
the bounded fast representation must never silently truncate that case.
No final-zero-impulse or selected-world identity admission is permitted.

## Concrete mapping and storage lifetimes

One32-thread CTA owns one world. All participating lanes complete each warp
rendezvous; no extra stream or multi-warp idle phase is introduced.

| Explicit shared storage | Bytes | Lifetime |
| --- | ---: | --- |
| F,32 ports by18 support values | 2304 | Formation through final decode |
| C, lower triangle of32 by32 | 2112 | Formation and solve; then alias as du43 |
| Lambda100 and lazy cross100 | 800 | Solve |
| Port h32 and impulse32 | 256 | Setup/solve/decode |
| Port keys32 | 128 | Formation through decode |
| Four metric-ready words | 16 | Solve |
| Three body bases, port count and error | 20 | Invocation |
| Total explicit payload | 5636 | Before compiler/runtime overhead |

No F32x43, full C32x32, private Z100x18 or private wrench-row panel is kept.
Existing global diagonal, incident and RHS outputs retain their original
meaning; they are not additionally cached100 times in shared memory.
Impulse storage first holds Pcurrent vhat during incident setup, then is
cleared for accumulated committed impulses. Dead C storage becomes du43 only
after all constraint visits finish; F is never recomputed for the decode.

F body construction has lanes own the existing support coordinates. A single
held-W load feeds six simultaneously accumulated current screw components;
present limits copy unsigned W columns. Static template intersection/position
tables are common topology metadata, not a world-scaled matrix. C uses
lane-strided triangular pairs and only their exact support intersections.
There is no sparse factor application per contact row.

Lanes then own independent row setup: charge a six-dimensional self quadratic
(approximately42 products per contact direction), Cii for a limit, the
six-term current incident, original restitution and all validation. On first
positive-radius metric use, three bilinear cross terms cost up to126 products;
the original root solve is unchanged and is not free. A contact residual has
six terms and three shuffle reductions, while a limit reads its signed port
velocity. For each committed combined contact delta, broadcast its six wrench
components and let each lane update one port using six shared C coefficients.
Decode adds at most P times18 factor products, plus the original434-W decode.

The global geometry output below prevents repeated anchor reconstruction.
Extend the ALREADY REQUIRED `prepare_world_contact_rows` owner using its
existing normal, tangents, world points and both anchor choices, emitting
only six signed wrench coefficients per reserved row. This is geometry,
not J, Z or response coefficients. At16K and100 rows its allocation is
39,321,600 bytes. The existing support array can carry raw row keys and then
fast port descriptors; one world-mode array adds65,536 bytes. F and C never
become global intermediates. The original118 MB-class Z allocation remains
available for fallback: this experiment claims no allocation retirement.
Geometry emitted for subsequently rejected worlds is explicitly wasted work
and included in the measured boundary.

Register planning is80--96 per thread, not a promise. Six body accumulators
and the original metric-root temporaries are the main pressure; AOT must
report actual spills, stack, registers and shared allocation before GPU use.
For illustration,5636 payload plus inherited128B scratch rounds to5888B at
256B allocation granularity; with a1KiB reservation that is6912B per CTA.
An available100/128KiB carveout would bound shared residency to14/18 warps,
before registers and other limits. This is a conditional arithmetic ceiling,
not achieved occupancy. The documented register file is64K entries and the
block ceiling32; actual carveout/allocation must be checked rather than
assuming the nominal SM capacity is usable shared memory. See the
[NVIDIA Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/).

## Dispatch before any old response work

1. A metadata-only stable limit prefix records original candidate order,
   row keys, counts and phase bounds, without W, Z, diagonal or incident work.
2. Retain the original allocator and row reservations. The augmented metadata
   owner writes the six geometry values from its existing calculations.
   Finalize counts and construct the ordinary global RHS as before.
3. The port kernel performs full structural admission before public velocity
   or impulse writes. Unsupported worlds remain mode0. Admitted worlds form
   current F/C, setup rows/restitution, solve and publish original outputs.
4. Complete mode0 fallback follows: original limit-response emission from
   recorded candidates, original contact-Z production, restitution and
   original metric solve. Each guard tests mode BEFORE W/Z work. Do not rerun
   allocation, renumber rows or repeat old responses on fast worlds.

This serial fallback tail, including guard-only launches, is fully charged.
Original fatal status/overflow handling must still prevent invalid public
finish; a numerical failure cannot silently turn into successful fast output.
Row metadata and RHS retain original semantics. Normal and tangent anchors
can differ. Raw friction-neighbor rules, material mixing, restitution trigger,
per-contact lambda-to-force mapping and all43-DOF/44-body publication survive.
F/C and mode are current per invocation: no cross-substep geometry, limit,
reset, notification, state-bank or graph-lifetime cache is assumed.

## Existing CPU evidence and the serious counter-cost

Reuse the original SHA-pinned sixteen current1600/held1601 payloads and their
existing metric sweep helper, not a new dataset or benchmark framework.
The helper is `/tmp/fpgs-g1-ordered-secant-cpu-HpvXKyo6/study.py`; the selected
payload loader is `tools/fpgs_bench/test_sparse_contact_block.py:saved_records`.
An inline read-only check found normalized J span defects at most2.57e-8:
hard RTX7410 has17 ports, hard GB4625 has22, and loaded8192 has21--23.
The selected cases contain no torso/self-contact coverage and are not a
population census. Limits-only and two-contact cases must remain represented
in cost evidence; their setup can cost more than original response formation.

Across these sixteen original finite-eight traces, residual scalar products
fall23564 to8371, but velocity-update products grow5212 to18092, plus up to
2556 combined-wrench products. Aggregate arithmetic is approximately28776
versus29019 BEFORE formation, self quadratics, cross setup and final decode.
The hypothesis therefore is NOT a60-percent reduction in all solve work.
Its differentiator is removing global row responses and replacing repeated
limit/contact warp reductions with a small shared port operator. The same
traces have643 limit and1288 contact residual dots: an ideal reduction count
is9655 to3864 before extra wrench broadcasts and setup. Root probes remain194.
These are logical selected-sample counts, not native population exposure,
bandwidth measurements, timing predictions or guarantees of the milestone.

Closed neighbors remain closed: sparse packets retired global Z but rebuilt
all private rows, increasing complete rows/GS from9.104 to11.837 ms RTX;
lazy compliance retained Z and added a lazy row Gram; spatial dynamics used
costly full tree actions. None implemented this present-limit/body operator,
but their serial-setup/shared-lifetime losses are genuine mapping risks here.

## Minimum qualification and stop condition

Regression first, reuse existing physical helpers and tolerances: current and
held operators including7410/4625; scalar/metric/delayed-friction and incoming
lambda; limit signs/deduplication; body14, static-side reversal and self-contact
fallback; more-than32 ports; original cap/error paths; actual odd-five-world
mixed fast/fallback reset, current/held and graph reentry. Preserve arbitrary
current external/control force behavior and original public-state outputs.
Use existing AOT targets and original checked paired whole/node protocol;
no new runner, census or proof framework. Freeze all runtime/source pins before
root's GPU runs. A first loss gets causal owner attribution, not a new grid.

Status at this commit: pre-code plan only. No native timing, new physical
qualification, promotion or all-task four-times result is claimed.

## First complete native readiness

The original driver-query results, already recorded in
`FRANKA_KINETIC_PAIRED16_20260915.md:321--330`, take precedence over generic
guide limits for these cards: RTX reports24 CTAs and102400 shared bytes per
SM; GB reports32 CTAs and233472 shared bytes. Both report65536 registers.
These are static limits, not observed occupancy. No new device query was run.

Initial source/AOT work is complete before the06:15 checkpoint. The first
source-matched offline report is
`/tmp/fpgs-g1-present-ports-offline-5jml9ov6/offline01/report.json`.
All six actual port/fallback entries compile on SM120/SM100. Port owner uses
80/78 registers,5764 shared bytes including inherited128-byte scratch,
zero stack and zero spills. Masked original contact is56 registers/128 bytes;
masked original metric is72 registers/1116 bytes. Root's review requested
placing the new contact mode guard after the original capacity/valid check;
that ordering is corrected in the final source. It still precedes all contact
geometry and held-W reads.

Final offline02 compiles all14 actual entries on SM120/SM100, including the
augmented metadata, original packet prefix and both remaining fallback owners:
`/tmp/fpgs-g1-present-ports-offline-5jml9ov6/offline02/report.json`, SHA256
`4286c57e835067f92096c31c7602e8625dc25f2e8cbdb5e848a8bad5cf23fa11`.
Port/fallback resources above are unchanged. Metadata costs106/96 registers
and1024 shared bytes; prefix48/40 registers and128 bytes; fallback limits48
registers/1024 bytes; fallback restitution30/32 registers/1024 bytes.
All14 entries have zero stack and spills. These static resources do not
establish achieved occupancy or a performance gain.

The new CPU regression first failed on missing explicit default-off ownership.
All10 CPU controls pass in3.481 seconds (session69442), including two new
controls: API/admission and metadata-only stable limit keys plus current
signed wrench geometry, with W/Z/diagonal/incident poisoned. An intervening
CPU run caught an exact decorator-cloning mismatch in the restitution wrapper;
the corrected wrapper retains the original disable-backward setting and law.
Existing numerical helper tolerances are unchanged. Root and the independent
test/source reviewer found no concrete F/C, committed-delta, sign or alias
lifetime blocker. Native physical correctness and whole cost remain unmeasured
at this readiness checkpoint; CPU/AOT is not a CUDA qualification claim.

## First native controls and inherited saved-payload criterion

Root's first nine-selector native batch on clean runtime `d03063f1` completed
with eight selectors passing on each card, including synthetic lifecycle and
actual five-world mixed graph checks. The saved-sixteen selector stopped seven
subcases at a newly added per-incident allclose, before their final physical
gates; the final tested-count assertion consequently also failed. Both parents
63538/55610 reaped1 with the aggregate Python source unchanged. Logs remain at
`/tmp/fpgs-g1-present-ports-native-5Mtt1tKj/gpu{0,1}.log`.

Root then ran accepted `b1bad06a` on exactly the same sixteen inputs. It failed
that new incident criterion on the same seven cases and indices, with nearly
identical 4e-6--1.31e-5 errors at near-cancelled entries. The control reaped0
(parent78386); evidence is `original_incident_gpu0.log` in the same directory.
For GB-fixture step1601/world4625, the original incident is0.064740613 versus
physical FP64 reconstruction0.064727635; candidate is0.064740717. This does not
establish a candidate runtime or geometry defect.

The test-only correction restores the inherited saved-payload component
diagnostic contract: log incident reconstruction differences and use the
actual FP32 incident in the same-input eight-sweep reference. Tight incident
checks remain on small synthetic cases. Diagonal and restitution checks,
momentum2e-6, cone3e-5 and final reference rtol3e-4/atol3e-5 are unchanged.
All production bytes remain those of `d03063f1`. Saved-sixteen final physical
qualification remains pending until the corrected selector completes.

The second native batch on clean test tip `3fcd90e1` completed twelve of the
sixteen saved cases through their final gates, but four hard cases stopped at
the newly added diagonal per-component check. Parents11299/57512 reaped1;
logs are `native02_gpu{0,1}.log` in the same native directory. Original accepted
`b1bad06a` also fails this criterion on the same four physical inputs, with
2.65e-5--3.50e-5 absolute errors versus candidate2.69e-5--3.48e-5. Root's control
parent38918 reaped0; `original_diagonal_gpu0.log` preserves it. Atomic contact
allocation can change row indices, so this comparison is by physical input
and coefficient values, not an assumed equal raw row order.

The second test-only correction records saved diagonal differences through
the same inherited component diagnostic. Synthetic diagonal checks and all
final velocity/impulse, cone and momentum tolerances remain unchanged. Runtime
still exactly matches `d03063f1`; four hard final physical gates remain open
until this corrected test runs.

## First whole cost screen: loss, qualification still open

Root ran the original paired whole protocol on clean `3fcd90e1` before spending
more sequential runs on new component criteria. Parent34672 reaped0; artifacts
are `/tmp/fpgs-g1-present-ports-whole-paired16k-20260915-01`. RTX graph time was
15.607145525 to18.255871400 ms, a2.648725875 ms loss. GB was20.457185200
to24.529586400 ms, a4.072401200 ms loss. The >=1.5 ms RTX milestone is missed.
This is an unqualified cost screen, not a promotion or completed physical
qualification. Matched original node attribution is pending; source work and
static resource counts alone do not locate the measured loss internally.
