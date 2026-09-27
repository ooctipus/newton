# Allegro keyed kinetic-row ownership correction

Pre-code card, 2026-09-14 21:20 UTC. Native/source checkpoint 21:50 UTC.
Base `55a0259e517e8cb9f19961df85986a9d29bead54` remains preserved in its
original worktree. This is one cause-specific correction, not a new mapping
or parameter sweep. Default-off `FEATHER_PGS_ALLEGRO_KINETIC_ROWS=1` remains
the opt-in; direction cells remain independently default-off.

## Measured cause and complete budget

Original cells `92507273` versus global kinetic-row `55a025` whole discovery:
RTX 16.499407 -> 18.444526 ms; GB300 16.421763 -> 19.581437 ms.
The strict matched node audit reports RTX row/solve exclusive boundary
9.034745 -> 10.739260 ms. Candidate components are contact producer 3.422595,
two fallback launches 0.502646, maps 0.303339, prefix 0.459509, retained rows
0.917408 and solve 5.132430 ms. GB boundary 9.173352 -> 12.567677 ms;
its solve grows 5.897736 -> 6.826553 ms. These are observed owner timings,
not selected-world percentages or summed overlapping kernel claims.

The contact producer serializes three rows of metadata in one warp lane,
writes compact global coefficients and support codes, then the parallel
consumer decodes those codes separately for every one of 22 coordinates.
Two separate fallback launches traverse their launch domains even when the
sampled worlds are all eligible. Geometry was already shared across the
three contact directions; repeated-direction geometry is not the old cause.

Target: original boundary 9.034745 -> <=7.034745 ms RTX, at least 2 ms whole
saving if unrelated work remains flat. This requires 3.704515 ms improvement
over the failed representation. Deleting the contact and two fallback owners
exposes 3.925241 ms, leaving only 0.220726 ms for replacement work unless
prefix/ingestion also improves. It is a deliberately falsifiable, unproven
budget, not an assumption that moved arithmetic becomes free.

Deeper root audit, 21:28 UTC: the apparent RTX solve-family reduction is
intermittent serial fallback, not a whitening gain. Actual parallel tiers
4.906914 -> 4.965593 ms RTX and 5.446110 -> 6.096680 ms GB. Original RTX has
two >128 fallback spikes (881.984/856.928 us), candidate none in that short
window; GB has one original versus two candidate spikes. Boundary snapshots
miss these transient worlds. Do not credit the correction with a prior
0.514 ms parallel improvement or call fallback universally unused.

## Complete replacement

- Reserve signed coordinate prefix keys and scatter raw contact ID/direction
  keys into the existing row slots. Do not allocate global contact Z, compact
  support codes, or separate fallback coordinate/sign arrays.
- In the actual existing parallel row lane, transform body-local anchors
  using CURRENT frames on EVERY Newton substep; compute prescribed target,
  material mixing, bias and restitution at the original seams. Contract
  current maps made with HELD L directly into register `Jr[22]` and publish
  the existing full float4 shared panel once. Retain the original 24-sweep
  parallel law, prescaling and final inverse action.
- One world-centric fallback pass reads the same typed keys, publishes full
  current physical J and metadata only when >128 rows or MF contacts require
  the original 12-sweep route. No active queue and no raw-contact scan for
  ordinary worlds. Retain the original raw allocator and MF producer.
- Keep map construction, factor/predictor/publication and all public physical
  settings. Retire old global contact-Z production and both old fallback
  kernels, not merely add a second representation beside them.

## Gates and accounting

Reuse the four pinned current/held Allegro payloads, existing matched-input
native tests and full actual Solver.step lifecycle including poisoned retired
J, >128/MF fallback, reset and graph grow/shrink. Numerical/physical tolerances
remain the existing ones; no bit-identity gate or extra iterations. CPU checks
and early SM120/SM100 offline compilation precede paired GPU authorization.
Report register/shared/local storage explicitly. All key emission, current
geometry, metadata, map, fallback, setup and solve costs belong to the gate.
One original guarded whole A/B against cells925 decides the hypothesis;
do not promote recovery versus the failed55 candidate as a gain.

Source/evidence: `ALLEGRO_KINETIC_ROWS_20260914.md`, original whole
`/tmp/fpgs-allegro-kinetic-rows-paired16k-20260914-01`, node captures
`/tmp/fpgs-allegro-kinetic-rows-nodes-paired16k-20260914-01`, strict reader
`/tmp/fpgs-allegro-kinetic-nodes-checked-W09prTAI/audit.py`.

## CPU/source readiness, 21:33 UTC

All eight existing CPU tests pass, including the new regression-first keyed
owner contract, four original current/held payloads, typed-key identities,
full current physical fallback J and existing actual constructor guards.
Fallback physical J defects are 2.57e-8 to 2.94e-8. Existing four CUDA
selectors remain pending root authorization; no GPU work was performed here.

First offline attempt exposed only a native CPU/CUDA cast-name portability
error, preserved as `offline01`; explicit casts and portable comparisons
correct it. Final `offline03` compiles all 16 SM120/SM100 records successfully:
`/tmp/fpgs-allegro-keyed-kinetic-offline-ZAcibf/offline03/report.json`.
RTX solve32/64/96/128 use 87/82/98/98 registers; GB 83/83/98/98. Shared memory
is unchanged 5868/10024/14172/18232 bytes. All stacks and spills are zero.
Maps/prefix/keys/fallback use 48/40/24/85 registers on both architectures.
This is resource evidence, not an occupancy or speed claim.

Final metadata values pass directly from the row geometry scope to shared
solver fields; canonical publication is retained for existing diagnostics and
contact exports, not reloaded as the private solve's source. Contact type is
derived from its key before ingestion, avoiding stale canonical reads. Final
eight CPU tests pass again (12.121 s including compilation), full pre-commit
passes, and source-matched offline03 passes. The earlier offline02 remains
preserved; only the final03 resources above describe the frozen candidate.

Actual data contract is `owner.keyed_rows=True`, `data.rowkeys[world,M]`,
negative `~(2*dof+side)` prefix tokens and nonnegative `4*raw_id+direction`
contact tokens. There is no global coefficient/support encoding allocation
or separate fallback dof/sign array. New solve keys start
`pgs_solve_kinetic_keyed_parallel_`; key emission is `scatter_contact_keys`,
fallback is `get_fallback_kernel__locals__allegro_kinetic_keyed_fallback`.

## Completed whole qualification, 21:55 UTC

Frozen runtime `7fca2fdf5ef090e1e28fc9e327ce9f4de5e879df` passes all four
existing CUDA selectors on both cards (18.745s RTX /18.764s GB including
module loading). Current/held momentum defects are3.39e-8--3.90e-8; maximum
velocity differences are3.81e-6--6.20e-6. Complete production reset, held
factor/current geometry, crowded fallback and seeded graph grow/shrink pass.
These checks retain their original physical tolerances and iteration budgets.

The guarded16K200/40/40 whole comparison against925 cells, enabled on BOTH
arms, measures RTX16.340675->15.201010ms (1.074973x) and
GB16.517619->16.719848ms (0.987905x). Wall times are24.023163->23.410114ms
RTX and24.794111->24.805209ms GB. The correction reverses55's regression,
but misses its planned2ms whole saving. It is a real RTX discovery gain, not
a cross-GPU incremental win. Artifact:
`/tmp/fpgs-allegro-keyed-kinetic-rows-paired16k-20260914-02`.
First01 stopped before any child because its required local launch card was
missing; that failed manifest remains preserved.

One original three-step node capture per arm/card, with the same strict
process/correlation/all-memory reader, confirms12 physics roots, zero auxiliary
roots,2064->1848 graph nodes and no unproven records. The complete row/solve
exclusive boundary is8.449931->7.549881ms RTX and8.969132->9.340897ms GB.
The original boundary differs from the earlier9.034745ms sample because of
transient fallback; do not subtract unrelated samples as matched savings.
Retained-row removal pays, but the actual parallel tiers grow4.889749->
5.852930ms RTX and5.508978->7.396244ms GB. All four tiers slow;64/96 dominate.
GB has one821.568us serial fallback call, raising exposed fallback0.081034->
0.351402ms; RTX fallback is0.084971->0.083808ms. Thus fallback does NOT explain
most of GB's solve growth. Actual executed sweeps were not recorded, so the
split between solve-local geometry cost and rounding-sensitive early stopping
is unresolved. No generic register-pressure explanation is supported.
Artifact `/tmp/fpgs-allegro-keyed-kinetic-rows-nodes-paired16k-20260914-01`,
with `strict_keyed_node_audit.json` in each capture directory; unchanged strict
reader extended only with exact owner names at
`/tmp/fpgs-allegro-keyed-nodes-checked-NV3Dj1lZ/audit.py`.

### Fresh corrected MJWarp reference: RTX target crossed

Three alternating FPGS/MJ, MJ/FPGS, FPGS/MJ rounds each run the two GPUs
simultaneously. Both backends use clean7fca and fixed Lab53ee. Original task
dt/substeps/decimation and maximum iterations remain unchanged; original
capacity, source, idle, finite-state and corrected-line-search checks pass.
All12 children return zero. Values below are medians, ms per environment step.

|GPU|FPGS physics|corrected MJ physics|physics ratio|FPGS wall|MJ wall|wall ratio|
|---|---:|---:|---:|---:|---:|---:|
|RTX PRO6000|15.103509|63.536818|4.206759x|23.131495|71.449856|3.088856x|
|GB300|16.664395|82.023917|4.922106x|24.920978|91.547777|3.673523x|

RTX FPGS samples are15.015833,15.331838,15.103509ms; MJ samples63.431669,
63.632510,63.536818ms. Every same-round RTX ratio exceeds4x. GB FPGS samples
are16.585155,16.664395,16.702456ms; MJ samples82.023917,81.965890,
94.850148ms. The last GB MJ sample is a large outlier and is retained, not
silently discarded. The reported GB median is82.023917ms. Environment wall
times exclude policy optimization: these are not full RL-training speedups.
Allegro meets the RTX physics target in this repeated fixed recipe; the
4x-across-all-representative-tasks target remains unmet.

Allegro's original MJ configuration defaults `use_mujoco_contacts=True`:
it has native MuJoCo contacts and NO Newton collision pipeline. Newton direction
cells therefore do not apply to it. Do not switch MJ to external contacts or
claim that disabling a relevant shared improvement funded the comparison.
MJ retains calibrated njmax112/nconmax22; FPGS retains raw286720,
broad524288,dense192/MF64/prop192 and original12/24 iterations.

Artifact and exact reproducible invocation:
`/tmp/fpgs-allegro-keyed-corrected-backends-paired16k-20260914-01` and
`/tmp/fpgs-allegro-keyed-backends-akOBZj8M/LAUNCH.md`.
The minimal adapter there has SHA-256
`e82a61022ee44ae92b57d0b336b862f8a3858b0a73f9c93503f948bc00f59f7a`;
its launch card SHA-256 is
`e19978bf30b9ea3fde847cf9ba0107f9cf912039344f61e50ddd9f3805384fba`.
It reuses original compare_backends798 and fixed-import adapterc062, routing
only FPGS to the existing keyed observer. MJ retains the original checked
native-contact/line-search path. Manifest driver hashes include every source
selection prerequisite. These retained `/tmp` prerequisites are required for
the exact local reproduction; the report does not claim a self-contained
download. A later report-only revision must be recorded separately from7fca.
