# First-hit physical-delta-velocity: causal cost closure

## Decision

The corrected physical mapping passes the unchanged sampled physical gates,
but this implementation is NOT a performance candidate. The completed actual
live40-step screen increases whole-physics time9.61% RTX /9.10% GB.
Do not tune launches/register limits or infer a remaining10% route from the
old high never-hit fraction. The producer still costs almost as much as the
already-pruned original; the physical-delta-velocity consumer adds1.44/1.63ms
of disjoint solve time in the source-identical node trace.

Both source-identical parents are reaped, all four arms exit0, source/idle
guards pass, and explicit post-reap compute queries were empty. No additional
GPU run or runtime correction is authorized by this note.

## Disjoint node accounting

Three actual traced environment steps,12physics graph roots and1920nodes
per arm/card; every kernel, memcpy and memset has verified process/launch
correlation. No auxiliary graph is discarded. Units are ms/environment step,
diagnostic only. Rows below sum exactly to complete node span.

| Disjoint contribution | RTX original | RTX first-hit | GB original | GB first-hit |
|---|---:|---:|---:|---:|
| collision | 1.877761 | 1.858552 | 2.144607 | 2.116778 |
| force_sensors | 0.551595 | 0.554166 | 0.611024 | 0.611194 |
| joined_solve | 1.043361 | 2.484227 | 0.945280 | 2.577408 |
| memory | 0.283808 | 0.284576 | 0.326352 | 0.325971 |
| mf_services | 0.444288 | 0.459755 | 0.497353 | 0.491898 |
| qualification_materialization | 0.376971 | 0.330773 | 0.368360 | 0.389619 |
| raw_allocation_zero | 1.399030 | 1.411361 | 1.352476 | 1.344369 |
| rows_and_response | 1.991467 | 1.958648 | 2.467497 | 2.232179 |
| state_and_held | 3.194656 | 3.202678 | 2.818601 | 2.813504 |
| cross_family_overlap_once | 0.000000 | 0.000000 | 0.010507 | 0.010069 |
| no_graph_work_gap | 0.258370 | 0.258604 | 0.280161 | 0.270489 |
| TOTAL node span |11.421307|12.803341|11.822218|13.183477|

Added span is1.382034/1.361260ms here. Joined solve increases
1.440866/1.632128ms; rows/response saves only0.032820/0.235318ms.
State/held changes+0.008023/-0.005097ms, raw/allocation/ZERO
+0.012332/-0.008107ms, force+sensors+0.002571/+0.000170ms.
No-work gaps change+0.000234/-0.009672ms; overlap is retained once.
This is not a host scheduling, new launch count, repair, or lost-join cause.

## Exact changed kernels and representation

| Kernel sum,8calls/environment step | RTX old | RTX new | GB old | GB new |
|---|---:|---:|---:|---:|
| contact_triplet → first_hit_packets |1.701793|1.659693|2.172373|1.940426|
| row_prefix → first_hit_prefix |0.125035|0.129931|0.132661|0.128971|
| offset_eight → first_hit_eight |0.992790|2.442478|0.897259|2.529291|
| retained compact generic |0.787702|0.261035|0.379712|0.530059|

Contact owners are serial with the other physics stages: their measured
sum is also their union, with no material overlap to credit. The stage's
exclusive deltas above include unchanged arm-map/validate and prefix.
Exact contact exclusive times are1.701793→1.659693ms RTX and
2.171605→1.939903ms GB. Offset/first-hit exclusive time alone grows
0.255659→2.223192ms RTX and0.565568→2.047349ms GB; these exclude the
overlapped generic interval and must not replace the joined union.
Generic and offset run concurrently: the generic duration changes must NOT
be added to offset changes as an independent saving/cost. Their exact union
and exclusive contribution is the joined-solve row above.

Actual resources, unchanged launch dimensions:
- Contact:4096CTAs×32threads,80→95registers,1280B shared,0local memory.
- Offset/first-hit:8192CTAs×64threads,60→102registers RTX and56→102GB,
  shared5760→7296B,0local memory.
- Retained generic:16384CTAs×32threads,92registers/4340B shared,0local.

These resource changes are measured; occupancy/compute/memory bottleneck
percentages are not established by this trace. No spill explanation applies.

The original triplet was NOT performing a dense29×29 matrix operation per
row. It already formed three directions together, gated joint support,
used common-arm relative moments/C-map cancellation, and restricted fingers
to their four-DOF branches. MF0 stored only Z29, not a full physicalJ panel.
The new packet skips those already-pruned J/Z/diagonal calculations but still
traverses the same raw records, runs the same lane0 rounded geometry and
material/prelude, and emits all three rows' current metadata/RHS. It writes
16floats/direction into the oldJ panel:48packet floats/contact versus87oldZ
floats/contact, not zero coefficient-like traffic. It also adds endpoint
moment construction. Hence the static omitted work was substantially
smaller than an assumed dense-J/Y producer, and its measured time deletion
is only0.042100ms RTX /0.231947ms GB at the contact-kernel level.

The consumer changes from factor-coordinate delta velocity, cheap dot(Z,z)
and Z-based updates, to physical delta velocity. Prefix rows are still eager
Z/diagonal; action(i) converts their response into physicalY on entry.
First-demand contact action reconstructs J, applies sparse T then T-transpose,
caches J/Y, and uses J·delta-v thereafter. Never-demanded normal residuals
gather endpoint delta twists, whose current-axis prefix scan is rebuilt only
when motion_dirty AND a packet residual is requested. It ALREADY coalesces
intervening changed cursors; proposing another dirty-bit/delay scan would
repeat existing work. The aggregate physical consumer costs2.46×/2.82× the
old offset kernel. This trace attributes the loss to that owner, but does
not apportion its internal time among first-demand action, prefix conversion,
scan/gathers, branch footprint or occupancy. No unsupported internal-cycle
percentage is claimed.

Model census remains comparable, not numerically identical trajectories:
about240–242kraw contacts,135–145kdense rows,3.7–4.0kactive worlds and55–76MF
worlds at checked endpoints. Exact before/after counts are in evidence.json.
A few-percent workload change is retained in the report; no same-seed
trajectory-identity or original-eight proof for every evolving16Kworld is claimed.

## Complete allowance and next decision

Authoritative40-step discovery, not these perturbed3-step numbers:
RTX11.299988→12.386436ms, GB11.666766→12.728927ms. Matched10% limits are
10.169989/10.500089ms, so current successor needs2.216446/2.228838ms recovery.

Even hypothetically restoring the entire original solve cost while retaining
this packet producer yields only the measured0.033/0.235ms row-stage saving,
far below1.130/1.167ms needed. Within this retained-producer scope, even a
free new solve almost exhausts the10% envelope: current packet-row stage is
1.959/2.232ms, versus original rows+solve minus10% of diagnostic whole span
1.893/2.231ms. This is a measured-scope allowance, not a universal hardware
lower bound; it rules out a consumer-only correction as a costed10% plan.

A genuinely larger next candidate would also need to remove current packet/
geometry/three-row metadata production or merge its current readers with
ZERO/allocation, while replacing the physical consumer. That is a different
complete ownership boundary, not a funded claim or implementation here.
No single additional correction has a demonstrated complete allowance.
Close this mapping; preserve the corrected numerical/cache findings for reuse.

## Evidence and source pins

- Node parent4819e3cb5fe2f26e6d5314ca26539bba50867351829093954d2160331973efd4:
  /tmp/fpgs-kuka-first-hit-live-nodes-paired16k-20260913-01/manifest.json.
- Cost parentd2357eaedc2563a0b06be327e99dc66cb4a040674cdfd067c185410560fa6c83:
  /tmp/fpgs-kuka-first-hit-live-discovery-paired16k-20260913-01/manifest.json.
- Candidate e96aaab35bd5b61578bc8a77dc8633f7bc80c68d, comparatorad42,
  benchmarka3a5, fixed Lab backend53ee; all unchanged during parents.
- Native first-hit9988059a713a8e46b7fe503c600ebe9a124a9e52afd92c11e0096a6d87179bbd.
- Original rowsae7085fe67416c1b759178a3a985883e1063943096734dba64006f2339294d31.
- Original triplet48c80721f9e7528256e2c56e66ee7260102e4605f30d2c7a975d0154c273452a.
- Readerf715ab10479e97c4215805e5cbd2056fbe9bb8216f1605f7819c1f9f8c952203,
  exact ce09 recovery after only three kernel names;3CPUtests PASS.
- baseline_gpu0.json f5625013570ea8ff018b18f65adbdfddd6751ec19c4bfa315aa0522f13cfac54
- baseline_gpu1.json127b838c7ef6bb208ae011154165e9a5eb0e1d578c4e8989730799f9c2e358de
- candidate_gpu0.jsonacc4ae0602ecd4af56a2cf7d8dbf18a6b309fe3478a9503373f8545e1988fd45
- candidate_gpu1.jsonb9e674f6fee64a6a55a521d8df107d1641677a16bb6420afdb240fbbcf7360b8

All original SQLite/capture/check hashes are embedded in the four audits and
the aggregate evidence.json. The passed512 physical gate, demanded-row census
and unchanged bounds are recorded in the discovery RESULTS.md. Numerical
correction success and performance rejection are separate conclusions.
