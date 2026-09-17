# G1 row-owned local formation — 2026-09-17

Status: CLOSED without an accepted gain. Default off; the accepted corrected-limit
baseline is unchanged. The one physical-width correction also lost. Isaac Lab,
runtime physics budgets and calibrated capacities are unchanged.

## Hypothesis and complete-cost gate

The compact register-residual experiment at `f489a7ce` remained slower than the
accepted corrected-limit owner: 12.5091 → 12.8689 ms RTX and 14.4305 → 14.8613 ms
GB300. This experiment must beat the accepted owner, not recover that loss and
call the recovery progress.

The candidate forms each eligible row in its owning lane, with up to 32 rows
concurrently in the existing 32 × 44 shared tile. Physical J is converted to WJ
in place, visiting output nodes in descending order and original input nodes in
ascending order. The checked lower-triangular support makes this safe without a
second tile. The existing corrected prefix, spectral contacts, iteration budget,
applied-impulse accounting and physical decode are retained.

Current RTX contact-triplet, value-emitting limit-prefix and restitution nodes
cost 1.437483, 0.557088 and 0.072464 ms respectively. Their 2.067035 ms sum is an
upper bound on displaced work, not a promised saving: large/rejected worlds
still need it. Allocation (0.156459 ms) and metadata (0.315643 ms) remain. The
compact register solve already owes 0.352563 ms versus the accepted solve. A
1 ms whole saving therefore leaves at most 0.714472 ms for new formation,
key production, filtered fallback and dispatch; assuming 90% of the displaced
time were eligible would reduce that allowance to about 0.508 ms. Row-count
eligibility has not established time eligibility.

This differs from the closed September 14 serial packet mapping: rows are formed
concurrently, not contact-by-contact by one warp. It is also not a memory-only
claim: geometry duplication and producer reductions may disappear, while sparse
W/index loads can increase across directions. No hardware-counter or roofline
claim is made.

## Ownership and fallback

`FEATHER_PGS_SPARSE_REGISTER_PACKETS=1` requires the corrected register-residual
owner and excludes the older body-basis row mode. Original `build_rows` source is
unchanged for all other modes; the candidate binds a separate method.

1. Original allocation and row metadata retain their order and calibrated
   capacity. Prefix/contact producers publish current row keys instead of Z.
2. The small solve rewrites routing every call, forms current geometry against
   the current/held W, and applies the original restitution trigger locally.
3. Success publishes canonical scalar row fields, impulses and physical velocity.
   Global Z is not produced for successful worlds. No cross-call response cache
   or global Gram is introduced.
4. Failure must leave keys, RHS, impulses and velocity untouched. Filtered original
   prefix/contact materialization and restitution then precede the unchanged
   corrected-limit fallback solve, on the same stream.

Global Z/incident allocations remain large enough for any world to fall back.
This retires production/traffic, not capacity. Private accepted-world Z is stale
and must not be used as a physical test oracle. Public contact-force export uses
canonical row identities and impulses, not that private Z.

## Qualification and timing

The completed screen includes every new launch and fallback cost and compares
against the accepted corrected-limit owner. Numerical and physical behavior,
not bit identity, are the gate. There was no MMA or layout grid; after the first
loss, one separately preserved width43 cause-falsification was authorized.

The pre-implementation factory/owner tests failed with the missing new native
module, as expected. Existing register-residual tests remain distinct from the
new packet tests. The native controls below pass, but complete measurement loses.

## Native checkpoint and test-oracle diagnosis

Runtime SHA256 `7ea45204f07a845ea767517719310963aec65f47010f054603121e1af59f5826`;
filtered services `871d2b41c5debc77b4535e4cb565fb446b079a0f8b2da74134fe8bae7ce1222f`.
Independent source review checks geometry, sorted-support triangular in-place
formation, transactional publication and fallback. Existing 13 CPU controls and
four new capture/dispatch controls pass. Initial current/held geometry, anchors
and restitution native selector passes on both cards.

Hidden-device AOT `/tmp/fpgs-register-packets-offline-boa5nu9z/offline01` reports
118 registers RTX /112 SM103, 5760 shared bytes and zero stack/spills. Executable
text is 82176/82304 bytes, versus compact 62976/62848. Resources increased; the
unchanged tile size is not an unchanged-occupancy or speed prediction.

Full native02 exposed a test-oracle mismatch in two saved states (RTX-source
world7410 at1600/1601), on both cards. A newly added component check substituted
independently reconstructed FP64 J for the inherited rounded production Z but
kept the old coefficient-level tolerance. Candidate and original control have
EXACTLY the same failing differences, 4.0260e-5 /3.7645e-5, and tolerance ratios
5.6129/2.6684. This is not a candidate-only physics discrepancy. Both cards'
sixteen-case diagnostic outputs agree: maximum inherited rounded-Z tolerance
ratio is0.073022 and maximum independent H backward defect is4.2574e-8, below
the existing2e-6 gate.

Test `75c98556fb27fd1e9d35954613720e604948dab9f3ac8f91deab86675c48370f`
restores the original coefficient check using original Z saved BEFORE candidate
sentinel poisoning, with unchanged rtol3e-5/atol3e-6. Independent physical J/H,
cone, metadata, scalar fields and original-output controls remain. FP64-J
component diagnostics and failed02/control logs are preserved; no tolerance or
runtime change was used to resolve this mismatch. Native03 passes all six groups
on both cards before whole timing (3.968s RTX /3.948s GB including load/tests,
not performance). Logs share prefix
`/tmp/fpgs-g1-row-owned-packets-`, with `native-{rtx,gb}-20260917-02.log`,
`momentum-control-{rtx,gb}-20260917-01.log`, and final native03.

These are finite-budget current/held, restitution, incoming/delayed-friction,
32/33/100 routing, numerical rollback and graph-transition controls, including
the existing saved16 J/H fixtures. They are not a converged-manifold reference
or a new full-population physical qualification. The candidate and original
controls both exhibit the preserved FP64-J component discrepancy; the inherited
rounded-Z check and independent H check answer different numerical questions.

## Complete whole-path result

Measured clean candidate `56608c664fb1f10161743b9b09865f29d5f9fb26` versus
corrected-limit `c6ab26f9687f64fd2acfe66f5328f16be1820692`, using fixed Lab
`53ee6b44c2334341305dbdf385a3916c6b140799`. Each card has one checked paired
round: 16,384 worlds, seed0, 200 warmup /40 unprofiled wall /40 graph-profile
steps. The .005 environment simulation step, decimation4, two .0025 solver
substeps and maximum8 passes remain unchanged. Capacities remain100 dense rows,
294,912 raw contacts,49,152 broad pairs and1,769,472 triangle pairs.

| Metric, ms per environment step | RTX baseline | RTX packets | GB baseline | GB packets |
| --- | ---: | ---: | ---: | ---: |
| Profiled whole physics graph | 12.509309075 | 13.346259925 | 14.415478075 | 15.731324800 |
| Unprofiled environment wall | 27.210253649 | 26.591046850 | 29.544967125 | 30.037355373 |

Physics costs increase **0.836950850 ms RTX /1.315846725 ms GB**; baseline/candidate
ratios are0.937289484 /0.916354996. The separate RTX wall decrease is not a
physics speedup, and one round does not establish repeated timing evidence.

Artifacts: `/tmp/fpgs-g1-row-owned-packets-whole-{rtx,gb}16k-20260917-01`.
Independent replay passes all original warning, capacity, source, process,
budget, result and final-idle guards:19 driver pins and28 artifact pins per card,
four captures/eight boundaries,160 physics and40 auxiliary graph launches per
capture. Only register-residual and register-packets flags change. The observer
checks exact key-only producers, the17-argument small ABI (CFM5, routing15,
PacketInput16),16-argument GS fallback, and all three filtered materializers.
All sticky flags are false; warnings passing is not physical convergence.

Candidate small/fallback counts at the two boundaries are15236/1148 and
15243/1141 RTX,15281/1103 and15284/1100 GB. Contact counts differ by at most
0.605%, row totals by0.466%; GB candidate has fewer contacts and rows at both
endpoints. All worlds have rows, MF counts are zero and host call cadence is
unchanged. This does not show a workload explosion explaining the loss, but
endpoint counts are not a sweep census or a proof of identical trajectories.
Post-environment invalid-cache counts are not cumulative reset counts.

## Strict node accounting and cause

Candidate traces are `/tmp/fpgs-g1-row-owned-packets-node-{rtx,gb}16k-20260917-01`.
The strict reader preserves the original analyzer rejection of auxiliary graphs;
it does not rewrite the failed parent as successful. Supplemental parsing proves
48 physics /12 auxiliary roots, exact process/correlation ownership, all graph
kernel/memory nodes, zero unproven nodes, capacities and frozen source identities.

| Candidate boundary, ms/env | RTX | GB |
| --- | ---: | ---: |
| Small fused formation + solve, owner sum | 3.583258750 | 4.317253083 |
| Original filtered GS fallback, owner sum | 0.913293750 | 0.969437333 |
| Formation + solve interval union | 4.496552500 | 5.286690417 |
| Formation + solve exclusive busy | 4.496552500 | 5.284589083 |
| Remaining row services, exclusive busy | 1.680504917 | 1.609456167 |
| Complete physics-root span | 13.466783750 | 15.935077000 |

The old value-emitting prefix, contact triplet and restitution owners are absent,
as are the old corrected-limit solve and compact small solve. Key prefix,
key contacts, filtered prefix, filtered contacts and filtered restitution each
run8 times/environment step; small and fallback each run8. They are all charged.
Their individual sums are not substituted for interval union/exclusive metrics.
The small kernel is **formation plus solve**, not GS-only. It retains118/112
registers and5760 shared bytes in the trace, with zero reported local memory.

Against the prior compact strict capture, row-service exclusive work falls only
**1.089366833 /1.179004417 ms**, not the2.067035 ms optimistic pre-code RTX
producer ceiling. New keys and filtered materializers remain. Formation+solve
exclusive work rises1.558681250 /2.043888167 ms against compact GS; its fused
owner must also recover compact's pre-existing deficit to the accepted owner.
This complete accounting explains why producer retirement did not pay. These
are separate short diagnostic captures with slightly different workloads,
not a sum-of-overlapping-components prediction of exact whole-run time.
No hardware stall fraction, memory-bandwidth bottleneck or instruction-cache
cause is established by these timings.

## One correction and closure

The separate branch `ooctipus/fpgs-g1-row-owned-width43-20260917` replaces exactly
105 shared-tile allocation/bound/address sites44→43 and no arithmetic. The
stride44 equal-node access pattern can place four distinct row addresses in a
bank; width43 removes that structural alias. This was an address-pattern
hypothesis, not measured stall evidence. Resources change5760→5632 shared bytes,
registers remain118/112, and all six native groups pass/card.

Its clean `b21cf507` paired whole result remains a loss: RTX12.502508425→13.333823950
ms and GB14.419813275→15.718051525 ms. No material recovery is observed. No43-wide
node capture, additional stride, MMA or mapping correction is needed or funded.
Both experiments remain default off and unpromoted; accepted production is
unchanged. Neither clean guards nor native comparison asserts full convergence
or a4× performance result.

## Reproducibility pins

- Whole manifests RTX `84a73f8d235486b985691f9c3ef3603010381130f8abf4369985c0fa788e4f6b`,
  GB `e64657b2ba6411464e3cb8039fd69fe4cf456efda751700524c7c0e53881db18`.
- Native03 logs RTX `e95ec8a092f41bb8083f7a5e4691d1cd25eee57395ea452e5a73df3740a50a1f`,
  GB `4740166b1e8b79bb6719008e1873b65667c5aa0364ef97fe8e1bb1fd96b77928`.
- Momentum-control logs RTX `ac190a47e4dbbcdc64b66ef264a627d013249c63a0ff15c7cf5cae3b53463e92`,
  GB `388e404973a02b02abdeb177464eb26711128a1325190cdba5d1f020e18f8620`.
- Bound owner `e56a42e85529961b61e39d40ecf0664fb80b323568d036487ff3b0d8097be9f2`;
  factor `708a99fbce60f4e2cc89fb4455c2d383b138f9ac575ee27aa66991e01bda2f10`;
  runner `649e08a5cd016b8035a9ff819da23b617a04116e5937e6fb508cd2c832bd98b2`.
- Timed observer `dfb2fcd7016098199962bb841c4ed33363014fc1f92b05a1f1203e7fe7648e9d`.
  This closure retains those exact bytes. The width43 branch has an import-order
  only observer change to
  `929464ada458dcdab7d9b6a0084d77ffcebefdcd5e10ede3682c860ea8829b8b`;
  its observation logic is unchanged. Runtime, tests and observer are not changed
  by this report-only closure; checkout56608 preserves the measured snapshot.

Strict reader `/tmp/fpgs-register-packets-strict-IPUKfZ4S/read_packets.py` final
SHA is `be03e7ac47b9c88b635923eb2f07724a8b093f747f37d207b645170b9df86d58`.
The original outputs RTX `e5fdd9ade4b4aa649b16829cff8f634d3c2b3cc8074cdde3ad48795615795949`
/GB `0642da8711a367c34a831e08db981db51729f6b364587c48098fb552fe029b7e`
embed the earlier reader `08847b551cfa10b35920fd6c8634191d0abe108cb0c2dd341c08585a0b7474fb`.
The only later edits strengthen the replacement-count guard from>0 to==8 and
rename an inherited GS comment to formation-and-solve; reversing those two edits
reconstructs the old reader hash exactly. No arithmetic or attribution changed.
Independent final-reader replay in `/tmp/fpgs-register-packets-node-review-ec8hASPb`
produces RTX `5ac994a815f532f2c887edc1b99572699bc39c49595825681efe52aac519e572`
/GB `45a976ba080aedb8fb8e6a99d3ba6c06e681ae1c2f58aa86b4fe8dcee61f1910`.
Recursive comparison differs only in the embedded reader hash. Both generations
of outputs and the original profiler failure remain preserved.
