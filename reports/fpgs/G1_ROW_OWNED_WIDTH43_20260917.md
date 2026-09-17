# One row-bank mapping correction — 2026-09-17

Status: CLOSED without an accepted gain; default off. This preserves the original
row-owned packet prototype at `56608c664fb1f10161743b9b09865f29d5f9fb26`.
The accepted comparator remains the corrected-limit owner, not that prototype.

## Why this one correction

Initial packets lost whole physics: RTX12.509309075 →13.346259925 ms;
GB14.415478075 →15.731324800 ms. All original source/idle/capacity/budget
guards and independent audits pass. Strict node attribution separates48
physics/12 auxiliary roots with zero unproven nodes. The small fused owner
costs3.583258750/4.317253083 ms, while complete row production retires only
about1.09/1.18 ms. The local formation cost cannot be called GS-only.

Compiled PTX/SASS retains the dependent shared-J load inside the triangular
WJ loop. With row stride44, equal-node accesses use only8 of32 banks: up to
four distinct addresses per bank. This establishes an address-pattern issue,
not a measured stall fraction; partial masks and mixed templates matter.

This single correction uses the exact physical width43. All105 tile allocation,
bound and row-address sites change44→43; arithmetic order, Gram registers,
prefix energy gate, contact law, publication/fallback and final dead-tile du
reuse remain unchanged. No new launch, matrix or copy is added. No stride grid
or further mapping correction is funded.

A1 ms whole saving would require recovering1.837 ms RTX /2.316 ms GB from the
losing prototype, more than its entire added small-owner cost versus compact.
The bank correction is a short causal test, not a credible promise of that
milestone. Recovery versus a losing candidate is not itself accepted progress.

## Qualification

Regression first fails on the44-wide source. Five CPU controls then pass;
the generated source is exactly the frozen original after only the105 checked
address substitutions. Native source SHA256:
`fe7539cbf85a5e111773b4b50101425179ccd392e91cd831892aed2262eac985`.
Test SHA256:
`28027e1714679499cb673a4051f0eeb8441e9daf671776467e43886ac3a38724`.

Hidden-device AOT `/tmp/fpgs-register-packets-width43-offline-MvplCiWc/offline01`
passes exact source guards, both SM120/103:118/112 registers unchanged,
5632 shared bytes (128 fewer), zero stack/spills/local memory. Text is
82048/82304 bytes. Six existing native groups pass on each card; complete paired
timing uses the original drivers, with unchanged physical tolerances and budgets.
Full precommit also reordered one observer import; observation logic is unchanged.
The observer SHA is
`929464ada458dcdab7d9b6a0084d77ffcebefdcd5e10ede3682c860ea8829b8b`.
Independent diff review confirms the9 generic and96 named-row replacements,
physical indices0..42, retained lane+32 bounds and full-warp read-completion
fence before the unchanged flat43-float final-du overlay. The test pins the
untouched padded generated source at
`7aeb40e66e7ddb64f317cab45a32688209b2faa98e695cbda5caa95939a821a6`.

## Complete result: no material recovery

Clean candidate `b21cf507` is compared with corrected-limit
`c6ab26f9687f64fd2acfe66f5328f16be1820692`, not with the losing44-wide prototype.
Both use fixed Lab53ee6b44,16,384 worlds, seed0, one paired round,200 warmup,
40 unprofiled wall and40 graph-profile steps. Simulation dt.005, decimation4,
two .0025 solver substeps, maximum8 passes and capacities100 rows/294,912 raw
contacts/49,152 broad pairs/1,769,472 triangle pairs remain unchanged.

| Metric, ms/environment step | RTX baseline | RTX width43 | GB baseline | GB width43 |
| --- | ---: | ---: | ---: | ---: |
| Profiled whole physics graph | 12.502508425 | 13.333823950 | 14.419813275 | 15.718051525 |
| Unprofiled environment wall | 27.151422025 | 27.606275125 | 28.862919498 | 30.756425200 |

Physics regresses **0.831315525 ms RTX /1.298238250 ms GB**;
baseline/candidate ratios are0.937653630 /0.917404632. These losses are essentially
the original44-wide losses0.836950850 /1.315846725 ms. Different paired runs and
workloads do not establish a precise tiny width benefit; there is no material
recovery. Wall timing is separate from profiled physics and also worsens here.

Artifacts: `/tmp/fpgs-g1-row-owned-width43-whole-{rtx,gb}16k-20260917-01`.
Independent replay passes original source, artifact, warning, capacity, process,
budget, result and final-idle guards:19 driver/28 artifact pins per card,
four captures/eight boundaries,160 physics/40 auxiliary graph launches per
capture. Only register-residual/register-packets flags change. Exact key-only
producers,17-argument small owner,16-argument original fallback and all filtered
materializers are observed. All sticky flags are false; this is not a convergence
certificate. Runtime, tests and observer bytes remain those measured atb21cf507.

Small/fallback endpoint counts: RTX15227/1157→15312/1072;
GB15277/1107→15239/1145. Endpoint contacts differ by at most0.681%, rows0.498%;
all worlds have rows, MF counts are zero and host call cadence is unchanged.
No population explosion is observed. This is not identical-trajectory or
executed-sweep evidence; invalid-cache snapshots are not cumulative resets.

## Preserved physical controls and original node cause

The width test does not replace the inherited six native groups. Original
native02 found two saved7410 component-gate failures on each card when a new
FP64 geometric-J reconstruction was substituted for rounded productionZ.
Diagnostic controls prove candidate and original have exactly the same errors
(4.0260e-5/3.7645e-5; tolerance ratios5.6129/2.6684). The original rounded-Z
tolerance remains unchanged at rtol3e-5/atol3e-6, using controlZ saved before
candidate poisoning. Independent J/H backward error remains gated at2e-6;
observed maximum is4.2574e-8. Cone, metadata, current/held, restitution,
incoming/delayed, threshold/fallback and graph-transition checks remain.
Original native03 and width43 native01 both pass all six groups/card. These
finite-budget controls are not a new converged-manifold reference or full
population convergence qualification. Failed02 and diagnostic logs are retained
under `/tmp/fpgs-g1-row-owned-packets-*`.

Original strict nodes `/tmp/fpgs-g1-row-owned-packets-node-{rtx,gb}16k-20260917-01`
prove48 physics/12 auxiliary roots, zero unproven nodes, exact process ownership,
capacity/source/factory guards and preservation of the original analyzer error.
The retired value prefix/contact/restitution owners are absent; all five key
and filtered-materializer owners run8 times/env, as do small and GS fallback.
Original formation+solve interval union is4.496552500/5.286690417 ms RTX/GB,
exclusive busy4.496552500/5.284589083; remaining row-service exclusive time is
1.680504917/1.609456167. The fused small owner includes formation and is not GS-only.

The pre-code RTX maximum displaced-producer sum was2.067035 ms, before charging
keys, filtered materialization, extra dispatch and compact's existing0.352563 ms
solve deficit. Actual row-service exclusive retirement against compact is only
1.089366833/1.179004417 ms, while formation+solve adds1.558681250/2.043888167 ms
against compact GS. This explains the unfavorable complete cost. Separate short
captures are not an exact sum-based predictor of whole time. The bank-address
pattern was real, but this one correction does not demonstrate it was a material
performance cause; no hardware stall fraction is measured.

No width43 node capture or further width/MMA/mapping grid is needed. Both branches
close default off and unpromoted. The accepted corrected-limit baseline is
unchanged; no4×, bit-identity or full-convergence claim is made.

## Artifact pins

- Whole manifests RTX `4b9e991ab7a057f917baa9a034b7b9647ea869fc2a414b94d1d0aa792f72935c`,
  GB `cbf44b95755a4bb6dc6a278c598a81a423fc780347dee1b72df27f9a4d8655b5`.
- Width43 native01 logs RTX `d8eb8358bfb3b252c334557f7c0783a34424673d2da7b6f27fc0db6918d793a0`,
  GB `9fb21b567691a60138c89ac04011257f8f882e6b7b5d6b43070e427355627300`.
- Original native03 logs RTX `e95ec8a092f41bb8083f7a5e4691d1cd25eee57395ea452e5a73df3740a50a1f`,
  GB `4740166b1e8b79bb6719008e1873b65667c5aa0364ef97fe8e1bb1fd96b77928`.
- Original node reader final SHA
  `be03e7ac47b9c88b635923eb2f07724a8b093f747f37d207b645170b9df86d58` at
  `/tmp/fpgs-register-packets-strict-IPUKfZ4S/read_packets.py`.
- Independent final-reader outputs in `/tmp/fpgs-register-packets-node-review-ec8hASPb`:
  RTX `5ac994a815f532f2c887edc1b99572699bc39c49595825681efe52aac519e572`,
  GB `45a976ba080aedb8fb8e6a99d3ba6c06e681ae1c2f58aa86b4fe8dcee61f1910`.

The root's original node outputs `e5fdd9ade4b4aa649b16829cff8f634d3c2b3cc8074cdde3ad48795615795949`
/`0642da8711a367c34a831e08db981db51729f6b364587c48098fb552fe029b7e` embed
reader `08847b551cfa10b35920fd6c8634191d0abe108cb0c2dd341c08585a0b7474fb`.
Only two later reader edits occurred: strengthen replacement-count>0 to==8 and
rename an inherited GS comment to formation-and-solve. Reversing these reproduces
the earlier reader hash; all output fields except the embedded reader hash are
identical. Original outputs and failed-parent evidence were not overwritten.
