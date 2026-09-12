# Packed geometry and contact-local sequential GS

Decision at 09:38 UTC, 2026-09-12: implement one bounded producer/consumer
correction to the [diagnosed private-island loss](PRIVATE_CONTACT_ISLANDS_20260912.md).
This is an unvalidated hypothesis, not a promoted optimization. Newton starts
at `ad3b7ec5f60a7743aeed7ac1558aca64b905fece` on the `ooctipus/newton` fork;
the original handoff remains an ancestor. Isaac Lab and all physics allowances
stay unchanged. Accepted E2 `5777558f`, not the slower private prototype,
remains the performance comparison.

## Work replaced

Move coupled-contact geometry back to raw-order contact workers. Only current,
fully admitted residual contacts are tagged; isolated scalar packets and the
complete E2 fallback remain unchanged. Reuse canonical J/Y and sparse fields,
explicitly writing dense zeros where required. The new producer also records
four local response products per contact: G10, G20, G21 and G22, where
Gij = Ji dot Yj using the actual dense-six and sparse coefficients.

In the coupled consumer, evaluate the three initial row velocities concurrently
in three eight-lane groups. Reproduce the original normal, tangent-one and
tangent-two GS updates, including disk projection and sibling changes, using
those four products. Publish the combined velocity response once per contact.
This is not a simultaneous cone solve or a reduced iteration allowance.

The dependency identity is:

- After the normal update, r1 += G10 * delta_normal and
  r2 += G20 * delta_normal.
- After tangent one's disk projection, r2 += G21 * delta_tangent1 +
  G22 * delta_tangent2_sibling.
- Tangent two's final sibling change contributes to the final velocity update;
  no further local row needs a refreshed residual.

Keep original denominators (including denominator-only CFM), omega, friction
delay, ordered contacts, prefix/coordinate limits, speculative inactive-contact
screening and the maximum eight sweeps. Record intermediate changes for
stationary exit even if their net impulse cancels. No symmetry of rounded
Ji/Yj products is assumed. Shared state omits private J/Y and sparse coefficient
panels; expected storage is roughly 4 KiB/world, subject to actual compilation.

## Whole-boundary cost and falsifiable gate

The current private node boundary is 2.940676 ms RTX versus E2 2.723458 ms.
Its coupled producer/consumer alone costs 1.434167 ms. Holding the other owners
fixed, RTX-minus-0.7-ms requires the new combined coupled boundary to fit
0.516949 ms (64.619 us per solver call). GB requires no regression: the
corresponding budget is 0.900575 ms, not an equal 0.7-ms saving.

The measured causes support two complementary mechanisms: roughly 8x more
issued geometry warp groups in the private layout, and three serialized row
reductions/publications per active frictional contact. Packing attacks the
first; contact-local blocking attacks the second. The slowest worlds need
roughly 2–2.5x recurrence improvement in the tail-limited scenario. Running the
three reductions concurrently and publishing once provides a concrete shorter
dependency chain, unlike a residency multiplier alone. Projection arithmetic
and inter-contact dependencies remain sequential, so a 3x speedup is not
assumed or promised.

Charge current raw tags, the geometry producer, four-product formation and
storage, all canonical coefficient traffic, consumer loads, retained prefix,
fallback, launches and publication. The earlier coupled-coefficient census
estimates about 19.2 MB/environment step for a write and one read, not measured
DRAM traffic; repeated sweeps add reads. New tag/product arrays are bounded by
the existing configured contact allocation, with no global capacity increase.
Extra tangent loads for inactive contacts, reduced cache reuse, producer
register pressure or projection-dominated tails can falsify the hypothesis.

## Implementation and acceptance

Keep the experiment default-off on a separate branch. Root owns integration,
paired GPU scheduling and this record; producer and recurrence agents own
separate modules; the test agent reuses existing fixtures. A thread-per-world
alternative is documented separately, not combined into this experiment.

Initial checkpoint: 11:08 UTC (90 minutes). Target the first integrated 4K
paired timing by 11:38 UTC; a later extension requires a newly documented
cause and revised measurable hypothesis. Begin with regression-first algebra,
current tag/ownership, mixed fallback, reset/notify and captured-graph controls.
Do not delay early complete timing for a larger telemetry framework.

Acceptance requires repeated paired whole-physics timing against E2 and loaded
convergence/physical checks at unchanged allowances. Check complementarity,
normal/friction response, momentum and reset stability rather than demanding
bit identity across reassociated arithmetic. Capacity losses remain failures.
Unsupported tasks retain existing dispatch and must not be described as gains.

The fixed Keyboard 2x target is still 11.258526 / 2 = 5.629263 ms on RTX;
the 0.7-ms structural milestone is not achievement of the multi-task 2–4x goal.
No Isaac Lab, MJWarp bridge or parent dependency pointer is changed.

## Source-ready checkpoint

The missing selector/owner regression failed before implementation. The
integration shares the original solve descriptor binding, scalar owner and
whole-world fallback. All 36 CPU-capable inherited controls pass (46 tests,
10 CUDA-only skips); this is not a GPU or physical acceptance result.

The packed producer compiles for both cards at 96/94 forward registers,
1024 bytes shared and no forward stack/spills. Its two new arrays consume
exactly 20 bytes per configured raw contact (2.8125 MiB at 147456), including
zero capacity without an artificial floor. Tags are cleared over that exact
allocation in the existing route pass; producer reads are also bounded by the
current contact owner. Smaller current buffers and regrowth do not retain an
old tag lease. No extra clear kernel or global capacity increase is introduced.

The thread-per-world alternative remains unimplemented. Its straightforward
32-world block mapping would launch only 128 blocks at 4K, fewer than the
RTX's 188 SMs; its lower per-world shared allocation is not an automatic
throughput multiplier. The local-block route is the single selected experiment.
Integrated whole timing remains pending.

## First paired GPU controls, 09:55 UTC

All **59 tests pass on each actual GPU, with no skips**. This includes the new
native eight-sweep versus original-GS comparison and deliberately wrong-Gram
negative control, prefix/empty/mixed-owner and delayed-friction cases, current
tag graphs, and the complete solver with mixed fallback, held-mass refresh,
reset/notify, two graph replays and no-contact transition. The new complete-step
fixture uses dt=0.005; inherited fixtures retain their original default dt.
CPU-only execution separately passes 45 controls with 14 CUDA-only skips.

Final consumer compilation uses 66 registers and 4048 shared bytes on both
architectures, with no stack/spills. The producer's forward resources above
are separate from unused generated backward functions; gradients are outside
the admitted path. Both new runtime modules pass independent source reviews.
Full pre-commit passes with all new files included. The paired parent is reaped,
source/idle guards pass, and no source bytes changed during these controls.

The controls do not establish full-size physical acceptance, performance gain,
or resolution of the inherited pooled-allocation FK memcheck discrepancy
documented in the private-island report. No allocator or FK behavior changed.

Paired manifest:
`/tmp/fpgs-contact-block-gpu-controls-20260912-01/manifest.json`, SHA256
`ea8d3274909cb14f66658f1ebd691277fa8f3e874fca68cbddfe45e45bc194bb`.
Exact runtime sources: producer
`972abdd0f934b7029484ee6121e87749468324c2760b53e123221274b67fcdc3`,
consumer `52b97f593c618666dbf08ee064db76654f98ba24101d7ef12006f2d2d3ea3f79`,
owner `f09616dedcbaa70dd8edaa92a6b713ca181da4dc07f1a178bff8d7014f3ffb17`.

For the early screen, reuse the accepted handoff's paired
`tools/fpgs_bench/compare_variants.py` with accepted E2 as baseline and this
checkout as candidate, task `keyboard-so101`, GPUs `0 1`, seed 0, 200 warmup,
40 synchronized steps, 40 graph-profile steps and one round. Both arms enable
`FEATHER_PGS_SPARSE_CONTACT_DIRECT=1`, `FEATHER_PGS_PRISMATIC_PUBLICATION=1`
and `FEATHER_PGS_COMPACT_CONTACT_BOUNDARY=1`; only the candidate enables
`FEATHER_PGS_CONTACT_BLOCK=1`. Neither older private/fused selector is enabled.
Keep capacities 704 rows, 147456 contacts and 57344 broad outputs. Run 512 then
4096 environments; repeat balanced 4K timing and full loaded quality only if
the early complete result qualifies. Manifests retain full runnable commands.

## Integrated screen and diagnosed loss, 10:14 UTC

Runtime `8e170074` passes the early capacity/source controls but **does not
meet the performance gate and is not promoted**. Both paired parents are
reaped. These are one-round discovery measurements, not repeated gains:

| Environments | RTX E2 / candidate physics ms | RTX ratio | GB E2 / candidate physics ms | GB ratio |
| --- | ---: | ---: | ---: | ---: |
| 512 | 3.426289 / 3.279365 | 1.04480x | 3.628322 / 3.506766 | 1.03466x |
| 4096 | 7.510662 / 7.157157 | 1.04939x | 6.181975 / 6.501475 | 0.95086x |

The 4K result saves only 0.353505 ms on RTX, below the 0.7-ms milestone, and
regresses by 0.319500 ms on GB. Whole-environment wall ratios are 1.00914x /
1.00547x; these are not training measurements. The 512 discovery deliberately
retains the same 4K capacities, not a proposed production capacity recipe.
All checked capacity flags are zero and all 4096 candidate worlds are admitted
at both existing metadata boundaries on each card. Boundary snapshots do not
establish full-window admission or numerical/physical acceptance.

A fresh three-step node capture charges all fifteen candidate owners and the
ten E2 owners, with 24 calls per label/card/arm. Process-correlated interval
unions, not overlapping kernel sums, give this complete contact boundary:

| Current node window, ms/environment step | RTX E2 | RTX candidate | GB E2 | GB candidate |
| --- | ---: | ---: | ---: | ---: |
| Complete contact interval union | 2.764919 | 2.817154 | 2.707618 | 2.882872 |
| Full physics graph span | 7.225392 | 7.331144 | 6.445044 | 6.563712 |

The node window is a different sample and instrumentation regime from the
whole-graph screen above. Its RTX loss must not replace that screen's small
gain, and the small screen gain must not be called a repeated win.

Candidate disjoint kernel sums, in RTX / GB milliseconds:

- Retained limit prefix: 0.071104 / 0.083467.
- Retained post-passes: 0.177195 / 0.214971.
- Routing and partition: 0.350421 / 0.321684.
- Masked preparation and schedule: 0.108533 / 0.124691.
- Packed geometry and four contact-local products: 0.466155 / 0.433227.
- Scalar private owner: 0.786017 / 0.821781.
- Contact-local residual: 0.827798 / 0.858507.
- Original fallback solve: 0.029931 / 0.029035.

The new coupled producer/consumer union is 1.293953 / 1.291733 ms. Original
E2 GS already solves both scalar and coupled work in 1.087489 / 1.271572 ms;
the candidate's split scalar, residual and fallback solves total 1.643745 /
1.709323 ms. RTX preparation/scheduling saves approximately 0.504 ms but the
split solve owners add 0.556 ms. GB saves approximately 0.258 ms but adds
0.438 ms. The ledger explains the complete loss without assuming an
all-fallback path, memory spill, or unmeasured hardware-occupancy cause.

The intended consumer resource reduction is real (66 registers, 4048 shared
bytes, no local memory). It does not eliminate serial nonlinear projections,
inter-contact dependencies or producer-to-consumer coefficient publication
and reloads. Packing itself costs 0.466/0.433 ms. Original GB production is
already substantially cheaper than RTX production in this window, leaving
less saved preparation to cover the additional split solves.

Holding other owners fixed, the RTX milestone now requires the complete
coupled producer/consumer to fit **0.541718 ms**, and GB no-regression requires
**1.116480 ms**. Keeping the current producer leaves just **0.075563 ms RTX**
for the entire consumer (9.45 us/solver call), before any new Gram formation
or scan. A new recurrence layered after this producer therefore has no
credible standalone cost case. Next work must change the complete equation,
not tune another consumer layout or waive the full preparation bill.

An independent historical audit also finds that register-held residuals,
column-major small Delassus matrices and final-only velocity reconstruction
were already implemented in the retained `response_block_sweep.py`. Prior
real-task losses were followed by a staged-load retry that did not repair
the sequential friction/synchronization costs. Those mechanisms alone are
not a novel next experiment. A differentiated coupled-only representation
would need to charge its construction, admission, scalar work and complete
contact-unit recurrence. Changed algorithms are permitted if convergence and
physical behavior pass with unchanged task allowances; exact original-GS
arithmetic or trajectory is not an additional user requirement.

Full loaded quality and balanced repeats remain unperformed for this failed
speed screen. The 59 actual-GPU controls per card remain the narrower
correctness evidence. Accepted E2, the fixed improvement reference and all
Isaac Lab sources/settings remain unchanged.

Artifact pins:

- 512 manifest: `/tmp/fpgs-contact-block-keyboard512-screen-20260912-01/manifest.json`,
  SHA256 `8ab28f92b28d0d2fb512e070e071743b18fcddb858efc67ff499dc9f5b326bbe`.
- 4K screen: `/tmp/fpgs-contact-block-keyboard4096-screen-20260912-01/manifest.json`,
  SHA256 `d55f7c13140dd7d18833fc2e1c8a4dfc47abf4d7b5583f4fa6505580dabb5706`.
- 4K nodes: `/tmp/fpgs-contact-block-keyboard4096-nodes-20260912-01/manifest.json`,
  SHA256 `93819acd6da6293861cd7a2f36995c45f5689586d650c99f36681942a2a9a3c7`.
- Complete audit: `/tmp/fpgs-contact-block-node-audit-fUsXEL/evidence.json`,
  SHA256 `b7d14bfac9017413bad08f52c35b5dc336032dc9e27570660228e28cc738b356`;
  sibling `FINDINGS.md` SHA256
  `4c8af429ff928dfc2ae7e12d0e4e42f0aa38d3865d3a2bcc0828974544325435`.
- Historical recurrence audit: `/tmp/fpgs-contact-space-history-ulX4DbnU/HISTORY.md`,
  SHA256 `522613a2785aa4b3c2bd56809c985266c48a5cb34d91f8a4c5b5a42f73ead695`.
