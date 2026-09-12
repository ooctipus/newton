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
