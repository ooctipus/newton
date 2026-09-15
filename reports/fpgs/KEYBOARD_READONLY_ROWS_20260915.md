# Keyboard read-only row staging: candidate card

Decision recorded 2026-09-15 08:13 UTC, before runtime changes.
Base: accepted `b1bad06ae0cd5b86ac07b2f2844ef62770e99075`.
This does not include the unsuccessful stationary-limit experiment.

## Hypothesis and exact replacement

Remove the sparse GS owner's shared copies of immutable row RHS, diagonal,
packed type/parent metadata, and friction coefficient. Retain mutable shared
velocity and impulses, every public producer/allocation, all admitted rows,
the independent-contact and speculative serial schedules, friction law,
limit projection, exact stationary-sweep check, and iteration allowance.
Read the same current authoritative row arrays at their original use sites.
Preserve the original packed type/parent interpretation. No new launch,
classification, global intermediate, routing tier, row cap, or fallback pass.

The candidate is default-off, distinctly keyed, and has the original native
argument ABI. Explicit unsupported runtime selection must not silently claim
that the candidate ran. Isaac Lab and dependency pointers remain unchanged.

## Complete cost case and uncertainty

Fresh accepted Keyboard RTX physics is 7.443361925 ms in the preceding
40-step screen. The corresponding separate node diagnostic measured sparse
GS at 1.078966 ms per environment step; this is ownership evidence, not an
exact subtraction from that whole-step sample. Eight calls execute per step.

The exact original compiled owner uses 14,664 B static shared memory,
94 registers/thread on RTX and 88 on GB300, without spills or stack.
Removing four 704-element staging arrays would leave approximately 3,400 B,
including the original 128 B compiler scratch. With unchanged registers,
256-register warp allocation and four register subpartitions give a
conditional resource ceiling of 20 one-warp blocks per SM on each GPU,
compared with shared-memory ceilings of six RTX and fifteen GB300 blocks.
These are resource ceilings, not measured achieved occupancy or speedup.

An ideal residency-proportional RTX owner would fall to approximately
0.324 ms, saving 0.755 ms before new costs. This gives a credible but narrow
route to a roughly 10% whole-step milestone. Global loads are now repeated
at use rather than copied once: cache misses, latency, address arithmetic,
metadata packing, changed registers, and reduced reuse can erase that gain.
An inactive tangent may avoid metadata loads entirely. No bandwidth or
latency-bound claim is established without counters. GB300 has a much
smaller conditional residency opportunity and is measured concurrently.

The historical corrected MJWarp RTX denominator of 28.224675875 ms implies
a 4x ceiling of 7.056168969 ms. Reaching that goal would need approximately
0.387 ms from the fresh baseline, but a new matched backend measurement is
required before claiming a new 4x comparison. The existing accepted table
does not change because a prototype or one screen crosses this value.

## Predeclared experiment and decision

One integrated variant, not a cache-policy or tier-size grid. Begin with
regression-first factory coverage and reuse the response-diagonal native
tests. Test eager and captured use, repeated calls with changed immutable
input values, empty/shrinking/growing rows, mixed prefix/contact rows,
friction delay, coupled and independent contacts, and mutable output stores.
Separate immutable arrays must remain unmodified. Numerical and physical
quality are the contract; bit identity is not required.

Then run the existing paired 4K Keyboard whole-step driver: seed 0, 200 warmup,
40 wall and 40 physics steps, fixed Lab, dt, two substeps, eight GS sweeps,
dense row cap 704, raw contact cap 147456, broad-phase cap 57344, and the same
three accepted Keyboard flags in both arms. No rows may be dropped.

The ambition is at least 0.7 ms RTX whole-physics saving. A smaller substantial
gain that demonstrably reaches the user's 4x task goal may proceed to
qualification; do not reject it solely for missing the internal 0.7 ms
milestone. A one-round gain is only triage. Promotion needs at least three
balanced paired rounds, matched post-change backend comparison, physical
and reset checks, and confirmation that unsupported task paths are unchanged.

If the screen loses or is marginal, use one existing node diagnostic and
compiled resource evidence to distinguish changed resource allocation from
new memory/instruction costs. A targeted correction needs a specific new
cause and revised whole-step prediction. No blind tuning or polishing an
unqualified path. Initial checkpoint is 90 minutes from this card; no silent
extension beyond two hours. The accepted runtime stays unchanged meanwhile.

## Ownership review before implementation

The complete sparse owner only initializes and reads the four staging
arrays. The production RHS, diagonal, type, parent and coefficient buffers
are separately allocated from velocity, impulse and fused-limit outputs.
Their producers precede this owner. Review the exact source and native
generated code while implementing; do not infer cache safety from a const
pointer alone or carry read-only assumptions across kernel boundaries.

Status: one candidate funded; implementation and GPU evidence pending.

## Implementation checkpoint

`FEATHER_PGS_SPARSE_READONLY_ROWS=1` explicitly selects the sparse owner;
it defaults off and raises for an unsupported solver owner. No stationary
mask or other part of that unsuccessful experiment is included.

A local source-generation helper selects each original shared read or a
cached read of the current authoritative argument. It removes only the four
read-only declarations and their initial copies when enabled. Mutable shared
velocity/impulses, all public arrays, native argument order, row schedules,
denominator branch distinctions, explicit rounding operations, friction and
limit arithmetic, iteration budget and stores remain unchanged. Packed type
and parent expressions are reconstructed literally; the passed `rhs_bias`
argument is consumed rather than a hardwired solver attribute. The selected
kernel has a `_readonly_rows` suffix.

Regression-first session 4373 exited 1 with the expected missing
`readonly_rows` factory argument before runtime changes. The corrected CPU
factory control passed (53009). Existing response-diagonal, sparse-response
owner and compact-contact modules then passed 11 CPU tests, with 17 CUDA
tests skipped (59756); these skips are not physical passes. Independent CPU
source review found default-off generated native snippets byte-identical to
the accepted base for plain, contact-group and speculative variants.

The physical scalar-reference and speculative-reference tests now exercise
the optional variant through their original fixtures and tolerances. One
additional test reuses the scalar test's actual launch arguments, expands
them to two worlds, captures once, and updates the same source arrays across
empty, growing, mixed prefix/contact and independent-contact states. RHS,
diagonal, packed type/parent and friction coefficients change after capture;
world strides, delayed friction, mutable outputs and read-only preservation
are checked against the original owner. No new benchmark or dataset is
introduced. Runtime source is held while root performs the actual compiled
resource query; full paired physical and whole-step qualification remain
pending at this checkpoint.

Root's actual CUDA-driver resource query completed on both cards without
kernel launches or timing. For the production 704-row/114-coordinate owner:

| GPU | Variant | Registers/thread | Static shared B | Maximum active blocks/SM |
| --- | --- | ---: | ---: | ---: |
| RTX PRO6000 | Accepted | 94 | 14,664 | 6 |
| RTX PRO6000 | Read-only rows | 88 | 3,400 | 20 |
| GB300 | Accepted | 88 | 14,664 | 14 |
| GB300 | Read-only rows | 86 | 3,400 | 20 |

Dynamic shared memory is zero and preferred carveout is -1. The actual GB300
driver ceiling is 14, correcting the pre-code rough estimate of 15. These
resource ceilings confirm the intended storage effect, not achieved
occupancy, physical correctness or a speedup. Preserved resource logs:
`/tmp/fpgs-keyboard-readonly-resource-4rw2va19/gpu0.log` and `gpu1.log`.
Their SHA256 values are, respectively,
`e210399acb6ed1121a0341da27ec201d2936b00de1c0c324c2c0a51b02aae148`
and `e936989da74e18e42eeb718b131823ed08633138747569d5c663dc88326e4453`.

## Closure: resource mechanism confirmed, whole-physics target missed

The candidate is not promoted. Runtime and tests remained frozen at
`46f24d32d6548ab6b8a550ade04874972cbf2cb9` for all qualification and timing.
This closure is report-only; accepted Newton remains
`b1bad06ae0cd5b86ac07b2f2844ef62770e99075`. No rescue, default change, or
combination with the stationary-limit path was made in this branch.

### Actual native qualification

Each GPU ran the complete response-diagonal module: 15/16 methods passed
on RTX and 15/16 on GB300. All four new methods passed, including unchanged
scalar/speculative physical tolerances and two-world changed-input graph
replay. Each module retained the inherited
`test_tiled_response_diagonal_matches_dense_reference` error: its existing
`accumulate_group_diag_worlds` call passes seven arguments but the kernel
requires nine. This unchanged failure is preserved, not an all-green suite.
Runs took 44.487 seconds RTX and 48.052 seconds GB300 and exited nonzero.

Native logs at `/tmp/fpgs-keyboard-readonly-native-IicjyVO3`:

- `gpu0.log`: SHA256 `be5bff82a1be70d44d173da5248623460619eb95cfb544bb624121cf5f346cb5`.
- `gpu1.log`: SHA256 `0fcf994f0522fab847b0687b5e848ebdcc04745ed3fa5a2d0cdf68bd50e5fc55`.

### Paired whole-physics screen

One discovery round at 4,096 environments, seed 0, 200 warmup, 40 synchronized
wall steps and 40 graph-profile steps. Values are ms per batched environment
step; the ratio compares accepted FPGS with this candidate, not MJWarp.

| GPU | Accepted physics | Read-only physics | Candidate minus accepted | Accepted/candidate |
| --- | ---: | ---: | ---: | ---: |
| RTX PRO6000 | 7.428472925 | 7.777718350 | +0.349245425 | 0.9551x |
| GB300 | 6.843528050 | 6.826663150 | -0.016864900 | 1.0025x |

The primary RTX result is a regression. Neither card approaches the 0.7 ms
milestone, and RTX does not reach the predeclared 4x historical-denominator
ceiling. No new matched MJWarp or repeated improvement claim is made.

| GPU | Accepted environment wall | Read-only environment wall |
| --- | ---: | ---: |
| RTX PRO6000 | 30.422373849 | 29.885451775 |
| GB300 | 28.021014400 | 26.822576026 |

These are separate synchronized environment wall windows, including host and
reset work, not full RL training updates. Their favorable one-round deltas
do not negate the physics loss or establish causal training improvement.

### One measured owner diagnosis

The separate matched three-step node capture confirms eight calls per step
of the actual `_readonly_rows` sparse GS owner and the intended compiled
resource reduction. Named kernel-duration sums per environment step:

| GPU | Accepted GS ms | Read-only GS ms | Candidate minus accepted |
| --- | ---: | ---: | ---: |
| RTX PRO6000 | 1.287765667 | 1.448032667 | +0.160267000 |
| GB300 | 1.426954333 | 1.435936000 | +0.008981667 |

The actual solver got slower despite registers falling 94 to 88 on RTX and
88 to 86 on GB300, shared memory falling 14,664 to 3,400 bytes, and zero
reported local memory in both arms. There is no new producer or launch to
blame: the changed owner itself costs more. The source replaces once-staged
metadata with repeated cached global reads and addressing at use sites.
Those are real source-level costs, but no hardware counters establish cache
misses, memory bandwidth or instruction issue as the specific limiting cause.
The driver residency ceiling improved; achieved occupancy was not measured.
Thus residency-proportional speedup was an unsuccessful prediction, not a
measured effect hidden elsewhere in the pipeline.

Short node physics spans were RTX 7.607929333 to 7.779795000 ms and GB300
6.897205667 to 6.854346333 ms. Both arms retain 420 device nodes per profiled
step. These three-step instrumented windows are not the 40-step graph-mode
screen above. Named duration sums can overlap and are not exclusive
critical-path savings; neither node wall noise nor cross-window subtraction
is used to manufacture a whole-step gain. No cache-policy, tile or parameter
rescue was attempted in this branch.

### Pins, unchanged protocol and checks

Fixed Lab: `53ee6b44c2334341305dbdf385a3916c6b140799`, with only its untracked
`.venv`; comparison tools: `961b7e2f751bcd1d8b03368e7956b54c81414897`.
The original fixed-import adapter and complete child commands/environments
are pinned in the manifests. Both arms retain
`FEATHER_PGS_SPARSE_CONTACT_DIRECT=1`,
`FEATHER_PGS_PRISMATIC_PUBLICATION=1`, and
`FEATHER_PGS_COMPACT_CONTACT_BOUNDARY=1`; only
`FEATHER_PGS_SPARSE_READONLY_ROWS` changes from 0 to 1. No Lab source or
dependency pointer changed. Caps remain dense/propagation 704, matrix-free
64, raw contacts 147,456, broad output 57,344. Budgets remain sim_dt 0.01,
decimation 4, two Newton substeps of 0.005, eight GS sweeps maximum.

All four whole children and all four node children exited zero. Each has two
complete passing available capacity boundaries and finite state; both final
source/idle guards passed. All 64 recorded per-run artifact hashes were
rechecked successfully. Available sticky flags and finite states do not
establish full convergence or complete demand calibration. No repeated
timing or performance acceptance is recorded.

- Whole artifacts: `/tmp/fpgs-keyboard-readonly-rows-whole-paired4k-20260915-01`, manifest SHA256 `7ee16bd0555f913022400c5ed877a8964bded646992d72c003a4625807c099f6`.
- Node artifacts: `/tmp/fpgs-keyboard-readonly-rows-nodes-paired4k-20260915-01`, manifest SHA256 `58b1da7a5d7b5c0f3a8b7271b665ec272361ef7048fadc22032ab39f026c84cc`.
- Frozen solver source SHA256: `61b41e4e694324534e5c74ddf81ecef0f03104bc9005912cd897605de204af31`.
- Frozen response-diagonal tests SHA256: `22edc41c23ba586b9ba4989edc1e27b714341dbebf950806af216c9081dccf73`.
