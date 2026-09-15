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
