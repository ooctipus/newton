# Keyboard: retire provably stationary, row-isolated limit updates

## Pre-code decision, 2026-09-15 07:30 UTC

Base: accepted Newton `b1bad06ae0cd5b86ac07b2f2844ef62770e99075`.
No Isaac Lab, timestep, substep, iteration, collision, or capacity changes.
Experimental default-off Newton path; root owns paired GPU execution.

The finite-eight scalar limit study at
`/tmp/fpgs-keyboard-finite8-limit-YwaJ5rHS/CARD.md` found a simpler actual
opportunity than its nonzero closed-form recurrence. Across four existing,
hash-pinned 4K checkpoints, 1,661,952 contactless scalar-key observations all
had feasible enabled limits and exactly zero lower/upper impulse in the
literal eight updates and saved native result. These historical inputs are
an exposure census, not current performance or full trajectory qualification.

The candidate checks this fixed point once. A coordinate must be outside the
dense response block and untouched by **every current row**, including both
sparse endpoints of noncontact prefixes and contact triples. Its native fused
limit impulses start at zero on each call. With finite supported inputs,
positive finite denominator, omega=1 and nonnegative enabled residuals, each
old update returns zero impulse forever. Clear only the local active flags;
leave velocity and both public impulses unchanged. Any unproved coordinate,
violated or inconsistent bound, or unsupported numerical condition retains
the complete original path. No tolerance-based screening, contact deletion,
iteration reduction, or algebraic reassociation of nonzero impulses.

Publish one current ownership mask in the existing schedule builder, before
coordinate heads are compacted. Reserve both endpoints of every actual row
after contact linking using compare-and-swap only on empty heads, preserving the original
independent and serial contact schedules. Allocate exactly ceil(D/32) uint32
words/world (16 bytes/world at D=114); refresh every call, including empty and
growing row sets and after resets. Do not alias unused capacity or add a new
producer/launch. Builder and solver cost both belong to the candidate.

The source-equivalent historical complete sparse owner was 1.087489 ms RTX.
The ambition is at least 0.7 ms whole-physics reduction, requiring about 64%
of that owner before mask/check overhead. This is **not** a measured limit
phase or a promised speedup. The real repeated no-op work on most 108 keys
justifies one bounded integrated intervention, not a mapping/tile grid.

Before timing, exercise complete row ownership (prefix, both endpoints,
dense coordinates, independent and coupled contacts), empty/grow/reset mask
refresh, stationary versus violated/inconsistent limits, public zeroing,
unchanged contact schedules, and existing sparse-response CUDA tests. Reuse
the fixed paired whole-physics runner and accepted Keyboard recipe. Compare
complete physics and environment wall time with all capacity/source guards.
If the first integrated result misses the substantial-gain milestone, inspect
the existing owner timing/code once to explain whether the mask/check cost or
remaining contact work dominates; do not launch a parameter sweep or polish
a sub-milestone kernel.

## Implementation checkpoint

`FEATHER_PGS_STATIONARY_LIMITS=1` selects this owner; the default is off.
Unsupported owners reject the explicit flag. The original factory argument
lists remain unchanged when off; candidate-only typed wrappers append the
mask and have a `_stationary_limits` key suffix. Runtime omega overrides
other than one, nonfinite or nonpositive response/denominator, negative or
nonfinite CFM, nonzero subnormal velocity, and nonfinite enabled residuals
retain the original limit updates. Only local active bits change. Public
limit flags, velocity/impulse publication, budgets and the original global
early exit remain unchanged. Numerical equivalence is claimed, not signed-zero
bit identity.

The complete all-row ownership scan charges two endpoint reads per actual
row, plus head checks; atomics are issued only for still-empty heads. At
the unchanged 704-row capacity and 4K worlds, the worst-case endpoint read
volume is 22 MiB per solve (176 MiB across eight solves), not a latency
prediction. The mask is exactly 64 KiB at 4K, written and read each solve.
No contact geometry, capacity, producer or launch is removed or added.

Regression-first: the new factory test failed on the base runtime with
`TypeError: unexpected keyword argument 'stationary_limit_dofs'` (session
27114, exit 1), before source edits. The corrected factory and existing
mixed-response CPU control then passed. The existing response-diagonal,
sparse-response-owner and compact-contact modules passed 11 CPU controls;
16 CUDA controls were skipped with CUDA hidden (session 10875, exit 0).
These are not CUDA physical passes.

Focused native tests cover an odd five-world mask owner, prefix-only and
tangent-only endpoints, later coupled/independent overlap, dense exclusion,
active schedule/link preservation, and captured-builder replay through
mixed, empty, growing and shrinking row counts on the same storage. The
native solve compares the original recurrence in eager and graph execution,
including stationary and violated/inconsistent limits, stale public impulses,
single/disabled bounds, runtime omega, zero CFM, and fixed invalid response
and denominator fallbacks. GPU qualification and integrated performance are
pending root-owned execution; no gain or promotion is claimed here.
