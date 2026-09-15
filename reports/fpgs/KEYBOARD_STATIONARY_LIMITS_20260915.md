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

## Closure: measured loss on primary RTX, not promoted

Runtime/test source remained frozen at
`fa0b41f8e1bc270a501c294ccc2734408982b0d1` throughout qualification and
both paired captures. This closure changes this report only. The accepted
baseline remains `b1bad06ae0cd5b86ac07b2f2844ef62770e99075`; there is no
composition into it and no default flag change.

### Native qualification

The complete response-diagonal unittest module ran on each actual card:
14/15 methods passed on RTX and 14/15 on GB300. All three new methods passed,
including captured-builder refresh and eager/captured native solve behavior.
The unsupported h/CFM cases test preservation of original fallback and
nonfinite propagation, not physical validity of those inputs.

Both module runs retain the same inherited error in
`test_tiled_response_diagonal_matches_dense_reference`: its unchanged
`accumulate_group_diag_worlds` call supplies seven arguments while the kernel
requires nine. The seven-argument call is also present in the accepted
baseline test source. This unrelated test was not repaired. The module runs
therefore exited nonzero; they are not reported as an all-green suite.
Actual CUDA compilation and execution qualified the new native entries;
no offline AOT run was performed or is claimed.

Logs at `/tmp/fpgs-keyboard-stationary-native-qGrY3s3m`:

- `gpu0.log`: SHA256 `1f4aca749d58f085b1bfbf7e709242a1717cd131b7feed50169bae1ed473c743`.
- `gpu1.log`: SHA256 `da5a51b6b395919608d26a49d5d6e8a6dbfd5b57e832e753bde3b7e41f329f0f`.

### Whole-physics screen

One paired discovery round, 4,096 worlds, seed 0, 200 warmup, 40 synchronized
environment steps and 40 graph-profile steps. Values are ms per batched
environment step; speedup is accepted baseline / candidate.

| GPU | Baseline physics | Candidate physics | Candidate minus baseline | Speedup |
| --- | ---: | ---: | ---: | ---: |
| RTX PRO6000 | 7.443361925 | 7.662634675 | +0.219272750 | 0.9714x |
| GB300 | 6.654806375 | 6.591330550 | -0.063475825 | 1.0096x |

Synchronized whole-environment wall time also lost in this window:

| GPU | Baseline wall | Candidate wall | Candidate minus baseline |
| --- | ---: | ---: | ---: |
| RTX PRO6000 | 28.682788397 | 31.592757002 | +2.909968604 |
| GB300 | 26.512718076 | 27.700290200 | +1.187572125 |

Environment wall includes reset/host work, not an entire RL training update.
These single windows do not establish a causal explanation for the wall
deltas or a repeated improvement. The primary RTX physics regressed and the
0.7 ms whole-physics milestone was missed on both cards.

### Measured owner diagnosis, not another optimization

A separate matched three-step node capture confirms the actual candidate
keys and charges both the changed solve and the existing builder. Values
below sum each named physics-graph kernel's durations per environment step;
each owner still has eight launches per step.

| GPU | Owner | Baseline ms | Candidate ms | Candidate minus baseline |
| --- | --- | ---: | ---: | ---: |
| RTX PRO6000 | Sparse GS | 1.078966000 | 1.129783000 | +0.050817000 |
| RTX PRO6000 | Contact schedule/mask builder | 0.265621667 | 0.285674667 | +0.020053000 |
| GB300 | Sparse GS | 1.274047667 | 0.993769667 | -0.280278000 |
| GB300 | Contact schedule/mask builder | 0.283882667 | 0.309386667 | +0.025504000 |

The two-owner sum increased 0.070870 ms on RTX and decreased 0.254774 ms on
GB300. On RTX the no-op work retirement did not reduce the measured solve
cost, and ownership construction also added cost. On GB300 the solve saving
was real in the node window but insufficient to deliver the milestone in
the complete whole-physics screen. This is the diagnosed scope; it does not
identify an instruction-level bottleneck or justify a packing/tuning sweep.

The solve retained 94 registers on RTX / 88 on GB300, 14,664 bytes shared
memory, and zero reported local memory on both arms. Builder registers rose
28 to 34 on RTX / 26 to 32 on GB300; its shared memory remained 1,024 bytes.
The candidate retired computations but not the solve's resident row-state
storage or launch shape. The mask/all-row scan costs are included above.

Do not substitute the short node capture for the 40-step physics screen.
Its physics graph span was RTX 7.215184 to 7.216275 ms and GB300 6.702837 to
6.425226 ms. Node sums may overlap; these are neither exclusive critical-path
savings nor additive predictions for the separate graph-mode measurement.
Both arms retain four physics graphs and 420 device nodes per profiled step.
No micro-rescue or further timing grid was run. Candidate closed, unpromoted.

### Reproduction and source/capacity evidence

Both captures use unchanged fixed Isaac Lab
`53ee6b44c2334341305dbdf385a3916c6b140799` (only untracked `.venv`), Newton
baseline/candidate pins above, and the original comparison tools at
`961b7e2f751bcd1d8b03368e7956b54c81414897`. The fixed import adapter remains
`/tmp/fpgs-kinetic-fixed-variants-TZRkIPYw/run.py`, SHA256
`c062388f23f82c524690f272fb2d14141ed4126db641de6dcd86f8ade5bfe416`.
Full commands, environments, imported paths and file hashes are in each
manifest. The original graph protocol is unchanged; the diagnosis changes
only steps/profile-steps from 40 to 3 and trace mode from graph to node.

Both arms retain `FEATHER_PGS_SPARSE_CONTACT_DIRECT=1`,
`FEATHER_PGS_PRISMATIC_PUBLICATION=1`, and
`FEATHER_PGS_COMPACT_CONTACT_BOUNDARY=1`; only
`FEATHER_PGS_STATIONARY_LIMITS` differs (baseline 0, candidate 1).
Identical limits: dense/propagation rows 704, matrix-free rows 64, rigid
contacts 147,456, broad-phase output 57,344. Budget: sim_dt 0.01,
decimation 4, two Newton substeps of 0.005, eight GS sweeps maximum.

All eight child captures exited zero with complete passing available sticky
capacity checks and finite boundary states. Both final source/idle guards
passed. All 64 per-run recorded artifact hashes were rechecked successfully.
Available overflow flags and finite state are not a full convergence proof
or complete demand calibration; the manifests correctly leave performance,
full physical quality and repeated timing acceptance false.

- Whole capture: `/tmp/fpgs-keyboard-stationary-limits-whole-paired4k-20260915-01`, manifest SHA256 `54bdba20980aabb35697cec0722590d1c69672f19db5881ce694ef5f51256c2d`.
- Node capture: `/tmp/fpgs-keyboard-stationary-limits-nodes-paired4k-20260915-01`, manifest SHA256 `c06c74739508ff5955beeedc9fe7004ab357a469b7a6effdf5e972daad3b7f11`.
- Frozen solver source SHA256: `89da3d3fad2896f99d024c6235c25771389de9c8ee85858bad236b12d64a08a3`.
- Frozen response-diagonal tests SHA256: `f19b2ecb1e3d26567003e2592011e5bf2b10e421086765285f5f719c87f09189`.
