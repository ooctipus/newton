# Keyboard shared row specialization: one integrated candidate

Decision: 2026-09-15 08:42 UTC, before code.
Base: accepted `b1bad06ae0cd5b86ac07b2f2844ef62770e99075`.
This branch contains neither the stationary-limit mask nor read-only-row loads.

## Evidence and hypothesis

The read-only-row trial actually reduced shared memory from 14,664 to 3,400 B
and raised driver residency ceilings from 6/14 to 20 blocks per SM on RTX/GB.
It nevertheless lost on RTX whole physics, 7.428473 -> 7.777718 ms. The separate
node sample locates a GS increase from 1.287766 to 1.448033 ms; there is no
additional producer or launch. Removing shared reuse introduced repeated
global reads. That source-level cost is real; its cache-hit/stall distribution
has not been measured. Increased theoretical occupancy alone was insufficient.

Test a distinct remedy: retain ALL original shared row reuse and arithmetic,
but separate the public global row stride/capacity (704) from one private
shared-row capacity (384). Count-admitted worlds execute the smaller owner;
larger worlds execute the original 704-private-capacity owner. Both owners
are disjoint before any input staging or output write. Empty worlds belong
to the small owner and still perform fused joint-limit work.

Four hash-verified historical 4K payloads show that 384 rows cover 99.66% of
RTX worlds and 97.62% of their rows (99.07% / 93.77% on GB). They are historical
adjacent checkpoints, not independent population or elapsed-time evidence.
All public row, contact, broad-phase and propagation capacities remain fixed;
no contact, tangent, parent, group link, prefix or impulse may be omitted.
Global M=704 and serial-normal stride235 must remain unchanged in BOTH owners.

This is not the closed private-island160 design, which added private J/Y
panels, or a cache-policy/tile-size grid. There are no new J/Y arrays,
classifiers, queues or global metadata producers. One fixed384 tier only.

## Complete cost and falsification

Private static shared storage is expected to be 8,264 B. Driver reservation
and allocation granularity may limit RTX to only ten blocks per SM, not the
unverified eleven/twelve estimated from incomplete arithmetic. Root queries
the actual compiled function before whole timing. Register changes and
achieved execution efficiency remain empirical.

Using the earlier 1.078966 ms owner, ideal 6-to-10 proportional scaling gives
0.647380 ms bulk time. To save 0.372304 ms and bring the recent 7.428473 ms
whole baseline to the historical corrected-MJWarp 4x ceiling of 7.056169 ms,
only about 0.059282 ms remains for fork/join, extra launch, exposed tail and
interference. The newer short-node owner is 1.287766 ms; that ideal model
leaves approximately 0.142802 ms. Neither estimate is a guaranteed subtraction
from whole time. They expose a narrow, falsifiable path, not a promised gain.

Avoid serially exposing the rare heavy tail. Reuse the existing size6 side
stream after all required producers and v_out initialization have joined.
Use existing otherwise-unused stable size events for ready/done ordering.
The disjoint full704 tail and384 bulk may overlap; join the tail before
returning to any impulse/velocity/publication consumer. Complete cost is
max(bulk, tail) plus fork/join and interference, not bulk alone. No new stream
or device-wide synchronization. Explicit unsupported opt-in must reject;
the original default-off dispatch remains unchanged.

## Qualification and stop rule

Default-off `FEATHER_PGS_SPARSE_SHARED_TIER=1`, distinctly keyed factory
variants and original native argument ABI. Preserve all original row order,
friction projection/delay, limit updates, exact stationary-sweep check and
iteration allowance. Regression-first factory/dispatch checks; reuse existing
scalar/speculative physical oracles. Focused native eager/captured controls
must exercise nonzero world strides, empty worlds, both sides of384, a valid
parent/tangent triple near the boundary, full704 fallback, changing counts,
independent/coupled contacts, and both output owners. Test actual ready/join
ordering and reuse across graph invocations. Do not build another framework.

Root then runs the unchanged paired4K whole-step driver: seed0,200 warmup,
40 wall/40 physics steps, fixed Lab, dt/substeps/eight sweeps and original
capacities. A first screen is not promotion. A substantial repeated gain,
especially one that actually reaches4x, may qualify even if it misses an
internal funding milestone. No physical tolerance or task parameter changes.
New backend ratios require matched denominator evidence.

First integrated checkpoint within90 minutes; no silent extension beyond
two hours. If resources do not improve as intended or complete timing misses,
diagnose with the existing node view; no tier-size/stream-policy grid or
unpriced fallback story. Accepted worktrees and the retained Franka branch
remain unchanged during this experiment.

Status: one candidate approved; runtime and native evidence pending.

## First implementation and resource checkpoint

The candidate changes only the existing solver factory and its host dispatch.
`FEATHER_PGS_SPARSE_SHARED_TIER` remains default off; explicit opt-in rejects
anything outside the original CUDA 704-row, 114-coordinate, 108+6 speculative
diagonal owner with parallel size streams. Its existing admission already
requires the original fused-limit and complete matrix-free solve contract.

The private384 owner handles clamped counts at most384, including empty-world
fused limits. The private704 owner handles counts385 through704. Both count
guards precede every other per-world input read and output write. The original
704 clamp, all global704/serial235 strides and native argument ABI are retained.
The host forks the tail on the existing size6 stream with stable size6/108
ready/done events, launches the bulk on the current stream, and joins in a
`finally` block before any downstream consumer. No new stream, event, global
buffer, producer or queue is allocated.

Regression-first CPU invocation session92387 exits1 before implementation:
the original factory rejects the unexpected `shared_row_capacity` keyword.
An independent generated-source comparison against b1bad then passes for all
three original variants (ordinary, contact-group and speculative). The
production default-off snippet SHA256 is
`0b9f4f0ab41be724c05a7abdd0672509aac39c9f6b7c00d4a646b40a89d69017`.
The native signature and Warp wrapper AST are unchanged. Each candidate
snippet differs from the original only in the five shared row-array bounds
and the uniform post-clamp count guards; the guard insertion checks its exact
single source seam. No floating-point expression or recurrence is changed.

Root-owned actual CUDA compilation and driver queries complete on both cards:

| Owner | Static shared B | RTX registers / maximum blocks per SM | GB300 registers / maximum blocks per SM |
| --- | ---: | ---: | ---: |
| Original704 | 14,664 | 94 / 6 | 88 / 14 |
| Bulk384 | 8,264 | 92 / 10 | 86 / 20 |
| Tail704 | 14,664 | 92 / 6 | 86 / 14 |

All use32 threads, zero dynamic shared memory and preferred carveout -1. These
are actual compiled-function residency ceilings, not achieved occupancy or a
throughput result. The minimum planned ten-block RTX resource gate passes.
Logs under `/tmp/fpgs-keyboard-shared-tier-resource-mlG4SwdP` are preserved:
`gpu0.log` SHA256
`797d6ed7e536b865a8f03643274910121d52ca00d447fb51c685d088256d5e22`,
`gpu1.log` SHA256
`a1140ac594b7b80f90dc2b9d8744de1b17d3dda8d5370c337bf03fc6fb7081e9`.

The inherited response-diagonal module passes its one CPU selector, with11
explicit CUDA skips (session88314). A broader37-selector CPU run passes25,
skips9 CUDA selectors and has three CPU-incompatible failures (session4573):
`test_articulated_contact_response_validation`,
`test_fuse_joint_velocity_limits_validation` and
`test_fixed_base_sibling_branches_use_diagonal_mass_path`. The exact same two
CUDA-required errors and diagonal-admission assertion failure reproduce on
untouched b1bad (session87423). These failures are preserved, not relabeled
as passes or repaired in this experiment. Focused tier tests and root-owned
native/whole qualification follow; no candidate timing is claimed yet.

### Early unqualified cost screen

Root authorizes the first unchanged whole-step timing screen immediately after
the source/ABI proof and actual compiled-resource checks above. The focused
five-world tier-boundary and fork/join graph tests are still being prepared
outside this frozen benchmark tree. Their absence keeps numerical qualification
open: the screen may establish a cost loss or motivate completing validation,
but cannot establish physical acceptance or promotion. Existing native
scalar/speculative and focused complete-owner tests remain mandatory before
retention. Freeze every source, test and report byte through root's whole-run
reap; no benchmark-tree edits are permitted during the screen.
