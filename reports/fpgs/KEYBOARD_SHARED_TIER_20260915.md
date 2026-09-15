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

## Closure: qualified native routing, insufficient whole gain

**Not promoted; no further tier grid.** Runtime remains the measured
`995e4ed8b0f686b875caa5f8eeee59a195c1f514`, solver SHA256
`94ce8a2cd08a08cfebba14034845db071d024d921ea8ddd2c5bcb91e0850b7d9`.
This closure adds only the focused test and these results. The accepted
baseline remains b1bad; increased driver residency did not deliver the
predicted complete-owner saving.

### Native checks and preserved fixture failure

Four focused CPU tests pass: unchanged generated math/public strides and ABI,
default-off/unsupported admission, the actual fixture's required call signature,
and ready/fork/finally-join ordering. The five-world native fixture uses counts
0/383/384/385/704, reversed dense groups, independent and coupled contacts,
loaded boundary triples, empty-world limits, and captured growth/shrink/reset.
Current-stream copies after the real solver method check downstream consumers.
Floating comparisons retain the inherited rtol2e-5/atol2e-6; schedule and
read-only-input checks remain exact.

The first paired native run passed both inherited physical oracles and every
original-versus-tier output/schedule/consumer comparison reached, but failed an
unjustified fixture expectation that every last normal impulse be positive.
On reversal, world1 has385 rows/prefix1: its first dense limit sets v0 near.1,
v1 remains -.1191564, and the key limit sets v6 near.15. The final normal's
residual is +.03766257, so its cold impulse correctly stays zero for all eight
sweeps. A CPU replay confirmed this. Only that case now explicitly expects
zero; positive loaded coverage elsewhere, fixtures and tolerances are unchanged.
The earlier missing required `mf_meta` fixture argument was corrected before
GPU execution and is covered by the CPU signature test.

Failed logs remain at `/tmp/fpgs-keyboard-shared-tier-native-s5xIqpXM`.
The exact failed test is preserved at
`/tmp/fpgs-keyboard-shared-tier-tests-Ziyox1pw/failed_8f79ae4b_test_sparse_shared_tier.py`,
SHA256 `8f79ae4b941d07e9e503ab5dad187da8eebca64f33b4f362f261c23001eb700a`.
The corrected external test SHA256 is
`2fbb49b0bccba9809b52dd55694a623c6344392310912cad818f31b98791edc1`.
All three native selectors subsequently pass on each GPU: RTX 0.138s, GB 0.173s,
without skips. Corrected logs under
`/tmp/fpgs-keyboard-shared-tier-native-corrected-v2Dw1234` have SHA256:

- gpu0: `78cd10f88fcbb1493d446ddee41c8dcbdc999f1175bd97041a5d8f502ab5b288`.
- gpu1: `74e2fa184d31b6b9de8e866d5640a74449639113cc5f0da37a032a793243ccc7`.

### Whole screen and separate node diagnosis

Milliseconds per batched environment step, accepted -> candidate:

| Measurement | RTX PRO6000 | GB300 |
| --- | ---: | ---: |
| Whole physics, 40-step graph screen | 7.852563150 -> 7.757965125 | 6.805157400 -> 6.715505425 |
| Synchronized environment wall | 30.621105 -> 31.789279 | 27.988174 -> 28.266865 |
| Separate three-step node graph span | 7.432094 -> 7.564035 | 6.758763 -> 6.929589 |
| Original GS, node sample | 1.112650667 | 1.353609667 |
| Candidate bulk384, node sample | 1.185835000 | 1.425930667 |
| Candidate tail704, node sample | .893557667 | .908533333 |

The whole screen saves only 0.094598025ms RTX/0.089651975ms GB, far below the
structural target and without reaching the historical RTX 4x ceiling. This is
one discovery round, not repeated evidence. Node timing is a separate window:
**bulk and tail overlap and must not be added**. Even the bulk alone is slower
than original GS in that sample. The resource gate passed, but the complete
dispatch did not reveal a substantial hidden gain. These observations do not
identify a particular hardware stall or prove achieved occupancy; no counters
or additional tier variants were run. Environment wall and physics are distinct.

### Reproduction and guard scope

Reuse `/tmp/fpgs-kinetic-fixed-variants-TZRkIPYw/run.py` (SHA256 prefix c062388f)
with the original settings and exact child commands/environments in:

- `/tmp/fpgs-keyboard-shared-tier-whole-paired4k-20260915-01/manifest.json`,
  SHA256 `6d80dbf50b2fdca949b9710603dd290a8a6a1520eef517645d98032ecbe79ad8`.
- `/tmp/fpgs-keyboard-shared-tier-nodes-paired4k-20260915-01/manifest.json`,
  SHA256 `c69b9392fa7cc230b9e35ebbb9d9d2d7d4a4d3674d4b1a1c34f7b1610d6eec9d`.

Both use 4096 worlds, seed 0, 200 warmup, fixed Lab contact-reset worktree,
sim_dt 0.01, decimation 4, two 0.005 substeps and at most eight GS sweeps. Whole
uses 40 wall/40 graph steps; the node run records 3 wall/3 node-profile steps.
Caps are dense/propagation 704, MF 64, raw contacts 147456 and broad output 57344.
Both arms retain sparse-contact-direct, prismatic-publication and compact-contact
flags 1, group lanes 16, masked rows 1 and narrow-phase threads 4; only shared-tier
changes 0->1. No Lab, capacity or physics-budget edits occurred.

Each screen completed all four children and eight capacity boundaries, with
finite state and final source/idle guards passing. Root reaped 37155, 14083 and
79028. Available sticky checks do not establish universal convergence or complete
collision-demand calibration. No repeated timing, backend-ratio update, or
performance promotion follows from this experiment.
