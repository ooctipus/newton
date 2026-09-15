# G1 paired-world solve candidate

Funded 2026-09-15 16:22 UTC, from retained ca0d427af809571bb5501f644c1a6e03990cd2a8.
Default-off Newton-only experiment, not a claimed speedup. Fixed Lab53ee6b44,
16K worlds, seed0, two substeps, eight sweeps and all calibrated capacities
remain unchanged. No parent pointer changes. RTX primary, paired GB when idle.

## Complete hypothesis and budget

Replace the single32-lane/world metric43 solve with two16-lane worlds/warp.
Each lane0/1 also owns support element16/17 when present; pre-adding those
products reproduces the original first shift16 pairing before shifts8/4/2/1.
No support truncation or skipped rows/impulses. Preserve current held W/Z,
normal-first metric tangents, scalar fallback, original row order inside each
world, stationary exit, all statuses, final impulses and complete43 velocity.

Use a fixed256-thread CTA containing16 worlds. One initial16-key local sort
by current row count pairs similar workloads, followed by one CTA join before
any tail-world return. No global queue, sorting producer, conversion or extra
kernel launch. Every subgroup owns isolated du43/lam100/contact caches and
correct16-lane synchronization masks. Current group/art/world maps and row
strides remain authoritative. All current row/factor producers stay unchanged.

Fresh full16K row-count proxy (sum counts / sum paired maxima) is1.5992RTX/
1.5957GB for adjacent pairs versus1.8952/1.8958 after sorting each fixed16-world
tile. These are row-count-only ratios, NOT time or sweep/branch predictions.
Sources: `/tmp/fpgs-g1-unprivileged-phase-cHINj73W/gpu{0,1}/phase.npz`.
The paired metric solve is distinct from the closed Gram, coordinate and packet
owners, which retain one32-lane world and change the representation instead.

Measured retained GS owns3.918784ms of15.685799ms RTX whole physics. A first
10% milestone requires >=1.568580ms saving, so the complete new GS owner must
fit approximately2.3502ms (>1.667x), with no excluded setup/publication cost.
The ideal2x limit leaves only0.3908ms above half-cost. This is a tight,
falsifiable target, not a forecast; it does not itself meet4x versus MJWarp.

Mutable solve scratch is16*988=15,808B plus64B permutation/compiler overhead.
At256 threads, <=80registers/thread conditionally permits three CTAs/SM under
the observed architecture limits;81 rounds to88 and permits only two. This is
a compiler/residency budget, not achieved occupancy. Extra secondary support
values, three43-coordinate decode stripes instead of two, row setup stripes,
divergent root/early-stop paths and slowest-warp CTA lifetime can erase gains.
No register cap, tile grid or physics relaxation will be used to hide that.

## Minimal gates

Check actual current/held G1 physical cases using the existing native tests;
exercise17/18-element support, odd/tail groups, mixed row counts, empty worlds,
fallback, within-pair early exits and graph replay. Validate physical results,
not arbitrary bit identity. CPU source/packing controls and an unchanged
original-kernel oracle precede the first whole paired16K discovery.

First native/whole checkpoint within45minutes. If theory and timing disagree,
use exact resources and same-input replay/strict existing nodes to locate the
loss, allowing one cause-specific correction. Do not expand qualification or
tune a losing mapping without a new complete cost case. No profiler privilege
or driver change is needed; unprivileged activity/event/low-overhead software
timing is sufficient for the planned first decision.

## First implementation resource check

Actual offline CUDA compilation passes for sm120 and sm103:94/92 registers,
16,896 bytes shared, no stack/spills, one CTA barrier. Thus the planned
three-CTA residency threshold is NOT met; only two CTAs are conditionally
admitted by register capacity. Do not interpret the count proxy as achieved
throughput. The first physical/native and integrated screen will still run.

Exact disassembly places the paired peak at87 live GPRs versus63 original,
around the metric root sqrt/div slow-path calls, not initial sorting (57 peak).
Secondary coefficient/address values and per-slot shared addresses stay live
across that branch. Reloading after the root is a potential lifetime tradeoff,
not an approved fix or a demonstrated route to<=80 allocated registers.
No register cap, block-size grid or additional runtime variant was tried.

Resource artifact: `/tmp/fpgs-g1-paired-offline-gkaix2ba/actual02/report.json`,
SHA256279c8cc9a8d32ad29ba003ab118996083fe4cf8b3ef6528fe0c0b207128dad5d.
Source review covers isolated half-warp masks/votes, complete support17/18,
sort/tail join, mapped group/art/world addresses and exclusive admission.
Pre-addition can legally permit different FMA contraction; physical tolerances
remain authoritative, not an assumption of bit-identical floating-point trees.
