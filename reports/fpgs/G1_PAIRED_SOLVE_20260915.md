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

## First native and whole result — loss, not promoted

Three CPU controls pass. All three focused CUDA selectors pass on both cards:
sixteen saved actual current/held cases; graph reuse/open/stick/sliding/scalar
fallback; and seventeen mixed worlds with complete17/18 support, count changes,
nonidentity group/art/world maps, singular blocks, failed halves and odd tail.
The first third-selector attempt failed BEFORE native execution because its
new fixture supplied a1D support table to a2D struct field. One-line fixture
correction e2f616478fba08a77c7aa6df9fd01a378afd8db4 fixes that binding. Runtime
remains unchanged from036762ae. No physical tolerances were relaxed.

First integrated paired16K graph comparison completed16:38 UTC, within the
45-minute budget. One discovery round, NOT accepted throughput:

| GPU | Retained physics ms | Paired physics ms | Retained/paired |
| --- | ---: | ---: | ---: |
| RTX PRO6000 |15.610799|18.741587|0.832950x|
| GB300 |20.437277|22.294128|0.916711x|

Whole environment wall time is29.534182 ->33.212016 ms RTX and34.924769
->36.683499 ms GB. Physics and environment wall scope remain separate; neither
table is a new MJWarp comparison. Both source/final-idle checks and original
capacity/finite/private-owner checks pass, including actual paired solve key.
No row, support or max-buffer budgets changed; no external process interrupted.

Full manifest: `/tmp/fpgs-g1-paired-solve-paired16k-20260915-01/manifest.json`,
SHA256e33d606fcf4fd0bf07ec3d94d100efe7c37a7a6cea5eddee2199391736d3b062.
It contains exact commands, environments, source pins and all per-card results.

## Same-input causal check, without privileged profiling

The reused untimed observer runs both original and paired kernels on exactly
the same retained last-solved inputs, with private impulses/velocity/status.
Thirty alternatingAB/BA turns discard the first ten. Graphs contain only the
solve; private resets occur before events. Every live input/output and plan
hash stays unchanged. Same-kernel eager/captured/final bytes match exactly.
Both cards return finite outputs, exact status agreement and ZERO numerical
difference in all16K impulse/velocity outputs. This rules out differing
trajectories as the cause of the measured isolated-kernel regression:

| GPU | Original solve ms | Paired solve ms | Paired/original time |
| --- | ---: | ---: | ---: |
| RTX PRO6000 |0.484096|0.896864|1.852657x|
| GB300 |0.577312|0.796432|1.379552x|

These are standalone kernel medians, not per-environment physics times. No
hardware counters, sudo or driver changes. The diagnostic data is under
`/tmp/fpgs-g1-paired-replay-run-wQfIcNQ1/gpu{0,1}`; both root sessions exited0
and final source/idle guards passed. Raw NPZ SHA256:

- RTX:1821debb9f002dfa8ac894608ef3e332dbeb6f4158df10f60d9cb750d6fc78df
- GB:6a5ebb234063ae423f3911608e7b012b48ae755127329dc1382066f387d113f2

Register-only recovery has no complete cost case: even an ideal32->48-world
residency benefit of1.5x leaves RTX1.2351x slower than original. This is a
conditional capacity thought experiment, not a measured occupancy model.
Count-only pairing also does not align executable work: accepted contacts
advance three rows per transaction, scalar limits one; root/zero-radius and
stationary-exit paths differ. Sampled17-row worlds already span9–13 transactions
per sweep and1–8 sweeps. Full-population metadata is retained for that causal
check; no performance percentage is assigned to divergence without evidence.

Full same-input metadata closure verifies the exact count/tie sort and current
group->art->world mapping. Of8192 sorted pairs,6549 RTX (79.94385%) and6443 GB
(78.64990%) have different limit-prefix lengths. Even among equal-total-row
pairs,60.8791% RTX and58.1281% GB differ in limit prefix. Prefix difference
median1/p954 on both cards; maxima8 RTX/7 GB.

Do not equate every scalar transaction with a limit: singleton normals also
occur. RTX has118829 limit rows +11781 singleton normals +3*65881 eligible
triplets =328253 rows; GB118882 +11205 +3*66386 =329245. Every observed triplet
has eligible metric metadata. This classification is not evidence of row drops.
The sampled clock-derived scalar count includes limits AND singleton normals.
Row-count packing proxies1.894995 RTX/1.895655 GB become1.882765/1.882448 for
scalar-plus-triplet transaction counts. That small proxy change does NOT
explain the measured1.8527x RTX slowdown. Prefix/path mismatch is widespread,
but its timing contribution remains unquantified. Actual compiled residency
cost, additional secondary-support/stripe work and path divergence are distinct
effects; no hardware-counter attribution is asserted.

Close count-only paired worlds as NO-FUND for the current large-gain milestone.
Do not promote it or fund a register cap, mapping grid, longer qualification,
or a register-lifetime-only rescue. A materially different representation or
scheduling proposal needs a new complete cost case. This is not proof that all
paired-world architectures are impossible. Retained ca0d, fixed Lab and original
worktrees/pointers remain unchanged. No accepted solver gain or new MJWarp ratio.

## Exact local reproduction

Use a NEW clean worktree at the measured runtime e2f616478fba08a77c7aa6df9fd01a378afd8db4;
do not reset any original worktree. The packaged scripts accept that runtime
path and fresh output paths, preserving the fixed Lab53ee checkout/interpreter
and retained ca0d baseline. They require the documented local frozen benchmark
adapters/manifests; this is a local handoff, not a portable asset distribution.

- `bash reports/fpgs/REPRO_G1_PAIRED_WHOLE.sh FRESH_OUTPUT CLEAN_E2F_WORKTREE`
  repeats the same paired two-GPU whole recipe through the original checked driver.
- `uv run --no-project --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python python reports/fpgs/REPRO_G1_PAIRED_SOLVE.py GPU_INDEX FRESH_OUTPUT CLEAN_E2F_WORKTREE`
  repeats only the diagnostic. Launch GPU_INDEX0/1 simultaneously with the same
  fresh output root; each gets a distinct child directory.

The committed diagnostic helper is byte-identical to the measured helper,
SHA2562a605df016f06c6f8da5a7c8d8b748cbba1d4b0d4098aeef41b1185734ab9539.
Packaged runtime/output-path CLI adaptations have not themselves been GPU
rerun; the exact measured launcher remains `run.py` in the artifact directory,
SHA256708b9616fa727c7b934d893b7a931824115adafdcaf492d3c1e3ed8688ff2d0f.
