# Unprivileged G1 solve diagnosis

Started 2026-09-15 15:51 UTC. Diagnostic only; no optimization or new backend
ratio is claimed. Retained Newton ca0d427af809571bb5501f644c1a6e03990cd2a8 and
fixed Lab 53ee6b44c2334341305dbdf385a3916c6b140799 remain unchanged. User requires
operation without sudo. No driver settings, privilege policy or external jobs
will be changed.

Reuse the original checked 16K G1 recipe and its first untimed boundary. Replay
the last solved rows through the original kernel and a separately generated
clock-only derivative, using private output/status storage and original cold
initial impulses. Preserve source, capacities, eight-sweep allowance, geometry,
held operator and physical equations. The diagnostic is never installed into
the simulation. Restore outputs before each standalone event-timed replay.

Use the existing marked-source-insertion method from the ANYmal diagnostic,
with a byte-exact strip-back check and no new synchronization barriers. Sample
lane-zero elapsed cycles and algorithm counts. Check numerical outputs against
the original same-input kernel, cycle conservation, unchanged live inputs and
instrumentation overhead. Save per-world records locally. These cycles are
within-world elapsed time, not hardware stall counters or additive GPU wall
time; replay timing is not whole-physics throughput. The original graph
ownership remains the whole-step cost reference.

First result checkpoint: approximately 20 minutes, before extending diagnosis
to other kernels or building another optimization. A costly diagnostic must
be identified as distorted, not used to claim a precise bottleneck fraction.
Any subsequent implementation still needs the existing complete-cost card,
large-gain threshold and original physical/performance gates.

## First paired native result, 16:02 UTC

Both runs complete without sudo (uid1002), child/parent exit0 and final
source/selected-device idle guards pass. Root sessions45625/71275 are reaped.
All original capacity/finite checks pass at both boundaries. The first-boundary
diagnostic replays all16K current last-solved worlds with private writes;
original and observed output bytes agree for eager, captured and final timed
launches. Every live input/output/plan digest remains unchanged.

Thirty alternating original/observed rounds discard the first ten. Captured
graphs contain only the selected solve; all private resets precede timing.
Median original/observed standalone costs are0.505440/0.718944 ms RTX and
0.555088/0.859232 ms GB, or42.24%/54.79% instrumentation overhead. This fails
the low-distortion requirement: do NOT treat observed phase percentages as
retained kernel costs or multiply them by whole GS time. Numerical equality
does not qualify profiling accuracy. These are not optimization ratios.

The256 fixed world%64 samples per card have consistent cycle and algorithm
accounting. RTX has7438 metric visits,2612 positive-radius transactions,
2072 sliding-root attempts,4866 probes,13718 scalar row visits and zero metric
fallback. GB has7185/2479/1887/4423/13439 respectively, also zero fallback.
Thus sliding roots average2.35/2.34 probes, not their maximum16. Of256 worlds,
187/185 exhaust eight sweeps. These are selected current diagnostic counts,
not a full-population or whole-trajectory claim. Sample rows average20.156/19.902
versus full16K19.981/20.094. Detailed clocks average236/229 reads per sampled
world and increase compiled state even for unobserved worlds.

Artifacts: `/tmp/fpgs-g1-unprivileged-phase-cHINj73W/gpu{0,1}`. Each contains
checked boundaries, parent manifest, current recipe metadata, log and phase.npz.
Raw phase SHA256 is b5cda3fe344b4e9f88b9a48f488906046ada3ef38b3c7ff401b4c434cbcf8528
on RTX and56c68a401b992143add162bb77ff8abca681e40ec68da9708ea09c370912d307 on GB.
Clock factory SHA25625563bf9fdb0974d6958222e5e7bc625d31f4dce4f61c6abf2aebe50d63a538a;
live observer02c0c16ee76580f657d235d76cb6c4742e9d2a1e3097c914d7e6b84e3c62e81f.
CPU factory source/ABI/strip/seam rejection controls pass, as do standalone
ruff checks. Numerical source and original barriers are exactly recoverable.

One corrective diagnostic is funded: compile-time selection of one timed
region with only elapsed/selected clocks and no18-field per-row counter state.
Measure its complete overhead again. This is not a phase/tolerance/register
tuning grid, and it does not change physics. Resource evidence and the final
distortion gate must precede any performance interpretation.

## Corrected unprivileged result, 16:19 UTC

Binary-region clock32 timing succeeds on both cards, with only0.52--1.21%
whole-kernel instrumentation overhead. All original checks, private replay
output-byte checks, sampled bounds and live-array/plan digests pass. Parents
65611/14006 exit0 and are reaped; both selected GPUs pass final idle. No sudo,
driver change, hardware-counter-policy bypass or external-job interruption.

| Region | RTX original / observed ms | Overhead | GB original / observed ms | Overhead |
|---|---:|---:|---:|---:|
| Complete scalar rows |0.524288 /0.529024|0.90%|0.547376 /0.554016|1.21%|
| Metric disk calculation |0.525744 /0.528496|0.52%|0.546528 /0.550560|0.74%|

On the256 sampled worlds, scalar rows account for35.89% RTX /33.85% GB of
summed within-world elapsed cycles. The metric disk calculation accounts
for12.30% /15.37%. The latter includes conditioning, the sticking test,
sliding roots and final verification, NOT roots alone. These are separately
measured sampled elapsed-cycle fractions, not additive whole-physics wall
fractions or hardware stall counters. No memory-/compute-bound or peak-CUDA
claim follows. Original source-index setup and final diagnostic writes are
outside the internal total; complete kernel events include all of them.

The concrete observer perturbation is supported by offline assembly of exact
cached NVRTC PTX: original72registers versus clock18 107RTX/108GB; both binary
versions return to72 on both architectures. Shared1116B and zero register
spills/stack remain. This is ptxas13.2 static evidence, not driver-loaded
resource telemetry or a measured causal fraction of the initial overhead.
The binary clock uses unsigned32 differences, checks selected<=total and
total<2^31, and rejects replay samples>=100ms; no long-interval claim is made.

Binary artifacts: `/tmp/fpgs-g1-unprivileged-binary-PehdrvlV/gpu{0,1}`.
The exact source-pinned local launcher is `run.py` in its parent; both0/1
arguments were launched simultaneously, one job/card, as normal uid1002.
The observer runs inside the original first untimed checked boundary and is
never installed into the simulation. Region3/5 raw NPZ SHA256 values:

- RTX scalar:1c9ef557f5c4f04dcebf4c0a94371b4e0abe4a4658df34b6c62631f4aa586812
- RTX disk:446a78a4ed03631419776b50df86314963143c4d704f0318185f2119718c326f
- GB scalar:a41f899c52695fd5f8855b0e4ae37c9864f96ee15287704aaba67f137abf219c
- GB disk:dd2a5d03e007445c8ae1b70402bf67fe0a0b689e64c1dc22cff2c51e73f5973f

Final binary helper SHA256 is
da12069cd62e4b0399ffdd270e4d2ae0a64d3723826ab42a85c50a8a70b7a279;
observer7a84f1b0cef3730fc36e6923ca843169d6e386d60501e5382d30d20d37820ed1.
The original detailed helper remains25563bf9; original measured observer bytes
are preserved by commit9da957ef. CPU factory and independent seam/ABI reviews
pass. One import-only correction makes the new binary helper load from its
diagnostic directory while Newton imports remain pinned to retainedca0d.

Profiling without sudo is now operational. This is diagnostic progress, not
a solver speedup. Hardware-counter restrictions do not block the next
optimization. No further profiler rewrite is funded here. The next experiment
is the separately carded paired-world G1 solve, not a friction-root micro-tune.

Local reproduction is included as `REPRO_G1_UNPRIVILEGED_20260915.py`.
Invoke with the fixed Lab interpreter through uv, arguments0 or1 and the same
fresh output-root path; launch one invocation per card concurrently. This
packaged copy adds an explicit output-root argument to the measured launcher
and has not itself rerun the GPU study. Its source/recipe checks and selected
device idle guards are unchanged; populated per-card output directories are
never overwritten. It requires the preserved local source-pinned manifest and
checked driver, not a new environment install or sudo.
