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
