# Representative-task timing extension

Requested 2026-09-13 05:55 UTC: extend the current measurements to G1,
ANYmal-D and SO101 Keyboard. This is a measurement and generalization study,
not evidence that the Kuka prototype already accelerates these tasks.

## Scope and decision

The experimental kinetic owner requires the exact Kuka trunk/four-finger
topology, 23 primary response coordinates and a six-coordinate free body.
Its contact-triplet producer and compact coupled solve cannot currently be
enabled as a valid replacement on these three tasks. Unsupported dispatch is
not an optimization result. Do not carry the Kuka 1.25x/1.16x section ratios
into another task or multiply them into an MJWarp comparison.

First recover each task's capacity-checked recipe, measure current FPGS versus
corrected MJWarp on both GPUs, and identify the actual existing solver path.
Use the existing Newton comparison driver and unchanged Isaac Lab harness.
Keep physics-graph, auxiliary sensor graphs and environment wall time separate.
One paired backend round is discovery, not repeated optimization acceptance.
Generalizing the new representation needs its own implementation and physical
checks before a baseline/candidate speedup can be reported.

The earlier Kuka live-integration 06:28 checkpoint is reprioritized by this
explicit user request. Its native/binding/lifecycle implementation remains
experimental and frozen, apart from separately diagnosed benchmark-guard fixes.
No live CUDA or full-task performance acceptance is implied by the CPU tests.

## Sources and first run

Runtime for the initial survey is clean Newton
`50dfa28d3aabe51f1b5b75721450efac2c35c5f1`, from
`/home/octi/Projects/newton-fpgs-composed-world-20260912`, in the ooctipus fork.
Isaac Lab remains clean at `1d8feb82d17dbfab8f0772de56f84deae2cb7974`.
The existing comparison driver's 44 CPU tests pass before launch.
No Lab source, timestep, substeps, iteration allowance or dependency pointer
changes. Device ownership stays one child per RTX PRO6000/GB300 concurrently.

The first launched batch is Keyboard4K, seed0, 200 warmup, 1000 synchronized
environment steps and 40 physics-graph steps, one backend round. It enables
the existing sparse-contact, prismatic-publication and compact-contact-boundary
FPGS flags, not the new Kuka kinetic flag. MJWarp uses the explicitly recorded
process-local line-search correction. Both backends use the same Newton source.

Established capacities are FPGS704 dense rows,147456 raw contacts and57344
broad output; MJWarp320 rows, pooled nconmax32 (131072 contacts) and45056
broad output. These are the previously calibrated per-task/world-count
settings, not newly chosen buffer inflation.

Raw run: `/tmp/fpgs-cross-task-keyboard-20260913-01`.
Its manifest records exact commands, flags, source hashes, process groups and
per-boundary checks. No timing is accepted while that manifest is incomplete.

ANYmal's established MJWarp16K capacities are32 rows, nconmax9 (147456
contacts),147456 Newton contacts and294912 broad output. The FPGS recipe and
its prior capacity evidence are being verified independently. G1 has not yet
been assigned a validated compact-capacity recipe; calibration must precede
any claim that its buffers are appropriately sized and drop-free.

## Keyboard discovery result

The four children complete with exit0 and checked boundaries passing; final
Newton/Lab source trees remain clean and both GPUs are compute-idle. Parent
manifest SHA256 is
`35072ab2506cc4be06aa46f32478c1ebcb2c845a58696f5cdec807f7a54dbc9f`.

| GPU | FPGS physics ms | MJWarp physics ms | MJWarp/FPGS | FPGS wall ms | MJWarp wall ms | Wall ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX PRO6000 | 6.966523 | 27.761017 | 3.98492x | 33.819385 | 32.134956 | 0.95019x |
| GB300 | 5.983209 | 27.985320 | 4.67731x | 29.633038 | 32.250968 | 1.08835x |

This is one fresh backend discovery round, not a repeated new optimization
gain or cross-backend physical parity. The Kuka kinetic path is not enabled.
Keyboard still has different reset/contact behavior between backends; a
physics ratio is not a training-throughput ratio.

## ANYmal calibration and discovery result

Full-run FPGS calibration completes on both GPUs for seed0 and held-out seed1:
16384 worlds,200 warmup,1000 steps x3 repeats and40 profile steps, including
12960 actual collision calls per run. Raw outputs:

- `/tmp/fpgs-anymald-capacity16k-20260913-01`: seed0,192 dense rows.
  Manifest SHA256 `59da9c805378fe0c4fd00d2e3ca7f2f6a8f9c3aa05a942cb3c7fc5b3abcf8b99`.
- `/tmp/fpgs-anymald-capacity16k-seed1-20260913-01`: seed1,72 dense rows.
  Manifest SHA256 `3125370a3d0032759abd0aff0eec481a31e780a9c3e91548d28843b178bac6a3`.

Across both devices/seeds the observed maxima are60 dense rows,176144 raw
contacts and245769 broad pairs. Chosen capacities72,212992 and294912 retain
20%,20.9% and20% reserve over these observations. All row drops, raw overflows
and13 sticky narrow flags are clear, states finite and source guards pass.
These horizons are capacity evidence, not a universal envelope or convergence
proof. Telemetry-instrumented timings are deliberately not used.

The actual MF storage is one row rather than a configured32-row panel;
propagation arrays are not allocated. Changing those settings would not remove
the large allocations their names might suggest. The much larger historical
11223040 raw-contact allocation is replaced by the measured212992 reservation.
This is capacity calibration, not a measured Kuka-architecture gain.

Clean timing then uses the existing comparison driver, no diagnostic kernels,
seed0,200 warmup,1000 synchronized wall steps and40 graph steps, one round.
The fixed ANYmal recipe retains grouped dynamics, in-kernel/world row handling,
eligible parallel24 sweeps and original fallback8, dt/substeps and controls.
MJWarp retains its calibrated32 rows/nconmax9,100/50 budgets and recorded
line-search correction.

| GPU | FPGS physics ms | MJWarp physics ms | MJWarp/FPGS | FPGS wall ms | MJWarp wall ms | Wall ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX PRO6000 | 9.202900 | 36.635007 | 3.98081x | 17.611016 | 46.860328 | 2.66085x |
| GB300 | 9.388857 | 41.892997 | 4.46199x | 19.069055 | 52.889402 | 2.77357x |

Raw comparison: `/tmp/fpgs-cross-task-anymald-20260913-01`.
Manifest SHA256 `d9463e7550decb61462678919cdfccbf5a05f959380ae9616155bca04314ee9a`.
All four children complete with checked finite/capacity boundaries passing.
This is a fresh current-FPGS comparison, not a baseline/candidate measurement
of the Kuka kinetic redesign, repeated stability or cross-backend parity.

## G1 calibration and support

G1 starts with a32-world asset/recipe smoke test, not a representative timing
claim: `/tmp/fpgs-cross-task-g1-smoke32-20260913-01`. Trial capacities are
4096 contacts,8192 broad outputs and262144 triangle pairs on both backends,
MJ300 rows/nconmax128. They are diagnostic starting reservations, not asserted
minimal or safe16K settings. The subsequent full-size calibration still needs
the actual grouped-dynamics recipe, warmup, reset horizon and held-out seed.
The smoke completes on both backends/cards with checked flags clear, finite
states and correctly separated auxiliary graphs. All children are reaped and
both GPUs are compute-idle. Its small-world ratios are not entered in the
representative table.

The Newton-only FPGS adapter reuses existing row/collision telemetry, installs
the exact per-task grouped recipe, preserves failure demand, and reads all13
current sticky narrow flags without clearing them. Its timed output is invalid
as performance evidence because telemetry adds work. Frozen adapter SHA256:
`66d50f1cbffa37887942383af3271b9377c61c4a340df5dac0d0c99123a64a7e`.
The independent G1 MJ observer preserves the current corrected numerical driver
and every warning bit; it passes seven CPU controls and a32-world paired GPU
instrumentation smoke before the full-scale checks recorded below.

G1's first16K FPGS capacity trial
`/tmp/fpgs-g1-capacity16k-20260913-01` is rejected: attempted triangle demand
reaches1467288 against1048576. Root interrupts both owned Python children after
the overflow is established; failure evidence is retained and both process
groups are reaped with GPUs idle. Public contact counts are censored by the
earlier triangle loss and cannot determine the final contact reservation.
The failed run's final row read is explicitly unavailable after Lab cleanup;
the earlier warmup row snapshot and final collision demand are retained.
No timing from this run is used.

Retry02 keeps192 rows and262144 contacts provisionally, requests49152 broad
outputs (the actual full explicit pair count, so no larger allocation is needed),
and uses1769472 triangle pairs,20.6% above the observed attempted peak. This is
still a trial: restoring upstream pairs can change contacts and trajectories.
The MJ full-size diagnostic initially retains300 rows until its actual tail is
measured; the32-world smoke's88-row maximum does not bound16K tails.

Retry02 completes both cards with12960 collision calls each, all capacities,
row drops,13 sticky flags and source checks passing. Manifest SHA256:
`9c5330051ddf37a81e0b3c95eef7d7eb453d31d93b2fc61701beaf4800203764`.
The uncensored paired maxima are83 dense rows,245966 public contacts,
49152 broad/mesh-convex pairs,1465626 triangles and247002 reducer hash entries.
Actual hash capacity is524288, chosen automatically by the existing reducer
from the triangle reservation; this is not an independently minimized setting.
Restoring upstream triangles changes contact/reset behavior, so the failed
trial's row/contact profile is also not a numerical baseline.

The held-out trial will use100 dense rows (20.5% over83),294912 raw contacts
(19.9% over245966),49152 broad outputs (the complete explicit pair-list bound)
and1769472 triangles (20.7% over1465626). No new solver algorithm is enabled.
Seed1 uses200 warmup,1000 steps and40 profile steps, one repeat; this is a
held-out reset horizon, not another three-repeat timing campaign.

G1 MJ seed0 calibration also completes on both cards:
`/tmp/fpgs-g1-mj-capacity16k-20260913-01`, manifest SHA256
`83517e6592c9cc882dea3cf9ca0abf78a908c24f2ac0e64cdb33b51989c02739`.
Both observe9920 actual solves (two solver calls per collision), with100 maximum
rows,1200 maximum stored sparse span, zero row/NNZ/contact overflows and zero
solver/line-search warning bits. Paired public contact demand peaks at254587;
triangle demand1463154 and reducer active entries246033. All boundary checks
and source/numerical-driver guards pass. The initial300-row reservation is not
retained as the final benchmark setting.

MJ held-out seed1 will use120 rows, automatically1440 sparse entries if the
current allocation rule holds, nconmax19/311296 pooled and raw contacts,
49152 broad pairs and1769472 triangles. These provide20% row/sparse and22.3%
contact reserve above the completed seed0 observations. The actual allocation
and full-run warning history must pass before clean timing.

FPGS held-out seed1 completes and passes on both GPUs at100 rows/294912
contacts, with4960 collision calls per card. Raw output:
`/tmp/fpgs-g1-capacity16k-seed1-20260913-01`, manifest SHA256
`e94c1ec413f9dbd6598ca64fc94a74767b1548a3baf37cdf0d1ffa9875df44b1`.
Paired seed1 peaks are83 rows,243928 contacts,1466598 triangles and247571
reducer entries. All row and collision overflow/sticky flags remain clear.
The task remains Rough-G1,43 generalized velocities per world, sim dt0.005,
decimation4, two substeps/solver dt0.0025 and eight FPGS sweeps. Its20-second
episode length makes the1200 warmup+held-out environment steps a reset horizon.

MJ held-out seed1 also completes and passes at120 rows/1440 actual sparse
entries, nconmax19/raw311296. Raw output:
`/tmp/fpgs-g1-mj-capacity16k-seed1-20260913-01`, manifest SHA256
`eb4525ac8836c20963022afc3da9efe9148d33bad4ce601bd806c6d8dac47a31`.
The RTX held-out tail reaches104 rows and1248 sparse entries, leaving15.4%
reserve at the chosen120/1440 rather than the seed0-based20%. There is no
overflow; all checked warning masks remain zero. Contact demand stays below
the previous254587 combined peak. No further capacity tuning is undertaken.

## G1 clean discovery result

The single paired clean comparison completes at approximately06:53 UTC:
`/tmp/fpgs-cross-task-g1-20260913-01`, manifest SHA256
`d28324bd2650d2b68df07d36daca842b0b82c39b8311bfd29e2ecc7ed1b48c78`.
All four children exit0, finite states and checked capacity/native-warning
boundaries pass, final source guards pass and both GPUs are compute-idle.

| GPU | FPGS physics ms | MJWarp physics ms | MJWarp/FPGS | FPGS wall ms | MJWarp wall ms | Wall ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX PRO6000 | 43.717024 | 52.264290 | 1.19551x | 57.466556 | 66.581503 | 1.15861x |
| GB300 | 72.387317 | 78.266767 | 1.08122x | 86.325911 | 92.563174 | 1.07225x |

Each profile has40 steps x4 physics graphs and one separate auxiliary graph
per step. Auxiliary graph means are1.027969/1.026174 ms FPGS/MJ on RTX and
1.026017/1.025291 ms on GB. They are excluded from physics totals; environment
wall comes from the separate1000-step synchronized window, not profiler wall.
The profile has no per-node graph timings, so it cannot identify the internal
collision/solver bottleneck or justify a stage-by-stage cost claim.

G1's1.20x/1.08x current physics advantage is much smaller than the roughly4x
ANYmal and Keyboard comparisons. The Kuka kinetic redesign remains unsupported
on all three tasks. These measurements establish broader current baselines,
not transferred optimization gains. All are one-round discovery comparisons
with corrected MJWarp and different trajectories, not matched convergence or
training-throughput results. Dependency/software warnings remain in logs even
though the native physics-warning masks are zero.

## Measurement latency correction

At approximately06:47 UTC the user asks why a timing request has taken50
minutes. The extension accumulated too much serial preparation: new capacity
observers and their reviews, initial/full/held-out calibrations, and only then
the final G1 timing. The G1 triangle overflow was real and its timing correctly
rejected, but that does not justify holding the complete survey response while
the two other tasks already have usable discovery results.

For quick timing requests, return completed task results immediately with their
actual scope; keep a task requiring new investigation separately pending and
timebox that work. Reuse existing reviewed measurement paths and do not add
new validation layers unless a concrete validity gap requires them. Do not
weaken overflow checks or substitute instrumented/invalid runs to save time.
The G1 preparation phase is closed; the one clean comparison completes and is
reported above without additional calibration or helper-review gates.

Source inspection finds existing ANYmal eligible in-kernel row/response work
and parallel24 sweeps; Keyboard already has diagonal108/arm6, fused key limits
and compact triplet solving. These cannot be credited again as new deletions.
G1's43-coordinate floating articulation needs a different topology/operator,
not the Kuka23+free6 factor. New broad benefit remains unimplemented/unmeasured.

The inherited Kuka/Franka accepted table and the unmet additional2x/4x target
remain unchanged.

## Reproduce the completed clean timing rounds

Run the existing Newton driver from the selected clean50dfa checkout. All three
commands measure RTX PRO6000 andGB300 concurrently for each backend; backend
batches do not overlap on either device. Output paths must be fresh and outside
all source checkouts. These are one-round discovery commands; use repeated
balanced rounds and separate numerical checks before accepting a new gain.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910/.venv/bin/python \
  python tools/fpgs_bench/compare_backends.py \
  --isaaclab /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910 \
  --fpgs /home/octi/Projects/newton-fpgs-composed-world-20260912 \
  --mjwarp /home/octi/Projects/newton-fpgs-composed-world-20260912 \
  --gpus 0 1 --task anymald --repeats 1 --num-envs 16384 \
  --warmup-steps 200 --steps 1000 --profile-steps 40 \
  --mjwarp-linesearch-fix \
  --fpgs-env 0:FEATHER_PGS_SIMPLE_WORLD_ZERO=1 \
  --fpgs-env 1:FEATHER_PGS_SIMPLE_WORLD_ZERO=1 \
  --capacity anymald:fpgs:dense_max_constraints=72 \
  --capacity anymald:fpgs:rigid_contact_max=212992 \
  --capacity anymald:fpgs:broad_phase_output_max=294912 \
  --capacity anymald:mjwarp:njmax=32 \
  --capacity anymald:mjwarp:nconmax=9 \
  --capacity anymald:mjwarp:rigid_contact_max=147456 \
  --capacity anymald:mjwarp:broad_phase_output_max=294912 \
  --output-dir /tmp/fpgs-cross-task-anymald-REPRO-FRESH

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910/.venv/bin/python \
  python tools/fpgs_bench/compare_backends.py \
  --isaaclab /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910 \
  --fpgs /home/octi/Projects/newton-fpgs-composed-world-20260912 \
  --mjwarp /home/octi/Projects/newton-fpgs-composed-world-20260912 \
  --gpus 0 1 --task keyboard-so101 --repeats 1 --num-envs 4096 \
  --warmup-steps 200 --steps 1000 --profile-steps 40 \
  --mjwarp-linesearch-fix \
  --fpgs-env 0:FEATHER_PGS_SPARSE_CONTACT_DIRECT=1 \
  --fpgs-env 1:FEATHER_PGS_SPARSE_CONTACT_DIRECT=1 \
  --fpgs-env 0:FEATHER_PGS_PRISMATIC_PUBLICATION=1 \
  --fpgs-env 1:FEATHER_PGS_PRISMATIC_PUBLICATION=1 \
  --fpgs-env 0:FEATHER_PGS_COMPACT_CONTACT_BOUNDARY=1 \
  --fpgs-env 1:FEATHER_PGS_COMPACT_CONTACT_BOUNDARY=1 \
  --capacity keyboard-so101:fpgs:dense_max_constraints=704 \
  --capacity keyboard-so101:fpgs:rigid_contact_max=147456 \
  --capacity keyboard-so101:fpgs:broad_phase_output_max=57344 \
  --capacity keyboard-so101:mjwarp:njmax=320 \
  --capacity keyboard-so101:mjwarp:nconmax=32 \
  --capacity keyboard-so101:mjwarp:rigid_contact_max=131072 \
  --capacity keyboard-so101:mjwarp:broad_phase_output_max=45056 \
  --output-dir /tmp/fpgs-cross-task-keyboard-REPRO-FRESH

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910/.venv/bin/python \
  python tools/fpgs_bench/compare_backends.py \
  --isaaclab /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910 \
  --fpgs /home/octi/Projects/newton-fpgs-composed-world-20260912 \
  --mjwarp /home/octi/Projects/newton-fpgs-composed-world-20260912 \
  --gpus 0 1 --task g1 --repeats 1 --num-envs 16384 \
  --warmup-steps 200 --steps 1000 --profile-steps 40 \
  --mjwarp-linesearch-fix \
  --capacity g1:fpgs:dense_max_constraints=100 \
  --capacity g1:fpgs:rigid_contact_max=294912 \
  --capacity g1:fpgs:broad_phase_output_max=49152 \
  --capacity g1:fpgs:max_triangle_pairs=1769472 \
  --capacity g1:mjwarp:njmax=120 \
  --capacity g1:mjwarp:nconmax=19 \
  --capacity g1:mjwarp:rigid_contact_max=311296 \
  --capacity g1:mjwarp:broad_phase_output_max=49152 \
  --capacity g1:mjwarp:max_triangle_pairs=1769472 \
  --output-dir /tmp/fpgs-cross-task-g1-REPRO-FRESH
```
