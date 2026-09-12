# KukaAllegro and Franka: progress toward 4x MJWarp

Goal: each task at least4x the corrected MJWarp whole-physics speed, with
RTX PRO6000 primary and simultaneous GB300 measurements. The target is
**not achieved**. Report environment wall time separately, not training speed.

This branch combines Kuka's independent-component/fork path from87af1f89
(report successor aa490863) and Franka's row packets plus numeric-notification
fix from c10f0e31. Combined runtime before this report is508aaad4. Both
experiments remain guarded and default-off. The original Newton handoff
31cf87f46 is an ancestor; all changes are in the ooctipus fork. Original
worktrees and Isaac Lab source/config/dependency pin remain unchanged.

## Repeated results on the individual frozen runtime commits

Three alternating AB/BA/AB rounds,16,384 worlds, seed0,200 warmup,40 wall
steps and40 graph-profiled steps per capture. Each arm launches both GPU
UUIDs simultaneously. Values are medians of three per-round means.

| Task / GPU | Previous FPGS ms | New FPGS ms | New / previous speed | Speed vs fixed MJ reference | Required ms for4x |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kuka / RTX | 15.453635 | 13.973622 | 1.1059x | 1.9655x | 6.866144 |
| Kuka / GB300 | 15.513112 | 13.884428 | 1.1173x | 1.8693x | 6.488661 |
| Franka / RTX | 6.246087 | 5.746087 | 1.0870x | 1.8635x | 2.677006 |
| Franka / GB300 | 5.804169 | 5.195039 | 1.1173x | 1.9920x | 2.587133 |

The fixed corrected-MJ references are27.464576525/25.954643650ms Kuka
and10.708024775/10.348531100ms Franka, RTX/GB. They are today's earlier
single-round discovery values, not fresh MJ captures in these FPGS A/B runs.
Another roughly50–53% of current candidate physics time must be removed.

| Task / GPU | Previous environment wall ms | New wall ms | Speed ratio |
| --- | ---: | ---: | ---: |
| Kuka / RTX | 35.371006 | 33.165392 | 1.0665x |
| Kuka / GB300 | 35.607505 | 33.506107 | 1.0627x |
| Franka / RTX | 24.263948 | 23.872308 | 1.0164x |
| Franka / GB300 | 23.921263 | 23.621120 | 1.0127x |

Franka's small wall difference is inconsistent per round: treat it as roughly
unchanged. The initial row-packet prototype regressed wall time to61.6ms.
That failure was traced to a38–39ms Python topology scan on each reset gravity
notification. The correction skips static topology scans only for documented
numeric notifications, preserving original gravity/FK/mass refreshes. Both
the failed prototype and its diagnosis remain in the branch history/reports.

All24 boundaries per task pass finite-state and four solver/thirteen collision
sticky checks, with identical original192/64/192 row capacities and4M raw
contact allocation. No buffer expansion or dropped-row allowance is used.
Substeps2, solver dt1/240, decimation4 and effective eight GS sweeps are fixed.
Independent audits rehash81 files and match1920 raw physics graph records per
task. All children exit0, are reaped, and source/idle guards pass.

## Numerical scope and integration status

Kuka same-current-input original eight-sweep replay on candidate-evolved
refresh/reuse states has maximum velocity difference4.77e-7 and dense impulse
difference8.28e-7; MF and coupled-fallback outputs are identical. Original
normal/limit residual tails are essentially unchanged. Those snapshots reach
nine MF rows, not every previously observed harder tail. See
[the Kuka report](INDEPENDENT_COMPONENTS_20260912.md).

Franka's25 tests pass on both actual GPUs, covering current prefixes/contacts,
held-factor physical action, complete fallback, native original/private GS,
graph replay and numeric versus structural notifications. Repeated scalar
snapshots show no candidate velocity/height explosion; small MF populations
and rare fallback counts differ between independent rollouts. See
[row ownership](FRANKA_ROW_PACKETS_20260912.md) and
[the notification correction](FRANKA_ROW_PACKET_NOTIFICATIONS_20260912.md).

The combined runtime passes45 tests on each actual GPU, including both feature
suites, fork dispatch, private API and existing local solves. Full pre-commit
passes. Logs are `/tmp/fpgs-fourx-progress-native-{rtx,gb}-20260912-01.log`.
The table above remains source-bound to the individual commits. The combined
`fb0f0759` runtime now also passes live16K compatibility checks for both tasks
on both GPUs: one round,200 warmup,40 wall and40 graph-profiled steps, with
all four experiment flags enabled. These are integration checks, not another
three-round speedup claim:

| Task / GPU | Individual candidate ms | Combined ms | Individual / combined |
| --- | ---: | ---: | ---: |
| Kuka / RTX | 14.025788 | 14.152383 | 0.9911x |
| Kuka / GB300 | 13.719741 | 13.648833 | 1.0052x |
| Franka / RTX | 5.711621 | 5.805963 | 0.9838x |
| Franka / GB300 | 5.182842 | 5.174058 | 1.0017x |

All children exit0 and are reaped; original capacity, source and final idle
guards pass. Compatibility manifests are
`/tmp/fpgs-fourx-combined-kuka16k-20260912-01/manifest.json`
(`0ad6d48115146ddbf3687c1e746914075928add580431c98deed3840a560793d`)
and `/tmp/fpgs-fourx-combined-franka16k-20260912-01/manifest.json`
(`5332703bbbca7442265dfeb9f144aa0102312ad2fe13550125d3fdf81d1fc300`).
None of these checks establishes universal convergence, identical trajectories,
training performance or physical parity with MJWarp.

## Actual Kuka zero-cohort measurement

A separate checked, untimed combined-runtime capture now records the actual
classifier decision immediately after its original call, not an inference
from zero row counts. It retains the original200 warmup and records two
adjacent original solver calls1600/1601 (mass refresh/reuse):

| GPU | Refresh resolved /16,384 | Reuse resolved /16,384 | Changed worlds |
| --- | ---: | ---: | ---: |
| RTX | 13,883 (84.735%) | 13,755 (83.954%) | 526 |
| GB300 | 13,921 (84.967%) | 13,783 (84.125%) | 550 |

Each observation stores only the65,536-byte binary decision array plus static
work counts. Selected fractions are the same for this model's32 bodies/FK
joint visits and35 global DOFs per world; the response map has only29 DOFs.
There are three articulations per world, including a prescribed free root.
This confirms a large eligible cohort, **not** that84% of publication time can
be hidden. Extra indexing, events, complementary work and GPU contention count.
The original free-body inverse still reads inertia for resolved worlds, so an
early finalizer must explicitly wait for that reader or safely exclude it.

Both children retain all eight original calls, exact source/alias/raw-capacity
guards and two passing original checked boundaries; all4 solver and13 collision
flags are clear. No alternate solve, live-array write or numerical-quality
claim is added by the recorder. Parent exits0, reaps both children and passes
final source/idle checks. Manifest:
`/tmp/fpgs-kuka-zero-cohort-paired16k-20260912-01/manifest.json`, SHA256
`b55b7b5190ed271752797df514c4197cab90bfb0710c9db59521e22ad8b3e089`.
Recorder: `/tmp/fpgs-kuka-zero-cohort-UFSd8PAS/run_cohort.py`, SHA256
`237e2929b491cb49113f7a53582f0017735f4172354aca2c4252e6bab7beb7ee`;
its adjacent READY.md contains the exact paired command and four CPU controls.

## Reproduce and continue

Use Lab1d8feb82d17dbfab8f0772de56f84deae2cb7974 and the original installed
environment (Python3.12.14, Warp1.17.0, Torch2.11cu130). GPU UUIDs:
RTX `GPU-883586b6-3100-0610-81e5-3b4c26f45639`,
GB `GPU-ebfac9e8-02d5-d8a9-3bfc-bac64c62ffd4`.
The checked paired reproducer is included at
`tools/fpgs_bench/compare_variants.py`; the exact recorded runs used tool
commit15158f3a30eead86ffd25e34bad73b5beab53988 and accepted baseline
42f1492ab99e2e8eedd157bd9893ed9b2fd4de61. Its README covers setup and guards.

From the tool checkout, with clean source trees and both GPUs idle:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
uv run --no-project --python /path/to/IsaacLab/.venv/bin/python python \
tools/fpgs_bench/compare_variants.py \
  --isaaclab /path/to/IsaacLab --baseline /path/to/accepted42 \
  --candidate /path/to/this/branch --task franka --gpus 0 1 \
  --rounds 3 --num-envs 16384 --warmup-steps 200 --steps 40 \
  --profile-steps 40 --trace-mode graph \
  --baseline-env FEATHER_PGS_SIMPLE_WORLD_ZERO=1 \
  --candidate-env FEATHER_PGS_SIMPLE_WORLD_ZERO=1 \
  --candidate-env FEATHER_PGS_LOCAL_ROW_PACKETS=1 \
  --output-dir /fresh/output/franka
```

For Kuka use `--task kuka` and replace the row-packet flag with both
`FEATHER_PGS_INDEPENDENT_COMPONENTS=1` and
`FEATHER_PGS_PAIRED_GENERAL_OVERLAP=1`. Do not edit pinned trees during a run.

Large inputs/traces remain local. Repeated manifests:

- `/tmp/fpgs-fourx-kuka-independent-fork16k-repeat-20260912-01/manifest.json`,
  SHA256 `83686f99e7a54ac6b526f591bcc12e1198638e1782b76ff3d1cd386c5b0c95f3`.
- `/tmp/fpgs-fourx-franka-row-packets-notify16k-repeat-20260912-01/manifest.json`,
  SHA256 `4c103836fe1f85fdfaf957e3702fe88238b4d18ddb704395abb91fb5667710c1`.
- Franka independent audit `/tmp/fpgs-franka-notify-repeat-audit-B8X5BIm5/RESULTS.md`,
  SHA256 `766b6789f92cef40e0426a132c262ac2fd1ab749fc4d6f52dbd31b02338fca16`.

Next large-gain investigation: independently completed world cohorts may
integrate/publish before harder worlds finish row preparation and solving.
This requires explicit protection against later impulse/velocity clears and
current-versus-next FK/cache races, plus final stream joins. Do not implement
an uncosted giant fused kernel or count already-overlapped local solves as
removable time. Standalone sparse-factor, reuse and packet-only alternatives
do not currently justify the requested large next step. Preserve their cost
and failure evidence; no lane/grid or sub-percent tuning sweep follows.
