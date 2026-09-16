# Franka complete world-lane state experiment — 2026-09-16

## Decision

**Unpromoted, default-off experiment.** The complete owner passed the focused
physical and lifecycle controls on both GPUs, but the successful paired whole
screen lost physics time: RTX 5.020306 → 5.166324 ms and GB300 4.693110 →
4.834694 ms per environment step. No accepted performance reference or inherited
handoff is changed. No predictor-only feature is extracted from this experiment.

## Boundary and invariant

`FEATHER_PGS_WORLD_LANE_STATE=1` selects one CUDA thread per world, block128,
inside the existing admitted Franka kinetic configuration. The generated
topology/type schedule replaces repair, current live-force prediction plus
masked primary/free factorization, and held/refresh finish. Private bias,
COM-offset and packed geometric-mass storage is field-major across worlds.
Physical model values remain live. Canonical factors, public joint/body state,
world screws, prescribed bodies and free-root transport remain available to the
original consumers. The retained p16 path is unchanged when the flag is off.

Factor/prediction runs at the original Stage1 CRBA boundary after current drive
data is ready; this preserves the Stage2 paired-inverse consumer. The old Stage3
predictor is skipped only for the new owner. Current force inputs and held-factor
cadence are preserved. Contact allocation, row order, eight GS passes,
timestep/substeps, limits and friction law are unchanged; no Lab edits or
capacity increases are used. The retained admission excludes joint-velocity
prescaling, so the alternate Stage1 velocity buffer is not active here.

This is a complete owner experiment, not a coordinate-only optimization or a
two-world subwarp mapping. Neither its kernels nor the retained p16 kernels use
CTA barriers, so removal of such barriers is not claimed as a measured benefit.

## Source and proof pins

Candidate base: `120c37fcbbd3f4ae4fc6ff4f3f8a4bb5c2256a57` on
`ooctipus/fpgs-world-lane-20260916`. Whole and node captures used the same frozen
dirty-source digest
`e423c8d53e7d97f7472852ae3e6335af13fd1d9e1200eb96b24e9a700ed75e10`.

| Frozen file | SHA256 |
|---|---|
| `newton/_src/solvers/feather_pgs/solver_feather_pgs.py` | `f7662424665a0f29e3c2dcdcf18a41aac77392e1c302fba6238cbae758532a49` |
| `newton/_src/solvers/feather_pgs/world_lane_state.py` | `ca6ec1cd6d86630fcfa641f4421f7ebfbe4a07cbceea94897a339dea1e96346b` |
| `tools/fpgs_bench/test_world_lane_state.py` | `0b3a0144db837f59bad8d8b9f737c7d4fb8d55f3e072020e383de9eeec82fb68` |
| `tools/fpgs_bench/world_lane_capture_20260916/run.py` | `01bf4ba65628a6b77bdb96d4f3b27aeff70dea5531757f04fc8149e159502000` |
| `tools/fpgs_bench/world_lane_capture_20260916/checked_world_lane.py` | `50d2c1faf86218696d3f13f3091e69a8b8183b62eeb19d59238ad2c5c018b70f` |
| `tools/fpgs_bench/world_lane_capture_20260916/nsys_checked.sh` | `c3c876e95b103e339478911d957e093af007fb4ad9967f2ce30657db4d433c95` |

Baseline Newton: `ca0d427af809571bb5501f644c1a6e03990cd2a8` in
`/home/octi/Projects/newton-fpgs-keyboard-linear-state-20260915`.
Unchanged Lab: `53ee6b44c2334341305dbdf385a3916c6b140799` in
`/home/octi/Projects/IsaacLab.wt/contact-reset-20260913`.
Benchmark checkout: `961b7e2f751bcd1d8b03368e7956b54c81414897`.

The thin capture adapter reuses
`/tmp/fpgs-kinetic-fixed-variants-TZRkIPYw/run.py`
(`c062388f23f82c524690f272fb2d14141ed4126db641de6dcd86f8ade5bfe416`)
and the existing checked-capture lifecycle. Only its two untimed metadata
boundaries inspect the actual owner, factory keys, private/canonical shapes and
status. The shell differs from the retained original only in the checked entry
path. Actual runtime keys are `world_lane_repair13_h45_3ec3b8a0fb`,
`world_lane_finish_held13_h45_3ec3b8a0fb`,
`world_lane_finish_refresh13_h45_3ec3b8a0fb` and
`world_lane_factor_predict9_6_3ec3b8a0fb`.

## Numerical and lifecycle evidence

The first missing-module regression failed before implementation. Five CPU
controls subsequently passed; CUDA tests were skipped in that CPU process.
Controls cover independently assembled current mass/bias/public state, held and
refreshed factor/force action, integration, masked repair, model notification
and unsupported admission. Existing physical tolerances were not widened.

Root ran both native selectors in
`tools.fpgs_bench.test_world_lane_state.TestWorldLaneStateCUDA` on both devices:
two tests passed per card (41.738 s RTX, 42.115 s GB, including compilation).
They compare against actual p16 through loaded eight-pass solves, free and
prescribed bodies, state banks, masked reset, held factors, notification and
graph replay. Maximum normalized joint/body velocity differences were
`3.7684e-5` / `5.2156e-5`; physical mass error `1.0169e-6`; held momentum error
`1.3725e-7`. Bad-force handling withholds invalid world prediction and latches
status. This is not a claim of transactional rollback or independent finite
certification of every factor entry; original and candidate factor publication
are not transactional.

Native logs: `/tmp/fpgs-world-lane-native-20260916-qIVxFvTQ/gpu{0,1}.log`.
SHA256: RTX `5c390c87f4f37879ebf1fee70c5458699eb3c92467afb95c90c6f84be5247ce9`,
GB `81c343e5d6dcdbc2400efd01d5faff38d4b7714520133ef7b9752965d5206954`.

## Whole screen and preserved setup failure

Both arms used 16,384 worlds, seed0, warmup200, wall40/profile40, one round,
graph execution, raw-contact capacity32768 and broadphase capacity7680.
Original row capacities and allowances were unchanged. Both set
`SIMPLE_WORLD_ZERO=1`, `LOCAL_ROW_PACKETS=1`, `FRANKA_KINETIC_STATE=1`,
`GROUP_LANES=16`, `ROWS_MASKED=1` (all `FEATHER_PGS_` prefixed), and
`NEWTON_NARROW_PHASE_THREADS_X=4`; only the world-lane feature differed.
Full commands and source guards are in the manifests.

First attempt:
`/tmp/fpgs-franka-world-lane-whole-paired16k-20260916-01/manifest.json`
(`ab95c27694ed708ae7a4f9e146b689e8e668eb97623a45646293e863f7585c8d`).
RTX **baseline** aborted with rc134, `malloc_consolidate(): invalid chunk size`,
during scene cloning, before model/solver creation and before any metadata or
capture boundary. GB baseline completed. No candidate launched. Source/idle
guards passed; the log does not establish the heap-corruption cause. No solver
fix was made in response. The fresh second attempt used identical frozen source
and settings, changing only the output directory.

Successful attempt:
`/tmp/fpgs-franka-world-lane-whole-paired16k-20260916-02/manifest.json`
(`2404277f0fc3615a6ab973fb93d990122daf6b99d2c656c4ce6f9afa9afb2fce`).
Parent and all four children returned zero. Source/idle, finite, actual-owner and
capacity checks passed at all eight boundaries.

| ms/environment step | RTX p16 | RTX world-lane | GB p16 | GB world-lane |
|---|---:|---:|---:|---:|
| Whole physics | 5.020306 | 5.166324 | 4.693110 | 4.834694 |
| Wall environment stepping | 30.668765 | 29.207772 | 28.611235 | 29.278558 |

Physics baseline/candidate ratios are **0.971737× RTX / 0.970715× GB**.
Wall stepping is not full RL and does not override the physics loss. This was a
single discovery screen, not repeated promotion evidence.

## Strict node diagnosis and closure

Node capture:
`/tmp/fpgs-franka-world-lane-node-paired16k-20260916-01/manifest.json`
(`5c0ae02109fd0e91bf68094dd19e7fa2343a9938d4c84aa91215c9e04c269ddb`).
All four children and guards passed. The external adapter
`/tmp/fpgs-franka-world-lane-strict-wX7i4mJ2/read_world_lane.py`
(`a2d68778063e125358de0a7696dd7805a89f2230285a363b2b5794a7457045f3`)
retains the original strict process/correlation/root-membership and interval
union/exclusive reader. Only normalization to 12 recorded environment steps and
exact owner/collision names changed. Each result proves 48 physics roots,
zero auxiliary roots and zero unproven graph nodes.

| Exclusive ms/environment step | RTX p16 | RTX world-lane | GB p16 | GB world-lane |
|---|---:|---:|---:|---:|
| Complete state/held family | 1.671307 | 1.774082 | 1.400978 | 1.540940 |
| Publication | 0.915504 | 1.205870 | 0.752968 | 1.042685 |
| Repair | 0.156627 | 0.152829 | 0.100739 | 0.122208 |
| Prediction + primary/free factors | 0.520656 | 0.336150 | 0.475679 | 0.304088 |

The family includes overlapping drive/mask work as a union: the component
columns are not blindly summed. These profiled values diagnose ownership;
they are not an additive whole-timing counterfactual. Family calls fall 56→40
per environment; all old prediction/factor9/factor6 launches disappear. Finish
remains eight calls, four held plus four refresh, with no duplicate producer or
changed cadence. Publication loses about 0.290 ms on each card, outweighing the
approximately 0.185/0.172-ms prediction/factor improvement.

CUDA-hidden offline resources are in
`/tmp/fpgs-world-lane-offline-VuSH1UvI/offline02/report.json`
(`92e58e789d186df0a1df43224fce92813923058c11e203b8e01580c8df7695cf`).
Candidate held/refresh finish uses 167/255 RTX registers and 168/255 GB registers,
512 B shared, and no spills; p16 finish uses 80 registers and 7296 B shared.
Candidate repair uses 255 registers with small 8/16-B spill loads/stores on
RTX/GB. Prediction uses 152/148 registers and no spills. Held and refresh finish
durations are similar despite their different register counts. These facts do
not establish achieved occupancy, bandwidth or stall bounds.

The source retains all mandatory body transform, motion, inertia/bias and public
publication work, serialized within each world thread. Private field-major
accesses coexist with canonical world-strided public/model accesses. Those are
plausible costs, not hardware-counter diagnoses. No evidence supports a
correction saving at least 1 ms: the p16 family would need to fall below
0.671307 ms RTX, versus candidate1.774082. Keeping its other measured work
would leave only about 0.103096 ms for the 1.205870-ms publication, roughly a
91% cut. An additional transpose or register/mapping sweep has no supported
complete bound of that size and is not pursued.

Detailed strict outputs, source-reader pins and resource interpretation are in
`/tmp/fpgs-franka-world-lane-strict-wX7i4mJ2/SUMMARY.md`
(`3abf71fc53701595fdbbafe85102534a3a7612de6368b9820d980af7e2910878`).
The experiment remains available only behind its default-off flag; no further
optimization, predictor-only extraction or performance promotion is authorized
by these results.

Closure checks: `uvx pre-commit run -a` passed, as did targeted pre-commit checks
including every new runtime, test, observer and report file. Runtime/observer
hashes remain the measured pins above. No commit or push was made during closure.
