# Kuka kinetic integration — 2026-09-14

## Pre-code scope and frozen references

Compose the already qualified Kuka runtime from
`ad42fea95fd44cd5ec72afa1be1cce570b45109a` onto accepted integrated
`baf0c57ac5ef22bef452acd625b1727dd73a50b5`, in this new worktree only.
Preserve both original worktrees, every accepted G1/Allegro/Franka feature,
and the original fixed Isaac Lab recipe. This is qualification/composition,
not a new numerical algorithm or a claim to have reached four times MuJoCo.

The runtime delta is the 21 kinetic modules plus three solver/owner seams.
Import the existing focused tests, not the external benchmark/probe tools.
Retain sparse-factor notification validation, and validate every owner before
invalidating any kinetic state. Preserve the shared notification snapshot.

## Source reconciliation and regression contract

Only the retained `_get_pgs_solve_mf_gs_kernel` AST pin differs in baf.
Its new single-factor branch defaults to false for the Kuka invocation.
The selected 192/64/29, factor/components, no-drive/no-dense-limit kernel has
identical ABI/key and identical nonblank emitted native source on both tips:
`9974a8fb73c1146fbfef95831c46f3dd444f80d6951ce3252626a81ae864cd3a`.
Baf inserts exactly one empty interpolation newline (21417 -> 21418 bytes).
Remove only that empty newline, preserving the new branch, then update only
the reviewed factory AST pin. The selected default native source must regain
the original exact SHA:
`1ea22a605c07900eb7365cf8c7a28b3dedc08de7ccefe3a0a7257c108882c649`.
The existing frozen native/descriptor oracle must pass without rebaselining.
Exercise focused kinetic, notification, publication and retained-factor CPU
controls; run the full repository pre-commit hooks before freezing.

## Pre-integration funding

Current-capacity original 16K whole screen (40 steps, fixed 200 warm) measured
RTX 12.357853 -> 10.909528 ms and GB300 11.934729 -> 11.218563 ms.
GB wall throughput regressed: 34.022117 -> 36.469050 ms per environment step.
The source-matched three-step node audit confirms complete graph spans
RTX 12.552401 -> 11.109275 ms and GB300 12.298603 -> 11.450399 ms;
these are diagnostic, not an independent long-run speed claim.

Existing current/held 4K physical readsets and loaded coupled/type-4 CUDA
controls qualify ad42's numerical owners. The subsequent integrated results
below supersede the initial pending-measurement status; the original screen
and its GB wall regression remain historical evidence, not repeat samples.

## Source/CPU readiness

The pre-reconciliation frozen-native test failed at the retained MF AST guard;
after the one-pin reconciliation it passes all 249 original native/descriptor
records without changing their expectations. The Kuka 192/64/29 default body
is byte-exact. The accepted G1 100/64/43 single-factor body retains identical
nonblank text (SHA `f6fd09a5b4efde69f992196aa88d0ab1f84fbbd6d015c2a4443486f72f104e34`)
and ABI. Its only textual difference is the removed empty interpolation line.
All 21 kinetic files match ad42 except the single reviewed AST pin in
`kinetic_source.py`; the two existing publication/joint owner files also match.
The final solver diff retains all other baf features and only adds the ad42
dispatch/reset/export/snapshot hooks plus this default formatting correction.

Focused CPU batch: 60 tests, 58 executed PASS and two CUDA skips. It covers
native closure, current world gravity/forces, held mass, ping-pong capture
allocation, masked reset, publication, notifications and retained Franka
notification validation. Retained-feature batch: 27 tests, ten executed PASS;
the seventeen CUDA-only mimic/connect/spring/pre-elimination cases were not
executed under the CPU-only lease. The sparse and single-factor CPU controls
pass. A new test requires sparse validation to reject before kinetic mutation.
Full `uvx pre-commit run -a` passes after formatting two test files; no native
numeric or source-oracle expectation was reformatted or regenerated.

The detailed ad42 qualification and disjoint owner accounting are preserved
separately in `KUKA_KINETIC_QUALIFICATION_20260914.md` on branch
`ooctipus/fpgs-kuka-qualification-report-20260914`, report commit
`169777492fcf82e45581c1742409f3df1ab0eff3`.

## Completed physical qualification and integrated measurements

The ad42 eager 200-warm 4K trajectory samples actual calls 1600 (refresh) and
1601 (held), before original finish/publication, and completes 1608 calls.
ZERO, MF0 and independent worlds occur. Original finish/public-state checks
pass on both cards. A saved external joint-limit test incorrectly penalized
reductions in positive separating speed; its fresh, CPU-only correction tests
increase in negative violation, preserving the 3e-5 bound and failed artifacts.
Physical contact-normal, complementarity, friction-cone, momentum and limit
nonregression pass. The failed legacy forward-velocity diagnostic remains
reported; this is not bit-identity or a blanket tolerance relaxation.
The existing loaded coupled/type-4 CUDA fixture separately passes on both
cards (one test/card, 2.077 / 2.114 s), including a nonzero limit impulse.
It is not claimed as naturally coupled trajectory coverage. Integration keeps
all 249 native/descriptor oracle records unchanged, as established above.

All measurements below compare clean baf against clean runtime
`6c2297ca3f7022df7e20ca0486f75c6ff818f29c`, with the fixed Lab and flags in
the reproduction recipe. Times are ms per environment step; ratios are
baseline/candidate. Repeat entries are medians of three alternating AB/BA/AB
rounds, not medians pooled with the discovery screen.

| Measurement | Card | Physics baseline -> candidate | Ratio | Wall baseline -> candidate | Ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| Paired discovery | RTX | 12.379235 -> 10.963438 | 1.129138x | 36.788461 -> 35.906270 | 1.024569x |
| Paired discovery | GB300 | 11.983935 -> 11.270214 | 1.063328x | 35.482980 -> 38.251759 | 0.927617x |
| Three-round repeat | RTX | 12.506653 -> 11.119767 | 1.124723x | 36.993766 -> 33.680188 | 1.098384x |
| Three-round repeat | GB300 | 12.033355 -> 11.339816 | 1.061160x | 36.255405 -> 35.576291 | 1.019089x |

The GB paired discovery has a wall-time regression. Its repeat median is only
a small wall improvement, and repeat round one also loses (35.363822 ->
37.513524 ms); do not claim a consistent GB end-to-end win. An attempted
paired repeat stopped at the initial idle guard, before children, because of
an external GB workload. The completed RTX and GB repeats were consequently
scheduled separately, each retaining its selected-device idle checks. They
are not three simultaneous paired rounds, nor full RL training measurements.

Closed artifacts, each containing the original manifest and child results:

- `/tmp/fpgs-kuka-kinetic-integrated-paired16k-20260914-01`
- `/tmp/fpgs-kuka-kinetic-integrated-repeat-rtx16k-20260914-01`
- `/tmp/fpgs-kuka-kinetic-integrated-repeat-gb16k-20260914-01`

All three manifests are complete, with final source/idle guards passing and
`physics_budgets_modified=false`. All 16 children return zero. Baseline,
candidate and external tool worktrees were clean. Fixed Lab's only recorded
untracked entry is its approved `.venv` symlink. Every run retains 16384
worlds, 200 warm steps, 40 wall and 40 graph-profile steps, seed zero, raw
311296 (19N), broad 442368 (27N), row capacities 192/64/192, sim dt 1/120,
two solver substeps at 1/240, eight GS sweeps and decimation four. Original
overflow checks pass; auxiliary graph time is zero and public state is finite.

This qualifies a substantial opt-in Kuka successor while retaining accepted
G1/Allegro/Franka source, not a four-times-MuJoCo result. Against the separate
27.504-ms RTX MJ reference, the 4x ceiling remains 6.876 ms; the repeated
11.119767-ms candidate still needs about 4.244 ms of further whole removal.

## Exact tested reproduction and prerequisites

The previous direct `compare_variants.py` command omitted mandatory fixed-Lab
import selection. Use the retained c062 adapter below, not that direct command
and not this integration tree's local benchmark files. The tested external
tool worktree is `newton-fpgs-structural-bench-20260913` at
`961b7e2f751bcd1d8b03368e7956b54c81414897`. Its original
`compare_variants.py` SHA-256 is
`48406c079588d3088fc25c1cdce2bc2659aaec28539013e5811c9c55378445b5`;
`compare_backends.py` and `checked_capture.py` are respectively
`79818fce1deab05b9409d386d7a240265cf2f7e5b7f019109f3ab99c9f52e403`
and `42b289bd0082d194180d5981651a108d11a66058ff4e67649ad7ddb408c9bea6`.
The current repository copies differ; silently substituting them is not this
measured protocol.

Required adapter `/tmp/fpgs-kinetic-fixed-variants-TZRkIPYw/run.py` has SHA-256
`c062388f23f82c524690f272fb2d14141ed4126db641de6dcd86f8ade5bfe416`.
It uses `/tmp/fpgs-contact-reset-compare-26Z8K3/sitecustomize.py`, SHA-256
`3b3e90f3f5870367e054c2bf4c515a4cd6d6557296efa637abb266fbb4dc4980`,
the fixed Lab backend package and its existing interpreter. It verifies the
selected Newton and backend imports and adds these inputs to source guards.
Preserve these external prerequisites; no replacement harness is implied.

The following is the recorded paired command. Its output already exists;
choose a fresh output directory for a reproduction, with both selected GPUs
idle and source roots clean. For an exact historical source comparison use
the stated baf/6c229 tips; a later report-only commit must be recorded as such.

```bash
env CUDA_VISIBLE_DEVICES='' GPU='' PYTHONDONTWRITEBYTECODE=1 \
  uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python \
  python /tmp/fpgs-kinetic-fixed-variants-TZRkIPYw/run.py \
  --isaaclab /home/octi/Projects/IsaacLab.wt/contact-reset-20260913 \
  --baseline /home/octi/Projects/newton-fpgs-tenhour-integrated-20260914 \
  --candidate /home/octi/Projects/newton-fpgs-kinetic-integrated-20260914 \
  --output-dir /tmp/fpgs-kuka-kinetic-integrated-paired16k-20260914-01 \
  --task kuka --gpus 0 1 --rounds 1 --num-envs 16384 \
  --warmup-steps 200 --steps 40 --profile-steps 40 --trace-mode graph \
  --capacity kuka:fpgs:rigid_contact_max=311296 \
  --capacity kuka:fpgs:broad_phase_output_max=442368 \
  --baseline-env FEATHER_PGS_SIMPLE_WORLD_ZERO=1 \
  --baseline-env FEATHER_PGS_LOCAL_ROW_PACKETS=1 \
  --baseline-env FEATHER_PGS_INDEPENDENT_COMPONENTS=1 \
  --baseline-env FEATHER_PGS_PAIRED_GENERAL_OVERLAP=1 \
  --baseline-env FEATHER_PGS_KUKA_JOINT_WORLD=1 \
  --baseline-env FEATHER_PGS_WORLD_SCAN_PUBLICATION=1 \
  --candidate-env FEATHER_PGS_SIMPLE_WORLD_ZERO=1 \
  --candidate-env FEATHER_PGS_LOCAL_ROW_PACKETS=1 \
  --candidate-env FEATHER_PGS_INDEPENDENT_COMPONENTS=1 \
  --candidate-env FEATHER_PGS_PAIRED_GENERAL_OVERLAP=1 \
  --candidate-env FEATHER_PGS_KUKA_JOINT_WORLD=1 \
  --candidate-env FEATHER_PGS_WORLD_SCAN_PUBLICATION=1 \
  --candidate-env FEATHER_PGS_KUKA_KINETIC_WORLD=1
```

The two recorded repeat commands use exactly the same interpreter, roots,
capacities, timing and environment arguments above, changing only these four
arguments. Their recorded output directories likewise must not be reused:

| Repeat | Python entry point | GPU arguments | Rounds | Output directory |
| --- | --- | --- | --- | --- |
| RTX | `/tmp/fpgs-original-rtx-variant-owner-CMP9QZpR/run.py` | `--gpus 0` | `--rounds 3` | `/tmp/fpgs-kuka-kinetic-integrated-repeat-rtx16k-20260914-01` |
| GB300 | `/tmp/fpgs-original-rtx-variant-owner-CMP9QZpR/gb_run.py` | `--gpus 1` | `--rounds 3` | `/tmp/fpgs-kuka-kinetic-integrated-repeat-gb16k-20260914-01` |

The wrappers have SHA-256
`6468763a06ebd6f9df518fe7f6cd05f0bf9404c524e1eece6e1dfb6757eef6c5`
and `716feec678807fe73254289691bd41525105efb715938c0c45a724198ca73640`.
They assert original driver 484 and adapter c062, replacing only the driver's
two-device argument restriction with the one selected device. Original
source, idle, capacity, timing and alternating-order checks remain intact;
this is a documented scheduling exception, not permission to ignore contention.
