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

## Existing funding and next gate

Current-capacity original 16K whole screen (40 steps, fixed 200 warm) measured
RTX 12.357853 -> 10.909528 ms and GB300 11.934729 -> 11.218563 ms.
GB wall throughput regressed: 34.022117 -> 36.469050 ms per environment step.
The source-matched three-step node audit confirms complete graph spans
RTX 12.552401 -> 11.109275 ms and GB300 12.298603 -> 11.450399 ms;
these are diagnostic, not an independent long-run speed claim.

Existing current/held 4K physical readsets and loaded coupled/type-4 CUDA
controls qualify ad42's numerical owners. The integration remains unmeasured
until root runs the original paired 16K wrapper on the frozen composed tip.
No GPU job, new benchmark framework, timestep/substep/iteration alteration,
or broader promotion is authorized by this source-composition step.

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
`ooctipus/fpgs-kuka-qualification-report-20260914`.

## Root-owned original whole comparison

Run only after this tree is frozen clean and root grants the paired GPU lease.
The existing driver records exact source identities and rejects changed
budgets/capacities. This is the original bounded one-round composition screen;
do not interpret it as a repeated promotion gate. No new wrapper is needed.

```bash
env CUDA_VISIBLE_DEVICES='' GPU='' PYTHONDONTWRITEBYTECODE=1 \
  uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python \
  python /home/octi/Projects/newton-fpgs-structural-bench-20260913/tools/fpgs_bench/compare_variants.py \
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
