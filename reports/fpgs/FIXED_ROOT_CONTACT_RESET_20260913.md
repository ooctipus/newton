# Fix fixed-root contact reset synchronization

## Outcome and ownership

Fixed in **Isaac Lab's Newton asset adapter**, not in FPGS, Newton's MuJoCo
solver, or MJWarp. The user explicitly expanded the earlier Newton-only edit
restriction to allow correctness fixes at the layer that owns the bug.

Lab nonfloating root-pose views alias `model.joint_X_p`. The root write APIs
updated that array and invalidated public FK, but never sent Newton's required
`ModelFlags.JOINT_PROPERTIES` notification. Newton already implements this
contract: see `docs/solvers/mujoco.rst` under fixed-root mocap bodies and
`SolverMuJoCo._update_joint_properties`. `reset(flags=0)` is a state-buffer reset,
not a notification of arbitrary model edits.

Consequently, current external collision points could be used with MJWarp
internal kinematics derived from an old root pose. Correct public Newton FK
did not reveal that internal inconsistency.

## Fix

Isaac Lab commit: `53ee6b44c2334341305dbdf385a3916c6b140799`.
Branch: `ooctipus/fpgs-contact-reset-20260913` on `ooctipus/IsaacLab`.

- Add immediate manager-owned model-change dispatch, retaining the existing
  initialization queue before solver construction.
- Notify after each nonfloating articulation/rigid-object root-pose write:
  link/COM, index/mask. Public and deprecated aliases already delegate here.
- Notify even with `skip_forward=True`: deferring data-cache invalidation does
  not defer a model mutation's notification.
- Record notification kernels alongside captured writes. A Python-only dirty
  set would disappear after its first host-side consumption, missing later
  graph replays.
- Leave floating-root and ordinary joint/velocity writes unchanged.

The production delta is 35 lines across three Lab files, plus regression tests,
a mock-view compatibility property, and a changelog fragment. No task physics,
iterations, substeps, threshold, installed package, or dependency pointer changed.
Direct low-level view/array mutation still requires caller notification.
Capture support remains dependent on the selected solver/constraint handler.

## Validation

### Regression-first active contact

The real Lab writer and real Newton selection view drive a two-world fixture:
FIXED or locked-D6 root, prismatic child sphere, frictionless plane contact.
Rotating only world 0 retains 0.01 m sphere penetration. A fresh solver built
from the updated model is the synchronization reference; no manual notification
is injected into the candidate.

Before the fix, a CPU run gave moved-world normal force **148.715118 N versus
180.118622 N** from the reference (17.44% difference); the untouched world's
148.714661 N matched. This is a synchronization regression, not a calibrated
force/convergence claim: uncorrected installed MJWarp 3.12 emits line-search
messages in CPU runs.

After the fix:

- 51 focused CPU tests pass: eight writers, fixed/locked-D6/free roots,
  `skip_forward`, initialization, and active contact response.
- 42 neighboring CPU manager/visualization tests pass.
- 65 existing Newton root-pose interface tests pass with the CPU manager
  device explicitly selected. Their mock required the existing Newton
  `is_floating_base` property; that fixture-only compatibility change is included.
- Four active-contact GPU cases pass **on each GPU**: FIXED/locked-D6 roots,
  eager writes and two successive graph replays with different rotations.
  Forces agree with fresh initialization within `rtol=1e-5, atol=1e-4 N`;
  generalized velocities and internal poses also agree. Both worlds retain
  active contact. All 32 checked solver calls/card have warning mask zero and
  passing external-contact capacity checks.
- Actual 32-world Keyboard MJWarp reset integration passes on RTX PRO6000 and
  GB300. Updated mocap pose is correct immediately after reset. Internal poses
  are correct after each of the next two original physics calls: zero position
  error and maximum rotation error below `1.7e-7 rad`. Internal derived poses
  may remain old before physics recomputes them; the updated mocap input is
  consumed before contact response. No observer adds a forward/reset/step.
- `uv run --no-sync isaaclab -f` passes, including the staged changelog.

GPU runs retain the reviewed process-local MJWarp 3.12 line-search correction,
not a new solver-budget change. The installed 3.12 versus declared 3.11 dependency
warning remains. Active fixture limits: `nconmax=2`, `njmax=8`; Keyboard 32-world
limits: raw 1152, broad 448, `nconmax=32`, `njmax=320`. No capacity drops reported.
All GPU children/parents were reaped and final source/idle guards passed.

This fixes the identified bug. It does not establish universal contact-law
parity, FPGS convergence, matched termination frequencies, or new performance.
Legitimate frequent contact terminations are acceptable. Previous task timing
tables have not been regenerated with this Lab fix.

## Pins and reproduction

Newton runtime stays `50dfa28d3aabe51f1b5b75721450efac2c35c5f1`, available on
`ooctipus/newton`, branch `ooctipus/fpgs-composed-world-20260912`. This report
branch has identical `newton/` and `tools/fpgs_bench/` runtime bytes. Lab's
original `1d8feb82d17dbfab8f0772de56f84deae2cb7974` worktree and interpreter
were preserved. Original handoff commits remain ancestors.

The checked-in regression is
`source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py` in Lab.
Using the original Lab interpreter and this fix's backend source first on
`PYTHONPATH`, run:

```sh
uv run --no-project --python /path/to/old-lab/.venv/bin/python \
  --with pytest --with pytest-mock python -m pytest \
  /path/to/fixed-lab/source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py \
  -k 'root_pose_writer or notify_model_change_before_solver_construction' -q
```

Set `CUDA_VISIBLE_DEVICES=''` for CPU. For GPU, select
`root_pose_writer_preserves_active_mujoco_contact_response and cuda`, one
process per GPU. Exact paired commands/source guards and local captures:

- Actual Keyboard: `/tmp/fpgs-keyboard-root-fixed-DFoCww/READY.md` and
  `/tmp/fpgs-keyboard-fixed-root-readback-paired32-20260913-01`.
  Manifest SHA256 `97d0b4935251f05e3ee2023f456e0df0e3e548fe86c3cee4a0cbd0e506582e6d`.
- Active GPU regressions: `/tmp/fpgs-contact-reset-gpu-tests-h8ylRG/run.py` and
  `/tmp/fpgs-contact-reset-active-paired-20260913-02`.
  Manifest SHA256 `62f4bc9f4e77dd5319858214c96b5fff69bc6caf2cdcd4b2a149d194d29900ea`.
  Attempt 01 failed in the diagnostic pytest-hook signature before any test;
  corrected wrapper attempt 02 passed. The failed record is preserved.
- Pre-fix Keyboard diagnostic: `/tmp/fpgs-keyboard-matched-reset-paired32-20260913-01`.
  That sample had zero robot force in changed reset worlds; it did not by
  itself demonstrate a contact-force consequence. The active fixture above
  closes that specific gap.

All timing produced by these diagnostic runs is discarded. Large artifacts
and temporary interpreter links remain local; no source worktree was overwritten.
