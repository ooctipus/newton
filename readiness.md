# PR #140 readiness qualification — 2026-09-27

Local author-run evidence, not GPU CI or training-equivalence certification.
The broad FPGS run is **not green**: six assertion failures reproduce on clean main.
The separate PR-introduced mock-fixture error found here has been repaired.

## Pins

| Source | Commit |
|---|---|
| Clean `fpgs-main` | `5238407d320823e71a5a4623c2c83c293d7b00fd` |
| Frozen pre-cleanup PR control | `f514526ab4139b1f1b051a04c9e654a71740c28e` |
| Runtime cleanup and canonical-row/oracle tests | `efbf1835adb55365cf414b3e3f43be8695d2dea3` |
| Final cleanup, including the mock-fixture fix | `c39323114a8999eefd6d8ea0f30daa54fe7cd229` |

The 78-test suites and candidate broad run used development source subsequently
committed as `efbf1835`. The broad run overlapped test-only Ruff whitespace
formatting; runtime code was unchanged. These are not clean exact-head captures.
The control and main checks used clean worktrees at the pins above.

## Runtime cleanup

Relative to the frozen pre-cleanup PR, production code is 18 lines shorter.
It no longer publishes an unused packed L, allocates dense triangular-solve
scratch for compact paths, or builds unused dense kernels. The factor builder
reuses ancestry masks and omits redundant inverse initialization; sparse PGS
removes unused local row arrays. Numerical updates and solver budgets are unchanged.

For G1 at 16,384 environments, the removed persistent float allocations are
`(434 + 2 * 43) * 16384 * 4 = 34,078,720 bytes = 32.5 MiB`: one packed factor
and two DOF scratch arrays. This is allocation accounting, not measured peak
VRAM or a whole-process memory reduction.

## Targeted tests

| Run | Discovered | Passed | Skipped | Failures/errors | Log |
|---|---:|---:|---:|---:|---|
| Clean final `c3932311`, RTX | 89 | 88 | 1 | 0 | `final-gpu0.log` |
| Clean final `c3932311`, GB300 | 89 | 88 | 1 | 0 | `final-gpu1.log` |
| Nine-module development suite, RTX | 78 | 77 | 1 | 0 | `fixed-gpu0.log` |
| Nine-module development suite, GB300 | 78 | 77 | 1 | 0 | `fixed-gpu1.log` |
| Compliance safety after mock fix, RTX | 11 | 11 | 0 | 0 | `compliance-fixed-gpu0.log` |
| Compliance safety after mock fix, GB300 | 11 | 11 | 0 | 0 | `compliance-fixed-gpu1.log` |

The clean-final runs completed in 21.542 s on RTX and 22.403 s on GB300.
They cover the nine modules below plus `test_feather_pgs_contact_compliance_safety`.

Nine modules: `sparse_mass_matrix`, `sparse_contacts`, `sparse_pgs`, `sparse_solver`,
`contact_controls`, `mimic`, `connect`, `springs`, and `preelim`, each prefixed by
`newton.tests.test_feather_pgs_`. The inherited skip is
`test_feather_pgs_preelim.TestFeatherPGSPreelimination.test_dense_warmstart_preserves_projected_closure`,
an explicitly obsolete drive-only warm-start fixture.

## Full-suite regression classification

| Source / GPU | Discovered | Passed | Failures | Errors | Skips | Log |
|---|---:|---:|---:|---:|---:|---|
| Cleanup candidate / RTX | 508 | 500 | 6 | 1 | 1 | `candidate-all.log` |
| Pre-cleanup PR / GB300 | 508 | 499 | 7 | 1 | 1 | `control-all.log` |
| Main / GB300, seven shared failing names only | 7 | 1 | 6 | 0 | 0 | `base-failing-names.log` |

All six shared assertion failures reproduce on main with matching discrepancies:

| Module / class | Test | Discrepancy |
|---|---|---|
| `test_feather_pgs_colored_unit_capacity.TestFeatherPGSColoredUnitCapacity` | `test_mixed_gap_cuda_graph_matches_eager` | Unit offset `4 != 2` |
| Same | `test_mixed_gap_matches_serial_propagation` | Unit offset `4 != 2` |
| `test_feather_pgs_friction_patches.TestFeatherPGSFrictionPatches` | `test_default_friction_preserves_free_rolling` (`geometry='mesh'`) | Difference 0.02379906 exceeds 0.02 |
| `test_feather_pgs_mass_update_interval.TestFeatherPGSMassUpdateInterval` | `test_cached_fk_id_matches_forced_recomputation` | Maximum `joint_q` difference 0.00014126 exceeds 2e-6 |
| Same | `test_cached_fk_id_matches_forced_recomputation_for_floating_base` | Maximum `joint_q` difference 0.00524180 exceeds 2e-6 |
| Same | `test_model_change_request_refreshes_a_reuse_step` | Mask shape `[0, 0] != [0]` |

The PR error was **not inherited**:
`test_feather_pgs_contact_compliance_safety.TestContactComplianceSafety.test_persistent_patch_guard`
passed on main but errored on both PR versions because its `SimpleNamespace`
omitted `_sparse_mass_matrix_size`. Commit `c3932311` adds the inactive-path field
to that mock; production guards and tolerances were unchanged. The subsequent
11-test passes do not retrospectively change the full-suite counts above.

## Canonical-row equivalence check

Only the pre-cleanup control additionally failed
`test_feather_pgs_sparse_solver.TestFeatherPGSSparseSolver.test_two_articulations_share_contact_solve`.
Independent contact allocation can give finite-budget PGS different row orders.
The corrected test matches semantic order while preserving contact bundles and
index mappings; production ordering and tolerances remain unchanged. It passes
in both clean-final 89-test suites, both earlier 78-test suites, and the candidate
full suite.

The corrected test also passed ten consecutive development runs on each GPU
(RTX 4.069 s; GB300 4.759 s). These used `f514526a` plus then-dirty cleanup/test
edits and are recorded only in tool transcripts, sessions `4749` and `33511`;
they are not clean-head captures or persisted log files.

## Reproduction and evidence scope

Logs named above are bundled in [readiness-test-logs.tar.gz](readiness-test-logs.tar.gz)
and originate from `/tmp/pr140-readiness-SrfL71/` on the test host.
Use `CUDA_VISIBLE_DEVICES=0` for RTX or `1` for GB300, `PYTHONPATH=<checkout>`, and
`uv run --no-sync --project /home/octi/Projects/fpgs-pr1-20260921/isaaclab python`.
Full discovery: `-m unittest discover -s <checkout>/newton/tests -p 'test_feather_pgs*.py'`.
Focused checks use `-m unittest` followed by the qualified module/test names above.

This is a failure-identity comparison, not cross-GPU performance measurement.
No running training job was interrupted. Passing targeted tests does not prove
training equivalence, universal physical convergence, or that inherited failures
are harmless; none of those claims is made here.

## Cleanup performance checks

These compare pre-cleanup PR `f514526a` with clean `c3932311`, **not main with PR**.
16,384 environments, CUDA graphs on, video off, unchanged task recipes, budgets,
capacities and shared Lab source. Both revisions explicitly enable the sparse path.
The first pass runs both GPUs simultaneously: 200 warmup steps, 40 environment
steps, followed by 40 separate physics steps; before revision first, after second.

| Task | GPU | Physics ms, before → after | Env FPS, before → after | Physics gain | Env FPS gain |
|---|---|---:|---:|---:|---:|
| G1 flat | RTX PRO6000 | 16.934 → 16.933 | 575,355 → 559,533 | 1.000× | 0.972× |
| G1 flat | GB300 | 14.586 → 14.559 | 602,501 → 599,625 | 1.002× | 0.995× |
| KukaAllegro lift | RTX PRO6000 | 22.521 → 22.426 | 398,935 → 381,868 | 1.004× | 0.957× |
| KukaAllegro lift | GB300 | 21.131 → 21.212 | 412,884 → 404,564 | 0.996× | 0.980× |
| Cassie flat | RTX PRO6000 | 8.367 → 8.326 | 882,787 → 856,990 | 1.005× | 0.971× |
| Cassie flat | GB300 | 9.191 → 9.161 | 849,827 → 828,027 | 1.003× | 0.974× |

Physics changes are all below 0.5%. Environment FPS was 0.5–4.3% lower in this
first pass. Matching call counts do not prove matching reset populations: the
capture records one batch-reset call per step, but not each batch size. The
increase lies in host-measured reset/reward/observation scopes, which can include
GPU waits and must not be interpreted as pure CPU cost.

The largest difference, Kuka, was repeated with **200 environment timing steps**
(200 warmup and 40 separate physics steps unchanged), in reversed revision order:
after first, before second, still both GPUs simultaneously.

| Task | GPU | Physics ms, before → after | Env FPS, before → after | Physics gain | Env FPS gain |
|---|---|---:|---:|---:|---:|
| KukaAllegro lift | RTX PRO6000 | 21.915 → 21.992 | 392,344 → 383,479 | 0.997× | 0.977× |
| KukaAllegro lift | GB300 | 20.852 → 20.999 | 394,761 → 393,880 | 0.993× | 0.998× |

Finally, the same longer Kuka check ran **RTX alone**, before then after, with
GB300 idle. Its environment FPS difference narrowed to 0.45%.

| Task | GPU | Physics ms, before → after | Env FPS, before → after | Physics gain | Env FPS gain |
|---|---|---:|---:|---:|---:|
| KukaAllegro lift | RTX PRO6000 | 22.102 → 21.932 | 398,459 → 396,658 | 1.008× | 0.995× |

The larger simultaneous-run difference did not reproduce in the isolated check;
this is evidence against treating the first pass as a stable code regression, not
proof of its cause or of universal performance neutrality. Small differences are
not repeat-qualified. This cleanup claims reduced code/storage, **not speedup**.
The primary 20-task main→PR table remains pinned to `a25b4c45`.

All 18 captures passed source/import admission, finite-state, public-constraint,
sticky row/factor and full-log warning checks. All 72 recorded artifact hashes
were verified. Collision boundary snapshots remain only boundary checks.
Unrounded results, source pins, matching Lab delta, capacities and artifact hashes
are in [readiness.json](readiness.json). Raw capture paths are local; large Nsight
captures are not distributed.

Reproduce using the existing `benchmark.py` command in [README.md](README.md),
with either PR revision above and `sparse_mass_matrix=True` for both, selecting
`--task g1-flat --task kuka --task cassie-flat`. Use a fresh output directory and
the appropriate `--gpu-index`/`--gpu-uuid` for each device. For the longer controls,
select `--task kuka --steps 200`; run on both GPUs concurrently or RTX alone as
labeled. No new benchmark implementation is required.
