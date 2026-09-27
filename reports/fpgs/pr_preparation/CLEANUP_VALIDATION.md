# PR-preparation cleanup validation — 2026-09-17

This is a cleanup checkpoint on the measured handoff lineage, not an upstream
restack or a new performance result. Original checkpoint `301b6688`, ledger
`3dc829a0`, worktrees and large captures are unchanged. No Isaac Lab code,
solver iteration/substep budget, parent dependency pointer, or installed
MJWarp package is modified.

## Changes and checks

1. Retire five unpromoted solver implementations and their exclusive probes
   and tests. Their selectors now raise a clear error when explicitly enabled;
   unset/zero retains the measured routes. See
   [solver cleanup](../PR_SOLVER_CLEANUP_20260917.md).
   Portable CPU batches: 32 passes/one CUDA skip and 15 passes/one CUDA skip.
   These batches overlap; they are not 47 unique tests.
2. Retire the unpromoted adaptive terrain exporter while preserving its shared
   selection score in the retained CSR implementation. The shared score and
   five synthetic scene definitions are AST-identical to the previous source.
   Retained CSR/shell CPU fixtures no longer require the archived qualification
   Python file. The expanded collision CPU batch has 62 passes and 28
   CUDA-related skips. Existing saved geometry and Allegro hull assets were
   present for that run; this is not a claim that every test is portable.
   A follow-up retired-selector/fixture suite has 16 passes and 14 explicit
   skips (eight CUDA, six archived). The retirement regression fails before
   the fix. Missing captured assets are explicit skips, never passing coverage.
3. Replace Python factory-source inspection/replacement/execution in the sparse
   spectral and limit entry points with a shared explicit factory and declared
   CFM-aware ABI. Native proposal fragments are unchanged; this does not remove
   all native-string specialization elsewhere in the solver.
   Six native-body SHA256 values, kernel names and argument order match the
   pre-refactor `301b6688` checkpoint, including scalar, metric and historical
   contact-block paths. The new factory regression fails before the refactor.
4. Preserve the original handoff and design guide, source pins, archived
   decisions and fair/incremental/total performance distinctions. See
   [evidence index](README.md). No performance measurement was repeated for
   cleanup.

### Sparse native checks on both GPUs

Run the following from this checkout, in the existing pinned runtime
(Python 3.12, Warp 1.17.0, CUDA toolkit 12.9). Substitute the interpreter path
and physical device ID for the host. Each invocation exposes one device,
which the tests address as `cuda:0`.

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
uv run --no-project --python /path/to/pinned/python3.12 python -m unittest -v \
  newton.tests.test_feather_pgs_sparse_factory \
  tools.fpgs_bench.test_sparse_spectral_tangents \
  tools.fpgs_bench.test_sparse_limit_jacobi
```

Repeat with `CUDA_VISIBLE_DEVICES=1` for the second device. Results:

| Device | Discovered | Passed | Skipped | Failed |
| --- | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | 16 | 16 | 0 | 0 |
| GB300 | 16 | 16 | 0 | 0 |

Coverage includes native contact transactions, anisotropy, CFM, incoming
impulses, current/held state, CUDA graph replay, empty state, simultaneous
limit prefixes, rejection fallback and the saved sixteen-input controls.
Those existing physical tests still depend on author-local assets; only
`newton.tests.test_feather_pgs_sparse_factory` is self-contained. A native-body
hash is a refactor guard, not proof of physical convergence or future timing.

### Retained collision native checks on both GPUs

The first broad module invocation omitted the original fixture's grouped
lane/masked-row recipe: 39 tests ran per card, with two loaded-case setup
failures per card. This inherited prerequisite is documented in
`HEIGHTFIELD_PAIR_CSR_20260916.md` and is not a physics regression. The test
now scopes construction-time lane/worker flags and explicitly checks the
import-time masked-row prerequisite. Runtime defaults and physical gates
are unchanged.

The selected retained shell/CSR class and retirement tests then passed all
23 tests per card, with no skips or failures, including nine loaded cases
for each of the corrected-CSR and shell/CSR arms. Reproduction:

```sh
CUDA_VISIBLE_DEVICES=0 FEATHER_PGS_GROUP_LANES=16 FEATHER_PGS_ROWS_MASKED=1 \
NEWTON_NARROW_PHASE_THREADS_X=4 NEWTON_HEIGHTFIELD_WELD_FLAT_SEAMS=1 \
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
uv run --no-project --python /path/to/pinned/python3.12 python -m unittest -v \
  tools.fpgs_bench.test_heightfield_shell_pair_csr.TestHeightfieldShellPairCSR \
  newton.tests.test_heightfield_retired_experiments
```

Repeat with physical device ID 1. Author-local log directory:
`/tmp/fpgs-pr-prep-validation-fgGIvg`. SHA256:

- `shell-rtx.log`: `b8aed5d5e82b913a6db8e7f7e507164fe9de84f95abdfc5bd7e9ca55dc5e1ffe`
- `shell-gb300.log`: `3aa7307bdc23b312fbbbaa8447318b6d64c5944a0e3cc11b120a0195b1f54854`

The older standalone CSR loaded comparator retains documented old-arm
sliding/yaw/step failures; it is not the selected shell/CSR comparison and
was not declared passing. No inherited failure was removed or threshold
relaxed. These are targeted cleanup controls, not a claim that the entire
repository test suite passes.

Full tracked-file pre-commit and scoped new-file checks pass. Towncrier draft
rendering initially exposed a legacy `.feature` filename; it and the legacy
Franka orphan identifier were renamed to the documented fragment convention.
The non-mutating draft now renders successfully.

## Upstream integration boundary

Live `dylanturpin/newton-collab:fpgs-main` was verified again on September 19 at
`65ab4b7973fafa81f6e17615600d2aae708b94e6`. It is not an ancestor of the
handoff or accepted runtime; their merge base is
`e452daa3a67e008f9dcef06fe1b809b36cfd386a`.

A snapshot of the accepted tree onto that branch would remove persistent
friction patches (enabled by default there), material contact compliance,
load-bounded torsion, shared friction/filter helpers and reviewed fixes in
restacked tranches #111/#114/#115. Broader replacement would also lose unrelated
upstream corrections. Specialized row builders and source-admission hashes
assume the older contact contract. Turning patches/compliance/torsion off is
not sufficient evidence of equivalence: upstream point-friction updates also
changed.

Consequently this checkpoint must not be called a mergeable `fpgs-main` PR.
The user selected `fpgs-main` on September 19. Start at its exact pin, port shared
correctness fixes first, preserve its existing features/defaults, then port
and requalify the opt-in representations with explicit admission/fallback.
For the handoff lineage, the accepted implementation can instead be restacked
by dependency group without pretending it includes later upstream features.

Remaining work before a complete review series: semantic port onto that base,
logical feature restack, remaining source-generation cleanup,
and portable physical fixtures/reproduction adapters. These are visible
limitations, not passing checks or evidence that the 4x target was met.

### September 19 evidence availability

The historical results above have not been rerun. Their `/tmp` cleanup logs,
the fixed-Lab adapter and all 13 benchmark manifests indexed in `evidence.json`
are now absent at their recorded locations. This does not change the recorded
results, but prevents rechecking their original raw evidence on this host.
Do not count missing prerequisites as passing tests or regenerate files under
historical hashes. Future captures must use persistent output directories.

### September 19 maintained benchmark source selection

The shared prepared Lab interpreter resolved `isaaclab`, `isaaclab_newton`,
`isaaclab_tasks` and `isaaclab_assets` to the older `fpgs-opt-20260910`
editable checkout. Selecting the corrected Lab directory and checking the
interpreter prefix/Newton import alone did not prevent this mismatch. The
interrupted Kuka diagnostic was marked failed and contributes no timing result.

The maintained driver now explicitly selects all Lab source package roots
alongside Newton, refuses foreign preloaded modules, and checks actual imports
and source hashes before/after capture. The parent validates the returned
evidence independently. Source-owned generated namespaces are traced to their
owning module without forcing lazy imports. No Lab source or shared install is
modified; import checks remain outside timing windows.

The editable-finder regression fails against the previous driver. All 76
portable tool tests pass with no skips:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=tools/fpgs_bench \
uv run --no-project --python /path/to/prepared/python3.12 python -m unittest \
  test_compare_backends test_compare_variants test_checked_capture
```

A real CUDA-hidden preflight on the prepared interpreter verified the selected
Newton and corrected `contact-reset-20260913` Lab packages, checking 386 loaded
modules. This is an import/reproduction check, not a simulator run, capacity
qualification, restored historical adapter or new performance measurement.

Independent review of the three earlier runtime cleanup commits found no
actionable regression. Its fresh CPU subset passed 21 tests with 14 explicit
skips (eight CUDA, six archived). It does not replace the unavailable loaded
physical evidence. The target-based port is maintained separately on
`ooctipus/fpgs-pr-series-20260919`; the first isolated external-contact seed fix
there is `7c0e3669`, not a new optimization of this research branch.
