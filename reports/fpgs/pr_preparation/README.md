# FPGS PR-preparation evidence index

Status: **cleanup/review checkpoint, not a PR, upstream restack, default-on
promotion, or new benchmark**. The accepted runtime is Newton
`301b6688ea74a7372a007f03989664890f0d2540`. Later evidence and rejected
experiments are recorded in ledger commit
`3dc829a0f8fecf07c6e8a0182ac7163c0533103d`, paths
`reports/fpgs/TWENTY_HOUR_20260916.md` and `reports/fpgs/PUBLIC_PGS_20260917.md`.
Those ledger files are not part of the accepted runtime checkpoint.

The user selected `dylanturpin/newton-collab:fpgs-main` on September 19.
Its remote tip was verified at `65ab4b7973fafa81f6e17615600d2aae708b94e6`.
The separate integration branch `ooctipus/fpgs-pr-series-20260919` starts at
that exact commit. This cleanup history is not a safe upstream cherry-pick
list: porting must preserve newer upstream contact features and defaults.
Preserve all existing branches and SHAs referenced by dependency pointers.

## History and disposition

- Original Newton handoff: `31cf87f4694f873a027e41e2ca5e9ad441234456`, retained
  as an ancestor. The original [handoff report](handoff/HANDOFF_fpgs_20260911.md)
  and [design guide](handoff/fpgs_solver_shape_and_spirit_20260909.md) are bundled
  verbatim from Isaac Lab `2129e5628596f9908cc307482a01700f45b555d7`.
  [Provenance and interpretation](handoff/README.md) record their hashes,
  historical RTX 5090 scope and later user-relaxed numerical-identity
  constraints. Other original validation/reference artifacts remain external.
- [Inherited integration](../INTEGRATED_HANDOFF_20260914.md),
  [qualified kinetic composition](../KINETIC_QUALIFIED_INTEGRATION_20260915.md),
  [Allegro/Kuka composition](../FOURX_QUALIFIED_INTEGRATION_20260914.md), and
  [Keyboard](../KEYBOARD_LINEAR_STATE_20260915.md) retain earlier accepted
  work and its limitations. Start-window baseline:
  `ca0d427af809571bb5501f644c1a6e03990cd2a8`.
- Accepted opt-in successors: [G1 shell/CSR](../HEIGHTFIELD_SHELL_PAIR_CSR_20260916.md),
  [spectral contacts](../G1_SPECTRAL_TANGENTS_20260916.md),
  [checked simultaneous limits](../G1_LIMIT_JACOBI_20260917.md), and
  [Franka compact workspace](../FRANKA_COMPACT_WORKSPACE_20260917.md).
  The [composition report](../STRUCTURAL_PROGRESS_20260917.md) is a historical
  snapshot; its then-pending repeats are superseded by the evidence below.
- Rejected/unsupported experiments remain archival evidence, not accepted
  recipe features or comparison baselines. Examples include full-warp world
  grouping, register-residual/row-owned packets, width-43, and the ANY spectral,
  residual-reuse and Krylov variants. Some experimental source is inherited;
  presence in history does not establish acceptance. Preserve the ledger's
  measured-loss versus unimplemented-screen distinction when splitting code.
- Public MuJoCo PGS research and the G1/ANY integer-age momentum CPU screens
  in ledger `3dc829a0` produced **no accepted native optimization**. Do not
  include their experimental policies in the accepted recipe.

### Cleanup checkpoint, not a newly qualified runtime

The preparation branch after `301b` removes five
unpromoted solver modules and their dedicated probe/test entry points, with
explicit rejection of their retired flags; see the
[solver cleanup note](../PR_SOLVER_CLEANUP_20260917.md). It also replaces Python
source cloning in the retained sparse spectral/limit factories with an
explicit shared factory, with portable native-body/ABI controls. Collision
cleanup removes the standalone adaptive-manifold module, retains the needed
selection helper in CSR, and adds repository-local synthetic heightfield test
fixtures. Changelog fragments are consolidated into the existing
`changelog/` convention.

See [cleanup validation and upstream boundary](CLEANUP_VALIDATION.md) for
current CPU/native checks and remaining integration work. These are bounded
cleanup commits, not the complete historical feature restack or a mergeable
upstream PR. The timings below continue to describe the original pinned
measured sources, not the cleanup checkout; no new benchmark was run.

The local branch is `ooctipus/fpgs-pr-prep-20260917`, based on `301b6688`:

| Commit | Scope |
| --- | --- |
| `7196fff4` | Retire five unpromoted solver implementations and their exclusive probes. |
| `91b0de50` | Replace sparse Python factory cloning with explicit functions; preserve native bodies and ABI. |
| `67fd12bd` | Retire adaptive terrain dispatch, retain CSR helpers and package synthetic test fixtures. |
| `1f47ea84` | Preserve the original handoff, index evidence and limitations, and normalize legacy changelog names. |

This is an additive cleanup stack. It preserves the original research history;
it does not yet replace that history with a complete base-to-feature PR series.
No PR has been opened and no existing published branch has been rewritten.

The September 19 continuation is on
`ooctipus/fpgs-structural-resume-20260919`, based on `1f47ea84`. Its benchmark
source-selection repair does not modify Isaac Lab or the solver, reconstruct
the historical adapters, or produce a new timing result.

## Fair backend results, not incremental gains

Whole physics-graph milliseconds per **batched environment step**; ratios are
MJWarp/FPGS. RTX PRO 6000 is primary; GB300 is a separate device. All rows are
single-round, same-Newton-source backend comparisons at the recorded source,
with corrected MJWarp and Isaac Lab `53ee6b44`. They are not a fresh six-task
benchmark of `301b`. Full source pins and original manifest hashes are in
[evidence.json](evidence.json).

| Task | Worlds | RTX FPGS / MJWarp ms | Ratio | GB FPGS / MJWarp ms | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Franka lift | 16384 | 4.763000 / 10.620445 | 2.229781x | 4.643132 / 10.382176 | 2.236029x |
| KukaAllegro lift | 16384 | 10.520933 / 27.460161 | 2.6101x | 10.665790 / 25.755104 | 2.4147x |
| Allegro reorientation | 16384 | 15.146122 / 63.862091 | 4.216399x | 16.531282 / 81.857731 | 4.951687x |
| ANYmal-D flat | 16384 | 9.243896 / 36.874710 | 3.9891x | 9.490690 / 42.296333 | 4.4566x |
| G1 rough | 16384 | 12.542372 / 36.567342 | 2.915505x | 14.403645 / 34.704481 | 2.409424x |
| SO101 Keyboard | 4096 | 6.960440 / 27.851542 | 4.001405x | 6.054350 / 28.557772 | 4.716902x |

These runs retain task dt/substeps/iteration allowances and calibrated
capacities, 200 warmup steps, 40 unprofiled steps and 40 profiled steps. Shared
collision improvements apply symmetrically where each backend uses Newton's
collision path; Allegro MJWarp uses its native contact pipeline. Recorded
source, capacity and warning checks pass. These are not identical-trajectory
comparisons. The six-task RTX 4x goal is **not met**; do not round ANYmal up to
4x or treat Keyboard's single-round margin as robust. Environment-wall timing
is separate; neither metric is end-to-end RL training throughput.

### Incremental and total FPGS evidence

Balanced three-round AB/BA/AB medians, separately from the backend table:

| Comparison | RTX original → candidate ms | RTX speedup | GB speedup |
| --- | ---: | ---: | ---: |
| G1 corrected limits versus its local baseline | 13.213031 → 12.563802 | 1.051675x | 1.032415x |
| Franka compact workspace versus its local baseline | 5.058346 → 4.801236 | 1.053551x | Whole-physics gain not established |
| G1 **total** start-window `ca0d` → accepted `301b` | 15.634412 → 12.520301 | 1.248725x | 1.418934x |

The direct total G1 GB medians are 20.476167 → 14.430670 ms. Its separate wall
speedups are 1.096526x RTX / 1.225739x GB. Franka's repeated RTX wall result
is flat; its GB physics screen is flat (1.000361x), with worse wall timing.
Do not multiply the total G1 result by the limit-only increment or multiply
ratios from unrelated captures. The direct total run changes five flags only:
chain scan, pair CSR, CSR shell, spectral tangents and corrected limit Jacobi.

## Correctness and external ownership

**Isaac Lab prerequisite, separate change.** Commit
`53ee6b44c2334341305dbdf385a3916c6b140799` fixes root-pose writes that mutate
`model.joint_X_p` without notifying the active solver. Immediate
`JOINT_PROPERTIES` notification (or queuing before solver construction)
prevents stale fixed-root/MuJoCo mocap state, including graph replay. Its
articulation/rigid-object/manager changes and regression tests belong in the
Lab review, not a Newton performance squash. No new Lab change or parent
dependency-pointer update is part of this checkpoint.

**MJWarp correction, separate ownership.** The existing
[compatibility helper](../../../tools/fpgs_bench/mjwarp_linesearch_compat.py)
is an experimental, opt-in, process-local workaround for reviewed MJWarp 3.12.0
line-search endpoint/bracket and elliptic friction-loss cost-difference
roundoff. It pins source hashes, fails closed for unknown versions, changes
in-memory source only, and preserves budgets. It is not an installed-package
patch, Newton default, confirmed upstream fix, or warning suppression. Keep
its regression tests and `physics_work_modified=true` disclosure; do not use
uncorrected warning-inflated MJWarp timings as fair denominators. An upstream
MJWarp submission or a documented pinned dependency is a separate review.

**Qualification is bounded.** The composition reports 37 CPU passes/15 CUDA
skips and seven native passes per device. The latter ran under external load
and are correctness-only. Current/held factors, reset/notification, graph
lifecycle, capacities, fallback and physical impulse/velocity controls remain
required. Finite states, unchanged contact counts and benchmark guard success
are not proofs of Coulomb convergence, arbitrary-model support, or task/RL
accuracy parity. The original handoff and
[known solver-parity limitation](../ACTIVE_WRENCH_20260916.md) document that
unequal tangent step metrics can yield a different fixed point; G1's shared
spectral tangent step does not retroactively fix every inherited ANY/parallel
path. Preserve finite-budget tradeoffs and distinguish cone feasibility from
physical convergence. New paths remain opt-in with their original fallbacks.

## Clean-checkout gaps and minimum portable package

This index, JSON and verbatim handoff are portable documentation, **not a
reproduction bundle**.
Artifact IDs in JSON identify author-local captures; no downloadable archive
is supplied here. Do not interpret their absence as a passing test. Large
Nsight/raw-state artifacts should remain separate, with license/privacy review
and a stable checksum index before distribution.

**September 19 availability audit:** all 13 distinct benchmark artifact IDs
in this JSON are missing at their recorded `/tmp/{artifact_id}/manifest.json`
locations. The fixed-Lab import adapter, G1 checked adapter and cleanup GPU
logs referenced below are also absent. Preserve the historical numbers and
hashes as reported evidence, not currently reverified captures. The maintained
harness repair cannot recover the lost files or establish equivalence to those
adapters. A new run must retain its own recipe, checks and artifacts outside
temporary storage. The interrupted September 19 Kuka diagnostic is failed and
excluded from all performance tables.

| Existing entry point | Concrete external dependency | Minimal packaging action |
| --- | --- | --- |
| `chain_capture_20260916/{run.py,checked_sparse.py}` | Hash-pinned runner/observer under `/tmp/fpgs-g1-paired-gs-checked-t0NFbw6H` | Bring the reviewed adapter into maintained benchmark tooling; parameterize checkouts, retaining guards. |
| `compact_workspace_capture_20260917` | `/tmp/fpgs-kinetic-fixed-variants-TZRkIPYw/run.py` and another worktree's `checked_capture.py` | Same; do not replace paths without reconciling the pinned observer behavior. |
| Franka compact/kinetic tests | Another worktree's `test_world_scan_publication.py`; `/tmp/fpgs-franka-current-rows-paired512-20260912-01` | Repository-local reference helper plus a small redistributable fixture; explicitly separate optional saved-capture tests. |
| G1 physical tests | Hardcoded downloaded G1 USD cache, saved sparse-operator captures and replay helper | Deterministic portable fixture or documented pinned asset acquisition; keep actual topology/physical coverage. |
| G1 CPU control import chain | `limit_jacobi_control`/spectral helpers reach `coulomb_block_hybrid`, which reads `/tmp/fpgs-anymal-lazy-reference-zE5yUe/method.py` at import | Extract the necessary maintained CPU reference without an unrelated archive import. |
| Later fair adapters / strict node readers | Author-local `/tmp` scripts and captures | Preserve a minimal reviewed recipe and small redacted summaries; do not bundle every experiment framework. |

The [base harness](../../../tools/fpgs_bench/README.md) already supports explicit
Lab/Newton checkouts and records settings, hashes, budget/capacity and process
guards. Reuse it rather than creating another benchmark framework. A portable
recipe must preserve actual-owner checks, source pre/post checks, sticky flags,
current validity, task budgets, calibrated capacities, device-idle checks and
physics-versus-auxiliary graph scope. Do not count overlapping kernel intervals
twice. Failed captures and skipped prerequisites must remain visibly excluded.

Minimum review/test package to finish before claiming fresh-checkout support:

1. Keep portable harness controls (`test_compare_backends`,
   `test_compare_variants`, `test_checked_capture`) and separate optional
   `test_mjwarp_linesearch_compat`, `test_mjwarp_huber`, `test_mjwarp_elliptic`.
   These tool tests are not discovered by the default `newton.tests` suite.
2. Retain accepted feature controls (G1 chain/spectral/limit, heightfield
   CSR/shell, Franka compact/kinetic) with repository-local imports/fixtures;
   preserve regression-first, malformed/unsupported routing and CPU/CUDA
   distinctions. Do not silently convert missing-data failures into success.
3. Publish one exact task-recipe/capacity/source manifest and a small sanitized
   result summary per comparison, with hashes of external raw artifacts.
   The numbers below do not themselves reconstruct all command-line flags.

## Suggested logical review order

1. Port onto the selected `fpgs-main` pin, reconciling inherited FPGS dependencies
   before restacking; preserve referenced history. This cleanup branch is not
   an upstream-ready series merely because its checkpoint was accepted locally.
2. Review the Lab stale-reset correction separately; retain its own tests and
   changelog. Keep any MJWarp correction/compatibility submission separate too.
3. Split Newton shared collision changes from solver changes, with matching
   regression tests and fallback/lifecycle contracts in each logical commit.
   Within solver changes, preserve dependency order: inherited qualified owners,
   G1 state/rows/spectral/limit composition, then Franka compact lifetime change.
4. Consolidate portable test helpers and existing capture adapters with their
   tests; follow with concise feature documentation, evidence index and proper
   changelog fragments. Exclude rejected prototypes from an accepted-feature PR
   unless explicitly proposed as experimental API, not by accidental ancestry.
5. Re-run the portable suite and selected native physical controls after any
   source/test relocation. New numerical edits or changed recipe/dispatch need
   new qualification; historical measurements do not validate a restack by
   assertion. No benchmark or runtime test was run for this documentation edit.
