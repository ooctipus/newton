# Qualified Allegro and Kuka composition

## Pre-code scope

Compose exact Allegro `7fca2fdf5ef090e1e28fc9e327ce9f4de5e879df` (including
direction cells `92507273fc6a6fdc7a219d928a3d6cf942e473d0`) onto Kuka
`298c038e16ab0151c6041d3ed3d6633eb179b2d7`, whose measured runtime is
`6c2297ca3f7022df7e20ca0486f75c6ff818f29c`. Work only in this new
`ooctipus/fpgs-fourx-qualified-20260914` worktree. Preserve both original
trees, their measurements and all accepted G1/Franka/collision features.
Use a real merge to retain both ancestors; inspect conflicts rather than
replacing the solver wholesale. Fixed Lab remains
`53ee6b44c2334341305dbdf385a3916c6b140799` without source or pointer changes.

This composes separately qualified task-specific implementations. It is not
a fresh measurement of the composed tree or completion of the user's
four-times-corrected-MJ goal across all six tasks. The initial source merge
uses no GPU job or new benchmark framework; the later root-owned focused CUDA
checks are recorded below. No timestep, substep, iteration or capacity change
is part of this integration.

## Required compatibility gates

Keep every Kuka retained source pin and all 249 native/descriptor expectations
unchanged. Its default MF native body and ABI must remain those of 6c229/ad42;
preserve the accepted optional single-factor branch. Keep Allegro's complete
keyed module and retained original parallel factory equivalent to 7fca, with
all geometry/provider modules byte-exact. Retain both notification paths,
source ownership and reset/publication semantics. Direction cells and both
task-specific kinetic owners remain opt-in; unrelated default paths must not
acquire an owner or new numerical algorithm.

Run the existing Kuka/retained-feature CPU controls, the eight Allegro CPU
controls and applicable cells controls, then full repository pre-commit.
Do not regenerate frozen oracles, introduce new numerical tolerances or call
CUDA skips physical validation. Root reviews the frozen source before any
integrated GPU comparison or push.

## Merge resolution and source proof

The only textual conflict was `notify_model_changed`: preserve Kuka's one
current shared `plan_snapshot` and every independent validator, including
Allegro, before `kinetic.invalidate_model_changed` or any cache mutation.
All constructor, step, row, fallback, reset and export seams otherwise merge
directly. There is no numerical-body or source-pin reconciliation in this
composition. The failed intermediate Allegro55 remains an ancestor, not the
comparison baseline or active final implementation.

All 21 Kuka kinetic modules, its publication/joint owners, `kinetic_source.py`
and the frozen 249-record test remain byte-exact against 6c229. The entire
`_get_pgs_solve_mf_gs_kernel` definition is also byte-exact. Its selected
192/64/29 factor/components native body retains SHA-256
`1ea22a605c07900eb7365cf8c7a28b3dedc08de7ccefe3a0a7257c108882c649`,
original key and ABI. The accepted optional G1 single-factor native-body
control also passes, without rewriting its expectation.

All geometry/sim files and `allegro_kinetic_rows.py` are byte-exact against
7fca. The entire original `_get_pgs_solve_parallel_kernel` definition is
byte-exact (SHA-256
`afd56eb96ad906b42a79ae05007b5a601cc066f34f6fa06ca6803d358cf201f0`).
CPU-side CUDA-source construction independently compares all eight original
and keyed 32/64/96/128-row kernels against 7fca: their generated native bytes,
keys and ordered parameter/type ABIs match. No GPU or AOT benchmark was run
for that comparison. The keyed ABI differs from the ordinary ABI as intended;
each matches its corresponding qualified original, not each other.

This eight-kernel identity proof covers the PARALLEL tiers, not every Allegro
native entry. Allegro's scalar fallback uses the retained Kuka MF factory,
whose restored default interpolation formatting differs from 7fca. The
observed generated module suffix changes from `03bd` to `772c`. Do not claim
scalar-fallback byte identity with 7fca: its physical fallback behavior passes
the integrated CUDA controls below. No frozen Kuka expectation is regenerated.

Default-off cells allocate no masks and retain the original query owners.
Their built-in box-support hook resolves to the original function for existing
providers. Both kinetic flags remain default off. Their admissions are
disjoint: Allegro requires the 16+6, 12/24-sweep parallel law; Kuka requires
23+6 and the eight-sweep nonparallel law. This does not promise that arbitrary
global opt-in combinations or unsupported tasks become newly supported.

## CPU qualification

Final selected batches run 119 tests: 91 execute PASS and 28 explicitly skip
CUDA-only coverage. They comprise 67 Kuka/publication/Franka notification
tests (66 executed), eight Allegro CPU plus eleven cells tests (15 executed),
and 33 sparse/single-factor/mimic/connect/spring/pre-elimination controls
(ten executed). The cells controls include all ten cooked hulls and 55,724
supported queries. The full frozen Kuka native/descriptor oracle passes.

Regression-first integration checks initially expose two old notification
`SimpleNamespace` fixtures without the new constructor-initialized
`_allegro_kinetic_rows` member. Add only `None` to those two fixtures; preserve
the actual 7fca production field access and all assertions. The final full
batch passes. No new test framework, expectation regeneration or numerical
tolerance change was introduced. CUDA-hidden initialization messages are
not GPU execution; skipped tests are not physical-validation passes.

Full `uvx pre-commit run -a` passes after the merge resolution and fixture
updates. No hook rewrites any native source or frozen oracle.

## Qualified predecessor results, not integrated measurements

The completed same-source Allegro backend comparison is
`/tmp/fpgs-allegro-keyed-corrected-backends-paired16k-20260914-01`.
Both source labels are clean 7fca, fixed Lab53, with 12 successful children,
the original checked-capture guards and three paired alternating rounds.
Medians in ms/environment step:

| Card | Allegro FPGS physics | Corrected MJ physics | Ratio | FPGS wall | MJ wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX PRO6000 | 15.103509 | 63.536818 | 4.206759x | 23.131495 | 71.449856 |
| GB300 | 16.664395 | 82.023917 | 4.922106x | 24.920978 | 91.547777 |

Every same-round RTX ratio exceeds 4x. GB MJ round three is 94.850148 ms,
a retained outlier; the displayed median does not discard it. Allegro's
incremental 925-cells-to-keyed discovery is a RTX gain, 16.340675 -> 15.201010
ms, but a GB loss, 16.517619 -> 16.719848 ms. Do not conflate the corrected-MJ
ratio with a cross-GPU incremental improvement. Environment wall is not
full RL-training throughput.

Original Allegro MJ uses native MuJoCo contacts, not Newton collision.
Direction cells are genuinely not applicable there; no contact-mode switch
or withheld relevant shared optimization funds the ratio. FPGS retains raw
286720/broad524288/dense192/MF64/prop192 and 12/24 iteration allowances;
MJ retains calibrated njmax112/nconmax22 and the pinned original line-search
correction. Exact local prerequisites and command remain at
`/tmp/fpgs-allegro-keyed-backends-akOBZj8M/LAUNCH.md` (adapter e82a6102,
original driver798/fixed-import c062). Root's full Allegro result report is
preserved separately by report-only commit
`a08f7f193d01f642dbf3c489a0f742d45a341a46`, included unchanged after the
code merge.

Kuka's qualified 6c229 successor retains repeated physics medians
12.506653 -> 11.119767 ms RTX and 12.033355 -> 11.339816 ms GB against baf.
Its GB wall evidence remains mixed, including the original paired regression;
see [the retained Kuka report](KUKA_KINETIC_INTEGRATION_20260914.md).
The focused integrated CUDA checks below pass; whole-physics throughput has
not been remeasured on the combined tree. Allegro's qualified predecessor has
crossed its measured RTX target; the all-six-task goal remains unmet.

## Focused CUDA qualification on the composed source

Root executes the six unchanged selectors below on clean report tip
`1921e214b59a80046cb3804df6aee7d95e23bb02`, whose runtime is code merge
17f8a19a. Both owned sessions finish successfully, with no skipped tests:

| Card | Tests passed | Test duration | Reaped session | Exit |
| --- | ---: | ---: | ---: | ---: |
| RTX PRO6000 | 6 | 16.141 s | 97544 | 0 |
| GB300 | 6 | 16.971 s | 89940 | 0 |

The four Allegro selectors retain current/held physical defects of
3.39e-8--3.90e-8 and pass actual production fallback, reset and graph
grow/shrink, including the merged notification/full-step path. The two Kuka
selectors exercise the retained joint/predictor/publication owner and queue
transitions. They are NOT full private-kinetic trajectory requalification.
These durations include test execution/module loading, not representative
whole-physics or environment-wall timing. No new performance claim follows.

Reproduce from the composed worktree using the fixed Lab interpreter, one
leased GPU UUID per process on each card, and require six passes with zero
skips. The only environment placeholders below are that process's same UUID:

```bash
env CUDA_VISIBLE_DEVICES='<leased GPU UUID>' GPU='<leased GPU UUID>' \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/home/octi/Projects/newton-fpgs-fourx-qualified-20260914 \
  uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python \
  python -m unittest \
  tools.fpgs_bench.test_allegro_kinetic_rows.TestAllegroKineticRowsCUDA.test_native_current_held_rows_and_fallback \
  tools.fpgs_bench.test_allegro_kinetic_rows.TestAllegroKineticRowsCUDA.test_native_parallel_matched_current_inputs \
  tools.fpgs_bench.test_allegro_kinetic_rows.TestAllegroKineticRowsCUDA.test_native_graph_refresh_and_grow_shrink \
  tools.fpgs_bench.test_allegro_kinetic_lifecycle.TestAllegroKineticLifecycleCUDA.test_production_step_held_refresh_fallback_reset_and_graph \
  tools.fpgs_bench.test_kuka_joint_world.TestKukaJointWorld.test_actual_native_cuda \
  tools.fpgs_bench.test_kuka_joint_world.TestKukaJointWorld.test_two_actual_graphs_empty_invalid_and_regrowth
```

## Frozen handoff

Code merge `17f8a19ae1a5c388838b1d9000d96f27ecd16b77` has parents 298c and
7fca. The subsequent merge of a08 changes only reports. The original Kuka
and Allegro trees remain untouched by this composition, including the failed55
ancestry and the inherited `31cf87f4694f873a027e41e2ca5e9ad441234456` fork.
No source pin or parent dependency pointer is advanced. Root's subsequent
focused CUDA results are recorded above; this report update changes no runtime
and launches no additional GPU work. Root owns review and pushing the handoff.
