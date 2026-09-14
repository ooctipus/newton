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
four-times-corrected-MJ goal across all six tasks. No GPU job, new benchmark
framework, timestep, substep, iteration or capacity change is part of this
source integration.

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
`a08f7f193d01f642dbf3c489a0f742d45a341a46` for inclusion after the code merge.

Kuka's qualified 6c229 successor retains repeated physics medians
12.506653 -> 11.119767 ms RTX and 12.033355 -> 11.339816 ms GB against baf.
Its GB wall evidence remains mixed, including the original paired regression;
see [the retained Kuka report](KUKA_KINETIC_INTEGRATION_20260914.md).
The combined source still requires root's integrated checks. Allegro has
crossed its measured RTX target; the all-six-task goal remains unmet.
