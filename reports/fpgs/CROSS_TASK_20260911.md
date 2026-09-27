# Cross-task structural optimization gate

This continues [the inherited handoff and reopening](REDO_20260911.md), from
Newton `a18951df4fb89e3632e1701e65ea63e59b4c7398` on the `ooctipus/newton`
fork. The user requested a multi-task benchmark before selecting the next
implementation. No new runtime optimization is claimed by this document.

## Contract

- Change only Newton collision/FPGS implementation; leave Isaac Lab sources,
  task physics, timestep, substeps and iteration allowances unchanged.
- Numerical convergence and task behavior are the acceptance criteria, not
  bit identity, unchanged contact counts or per-operation formal certificates.
- RTX PRO 6000 is primary. Launch the same batch on RTX and GB300 concurrently,
  one owned job per device; reap profilers and verify idle before the next batch.
- Preserve original worktrees, captures and handoff lineage. Do not update any
  parent dependency pointer during this study.
- Keep two distinct references: pre-redo FPGS for additional improvement, and
  MJWarp for the backend comparison. Never compare against a slow prototype
  as though it were the accepted runtime.

## Mandatory decision gates

1. Benchmark KukaAllegro, Allegro, Franka, Keyboard SO101 and AnymalD before
   selecting the next structural implementation. Reuse the existing unchanged
   Lab harness and Newton benchmark driver. A one-round sweep is triage only;
   confirm speedup claims with balanced repeated whole-physics timings.
2. Write a candidate card before coding: exact work removed, affected tasks,
   expected whole-step saving in milliseconds, added work and fallback cost,
   physical checks, and a falsifiable stop/go test. A component acceptance
   percentage or operation-count reduction is not an end-to-end cost model.
3. Require a credible route to the requested 2–4x whole-physics target. A first
   structural milestone should plausibly remove at least 10% of whole time;
   do not spend this effort on sub-percent tuning. This is a prioritization
   threshold, not permission to report a 10% gain as achieving 2x.
4. Test the smallest integrated causal implementation early. Measure classifier,
   representation conversion, materialization, synchronization, fallback,
   integration and publication costs as well as the proposed fast kernel.
   Keep efficient existing producers unless replacing them is the hypothesis.
5. If theory and timing disagree, diagnose the specific gap before rejecting
   the algorithm. Allow one targeted corrective experiment per identified
   cause. Reopening then needs new evidence, not another layout/tile sweep.
   Set an initial 90-minute experiment checkpoint; no silent extension beyond
   two hours without a recorded new cause and revised measurable hypothesis.
6. Correctness gates must address actual numerical/physical harm. Compare
   residuals, complementarity, momentum and stability with the original law
   and allowance; include fallback and reset transitions. Do not build a
   stronger proof contract than the user requested.
7. At every checkpoint record one of: measured integrated gain, concrete
   diagnosed loss, or unvalidated hypothesis. Report elapsed time, next test
   and stop condition. Negative results must include why, not just 'slower'.
8. Before promotion, complete repeated paired whole timing and representative
   cross-task/reset validation. A task-specific path must be labeled as such;
   unsupported dispatch is fallback, not evidence of generalization.

Node-time sums can overlap and are not critical-path attribution. Amdahl bounds
must not treat overlapping stages as independently removable whole time.
For the existing Kuka reference, RTX pre-redo time is 17.757384325 ms and the
current zero-only measurement is 15.3753745 ms. The fixed 2x/4x targets remain
8.8786921625/4.43934608125 ms, not half/quarter of a newly reset baseline.

## Current status

The corrected one-bound and independent-component ideas remain candidates,
not the selected implementation. The existing classifier supports only the
Kuka23+free6 recipe. Cross-task timing and dispatch must determine whether to
generalize that design or prioritize a broader producer/collision change.

Fresh local work log and artifacts:
`/tmp/fpgs-cross-task-20260911-yqkk2O`.

## Fresh baseline survey

The five-task paired survey completed on both GPUs using clean Newton
`a18951df4fb89e3632e1701e65ea63e59b4c7398` for both backends and unchanged,
clean Isaac Lab `1d8feb82d17dbfab8f0772de56f84deae2cb7974`. This is one
discovery round, seed 0, 200 warmup steps, 40 synchronized steps and 40
whole-physics graph steps. One repeat cannot estimate run-to-run spread or
establish a new optimization gain.

All values below are milliseconds per batched environment step. The first
four tasks have 16,384 environments; Keyboard uses its established 4,096 scale
and must not be ranked by absolute time as an equal-sized workload.

| Task | RTX FPGS | GB300 FPGS | RTX MJWarp | GB300 MJWarp |
| --- | ---: | ---: | ---: | ---: |
| Franka | 6.232 | 5.790 | 51.379 | 12.602 |
| Allegro | 19.222 | 21.241 | 64.867 | 81.671 |
| KukaAllegro | 15.420 | 15.395 | 170.364 | 30.853 |
| AnymalD | 9.888 | 10.162 | 38.961 | 43.044 |
| Keyboard SO101 (4K) | 11.567 | 11.356 | 77.654 | 31.441 |

FPGS uses the original per-task Lab recipes plus
`FEATHER_PGS_SIMPLE_WORLD_ZERO=1`; the selector applies only to Kuka here.
The historical optional Franka `PAIR_SHAPE_PREP` and AnymalD
`REGISTER_WHITENING` switches are not enabled in this default-recipe survey.
The separate Allegro rejection-only collision candidate is also not included.
Do not attribute differences versus those older opt-in runs to the zero path.

The MJWarp columns are warning-affected fixed-recipe costs, not accuracy-matched
solver comparisons. Full-run line-search-warning counts are 724,727/724,918
for Franka and 3,674,913/3,675,333 for Kuka (RTX/GB). In particular, Kuka's
apparent greater-than-10x RTX ratio is **not evidence that the algorithmic 10x
goal has been achieved**. Allegro has 4/6 such warnings. The inherited installed
MuJoCo/MJWarp 3.12 versus Newton's declared ~=3.11 mismatch remains; no package
or physics setting was changed to make these measurements look better.

Original FPGS budgets are preserved. All tasks use environment decimation 4
and two Newton substeps. Franka/Kuka use sim_dt 1/120 and maximum 8 GS sweeps;
Allegro uses sim_dt 1/120, fallback 12 and the existing eligible parallel-24
route. AnymalD uses sim_dt 0.005, fallback 8 and eligible parallel-24.
Keyboard uses sim_dt 0.01 and maximum 8 GS sweeps. Generic `pgs_iterations`
metadata alone does not describe the parallel route's budget.

Synchronized whole-environment wall time is separate and includes host/reset
work; it is not RL training throughput:

| Task | RTX FPGS / MJWarp wall ms | GB300 FPGS / MJWarp wall ms |
| --- | ---: | ---: |
| Franka | 24.136 / 72.315 | 25.341 / 32.829 |
| Allegro | 26.641 / 73.275 | 28.583 / 89.847 |
| KukaAllegro | 34.584 / 202.983 | 35.826 / 63.267 |
| AnymalD | 18.174 / 48.344 | 18.849 / 52.679 |
| Keyboard SO101 (4K) | 33.527 / 83.400 | 30.813 / 38.202 |

Keyboard still reaches the 192-row cap in both current boundary snapshots.
This uninstrumented timing survey does not count dropped rows or resolve the
previous 423-row high-water loss. Sparse-diagonal size 108 and compact diagonal
mass size 108 are present in both new captures, confirming the existing owner.
Keyboard MJWarp emits 985,365/985,879 full-run line-search warnings (RTX/GB),
plus one dependency warning each. AnymalD emits 2/1 line-search warnings.
All five tasks have finite before/after states, which does not establish physical
parity. Both drivers completed, their parents were reaped, the original source
trees remained clean, and both GPUs were verified compute-idle afterward.

The current Kuka result reproduces the previous zero-only measurement closely;
it does not demonstrate another speedup. The fixed pre-redo reference and
2x/4x targets above remain unchanged.

### What this changes about selection

- Kuka's zero/one implementation cannot be copied unchanged to the other tasks.
  Franka has mimic/local owners; Allegro has velocity limits and the existing
  parallel-24 route; Keyboard already has diagonal-key response/fused limits.
- Do not begin with a one-bound-only polish or a dense-108 Keyboard rewrite.
  The first is not yet a credible large whole-step saving; the second attacks
  computation already avoided by the current solver.
- Existing profiles identify Allegro collision, row production and solve as
  substantial; AnymalD is solve-heavy. Kuka still retains significant dynamics,
  publication and mixed-component solve work. Their different owners need
  explicit costed hypotheses, not a universal easy-world percentage.
- Franka's available detailed profile is stale, and no valid fixed-source
  Keyboard node breakdown was identified. Refresh only the missing attribution
  needed by a shortlisted candidate. Keyboard's known row drops remain a
  separate physical-quality issue; finite state is not full admission.
- The next implementation decision must satisfy the candidate-card and early
  integrated timing gates above. No new structural path is promoted here.

The existing-profile inventory, including exact revisions and windows, is at
`/tmp/fpgs-cross-task-20260911-yqkk2O/profile_inventory.md`, SHA256
`533c7465b39458f4433072ff4cf008cd817ca78807d373e8dc690adbd7530f62`.
These old node sums are diagnostic ownership evidence, not fresh critical-path
budgets for this survey.

### Reproduction

Use the inherited `tools/fpgs_bench/compare_backends.py` with explicit clean
Newton and Isaac Lab checkouts at the pins above. No new benchmark framework
or Lab modifications are needed. The driver records actual imports, source
hashes, per-task recipes, GPU UUIDs, process cleanup and timing artifacts.
Its 31 CPU tests passed before launch. Driver SHA256:
`40b411196cfacfb012e29429db6f0570f3aec5ed132cd432464a82da197630b2`.

```sh
uv run --no-project --python /path/to/IsaacLab/.venv/bin/python \
  python /path/to/Newton/tools/fpgs_bench/compare_backends.py \
  --isaaclab /path/to/IsaacLab --fpgs /path/to/Newton --mjwarp /path/to/Newton \
  --task franka --task allegro --task kuka --task anymald --gpus 0 1 \
  --fpgs-env 0:FEATHER_PGS_SIMPLE_WORLD_ZERO=1 \
  --fpgs-env 1:FEATHER_PGS_SIMPLE_WORLD_ZERO=1 \
  --repeats 1 --num-envs 16384 --warmup-steps 200 --steps 40 --profile-steps 40 \
  --output-dir /path/outside/checkouts/new-survey
```

Run Keyboard separately with `--task keyboard-so101 --num-envs 4096`, retaining
the other sampling options. For a promotion comparison use at least three
balanced rounds, not this one-round triage protocol.

Local raw results: `/tmp/fpgs-cross-task-survey16k-20260911-01` and
`/tmp/fpgs-cross-task-survey-so101-20260911-01`. Large captures remain local.

Independent final metadata/source audit covers all 20 captures and 3,200
physics graph roots, with matching commands, budgets, source/driver hashes and
finite boundaries. This is not a new independent raw-SQLite re-export or a
physical-convergence audit. Audit note:
`/tmp/fpgs-cross-task-20260911-yqkk2O/capture_audit.md`, SHA256
`f250403a1c961f44dbf2998b04776b12758a39703f246fbf4cf93f5f85cc7154`.

Manifest SHA256 values:

- Four-task 16K: `35baf4c3897e2e75ba11f246fe899592911eba13a1ce8732625cd5f116d60963`.
- Keyboard 4K: `5dd9ddb989d3db34aceefa955bc1798d527a1ef6b6b81d576103b3b642dc7829`.

This checkpoint changes documentation and the workstream instructions only.
The original handoff remains an ancestor; Newton runtime remains the a189
baseline. The benchmark driver's 31 CPU tests and `uvx pre-commit run -a`
pass. No Isaac Lab source or dependency pointer was modified.
