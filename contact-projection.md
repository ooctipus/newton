# Sparse FPGS: shared contact projection

Final candidate `a25b4c45f86ae9ea4d3ff8fb538cb85ece7dcac7`; clean main `5238407d320823e71a5a4623c2c83c293d7b00fd`; immediate prior PR `1472112114f70e288012731a7639d69228e258de`.

## What changed

The sparse contact producer computes supported endpoint Jacobians once and reuses geometry and packed factor loads across the normal and two tangent directions. Eight lanes cooperate on each contact; factor traversal skips unrelated branches. Outputs retain the compact factor-row layout, with no global Jacobian staging, extra conversion launch, task-name dispatch, or change to the PGS budget. Same-articulation endpoints combine before projection; free-body responses retain their existing mass metric.

Isaac Lab, collision, task recipes, timesteps, substeps, iterations and capacities are unchanged. Runtime representation selectors intentionally differ from dense main. Twelve native tasks use sparse response; eight keep their existing paths. Multiple same-topology articulations, free bodies and prescribed endpoints are supported; the one-topology-group/64-DOF admission limits remain.

## Full 20-task RTX comparison

RTX PRO 6000 Blackwell Max-Q; **16,384 environments per task**, seed 0, 200 warmup steps, 40 environment timing steps and a separate 40-step physics window. CUDA graphs on; video off. One final-head capture per task against the recorded clean-main baseline, which was not rerun.

| Task | Env FPS (main → PR) | Env FPS gain | Physics ms (main → PR) | Physics gain |
| --- | ---: | ---: | ---: | ---: |
| Franka lift† | 466,574 → 474,243 | 1.02× | 16.775 → 16.737 | 1.00× |
| KukaAllegro lift | 360,294 → 392,756 | 1.09× | 26.423 → 22.441 | 1.18× |
| Allegro reorient† | 286,291 → 283,602 | 0.99× | 49.482 → 49.419 | 1.00× |
| ANYmal-D flat | 535,626 → 643,163 | 1.20× | 22.467 → 16.982 | 1.32× |
| G1 rough | 279,894 → 364,505 | 1.30× | 44.191 → 30.321 | 1.46× |
| SO101 Keyboard† | 97,341 → 98,013 | 1.01× | 119.171 → 120.234 | 0.99× |
| Ant† | 1,640,350 → 1,664,562 | 1.01× | 4.767 → 4.764 | 1.00× |
| ANYmal-D rough | 163,410 → 178,575 | 1.09× | 91.267 → 82.639 | 1.10× |
| Cartpole† | 2,594,743 → 2,806,341 | 1.08× | 0.471 → 0.472 | 1.00× |
| Cassie flat | 795,500 → 909,985 | 1.14× | 11.209 → 8.372 | 1.34× |
| Cassie rough | 482,384 → 545,880 | 1.13× | 22.537 → 18.738 | 1.20× |
| Franka reach† | 1,142,546 → 1,155,847 | 1.01× | 10.621 → 10.550 | 1.01× |
| G1 flat | 406,129 → 574,802 | 1.42× | 27.321 → 17.005 | 1.61× |
| Go2 flat | 783,237 → 957,394 | 1.22× | 16.879 → 12.501 | 1.35× |
| Go2 rough | 287,158 → 331,480 | 1.15× | 49.671 → 41.451 | 1.20× |
| H1 flat | 658,204 → 806,158 | 1.22× | 13.309 → 9.907 | 1.34× |
| H1 rough | 363,385 → 402,127 | 1.11× | 31.971 → 27.310 | 1.17× |
| Humanoid† | 323,338 → 322,853 | 1.00× | 42.509 → 42.628 | 1.00× |
| UR10 reach† | 1,958,659 → 1,923,289 | 0.98× | 5.119 → 5.125 | 1.00× |
| KukaAllegro reorient | 347,020 → 380,739 | 1.10× | 26.512 → 22.452 | 1.18× |

Env FPS is random-action environment transitions/second, not RL training FPS. Its gain is candidate FPS / main FPS. Physics is the main simulation graph per environment step; its gain is main time / candidate time. Terrain-sensor graphs and eager reset/environment work are outside this physics timing but included in environment FPS.

† Existing solver path retained; timing variation is not credited to the sparse change. Ant, Humanoid and Cartpole keep their native split recipes. Keyboard keeps the accepted 736-row capacity on both revisions; the rejected historical 704-row capture is excluded.

## Incremental contact-projection check

Fresh controls compare the immediate prior PR `14721121` with final `a25b4c45`, not with main. Same 16,384-environment workload and timing windows as above; one fresh capture per arm/task/GPU. RTX candidate measurements are reused from the final20 table.

| Task | GPU | Prior PR → latest physics (ms) | Physics gain | Env FPS gain |
| --- | --- | ---: | ---: | ---: |
| KukaAllegro lift | RTX PRO6000 | 25.044 → 22.441 | 1.12× | 1.05× |
| KukaAllegro lift | GB300 | 24.480 → 21.179 | 1.16× | 1.06× |
| G1 rough | RTX PRO6000 | 30.998 → 30.321 | 1.02× | 1.00× |
| G1 rough | GB300 | 44.990 → 43.690 | 1.03× | 1.01× |
| Cassie flat | RTX PRO6000 | 8.802 → 8.372 | 1.05× | 0.99× |
| Cassie flat | GB300 | 9.427 → 9.176 | 1.03× | 0.99× |

Cassie physics improves while environment time is 1.2% worse on RTX and 0.8% worse on GB300 in these pairs. These small loop differences are not repeat-qualified. Do not convert physics gains into environment gains.

An earlier discovery capture at `271e5e9c` used identical runtime/test bytes (only the final SVG differs). Kuka physics measured 22.454/21.074 ms RTX/GB300 there and 22.441/21.179 ms at the final head. The chronology is initial147 controls → 271 discovery → fresh147 controls → a25 captures, not a randomized or interleaved ABBA experiment. Exact raw values and hashes are in [contact-projection.json](contact-projection.json).

## Attribution and memory

In a separate five-step GB300 node trace, Kuka's contact producer falls **5.686 → 2.591 ms per environment step (2.19×)**. Both traces contain 40 producer calls (eight per step); the candidate has slightly more, not fewer, ordinary rows at the checked boundaries. Registers change 107 → 89, shared memory 1,024 → 6,080 B, and reported local memory remains zero. These are summed kernel-activity times, not a whole-physics speedup or additive critical-path breakdown. The SQLite query, resource values, workload counts and hashes are retained in the JSON.

The existing sparse representation reduces Kuka's large ordinary contact-row allocations **1,392 → 492 MiB** at 16,384 worlds and 192 rows/world. This follow-up does not add global row buffers. This is allocated row storage, not peak/total VRAM or memory traffic.

## Validation and limits

All 20 final-head captures pass their checks: 40 boundaries, finite states, public constraint checks, sticky row flags, factor status, full-log warning rejection and source admission. All 80 final capture hashes were independently verified. All six fresh parent/final comparisons also pass. Collision snapshots and sticky warnings are not a lifetime no-drop certificate; short random-action captures do not establish training equivalence or qualify small timing differences.

Final nine-module test suite: **RTX 77 passed / 1 inherited skip; GB300 76 passed / 1 skip / 1 known failure**. All pre-commit hooks pass. Direct dense-response oracles cover 8/16/32-lane groups, partial groups, bit 63, 1/3 reserved rows, distinct anchors, same/different articulation contacts, free metrics and offsets.

The GB300 failure is `test_two_articulations_share_contact_solve`: independent dense/sparse rollouts allocate limit/contact bundles in different atomic orders. Substituting only the old scalar producer reproduces the first-step mismatch (max velocity difference 0.089762628 versus 0.08976269). Across 34 identical-input old/new producer replays, compact indices match and factor/incident/diagonal differences stay within 2.4e-7. The final GB suite still fails this comparison (max velocity difference 0.02272791); no tolerance or fixture was changed. This is not reported as a clean GB test pass.

## Reproduce

Use the pinned command and prepared-environment requirements in [README.md](README.md). Set sparse selection to false for clean main and true for the candidate. Use `--gpu-index 1` and the matching UUID for GB300. Historical initial and generalized-PR results remain in [results.json](results.json) and [extension.json](extension.json), respectively.

At the candidate checkout, the local regression suite is:

```sh
CUDA_VISIBLE_DEVICES=0 uv run --extra dev python -m unittest \
  newton.tests.test_feather_pgs_sparse_mass_matrix \
  newton.tests.test_feather_pgs_sparse_contacts \
  newton.tests.test_feather_pgs_sparse_pgs \
  newton.tests.test_feather_pgs_sparse_solver \
  newton.tests.test_feather_pgs_contact_controls \
  newton.tests.test_feather_pgs_mimic \
  newton.tests.test_feather_pgs_connect \
  newton.tests.test_feather_pgs_springs \
  newton.tests.test_feather_pgs_preelim
uvx pre-commit run -a
```

Repeat the unittest command with `CUDA_VISIBLE_DEVICES=1` for the second GPU; the known comparison failure above remains. Raw captures stay local; compact numbers, source pins, artifact hashes and reproduction helpers are published separately from the solver PR.
