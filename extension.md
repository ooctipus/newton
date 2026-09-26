# Sparse FPGS extension results

RTX PRO6000, **16,384 environments per task**, CUDA graphs **on**, video **off**.
FPS means random-action environment transitions per second, including the
environment loop—not RL training FPS. Clean candidate `14721121` is compared
with the recorded clean `fpgs-main` baseline `5238407d`; the baseline was not rerun.

| Task | main FPS | PR FPS | Gain |
| --- | ---: | ---: | ---: |
| Franka lift† | 466,574 | 475,458 | 1.02× |
| KukaAllegro lift | 360,294 | 375,398 | 1.04× |
| Allegro reorient† | 286,291 | 286,994 | 1.00× |
| ANYmal-D flat | 535,626 | 632,277 | 1.18× |
| G1 rough | 279,894 | 363,343 | 1.30× |
| SO101 Keyboard† | 97,341 | 96,815 | 0.99× |
| Ant† | 1,640,350 | 1,585,489 | 0.97× |
| ANYmal-D rough | 163,410 | 176,419 | 1.08× |
| Cartpole† | 2,594,743 | 2,801,590 | 1.08× |
| Cassie flat | 795,500 | 886,354 | 1.11× |
| Cassie rough | 482,384 | 531,132 | 1.10× |
| Franka reach† | 1,142,546 | 1,167,116 | 1.02× |
| G1 flat | 406,129 | 566,595 | 1.40× |
| Go2 flat | 783,237 | 986,486 | 1.26× |
| Go2 rough | 287,158 | 332,520 | 1.16× |
| H1 flat | 658,204 | 768,677 | 1.17× |
| H1 rough | 363,385 | 400,437 | 1.10× |
| Humanoid† | 323,338 | 324,484 | 1.00× |
| UR10 reach† | 1,958,659 | 1,936,843 | 0.99× |
| KukaAllegro reorient | 347,020 | 351,743 | 1.01× |

Native task recipes, substeps, iteration budgets, contact laws and capacities
are matched. Each arm has one discovery capture: 200 warmup steps, 40 timed
environment steps and a separate 40-step physics profiling window.
† Existing path retained; its timing variation is **not credited** to sparse dynamics.
Ant, Humanoid and Cartpole retain their native `split` presets.

<details>
<summary>Physics graph time per environment step</summary>

| Task | main physics (ms) | PR physics (ms) | Gain |
| --- | ---: | ---: | ---: |
| Franka lift† | 16.775 | 16.733 | 1.00× |
| KukaAllegro lift | 26.423 | 24.751 | 1.07× |
| Allegro reorient† | 49.482 | 49.324 | 1.00× |
| ANYmal-D flat | 22.467 | 17.240 | 1.30× |
| G1 rough | 44.191 | 31.022 | 1.42× |
| SO101 Keyboard† | 119.171 | 119.668 | 1.00× |
| Ant† | 4.767 | 4.762 | 1.00× |
| ANYmal-D rough | 91.267 | 83.415 | 1.09× |
| Cartpole† | 0.471 | 0.470 | 1.00× |
| Cassie flat | 11.209 | 8.792 | 1.27× |
| Cassie rough | 22.537 | 19.047 | 1.18× |
| Franka reach† | 10.621 | 10.546 | 1.01× |
| G1 flat | 27.321 | 17.408 | 1.57× |
| Go2 flat | 16.879 | 12.558 | 1.34× |
| Go2 rough | 49.671 | 41.889 | 1.19× |
| H1 flat | 13.309 | 10.090 | 1.32× |
| H1 rough | 31.971 | 27.504 | 1.16× |
| Humanoid† | 42.509 | 42.552 | 1.00× |
| UR10 reach† | 5.119 | 5.125 | 1.00× |
| KukaAllegro reorient | 26.512 | 24.769 | 1.07× |

Physics includes the main simulation graph with native decimation/substeps.
It excludes eager environment/reset work and separately captured terrain-sensor
graphs; environment FPS includes those costs. The two windows must not be
subtracted to infer reset cost.

</details>

## Incremental extension check: both GPUs

Two balanced rounds per GPU compare the prior PR `d5c2177d` with its recorded
pre-finalization extension patch (`d1ab0802…`). These are **not** clean-main
comparisons or exact-`14721121` repeats; the source hashes are in
[extension.json](extension.json). Values below are medians of two captures per
arm; per-round values and ranges are retained in JSON.

| Task | GPU | Prior PR → extension physics (ms) | Physics gain | Environment FPS gain |
| --- | --- | ---: | ---: | ---: |
| KukaAllegro lift | RTX PRO6000 | 26.637 → 24.921 | 1.07× | 1.05× |
| KukaAllegro lift | GB300 | 25.895 → 24.350 | 1.06× | 1.04× |
| Cassie flat | RTX PRO6000 | 11.214 → 8.767 | 1.28× | 1.11× |
| Cassie flat | GB300 | 12.672 → 9.425 | 1.34× | 1.13× |
| G1 rough | RTX PRO6000 | 30.888 → 30.952 | 1.00× | 0.99× |
| G1 rough | GB300 | 44.282 → 44.964 | 0.98× | 1.01× |

G1 rough is essentially unchanged on RTX; on GB300 the extension takes **1.5%
more physics time** (1.25–1.84% across the two pairs). This incremental regression
is distinct from G1's gain over clean main.

## Representation and allocated storage

Articulated coordinates use sparse factors; free bodies retain physical
six-coordinate responses. Existing native free/free rows feed the same sweep
without a projected copy. This removes dense factor/response intermediates,
not a global Delassus matrix—the baseline already uses matrix-free PGS here.

KukaAllegro's allocated ordinary contact-row arrays decrease **1,392 → 492 MiB**
at 16,384 worlds and 192 rows/world. The old separate group and world `J/Y`
arrays total 696 + 696 MiB; new row indices/factors, six-coordinate free response
and incident velocity total 204 + 204 + 72 + 12 MiB. This is allocated row storage,
**not** memory traffic, total/peak VRAM, or all solver memory.

Selection uses topology and supported features, not robot names. The extension
supports several same-topology articulations in a world, independent free bodies
and prescribed contact endpoints. Native features are not disabled to gain
admission. The remaining one-topology group and 64-bit ancestry limits are
unfinished representation work, not limits of sparse mechanics; unsupported
scenes retain the existing path.

<details>
<summary>Why Kuka's time gain is smaller than its storage reduction</summary>

A separate clean `d5c2177d` → clean `14721121` GB300 trace (200 warmup,
5 node-profile steps, graphs on) records PGS kernel activity **6.960 → 4.372 ms**,
but row production/response/clearing grows **4.639 → 6.070 ms**. Repeated
joint-contact Jacobian calculations in the new producer consume much of the
solve saving. These are kernel-activity sums, which can overlap—not graph
critical-path times or replacements for the 40-step measurements above.

</details>

## Checks and reproduction

The final code passed **75 tests**, with one inherited obsolete-fixture skip;
all pre-commit hooks passed. All 20 final-head captures passed finite-state,
public-constraint, sticky row-warning and full-log warning checks. Collision
boundary snapshots are not a lifetime no-drop certificate; these short runs do
not establish training equivalence or qualify small timing differences.

[README: command and exact pins](README.md) · [Unrounded data and hashes](extension.json).
The same portable command selects either GPU with `--gpu-index` and its matching
`--gpu-uuid`; `--plan-only` has been checked on both indices without GPU work.
Raw captures remain local. [Original results](results.md) and their Ant/Humanoid
matrix-free supplements describe only the initial `d5c2177d` candidate.
