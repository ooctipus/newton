# Elliptic line-search numerical evidence

This supplements the [cross-task study](CROSS_TASK_20260911.md). It separates
two diagnosed numerical losses from physical-quality and performance claims.
The experimental compatibility path retains the original elliptic cone evaluator;
the more expensive direct-residual evaluator remains a diagnostic candidate.
No task timestep, substeps, iteration allowance or gradient tolerance was changed.

## Scope and outcome

- Allegro's saved failures lose the sign of a small friction-cost decrease.
  Computing the same Huber difference from displacement fixes that cancellation
  without changing its derivatives, Hessians or nonfriction rows.
- ANYmal-D also has cancellation in the prepared elliptic tangent-norm polynomial.
  Corrected bracket acquisition and representable-step stopping remove observed
  line-search warnings, but do not make the inherited cone evaluator exact.
- The observed objective/state errors do not justify requiring exact FP64 roots
  or permanently paying for a direct per-contact evaluation. Conversely, absence
  of warnings is not a complete physical-quality verdict.
- Corrected ANYmal rollout bulk gradients are comparable, but their largest
  tails are not uniformly better. The quantitative caveat below remains open;
  this report does not claim globally exact convergence or uniform improvement.

These are paired RTX PRO 6000 / GB300 diagnostics, 16,384 environments, seed 0,
200 warmup steps, three repeats of 40 synchronized steps and 40 graph-profile steps. The unchanged
Allegro and ANYmal-D recipes both use a **50-iteration line-search allowance**.
Each final diagnostic covers 47,185,920 world-solves per device, including
warmup and graph execution. These are not balanced whole-performance benchmarks.

## Allegro: a false positive shifted cost

Each stock failed ray has 16 friction-loss rows in linear Huber branches.
The initial total objective is about 824 / 562, while the actual decrease is a
few millionths. At the initial Newton point the directional gradient is already
below the unchanged `gtol=1e-6`; erroneous positive shifted cost prevents the
cost-and-gradient acceptance condition.

| Initial-point quantity | RTX | GB300 |
| --- | ---: | ---: |
| Recorded GPU shifted cost | +3.41869e-6 | +1.15399e-5 |
| Independent raw-row FP64 shifted cost | -6.44798e-6 | -2.86804e-6 |
| Original CPU rows, FP64 sum | +7.96674e-7 | +5.17242e-6 |
| Same CPU rows, stable friction difference only | -6.45252e-6 | -2.86076e-6 |
| Original CPU friction error | +7.25e-6 | +8.03e-6 |
| Original CPU elliptic error | -4.56e-9 | +7.28e-9 |

The CPU point and three-point controls preserve every original gradient and
Hessian and all nonfriction outputs. CPU reduction/FMA ordering is not claimed
identical to the recorded GPU evaluation. The causal actual-GPU control changes
only the friction difference with the **stock bracket**, removing the 9 / 4
stock warnings. Thus this fix is not justified by suppressing positive-cost
failures or by broadening numerical tolerance.

## ANYmal-D: conditioning and the representable alpha grid

The two saved stock rays have no friction-loss rows. Re-evaluation distinguishes
raw FP32 input rows from stored FP32 prepared coefficients, the latter evaluated
with Decimal precision 80. This is high-precision numerical evaluation, not
literally exact square-root arithmetic.

| Alpha | RTX | GB300 |
| --- | ---: | ---: |
| Raw-row FP64 minimizer | .917176982436 | .995524300148 |
| Stored-coefficient high-precision minimizer | .917192108915 | .995537128974 |
| Recorded low endpoint | .917186796665 | .995540142059 |
| Recorded high endpoint | .917191028595 | .995557129383 |
| Endpoint width in FP32 steps | 71 | 285 |

Neither recorded sign bracket contains the raw minimizer; the stored-coefficient
minimizer is also outside it, on the other side. There are two errors: coefficient
construction and cancellation during evaluation. The expression
`T² = uu + 2*alpha*uv + alpha²*vv` combines terms of order 4e5–5e5:

| Tangent squared norm at recorded low | RTX contact 77016 | GB300 contact 35904 |
| --- | ---: | ---: |
| Raw-row FP64 | 2.7261131 | .2360850 |
| Stored coefficients, precision 80 | 2.7520548 | .2288827 |

Both stored Gram matrices remain positive definite in these examples. This is
not evidence of a negative Gram determinant. Small coefficient errors are
amplified near the tangent residual minimum.

At the FP32 neighbors of the raw minimizer, directional derivatives are
`(-7.01, +13.33)` / `(-1.89, +6.84)`, versus `gtol=.00417 / .00525`.
The absolute tolerance is unattainable on that local alpha grid. Representable
stopping is justified, but the sign of an inaccurately evaluated derivative is
not an independent raw-objective certificate. The outer solver also uses
objective/model improvement stopping; gradient tolerance is not its sole rule.

### Physical scale, not exact-root acceptance

For the recorded stock first rays, keep the original search and all other inputs
fixed and compare published alpha with the raw-ray optimum:

| One frozen-ray consequence | RTX | GB300 |
| --- | ---: | ---: |
| Relative acceleration-step difference | 1.531e-5 | 1.591e-5 |
| Largest joint acceleration difference, rad/s² | .03541 | .04279 |
| Largest base linear acceleration difference, m/s² | 1.94e-5 | 2.55e-5 |
| Objective regret | .03350 | .01863 |
| Regret / achieved decrease | 1.99e-9 | 9.28e-10 |
| One-substep max joint velocity change, rad/s | 8.85e-5 | 1.07e-4 |

The last row uses actual ANYmal `sim_dt=.005`, two substeps, solver `dt=.0025`.
It is a one-substep perturbation, **not a trajectory-error bound**. Large gradient
errors alone do not demonstrate material physical corruption when curvature is
roughly 1.5e8–3.4e8 and objective regret is this small.

A separate complete CPU LS50 replay compared the corrected coefficient evaluator
with corrected direct-FMA residual evaluation. Raw objective regret fell from
.02578 / .43566 to 7.08e-8 / 1.49e-8. Both coefficient regrets are below one FP32
ULP of the full objective in these examples. The maximum one-substep velocity
differences were 7.78e-5 / 5.36e-4. These CPU replays recompute preparation and
reduction and are not the original GPU trajectory. Direct evaluation reads and
contracts all 3/4/6 contact rows per alpha instead of using prepared O(1)
coefficients. Its demonstrated numerical improvement has no established physical
benefit sufficient to require this permanent overhead.

## Full-run warning and gradient evidence

| Task / variant | LS warnings RTX / GB300 | Other solver/storage/nonfinite failures |
| --- | ---: | ---: |
| Allegro stock | 9 / 4 | 0 / 0 |
| Allegro corrected bracket only | 6 / 8 | 0 / 0 |
| Allegro stable friction, stock bracket | 0 / 0 | 0 / 0 |
| Allegro stable friction + corrected bracket | 0 / 0 | 0 / 0 |
| ANYmal stock | 1 / 2 | 0 / 0 |
| ANYmal corrected bracket | 0 / 0 | 0 / 0 |

All referenced final reports pass their source guards. ANYmal corrected records
90,522,394 / 90,503,706 representable-convergence events across outer iterations;
this is common stopping behavior, not a rare fallback. ANYmal has 22 active rows
at peak. Allegro stable-friction/stock-bracket GB reaches 80 rows at its 80-row
capacity: no overflow was observed, but this is **zero reserve**, not held-out
right-sizing acceptance. Capacity calibration remains a separate requirement.

The independent force-balance observer computes
`g = M*qacc - qfrc_smooth - Jᵀ*stored_efc_force`, using the official fresh FP32
mass/transpose products and FP64 residual recombination/norm. It does not
independently prove that stored constraint force satisfies the constitutive law.
The following scale is `||g||₂ / (meaninertia * nv)`. A "peak" is each world's
maximum across the run; its p99 is **not** an all-call p99.

| Task / device | Final p99, stock → corrected | Per-world peak p99 | Largest per-world peak |
| --- | ---: | ---: | ---: |
| Allegro RTX, including stable friction | .00324309 → .00234372 | .03180136 → .01044481 | .14431738 → .08382274 |
| Allegro GB300, including stable friction | .00309473 → .00240225 | .03166482 → .01042109 | .14392882 → .04119763 |
| ANYmal RTX | .000531950 → .000555905 | .00747697 → .00746711 | .04409313 → .03430669 |
| ANYmal GB300 | .000532607 → .000537296 | .00746643 → .00760473 | .02667264 → .08257709 |

Allegro's measured bulk and peak distributions improve. ANYmal's bulk remains
close, but GB's largest per-world peak grows **3.10x**. Final maximum gradients
grow .00222487 → .00388413 on RTX and .00262653 → .00600804 on GB. These are
different evolved rollouts, not matched same-input per-world comparisons; the
aggregates neither prove harmful physics nor establish uniformly nonworse tails.
Contextual tail and held-out task-state checks remain necessary before an
unqualified quality claim. Exact FP64 root matching is not the missing gate.

## Durable tests and claim boundary

The repository-local [native fixture and independent oracle](../../tools/fpgs_bench/mjwarp_elliptic_fixture.py)
and [elliptic tests](../../tools/fpgs_bench/test_mjwarp_elliptic.py) contain no
scratch imports. They cover actual native contacts of dimensions 3/4/6, including
torsional and rolling rows, all three constitutive zones via native
`mj_constraintUpdate`, dense/CSR fused-Jv paths, the public installed compatibility
driver at LS50, and an actual LS1 exhaustion that must retain warning bit 1024.
The original helper's ELLIPTIC scope rejection was recorded before extension.

```sh
cd tools/fpgs_bench
CUDA_VISIBLE_DEVICES='' uv run --no-project --python /path/to/IsaacLab/.venv/bin/python \
  python -m unittest -v test_mjwarp_elliptic test_mjwarp_linesearch_compat
```

The independent scratch studies passed six native-fixture, four saved-ray and
six direct-full-line-search CPU tests. They retain genuine budget failure and an
explicit stock-bracket/direct-point negative control. The supported claim is
experimental numerical robustness for diagnosed line-search failures with
unchanged budgets and visible warnings—not exact cone evaluation, global
convergence, universally warning-free simulation, or a new performance gain.

## Local evidence index

Large raw captures remain local. This report preserves their exact identities
and the small numerical findings rather than vendoring temporary environments.
For the six analyzed files, append `/gpuN/audit.first_ray.npz` to these roots:

| Root under `/tmp/` | GPU | SHA-256 |
| --- | ---: | --- |
| `mj-elliptic-stock-allegro16384-20260912-01` | 0 | `1e6c844b8440762dcf0dc66a0a6f6ac01bfb673851910d9529faa2eef697eb3c` |
| same | 1 | `34ad58068c7d81239913c7cc94741aaa472db6af604ce3eb39292746c48326a6` |
| `mj-elliptic-corrected-allegro16384-20260912-01` | 0 | `e134ff9c248da85201f6bace14888faece93b44e8b832ae6d1d3165e9253fc4a` |
| same | 1 | `98ace17cdf2861dbe6a275dc7121e3e1586f6f7bdb63ac361cf1bfffbca2d3f5` |
| `mj-elliptic-stock-anymald16384-20260912-01` | 0 | `a28c5aa20b895acc7ced71592700724f7b02615ab234403c9cf0b8175ff0c7bc` |
| same | 1 | `78f8e280b539d08666a0a208f6e9269ca53d0ef86be7a58bb63567acd6325115` |

Full-run aggregates are `gpuN/audit.json` in those roots, plus
`/tmp/mj-elliptic-{delta-stock-allegro,delta-corrected-allegro,corrected-anymald}16384-20260912-01`.
They record actual source pins, capacity, call counts and observer arithmetic.

Frozen analysis root: `/tmp/fpgs-elliptic-ray-analysis-3l5sK8`.

| File / source | SHA-256 |
| --- | --- |
| Original `FINDINGS.md` | `979a786db2ac2fd83281eee11d78e1f99b110d8d2b3d7a424924db3241398dca` |
| `saved_rays_01.json` | `5ac1b7bb9b9f617f1971cb4e2ea655fcf7bb39183dc7c6eb5888c2dea50adfdf` |
| Raw oracle `ray.py` | `d5b312670753cb7be1286a12d5c2472cc4b33a44208aac0c916dcf6a896d270d` |
| Prepared oracle `prepared.py` | `e4f0add0ff4165042233b46b5dbe85189d0f40e09aa4d2001152f627ca272b86` |
| Actual CPU point `native_point.py` | `9ce040156b217a1ab99a382385ab0a9dc5ce075b5bd571a446aec9c878972a40` |
| Diagnostic-only `direct_cone.py` | `72e1bc96757726a5fce8e941b84b7a802c5f492338007f3ecbfdbd73171c8874` |
| Reviewed installed MJWarp 3.12 `solver.py` | `509b43da297dc6e49efeacdbb59df675213e84afbdb26770471a5ba1855a24b3` |
| Official mass/J-transpose `support.py` | `1a08ef15f149d1cf3c9ec5df6c688669fe86aa237ee2512a7463caf1b74e3d54` |

Diagnostic-only full-LS adapter:
`/tmp/fpgs-elliptic-direct-linesearch-axJGKG/READY.md`, SHA-256
`8dfa6c8ef917c4acd4ba7947cd2d2d51f98692758690b48afab892da64b9fa07`.
It pins the actual private driver, direct evaluator, replay and six tests.
