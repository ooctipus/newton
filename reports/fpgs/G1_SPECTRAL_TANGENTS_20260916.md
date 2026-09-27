# G1 spectral tangents: reopened native cost screen

Default-off experiment on qualified-chain `ef9481bcc7b17fc9496f28a206ab6d0c72331507`.
Enable `FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS=1` alongside the existing metric
owner. This reopens the **already studied** frozen `spectral_gs_control.py`
policy (`a39eb83e4f5b240a24416f96683df0cc5e7453c8f180cbf539f9d752141256c7`),
not a new algorithm claim. No task timestep, substep, maximum eight sweeps,
geometry, collision, current/held operator or row ordering changes.

The prior CPU G1 screen passed eleven of sixteen pointwise comparisons and
had no hard cone/sign/momentum/nonfinite failures. Those comparisons alone
do not establish unacceptable physics under the current user contract.
ANYmal's closed ordered-control cost was against a parallel baseline; G1's
accepted owner is already ordered. Native cost was therefore a distinct
unmeasured question. The completed screen below is positive but misses the
1 ms RTX standalone milestone. The option remains default-off and unpromoted;
bounded native correctness evidence is not full convergence qualification.

## Exact change and charged boundary

Keep the original fused scalar-normal transaction, physical normal/tangent
cross corrections, coefficient validation, scalar fallback, delayed-friction
semantics, incoming-lambda/zero-delta lifecycle, combined response update and
complete43 decode. Replace only the lane-zero metric-disk inverse, eigenmin,
root loop and local KKT acceptance with one projected tangent gradient step.

For physical tangent Gram G, use `D=lambda_max(G)+max(CFM1,CFM2)` and
`t_new=project_disk(t_old-r_t/D, mu*normal_new)`. The tangent residual includes
the just-proposed normal response. CFM appears only in D, never in the
physical residual. The existing global row-CFM array is added to the native
solve ABI immediately after diagonal; physical diagonal is diagonal-CFM.
The original tangent-cross shared slot caches D after its first use, so no
new shared or global allocation is introduced. Invalid finite/positive
denominator data rejects before publication to the original scalar path.

At a fixed normal radius this has the own-parent Coulomb fixed point.
For an old tangent feasible in that radius, the tangent quadratic decreases
by at least `(D-lambda_max(G)/2)*||delta_t||^2`. If the normal shrinks the disk,
the old tangent can be infeasible and that particular descent comparison
does not apply. No global Coulomb convergence claim follows.

The full row residual reductions, cross-term setup, combined response actions,
row visits, original exact-unchanged exit and decode remain. No physical-stop
scan, additional solver sweep, mapping change or contact drop is introduced.
The at-least1ms RTX whole-physics milestone requires removing more than25%
of the approximately3.96ms accepted GS owner. Its scalar-proposal fraction
is not measured; native whole timing is the falsification, not register or
root-count arithmetic.

## Initial evidence

Missing-module regression12675 failed before runtime creation. Three CPU
tests pass: source/admission and unchanged-default ownership, plus reused
frozen anisotropic sliding, duplicate-normal and incoming-delta controls.
Native selectors reuse existing same-row/current-held/graph/empty fixtures
and sixteen pinned G1 current/held payloads with independent J/H momentum
and physical diagnostics. No new high-iteration oracle or cohort is created.

CUDA-hidden AOT73982 exited0 on SM120/103:64 registers,1116 bytes shared,
zero stack/spills/barriers. Accepted metric owner uses72 registers with the
same shared allocation. This is not timing evidence. Artifacts:
`/tmp/fpgs-g1-spectral-offline-9kUPSrcy/offline01/report.json`.
Final source-matched repeat39827 has identical resources at
`/tmp/fpgs-g1-spectral-offline-9kUPSrcy/offline02/report.json`.
The full pre-commit rerun and targeted new-file checks pass; CPU74717 passes
all four candidate/original controls, with three native selectors skipped.
Independent source review finds no concrete ordering/ABI/cache blocker.
If physical diagonal is lost to diagonal-minus-CFM cancellation, the cached
denominator rejects safely to the retained scalar fallback; extreme
cancellation is not claimed as a newly qualified input regime.

## Native scope and physical diagnostics

Root-owned sessions41861/64697 both exited0: all three native selectors pass
on RTX and GB300. Logs are
`/tmp/fpgs-g1-spectral-native-20260916-jZmTfTOe/gpu{0,1}.log`.
They cover native transaction controls, current/held operator and graph/empty
lifecycle controls, and the existing sixteen saved G1 cases. The candidate
reproduces the frozen CPU spectral policy within the existing velocity/impulse
tolerances, with independent current-J/held-H momentum and cone checks.
No new high-iteration reference, trajectory or full-cohort physical study
was run.

The legacy log label `metric_tangent_saved_native` is retained by fixture
reuse: its `metric_current` field now denotes this spectral candidate, and
`scalar_current` is the old scalar ordered reference, not the accepted
metric-eight-sweep baseline. CPU `oracle_transactions` are reference work
counts, not measured GPU counters or extra native stopping scans.

Across the sixteen cases on each execution device, maxima are:

| Diagnostic | Maximum |
| --- | ---: |
| Natural residual | 2.2976938e-4 |
| Normal violation | 1.6404271e-3 |
| Complementarity | 7.0045977e-4 |
| Maximum-dissipation defect | 8.0912203e-4 |
| Cone violation | 7.4505806e-9 |
| Momentum defect | 4.2573383e-8 |

All finite/cone/momentum hard checks and CPU-policy reproduction pass.
The residual maxima do **not** establish convergence to 3e-5 in all cases,
nor a new nonregression result against a converged physical reference.

## Whole-physics result and closure

Root-owned paired single-round 16,384-world capture completed with all
source, actual-owner, capacity and idle guards passing:
`/tmp/fpgs-g1-spectral-tangents-whole-paired16k-20260916-01`.
Both arms retain the qualified chain recipe and unchanged iteration budgets;
only the explicit spectral flag differs. No CSR, shell or new manifold is
composed into this result.

| Device | Original metric (ms) | Spectral (ms) | Saving (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| RTX | 15.031834625 | 14.480169350 | 0.551665275 | 1.038098x |
| GB300 | 19.837211200 | 18.904325600 | 0.932885600 | 1.049348x |

These are complete physics times, not an isolated root-loop timing. The
positive RTX saving is below the required 1 ms standalone milestone. There
is no promotion, three-round timing claim, measured root-work attribution
or full convergence claim. Preserve this default-off candidate for possible
later composition; no mapping, local-step or cohort retry is funded here.

The existing checked observer and parent were minimally extended to require
the explicit flag, actual factory identity, added-CFM ABI and exact kernel
key. CPU75155 exercised their literal checks using an actual fixture owner:
both flag modes pass; wrong factory, missing flag and wrong parent metadata
reject. This is not a claim of separate full Lab construction coverage.

Measured source and artifact pins:

- `sparse_spectral_tangents.py`: `f9c0f5dd0706e9c3d2f201057f3b0bd4fedc8a9c0af4921da1df404b0cc28309`.
- `sparse_factor.py`: `086a6064a0872ad0a2a46e8bc7fb6613e667536a539b65d725f332c26cdf4227`.
- `test_sparse_spectral_tangents.py`: `4c935b770ffe14c71aee422bb3837d6035b9d92a7eed1c651a40a23067c035f2`.
- `checked_sparse.py`: `38fb283b1f99a613cd14475fe274f6e53e0928379bf92e42961bef3aa9e3ce42`.
- `run.py`: `26191de94dc39793cd8aa64478a69836717107649a220efcd7bceb9ddd7ca73c`.
- Source-matched AOT report: `57b6cf99415e07c865e5337f9d64d2215140ec9b13a2c7ec2415fe32891879c6`.
- Native RTX log: `05ebe5a2b296dbb509762a385c222cf29122eac18c73344f393ee4ab86a72a80`.
- Native GB300 log: `28e07df1317d72eaed0dea9f7103a38b38e83a7c7400ff289e95f2d6b0506013`.
- Whole manifest: `81666da740eb459ab2d9bcc9c816fa46cd00baf128c22bd26ca452a93256e398`.
