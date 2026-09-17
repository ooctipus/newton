# Projection design: simplicity, maintenance and actual generality

2026-09-17. Read-only design/source assessment, independently reviewed. No
runtime refactor, new friction model, broader admission or performance claim.
The [public PGS study](PUBLIC_PGS_20260917.md) remains a separate convergence
transfer experiment, not evidence that one projection API can replace all laws.

## Keep three contracts distinct

| Layer | Responsibility | What it does not prove |
| --- | --- | --- |
| Feasibility repair | Restore impulse bounds after extrapolation | Residual convergence or optimal friction |
| Numerical proposal | Choose an update with a step metric/preconditioner | That every feasible endpoint solves the contact law |
| Contact law | Define normal complementarity, friction and regularization | That one particular iteration converges within its allowance |

Yuval's simple momentum repair does not replace MuJoCo's subsequent local
optimization. Likewise our disk clip is not the whole FPGS solver. These layers
can remain separately named even when a native transaction inlines them.

## The simple isotropic case and its metric condition

At fixed nonnegative normal impulse n, isotropic friction gives a disk of
radius mu*n. Radial clipping is its exact Euclidean projection, with explicit
zero-radius/zero-length handling. A **shared scalar tangent step** followed by
this projection has the intended fixed-normal isotropic friction stationarity.
This is the simpler building block in the admitted, accepted G1 spectral path:
a closed-form 2x2 largest eigenvalue and one disk clip, without an inner root.
It is one projected-gradient action, not an exact local quadratic minimizer
or a global finite-eight convergence guarantee.

The older metric path instead solves a fixed-normal local proximal quadratic,
including tangent coupling and denominator-only CFM. It has an unconstrained
2x2 solution or a safeguarded multiplier search, followed by acceptance checks
and scalar fallback. Its extra machinery can earn better finite-budget local
accuracy; simplicity alone does not make it universally inferior.

Metric compatibility is essential. For diagonal positive tangent steps D,
a sliding fixed point of `t = project_disk(t - D*r)` generally satisfies
`r = -gamma * inverse(D) * t`, not the isotropic maximum-dissipation condition
`r = -gamma * t`. Use a common scalar step with Euclidean projection, or the
projection in the matching inverse-step metric. The accepted ANY EX1 unequal
tangent-step limitation was [already documented](</home/octi/Projects/newton-fpgs-krylov-chord-cpu-20260916/reports/fpgs/ACTIVE_WRENCH_20260916.md:775>);
this note does not discover a new bug or authorize changing that solver.

## MuJoCo's implemented generality is broader

MuJoCo C supports contact dimensions 1/3/4/6, anisotropic friction, torsional
and rolling components, pyramidal coordinates, and separate equality/bounded
friction-loss semantics. Its elliptic momentum helper holds a valid normal
fixed and clips in normalized friction coordinates. The actual elliptic PGS
update separately uses a ray/normal step and tangential QCQP; the repair is
not a substitute for that optimizer or its soft-contact objective.
[Pinned C source](https://github.com/google-deepmind/mujoco/blob/c499f7f2b0f3d78b47c48fd487d22c5544ae4f6d/src/engine/engine_solver.c#L369-L719),
[contact dimensions](https://mujoco.readthedocs.io/en/stable/computation/index.html#condim).

For positive friction scaling S, normalized radial clipping projects the
ellipsoid in the S^-2 weighted metric, not generally the original Euclidean
metric. A normalized-gradient design needs matching S^2 scaling, or an
appropriate metric/QCQP proposal. Disabled axes need an explicit zero-coordinate
contract, not division by zero. Full Euclidean second-order-cone projection
can also change the normal: it is not a drop-in fixed-normal repair for our
own-disk law.

A longer norm loop does not add torsion or rolling to today's two-tangent
FPGS path. Those features also require angular Jacobians, coefficient units,
response coupling, force/torque publication and physical validation. Current
G1 admission/fallback remains unchanged; a formula's generality is not proof
of implemented support.

## Maintenance recommendation, not a refactor

Prefer a small named feasibility primitive with explicit geometry and metric;
keep shared-spectral and metric/QCQP proposals distinct. Reuse the impulse-delta
response/fallback/publication contract. Compile-time specialization can keep
the common isotropic case small without introducing a dynamic hot-path framework.

The accepted G1 math is simpler, but current source composition is still
prototype-style: spectral and limit modules perform guarded exact replacements
of generated metric source. Fail-closed counts catch drift, but named typed
preparation/proposal/publication fragments would be easier to maintain. Such
cleanup should follow algorithm selection and preserve measured arithmetic;
none is implemented or funded here.

Future behavioral coverage should distinguish feasibility from stationarity:
zero/interior/boundary disks, sliding/sticking, incoming impulses, tangent-basis
rotation, unequal step metrics and response consistency. Anisotropic/disabled
axes or higher contact dimensions need tests only with real implemented rows,
not fictitious claims that current production already supports them.

## Preserved evidence

- Root [math/design note](</tmp/fpgs-public-pgs-source-study-bxo2dQ73/PROJECTION_DESIGN.md>), SHA `af3250da769870961221087cd44211dcfd7705abef711ddee5b26e39a749a199`.
- Independent [source comparison](</tmp/fpgs-public-pgs-source-study-bxo2dQ73/PROJECTION_COMPARISON.md>), SHA `90e3ee33f882762a38f517d91e91d01e39ef1c7b470b0bf62713abbb9ab34b45`.

The comparison inspected `sparse_metric_tangents.get_fragments`,
`sparse_spectral_tangents.get_fragments`, the accepted limit composition and
pinned MuJoCo repair/QCQP routines. The independent review found no concrete
math/source issue in the design note. This archive changes neither accepted
runtime nor the separate G1/ANY momentum-transfer decisions.
