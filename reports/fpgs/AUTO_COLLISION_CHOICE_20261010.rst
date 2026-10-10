Automatic collision choice (experimental)
=========================================

``NEWTON_NARROW_PHASE_COHERENT_CONVEX=auto`` selects ordinary split
queries or certified rejection from current geometry and independently
measured GPU primitive costs. No task name, robot name or task timing enters
the decision. This experimental mode may change without compatibility guarantees.

Eligible CUDA pipelines sample up to 512 queries every 64 collision calls.
The sample counts support calls and visited vertices, estimates separable
work, and subtracts screening, lookup, launch and amortized sampling costs.
A sampling uncertainty bound and 15% query-stage margin favor ordinary
queries when the estimated benefit is small. Device-side queue counts
select the method during replay without host synchronization. Empty kernels
are included in the cost estimate and measured overhead.
Sampling never produces contacts. The existing current-geometry certificate
and cold query/manifold implementations continue to own contact generation.

Unsupported pipelines or models without eligible box/convex pairs take the
ordinary route without allocating a selector or running its kernels.
Existing explicit modes ``0`` and ``reject_only`` remain available.
Auto mode currently excludes convex-cell acceleration. The same device-side
selection works in eager execution and captured simulation. It preserves
existing cross-stream capture events.

Calibration runs once per process/device before capture. It uses synthetic
convex shapes, support-query timings, bound scans and key lookups. Startup
calibration and compilation are separate from steady simulation timings.
The estimate covers query work only, not full physics. It approximates
steady separability; rapid motion and transitions can invalidate a cost
forecast but cannot bypass current-geometry contact validation. Switching
can lag a change by up to 63 collision calls.

Validation artifacts and task measurements are recorded in the workspace's
``output/fpgs-auto-collision-choice-20261010``. This branch is an experimental
implementation, not a promoted replacement for the qualified benchmark routes.
