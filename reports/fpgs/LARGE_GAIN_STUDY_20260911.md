# FPGS large-gain research checkpoint, 2026-09-11

This documentation-only branch records an unsuccessful large-gain search,
not a newly enabled solver optimization. It derives from the accepted Newton
checkpoint `a7eb7d15589a3c6032288a488463eb8e028d88e6` on the `ooctipus/newton`
fork. The original handoff `31cf87f4694f873a027e41e2ca5e9ad441234456` remains
an ancestor. No Isaac Lab dependency pin is advanced.

The maintained [study and reproduction entry point](https://github.com/ooctipus/IsaacLab/blob/ooctipus/fpgs-large-gain-20260911/reports/fpgs/LARGE_GAIN_STUDY_20260911.md)
contains the comparison contract, paired RTX PRO 6000 / GB300 results, rejected
architectures, compiler and lifecycle findings, retained failures and remaining
proof obligations. The requested work window is 10:37:35–20:37:35 UTC.

The additional 2–4× whole-physics target has not been demonstrated. The last
complete strict outward-FP32 screen passes its isolated numerical gates
but costs 1.094/1.113 ms per call on RTX and 1.534/1.560 ms on GB. That is
already comparable to the original entire constraint stage, before paying
fallback and integration costs. Its proposal/body producers dominate enough
work that a contact-only rewrite cannot rescue the design. No tuning or live
speedup claim is promoted from this result.

A later, genuinely different primitive uses a direct FP32 one-limit candidate
and parallel directed residual matvecs instead of interval triangular proof
solves and unique rounding-cell demands. Its complete paired 16K component
gate passes at 34.048/34.352 microseconds RTX and 35.168/35.312 GB for the
refresh/reuse captures. This includes the candidate and residual stores only;
inverse-error bounds, body/contact screening, allocation, fallback and live
integration are excluded. It is not an accepted whole-physics improvement.
Native source SHA256:
`750c0c2862d3198b2204013a403b73bf4f54a22ca864a9d47ddb50396a4882b9`.

The constructive CPU affine screen covers 58,051/65,536 sampled worlds, but
only 74.66% of the same-eight row-visit proxy; all mixed-MF snapshots still
fall back. The next major design must remove unnecessary body/J publications
and reduce the expensive fallback, not tune the closed interval mapping.
Per-operation formal certificates are a stronger optional contract, not a
requirement imposed by numerical convergence. A cheaper approximate solver
path would need separate physical and held-out trajectory qualification.

The final consumer audit removes another self-imposed constraint: canonical
dense/MF row arrays are internal storage, not mandatory public sensor output.
A new simple-world owner can publish solved velocity and zero contact forces
directly, leaving row allocation/operator work to fallback worlds. Raw contact
geometry, last-solve force identity, stale-buffer transitions, diagnostics and
the original cold iteration allowance must remain correct. The old 52 µs
allocator cost is therefore not an immutable lower bound, but no implementation
or measured saving from this new boundary is claimed here. Dense-only worlds
already exit before expensive MF work, so queue compaction alone cannot be
credited with the measured mixed-world solver cost.

The default-off experimental source trees and raw results remain local, with
verified copies under
`/home/octi/Projects/fpgs-large-gain-evidence-20260911-ri5D08/closed-core`.
The manifest SHA256 is
`91b244a974bf7803b9f9c83847ba53ae9601587d73cfdcaaa837c183f5e0bfa5`.
The native strict carrier source SHA256 is
`8c38b3e1151dd38315e00f322e916ad81d3d390a3eab606f76e963ba8cdcaf44`;
the original interval-certified proposal source SHA256 is
`ad8b4b7ceafbdb06073ea80b7351dc970e961ff7accc35605bb064589fbbe854`.
These identify archived research, not source installed by this branch.

The newer direct candidate and completed paired results are in sibling archive
section `new-proposal`, manifest SHA256
`5284f3f39ace35bde470beede91732ffba327d34e3a01ea6d31693c649b852fd`.
The final representation/MF/public-force audits and separate-seed check are in
`final-scope`, manifest SHA256
`c1b73646ef7e5766bc9b5fbee4e66cc1da08e6e222c3cf9add5b6e3d1c4d7236`.
All archived payloads pass their complete checksum rechecks. Their absolute
source paths and existing runtime remain reproduction prerequisites.

The original finite-GS iterate is not a correctness oracle. Future replacement
work may change matrix representation and internal algorithms while retaining
the timestep, substeps and iteration allowance. It must demonstrate physical
feasibility, complementarity, dissipation, momentum error and stability, then
complete reset-inclusive whole-physics cost on both devices. The inherited
FPGS/P25 gates still block any parent dependency update.
