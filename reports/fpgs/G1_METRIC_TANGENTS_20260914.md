# G1 fused normal and metric-disk tangent experiment

Pre-code card, 2026-09-14 05:40 UTC; checkpoint07:10, no silent extension past
two hours. New branch from f2baf6de, all earlier candidates preserved. Default-off
FEATHER_PGS_SPARSE_METRIC_TANGENTS=1, owner.metric_tangents, native key
sparse_metric_tangent43_s18_c100. Only capacity100; incompatible packet/contact-
block modes reject before allocation. Original scalar fallback remains complete.

New numerical cause: the old scalar tangent/sibling projection can plateau at
nonzero physical maximum-dissipation residual. Stopping-only is closed. Replace
normal plus two tangent publication cycles by one fused triplet, retaining the
original scalar normal update, current normal radius, eight maximum sweeps and
exact-stationary exit. No new stopping check, global solve or3x3 sticking solve.

At old kinetic state compute normal residual and scalar delta_n FIRST. If its
new radius is zero, clear old tangents without tangent residual/root/cache work.
For positive radius, form current tangent residuals and add G_tn*delta_n, then
solve min(.5*x^T K*x+b^T*x), ||x||<=mu*new_lambda_n, where K uses existing tangent
denominators and physical cross term, b=r_t-K*old_lambda_t. CFM is proximal-delta
damping only; no physical CFM*lambda appears at the fixed point. Publish one
combineddu = Z_n*delta_n + Z_t1*delta_t1 + Z_t2*delta_t2 after full acceptance.

Reuse three lazy cross coefficients/contact plus four ready words:416 bytes
additional shared at capacity100, no new global arrays or producer launch.
Finite/SPD/omega==1/complete identical-support triplet guards fall back BEFORE
any impulse/du mutation. Solve the2D disk by one bounded reciprocal-norm Newton
root (16 maximum probes), fixed FP32 root/KKT guards and only tiny inward repair
of a converged boundary root. Never radial-clip an unconstrained anisotropic
solution. Unsupported/sliding-root failures retain the original scalar triplet.

CPU selected16 prior evidence: /tmp/fpgs-g1-metric-tangent-cpu-FuzScC8G, exact
four original current/held payloads and existing physical scorer. RTX7410 natural
scalar8->metric8 is7.094e-4->1.722e-4 current and.004134->9.563e-5 held; GB4625
.001210/.004108->6.05e-10/2.94e-9. No root/SPD fallback, max4 root evaluations,
cone2.78e-17, momentum3.89e-16. Some tiny normal/complementarity components worsen;
velocity changes reach.18354, so actual physical behavior needs validation.

Own-stationary-truncated FP64 work:3227->1927 residual dots and550->191 nonzero
publication transactions, removing123 sibling updates. Add17 three-cross caches,
114 unconstrained2D checks and194 ADDITIONAL sliding-root evaluations. These are
selected-case operation counts, not population rates/nativeFP32 execution/time.
Normal-first zero-radius behavior is required to retain the dot removal. Both
tangent delta multiplies, all limit/normal rows, setup/decode and global services
remain. Existing mass/rows/predictor/publication producers are unchanged.

Target: approximately2ms RTX whole saving; scalar GS baseline4.84ms requires
complete solve near2.84ms including all setup/root/fallback. The prior local3x3
4.24ms owner is not the baseline. First physical original-owner tests then paired
whole timing at fixed Lab53ee, timestep/substeps/max8/capacities. Charge all
checks/fallback and use original baselinelevel with contactblock0. A first loss
gets one bounded source/node causal diagnosis, not a layout/root-tolerance grid.

Regression-first constructor and local metric-oracle tests precede native code.
Physical gates cover current/held operator, actual rounded-Z law, independent
momentum/cone/residuals, open/slip/zero-radius/fallback and graph/empty transitions.
Do not reinstate failed elementwise warm coefficient gates or demand original
eight-vector identity. Same-input scalar numerical diagnostics remain visible.
No new benchmark framework or changes to Isaac Lab, original trees or allowances.

## CPU/native readiness checkpoint

Constructor regression first failed on the old runtime with missing
SparseFactor.metric_tangents; it passes after admission/kernel hooks. Native
fragments are isolated in sparse_metric_tangents.py; original scalar source and
fallback remain unchanged, with no new global allocation or producer pass.

CPU-only offline compilation of original and metric kernels passes SM120 and
SM100, with CUDA devices hidden. Original uses40/37 registers and700B shared;
metric72/72 and1116B shared; all stack/spill stores/spill loads are zero. These
are compiler resource facts, not measured occupancy or timing. Existing compile
helper d76cbfa8e4462d29d4843c1b3e5a006a81b6451fad673e8299f2f9413ed1541d was
reused in memory, executed-source SHA
98694d49d98f6bfc193007c327fe3209e519fda38631085372c9cc51f61c14d1.
Source-pinned report: /tmp/fpgs-g1-metric-tangent-native-BPVNFDmF/report.json.
No GPU or physical/whole timing has run at this checkpoint.

The eight existing CPU factor/level/parallel-prefix/contact-block controls pass
with metric mode disabled. Independent native source review found no blocker;
original recurrence, held/current ownership, metadata and publication are intact.
New physical selectors (original paired owner; root owns the GPU lease):

- tools.fpgs_bench.test_sparse_metric_tangents:TestSparseMetricTangentsCPU.test_owner_admission
- tools.fpgs_bench.test_sparse_metric_tangents:TestSparseMetricTangentsCUDA.test_native_stick_open_slip_and_guarded_fallback
- tools.fpgs_bench.test_sparse_metric_tangents:TestSparseMetricTangentsCUDA.test_native_saved_sixteen_current_held_epochs
- tools.fpgs_bench.test_sparse_metric_tangents:TestSparseMetricTangentsCUDA.test_native_current_held_graph_and_empty

The existing original155eba parent/027da child adapter must inject and record
SPARSE_FACTOR=1, SPARSE_PACKETS=0, SPARSE_PARALLEL_LIMITS=1,
SPARSE_LEVEL_UPDATE=1, SPARSE_CONTACT_BLOCK=0, SPARSE_METRIC_TANGENTS=1 (all with
FEATHER_PGS_ prefix), retain --fpgs and fixed Lab53ee, and pin the clean source,
existing replay helper, four payloads/capture audits and G1 asset. Outer flags
alone are insufficient because the parent cleans FEATHER_ child environments.
