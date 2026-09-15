# Cached world dynamics: causal control before a larger replacement

September 15, 19:17 UTC. Base retained Newton
`ca0d427af809571bb5501f644c1a6e03990cd2a8`, fork `ooctipus/newton`.
Isaac Lab stays at `53ee6b44c2334341305dbdf385a3916c6b140799`.
No sudo, driver changes, capacity enlargement or physics-budget changes.

## Candidate and full cost

ANYmal's current fused K1 recomputes poses, motion subspaces, motion and bias
from joint state without reading the existing FK/ID validity bit. Final
publication already produces those values. This is actual duplicate work,
but lazy publication occurs only after the second substep: only half the K1
calls follow publication. Do not claim all K1 time is removable.

Existing strict RTX node evidence at
`/tmp/fpgs-anymal-active-dual-nodes-paired16k-20260915-01/round_01_baseline/round_01_anymald_fpgs_gpu0/strict_active_dual_node_audit.json`
reports K1 0.808864 ms, template FK 0.273515 ms, finalization 0.081579 ms,
composite inertia 0.239275 ms, and gather/triangular solve/scatter 0.326 ms
per environment step. These are owner sums, not additive critical-path
savings. The source is equivalent for these owners; the retained recipe
uses lazy kinematics, 16K worlds, two substeps, original eight-iteration
fallback and eligible parallel24 solver.

First use the existing `FEATHER_PGS_FUSED_K1=0` path as a causal control,
not a newly implemented optimization. It reuses valid cached FK/ID, then
computes current external/passive/control forces through grouped tau.
It also publishes after BOTH substeps, adding approximately 0.355 ms at
the old node prices. Mandatory torque work remains. This control's ceiling
is therefore less than 0.454 ms, and it cannot itself support a large-gain
claim. One paired whole-task batch is sufficient to select the next step.

A broader candidate, if supported, would cache generalized bias18 and use
one held-L predictor for current force through v-hat, together with compact
13-moment composite input consumed directly by the L18 factor builder.
It must replace the old producers and conversions, not retain both paths.
Keep original L324, row response, parallel24, public state and held/current
mass semantics. Charge new bias projection, live external-force traversal,
two triangular solves, cold repair and factor-input work. This larger
implementation is not funded by the K1 upper bound alone.

## Decision and qualification

Keep baseline and control on the same retained runtime. Reuse the fixed
paired benchmark owner and calibrated ANYmal capacities: dense72, raw212992,
broad294912. Seed0, warmup200, wall40, graph40, both idle GPUs concurrently.
No fresh MJWarp or six-task rerun is needed for this causal check.

Check actual source pins, GPU idleness, budgets, warnings and row capacity.
Passing these checks does not prove physical equivalence. Before retaining
any implementation, reuse the existing mass-update/cache and inertial-change
tests plus loaded-contact controls, including graph replay, mixed resets,
state identity changes, live forces and requested mass refresh.

The first control is bounded to one batch. Any implementation that loses
must receive a cause-specific diagnosis before closure; do not start a
mapping grid or new validation framework. No new speedup is claimed here.
