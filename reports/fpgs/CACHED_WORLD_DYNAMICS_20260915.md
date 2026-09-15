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

## First control and cause-specific correction

The paired control completed with source, idle and capacity guards passing.
RTX physics: 9.316078 -> 9.381486 ms (0.9930x); GB: 9.470825 -> 9.548018 ms
(0.9919x). This is one discovery run, not repeated evidence. Artifact:
`/tmp/fpgs-cached-world-control-paired16k-20260915-01/manifest.json`.
No improvement is retained. The source establishes an additional publication
after the intermediate substep; its causal cost is not separately measured
in this graph-only control.

Fund one bounded correction: reuse valid published inputs inside K1 while
retaining the original lazy-publication schedule and current-force Pass3.
Use whole-warp hit admission to keep existing full-warp synchronization safe
under mixed masked resets. Load poses/S/bias/origin on a hit, otherwise run
the original cold path. Consume validity before integration so same-buffer
lazy steps cannot reuse pre-integration state. Ordinary publication restores
validity. Limit admission to existing non-snapshot FK caching, no fused mass.

This exposes only a proper subset of 0.404432 ms in the old RTX K1 owner,
less current-force work and cache loads. It is a small bounded dataflow
correction, not a claimed route to another 2--4x. Its specific value is that
ANYmal's last accepted RTX ratio (3.91x, historical MJWarp denominator) is
close to 4x. Do not report crossing 4x without repeated fresh backend timing.
Target one integrated screen within 30 minutes; do not pursue a mapping grid
or promote merely because the kernel is faster in isolation.

### Native correctness before the first screen

The cache reuse exposed a publication-cadence defect: public lazy publication
runs after `step()` increments `_step`, but the old preparatory inertia flag
uses `_step + 1`. K1 previously hid stale compact terms by recomputing them.
The opt-in owner now prepares the actual upcoming `_step` on forced
publication; normal/default publication is unchanged.

Regression-first RTX test failed before that repair: 85/90 velocity entries,
maximum difference 8.460134e-5 with unchanged rtol/atol3e-6. Source patch and
failure log are preserved in `/tmp/fpgs-cached-world-native-X47xqnXj/`.
After the repair all three selectors pass on each GPU, covering current
forces, mixed graph resets, held/requested mass and foreign state identity.
These compact floating18 fixtures are contact-free; this is dynamics/cache
qualification, not an assertion of loaded-contact or task quality.

## Close the bounded cache-only experiment without promotion

Measured runtime: `f5086ec92c6559435bb37b6a24668268647dac31`.
The final three native selectors also cover same-buffer lazy integration;
all pass on RTX and GB300. Full pre-commit passes.

| GPU | Retained physics | Cache-aware physics | Ratio |
| --- | ---: | ---: | ---: |
| RTX PRO6000 | 9.222728 ms | 9.195483 ms | 1.002963x |
| GB300 | 9.480967 ms | 9.341983 ms | 1.014877x |

One discovery run, not repeated evidence. Source, idle, finite-state and
capacity checks pass. The RTX result is not a meaningful large gain; do not
promote, rerun a timing grid, or update the MJWarp standings from this sample.
Retained Newton remains ca0d; no Lab or parent-pointer change.

One existing process-correlated node reader diagnoses the executed path:
24 K1 calls and 12 publication calls over three environment steps in both
arms. Actual `_c1` K1 is present; it is not a silent fallback. RTX K1 falls
0.805248 -> 0.572075 ms; GB 0.648640 -> 0.465984 ms. Warm/cold RTX launches
alternate around 44.5/98.5 us versus original 100.7 us average. Registers
remain 80 RTX (80 -> 79 GB), with unchanged shared storage and zero spills.

However, required next-publication cost rises 0.082485 -> 0.199392 ms RTX
and 0.047424 -> 0.094272 ms GB. Correct reusable inertia terms have a real
production cost. RTX composite cost also rises 0.238603 -> 0.287701 ms.
The manifold kernel changes 0.016533 -> 0.147616 ms RTX and
0.019179 -> 0.181120 ms GB despite identical collision source/resources.
The latter increase is observed, not causally explained; boundary contact
counts do not establish the inner queried workload. Do not blame register
pressure or claim physical equivalence from these timings. A future larger
replacement needs same-input attribution of that collision difference.

Local artifacts (preserved, no counters or sudo):

- Control manifest: `/tmp/fpgs-cached-world-control-paired16k-20260915-01/manifest.json`,
  SHA256 `61c5905ccf4c39e5df79cf2f7757f164c22ff07dfd38c9f4d79554eff4032563`.
- Whole manifest: `/tmp/fpgs-cached-world-native-paired16k-20260915-01/manifest.json`,
  SHA256 `1403544b41afaa98e61435b3ef6605c75767638844385ee4a35feef410624cdc`.
- Node manifest: `/tmp/fpgs-cached-world-native-nodes-paired16k-20260915-01/manifest.json`,
  SHA256 `ee2530a1f3b7a800a93c42fcbda51df67414eca114c6da34f75a7ca4b4d67c1d`.
- Reader output: `/tmp/fpgs-cached-world-native-X47xqnXj/nodes.json`,
  SHA256 `8c85b3cd755d788f06830d0f41fd89b756e36472e8e6c18c65da8630aa62a3c1`.
- Reader: `/tmp/fpgs-franka-articulated-attribution-Y8sTiAgS/audit.py`,
  SHA256 `e76d53f685871f8d3e38a221ce89485d3d87014b9c8af1f6ac22e10af8e74dc6`;
  reused `read()` with `categories=lambda name: ['all']`, unchanged root,
  process, source and capacity checks. Node traces are not throughput evidence.

The adjacent reproduction script reruns the checked whole or node experiment
using the existing pinned local driver/adapter; raw captures remain local.

## User-proposed next architectural direction: islands and sleeping

The retained Keyboard path already splits the 108 scalar key responses from
the six robot DOFs and solves independent chains. It does not implement
dynamic sleeping. Current key force, integration and next-publication work
still runs for inactive keys. A complete active-key pipeline could remove
more work than solver-only islanding, but needs collision/force/reset/model
wake handling and valid public/contact-force state. No speedup or sleepable
fraction has been measured. This is not the previously failed stationary
limit-only arithmetic experiment.

Current MJWarp comparisons use external Newton contacts. The installed
SolverMuJoCo rejects sleeping with that contact mode, so existing ratios are
not comparisons against sleeping-enabled MJWarp. Treat an internal-contact,
sleep-enabled comparison as a separately labeled backend configuration.

Matrix-free contact response is already present in FPGS; this does not mean
all retained paths have linear contact-count cost. Franka local residuals
form a small packed response matrix, and ANYmal's exact row-sum preparation
still computes all row-pair products. Dense DOF-factor work is distinct from
contact count. A newly cited linear-time method must be identified before
claiming it is implemented or an unexploited gain.
