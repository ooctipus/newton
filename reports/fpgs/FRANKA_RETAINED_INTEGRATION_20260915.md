# Retain the qualified Franka structural gain as an opt-in

Decision: 2026-09-15 08:22 UTC, before integration changes.
Base: accepted `b1bad06ae0cd5b86ac07b2f2844ef62770e99075`.
Source: runtime `865dbd6c7c5009b0b1f381c2a62baa55a125ce6e`, report tip
`106a149fe5db78d876f834e1f60ff8bfc7709118`.

## Why reconsider a closed result

The original 0.55 ms funding milestone was missed. Three balanced paired
rounds nevertheless establish a substantial structural physics gain:
RTX median 5.545875575 -> 5.070682400 ms (1.093714x), GB300 median
4.999355200 -> 4.721928800 ms (1.058753x). An internal prioritization threshold
is not a correctness criterion and should not discard this useful result.
This is an explicit acceptance-policy reconsideration, not a new optimization,
new measurement, relaxed physical tolerance, or claim of reaching 4x.

RTX environment wall time is approximately unchanged. GB300 is slower in
all three pairs, mean +0.492922 ms (about 1.73%). Existing nested event timers
and the 22 exact device comparisons plus one four-byte readback are consistent
with residual notification overhead; they do not isolate its full causal cost.
Do not call this an environment or training speedup, hide the GB tradeoff,
or reuse the earlier corrected 16 ms/1.67 s prototype regressions as its cause.

Retain the existing default-off option. It is an available physics-focused
RTX recipe; GB300 users retain the opt-out and the explicit wall-time caveat.
No hardware-dependent numerical law or automatically selected weaker proof.

## Exact integration and qualification

Merge the existing qualified source without changing its native mathematics,
held-factor lifetime, device notification checks, capacities, or allowances.
Preserve both disjoint G1 and Franka owners at all eight shared host-hook
conflicts, including prevalidation-before-mutation, reset, current/held mass
request handling, force/size-stream ordering, and publication. Keep Kuka's
accepted current-contact path and the existing teardown behavior.

The original independent physical, current/held, loaded/reset, five-world
odd-tail, two-bank graph and notification tests all passed on both GPUs;
all 12 repeated timing children and 24 capacity boundaries passed. Reuse those
artifacts rather than pretending this policy decision is fresh timing.

In this new tree, run the existing eleven Franka CPU controls and four CUDA
selectors plus the relevant retained G1/Kuka integration selectors. Verify
native files against the qualified source and unchanged unsupported dispatch.
Root owns GPU execution. No new physical test framework, mapping experiment,
cache policy, parent dependency pointer, or Isaac Lab modification is needed.
Any source/physical integration failure blocks retention until resolved.

The accepted b1bad worktree stays untouched during integration and Keyboard
timing. Push this new ooctipus branch only with its qualification status and
source ancestry explicit. Do not substitute an inferred current MJWarp ratio
for a matched backend measurement or rewrite the original negative report.

Status: opt-in integration approved; combined-source checks pending.

## Combined-source CPU checkpoint

The merge preserves both parent histories. Only `solver_feather_pgs.py` needed
conflict resolution: eight host-hook overlaps, with both disjoint owners
retained. All notification validation remains before mutation. G1 keeps its
current-S/held-W producer, drive-event ordering and request consumption;
Franka keeps its original current-force/L9/L6 path and complete publication.
Kuka's current-contact owner, other production modules and the common
device-drain destructor are unchanged from b1bad. Independent hook review
found no blocker. The option remains default off.

Both Franka native files and both existing test files are byte-identical to
865dbd6c. Every existing top-level solver/native function has an unchanged AST;
the original factor factory retains its exact `5d220f15...` source pin. No
native equation, tolerance, test oracle, capacity or iteration was modified.

| File | SHA256 |
| --- | --- |
| `franka_kinetic_state.py` | `fa9aa0e1888c1b6bf0e05e73b89e92175ee5fa6de40ee08286e8e952e85c1cfe` |
| `franka_kinetic_factor.py` | `eda0a22bb7328cef4f691970a8551fb5bc98d209c51977dbca60123ee91c9b8e` |
| `test_franka_kinetic_state.py` | `a19104b5f56e84b92afb55aca324fbae66cdd250c8a7e57f6d0a2f5cb4aa17a7` |
| `test_franka_kinetic_factor.py` | `f50240b2a4ae33008d30ee12a2405df8063c5349988350e20e4b1e569496c186` |
| Resolved `solver_feather_pgs.py` | `40e030b0a67ee5f6205f9efabb7ea4a266b8145ab689a942f5dde69aed90ae55` |

The first combined CPU run passes 55 executed tests with one explicit CUDA
skip (56 selected, 4.696 seconds, session71536 exit0). It includes the eleven
Franka controls, five retained G1 controls, Kuka binding/lifecycle/current-contact
controls and two teardown controls. A second run with the accepted Kuka
current-contact flag enabled passes all 22 binding/lifecycle tests (2.077
seconds, session27582 exit0). These are 77 executed tests across the two runs,
not 77 distinct selectors. The CUDA-hidden initialization notice and inherited
joint-target-layout deprecation are not GPU execution or numerical failures.

Exact commands, from this worktree:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python \
  python -m unittest \
  tools.fpgs_bench.test_franka_kinetic_state.TestFrankaKineticStateCPU \
  tools.fpgs_bench.test_franka_kinetic_factor \
  tools.fpgs_bench.test_g1_kinetic_state.TestG1KineticStateCPU \
  tools.fpgs_bench.test_kinetic_live_bindings \
  tools.fpgs_bench.test_kinetic_live_owner \
  tools.fpgs_bench.test_kinetic_current_contact \
  newton.tests.test_feather_pgs_teardown

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
FEATHER_PGS_KUKA_CURRENT_CONTACT=1 uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python \
  python -m unittest tools.fpgs_bench.test_kinetic_live_bindings \
  tools.fpgs_bench.test_kinetic_live_owner
```

Combined-source CUDA checks remain root-owned and pending. This checkpoint
does not relabel the inherited timings as a new benchmark or establish a
new backend ratio. The original negative reports remain unchanged.
Full `uvx pre-commit run -a` passes (session3475, exit0); the native and resolved
solver hashes remain unchanged after formatting and checks.

## Combined-source native qualification and retention

The root-owned paired native run on clean merge
`b9ba04a471aa659206d4e67518e421d527e41aad` passes all eight selectors on each
card: RTX 17.882 seconds (session40388), GB300 18.021 seconds (session49686).
Both sessions were reaped with exit0, with no skipped tests or errors. The only
warnings are the inherited joint-target-layout deprecations. These durations
include test/module loading and are not whole-physics benchmark measurements.

The four Franka controls cover loaded current/held forces, publication, exact
device-proof rejection/rebinding, five-world odd-tail/reset and two-bank graph
lifetime. The three retained G1 controls include anchored-root current-force
prediction, current/held geometry, all sixteen saved inputs and the actual
five-world mixed-cache graph. The retained Kuka current-contact boundary
control also passes. Its scope is the existing boundary control, not a new
complete Kuka qualification campaign. All original physical tolerances remain
unchanged. Franka's reported maximum current-H9 error is 1.3396387e-6, bias
error 2.0842464e-6, held momentum9 error 1.7301406e-7 and public velocity error
7.1193080e-7; the retained G1 independent-H error is 8.906245e-8.

Verified complete logs under `/tmp/fpgs-franka-retained-native-jLfTopU1`:

| Card/log | SHA256 |
| --- | --- |
| RTX `gpu0.log` | `6b911ff277707e6903ecfa11bc7fca186e4a50b0134137ca05cbe880699bcbd4` |
| GB300 `gpu1.log` | `0d62ed72f75843dcff98d68628aa56f66108cd6c1ec8adda4aad501e20dbaf6b` |

Exact RTX invocation, from the retained worktree:

```bash
set -o pipefail
env CUDA_VISIBLE_DEVICES=GPU-883586b6-3100-0610-81e5-3b4c26f45639 \
  PYTHONPATH=/home/octi/Projects/newton-fpgs-franka-retained-20260915 \
  FPGS_TEST_DEVICE=cuda:0 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python \
  python -m unittest \
  tools.fpgs_bench.test_franka_kinetic_state.TestFrankaKineticStateCUDA \
  tools.fpgs_bench.test_g1_kinetic_state.TestG1KineticStateCUDA \
  tools.fpgs_bench.test_kinetic_current_contact.TestKineticCurrentContact.test_cuda_current_contact_boundary \
  -v 2>&1 | tee /tmp/fpgs-franka-retained-native-jLfTopU1/gpu0.log
```

The simultaneous GB300 command is identical except
`CUDA_VISIBLE_DEVICES=GPU-ebfac9e8-02d5-d8a9-3bfc-bac64c62ffd4` and the output
path ends in `gpu1.log`. No external solver feature flags were set; the
existing selectors manage their explicit opt-ins. Use fresh log destinations
when reproducing to preserve these original artifacts.

Final decision: retain this qualified integration as the existing default-off,
physics-focused opt-in. The reported repeated speedups remain measurements of
865dbd6c, not new timings of this merge. RTX wall neutrality and the measured
GB300 mean wall regression of 0.492922 ms remain explicit tradeoffs. This is
neither an environment/training throughput claim nor achievement of the
all-task 4x goal. The report-only qualification update changes no runtime,
test, original negative report, Isaac Lab source or parent dependency pointer;
all source/test hashes above remain exact.
