# Kuka full-visit lazy physical response

## Pre-code decision, 2026-09-15 11:07 UTC

Conditional GO for one integrated structural prototype, not a measured gain.
Base is retained `ca0d427af809571bb5501f644c1a6e03990cd2a8` on
`ooctipus/newton`; branch `ooctipus/fpgs-kuka-lazy-response-20260915`.
Preserve the handoff and original worktrees. Newton only; Lab, timestep,
substeps, eight-sweep allowance and all capacities stay unchanged.

### Work replaced

Maintain physical generalized velocity increment instead of a whitened
increment. Publish current full physical J, original endpoint-derived r0,
CFM and row metadata using the existing raw-striped producer. For MF0 and
independent primary/free worlds, do not eagerly construct every T*J response
or denominator. Every original normal/limit visit remains in every original
sweep. When a row first needs an impulse, form z=T*J, R=T-transpose*z and
diag=dot(z,z)+CFM; cache R and diag for the remaining visits. Residuals are
r0+J*delta-v and updates are delta-v+=R*delta-lambda. Preserve original
normal/tangent order, sibling disk updates and stationary exit semantics.

This is not the closed once-per-sweep admission screen: contacts activated
by earlier contacts are handled immediately at their canonical visit. It is
not the closed private/cooperative owner: no eager full-world response panel,
new row-slot map, or altered positive-MF stream dependency is introduced.

Independent qualification uses all-row physical J_free==0 and the original
current MF endpoint/metadata predicates. The held operator is block diagonal
between primary23 and free6, so these dense responses cannot affect free6.
The original negative-selector general owner reads no dense rows and retains
its original free6 law concurrently. Coupled/unsupported positive worlds
receive eager responses and the original hybrid/general fallback law; their
production, numerical qualification, bridge and exposed time are charged.

### Whole-step budget and falsifier

Accepted source-matched RTX node capture is
`/tmp/fpgs-kuka-cooperative-response-nodes-paired16k-20260915-01/round_01_baseline/round_01_kuka_fpgs_gpu0/strict_cooperative_response_node_audit.json`.
Whole physics is 10.538243 ms in that diagnostic, complete affected union
3.595766 ms: rows/response1.746955, joined solve1.061920, MF services0.451136,
qualification/materialization0.335755 ms. Do not add overlapping solve sums.
Accepted repeated whole physics is separately 10.480220 ms RTX/10.689198 GB.

First milestone: approximately 1.05 ms RTX whole saving (at least 10%),
requiring affected union at most 2.545 ms with retained work unchanged.
After existing MF/qualification, raw-J production, lazy solve including
first-use actions/cache, and positive fallback/exposure have at most1.758 ms.
An engineering allocation is0.55+0.85+0.35 ms respectively. These are tight
conditional allowances, NOT measured stage costs or a population forecast.
MF0 alone is not funded: the masked positive-MF row control alone retained
0.625889 ms before new full-J publication and first-use transpose work.

Added work includes complete common-arm physical J, J memory reads/writes,
first-active T-transpose action, response-readiness lifetime, actual fallback
materialization and any changed overlap. Reuse existing J/response storage;
do not increase contact/row capacities. Preserve current geometry, MF forces,
publication and public contact impulses. This milestone is not the whole
4x target: older corrected MJWarp27.504273 ms implies a6.876068 ms ceiling.

### Numerical and lifetime contract

Keep original incident r0 rather than newly rounding J*vhat+bias. Complete
MF0 common-arm J and signed-coordinate prefix J. Readiness is separate from
geometry validity and is reset every invocation; no stale response survives
reset, withdrawal/regrowth, refresh or held-factor reuse. Negative-selector
dense ownership writes primary only; MF retains free6 and original joins.

Unused-response overflow checking changes deliberately: check finite current
J/r0/material/held inputs eagerly, and finite z/R/positive denominator when
first needed. Do not claim unchanged eager validation of arithmetic that is
no longer executed. Positive fallback must complete its original numerical
qualification on freshly materialized operands before selecting its bridge.
Zero/unsupported CFM conditions need an explicit safe decision, not a stale
denominator read or unjustified positivity assumption.

Selected CPU algebra evidence before coding:11 retained current-contact
worst-world readsets match original FP64 eight-sweep velocity/lambda (maximum
scaled velocity difference9.55e-16), building32/348 responses while retaining
all visits. The nine-contact late-activation counterexample also matches;
the closed omission law leaves residual-0.08290845. This is neither native
FP32 qualification nor an all-world activity census.

Reuse existing fixtures: original-eight physical residual/complementarity,
momentum/cone law, late activation, common-arm/prefix, independent MF, coupled
fallback, reset/stale/withdraw-regrow. Run a small native paired smoke first,
then existing 16K integrated paired graph screen early. Freeze all checkout
bytes during benchmark. Root owns both GPUs, one process per card; do not
interfere with external jobs. No new general benchmark framework.

Initial checkpoint12:37 UTC (90 minutes). A first loss gets one bounded
node diagnosis and a correction only for a demonstrated cause. No silent
extension past13:07 UTC; record integrated gain, diagnosed loss or unvalidated
hypothesis. No mapping grids or sub-percent polish. Promotion requires
repeated paired whole measurements and relevant physical/reset controls.

## First integrated native checkpoint, 11:28 UTC

The full source path is implemented behind
`FEATHER_PGS_KUKA_LAZY_RESPONSE=1`, default off. Nonpositive/nonfinite CFM
retains the complete original path. The existing materialize dispatch now
owns positive eager response, numerical qualification and the original
bridge, with explicit inter-function warp barriers including failure paths.
There are no additional launches or enlarged row/contact buffers. A redundant
new full-J/held/metadata validation scan was removed during review: existing
producers, row validation and original entry guards already own those checks.
First-executed residual and first-used response finiteness remain checked.

Regression-first missing-module failure was observed. Five focused CPU tests
pass after the final changes (3.595 s). Two native tests pass on each GPU,
9.305 s RTX/9.481 s GB, no skips, both processes reaped with exit0. These
exercise original-eight late activation, full common-arm/prefix J, actual
independent/coupled routing, momentum/cone bounds and captured readiness
changes. They are focused physical checks, not a loaded-task qualification
or performance claim. Whole-task timing is next.

Frozen runtime SHA256:
`0760c8b628dc943d092c92a3772edf121e697a4aca3a864eb99253d0a595228d`;
binding `e81e9e7f42239495b9b63600505ec7f8d817b5ee72314e003fe2d32c3b8216c7`;
test `0c710b6c0f7405f3e8de0fdeda798eff5c5ff6bd426dabddd228500b6614a4fa`.
Logs: `/tmp/fpgs-kuka-lazy-response-native-63BcHTo8`, RTX SHA256
`70c6cff8d0c7cdf7ba626da73a768e59b896ce0dcd142c146a9f084b8512fa49`, GB
`906d87399ff66748a4f7abdcd2fab4513729235b137b4ef91a0f82c53645f4fd`.

Reproduce the focused test module with the fixed Lab interpreter via
`uv run --no-project --python
/home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python
python -m unittest -v tools.fpgs_bench.test_kinetic_lazy_response`.
Run from this Newton checkout, explicitly set its PYTHONPATH and
`FEATHER_PGS_KUKA_LAZY_RESPONSE=0`; each fixture chooses its tested arm.
For each isolated GPU UUID set `FPGS_TEST_DEVICE=cuda:0`; for CPU hide CUDA
and leave that selector unset. Root alone owns paired GPU launches.

## Whole-task decision, 11:44 UTC: do not promote this version

Measured runtime is clean `eb9c0ec8f9a57dcf1c574ed5fcc5591d8f686eef`;
retained baseline is `ca0d427af809571bb5501f644c1a6e03990cd2a8`.
The first complete graph A/B finished at11:34, about27 minutes after the
pre-code decision. All four children exit0, all eight finite/capacity
boundaries pass, and final source/idle guards pass. This is one discovery
round, not repeated acceptance evidence.

| Whole physics, ms/environment step | RTX | GB300 |
| --- | ---: | ---: |
| Retained | 10.508200 | 10.634408 |
| Lazy response | 10.645301 | 10.393238 |
| Retained/candidate | 0.987121x | 1.023205x |

Environment wall time is separately35.137717→36.750310 ms RTX and
36.645139→34.758045 ms GB. Neither is a full training measurement.
The ≥1.05 ms whole-saving milestone is missed; no new accepted gain or
updated MJWarp ratio is claimed. No long qualification/repeat campaign is
funded for this version.

The immediate three-step node diagnosis finishes at11:37. All four strict
audits pass:12 physics roots and1728 nodes each, zero auxiliary/unproven
nodes, same capacities and eight calls per changed owner. These diagnostics
are not additional whole-timing repeats.

| Complete affected family, ms/env step | RTX retained | RTX lazy | GB retained | GB lazy |
| --- | ---: | ---: | ---: | ---: |
| Rows/response | 1.731265 | 1.435265 | 2.157866 | 1.510495 |
| Joined solve union | 1.157664 | 1.132097 | 1.040373 | 1.333973 |
| Qualification/materialization | 0.329248 | 0.514188 | 0.384198 | 0.642000 |
| MF services | 0.450102 | 0.456972 | 0.497487 | 0.489594 |
| Complete affected union | 3.668279 | 3.538522 | 4.079923 | 3.976062 |

The failed premise is identifiable. RTX raw contact production only falls
1.432822→1.178934 ms, not to the conditional0.55 ms allowance. Original
MF0 production already constructs current J in shared memory and publishes
Z; this version publishes equally sized J instead. It retains current
axis/anchor projection, endpoint incident/restitution and row metadata.
Unused responses therefore do not mean unused producer work. Positive
materialization adds0.207115 ms RTX, consuming most of the modest retirement.
RTX primary solve is roughly unchanged1.115776→1.090220 ms; register tuning
is not justified as the explanation for the missing millisecond.

On GB, primary solve grows0.991659→1.284810 ms and positive materialization
adds0.292661 ms. Reduced general-owner sum is offset by lost overlap; use the
joined union above. Actual traced lazy solve uses120/118 registers versus
60/56, shared6576 versus5760 bytes, local0. These do not establish achieved
occupancy or a unique hardware stall cause.

Decision: close this version unpromoted. Eliminating only the added fallback
cost cannot recover the missing≥0.92 ms RTX. A subsequent candidate must
retire J/metadata production itself or another substantial owner, and price
its replacement consumers; it requires a new pre-code budget. No mapping
grid, micro-optimization or solver-law change is authorized by this result.

Artifacts: `/tmp/fpgs-kuka-lazy-response-paired16k-20260915-01/manifest.json`
SHA256`d3b7ebebbdaa73ddb1a5bc53f28bba60f29c6ac159cbcada7d630a391f5214f1`;
node manifest in `/tmp/fpgs-kuka-lazy-response-nodes-paired16k-20260915-01`
SHA256`6122be56a14b7273cdd7b9a555a2e45f666bc0f511a04c5f61f4efeba0b358f2`;
strict unchanged-reader alias audit `/tmp/fpgs-kuka-lazy-node-audit-AQ9Y0jsn`.
Large captures remain local.

Reproduce the graph screen with a NEW output directory:

```bash
kuka_variant_args=()
for kuka_flag in SIMPLE_WORLD_ZERO LOCAL_ROW_PACKETS INDEPENDENT_COMPONENTS \
  PAIRED_GENERAL_OVERLAP KUKA_JOINT_WORLD WORLD_SCAN_PUBLICATION \
  KUKA_KINETIC_WORLD KUKA_CURRENT_CONTACT; do
  kuka_variant_args+=(--baseline-env "FEATHER_PGS_${kuka_flag}=1")
  kuka_variant_args+=(--candidate-env "FEATHER_PGS_${kuka_flag}=1")
done
uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python \
  python /tmp/fpgs-kinetic-fixed-variants-TZRkIPYw/run.py \
  --isaaclab /home/octi/Projects/IsaacLab.wt/contact-reset-20260913 \
  --baseline /home/octi/Projects/newton-fpgs-keyboard-linear-state-20260915 \
  --candidate /home/octi/Projects/newton-fpgs-kuka-lazy-response-20260915 \
  --task kuka --gpus 0 1 --rounds 1 --seed 0 --num-envs 16384 \
  --warmup-steps 200 --steps 40 --profile-steps 40 --trace-mode graph \
  --capacity kuka:fpgs:rigid_contact_max=311296 \
  --capacity kuka:fpgs:broad_phase_output_max=442368 \
  "${kuka_variant_args[@]}" \
  --baseline-env FEATHER_PGS_KUKA_LAZY_RESPONSE=0 \
  --candidate-env FEATHER_PGS_KUKA_LAZY_RESPONSE=1 \
  --output-dir /tmp/fpgs-kuka-lazy-response-reproduction-new
```

For the diagnostic only, select`--profile-steps 3 --trace-mode node` and a
different new output directory. Adapter SHA256 is
`c062388f23f82c524690f272fb2d14141ed4126db641de6dcd86f8ade5bfe416`;
unchanged benchmark tools are at`961b7e2f751bcd1d8b03368e7956b54c81414897`.
