# Keyboard translation-only state: pre-code decision

Decision: 2026-09-15 09:36 UTC. Base `28f93014520504e90be34a304507368c7c82c4cc`
(retained Franka integration, accepted Keyboard runtime unchanged from b1bad).
Branch `ooctipus/fpgs-keyboard-linear-state-20260915`. No shared-tier,
read-only-row or stationary-limit experiment is included.

## Work to replace

The existing Stage7 prismatic plan already admits fixed roots and independent
translation-only leaves, but its leaf publisher still invokes the complete
general spatial-body dynamics law. Cold Stage1 still traverses all leaves
serially. The direct scalar mass owner reads twelve inertia terms to project
them against a motion vector with identically zero angular part.

Replace that complete state-production boundary for admitted leaves:

- Reuse the topology-only plan and current frames, axes, q and passed Stage1
  velocity (including any existing prescaling). Compute leaf poses/motion in
  parallel during cold repair and next-state publication.
- For zero angular velocity and zero bias acceleration, Coriolis and inertial
  bias vanish. Publish the full gravity wrench, current public pose and COM
  velocity directly; do not rotate inertia merely to multiply it by zeros.
- The mass projection is `m * dot(axis_world, axis_world) + armature + max(K,0)`.
  Preserve non-unit axes, current drive coefficients and held-mask behavior.
  Remove ordinary key inertia materialization only when no retained consumer
  needs it. Preserve all generic articulated-arm and other unsupported paths.

This is not a contact algorithm or capacity change. Every contact, impulse,
force callback, timestep, substep and original iteration allowance remains.
No Isaac Lab or parent dependency pointer changes.

## Consumer and lifetime requirements

The global parallel composite schedule already excludes the direct-diagonal
key articulation. Exceptional masked refresh/composite paths do not. Retain
their original complete materialization for this first implementation; do not
silently feed them absent buffers. The scalar mass path must cease reading the
ordinary key compact-inertia terms before those terms are retired.

Cold root traversal currently stamps cache validity. Preserve pre-launch
validity in one small persistent device buffer before shortened root repair;
the leaf pass must read that snapshot, not the newly stamped validity. Charge
its copy and launch. Finish all leaf writes on the current stream before any
inertia-ready event or force consumer. Keep reset and model-notification
invalidation, cache source identity and snapshot-backed modes correct; reject
unsupported opt-in combinations rather than silently changing their owners.
Prefer the existing cache contract over a new state machine or proof system.

## Cost and stop rule

Source-equivalent RTX node evidence is the baseline under
`/tmp/fpgs-keyboard-shared-tier-nodes-paired4k-20260915-01`:
next publication 0.867157 ms, cold/cache FK 0.712224 ms and direct mass
0.124224 ms per environment step. Only the first cold FK launch is expensive
(about 0.67472 ms); later cache hits are already cheap. Publication runs eight
times. The 1.703605 ms sum is an upper envelope, not an exclusive saving;
mass overlaps other work and required pose/state production remains.

Fund one integrated candidate targeting at least 0.7 ms RTX whole-physics
saving (roughly 9-10% at the current 7.4-7.9 ms baseline). This needs a large
cut across the cold and next-state producers, not a scalar-mass micro-tune.
All new copies, leaf launches, fallback materialization and joins are charged.
First integrated screen within 90 minutes; no silent extension beyond two
hours. If the full screen misses, use one existing node diagnosis to locate
the gap; no launch-size grid or follow-on polish without a new costed cause.

Use the unchanged paired4K driver, 200 warmup and 40 wall/physics steps,
fixed 704 rows, 147456 contacts and 57344 broad-phase pairs. Baseline is the
accepted runtime, not a failed prototype. A first screen is discovery only.

## Minimal qualification

Extend existing prismatic-publication and scalar-mass tests, regression-first.
Use the generic solver and independent `newton.eval_fk` oracle. Cover poisoned
cold buffers, warm Stage7-to-Stage1 reuse, partial reset and captured replay;
actual changes to root/child frames and non-unit axes; COM, mass, inertia and
gravity notification; external/passive/actuator forces; held and requested
mass refresh; and unsupported revolute/free articulations. Check retained
physical/public outputs, not obsolete unread inertia buffers or bit identity.
No tolerance widening or separate test framework. Default off until repeated
paired timing and physical qualification justify retention.

## First implementation and early-screen gate

The first implementation adds `prismatic_linear_state.py` and narrow host
hooks. `FEATHER_PGS_PRISMATIC_LINEAR_STATE=1` requires cached matrix-free direct
diagonal response, with every direct-size articulation exactly covered by the
existing prismatic plan. Unsupported/grouped/fused/snapshot owners reject the
opt-in. The default remains off; the old publication module is unchanged.

Cold repair copies the original articulation validity before the shortened
root launch and completes its body-parallel leaf pass before the existing
inertia/force events. The persistent addition is one int per articulation
(32 KiB for the 4K two-articulation recipe), copied each substep. Warm leaf
repair returns from that snapshot. Next publication retains the original
body-parallel mapping. Exceptional masked full inertia and composite work,
all drives and force consumers, and all constraint owners remain unchanged.

Source checks confirm the default direct-mass native function is unchanged,
the new scalar mass retains its argument ABI, and the new next-state finalizer
retains the original prismatic finalizer ABI. Independent runtime and lifetime
reviews found no concrete blocker. Native source SHA256:

- `prismatic_linear_state.py`:
  `f0cd1d10a516b182630f6fd97137e8d6514f53accb4d8bffcc06733ebd687da7`.
- `solver_feather_pgs.py`:
  `f5fe22374ec9f60a9400a5e73cb8e334195cc6833afb028b0b460d6c27493d3e`.

Regression-first CPU session 80969 failed on the missing
`_PRISMATIC_LINEAR_STATE` API before implementation. Focused tests reuse the
existing prismatic-publication and scalar-response modules; small CPU forced
state-boundary fixtures do not feed elided inertia into generic mass consumers.
Focused four CPU controls passed (session 34821). The complete existing
prismatic module plus new scalar-mass oracle passed seven tests with one
explicit CUDA skip (session 57668, eight tests total). These include cold/warm
repair, partial reset, notified current frames/inertias/gravity, and scalar
mass with poisoned retired terms. Existing external/passive full-step coverage
remains a legacy-path control, not qualification of the new CUDA owner.
Full `uvx pre-commit run -a` passed (session 97622); the first pass only
formatted the two test files and did not change runtime source.

After focused CPU/source checks, the first clean paired whole run is expressly
an **unqualified cost screen**, not promotion. Full native cold/reset/current
mass and captured lifecycle controls remain a retention gate. This sequencing
does not claim that source review or CPU compilation proves CUDA physics.

## First screen: constructor failure, no candidate timing

The first paired screen at runtime `cd3ec68d5a5c65eb54abc1faaef028fcbd4f794c`
failed before either candidate was constructed. Root parent 58617 was reaped
with exit 1; both baseline captures completed, but neither candidate produced
a timing report. Preserve the original artifacts at
`/tmp/fpgs-keyboard-linear-state-whole-paired4k-20260915-01` (manifest SHA256
`f0319b699acf1285ccab3b246d80a544a6fe2655520b7d58889061d2ca3a9f91`).

The admission check incorrectly boolean-indexed the N+1 articulation-start
array with an N-articulation mask (8193 versus 8192 in the real task). The
intended comparison is against `starts[:-1][admitted]`: the final entry is
the end sentinel, not another articulation. Correct only this host slice and
add a real-plan CPU admission regression. No native math, cache ordering,
physical tolerance or solver budget changes are involved.

The new actual-plan CPU regression first reproduced the same `IndexError`
(session 36463, four starts versus three articulation flags). After the slice
fix, all eight CPU controls passed with one explicit CUDA skip (session 69208,
nine tests total, 0.364 s). An earlier invocation used a nonexistent scalar
test-class name; its loader error was corrected without source changes.
AST comparison confirms every native function is unchanged from `cd3ec68d`;
only `configure_linear_state` differs. Corrected module SHA256:
`e3bfb42113b651aabcce92902771b025109e1d7186a31edeae02b225de733e1c`.
The solver-hook hash remains `f5fe22374ec9f60a9400a5e73cb8e334195cc6833afb028b0b460d6c27493d3e`.

## Retained result: repeated physics gain and matched MJWarp comparison

The measured runtime is `94153754aeaca76f92d3fbf8cb0ebc0de03160be`.
No runtime source changed after its constructor correction. Retain this
default-off path with `FEATHER_PGS_PRISMATIC_LINEAR_STATE=1`; it is not enabled
for other representative tasks or unsupported solver configurations.

Three alternating before/after pairs, unchanged seed0/4K recipe:

| GPU | Baseline physics median | Linear physics median | Speedup | Saving |
| --- | ---: | ---: | ---: | ---: |
| RTX PRO6000 | 7.755844 ms | 6.803919 ms | 1.139908x | 0.951925 ms |
| GB300 | 6.654423 ms | 5.945697 ms | 1.119200x | 0.708727 ms |

Every physics pair improves. RTX baseline samples: 7.755844, 7.758568,
7.676726 ms; candidate: 6.772816, 6.803919, 6.817308 ms. GB baseline:
6.740669, 6.552950, 6.654423 ms; candidate: 5.945697, 5.928910, 5.999382 ms.
Environment-wall median speedups are 1.049277x RTX and 1.019566x GB, but
GB's third wall pair regresses about 4.15%. These are not training timings.

All twelve children and 24 capacity boundaries pass, with unchanged budgets
and successful source/idle guards (session24599, exit0). Preserve
`/tmp/fpgs-keyboard-linear-state-repeated-paired4k-20260915-01`, manifest SHA256
`f9514371a72fe2ffd4122e6eeff9a776d5b17013ed7938ce54d133e6f21108ab`.

### Discovery and the one component diagnosis

The earlier corrected single whole screen passed all four children/eight
boundaries: RTX7.600451 -> 6.990566 ms, GB6.814330 -> 6.120821 ms.
Its RTX0.609886 ms saving missed the initial0.7 ms target. It was not hidden
or automatically rejected. Directory
`/tmp/fpgs-keyboard-linear-state-whole-paired4k-20260915-02`, manifest SHA256
`7563b5963dbed8ba091af6f0bb0957cec0205940dad6c3b044c3695ff4955eee`
(session3010, exit0).

The single node diagnosis passed all guards (session11320, exit0),
`/tmp/fpgs-keyboard-linear-state-nodes-paired4k-20260915-01`, manifest SHA256
`f355360218d2350a2fea811ac2a1b0f73e6ae81767cee757429a21963d738476`.

| Replacement owner, ms/step | RTX old -> new | GB300 old -> new |
| --- | ---: | ---: |
| Cold/cache FK | .442304 -> .066166 | .697077 -> .056352 |
| Added leaf repair | 0 -> .125323 | 0 -> .077557 |
| Added validity copy | 0 -> .019723 | 0 -> .015712 |
| Next publication | .830219 -> .723638 | .361984 -> .285675 |
| Direct mass | .131989 -> .080192 | .102378 -> .055018 |

Owner sums overlap; they are not exclusive whole savings. Every listed owner
runs eight times, but FK is not cold eight times. In this three-step RTX span
the baseline resets twice and candidate three times; the smaller baseline
cold exposure also explains most of the difference from the pre-card envelope.
Required state publication remains substantial. No hardware-counter stall or
bandwidth claim is established. Repeat the unchanged path, not a tuning grid:
the repeated result subsequently exceeds the original RTX target.

### Fresh corrected MJWarp: Keyboard physics clears 4x

Both arms use Newton94153754 and fixed Lab53ee6b44, the same shared
`NEWTON_NARROW_PHASE_THREADS_X=4`, and the original calibrated capacities.
Old Keyboard MJ comparisons omitted that shared flag and must not be silently
substituted for this new denominator. The small external adapter only matches
the shared flag, records it, and adds its own source guard; original checked
capture, alternating order, fixed imports and timing are retained.
The capacity report records constructor thread counts before the manager
applies the multiplier. The same shared manager path applies it on both arms.

| GPU | FPGS physics median | MJWarp physics median | Physics speedup | Environment wall speedup |
| --- | ---: | ---: | ---: | ---: |
| RTX PRO6000 | 6.914322 ms | 28.227042 ms | 4.082402x | 1.050538x |
| GB300 | 6.068582 ms | 28.517297 ms | 4.699169x | 1.191544x |

All three pairs exceed4x on both GPUs; the lowest RTX pair is4.051701x.
Every state-finite check, twelve child exits and24 capacity boundaries pass.
Session49962 exits0. Original source guards complete, and root separately
checks GPUs idle; this backend driver has no final-idle boolean field.
Directory `/tmp/fpgs-keyboard-linear-state-corrected-backends-paired4k-20260915-01`,
manifest SHA256
`dc35d8ff7510ab1711c1c27b7207a996ddb0dec6d5734f5b00eaf69a3a677c6e`.

The existing MJWarp3.12 correction remains
`newton-bracket-elliptic-huber-v3`, helper SHA256
`7506361f334566dfb51a3d46880bf7a4f919c1fd45513fcfa44f7a4d3e60a6eb`,
guarded against installed solver SHA256
`509b43da297dc6e49efeacdbb59df675213e84afbdb26770471a5ba1855a24b3`.
It changes neither the iteration budget nor gradient tolerance and suppresses
no warning bits. This is a comparison with that documented correction, not
a claim about unmodified upstream MJWarp or general trajectory equivalence.

FPGS capacities remain dense704/contact147456/broad57344; MJWarp remains
njmax320/nconmax32/contact131072/broad45056. Both retain sim_dt0.01,
decimation4, two solver substeps and the original iteration allowances.
No Isaac Lab, parent dependency pointer or installed MJWarp package is edited.

### Physical qualification preserved in the repository

The original actual-constructor test passes on both GPUs at runtime94153754
(sessions37328/89433, exit0). External source SHA256
`f3ccab467cf28dada66b081613b942868039507c2b0f26c72378a2efddcec59e`.
Logs under `/tmp/fpgs-keyboard-linear-native-run-ycj8u078`:
RTX `195287c9fa2beb1c958f68e068b4030415bb5db2511eaddf2f2af99995967c02`;
GB `fe3da794781afc1ad45392d2507f14b21880a7cc9efcebdd3507d57a6c673005`.

It is now one helper and one GPU test in the existing
`newton/tests/test_feather_pgs_prismatic_publication.py`; no new test framework.
The helper and method ASTs are unchanged except portable imports/names.
Test SHA256 `b88973d9abbf498c7856c96015a581ac1c6b887dfe5f00d9e76faf0bbbc23575`.
It covers actual108+6 constructor admission, loaded contacts, non-unit axes,
current forces/targets, passive terms, held/requested mass refresh, notified
inertial/axis/drive changes, the retained angular factor, independent public FK,
and captured partial-reset replay. Original tolerances and eight sweeps remain.

The complete prismatic module plus scalar-mass oracle passes10/10 on each
GPU, no skips (RTX54460,0.997s; GB84595,1.054s; both exit0). Logs under
`/tmp/fpgs-keyboard-linear-integrated-vAKSDPdl`:
RTX `ae9f24ba21d68620c604a644a09ec203fc2506766b25c2f0af5a64669abc4e2e`;
GB `c39c77e8d352cf1feb1a93d62ad0332bf6947f1c614400916826c60ba87e5273`.
The pinned-interpreter CPU module passes seven tests with two explicit CUDA
skips (38125,0.376s). Full precommit remains the final commit gate.

### Exact local reproduction

The commands below record the measured frozen94153754 source state.
For a rerun, use separate pinned worktrees, matching adapter path settings
and a fresh output directory; preserve existing worktrees and artifacts.
The fixed Lab interpreter is mandatory; do not create/sync another project
environment while qualifying this recipe.

Fixed paired adapter:
`/tmp/fpgs-kinetic-fixed-variants-TZRkIPYw/run.py`, SHA256
`c062388f23f82c524690f272fb2d14141ed4126db641de6dcd86f8ade5bfe416`.
Fixed backend adapter:
`/tmp/fpgs-allegro-fixed-backends-EQirljZU/run.py`, SHA256
`514137a2b336d1668f501c39488669da768964bb1b30d9386ff41fc45af43427`.
Shared adapter:
`/tmp/fpgs-keyboard-shared-backends-u1bZoGuc/run.py`, SHA256
`9f2e361692ea9df502b2c5c8f3c555bb146ebb5dff574dd859592308674c81d0`.
These auxiliary adapters and large captures remain local; source/version
and per-child environment records are preserved in the manifests.

Before/after:
```bash
env CUDA_VISIBLE_DEVICES='' GPU='' OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python python /tmp/fpgs-kinetic-fixed-variants-TZRkIPYw/run.py --isaaclab /home/octi/Projects/IsaacLab.wt/contact-reset-20260913 --baseline /home/octi/Projects/newton-fpgs-franka-retained-20260915 --candidate /home/octi/Projects/newton-fpgs-keyboard-linear-state-20260915 --output-dir /tmp/fpgs-keyboard-linear-state-repeated-paired4k-20260915-01 --task keyboard-so101 --gpus 0 1 --rounds 3 --num-envs 4096 --seed 0 --warmup-steps 200 --steps 40 --profile-steps 40 --trace-mode graph --capacity keyboard-so101:fpgs:dense_max_constraints=704 --capacity keyboard-so101:fpgs:rigid_contact_max=147456 --capacity keyboard-so101:fpgs:broad_phase_output_max=57344 --baseline-env FEATHER_PGS_SPARSE_CONTACT_DIRECT=1 --candidate-env FEATHER_PGS_SPARSE_CONTACT_DIRECT=1 --baseline-env FEATHER_PGS_PRISMATIC_PUBLICATION=1 --candidate-env FEATHER_PGS_PRISMATIC_PUBLICATION=1 --baseline-env FEATHER_PGS_COMPACT_CONTACT_BOUNDARY=1 --candidate-env FEATHER_PGS_COMPACT_CONTACT_BOUNDARY=1 --baseline-env FEATHER_PGS_PRISMATIC_LINEAR_STATE=0 --candidate-env FEATHER_PGS_PRISMATIC_LINEAR_STATE=1
```

Matched corrected backend comparison:
```bash
env CUDA_VISIBLE_DEVICES='' GPU='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 uv run --no-project --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python python /tmp/fpgs-keyboard-shared-backends-u1bZoGuc/run.py --isaaclab /home/octi/Projects/IsaacLab.wt/contact-reset-20260913 --fpgs /home/octi/Projects/newton-fpgs-keyboard-linear-state-20260915 --mjwarp /home/octi/Projects/newton-fpgs-keyboard-linear-state-20260915 --output-dir /tmp/fpgs-keyboard-linear-state-corrected-backends-paired4k-20260915-01 --task keyboard-so101 --gpus 0 1 --repeats 3 --num-envs 4096 --warmup-steps 200 --steps 40 --profile-steps 40 --mjwarp-linesearch-fix --capacity keyboard-so101:fpgs:dense_max_constraints=704 --capacity keyboard-so101:fpgs:rigid_contact_max=147456 --capacity keyboard-so101:fpgs:broad_phase_output_max=57344 --capacity keyboard-so101:mjwarp:njmax=320 --capacity keyboard-so101:mjwarp:nconmax=32 --capacity keyboard-so101:mjwarp:rigid_contact_max=131072 --capacity keyboard-so101:mjwarp:broad_phase_output_max=45056 --fpgs-env 0:FEATHER_PGS_SPARSE_CONTACT_DIRECT=1 --fpgs-env 1:FEATHER_PGS_SPARSE_CONTACT_DIRECT=1 --fpgs-env 0:FEATHER_PGS_PRISMATIC_PUBLICATION=1 --fpgs-env 1:FEATHER_PGS_PRISMATIC_PUBLICATION=1 --fpgs-env 0:FEATHER_PGS_COMPACT_CONTACT_BOUNDARY=1 --fpgs-env 1:FEATHER_PGS_COMPACT_CONTACT_BOUNDARY=1 --fpgs-env 0:FEATHER_PGS_PRISMATIC_LINEAR_STATE=1 --fpgs-env 1:FEATHER_PGS_PRISMATIC_LINEAR_STATE=1 --fpgs-env 0:NEWTON_NARROW_PHASE_THREADS_X=4 --fpgs-env 1:NEWTON_NARROW_PHASE_THREADS_X=4
```

Portable qualification from the candidate checkout (repeat with each GPU UUID):
```bash
env CUDA_VISIBLE_DEVICES=GPU-883586b6-3100-0610-81e5-3b4c26f45639 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python python -m unittest newton.tests.test_feather_pgs_prismatic_publication newton.tests.test_feather_pgs_response_diagonal.TestFeatherPGSResponseDiagonal.test_linear_diagonal_mass_ignores_retired_terms_and_holds_masks -v
```

This closes the recorded Keyboard4x physics target, not all representative
tasks. No follow-on micro-tuning is funded. Removing more spatial key
intermediates does not yet establish another0.7ms after required public state.
Franka/Kuka remain separate larger targets; their current contact/GS boundaries
are1.295/3.619ms exclusive, with prior private variants already losing.
A new attempt needs a distinct measured cause, not a recycled failed path.
