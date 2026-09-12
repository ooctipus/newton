# Integrated contact preparation and solve experiment

Status at 08:12 UTC: the first integrated mapping is slower on both GPUs;
node attribution identifies its losses. It is not promoted. The additional
2–4x whole-physics goal remains unmet and is not redefined by this experiment.

The source/cost card preceded implementation:
`/tmp/fpgs-e2-attribution-d7H4nC/CARD.md`, SHA256
`dc8d07abbbb9380e31fadc1c14e23f59616b0b34952f1ef63d59fc281e88e5fe`.
The branch starts at handoff `d69d9965`, whose Newton runtime exactly matches the
checked opt-in E2 source `5777558f`. Isaac Lab remains at `1d8feb82` unchanged.

## Removed work and cost hypothesis

The current Keyboard E2 node capture exposes 3.336255 ms RTX / 2.707948 ms GB
of non-overlapped contact preparation, bias/restitution, impulse initialization,
schedule construction and sparse solve. These are measured node intervals, not
guaranteed critical-path savings. The repeated graph baseline is separately
6.985432 ms RTX / 6.147953 ms GB; do not replace it with the slower node-profile
window or count logical row percentages as elapsed-time savings.

One integrated per-world owner will build current contact geometry/response,
prepare bias and schedules, and solve with shared prepared values. Its first
mapping uses 128 cooperative threads per admitted world, private storage for
384 current rows and 32 dense-six contacts, and the complete original 704-row
path for unsupported worlds. Global capacity stays 704. The private panel is
an implementation domain, not permission to discard any row or contact.

The first revision adds an explicit raw-to-row map pass after the original
successful slot allocation. Its complete cost is charged; this avoids changing
the allocator ABI before the first causal measurement. Folding it into the
allocator later is not a standalone optimization target. All fallback launches,
classification, panel construction and canonical publication remain charged.
The design must avoid both naive full-704 shared J/Y storage and serial contact
geometry construction—the latter invalidated the ANYmal lazy-row cost forecast.

The first causal prototype reuses the E2 contact geometry as a cooperative Warp
function, publishes canonical geometry once, then loads the bounded prepared
panel into shared memory for the unchanged generated native recurrence. That
geometry-to-panel global read remains charged; this is not a claim to eliminate
all intermediate traffic. Original prefix clearing and limit-response launches
also remain charged. This controlled seam avoids introducing a second geometry
law before the first complete-graph measurement. The net timing milestone is
unchanged despite these deliberately retained costs.

Milestones, measured with the complete paired graph benchmark:

- First: at least 0.7 ms net RTX reduction (roughly 10% of E2).
- Ambitious: at least 1.4 ms net, toward the unchanged original 2x target.
- Neither milestone alone establishes the full multi-task 2–4x objective.

## Numerical and implementation contract

The default-off flag is `FEATHER_PGS_FUSED_CONTACT_SOLVE=1`. Initial admission
requires the existing small-dense/diagonal topology, cold matrix-free solve,
and no incompatible postsolve, regularization, gradient or debug consumer.
Unsupported topology/options/worlds use the original complete E2 route.

Preserve original eight sweeps, friction law/delay, omega, CFM update semantics,
ordered scalar chains, coupled-contact ordering, limit projections and early
exit. Restitution uses incident predictor velocity, never partially solved
velocity. Retain current material/anchor/prescribed-motion laws, actual contact
identities and force publication. Convergence and physical outputs matter, not
cross-trajectory bit identity.

Current allocation counts—not previous finalized counts—govern ownership.
Every original and new writer must use the same current fallback decision.
Join original factor/collision events, retain held-mass epochs and reset/notify
semantics, and preserve stable graph pointers. Canonical outputs needed by any
retained reader must be published; avoiding intermediate reads is not authority
to leave public or diagnostic state stale.

## Early verification and stop/go

Start: 07:10 UTC; first experiment checkpoint: 08:40 UTC. Do not extend beyond
two hours without a new diagnosed cause and a revised measurable hypothesis.

Use regression-first CPU/native controls for geometry, friction/limits,
prefix/empty/growing lists, 384/385-row and 32/33-dense-contact boundaries,
and complete fallback. Then run root-owned paired 512-world actual-input
controls and an early 4K complete-graph screen. Do not delay that screen to
build an unnecessary per-operation proof framework. Compare original E2
same-input velocity/impulse/force and meaningful physical residuals, including
refresh/reuse and reset transitions, at existing justified tolerances.

Both GPUs retain 704 rows, 147456 public contacts, 57344 broad output,
200 warmup steps, original substeps/iteration allowances and task settings.
The first graph screen uses one round with 40 synchronized/40 profile steps;
promotion requires repeated alternating long runs and actual full-size checks.
All source/UUID/idle/process and four FPGS/thirteen collision capacity gates
remain mandatory. A failed first timing gets one diagnosed causal retry, not a
block-size/register-cap/tier sweep. No unmeasured cross-task gain is implied.

MJWarp root synchronization remains a separate comparison-correctness issue;
this experiment does not repair it or authorize changing the MJWarp bridge.

## First device checkpoint

At 07:41 UTC the paired focused gate ran 31 tests per GPU. Thirty passed on
each, but the new complete-step comparison failed on sixteen non-contact key
coordinates, with a maximum joint-position difference of 0.004068 m. The failed
run is preserved at `/tmp/fpgs-fused-contact-gpu-controls-20260912-01`; both
children were reaped and final source/idle guards passed. No timing was accepted.

Source diagnosis found an unconditional per-row weight store in the fused
preparation code. This admitted nonregularized mode uses the original one-element
`_contact_row_w_dummy`; the original bias kernel stores weights only when the
contact weight is below one. The prototype's write was therefore out of bounds.
The unused descriptor/store was removed, with the same full-step regression
and tolerances retained. The corrected paired gate passed all 31 tests per GPU,
without skips. This includes eager selected/fallback/empty/growing transitions,
partial reset/inertia notification and two captured graphs. It is a focused
implementation gate, not full-task numerical/physical acceptance.

Corrected runtime pins:

- `fused_contact_solve.py`:
  `c4efde55359f8ae5b33292dad59b3ada7b1fcb881bcd34456cb1b040e7bfa501`.
- `compact_contact.py`:
  `fc92c0ca35b4e624142935e43de74871e896d3765e7843f62ffd01b512d94150`.
- `solver_feather_pgs.py`:
  `151caace222cb0f3a81d04f4239d5454128df7345ace2f1fca5b99bc88937ae5`.
- Focused test:
  `879dd3588118e918dee4e87ebedc68571e721778b609bec44c6ef18e2a685077`.
- `/tmp/fpgs-fused-contact-gpu-controls-20260912-02/manifest.json`:
  `ebdbc6f976374d546ddcb9e3d493893742203785940f1df2f6722cc2c0773f46`.

## Early complete-graph measurements

Both screens compare E2 `5777558f` against the corrected prototype with the
unchanged budgets and capacities above. Each is one discovery round, seed 0,
200 warmup, 40 synchronized and 40 profiled steps. All source/idle/finite and
four FPGS/thirteen collision capacity checks passed; these are not repeated
speedup estimates or physical-parity gates.

| Worlds | GPU | E2 physics ms | Prototype physics ms | E2 / prototype |
| --- | --- | ---: | ---: | ---: |
| 512 | RTX | 3.280515 | 4.596288 | 0.714x |
| 512 | GB300 | 3.582637 | 4.949483 | 0.724x |
| 4096 | RTX | 7.379918 | 9.554181 | 0.772x |
| 4096 | GB300 | 6.232308 | 9.182108 | 0.679x |

The 4K synchronized environment times were 32.0111 -> 33.4314 ms RTX and
26.4465 -> 30.8152 ms GB. More than 99% of worlds were admitted in each of
the two current-ownership snapshots. This rules out an all-fallback dispatch
mistake, but admission percentages do not measure fallback cost.

Raw screen manifests:

- `/tmp/fpgs-fused-keyboard512-screen-20260912-01/manifest.json`:
  `5d6f20d2f969a8400577d934334b42dd79321a3dcdbe91c81a6e5ea38acec740`.
- `/tmp/fpgs-fused-keyboard4096-screen-20260912-01/manifest.json`:
  `c1bcee6b2251bf5110b68a5f189176b162fc77103f3596259bceb8181c668026`.

## Diagnosed cost mismatch, not a first-loss rejection

A separate paired 4K node run used three profiled environment steps. The audit
joins actual CUDA correlations to twelve graph roots, uses interval unions and
exclusive busy intervals, and checks all ten baseline/thirteen candidate labels.
The original graph screen remains the timing result; this shorter run provides
causal attribution. Values below are summed kernel milliseconds per environment
step; complete-boundary unions account for the small GB overlap separately.

| Owner | RTX E2 | RTX prototype | GB E2 | GB prototype |
| --- | ---: | ---: | ---: | ---: |
| Complete boundary, interval union | 2.730370 | 4.870768 | 2.765663 | 5.472352 |
| Raw-to-row inversion | absent | 0.035179 | absent | 0.031977 |
| World classifier | absent | 0.988865 | absent | 1.021362 |
| Schedule | 0.271115 | 0.265056 | 0.307968 | 0.272128 |
| Original / masked fallback solve | 1.048470 | 0.808897 | 1.327978 | 0.855339 |
| Integrated admitted owner | absent | 2.201463 | absent | 2.619712 |

The classifier serially follows each world's contact-to-articulation pointer
chains with only sixteen 256-thread CTAs. Its logical small bookkeeping count
did not imply cheap execution. Raw inversion itself is not the main routing
loss. The masked schedule and full-capacity fallback still cost substantially:
skipping most worlds does not remove difficult-world tails or the full grid.
The admitted owner also retains canonical coefficient stores/reloads and a
384-row shared panel; only warp zero executes the recurrence after four warps
cooperatively prepare geometry. These are architectural costs, not evidence
that register-cap or block-size tuning will deliver the target.

Even deleting routing and every masked fallback pass for free would leave
admitted owner plus retained prefix at about 2.274 ms RTX / 2.702 ms GB.
The first milestone requires approximately 2.030 / 2.066 ms from this baseline
boundary. Thus a classifier-only rescue cannot satisfy the milestone. A
credible retry must remove work inside the prepared representation/solve too;
private independent-key chains plus a compact coupled residual panel are being
evaluated as an integrated alternative, not an unchanged scalar fast path beside
the original full solver. No gain is claimed for that unimplemented design.

Attribution pins:

- `/tmp/fpgs-fused-keyboard4096-nodes-20260912-01/manifest.json`:
  `bbd3aaec2af46983cbc3d5b08720c1ffaf4e2e0a7f1a1f8f8e7c91f55273f137`.
- `/tmp/fpgs-fused-node-audit-gYjutA/audit.py`:
  `5ab6f1f45da57f94bbb9379da10ae7bdb7ed9ec0fadebf8375df68fe0196892e`.
- `/tmp/fpgs-fused-node-audit-gYjutA/evidence.json`:
  `1b171aa42a6b619238974ddf774d0ce616f333a46dd85acea0ecbfbf1536c50d`.

## Separate memory-check failure under investigation

The exact full-step fixture was also run under Compute Sanitizer memcheck on
both GPUs. This failed: the first report is a four-byte read in captured
`eval_rigid_fk_id` potentially preceding a sixteen-byte allocation, followed by
sticky launch failures. The ordinary focused gate above passed, but that does
not override this failure. Its origin (fixture, inherited runtime, or prototype)
is not established yet. Do not claim memory-safety acceptance or dismiss it as
an instrumentation artifact. Both child processes were reaped and final
source/idle checks passed. Preserve the failed evidence:
`/tmp/fpgs-fused-contact-memcheck-20260912-01/manifest.json`, SHA256
`a47f24b1ff9ce42ab27eb7fc6ddaf3f96fbf06a86d1c4528d7d81ebed671d90e`.
