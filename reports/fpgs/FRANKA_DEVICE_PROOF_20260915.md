# Franka exact device proof: bounded host-regression repair

Pre-code card approved by root at 2026-09-15 02:44 UTC. Isolated branch
`ooctipus/fpgs-franka-device-proof-20260915` starts at frozen paired16 native
`91f505f5dd6d0479a1d1845036f0598a72139703`. Original trees remain unchanged.
First checkpoint 03:20 UTC; no silent extension beyond 03:35 UTC.

## Cause, scope and complete cost

Fixed-root Lab pose writes immediately notify `JOINT_PROPERTIES`. The current
Franka owner copies and hashes 22 full arrays at each notification. Saved512
CPU validation measured 0.488 ms median; exact proof payload is 813,392 bytes.
At the existing 16,384-world shape this is **26,017,792 bytes**. CPU linear
scaling is consistent with the observed approximately16 ms wall regression;
it is not GPU timing or conclusive critical-path attribution.

The unchanged 22 fields are:

- Model: articulation_start, joint_type, joint_parent, joint_child,
  joint_q_start, joint_qd_start, joint_dof_dim, body_flags, joint_axis,
  joint_X_p, joint_X_c.
- Solver: art_to_world, articulation_joint_end, articulation_dof_start,
  articulation_response_dof_count, body_to_articulation,
  _prescribed_articulation, body_response_dof_mask, _free_root_joint_indices.
- Groups: group_to_art[9], group_to_art[6], _crba_source_dof_by_size[9].

Constructor-frozen, exact-sized device clones add **26,017,792 bytes** of
snapshot payload, not maximum-capacity allocations. Unchanged notifications
perform **52,035,584 bytes of device reads**, 22 simple comparison launches,
one mismatch clear and **one four-byte readback** before validity/cache
mutation. No new hashing kernel, pointer-table framework, or per-notification
snapshot copies. Charge launch/Python overhead, device bandwidth and stream
waiting; under1 ms notification cost is plausible, not a measured guarantee.

Use bitwise uint32 views, not float equality, preserving signed-zero and NaN
payload distinctions in the existing fingerprint contract. Retain scalar,
shape/type and all22 changed-input checks, the numeric-flag allowlist, and
existing reconstruction errors. Rebind views when source objects change;
pointer identity may cache a descriptor but never prove unchanged contents.
Warp zero-copy views retain their source allocation; snapshots and live views
must remain owned until the single readback completes. Unsuitable CPU,
non-contiguous or other-device bindings retain the original fingerprint
fallback. No new support for notification capture is claimed.

Do not change the three paired16 physics owners, factor arithmetic, Lab,
notification flags, physics allowances, capacities, or default-off admission.
Legacy row-packet validation remains unchanged and charged in both arms.
Numerical axis/anchor pinning is stronger than the math requires, but removing
it is deliberately deferred: it would expand notification behavior and alone
leaves approximately10 MB of host proof per notification.

## Qualification and falsifiable outcome

Extend the existing test file only: regression-first CPU checks of exact
comparison, descriptor fallback and all existing counterexamples; CUDA checks
of all22 changed fields, last-entry writes, signed-zero changes, unchanged
NaNs, equal-value rebind, wrong shapes, and zero model-array fingerprint calls
on the unchanged fast path. Root exclusively owns actual CUDA tests/timing.
Reuse the existing10 CPU controls, three native physical controls and original
paired whole benchmark. No benchmark framework or new physical test campaign.

Success requires removing at least12 ms of the RTX wall regression without
unexplained physics loss relative to paired16. Final performance comparator
remains accepted `1a9efc33`, not the broken prototype. Promotion requires
repeated good physics and wall results. Paired16 remains below its0.55 ms
physics milestone; this repair does not change that or claim the4x goal.

## CPU/source checkpoint, 02:54 UTC

Regression first: with only the new focused test present, frozen91f505f5
failed `test_device_proof_skips_host_copies_without_relaxing_rejection` with
`AttributeError: 'FrankaKineticState' object has no attribute
'_initialize_device_proof'`. After the fix, the same test passes, including
all22 last-entry mutations, restoration after rejection, equal-value rebinding,
wrong-shape rejection, non-contiguous exact fallback, signed-zero distinction
and unchanged NaN-payload words. Its normal path asserts22 comparison launches,
one status readback and zero calls to the model-array fingerprint routine.

All11 CPU tests pass (eight owner, three retained factor); the CUDA test class
is explicitly skipped with no visible device. Full pre-commit passes. The
comparison kernel compiled and executed on CPU. No actual CUDA or new AOT
campaign was run by this implementation agent. Root owns the four CUDA
selectors and original whole benchmark, so no wall improvement is claimed.

```sh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python \
  python -m unittest tools.fpgs_bench.test_franka_kinetic_state \
  tools.fpgs_bench.test_franka_kinetic_factor
uvx pre-commit run -a
```

Formatted runtime SHA256:
`bcf571dffd90ba057fd54cbbd4d60a9dc85a189870db35f11d0f8a5931ccba93`.
Test SHA256:
`7b0a3a1ad0e73485add0e38dda468c1f040670d7f63ee3bf4d58ac0778f69f50`.
Factor file remains byte-identical to91f505f5, SHA256
`eda0a22bb7328cef4f691970a8551fb5bc98d209c51977dbca60123ee91c9b8e`.
The runtime diff starts after all three paired16 physics factories and changes
only the proof kernel, constructor proof setup and validator helpers/dispatch.
CPU generated repair/finish/predict kernel identities remain unchanged.

## Preserve the original stream envelope before CUDA qualification

Pre-correction commit `04254d65e0794777523d2d97f8fa4a224e914775` was not
GPU-tested or promoted. Independent review found that Warp `array.numpy()`
enters `ScopedStream(device.null_stream)` before reading model data, whereas
the initial device comparison used the caller stream before its final
readback. The fixed Lab same-stream path was ordered, but preserving the
generic existing contract requires the null-stream envelope before comparison.

The minimal correction wraps `_validate_device_proof()` in
`wp.ScopedStream(self._proof_device.null_stream)`. Its existing `sync_enter`
orders after the caller stream and exit restores that caller. This does not
claim synchronization with every independent non-blocking stream and adds no
new global-device synchronization API or additional host payload.

Regression first: the focused test extension fails on04254d65 because the
first comparison has no preceding stream envelope. After correction, all11
CPU controls and full pre-commit pass again. The existing CUDA test helper now
uses an explicit caller stream, requires comparisons on the null stream and
caller restoration, and queues each changed last word on the caller stream
before notification without an intervening readback. Root alone executes it.

Corrected runtime SHA256:
`fa9aa0e1888c1b6bf0e05e73b89e92175ee5fa6de40ee08286e8e952e85c1cfe`.
Corrected test SHA256:
`a19104b5f56e84b92afb55aca324fbae66cdd250c8a7e57f6d0a2f5cb4aa17a7`.
All paired16 physical factories and retained-factor sources remain unchanged.

## Final qualification and closure

Runtime and tests were frozen at
`865dbd6c7c5009b0b1f381c2a62baa55a125ce6e` throughout the following checks.
This closure changes only this report. The runtime/test/factor hashes above
remain the exact measured sources. All original physics/guard/plan callables
are unchanged from91f505f5; an AST comparison also confirms the original
constructor prefix and complete numeric-flag/shape/fallback validator path
are unchanged apart from the new device-proof dispatch.

### Native physical and notification checks

Root ran all four selectors in `TestFrankaKineticStateCUDA`: the new exact
proof/stream test plus the three existing current geometry/held-force,
integration/publication/graph and actual loaded/reset/odd-tail controls.
All four pass on each GPU: 2.439 seconds RTX and2.467 seconds GB300. Root
sessions65231/13136 were reaped with exit0. The proof comparison was the only
new binary; paired16 physical errors and kernels are unchanged. Selected
unchanged normalized maxima are current H9 `1.3396386644807769e-6`, positive
bias `2.084246440695394e-6`, matched8 joint velocity
`1.1976400452964131e-4`, and matched8 body velocity
`1.3237149835899564e-4`. These are the existing tests' measured errors, not
a new end-to-end training/convergence claim.

Local complete native logs:

- `/tmp/fpgs-franka-deviceproof-native-mHEiW6v9/gpu0.log`, SHA256
  `1c5d1db80e9f05b7c59fc9b23cd15bb9aefa0d7f6931009f37b0af51ac9a37b4`.
- `/tmp/fpgs-franka-deviceproof-native-mHEiW6v9/gpu1.log`, SHA256
  `4c80afd3d492f2abd05f4ee6ad6e6d43afbf5abb55dc03c0733be33e004a6c96`.

### Fixed comparison and preserved failed launch

Both arms use fixed Lab `53ee6b44c2334341305dbdf385a3916c6b140799` and its
unchanged root-write notification. The accepted Newton baseline is
`1a9efc33efbc0f23e1e7676a5edded795f224c97`; the candidate is865dbd6c above.
The existing driver source is961b7e2f751bcd1d8b03368e7956b54c81414897,
`compare_variants.py` SHA256
`48406c079588d3088fc25c1cdce2bc2659aaec28539013e5811c9c55378445b5`.
The fixed Lab profiling entry retains SHA256
`586aecb18d95e317991f17e62fb2c9527b2eb76220d76d464bd4c67e9ec8b954`.
Exact commands, imports, GPU UUIDs, source hashes and guards are in each
manifest; the complete recorded command is the reproduction authority.

Recipe: Franka16,384 worlds, seed0,200 warmup/40 synchronized environment
steps/40 graph steps, environment decimation4, sim_dt1/120, two Newton
substeps (solver_dt1/240), matrix-free/immediate contact response, eight PGS
iterations and unchanged parallel streams. Both arms retain rigid contacts
32768, broad-phase output7680, dense192, matrix-free64 and propagation192.
Both enable `FEATHER_PGS_SIMPLE_WORLD_ZERO=1` and
`FEATHER_PGS_LOCAL_ROW_PACKETS=1`; the candidate alone sets the existing
default-off `FEATHER_PGS_FRANKA_KINETIC_STATE=1` (baseline0). No Lab, budget,
capacity, material, collision or solver-law changes were made for timing.

Initial whole01 failed before the candidate ran: RTX baseline returned0;
GB baseline returned139 during setup before timing/CUDA initialization, with
no stack or numerical/capacity failure identified. Root reaped session36190
with exit1; final source and idle guards passed. No performance conclusion is
drawn from that failure. Preserved manifest:
`/tmp/fpgs-franka-device-proof-whole-paired16k-20260915-01/manifest.json`,
SHA256 `811aaba9b831fb3cb7323709240ff65dab9962cc187d89ae8c31096978d78d10`.

Retry02 changed only the output directory. All four runs and eight checked
boundaries passed; root85732 was reaped with exit0. Discovery measurements:

| GPU | Baseline physics ms | Candidate physics ms | Baseline wall ms | Candidate wall ms |
| --- | ---: | ---: | ---: | ---: |
| RTX PRO6000 | 5.468874475 | 5.030387075 | 29.411064551 | 29.365427076 |
| GB300 | 4.989347100 | 4.686589625 | 27.546460851 | 29.650233375 |

The RTX wall regression repair exceeds the12 ms criterion relative to the
earlier broken paired16 prototype's44.762 ms wall, removing about15.4 ms.
That is not an accepted-baseline speedup: retry02 RTX wall is essentially
unchanged against accepted1a9, while GB wall is2.103772524 ms slower.
Preserved retry manifest:
`/tmp/fpgs-franka-device-proof-whole-paired16k-20260915-02/manifest.json`,
SHA256 `1388a9d84c85ea8e7a659a3be50c02b7aaac96c2449b9aa2bdbf809f0d26fa2b`.

### Three-round balanced repeat

The repeat changes only the output directory and rounds1→3. Order is
baseline/candidate, candidate/baseline, baseline/candidate; each arm runs on
both GPUs together. Root reaped session92243 with exit0. All12 runs return0
without cleanup signals; all24 solver/collision boundaries pass, states are
finite, actual budgets/capacities and host call counts match, and final source
and idle guards pass. Large captures remain local.

All times below are milliseconds per batched environment step. Wall delta
is candidate minus baseline, so positive means slower:

| Round | GPU | Baseline physics | Candidate physics | Baseline wall | Candidate wall | Wall delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | RTX PRO6000 | 5.545875575 | 5.007319825 | 29.034897248 | 29.401579627 | +0.366682379 |
| 2 | RTX PRO6000 | 5.577463450 | 5.070682400 | 29.771972052 | 29.140999549 | -0.630972502 |
| 3 | RTX PRO6000 | 5.493414325 | 5.124493950 | 28.787714726 | 28.783619098 | -0.004095628 |
| 1 | GB300 | 4.983208450 | 4.728083100 | 27.333656178 | 28.528939001 | +1.195282824 |
| 2 | GB300 | 4.999355200 | 4.710292000 | 28.657098752 | 28.705157625 | +0.048058873 |
| 3 | GB300 | 4.999479250 | 4.721928800 | 29.309731274 | 29.545154702 | +0.235423428 |

Published ratio-of-medians physics results are RTX5.545875575→5.070682400 ms,
**1.093713851×**, and GB4.999355200→4.721928800 ms, **1.058752771×**.
Median savings are0.475193175 ms RTX and0.277426400 ms GB. **No round on
either GPU reaches the0.55 ms physics milestone.**

Whole-environment results do not establish a gain. RTX varies around parity:
mean29.198194675→29.108732758 ms, while ratio of medians is0.996359003×.
GB is slower in all three paired rounds. Its mean wall time is
**28.433495401→28.926417109 ms, +0.492921708 ms** (about+0.492922 ms;
mean-time throughput ratio0.982959462×). The GB ratio of medians0.998325776×
looks nearly neutral but must not hide the consistently positive paired
losses; median paired loss is0.235423428 ms. Inherited host timers show mean
candidate `event.apply` overhead of0.397329173 ms RTX /0.495626450 ms GB,
but nested timers overlap and are not an exact standalone proof-kernel cost.

Repeat manifest:
`/tmp/fpgs-franka-device-proof-repeat-paired16k-20260915-01/manifest.json`,
SHA256 `01527229b579d1353e5b670340b4dd78307f69f4424d80442aa523b90f8df45d`.

### Decision

**No promotion.** The large prototype host regression is repaired and the
paired16 physics gains repeat, but the physics milestone remains missed and
whole-environment throughput does not improve; GB retains a measured wall
regression. This candidate stays on its separate ooctipus branch and is not
composed into acceptedb1bad06a. It does not establish an additional2–4× gain
or4× across representative tasks. No further packing, benchmark campaign or
micro-fix is part of this closure; original trees, Lab and measured runtime
remain unchanged.
