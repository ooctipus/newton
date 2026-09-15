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
