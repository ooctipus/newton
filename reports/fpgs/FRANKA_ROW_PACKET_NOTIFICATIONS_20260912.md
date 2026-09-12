# Local row-packet numeric notification correction

Parent runtime: `b13c52bd2afb6ed34064b1177364fb5902f0977e`. Original
prototype and failed wall-time captures are preserved. No numerical kernel,
iteration budget, capacity, task parameter or Isaac Lab source changes.

## Measured failure and source cause

The first 16K paired screen is
`/tmp/fpgs-fourx-franka-row-packets16k-20260912-01`. Both source/capacity
guards passed and the parent completed. Physics improved
6.393721 to 5.788528 ms on RTX and 6.056340 to 5.358144 ms on GB300, but
whole-environment wall time regressed 22.193953 to 61.644536 ms and
22.311060 to 61.565150 ms. These are a single screen, not promotion evidence.

RTX host `sim.step` grew 0.938769 to 40.638519 ms per environment step;
reset/event timings stayed close. The candidate unconditionally invoked nine
model-array topology readbacks, nine cached-plan readbacks, then a Python
per-articulation/per-mimic admission loop on every model notification.

A CPU-only reproduction of the exact unchanged `validate_prefix_topology`
function on 16,384 valid nine-DOF worlds with two mimics took
38.581, 38.662, 39.081, 38.322, 38.539 and 38.855 ms. This excludes device
readback cost and already accounts for the observed extra host time.

The recurring actual source path is the lift reset-mode `variable_gravity`
event (`lift_env_cfg.py:315`), then Newton gravity assignment and
`add_model_change(MODEL_PROPERTIES)` (`events.py:1419`), then the existing
manager notification flush inside `simulate` (`newton_manager.py:1019`).
Joint gain randomization is startup-only in this task: a fix excluding only
joint/material notifications would not address the recurring gravity path.

## Bounded correction

Pass the actual flags into packet validation and bypass only nonzero subsets
of `JOINT_DOF_PROPERTIES`, `BODY_INERTIAL_PROPERTIES`, `SHAPE_PROPERTIES`
and `MODEL_PROPERTIES`. The current `ModelFlags` contract describes numeric
DOF, inertia/material and global parameters (including gravity), not the
structural indices pinned by packet admission.

`BODY_PROPERTIES` can change body flags and retains complete validation.
`JOINT_PROPERTIES`, `CONSTRAINT_PROPERTIES`, all other known flags, unknown
bits, mixed numeric/structural flags, zero and `ALL` also retain exactly the
original signature plus admission checks. Calling validation without an
argument defaults to `ALL`. Structural edits still require reconstruction
and a proper structural notification; mislabeling a topology edit as gravity
is not supported.

All original FK invalidation, kinematic/armature refresh, inertia rebuilding
and mass-update requests still run. The skip affects only private topology
validation, not `notify_model_changed` as a whole.

## Regression and next gate

The new notification tests were run against the unchanged parent first:
all 15 numeric subsets and the gravity case failed on the forbidden topology
readback, while structural rejection controls passed. After the fix the
CPU controls preserve each original refresh call and reject changed actual
CPU body-flag arrays under BODY and ALL. Existing packet physical/native
controls are unchanged. Root owns actual-device regression and the fresh
complete 16K physics plus wall-time gate; this host correction is not yet
claimed as a measured recovery.
