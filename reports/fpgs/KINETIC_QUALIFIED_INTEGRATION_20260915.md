# Compose qualified G1 and Kuka kinetic owners

This isolated `ooctipus/fpgs-kinetic-qualified-20260915` worktree composes the
already measured G1 current/next dynamics owner with Kuka's current-contact
producer. It excludes the unqualified Franka kinetic-state and Kuka private
response experiments. No Isaac Lab source or dependency pointer changes.

## Exact lineage and retained implementation

- First parent: G1 `8cd568868bb51f12b2f4700006ed71cc06a07c2e`.
- Second parent: Kuka `629dedb057cc34cb05139b98d6dece5609ae306e`.
- Common qualified base: `1a9efc33efbc0f23e1e7676a5edded795f224c97`.
- Inherited handoff ancestor: `31cf87f4694f873a027e41e2ca5e9ad441234456`.

The real merge is conflict-free. Relative to G1 it adds only Kuka's existing
module, binding hook, test module and report, plus this integration note.
G1's two runtime modules, solver hooks, sparse factor, teardown test and
physical test remain byte-identical to the first parent. Kuka's runtime
module, bindings and current-contact test remain byte-identical to the second
parent. No frozen source oracle, tolerance, capacity or iteration budget is
regenerated. All original worktrees and their local captures are preserved.

Retained SHA256 values for the runtime composition:

| File in `newton/_src/solvers/feather_pgs` | SHA256 |
| --- | --- |
| `g1_kinetic_owner.py` | `3696335a856642d8af4aac0bb3e3cd9f8ecc91bc85d04fce45ee6d46b7d1a348` |
| `g1_kinetic_state.py` | `7a9c7fb0603706e71eb60a1e998e0861f5ed05123bcf06983643501c42047f04` |
| `solver_feather_pgs.py` | `106ee29242ca31f42559f77929d3b9b1a9e92707a66c79ce0934fe866239b83d` |
| `sparse_factor.py` | `0ae4b1ecf495b69412e41ae11f7c377a4da47c13bd8a09214750ee47e7ac5650` |
| `kinetic_current_contact.py` | `17ad7f0827da875ae4ba77b0eeb4653c0c323077c4feaf1260860ac3bbd81196` |
| `kinetic_live_bindings.py` | `8c75cb5c64b97665db9bdc3d619d40b20beff3fe318acf3d1f00ef8e0cb74bb5` |

## Checks and reproduction

Existing CPU controls execute 106 passes in three batches: 51 passes plus one
explicit CUDA skip for G1 geometry/admission/mass requests, teardown, Kuka
current rows/default-off behavior, bindings, lifecycle and frozen native
source; 22 binding/lifecycle passes repeated with current-contact enabled;
and 33 mapping/notification/publication passes plus one CUDA skip. All three
processes exit zero. CUDA is hidden throughout; initialization's unavailable
device message and inherited target-layout deprecation are not GPU tests.
There are no new tests or benchmark tools in this integration.
Full `uvx pre-commit run --all-files` passes without modifying runtime files;
both staged and unstaged whitespace checks pass.

Use the fixed Lab interpreter with `CUDA_VISIBLE_DEVICES=''` and
`OPENBLAS_NUM_THREADS=1` to run the existing `unittest` modules:

```text
tools.fpgs_bench.test_g1_kinetic_state.TestG1KineticStateCPU
newton.tests.test_feather_pgs_teardown
tools.fpgs_bench.test_kinetic_current_contact
tools.fpgs_bench.test_kinetic_live_bindings
tools.fpgs_bench.test_kinetic_live_owner
tools.fpgs_bench.test_kinetic_native_port
tools.fpgs_bench.test_model_notification_snapshot
tools.fpgs_bench.test_franka_packet_notifications
tools.fpgs_bench.test_kuka_joint_owner
tools.fpgs_bench.test_world_scan_owner
```

Repeat the binding/lifecycle pair with `FEATHER_PGS_KUKA_CURRENT_CONTACT=1`.
Both new flags remain default off. For performance reproduction, retain the
complete respective qualified recipes and fixed Lab
`53ee6b44c2334341305dbdf385a3916c6b140799`: add
`FEATHER_PGS_G1_KINETIC_STATE=1` only for the admitted G1 path, or
`FEATHER_PGS_KUKA_CURRENT_CONTACT=1` alongside the existing Kuka kinetic-world
recipe. See the unchanged [G1 report](G1_KINETIC_STATE_20260914.md) and
[Kuka report](KUKA_CURRENT_CONTACT_20260915.md) for exact flags, capacities,
physical evidence, three-round measurements and original capture manifests.

This source composition is not a fresh CUDA qualification, all-task timing,
MJWarp denominator or achievement of the all-task four-times target. Root
owns subsequent GPU checks, review and publication. No parent pointer is
advanced by preparing this merge.
