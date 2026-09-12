# Kuka joint-world predictor, ZERO screen and active response

This is a **default-off, unqualified structural experiment**, based on accepted combined Newton `064ec8ac455fc4cde557a3b54a1a62624cf56441`. No new performance gain, full-task numerical acceptance, or 4× completion is claimed. Isaac Lab source, task configuration, capacities, timestep, substeps and iteration allowances remain unchanged.

`FEATHER_PGS_KUKA_JOINT_WORLD=1` selects the experiment only for the admitted Kuka 23+free6/prescribed6 layout. The existing accepted ZERO, independent-component and paired/general-overlap switches remain enabled in the comparison. Unsupported construction or per-call source ownership retains the existing implementation.

## Complete cost target, not a component-speedup claim

The source-checked current Kuka boundary from current-tau input through predictor/ZERO/rows/MF/8GS to generalized q/qd is **6.031064 ms RTX exclusive busy**, 6.166819 ms union. Against 13.973622 ms whole physics, its replacement must be **≤3.236340 ms** for a modelled 20% whole-step gain. These figures come from the retained joint-mapping card and correlated node evidence; overlapping node sums are not additive savings.

The larger mapping's hypothesis ledger is bucket/queues 0.18, light predictor/ZERO/selected integration 0.65, active geometry/response/MF/counts 1.10, retained concurrent GS 1.127, final active publication/joins 0.15 = 3.207 ms. These are targets, not measurements. In particular the old approximately 0.800 ms whitening arithmetic remains real work, and the initial slice below retains more row preparation than the complete proposal. It must be timed as integrated, not credited with unimplemented direct geometry or mixed-world conversion removal.

Current FK/RNEA/tau, held mass/Cholesky/inverse-lower production and final public FK are outside the removed ownership. Neither current tau's 1.331862 ms duration (0.150069 exclusive) nor final public FK's approximately 1.438347 ms duration is credited as saved. Even removing the complete boundary cannot alone achieve the fixed 4×-MJ objective.

## What the first implemented slice changes

| Boundary | First-slice implementation | Work explicitly retained or charged |
|---|---|---|
| Raw ownership | Count/scan/scatter/validate original raw IDs by world, with exact existing contact capacity | All raw geometry and IDs retained; all bucketing, scan, clears and joins charged |
| Predictor | One 32-lane world owner computes held `Wᵀ(Wτ)`, where `W=L⁻¹`, then original transported predictor for all 35 global DOFs | Current tau and original held-factor epoch; original-L solve on invalid inverse action |
| ZERO decision | Same current normal, restitution, depenetration, finite-margin and signed-limit laws, with body/lane current twists and bounded parent-prefix sums | Every raw ID in an arbitrary-length segment; original raw-ID anchor neighborhood; conservative unresolved fallback |
| Selected generalized output | Original qdd/free-root transport/q/qd equations for ZERO worlds inside the light owner | No early public body FK, inertia/cache publication, force or sensor mutation |
| Active response | Compact unresolved prefix is checked before factor loads; original four row-warps compute direct Z for MF-zero worlds | Current grouped J and canonical J/Y/diagonal-CFM publication; mixed worlds still compute physical Y |
| Active solve/publication | Original paired/general independent overlap, followed by original generalized equations for unresolved worlds | Original 8-GS law, friction coupling, mixed qualifier, current MF inertia, all-world public FK/cache finalization and force/sensors |

This first slice **does not** implement direct raw/prefix-to-Z geometry, remove grouped Jacobians, move the mixed qualifier before MF preparation, or replace physical-Y→Z qualification in mixed worlds. The ordinary dense/MF allocators, contact metadata/prelude, masked Jacobian builder, bias/restitution, MF response and original `v_out←v_hat` copy remain. Active endpoint-twist storage is additional current data, not evidence that its later consumers have been fused. The original contact path/slot and solved impulse identity remain authoritative for public forces.

The active response source is recovered from the original paired factory. Removing only the new empty-row guard and serial CPU-control branch reproduces the CUDA numerical snippet byte-for-byte. For MF-zero worlds it publishes original Z; mixed worlds retain original physical Y before the unchanged independent-component qualifier. It avoids 565 factor-float loads per resolved/empty world, but cache effects and fixed dispatch slots prevent translating skipped bytes into a timing claim.

## Ownership and failure behavior

Static admission checks exactly three articulations and 32 bodies per world, a parent-first 30-body/23-DOF primary, responsive free6 and prescribed6 roots, all 35 disjoint physical DOFs, and equality between original response masks and ancestor sets. Response width 29 is not the global publication width 35. No per-world raw-contact limit is introduced.

Only integer raw-ID indexing overlaps mass work on a separate stream. The light owner waits for its current completed bucket and current held factors/tau. Original q/qd output must be storage-disjoint from every still-read current-state field. Public body poses, current S/origins, inertia and FK-cache state remain untouched until the original late all-world publication after the paired/general join. Reset/notification paths join outstanding bucket work; structural changes require revalidation or reconstruction. The controller's recorded-current-event guard prevents waiting on an unrecorded prior call.

Malformed or ambiguous raw ownership sets a private invalid flag. The current implementation then uses the original-L predictor, makes every world unresolved and retains complete original row/solve work; it does not claim to rerun the original ZERO classifier. A private invalid flag neither clears nor replaces existing persistent capacity failures. Raw overflow or other rejected storage remains rejected by the checked runtime.

Public contacts remain **4,000,000**, dense rows **192/world**, MF rows **64/world**, propagation rows **192/world**. Kuka retains sim_dt **1/120 s**, environment decimation **4**, Newton substeps **2**, and the original maximum **8 GS sweeps** with current friction and augmented drives. No extra sweeps or altered timestep are allowed to make the candidate pass.

The raw-ID bucket adds 16,131,084 bytes at the 16K/public-4M recipe, including its two W+1 integer arrays and flag. Active endpoint twists add 12 MiB; immutable topology maps and active-list/decision arrays are also new allocations. All are preallocated before graph capture. This is not a contact-capacity reduction and no added storage is hidden from the cost gate.

## Validation checkpoint and next gate

Active-response module controls passed on CPU (five tests; its CUDA test skipped there), followed by **all six tests on each actual GPU**, including original-kernel comparison and captured count/list transitions. Root reaped both children and verified idle. Tests cover direct Z versus physical Y, nonidentity groups/world maps and both component offsets, held-factor refresh, CFM, current 0→192→1→0→8 counts, untouched poisoned tails and input ownership. Offline sm120/sm103 compilation uses 71/72 registers, 2,772 shared bytes and zero stack/spills; these are compiler resources, not timing or occupancy.

The complete light-owner/controller and integrated current-input physical gates are separate and are not implied by that component pass. Required controls include original held-L/FP64 predictor comparison; parent-prefix/current-S closure; actual raw normals and near-margin decisions; invalid-bucket/uncertain-action fallback; 35-DOF transport/publication; mixed MF and prescribed endpoints; reset/notify and actual eager/graph lifecycle; and force/constraint residual distributions at unchanged budgets. Numerical/physical behavior is the gate, not identical trajectories or a stronger per-operation bit-identity requirement.

Root subsequently completed **86 tests on each actual GPU**, all passing, including the 45 retained combined-path controls, ten original ZERO controls and 31 new raw/light/response/publication controls. Both children were reaped and both GPUs checked idle. Logs: `/tmp/fpgs-kuka-joint-native-{rtx,gb}-20260912-01.log`. These include all four saved 512-world inputs, empty/invalid/regrown raw prefixes, active-owner withdrawal/reentry, two graph objects, unchanged excluded publication outputs, numeric reset notifications and injected raw-stream failure recovery. They do not replace current-rollout convergence or whole-physics timing. An additional regression-first CPU constructor test checks that unsupported topology retains the original implementation.

The actual light kernel matches original ZERO selections in all four saved inputs (422/415/431/429 worlds). Maximum predictor-velocity difference is 1.1920929e-7. Offline compilation reports 70/72 registers, 2,304 shared bytes, 96 stack bytes and no register spills; stack storage is not claimed absent. Runtime SHA256: `fd2b0c47f625c27e8a07d23f67b099a8bf19f90cc9c750da07582b6786e626ab`.

Use benchmark tools `32164ded363de6d79e65371f8c89ffad411e8895`, which check the actual ZERO/active/predictor-fallback partition and raw invalid flag only at the two existing untimed metadata boundaries. Its 74 CPU tests and full pre-commit checks pass. This reports recent buffer contents, not whole-run admission; no timed kernel or host readback is added.

After complete source/numerical controls, root owns one integrated 512→16K timing screen charging every bucket/light/row/fallback/solve/publication pass. Repeated balanced whole timing and current-input tail validation are required before promotion. A material miss of the ≤3.236 ms complete replacement target requires measured dominant-loss attribution before any corrective experiment, not an independent tile/grid tuning sweep.

## Why this is not the rejected early-publication mapping

The separate early-K candidate `fb32ce78d1e0d4242d274f67e492d8ed7478af2e` regressed **13.938037→14.946284 ms RTX**, **14.121523→15.030761 ms GB**. It overlapped a second public FK/finalizer owner with row preparation. The final join had zero exposed tail in all 24 measured subsolves per card; early selected FK alone was slower than original all-world FK, and allocator/prelude/J durations inflated by approximately 0.827 ms on each card. It is not accepted.

This experiment leaves public FK/cache publication late, and instead replaces the predictor/ZERO/generalized-output ownership. It does not assume the earlier 84–85% two-call cohort persists: early-publication timing boundaries were only approximately 76–77%. Neither a high cohort fraction nor source-level work reduction establishes a whole-step gain.

## Retained evidence

- Complete mapping card: `/tmp/fpgs-kuka-joint-mapping-k0cb19eU/CARD.md`, SHA256 `28d13d773bc4c70441599a4d5c93b8cc273011ae0c8f65852bed5f50b93ea7ec`.
- Active-response source/ABI/tests/offline resources: `/tmp/fpgs-kuka-active-response-sE9Nm4eu/READY.md`, SHA256 `649da6d4f6158c5a503080dacadc8478be2299a481527b5b407645e9ecae2b61`.
- Active-response runtime/test SHA256: `f9f33245ce5e323ad2c44b6429a1c89033ca4a22a3efd4abb0fc774ba5ae6e19` / `bfe99b2d19da5a549a2f96b5fd2b194a8b027417e92a9a578a911b7281477af1`.
- Failed early-publication diagnosis: `/tmp/fpgs-early-kuka-node-audit-soIL49Ox/FINDINGS.md`, SHA256 `f00b876bd4977a58ef89dd7771df30bbc50338de6372916431cef9fbc1bcd019`; evidence JSON `1db6f0ff49b67b3415c63695b031e6d9700514d9c1b779b3ded5e196e474659d`.
- Failed early-publication run: `/tmp/fpgs-fourx-early-kuka16k-20260912-01`, manifest SHA256 `015fc6dad4501f6f5e03e29284752e0c64effa7685c86241ada5f6948dc6fdf7`.

Large raw captures remain local. This report records the initial implementation checkpoint, not a completed whole-experiment acceptance decision.
