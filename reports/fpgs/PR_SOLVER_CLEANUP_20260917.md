# Bounded solver cleanup for PR preparation

The preparation tree removes five unpromoted runtime modules:
`body_basis_rows.py`, `sparse_packet_rows.py`, `sparse_paired_gs.py`,
`coupled_contact.py` and `coupled_jacobi.py`. Their five dedicated test modules
and the coupled-only saved probe/profile scripts are removed together.
The original research branches, including checkpoint `301b6688`, preserve
their source and evidence. No experiment has been requalified or optimized.

The corresponding `FEATHER_PGS_BODY_BASIS_ROWS`,
`FEATHER_PGS_SPARSE_PACKETS`, `FEATHER_PGS_SPARSE_PAIRED_GS`,
`FEATHER_PGS_COUPLED_CONTACT` and `FEATHER_PGS_COUPLED_JACOBI` flags now fail
explicitly when set to `1`, before solver construction accesses the model.
Direct sparse-owner construction performs the same check. Unset and `0`
preserve the retained routes. False owner markers remain for existing checked
capture readers; no removed producer, allocation or solver dispatch remains.

All accepted G1 sparse, parallel-limit, level-update, metric, kinetic-state,
chain, spectral and corrected-limit paths remain. The Kuka kinetic module
family, Franka packet/state/compact owners, Allegro keyed rows, Keyboard
linear state and their complete fallback dependencies remain. The separate
contact-block implementation is retained for now, along with the shared
fixtures in its test module.

## Validation

The two portable retired-selector regressions fail before the implementation
and pass afterward. A CUDA-hidden CPU batch discovers 33 tests: 32 pass and
one CUDA-only constructor test skips. It covers those regressions, dense
constructor dispatch, Kuka live bindings/lifecycle and the frozen native and
descriptor oracle. This is correctness/source evidence, not new timing or
GPU physical qualification.

A second batch with all five retired flags explicitly `0` discovers 16 tests:
15 pass and one CUDA-only test skips. It includes the frozen Kuka oracle,
Franka factor native/ABI controls, dense construction and portable sparse
admission controls. The changed Python files pass Ruff lint and formatting
checks; `git diff --check` passes.

## Remaining preparation work

Kuka and Franka still construct specialized kernels through inspected source,
AST transformations, string substitutions and definition hashes. Relevant
owners include `kinetic_source.py`, `kinetic_guard.py`, `kinetic_hybrid.py`,
`kinetic_solve.py`, `kinetic_compact_coupled.py`, `kinetic_current_contact.py`,
`kuka_joint_world.py`, `world_scan_publication.py`,
`franka_kinetic_state.py`, `franka_kinetic_factor.py` and
`franka_compact_workspace.py`. Their redesign is outside this bounded cleanup.
The sparse factory cleanup is tracked separately.

Several retained tests are not portable yet. `test_sparse_factor.py` requires
a cached G1 USD under `/tmp/https`; saved G1 controls import an external replay
helper and captures through `test_sparse_contact_block.py`. Franka packet
tests load `/tmp` captures and paths embedded in their audit JSON; Franka
kinetic-state tests also import a pinned Python reference from another local
checkout. Allegro and Kuka saved-input tests require `/tmp` captures. These
need maintained fixtures or explicit downloadable assets before they can be
ordinary checkout-independent CI gates. Removing or weakening their physical
oracles is not part of this cleanup.

This tree follows the measured handoff lineage. Porting it onto `fpgs-main`
requires preserving that branch's persistent patches, compliance, torsion,
contact filters and reviewed owner fixes; a source snapshot is not a safe
integration of those different contact contracts.
