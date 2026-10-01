# Newton Development Guidelines

Read and follow the canonical [source code and public API guidelines](CODING_GUIDELINES.rst) before changing or reviewing Newton code. This file contains agent-specific workflow instructions; the linked guide is authoritative for coding and API design.
For reviews, also read and apply the [review guidelines](REVIEW_GUIDELINES.rst) and follow `.claude/skills/code-review-newton/SKILL.md`.

## Dynamic world boundaries

- Generic instance relations, typed storage, byte backing and graph updates belong to the standalone `gpu-components` package. Newton imports its public concept operations and passive records; do not restore `newton.worlds`, generic owner files, aliases or method-forwarding facades.
- Native MJWarp stage binding has one `StepBindings` record shared with the workspace. Do not duplicate its count declarations, binding/operation lists or failure latch in Newton. Native population recording composes physics and application callbacks; the MJWarp stage operation owns launch binding.
- Application recording delegates atomically to `mujoco_warp.launch_step_kernel`; never reconstruct a raw Warp launch followed by separate binding. Native workspaces are passive scratch records, with validation and reporting at MJWarp's operation boundary.

- Use the cloner's prototype/occurrence/instance relations as the architectural reference: resolve names and paths to integer IDs before entering numeric relation APIs. Do not accept `int | str` identities or repeat symbolic resolution downstream.
- Keep lifetime validity, physical readiness and task participation with their respective owners. Public world records describe numeric relations and the domain publication protocol; allocation and scheduling scratch stays private.
- Physics callbacks consume the supported population recording interface. They must not borrow private field-storage owners or separately notify a graph updater after launching application kernels.
- Test inverse mappings, multiplicity, stale generations and allowed writes, and add structural gates for rejected boundary crossings. Matching numerical outputs alone does not establish architectural completeness.
- Name immutable definitions prototypes and mutable instances populations. Distinguish a reusable world ID, a lifetime handle (ID and generation), a storage slot and a request index.
- Generation exhaustion must never make a live world impossible to destroy. Retire a terminal generation without wrapping; permanently exclude its dead identity from reuse.
- Invalid transaction or compaction phases cancel publication and advancement until a fresh batch; later stages must not restore permission from the rejected transaction.
- Keep live counts, protected prefixes, certified ready prefixes and reserved capacities distinct. Readiness can be withdrawn without unmapping memory; mapped bytes do not establish safe access.
- Keep batch diagnostics on batch results, not membership relations. Expose borrowed directory data from a physics composition root; keep its mutating directory owner private.
- Keep borrowed relation and batch diagnostics readable after quarantine or closure. Read access does not grant submission rights or certify physical readiness.
- Reject Boolean and lossy numeric coercions at integer identity, capacity and byte-count boundaries. Name byte units and mapping directions explicitly.
- Validate launch declarations before emitting kernels. An irreversible preparation failure invalidates the entire graph; another updater must not publish it after a caught error. Keep preparation callbacks out of retained execution state.
- Roll back physical-handle acquisition on every exceptional exit, including interruption. Keep cleanup survivors in the authoritative ledger and keep failed-retirement diagnostics readable for retry.
- Reject unsafe copy aliasing before launch. Exact self-copy can do no work; disjoint packed fields remain valid even when their enclosing address spans overlap.
- Key typed fill patterns by canonical element bytes, never display representations. Equivalent typed values share prepared patterns; distinct bytes remain distinct, including during capture.
- Numerical parity must reject nonfinite state explicitly. Include allocation reuse and poisoned scratch in first-write regressions; matching NaNs are not evidence of correct dynamics.
- Keep test-only fault injection out of public constructors. Patch private driver loaders in tests instead of exporting a fake-driver callback ABI.
- Document every supported public owner method and property so the API documentation filter cannot silently hide operations; gate this without asserting incidental prose or inherited members.
- Verify decorated Warp records in rendered Sphinx pages: every field needs its source description and type, and compiler metadata must not appear as public record members.
- A homogeneous world prototype must broadcast each nonempty batched Model parameter from one row. Compaction relocates live Data, not immutable Model parameters; reject slot-dependent model batches before allocation.
- After graph retirement and consumer joins, invalidate borrowed native views and detach owned program metadata; invalid views must not prolong update-table buffers. Preserve only documented diagnostics and retryable resource ledgers.

## Workflow

- Create a feature branch on your fork before committing—never commit directly to `main`. Give the pull request a concise, descriptive title.
- Use imperative mood in commit messages ("Fix X", not "Fixed X"), with a roughly 50-character subject and a body wrapped at 72 characters that explains what and why.
- Verify regression tests fail without the fix before committing.

Run `uvx pre-commit run -a` to lint and format before committing. Use `uv` for all commands; fall back to `venv` or `conda` only if `uv` is unavailable.

```bash
# Examples
uv sync --extra examples
uv run -m newton.examples basic_pendulum
```

## Tests

```bash
uv run --extra dev -m newton.tests
uv run --extra dev -m newton.tests -k test_viewer_log_shapes           # specific test
uv run --extra dev -m newton.tests -k test_basic.example_basic_shapes  # example test
uv run --extra dev --extra torch-cu12 -m newton.tests                  # with PyTorch
```

```bash
# Benchmarks
uvx --with virtualenv asv run --launch-method spawn main^!
```

## PR Instructions

- If opening a pull request on GitHub, use the template in `.github/PULL_REQUEST_TEMPLATE.md`.
- Follow `changelog/README.md`: add a Towncrier fragment for user-facing changes instead of editing `CHANGELOG.md` directly. A `.skip` reason is optional for changes without user-facing impact.
- Preview fragments with `uvx --from towncrier==25.8.0 towncrier build --draft --version X.Y.Z --date YYYY-MM-DD`.

## Examples

- Follow the `Example` class format.
  - Implement `test_final()` or `test_post_step()`; an example may implement both.
  - In test mode, `test_post_step()` runs after each simulation step and `test_final()` runs after the example completes.
- Register the example in `README.md` with its `python -m newton.examples <name>` command and a 320x320 JPEG screenshot.
