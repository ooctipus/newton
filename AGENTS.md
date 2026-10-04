# Newton Development Guidelines

Read and follow the canonical [source code and public API guidelines](CODING_GUIDELINES.rst) before changing or reviewing Newton code. This file contains agent-specific workflow instructions; the linked guide is authoritative for coding and API design.
For reviews, also read and apply the [review guidelines](REVIEW_GUIDELINES.rst) and follow `.claude/skills/code-review-newton/SKILL.md`.

## Dynamic world boundaries

- Generic instance relations, typed storage, byte backing and graph updates belong to the standalone `gpu-components` package. Newton imports its public concept operations and passive records; do not restore `newton.worlds`, generic owner files, aliases or method-forwarding facades.
- MJWarp owns numerical types, work layouts, defaults, alignment, reset and transfer semantics. Newton composes its native operations with generic storage and graph operations; MJWarp never imports GPU Components. Do not restore StepBindings, Data.scratch, a native managed workspace, or a second native field/default catalog.
- The MuJoCo world composition consists of passive records and free operations. Do not add forwarding methods, computed properties, custom allocation constructors or a generic runtime manager. Borrowed population records contain actual descriptors; validate their lifetime before allocating or recording consumers.
- Newton's MuJoCo preparation emits the authoritative numeric model mapping. Tasks select from that mapping and keep their own scalar-joint or control restrictions; they must not reconstruct private actuator encodings or import private native solver APIs.
- MJWarp's array_fields, replace_arrays and metadata_signature own native enumeration, rebinding and metadata validation. Newton consumes declared dimensions and freezes concrete and symbolic descriptors; do not introduce a native schema walker in Newton.
- Native work declarations have one source and drive both ordinary allocation and externally supplied arrays. Names resolve only during preparation. Numerical stages receive native arrays or contexts; backing, readiness, graph bindings and publication belong here.
- Newton supplies concrete capacity Data and authoritative population scalars, and constructs its shallow execution projection with exact Warp CountParameter identities. Equal maxima never establish domain identity. Concrete VMM descriptors still need accessible-range authorization for IO.
- Capture ordinary Warp launches with canonical operand and memory-operation records. This root owns one update table and one binding ledger per population, exact native/application ranges, and the ordered lifecycle program. A callback's native-looking kernel is still application work and needs an exact declaration. Reject callback memory operations until their storage admission is established.
- Keep generic binding validation, byte-layout checks, graph identity, native dependency proofs and retention in GPU Components; do not move an old managed workspace class wholesale into this root. Freeze storage/count/updater relations before recording and invalidate failed captures. Release preparation callbacks after binding.
- Generated-kernel invalidation uses MJWarp's cache operation, never its private cache representation. Execution count meaning comes from canonical operands, independently of memoization or kernel identity.

- Use the cloner's prototype/occurrence/instance relations as the architectural reference: resolve names and paths to integer IDs before entering numeric relation APIs. Do not accept `int | str` identities or repeat symbolic resolution downstream.
- Keep lifetime validity, physical readiness and task participation with their respective owners. Public world records describe numeric relations and the domain publication protocol; allocation and scheduling scratch stays private.
- Physics callbacks consume borrowed population data and use ordinary Warp launches. They must not borrow private field-storage owners or separately notify a graph updater. Numeric launch declarations do not certify memory-operation storage: reject callback fills/copies until their exact operands have an explicit storage-admission path.
- Every kernel emitted inside a physics callback must have an explicit application declaration at that exact record index. Native kernel identity never exempts callback work from this boundary; gate each callback's complete emitted range before classifying the remaining native program.
- Validate prepared native descriptors after the final physics callback at the population composition boundary, before pose recording or publication. Ordinary kinematics receives numerical data only.
- Test inverse mappings, multiplicity, stale generations and allowed writes, and add structural gates for rejected boundary crossings. Matching numerical outputs alone does not establish architectural completeness.
- Name immutable definitions prototypes and mutable instances populations. Distinguish a reusable world ID, a lifetime handle (ID and generation), a storage slot and a request index.
- Generation exhaustion must never make a live world impossible to destroy. Retire a terminal generation without wrapping; permanently exclude its dead identity from reuse.
- Invalid transaction or compaction phases cancel publication and advancement until a fresh batch; later stages must not restore permission from the rejected transaction.
- Fresh W/C/D growth checks every domain before mapping, maps the complete batch before readiness publication, and initializes contact/CCD suffixes before directory admission. Consume ordered directory-service receipts into sticky native health before their buffer can be reused. Historical growth joins readers and publication but only adds mappings; never delegate growth to resizing or release backing. Shrink retains joined maintenance; no task may inspect backing ledgers to choose that path.
- Capture independent prototype initialization and relocation as sibling branches, joining before shared directory publication. Branch callbacks may write only their admitted prototype destinations and acknowledgements; complete global update/error guards remain before any physics branch.
- Keep live counts, protected prefixes, certified ready prefixes and reserved capacities distinct. Readiness can be withdrawn without unmapping memory; mapped bytes do not establish safe access. Order shared-budget reclamation by actual mapped byte ranges, including unpublished headroom retained after clean failed growth, rather than by readiness.
- Keep batch diagnostics on batch results, not membership relations. Expose borrowed directory data from a physics composition root; keep its mutating directory owner private.
- Keep borrowed relation and batch diagnostics readable after quarantine or closure. Read access does not grant submission rights or certify physical readiness.
- Reject Boolean and lossy numeric coercions at integer identity, capacity and byte-count boundaries. Name byte units and mapping directions explicitly.
- Validate launch declarations before emitting kernels. An irreversible preparation failure invalidates the entire graph; another updater must not publish it after a caught error. Keep preparation callbacks out of retained execution state.
- Roll back physical-handle acquisition on every exceptional exit, including interruption. Keep cleanup survivors in the authoritative ledger and keep failed-retirement diagnostics readable for retry.
- Reject unsafe copy aliasing before launch. Exact self-copy can do no work; disjoint packed fields remain valid even when their enclosing address spans overlap.
- Key typed fill patterns by canonical element bytes, never display representations. Equivalent typed values share prepared patterns; distinct bytes remain distinct, including during capture.
- Numerical parity must reject nonfinite state explicitly. Include allocation reuse and poisoned scratch in first-write regressions; matching NaNs are not evidence of correct dynamics.
- Omit a generic derived-property refresh only when the prepared bank proves its edit provenance remains unchanged. Keep that certificate in the existing bank relation; do not add a mirrored cache flag or bypass later generic invalidation and mode transitions. Withdraw it before a world transfer can replace that history, including transfers that subsequently fail.
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
