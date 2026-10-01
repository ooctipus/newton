# Newton Development Guidelines

Read and follow the canonical [source code and public API guidelines](CODING_GUIDELINES.rst) before changing or reviewing Newton code. This file contains agent-specific workflow instructions; the linked guide is authoritative for coding and API design.
For reviews, also read and apply the [review guidelines](REVIEW_GUIDELINES.rst) and follow `.claude/skills/code-review-newton/SKILL.md`.

## Dynamic world boundaries

- Use the cloner's prototype/occurrence/instance relations as the architectural reference: resolve names and paths to integer IDs before entering numeric relation APIs. Do not accept `int | str` identities or repeat symbolic resolution downstream.
- Keep lifetime validity, physical readiness and task participation with their respective owners. Public world records describe numeric relations and the domain publication protocol; allocation and scheduling scratch stays private.
- Native callbacks consume the supported prototype recording interface. They must not borrow private row-storage owners or separately notify a graph updater after launching application kernels.
- Test inverse mappings, multiplicity, stale generations and allowed writes, and add structural gates for rejected boundary crossings. Matching numerical outputs alone does not establish architectural completeness.

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
