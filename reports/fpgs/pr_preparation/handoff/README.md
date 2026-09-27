# Original handoff: provenance and historical scope

**Historical material, copied verbatim.** These September 2026 snapshots
describe the original handoff, principally measured on an **RTX 5090**. They
are not current RTX PRO 6000/GB300 results, accepted cleanup benchmarks, or
fresh numerical qualification. Read the [current evidence index](../README.md)
for later results, limitations and preparation status.

Source repository: Isaac Lab; exact source commit:
`2129e5628596f9908cc307482a01700f45b555d7`.
Both files were extracted with `git show COMMIT:PATH` and added without text,
whitespace or line-ending changes. Their source paths are under `reports/fpgs/`.

| Verbatim file / source basename | Bytes | SHA256 |
| --- | ---: | --- |
| [HANDOFF_fpgs_20260911.md](HANDOFF_fpgs_20260911.md) | 12243 | `8adc184ee72f78e854d31c07f6dab56b32189d4d64ec93f616998b0fa2a98506` |
| [fpgs_solver_shape_and_spirit_20260909.md](fpgs_solver_shape_and_spirit_20260909.md) | 112409 | `4ec51734cb58194a157043f908729f6d8e41376c9bba1245231c59160d47b2ab` |

Both are below 500 KB; neither was truncated. The later original Newton
handoff checkpoint is `31cf87f4694f873a027e41e2ca5e9ad441234456`.
The later required Isaac Lab root-reset notification correction is
`53ee6b44c2334341305dbdf385a3916c6b140799`, separately described in the current
index. Copying these documents does not copy or modify either repository's
runtime or change a parent dependency pointer.

## How to interpret the originals

- The originals preserve the author's then-current principles, measurements,
  failed experiments and known solver-parity limitations. Statements such as
  “nothing is committed,” absolute worktree paths and historical hardware
  labels remain deliberately unchanged; they are not claims about this branch.
- The original “same math/order/projections” and numerical-identity constraints
  were later relaxed by the user for the structural study: changed algorithms
  and nonidentical finite iterates were permitted, subject to convergence and
  physical-behavior checks, retained task timestep/substeps/maximum iteration
  allowances, calibrated capacities and complete fallback costs. Those later
  permissions do not retroactively validate a historical experiment.
- Unchanged contact counts, finiteness or cone feasibility alone do not prove
  Coulomb fixed-point correctness or parity with another solver. Preserve the
  originals' explicit fixed-point warnings and the later qualification limits.
- Old performance ratios are not corrected-MJWarp ratios from the later
  six-task comparison. Do not combine hardware, timing scopes or denominators
  across the original and current tables, or label them RL-training throughput.
- Absolute paths and links inside these verbatim archives may not resolve in a
  clean checkout. They are historical provenance, not supported reproduction
  commands. The other Lab validation/reference artifacts remain identified by
  source commit and path in [evidence.json](../evidence.json), not bundled here.

Any correction or interpretation should go in this adjacent note or the
current index, not silently alter these two archival files.
