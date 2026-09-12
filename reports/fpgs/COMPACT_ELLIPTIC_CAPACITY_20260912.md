# Compact elliptic capacity: held-out seed 1

Both paired runs pass the **observed capacity and line-search checks** on RTX
PRO 6000 and GB300. This is not physical-equivalence, trajectory-convergence or
speed acceptance. See the [numerical limitations](ELLIPTIC_NUMERICS_20260912.md)
and the [machine-readable audit](COMPACT_ELLIPTIC_CAPACITY_20260912.json).

These are the newly completed `cap112-ncon22` native Allegro and `cap32` external
ANYmal-D runs, not an earlier truncated native-collision control. Only the exact
input files indexed in the JSON are used; no clamped contact count is substituted
for raw pair demand.

## Actual workload and ownership

Each task uses 16,384 worlds, seed 1, 200 warmup steps, `steps=1000`, **repeats=3**,
and 40 graph-profile steps. Every device records **25,920 batched solver calls =
424,673,280 world-solves**. These cumulative counters include warmup/capture/replay;
they are not inferred by multiplying only the nominal measurement window.
ANYmal records 12,960 actual Newton collision calls per device.

Both use the original NEWTON + ELLIPTIC recipe, outer budget 100 and line-search
budget 50. Allegro uses corrected brackets plus stable Huber cost differences;
ANYmal uses corrected brackets and has no friction-loss rows. Neither changes
the cone law, task budget or derivative tolerance. Environment decimation is 4,
with two substeps: Allegro `sim_dt=1/120`, solver `dt=1/240`; ANYmal `sim_dt=.005`,
solver `dt=.0025`.

| Task | Collision owner | Explicit requested caps | Observed allocated layout |
| --- | --- | --- | --- |
| Allegro | Native MJWarp collision | `njmax=112`, `nconmax=22` | 112 rows/world; pooled contacts 360,448; native pair buffers also 360,448 |
| ANYmal-D | Newton collision → MJWarp conversion | `njmax=32`, `nconmax=9`, Newton contacts 147,456, broad output 294,912 | 32 rows/world; pooled MJ and public/pipeline Newton contacts 147,456; broad/split queues 294,912 |

`nconmax` is a per-world API multiplier; the resulting `naconmax` and collision
output queues are **batch-global**. The row cap is per world. In native Allegro,
all actual non-flex contact arrays have first dimension 360,448; `efc_address`
has shape `[360448, 4]`. Its recorded `naccdmax` is also 360,448. Transient native
pair-buffer sizes are inferred from the pinned `create_collision_context` source,
not intercepted allocation objects. This is a capacity/layout audit, not a
deduplicated allocated-byte inventory or separate CCD-buffer watermark study.

For ANYmal, the full explicit input list still contains 2,244,608 pairs. The
294,912 limit applies to its **output**, not input traversal. Actual split-query,
GJK and manifold output capacities are each 294,912, with buffer verification on.

## Measured high waters and remaining reserve

Numbers are RTX / GB300. Reserve means `(capacity / peak demand - 1) * 100`,
not the fraction of capacity unused. The last column uses the larger paired peak.

| Owner / quantity | Capacity | Measured peaks | Worst-device reserve |
| --- | ---: | ---: | ---: |
| Allegro rows/world | 112 | 73 / 82 | 36.59% |
| Allegro raw native `ncollision` | 360,448 | 271,440 / 271,457 | 32.78% |
| Allegro raw native `nacon` | 360,448 | 153,295 / 153,328 | 135.08% |
| ANYmal rows/world | 32 | 24 / 25 | 28.00% |
| ANYmal raw Newton/MJ contacts | 147,456 | 119,614 / 119,615 | 23.28% |
| ANYmal broad output | 294,912 | 218,169 / 217,794 | 35.18% |
| ANYmal GJK candidates / split-query results | 294,912 | 123,006 / 123,108 | 139.56% |
| ANYmal split GJK work | 294,912 | 122,955 / 123,058 | 139.65% |
| ANYmal split manifold work | 294,912 | 175 / 195 | 151,136.92% |

Allegro's native `naconmax` also sizes the transient pair outputs. Consequently,
the apparently large contact-only reserve is **not** evidence that the shared
allocation can safely be reduced to the contact peak: raw pairs are the limiting
observed owner. The native recorder retains unclamped `ncollision`/`nacon` after
every original solve and rejects unsupported sleeping/flex/contact-filter paths.
Its CPU regression uses the original atomic pair emitter to observe demand 7
against capacity 3, without changing the demand to the stored prefix.

ANYmal's shared split queues remain sized from the broad output cap, so their
individual spare percentages do not establish independently reducible storage.
Soft and SDF-SDF demand and capacity are zero in these captured scopes.
The observations validate these allocations for this horizon; they neither prove
minimal allocations nor a guarantee for every future state or task variant.

## Flags and acceptance

On **all four device/task runs**:

- Line-search exhaustion, outer exhaustion, row overflow, pooled-contact
  overflow, other solver overflow and nonfinite-solve counts are all zero.
- Public MJ overflow masks are zero. The capacity-only wrapper excludes LS/outer
  bits from its own verdict, but this audit separately checks their full-run
  counters; no warning is silently excluded from the combined result.
- Independent stored-force-balance diagnostics are finite and in bounds on all
  424,673,280 world-solves per device. This is not an independent constitutive-law
  or trajectory-equivalence test.
- Both boundary states are finite; source, loader and collision-hook ownership
  guards pass. Both parent manifests report clean source trees, exit code 0,
  no cleanup signals, and successful final source/device-idle checks.

Allegro also has zero negative native counters and zero raw-pair/raw-contact
overflow calls. No Newton narrow-phase pass is claimed: it is explicitly not
applicable because the actual collision owner is native MJWarp.

ANYmal has zero overflow calls for every observed collision queue and all 13
sticky narrow-phase flags false: broad, query, GJK/split-GJK/split-manifold,
mesh/triangle/mesh-plane/mesh-mesh/SDF, contacts, and reduction hash load/insert.
Its exact pipeline/public/MJ capacities match the requested allocations.

The maximum observed outer iteration counts are Allegro 34 / 40 and ANYmal
26 / 30, below the unchanged 100 allowance. This does not establish exact
global convergence; the warning-free claim is restricted to these diagnostics.
No instrumented timing values are used or promoted here.

## Evidence identity

Local run roots:

- `/tmp/mj-native-allegro16384-cap112-ncon22-seed1-20260912-01`
- `/tmp/mj-elliptic-anymald16384-cap32-seed1-20260912-01`

The JSON pins all 16 consumed manifests/captures/audits by SHA-256 and records
per-device counts, flags, allocated layouts, reserves and source identity.
This audit also independently rehashed 26 declared helper/source pins. Source
checks for the larger installed package remain the captured diagnostics' guards;
this document does not pretend to rerun those GPU observations.

Both manifests identify clean Newton `5c31305dbba402928c76229670087f6642ef50e6`
and unchanged Isaac Lab `1d8feb82d17dbfab8f0772de56f84deae2cb7974`.
The orchestration owner is `35b19928df46465137ecc2cf92da4e6f8a8bd17b`.
These controls use pinned private diagnostic implementations; they are not
evidence that a later feature-worktree revision was run identically.
The paired physical devices are indexed by exact UUID in the JSON.

The raw artifacts were only read; no source/runtime edits, GPU launches or
commits were performed for this audit. The root capacity report is untouched.
