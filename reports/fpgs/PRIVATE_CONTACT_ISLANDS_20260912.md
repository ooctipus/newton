# Private key contacts and compact coupled solve

Status: initial GPU controls pass; early whole timing misses the structural
milestone. Residual preparation/iteration diagnosis authorized at 09:04 UTC,
2026-09-12. No new accepted performance gain or promotion.
This is a structural producer-and-consumer replacement following the diagnosed
loss in [the full-world fusion experiment](FUSED_CONTACT_SOLVE_20260912.md).
It does not promote that slower implementation or reset the performance target.

The branch starts at `34e134b4281febbb7483e3e3e8854f721e7fdeb3`, preserved on
`ooctipus/newton`, and retains original handoff ancestor `31cf87f46`.
Isaac Lab remains unchanged at `1d8feb82d17dbfab8f0772de56f84deae2cb7974`.
E2 `5777558f` remains the accepted comparison implementation.

## Work removed and measured budget

The first full-world fusion paid roughly 1 ms for serial, underfilled world
classification and substantial separate fallback work. Its fused kernel alone
also exceeded the complete replacement budget. Classifier-only or resource-only
tuning cannot meet the first milestone, even under free-fallback assumptions.

Instead, independent scalar key coordinates own their geometry and all eight
GS sweeps privately; coupled coordinates own only a compact residual panel.
This removes intermediate canonical coefficient traffic for admitted contacts
and avoids staging all 384/704 rows and all 114 coordinates for every world.
The existing force converter and Stage7 remain unchanged. Global coefficient
arrays stay allocated for complete fallback; this is not an allocation-saving
claim.

The decisive gate is at least 0.7 ms net whole-physics improvement over E2 on
RTX, with no GB regression. Approximately 1.4 ms would approach the unchanged
fixed Keyboard 2x reference: 11.258526 ms / 2 = 5.629263 ms. Neither a 10%
milestone nor Keyboard-only dispatch establishes the full multi-task 2–4x goal.

Fresh node guidance is a complete boundary budget near 2.030 ms RTX / 2.066 ms
GB, from baseline boundaries 2.730 / 2.766 ms. Charge routing, partitioning,
private preparation/solve, all fallback passes, synchronization and publication.
Only complete-graph timing decides performance; node sums are not automatically
critical-path savings. Never compare to the slower prototype as a new baseline.

## Evidence-backed bounds, not topology guarantees

The saved successful E2 4K refresh/reuse snapshots have 94.79% / 94.97%
independent contacts, at most four contacts per independent key, residual row
maxima 153 / 108 and reserved-key maxima 13 / 10 (RTX / GB). Residual-row p99
is 36 / 39. Topology-derived membership agrees exactly with the original
negative-link scheduler. These are historical same-input observations, not the
current timing window or removable time fractions.

The fixed initial private bounds are:

- Four contacts per independent key.
- 160 residual rows, including the original dense-six limit prefix.
- Six dense coordinates plus at most sixteen reserved scalar coordinates.

The 160-row bound deliberately covers the observed 153-row tail; choosing 128
would send two observed RTX worlds to the expensive complete fallback. This is
one chosen representation, not a tier/block-size search. Any exceeded bound
routes the entire world through original E2 before private output writes.
Global capacities remain 704 rows, 147456 contacts and 57344 broad outputs.

## Ownership and numerical contract

Classify current successful raw reservations from actual body/articulation
response topology, never from a small numerical Jacobian. Reserve every key
touched by a dense or two-key contact. All contacts on reserved coordinates
belong to the coupled residual. Preserve original row order within each
component, duplicate-coordinate merging and current prescribed-motion targets.

Isolated owners include zero-contact keys with active lower/upper limits.
Residual projection and publication must exclude every isolated coordinate;
running the old 114-coordinate writer beside islands would double-solve limits
and overwrite velocities. Cold impulses, omega, CFM, friction delay/disk law,
eight sweeps and original incident predictor velocity remain authoritative.
No cross-trajectory bit identity is required; meaningful convergence and
physical behavior are required.

Refactor the current E2 producer into common geometry/row-packet functions.
Both original publication and private owners use those functions, avoiding a
second handwritten contact law. Retained fallback bias, restitution, schedule
and impulse passes skip all admitted worlds before touching omitted buffers.
Publish canonical counts, row metadata and impulses required by the original
contact-force reader. Debug/warm/postsolve modes requiring full coefficients
are outside admission and retain complete original behavior.

Preserve current raw-contact/factor lifetimes, held-mass refresh/reuse, reset and
model-change notifications, original stream joins and stable graph pointers.
No Isaac Lab or MJWarp bridge change is part of this implementation.

## Work split and early checkpoints

Root owns the solver integration and this report. The solver agent owns common
geometry, partitioning and scalar owners; the dynamics agent owns the compact
residual engine. Tests reuse existing fixtures, the checked paired parent and
the current benchmark driver. No agent launches GPU work independently.

First checkpoint: 09:48 UTC (90 minutes). Report source-ready state, actual
compiler resources and any diagnosed obstacle. Initial integrated paired
512/4K timing is targeted by 10:48 UTC; an extension requires a new recorded
cause and revised measurable hypothesis, not silent framework growth.

Controls cover actual scalar and coupled geometry, limit-only coordinates,
fifth-contact/161st-row/17th-key fallback, empty/growing lists, reset/notify,
original public forces and poisoned omitted coefficients. Time the integrated
implementation early, then require repeated paired timings and full-size
physical/reset validation before promotion. A first loss gets one
cause-directed retry with a quantified route across the gap, not micro-tuning.

The source/data-backed precursor card is
`/tmp/fpgs-private-island-card-dKvGv8/CARD.md`, SHA256
`dc503436c8a317e7ae747cf34982ae0ffab6a273abac09c7522f2b39f76792c7`.
This checked-in decision amends that card's initial 128-row/worst-case-coordinate
proposal to the fixed 160-row/16-reserved-key bounds above.

## Initial implementation and early measurement

The source-ready implementation reached paired GPU testing before the first
90-minute checkpoint. Twelve new CPU controls passed; three CUDA-only controls
were initially skipped on CPU. All **46 actual GPU tests passed on each card**,
without skips, including original compact/coalesced/response/publication tests,
current routing thresholds, loaded scalar friction/limits, mixed complete
fallback, reset/notify and captured-graph replay. The regression-first missing
private flag and common row-packet API failures were observed before integration.
Full pre-commit passes after its two test-only formatting corrections.

An inherited graph-memcheck discrepancy is still open: the prior full-world
fixture reports a pooled-allocation read at the original E2 FK-valid flag;
the matched nonpooled run passes after a test-only event reseed following graph
replays. That does not establish this private implementation's sanitizer
acceptance or prove the pooled report spurious. Production allocation policy
and FK code are unchanged. The detailed retained diagnosis is at commit
`f3b17bc6`, `reports/fpgs/FUSED_CONTACT_SOLVE_20260912.md` on the separate
full-world experiment branch; the private fixture includes the same event reseed.

Offline and loaded resource evidence agrees: partition uses 38/32 registers
and 4676 shared bytes; scalar uses 128/128 and 8448 bytes; residual uses 124/120
and 15568 bytes (RTX/GB). No stack/spill/local-memory allocation was reported.
Scalar packets use one 65-float shared stride per lane, retaining 64 live
fields; the extra word avoids the source-obvious same-bank stride, not a
measured standalone optimization claim. Its CPU-only storage uses malloc/free
because native host thread-local storage failed to link in the installed Warp
JIT; CUDA never uses that host heap path.

The one-round graph screens use identical source-pinned budgets and capacities,
200 warmup, 40 synchronized wall steps and 40 physics graph steps. They are
discovery measurements, not repeated speedups or full loaded physical approval.

| Environments | RTX E2 / private physics ms | RTX ratio | GB E2 / private physics ms | GB ratio |
| --- | ---: | ---: | ---: | ---: |
| 512 | 3.198499 / 3.060291 | 1.045x | 3.580306 / 3.405952 | 1.051x |
| 4096 | 7.280123 / 7.064237 | 1.031x | 6.364789 / 6.466738 | 0.984x |

At 4K, synchronized environment wall times were 30.170008 / 29.626255 ms RTX
and 28.597950 / 27.945852 ms GB. These are not training throughput. All four
FPGS and thirteen collision capacity flags remained clear, source/idle guards
passed, and parents were reaped. Every world was selected in both boundary
snapshots on each card; that is not whole-window admission telemetry.

### Theoretical saving versus measured cost

A fresh three-step node capture charges the exact ten E2 and fourteen private
boundary owners, each called 24 times. The complete boundary is **slower**:
2.723458 to 2.940676 ms RTX and 2.493229 to 2.775832 ms GB. Its own complete
graph spans are 7.190618 to 7.464201 and 6.154613 to 6.441312 ms. Do not mix this
profiled window with the forty-step graph screen to claim a precise saving.

| Private owner, ms/environment step | RTX | GB |
| --- | ---: | ---: |
| Reverse route plus partition | 0.340523 | 0.316115 |
| Independent scalar preparation and solve | 0.778284 | 0.823488 |
| Coupled residual preparation and solve | 1.434167 | 1.183179 |
| All masked fallback preparation, schedule, post passes and solve | 0.315766 | 0.369093 |
| Fallback solve alone, included above | 0.029077 | 0.029088 |

The new residual alone exceeds the original **complete** GS node, which costs
1.059852 / 1.084512 ms. Empty fallback dispatch is not the cause. The original
GS uses 14664 shared bytes versus the residual's 15568: fewer logical rows did
not lower its shared-memory residency ceiling, because private coefficients
were added. Sparse per-world geometry preparation is another source-backed
packing hypothesis, not yet a measured phase attribution.

One bounded phase-clock diagnostic is authorized, with source-ready target
09:19 UTC. It must preserve the actual same-input law and distinguish mapping,
geometry preparation and recurrence/publication without adding barriers or
changing physics. A residual-only correction needs roughly 0.917 ms removal
(64%) on RTX to reach the original E2-minus-0.7-ms milestone. GB requires no
regression, not an invented equal 0.7-ms gain. Routing/layout micro-tuning is not
the selected retry. A loaded-quality adapter is paused until a credible large
correction qualifies; no full-task physical-equivalence claim is made here.

### Reproduction and evidence

Use the accepted handoff's `tools/fpgs_bench/compare_variants.py`, with E2 as
`--baseline`, this checkout as `--candidate`, task `keyboard-so101`, GPUs `0 1`,
the capacities and sample counts above, and `--rounds 1 --trace-mode graph`.
Both arms enable `FEATHER_PGS_SPARSE_CONTACT_DIRECT=1`,
`FEATHER_PGS_PRISMATIC_PUBLICATION=1` and
`FEATHER_PGS_COMPACT_CONTACT_BOUNDARY=1`; only the candidate enables
`FEATHER_PGS_PRIVATE_CONTACT_ISLANDS=1`. The earlier full-world fusion switch
is not enabled. For the diagnostic window use `--profile-steps 3 --trace-mode node`.
Manifests contain complete runnable commands, actual imports, hashes and UUIDs.

Local evidence paths and SHA256:

- `/tmp/fpgs-private-island-gpu-controls-20260912-01/manifest.json`:
  `4cd90390b978a7a0f0334e0efe3c550bc167e22cba3b452db69d692187fa0bf8`.
- `/tmp/fpgs-private-island-keyboard512-screen-20260912-01/manifest.json`:
  `c7c1d00e74a08693293b4fd9010f7b6fba86960b9e09902968724a08826e3800`.
- `/tmp/fpgs-private-island-keyboard4096-screen-20260912-01/manifest.json`:
  `35e9f823863e864918046f6a2ea1502bca39f442848e2d23c0840a1e4def34d0`.
- `/tmp/fpgs-private-island-keyboard4096-nodes-20260912-01/manifest.json`:
  `dbe59c80283554d27c8b87cc14a2a34e2cadee986c2ee6bba5fffbf528997df1`.
- `/tmp/fpgs-private-island-node-audit-1zHbgZHA/evidence.json`:
  `88409c4b85d4a90145d260f780ae5bcc0f76641bd79ea63babf092581e2fb22b`.

The exact-label audit `388338bb0a559400bc1c91dd4ed39687e4642bafd2a2cfab882b532b96ae0238`
retains the existing process correlation and interval algebra; its three CPU
controls pass independently. Node sums are not guaranteed critical-path savings.
No Isaac Lab, MJWarp bridge, production pointer or accepted runtime changed.
