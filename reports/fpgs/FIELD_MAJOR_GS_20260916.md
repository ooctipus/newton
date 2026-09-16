# Field-major world-lane sparse metric GS component screen

## Scope and falsification target

This is a research-only component experiment, not a production dispatch
change or a whole-physics speed claim. The accepted current-row/held-W
representation and original eight-sweep scalar-normal/metric-disk law are
retained. Root funds one fixed block32 mapping: one world per lane,32
worlds per CTA. No block/grid search, changed iteration budget, root
tolerance, contact ordering or numerical routing is included.

The intended retirement is the original solve's per-row warp reductions,
broadcasts and synchronization, while allowing all32 lanes to execute
independent tangent roots. Each lane owns its world's43-coordinate update
in shared `du[43][32]`, with private global field-major lambda and lazy
cross coefficients. Four scalar bitmap words retain the original
row/32 readiness lifetime. Metadata and Z are transposed only as explicit
component inputs; their preparation is excluded from timing, not claimed
free in a complete implementation.

At16K worlds the fixed main launch has512 CTAs/512 warps, versus actual
RTX188 and GB152 SMs. This leaves only about2.7/3.4 total warps per SM even
before imbalance: root divergence and poor latency hiding are serious
falsification risks. Existing adjacent-world counts imply about2.08x
padded row visits (median world18 rows versus median warp maximum42).
A uniform row-slot induction variable with per-lane `skip_until` preserves
triplet transaction order without allowing accepted triplets to drift to
different row indices. It does not remove padded inactive visits.

The solver-only target is at least about1.3ms equivalent environment-step
saving, leaving at least1ms after a future, separately funded complete
producer. Packing is excluded, so even a successful screen is only a lower
bound on complete cost. There is no whole-runtime or MJWarp comparison in
this experiment.

## ABI and charged publication

New standalone module: `sparse_field_major.py`. It has no environment flag,
`install()` function, constructor hook or production owner mutation.

- `pack_rows(data,rhs,diagonal,row_type,parent,mu)` creates
  `FieldMajorRows`: Z[18,100,world], row fields[100,world], lambda/cross
  scratch[100,world] and final du[43,world]. This explicit host conversion
  is outside the component timer.
- `get_solve_kernel()`: key `sparse_field_major_metric43_s18_c100`;
  ordinary launch dim=groups, block32; arguments
  `(plan,data,packed,counts,canonical_impulses,iterations,omega,friction_start)`.
- `get_decode_kernel()`: key `sparse_field_major_decode43_s18_c100`;
  tiled launch dim=groups, block32; arguments
  `(plan,data,packed,counts,canonical_impulses,vhat,vout)`.

Both kernels must be included in every candidate timing. The second owner
uses the original cooperative group-indexed W434 decode, writes complete43
velocity and exports active lambda into the canonical world-major array.
W is never transposed outside timing. The extra du write/read costs about
2.8MB each at16K worlds, plus its extra launch and canonical publication;
these are charged. No dense Gram or other world-sized matrix is produced.

Every launch loads incoming lambda and initializes du to zero. It does not
reapply an incoming impulse. The original delayed-friction reset behavior
and exact-unchanged exit remain intact. Both kernels guard group overlaunch,
valid/status and capacity before consuming private data; main validation
sets the original bad-row status and decoder vetoes stale private outputs.
Only active canonical impulses are overwritten; inactive tails remain
untouched. Group/art/world maps remain explicit and independent.

The original16-probe root, eigenvalue floor, repair and local KKT guard are
extracted directly from `sparse_metric_tangents.get_fragments(100)`; only
field accesses change. Normal and tangent trials remain transactional.
Positive-radius tangents use old du plus cached normal cross response
exactly once; zero-radius transactions remove old tangents; invalid or
rejected trials use the original scalar/sibling sequence. CFM remains in
the denominator/proximal delta, never added as physical CFM*lambda.

Sequential support accumulation replaces warp-tree summation. Floating
point association/FMA differences are explicit and require the existing
physical/backward and output gates; bit-identical lambda is not assumed.
The CPU body uses finite-equivalent local math helpers for CUDA/libm names
missing from Warp CPU headers; native CUDA math expressions are unchanged.

## Regression, CPU and offline checkpoint

The missing-module regression failed before implementation. Two CPU
controls pass, including execution of the complete scalar native CPU body
and decoder across35 mapped worlds. Cases cover support17/18, different
sibling support, singular metric fallback, open disks with old tangents,
nonzero incoming impulses, scalar relaxation, delayed friction, zero
iterations, empty worlds, overlaunch, preserved inactive tails and stale
decode veto after invalid/status/overflow/malformed rows. The independent
metric reference and actual held sparse factor check velocity/impulse
behavior, not only source text. CUDA selectors additionally reuse the
original16 captured current/held physical cases and continuing graph guards.

Independent read-only review found no concrete transaction, mapping or
shared-memory blocker. CPU is not native qualification. Root alone owns
GPU launches and matched actual-input timing.

| Hidden AOT, fixed block32 | SM120 | SM100 |
| --- | ---: | ---: |
| Main registers | 80 | 80 |
| Main shared bytes | 5,632 | 5,632 |
| Decode registers | 40 | 32 |
| Decode shared bytes | 128 | 128 |
| Stack / spill loads / spill stores, both kernels | 0 / 0 / 0 | 0 / 0 / 0 |
| Reported barriers, both kernels | 0 | 0 |

Source-matched compiler artifacts:
`/tmp/fpgs-field-major-offline-27u7sUkV/offline01/report.json`.
These resource numbers do not establish occupancy, latency or a speed gain.

## Existing full-population inputs

Root's original private cold-eight replay exactly reproduced all retained
velocity and active impulses on both cards, leaving live arrays unchanged.
Snapshots preserve original arrays, not pretransposed fields:
`/tmp/fpgs-g1-full-solve-input-paired16k-20260916-01/gpu{0,1}/inputs.npz`.
RTX SHA256 `9c1c81c665e38a170e85836fd534dc347598f5af8fa2be0e58fb23fba80ae223`;
GB `97dca4278c7f91a2c15fc43319f0fc29d4036036b9940b7ae781e8a97911ba25`.
Original source is qualified-chain `ef9481`; iterations8, friction_start0,
explicit cold initial impulses and all plan fields are retained. No new
dataset selection or contact removal is used. Native result and complete
component cost remain pending at this checkpoint.

## Native and valid component closure

Frozen module SHA256:
`c0f4557f5238f79ead9d317ee8b817e6ecdc03d8cc1084f4051079a9ef795856`.
Basic test SHA256:
`01737e1b9f4802647fa506b41d0a690aa96ca7f3cd092ac8da6c8ca0e3d2aab3`.
The imported native-root dependency is separately pinned:
`sparse_metric_tangents.py`,
`344461fb7e1e256801766b757f095742247ee0e047193d43919757ae7f105bd5`.
The source-matched offline report SHA256 is
`dfbd18f0337f86100d6014fc96fc56e45c5541746d34474d53ff6d5ad4cb1621`.

Both native selectors passed on both GPUs, including the original16
current/held physical cases and35-world mapped/graph controls. Logs:
`/tmp/fpgs-field-major-native-20260916-ElVXVIYU/gpu{0,1}.log`.
The initial component run01 is invalid: a dependency-hash addition changed
the external replay helper while its source-pinned parent was active. Its
failed manifest is preserved; unchanged numerical code does not excuse
that source-guard failure. No result below relies on run01.

Valid run02 completes with exit0 and original source, input, readonly-array
and final idle guards passing:
`/tmp/fpgs-g1-field-major-component-paired16k-20260916-02/`.
It uses10 warmup rounds and40 measured balanced AB/BA rounds. Canonical
incoming impulses, status and outputs are restored outside every timed
event. Both candidate kernels are inside each event; all packing is
excluded. Original replay reproduces the captured outputs exactly and
each owner's captured graph agrees byte-for-byte with its direct launch.

| Complete component median, ms per solve call | RTX | GB |
| --- | ---: | ---: |
| Original cooperative metric GS plus its decode | 0.507535994 | 0.557071984 |
| Field-major GS plus separate cooperative decode | 3.791040063 | 3.533216000 |
| Original / candidate | 0.13387777 | 0.15766712 |

All16,384 worlds on each card pass finite/status, nonnegative-normal,
cone and captured-operator momentum consistency gates. All published
velocities pass the original-reference tolerance. Pointwise physical
diagnostics have zero RTX regressions and one GB complementarity
regression, world15845; normal, natural and MDP diagnostics have none.
These are finite-eight consistency/comparison results, not proof of a
converged contact solution or an all-pointwise promotion pass. Independent
current-J/H qualification is the separate selected16 native gate.

Final report hashes:

- `gpu0/component/report.json`:
  `1f4250a8e26b237bad8596d04224e7c397671fca4ac313bc89621ad884494ee9`.
- `gpu1/component/report.json`:
  `fd3b4652b3fc00ac55af8ce96f3b5e8685cd68310cee97efd766f4980d51c20c`.
- Frozen external replay helper:
  `/tmp/fpgs-g1-field-major-replay-rm9eNdqx/replay.py`,
  `6c8b6a2cf23d289d733f8164be3cf068da756024e2d84f0b9ed8f5f8931ed682`.

## Cause assessment and funding decision

There is no stack/spill or explicit barrier/shuffle in the new main kernel.
Static disassembly confirms zero LDL/STL, SHFL and BAR on both targets.
Eliminating those original collective instructions therefore happened;
it did not produce a useful total solve. The source exposes several
countervailing costs:

- Main-grid population falls from16,384 original solve warps to512,
  independent of the later cooperative decoder. The latter cannot hide
  latency during main GS. This is a population bound, not a measured
  achieved-occupancy or stall attribution.
- Each world serializes its up-to18-coefficient dot products, cross products and
  response updates, rather than distributing them over coefficient lanes.
  The new metric path reads normal coefficients three times and tangent
  coefficients twice where the original keeps z0/z1/z2 in lane registers.
- Lambda and cached cross coefficients now reside in global field-major
  storage rather than a private warp's shared array. Divergent contact
  types, root exits and world row counts reduce useful common work; the
  approximately2.08x row-slot padding remains.
- The included cooperative decoder reads coordinate-major private du
  across coordinates at a world-count stride. W itself remains canonical
  and cooperative; the new private du gathers and extra publication launch
  are not free.

No phase timer or hardware stall counter separates these effects, so this
report does not assign causal percentages. Nor does it infer that resource
counts alone predict performance. However, the complete lower bound is
already decisively adverse before any row-packing cost.

At the budget's eight solves per environment step, saving1.3ms would require
this captured component to reach at most0.345036ms RTX or0.394572ms GB.
Those are prospective budget conversions, not a measured whole-physics
extrapolation. Relative to the actual component they require approximately
11.0x and9.0x reductions. No single justified correction identified here
can plausibly provide that before packing. Moving to4/8/16 coefficient
lanes per world changes the central ownership choice and overlaps prior
subgroup mapping work; it is not authorized as a corrective grid.

The field-major experiment is closed without production integration or
further mapping variants. Accepted default dispatch remains untouched.
There is no new whole-physics speedup, MJWarp comparison, or cross-task
qualification claim.
