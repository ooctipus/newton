# Finite-shell query with corrected pair CSR — 2026-09-16

Status: default-off candidate with passing bounded native physical controls and
one measured whole-physics gain: 1.262675 ms RTX / 3.924349 ms GB. The first RTX
milestone is met; repeated timing, further composition and general promotion
remain unqualified. No additive speedup is claimed.

## Source and ownership

Isolated branch `ooctipus/heightfield-shell-pair-csr-20260916`, based on corrected
CSR checkpoint `1e16a323876778aaeddf6c4e8420e3f155234bfc`. The measured CSR tree
is untouched. `NEWTON_HEIGHTFIELD_PAIR_CSR_SHELL=1` selects the existing finite
shell geometry only after ordinary pair-CSR admission succeeds; its default is
zero. The actual dispatch is `narrow._heightfield_pair_csr_shell` and
`narrow._pair_csr.shell_support`. Cached `get_query_kernels(shell_support)` takes
an explicit Boolean; the environment is read only during pipeline construction.

The finite factory is AST-identical to the proven factory at shell checkpoint
`bebf06f8fcdf49bb1c93523ca994447efb779a7b` (comments excluded). Its shell branch
uses `query` instead of the conservative `query_contacts` wrapper. The latter
deliberately falls back for positive-clearance face witnesses because the old
hash reducer did not retain their spatial support. Corrected CSR now owns those
witnesses directly and prioritizes active/nearest-positive arithmetic intervals.

The tagged triangle producer, finite eligibility checks, handled markers, flat
seam filter, complete generic fallback, stock writer, CSR routing, interval
priority and four-witness export are retained. No old shell hash reducer or
priority-bit encoding is imported. There are no additional arrays, capacities,
launches or host synchronization points. Direct/unreduced and CSR-off pipelines
keep the original conservative query; unsupported/ambiguous finite queries
continue through the original generic route. Existing immutable hull/source
identity and current pose/height/scale/margin contracts remain unchanged.

Finite shell geometry can replace generic MPR witnesses on newly handled
triangles; this is not an identical raw pool or an exact old-manifold claim.
The four-witness cap prevents the old shell output expansion only for pairs
already having at least four accepted witnesses. All membership/export and
changed row/GS costs remain part of the whole comparison.

## Causal cost card

Prior shell nodes reduced finite+generic query cost from 2.211680 to 1.400044 ms
RTX and 6.935125 to 3.023003 ms GB, but rows+GS grew .528779/.488086 ms under
the old hash/export path. The initial CSR node query sums were already
2.093570/6.524651 ms; their difference from the shell sample is only an
optimistic affected-owner reference, not a matched forecast. Corrected CSR
alone saved .679287/.683943 ms in one whole comparison. This composition tests
whether query retirement survives without the old manifold expansion. The
milestone remains at least 1 ms complete RTX saving, with all original budgets,
fallbacks, physical controls and publication costs included. No mapping grid.

## Regression and CPU evidence

Before runtime edits, session80235 failed the new explicit-factory and actual
binding controls on the missing Boolean argument/owner marker. After edits,
session50248 returned0: 21 CPU tests passed, eight CUDA selectors skipped with
devices hidden. These cover the inherited variable-pool, exact writer,
interval/cancellation, mixed fallback, overflow/sticky status, reset and nearest
footprint tests; new explicit cache modes, actual binding and handled-marker
controls; existing finite query tests and independent synthetic geometry.

The new observer test executes the literal production collision extension on
real `CollisionPipeline` objects. Both shell modes pass; wrong query identity,
actual marker, owner marker, missing flag and sticky status are rejected in
each mode. Existing array/device/capacity/source guards are retained.

On both support and positive-gap rebound CPU fixtures, the exact tagged stream
contains the same 30 triangles. Finite handled entries increase 22→30; raw
witnesses change 58→54; published witnesses remain four. These two fixtures
demonstrate the intended route change, not a full-task population census.

The inherited nine loaded cases are unchanged. Their comparison keeps corrected
CSR enabled in both arms and toggles only the shell flag. Rebound speed/energy
and spin must be inspected explicitly: the inherited free-foot failure list is
not by itself an analytic restitution gate. Production anchor2 sliding and the
anchor0 analytic friction control remain distinct.

## Hidden-device compilation

Existing offline compiler reused at
`/tmp/fpgs-shell-pair-csr-offline-zfeb2zHu/compile.py`; output is
`offline01/report.json`. Session38624 returned0 with no visible CUDA devices.
Report SHA256: `0a7cf8785f247993d5dc890ced2ffa3cd05f2277337d5e168f0719e6157f3a52`.
Actual midphase, finite, generic, membership and stock export entries compile
for SM120 and SM103. Finite-shell uses 128/130 registers, 496 B stack and zero
spill loads/stores, versus corrected conservative finite 128/133 registers and
592 B stack. Generic remains 168 registers/432 B stack; export remains
72 registers, 128 B static plus unchanged 336 B dynamic shared, zero spills.
These are compiler resources, not occupancy or stall measurements.

## Bounded native physical qualification

Root sessions18514/36320 ran the complete `TestHeightfieldShellPairCSR` class
and `TestIndependentGeometry.test_synthetic_native_cuda`, with the inherited
fixture environment `FEATHER_PGS_GROUP_LANES=16`, `FEATHER_PGS_ROWS_MASKED=1`,
`NEWTON_NARROW_PHASE_THREADS_X=4`. Both cards passed22 tests (60.046/64.193 s),
including all18 loaded records: nine corrected-CSR reference and nine shell+CSR.
The independent read-only source review also found no concrete blocker.

Logs: `/tmp/fpgs-shell-pair-csr-native-20260916-CxMjp4Ok/gpu{0,1}.log`.
SHA256 RTX `25c6f0b1648e570e0aa140a01e7dc4af410ad7d135b1e92761ab464bdd56ed72`;
GB `de117ad4ee16a49baa413cf7f5afb6f424f640855eaed8c45b0ea6c5f270e19c`.

Candidate physical metrics match across the two cards:

- Free rebound: normal velocity +1.800000191 m/s and energy ratio .360000076294
  for authored restitution .6 and incoming speed3 m/s. Spin is 2.96623e-6 rad/s,
  versus corrected CSR-only 3.35866e-5. Scalar rebound has the same normal
  velocity/energy and zero spin. These were inspected explicitly, not inferred
  from the inherited free-foot failure list.
- Flat support: final speed1.89358e-6 m/s, spin2.69729e-5 rad/s; complete80-step
  relative weight error9.19379e-7 and momentum error5.07842e-8. Tilt and yaw
  settle. Step recovery final speed .000370068 m/s and spin .00696217 rad/s;
  its tail maximum spin is .0143672 and relative weight error7.56588e-5.
- Across the nine candidate cases: maximum cone excess7.45058e-9, no negative
  normal impulses, maximum scaled momentum error3.82112e-7.
- Full-friction sliding stops at x=.50718540 m versus analytic .509684 m,
  final speed2.93472e-7 m/s. Production anchor2 sliding is a distinct law: its
  x=1.259647 m, vertical speed−.240015 m/s and spin2.95255 occur after the finite
  COM edge x=1.2 m. Samples at x=1.12647/1.17461 remain at z=.03 m with near-zero
  vertical speed; tipping begins beyond the edge. No sampled interior
  fall-through is observed; exact trajectory parity is not claimed. The
  separate finite-border case reaches x=1.55317 m with zero contacts.

## First integrated comparison

Artifact: `/tmp/fpgs-g1-heightfield-shell-pair-csr-whole-paired16k-20260916-01/manifest.json`.
SHA256 `d9644cb15b223dfb77253b4da4a08984ebfdce4b6f14ef67f5440e674d4a0d71`.
Root session33299 and all four children returned0, with no cleanup signals.
All eight capacity/actual-owner boundaries and final source/idle guards passed.

The reference is the unchanged qualified-chain tree `ef9481bc`, not the slower
CSR or shell prototype. Same seed0, 16,384 worlds, 200 warmup, 40 wall and40 graph
steps, two Newton substeps, eight GS sweeps and original contact/broad/triangle
capacities294912/49152/1769472. Row capacity100 and all original flags are
unchanged; only `NEWTON_HEIGHTFIELD_PAIR_CSR` and
`NEWTON_HEIGHTFIELD_PAIR_CSR_SHELL` differ0→1. This tree contains no spectral
tangent composition. The checked observer proves the actual Boolean/query
factory and all previous integer/interval ownership and status guards.

| Device | Qualified-chain physics ms | Shell+CSR physics ms | Saved ms | Ratio |
| --- | ---: | ---: | ---: | ---: |
| RTX | 15.044929850 | 13.782255325 | 1.262674525 | 1.091615958 |
| GB | 19.807130175 | 15.882781250 | 3.924348925 | 1.247081973 |

Synchronized wall stepping is separate: RTX27.585170→27.875054 ms (slower),
GB33.608828→29.704653 ms. Neither is full RL throughput. Query, raw membership,
interval selection, fallback, publication, rows, GS and final state are all
inside the whole-physics measurement; this is not a sum of component savings.
There is no new component-only node capture, repeated run, MJWarp comparison
or multi-task qualification for this standalone checkpoint. Defaults remain
off. The next authorized scope is a separate composition/repeated screen.

## Frozen source pins

- `heightfield_finite.py`: `c82d3d1ef5e0b2e3e738ac523b40bb5c4f0992f8db500ca05c72cf0e9e86a866`
- `heightfield_pair_csr.py`: `a3891ca6290f2a9373ce065aabc3b4d2818a7eadf9b344e145683150731bb9e3`
- `narrow_phase.py`: `e8cfddf64154bca88af5839fb95530934e9a82aa018e3366379ef477a2e2e222`
- `test_heightfield_shell_pair_csr.py`: `c2acd5be81f73dc6496004dd94b3f7a05d5cfacd8a1e48053f86d1a10bcc9aa7`
- `checked_sparse.py`: `bdeb1d0b58772d79e5503274d174d89c8db0a02e9733031e578ce9dcaa8a0a0d`
- `run.py`: `84d3c3be89dcd553285de87f1142583e9ff74057ee7b5f38a24f0f0cd5897006`

Runtime/test/observer sources remain unchanged since the native/whole freeze.
