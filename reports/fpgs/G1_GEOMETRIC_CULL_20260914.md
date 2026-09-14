# Current cuboid / heightfield geometric rejection

Pre-code card, 2026-09-14 11:04 UTC. Base Newton `7df75c46`, unchanged Lab
`53ee6b44`. First checkpoint 11:45 UTC. This shared collision change applies
equally to FPGS and MJWarp; no solver-only backend-ratio gain is claimed.

## Boundary and measured reason

Retire geometrically separated triangle queries before the existing midphase
atomic append. The original finite query, marked generic fallback, reducer and
export remain unchanged for every surviving triangle. No additional queue,
mask, compaction pass, kernel, capacity increase or cached current geometry.

The complete paired saved-input diagnostic is preserved at
`/tmp/fpgs-g1-geometric-packed-native-paired-20260914-YoLrRO`.
Its source is `/tmp/fpgs-g1-geometric-query-packed-QeGZaZvD/diagnostic.py`,
SHA256 `68239bb36a9bb96499732ab99cb013574b82e6dbed42600fc9edd0e2514064cb`.
Original versus compacted surviving queries including separator, per four
collisions: RTX 3.224064 -> 2.247424 ms; GB300 11.502592 -> 7.336320 ms.
These are query-boundary diagnostics, not whole-physics measurements. The
earlier masked result had no saving because its holes retained the old stream
mapping. Production append supplies compactness without a separate pass.

## Admission and added work

Default-off `NEWTON_HEIGHTFIELD_GEOMETRIC_CULL=1` requires existing cell reject,
finite query, nonpredictive stock global reduction, and no ordinary meshes.
Per pair, admit only the existing immutable recognized cuboid with matching
source pointer, positive current scale, a current enclosing collision AABB,
finite geometry and unit-enough current orientations. Unknown cases call the
unchanged cell midphase. Hull refit/replacement requires rebuilding the same
experimental finite-query pipeline; this change does not weaken that contract.

Prepare the normalized current cuboid axes and projection radii once per pair.
After the original cell-height test, examine horizontal box/triangle-edge axes,
upward box axes and the upward triangle plane. Never reject below the prism.
Retain all uncertain/tied/nonfinite tests. Use the larger of the complete
current shell and current scaled reducer beta, plus the audited conservative
world-coordinate padding; no physical smoothing or modified height samples.

Added pair preparation and per-triangle axis tests are charged in whole timing.
There is no credit for clear/reduction/export until actually measured.

## Far-witness and physical obligations

The saved FP32 all-callback audit omitted no callback whose raw depth was at or
below max(current shell, current scaled beta) plus conservative padding; minimum
slack was 1.357 mm RTX / 1.255 mm GB. Such far witnesses cannot enter spatial
competition, cannot beat an exportable contact in depth competition, and are
filtered on export when alone. This is not permission to generalize to unknown
hulls, custom writers, predictive admission or arbitrary postprocessors.
Original reducer keys, slots, tie handling and finite-query markers stay intact.
Equal-score scheduling ties need not remain bitwise identical.

Regression-first constructor/API control must fail on base. Then test actual
emitted triangles, near/penetrating/rotated/nonflat/current geometry, source or
AABB admission withdrawal, current scaled beta, empty/regrow and graph replay.
Reuse original finite geometry and physical collision controls; do not impose
unchanged contact count. Root owns paired native physical and original guarded
whole-environment timing. Stop on harmful geometry loss or a whole-cost loss;
only a measured causal issue authorizes a correction. No GPU launch by agents.

## CPU / source checkpoint, 11:24 UTC

The constructor regression failed on base with the missing Boolean owner, then
passed with the implementation. Twenty-one bounded CPU tests pass: all five new
controls, original cell rejection, packed pair mapping/admission and original
finite query controls. The new tests also exercise nonstock writer fallback,
source withdrawal, a too-small local AABB, each nonfinite endpoint gap/margin,
live scaled beta, rotated support radius against all eight independent corners,
actual production emission, empty/regrow and current height-array mutation.

Both pinned saved 96-pair scenes pass the original real reducer-buffer callback
audit at two current-pose epochs. RTX: 1208 -> 605 and 1204 -> 580 triangles;
GB fixture: 1190 -> 669 and 1188 -> 648. All 1381/1341/1403/1372 original callbacks
were inspected, including 603/624/521/540 from actually omitted triangles.
Minimum omitted raw-depth slack above current max(shell, scaled beta) is
2.948 mm. No lost contacting shape or deepest separation change above 0.2 mm.
Counts describe saved demand only, not runtime speedup or contact-count parity.

Offline native compilation at the original block128 passes SM120 and SM100:
116 registers, 512 bytes shared, zero stack/spills/barriers. Initial evidence
is `/tmp/fpgs-g1-geometric-cull-offline-34dTurFY/report.json`; that compilation
precedes source formatting only. GPU physical and whole-physics gates remain
root-owned and pending. This implementation is not promoted.

Native selectors in `newton.tests.test_heightfield_geometric_cull`:
`TestHeightfieldGeometricCull.test_actual_emission_and_graph_cuda` and
`TestHeightfieldGeometricCull.test_saved_current_cuda`. The former includes the
same current admission guards on the GPU; the latter audits original callbacks
for all actual omitted triangles, not a diagnostic mask oracle. Reuse the
existing accepted loaded support/tilt/sliding/border qualification under the
explicit new flag for the physical dynamics gate.

## Native and repeated whole result, 12:22 UTC

Frozen runtime `7790c35b348dd1885a90f6c2def9b336cde80399` passes the four
original/current physical selectors on each GPU with no skips or failures.
All 28 original loaded-support/rebound checks also pass. Artifact:
`/tmp/fpgs-g1-geometric-cull-physical-paired-20260914-01`, manifest SHA256
`1810c0d55e34f25f64dc4dc04461dd31f0c877d7dfcd8baa1d7465ffb93269ce`.
The actual welded-reducer callback audits inspect every emitted callback at
both current poses, not just a selected saved subset; minimum omitted slack
above the complete current shell/beta bound is 3.163 mm.

The original paired 16,384-world protocol keeps Lab, timestep, substeps,
iteration allowances, warmup, current contact capacity and timing ownership
unchanged. Three complete FPGS A/B rounds have no source, capacity or idle
guard failures. The two alternating repeat rounds give:

| Whole FPGS physics | Before | Cull enabled | Speedup |
| --- | ---: | ---: | ---: |
| RTX PRO 6000 | 19.4992 ms | 18.1763 ms | 1.0728x |
| GB300 | 25.1200 ms | 20.9576 ms | 1.1986x |

These are medians from
`/tmp/fpgs-g1-geometric-cull-paired16k-20260914-repeat02`.
The separate discovery round was 1.0692x / 1.2005x. Environment-wall repeat
medians improve 33.7319 -> 31.9614 ms RTX and 39.2986 -> 35.9180 ms GB300.
This is a measured shared-collision gain, not the 4x solver target.

## Fair shared MJWarp denominator

The refreshed shared comparison explicitly enables the same five collision
changes and `NEWTON_NARROW_PHASE_THREADS_X=4` for both backends. The earlier
G1 MJWarp recipe omitted THREADS_X and inherited Lab's default 1; FPGS used 4.
The new common value changes the actual launch size and stride from 49,152
to 196,608 on MJWarp, so changes in its denominator cannot be attributed to
the cull alone. This recipe correction is disclosed, not credited as a new
FPGS algorithm gain. MJWarp uses the shared Newton contact path, verified by
its actual conversion owner and `_use_mujoco_contacts=False`.

| Whole physics, two-round medians | FPGS | Corrected MJWarp | MJWarp/FPGS |
| --- | ---: | ---: | ---: |
| RTX PRO 6000 | 18.1758 ms | 37.8185 ms | 2.0807x |
| GB300 | 20.9272 ms | 38.2270 ms | 1.8267x |

Artifact: `/tmp/fpgs-g1-shared-geometric-paired16k-20260914-repeat02`.
An independent first round gives 2.0801x / 1.8311x. All original source,
capacity, actual feature and final idle checks pass on both cards. Complete
environment-wall repeat medians are 32.7126 versus 51.5760 ms RTX and 35.1819
versus 52.6833 ms GB300; these exclude policy inference and learning.

The shared helper is `/tmp/fpgs-g1-shared-geometric-CHbQCtkF`, with run SHA256
`672e9eb909e53c645512ab4237259e06a6c41961fe11f91700f50e0de2db3214`
and observer SHA256
`e638ce3fa1bb122f5aa8c93a8b86f24427cd330f768978fb1a2b28a6c0f0ea22`.
It preserves original timing/physical/source/capacity ownership and adds
untimed actual shared-collision admission checks. Existing captures retain
the clean runtime pin above; this report-only update does not change it.
