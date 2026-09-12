# Independent hand/object owner: bounded experiment

Default-off experiment from accepted42, not a performance promotion. Root
authorized one same-input owner-cost test before a large live rollout.

Fresh accepted42 inputs at `/tmp/fpgs-kuka-mf-current-paired16k-20260912-01`
passed both source/idle guards, all capacity checks and exact adjacent held-factor
checks. RTX refresh/reuse qualify40/42 and44/46 worlds: independent dense max85,
coupled max31 (world2856; other coupled world8000 has27). GB qualify46/47 and49/50:
independent max95/77, coupled world6872 has66 dense+9MF rows. Only1/40,1/44,0/46,
1/49 independent hand predictors are initially feasible. There is no zero-only
or one-limit shortcut. The GB coupled fallback may retain the critical tail.

Selected implementation retains original eight-sweep/current-friction laws:

- After current physical response/RHS preparation, one bounded-worker pass
  qualifies exact operator separation and converts admitted hand physicalY to
  the existing paired solver's whitenedZ using heldL. Full dense secondaryJ/Y
  must be exactly zero, current MF endpoints only free6, all active rows finite
  and supported, with complete friction parent/sibling metadata. The current
  source's block-diagonal23+6 topology is required. Fresh raw-contact census
  independently agreed with this exact current-operator predicate; no extra
  scan over4M allocated raw contacts is introduced.
- A separate signed selector is0 for noMF, positive actualMF count for complete
  fallback, and-1 for independent components. Canonical counts/IDs never change.
- The existing paired hand variant consumesZ and publishes only its primary23
  velocity coordinates on split worlds. The existing general MF node uses an
  original-law variant with zero dense rows on split worlds and publishes only
  the secondary6. All other MF worlds retain its complete mixed fallback. No concurrent
  read/write across components, even if their launches later overlap.
- Initial mapping is sequential and charges all qualification/whitening,
  extra dispatch, object solve, fallback and publication. No overlap gain is
  assumed. Each independent component retains its own original stationary exit
  and maximum eight sweeps, including friction-start timing.

This removes repeated hand work driven by an independent object and permits the
existing paired contact-triple recurrence. It does not reintroduce failed full
response-block/Gram methods or change held dense versus current MF operators.

Fresh baseline RTX paired+general nodes sum3.213106ms/env. A10% whole15.769ms
milestone needs complete replacement at most1.636206ms, leaving0.620933ms beyond
the existing1.015273ms paired node. These sums are not exclusive wall attribution;
whole-graph measurement is mandatory. A coupled world may defeat the gate. No
whole4x-MJ claim follows from this slice alone.

First gate: all four current saved refresh/reuse inputs, original versus complete
new owner including preparation; actual CUDA eager/captured output/force/action
checks and event cost, no GPU by implementation agents. Include explicit
coupling/owner transitions and disjoint publication controls. Numerical tolerance
and physical residual/action checks, not bit identity. If a remaining coupled
tail explains the loss, retain evidence and stop this slice instead of tuning
its queue/block size. No new iteration/timestep allowance, Lab mutation or
public buffer-capacity change is authorized.

## CPU/source readiness

The owner additionally excludes velocity-post iterations and physical-response
debug/capture readers. The inherited paired-factor guard already excludes the
velocity/debug recipe; these exclusions are explicit here because admitted MF
worlds change their internal Y representation after RHS preparation. Contact
force publication consumes canonical impulses and original contact metadata,
not this private-stage response representation. No public force law is changed.

The classifier has actual nonidentity-factor CPU controls across 517 worlds,
including offset/group permutations, current grow/shrink/empty transitions and
malformed-row fallback. Its finite-product admission bound only falls back; it
never clips or changes the fallback's data. An explicit logical-lane CPU guard
keeps the in-place conversion single-owned. The CUDA native path is unchanged.

Final offline builds of both original/selected consumer variants and preparation
pass on sm120/sm103 have zero stack/spills. Preparation uses 40/32 registers and
516B shared; paired original/selected uses 92/94 RTX and 88/91 GB registers with
5760B shared; generic original/selected uses 80/82 registers and 4340B shared on
both architectures. These are compile resources, not runtime occupancy or speed.
The only new allocation is four bytes per world (65536B at 16384 worlds).

Root ran the complete eight-test module on actual RTX PRO6000 and GB300:
eight passes on each, including both CUDA controls. The device cases cover
517-world graph transitions and the actual original/selected paired plus MF
recurrences with one and eight sweeps, contact triples and delayed friction.
Both processes completed successfully before the next task-level capture.
Default generated sources match the accepted42 controls, and the native
full-step recurrence text is unchanged. Regression-first import fails on
accepted42, where the new module is absent. Tests and compiler output do not
establish loaded physical quality or a task-level speedup.

## First loaded current-input result

The frozen `d49d0715` runtime completed all four saved refresh/reuse cases on
their original devices. Source guards, eager/two-graph agreement, immutable
input checks and original-versus-saved output controls pass. Maximum split
velocity difference is 7.16e-7 and dense impulse difference 5.54e-7; MF
impulses are identical. Physical impulse-action reconstruction errors remain
below 6e-7. The larger GB normal-velocity deficits already exist in the
baseline and are essentially unchanged. These are loaded component checks,
not downstream force/sensor or trajectory acceptance.

Median event times below include preparation, paired hand and general MF
nodes together. Each uses the original 16K grid and original captured world
IDs, but **uncaptured non-MF worlds are empty in both arms**. Ordinary paired
work is therefore absent. These one-substep, original-then-split samples
cannot be converted into whole-physics gains or added to another profile.

| Device / saved phase | Original us | Split us | Original / split |
| --- | ---: | ---: | ---: |
| RTX / refresh | 344.064 | 263.872 | 1.304x |
| RTX / reuse | 239.648 | 238.944 | 1.003x |
| GB300 / refresh | 406.560 | 442.400 | 0.919x |
| GB300 / reuse | 310.400 | 396.912 | 0.782x |

This does not clear the large-gain gate. One bounded diagnostic will separate
preparation, paired and general cost, then retain only independent or coupled
captured worlds to test the predicted fallback-tail explanation. These
subset timings are diagnostic only, never a reduced-work performance claim.
If the unchanged coupled tail dominates, stop this slice without grid tuning.

Pinned reports:

- `/tmp/fpgs-independent-components-loaded-20260912-01/gpu0/report.json`,
  SHA256 `34c972521bf2d9bfde7223ecb918dab04c84180df01850e58a889566c6d5bc3f`.
- `/tmp/fpgs-independent-components-loaded-20260912-01/gpu1/report.json`,
  SHA256 `141014e492f8e67c4b891a143cc4794eac3e83724f24ba54eb369aa7abdd6c44`.
- Replay and controls: `/tmp/fpgs-independent-components-replay-Jmjo9kWu/READY.md`.

## Diagnosed scheduling loss and one fork/join retry

The attribution controls confirm the sequential mapping's stop condition.
On GB, split general costs 291.328/291.296 us for full refresh/reuse inputs,
291.296/291.328 us with only coupled world6872 retained, and 18.912/18.976 us
with only independent worlds. The one coupled world determines the general
tail. Preparation plus paired hand work adds 158.624/116.544 us before it.
Original general co-scheduled those worlds within one grid; the sequential
split removed that overlap. RTX has the same mechanism, with about133 us
full/coupled general versus20.48 us independent-only. Subsets preserve every
selected world's inputs and outputs. These are diagnostic timings, not a
reduced-work optimization claim.

This concrete cause justifies one dependency-removal retry, not another grid
search. Default-off `FEATHER_PGS_PAIRED_GENERAL_OVERLAP=1` creates a persistent
paired stream and two events outside capture. Current preparation completes
before the ready event; general runs on the caller stream and paired runs
independently, with a mandatory caller join before return/publication. The
original owners are already disjoint by world. With the component flag, the
selected worlds are disjoint by velocity coordinates instead. Both original
and split fork variants therefore have explicit same-input controls.

No arithmetic, kernel resources, capacity or iteration allowance changes.
Admission requires the existing23+6 factor recipe and parallel-stream setting;
gradient, extra sweep owners, warm response readers and synchronous/debug
instrumentation are excluded. Unsupported recipes keep the original path.

The prospective RTX whole-cost card uses the fresh3.213106 ms paired/general
boundary and a1.636206 ms ceiling for a10% whole-time step. Subtracting about
.321216 ms preparation leaves **1.314990 ms** for the slower full concurrent
branch plus every event/contention cost. The existing ordinary paired node
is about1.0–1.1 ms, so this is plausible but tight; its added hand work is
not present in the sparse replay and must be measured live. These node sums
are not exclusive critical-path savings. GB's longer fallback tail requires
an explicit no-regression gate.

The complete sparse replay, including fork/join, now gives:

| Device / phase | Original sequential us | Original fork us | Split sequential us | Split fork us |
| --- | ---: | ---: | ---: | ---: |
| RTX / refresh | 343.312 | 338.016 | 264.192 | 172.096 |
| RTX / reuse | 238.896 | 234.976 | 237.776 | 173.152 |
| GB300 / refresh | 405.600 | 401.792 | 442.464 | 339.328 |
| GB300 / reuse | 311.424 | 305.552 | 398.832 | 334.032 |

All four arms pass native five-world ownership controls, including ordinary
non-MF, coupled fallback, both offsets and delayed friction; actual saved
refresh/reuse outputs pass eager and two captured graphs on both cards.
Fork output is identical to the corresponding sequential output. Original
saved-output and before/after source guards pass. Root reaped both children
and checked GPU idle. Ordinary non-MF work remains absent from this replay;
this is not yet a whole-physics gain. The next gate is full16K on both GPUs.

Three Python admission/dispatch tests pass on the new integration, including
original-order preservation, prepare-before-fork and join-before-return.
The corrected regression control passes default order but fails missing fork
behavior on predecessor563ebc3d. An initial stub omitted a required boolean;
that harness error was fixed before the meaningful regression check.

Fork reports and exact input/source closure:

- `/tmp/fpgs-independent-components-fork-20260912-01/gpu0/report.json`,
  SHA256 `a9db9ba413fe9259332e1b5c3f297508ee97a7e6c06c4d4677177c773838a27b`.
- `/tmp/fpgs-independent-components-fork-20260912-01/gpu1/report.json`,
  SHA256 `fc24127b45cd53e7814f97dbdfb5d0be6dbc2f14449c8f32811674110ebd2209`.
- `/tmp/fpgs-independent-components-fork-jslpJHdd/READY.md` records the
  prospective cost card, full timing scope and native controls.
