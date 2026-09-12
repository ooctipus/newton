# Structural continuation: measured results and open gates

This is the continuation after the correctness checkpoint, not a completion of
the additional 2–4x target. The earlier
[structural plan](STRUCTURAL_V2_20260912.md) records the pre-implementation
hypotheses. This report supersedes its statement that no new timing gain exists.
Isaac Lab is unchanged at `1d8feb82d17dbfab8f0772de56f84deae2cb7974`.
All Newton branches retain the original handoff `31cf87f46` and use the
`ooctipus/newton` fork. Experimental flags remain default-off.

## Keyboard SO101: repeated whole-physics timing

The baseline is the capacity-corrected Newton
`108459ec7420fb62f79bc2b2728c974548142105`; candidate runtime is
`654894cb9ee623c9d516a535eb4de7bbb87ae285`. Both use 4,096 worlds, seed 0,
200 warmup steps, 1,000 synchronized environment steps and 40 graph-profile
steps per arm. Three rounds alternate baseline/candidate, candidate/baseline,
baseline/candidate. Each arm starts RTX PRO 6000 and GB300 simultaneously,
with one owned process per device. These are per-device medians, not pooled
cross-hardware samples.

| Hardware | Baseline physics ms | Candidate physics ms | Baseline/candidate | Baseline wall ms | Candidate wall ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | 11.258526 | 7.871787 | **1.43024x** | 38.377142 | 35.700507 |
| GB300 | 10.749849 | 7.124362 | **1.50889x** | 34.095171 | 31.407152 |

Physics samples in round order, in milliseconds:

- RTX baseline: 11.1186283, 11.2827297, 11.25852635;
  candidate: 7.683744825, 8.079784575, 7.871786775.
- GB300 baseline: 10.7663184, 10.5390982, 10.7498492;
  candidate: 7.124362225, 7.0904303, 7.485207275.

The graph speedup is not end-to-end training throughput. Measured environment
wall improvement is only about 1.08x/1.09x; reset/IK, observation, host and other
non-physics work remain. No Lab changes are included. Do not divide these new
samples by an old, unpaired MJWarp sample and call that a fresh measured ratio.

All 12 captures completed with finite states, zero four FPGS sticky capacity
flags and zero 13 Newton narrow-phase flags at both checked boundaries. All
source/idle guards passed and children were reaped. The original timestep,
two substeps, effective eight sparse PGS sweeps and solver law remain unchanged.
Both arms use the same calibrated capacities: 704 rows, 147,456 contacts and
57,344 broad-phase output pairs. The full input pair list is not reduced.

**This is repeated timing evidence, not automatic physical-quality acceptance.**
Actual loaded-input producer controls now pass in two separate 512-world paired
runs on both GPUs: five post-warmup checkpoints per child, 320 actual solver
calls, original sparse-response and Stage 7 publication controls, and physical
J6-plus-two-sparse-coefficient row diagnostics. Source, capacity and process
ownership checks pass. The same loaded-input gate also passes at 4,096 worlds
on both devices, with five checkpoints in each baseline/candidate child and all
four FPGS / 13 collision flags clear. Independent 4K runs have different target
positions and reset traces, so they are not matched-control trajectories.
Finite states and zero capacity flags alone do not establish numerical quality.

Independent rollout differences must not be attributed to the candidate without
a control: two unmodified-path baseline runs also have different reset traces and
about 1.6 cm median all-body translation differences. On RTX, baseline run 2
versus candidate run 1 instead has identical reset traces, zero median body
translation difference and at most 0.211 mm translation difference at the five
checkpoints. Joint target positions still differ; applied joint-force and
target-velocity channels agree. GB realizations group differently. This supports
the same-input implementation checks but is not a matched-control trajectory
equivalence claim. The sampled fixed-eight-sweep physical residuals are not
required to be exactly zero; numerical convergence, not bit identity, is the gate.

## Fresh balanced Keyboard comparison with MJWarp

Three long alternating FPGS/MJWarp rounds have now completed on both devices,
using the same 4K task, 200 warmup / 1,000 synchronized / 40 graph-profile steps.
FPGS is the retained A+B runtime `654894cb`; MJWarp uses correctness-fixed
Newton `108459ec` and its process-local line-search correction. Original task
timestep, substeps and each backend's effective solver allowances are unchanged.
FPGS capacities are the 704 / 147,456 / 57,344 values above. MJWarp uses
`njmax=320`, `nconmax=32`, 131,072 public contacts and 45,056 broad output slots.

| Hardware | FPGS physics ms | MJWarp physics ms | MJWarp/FPGS | FPGS wall ms | MJWarp wall ms | Wall ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | 7.819561 | 27.760556 | **3.55014x** | 35.464991 | 32.026520 | **0.90305x** |
| GB300 | 7.176602 | 27.995816 | **3.90098x** | 31.160383 | 32.640041 | **1.04749x** |

These are per-device medians of three graph-only captures, not a ratio to old
unpaired samples. All 12 captures have finite states and pass both checked
boundaries: four FPGS flags or all MJWarp warning-mask positions, plus all 13
collision flags. Sources pass the checked launcher guards, children exit zero
and their process groups are reaped. This is not a cross-backend trajectory or
solver-parity claim. In particular, the RTX whole-environment wall result remains
slower despite the much faster physics graph; do not advertise training speedup.

The subsequent read-only wall audit found a material workload difference:
FPGS enters the batched reset path on 966–977 of 1,000 timed steps, versus 24
for MJWarp. This means at least one world resets, not that 97% of worlds fail.
Inclusive reset host elapsed time is about 23.6 ms/step RTX and 20.0 ms GB for
FPGS, versus about 0.56/0.52 ms for MJWarp. It includes nested GPU waits and
must not be added to graph durations. The separate 40-step traces contain
40 reset calls for FPGS and none for MJWarp; the RTX reset ranges contain about
84–85 stream synchronizations per reset. Existing captures do not identify
per-world reset causes or episode ages. A read-only termination-mask diagnostic
is being prepared without changing task rules, actions or reset behavior.
This is not evidence that Isaac Lab alone owns the end-to-end shortfall.

Ten times the measured RTX MJWarp physics would require approximately 2.776 ms,
versus the current 7.820 ms: about 5.044 ms / 64.5% further removal. The current
contact and solve owners cannot deliver that alone. It is an architecture-wide
target, not evidence that lowering iteration counts or dropping rows is justified.

## What changed, and why it was worth implementing

`FEATHER_PGS_SPARSE_CONTACT_DIRECT=1` assigns one bounded worker to each sparse
scalar-contact slot. The old scalar kernel accidentally inherited the dense
warp-per-contact producer's 4,096-worker limit, giving it only 16 CTAs. The
existing native arithmetic and output ownership are unchanged; there is no new
allocation, host contact-count read or kernel launch. The separate first graph
A/B measured 1.126x RTX / 1.142x GB whole-physics speedup, not just a kernel gain.

`FEATHER_PGS_PRISMATIC_PUBLICATION=1` admits exact fixed-root, direct-prismatic-
leaf stars. Original FK handles the root; the following body-parallel publisher
produces current canonical key kinematics/dynamics. Unsupported articulations
retain their original path. The mass/solver/integration algorithms are unchanged.

The combined actual node profile confirms these owners on the real task:

| Kernel, per environment step | Original RTX ms | Combined RTX ms | Combined GB ms |
| --- | ---: | ---: | ---: |
| Final FK kinematics | 2.824845 | 0.183211 | 0.176191 |
| Sparse diagonal contact response | 1.508556 | 0.190133 | 0.155359 |
| Body finalizer / replacement | 0.419627 | 0.831435 | 0.360107 |

The replacement finalizer is more expensive than the old finalizer; this cost
is included, not hidden behind the faster FK producer. The combined graph still
has 344 kernel and 116 memory nodes per environment step. Node profiling is
attribution, not a substitute for the three-round graph-only timing above.
The remaining RTX sparse solve costs 1.125313 ms in that node capture.

Twenty-one targeted implementation tests pass on **each** actual GPU: sparse
response ownership, prismatic publication and mass-interval/reset behavior.
The wider response suite also exposes one inherited test-signature error:
`test_tiled_response_diagonal_matches_dense_reference` supplies seven arguments
to a nine-argument kernel. It was reproduced on both GPUs at clean baseline
108459ec; it is not hidden or counted as a passing test.

## Next substantial opportunities and stop rules

The additional 2x target against the fixed RTX baseline means approximately
5.629263 ms. The current 7.871787 ms still needs about 2.242524 ms net removal.
Do not restart the target from a slow prototype, count restored dropped work as
an optimization gain, or assume one more producer specialization will reach it.

The scalar Keyboard experiment changed the admitted keys' representation across
both producers and real scalar mass/force consumers. It must exclude legacy
masked spatial-inertia/composite readers before omitting their inputs, retain
current public poses/velocities and sparse-contact geometry, and preserve cold
cache, reset, notification, held-mass and stream-event contracts. Merely skipping
matrix stores or tuning the finalizer alone is not the proposed experiment.
The six-DOF arm's selected-tau owner is not removable key work. Its source is now
frozen at `382f659dbd01f6dea468db1ff37ee6ab88da1f31`, with 29 targeted tests
passing on each actual GPU, including cold/reset/model-notification and real
multi-stream graph execution. Loaded-task controls now pass at 512 worlds on
both devices: all 108 scalar keys per world, actual current force/mass inputs,
both refresh/held mass phases, canonical publication and unchanged eight sweeps.
The first 4K graph A/B is only 1.01278x RTX / 0.99821x GB (one short sample),
not the required large gain. Exact node-overlap accounting finds only
0.302315/0.108459 ms exposed savings in the three replaced owners. Current
canonical transforms and publication remain substantial; the legacy masked
composite work was already mostly skipped. This direction is deferred, not
micro-tuned. Reopening requires a different producer/consumer boundary with
newly measured headroom. Different loaded contact counts prevent attributing
every other-owner increase to this candidate.

ANYmal's 18-DOF parallel solver owns about 4.075/4.459 ms in the fresh RTX/GB
profiles. A source-isolated diagnostic records **every executed sweep's** fresh
positive parent/friction-radius history and incoming Nesterov momentum. A final
zero impulse is not proof that a row was unnecessary earlier. Only after this
census will a lazy-row algorithm be considered, charging detection, construction,
changed preconditioning, convergence and fallback costs. This diagnostic has no
timing claim and changes no live solver statement. The census found that only
about 38% of tier-48 tangents ever activate, but activation can occur as late as
sweep 22 and incoming momentum can remain when the current friction radius is
zero. All mandatory plus ever-activated rows fit within 32 lanes in the sampled
16K world-calls; this is not a universal capacity guarantee.

A second, source-isolated phase census passed original/traced numerical and
graph controls at 512 and 16K worlds on both GPUs. Instrumented same-CTA elapsed
cycles put sweeps at about 45-47% of the RTX owner and 46-51% on GB. These are
not exclusive GPU wall-time fractions; instrumentation changes registers and
the final clock is lane-zero publication progress. They are nevertheless too
small to support the proposed >=10% whole-step gain from merely reducing sweep
row visits to 60% while retaining full preparation. That proposal is deferred
before implementation. A replacement must also address preparation/majorizer
cost, not spend hours polishing the same limited hypothesis.

A bounded CPU reference of that replacement is now complete on all 2,048
world-call observations from both devices' 512-world captures. It causally
materializes tangents when needed within the current sweep, retains activated
rows and momentum, and incrementally forms exact restricted Gram row sums.
It changes the pair preconditioner explicitly; it is not an old-iterate shortcut.
At the unchanged maximum 24 sweeps, materialized rows are 60.62% of full and
ordered Gram products 36.41%. Natural residual median/p99 improves from
0.002593/0.02037 to 0.0002817/0.001912 against an independent physical reference.
Individual losses remain: 90 of 2,048 natural-residual cases worsen, and one
contact's frozen-geometry next-gap prediction worsens to about -80 micrometres.
All reference solves converge offline, but those additional reference iterations
are not a production allowance. These are numerical/logical-work results, not
GPU timing or general trajectory acceptance. One native full-owner experiment
is now being implemented; parked-warp, preparation, insertion and fallback costs
remain charged. Its append-only panel changes FP32 reduction order relative to
the CPU reference and must be evaluated on actual outputs.

Allegro's rejection-only temporal support-axis carry-forward now has repeated
timing evidence, described below. It is an old algorithm carried onto the
correctness-fixed source, not a newly invented collision method.

The other active Keyboard prototype replaces seven separate contact producers
with one complete compact contact owner plus a bounded dense-limit prefix.
It retains actual contacts, original allocation/scheduling/RHS and eight sweeps;
zero arm coupling does not mean a contact is inactive. The measured seven-owner
union is 1.782/1.896 ms, and 92.9% of sampled contact triples have zero J6 arm
coupling. The first target is >=0.787 ms net RTX removal after charging all new
producer, clear, prefix and synchronization costs. Its source is frozen at
`48c0add23be9426cfff3a9fa5a6db02a3a2aea29`; 29 targeted device tests pass on
each GPU. The first 4K graph A/B is 0.99923x RTX / 1.22409x GB (one short
sample): no RTX gain, so it is not promoted. A successful source-bound node
capture localizes the mismatch: the seven old owners' exposed time is
1.782380/1.894629 ms; the complete three-owner replacement is
1.716397/0.771274 ms. The new contact producer alone takes
1.641485/0.693109 ms; prefix/clear and added idle time do not explain the RTX
loss. Its actual 113/114-register kernels have zero reported local spill bytes.
The old J-clear sum was almost entirely overlapped and cannot be credited as
exposed savings. Raw-index warp mixing is being measured before selecting one
causal retry; the earlier zero-J6 fraction is not itself a divergence diagnosis.

The first loaded 512-world compact-contact quality run stops at the independent
FP64 response oracle on both devices (maximum violating Y6 difference about
1.28e-5/1.10e-5). The preceding actual/private/original-producer comparison had
completed, but the full gate did not. Original failure snapshots are retained;
geometry-rounding amplification versus a runtime error must be diagnosed before
changing an oracle or claiming this candidate is quality-accepted.

That diagnosis is now complete. Maximum J32-versus-raw64 geometry differences
are about 6e-7; applying the actual held-H inverse amplifies them to the observed
1e-5 response difference. Solving against the independently checked published
J32 passes the original Y tolerance, with componentwise backward error about
1.4e-7. The original seven kernels on the same CPU inputs fail the old raw64-Y
criterion too. A fresh oracle separates geometry and solve contracts without
changing tolerances, runtime or physics; wrong-J/exact-Y and wrong-Y controls
reject corruption. The fresh two-call loaded GPU gate now passes on both cards,
including original/private/live publication, prefix ownership and all capacity
checks. Original failed captures remain retained.

The raw-order census rules out widespread dense/key-only warp interleaving:
only one mixed warp per device. It does reveal poor instruction-level store
coalescing, including scattered J/Y zeros for about 95% of contact triples.
A source/PTX-backed address model motivates one output-ownership retry: a
coalesced current-contact J/Y clear, followed by dense-endpoint-only J/Y stores
in the raw producer. Canonical consumers, contacts and eight sweeps remain.
The modeled scalar-sector-request reduction is about 43%, not measured DRAM
traffic or a proven explanation of elapsed time. The complete new clear and
producer must be timed before claiming a gain; no block-size tuning grid follows.

The single coalesced retry is now implemented in experimental commit
`5777558f98739c8afade7f970d79dc8a4cf52044`. All 36 targeted tests pass on each
actual GPU, including the complete clear/producer, original solver reset/notify,
held mass and repeated graph controls. Its first complete same-capacity 4K
screen (200 warmup / 40 synchronized / 40 graph steps) is:

| Hardware | Retained A+B physics ms | A+B+coalesced contact ms | Ratio |
| --- | ---: | ---: | ---: |
| RTX PRO 6000 | 8.116740 | 7.408049 | 1.095665x |
| GB300 | 7.582394 | 6.234964 | 1.216109x |

All four captures pass capacity/finite-state checks and checked source/process
guards; the parent is reaped. The added clear is charged. This removes about
0.709 ms / 8.73% on RTX in one screening sample, after the first compact version
showed no RTX gain. It is not a repeated retained result or the additional 2–4x
target. A fresh loaded-input gate observes the actual clear independently and
replays private clear+raw from the pre-clear seed; 12 CPU observer tests pass.
Loaded GPU checks and repeated whole-step timing remain pending at this entry.

The subsequent three long alternating A/B rounds are complete (same 4K recipe,
200 warmup / 1,000 synchronized / 40 graph steps). All 12 captures pass both
capacity boundaries, finite-state checks and the checked source/process guards;
the parent has been reaped. Medians are:

| Hardware | A+B physics ms | A+B+coalesced physics ms | Physics ratio | A+B wall ms | Coalesced wall ms | Wall ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | 7.804528 | 6.985432 | 1.117258x | 35.683837 | 34.405500 | 1.037155x |
| GB300 | 7.238613 | 6.147953 | 1.177402x | 30.933954 | 29.811093 | 1.037666x |

Thus the measured RTX reduction is 0.819096 ms / 10.50% of physics time, while
wall throughput improves only about 3.7%. This is a structural gain, not the
additional 2–4x target and not a new paired MJWarp comparison. Fresh loaded512
checks now pass on both GPUs: current-range clear, prefix/tail ownership,
private original-seven and clear+raw controls, independent J/held-H response,
original solve continuation and all capacity flags. A population-only 4K loaded
gate is being prepared before adding this experimental runtime to the handoff.
The experimental commit is pushed to `ooctipus/fpgs-compact-coalesced-20260912`.

The Keyboard broad-phase neighbor-list idea was closed before implementation:
its measured 0.370/0.434 ms owner cannot meet a 10% whole-step milestone even if
free. This is the systematic prevention of unproductive micro-optimization:
write the ownership/cost hypothesis first, run an early integrated causal test,
diagnose a theory/timing mismatch, allow one retry only for a new concrete cause,
and time-box work rather than repeatedly polishing a sub-percent opportunity.

## Allegro: repeated collision carry-forward result

Baseline runtime is again `108459ec`; candidate is
`fba9fead70d17728f842954d66cf1f05f4617d40`, pushed to
`ooctipus/newton` branch `ooctipus/fpgs-rejection-port-20260912`.
Enable `NEWTON_NARROW_PHASE_COHERENT_CONVEX=reject_only` only for the candidate.
The previous axis is only a hint: strict separation is checked against complete
current-shape support and the original contact shell. Everything else uses the
original cold MPR/GJK/manifold path. No accepted warm simplex/contact patch is
reused. Reset generations, graph ownership and all sticky capacity checks remain.

Three long graph rounds use AB/BA/AB, 16,384 worlds, seed 0, 200 warmup steps,
1,000 synchronized steps and 40 graph-profile steps per arm, paired GPUs.
Original timestep, substeps and effective solver allowances are unchanged.

| Hardware | Baseline physics ms | Candidate physics ms | Baseline/candidate | Time removed | Wall ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | 19.156731 | 17.334779 | **1.10510x** | 1.821952 ms / 9.51% | 1.06833x |
| GB300 | 21.150929 | 17.963830 | **1.17742x** | 3.187099 ms / 15.07% | 1.12059x |

Physics samples in round order, in milliseconds:

- RTX baseline: 19.076361425, 19.156731125, 19.182069325;
  candidate: 17.359969075, 17.334778775, 17.30559845.
- GB baseline: 21.1509293, 20.982508, 21.22893395;
  candidate: 17.963829875, 17.76699725, 18.2365032.

All 12 captures pass finite-state, four FPGS and 13 collision sticky checks,
source and idle guards. Native device tests pass 38/38 on each GPU. Separate
same-current-input contact-publication controls pass at 512 worlds, 16K worlds
and held-out seed 1, including natural resets and private partial/global cache
invalidations. Each device/phase checks eight adjacent actual collision calls
against current cold producer controls and complete support geometry. This is
contact-publication quality evidence, not a blanket whole-trajectory guarantee.

Both timing arms retain the same inherited Allegro FPGS allocation: public
contact capacity 14,172,160 and broad/query capacity 2,834,432. These are **not**
claimed to be minimal or newly right-sized. The candidate adds 205,717,556 bytes
of exact eligible-pair/query cache storage; this cost is not hidden. The separate
MJWarp Allegro capacity calibration must not be misreported as calibration of
this FPGS pool. No fresh balanced Allegro/MJWarp speedup is claimed here.

Reproduce with this branch's portable `compare_variants.py`, selecting the two
source commits above and `--task allegro --num-envs 16384 --rounds 3
--warmup-steps 200 --steps 1000 --profile-steps 40`, plus
`--candidate-env NEWTON_NARROW_PHASE_COHERENT_CONVEX=reject_only`.
Use explicit Lab/baseline/candidate paths and a fresh output directory as in
`tools/fpgs_bench/README.md`. No capacity overrides were used for this A/B.

## Local evidence and reproduction identities

Large captures remain local. The measured runtime/source commits and the exact
commands, environment flags, GPU UUIDs, source hashes, results and checked
capacity reports are retained in these manifests:

- `/tmp/fpgs-keyboard-structural-balanced-20260912-01/ab_manifest.json`:
  three alternating long rounds, all six paired arm manifests.
- `/tmp/fpgs-keyboard-structural-nodes-20260912-01/manifest.json`:
  combined node attribution, all checks complete.
- `/tmp/fpgs-response-owner-keyboard4k-ab-20260912-01/ab_manifest.json`:
  response-only first causal A/B.
- `/tmp/fpgs-prismatic-keyboard4k-ab-20260912-01/ab_manifest.json`:
  publication-only node A/B.
- `/tmp/fpgs-keyboard-structural-gpu-tests-20260912-01`:
  21 targeted tests per GPU.
- `/tmp/fpgs-response-diagonal-baseline-control-20260912-01`:
  clean-baseline reproduction of the inherited test error.
- `/tmp/fpgs-keyboard-quality-paired-512-01/paired.json` and
  `/tmp/fpgs-keyboard-quality-paired-512-02/paired.json`:
  actual loaded producer/physical diagnostics; independent rollout caveat above.
- `/tmp/fpgs-keyboard-quality-paired-4k-01/paired.json`:
  loaded producer checks at the actual 4K timing population, all children and
  final source/process/idle checks pass; independent controls/resets differ.
- `/tmp/fpgs-mj-keyboard-balanced-20260912-01/manifest.json`:
  three long alternating backend rounds; SHA256
  `2c567eaf412e3b16c16afe68281d941d64777b051d44bab95eb9def67c51b043`.
  Its `summary.json` has SHA256
  `434a8ec3b27a6a56370a5551f299fbe69c1c80877b3fa1459aa509d762f8657a`.
- `/tmp/fpgs-keyboard-wall-audit-c27E1E/AUDIT.md`:
  all twelve timed reset counters, trace containment and measurement limits.
- `/tmp/fpgs-compact-coalesced-keyboard4k-first-ab-20260912-01/manifest.json`:
  first complete coalesced-retry timing screen, not repeated promotion.
- `/tmp/fpgs-compact-coalesced-keyboard4k-balanced-20260912-01/manifest.json`:
  three long alternating rounds, SHA256
  `efa126b90f0ff441030cfa87922c9c54dbebf4cadf26952325726286e80c398c`.
- `/tmp/fpgs-compact-e2-loaded-paired-512-01/paired.json`:
  fresh loaded E2 gate, both devices pass, SHA256
  `efdd4418bad6af5211d2abb33e7914a2b7eb86e5e6c3a5dabc5ff7325d733660`.
- `/tmp/fpgs-compact-coalesced-gpu-tests-20260912-01/manifest.json` and
  `/tmp/fpgs-compact-e2-loaded-2OBF0T/READY.md`:
  36 GPU tests per device and source-frozen loaded-check observer.
- `/tmp/fpgs-scalar-key-gpu-tests-20260912-02`:
  29 targeted scalar/publication/response/mass tests per GPU. The prior `-01`
  failure is retained: the root launcher named a nonexistent test module.
- `/tmp/fpgs-scalar-key-quality-paired-512-01/paired.json` and
  `/tmp/fpgs-scalar-key-keyboard4k-first-ab-20260912-01/manifest.json`:
  loaded scalar quality and first integrated timing, respectively.
- `/tmp/fpgs-scalar-key-attribution-HO3pyh/evidence.json`:
  exact successful-NVTX-graph-launch node accounting, independently recomputed.
- `/tmp/fpgs-compact-contact-keyboard4k-first-ab-20260912-01/manifest.json`
  and `/tmp/fpgs-compact-contact-keyboard4k-nodes-20260912-01/manifest.json`:
  first compact-owner graph A/B and separate node diagnosis.
- `/tmp/fpgs-compact-contact-attribution-3LyzxT/DIAGNOSIS.md` and
  `/tmp/fpgs-compact-loaded-paired-512-01/paired.json`:
  cost diagnosis and retained unsuccessful loaded FP64-oracle gate.
- `/tmp/fpgs-compact-loaded-oracle-diagnosis-Usf5dO/result.json` and
  `/tmp/fpgs-compact-loaded-v2-1Ds6Zb/READY.md`:
  independently recomputed geometry/solve decomposition and corrected oracle.
- `/tmp/fpgs-compact-loaded-paired-512-02/paired.json`:
  fresh loaded original-E GPU gate, both cards complete and all checks pass.
- `/tmp/fpgs-compact-raw-analysis-cyceGo/RAW_AND_STORE_MODEL.md`:
  raw-order census and explicit scalar-store address model, not a timing result.
- `/tmp/fpgs-anymal-phase-paired16k-20260912-01/manifest.json`:
  source-isolated preparation/sweep/publication census, not performance timing.
- `/tmp/fpgs-anymal-lazy-reference-zE5yUe/RESULT.md`:
  fixed-24 CPU method, independent physical references, individual losses and
  source/input/output pins. SHA256
  `5781a0ca6c5f03c909de13f22fcf186abab3531000f105cda6056c6bff8028e2`.
- `/tmp/fpgs-rejection-port-allegro16k-balanced-20260912-01/manifest.json`:
  repeated Allegro timing, SHA256
  `f25866e19e02fe544b5045c04ec22491a0b71e0fb6b86702a669d8e1f35b5321`.
- `/tmp/fpgs-rejection-port-gpu-tests-20260912-02` and
  `/tmp/fpgs-rejection-port-quality-{512,16k,heldout}-01/paired.json`:
  actual CUDA and contact-publication controls. Original failed device controls
  remain retained; their fixes were test launch/expected-warning capture only.

The paired benchmark helper and handoff report are retained in this branch;
no Lab dependency pin, original worktree or installed backend package is changed.
The overall structural optimization mission remains open.
