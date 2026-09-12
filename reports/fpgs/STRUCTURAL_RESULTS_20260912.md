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
ownership checks pass. The 4K loaded-input gate remains pending. Finite states
and zero capacity flags alone do not establish numerical quality.

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

The next Keyboard experiment changes the admitted keys' representation across
both producers and real scalar mass/force consumers. It must exclude legacy
masked spatial-inertia/composite readers before omitting their inputs, retain
current public poses/velocities and sparse-contact geometry, and preserve cold
cache, reset, notification, held-mass and stream-event contracts. Merely skipping
matrix stores or tuning the finalizer alone is not the proposed experiment.
The six-DOF arm's selected-tau owner is not removable key work. Its source is now
frozen at `382f659dbd01f6dea468db1ff37ee6ab88da1f31`, with 29 targeted tests
passing on each actual GPU, including cold/reset/model-notification and real
multi-stream graph execution. Loaded-task quality and first timing are still
pending; no scalar-representation speedup is claimed yet.

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
`48c0add23be9426cfff3a9fa5a6db02a3a2aea29`; device gates and timing are pending.
Serializing three small solves per contact and losing old overlap are explicit
risks to measure, not reasons to guess either success or failure.

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
- `/tmp/fpgs-scalar-key-gpu-tests-20260912-02`:
  29 targeted scalar/publication/response/mass tests per GPU. The prior `-01`
  failure is retained: the root launcher named a nonexistent test module.
- `/tmp/fpgs-anymal-phase-paired16k-20260912-01/manifest.json`:
  source-isolated preparation/sweep/publication census, not performance timing.
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
