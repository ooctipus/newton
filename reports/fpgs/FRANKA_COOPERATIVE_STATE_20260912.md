# Franka complete private owner: phase diagnosis and cooperative retry

2026-09-12, 20:27 UTC. Report-only successor; accepted runtime is unchanged.
The cooperative successor passes the bounded physical controls and reduces the
measured private-owner cost, but still misses the entire proposed replacement
allowance. Neither experiment is integrated, promoted, or a whole-physics gain.

## What the completed phase diagnostic established

The frozen predecessor is `private_state.py` at `10f3f87f`. Its source-identical
clock clone preserves eager/graph outputs and read-only inputs, but materially
perturbs resources and elapsed time:

| GPU | Original owner | Clock clone | Probe overhead | Registers / resident CTAs per SM |
| --- | ---: | ---: | ---: | --- |
| RTX PRO6000 | 2.596864 ms | 2.941056 ms | +13.254% | 85→107 / 20→16 |
| GB300 | 2.650432 ms | 3.232000 ms | +21.942% | 79→106 / 24→16 |

These are eight-substep-equivalent component times. All forty paired probe
deltas are positive, including both AB and BA orders. Shared memory remains
3712 B, local memory 72 B, and the block size 32. Phase cycles therefore support
**qualitative dependency analysis only**; multiplying cycle fractions by the
original elapsed time does not price a replacement or establish savings.

Current input/force, free6 prediction, prefix preparation/solve, and next kinetics
all remain substantial. Reuse primary-factor work is only about 1% of summed
selected-world phase cycles. Most worlds qualify for the rank case; the measured
population does not run a twenty-row eight-sweep fallback everywhere. Rare
fallback worlds nevertheless dominate much of the reuse tail. The independent
audit retains every row/rank stratum and does not dismiss that tail.

## One complete successor, with two important corrections

The fixed32-thread successor caches current force geometry while producing the
previous step's next geometry, cooperates on current validation and free6 held-L
action, compacts the current sparse prefix cooperatively, and builds complete
prefix responses only when the signed rank case fails. Arbitrary current body
wrenches, controls, passive terms, held epochs, the original complete eight-sweep
fallback, all21 generalized coordinates, next public body state, and private
kinetics remain in scope. Two new force-geometry buffers are genuinely chained
and restored; there is no assumed zero-force shortcut or extra timed seed call.

The original mapping proposal used differences of global prefix sums for
subtree forces. Root rejected that representation: a huge force outside a
subtree can erase its small internal force before subtraction. The implemented
layout instead validates exact suffix-or-singleton masks, uses reverse inclusive
suffix sums for the arm, and each finger's original own wrench for its singleton.
The same rule applies to next bias. A native CPU/CUDA negative control shows that
`1e20` outside must not erase `1.25` inside; the deliberately wrong prefix-
subtraction oracle returns zero and fails. Unsupported masks reject at binding.

The authorized sum reassociation also makes bit-identical next bias an invalid
requirement. A checked source adapter replaces **only** the predecessor's exact
comparison to its frozen producer with a reported finite scaled delta. The
independent physical predicate remains unchanged:
`isfinite(bias_error) and bias_error <= max(3e-6, original_bias_error)` at the same
candidate next state. The existing current-force, held-H/action, eight-sweep,
momentum/residual/bounds, public-state and epoch gates remain unchanged. Source
recovery proves that narrow change; a finite wrong-bias negative still fails.
This is not a blanket relative-tolerance relaxation or a new convergence claim.

## Completed numerical gate

Both root-owned physical GPU suites pass: one actual CUDA method per GPU, zero
skips, failures, errors, expected failures or unexpected successes. Together
they cover both actual512 refresh/reuse pairs, changed force/torque/COM/gravity,
full20 and zero/one-row prefixes, original finite-eight controls, free limits,
raw/epoch/late rejection without partial next-state commit, poisoned primary
canonical caches, next force geometry, and two-state graph replay. The added
huge-outside-force and stale-pose/nonfinite-prescribed-q controls also execute on
the actual GPU. These remain finite input/operator/state-transition controls,
not a full rollout certificate.

Across the fourteen printed physical cases, current physical-force scaled error
is at most 1.087e-6; next geometric-H relative error at most 2.189e-6; original-eight
velocity difference at most 1.228e-7. Next-bias scaled error reaches 3.016e-6 in
the changed RTX case, versus 1.699e-5 for the original same-state law, satisfying
the unchanged non-regression predicate. The full20 stress case retains original
finite-eight tails: maximum reported bound violation is 0.01123 and
complementarity measure 0.02968. A passing suite does not mean those tails vanish.

## Cost: improved component, still outside the complete allowance

| GPU | Prior private owner | Cooperative owner | Separate-window reduction | Excess over entire 1.634704 ms allowance |
| --- | ---: | ---: | ---: | ---: |
| RTX PRO6000 | 2.588799953 ms | 2.048000097 ms | 0.540799856 ms / 20.890% | 0.413296097 ms |
| GB300 | 2.642175913 ms | 2.245120049 ms | 0.397055864 ms / 15.028% | 0.610416049 ms |

The old/new numbers are separate-window measurements of the same expanded
private-owner scope, not an interleaved old/new whole-physics experiment. New
two-owner medians are 0.512000024 / 0.561280012 ms; the exact multiplier is four
for the unchanged eight physical substeps, dt=1/240, eight PGS sweeps, and zero
velocity-only iterations. Forty samples follow ten warmups. Each GPU's actual512
scene pair is materially replicated32 times with rebased independent storage,
not modulo-indexed and not a fresh16K collision trajectory. Selected counts stay
15,552 RTX / 15,424 GB per owner. The second owner consumes the first's actual
next q/qd, bias and force geometry; held L/G/H epochs and refresh/reuse cadence
remain shared correctly. Eager/graph all-output and read-only guards pass.

The **entire** charged selected-family allowance is 1.634704 ms for the proposed
20% whole-physics reduction. The 1.20 ms owner allocation is only planning, not
an independent rejection threshold. This retry misses even the entire allowance
before actual free-L6 refresh, raw-count/demand processing, unresolved canonical
preparation, cold/reset/odd-refresh repair, cache reseeding and live fences.
Current free forcing/response, transport, integration and next free inertia are
already timed. Retained raw-contact/local20/40/general work outside the selected
allowance is not double-charged; its possible dilation still matters to any
future whole measurement. No integrated performance conclusion follows.

## Balanced matched repeat

A subsequent alternating old/new run confirms the component improvement in
one measurement window, on the same two saved states and independently allocated
original/candidate bundles. Forty measured pairs contain twenty AB and twenty BA
orders. Every mutable input/output is restored outside every event; complete
eager/graph hashes, immutable inputs and all source/UUID/idle guards pass.

| GPU | Matched predecessor | Matched cooperative | Component speedup | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| RTX PRO6000 | 2.596863985 ms | 2.072576046 ms | **1.25296x** | 20.189% |
| GB300 | 2.650624037 ms | 2.259456038 ms | **1.17312x** | 14.758% |

These remain eight-substep-equivalent private-owner times, not whole physics.
The matched candidate exceeds the entire 1.634704 ms allowance by 0.437872 ms
RTX and 0.624752 ms GB before the missing services listed above. The substantial
component reduction is retained as experimental evidence; it does not establish
the proposed 20% whole-physics gain or the user's 4x-MJWarp objective.

The next structural study examines whether a complete direct rigid-body state
service can delete generic free-body preparation, factor refresh, transport and
publication work while preserving held rotational response and current forces.
No free-body shortcut or new runtime promotion has been accepted.

## Frozen reproduction evidence

All four parents are complete; both children in each exited zero, and recorded
source/UUID/final-idle guards pass. The Franka accepted-reference tree remains
`064ec8ac455fc4cde557a3b54a1a62624cf56441`; this report branch's runtime remains
`50dfa28d3aabe51f1b5b75721450efac2c35c5f1`. Isaac Lab remains clean at
`1d8feb82d17dbfab8f0772de56f84deae2cb7974`. No Lab code/config, dt, substeps,
iterations, public capacities or accepted runtime changed.

- Phase root: `/tmp/fpgs-franka-private-state-phase-paired16k-20260912-01/`,
  manifest `cb5559f494e9acfe268a66cedb625aa48ed347228b86b0e8e23239baa0ec1c5d`.
- Independent phase audit: `/tmp/fpgs-franka-private-state-phase-audit-979Auseg/`,
  evidence `7c82b0a263a57cea8818bfffe9dda5fc9721dd2c050f1ccb2423a27307ce9c85`.
  Its220 source pins and five input hashes were rechecked for this draft.
- Cooperative source: `/tmp/fpgs-franka-cooperative-state-FMucSlN7/cooperative_state.py`,
  `145718e55cd2f0d8413f17a1fe31b38de4a551c5f1d2eed8e2510bd5e6d9fc5c`.
- Root chain and physical adapter: `/tmp/fpgs-franka-cooperative-root-LuJCkuHQ/`.
  `pins01.json` is `b80cd8398858c8561bd2fc848a1d872a9241b7bb8c69c7739bcf3aaeac2992a9`:
  224 entries, plus the pin file itself in each225-entry result. All rechecked.
- Correctness root: `/tmp/fpgs-franka-cooperative-correctness-paired512-20260912-01/`,
  manifest `4153f44e468421df97f1710af3e0d3cfa389e7611e0e4311ea6e0d8021de95cc`;
  RTX audit `39f72f6c51c6213ce0a6a57def54f5a7017873b3f4b51979b09755dc58ee0805`;
  GB audit `1370deaddb0b062a8a39370fdbce747e98deb7710c94a7290a800c59bcd0fa2c`.
- Cost root: `/tmp/fpgs-franka-cooperative-cost-paired16k-20260912-01/`,
  manifest `2f67fdab6a3dca2cdaaff8d64b985dd64a921c602bafaf37133956582fa2c768`;
  RTX audit `78411280b85d8bef2eaa6aee387b2a50ad6b22858ebf5f1e005b08bb06836046`;
  GB audit `f9033b546c8bb504d7eddeaea75f67807742958a6eb727eab6d35857d5091c25`.

- Matched root: `/tmp/fpgs-franka-cooperative-matched-paired16k-20260912-01/`,
  manifest `cbaee838521e821ed4ec655af5a12c9d250f06ac6f690e9daf4c04d76a260e83`;
  RTX audit `a3b866524ea3c430c051281934b8190f8f36e8bebd6e8fd35b7931c3386cf67f`;
  GB audit `268fbdf55c7fc32d2a2003bd34adc4fae59fc42adf512ce7b2131c476dd587c1`.
- Matched wrapper: `/tmp/fpgs-franka-cooperative-matched-KEjVmQKM/run_matched.py`,
  `06795bbc98a1f52a35798ca1b841e615e11afde6ebf1bf19e715d26c245feeb9`;
  its `pins01.json` is
  `e1de5bc6c502115d669ea15d1e605b87373e0a2d5385ecb4700c5a32708dce9c`.
  The result explicitly checks 225 entries plus that pin file itself.

Read-only local integrity and raw event readback commands (no GPU launch):

```sh
jq -r 'to_entries[] | "\(.value)  \(.key)"' /tmp/fpgs-franka-cooperative-root-LuJCkuHQ/pins01.json | sha256sum --check
sha256sum /tmp/fpgs-franka-cooperative-root-LuJCkuHQ/pins01.json
jq '{environment_equivalent_ms,environment_multiplier,two_physical_owner_event_ms,eager_validation,readonly_unchanged,source_guard_pass}' /tmp/fpgs-franka-cooperative-cost-paired16k-20260912-01/gpu0/audit.json
jq '{environment_equivalent_ms,environment_multiplier,two_physical_owner_event_ms,eager_validation,readonly_unchanged,source_guard_pass}' /tmp/fpgs-franka-cooperative-cost-paired16k-20260912-01/gpu1/audit.json
```

The parent manifests preserve exact root-owned commands. Large captures,
experimental sources and generated binaries remain local. Further experiments
must price the complete changed boundary; no whole-physics or wall-time gain is
recorded here.
