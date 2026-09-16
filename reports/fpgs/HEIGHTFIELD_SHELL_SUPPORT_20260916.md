# Buffered heightfield shell support — 2026-09-16

## Decision

Default-off, unpromoted experiment. Bounded geometry and loaded eight-sweep
controls pass on both cards. One complete G1 discovery round saves only
0.137957 ms on RTX (1.009250x), but 3.254540 ms on GB300 (1.196463x).
This is not an accepted RTX gain or a repeated performance result. Further
qualification and a same-collision, corrected MJWarp comparison are required
before any later GB promotion; inherited accepted references remain unchanged.

## Representation and scope

`NEWTON_HEIGHTFIELD_SHELL_SUPPORT=1` pairs the existing analytical finite query
with a replacement buffered reducer. Positive-shell top-face witnesses no
longer require the conservative generic face fallback. Spatial support slots
admit witnesses within the unchanged live gap-plus-margin envelope; their
priority uses the exact legacy inner predicate
`depth < 0.0001 * length(shape_a_AABB_diagonal)`. Depth is geometric signed
distance, not margin-subtracted clearance. Spatial eligibility is separate
from priority, preventing an out-of-envelope legacy-inner witness from hiding
eligible support. Every eligible inner outranks every outer in the same
shape-pair/normal bin. Deepest and voxel competition remain unchanged.

The new encoding is used consistently for analytical and generic fallback
witnesses of an admitted pair. It is restricted to heightfield versus box or
immutable cuboid convex hull with matching source identity, positive current
scale and the existing finite-query margin admission. Unsupported/refitted
sources and sphere/capsule pairs retain the old encoding. Direct/unreduced
queries retain their original fallback. The off-mode factory identity is
preserved; no buffer inflation, additional launch, changed gap/material,
modified flat-seam writer, solver allowance or Lab edit is introduced.

The priority packing sacrifices one low score bit; bit identity is not claimed.
One inner witness can suppress all outer spatial winners in its normal bin.
Thus this policy fixes the all-positive support hole, not every possible mixed
manifold or finite-iteration ordering. The loaded controls remain necessary.

Earlier analytic-manifold and direct-speculative-face experiments retained the
old reducer or original manifold ownership. They did not implement this paired
finite-shell support policy. Their losses and prior rebound failures remain
preserved; this report does not retroactively qualify them.

## Source and setup pins

Isolated branch `ooctipus/heightfield-shell-support-20260916`, base
`f22128016220fb4a4424b09f257833bef893a652`. Whole and node captures used frozen
source-diff digest `0a310dfefe9af08d7abedd9f7840c99231ccbce328138ff9f36f74b3fff84a80`.
Baseline is qualified-chain `ef9481bcc7b17fc9496f28a206ab6d0c72331507`; Lab is
`53ee6b44c2334341305dbdf385a3916c6b140799` and benchmark checkout is
`961b7e2f751bcd1d8b03368e7956b54c81414897`.

| Frozen file | SHA256 |
| --- | --- |
| `contact_reduction_global.py` | `35f853f43e1e97c3f8574e6f01215f261f29fc8ac021d7a0de972952384220d8` |
| `heightfield_finite.py` | `3c7eab8073ad804209d5377e7e19a46fd7360ed88c77691f9a12aa60e8208da4` |
| `narrow_phase.py` | `0116b08ba8f10bc711b592296ee0a9901e974dad9d9d8b499fd0d8e70adfaadf` |
| `test_heightfield_shell_support.py` | `50efa9998fd153ac4413bd4ac23af6fda1f38c138996939dadc8a387cc8426f7` |
| `chain_capture_20260916/checked_sparse.py` | `c4ceb84cf2a9d7e76aa8f479ebe8ef46d9df0857f8e852aed59df7ed998a6fb0` |
| `chain_capture_20260916/run.py` | `509bff7da1b873cac1a66d0ab90dbbb4fdfc7438be23592a7122b47a58badcb8` |

The last two files are root-owned minimal observer/parser changes. Actual
query/reducer keys are `heightfield_finite_shell_contacts` and
`reduce_heightfield_shell_contacts_kernel`; both checked boundaries verify the
paired owner. The existing runner only gained the Boolean flag whitelist.

Initial native logs `/tmp/fpgs-shell-native-20260916-x80fqRsw/gpu{0,1}.log`
preserve two geometry passes and setup-only failures in every loaded case:
the tiny width-one articulation defaults to two lanes without the historical
recipe. No physics record was produced. The identical historical failure is
documented in `RESULT_A01_SETUP.md` under
`/tmp/fpgs-heightfield-finite-qualification-J9kBvfGP`. Retrying with the original
`FEATHER_PGS_GROUP_LANES=16`, `FEATHER_PGS_ROWS_MASKED=1` and
`NEWTON_NARROW_PHASE_THREADS_X=4` required no physics or test modification.

## Physical evidence

Regression-first CPU checks reproduced missing positive-shell spatial support
and the out-of-envelope suppression before the fix. The seven shell tests plus
seven retained finite-query tests then give 13 CPU passes and one CUDA skip.
Targeted pre-commit passes. Independent saved96 geometry/lifecycle CPU checks
also pass; the matrix-free loaded fixture remains CUDA-only, not silently
switched to a different CPU solver.

Root's corrected native run executes all four selectors on each card, with no
skips or failures (14.752 s RTX, 15.119 s GB):

1. Shell mixed-inner/outer CUDA support.
2. Existing actual96 complete collision/query/lifetime controls.
3. Three same-state original/candidate reduced rebound pairs.
4. Existing reduced support, tilted-foot, sliding and finite-border controls.

In these inherited physical fixtures, `enabled=False` selects the generic
collision control; the whole-run baseline below retains the accepted
conservative finite query with shell support disabled.

Logs: `/tmp/fpgs-shell-native-recipe-20260916-tFKrSCm2/gpu{0,1}.log`, SHA256
`75bcdc7df49617cdf18d15a02df70b59c74d9a913668460b76754655ee574ed3` /
`c8dd4caebdf29b081068ad8ec5477d1060522c101144235798069cb768262917`.
The original qualification source is unchanged `bd6d47727eeee2bb29a9f7e113b9c4282373b4b730e35f9179f38964c7265762`;
rebound collector is unchanged `05eea5063b43fcc71ab99c9366b36beb16b85d1ec19e19718c585d66bdf9ec2d`.

Collector exit zero alone is not rebound acceptance: its test asserts collection.
The final `QUALIFICATION_REPORT.prior_qualification` retains all six rebound
records per card. Post-return `J_world` is cleared by original maintenance, so
its stored residual is invalid. Read-only reconstruction uses `H=L L.T`,
`J=Y H`, and physical residual `r=J v_out+rhs`, with denominator-only CFM absent
from the numerator. All held R and friction coefficients are zero here.

| Maximum across three repeats | Original, each card | Shell RTX | Shell GB |
| --- | ---: | ---: | ---: |
| Public spin, rad/s | .009843941 | .001921619 | .000005742 |
| Negative normal residual, m/s | 2.82504e-4 | 5.32622e-5 | 4.32877e-7 |
| Absolute normal complementarity, `abs(lambda*r)` | 9.74995e-4 | 4.47002e-4 | 2.28740e-6 |
| Absolute foot-corner restitution residual, m/s | 1.40012e-3 | 2.50225e-4 | 8.76846e-7 |

Candidate energy ratios are .360000076–.360027305 for authored restitution
squared .36. Candidate normal speeds are 1.800000191–1.800068259 m/s against
1.8. Candidate rebound has seven contacts/21 rows versus four/12 in original,
and marks all32 current triangle entries. No large one-sided rebound defect is
reproduced in these repeats; this is not an all-order or long-trajectory proof.

Across all12 records, reconstructed normal J agrees with independently formed
raw `[n, (p-COM) cross n]` within7.451e-8; diagonal agreement with `J dot Y+CFM`
is within1.70e-7. Exactly eight FP64 replay sweeps reproduce actual `v_out`
within4.863e-6. Held impulse closure is at most2.981e-7, public angular-impulse
closure3.286e-8 and scaled linear momentum error1.152e-7. No gate was widened.

All eight loaded records per card have empty original failure lists. Candidate
support's complete80-step tail has relative mean-weight errors1.007e-7 RTX /
4.929e-8 GB, zero duration beyond the original2% force band, and momentum
closure below5.05e-8. Sliding and finite-border release pass unchanged gates.

## Whole cost and strict attribution

One round,16,384 environments, seed0, warmup200, wall40/profile40; same raw
capacity294912, broadphase49152, triangle1769472, dense100 and MF32. Original
eight sweeps, two substeps, dt.0025 and all other flags remain fixed.
All four whole children return0 and source/idle/capacity/finite checks pass.
Manifest `/tmp/fpgs-g1-heightfield-shell-whole-paired16k-20260916-01/manifest.json`,
SHA256 `b1531d47aaae3f1b86c54df41f1a1510991f6e4ed812907e0d069f8362cc6ab0`.

| ms/environment step | RTX original | RTX shell | GB original | GB shell |
| --- | ---: | ---: | ---: | ---: |
| Whole physics | 15.051505 | 14.913548 | 19.820232 | 16.565692 |
| Wall environment stepping | 29.681906 | 28.669082 | 35.422072 | 31.058402 |

Wall stepping is not full RL. These single-round physics ratios are1.009250x /
1.196463x, not repeated promotion evidence.

Node capture `/tmp/fpgs-g1-heightfield-shell-node-candidate16k-20260916-01`
preserves the expected legacy auxiliary-analyzer exit1. The strict supplemental
reader verifies all48 physics/12 auxiliary roots, every process/correlation,
source and original capacity, both shell snapshots, and zero unproven nodes.
External reader `/tmp/fpgs-g1-shell-strict-ArrDQjFl/read_shell.py`, SHA256
`e6950d83f6b04eb33f2e0bf00e13c36d57129bc8abadc1204a7418a2bc84474f`,
only extends the pinned existing chain reader's exact aliases and sub-boundaries.
Its result hashes for gpu0/1 are
`8acf1ccd29912493e22239bfea2035ec9c79fa8094890771dfd6cac0df8805b6` /
`1b03983b35e09a79bf5b358605ba955788a55e06ff01522fd8d0d7d087f0d4db`.

The comparison below uses the preserved qualified-chain node reference
`/tmp/fpgs-g1-chain-vector-strict-Os6BOg4H/candidate_gpu{0,1}.json`, not a fresh
paired node baseline. Family values are exclusive interval ownership; individual
owner values are node sums. Neither is substituted for whole throughput.

| Owner/family ms per environment | RTX original | RTX shell | GB original | GB shell |
| --- | ---: | ---: | ---: | ---: |
| Collision, exclusive | 3.743738 | 3.044506 | 8.378876 | 4.527282 |
| Finite query, summed | 1.388055 | 1.220585 | 4.011024 | 2.809747 |
| Generic fallback, summed | .823625 | .179459 | 2.924101 | .213256 |
| Buffered reducer, summed | .201310 | .286611 | .289188 | .342178 |
| Row family, exclusive | 2.527910 | 2.793490 | 2.554350 | 2.824500 |
| Contact-triplet producer, summed | 1.251471 | 1.494044 | 1.342003 | 1.616136 |
| GS, exclusive | 3.957767 | 4.220966 | 4.421050 | 4.638986 |

Old query/reducer owners disappear; new owners each execute four times per
environment. Original GS remains eight calls; dynamics and state publication
are essentially flat. RTX collision saves .699232 ms, while row preparation
and GS add .528779 ms; the reducer's .085301-ms increase is smaller than the
downstream increase. GB collision saves3.851594 ms while rows/GS add .488086 ms.
These ownership deltas explain the direction of complete timing without a
hardware-bound claim or a summed-node whole-time counterfactual.

Whole checked endpoints show RTX contacts77,558/78,072 becoming95,909/95,618;
total rows328,268/329,161 become370,823/370,534. GB shows a similar increase.
These are two current boundary samples, not time-averaged populations. More
support witnesses impose real downstream work. Automatic hull-interior removal
is not justified: frictionless constraint redundancy does not establish general
pointwise Coulomb/torsional equivalence or unchanged finite-eight convergence.

Query register counts decrease139→130 RTX /133→131 GB; reducer counts rise
42→47 /55→56, with unchanged launch dimensions/shared storage and reported
local memory0. No occupancy, bandwidth or stall cause is inferred from these
resource facts alone.
