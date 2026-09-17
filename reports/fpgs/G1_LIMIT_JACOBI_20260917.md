# G1 limit-prefix Jacobi: bounded native cost screen

Status: default off and unpromoted. The first native implementation lost
0.180 ms whole RTX. This is one fixed limit-only iteration change, not a
contact Jacobi solver. A response-representation correction is proposed below.
Base: `a61ea916ab55ee83109accf1a0360a02f7e83f7f`.

## Complete ownership and numerical scope

`FEATHER_PGS_SPARSE_LIMIT_JACOBI=1` requires the existing spectral-tangent
owner. `SparseFactor.limit_jacobi` records actual ownership. The cached no-arg
factory produces `sparse_spectral_limit_jacobi43_s18_c100`, with the unchanged
15-argument spectral ABI and one 32-thread warp per world. Off mode keeps the
original spectral factory. No global allocation, launch, producer, factor,
decode, contact transaction, maximum-eight policy, or public output changes.

For each outer pass, all admitted prefix rows read the same old kinetic
velocity. Eight four-lane groups propose projected scalar updates; each group
uses two shuffle stages after its strided sparse dot. CFM remains only in the
proposal denominator. Using the original signed coordinate column of W,
node-owned deterministic gathers construct the complete proposed kinetic
response. With physical residual r and impulse change delta, admission is

```
energy = r dot delta + 0.5 * norm(delta_du)^2
scale  = abs(r dot delta) + 0.5 * norm(delta_du)^2
accept exact-zero delta, or finite energy <= -(64 * 2^-24) * scale
```

This is a scale-aware numerical descent guard, not a formal rounding-error
certificate or a physical stopping test. Two scalar warp reductions, proposal
votes/fences, indexed W gathers, and 688 bytes of shared scratch are charged.
No Gram matrix is formed. Rejection leaves lambda/du untouched and executes the
original serial prefix in the same outer pass. General omega != 1 also keeps
the serial prefix. Incoming impulses are not reapplied: only their delta
changes velocity. All contact fragments remain exactly the spectral source.

The immutable admission checks identify source-produced signed W columns from
the first support node and coefficient. This uses the existing limit-producer
contract; it is not an independent coefficient-by-coefficient validator for
arbitrary external row buffers. Capacity 86 is scratch capacity, not a changed
production row limit. The 86-row test is synthetic capacity stress (37 scalar
coordinates permit at most 74 distinct signed limits); 87-prefix rows fall
back to the unchanged solver.

## Frozen CPU screen before implementation

External card and artifacts: `/tmp/fpgs-g1-limit-block-card-GowrgB7P/`.
The helper was frozen before candidate evaluation. Saved sixteen current/held
physical cases passed finite/cone/momentum checks; their endpoint differences
were FP64 roundoff. These alone were weak interaction coverage.

The additional sample froze IDs before evaluation: 64 strongest normalized
limit correlations, 64 largest original active-limit counts, and uniform 512
worlds per archived card, deduplicated to 635 RTX and 633 GB worlds. Inputs are
the existing full-16K original archives, not a new current/held capture. All
1,268 cases passed finite/cone/captured-operator momentum checks. There were
8,554 attempts, 3,136 zero steps, 5,418 nonzero admissions and no rejection;
the scaled energy range was [-0.919605, -0.0411981]. Original/candidate outer
counts were 8,489/8,554. Finite-eight numerical tradeoffs remain: worst normal
residual increased 1.76001602 to 1.76238176; maximum H-scaled velocity difference
was 0.00264837. This is not a new converged reference or full qualification.

Uniform-512 source-path counts estimate removal of 77.1--77.2% of prefix
shuffle/add stages including the two guards, or 34.4% of all old sparse-dot
stages. Prefix fence counts change 74,661 to 6,850 RTX and 74,148 to 6,832 GB.
These are prospective native instruction-path counts on CPU iterates, not
measured GPU instructions, timings, or literal dense NumPy product counts.
The existing approximately 0.560 ms limit producer is retained. The cost gate
is a measured whole RTX saving of about 1 ms; the approximately 3.22 ms old GS
owner would need to fall to approximately 2.22 ms with other owners unchanged.

Important immutable artifact hashes:

- CPU card: `f2445ed73fbf3fbbecf6406bf33d5fa6813d7c30609339c7fd0efaa05a2a68ad`.
- Original helper: `4b79557f248c85f610face864417059ed55aa8d86302df6650aadc292bfb4364`.
- Predetermined IDs: `9999d7037589169ddb3cb744fc27b8f0a13a9966fb8b6724e4ea60d75fd0be2b`.
- Interacting result: `d04fac1c9b98d248ce232a3409e15720e4eb70499290ca9c40cd23b80e98b216`.
- Work ledger: `c88a3a950e66c921e4648b7f047198f1fa758842e451d2a630faa684b196f04e`.

## Implementation controls and frozen native checkpoint

The missing-module regression failed first (session 32398, exit 1). Final CPU
session 23094 passed eight controls, with three CUDA controls skipped. This
includes four real-owner observer controls, not fabricated owner attributes.

Native selectors in `TestSparseLimitJacobiCUDA` cover prefix sizes 0/1/13/86/87,
signed columns, incoming impulses, late contacts, omega != 1, zero iterations,
invalid incoming/metadata fallback, current/held rows, graph replay,
empty/regrow, and the original saved-sixteen physical J/H checks. Genuine
energy rejection uses W=I with W[42,0:3]=2: three proposals of one have energy
+4.5 and must roll back to the original one-pass [1, 0.2, 0.04], not [1,1,1].

GB correctness-only native execution passed all three selectors, root session
33574 exit 0. It ran under unrelated external GPU load, so its elapsed time is
not performance evidence. Log:
`/tmp/fpgs-limit-jacobi-native-0v3V7vRK/gb-shared-correctness.log`, SHA256
`bf92a5301db68aed172870e4c83528c5eceb73820adc37e5e192ee42077a16e5`.
RTX native also passed all three selectors, root session 43847 exit 0. Log
`/tmp/fpgs-limit-jacobi-native-0v3V7vRK/rtx.log`, SHA256
`7ff54cbe8bed58e819e993298cc54fc2b701a787772a1049cbf276c98cbe9b81`.

Hidden device-free compilation also passed. Reused adapter:
`/tmp/fpgs-limit-jacobi-offline-LT9vM33W/compile.py`; final result:
`/tmp/fpgs-limit-jacobi-offline-LT9vM33W/offline02/report.json`, SHA256
`057a6e3c1920f704b3483279b89d27248e4e57e88aeb3a36bcd84781b9bd7582`.

| Owner | SM120 registers | SM103 registers | Shared bytes | Stack/spills |
|---|---:|---:|---:|---:|
| Original spectral | 64 | 64 | 1,116 | 0 |
| Limit proposal | 62 | 61 | 1,804 | 0 |

These are complete-owner compiler resource reports, not occupancy or stall
counters. Frozen source pins:

- Runtime `sparse_limit_jacobi.py`: `d806a20de5d7a637b31de0f164cf43ad9d981e0f30fe55f062105f0aa5b497db`.
- Binding `sparse_factor.py`: `4c94ee8a3b2f3da3c98641c07db55c634421c5319a6cc382e1ba47efe4b3bf16`.
- Focused tests: `288bb0144847f3230e6ad11e0200cf90def8609f0395f82bc8e68ca4b359348a`.
- Local CPU control: `a9f1bc22712cb716ce6d5c403e9b44cafc389e67d65d99173d5b039a970aa38d`.
- Whole observer: `33768081c67a994672bdd7e7c5990f083f7a0c410bca920142b772eb7f1a941b`.
- Whole runner: `c8439e7b424b825d184c743288c16acf8a7a6dde78cee5f7cc39f11d08336da5`.
- Observer tests: `af75f9cccf4bf392b853930c2f21dab77f4745de4077bf2f2daede36db76a034`.

## First measured closure and single proposed correction

Whole RTX root session 52829 completed with all source/idle/capacity guards:
13.190505650 to 13.370332925 ms, ratio 0.986550, a 0.179827275 ms loss.
Artifact: `/tmp/fpgs-g1-limit-jacobi-whole-rtx16k-20260917-01`, manifest SHA256
`9b5f968ba5428fd58c6898933cbffebf5d334760914087ce2188b93e3d234833`.

The following source-frozen node run preserved its original auxiliary-analyzer
exit 1: `/tmp/fpgs-g1-limit-jacobi-node-candidate16k-20260917-01`. External
strict reuse `/tmp/fpgs-g1-limit-jacobi-strict-PvgHD2g3/read_limit.py` passed
48 physics roots, 12 separate auxiliary roots, zero unproven graph nodes,
process correlations, exact owner/cadence, model budgets, capacities, and
manifest/current source and idle guards. Output `candidate_gpu0.json` SHA256:
`fa66ad2ee902eb3542eb0bb8209da3cfeaa4c5b328ddf3418e6ca5f74e085d26`;
reader SHA256:
`25654e62120f71965ae8656d1cf4875bfeeaf66388e5d8bf7b04f401d42f6b7f`.
The reader's first attempt rejected an incomplete expected observer dictionary;
adding its actual exact `kernel_key` field resolved the reader defect without
relaxing any capture guard.

Compared with the existing a61 strict capture, GS increases 3.221638750 to
3.390996333 ms (+0.169357583), while collision is 2.330610 to 2.338094, rows
2.782465 to 2.768694, publication 1.842556 to 1.842952, and dynamics 2.663408
to 2.657261 ms. The new GS owner has exactly eight calls, block32/grid16384,
62 registers, 1,804 shared bytes and zero reported local memory; the original
spectral key is absent. This isolates the loss to the replacement solve owner,
not a new producer or inflated invocation cadence. It does not identify a
specific hardware stall.

The one proposed structural correction retains the fixed iteration/guard and
uses actual packed Z to accumulate response into shared step[43]. It removes
the signed W-column metadata and irregular 43-node index probes, replacing
them with at most 18 contiguous coefficient products per changed row in original
prefix order and one warp fence per changed row. Shared scratch becomes 516
bytes (delta[86] plus step[43]) instead of 688. Two scalar energy reductions,
all proposal work, response finite checks, and transactional fallback remain.
The existing sample has 2,789/2,768 changed rows over 3,425/3,416 candidate
sweeps: the additional changed-row fences average approximately 0.81 per sweep.
This is a source-directed cost hypothesis, not measured recovery or a changed
numerical policy. The initial source and native results must remain recoverable
before this correction is implemented.

Full-repository and targeted precommit checks, focused CPU controls, and diff
checks passed before this initial measured checkpoint.
