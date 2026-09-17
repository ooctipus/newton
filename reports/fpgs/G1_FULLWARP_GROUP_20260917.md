# G1 four-full-warp ownership screen

Experimental, default off, unpromoted and closed as a diagnosed cost loss.
Native controls passed on both GPUs, but the first whole-task RTX screen
loses. No performance gain or accepted runtime change is claimed.

## Scope and cost hypothesis

Base `a61ea916ab55ee83109accf1a0360a02f7e83f7f`, isolated branch
`ooctipus/fpgs-g1-fullwarp-group-20260917` at
`/home/octi/Projects/newton-fpgs-g1-fullwarp-group-20260917`.

`FEATHER_PGS_SPARSE_FULLWARP_GROUP=1` groups four independent complete 32-lane
worlds into one 128-thread CTA for the existing spectral GS and parallel limit
prefix. It changes no row mathematics, eight-pass allowance, global layout,
factor, contact producer, CFM convention, or publication. Prefix and solve
are both replaced; no old producer, conversion, or additional launch remains.

The CUDA group is `pack*4 + (threadIdx.x>>5)`, guarded against the plan group
count before the articulation lookup. Each solve warp has its own du43,
lambda100, cross100 and ready4 shared slice. All reductions and synchronization
remain the original full-warp collectives. CPU prefix retains its original
one-world serial fallback; production CUDA launch is ceil(world_count/4)
blocks. Factor refresh remains its original 128-thread owner with CTA barriers.

Actual context-free device attributes reported RTX 188 SMs/24 blocks/48 warps
per SM and GB 152 SMs/32 blocks/64 warps per SM, with 65536 registers per SM.
Current strict RTX owner costs are GS 3.221639 ms and prefix 0.560013 ms. At the
recorded original registers, grouping's optimistic residency-scaled estimate
is GS 24-to-32 warps (0.805410 ms) plus prefix 24-to-40 warps (0.224005 ms):
1.029415 ms combined, not a prediction or a 1.2 ms claim. CTA resources remain
occupied until the slowest of four worlds finishes. GB has no corresponding
solve block-cap benefit and may lose register residency.

## Source and compiled resources

- `sparse_fullwarp_group.py`:
  `54bffa40547910d6d265d28a4d9b00206ce371c63662451971ff4760fa037f1b`.
- `sparse_factor.py`:
  `d58d4c2176b124f5e28d3580e81433c52c83e9d7cb7cd8b3a1c79016321a687b`.
- `test_sparse_fullwarp_group.py`:
  `9357159014a06547151355d3a101957b4e0d8af9fdec147b55d28de4fac85315`.
- Root observer `checked_sparse.py`:
  `85afa8cd5705af46d5056093bba44bf8900af912982483f1ebc252351adda107`.
- Root runner `run.py`:
  `ef8b01dc6588a6a9eb1c0c503eaec446d05710293bfc4fbf3d58ee29798a408f`.

Original row, spectral, and metric numerical source files remain unchanged.
Exact block-128 hidden-device AOT report:
`/tmp/fpgs-fullwarp-group-offline-Zaw7en5f/offline01/report.json`, SHA256
`dc803bd5ac7b2feed419f7b5fcee0f7361bb06433dc846d7977a6ad387e33b69`.

| Owner | SM120 registers/shared B | SM103 registers/shared B |
| --- | ---: | ---: |
| Spectral GS `_w4` |64/4464|66/4464|
| Parallel limit prefix `_w4` |48/512|40/512|

All four compiles have zero stack, spills and CTA barriers. RTX GS preserves
the proposed 32-resident-warps register bound. GB's 66 registers may round to
a 72-register allocation and reduce the solve bound from 32 to 28 warps.

## Controls and whole measurement

The missing-module regression failed before implementation. Three new CPU
controls and five existing spectral/prefix controls pass; scoped Ruff and
diff checks pass. Root reported RTX native 4/4 pass (session 94295), covering
world counts 1/3/4/5,
nonidentity group/art/world maps, empty/100/101-row worlds, neighboring
valid/status exits, clipped prefix capacities, stable prefix ordering, graph
replay, and existing current/held independent-J/H saved-16 controls. Root's
observer/CLI 4/4 controls pass, including exact factory identity and rejection
of false markers. RTX native evidence is root's session 94295 terminal
transcript, not a filesystem log; no log digest is claimed.
GB native 4/4 passes in 3.183 seconds (root session 52987, exit 0):
`/tmp/fpgs-fullwarp-group-native-Y5sw8c/gb.log`, SHA256
`dc9186ec91a706e27174eb87737e2dce32fd49bd1f8454e73d027ec2287db445`.

The frozen first whole RTX A/B run is root session 11690 at
`/tmp/fpgs-g1-fullwarp-group-whole-rtx16k-20260917-01`. It uses the original
source-pinned whole harness and explicit grouped 0/1 arms. Unrelated external
jobs limited initial GPU availability; none was interrupted. No GB whole
screen is funded following the RTX loss.
The run completed with source/idle/capacity checks passing: physics
13.144145525 -> 13.961268150 ms (0.941472177x), wall
26.772528101 -> 27.267863124 ms. Manifest SHA256:
`0ef952f3ada44b122241d3c6a464f68052c03594e814b07305539a4cae6ba1d2`.

The authorized diagnostic reuses the strict shell/CSR/spectral reader at
`/tmp/fpgs-g1-shell-csr-spectral-strict-R80XhYQP/read_shell_combined.py`
(SHA2569952ff395df373b63a4c400facc6558361ffb23897fb6a1048ccc5106ce77bb5).
Its minimal external adaptation is
`/tmp/fpgs-g1-fullwarp-strict-3xS3kf6f/read_fullwarp.py`. Candidate node capture
`/tmp/fpgs-g1-fullwarp-group-node-candidate16k-20260917-01` completed simulation
and capture; parent exit 1 is the preserved known auxiliary-graph analyzer
rejection, not a simulation failure. The strict reader passes source/capacity,
process/correlation scope, 48 physics plus 12 auxiliary roots, zero unproven
nodes, 8 calls each `_w4` owner, block 128/grid 4096, and old-owner absence.
Output `candidate_gpu0.json` SHA256
`d9729fdb339c4236bb04dbed39691a170fcbc5f03582e0b5ba9098d121ed80c6`.

Against the earlier source-matched a61 strict baseline (NOT paired node timing):

| RTX owner | Original ms | Grouped ms | Delta ms |
| --- | ---: | ---: | ---: |
| GS |3.221639|4.048535|+0.826896|
| Limit prefix |0.560013|0.520920|-0.039093|
| Refresh |1.780706|1.777397|-0.003309|
| Contact triplets |1.442321|1.436210|-0.006111|

Changed-owner gross +0.787802 ms accounts for the whole +0.817123 ms loss.
Actual RTX grouped GS resources match AOT 64 registers/4464 B shared; no
register growth explains the regression. Both node cohorts have all worlds
nonempty, but current row count 384309 versus prior 385818 and trajectories differ;
this is ownership attribution, not a controlled component performance claim.

## Cause evidence and remaining limitation

The existing older coherent spectral8 CPU-FP32 work arrays at
`/tmp/fpgs-g1-spectral-stop-population-6jOtK9kO/gpu0.npz` (SHA256
`19bcd80bc33be44427f2aa51e0a32f1261124d3fb5629b097acf3bbc2352bf1d`)
contain actual per-world work for 329956 rows. Their group-to-world map is
identity. Consecutive groups of four give `4*sum(group_max)/sum(world_work)` of
1.487 for row counts, 1.561 for transactions, 1.633 for row-plus-cross dots,
and 1.623 for their products. Dot-work utilization is 61.25%, equivalent to
19.60 useful warps from 32 reserved slots under this simplified proxy.
GB older-cohort dot inflation is 1.627. The simple RTX 24/32 residency ratio
times 1.633 gives 1.224x work inflation, similar in magnitude to the nonpaired
observed GS 1.257x loss.

This supports a straggler mechanism but does NOT measure stalls or prove the
distribution on the current shell/CSR cohort: current captures contain only
aggregate row counts, not per-world work. A persistent independent-warp queue
could refill a finished warp without waiting for its three neighbors. It
would add a per-solve counter reset, atomic work claims, repeated world guards
and dynamic indexing, and must retain all original per-world scratch resets.
Root declined that correction: the original residency-only estimate is about
0.805 ms before queue costs, and no evidence establishes at least 1 ms of
whole-step saving from that correction. No queue correction, mapping
grid, further numerical policy, or hardware-bound claim is made. Preserve
this attempted scheduling path without promoting it.
