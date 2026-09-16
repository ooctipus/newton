# G1 stage-once spectral rows: fixed-cohort experiment

Default-off prototype based on the clean shell-CSR/spectral composition
`a61ea916ab55ee83109accf1a0360a02f7e83f7f`. Enable
`FEATHER_PGS_SPARSE_STAGED_ROWS=1` with the existing spectral owner. No collision,
row producer, timestep, substep, capacity, maximum sweep count or physical law
changes. Native controls pass, but the whole-physics saving is only0.147ms RTX
and0.310ms GB300. The experiment closes default-off and unpromoted, below the
1ms RTX material-change target. No mapping, tile or instruction retry is funded.

## Exact boundary and cost hypothesis

For an actual world row count at most32, cooperatively stage its existing
18-wide Z rows, support nodes, incident, RHS, diagonal and friction into shared
memory before the first sweep. Pack type, validated tangent parent, support
length and the original immutable triplet-eligibility predicate into one word
per row. Incident and RHS remain separate with their original arithmetic
association. Current CFM remains the original lazy global read on first block
setup; the already cached spectral denominator/cross terms are not claimed as
new savings. Initial lambda, zero initial delta velocity, delayed friction,
normal-first proposal, scalar/sibling fallback, exact-unchanged exit and final
43-coordinate decode remain unchanged.

Worlds above32 rows execute the original generated spectral body, changing
only its shared-storage aliases. The initial mapping/status/type/parent checks
are common and unchanged. One3952-byte shared union holds either path; no
persistent array, row transpose or extra dispatch is introduced. The fallback
is charged the same kernel resource footprint. Staging also reads active-row
tangent coefficients that an original zero-radius visit might not touch.

The current shell-CSR/spectral fair-composition row histograms cover16384
worlds per boundary: RTX eligible worlds92.92–93.46%, eligible rows89.91–90.64%;
GB30093.10–93.12% and90.16–90.20%, respectively. These are from
`/tmp/fpgs-g1-shell-csr-spectral-fair-paired16k-20260916-01/round_01_g1_fpgs_gpu*/capture_checks.json`,
not a new census. They supersede the pre-shell composition's approximately94%
eligible-world estimate; the fixed32 threshold is not retuned. The earlier
pre-shell strict spectral GS owner costs3.182044ms RTX; the subsequently measured
shell-phase baseline is3.221615ms. Using the earlier budget, a row-proportional model
requires approximately1.54x eligible-owner speedup
to save1ms whole. This is a falsifiable cost target, not a prediction: original
loads may already hit L1, and reductions, dot products, projection and decode
remain. Shared staging plus immutable index/eligibility retirement is the only
funded change, with no mapping or capacity grid.

The older sparse-packet attempt also used shared Z, but fused its production
into the solve at capacity100, used9100 bytes shared and lost whole time.
This prototype preserves that experiment's faster separate producer and bounds
the staging cohort/storage. Present-ports, paired16 and field-major proposals
changed other representations or ownership; their closures are not evidence
of a staging win.

## CPU/source and compilation evidence

Missing-module regression6258 exited1 before implementation. Final CPU57625 exits0:
seven original/new tests pass, seven native tests skip. New tests establish
the unchanged15-argument ABI, actual flag/factory admission, source-identical
large-cohort body after alias substitution, retired fast-path global reads,
shared-storage bound, and actual CPU-owner observer checks. Missing flag,
wrong flag and wrong factory reject. Independent source review found no
concrete metadata, barrier, byte-node, sibling or incoming-state blocker.

CUDA-hidden AOT86833 exits0:

| Target | Registers | Shared bytes | Stack/spills | CTA barriers |
| --- | ---: | ---: | --- | ---: |
| SM120 | 63 | 4080 | 0 / 0 | 0 |
| SM103 | 64 | 4080 | 0 / 0 | 0 |

The original spectral owner has64 registers and1116 bytes shared. These are
static compiler resources, not measured occupancy or speed. The first AOT
failed only because Warp rewrites a C++ `int(...)` spelling; explicit
`static_cast` corrected that code-generation seam. Both attempts are retained
at `/tmp/fpgs-g1-staged-rows-offline-HUhkYWfG/`.
Successful report SHA256:
`9d2df17d1a6df276e600d38e000a37cae7920e58c00b0cda96247a41db7b7973`.

Native selectors reuse frozen spectral transaction/current-held/graph-empty
and saved16 independent J/H controls. One additional selector compares the
original spectral kernel with0/32/33/100 rows,18-coefficient support, incoming
lambda, delayed friction, mismatched-template scalar fallback, empty/regrow
and original count-overflow return. No new reference or physical tolerance is
introduced. Root alone runs these selectors and the unchanged paired capture.

```sh
env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
  FEATHER_PGS_GROUP_LANES=16 FEATHER_PGS_ROWS_MASKED=1 FEATHER_PGS_FUSED_K1=0 \
  uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python \
  python -m unittest tools.fpgs_bench.test_sparse_staged_rows.TestSparseStagedRowsCUDA -v
```

The existing checked observer/parent require explicit staged flags on both
arms, actual owner marker, exact factory identity and the unchanged spectral
CFM argument/layout. The measured composition stays untouched and uses flag0;
this isolated tree uses flag1 for the single whole-cost comparison.

## Native and whole-physics closure

Root-owned native sessions65354/23332 both exit0: all four selectors pass on
both cards, including the saved16 independent current-J/held-H checks and the
fixed-cohort boundary controls. Logs:
`/tmp/fpgs-g1-staged-rows-native-20260916-y7IhwGVf/gpu{0,1}.log`.
This validates the unchanged spectral transaction within the reused physical
tolerances; it does not newly establish convergence for every spectral case
or a full-cohort high-accuracy reference result.

The single paired16K whole run, root20145, completes with all source,
actual-owner, capacity and idle guards passing:
`/tmp/fpgs-g1-staged-rows-whole-paired16k-20260916-01`.
Both arms use the same shell-CSR/spectral composition and original budgets;
only the staged solve flag differs.

| Device | Baseline physics (ms) | Staged physics (ms) | Saving (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| RTX | 13.168334400 | 13.021441900 | 0.146892500 | 1.011280817x |
| GB300 | 14.882789850 | 14.573015150 | 0.309774700 | 1.021256734x |

The separately observed environment wall times are27.174752→28.218408ms RTX
(worse) and28.465651→28.074566ms GB300. This one-round physics improvement
does not meet the1ms RTX target and is not promoted or claimed as a repeated
wall-time improvement.

## Strict node attribution and interpretation

The candidate-only12-step node capture is
`/tmp/fpgs-g1-staged-rows-node-candidate16k-20260916-01`. Its original parent
exits1 at the preserved auxiliary-graph analyzer rejection; this is not a
runtime or capacity failure. The minimal supplemental reader is
`/tmp/fpgs-g1-staged-rows-strict-MZC6tvqu/read_staged.py`. It retains the complete
pinned shell→combined→CSR→chain→original interval-reader chain and adds only
the exact staged factory/source/flag/dictionary expectations. All48 physics
roots,12 auxiliary roots, process correlations, input hashes, source snapshots,
capacities, zero statuses and exact owner checks pass. It observes eight staged
solve launches per environment step and no original metric or spectral solve.
Root independently executes an exact reader copy at
`/tmp/fpgs-staged-root-replay-q2nNeKBN` and reproduces both outputs byte-for-byte.
An initial replay without a physical reader file failed its self-digest guard;
that failed invocation is preserved, and no guard was weakened for the replay.

The baseline is the already strict shell-composition capture summarized at
`/tmp/fpgs-g1-shell-csr-spectral-strict-R80XhYQP/candidate_gpu{0,1}.json`.
Exclusive owner times are:

| Device/owner | Baseline (ms) | Staged (ms) | Difference (ms) |
| --- | ---: | ---: | ---: |
| RTX GS | 3.221614750 | 3.060125833 | -0.161488917 |
| RTX rows | 2.782452000 | 2.781475333 | -0.000976667 |
| RTX collision | 2.330609833 | 2.334253750 | +0.003643917 |
| GB300 GS | 3.250951500 | 2.890316833 | -0.360634667 |
| GB300 rows | 2.790419750 | 2.776954417 | -0.013465333 |
| GB300 collision | 4.301366917 | 4.299118917 | -0.002248000 |

Complete node spans are13.298025167→13.128546417ms RTX and
15.064496917→14.692305000ms GB300. The observed solve resources match AOT:
block32/grid16384,63/64 registers,4080 shared bytes and zero reported local
memory, versus64 registers/1116 shared bytes in the original solve. The same
76 memory nodes remain per environment step; no new global staging dispatch
or persistent allocation was hidden outside the solve.

These separate diagnostic captures do not have identical endpoint workloads.
RTX active contacts are89865→89080 and total rows385818→383509; GB300 contacts
are89429→89434 and rows384219→384262. Thus the owner differences attribute the
complete captured pipeline but are not an identical-input microbenchmark.
The substantial retained-owner stability and smaller GS time locate the small
gain in the intended owner, without proving how much is due to load/index
retirement versus workload variation.

The measurement includes all once-per-solve staging, byte-node conversion,
metadata packing, synchronization and enlarged shared-storage costs, including
the large-world fallback's resource footprint. Original row visits, response
dot products, warp reductions, physical updates, spectral projection and
final decode remain. It does not establish an L1-hit, bandwidth, bank-conflict,
occupancy or stall cause. The complete tested retirement simply delivers much
less than the proposed material whole-time saving; this closes the fixed
prototype without further tuning.

Measured runtime and artifact SHA256 pins:

- Runtime: `11c3ab9aa70cad96ea34198210b9067230c778c0417a8e4f90500028681f35a8`.
- Sparse binding: `0e607c54949ad03bb2caa78e54ce24ba9399551654e3a15a576e317c10361cc3`.
- Tests: `c0626a087028c300b2f4464285d52517ac9013a4afd0f47d06c5c1e5f7e5ccde`.
- Observer: `3e13b49aef8baaf2189d5467109b6b36f94b31c945c2481f0b0cb440d53d5ca6`.
- Capture parent: `5c9c243584de0eed4d0529af2e0a2141c0d646d2cff43039ddc0bd2189667af4`.
- Native RTX log: `1224fe1953b85351256a18a002925b471c0f6dffbce949711c16de4579a7547b`.
- Native GB300 log: `b0c54b609f34d42c277d2a1dd8494a172e0a11ec6dcf9b344dbf247afe8fca4a`.
- Whole manifest: `378e5527713bd2e34a2d7c1098394738470789de4ba73d64ddbde073ab6e40a9`.
- Node manifest: `bd3037bb11a856c16537c618ae376c390b5fad200641ab0362f5107d6e9b7683`.
- Supplemental reader: `d3e0cfcceabd67eb1262e79973ab5771a0341851f38231b88569a907d07fb5b5`.
- Strict RTX output: `5c0be57df59c4449dd3e1d2edd2290c3837f358f175eb275b2a75d39d0380553`.
- Strict GB300 output: `6bf7aef39a2f171f1bf9eea5fdbc241e71b40faf5dcda9f3c9c5e83016df4c08`.

Full and scoped pre-commit checks pass. Measured runtime/test/observer bytes
remain unchanged during report closure. The branch preserves the experiment
for reproducibility, not as an accepted optimization.
