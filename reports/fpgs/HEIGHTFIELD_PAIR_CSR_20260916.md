# Raw-witness heightfield pair CSR: measured, physically unqualified

This checkpoint preserves the first `NEWTON_HEIGHTFIELD_PAIR_CSR=1`
prototype before any selection correction. The flag defaults off. It is **not
promoted**: one whole-physics screen saves less than the funded 1 ms RTX
milestone, and the loaded support/impact behavior is unacceptable. Cone and
momentum checks do not override those physical failures.

## Owner and scope

Branch `ooctipus/heightfield-pair-csr-20260916` starts from adaptive-manifold
commit `c59c10bcf732ec719ba5a353c7b1bf1fe7996ca5`, not the separate shell-support
branch. Packed finite and generic triangle queries retain separate lifetimes
and the existing raw witness buffer. Their callbacks additionally tag the
originating pair. Count/scan/scatter builds an uncapped per-pair raw-ID list;
the replacement export selects at most four writer-accepted witnesses with
their own normals/materials. It is a new discretization, **not** the old
reducer-survivor set and not a pooled-friction law.

Admission requires the stock writer, supported finite/geometric heightfield
route, and unique pair ownership: generated NXN pairs or a construction-time
verified unique explicit list. HFIELD/BOX and HFIELD/CONVEX pairs use CSR;
unsupported pairs retain the original hash reduction/export. Deterministic,
speculative, hydroelastic, mesh and unsupported writer configurations retain
the old owner. Same-pointer explicit-pair mutations that violate uniqueness
require pipeline reconstruction; immutable-hull constraints are unchanged.

Additional int32 arrays are `triangle_pair[T]`, `raw_pair[R+1]`, `ids[R]`,
`counts[P+1]`, `offsets[P+1]`, `cursors[P]`, and `status[1]`: exactly
`4*(T+2*R+3*P+4)` logical bytes. There is no pair-local candidate cap or timed
host synchronization. Old hash storage and fallback launches remain, but
admitted pairs no longer insert or publish through that hash. Raw overflow
and malformed membership latch a public sticky capacity flag; `clear=True`
returns the failure before acknowledging it. A failed result must be discarded.

## Validation and preserved failures

- Missing-module regression failed before implementation. Seven focused CPU
  controls passed, including unique-pair admission/fallback, mixed supported
  and unsupported shapes, more than 252 witnesses, reset/empty/regrow,
  writer acceptance, actual raw-overflow and return-before-clear status.
- Four native routing/lifecycle/pool/overflow controls passed on each GPU.
  Initial loaded execution in
  `/tmp/fpgs-pair-csr-native-20260916-HlZyJVHI` failed before physics because
  the inherited fixture lacked its original grouped-lane environment. This
  is a preserved setup failure, not a CSR physical result.
- Corrected original fixture environment (`FEATHER_PGS_GROUP_LANES=16`,
  `FEATHER_PGS_ROWS_MASKED=1`, `NEWTON_NARROW_PHASE_THREADS_X=4`) completed
  all 18 records/card in
  `/tmp/fpgs-pair-csr-loaded-20260916-HBoLF6K6/gpu{0,1}.log`. CSR support,
  tilted-foot, yaw and step settling fail. Flat support ends at approximately
  0.5844 rad/s with 0.675 mm penetration, versus essentially zero baseline
  spin. Good tail force balance, momentum and cone errors certify only the
  reduced instantaneous problem, not a correct manifold.
- Free rebound exposes the same support asymmetry: candidate final vertical
  velocity is about -1.9873 m/s and spin 33.8569 rad/s, versus baseline upward
  velocity about 1.8 m/s. The existing record's empty failure list is not a
  physical clearance of that observation. Full-friction sliding improves in
  this fixture while its baseline misses the analytic gate; it does not
  compensate for support/impact failure. Neither baseline nor gates changed.
- Hidden AOT is preserved at
  `/tmp/fpgs-pair-csr-offline-2y6ompmN/offline01/report.json`
  (SHA256 `77769cda63eaf8cf40686dee288ee6ed0e211ff31ed01c269e3403946228b1b9`).
  Export uses 72 registers, 128 B static plus 320 B dynamic shared memory,
  no stack/spills in that compilation. Finite/generic query stack frames
  remain; absence of reported spill loads/stores is not absence of a stack.
- Full `uvx pre-commit run -a`, targeted hooks including both new files, and
  `git diff --check` passed before the measured checkpoint commit. Runtime
  and test hashes remained equal to the captured versions.

## Whole screen

The unchanged paired driver used 16,384 G1 worlds, seed 0, 200 warmup steps,
40 wall steps and 40 graph steps; maximum eight GS sweeps, two Newton substeps,
100 dense rows, rigid capacity 294,912, broad-phase capacity 49,152 and
triangle capacity 1,769,472. Both arms use the qualified chain state owner,
compiled coordinates off, and old adaptive export off. The only experimental
arm switch is CSR 0/1. No MJWarp or other-task comparison was run here.

| Whole physics, ms/env step | Qualified chain | CSR | Saving | Speedup |
| --- | ---: | ---: | ---: | ---: |
| RTX | 15.036678875 | 14.161686525 | 0.874992350 | 1.061786x |
| GB300 | 19.870284175 | 18.936348000 | 0.933936175 | 1.049320x |

Valid run:
`/tmp/fpgs-g1-heightfield-pair-csr-whole-paired16k-20260916-02/manifest.json`,
SHA256 `f6caaf3b02664b01143cba7b89d9e447d1d08d6abe0c95d314c5de32bc3c6d49`.
All source/idle/capacity/actual-owner guards passed. Run `-01` remains failed:
the original observer's exact capacity-key set rejected the newly named
public CSR flag. The corrected observer extends that exact set only when CSR
is requested; it does not ignore capacity failures. One screen is not repeat
qualification, and the result remains physically unqualified.

## Strict node attribution

Input `/tmp/fpgs-g1-heightfield-pair-csr-node-candidate16k-20260916-01`
retains the parent's expected auxiliary-graph analyzer rejection. Both SQLite
captures are valid. The minimal reader at
`/tmp/fpgs-pair-csr-strict-UGnL9ijL/read_pair_csr.py` has SHA256
`c2daaeea7405612d8577e23c1d176830a94ae803285a4be8331e93adc08bb497`.
It pins the existing strict reader, exact runtime/observer hashes and source
snapshots, and retains the 48 physics/12 auxiliary scopes, correlations,
budgets and capacity guards. Independent CPU replay to
`/tmp/fpgs-pair-csr-independent-gXGjU3` passed before this report was added.
No GPU was used for the independent review. Both replays produce identical
result hashes:

- RTX `8c36a118acc27607516e233a72dfdd8484747ebd8a21e0e762babbb7b94f3af3`
- GB `aba37ed8370132b7a35d6cf0d909b0b84f814d08e623abade055a349a2e4c356`

Every new query/membership/export owner runs four times per environment step;
original metric GS still runs eight. Old full buffered hash reduction and old
adaptive export are absent. Filtered fallback reduction and original fallback
export remain four calls each. Exclusive family times versus the existing
qualified-chain strict capture, in ms/env step:

| Family | RTX original | RTX CSR | GB original | GB CSR |
| --- | ---: | ---: | ---: | ---: |
| Collision | 3.743738 | 3.030802 | 8.378876 | 7.733085 |
| Row/response | 2.527910 | 2.703490 | 2.554350 | 2.743940 |
| Metric GS | 3.957767 | 3.565301 | 4.421050 | 3.956226 |

RTX new costs: count 0.059784, scan including init 0.021837, scatter 0.033669,
fallback reduction 0.012413 and CSR export 0.368163 ms. GB CSR export costs
0.627026 ms. Active hash clear drops to 0.007328/0.008955 ms RTX/GB; original
fallback export remains about 0.06232 ms RTX. Thus actual admitted hash work
retirement survives integration, but rows get slower and the reduced
manifold's GS saving is smaller than the prior proxy suggested. Node sums are
ownership diagnostics, not a new critical-path or whole-throughput claim.

## Concrete physical cause and next gate

CPU inspection of the actual initial support fixture finds many distinct
near-vertical footprint witnesses at raw depth 0.00500000082 m. The selected
set instead contains one near-active vertical contact and three positive-gap
side witnesses with physical separations approximately 0.03531, 0.03772 and
0.03772 m, and normal z components about 0.124, 0.117 and 0.117. Stages 1--3
maximize geometric spread without activation/depth priority. These distant
speculative rows are inactive at rest and displace load-bearing support.
This is not a duplicate-ID or routing-loss diagnosis.

Initial old-reducer geometry is unchanged by geometric-cull 0/1 in the CPU
control. Old adaptive-four export also loses the active footprint in that
initial scene. Therefore the defect is not uniquely introduced by raw CSR,
nor does this control explain earlier trajectory differences solely by the
geometric-cull flag. MJWarp uses different zero-cutoff, margin-inflated witness
generation; its inspected spread has no deepest-depth band to copy. No exact
MJWarp candidate-filter equivalence is claimed.

One activation/nearest-separation priority correction is proposed, pending an
explicit arithmetic-bound and complete added-cost review. No corrected result
exists at this checkpoint. It must retain upcoming separated contacts,
per-witness laws and the unchanged loaded physical gates; successful cone or
momentum checks alone cannot qualify it. The feature remains default off.
