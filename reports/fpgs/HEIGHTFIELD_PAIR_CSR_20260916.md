# Raw-witness heightfield pair CSR: unpromoted checkpoints

The initial checkpoint preserves the first `NEWTON_HEIGHTFIELD_PAIR_CSR=1`
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

## One cause-directed interval-priority correction

The first prototype is recoverable at pushed commit
`eabc4f8c3d2aacae78e78a6004d02047b175f7f5`. The correction changes selection
priority, not the query witness pool, stock writer, fallback, friction policy,
capacities or solver allowance. A fail-first actual-query regression reproduced
the side-witness displacement in both support and positive-gap rebound before
the runtime change.

Count now caches one outward `(L,U)` separation interval per admitted raw ID,
reusing the stock decode and its second normalization. With represented normal
`n`, center `p`, depth `t`, effective radii `ra/rb` and margins `ma/mb`, define
`S = sum(abs(n)*(2*abs(p)+abs(n)*(abs(t)+abs(ra)+abs(rb))))
+abs(ra)+abs(rb)+abs(ma)+abs(mb)`. For `u=2^-24` and
`gamma(k)=k*u/(1-k*u)`, the upward-rounded multiplier is
`K=up(gamma(16)/(1-gamma(8)))=9.536757943351404e-7f`. The implementation uses
`E=up(up(K*S_hat)+64*FLT_MIN)`, `L=down(phi-E)`, `U=up(phi+E)`, with literal
stock reconstructed `phi`. CUDA uses `nextafterf`; CPU uses tested binary32
successors because Warp's CPU module has no C-runtime declaration for it.
The count module explicitly disables fast math. Nonfinite arithmetic latches
failure. This bounds represented-witness endpoint arithmetic only: it is not
a query-error, body-local roundtrip or geometric-support certificate, and the
underflow slack is not a physical depth band.

One preliminary bucket scan computes `cutoff=max(0,min(U))`. The original four
scans choose minimum U first, then prioritize nonnegative spread scores from
`L<=cutoff`; outside witnesses have negative `-U` scores and fill remaining
slots by nearest separation only after that cohort is exhausted. There is no
normal-angle threshold, extra per-stage reduction or score-bit truncation.
All-positive closest footprints are retained. The single new vec2 array costs
`8*(R+1)=14,155,784` bytes at the original raw capacity, with at most 52 added
logical bytes per accepted record across count/export, no new launch or extra
normal decode. Shared export scratch increases by 16 bytes.

Final runtime SHA256 is
`0be8626e13df2995d6b3a9f00a148cf612726b69ed83cda734f4d2d3a5608e3f`;
focused tests are
`9607a44dc57112feee32de4cd664e1adfd7a82b0e1bbc18ff963f8743914c4b0`.
CPU nine controls passed, including active/positive footprint geometry,
outward rounding at signed zero/infinities/NaN, large-center cancellation,
radii, zero/negative margins, unchanged gap acceptance and nonfinite interval
failure. SM120/103 hidden AOT passed at
`/tmp/fpgs-pair-csr-offline-2y6ompmN/offline02/report.json`, SHA256
`ed1de0df32f88b9b6a293389755c4859491ae357e4083592c98f0f743c848202`:
count uses 38 registers; export uses 72 registers and 128+336 B shared;
neither has stack/spill loads/stores. Original query resource profiles remain.

### Native physical result: inspect outcomes, not only failure lists

Seven native selectors/card ran with the unchanged original fixture environment
at `/tmp/fpgs-pair-csr-priority-native-20260916-1Tn0yyls/gpu{0,1}.log`.
Log hashes are RTX
`9ebe060b3fdf654d7193c0028114e2b97aa64f37d56e4650632a62635c90118b`
and GB `8914ae3f7163a4f2832cf510d0c09319013dd1b8514ccadc44888a0fb54539e7`.
Six standalone native controls pass on both cards. The loaded selector still
exits 1, solely for retained old-arm failures: full-friction sliding/yaw/step
on RTX and full-friction sliding/step on GB. All nine candidate case records
have no listed failures; no old failure was removed or threshold changed.

Independent log inspection confirms more than that empty candidate list:

- Free rebound now has vertical velocity +1.80000019 m/s and energy ratio
  0.360000076, consistent with authored restitution 0.6 and incoming -3 m/s;
  spin is 3.36e-5 rad/s. The original catastrophic asymmetric rebound is gone.
- Flat support final speed/spin are 2.04e-6 m/s / 1.97e-5 rad/s. Its full
  80-step mean-weight relative error is 5.33e-7 and momentum error 5.65e-8.
  Tilt/yaw settle. Step final spin is 0.00646 rad/s (tail maximum 0.01391),
  weight error 9.67e-5 and minimum separation about -16.3 micrometers.
- Candidate maximum recorded cone excess is 7.46e-9, negative normal impulse
  is zero, and maximum momentum error across the nine cases is 4.57e-7.
- Full-friction sliding stops at 0.507185 m versus analytic 0.509684 m, with
  final speed approximately 1.01e-7 m/s. Production anchor-limit-two sliding
  is a different selected-anchor law: it reaches x=1.27577 m beyond the
  finite terrain edge, with vertical velocity -0.3723 m/s and continued
  tipping. Samples remain near the support height through center x=1.18742;
  descent appears after the center crosses edge x=1.2. This is not an observed
  interior terrain fallthrough, nor an assertion of trajectory equivalence.
  The separate finite-border case finishes outside with zero contacts.

These are bounded loaded controls, not cross-task or long-horizon promotion.
In particular the inherited free-rebound failure list alone lacks an analytic
free-foot gate; the explicit velocity/energy/spin inspection above is retained.

### Corrected whole screen and disposition

`/tmp/fpgs-g1-heightfield-pair-csr-priority-whole-paired16k-20260916-01/manifest.json`
is complete, SHA256
`e2f6cfee1e7e5f9b44f6969214723209bfc4cd5c2a50855a902f2ac5c2729b3a`.
All original source, idle, budget, capacity and actual-owner guards passed,
including exact interval array ownership. Same paired 16K recipe as above:

| Corrected whole physics | Qualified chain | Priority CSR | Saving | Speedup |
| --- | ---: | ---: | ---: | ---: |
| RTX ms/env step | 15.005645075 | 14.326358525 | 0.679286550 | 1.047415x |
| GB300 ms/env step | 19.825631200 | 19.141688100 | 0.683943100 | 1.035731x |

Environment wall time is separate: RTX 28.13218 to 28.37751 ms; GB 35.20044
to 33.68487 ms. The arithmetic/selection correction costs some of the initial
prototype's saving while resolving the diagnosed loaded failures. It remains
below the 1 ms standalone milestone and is **default off, not promoted**.
No corrected node attribution, repeated qualification, new MJWarp comparison
or shell/spectral composition result is claimed at this checkpoint.
