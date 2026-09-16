# Concurrent spectral Jacobi: unpromoted first implementation

This default-off experiment replaces EX1/Nesterov only for the original cold
ANY18 admission with simultaneous normal-first, root-free spectral tangent
proposals. All proposals see the incoming pass state. The fixed permanent
1-to-0.5 merit latch, maximum 24 passes, denominator-only CFM, physical merit
tolerances, current rows/held factor, staging, capacity, fallback and physical
decode remain intact. No contact roots, history or extra pass is introduced.
It is not the previously closed ordered spectral GS or rooted contact Jacobi.

## Numerical evidence is separate from cost

The fixed CPU all2048 evaluation reused existing converged held-H references,
with source/input guards; no reference was solved again. All hard laws pass.
Compared with the saved native result, 1982 cases are closer to the reference,
but **66 genuine finite-budget H-distance regressions remain**. Scaled H error
median/p99/max changes from 0.00638992/0.03199576/0.05573845 to
0.00011879/0.00662368/0.01813002. Maximum excess error is 0.00791096; the worst
error ratio is 5.1093. These tradeoffs are retained, not silently relabeled.
The original pointwise gates pass 1390 and flag 658 cases. Maximum biased
normal defect improves from 0.0343720 to 0.0290156.

CPU passes are 45,557 versus 49,059 original, with 614 physical stops and
431 permanent latches. The two operators plus setup count 49,830,876 products
versus 84,268,494 original: a 40.87% product reduction, **not** a timing model.
This excludes the substantial scalar projection/merit cost. Retained geometry,
whitening, final decode and unsupported fallback are not claimed removed.

Native replay passes all2048 hard finite/sign/cone/momentum controls on both
cards. Eight CPU translation controls have maximum scaled held-H difference
1.2114591e-5. Native pointwise diagnostics flag 664 cases. Same-input complete
owner replay is roughly 0.095 ms versus 0.048–0.052 ms original on RTX, and
0.101 ms versus 0.050–0.054 ms on GB. No numerical or performance promotion.

## Whole physics and causal phase evidence

One paired discovery round, all source/idle/observer/capacity guards passing:

| Complete physics, ms | Original | Candidate | Original/candidate |
| --- | ---: | ---: | ---: |
| RTX | 9.218097850 | 12.435956575 | 0.741246 |
| GB | 9.483441025 | 14.137671900 | 0.670792 |

An output-only adaptation of the existing five-phase observer preserves every
physical output bit and strips exactly to production native source. All2048
worlds are admitted; all clock sums are exact. Summed within-world clock shares:

| Instrumented phase | RTX | GB |
| --- | ---: | ---: |
| Preparation/staging/whitening | 15.498% | 14.206% |
| Concurrent proposals and join | 15.006% | 15.246% |
| Two operators and joins | 11.361% | 11.325% |
| Physical merit, latch and commit | 53.172% | 54.230% |
| Final decode/publication | 4.963% | 4.994% |

The instrument changes resources, so these are causal clock observations,
not exclusive GPU wall attribution. Native actual passes are 45,558, with
431 midpoints and 614 stops. There are no root/probe/metric-fallback calls.
Production AOT reports RTX 86/91 registers and GB 87/101 for tiers32/48,
zero stack/spills. The observer uses 103/128 and 103/100 registers respectively.

The old loop caches its row step and uses a cheap impulse-change exit test;
the new loop evaluates the complete physical merit each pass. Source/PTX
inspection identifies 6,450,712 repeated invariant-denominator divisions and
1,397,617 repeated diagonal square roots in the CPU-counted cohort. A single
approved follow-up will cache invariant proposal/merit coefficients, retaining
the same mathematics and gates. Dynamic hypot/projection, operators, joins,
finite checks and fallback remain. This is not a mapping/parameter grid.
It must recover at least **4.217858725 ms RTX** from this losing candidate to
beat the original by 1 ms; a gain versus the losing candidate is insufficient.
No correction result is claimed at this checkpoint.

## Reproduction and pins

Flag `FEATHER_PGS_SPECTRAL_JACOBI=1`, marker `_fpgs_spectral_jacobi`, suffix
`_sj24`; original production CTA32/64 and live72/32 capacity ABI are retained.
The saved-probe ABI remains its exact archived192/1 allocation. Root's observer
checks actual factory/constructor attributes; CPU matrix-free construction is
unsupported and was not falsely reported as a constructor test.

- CPU evidence: `/tmp/fpgs-spectral-jacobi-cpu-etPRAs/summary.json`, SHA256
  `0fa7bd683564513067cabff3108300a58060a5bbb4ce6614f45ee6564d89dd18`;
  full `result.json.zlib.base64`, SHA256
  `a90ac1e8586911d2f14380e36d91b41e108a3d86b2e6fa50bc0c0e5a28be4014`.
- Native: `/tmp/fpgs-spectral-jacobi-native-20260916-AbNbPv7B/gpu{0,1}.log`,
  SHA256 `1c51a73f3532447b3a633bb5c1c40095643f5805ff7144359574781f42380f5c`
  and `caab75c76b6ffdfe6437e7006733ccdb0bee8a03502262d4d52e66e431cc622f`.
- Whole: `/tmp/fpgs-anymal-spectral-jacobi-whole-paired16k-20260916-01/manifest.json`,
  SHA256 `595beb3a5fe0acd1ae5b00a9f095923881d23ad0cdf11c82b4b4ef9e8a624569`.
- AOT: `/tmp/fpgs-spectral-jacobi-cpu-etPRAs/native_aot/report.json`, SHA256
  `b1fe089bbb69be83be94339893ebdec68a1b00da4b4ae53ca9a3a7701dc214ba`.
- Phase observer/runner/readme and logs:
  `/tmp/fpgs-spectral-jacobi-phase-f78F1Iy2/`; observer SHA256
  `b0a03c3994080e4fb8819e80af6c59193c7cf8d1e70aa0876021ea175042a0f0`,
  runner `d6936e8513c3868bea9fa78da3b11dade969ad0ee75ae59e646dcb8adb2fe106`;
  logs `907de2a3ee6e3275ae6a4f85ad5779ed2a976caf5b324a25c243a82768997e75`
  and `e43b2a002295b5b21741f4f4e02b1f969a223ff29d5842199c77f711c0cb205f`.

Measured runtime `spectral_jacobi.py` SHA256
`8fd17fcbdb8ffce9b2df33d4ce226636ebf82047b5df7dc7f983e121a57b0065`;
solver `80d13dc97ddd28ee07a213c73989a13ebe2ebd7ede50330f913b3fa246322b57`;
probe `98713690d835c733c26fc77b0dd1a596088ac1389fc4163672b5cc2886e722e7`;
CPU helper `f3cb2e5e15de8ff7d5b876bbf966d69139c134226a2338d15072ffa998385a70`;
observer `0dd420c49462492f36a5248061570e4bb79cd2ebc6faabbc220a9b9b6165c598`.
Dependencies are pinned by the runtime and probe. Nine focused CPU controls,
missing-module regression, hidden AOT, full and targeted pre-commit passed.
No Isaac Lab edits, GPU launches by the implementation agent, PR or parent
pointer changes; the inherited handoff remains unchanged.
