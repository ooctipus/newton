# Public PGS lead and bounded G1 transfer

2026-09-17. Public-source investigation and CPU diagnostics only; no Newton/Lab
runtime change, GPU port, new capture, tolerance change or performance promotion.
This is separate from the [whole-timing ledger](TWENTY_HOUR_20260916.md).
The companion [projection design comparison](PROJECTION_DESIGN_20260917.md)
separates repair, numerical metric and contact law, and discusses maintenance
and implemented generality without authorizing a runtime refactor.

## Strong public match, not identification of a private conversation

Yuval Tassa's June 23 commit
[`c499f7f2b0f3d78b47c48fd487d22c5544ae4f6d`](https://github.com/google-deepmind/mujoco/commit/c499f7f2b0f3d78b47c48fd487d22c5544ae4f6d)
adds restarted Nesterov momentum to C MuJoCo PGS. The published benchmark reports
1.8–2.1x solver and 1.6–1.7x complete-step speedups, with iterations 95→46 without
islands. This is fewer sweeps to its stopping criterion, not a twice-cheaper
individual sweep or a GPU result.

The mechanism uses the previous two force iterates, positive integer-age
coefficient `(age-1)/(age+2)`, feasibility projection, a complete sequential PGS
sweep, then a negative direction-dot restart. Restart resets the age but does
not roll back that sweep. The first two sweeps have no positive momentum.
The benchmark uses a two-humanoid/100-object scene, warmstart, a 100-iteration
maximum and tolerance 1e-8; it does not establish an eight-pass FPGS result.
[Introducing source and tests](https://github.com/google-deepmind/mujoco/commit/c499f7f2b0f3d78b47c48fd487d22c5544ae4f6d).

The archived source study checks the 3.11 release entry and extracts byte-identical
`solPGS` regions at introduction, 3.12 commit
`13827e9ee56f097f57acf69ae52b078f9839682d`, and September 16 main
`71d430c71f8593a977136485e81314ec19a66e7e`; region SHA is
`f247a253f236e8de93b7f350b5734936c19cbff205b3aff8524ce458c48338d0`.
[Pinned release notes](https://github.com/google-deepmind/mujoco/blob/13827e9ee56f097f57acf69ae52b078f9839682d/doc/changelog.rst).
Separate earlier changes concern island dispatch, constraint permutation and
CPU thread-pool dispatch; their gains cannot be multiplied into this forecast.

MJWarp is a different case: its public README still lists PGS as unsupported.
Taylor Howell's May prototype PR1331 was closed unmerged; it is not the June
C momentum change. This establishes neither the reported speaker's intended
method nor the absence of unpublished work.
[MJWarp compatibility](https://github.com/google-deepmind/mujoco_warp#mujoco-api-compatibility),
[prototype PR1331](https://github.com/google-deepmind/mujoco_warp/pull/1331).

MuJoCo's regularized cost-decrease exit is not our physical stop. Momentum can
raise the cost before the sweep's counted decrease, while FPGS retains separate
normal complementarity, each parent's friction disk and denominator-only CFM.
Transfer the proposal only; do not import that stopping test as a proof of the
FPGS contact law. Full source, model and stopping-order analysis is in the
[archived study](</tmp/fpgs-public-pgs-source-study-bxo2dQ73/REPORT.md>), SHA
`f422855848c664ed571e261b61b25d73cdfdad3406d7d73e90de7e1753468ea6`.

The added scans are amortized differently: C PGS re-dots each visited AR row
against constraint forces, while accepted G1 carries a 43-coordinate kinetic
response and uses at most 18 coefficients per row, often skipping open tangents.
The public sparse-AR benchmark does not justify assuming dense quadratic work;
nevertheless its residual cost scales with visited AR nonzeros, versus linear
history/projection scans. This explains a possible overhead difference, not a
cross-model/hardware speed ranking. ANY24 already pays FISTA-style momentum, so
its separate study replaces that policy rather than adding acceleration anew.

## Fixed-eight G1 screen: numerical promise, no native speed case

The [frozen CPU results and causal note](</tmp/fpgs-g1-yuval-momentum-chQosOcW/RESULTS.md>)
reuse corrected simultaneous limits followed by ordered normal-first spectral
contacts, own-parent disks and denominator-only CFM. Scope is omega 1,
friction-start 0 and maximum 8. Four synthetic controls pass after a preserved
missing-module regression. Saved16 uses independent current/held physical J/H;
historical1268 uses the unchanged 635 RTX/633 GB selection from ef9481 W/Z,
not today's shell/CSR population. No new reference solve is added.

| Historical1268 diagnostic | Original → momentum |
| --- | ---: |
| Finite/cone/negative/momentum hard failures | 0 → 0 |
| Cases passing the physical stop at endpoint | 709 → 740 |
| Natural median | 6.0593e-7 → 2.6484e-7 |
| Natural p95 | 0.0051071 → 0.0031126 |
| Natural maximum | 0.0800948 → 0.1464191 |
| Hypothetical first-stop-or8 pass sum | 6496 → 6419 |

That 77-pass saving is only 1.19% **before** extra history, projection, response
updates, restart reductions and physical checks. The latter are diagnostics,
not native G1 termination. Candidate actually calls the map 10144 times versus
8554 original calls with exact-unchanged exits; these different termination
semantics are not an equal-work timing comparison.

The existing curves explain the small saving: 408 cases stop by pass 2 before
momentum, 479 stop in neither arm by 8, so 70.0% cannot save a pass here. Natural
residual improves in 375 of those 479 without satisfying all stop gates. The
remaining groups save 24+70−17=77 passes. The first boost is pass 3; 606 cases
restart then, leaving an average 3.98 positive-beta passes. Observed overshoot
and recovery in RTX9970/held7410 do not prove asymptotic divergence, but show
that restart is neither rollback nor a componentwise physical safeguard.
Twenty-eight candidate cases reach a stop and later leave it; ever-hit counts
are not endpoint qualification. Tails are reported, not hidden by median gains.

Complete added service includes 19853 tangent-disk norms, 116930 sparse response
products for 9820 projection-corrected rows, 179298 restart dot terms and 8876
reductions, plus history traffic. Prospective history is `4*(2*n+43)` bytes,
up to 972 at cap 100; no occupancy or native timing is inferred. CPU triangular
preparation and diagnostic physical scans are separately labeled reference
overhead, not repeatedly mandatory native work. All original hard gates pass;
no pointwise monotonicity requirement or tolerance grid is introduced.

Conclusion: the public result demonstrates acceleration on its reported
benchmark, but this G1 screen does not fund a GPU port or establish a gain.
It does not rule out other tasks. The separate ANY24 projected integer-age
replacement study is closed below; the existing ANY FISTA-like recurrence is
not silently treated as an identical prior experiment.

## Exact local pins

- Final `RESULTS.md`: `81411d2cd2079bc77733924fd9ad31c097fe4d7dddba1cea13d4d0752ea8b3f0`.
- [Control](</tmp/fpgs-g1-yuval-momentum-chQosOcW/control.py>): `1f8a4b1a6016e00ba3c13b27bf52301b7daae6c027b54359e7af38bbe7c05c49`.
- Tests: `e9718e8cc4926205b14cb8f4b58e11b1bcfb48dd7d5261806a19ccb977d8c0aa`.
- Evaluator: `5ddddede660ac92e4a954d5e86f5f12cd73252b19557684b65352f0ed09b4ee8`.
- [Saved16](</tmp/fpgs-g1-yuval-momentum-chQosOcW/saved16.json>): `98f4bcd1901caca80ec2fed3d71cdf8c6f5bfc76bd6883535f9e71828d855995`.
- [Historical1268](</tmp/fpgs-g1-yuval-momentum-chQosOcW/historical1268.json>): `16cf5727b4c12287ba6e731e342c9eb4c87bbde2c0a6cbaa2f1edf8655fb5859`.

Two evaluator-only serialization/optional-counter errors preceded successful
outputs; numerical helper bytes stayed fixed. JSON label `above_original_plus_3e5`
is a naming typo only: the expression uses 3e-5. Use per-case curves and explicit
maxima, not sums of first-stop or history-size fields. No longer-horizon followup
or GPU run is implied by this archive.

## ANY24 replacement closure

The [final ANY report](</tmp/fpgs-anymal-yuval-ex1-7CPHmcbl/RESULTS.md>) closes
this transfer without a native port. Six CPU controls pass; the baseline
matches the pinned literal CPU EX1/FISTA method exactly on all 2048 historical
current/held cases, not CUDA bit equivalence. The replacement retains the EX1
map, denominator-only CFM, incoming-delta publication, original cheap stop and
24-pass cap. Independent archive verification passes all 2077 recorded
source/payload/reference hashes.

Executed passes change **49059→49064**, with **2026→2031** cases reaching 24.
Core EX1/recurrence/publication products change **84704706→84708576**, before
**397785 additional momentum disk norms**. Existing operator work is not
retired. Actual restarts fall **2200→2139**; median positive-beta exposure is
20 passes, so G1's short acceleration horizon does not explain this result.
Removing FISTA's scalar square root does not establish a complete speed case.

All candidate/baseline/native finite, cone, negative-impulse and momentum hard
checks pass. Only 874 cases are closer to the existing H-reference; other
physical diagnostics show mixed tradeoffs, not a componentwise rejection rule.
Neither arm meets the separate offline full physical stop in any case.
Those residuals use the impulse-implied FP64 physical response, while published
FP32 velocity is checked separately for momentum consistency. Feasibility is
not convergence, and the offline stop never terminates the map.

No GPU timing, new reference, policy grid or runtime change follows. This
closes the tested G1 and ANY transfers only, not every task or an unidentified
private method. Accepted sources and fair/paired timings remain unchanged.

Frozen directory: `/tmp/fpgs-anymal-yuval-ex1-7CPHmcbl`.

- `RESULTS.md`: `05bc29ed077047d5f1649841b66126bb9a4a54d49dad6d92c08c514e4ea89c01`.
- `control.py`: `d140e1b598fd58931c94808d3ab730035f6427721d0df5b3ade4ae1409d453e5`.
- `test_control.py`: `4d4472c1ec9f79ddf722c2abbffcd927267665cbb6b07aa920a45199ac91ce61`.
- `check.py`: `3f947280dd6d1114b17e0bc32d8c01a3dc6b3973aeb43c0afac9af11115dd450`.
- `first8.json`: `c6a37727f5cf73a04acbd8598f37344a39201511a91e65c563d93498eab97cac`.
- [All2048](</tmp/fpgs-anymal-yuval-ex1-7CPHmcbl/all2048.json>): `8812daecec45fc11588384ca6d9235c832cc1af8cc82b3c0a547abddae6311c6`.
