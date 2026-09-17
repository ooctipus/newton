# Combined opt-in structural checkpoint

This branch combines G1 limit Jacobi checkpoint
`81c0cb201453df67daf120438b05801906f2878c` with the exact nine-file Franka
compact-workspace change from `8ec4db342c0c5596060e89ef4923412a4a0ef1c1`.
The G1 parent includes the measured shell/CSR/spectral composition `a61ea916`.
Original handoff Newton `31cf87f4694f873a027e41e2ca5e9ad441234456` remains an
ancestor. Remote is the user's `ooctipus/newton` fork. No Isaac Lab source,
parent dependency pointer, task timestep, substep count, iteration allowance
or calibrated capacity changes are included.

This is a usable composition checkpoint, not a claim of new combined timing,
cross-task performance promotion, sustained physical qualification or 4x
across all tasks. New paths remain opt-in. Nonselected task recipes keep their
existing owners; the original complete unsupported-mode routes remain.

## Measured component-branch evidence

All times below are whole physics per batched environment step, not sums of
individual kernels. Franka and G1 use their established 16,384 environments.

| Change | GPU | Original ms | Candidate ms | Ratio |
| --- | --- | ---: | ---: | ---: |
| Franka compact, run 01 | RTX | 5.108246725 | 4.797041475 | 1.064874413x |
| Franka compact, run 03 | RTX | 5.049426675 | 4.864011150 | 1.038119881x |
| Franka compact, run 02 | GB300 | 4.690438375 | 4.688745475 | 1.000361056x |
| G1 actual-Z limit Jacobi, run 01 | RTX | 13.160521725 | 12.535854600 | 1.049830438x |

Franka's GB result is flat, not a cross-card speedup. Its environment wall time
in that GB run changes 27.596603299 to 28.507825048 ms, separately from physics.
The G1 gain follows a diagnosed initial whole-step loss and one targeted
response-representation correction. It has not yet been repeated or measured
successfully on GB300. G1's GB attempt and Franka's first RTX paired node run
stop between arms after unrelated GPU jobs appear; neither gives a comparison.
Do not divide these new FPGS values by an old MJWarp denominator and call the
result a fresh matched-backend measurement.

Detailed algorithm, numerical limits, initial failures and artifact locations:

- [Franka workspace](FRANKA_COMPACT_WORKSPACE_20260917.md); later GB/abort
  report-only update is retained separately as `ee2a50d5` on the same feature
  branch. The exact GB outcome is recorded above.
- [G1 limit-prefix update](G1_LIMIT_JACOBI_20260917.md).
- [G1 spectral contacts](G1_SPECTRAL_TANGENTS_20260916.md).
- [Heightfield shell/CSR](HEIGHTFIELD_SHELL_PAIR_CSR_20260916.md).

The maintained capture entry points are
`tools/fpgs_bench/compact_workspace_capture_20260917/run.py` and
`tools/fpgs_bench/chain_capture_20260916/run.py`. They accept `--gpus 0 1`
for concurrent paired-card measurement when both cards are idle. Preserve the
complete environment dictionaries and capacity arguments from the respective
successful manifest, changing only candidate checkout and output directory.
Franka adds `FEATHER_PGS_FRANKA_COMPACT_WORKSPACE=1` to its selected kinetic
recipe. G1 adds `FEATHER_PGS_SPARSE_LIMIT_JACOBI=1` to its selected a61
shell/CSR/spectral/chain recipe. These are not global flags for other tasks.
Use the same shared collision recipe in any new MJWarp comparison.

## Composition checks

The focused CPU composition suite passes 37 tests, with 15 CUDA-only skips
(52 discovered tests). Seven selected native tests then pass on each GPU:
four Franka physical/current-held/publication/reset tests and three G1
prefix/energy-fallback/current-held/graph/saved-case tests. Both native runs
share the device with an unrelated job and are correctness evidence only.
They are not benchmark timing or a new long-horizon convergence study.

Exact measured runtime SHA256 values are unchanged by composition:

- Franka compact: `ad6e62f07e5e7f9af65b8468ab34374183f345afe395dd922899dc3aba763716`.
- Franka binding: `66b8b5e7d9d4784d3da7104d204627a2e7965b8468aed6e93eb1dc618cb7daf2`.
- G1 limit response: `5a15a0d64f914466df6e847426f5b9e83173d33fab490ada4aaa04c6211e3ac0`.
- G1 factor binding: `4c94ee8a3b2f3da3c98641c07db55c634421c5319a6cc382e1ba47efe4b3bf16`.

Local test logs and SHA256:

- `/tmp/fpgs-structural-progress-cpu-20260917-01.log`:
  `b93e1fbba3bba695d2ac1ccc7f42480f9bb771687004e611c5b4b32cac7c9ea6`.
- `/tmp/fpgs-structural-progress-native-rtx-shared-correctness-20260917-01.log`:
  `61ed4aa920928363ebe63604770cb251374170a435c290a8b982d66999072b54`.
- `/tmp/fpgs-structural-progress-native-gb-shared-correctness-20260917-01.log`:
  `fd7ceb5c0b6063413fccebd699c351d369637142dbbb48374f7b260e6675b321`.

Independent source/dispatch review passes: both sets of runtime dependencies
are byte-identical to the measured branches, default ownership is unchanged,
and actual CPU plane/box collision controls with heightfield options off and
on retain the original no-heightfield path. Full and owned pre-commit pass.
Original worktrees and large reference captures remain in place.
