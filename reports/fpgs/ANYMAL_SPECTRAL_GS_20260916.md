# ANYmal ordered spectral GS: native cost closure

## Decision and scope

Closed as an **unpromoted, default-off experiment**. `FEATHER_PGS_SPECTRAL_GS=1`
translates the frozen CPU spectral-GS control; the default retains the original
solver. No timestep, substep, iteration allowance, capacity, Isaac Lab source or
inherited handoff change. The strong CPU reference-distance result is not a
performance result, and neither it nor the native hard checks establish full
trajectory qualification.

The admitted cold ANY18 path retains the original current-row geometry,
whitening and final physical decode. It replaces EX1/Nesterov with at most 24
complete ordered sweeps. Each contact receives a normal-then-spectral-tangent
transaction using a shared tangent step and Coulomb disk, denominator-only CFM
and the frozen physical stopping policy. There are no nested roots, Schur solves
or history buffers.
Unsupported static/dynamic configurations retain the original complete owner;
dynamic admission is decided before impulse mutation. The existing tier32/48
block32/64 mapping is unchanged.

## Numerical evidence, separate from cost

- The frozen CPU study covered 2,048 saved current/held cases. Candidate velocity
  was closer to the existing converged held-H reference in 2,035 cases; preserve
  the 13 finite-budget regressions. Median reference error changed from
  0.00638992 to 0.0000690685, maximum from 0.0557385 to 0.00677731.
- Native saved-input runs completed on both cards: **2,048/2,048 hard finite,
  impulse-sign, cone and momentum checks pass**. Eight prescribed native/CPU
  translation controls had scaled held-H velocity disagreement between
  1.989e-7 and 1.2112e-5. These eight controls are not a new all-case converged
  reference comparison.
- Native has 500 pointwise diagnostic misses against original native24; the CPU
  study had 494. These counts are distinct from the 13 independent-reference
  regressions and must not be relabeled as either hard-law failures or passes.
- Regression-first missing-module failure was observed before implementation.
  Three new CPU factory/local-transaction controls and three original spectral
  controls pass. They cover original ABIs/fallback, normal-to-tangent coupling,
  unequal CFM, open contact and zero friction. Independent source review found
  no concrete math/collective blocker; actual native inputs additionally cover
  both row tiers and current/held operators.
- Archival CPU rerun: all six tests pass with CUDA hidden. Required
  `uvx pre-commit run -a` and targeted checks covering all new/modified files pass.
  Measured runtime and observer hashes remain unchanged.

## Measured cost

Saved-input replay timings include the same original input restoration and
complete owner in both arms. Ranges below span current/held partitions, not
confidence intervals; units are milliseconds per replay launch.

| Device/tier | Original | Spectral GS |
| --- | ---: | ---: |
| RTX / 32 | 0.0471065–0.0475341 | 0.1986964–0.2000068 |
| RTX / 48 | 0.0510597–0.0511124 | 0.2503829–0.2506221 |
| GB / 32 | 0.0501414–0.0502306 | 0.2088852–0.2093246 |
| GB / 48 | 0.0544313–0.0544337 | 0.2595290–0.2595921 |

Whole-task discovery used 16,384 environments, seed 0, 200 warmup steps and
40 measured/profile steps, one paired round. Baseline is accepted `ca0d427a`;
candidate is `57e1dcfa` plus the pinned files below. Both arms use
`FEATHER_PGS_SIMPLE_WORLD_ZERO=1`, explicit spectral 0/1, dense capacity 72,
raw-contact capacity 212,992 and broadphase capacity 294,912.

| Whole physics, ms | Original | Spectral GS | Original/candidate |
| --- | ---: | ---: | ---: |
| RTX | 9.219848000 | 22.605269075 | 0.407863 |
| GB | 9.501435475 | 24.751986800 | 0.383866 |

Whole02 completed with all final source/idle guards true and matching capacities.
Whole01 is preserved as a failed observer attempt: both baseline arms raised
`AttributeError` for the nonexistent `mf_warmstart` attribute before physics
timing. Only the observer was corrected to inspect `_mf_warmstart_enabled`;
measured runtime files were unchanged. Whole01 is not performance evidence.

## Cause diagnosis and corrected work accounting

The original recurrence evaluates row residuals and projections concurrently;
EX1 is a one-time row-parallel setup. The candidate serializes each contact
transaction through three dot reductions (15 shuffle instructions), lane-zero
normal/disk projection, three broadcasts and response publication before the
next contact. Tier48's second warp waits during this loop. Every pass also
performs the full physical residual/projection stopping scan. The CPU cohort
used 44,632 sweeps versus 49,059 original passes; 1,372/2,048 cases still used 24.
There was no few-sweep reduction large enough to offset this dependency chain.

The prior CPU product comparison needs a native accounting correction:

| Saved-visit product proxy | Products |
| --- | ---: |
| Original EX1 + recurrence | 84,268,494 |
| CPU candidate, changed response updates only | 57,506,292 |
| Native-equivalent unconditional response + extra cross setup | 72,007,650 |

The CPU counts only 527,826 changed response-row actions; the native loop performs
response arithmetic for all 1,293,603 visited rows. Native also prepares two
additional normal/tangent cross terms per triplet: 717,372 products. Thus the
same-visit proxy reduction is **14.55%, not 31.76%**. This is source-derived work
accounting, not native instruction counters; it excludes retained geometry,
whitening/decode and projection/collective costs. Native FP32 stopping can also
differ from CPU stopping.

Source-matched hidden AOT (SM120/SM103) reports tier32 83/89 registers and
5,924/4,624 shared bytes; tier48 128/95 registers and 8,264 shared bytes. All four
have zero stack/spill loads/spill stores. These are resource facts, not measured
occupancy, bank-conflict or stall attribution.

Register-resident response accumulation could remove per-transaction shared
loads/stores and a fence, but cannot retire ordered contact dependence, dot
reductions, projections or stopping scans. No evidence prices that removable
work at the 76–80% owner reduction needed merely for replay parity. Reaching a
1 ms whole saving would require removing 14.385 ms RTX / 16.251 ms GB from this
candidate. No further mapping, register, stopping or parameter variant is funded.

## Reproduction and pins

Native records: `/tmp/fpgs-spectral-gs-native-20260916-S83DC8tM/gpu{0,1}.log`.
Whole records: `/tmp/fpgs-anymal-spectral-gs-whole-paired16k-20260916-{01,02}`.
AOT: `/tmp/fpgs-spectral-contact-offline-tVyCzVM2/offline02/report.json`.
Prior CPU study: `/tmp/fpgs-anymal-spectral-gs-qualification-6Nxtdj/summary.json`.
The whole manifest pins the original runner, inputs, source fingerprints and
capacity/owner checks. The observer/launcher is preserved under
`tools/fpgs_bench/spectral_capture_20260916/`; it reuses the original fixed-variant
parent, rather than introducing a benchmark protocol.

SHA-256 values:

```text
spectral_contact.py       aedd42a5ef76041620c7333bf34029c06d12d1410aeed1deaf366befa8f69943
solver_feather_pgs.py     86d80aefc3ae9ebf17fa1929bf392052eb0d4b9c2dbeacb50c1c230bc93819e0
coupled_contact.py        3e972df7b7e2659716718595a49164a3a7ed9f94c2d874c0e29855dca3384015
spectral_gs_control.py    a39eb83e4f5b240a24416f96683df0cc5e7453c8f180cbf539f9d752141256c7
test_spectral_contact.py 0033c026704319795cc332c0afd62ac0196b8d4765d01f84fd834e1c64100988
coupled_saved_probe.py   10da54f609ee3e098a188bd6c92ad705f2fb58a9e534e900f152bba2bda66fc9
checked_capture.py       ee59f76a4a1d10671503a0f08e8ce1706f2c1843592c5bd06ab533fc81d089dd
nsys_checked.sh          0e5c5b3153be3be947c8fa3f297bec20fd1a18b8e4a604a73ae3a2a93d7bb30f
run.py                   e97e276e0891e125d1484008f71a433db4e072030a23735b0080b679f8cd2b65
native gpu0.log          e57f5956fdb647aac48552e446090380716db4d3a38a8612df432df8c13efe30
native gpu1.log          3d2f04a7d5e570227ef0e4cdcf11909820604acb5566d8b95e05d2de53dd12d2
whole01 manifest.json    ea6c90f21d5574a59a868587dc5a15097ab31294119c6185fbab3582f417d9cb
whole02 manifest.json    fbed5aa677ff0283a141d6eb4e1f56b23c09529405fc58a9068310ab6c09e3eb
AOT report.json          b7b7cc0d0b8d8732d2f8998c7fe02a6efff6e62b0e6149a3992ab63809425872
CPU summary.json         515fddfc5565ce179309a1b0c29ea3081b4db47e6119c6dd1f7256b0021acf82
```

CPU controls (CUDA hidden):

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python \
  -m unittest tools.fpgs_bench.test_spectral_contact \
  tools.fpgs_bench.test_spectral_gs_control
```

Native saved-input command, run only by the root GPU owner, substitutes the
appropriate visible UUID and `--gpu-source 0` or `1`:

```bash
uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python \
  -m tools.fpgs_bench.coupled_saved_probe --gpu-source 0 \
  --candidate spectral_contact \
  --candidate-sha aedd42a5ef76041620c7333bf34029c06d12d1410aeed1deaf366befa8f69943 \
  --register-whitening
```
