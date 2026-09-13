# G1 sparse + finite: first complete composition result

The first complete same-window A/B shows a material combined saving against the cell-only baseline. This is one discovery round; it is not repeated promotion, a new coupled-trajectory qualification, or the 4× target. The manifest explicitly keeps performance/physical-quality/repeated-timing acceptance false.

| Per environment step, 16,384 worlds | RTX cell-only | RTX composed | Baseline/candidate | GB cell-only | GB composed | Baseline/candidate |
|---|---:|---:|---:|---:|---:|---:|
| Whole physics, ms | 37.862898900 | 26.969148400 | 1.403933796× | 52.027578975 | 26.897414575 | 1.934296653× |
| Whole environment wall, ms | 52.043408675 | 41.402623575 | 1.257007508× | 66.504235224 | 41.987810076 | 1.583893875× |

Whole-physics savings against this cell-only baseline are 10.893750500 ms RTX and 25.130164400 ms GB. These are direct complete-window differences, not multiplied standalone gains or sums of overlapping node durations.

## Exact source and protocol

Baseline is frozen cell-only `a994b1b24b6073dcb10e855c10e8197ee116ef14`. Candidate is clean composition `8ecd28b1a2b9f0ebf2004d8953b89531c6ab2b96`, from pushed sparse report tip `954895ec` plus exact finite `882e468a`/`70f17773` cherry-picks. Candidate solver bytes equal measured sparse 3bae; collision bytes equal finite 70f. No packing/direct-query experiment or source edits were included.

Both arms use CELL=1. Baseline has SPARSE=0/FINITE=0; candidate has SPARSE=1/FINITE=1. Fixed Lab backend 53ee with original core 1d8, seed 0, 16,384 worlds, 200 warmup/40 wall/40 graph steps, one AB round. Sim dt 0.005 s, decimation 4, two 0.0025 s Newton substeps, original matrix-free finite-eight maximum and grouped dynamics. Same dense 100/MF 32/propagation 100, raw 294912, explicit broad 49152 and triangle 1769472. No physics budget or capacity change.

## Physical gate and actual dispatch

Paired physical parent 1528 ran at 16:21:46 and reaped exit 0 by 16:22:45. Both GPUs passed all 12 existing methods with zero skips/errors/failures. This covers unchanged sparse operator/current-row/held action, real Solver continuing serial/parallel graphs, reservation regression and original offending raw geometry, plus finite synthetic and both actual 96-pair pipelines/graphs. It is same-checkout physical/geometry evidence, not a newly added coupled-trajectory oracle. Historical warmed sparse component net-response caveats remain recorded in the sparse report; no tolerance was changed.

Physical output `/tmp/fpgs-g1-sparse-finite-physical-paired-20260913-01`, manifest SHA `a8a6826ae5aa66630bb7033264e9712f76f9184e09b6e4c90bfda3ed596250c5`.

Every original live capture check and both actual activation observers passed at both boundaries on both GPUs. Each candidate boundary had 16,384 valid sparse operators and zero sticky status. Finite handled/current-triangle counts were RTX 608087/609182 then 612225/613380; GB 609254/610340 then 614946/616084. Baseline had no sparse owner and no marked finite entries. Contact/row/collision status flags stayed clear. Manifold/count identity is not asserted.

## Timing and completion authority

Exact commands and helper pins are in `READY.md` (d3880fc9). Live parent 25303 ran after the physical gate without source or tolerance changes and reaped exit 0 by 16:25:10 UTC. Both source and idle guards passed; a fresh compute query was empty, and both GPUs were explicitly released to root.

Live output `/tmp/fpgs-g1-sparse-finite-live-paired16k-20260913-01`, manifest SHA `b9542b7d51e407a9f9cc9abc682af76a68c271420f8c3aad4aff879d1377fe5a`.

The sparse-only durable report is pushed at `ooctipus/g1-sparse-kinetic-20260913` tip 954895ec, file `reports/fpgs/G1_SPARSE_FACTOR_20260913.md`. The isolated composition and measured helpers remain frozen. Balanced repeated timing, same-collision corrected-MJ denominator and broader physical qualification remain separate next decisions. No additional GPU job is scheduled by this agent.

The full reproducible commands, source-bound activation mechanics and exact helper SHA256 values are in `/tmp/fpgs-g1-sparse-finite-compose-ATSRjnR2/READY.md`, SHA `d3880fc9f6b0fc0489d4d628014cec6a018a3e18bd04bf0550e702e34c5ff0a1`. The measured source tree stays at 8ecd28b1; this report is committed only on a separate report-successor worktree.
