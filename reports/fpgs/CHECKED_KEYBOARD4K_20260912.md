# Production checked Keyboard 4K audit

Read-only audit of `/tmp/fpgs-checked-keyboard4096-20260912-01`; no GPU work or source changes. [checked_keyboard4k_20260912.json](checked_keyboard4k_20260912.json) records exact source, capture, log, SQLite and Nsight hashes. All eight recorded driver/helper hashes and both clean checkout identities were independently rechecked. All 640 selected direct graph records were independently matched by process/correlation in read-only SQLite and their durations and 40-step means recomputed.

## Result and scope

All four captures completed with return code 0 and no cleanup signals. Each has two successful checked boundaries: eight boundaries total. State is finite at both boundaries; all available FPGS/MJWarp flags and all 13 collision flags are zero, with no unsupported collision owners. This is successful boundary/sticky-flag validation, not a per-substep convergence, constitutive-law, trajectory-parity or complete-demand proof. The separate long diagnostic run supplies the stronger per-substep observations; do not attribute its 40.6 million world-solves to this checked wrapper's boundary mechanism.

Both backends used clean Newton `360388d1b072663d499c41c3ffec4ad63dd6f0b4`, and unchanged Lab `1d8feb82d17dbfab8f0772de56f84deae2cb7974`. The MJ arm loaded helper `c5f0a6aef071e9d47baf3c646d1f8bfe715b3a31f6a079f9b244b09f080e368c`, patch `pyramidal-sign-acquisition-adjacent-float-v2`, generated factory `485f85ccb4af6b8d0e4ec45be56e2b6c7e3300612e203d5e59c5dea98b934d0d`. Actual PYRAMIDAL+NEWTON support passed both boundaries. Iteration budgets, tolerances and warning bits were not changed; friction-delta was off. Installed packages were not edited. MJ numerical work is explicitly marked modified, not presented as stock MJWarp.

## Timing: graph advantage is not environment throughput

| GPU | FPGS physics graph ms | MJWarp physics graph ms | MJ/FPGS graph ratio | FPGS environment wall ms | MJWarp environment wall ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | 11.186489 | 27.797029 | 2.48488 | 37.411073 | 32.118683 |
| GB300 | 10.801294 | 27.977880 | 2.59023 | 34.040587 | 32.438661 |

Graph numbers are **means of 40 per-environment-step sums of four direct physics graph durations**, 160 graph roots per capture and no auxiliary roots. They are not the median of those 40 observations. `summary.json` calls them `median_us` because it takes a median across one repeat. Its zero spread is not evidence of repeatability. Wall numbers are the separate unprofiled 1000-step environment averages. FPGS is slower in that full-environment measurement despite the graph advantage; host/reset work and different resulting trajectories are included. Do not claim training throughput or a speedup over the earlier broken-capacity baseline.

This is one round, FPGS first then MJWarp, each backend paired across the two UUID-pinned GPUs. Both use 4096 worlds, seed 0, 200 warmup steps, 1000 unprofiled steps and 40 profiled steps; sim dt 0.01, two substeps (solver dt 0.005), environment decimation four. FPGS retains its task recipe flags (group lanes 16, masked rows 1, narrow threads 4); MJ inherits none of them. Presets, trajectories, contact counts and finite-iteration solver behavior differ; this is not a same-input quality comparison or equal-work experiment.

## Capacities and process evidence

FPGS: dense rows 704, rigid contacts 147456, explicit broad output 57344. Boundary dense maxima are 582/570 (RTX) and 573/575 (GB); these are snapshots, not full-run high waters. Configured MF/propagation attributes are 64/704, not proof of physically allocated buffer sizes.

MJWarp: rows/world 320, pooled contacts 32 per world = 131072, Newton rigid contacts 131072, explicit broad output 45056; recorded sparse row capacity `njmax_nnz=2246`. Both pipeline constructors recorded effective requested broad capacity, split GJK/MPR, block 128 and full input pair count 14864384. Capacities are recipe/scale-specific, not universal upper bounds.

The driver enforces GPU idle before batches, source guards before/after batches, and reaps owned process groups. The four `cleanup_signals=[]` records are consistent with groups already gone. The manifest does **not** store a final GPU-wide idle sample. Root separately reports an empty compute-app query immediately after reaping at 01:41 UTC, before the next launch; this is attributed parent evidence, not independently recoverable from these capture files.

No solver capacity/line-search warning was observed in the capture logs. Unrelated shape-color FutureWarnings remain. Both MJ logs still explicitly warn that installed MuJoCo and MJWarp 3.12.0 differ from Newton's declared `~=3.11.0`; the helper's exact 3.12 source guard is not a general dependency-compatibility waiver.

## Reproduction and durable retention

Run only with the two GPUs available, the pinned clean sources and exact installed dependencies; choose a new output path. This is the same one-round recipe, not a balanced confidence benchmark:

```bash
uv run --no-project --python /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910/.venv/bin/python python \
  /home/octi/Projects/newton-fpgs-checked-frozen-20260912/tools/fpgs_bench/compare_backends.py \
  --isaaclab /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910 \
  --fpgs /home/octi/Projects/newton-fpgs-checked-frozen-20260912 \
  --mjwarp /home/octi/Projects/newton-fpgs-checked-frozen-20260912 \
  --output-dir /tmp/fpgs-checked-keyboard4096-REPRO-NEW \
  --task keyboard-so101 --gpus 0 1 --repeats 1 --num-envs 4096 \
  --warmup-steps 200 --steps 1000 --profile-steps 40 \
  --check-overflow --mjwarp-linesearch-fix \
  --capacity keyboard-so101:fpgs:dense_max_constraints=704 \
  --capacity keyboard-so101:fpgs:rigid_contact_max=147456 \
  --capacity keyboard-so101:fpgs:broad_phase_output_max=57344 \
  --capacity keyboard-so101:mjwarp:njmax=320 \
  --capacity keyboard-so101:mjwarp:nconmax=32 \
  --capacity keyboard-so101:mjwarp:rigid_contact_max=131072 \
  --capacity keyboard-so101:mjwarp:broad_phase_output_max=45056
```

Minimal durable report evidence: this Markdown plus [checked_keyboard4k_20260912.json](checked_keyboard4k_20260912.json) (about 13 KB, exact sources, capacities, aggregate metrics and raw hashes). Prefer additionally retaining the original manifest/summary/ratios and all four capture/check/analysis JSON triplets: only 323137 bytes total, including the 40-step samples and launch membership. Raw Nsight, SQLite and logs remain local and are hash-addressed by the compact JSON. Hashes establish identity but cannot recover deleted traces; the compact summary alone cannot rerun raw graph-membership analysis. Nothing was copied into the feature/report or archived by this audit.
