# Existing box-box SAT discovery: no accepted gain

The existing SAT flag produced no validated speedup. Kuka failed unchanged dense-row capacity; Franka completed slower. The requested whole-physics 4x target remains unmet. These results preserve the distinction between a failed mapping, an unmeasured capacity-invalid run and rejection of an entire collision algorithm.

## Fixed experiment and native controls

Only `NEWTON_COLLISION_BOX_SAT=0` versus `1` changed between paired FPGS arms. SAT uses the existing box-box separating-axis/reference-face clipping and feature reduction, publishing at most four contacts per processed pair. Other shape routes remain unchanged; no convex-to-box conversion was implemented. Identical manifold points/counts are not required, but valid geometry, physical behavior and complete constraint capacity are.

The existing six SAT tests passed on EACH GPU, with zero skips, failures or errors. They cover wrapper geometry parity, sliding feature identity, corner support, speculative approach, shape margins and distinct corners at ordinary/millimeter scales. Their audits explicitly retain `physical_quality_accepted=false`: they are not full-task convergence/trajectory or unsorted-workload capacity certification.

Sources and recipe:

- Native gate and both Kuka arms: Newton `50dfa28d3aabe51f1b5b75721450efac2c35c5f1`.
- Both Franka arms and node diagnostic: Newton `064ec8ac455fc4cde557a3b54a1a62624cf56441`.
- Benchmark tools: `b22fa50583478a0e6b7b020e8b8f29d488934207`; unchanged Lab: `1d8feb82d17dbfab8f0772de56f84deae2cb7974`.
- RTX UUID `GPU-883586b6-3100-0610-81e5-3b4c26f45639`; GB300 UUID `GPU-ebfac9e8-02d5-d8a9-3bfc-bac64c62ffd4`.
- Each discovery: 16,384 worlds, seed 0, 200 warmup, 40 synchronized wall steps and 40 physics-graph steps, one AB round. This is not a balanced-repeat promotion.
- No capacity overrides: decimal 4,000,000 raw contacts; dense/MF/propagation rows 192/64/192. No inflated buffers, Lab/runtime edits, timestep or iteration changes.
- Original sim_dt 1/120, decimation 4, two solver substeps at 1/240, eight GS sweeps maximum and held-mass interval 2: four collision calls/eight substeps per environment step.
- Both tasks retain `FEATHER_PGS_{SIMPLE_WORLD_ZERO,INDEPENDENT_COMPONENTS,PAIRED_GENERAL_OVERLAP,LOCAL_ROW_PACKETS}=1`; Kuka additionally retains `FEATHER_PGS_{KUKA_JOINT_WORLD,WORLD_SCAN_PUBLICATION}=1`. Both retain group lanes16, masked rows1 and narrow-phase worker multiplier4.
- Checked capture binds explicit SAT requests to actual primitive-module `_sat0`/`_sat1` at untimed boundaries. SAT1 changes physics work, not physics budgets.

## Kuka: capacity-invalid, not a measured slowdown

Parent `/tmp/fpgs-kuka-sat-paired16k-20260912-01` failed: both SAT1 children exited 1 at the first post-warmup boundary with sticky dense overflow at 192. MF, propagation and incoming-contact overflow flags were false. Parent completion/reaping and final source/idle checks passed. No capacity was increased.

The solver check raised BEFORE collision metadata capture. Consequently SAT1 has neither a valid timing result nor actual-module/13-narrow-phase-flag proof; the environment request alone is not activation evidence. No SAT0/SAT1 ratio is valid. SAT0 itself completed cleanly: physics 12.801808/12.214191 ms and wall 31.784256/32.446695 ms (RTX/GB).

Existing evidence cannot distinguish legitimate additional manifold rows from duplicate/incorrect geometry. The retained first-overflow design would keep allocation unchanged, verify the live module before stepping, save unclamped demand (`slot_counter + dropped_dense`) and raw identity/geometry at first overflow, and stop BEFORE GS. It must preserve collision-generation poses separately from current substep poses. That observer is proposed, not implemented or executed here; blind capacity inflation is not a remedy.

## Franka: completed 40-step screen

Parent `/tmp/fpgs-franka-sat-paired16k-20260912-01` completed; all children exited 0, source/UUID/idle checks passed, actual SAT0/1 matched requests, and both boundaries passed all four solver and thirteen narrow-phase checks. Observed capacities matched the fixed recipe.

| Device | Physics SAT0 → SAT1 ms | Baseline/candidate | Wall SAT0 → SAT1 ms | Baseline/candidate |
| --- | ---: | ---: | ---: | ---: |
| RTX | 5.732994675 → 6.139955800 | 0.933719x | 22.953526 → 24.349741 | 0.942660x |
| GB300 | 5.191205275 → 5.490427925 | 0.945501x | 22.092088 → 23.296999 | 0.948280x |

Physics time increased 7.10%/5.76%; wall is a separate metric, not training throughput. Current raw counts increased 13.7–20.4%, dense rows 0.61–1.07%, while GJK queue items fell about 80% and generic-manifold items 72–73%. RTX's final maximum dense count 42 exceeds local40, but owner-population counts were not captured; this alone cannot price general-fallback work.

## One unchanged node diagnostic: why the forecast missed

`/tmp/fpgs-franka-sat-nodes-paired16k-20260912-01` changed only profiling to three steps/node tracing. It completed with source/idle/activation/capacity guards passing. Its whole deltas are +0.154788 ms RTX/+0.226306 ms GB, NOT the authoritative 40-step screen's +0.406961/+0.299223 ms. Do not extrapolate node fractions into the 40-step result.

The reader retains EVERY graph kernel and memory node: per arm/card, twelve process-correlated physics roots, 1,224 kernels and 660 memory records (540 memsets/120 copies). Unions and exclusives include memory; exclusive families plus multi-owner overlap and graph gaps close exactly to the measured whole window. Four CPU accounting controls pass.

| Node-window measure, ms/environment step | RTX SAT0 → SAT1 | GB300 SAT0 → SAT1 |
| --- | ---: | ---: |
| Complete collision union/exclusive | 1.239594 → 1.378614 | 0.996968 → 1.090474 |
| Primitive collision | 0.046208 → 0.252960 | 0.040885 → 0.185045 |
| MPR+GJK+generic-manifold time removed | 0.069097 | 0.050815 |
| Complete row/response/solve union/exclusive | 1.932864 → 1.942870 | 1.941002 → 2.074273 |
| Current typed-contact producer union | 0.176672 → 0.186422 | 0.212832 → 0.267167 |
| All local9/20/40 union | 0.582720 → 0.575552 | 0.601941 → 0.664437 |

SAT primitive cost overwhelms the modest generic savings on both cards. Primitive registers rise 72→123 RTX/72→114 GB, with unchanged 128-thread blocks, 3008/2432-CTA grids and 512 B shared memory. Generic grids also remain fixed despite fewer queued queries. Registers alone are not a demonstrated causal explanation; fixed dispatch and the surviving query mix are unresolved contributors.

GB additionally grows the producer/residual40 path; general GS is fully overlapped in both arms and is NOT the exposed regression. Local9/20 durations must not be added to local40. Dynamics/publication and other-memory exclusive shifts are small; exact additive decomposition/resources are retained in the audit. Changed manifolds/trajectories remain a workload caveat, not an established physical failure. No tuning sweep or algorithm-wide rejection follows from this diagnostic.

## Backend denominator and status

Both corrected Franka AND Kuka MJWarp reference captures use `external_newton_prefix` collision, as does FPGS; the earlier claim that Franka bypassed Newton collision was incorrect. These SAT runs are FPGS-only and do not retime MJ. Any promoted shared collision change requires a like-for-like corrected-MJ retiming, while the historical denominator/4x target remains explicitly labeled. No SAT gain, whole-task physical acceptance or achievement of 4x is claimed.

## Artifact pins

The following SHA256 values were rechecked for this draft. Detailed child/capture/source/resource hashes remain in the parent manifests and independent reports; final idle status is recorded parent evidence, not a new GPU query.

| Artifact | SHA256 |
| --- | --- |
| `/tmp/fpgs-sat-native-paired-20260912-01/manifest.json` | `f495ce2eb11083f2141f0303240bd170c09f1b2f1d47ec404097125c39cb7c57` |
| `/tmp/fpgs-kuka-sat-paired16k-20260912-01/manifest.json` | `830cbe38c888977a17918db7fba1b00cdfd5c202ff378778a568a2f09c032321` |
| `/tmp/fpgs-franka-sat-paired16k-20260912-01/manifest.json` | `ca334f617083ef5b44d485df087d827cf6491ecc438cd14b9e013231e9e99519` |
| `/tmp/fpgs-franka-sat-nodes-paired16k-20260912-01/manifest.json` | `4f8331056c36aeee805ea294372a610d1508ba0379637937d65be76689bad105` |
| `/tmp/fpgs-sat-algorithm-card-dJZQQVIg/CARD.md` | `dff9e7006e7bea648214be45d724ba4b4c3d2311ea8b8d64335ab5e3f451be28` |
| `/tmp/fpgs-sat-screen-report-bDuwIy8V/KUKA_FAILURE.md` | `b984ede6c18b48775877df65282eaaf02ad5232c9f701a53d4c5b39646ff9e5b` |
| `/tmp/fpgs-sat-screen-report-bDuwIy8V/FRANKA_SCREEN.md` | `73801bd7e6c6c50dc58e8ce9970300ddee6e636c604f53ca34614f15d30afb12` |
| `/tmp/fpgs-franka-sat-node-audit-UaqzO7em/FINDINGS.md` | `e0bf72afc343e0aaa4c94c2f8beb3e790ca9ffde28675c9730a2731ac5853652` |
| `/tmp/fpgs-franka-sat-node-audit-UaqzO7em/evidence.json` | `962ab1e461796d06efa41a1f0c29262e4e97001722c94e447a9f098429b67590` |
| `/tmp/fpgs-franka-sat-node-audit-UaqzO7em/audit.py` | `fac6f151a47fafe59dede84e4d8dceaad1ca312856de268e3b7a9db7a47d0a35` |
| `/tmp/fpgs-franka-sat-node-audit-UaqzO7em/test_audit.py` | `46162c56665e508128fcb0bc4fd7f3a434a9eabcd72c838b4653e64b3832b71c` |
| `/tmp/fpgs-corrected-mj-scope-3o6CabAP/FINDINGS.md` | `3f8ee6dc95690381a7adffd86fbc5cf935d7daa7d9f22d68dff646ca39ed8ea8` |

Readback: `sha256sum <exact artifact paths above>`. This report preparation used CPU/file reads only and changed no existing frozen report, checkout or GPU state.
