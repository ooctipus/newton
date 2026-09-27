# Franka kinetic paired16: one complete-owner correction, pre-code

Status: proposal only; no runtime edit, new worktree, or GPU execution authorized
by this card. Root must read and approve before implementation.

## Frozen cause and comparison

Read the complete latest report at
`/home/octi/Projects/newton-fpgs-franka-kinetic-state-20260915/reports/fpgs/FRANKA_KINETIC_STATE_20260915.md`,
report HEAD `edf8b25200afbc53fe974ec137f0ab470e3bc339`.
If funded, create one isolated branch/worktree from host-fixed native parent
`edd49d0b407a7e0720e0f028e067de90861fb363`; preserve the old tree and all failures.
Accepted whole-performance comparator remains qualified `1a9efc33`, not the
unsuccessful kinetic prototype. Reuse its exact original runner and capacities.

Post-host-fix whole02 measured RTX 5.457961 -> 5.466557 ms and GB
5.025921 -> 5.383826 ms. Wall remained 28.784147 -> 44.972803 ms RTX and
27.884037 -> 44.296596 ms GB. No promotion. The remaining approximately 16 ms
host-proof cost is separate and remains charged; do not optimize hashes in
this experiment or claim improvement over the broken 1.69-second prototype.

Matched nodes are at
`/tmp/fpgs-franka-kinetic-state-nodes-paired16k-20260915-01`.
The original `ce09a496` strict reader confirms 12 physics roots per capture,
three environment steps, and 24 calls to each of the three new owners. These
are three diagnostic steps, not three repeated throughput benchmarks.

| Complete state/factor/drive/mask family (ms) | RTX old | RTX new | GB old | GB new |
| --- | ---: | ---: | ---: | ---: |
| Summed kernels | 2.513538 | 2.055565 | 2.089395 | 2.141363 |
| Union | 2.105090 | 2.024397 | 1.695736 | 2.113129 |
| Exclusive busy | 2.026978 | 2.019715 | 1.635162 | 2.105043 |

New RTX finish/predict/repair sums are 1.047489 / 0.538603 / 0.216000 ms.
GB values are 1.177771 / 0.534379 / 0.191637 ms. No repeated full-repair or
H-request bug was found: one expensive first repair per environment step,
then short current/held guard calls. All retired global producers are absent.
Old tau/factor/composite overlap was 0.271510 ms RTX / 0.260445 ms GB.
**Packing does not restore this overlap, and receives zero credit for it.**

## One fixed mapping across all three owners

Use one 32-thread CTA for exactly two independent worlds. Hardware lane
`h` selects subgroup `g=h>>4`, local lane `l=h&15`, and world `2*CTA+g`.
The one mapping applies to repair, finish, and current-force/held-factor
prediction. Keep the three launches separate, eight calls each per environment
step; no phase fusion, new producer, persistent queue, or block-size sweep.
Keep the existing default-off `FEATHER_PGS_FRANKA_KINETIC_STATE` admission and
factory ABI. Add a distinct `_p16` suffix to these three kernel names for the
existing untimed actual-owner checks; factor9 and its name stay unchanged.

Each subgroup owns its own original 896-float state scratch, or 120-float
predictor scratch. Retain the existing offsets and global KineticPlan/cache
layout; this is not a scratch-relayout or cache-lifetime experiment. CUDA grid
size becomes ceil(worlds/2) CTAs, block32. The CPU reference path continues
one complete world per logical item and its existing arithmetic.

Preserve all 13 physical bodies, 21 generalized coordinates, primary9/free6/
prescribed6 ownership, and the exact parent-jump composition order. Pose and
motion scans still perform four rounds; only padding lanes13..15 participate.
Use subgroup masks `0x0000ffff`/`0xffff0000` and shuffle width16. Replace every
full-warp sync/vote used by these owners with its corresponding subgroup
operation; no block barrier may depend on another subgroup's cache admission.
Out-of-range odd tails and current-valid early returns must be uniform within
their own 16 lanes and independent of the other subgroup. A valid subgroup
must not access the absent world's plan, memory, or source tags.

Keep the same body/subtree, per-row and per-factor arithmetic order. In
particular, subtree reductions remain body10 down to1; L9/L6 solves remain
the original forward/back substitutions. They now occupy four lanes per
hardware warp across two worlds instead of two lanes for one world. No
precomputed inverse, new factor arithmetic, coefficient truncation, changed
PGS projection, iteration, tolerance, timestep, or contact law is allowed.

## Complete cost allowance and adverse work

All three owners total 1.802093 ms RTX. Ideal 2x throughput, not a prediction,
would yield 0.9010465 ms and save 0.9010465 ms. The requested conservative
owner reduction is at least 0.750000 ms: combined time <=1.052093 ms, or
1.713x faster. Thus the mapping has only 0.1510465 ms headroom above ideal
halving (about 16.8% of that ideal replacement cost).

Explicitly charge:

- 21-coordinate update/predict output requires two local-lane rounds, not one.
- Refresh collection of 19 independent components requires two rounds, not
  one; bias-only collection still has six components.
- H81 output takes up to six iterations per lane instead of three. H81 values,
  complete nine-entry rotated inertias and factor9 input are unchanged.
- State payload doubles from 3584 to 7168 shared bytes per CTA; including the
  old 128-byte generated overhead projects about 7296 bytes, not an AOT fact.
  Predictor payload doubles from480 to960 bytes (about1088 with old overhead).
  Compile and charge actual registers, local stack, shared bytes and barriers.
  Old finish uses80/90 registers and72 stack bytes; no occupancy claim follows
  from packing, and larger shared storage can reduce resident CTAs.
- All physical memory traffic, four pose/motion rounds, required full free
  I36 and prescribed V6 services, source/status checks and live external-force
  reduction remain. The split intrinsic/external reductions remain separate.
- Retained factors, drives, row formation, PGS, sensor/publication consumers,
  clears and all overlap remain charged. Do not assume the observed RTX mixed
  packet growth or remaining host regression disappears.

Finish plus repair alone could save only 0.631745 ms even under ideal halving;
that does not support the 0.75 ms margin. Predictor must be included. Its
additional ideal-halving opportunity is 0.269302 ms. The previous paired
publication trial's approximately0.2 ms gain is not proof of this result:
this proposal also packs the measured0.539 ms live-force/held-L owner and
the compact bias/H producer, whose caches did not belong to that old boundary.
It remains a packing correction, not a globally new algorithm.

## Verdict and bounded qualification if funded

**Plausible enough for one causal experiment, but high risk.** No source bound
makes <=1.052093 ms impossible: most body work uses only13 lanes today, and
serial predictor solves use only two. There is also no measured evidence that
the extra rounds/shared-memory cost fits the narrow0.151 ms overhead allowance.
Neither 2x owner scaling nor >=0.55 ms whole saving is promised.

Reuse the original CPU and three native physical selectors, independent FP64
H/bias/current-force/public-velocity references, original8-sweep loaded solve,
held/current factor epochs, reset/model notification/request consumption and
graph/source-bank checks. Add only the necessary existing-fixture odd-world
and mixed-validity controls (e.g. five worlds: cached/invalid subgroups and an
absent paired tail); do not weaken the physical oracle. Verify all13 outputs,
free/prescribed services and both state banks. Confirm unchanged factor source.

After source/CPU/AOT review and freeze, root owns paired native runs, then one
original whole A/B against qualified1a9 at16K/200warm/40wall/40physics, followed
by source-matched nodes only if needed for the decision. Success requires an
actual >=0.55 ms RTX whole-physics saving with all original guards and no
unexplained GB regression. Host-cost repair is still needed before any final
promotion. If the combined mapping misses, close it; no tile/register grid.

Current reviewed module SHA256:
`e776ea36c9bc03fbfdebce4e7ea917efd6e4b352ef1a5653b57f05adc9a78f65`.
Retained factor SHA256:
`eda0a22bb7328cef4f691970a8551fb5bc98d209c51977dbca60123ee91c9b8e`.
Existing physical test SHA256:
`25931d1c3b1c434bfd6bd8315b62a517b50c17b8f99ea05f443b3bbec6d06c4d`.

## Funding record

Root approved this single correction at 02:18 UTC on 2026-09-15, with a
03:45 UTC checkpoint. Implementation is isolated on branch
`ooctipus/fpgs-franka-kinetic-paired16-20260915` from `edd49d0b`.
The original pre-code card remains unchanged at its external path.

## Implementation and CPU/offline checkpoint

All three owners now map two independent CUDA worlds to 16-lane subgroups of
one 32-thread CTA. Each subgroup has its own storage slice, masked scans,
barriers and finite-value votes. The incomplete final pair returns before any
world array access. The original serial CPU owner and numerical operations are
unchanged. Factor source, solver hooks, iteration budget and host notification
proof are unchanged.

The missing paired-owner key regression failed before implementation. The
seven owner CPU controls and three retained-factor controls now pass (10 tests,
1.665 seconds). Existing saved/current/held and physical tolerances are retained.
The actual native lifecycle fixture now has five worlds, mixed repair validity
in both pair orders, an absent upper tail, graph empty/regrow, and an upper
subgroup failure-vote isolation check. These CUDA controls have not yet run.

Offline compilation passed all eight SM120/SM100 entries without GPU use:

| Owner | Registers SM120/SM100 | Shared bytes/CTA | Stack bytes | Spills / CTA barriers |
| --- | --- | --- | --- | --- |
| Repair | 64 / 64 | 7296 | 72 | 0 / 0 |
| Finish | 80 / 80 | 7296 | 72 | 0 / 0 |
| Predictor | 42 / 40 | 1088 | 0 | 0 / 0 |
| Retained factor9 | 40 / 40 | 1808 | 0 | 0 / 0 |

The report is `/tmp/fpgs-franka-kinetic-paired16-offline-M62eLv/offline01/report.json`,
SHA256 `a860cc987b0ca1daf0b5920f98655e7d2165ea4d853d18a9fb04bfb5ed4250b1`.
Runtime SHA256 at this checkpoint is
`544266cd56498da2525d17503e17edf6bd01a9dbf438553ca4d6676d25b88ce9`.
These are compilation/resource facts, not occupancy or throughput evidence.
Paired native physical checks and the original whole A/B remain root-owned and
required before any performance or qualification claim.
