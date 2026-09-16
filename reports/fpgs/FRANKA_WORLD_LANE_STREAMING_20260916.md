# Franka streaming world-lane correction — 2026-09-16

## Decision

**Closed, unpromoted and default-off.** The single streaming correction passed
the existing physical/lifecycle controls on both cards, but lost the complete
paired whole boundary: RTX 5.141021→5.286214 ms and GB300 4.715610→5.077819 ms.
The approximately 0.5-ms whole-saving target was not met. No further mapping,
liveness or predictor-only variant is pursued. The original p16 baseline and
the earlier world-lane experiment remain intact.

## Complete change and retained contract

This isolated branch starts at `cd9315da48fba515944f4ac64d69e6a50c21720e`.
`FEATHER_PGS_WORLD_LANE_STATE=1` plus
`FEATHER_PGS_WORLD_LANE_STREAMING=1` selects the correction. With the new flag
off, the original world-lane path is unchanged; with both off, the accepted
p16 path remains selected.

The generated forward walk writes each primary body's own wrench and, on
refresh, 13 mass/moment/inertia scalars into lane-private shared fields. Current
screws use the same allocation. The reverse dependency walk holds completed
child suffixes, projects each completed subtree column immediately into its
ancestors' cached screws, and writes the original private bias/packed mass.
It no longer keeps all ancestor own terms and descendant columns live together.
No additional global own-term matrix, conversion launch or transpose is added.

One fixed 32-thread block owns 32 worlds. Uniform finish kernels stage primary
body pose/velocity in a reusable shared tile and cooperatively publish the
canonical AoS outputs using actual body IDs. Eleven primary bodies add two
full-warp joins each: 22 joins per finish call. Masked repair retains direct
publication because individual lanes can return early. The two original free
root helpers, current screws and other canonical outputs also retain direct
publication. Their complete costs are included; no public state is skipped.

All dynamic model/force inputs, free-root transport, current/held factor cadence,
state-bank/reset/model notifications, original contact preparation and finite
eight GS passes remain unchanged. The existing field-major private cache stays
`bias[9,W]`, `com_offset[13,W]` of vec3 and `geometric[45,W]`; canonical9×9/6×6
factors and public state are unchanged. No Lab changes or capacity increases.

## Physical and source evidence

The new-module regression failed before implementation. All five inherited CPU
controls passed, including independently assembled current mass/bias, held
momentum, original integration, masked reset and notifications. Root ran the
two inherited native controls on both cards: 2/2 passed per card in 39.235/40.261 s
including compilation. The loaded five-world p16 comparison exercises the
padded final warp, free/prescribed bodies, state banks, reset and graph replay.
No tolerance was widened.

Maximum normalized errors across the native logs: matched-eight joint velocity
`3.80599e-5`, body velocity `5.25445e-5`, current physical mass `1.06350e-6`,
held momentum `1.37255e-7`. This is the retained physical-control scope, not a
new independent full-task convergence claim.

Native logs: `/tmp/fpgs-world-lane-streaming-native-20260916-jLCQ8ben/gpu{0,1}.log`.
SHA256: RTX `721d9b46d96e35472dbd405b0f42aa23f84de116eb723144ed2be96ecd0c84fc`;
GB `5d7e052a178a7b79616fdd96b3ea1670dbb0ca56289ee7abac8a2cb705d100a9`.

| Measured source | SHA256 |
|---|---|
| `world_lane_streaming.py` | `27b78ebdd0184f2f23c686dd5041a12e8d961b02052a0deabc19a3371cfbe22c` |
| `solver_feather_pgs.py` | `fcdebb4a86841c5af0011a59efa4ceb231e308ecb5b6405c830879fd90876d8e` |
| `test_world_lane_streaming.py` | `e4ccad58133c056844b971cd77135acd8302666cfa64a9cca5713f1173c2e373` |
| `world_lane_capture_20260916/checked_world_lane.py` | `946cb5b476d3e541ad7b9b993d6f4bf109d6f97d11408f86771b8b2772f979c0` |
| `world_lane_capture_20260916/run.py` | `6e56efc60ef13b62fa9523a9d65edd647cecb0c9b5cc2baa3812436f03ddac29` |
| Unchanged `world_lane_state.py` generator/helpers | `ca6ec1cd6d86630fcfa641f4421f7ebfbe4a07cbceea94897a339dea1e96346b` |

The whole and node captures share candidate dirty-source digest
`85b8bec8c0cb615d3f0ce5d1b6c7a99168a80e2820a6c8c3c3822aa22d917386`.
Baseline Newton is `ca0d427af809571bb5501f644c1a6e03990cd2a8` in the retained
keyboard-linear-state checkout. Unchanged Lab is
`53ee6b44c2334341305dbdf385a3916c6b140799`; benchmark tool is
`961b7e2f751bcd1d8b03368e7956b54c81414897`.

## Paired whole result

The original p16 versus streaming capture used 16,384 worlds, seed 0,
warmup 200, wall 40/profile 40, one paired round, original raw capacity 32768 and
broadphase capacity 7680. Both arms explicitly retained `SIMPLE_WORLD_ZERO=1`,
`LOCAL_ROW_PACKETS=1`, `FRANKA_KINETIC_STATE=1`, `GROUP_LANES=16`, `ROWS_MASKED=1`
and narrow-phase threads 4. Only the two world-lane flags differed 0/0 versus 1/1.
All children and the parent exited 0; source/idle, finite, actual owner and
capacity guards passed.

| ms/environment step | RTX p16 | RTX streaming | GB p16 | GB streaming |
|---|---:|---:|---:|---:|
| Whole physics |5.141021|5.286214|4.715610|5.077819|
| Environment wall |27.704438|28.247018|28.006917|28.478986|

Physics baseline/candidate ratios are 0.972534×/0.928668×. These are single-round
discovery results, not repeated promotion evidence or RL throughput.
Manifest: `/tmp/fpgs-franka-world-lane-streaming-whole-paired16k-20260916-01/manifest.json`,
SHA `1b6d7a16c16069a57cc0bda4771ffccb9e9c38b5cfa3bd6fc77d36810d5b92e8`.

## Strict ownership diagnosis

The paired node capture changes only profile 40→12 and graph→node tracing.
Parent/all children exited 0 and all guards pass. Each strict result proves 48
physics roots, zero auxiliary roots and zero unproven graph nodes. The original
process/correlation/root scope, interval union and exclusive-busy algebra are
unchanged. Only three exact streaming owner aliases and streaming/block checks
were added to the pinned Franka reader.

| Exclusive ms/environment step | RTX p16 | RTX streaming | GB p16 | GB streaming |
|---|---:|---:|---:|---:|
| Complete state/held family |1.673867|1.840900|1.400349|1.764111|
| Publication |.916125|1.271590|.752576|1.240138|
| Repair |.156589|.155493|.099675|.148131|
| Prediction plus primary/free factor |.520379|.331840|.476423|.305240|

The family is an exclusive interval union, including drive/mask overlap; do not
blindly add the component rows. Family calls remain 56→40. All old separate
prediction/factor9/factor6 calls are retired; no duplicate producer appeared.
Finish remains eight calls, four held plus four refresh, with unchanged cadence.

Candidate raw owner-duration sums, separately labelled because small GB overlaps
are removed only in the exclusive table above:

| Four calls per environment | RTX streaming ms | GB streaming ms |
|---|---:|---:|
| Held finish |.618771|.590511|
| Refresh finish |.652819|.650381|

The measured loss belongs primarily to publication: +.355465ms RTX and
+.487562ms GB. Factor/prediction retains a .188539/.171183-ms improvement;
repair is approximately flat on RTX but adds .048456 ms on GB. The resulting
state-family regressions are .167033/.363763 ms. This attribution explains the
whole miss without assuming that all other independent node times add to wall.

### What the correction established, and what it did not

The intended register-lifetime change actually occurred. Both AOT and captured
native metadata show 80 registers for both finish modes on both devices, versus
167/255 RTX and 168/255 GB in the earlier world-lane owner. Static shared memory
increases from 512 B to 16,384 B held/33,024 B refresh. Repair uses 210/206 registers
and 33,024 B shared; predictor 152/148 registers and 128 B shared. All final AOT
kernels have zero spills. The original p16 finish uses 80 registers and 7296 B
shared. The final streaming grid is 512 blocks×32 threads; p16 uses 8192×32.

The first offline draft allowed the compiler to forward lane-private shared
terms back into registers in repair, producing 216 B of spill stores. Explicit
volatile own-term access restored the intended lifetime boundary before any
native timing. Final finish registers became 80/80. This was a source-realization
fix, not a timed mapping or parameter sweep.

The complete correction adds shared own-term/screw traffic and 22 full-warp
publication joins per finish while retaining serialized per-world body math,
live canonical model reads, current screws and original free-body publication.
It improves block coverage 128→512 but still has only 512 world-lane warps for
16K worlds. These are source/resource facts. No hardware counters establish
whether shared traffic, publication instructions, latency or another stall
dominates; lower registers alone did not translate into lower publication time.
The causal result is therefore a failed complete lifetime/publication correction,
not proof of a bandwidth limit or permission for another mapping grid.

## Diagnostic artifacts

- Node manifest: `/tmp/fpgs-franka-world-lane-streaming-node-paired16k-20260916-01/manifest.json`,
  SHA `5d28fa71376002c3b974d8e9578e878effb359af87d42cbdb54fc774865806c7`.
- External wrapper: `/tmp/fpgs-world-lane-streaming-node-Kb3s9LxD/run_node.py`,
  SHA `04363132f592c5b0d40fbbfa41b7ea0ba6ca17595cecea1155aded3576c227d3`.
- Strict reader in the same directory: `read_streaming.py`,
  SHA `2ba85fcbc53bd69ea585249ebd705bdf9c31d58fda6b6cd79bb07825144b4cc2`.
  It pins original reader `a2d68778063e125358de0a7696dd7805a89f2230285a363b2b5794a7457045f3`.
- Strict baseline RTX/GB JSON hashes:
  `0bd8028aa95466411311ee10a55a0f2e8e567713cc55d89eb74877e493984a50`,
  `2fc509f7e1612535757a73821e7003aa599681eb8ee96e029ae8961698ed40ee`.
- Strict candidate RTX/GB JSON hashes:
  `a39af8738f9ad0a239b7a6a860c6421dcc67ea98c1f381921483d93701fd0d36`,
  `e838d3e52ae7a5e6bbbb62f06e545b50eceba4f03c7757378b75e6ee8355d888`.
- Final CUDA-hidden resources:
  `/tmp/fpgs-world-lane-streaming-offline-aLghKm8f/offline02/report.json`,
  SHA `ff40fb1c3d347894e68354ea78d3c2e20e0dfccbcedc7c1f44ffc89b9a121936`.

Runtime and observer hashes remain the measured pins. The inherited handoff and
original experiment report are unchanged. Required `uvx pre-commit run -a` and
targeted pre-commit including every new/untracked owned file passed. No commit
or push is made before root review; no further performance variant is authorized
by this closure.
