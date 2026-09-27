# Late all-world Kuka publication experiment

This default-off Newton-only experiment targets the complete late generalized
integration, forward-kinematics and body-dynamics publication boundary. It is
not yet a measured or accepted optimization. The fixed goal remains **Kuka
Allegro and Franka at least 4× corrected MJWarp whole-physics throughput**.
Whole-environment wall time is separate; neither metric is full RL training.

Branch: `ooctipus/fpgs-world-scan-publication-20260912`, based on preserved
joint-world experiment `5067ae8dc26f84f276d2032336018b9c7417a2f1` and accepted
combined progress `064ec8ac455fc4cde557a3b54a1a62624cf56441`. The inherited
handoff remains in the ancestry. Only the `ooctipus/newton` fork is used.
Isaac Lab remains unchanged at `1d8feb82d17dbfab8f0772de56f84deae2cb7974`;
no task settings, dependency pointers, contact capacities or GS allowance change.

## Fixed reference and measurable checkpoint

Accepted balanced three-round physics medians, milliseconds per 16K batched
environment step, remain:

| Task / GPU | Accepted FPGS | Fixed corrected MJWarp | MJWarp/FPGS |
| --- | ---: | ---: | ---: |
| Kuka RTX PRO6000 | 13.973622 | 27.464577 | 1.9655× |
| Kuka GB300 | 13.884428 | 25.954644 | 1.8693× |
| Franka RTX PRO6000 | 5.746087 | 10.708025 | 1.8635× |
| Franka GB300 | 5.195039 | 10.348531 | 1.9920× |

The MJWarp denominators are the earlier-today corrected single-round discovery
references, not new repeated accuracy-matched runs. The old warning-heavy
Kuka 170 ms/11.05× claim is not a valid target comparison. Target ceilings
are Kuka 6.866144/6.488661 ms and Franka 2.677006/2.587133 ms (RTX/GB).
The 4× objective is not achieved.

Source-matched prior Kuka publication costs 2.227286 ms exclusive: generalized
qdd/transport/integration .311659, FK 1.438347, full finalizer .479542.
To save 10% of accepted 13.973622 ms physics, the complete replacement must
fit **.829924 ms**, counting all added work. The unmeasured planning range
is .70–1.15 ms, so the hypothesis is tight, not a promised gain. Replacing
only FK/finalization would leave an implausible .518265 ms allowance and is
not the selected narrow experiment. No factor, collision, row, GS or public
contact-force work is credited as removed.

## Mechanism and ownership

`FEATHER_PGS_WORLD_SCAN_PUBLICATION=1` enables one 32-thread block per complete
Kuka world, after the existing paired/general/MF joins and velocity clamp:

1. Execute the original qdd, free-root transport-removal and integration laws
   for all 35 physical DOFs and 32 joints, including prescribed free motion.
2. Form each current local joint transform and compose the parent tree with
   four doubling rounds, instead of serial/global parent traversal.
3. Form current solve-frame motion subspaces and scan associative segments
   `(Vp,Ap) ∘ (Vc,Ac) = (Vp+Vc, Ap+Ac+cross(Vp,Vc))` for velocity and bias
   acceleration. Both dependent tree traversals are replaced.
4. Feed current local pose/velocity/acceleration values into the original
   full inertia, Coriolis, gravity and COM-velocity finalizer, and publish
   all canonical outputs. Mark cache-valid only after complete block stores.

All 15 output fields remain: v_out, qdd, q/qd, body pose/COM velocity, COM
pose, S, articulation origin, body spatial velocity/acceleration/bias wrench,
full inertia, compact inertia terms and cache-valid. Body bias wrench is
next-step dynamics data, not the sensor contact force. Original impulse-to-
public-force conversion remains unchanged. Full inertia is always refreshed
for free roots; other full/compact writes use the exact original next-mass-
refresh predicates. Held fields remain untouched on reuse.

The constructor reuses the checked complete world32/global35 parent plan,
rejecting depth beyond fifteen parent edges. Runtime binds current frames,
COM, inertia, gravity and state arrays. Numeric model notifications avoid
static readbacks; structural ownership changes require solver reconstruction
and graph recapture. Original reset and FK dirty/cache-source logic remains.
Unsupported models, gradient states, noncontiguous states or source/output
storage overlap retain the original complete owner before any new write.

The **joint-world predictor flag stays OFF** in this first comparison;
combining both owners is explicitly unsupported to prevent double integration
and double-crediting already-removed ZERO publication. No private early stream,
mixed early/late body list, row panel or factor panel is introduced.

Native offline compilation on sm120/sm103 reports 77/79 registers, 2624 shared
bytes, 72 stack bytes and zero spills. This is not an occupancy or speed claim.
The source-level logical publication traffic cannot be deleted; the hypothesis
is removal of dependent traversals and intermediate reloads, not skipped state.

## Prior joint-world checkpoint: measured small gain, not promoted

The predecessor first slice completed one 512 and one 16K paired node-profile
screen. At 16K it measured **14.500464→13.921781 ms RTX** and
**14.459837→13.864383 ms GB**, only 1.04157×/1.04295×. It is not repeated
promotion evidence and does not meet its proposed 20% whole-physics target.

The complete diagnosis is preserved: its light owner costs 1.249/1.093 ms,
versus a .65 ms forecast; raw indexing mostly overlaps, but the combined
retained-dynamics/indexing union grows .533/.346 ms. Active response saves
.244/.286 ms. Exact additive whole-span changes are replaced-exclusive
−.911/−.958, other device union +.300/+.302 and uncovered span +.033/+.060
= −.579/−.595 ms. Different current MF tails prevent attributing all solve
duration changes to architecture. The smaller changed-owner subset must not
be compared with the larger full row/MF/solve budget.

All source/idle, finite, four solver and thirteen collision flags passed;
current ZERO/active/fallback partitions were valid and predictor fallback zero.
These controls are not long-rollout convergence or backend physical parity.
No capacity inflation, warning suppression, micro launch/grid retry or
promotion followed this 4% screen.

## Validation and reproduction boundary

The independent native tests compare the complete original five-kernel law
against all 15 new outputs on four saved actual512 inputs, refresh/reuse,
moving free/prescribed roots and changed current frames/COM/inertia/gravity.
The host lifecycle tests were verified failing before implementation and
then all seven passed on CPU. CUDA tests and integrated timings are separate
gates; no runtime acceptance is implied by the initial offline compilation.

The numerical gate permits reassociation. An inherited 3e-6 original-relative
test failed at approximately 15 micrometres/s COM velocity difference. On
the same inputs, FP64 original-equation checks instead show maximum pose
error original 23.44 micrometres versus candidate 6.46, velocity 19.29 versus
8.11 micrometres/s, acceleration 11.04 versus 6.42 micrometres/s². The original
failure is retained; it does not justify rejecting the more accurate result.
Fixed physical/local error checks and exact held/cache ownership remain.
These numbers are CPU component evidence, not CUDA or rollout acceptance.

Root subsequently ran the retained combined/joint-world controls plus both new
test modules on each actual GPU: **100 tests passed on RTX and 100 on GB300**,
24.945/25.077 seconds. This includes native eager/graph execution of all four
saved publication inputs and numeric reset/replay checks; some retained tests
are CPU controls, not 100 distinct GPU kernels. Both children were reaped and
compute-idle verified before any profiler launch. Logs are
`/tmp/fpgs-world-scan-native-{rtx,gb}-20260912-01.log`. Full
`uvx pre-commit run -a` passes with every new file staged. Integrated timing
and physical rollout behavior remain separate gates.

Root owns paired RTX/GB runs with one job per device and source-pinned tools
`32164ded363de6d79e65371f8c89ffad411e8895`. First compare accepted064 against
this flag plus the same accepted ZERO/independent/paired-overlap/local-packet
flags, with JOINT_WORLD off. Use 200 warmup steps, 40 synchronized wall steps
and three node-profile steps at512 then16K. Reap both jobs and verify idle.
Only a qualified gain proceeds to balanced repeated40-graph timing and
current rollout/reset checks. A miss requires causal attribution before one
targeted retry; no tile/grid sweep substitutes for a structural cost model.

Evidence (large captures remain local):

- Proposal `/tmp/fpgs-late-world-publication-tL6uLwMG/CARD.md`, SHA256
  `843f56727f8d77a73e85555bfa9da90e72afee9aa9476cb83997e4800baa709a`.
- Joint-world diagnosis `/tmp/fpgs-kuka-joint-node-audit-uhDHSiXF/FINDINGS.md`,
  SHA256 `05bc0c2cf84c72c7c696aa3f88fe92fed3c65d24230161691b4155c8fbd18f29`.
- Joint-world16K manifest `31103877f30e00a645631e25379eac63b2f3ac81e9a51d5828952c05f56b9ba9`;
  joint-world512 manifest `2431ec03162a1d4d15d4a448b8d3d353218e4f8bee3b6d36cbc821c433f9f029`.
- Offline resources `/tmp/fpgs-world-scan-ready-JBp9U0tl/offline01/report.json`.
- Initial numerical failure/FP64 isolation `/tmp/fpgs-world-scan-controls-SFVT0E6Q/`.

This checkpoint preserves the hypothesis and its predecessor's measured miss;
it does not claim the next performance gate has passed.
