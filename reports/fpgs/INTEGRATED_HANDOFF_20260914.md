# Ten-hour accepted-runtime integration

Work window: September14, approximately08:58--18:58 UTC. Objective remains
4x corrected MJWarp whole-physics throughput across Franka, KukaAllegro,
Allegro, ANYmal-D Flat, G1 Rough and SO101 Keyboard. This integration is
being qualified; composition alone is not a new performance result.

## Source ownership

Branch `ooctipus/fpgs-tenhour-integrated-20260914` on `ooctipus/newton`,
based on `7df75c46bd468b64c17958190fc800cfeb62201e`. The inherited handoff
and original ancestry remain intact. Isaac Lab stays at
`53ee6b44c2334341305dbdf385a3916c6b140799`, with no source or parent-pointer
edits. Its existing local interpreter symlink is not a Lab runtime change.

Composed accepted source changes, with original branches and artifacts retained:

- `7790c35b348dd1885a90f6c2def9b336cde80399`: conservative current G1
  terrain-query culling inside the existing append owner. Default-off flag
  `NEWTON_HEIGHTFIELD_GEOMETRIC_CULL=1`, requiring the existing cell/finite
  pipeline and supported original reducer. Report-only update `2f84ae50`.
- `99c796ca`: complete vectorized Franka packet-topology notification
  validation. No numerical, kernel or notification-law change. Report-only
  repeated result `04b2328e`.
- `b1799b6ccd91388a672b76692d608127654dfb2c` and
  `fba9fead70d17728f842954d66cf1f05f4617d40`: previously accepted Allegro
  coherent current-shape rejection and its physical test controls. This is
  carried forward, NOT a newly developed gain in this ten-hour window.

The geometry merge has two disjoint opt-in owners: coherent rejection
explicitly excludes heightfields, while geometric culling requires them.
No G1 compact, Allegro hill or unqualified Kuka kinetic source was included
at the original baf checkpoint. The qualified Kuka successor below is a later
composition. Failed prototypes must never become the comparison baseline.

## Qualification and reporting contract

Keep original task timestep, two Newton substeps, decimation and maximum
iteration allowances. Use current calibrated capacities and original sticky
overflow checks. No dropped contacts/rows or silent maximum-buffer inflation.
RTX PRO6000 is primary; qualify on GB300 as well, one process/device. The Kuka
repeat scheduling exception below preserves both-card coverage and idle guards.

Measure incremental FPGS improvement separately from corrected MJWarp ratios.
Shared collision changes and explicit thread mapping must apply to both
backends when their actual collision path uses Newton. Physics and synchronized
whole-environment wall time are separate; neither measures full RL training.
Numerical convergence and physical behavior, not bit-identical trajectories,
are the acceptance criteria.

Initial combined CUDA-hidden test run:26 selected tests,22 executed PASS and
four CUDA-specific skips. This checks merged source composition but does not
substitute for paired GPU physical controls or representative environment runs.
The combined runtime must receive those checks before promotion.

## Established results before composition

Franka's two alternating repeat medians improve environment wall time from
62.881182 to28.905997ms RTX (2.175368x) and63.050704 to28.939305ms GB
(2.178722x). Physics remains effectively unchanged. See the retained
[Franka report](FRANKA_NOTIFICATION_VECTORIZE_20260914.md).

G1's terrain cull repeats at1.072781x RTX /1.198608x GB incremental physics.
The fair shared-collision comparison is2.080706x /1.826664x corrected MJWarp;
both backends explicitly use the same thread multiplier. See the retained
[G1 report](G1_GEOMETRIC_CULL_20260914.md).

These task-specific results are not new measurements of this composed tree,
not evidence that other environments improved, and not the4x-across-tasks goal.
The full ongoing decision ledger remains on
`ooctipus/fpgs-structural-fourx-20260913` at report commit `8ac6e50d`.

## Qualified Kuka successor, September 14

Branch `ooctipus/fpgs-kinetic-integrated-20260914` composes qualified ad42 Kuka
onto `baf0c57ac5ef22bef452acd625b1727dd73a50b5`; measured runtime is
`6c2297ca3f7022df7e20ca0486f75c6ff818f29c`. Original worktrees and accepted
G1/Allegro/Franka features remain intact. Kuka's default emitted native source
and all 249 frozen native/descriptor records match its qualified predecessor;
the optional accepted single-factor branch remains available. Current/held,
reset and loaded coupled/type-4 controls pass. See the source reconciliation,
retained failed diagnostic and exact artifacts in the
[Kuka integration report](KUKA_KINETIC_INTEGRATION_20260914.md).

At fixed current 16K capacities, three alternating repeat medians against baf
are 12.506653 -> 11.119767 ms RTX physics (1.124723x) and
12.033355 -> 11.339816 ms GB physics (1.061160x). Wall medians are
36.993766 -> 33.680188 ms RTX (1.098384x) and
36.255405 -> 35.576291 ms GB (1.019089x). The original paired discovery's GB
wall regression, 35.482980 -> 38.251759 ms, remains explicit; one GB repeat
round also loses. The GB wall result is mixed, not a consistent throughput win.
External GB work blocked the first paired-repeat idle guard before children;
the completed three-round card repeats were scheduled separately with the
documented original-driver scheduling exception. All source, capacity and idle
guards pass. This is a qualified opt-in gain, not the 4x-across-tasks goal.

Reproduction requires the pinned external original driver plus the c062
fixed-Lab import adapter documented in the Kuka report. The prior direct
driver command omitted that prerequisite; local benchmark copies are not
interchangeable with the measured external tool revision. No timestep,
iteration or contact-capacity budget changes fund these results.

The successor remains on the `ooctipus/newton` fork, with inherited
`31cf87f4694f873a027e41e2ca5e9ad441234456` verified as an ancestor.
The original Lab handoff remains unchanged at
`/home/octi/Projects/IsaacLab.wt/contact-reset-20260913/reports/fpgs/HANDOFF_fpgs_20260911.md`.
Its historical source/timing context is preserved, not relabeled as today's
baseline. Lab remains pinned to `53ee6b44c2334341305dbdf385a3916c6b140799`.

## Current composed successor, September 14

Branch `ooctipus/fpgs-fourx-qualified-20260914` now combines the qualified Kuka
and Allegro sources. Code merge `17f8a19ae1a5c388838b1d9000d96f27ecd16b77`
retains parents 298c and 7fca; report merge
`1921e214b59a80046cb3804df6aee7d95e23bb02` additionally preserves a08's
completed Allegro evidence. Original trees and the inherited fork remain intact.
See the [composed integration card](FOURX_QUALIFIED_INTEGRATION_20260914.md)
for source proofs, exact selectors, results and reproduction prerequisites.

On composed 1921, six focused CUDA tests per card pass with zero skips:
16.141 s RTX and 16.971 s GB. Four cover Allegro current/held rows, physical
fallback and actual production reset/graph behavior; two cover retained Kuka
joint/predictor/publication queue controls, not full private-kinetic trajectory
requalification. Kuka's 249-record oracle and all eight Allegro original/keyed
PARALLEL native contracts remain unchanged. Allegro scalar fallback instead
uses the retained Kuka MF factory; its physical controls pass, but its native
bytes are not claimed identical to 7fca.

The qualified 7fca predecessor measures 4.206759x RTX / 4.922106x GB against
corrected native-contact MJWarp, with the retained GB outlier and incremental
GB loss documented. These are not new timings of the composed tree. The
all-six-task 4x goal remains unmet; neither test durations nor environment
wall comparisons measure full RL-training throughput. This update changes
reports only, leaving runtime, source pins and physics budgets unchanged.
