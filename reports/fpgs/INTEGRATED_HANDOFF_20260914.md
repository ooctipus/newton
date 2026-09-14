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
No G1 compact, Allegro hill or unqualified Kuka kinetic source is included
at this checkpoint. Failed prototypes must never become the comparison baseline.

## Qualification and reporting contract

Keep original task timestep, two Newton substeps, decimation and maximum
iteration allowances. Use current calibrated capacities and original sticky
overflow checks. No dropped contacts/rows or silent maximum-buffer inflation.
RTX PRO6000 is primary; every GPU batch also runs GB300, one process/device.

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
