# Positive-shell face manifold correction

This isolated successor starts at `70f17773ea15cc03724635e3e1e9a270332865f6`.
The original finite query remains unpromoted after loaded Gate A02. Neither
solver iterations, physical tolerances, contact capacities, the reducer nor
Isaac Lab changes here.

## Observed cause

The original finite query already clips separated top-face patches and its
unreduced contacts cover both sides of the foot. The retained reducer does
not use its spatial support slots for positive-clearance contacts. Different
depth/voxel winners can leave a one-sided close manifold. On the exact same
saved finite geometry and held operator, changing only four close-row orderings
at the original eight sweeps spans near-zero to9.69rad/s spin. One ordering
reproduces the prior GB9.67rad/s failure within1.89e-6 in public velocity.
The reduced-only diagnostic also captured actual RTX4.86rad/s spin.
Jacobian reconstruction from physicalY/heldH agrees with raw geometry within
8.20e-8; the bad RTX angular impulse agrees with public force torque within
8.18e-9. This is poor fixed-sweep manifold/order robustness, not force export.

Full immutable evidence and CPU derivation:
`/tmp/fpgs-heightfield-finite-qualification-J9kBvfGP/RESULT_REBOUND_CAUSE.md`,
SHA256 `41635503e2c8fdb96159f68295a0366538aa36143b8d19afec8bea02080bfbd8`.

## Bounded corrective experiment

Preserve the pure finite geometry query and add one internal writer-policy
wrapper. A returned top-face manifold whose minimum geometric distance exceeds
the current summed shape margins, but is within the current summed gap+margin
threshold, remains unhandled for the original generic query. No marker or
contact is written before this handoff. Far witnesses, finite edge contacts,
touching/penetrating manifolds and existing unsupported fallbacks retain their
previous paths. The original global triangle stream, full fallback scan,
writer, reducer and capacities remain unchanged.

This is conservative current-law admission, not a claim that clipping was
missing or that original contact generation is universally robust. It adds
analytical evaluation followed by generic work for these face queries; that
duplicate work, additional contacts and any solver cost must be included in
the next complete A/B. Previous finite discovery savings are not transferred
to this successor without measurement.

The same96-fixture audit now queries the actual policy with current margins,
so its admission/marker identity still checks the production contract. Pure
primitive geometry checks remain unchanged. A focused existing-module test
first failed on the old positive-face admission and then passed with the
policy; touching, penetration, far and border controls remain analytical.

## CPU checkpoint

Seven native/dispatch tests pass; three independent geometry controls pass
with the two CUDA-only methods explicitly skipped. Both existing actual96
fixtures pass complete direct/reduced collision, empty/regrow ownership,
contacting-shape/minimum-separation and native-query witness checks on CPU.
The GB fixture retains1145/1190 analytical entries; the45 extra face fallbacks
are charged work, not dropped contacts. No GPU physical or timing acceptance
is claimed at this checkpoint.

All four complete dispatch kernels compile offline for SM120 and SM103.
Analytical direct uses127/128 registers and reducer132/128, with592-byte stack
and no spill loads/stores; original fallback remains168 registers. These are
resource observations, not a performance prediction.

Reuse the frozen Gate A and reduced-rebound diagnostic, original paired runner,
and existing96 physical tests on both cards. Preserve known unreduced-capacity
and friction-control failures rather than changing them. First checkpoint is
17:40 UTC physics-ready;18:00 is complete cost or a concrete diagnosed blocker.
No new framework, extra solver sweeps, global-buffer inflation, favorable-order
sorting, or Gate B development is part of this correction.
