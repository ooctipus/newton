# Current flat-terrain internal seams

Default-off correctness experiment, based on caece007. Original worktrees,
Isaac Lab, collision cadence, margins/gaps and eight solver sweeps are unchanged.
No performance improvement is claimed.

The unchanged loaded-scene helper (qualification.py SHA256
bd6d47727eeee2bb29a9f7e113b9c4282373b4b730e35f9179f38964c7265762)
reproduces the suspected geometry on CPU without a solver: level foot at
x=.015m, bottom at .005m, on the flat21x21 heightfield. Both original and
corrected finite query publish normal(-.7071069,0,.7071067), margin-adjusted
distance .00207107m. At vx2m/s and dt.0025 the speculative normal residual is
-.58578m/s. The continuous terrain has no obstacle or physical edge there.
This is a reproducible nonphysical internal feature, not yet a causal replay
of the previously recorded sliding rollout's impulse burst.

One bounded correction will reject separated oblique query witnesses only
when their closest terrain feature is a provably interior **flat** edge or
flat interior vertex fan. Current decoded elevations must be equal; no
near-coplanar smoothing, static height cache or tolerance-widened terrain.
Finite exterior borders, nonflat terrain/creases, uncertain feature locations
and unsupported or uncertain source features retain original query ownership. The
normal-cone argument applies to each supported convex shape against the prism;
the analytical path retains its original BOX/cuboid-hull admission. Apply the same
qualification to original GJK and finite queries, before any contact write.
Do not clamp impulses, change friction, or return partially written manifolds.

Reuse existing geometry, reduced loaded sliding/rebound and border controls;
add a focused regression in the existing heightfield test module. No new
benchmark/validation framework. Regression must fail on original geometry.
Then one paired physical check and complete fixed-protocol timing, including
the shared corrected MJWarp collision arm if this path is retained for G1.
Any query/row reduction is removal of false terrain features, not capacity
truncation; original sticky buffer checks remain required.

The implementation checkpoint is00:30UTC. This can unblock the previously held
finite-query path, whose measured GB saving was10.28ms but RTX saving only1.06ms
on its older cell-only baseline. Those gains are not transferred to this new
source, and neither they nor a contact-count reduction establish4x versus MJWarp.

The import-time opt-in is `NEWTON_HEIGHTFIELD_WELD_FLAT_SEAMS=1`; set it before
importing Newton, in a fresh process. It does not alter the default path.
Original generic query and analytical query share the same current-feature
predicate, and penetration/uncertain nonfinite normals retain original handling.

## Initial regression and CPU checks

The new complete collision regression failed in all four original configurations
(original/finite, direct/reduced) before the filter. Minimum normal-z was about
.123 instead of the required upward direction on this flat interior surface.
The same regression passes after the correction. Eight further controls pass:
current-height/real-crease/finite-border feature ownership, box/sphere/rotated/
sloped penetration and the existing finite border/interior-surface cases.
Final repeated CPU run: nine tests, no failures/errors/skips. The source review
independently checked all six edge-neighbor mappings and the complete vertex fan.

The existing96 geometry test module now has one reduced loaded CUDA selector,
reusing the original pinned qualification helper and unchanged physical gates.
It records eight original/finite support/tilt/sliding/border cases and preserves
the preceding rebound report. No GPU physical or whole performance result exists
for this correction yet. The previous unreduced fixture-capacity failure has
not been hidden or used to inflate the live buffers.
