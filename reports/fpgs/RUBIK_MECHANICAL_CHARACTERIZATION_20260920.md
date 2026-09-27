# Rubik mechanical characterization — classic non-magnetic 57 mm reference

Status: **not calibrated to a real cube**. Isolate the cube and characterize its
response to applied forces before qualifying the Franka scene. A rendered cube,
an FK permutation test, or a successful robot turn is not mechanical validation.

## What the current model represents

The geometry follows the [pinned MuJoCo cube](https://github.com/google-deepmind/mujoco/blob/5ceb72b1bbe87612a2859304d2c22a15cd3c180d/model/cube/cube_3x3x3.xml):
26 chamfered rigid cubies, six center hinges, and twenty core-centered ball
joints. The 56.92 mm shell has 80 micrometre seams and 1.42 mm chamfers. These
are **model choices**, not measured manufacturing clearances of a real cube.
The shared 44-triangle mesh is cooked to a 0.25 mm SDF; voxel pitch and physical
clearance are distinct quantities and must be checked independently.

The [OpenAI cube-model paper, section 4.2](https://arxiv.org/pdf/1910.07113)
describes interlocking tabs, preload-induced friction, and limited deformation
in real cubes. Its simulation relied on soft contacts to approximate some
compliance and used only rough calibration before domain randomization.
Therefore copying its geometry does not establish matched physical response
under a different solver.

Current limitations:

- Internal contact friction is zero. FPGS does not consume the MuJoCo reference's
  joint friction-loss parameter. Viscous damping is not static friction: its
  resisting torque vanishes at zero velocity.
- Core-centered joints prohibit radial travel. A zero radial displacement under
  load would reflect that constraint, not measured infinite material stiffness.
- Contact deformation can arise from geometric clearance, compliance, or solver
  error. Increasing penetration tolerance is not a calibrated spring model.
- No magnetic alignment should be added for the chosen non-magnetic reference.

## Inertia audit

For the top layer, with nine cubies of mass `m = 0.00253704 kg`, pitch
`p = 0.019 m`, and mesh-derived COM inertia:

`I_layer,z = sum(I_cubie,z + m (x² + y²)) = 1.2304e-5 kg m²`.

The reference adds `1e-4 kg m²` to every joint rotational axis. A coherent top
turn therefore adds `9e-4 kg m²`, making effective inertia **74.15 times** the
geometry-derived value. This is an added numerical/model inertia, not mass
hidden inside the plastic. The layer's corresponding viscous damping is
`9 × 5e-4 = 0.0045 N m s/rad`: it resists a 1 rad/s turn with 0.0045 N m but
provides no static breakaway threshold.

Keep the reference-stabilized baseline identifiable. Compare it with zero added
armature and with physical inertia plus separately identified friction/damping;
do not silently relabel the stabilized version as physically calibrated.

## Applied-load experiments

Clamp only the core fixture. Do not weld a turning layer, prescribe cubie
orientations, or use the robot during identification. Apply a torque to the
top-center cubie, then repeat with a two-point force couple on opposite edge
cubies. The other eight top-layer pieces must follow through contact.

For opposite tangential forces at radius `r = 28.46 mm`, `tau = 2 r F`:

| Applied torque | Force at each edge |
| ---: | ---: |
| 0.001 N m | 0.0176 N |
| 0.005 N m | 0.0878 N |
| 0.010 N m | 0.1757 N |
| 0.020 N m | 0.3514 N |
| 0.050 N m | 0.8784 N |

These are exploratory load levels, **not measured classic-cube torque targets**.
Report actual moment arms and net applied wrench if the pieces lose alignment.

The mechanical quantities to identify are different:

- **Turning:** a simple bookkeeping model is
  `tau_drive = I alpha + b omega + tau_dry + tau_geometry(theta)`.
  Sliding resistance has scale `tau_dry ~ mu N_preload r_effective`; cubie mass
  alone cannot predict it. Removing the numerical armature fixes the inertia
  term, not the missing preload/friction terms.
- **Clearance:** the 80 micrometre seam controls local geometric free play. A
  small-angle displacement has scale `r theta`, so at a 19 mm lever this seam
  corresponds to about 0.24 degrees. This is a local scale estimate, not a
  measured face backlash or allowable corner-cutting angle.
- **Deformation:** distinguish motion through existing free play from elastic
  travel after engagement. In an approximately linear engaged range,
  `delta F = k_effective delta x`; loading and unloading reveal hysteresis.
  Rigid ball retainers remove radial travel from the model and cannot identify
  that stiffness, regardless of how plausible the rendered displacement looks.

| Experiment | Measurement | What it separates |
| --- | --- | --- |
| Slow torque ramp and hold, both directions | Angle, speed, onset and creep | Breakaway versus viscous resistance or finite observation time |
| Several low-speed turns | Torque versus angle and angular speed | Rate-independent friction versus damping; geometric jams |
| Force couple versus center torque | Top-layer rigid-fit error, lower-layer drift | Genuine contact transmission versus artificial layer driving |
| Off-axis 1, 2, 5 N load/unload | Displacement, tilt, residual deformation | Free play, compliance and irreversible rearrangement |
| Misaligned adjacent layer | Torque versus initial misalignment | Forgiveness and jamming, not just aligned quarter turns |

Track all 26 pieces using transformed COMs; body origins coincide with the core.
Use lower-layer drift and top-layer shape error alongside the fitted turn angle:
a best-fit angle alone can hide a severely distorted layer.

## Numerical checks before physical fitting

Repeat selected loads with half timestep and increased iterations. Refine SDF
pitch independently. Report contact and constraint overflow/drop counters.
If the force–displacement curve changes materially with those numerical choices,
it is not yet a stable property of the mechanical model. Keep loading history
and observation duration fixed; a slowly creeping frictionless system must not
be mistaken for static friction.

## First isolated results

The probe removes added armature, viscous damping and internal contact friction.
Masses remain the reference model's values, **not measurements of the selected
commercial cube**. Each trial settles for 0.5 s, ramps for 0.25 s, holds for 2 s,
unloads over 0.25 s, then relaxes for 1 s. A two-point couple applies a nominal
0.001 N m; actual torque is measured from the moving application points.

| Gravity-on couple | Timestep / iterations | Mean fitted angle during hold¹ | Top-layer fit RMS¹ | Final fitted angle |
| --- | --- | ---: | ---: | ---: |
| Initial diagnostic | 1/600 s / 24 | 6.189° | 0.266 mm | 6.221° |
| Combined refinement | 1/1200 s / 48 | 6.207° | 0.268 mm | 6.223° |

¹ Eighteen saved samples with `1 < time_s < 2.7`; final values at 4 s.
Both captures have zero recorded row drops or capacity flags. This combined
refinement does not independently establish timestep or iteration convergence,
and it does not test SDF discretization error.

The refined hold has about **−0.00104 N m of cross-layer normal-contact torque**
opposing the applied couple, with a 0.203 N sum of cross-layer force magnitudes.
Thus the roughly 6° stall survives this numerical refinement; it is not evidence
of realistic dry friction, which is absent from this model. The larger loads
in the initial, unguarded sweep produced fast or incoherent motion and are not
accepted quasistatic measurements.

For an ideal coherent turn about vertical, every cubie's height is constant:
`U_g(theta) = sum(m g z)` and `dU_g/dtheta = 0`. Gravity-dependent resistance
therefore requires an off-path configuration (such as sag and wedging), a
contact-representation effect, or numerical response; it is not an intrinsic
friction law for the intended face turn. Zero-gravity and center-torque controls
separate these possibilities without changing the geometry to force a pass.

The force couple and center torque have the same net turning moment, but not
the same load on each piece. With the force points 19 mm above the core, each
edge cubie receives an off-axis moment `z F = z tau / (2 r)`: **0.334 mN m per
piece for a 1 mN m couple**, with opposite signs on the two pieces. For scale,
one modeled cubie's gravity moment at a 19 mm lever is 0.473 mN m. Cancellation
of the net off-axis moment does not cancel those separate ball-joint loads.
The center hinge cannot directly rotate about these off-axis directions.
This makes load-path sensitivity an important retainer/contact-model check,
not merely a test of whether the solver can rotate nine cubies together.

Follow-up controls, with the same zero-armature SDF model:

| Control | Actual event | Interpretation |
| --- | --- | --- |
| Zero gravity, edge couple | At 0.6775 s, the 1 mm coherence guard trips at 13.40° center yaw and 0.000667 N m. The requested 0.001 N m is never reached. | The gravity-on plateau disappears, but this is not a clean quarter turn. |
| Gravity on, center torque | Full 0.001 N m is reached; the speed guard trips at 0.7683 s and 15.39° center yaw. Subsequent coasting crosses 90° with 0.20–0.23 mm top-layer RMS and 0.25–0.27 mm lower-layer drift. | The model transmits a center-driven turn through contact. It is not a controlled 90° stop or a real torque calibration. |
| Gravity on, off-axis corner force | A nominal 1 N ramp is cut at **0.133 N**, after 1.03 mm maximum coherence error. The loaded corner moves approximately (+1.16, −0.68, −0.53) mm from its ideal position. | This is rearrangement under load, not a completed 1 N stiffness measurement. |

The off-axis trial lasts 1.5 s, with a shorter hold and relaxation than the
4 s torque trials; its guard trips during the initial ramp. At 1.5 s the loaded
corner retains about 2.63 mm displacement while its distance from the core
remains fixed to numerical precision. The measured displacement is therefore
largely angular rearrangement, not modeled radial spring compression.
Its unloaded, settled corner was already displaced (+0.19, +0.12, −0.31) mm;
the quoted trip displacement is relative to ideal geometry, not that settled
baseline. No follow-up control reports row drops or capacity flags.

Later probes cut the applied load on excessive speed, loss of piece coherence,
or the first quarter turn. They preserve the actual trip load and time; motion
after cutoff is coasting, not continued response to the nominal requested load.
These diagnostic guards are not measured material limits. Earlier captures
retain their original, less comprehensive guard configuration.

Raw captures are saved in
`/home/octi/Projects/fpgs-recipes-evidence-20260919/scene-cad-upgrade-20260920-46PIqu`:
`cube-isolated-physical-unbounded-dt600-it24.json` and
`cube-isolated-physical-couple-dt1200-it48.json`.
Follow-up files are `cube-isolated-zero-gravity-trip-detail.json`,
`cube-isolated-center-torque-dt1200-it48.json`, and
`cube-isolated-off-axis-1n-dt1200-it48.json`. The zero-gravity trip-detail run is
a short one-world RTX trial, not the earlier two-world GB300 capture; compare
configuration metadata rather than assuming their trip times are identical.

## Collision-representation audit

A static check compares the cooked field with independent double-precision
point-to-triangle distances to the source mesh. It samples 446 surface sites
and 45,296 offset points, including faces, bevels, edges and vertices; these
are sampled errors, not global bounds or comparison with a real cube.

The requested 250 micrometre grid has an actual pitch of 237.7 micrometres.
With the collision texture sampler, sampled edge zero surfaces retreat inward
by about **24–32 micrometres**, and vertex zero surfaces by **67–68 micrometres**.
Those displacements are comparable to the intended 80 micrometre seam. Flat
and bevel interiors are much more accurate under software trilinear sampling;
the hardware sampler adds up to 10.9 micrometres of sampled flat-face distance
error. Voxel pitch itself is therefore not a useful substitute for an error
measurement.

This can alter tight edge coupling, but does not establish the sole cause of
the force-couple stall. The source collision polygons are not themselves moved
with the SDF's effective surface. A convex-hull control can retain this already
convex 24-vertex geometry without grid discretization; contact generation and
solver tolerances still require checking. Concave hardware and dishes do not
have that simple convex representation.

The matched convex-couple control does **not** fix the mechanics: during the
ramp, the 1 mm coherence guard trips at 0.5458 s under only 0.000134 N m
(0.00234 N per point), with center yaw 0.681 degrees. It does not trip during
the unloaded settling interval, although lower-layer drift is already 0.482 mm
at 0.5 s. It records zero drops/capacity flags, with peaks of 348 contacts and
1,044 dense rows. The requested 0.001 N m is never reached.

The control removes grid discretization **and changes contact generation and
the manifold**, so it does not independently isolate SDF resolution error.
It rules out adopting this direct-convex configuration as an immediate fix;
it does not show that SDF error is irrelevant. The optional `--collision convex`
probe remains available; scene defaults have not changed. Raw capture:
`cube-isolated-convex-couple-dt1200-it48.json`.

Reproduction script and measurements: `audit_cube_sdf.py` and
`cube_sdf_accuracy_audit.json` in the evidence directory above.

## Real-world comparison still needed

We have mechanism references but no verified torque–angle and off-axis
force–displacement dataset for the selected classic cube. A university project
[design estimate](https://courses.grainger.illinois.edu/ece445/getfile.asp?id=20853)
mentions approximate turning torques, but does not supply measured curves or a
specified classic-cube sample; it is not a calibration target.

Useful measurements are cube mass, slow CW/CCW torque–angle traces, breakaway
and steady torque at stated speeds, and force–displacement loading/unloading
with the fixture and load application points specified. Record cube condition,
lubrication and tension. Fit parameters to part of that data and validate on
held-out speeds, directions and loads; do not invent a universal real-cube band.

## Decision before robot integration

Do not accept the current surrogate as a mechanically matched classic cube.
It can transmit a face rotation, but the force-couple and off-axis controls
expose load-sensitive rearrangement. Neither larger numerical inertia nor a
single collision-backend switch establishes the missing mechanics.

The next structural model work is preloaded, compliant retention and an
explicit dry-friction response, with contact accuracy checked at the intended
clearance scale. Retest both center torque and edge forces with a defined
fixture before tuning the robot grasp. Until real reference measurements are
available, any selected preload, friction or stiffness must remain labeled
assumptions rather than validated classic-cube parameters.
