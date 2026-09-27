# Stronger contact-driven container shaking

Initial scope: controller changes for the hardware bin, coin tray and dish
rack only. Cube geometry, mechanics and actuation remain unchanged. Later
qualification exposed collision/target-indexing defects and motivated the
cross-backend benchmark integration described below; the original controller
comparison did not include those subsequent changes.

**Visual acceptance:** the first stronger X-stroke captures below passed their
numerical/geometry checks but were rejected as insufficiently vigorous: the
coin tower remained standing and the hardware barely rearranged. Millimetre
excursion counts alone are not the success criterion. The next coin trial uses
an aligned Y stroke, ±100 mm, and 2× arm stiffness (damping ×√2) to reduce table
clearance risk from sag. It retains all masses, friction, finger gains, motor
effort limits and solver budgets. The first-trial statement about unchanged
gains does not apply to this explicitly labelled second coin trial.

## Motion change

The old 100 mm reversal lasted 0.90 s. For the quintic interpolation
`s(u) = 10u³ − 15u⁴ + 6u⁵`, a straight stroke of length `L` and duration `T`
has peak speed `1.875 L/T` and peak acceleration `(10/sqrt(3)) L/T²`.
These are nominal task-space estimates; IK joint interpolation and tracking
make actual container motion different.

| Motion | Span | Time between extrema | Nominal peak speed | Nominal peak acceleration |
| --- | ---: | ---: | ---: | ---: |
| Previous | 100 mm | 0.90 s | 0.208 m/s | 0.713 m/s² |
| New default | 120 mm | 0.40 s | 0.563 m/s | 4.330 m/s² |

The default now makes four complete back-and-forth cycles while gripping.
Approach, closing, release and retraction retain their previous timing.
`--shake-amplitude` controls half-span and `--shake-cycles` controls repetition.
Within the 36% shaking interval of a period `P`, each leg lasts
`T = 0.36 P / (2 cycles + 1)`; the first and last legs cover only half-span.

The controller also writes the analytic joint-velocity target
`qd_target = (q_next − q_previous) 30u²(1−u)²/T`.
Previously zero velocity targets opposed the intended motion through damping.
Container robots use position-plus-velocity actuation so MJWarp can consume
both targets as well. This is a desired velocity, not applied body velocity,
torque feed-forward or object pose control. Settling and `--no-motion` retain
zero velocity targets; cube actuation is unchanged.

## Why acceleration matters

With the current FPGS arithmetic-mean friction mixing, ideal horizontal slip
thresholds are approximately 2.35 m/s² for coin/tray and steel/rack, 3.09 m/s²
for hardware/bin and ceramic/rack, and 3.92 m/s² for coin/coin contact. Coupled
piles, rolling and rack supports complicate these estimates. The old measured
hardware-bin acceleration peaked near 0.56 m/s², well below these scales.

The 7.616 kg loaded bin requires about 39 N at the new nominal acceleration,
including estimated table drag. The ideal two-pad friction ceiling is 60 N,
but the offset handle also creates yaw torque. A CPU inverse-dynamics estimate
including arm motion predicts about 10.8 N m peak in the limiting 12 N m wrist
joint at a 0.40 s leg. A 0.35 s leg predicts saturation (~13.1 N m), so simply
requesting still faster strokes is not guaranteed to move the loaded bin more.

The first stronger hardware trial confirms that limitation: the TCP reaches
124 mm of X travel and 4.30 m/s², but bin acceleration reaches only 1.81 m/s²
(previously 0.56). Bin yaw grows from roughly ±8 to ±20 degrees. Median part
excursion rises only from 1.07 to 1.48 mm, with 14/238 initially contained parts
moving over 5 mm versus 11/235 before. No part moves over 20 mm. Both fingers
retain geometric contact, so this is not a free-escape grasp failure. This trial
passes capacity and geometry checks, but does **not** demonstrate vigorous
internal agitation. It must not be presented as equivalent to the coin/rack win.

A second hardware-only trial aligns the stroke with the Y handle-to-COM line
and inclines the grasp by 30 degrees, aiming to reduce yaw and share the pull
between jaw-normal and friction forces. This remains a physical grasp, with unchanged efforts,
geometry and materials. The first trial is preserved.
Explicit `--shake-axis x --grasp-tilt 0` reproduces the first stronger motion.
The initially proposed 45-degree inclination was rejected before simulation:
nominal hand-housing clearance was only 9.49 mm, and a first-order loaded-PD
estimate predicted table intersection. An IK solution alone does not establish
physical clearance under load.
At 30 degrees, estimated clearance is 26.37 mm nominal / 12.43 mm with loaded
PD deflection. Peak estimated wrist torque is 9.67/12 N m. The requested
~39.7 N handle force exceeds the ideal 34.6 N grip bound **if net vertical
handle force is zero**. A downward component can increase the feasible Y force
while the table supports the load, so this is not an unconditional force cap.
The trial must establish actual container acceleration and content motion, not assume
the requested trajectory is achieved. These are first-order feasibility
estimates, not collision/contact certificates. Second-trial evidence:
`/home/octi/Projects/fpgs-recipes-evidence-20260919/scene-hardware-aligned-20260920-oRlBTs`.

The completed 780-frame aligned hardware capture passes all state/capacity
checks: 11,683 raw contacts, 6,747 reduced contacts, 20,190 MF rows, zero drops
or overflow/reduction violations. Measured Y span is 158.6 mm and peak
acceleration 5.91 m/s²; yaw stays between −0.96 and +3.41 degrees. These are
actual dynamic values, not the quasi-static force-bound prediction above.
Of 243 initially contained parts, 81 move over 5 mm and 16 over 20 mm; median
excursion is 3.75 mm. Median relative rotation is 18.8 degrees and 66% rotate
more than 15 degrees. This is more activity, **not wholesale pile rearrangement**.
Sampled hand/finger distance from the bin shell stays above 18.56 mm, table
clearance above 5.59 mm, and worst bin-floor penetration is 0.238 mm. Checking
all finger collision boxes confirms intermittent separation: nearest sampled
right-finger/handle gap reaches 9.20 mm. Thus this is contact-driven agitation
with intermittent grasp slip, **not a continuously maintained bilateral pinch**.
The increased activity is not caused by housing/bin or housing/table bumps.
The updated 1× video uses an opposite-side detail camera so the hand does not
hide the contents.

A CPU IK check compares peak commanded joint speeds with the limits authored
in the FR3 URDF: maximum utilization is 72.45% hardware (first X/zero-tilt trial),
71.40% coins and 71.63% dishes, all on joint 4. Fingers peak at 0.0539 m/s versus
the authored 0.2 m/s.
For the aligned 30-degree hardware path, peak commanded utilization drops to
33.93% (joint 4); see `hardware_cpu_feasibility.json` in the second-trial directory.
The imported model currently contains placeholder limits of `1e6`, so those
model values would not provide a meaningful check. This is a command check,
not a claim of runtime speed-limit enforcement or measured joint velocities.
See `joint_velocity_check.json` in the evidence directory.

## Qualification

Current captures are single-world, seed 42, 10 substeps/frame, 24 iterations,
60 Hz simulation frames. Qualification must measure container acceleration,
object motion relative to that container, grasp retention, mesh/floor integrity
and capacity counters. Genuine rim spills are allowed; dropped rows and floor
fall-through are not. Static parts already outside the bin must not be counted
as activated contents merely because the bin moves nearby.

The earlier, gentle-motion results in
`FRANKA_CLUTTER_SCENES_20260920.md` do not qualify this stronger controller.
New captures, comparisons and videos are stored under:
`/home/octi/Projects/fpgs-recipes-evidence-20260919/scene-vigorous-shake-20260920-sNYaR6`.

Completed measurements (old gentle motion → new controller):

| Scene | Measured peak X acceleration | Median object excursion in container frame | Objects moving >5 mm |
| --- | ---: | ---: | ---: |
| Coins, RTX, 256 objects | 0.74 → 2.32 m/s² | 2.61 → 7.56 mm | 2 → 200 |
| Dishes, RTX, 40 objects | 1.01 → 2.83 m/s² | 0.18 → 14.57 mm | 3 → 36 |

Coin/dish measurements use the same 30 Hz pose sampling and five-point least-squares
derivatives in both captures, over the 5.0–8.6 s shaking interval. Twenty-three
coins now move more than 20 mm relative to the tray (previously none). All 256
stay inside. Actual X span is 88.3 mm, **not** the requested 120 mm; the robot,
grasp and container dynamics attenuate the target motion. Rotation-induced
acceleration of individual contact points is not included in the origin's
X-acceleration statistic.

The coin capture passes all frame checks: peaks 2,176 contacts / 6,474 MF rows,
zero drops or capacity flags. Sampled coin/floor penetration is at most 0.24 mm;
TCP-to-handle offset variation is 2.29 mm RMS, 4.68 mm maximum about its mean.
These sampled geometry/offset checks are not per-substep contact residuals or
a direct grip-force measurement. Detailed data: `coins_motion_comparison.json`.

The dish capture also passes all frame checks: peaks 1,846 contacts / 5,466 MF
rows, zero drops or capacity flags. Eight objects move over 20 mm relative to
the rack (previously one). Cutlery spreads; one plate leaves its original slot
and leans up to 31.6 degrees. Keeping every plate upright is not a requirement
for this stronger agitation. The sampled object-surface minimum world height
is −0.016 mm and rack minimum is −0.008 mm, with no observed tabletop
fall-through. Grip-offset variation is 1.87 mm RMS; measured span is 110.4 mm.
See `dishes_motion_comparison.json`. Both videos were rendered from saved
simulation poses, visually inspected and fully decoded successfully.

### Why the first coin shake was visually weak

The measured TCP and handle X acceleration reached 4.31 and 4.17 m/s², while
the tray origin reached only 2.32 m/s². Tray yaw spanned 30.72 degrees. At peak
TCP acceleration, 3.15 m/s² came from rotating the offset handle and only
1.01 m/s² from translating the tray origin; relative TCP/handle motion accounts
for another 0.15 m/s². Increasing gains alone would not fix this force-line
problem. See `coins_rotation_decomposition.json` beside the first captures.
The second coin trial pulls toward/away from the centre, with a nominal 20 cm
full stroke in 0.40 s. It **does** break up the tower: by 6 s, the number of
tower-coin centres above 30 mm drops from 72 to 24; their 90th-percentile height
above the tray floor drops from 43.0 to 30.8 mm. However, the run stops at frame
398 when coin 167 genuinely falls off the workbench. It first crosses the tray
rim, slides on the tabletop, crosses the table edge at Y=0.40 m, then falls.
This is not observed floor tunnelling; all capacity/drop counters remain clear.
The partial capture stays labelled **failed containment**, not a completed pass.
Evidence directory:
`/home/octi/Projects/fpgs-recipes-evidence-20260919/scene-coins-aligned-20260920-Qn0w0B`.

A 16 cm full-stroke follow-up retains the same 0.40 s leg and doubled arm
stiffness. Nominal acceleration is still 5.77 m/s², above the ~3.92 m/s²
coin/coin slip estimate, but nominal speed falls from 0.94 to 0.75 m/s. An ideal
tabletop-friction stopping-distance estimate then leaves more room before the
table edge. This is a bounded containment adjustment, not a relaxation of the
required visible tower breakup. No floor, larger table or weaker escape check
has been introduced.

The user subsequently requested a taller tower. The 256-coin/16-cm follow-up
was stopped before shaking and is not a qualification. The new default is
520 coins: the same 64-coin bed plus 38 complete 12-coin layers, versus 16
layers before. Coin dimensions, tray footprint and individual masses are
unchanged; the ideal settled height is about 109 mm and total tray load rises
to 6.507 kg. The taller trial uses the same 16 cm Y stroke and 2× arm stiffness.
Its CPU clearance/rate/torque checks pass, but predicted handle demand (~43.5 N)
exceeds the simple one-finger normal-force bound; actual motion and tower
breakup must be measured. Evidence:
`/home/octi/Projects/fpgs-recipes-evidence-20260919/scene-coins-tall520-20260920-zrrrrX`.

**Tall-stack result: NOT QUALIFIED.** The 520-coin tower visibly collapses,
but coin 29 crosses the tray floor at 4.60–4.633 s, before shaking starts.
It stays inside the tray footprint and settles on the workbench underneath
the 5 mm floor. This is distinct from coin 240's later legitimate table-edge
spill, which stops the run at frame 490. All capacity/drop counters remain
clear. The partial video explicitly labels both failures; no successful
qualification or performance number is claimed.

Collision-only replay finds upward support contacts before the crossing,
with penetration growing from 0.60 to 1.63 mm. Coin/tray uses triangle-based
mesh–cylinder collision, not the tray's cooked SDF. Center-based backface
rejection then removes every top-floor triangle when the coin center falls
below local z=5 mm. Synthetic interpolants confirm 8 candidates at 5.1915 mm
and zero at 4.9925 mm. Raw/reduced contacts agree, excluding reduction or
capacity as the cause. A general closed-mesh robustness fix is being tested;
the earlier penetration growth still needs convergence/compliance analysis.
See `coin29_collision_replay.json` and `coin29_floor_crossing_diagnosis.json`
in the tall-stack evidence folder. No room floor, enlarged table or relaxed
escape check has been added to conceal the real penetration.

An independent CPU check reproduced a pre-existing FPGS coordinate-target
indexing issue in replicated worlds containing free joints. A temporary guard
rejected FPGS world counts above one; single-world Franka-first captures were
unaffected. The subsequent Newton fix reads position targets through the
model's coordinate/DOF-aware target starts, while velocities remain DOF-indexed.
The evidence directory's `indexing_issue.md` preserves the original reproduction.

The guard was removed after two-world FPGS coins and hardware smokes, each with
eight loose objects per world, 48 frames / 480 substeps and CUDA graph replay.
Distinct first-joint targets produced opposite approximately 0.04149 rad
displacements. Every frame passed finite-state and capacity checks; no
cross-world contacts occurred. Physical coordinates coincide across replicated
worlds, isolated by world IDs; only visualization is spatially offset. Reports
`multiworld_coins_fpgs.json` and `multiworld_hardware_fpgs.json` and the diagnostic
script are in the evidence directory. These are small correctness checks, not
full-count agitation, scaling or performance qualification.
The matching N=2 MJWarp coins smoke also passed, with approximately
0.04146 rad independent joint displacements. Its installed MuJoCo/MJWarp
3.12.0 versions differ from Newton's declared `~=3.11.0` requirements;
that dependency warning prevents treating this smoke as a pinned performance
baseline. The report is `multiworld_coins_mjwarp.json`.

The 33 geometry, wrench and motion-control tests passed before the taller-stack
change, including separate
coordinate/DOF array strides, analytic target velocities, stationary intervals,
cycle schedules, axis selection, inclined-grasp IK, unchanged cube actuator
modes and early rejection of the then-unsupported multiworld case. That guard
test now checks that replicated construction is accepted; solver regressions
cover independent targets after free/ball joints in both target layouts.
Ruff checks and formatting pass. These tests do
not replace the contact-driven rollout checks.
The subsequent arm-stiffness regression passes, and the six tray/layout tests
pass with the 520-coin default and non-overlap checks for all coin pairs.

```bash
python -m newton.examples robot_clutter --scene hardware --device cuda:1 --num-frames 780 --viewer null --test --report hardware.json
python -m newton.examples robot_clutter --scene coins --device cuda:0 --num-frames 720 --viewer null --test --report coins.json
python -m newton.examples robot_clutter --scene dishes --device cuda:0 --num-frames 720 --viewer null --test --report dishes.json
```

## Cross-backend integration and large-world diagnosis

The isolated Lab worktree `../franka-clutter-benchmarks-20260920` adds normal
scripted DirectRLEnv tasks for hardware, coins and dishes. They load shared CAD
bundles, clone N environments, use the same saved FR3 position/velocity targets,
and expose sibling FPGS, MJWarp and native PhysX presets at ten 1/600 s steps per
action. The exporter preserves source masses, COMs and inertias explicitly;
native SDF fields are independently cooked, not assumed identical. Rubik is
excluded pending mechanical qualification.

The ordinary runtime benchmark is reused, not replaced by a new timer. A thin
launcher runs the two GPUs simultaneously and competing backends sequentially
on each card. All twelve N=2 and N=8 / eight-coin integration jobs passed.
The N=8 jobs also exported clean capacity/finite-state diagnostics and correct
physical GPU UUIDs. These 20-step smokes do **not** establish large-world motion
quality or speedups. Separate
Lab fixes admit the previously rejected FPGS benchmark label and attribute GPU
identity/resource samples by physical UUID instead of a visibility-remapped
ordinal. Optional diagnostics report actual capacities and sticky failures;
per-frame validation is kept outside performance runs.

Evidence root:
`/home/octi/Projects/fpgs-recipes-evidence-20260919/benchmark`.
The parent directory's `README.md` is the user-facing index. Historical
experiment folders are now under `archive/`, and the four retained videos
are in `videos/`; old capture manifests retain their original paths.
See the Lab guide `docs/source/developer-tools/benchmarking/franka_clutter.rst`
for the export, cloning, profiling, qualification and paired-run commands.
Native PhysX wall timing and CPU/NVTX profiling work; complete native CUDA
tracing remains unresolved on this OVStage stack, even with a driver-compatible
Nsight release. Do not interpret a startup-only GPU trace as a solver profile.
Actual Lab CUDA/NVTX traces were verified for FPGS and MJWarp, including their
solver kernels and the concrete task's trajectory, observation and reset hooks:
`lab_fpgs_profile_verified.nsys-rep` and `lab_mjwarp_profile_verified.nsys-rep`.
These profiled short smokes are diagnostics, not throughput results.

The closed-mesh collision fix now preserves convex support contacts when the
center passes behind a closed surface but the shape still straddles its nearest
face. Open-sheet semantics and fully embedded cases retain their prior rules.
Six new failures reproduced before the fix; 57 collision/geometry tests passed
afterward. Frozen loaded-coin replay measures about 30 microseconds additional
collision time per step (+8.8%), not a full-solver improvement. A short restart
does not yet qualify the full 520-coin rollout.

The loaded 520-coin solver profile exposes a structural scaling limit. At 96
iterations on GB300, one frame (ten substeps) takes about 7.85 s CUDA physics
time. One MF-GS kernel accounts for 99.83% of summed GPU kernel time, launched
as **one warp for the entire world**. Free-body rows already use sparse 6+6 DOF
responses; it serially processes roughly 14,000 rows, while the small dense
articulation portion retains world-wide state. The profile does not separately
measure these two costs or establish register spilling. A matched one-frame
MJWarp24 diagnostic takes 69.70 s and has worse floor penetration; different
budgets/residuals and the dependency-version warning prohibit a qualified
speedup claim. Full details and raw profiles are in
`archive/scene-coins-tall520-20260920-zrrrrX/coins_solver_hotspot.md` under the evidence
parent. The next structural target is within-world constraint scheduling and
compact articulation response, not another sparse free-body representation.

The full hardware recipe exceeds RTX shared memory if all MF impulses remain
resident. A construction-time storage selection now streams the existing
impulse array only when resident storage cannot fit, without clipping rows or
changing projection order. One lane owns global impulse accesses and broadcasts
values to the warp. Five-world tests match resident velocities and impulses
exactly across four projectors, two metadata modes, six row phases and graph
replay. Nineteen CPU checks, including guarded kinetic-source compatibility,
pass. The actual RTX hardware256 Lab task also passes one environment step
with all 37,248 MF row slots retained: compiled shared memory is 7,356 bytes
versus the 101,376-byte device limit. Only 45 MF rows are active at that initial
instant, so this is **capacity admission**, not loaded-pile qualification or a
performance improvement. The 520-coin resident path remains unchanged.

A second large-capacity audit found signed 16-bit decoding of MF friction
parents: row 32,768 became negative even though storage admitted it. General
GS, propagation-full, local-pair and runtime-sized admission now decode the
upper bits as unsigned, reserving 65,535 for the no-parent sentinel. Capacity
above 65,535 is explicitly rejected, never clipped. Real active triples at
parents 32,767 / 32,768 / 37,247 / 65,532 match rebased low-index controls
exactly for current and bisection projectors, with nonzero friction impulses.
The focused 24 CPU checks and GPU regression pass; smaller-capacity GS CUDA
remains unchanged. See `fpgs_mf_parent_encoding.md` in the Lab evidence root.

The corrected-phase native PhysX loaded-stack restart still shows severe
penetration despite matching native poses, velocities and robot FK. A runtime
query confirms the cooked tray SDF is hollow and correctly signed at sampled
points. One-coin and eight-coin fixed-tray controls produce upward contacts and
settle without penetration. Thus a missing cavity or universally broken
cylinder contact is not supported by the evidence; the compressed, tilted
FPGS snapshot/free-tray cold restart needs separate analysis from a fresh native
settle. The subsequent fresh 720-step native run also loses all 64 bed coins
through the floor during settling, before robot contact. Cold FPGS history is
therefore not the sole cause. Dynamic-tray/primitive-pair isolation remains
necessary; this run is explicitly unqualified. See
`NATIVE_COINS_RESTART_DIAGNOSIS_V2.md` and `coins520_fresh_ovphysx_v1.json` in the
Lab evidence root.

Minimal native controls subsequently isolate the pair: an analytic cylinder
falls through the **dynamic** SDF tray even with one coin, while an explicit
convex coin mesh stays supported. Static-tray controls pass. Both documented
cylinder-approximation settings, including application before OVStage population,
failed to alter this result in the installed SDK. No hidden runtime workaround
has therefore been promoted.

The Lab exporter now authors the same 32-sided / 64-vertex convex coin for
every backend, preserving original mass, COM, inertia, dimensions and material.
The authored radial chord error is 60.19 micrometres. Both hull limits are
explicitly 64 vertices, within documented native GPU hull limits; the earlier
64-sided control was a geometry diagnostic, not proof of GPU-native cooking.
Seven export/round-trip tests pass, including watertight outward geometry and
retention of its support vertices through Newton import. Immutable bundles
`coins8-v5-convex` and `coins520-v5-convex` retain `qualified=false`. They are
**not** interchangeable with earlier analytic-cylinder
captures. The standalone Newton example remains an analytic-cylinder scene.

A fresh native PhysX 520-coin run with the shared convex bundle completes all
720 steps and one scheduled reset, observing the final state before reset.
No coin centers cross the tray floor or leave its footprint. Actual tray
travel is 16.79 cm; settled and final floor penetration are 0.0106 and 0.0150 mm.
One tumbling top-layer coin reaches 1.297 mm transient penetration at 7.55 s;
neighboring samples are near 0.015 mm. This is an impact-resolution issue to
qualify, not the analytic-cylinder bed's wholesale pass-through. The tower
still largely stands: 120 centers above 80 mm after settling become 118 at
the end. Thus dramatic rearrangement remains unmet, and this diagnostic is
not a throughput or convergence result. See
`coins520_fresh_convex32_ovphysx_v1.json` and its saved poses in the evidence root.
The corrected eight-coin bundle also passes N=2 integration on all three
backends on GB300 (`convex-coins-n2-integration`). These are short integration
tests, not qualification of the 520-object case. The combined Lab task and
benchmark regression suite now passes 116 tests, including matched trajectory
knots/timing for four- and six-cycle export recipes.

Saved-pose analysis shows that the four-cycle tower was not immobile:
518/520 coins finish more than 10 mm from their settled tray-relative locations,
but most of the upper stack slides coherently. A bounded six-cycle candidate
(`coins520-v6-six-cycles`) keeps geometry, the 16 cm requested stroke, materials,
drive gains, effort limits and solver budgets unchanged. It only fits six
reversing cycles into the same grasp interval. The fresh native 720-step run
now breaks up the tower: settled/final height is 109.12/22.60 mm and the final
count above 80 mm is zero. Actual tray span is 16.74 cm; 100 ms-smoothed peak
horizontal acceleration rises from 8.11 to 12.46 m/s². Sampled TCP/handle-center
separation stays within 4.64 mm during agitation; this is a tracking measure,
not a direct uninterrupted contact-force measurement.

No interior coin center crosses the tray floor; maximum sampled interior
floor penetration is 0.669 mm. Of the 187 coins leaving the tray, 122 fall
below the workbench. Every first crossing of the checked table-height planes
is outside its authored XY footprint, and no saved center lies inside the
table slab. This supports genuine edge spills, not observed table pass-through.
There is no room floor, so these coins continue falling until the scheduled
reset; that scene limitation remains explicit. All 720 states are finite and
native logs contain no overflow/cooking error. This establishes stronger
physical agitation on native PhysX, **not** cross-backend convergence or
accepted timing. The six-cycle bundle remains unqualified. Full results and
sampling limitations are in `NATIVE_COINS_RESTART_DIAGNOSIS_V2.md` and
`coins520_fresh_convex32_sixcycles_ovphysx_v1.json`.

Native PhysX's generic GPU buffers remain uncalibrated. The full dish-rack
MJWarp smoke also exposed an insufficient shared Newton contact capacity:
up to 4,567 requested contacts versus 4,096 allocated across two worlds. The
launcher rejected the result instead of admitting dropped-contact timing.
The provisional dish recipe now allocates 2,688 contacts/world, shared by both
Newton solvers, and supports an explicit per-world qualification override.
The repeated N=2 MJWarp smoke passes with zero contact/capacity/iteration flags;
the matching native PhysX smoke passes its available checks and log review.
This modest headroom over the short observed demand is not full-cycle sizing.
These are explicit remaining
limits to a three-way qualified comparison. No bundle has been promoted to
qualified and FPGS being fastest is not claimed.

The final source audit distinguishes repeated dense-row work from MF work:
only dense rows traverse all 3,135 world DOFs; MF rows already touch their
sparse endpoints. A robot-only nine-DOF replacement would be incorrect for
robot–container contacts, which also need the container's six coordinates.
Actual per-row articulation/support spans (nine plus six here) could remove
dense zero-work while preserving serial GS order; dependency-safe parallel
MF scheduling remains the larger candidate. Neither improvement has been
implemented or timed in this benchmark-integration work.
