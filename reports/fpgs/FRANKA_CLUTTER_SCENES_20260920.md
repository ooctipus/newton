# Franka clutter — CAD upgrade and archived smoke results, 2026-09-20

Local branch: `ooctipus/franka-clutter-scenes-20260920`, based on
`792a0ec66d4b2307d98c84e142ee556dbd07d3b8`. Changes are examples, geometry
tests, screenshots, and documentation; no Isaac Lab or solver-core edits.

The CAD rollout numbers below use the earlier **gentle controller**. The later
four-cycle, velocity-targeted shaking change has separate
[measurements and qualification](FRANKA_CLUTTER_VIGOROUS_20260920.md); these
captures must not be presented as evidence for its stronger motion.

## Current CAD/SDF upgrade

The geometry and control have changed materially since the archived table
below. Its successful runs **do not qualify the current scenes**.

- Hardware: 256 independent steel parts; pinned Factory threaded M12/M16 nuts
  and bolts, original involute gears and bevelled washers. Watertight meshes
  are simplified before SDF construction. Source-to-simplified bidirectional
  sampled errors were 6–39 micrometres (4,000 samples, measured in millimetres
  to avoid small-coordinate numerical error), not a guaranteed Hausdorff bound.
  Solid volume is about 0.919 L in a 1.798 L bin interior. Seeded, disjoint
  layers start below 159 mm and settle into the 55 mm-high bin.
- Coins: the 256 coins now sit in a free tray with a physical handle; the loaded
  assembly is approximately 3.68 kg.
- Dishes: 40 free objects in one free rack assembly. Continuous ceramic shells,
  four-tine forks and genuinely concave spoons replace primitive approximations.
- Cube: 57 mm contact-coupled model with 80 micrometre seams, 1.42 mm chamfers,
  54 visual-only facelets and a 44-triangle shared collision mesh. Six center
  hinges and twenty core-centered ball joints follow the pinned MuJoCo model.
  Internal contacts are frictionless; finger friction supplies the grasp.
  Geometry tests permit U then R turns and reject isolated cubie twisting.

All containers are free bodies moved only by the Franka's contact with their
attached handles. The controller approaches, closes, slides back/forth,
releases and retracts; it does not grasp individual utensils. CUDA SDF cooking
happens once per unique low-polygon mesh before world replication, with cached
cooks and feature-specific voxel sizes. Winding-number signs preserve the
union of intersecting rack wires. Coins, dishes and hardware have completed
loaded-container rollouts; cube mechanics remain unqualified.
Initial tests exposed insufficient old contact/triangle capacities, which fail
explicitly rather than silently dropping rows.

Current measured checks (not performance benchmarks):

| Current CAD scene | Completed check | Result |
| --- | --- | --- |
| Coins, 256, current storage | RTX, 720 frames, 10 substeps, 24 iterations | Actual tray X travel 93.23 mm; 2,187 peak contacts, 6,507 MF rows, zero dropped rows/capacity flags. All coins stay inside the tray; sampled coin/floor penetration at most 0.193 mm and tray/table penetration 0.012 mm. |
| Dishes, 40, first CAD rack | RTX, 720 frames | Actual rack X travel 99.2 mm; 1,594 contacts, 4,728 MF rows, zero dropped rows/capacity flags. Plate guides were subsequently corrected, so this is not the final-rack qualification. |
| Dishes, corrected guides | RTX, 720 frames | Actual X travel 99.0 mm; all 12 plates/bowls remain within 8.91 degrees of upright after settling. Peak contacts 1,586, MF rows 4,704, dense rows 55; no drops/capacity flags. Mesh-height minimum −0.089 mm. |
| Hardware, 256 | GB300, 780 frames, 3 s settling + 10 s motion | Actual bidirectional bin X travel 69.10 mm. Peaks: 11,980 raw / 6,958 reduced contacts, 20,823 MF rows, 64 dense rows. No drops, capacity flags or reducer-invariant failures. Worst sampled interior-floor penetration is 0.120 mm. |
| Cube, reference-stabilized | GB300, 720-frame initial robot script | No capacity failures with 1,024 contact / 3,072 dense-row storage (peaks 663 / 1,966), but the turn failed. Robot sag caused a multi-layer grasp; this is not accepted mechanics validation. |

The hardware scene's measured storage was adjusted independently: 12,416 raw
contacts, 22,528 MF rows and 131,072 triangle candidates at 256 objects. An RTX
one-frame launch test passes; this is **not** a full RTX rollout. The 24,576-row
diagnostic failed PTX admission on RTX (105,660 bytes shared memory versus the
101,376-byte limit), although it ran on GB300. Do not increase row storage
without checking per-device admission, and never silently drop physical rows.
The completed hardware rollout leaves only 3.5% raw-contact and 7.6% MF-row
headroom; it qualifies this seed and motion, not arbitrary seeds. Across all
301 post-settling saved poses, 240–243 pieces overlap the bin's interior
footprint and others rest outside after spilling over the rim. Mesh minimum
world height is −0.0041 mm; no floor fall-through was observed. The grasp is
physical but not perfectly centered, and actual travel is below the commanded
100 mm span. `hardware_pose_checks.json` records the measured trajectory.

Accepted container poses/videos and cube diagnostic captures live under
`/home/octi/Projects/fpgs-recipes-evidence-20260919/scene-cad-upgrade-20260920-46PIqu`.
The hardware video shows saved poses from t=3–13 s at 1× and explicitly labels
the omitted settling interval. The full 0–13 s capture is retained. All three
container videos passed full decode checks; the cube has an initial-geometry
preview only, not an accepted robot-turn video.
The coin capture uses 2,688 raw-contact and 8,064 MF-row storage, reduced after
the earlier larger-capacity run. During closed-grip sliding, TCP-to-handle
offset varies by 0.53 mm RMS; the tray is moved by contact, not a pose target.

The current geometry/wrench suite passes all **24 tests**, including sampled
GPU SDF cavity/gap checks; deprecation warnings are treated as errors. Ruff
check and formatting checks pass. These checks do not certify real-cube
mechanics or qualify a backend performance comparison.

For the cube, the user selected a classic non-magnetic 57 mm reference and
requested isolated force testing **before** further robot integration. The
[mechanical characterization note](RUBIK_MECHANICAL_CHARACTERIZATION_20260920.md)
separates real-data requirements, force-couple experiments, and the 74.15-fold
added-inertia mismatch. No current cube variant is declared calibrated.

Asset provenance:

- [Factory nuts/bolts acknowledgment](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/assets/licenses/factory-acknowledgments.txt)
  and [BSD-3-Clause license](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/LICENSE.txt).
- [MuJoCo cube reference, Apache-2.0](https://github.com/google-deepmind/mujoco/blob/5ceb72b1bbe87612a2859304d2c22a15cd3c180d/model/cube/cube_3x3x3.xml).
- Other meshes are original parametric CAD under this repository's Apache-2.0
  license, not imported NIST or product-certified household models.

## Archived primitive prototype

Each previous scene used one torque-driven Franka and procedural collision geometry.
Seeded lattices, convex annular sectors, and polar ring packing express the
layouts. IK runs at setup; a periodic quintic joint-target interpolation drives
the robot. Loose-object poses are never prescribed during simulation.

### Completed checks — previous geometry only

Single-world FPGS runs: 720 frames / 12 simulated seconds, 10 substeps per frame,
24 solver iterations, seed 42, CUDA graphs, matrix-free mode. Counts below are
independent peak watermarks, not values necessarily reached simultaneously.

| Scene | Objects | GPU | Peak contacts consumed | Peak articulated rows | Peak free-body rows |
| --- | ---: | --- | ---: | ---: | ---: |
| Mixed hardware | 128 | RTX PRO6000 | 1,592 | 37 | 4,725 |
| Coin bed/tower | 256 | GB300 | 2,059 | 100 | 6,117 |
| Dish rack | 40 | RTX PRO6000 | 532 | 64 | 1,542 |
| Retained cube | 26 | GB300 | 286 | 835 | 0 |

All completed without escaped/non-finite bodies or solver row overflow/drop
counters. Hardware, coins, and dishes also passed the runner's sticky
narrow-phase capacity checks. Hardware reduction peaked at 7,248 raw contacts
before patch reduction; no reducer invariant failures or fallback frames.
These are smoke checks, not convergence certificates, RL success rates, or
performance comparisons. Not every scene has been checked on both GPUs.

A separate 2-world × 16-coin, 30-frame RTX run passed state/capacity checks.
Eight CPU tests pass with deprecation warnings treated as errors: positive
mass/inertia, seeded/count behavior, initial clearance, open hardware bores,
and all six cube-face quarter-turn FK permutations. Ruff check/format pass.

The first dish script swept in Y and ejected cutlery while descending across
the front rail. The final X sweep stays in the cutlery bay; nominal front-pad
clearance including contact gaps is about 2.2 mm. Its full cycle passed.

## Running the current scenes

These commands use the current CAD geometry, not the archived primitive
prototype. From this worktree with its dependencies and CUDA:

```bash
python -m newton.examples robot_clutter --scene hardware --device cuda:1 --viewer null --num-frames 780 --test --report hardware.json
python -m newton.examples robot_clutter --scene coins --device cuda:0 --viewer null --num-frames 720 --test --report coins.json
python -m newton.examples robot_clutter --scene dishes --device cuda:0 --viewer null --num-frames 720 --test --report dishes.json
python -m newton.examples.robot._clutter_cube_probe --device cuda:0 --mode couple --amplitudes 0 0.001 --substeps 20 --iterations 48 --report cube_probe.json
python -W error::DeprecationWarning -m unittest newton.tests.test_clutter_cube newton.tests.test_clutter_hardware newton.tests.test_clutter_layouts newton.tests.test_clutter_tray newton.tests.test_clutter_dishes_cad
```

The executed environment was Warp 1.17.0, through
`uv run --no-project --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python`.
Archived primitive-prototype JSONs: `/tmp/clutter-hardware-fpgs-720.json`,
`/tmp/clutter-coins-fpgs-720.json`, `/tmp/clutter-dishes-x-sweep-fpgs-720.json`,
and `/tmp/clutter-coins-2x16-fpgs-30.json`.
The archived primitive-cube result came from a diagnostic wrapper that called
the Example step/test hooks for 720 frames and printed its report; it was not
saved to JSON. Those files do not reproduce or qualify the current geometry.
The README now uses current simulator views, including a settled hardware frame.

## Important limitations

- Split FPGS hardcodes a 64-body MF map in `solver_feather_pgs.py`; the map can
  silently omit additional free bodies even when row capacity checks pass.
  Matrix-free immediate solving bypasses it with warm start disabled (default).
  This example rejects split mode above 64 free objects; the underlying solver
  bug is not fixed here. A bare 128-piece case isolated the issue without Franka
  or CUDA capture. Do not hide it by shrinking the object count.
- Hardware enables existing FPGS-only body-pair reduction (40 mm cells, zero
  hysteresis); other scenes do not. Raw collision storage remains separately
  sized. This is an approximate manifold reduction, not lossless geometry
  simplification. Its default hardware settings are not backend matched.
- MJWarp external-contact hardware is unstable before Franka moves, with both
  4 and 128 pieces, under versions 3.12 and required 3.11. Positive masses and
  inertias and external contact capacity were checked. Native contacts stayed
  finite longer but overflowed their buffer (12,660 requested / 8,320 available),
  so that diagnostic is not a qualified alternative or proof of root cause.
- The cube uses spherical retainers, not commercial puzzle tracks. The archived
  prototype's 720-frame diagnostic kept cubie radii within 3.2e-8 m and observed robot contact, but its
  final fitted top rotation was only 7.9 degrees, with 5.7 mm rigid-fit error.
  This is retained articulated manipulation, not a completed face turn.
