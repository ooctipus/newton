# 20 matched task recipes

Clean main `5238407d` versus clean candidate `d5c2177d`; 16,384 environments, seed 0, 200 warmup steps, 40 wall steps then 40 graph-profile steps, one independent capture per arm. No video or visualizer. FPS is inclusive random-action environment stepping, not PPO. All 20 matched pairs passed the supported checks. Small inactive-path differences are not established gains or regressions.

| Task | main FPS | PR FPS | gain |
|---|---:|---:|---:|
| Franka lift | 466,574 | 481,409 | 1.03× |
| KukaAllegro lift | 360,294 | 365,223 | 1.01× |
| Allegro reorient | 286,291 | 286,404 | 1.00× |
| ANYmal-D flat | 535,626 | 625,378 | 1.17× |
| G1 rough | 279,894 | 366,709 | 1.31× |
| SO101 Keyboard | 97,341 | 98,633 | 1.01× |
| Ant | 1,640,350 | 1,663,119 | 1.01× |
| ANYmal-D rough | 163,410 | 174,379 | 1.07× |
| Cartpole | 2,594,743 | 2,761,774 | 1.06× |
| Cassie flat | 795,500 | 777,294 | 0.98× |
| Cassie rough | 482,384 | 474,593 | 0.98× |
| Franka reach | 1,142,546 | 1,151,362 | 1.01× |
| G1 flat | 406,129 | 551,365 | 1.36× |
| Go2 flat | 783,237 | 924,426 | 1.18× |
| Go2 rough | 287,158 | 311,144 | 1.08× |
| H1 flat | 658,204 | 774,321 | 1.18× |
| H1 rough | 363,385 | 403,832 | 1.11× |
| Humanoid | 323,338 | 324,367 | 1.00× |
| UR10 reach | 1,958,659 | 1,935,591 | 0.99× |
| KukaAllegro reorient | 347,020 | 343,866 | 0.99× |

Across three independent captures per revision (two additional balanced rounds), main-physics-graph throughput improves **1.24–1.31× on ANYmal-D flat** and **1.436–1.439× on G1 rough**. Inclusive stepping improves 1.17–1.19× and 1.27–1.31×, respectively. These are observed paired ranges, not confidence intervals; the 20-task table retains its original single-capture values.

<details>
<summary>Main physics graph timings and path selection</summary>

Milliseconds per environment step; excludes sensor/eager work. These GPU timings are from a separate 40-step window, not a component to subtract directly from the wall measurement.

| Task | main graph ms | PR graph ms | graph gain | Sparse path |
|---|---:|---:|---:|---|
| Franka lift | 16.775 | 16.779 | 1.00× | inactive |
| KukaAllegro lift | 26.423 | 26.889 | 0.98× | inactive |
| Allegro reorient | 49.482 | 49.377 | 1.00× | inactive |
| ANYmal-D flat | 22.467 | 18.084 | 1.24× | active |
| G1 rough | 44.191 | 30.774 | 1.44× | active |
| SO101 Keyboard | 119.171 | 117.886 | 1.01× | inactive |
| Ant | 4.767 | 4.762 | 1.00× | inactive |
| ANYmal-D rough | 91.267 | 84.549 | 1.08× | active |
| Cartpole | 0.471 | 0.471 | 1.00× | inactive |
| Cassie flat | 11.209 | 11.251 | 1.00× | inactive |
| Cassie rough | 22.537 | 22.551 | 1.00× | inactive |
| Franka reach | 10.621 | 10.623 | 1.00× | inactive |
| G1 flat | 27.321 | 17.406 | 1.57× | active |
| Go2 flat | 16.879 | 13.278 | 1.27× | active |
| Go2 rough | 49.671 | 44.969 | 1.10× | active |
| H1 flat | 13.309 | 10.075 | 1.32× | active |
| H1 rough | 31.971 | 27.433 | 1.17× | active |
| Humanoid | 42.509 | 42.673 | 1.00× | inactive |
| UR10 reach | 5.119 | 5.123 | 1.00× | inactive |
| KukaAllegro reorient | 26.512 | 26.478 | 1.00× | inactive |

- Franka lift: Inactive: free rigid body, three articulations/world and multiple size groups.
- KukaAllegro lift: Inactive: free rigid body, three articulations/world and multiple size groups.
- Allegro reorient: Inactive: native joint velocity limits enabled; also a free object and multiple articulations/world.
- ANYmal-D flat: Active: 18-DOF branched single-articulation topology.
- G1 rough: Active: 43-DOF branched single-articulation topology.
- SO101 Keyboard: Inactive: separate SO101 and 108-DOF keyboard articulations; multiple articulation/size-group gate (keyboard also exceeds 64-bit mask domain).
- Ant: Inactive: native pgs_mode=split; sparse admission requires matrix_free.
- ANYmal-D rough: Active: 18-DOF branched single-articulation topology.
- Cartpole: Inactive: native pgs_mode=split; also a full-chain rather than branched sparse opportunity.
- Cassie flat: Inactive: two toe endpoint supports jointly cover all 18 DOFs, violating max_support<size; derived from cached USD topology, not runtime-logged support.
- Cassie rough: Inactive: two toe endpoint supports jointly cover all 18 DOFs, violating max_support<size; derived from cached USD topology, not runtime-logged support.
- Franka reach: Inactive: native joint velocity limits enabled.
- G1 flat: Active: 43-DOF branched single-articulation topology.
- Go2 flat: Active: 18-DOF branched single-articulation topology.
- Go2 rough: Active: 18-DOF branched single-articulation topology.
- H1 flat: Active: 25-DOF branched single-articulation topology.
- H1 rough: Active: 25-DOF branched single-articulation topology.
- Humanoid: Inactive: native pgs_mode=split; sparse admission requires matrix_free.
- UR10 reach: Inactive: native joint velocity limits enabled.
- KukaAllegro reorient: Inactive: free rigid body, three articulations/world and multiple size groups.

</details>

Keyboard uses 736 rows on both arms. The rejected 704-row attempt is excluded. Exact sampling, unrounded results, storage shapes, per-task configuration, supported-check outcomes and raw artifact hashes are in [results.json](results.json). The primary graph is not all environment GPU work; rough-task sensor graphs are separately recorded in JSON. No lifetime collision-capacity or training-convergence claim is made.

## Matched matrix-free controls

Both revisions receive only the same additional `pgs_mode='matrix_free'` override; all other inputs, capacities, budgets and features remain matched. These supplementary rows are single captures, separate from the 20-task table.

| Task (matrix-free on both) | main FPS | PR FPS | gain |
|---|---:|---:|---:|
| Ant | 2,176,772 | 2,292,864 | 1.05× |
| Humanoid | 941,266 | 1,003,308 | 1.07× |

Main physics graph: Ant 2.808 → 2.180 ms (1.29×); Humanoid 9.339 → 7.994 ms (1.17×). Both activate the sparse representation and pass the supported checks. The much larger split-to-matrix-free improvement already exists on main; it is a configuration gain, not a PR gain. The comparisons above are matrix-free main → matrix-free PR only.
