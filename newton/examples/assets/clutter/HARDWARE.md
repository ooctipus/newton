# Hardware CAD provenance and approximation

The clutter example downloads the M12/M16 loose nut and bolt OBJ files from
[NVIDIA IsaacGymEnvs Factory](https://github.com/isaac-sim/IsaacGymEnvs/tree/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/assets/factory/mesh/factory_nut_bolt),
pinned at `aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3`. These are the CAD asset
family used by Isaac Lab's NIST tasks, not meshes created by this example.
Factory identifies its nut and bolt assets as custom NVIDIA designs distributed
under the repository's [BSD-3-Clause license](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/LICENSE.txt);
see the [asset-specific acknowledgment](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/assets/licenses/factory-acknowledgments.txt).
Downloaded source assets remain in Newton's asset cache, not this repository.

The beveled washers, 20-degree involute gears, rounded bin, and handle are
authored in this example under Apache-2.0. In particular, the gears are **not**
the NIST gear assets. All dimensions are SI meters; M12/M16 fasteners retain
their source dimensions and threads without arbitrary size randomization.

Before SDF cooking, the loader welds duplicated export vertices and splits
outer-surface T-junctions, without filling holes. Quadric decimation then reduces
the source CAD to at most 3,200 triangles per fastener. It checks closed-surface
topology, consistent winding, unchanged Euler characteristic, and <1% volume
change. One mesh and one cached SDF are shared by all instances of each variant.

Sampled bidirectional surface error against the repaired source (4,000 points
per direction, seed 42; nearest-triangle distances evaluated in millimeters to
avoid small-coordinate numerical tolerances):

| Part | Source triangles | Reduced triangles | Maximum sampled error |
|---|---:|---:|---:|
| M12 bolt | 23,506 | 3,200 | 22.3 micrometers |
| M16 bolt | 25,668 | 3,200 | 39.1 micrometers |
| M12 nut | 13,500 | 3,198 | 6.4 micrometers |
| M16 nut | 15,080 | 3,196 | 10.4 micrometers |

These are sampled errors, not a global Hausdorff bound or thread-engagement
validation. Hardware SDF voxel size is 0.1 mm. This is a bin agitation scene,
not a metrology or precision screw-assembly benchmark.

The default 256 pieces occupy about 0.919 liters of solid material, or 51% of
the bin's approximately 1.798-liter interior (before rounded-corner correction).
The steel payload is about 7.17 kg and remains table-supported. Initial packing
uses shallow per-kind layers with sideways bolts: no pair of object AABBs
overlaps, and the highest piece is 159 mm above the tabletop before settling.
