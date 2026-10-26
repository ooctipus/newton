# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Time each tree pass as its existing kernel and as its ``tree_scan`` configuration, with one launch.

``passes``: every held-tree pass on every shape it runs on. A block per group
of trees (the held tree's launch) stages each tree's state block in shared
memory, as the held-tree kernels hold it, runs the pass ``--repeat`` times and
writes the block back; the arms differ only in the pass text: ``existing``
(the module text), ``generated-held`` (the configuration on the existing plan
structs) and ``generated-scan`` (on the scan plan, levels by CSR).

``schedules``: every schedule of the pose, motion and subtree scans on every
shape, generated on the scan plan, for the rule that picks a schedule from the
plan's data.

``general``: the general route's passes (``tree_passes``) on a solver's own
arrays and binding, one thread per articulation (CRBA: per column), at the
solver's block sizes.

Each timing is the median of ``--samples`` CUDA-event-timed launches after a
warm-up, in microseconds per launch. Run on an otherwise idle GPU::

    python tools/fpgs_bench/tree_scan_bench.py passes --groups 16384 --repeat 8
    python tools/fpgs_bench/tree_scan_bench.py schedules --groups 16384 --repeat 8
    python tools/fpgs_bench/tree_scan_bench.py general --worlds 4096
"""

import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from newton._src.solvers.feather_pgs import sparse_factor, tree_chain_scan, tree_kinetic_state
from newton._src.solvers.feather_pgs import tree_scan as ts
from newton._src.solvers.feather_pgs.device_preludes import dual_native, lane_group
from newton.tests import feather_pgs_tree_scan_fixture as fx

_STAGE = """
#if defined(__CUDA_ARCH__)
    __shared__ float stage_values[$STAGED];
    const int stage_lane=$STAGE_LANE,stage_stride=$STAGE_STRIDE;
    float* stage=stage_values+$STAGE_OFFSET;
    float* stage_global=reinterpret_cast<float*>(base)+group*$TOTAL;
    for(int k=stage_lane;k<$TOTAL;k+=stage_stride)stage[k]=stage_global[k];
    $STAGE_SYNC
    const uint64_t address=reinterpret_cast<uint64_t>(stage);
    for(int repetition=0;repetition<$REPEAT;++repetition) {
$PASS
    }
    $STAGE_SYNC
    for(int k=stage_lane;k<$TOTAL;k+=stage_stride)stage_global[k]=stage[k];
#endif
"""


def _pass_text(item, plan, layout, arm, schedule=None):
    if arm == "existing":
        return fx.existing_text(item, layout)
    if schedule is not None and schedule is not item.schedule:
        item = fx.Pass(item.name, item.combine, item.direction, schedule, item.site, item.plan_struct)
    return fx.generated_text(item, plan, layout, dialect=arm.split("-", 1)[1])


@functools.cache
def staged_kernel(item, layout, arm, plan, repeat, schedule=None):
    """The pass's kernel with each tree's state block staged in shared memory, run ``repeat`` times."""
    chain = (item.plan_struct == "chain" and arm == "existing") or (
        schedule or item.schedule
    ) is ts.Schedule.HEAVY_CHAIN
    sizes = layout.sizes()
    text = _pass_text(item, plan, layout, arm, schedule)
    body = (
        _STAGE.replace("$PASS", text)
        .replace("$STAGED", str(sizes["TOTAL"] * (1 if chain else layout.pack)))
        .replace("$STAGE_LANE", "(int)threadIdx.x" if chain else "LANE")
        .replace("$STAGE_STRIDE", "(int)blockDim.x" if chain else "STRIDE")
        .replace("$STAGE_OFFSET", "0" if chain else f"SUB*{sizes['TOTAL']}")
        .replace("$STAGE_SYNC", "__syncthreads();" if chain else "SYNC();")
        .replace("$REPEAT", str(repeat))
    )
    preludes = item.combine.preludes() if arm != "existing" else ()
    snippet = dual_native(layout.render(body), preludes=(lane_group(layout.lanes), *preludes))
    if arm == "generated-scan":
        plan_type = ts.ScanPlan
    else:
        plan_type = tree_chain_scan.ChainPlan if item.plan_struct == "chain" else tree_kinetic_state.KineticPlan
    reduction = item.direction is ts.Direction.UP or item.name == "twist_shfl"

    if reduction:

        @wp.func_native(snippet)
        def run_staged(
            base: wp.uint64,
            p: sparse_factor.SparsePlan,
            plan: plan_type,
            f: tree_kinetic_state.ForceInput,
            flag: int,
            group: int,
        ): ...

    else:

        @wp.func_native(snippet)
        def run_staged(
            base: wp.uint64,
            p_unused: sparse_factor.SparsePlan,
            p: plan_type,
            f: tree_kinetic_state.ForceInput,
            flag: int,
            group: int,
        ): ...

    _lanes, _, _, _ = tree_kinetic_state._thread_helpers(layout)
    pack = 0 if chain else layout.pack

    def run(
        base: wp.uint64,
        sparse: sparse_factor.SparsePlan,
        structure: plan_type,
        f: tree_kinetic_state.ForceInput,
        flag: int,
    ):
        block, _logical = wp.tid()
        lanes = _lanes()
        group = block
        if wp.static(pack > 0):
            group = block * wp.static(pack) + lanes[2]
        run_staged(base, sparse, structure, f, flag, group)

    label = (schedule or item.schedule).value
    run.__name__ = run.__qualname__ = (
        f"bench_{item.name}_{label}_{arm.replace('-', '_')}_{layout.bodies}_{layout.lanes}"
    )
    return wp.kernel(module="unique", enable_backward=False)(run)


def _time(launch, samples, device):
    """Median and 90th percentile of ``samples`` launches, each timed by CUDA events [us]."""
    for _ in range(3):
        launch()
    wp.synchronize_device(device)
    times = []
    for _ in range(samples):
        start, end = wp.Event(device, enable_timing=True), wp.Event(device, enable_timing=True)
        wp.record_event(start)
        launch()
        wp.record_event(end)
        wp.synchronize_event(end)
        times.append(wp.get_event_elapsed_time(start, end) * 1000.0)
    return float(np.median(times)), float(np.percentile(times, 90))


def _launcher(item, plan, layout, arm, groups, repeat, device, schedule=None, flag=1):
    data = fx.inputs(item, plan, layout, groups, 3, device)
    kernel = staged_kernel(item, layout, arm, plan, repeat, schedule)
    chain = (schedule or item.schedule) is ts.Schedule.HEAVY_CHAIN or (
        item.plan_struct == "chain" and arm == "existing"
    )
    pack = 1 if chain else layout.pack
    structure = data.scan if arm == "generated-scan" else data.held
    block = fx.block_dim(item, plan, layout) if chain else layout.block
    if chain and arm == "generated-scan":
        packing = plan.chain_packing(2)
        block = max(layout.block, packing.lanes if packing is not None else 32)

    def launch():
        wp.launch_tiled(
            kernel,
            dim=[groups // pack],
            inputs=[wp.uint64(data.scratch.ptr), data.sparse, structure, data.force, flag],
            block_dim=block,
            device=device,
        )

    return launch


def bench_passes(args, device):
    rows = []
    for shape in args.shapes or fx.SHAPES:
        plan = fx.plan_of(shape)
        for packed in (0, 8):
            lanes = fx.lanes_of(plan, packed)
            if packed and lanes >= 32:
                continue
            layout = fx.layout_of(plan, lanes)
            for name, item in fx.PASSES.items():
                if not fx.applies(item, plan, lanes) or (item.plan_struct == "chain" and packed):
                    continue
                if args.passes and name not in args.passes:
                    continue
                row = {
                    "shape": shape,
                    "bodies": plan.count,
                    "lanes": lanes,
                    "pass": name,
                    "schedule": item.schedule.value,
                }
                for arm in ("existing", "generated-held", "generated-scan"):
                    launch = _launcher(item, plan, layout, arm, args.groups, args.repeat, device)
                    row[arm] = _time(launch, args.samples, device)
                rows.append(row)
                print(_format(row), flush=True)
    return rows


def bench_schedules(args, device):
    combos = {
        "pose": (
            fx.POSE,
            ts.Direction.DOWN,
            (ts.Schedule.JUMP_SHFL, ts.Schedule.JUMP_SMEM, ts.Schedule.HEAVY_CHAIN, ts.Schedule.LEVEL),
        ),
        "motion": (
            fx.MOTION,
            ts.Direction.DOWN,
            (ts.Schedule.JUMP_SHFL, ts.Schedule.JUMP_SMEM, ts.Schedule.HEAVY_CHAIN, ts.Schedule.LEVEL),
        ),
        # The subtree sum of every step (6 wrench channels) and of a geometry refresh (19 with the moments).
        "reduce6": (
            fx.WRENCH_MOMENTS,
            ts.Direction.UP,
            (ts.Schedule.DFS_INTERVAL, ts.Schedule.HEAVY_CHAIN, ts.Schedule.LEVEL),
        ),
        "reduce19": (
            fx.WRENCH_MOMENTS,
            ts.Direction.UP,
            (ts.Schedule.DFS_INTERVAL, ts.Schedule.HEAVY_CHAIN, ts.Schedule.LEVEL),
        ),
    }
    rows = []
    for shape in args.shapes or fx.SHAPES:
        plan = fx.plan_of(shape)
        lanes = fx.lanes_of(plan)
        layout = fx.layout_of(plan, lanes)
        for scan, (combine, direction, schedules) in combos.items():
            row = {"shape": shape, "bodies": plan.count, "depth": plan.max_depth, "scan": scan}
            for schedule in schedules:
                if schedule in (ts.Schedule.JUMP_SHFL, ts.Schedule.DFS_INTERVAL) and plan.count > 32:
                    continue
                if schedule is ts.Schedule.HEAVY_CHAIN and plan.chain_packing(2) is None:
                    continue
                item = fx.Pass(
                    f"{scan}_{schedule.value}",
                    combine,
                    direction,
                    schedule,
                    "",
                    "chain" if schedule is ts.Schedule.HEAVY_CHAIN else "kinetic",
                )
                flag = 0 if scan == "reduce6" else 1
                launch = _launcher(
                    item, plan, layout, "generated-scan", args.groups, args.repeat, device, schedule, flag
                )
                row[schedule.value] = _time(launch, args.samples, device)
            row["rule"] = ts.choose_schedule(plan, direction, combine, cuda=True, lanes=lanes).value
            rows.append(row)
            print(_format(row), flush=True)
    return rows


def bench_general(args, device):
    from newton.tests.feather_pgs_tree_scan_fixture import (  # noqa: PLC0415
        GENERAL_PASSES,
        articulated_model,
        general_route,
        pass_launch,
        run_inputs,
    )

    rows = []
    for shape in args.shapes or ("anymal13", "humanoid13", "g1_30", "chain64", "random60", "pendulum2"):
        route = general_route(articulated_model((shape,), args.worlds, device))
        for name in args.passes or GENERAL_PASSES:
            run_inputs(route, name)
            row = {"shape": shape, "articulations": route.model.articulation_count, "pass": name}
            row["time"] = _time(pass_launch(route, name, write_inertia_records=1), args.samples, device)
            rows.append(row)
            print(_format(row), flush=True)
    return rows


def _format(row):
    parts = []
    for key, value in row.items():
        if isinstance(value, tuple):
            parts.append(f"{key}={value[0]:.1f}us")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("passes", "schedules", "general"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--groups", type=int, default=16384)
    parser.add_argument("--worlds", type=int, default=4096)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--shapes", nargs="*")
    parser.add_argument("--passes", nargs="*")
    parser.add_argument("--json")
    args = parser.parse_args()
    wp.init()
    device = wp.get_device(args.device)
    rows = {"passes": bench_passes, "schedules": bench_schedules, "general": bench_general}[args.mode](args, device)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
