# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Time the tree scans, the held tree's launch widths and the general route's passes, one launch each.

``schedules``: every schedule of the pose, motion and subtree scans on every
shape, generated on the scan plan, for the rule that picks a schedule from the
plan's data (``tree_scan.choose_schedule``). A block per group of trees (the
held tree's launch) stages each tree's state block in shared memory, as the
held-tree kernels hold it, and runs the scan ``--repeat`` times.

``lanes``: the held tree's finish, predictor and repair on replicated trees of
every shape at each lane width a tree admits (the scans' schedules chosen at
that width) and on the heavy chains, for the rule that picks the width from the
tree's data (``tree_kinetic_state.held_lanes``); ``rule`` marks its pick.

``general``: the general route's passes (``tree_passes``) on a solver's own
arrays and binding, one thread per articulation (CRBA: per column), at the
solver's block sizes.

``serial``: the one-articulation-per-thread passes at every block size from 1
to 256 and each ``--world-counts``, for their fixed block size
(``tree_passes.SERIAL_BLOCK_DIM``).

Each timing is the median of ``--samples`` CUDA-event-timed launches after a
warm-up, in microseconds per launch. Run on an otherwise idle GPU::

    python tools/fpgs_bench/tree_scan_bench.py schedules --groups 16384 --repeat 8
    python tools/fpgs_bench/tree_scan_bench.py lanes --worlds 4096
    python tools/fpgs_bench/tree_scan_bench.py general --worlds 4096
    python tools/fpgs_bench/tree_scan_bench.py serial --world-counts 4096 16384
"""

import argparse
import contextlib
import functools
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from newton._src.solvers.feather_pgs import sparse_factor, tree_kinetic_state, tree_passes
from newton._src.solvers.feather_pgs import tree_scan as ts
from newton._src.solvers.feather_pgs.device_preludes import dual_native, lane_group
from newton.tests import feather_pgs_tree_scan_fixture as fx

_STAGE = """
#if defined(__CUDA_ARCH__)
    __shared__ float stage_values[$STAGED];
    float* stage=stage_values+SUB*$TOTAL;
    float* stage_global=reinterpret_cast<float*>(base)+group*$TOTAL;
    for(int k=LANE;k<$TOTAL;k+=STRIDE)stage[k]=stage_global[k];
    SYNC();
    const uint64_t address=reinterpret_cast<uint64_t>(stage);
    for(int repetition=0;repetition<$REPEAT;++repetition) {
$PASS
    }
    SYNC();
    for(int k=LANE;k<$TOTAL;k+=STRIDE)stage_global[k]=stage[k];
#endif
"""


@functools.cache
def staged_kernel(item, layout, plan, repeat):
    """The pass's kernel with each tree's state block staged in shared memory, run ``repeat`` times."""
    sizes = layout.sizes()
    body = (
        _STAGE.replace("$PASS", fx.generated_text(item, plan, layout))
        .replace("$STAGED", str(sizes["TOTAL"] * layout.pack))
        .replace("$REPEAT", str(repeat))
    )
    snippet = dual_native(layout.render(body), preludes=(lane_group(layout.lanes), *item.combine.preludes()))
    if item.direction is ts.Direction.UP or item.name == "twist_shfl":

        @wp.func_native(snippet)
        def run_staged(
            base: wp.uint64,
            p: sparse_factor.SparsePlan,
            plan: ts.ScanPlan,
            f: tree_kinetic_state.ForceInput,
            flag: int,
            group: int,
        ): ...

    else:

        @wp.func_native(snippet)
        def run_staged(
            base: wp.uint64,
            p_unused: sparse_factor.SparsePlan,
            p: ts.ScanPlan,
            f: tree_kinetic_state.ForceInput,
            flag: int,
            group: int,
        ): ...

    _lanes, _, _, _ = tree_kinetic_state._thread_helpers(layout)
    pack = layout.pack

    def run(
        base: wp.uint64,
        sparse: sparse_factor.SparsePlan,
        structure: ts.ScanPlan,
        f: tree_kinetic_state.ForceInput,
        flag: int,
    ):
        block, _logical = wp.tid()
        lanes = _lanes()
        run_staged(base, sparse, structure, f, flag, block * wp.static(pack) + lanes[2])

    run.__name__ = run.__qualname__ = f"bench_{item.name}_{layout.bodies}_{layout.lanes}"
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


def _launcher(item, plan, layout, groups, repeat, device, flag=1):
    data = fx.inputs(plan, layout, groups, 3, device)
    kernel = staged_kernel(item, layout, plan, repeat)

    def launch():
        wp.launch_tiled(
            kernel,
            dim=[groups // layout.pack],
            inputs=[wp.uint64(data.scratch.ptr), data.sparse, data.scan, data.force, flag],
            block_dim=layout.block,
            device=device,
        )

    return launch


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
        for scan, (combine, direction, schedules) in combos.items():
            row = {"shape": shape, "bodies": plan.count, "depth": plan.max_depth, "scan": scan}
            for schedule in schedules:
                lanes = fx.chain_lanes(plan) if schedule is ts.Schedule.HEAVY_CHAIN else fx.lanes_of(plan)
                item = fx.Pass(f"{scan}_{schedule.value}", combine, direction, schedule)
                if not fx.applies(item, plan, lanes):
                    continue
                flag = 0 if scan == "reduce6" else 1
                launch = _launcher(item, plan, fx.layout_of(plan, lanes), args.groups, args.repeat, device, flag)
                row[schedule.value] = _time(launch, args.samples, device)
            row["rule"] = ts.choose_schedule(plan, direction, combine, cuda=True, lanes=fx.chain_lanes(plan)).value
            rows.append(row)
            print(_format(row), flush=True)
    return rows


def _held_tree(shape, worlds, device, lanes=None, schedules=None):
    """A sparse-route solver whose held tree runs replicated trees of ``shape`` at the given lanes and schedules
    (the rules' choices where ``None``)."""
    from newton.tests.test_feather_pgs_sparse_solver import _make_solver  # noqa: PLC0415
    from newton.tests.test_feather_pgs_tree_scan import _held_tree_model  # noqa: PLC0415

    model = _held_tree_model(shape, worlds, device)
    with contextlib.ExitStack() as stack:
        if lanes is not None:
            stack.enter_context(
                mock.patch.object(tree_kinetic_state, "held_lanes", lambda tree, groups, *, cuda: lanes)
            )
        if schedules is not None:
            stack.enter_context(
                mock.patch.object(
                    tree_kinetic_state, "held_schedules", lambda tree, lanes, *, chains, reference=False: schedules
                )
            )
        solver = _make_solver(model, True)
    return model, solver


def _repair(owner, state_in, augmented, dt):
    """A repair of every group: the caches invalidated, then a begin."""
    owner.invalidate()
    owner.begin(state_in, augmented, state_in.joint_qd, dt, True)


def bench_lanes(args, device):
    import newton  # noqa: PLC0415

    rows = []
    for shape in args.shapes or [name for name, parents in fx.SHAPES.items() if len(parents) <= 44]:
        plan = fx.plan_of(shape)
        rule = tree_kinetic_state.held_lanes(plan, args.worlds, cuda=True)
        widths = [64] if plan.count > 32 else [w for w in (4, 8, 16, 32) if w >= plan.count]
        configurations = [(lanes, None) for lanes in widths]
        packing = plan.chain_packing(2)
        if packing is not None:
            chain_lanes = max(packing.lanes, 32 if plan.count <= 32 else 64)
            configurations.append((chain_lanes, tree_kinetic_state.HeldSchedules(*(ts.Schedule.HEAVY_CHAIN,) * 4)))
        for lanes, schedules in configurations:
            model, solver = _held_tree(shape, args.worlds, device, lanes, schedules)
            owner = solver._tree_kinetic_state
            if owner is None:
                continue
            (group,) = owner.groups
            rng = np.random.default_rng(5)
            state_in, state_out, control = model.state(), model.state(), model.control()
            state_in.joint_qd.assign(rng.uniform(-0.5, 0.5, model.joint_dof_count).astype(np.float32))
            state_in.body_f.assign(rng.normal(0, 0.1, (model.body_count, 6)).astype(np.float32))
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            dt = 1.0 / 240.0
            solver._last_step_dt = dt
            augmented = solver._prepare_augmented_state(state_in, model.state(), control)

            repair = functools.partial(_repair, owner, state_in, augmented, dt)
            repair()
            row = {
                "shape": shape,
                "bodies": plan.count,
                "dofs": plan.dof_count,
                "lanes": group.layout.lanes,
                "schedules": group.schedules.tag[1:],
                "rule": group.layout.lanes == rule and schedules is None,
            }
            finish = functools.partial(owner.finish, state_in, augmented, state_out, dt, False)
            predict = functools.partial(owner.predict, state_in, augmented, control, state_in.joint_qd, dt)
            row["finish"] = _time(finish, args.samples, device)
            row["predict"] = _time(predict, args.samples, device)
            row["repair"] = _time(repair, args.samples, device)
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
            launch = pass_launch(route, name, block_dim=args.block_dim, write_inertia_records=1)
            row["time"] = _time(launch, args.samples, device)
            rows.append(row)
            print(_format(row), flush=True)
    return rows


def bench_serial(args, device):
    from newton.tests.feather_pgs_tree_scan_fixture import (  # noqa: PLC0415
        articulated_model,
        general_route,
        pass_launch,
        run_inputs,
    )

    rows = []
    for worlds in args.world_counts:
        for shape in args.shapes or ("anymal13", "g1_30", "chain64", "random60", "pendulum2"):
            route = general_route(articulated_model((shape,), worlds, device))
            for name in args.passes or ("fk_id", "tau", "composite_inertia"):
                run_inputs(route, name)
                row = {"worlds": worlds, "shape": shape, "pass": name, "fixed": tree_passes.SERIAL_BLOCK_DIM}
                for block in (1, 2, 4, 8, 16, 32, 64, 128, 256):
                    launch = pass_launch(route, name, block_dim=block, write_inertia_records=1)
                    row[str(block)] = _time(launch, args.samples, device)
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
    parser.add_argument("mode", choices=("schedules", "lanes", "general", "serial"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--groups", type=int, default=16384)
    parser.add_argument("--worlds", type=int, default=4096)
    parser.add_argument("--world-counts", type=int, nargs="*", default=[4096, 16384], help="serial: the world counts")
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--shapes", nargs="*")
    parser.add_argument("--passes", nargs="*")
    parser.add_argument("--block-dim", type=int, help="general: one block size for every pass (default: the solver's)")
    parser.add_argument("--json")
    args = parser.parse_args()
    wp.init()
    device = wp.get_device(args.device)
    modes = {"schedules": bench_schedules, "lanes": bench_lanes, "general": bench_general, "serial": bench_serial}
    rows = modes[args.mode](args, device)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
