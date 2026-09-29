# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure exact-size homogeneous model replacement at fresh-episode boundaries.

Import and prepare each USD prototype once. Replication expands the prepared
Model and solver directly; ``--construction-mode rebuild`` measures the ordinary
builder and solver construction reference. No mode caches population sizes.
The previous batch remains alive until its replacement has been validated.
``--graph-mode update`` still records each fresh graph, then attempts to reuse
the previous executable through CUDA's graph update API. Incompatible updates
fall back to fresh instantiation; no size-specific model or solver is cached.
``--scratch-mode pool`` recycles temporary storage within each exact-size batch.
It uses Warp's public allocator hooks and persistent CUDA allocations during
relaxed capture, avoiding allocation/free graph nodes without a planning capture.
This diagnostic assumes one host thread and one CUDA stream per batch.
``--capture-substeps 2`` records a repeating two-substep window and replays it
four times for the default eight-substep control step. The window must preserve
both state-buffer bindings and the solver's state-synchronization cadence.
``--setup-mode direct`` captures the fresh batch without a warmup or reset.
Both modes retain post-capture replay, reset, finiteness and independence checks;
``capture_ready_ms`` excludes those diagnostic checks and the first physics step.

This diagnostic uses Warp 1.17's private native graph-instantiation entry point
to measure recording separately from instantiation/upload and first replay.
The update experiment additionally requires the optional ``cuda-python`` package.
These low-level handles and ownership conventions are version-dependent.

Example:
    python scripts/benchmark_model_rebuild.py --prototype keyboard12.usda \
        --prototype keyboard108.usda --world-counts 128 512 256 --output rebuild.json
"""

# Timed callbacks execute immediately, before their enclosing loop advances.
# ruff: noqa: B023

from __future__ import annotations

import argparse
import cProfile
import ctypes
import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo


class CaptureScratchPool:
    """Own persistent scratch blocks, recycling equal-size temporary arrays.

    CUDA malloc is permitted during relaxed capture but is not recorded. Keeping
    every backing allocation alive with the graph makes replay independent of the
    allocation call. Descriptor destruction only returns a block to this pool;
    backing storage is released after the graph and its last array owner retire.
    ScopedAllocator changes the device's allocator, so concurrent host allocation
    and multi-stream use are intentionally outside this diagnostic's contract.
    """

    deallocate_requires_context_guard = False

    def __init__(self, device):
        self.device = wp.get_device(device)
        self.stream = wp.get_stream(self.device).cuda_stream
        self.blocks = {}
        self.free = {}
        self.live = {}
        self.backing_bytes = 0
        self.live_bytes = 0
        self.peak_live_bytes = 0
        self.reuses = 0
        # Select cudaMalloc without changing the allocator used by callers.
        with wp.ScopedAllocator(self.device, None), wp.ScopedMempool(self.device, False):
            self.backing = wp.get_device_allocator(self.device)
        self.memory_kind = self.backing.memory_kind

    def allocate(self, size):
        """Lease a block on this pool's single ordered stream."""
        if wp.get_stream(self.device).cuda_stream != self.stream:
            raise RuntimeError("CaptureScratchPool requires a single CUDA stream.")
        available = self.free.get(size)
        if available:
            ptr = available.pop()
            self.reuses += 1
        else:
            ptr = self.backing.allocate(size)
            self.blocks[ptr] = size
            self.backing_bytes += size
        self.live[ptr] = size
        self.live_bytes += size
        self.peak_live_bytes = max(self.peak_live_bytes, self.live_bytes)
        return ptr

    def deallocate(self, ptr, size):
        """Recycle only the descriptor's block; retain the CUDA allocation."""
        if wp.get_stream(self.device).cuda_stream != self.stream:
            raise RuntimeError("CaptureScratchPool requires a single CUDA stream.")
        assert self.live.pop(ptr) == size
        self.live_bytes -= size
        self.free.setdefault(size, []).append(ptr)

    def __del__(self):
        if not hasattr(self, "backing"):
            return
        try:
            with self.device.context_guard:
                for ptr, size in self.blocks.items():
                    self.backing.deallocate(ptr, size)
        except (TypeError, AttributeError):
            # Match Warp's cleanup convention during Python interpreter exit.
            pass


@dataclass
class SimulationBatch:
    """Own all resources used by one exact-sized captured simulation."""

    model: newton.Model
    solver: SolverMuJoCo
    state_0: newton.State
    state_1: newton.State
    control: newton.Control
    substeps: int
    dt: float
    capture_substeps: int | None = None
    graph: wp.Graph | None = None
    scratch_pool: CaptureScratchPool | None = None

    def __post_init__(self):
        self.capture_substeps = self.substeps if self.capture_substeps is None else self.capture_substeps
        if self.substeps < 2 or self.capture_substeps < 2 or self.capture_substeps % 2:
            raise ValueError("Capture substeps must be positive, even, and divide the control-step substeps.")
        if self.substeps % self.capture_substeps:
            raise ValueError("Capture substeps must be positive, even, and divide the control-step substeps.")
        interval = self.solver.update_data_interval
        if interval > 0 and self.capture_substeps % interval:
            raise ValueError("The capture window must preserve the solver update_data_interval cadence.")

    def simulate(self, *, substeps=None):
        """Advance a control step, or its repeating window while recording."""
        scope = wp.ScopedAllocator(self.model.device, self.scratch_pool) if self.scratch_pool else nullcontext()
        with scope:
            for _ in range(self.substeps if substeps is None else substeps):
                self.state_0.clear_forces()
                self.solver.step(self.state_0, self.state_1, self.control, None, self.dt)
                self.state_0, self.state_1 = self.state_1, self.state_0

    def replay(self):
        """Advance one full control step with the recorded window."""
        for _ in range(self.substeps // self.capture_substeps):
            wp.capture_launch(self.graph)

    def reset(self):
        """Restore fresh episodes after warmup and graph validation."""
        self.solver.reset(self.state_0)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)


def main():
    """Profile complete replacements and write stage timings with validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype", type=Path, action="append", required=True)
    parser.add_argument("--world-counts", type=int, nargs="+", default=[128, 512, 256])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--capture-substeps", type=int, help="Even recording window dividing --substeps.")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--nconmax", type=int, default=600, help="Native contact capacity per world.")
    parser.add_argument("--njmax", type=int, default=600, help="Native constraint capacity per world.")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--no-sleeping", action="store_true")
    parser.add_argument("--graph-mode", choices=("fresh", "update"), default="fresh")
    parser.add_argument("--construction-mode", choices=("replicate", "rebuild"), default="replicate")
    parser.add_argument("--scratch-mode", choices=("warp", "pool"), default="warp")
    parser.add_argument("--setup-mode", choices=("warmup", "direct"), default="warmup")
    args = parser.parse_args()
    if min(args.world_counts) < 1 or args.repeats < 1 or args.steps < 1:
        parser.error("World counts, repeats and steps must be positive.")
    if args.nconmax < 1 or args.njmax < 1:
        parser.error("Native contact and constraint capacities must be positive.")
    if args.substeps < 2 or args.substeps % 2:
        parser.error("Substeps must be positive and even to preserve graph state bindings.")
    args.capture_substeps = args.substeps if args.capture_substeps is None else args.capture_substeps
    if args.capture_substeps < 2 or args.capture_substeps % 2 or args.substeps % args.capture_substeps:
        parser.error("Capture substeps must be positive, even, and divide --substeps.")
    wp.init()
    wp.config.enable_backward = False
    device = wp.get_device(args.device)
    if not device.is_cuda:
        parser.error("The MJWarp experiment requires a CUDA device.")
    if args.graph_mode == "update":
        try:
            from cuda.bindings import driver as cuda_driver  # noqa: PLC0415
        except ImportError:
            parser.error("--graph-mode update requires the optional cuda-python package.")
        # cuda-python discards resultInfo when CUDA reports an incompatible
        # update. Call the Linux driver entry point to retain that diagnostic.
        cuda_library = ctypes.CDLL("libcuda.so.1")
        graph_exec_update = cuda_library.cuGraphExecUpdate_v2
        graph_exec_update.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        graph_exec_update.restype = ctypes.c_int
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "device": {
            "alias": device.alias,
            "name": device.name,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "warp_version": wp.__version__,
        "newton_source": newton.__file__,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "imports": [],
        "samples": [],
        "timing_note": (
            "Synchronized wall ms; capture_record, graph_exec_update, graph_instantiate_upload and first_replay "
            "are separate. Update mode records a fresh graph and reuses only a compatible executable."
        ),
    }
    result["config"]["prototype"] = [str(path.resolve()) for path in args.prototype]

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def timed(row, name, operation):
        wp.synchronize_device(device)
        start = time.perf_counter()
        value = operation()
        wp.synchronize_device(device)
        row[name] = 1000 * (time.perf_counter() - start)
        print(f"STAGE {name}: {row[name]:.3f} ms", flush=True)
        return value

    def fresh_builder():
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.gap = 0.01
        builder.default_shape_cfg.margin = 0.0
        return builder

    solver_options = {
        "solver": "newton",
        "integrator": "implicitfast",
        "njmax": args.njmax,
        "nconmax": args.nconmax,
        "nccdmax": 128,
        "iterations": 100,
        "ls_iterations": 15,
        "cone": "pyramidal",
        "update_data_interval": 2,
        "use_mujoco_contacts": True,
        "enable_sleeping": not args.no_sleeping,
    }
    with wp.ScopedDevice(device):
        prototypes = []
        for path in args.prototype:
            prototype = fresh_builder()
            timing = {"path": str(path.resolve())}
            timed(
                timing,
                "import_ms",
                lambda: prototype.add_usd(
                    str(path), root_path="/World", load_visual_shapes=False, enable_self_collisions=False
                ),
            )
            if args.construction_mode == "replicate":

                def prepare_model():
                    builder = fresh_builder()
                    builder.replicate(prototype, 1)
                    return builder.finalize(device)

                model = timed(timing, "prepare_model_ms", prepare_model)
                prototype = timed(timing, "prepare_solver_ms", lambda: SolverMuJoCo(model, **solver_options))
            prototypes.append(prototype)
            result["imports"].append(timing)
        save()

        current = [None] * len(prototypes)
        reference_shapes = {}
        for repeat in range(args.repeats):
            for count in args.world_counts:
                for prototype_id, prototype in enumerate(prototypes):
                    row = {"repeat": repeat, "world_count": count, "prototype": prototype_id, "stages_ms": {}}
                    stages = row["stages_ms"]
                    old = current[prototype_id]
                    old_graph = old.graph if old is not None else None
                    old_arrays = (
                        (old.state_0.joint_q, old.state_1.joint_q, old.solver.mjw_data.qpos, old.solver.mjw_data.qvel)
                        if old is not None
                        else ()
                    )
                    old_values = [array.numpy().copy() for array in old_arrays]
                    profiler = cProfile.Profile() if args.profile else None
                    if profiler is not None:
                        profiler.enable()
                    overall_start = time.perf_counter()
                    if args.construction_mode == "replicate":
                        solver = timed(stages, "clone_model_solver", lambda: prototype.replicate(count))
                        model = solver.model
                    else:
                        builder = timed(stages, "builder", fresh_builder)
                        timed(stages, "replicate", lambda: builder.replicate(prototype, count))
                        # Reading array-backed builder attributes here materializes lists.
                        model = timed(stages, "finalize", lambda: builder.finalize(device))
                        solver = timed(stages, "solver", lambda: SolverMuJoCo(model, **solver_options))
                        del builder

                    def initialize_state():
                        state_0, state_1, control = model.state(), model.state(), model.control()
                        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
                        state_1.assign(state_0)
                        pool = CaptureScratchPool(device) if args.scratch_mode == "pool" else None
                        return SimulationBatch(
                            model, solver, state_0, state_1, control, args.substeps, args.dt,
                            capture_substeps=args.capture_substeps, scratch_pool=pool
                        )  # fmt: skip

                    batch = timed(stages, "state_control", initialize_state)
                    if args.setup_mode == "warmup":
                        timed(stages, "warmup", batch.simulate)
                        timed(stages, "reset", batch.reset)
                    else:
                        stages["warmup"], stages["reset"] = 0.0, 0.0

                    def record():
                        mode = wp.CaptureMode.RELAXED if batch.scratch_pool else wp.CaptureMode.THREAD_LOCAL
                        with wp.ScopedCapture(device=device, capture_mode=mode) as capture:
                            batch.simulate(substeps=batch.capture_substeps)
                        batch.graph = capture.graph
                        # A caller retaining only Graph must also retain its pointers.
                        batch.graph._scratch_pool = batch.scratch_pool

                    timed(stages, "capture_record", record)
                    graph_update = {
                        "attempted": False,
                        "reused_executable": False,
                        "cuda_result": None,
                        "update_result": None,
                    }
                    if args.graph_mode == "update" and old_graph is not None:
                        assert old_graph.graph_exec is not None
                        update_info = cuda_driver.CUgraphExecUpdateResultInfo()
                        update_code = timed(
                            stages,
                            "graph_exec_update",
                            lambda: graph_exec_update(
                                old_graph.graph_exec,
                                batch.graph.graph,
                                update_info.getPtr(),
                            ),
                        )
                        graph_update.update(
                            attempted=True,
                            cuda_result=update_code,
                            update_result=int(update_info.result),
                            update_result_name=str(update_info.result),
                        )
                        if int(update_info.errorNode):
                            node_type = cuda_driver.cuGraphNodeGetType(update_info.errorNode)
                            if int(node_type[0]) == 0:
                                graph_update["error_node_type"] = str(node_type[1])
                        if graph_update["cuda_result"] == 0 and graph_update["update_result"] == 0:
                            # Each Graph destructor owns graph_exec. Move the handle,
                            # retaining modules until its new owner is retired.
                            batch.graph.module_execs.update(old_graph.module_execs)
                            batch.graph.graph_exec, old_graph.graph_exec = old_graph.graph_exec, None
                            graph_update["reused_executable"] = True
                        elif update_code != int(cuda_driver.CUresult.CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE):
                            raise RuntimeError(f"Unexpected CUDA graph update failure: {graph_update}")

                    if batch.graph.graph_exec is None:

                        def instantiate():
                            graph_exec = ctypes.c_void_p()
                            runtime = wp._src.context.runtime
                            if not runtime.core.wp_cuda_graph_create_exec(
                                device.context,
                                wp.get_stream(device).cuda_stream,
                                batch.graph.graph,
                                ctypes.byref(graph_exec),
                            ):
                                raise RuntimeError(runtime.get_error_string())
                            batch.graph.graph_exec = graph_exec

                        timed(stages, "graph_instantiate_upload", instantiate)
                    else:
                        stages["graph_instantiate_upload"] = 0.0
                    row["graph_update"] = graph_update
                    row["capture_ready_ms"] = 1000 * (time.perf_counter() - overall_start)
                    timed(stages, "first_replay", batch.replay)
                    timed(stages, "fresh_episode", batch.reset)
                    row["ready_ms"] = 1000 * (time.perf_counter() - overall_start)
                    if batch.scratch_pool is not None:
                        row["scratch_pool"] = {
                            "backing_bytes": batch.scratch_pool.backing_bytes,
                            "backing_allocations": len(batch.scratch_pool.blocks),
                            "peak_live_bytes": batch.scratch_pool.peak_live_bytes,
                            "live_bytes": batch.scratch_pool.live_bytes,
                            "reuses": batch.scratch_pool.reuses,
                        }
                    if profiler is not None:
                        profiler.disable()
                        profile_path = args.output.with_name(
                            f"{args.output.stem}.{prototype_id}.{count}.{repeat}.pstats"
                        )
                        profiler.dump_stats(profile_path)

                    if args.graph_mode == "update":
                        # Keep graph inspection and file I/O outside ready_ms.
                        _, _, node_count = cuda_driver.cuGraphGetNodes(batch.graph.graph.value)
                        node_status, nodes, _ = cuda_driver.cuGraphGetNodes(batch.graph.graph.value, node_count)
                        if int(node_status) != 0:
                            raise RuntimeError(f"Cannot inspect captured CUDA graph: {node_status}")
                        node_types = {}
                        for node in nodes:
                            node_status, node_type = cuda_driver.cuGraphNodeGetType(node)
                            if int(node_status) != 0:
                                raise RuntimeError(f"Cannot inspect captured CUDA node: {node_status}")
                            name = str(node_type)
                            node_types[name] = node_types.get(name, 0) + 1
                        row["graph_top_level_node_types"] = node_types
                        if batch.scratch_pool is not None:
                            assert not any("MEM_ALLOC" in name or "MEM_FREE" in name for name in node_types)
                        if graph_update["attempted"] and not graph_update["reused_executable"]:
                            stem = f"{args.output.stem}.{prototype_id}.{count}.{repeat}"
                            old_dot = args.output.with_name(f"{stem}.previous.dot")
                            new_dot = args.output.with_name(f"{stem}.replacement.dot")
                            wp.capture_debug_dot_print(old_graph, str(old_dot))
                            wp.capture_debug_dot_print(batch.graph, str(new_dot))
                            graph_update["diagnostic_graphs"] = [str(old_dot), str(new_dot)]

                    row["counts"] = {
                        "bodies": model.body_count,
                        "shapes": model.shape_count,
                        "joints": model.joint_count,
                        "dofs": model.joint_dof_count,
                        "articulations": model.articulation_count,
                    }
                    assert model.world_count == count
                    assert solver.mjw_data.nworld == count
                    per_world = tuple(value // count for value in row["counts"].values())
                    assert all(value % count == 0 for value in row["counts"].values()), row["counts"]
                    if prototype_id in reference_shapes:
                        assert per_world == reference_shapes[prototype_id]
                    else:
                        reference_shapes[prototype_id] = per_world
                    # Validate graph state and independence before replacing the old batch.
                    if old is not None:
                        assert old.state_0.joint_q.ptr != batch.state_0.joint_q.ptr
                        assert old_graph is not batch.graph
                    q_initial = batch.state_0.joint_q.numpy().copy()
                    np.testing.assert_array_equal(q_initial, model.joint_q.numpy())
                    np.testing.assert_array_equal(batch.state_0.joint_qd.numpy(), model.joint_qd.numpy())
                    np.testing.assert_array_equal(batch.state_1.joint_q.numpy(), q_initial)
                    graph_before, executable_before = batch.graph, batch.graph.graph_exec
                    replay = []
                    for _ in range(args.steps):
                        sample = {}
                        timed(sample, "step", batch.replay)
                        replay.append(sample["step"])
                    q_final = batch.state_0.joint_q.numpy()
                    for state in (batch.state_0, batch.state_1):
                        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                            assert np.isfinite(getattr(state, name).numpy()).all(), name
                    assert np.isfinite(solver.mjw_data.qpos.numpy()).all()
                    assert np.isfinite(solver.mjw_data.qvel.numpy()).all()
                    overflow = int(np.bitwise_or.reduce(solver.mjw_data.overflow.numpy(), initial=0))
                    assert overflow == 0, f"Native overflow bitmask: {overflow}. Check capacities and iteration limits."
                    assert batch.graph is graph_before and batch.graph.graph_exec is executable_before
                    del state, graph_before, executable_before
                    for array, expected in zip(old_arrays, old_values, strict=True):
                        np.testing.assert_array_equal(array.numpy(), expected)
                    if old_arrays:
                        del array, expected
                    if old_graph is not None and graph_update["reused_executable"]:
                        assert old_graph.graph_exec is None
                        assert batch.graph.graph_exec is not None
                    row["validation"] = {
                        "finite_state": True,
                        "native_overflow": overflow,
                        "reset_matches_model": True,
                        "old_batch_unchanged": True,
                        "replay_keeps_graph": True,
                        "max_q_change": float(np.max(np.abs(q_final - q_initial), initial=0)),
                    }
                    row["replay_ms"] = {"median": float(np.median(replay)), "min": min(replay), "max": max(replay)}
                    timed(stages, "final_reset", batch.reset)
                    # Replaying the same bundle never records a replacement graph.
                    assert batch.graph is not old_graph
                    current[prototype_id] = batch
                    retirement_start = time.perf_counter()
                    # All preceding validation is synchronized. Destroy the old
                    # graph before dropping its pool; successful updates moved
                    # the executable to the replacement, with fresh pointers.
                    if old is not None:
                        old.graph = None
                    del old_graph
                    del old, old_arrays, old_values, model, solver, batch
                    wp.synchronize_device(device)
                    row["retire_ms"] = 1000 * (time.perf_counter() - retirement_start)
                    result["samples"].append(row)
                    save()
                    print(json.dumps(row), flush=True)
        wp.synchronize_device(device)


if __name__ == "__main__":
    main()
