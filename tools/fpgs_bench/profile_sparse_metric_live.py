# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Append an unprivileged same-input solve diagnosis to an untimed boundary."""

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path("/tmp/fpgs-g1-pair-terrain-checked-lDHRWzAJ/checked_sparse.py")
BASE_SHA = "71521991361ca3d4c8b42d10d96b4098609d5253557ec7df46923c3a886ecf34"


def digest(array):
    """Hash exact array bytes, including inactive storage, outside timing."""
    return hashlib.sha256(np.ascontiguousarray(array.numpy()).view(np.uint8)).hexdigest()


def exact(actual, expected, label):
    """Require instrumentation to leave all numerical output bytes unchanged."""
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual).view(np.uint8),
        np.ascontiguousarray(expected).view(np.uint8),
        err_msg=label,
    )


def profile(solver):
    """Replay original last-solved inputs without writing live simulation arrays."""
    import warp as wp  # noqa: PLC0415
    from profile_sparse_metric_clock import build_clock_kernel  # noqa: PLC0415

    from newton._src.solvers.feather_pgs.sparse_factor import SparseData  # noqa: PLC0415

    owner = solver._sparse_factor
    if (
        solver.pgs_warmstart
        or solver.pgs_velocity_iterations != 0
        or solver.pgs_omega != 1.0
        or solver.pgs_iterations != 8
        or owner.packet_rows
        or owner.block_contacts
        or not owner.metric_tangents
        or getattr(owner, "present_ports", False)
    ):
        raise RuntimeError("Require the retained cold eight-sweep metric owner")
    if solver.world_count != 16384 or solver.dense_max_constraints != 100:
        raise RuntimeError("Require the fixed 16K/100-capacity live recipe")
    if solver.model.device.is_capturing:
        raise RuntimeError("Diagnostics must remain outside simulation capture")
    output = Path(os.environ["FPGS_CLOCK_OUTPUT"])
    if output.exists() or output.suffix != ".npz":
        raise RuntimeError("Require a fresh explicit .npz diagnostic output")
    kernel, metadata = build_clock_kernel()
    original = owner.kernels.solve
    if original.key != "sparse_metric_tangent43_s18_c100":
        raise RuntimeError("Unexpected retained solve owner")
    fields = tuple(metadata["fields"])
    device = solver.model.device
    live = {
        "W": owner.data.W,
        "Z": owner.data.Z,
        "support": owner.data.support,
        "incident": owner.data.incident,
        "valid": owner.data.valid,
        "status": owner.data.status,
        "counts": solver.constraint_count,
        "rhs": solver.rhs,
        "diag": solver.diag,
        "type": solver.row_type,
        "parent": solver.row_parent,
        "mu": solver.row_mu,
        "vhat": solver.v_hat,
        "vout": solver.v_out,
        "impulses": solver.impulses,
    }
    before = {name: digest(array) for name, array in live.items()}
    counts = solver.constraint_count.numpy()
    plans = {
        name: digest(getattr(owner.plan, name))
        for name in owner.plan._cls.vars
        if hasattr(getattr(owner.plan, name), "numpy")
    }
    arms = []
    stream = wp.Stream(device)
    with wp.ScopedStream(stream):
        for traced, selected in enumerate((original, kernel)):
            data = SparseData()
            for name in ("W", "Z", "support", "incident", "valid"):
                setattr(data, name, getattr(owner.data, name))
            data.status = wp.clone(owner.data.status)
            impulses = wp.zeros_like(solver.impulses)
            vout = wp.zeros_like(solver.v_out)
            diagnostic = wp.zeros((solver.world_count, len(fields)), dtype=wp.uint64, device=device)
            inputs = [
                owner.plan,
                data,
                solver.constraint_count,
                solver.rhs,
                solver.diag,
                impulses,
                solver.row_type,
                solver.row_parent,
                solver.row_mu,
                8,
                1.0,
                int(solver._contact_friction_start_iteration(8)),
                solver.v_hat,
                vout,
            ]
            if traced:
                inputs.extend((diagnostic, 64))

            def execute(selected=selected, inputs=inputs):
                wp.launch_tiled(selected, dim=[solver.world_count], inputs=inputs, block_dim=32, device=device)

            def reset(impulses=impulses, vout=vout, data=data, diagnostic=diagnostic):
                impulses.zero_()
                vout.zero_()
                data.status.zero_()
                diagnostic.zero_()

            execute()
            expected = (impulses.numpy(), vout.numpy(), data.status.numpy())
            if np.any(expected[2]) or not np.isfinite(expected[0]).all() or not np.isfinite(expected[1]).all():
                raise RuntimeError("Nonfinite or failed replay")
            if traced:
                for name, actual, reference in zip(
                    ("impulses", "vout", "status"), expected, arms[0]["expected"], strict=True
                ):
                    exact(actual, reference, "eager:" + name)
            reset()
            wp.synchronize_stream(stream)
            with wp.ScopedCapture(device=device, stream=stream) as capture:
                execute()
            reset()
            wp.capture_launch(capture.graph, stream=stream)
            for name, array, reference in zip(
                ("impulses", "vout", "status"), (impulses, vout, data.status), expected, strict=True
            ):
                exact(array.numpy(), reference, "graph:" + name)
            arms.append(
                {
                    "graph": capture.graph,
                    "reset": reset,
                    "expected": expected,
                    "outputs": (impulses, vout, data.status),
                    "diagnostic": diagnostic,
                    "samples_ms": [],
                    "kernel_key": selected.key,
                }
            )
        events = (wp.Event(device, enable_timing=True), wp.Event(device, enable_timing=True))
        for turn in range(30):
            for index in (0, 1) if turn % 2 == 0 else (1, 0):
                arm = arms[index]
                arm["reset"]()
                wp.record_event(events[0], stream)
                wp.capture_launch(arm["graph"], stream=stream)
                wp.record_event(events[1], stream)
                wp.synchronize_event(events[1])
                if turn >= 10:
                    arm["samples_ms"].append(float(wp.get_event_elapsed_time(*events)))
        for arm in arms:
            for name, array, reference in zip(
                ("impulses", "vout", "status"), arm["outputs"], arm["expected"], strict=True
            ):
                exact(array.numpy(), reference, "last:" + name)
        raw = arms[1]["diagnostic"].numpy()
    after = {name: digest(array) for name, array in live.items()}
    if before != after or plans != {name: digest(getattr(owner.plan, name)) for name in plans}:
        raise RuntimeError("Replay modified live input, output or plan")
    sampled = np.arange(solver.world_count) % 64 == 0
    values = {name: raw[sampled, i] for i, name in enumerate(fields)}
    if np.any(raw[~sampled]) or not np.array_equal(values["rows"], counts[sampled]):
        raise RuntimeError("Unexpected sampled-world publication")
    if not np.array_equal(values["total_cycles"], raw[sampled, 1:7].sum(axis=1)):
        raise RuntimeError("Phase cycles do not conserve total")
    if (
        np.any(values["total_cycles"] == 0)
        or np.any(values["sweeps"] > 8)
        or np.any(values["root_probes"] > 16 * values["sliding_roots"])
        or np.any(values["self_block_builds"] > values["positive_radius"])
        or np.any(values["positive_radius"] > values["metric_visits"])
        or not np.array_equal(values["metric_accepted"] + values["metric_rejected"], values["metric_visits"])
        or not np.array_equal(
            values["scalar_row_visits"] + 3 * values["metric_accepted"], values["rows"] * values["sweeps"]
        )
    ):
        raise RuntimeError("Algorithm-count conservation failed")
    np.savez(output, diagnostic=raw, counts=counts, fields=np.asarray(fields))
    return {
        "scope": "Last-solved full16K standalone replay; not whole-physics timing or hardware stall counters",
        "metadata": {
            k: v for k, v in metadata.items() if k not in ("original_kernel", "original_native", "instrumented_native")
        },
        "sample_stride": 64,
        "rounds": 30,
        "discard": 10,
        "sampled_worlds": int(sampled.sum()),
        "field_totals": {name: int(value.sum()) for name, value in values.items()},
        "phase_fraction_of_sampled_elapsed_cycles_not_wall": {
            name: float(values[name].sum() / values["total_cycles"].sum()) for name in fields[1:7]
        },
        "arms": [
            {"key": a["kernel_key"], "samples_ms": a["samples_ms"], "median_ms": float(np.median(a["samples_ms"]))}
            for a in arms
        ],
        "instrumented_over_original": float(np.median(arms[1]["samples_ms"]) / np.median(arms[0]["samples_ms"])),
        "all_outputs_byte_equal": True,
        "live_arrays_unchanged": True,
        "live_input_sha256": before,
        "plan_sha256": plans,
        "raw_output": str(output),
        "raw_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "check_pass": True,
    }


def main():
    """Retain the original checked CLI and all existing boundary guards."""
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA:
        raise RuntimeError("Changed original checked observer")
    spec = importlib.util.spec_from_file_location("_pinned_metric_profile_base", BASE)
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    original = base.snapshot
    completed = False

    def observed(solver, requested):
        nonlocal completed
        result = original(solver, requested)
        if requested and not completed:
            result["unprivileged_metric_profile"] = profile(solver)
            completed = True
        return result

    base.snapshot = observed
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
