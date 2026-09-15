"""Diagnose retained versus paired solve on one actual last-solved boundary."""

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path("/tmp/fpgs-g1-paired-solve-checked-G8fkEVEo/checked_sparse.py")
BASE_SHA = "9200384855af4c94a80ea4391430c11c44e26099592128d98d44972f7c2e3f40"
PARENT = Path(
    "/home/octi/Projects/newton-fpgs-g1-unprivileged-profile-20260915/tools/fpgs_bench/profile_sparse_metric_live.py"
)
PARENT_SHA = "7a84f1b0cef3730fc36e6923ca843169d6e386d60501e5382d30d20d37820ed1"
SOURCE_PINS = {
    "sparse_factor": "aadbff86ef0b3065673d0701db0b5ae3a62bda52a7b1c3398f50189958eb9403",
    "sparse_factor_rows": "ec1303a009bab879b67e11d6521173f9441c3aee9e6e77da05c068c33677564b",
    "sparse_paired_worlds": "95fc2ec99884d7186ac36985520c2451f73b2fa2abb1161aef0732c2802997db",
}


def digest(array):
    """Hash all exact bytes outside event timing, including inactive storage."""
    return hashlib.sha256(np.ascontiguousarray(array.numpy()).view(np.uint8)).hexdigest()


def exact(actual, expected, label):
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual).view(np.uint8),
        np.ascontiguousarray(expected).view(np.uint8),
        err_msg=label,
    )


def difference(actual, reference):
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    tolerance = 3e-5 + 3e-4 * np.abs(reference.astype(np.float64))
    return {
        "max_abs": float(np.max(delta, initial=0)),
        "max_scaled_by_one_plus_reference": float(np.max(delta / (1 + np.abs(reference)), initial=0)),
        "max_tolerance_ratio": float(np.max(delta / tolerance, initial=0)),
        "outside_tolerance": int(np.count_nonzero(delta > tolerance)),
        "allclose": bool(np.allclose(actual, reference, rtol=3e-4, atol=3e-5)),
        "rtol": 3e-4,
        "atol": 3e-5,
    }


def profile(solver):
    """Keep the original fourteen arguments; privatize every written array."""
    import warp as wp  # noqa: PLC0415

    from newton._src.solvers.feather_pgs import sparse_factor, sparse_factor_rows, sparse_paired_worlds  # noqa: PLC0415

    modules = (sparse_factor, sparse_factor_rows, sparse_paired_worlds)
    source_files = {module.__name__.rsplit(".", 1)[-1]: Path(module.__file__) for module in modules}
    sources = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_files.items()}
    if sources != SOURCE_PINS:
        raise RuntimeError("Changed replay runtime source")
    owner = solver._sparse_factor
    if (
        os.environ.get("FEATHER_PGS_SPARSE_PAIRED_WORLDS") != "0"
        or owner.paired_worlds
        or not sparse_paired_worlds.supported(owner)
        or solver._mf_warmstart_enabled
        or solver.pgs_warmstart
        or solver.pgs_velocity_iterations != 0
        or solver.pgs_omega != 1.0
        or solver.pgs_iterations != 8
    ):
        raise RuntimeError("Require admitted retained cold eight-sweep owner with paired flag0")
    if solver.world_count != 16384 or solver.dense_max_constraints != 100:
        raise RuntimeError("Require the fixed16K/100-capacity recipe")
    if solver.model.device.is_capturing:
        raise RuntimeError("Replay must remain outside simulation capture")
    output = Path(os.environ["FPGS_PAIRED_REPLAY_OUTPUT"])
    if output.exists() or output.suffix != ".npz":
        raise RuntimeError("Require a fresh explicit .npz output")
    original = owner.kernels.solve
    candidate = sparse_paired_worlds.get_solve_kernel()
    if original.key != "sparse_metric_tangent43_s18_c100" or candidate.key != "sparse_metric_paired43_s18_c100_w16":
        raise RuntimeError("Unexpected actual solve owners")
    device = solver.model.device
    live = {
        **{name: getattr(owner.data, name) for name in ("W", "Z", "support", "incident", "valid", "status")},
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
        for index, selected in enumerate((original, candidate)):
            data = sparse_factor.SparseData()
            for name in ("W", "Z", "support", "incident", "valid"):
                setattr(data, name, getattr(owner.data, name))
            data.status = wp.clone(owner.data.status)
            impulses, vout = wp.zeros_like(solver.impulses), wp.zeros_like(solver.v_out)
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
            dim = solver.world_count if index == 0 else (solver.world_count + 15) // 16
            block = 32 if index == 0 else 256

            def execute(selected=selected, inputs=inputs, dim=dim, block=block):
                wp.launch_tiled(selected, dim=[dim], inputs=inputs, block_dim=block, device=device)

            def reset(impulses=impulses, vout=vout, data=data):
                impulses.zero_()
                vout.zero_()
                data.status.zero_()

            execute()
            expected = (impulses.numpy(), vout.numpy(), data.status.numpy())
            if np.any(expected[2]) or not np.isfinite(expected[0]).all() or not np.isfinite(expected[1]).all():
                raise RuntimeError("Nonfinite or failed replay")
            if index:
                exact(expected[2], arms[0]["expected"][2], "cross-kernel status")
            reset()
            wp.synchronize_stream(stream)
            with wp.ScopedCapture(device=device, stream=stream) as capture:
                execute()
            reset()
            wp.capture_launch(capture.graph, stream=stream)
            for name, array, reference in zip(
                ("impulses", "vout", "status"),
                (impulses, vout, data.status),
                expected,
                strict=True,
            ):
                exact(array.numpy(), reference, "same-kernel graph:" + name)
            arms.append(
                {
                    "graph": capture.graph,
                    "reset": reset,
                    "expected": expected,
                    "outputs": (impulses, vout, data.status),
                    "data": data,
                    "inputs": inputs,
                    "samples_ms": [],
                    "kernel_key": selected.key,
                    "dim": dim,
                    "block_dim": block,
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
                ("impulses", "vout", "status"),
                arm["outputs"],
                arm["expected"],
                strict=True,
            ):
                exact(array.numpy(), reference, "same-kernel last:" + name)
    after = {name: digest(array) for name, array in live.items()}
    if before != after or plans != {name: digest(getattr(owner.plan, name)) for name in plans}:
        raise RuntimeError("Replay modified live input, output or plan")
    if sources != {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_files.items()}:
        raise RuntimeError("Runtime source changed during replay")
    comparisons = {
        name: difference(arms[1]["expected"][index], arms[0]["expected"][index])
        for index, name in enumerate(("impulses", "vout"))
    }
    np.savez(
        output,
        counts=counts,
        original_impulses=arms[0]["expected"][0],
        original_vout=arms[0]["expected"][1],
        paired_impulses=arms[1]["expected"][0],
        paired_vout=arms[1]["expected"][1],
        original_status=arms[0]["expected"][2],
        paired_status=arms[1]["expected"][2],
        row_type=solver.row_type.numpy(),
        row_parent=solver.row_parent.numpy(),
        mu=solver.row_mu.numpy(),
        diagonal=solver.diag.numpy(),
        support=owner.data.support.numpy(),
        group_to_art=owner.plan.group_to_art.numpy(),
        art_to_world=owner.plan.art_to_world.numpy(),
    )
    medians = [float(np.median(arm["samples_ms"])) for arm in arms]
    return {
        "scope": "LAST-SOLVED full16K cold standalone kernel replay; not next-reset state or whole physics",
        "source_commit": "e2f616478fba08a77c7aa6df9fd01a378afd8db4",
        "source_sha256": sources,
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parent_helper_sha256": PARENT_SHA,
        "checked_adapter_sha256": BASE_SHA,
        "worlds": int(solver.world_count),
        "capacity": int(solver.dense_max_constraints),
        "iterations": 8,
        "omega": 1.0,
        "friction_start": int(solver._contact_friction_start_iteration(8)),
        "rounds": 30,
        "discard": 10,
        "order": "AB then BA alternating; resets before GPU events",
        "arms": [
            {
                "key": arm["kernel_key"],
                "dim": arm["dim"],
                "block_dim": arm["block_dim"],
                "samples_ms": arm["samples_ms"],
                "median_ms": medians[index],
            }
            for index, arm in enumerate(arms)
        ],
        "paired_over_original_time": medians[1] / medians[0],
        "original_over_paired_speedup": medians[0] / medians[1],
        "cross_kernel": comparisons,
        "cross_kernel_status_exact": True,
        "cross_kernel_numerical_pass": all(value["allclose"] for value in comparisons.values()),
        "same_kernel_graph_and_last_byte_equal": True,
        "live_arrays_unchanged": True,
        "live_input_sha256": before,
        "plan_sha256": plans,
        "raw_output": str(output),
        "raw_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "check_pass": True,
    }


def main():
    """Reuse the existing checked CLI and every original boundary guard."""
    if (
        hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA
        or hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA
    ):
        raise RuntimeError("Changed source-pinned checked adapter or diagnostic parent")
    spec = importlib.util.spec_from_file_location("_pinned_paired_profile_base", BASE)
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    original, completed = base.snapshot, False

    def observed(solver, requested):
        nonlocal completed
        result = original(solver, requested)
        if requested and not completed:
            result["paired_same_input_profile"] = profile(solver)
            completed = True
        return result

    base.snapshot = observed
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
