# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Root-owned saved-operator feasibility, not task timing or qualification.

The hybrid currently has known CPU physical failures. This bounded probe tests
the actual native ABI and includes identical output restoration in each timing.
It does not extrapolate a 512-world replay into a whole-environment speedup.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import coupled_contact
from newton._src.solvers.feather_pgs import solver_feather_pgs as original

NATIVE_SHA = "3e972df7b7e2659716718595a49164a3a7ed9f94c2d874c0e29855dca3384015"
HELPER = Path(
    "/home/octi/Projects/newton-fpgs-anymal-contiguous-response-20260914/tools/fpgs_bench/test_branch_response.py"
)
HELPER_SHA = "461563b3252bf42ffacaccbb66c479ea19d7504b57ad7e9f97fe1840f2d217a1"
WRITTEN = ("v_out", "world_impulses", "world_row_type", "world_row_parent", "world_row_mu")


def bind(arrays, scalar, tier, candidate, device, candidate_module=coupled_contact):
    """Bind exact captured strides and keep every iteration independently cold."""
    factory = original._get_pgs_solve_parallel_kernel
    if candidate:
        factory = candidate_module.get_parallel_factory(factory)
    kernel = factory(
        arrays["world_impulses"].shape[1],
        arrays["mf_impulses"].shape[1],
        18,
        wp.get_device(device).arch,
        rows=tier,
        min_rows=0 if tier == 32 else 32,
        sweeps=24,
        matrix_free=True,
        inkernel_response=(18, 0, 0, 0),
        exact_row_sums=True,
        world_rows=True,
    )
    marker = f"_fpgs_{candidate_module.__name__.rsplit('.', 1)[-1]}"
    assert bool(getattr(kernel, marker, False)) == candidate
    buffers, arguments = {}, []
    for argument in kernel.adj.args:
        name = argument.label
        if name in arrays:
            buffers[name] = wp.array(arrays[name], dtype=argument.type.dtype, device=device)
            arguments.append(buffers[name])
        else:
            arguments.append(scalar[name])
    seeds = {name: wp.clone(buffers[name]) for name in WRITTEN}

    def launch():
        for name in WRITTEN:
            wp.copy(buffers[name], seeds[name])
        wp.launch_tiled(
            kernel,
            dim=[len(arrays["world_constraint_count"])],
            inputs=arguments,
            block_dim=kernel._fpgs_block_dim,
            device=device,
        )

    launch()
    output = {name: buffers[name].numpy() for name in WRITTEN}
    for name, value in buffers.items():
        if name not in WRITTEN:
            np.testing.assert_array_equal(value.numpy(), arrays[name], err_msg=name)
    with wp.ScopedCapture(device=device) as capture:
        for _ in range(32):
            launch()
    # A captured graph stores device pointers, not Python ownership of these
    # allocations. Keep both live buffers and immutable reset seeds alive.
    return output, capture.graph, (buffers, seeds, arguments, kernel)


def timed(graph, device):
    """Measure one restored-operator graph batch, including all reset copies."""
    start, end = wp.Event(device=device, enable_timing=True), wp.Event(device=device, enable_timing=True)
    wp.record_event(start)
    for _ in range(16):
        wp.capture_launch(graph)
    wp.record_event(end)
    return wp.get_event_elapsed_time(start, end) / (16 * 32)


def main():
    """Report old/new native physics and balanced replay costs without promotion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-source", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--candidate",
        choices=("coupled_contact", "coupled_jacobi", "spectral_contact", "spectral_jacobi"),
        default="coupled_contact",
    )
    parser.add_argument("--candidate-sha", help="Required frozen source SHA for a new native candidate")
    parser.add_argument("--register-whitening", action="store_true", help="Retain the accepted register row producer")
    args = parser.parse_args()
    if args.candidate != "coupled_contact" and args.candidate_sha is None:
        parser.error("A new native candidate requires its explicit frozen --candidate-sha")
    candidate_module = importlib.import_module(f"newton._src.solvers.feather_pgs.{args.candidate}")
    native_sha = args.candidate_sha or NATIVE_SHA
    source_path = Path(candidate_module.__file__)
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == native_sha
    for name, digest in getattr(candidate_module, "SOURCE_PINS", {}).items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256(Path(coupled_contact.__file__).read_bytes()).hexdigest() == NATIVE_SHA
    assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA
    spec = importlib.util.spec_from_file_location("coupled_existing_physics", HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    rows, physical = helper.reference("rows"), helper.reference("physical")
    device = "cuda:0"
    assert wp.get_device(device).is_cuda
    for name in ("_INK_CHECK", "_WR_CHECK", "_WR_WARM", "_REGISTER_WHITENING", "_SHADOW_LEAN"):
        setattr(original, name, False)
    original._REGISTER_WHITENING = args.register_whitening
    cpu_control = None
    if args.candidate == "spectral_contact":
        cpu_control = importlib.import_module("tools.fpgs_bench.spectral_gs_control")
        assert hashlib.sha256(Path(cpu_control.__file__).read_bytes()).hexdigest() == (
            "a39eb83e4f5b240a24416f96683df0cc5e7453c8f180cbf539f9d752141256c7"
        )
    elif args.candidate == "spectral_jacobi":
        cpu_control = importlib.import_module("tools.fpgs_bench.spectral_jacobi_control")
        assert hashlib.sha256(Path(cpu_control.__file__).read_bytes()).hexdigest() == (
            "f3cb2e5e15de8ff7d5b876bbf966d69139c134226a2338d15072ffa998385a70"
        )
    candidate_module.get_parallel_factory.cache_clear()
    print(
        "SCOPE",
        json.dumps(
            {
                "known_unqualified": True,
                "candidate": args.candidate,
                "native_sha": native_sha,
                "device": wp.get_device(device).name,
                "source_gpu": args.gpu_source,
                "worlds": 512,
                "includes_output_restore": True,
                "register_whitening": args.register_whitening,
            }
        ),
        flush=True,
    )
    for step in (0, 1):
        for tier in (32, 48):
            arrays, saved, scalar, owned, pins = rows.load(args.gpu_source, step, tier)
            outputs, graphs, leases = zip(
                *(bind(arrays, scalar, tier, candidate, device, candidate_module) for candidate in (False, True)),
                strict=True,
            )
            failures, hard_failures, translation_controls = [], [], []
            count = arrays["world_constraint_count"]
            inactive = (~np.isin(np.arange(len(count)), owned)[:, None]) | (
                np.arange(arrays["world_impulses"].shape[1])[None, :] >= count[:, None]
            )
            np.testing.assert_array_equal(outputs[1]["world_impulses"][inactive], arrays["world_impulses"][inactive])
            unowned_dofs = arrays["world_dof_indices"][~np.isin(np.arange(len(count)), owned)].reshape(-1)
            unowned_dofs = unowned_dofs[unowned_dofs >= 0]
            np.testing.assert_array_equal(outputs[1]["v_out"][unowned_dofs], arrays["v_out"][unowned_dofs])
            for name in WRITTEN[2:]:
                np.testing.assert_array_equal(outputs[0][name], outputs[1][name], err_msg=name)
            for world in owned:
                problem = rows.world(arrays, saved, scalar, int(world))
                n, ids = len(problem["J"]), arrays["world_dof_indices"][world]
                scores = [
                    physical.residuals(problem, out["world_impulses"][world, :n], out["v_out"][ids]) for out in outputs
                ]
                old, new = scores
                impulse = outputs[1]["world_impulses"][world, :n]
                backward = helper.momentum_backward(problem, impulse, outputs[1]["v_out"][ids])["backward_scaled"]
                if (
                    not all(np.isfinite(value) for value in new.values())
                    or max(new["momentum_scaled"], backward, new["disk_excess"]) > 3e-5
                    or new["negative_normal_impulse"] > 1e-7
                ):
                    hard_failures.append({"world": int(world), "score": new, "backward": backward})
                failed = [
                    name
                    for name in ("natural_residual_scaled", "normal_negative_velocity", "complementarity", "mdp_gap")
                    if new[name] > old[name] + 3e-5 * max(1.0, old[name])
                ]
                if failed:
                    failures.append({"world": int(world), "failed": failed, "old": old, "new": new})
                if cpu_control is not None and world == owned[0]:
                    # One predeclared first-owned world per partition checks
                    # translation against the frozen double CPU map. Preserve
                    # FP32 stopping/association differences as diagnostics.
                    expected = cpu_control.solve(
                        problem["J"],
                        problem["L"],
                        np.sum(problem["Z"] ** 2, axis=1) + problem["cfm"],
                        problem["bias"],
                        problem["kind"],
                        problem["parent"],
                        problem["mu"],
                        problem["predictor"],
                        iterations=24,
                    )
                    kinetic_reference = problem["L"].T @ expected.velocity
                    kinetic_error = problem["L"].T @ (outputs[1]["v_out"][ids] - expected.velocity)
                    translation_controls.append(
                        {
                            "world": int(world),
                            "cpu_sweeps": expected.work["sweeps"],
                            "cpu_physical_stop": expected.work["physical_stop"],
                            "held_H_velocity_error_scaled": float(
                                np.linalg.norm(kinetic_error) / (1 + np.linalg.norm(kinetic_reference))
                            ),
                            "impulse_error_inf": float(np.max(np.abs(impulse - expected.impulses))),
                            "cpu_physical": physical.residuals(problem, expected.impulses, expected.velocity),
                            "native_physical": new,
                        }
                    )
            samples = [[], []]
            for graph in graphs:
                wp.capture_launch(graph)
            for round_ in range(6):
                for which in (0, 1) if round_ % 2 == 0 else (1, 0):
                    samples[which].append(timed(graphs[which], device))
            medians = [float(np.median(sample)) for sample in samples]
            print(
                "RESULT",
                json.dumps(
                    {
                        "step": step,
                        "tier": tier,
                        "cases": len(owned),
                        "failures": failures,
                        "hard_failures": hard_failures,
                        "translation_controls": translation_controls,
                        "samples_ms": samples,
                        "median_ms": medians,
                        "replay_speedup": medians[0] / medians[1],
                        "pins": pins,
                    }
                ),
                flush=True,
            )
            assert not hard_failures, "Native hard physical failure; no further timing funded"
            assert len(leases) == 2
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == native_sha
    for name, digest in getattr(candidate_module, "SOURCE_PINS", {}).items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256(Path(coupled_contact.__file__).read_bytes()).hexdigest() == NATIVE_SHA
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
