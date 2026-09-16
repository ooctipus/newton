# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read owner keys and layouts only at the existing two untimed metadata boundaries."""

import hashlib
import importlib.util
import os
from pathlib import Path

ORIGINAL = Path("/home/octi/Projects/newton-fpgs-structural-bench-20260913/tools/fpgs_bench/checked_capture.py")
PIN = "42b289bd0082d194180d5981651a108d11a66058ff4e67649ad7ddb408c9bea6"


def snapshot(solver):
    """Validate actual factories, private cache shape, public factor shape and sticky status."""
    from newton._src.solvers.feather_pgs import franka_kinetic_state as retained  # noqa: PLC0415

    mode = os.environ.get("FEATHER_PGS_WORLD_LANE_STATE")
    if mode not in ("0", "1") or os.environ.get("FEATHER_PGS_FRANKA_KINETIC_STATE") != "1":
        raise RuntimeError("Require explicit world-lane mode and retained kinetic ownership")
    requested = mode == "1"
    owner = getattr(solver, "_franka_kinetic_state", None)
    world_lane = getattr(solver, "_world_lane_state", None)
    worlds = int(solver.world_count)
    if owner is None or owner.solver is not solver:
        raise RuntimeError("Missing actual kinetic owner")
    observed = world_lane is not None
    if observed != requested or (observed and world_lane is not owner):
        raise RuntimeError("Unexpected world-lane fallback or ownership alias")
    if requested:
        from newton._src.solvers.feather_pgs import world_lane_state  # noqa: PLC0415

        if type(owner) is not world_lane_state.WorldLaneState or owner.world_lane_state is not True:
            raise RuntimeError("Wrong world-lane implementation")
        expected = {
            "repair": world_lane_state.get_state_kernel(owner.schedule),
            "finish_held": world_lane_state.get_state_kernel(owner.schedule, finish=True, refresh=False),
            "finish_refresh": world_lane_state.get_state_kernel(owner.schedule, finish=True, refresh=True),
            "predictor": world_lane_state.get_predictor_kernel(owner.schedule),
        }
        actual = {
            "repair": owner.repair_kernel,
            "finish_held": owner.finish_held_kernel,
            "finish_refresh": owner.finish_refresh_kernel,
            "predictor": owner.predictor_kernel,
        }
        shapes = {"bias": (9, worlds), "com_offset": (13, worlds), "geometric": (45, worlds)}
        plan_shape, block_dim = (32, worlds), 128
    else:
        if type(owner) is not retained.FrankaKineticState:
            raise RuntimeError("Baseline is not the retained p16 owner")
        expected = {
            "repair": retained.get_state_kernel(False),
            "finish": retained.get_state_kernel(True),
            "predictor": retained.get_predictor_kernel(),
        }
        actual = {"repair": owner.repair_kernel, "finish": owner.finish_kernel, "predictor": owner.predictor_kernel}
        shapes = {"bias": (worlds, 9), "com_offset": (worlds, 13), "geometric": (worlds, 81)}
        plan_shape, block_dim = (worlds, 32), 32
    if any(actual[name] is not kernel for name, kernel in expected.items()):
        raise RuntimeError("Actual kinetic factory differs from the requested factory")
    if owner.plan.body_ids.shape != plan_shape:
        raise RuntimeError("Wrong private body-plan layout")
    if any(getattr(owner.data, name).shape != shape for name, shape in shapes.items()):
        raise RuntimeError("Wrong kinetic private-cache layout")
    if owner.data.current_valid.ptr != solver._fk_id_cache_valid.ptr:
        raise RuntimeError("Kinetic current validity lost canonical ownership")
    if any(solver.L_by_size[size].shape != (worlds, size, size) for size in (9, 6)):
        raise RuntimeError("Canonical factor layout changed")
    if owner.status.device.is_capturing:
        raise RuntimeError("Owner inspection must remain outside capture")
    owner.check()
    status = owner.status.numpy()
    current = owner.data.current_valid.numpy()
    geometry = owner.data.geometry_valid.numpy()
    if any(value not in (0, 1) for value in current) or any(value not in (0, 1) for value in geometry):
        raise RuntimeError("Invalid kinetic cache-validity values")
    return {
        "check_pass": True,
        "requested": requested,
        "observed": observed,
        "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Untimed owner/layout/status observation; not trajectory or convergence proof",
        "world_count": worlds,
        "keys": {name: kernel.key for name, kernel in actual.items()},
        "declared_factory_block_dim": block_dim,
        "private_shapes": {name: list(shape) for name, shape in shapes.items()},
        "plan_body_shape": list(plan_shape),
        "factor_shapes": {str(size): list(solver.L_by_size[size].shape) for size in (9, 6)},
        "status_nonzero": int((status != 0).sum()),
        "current_valid_count": int((current == 1).sum()),
        "geometry_valid_count": int((geometry == 1).sum()),
    }


def main():
    """Extend the original boundary hook without touching timed kernels or calls."""
    if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != PIN:
        raise RuntimeError("The retained checked capture has changed")
    spec = importlib.util.spec_from_file_location("retained_world_lane_checked", ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.install_boundary_check

    def install(harness, report, save, get_manager, *args, **kwargs):
        original(harness, report, save, get_manager, *args, **kwargs)
        preceding = harness._model_meta

        def observed(physics):
            metadata = preceding(physics)
            entry = report["boundaries"][-1]
            try:
                if physics != "feather_pgs":
                    raise RuntimeError("The world-lane comparison is FPGS-only")
                entry["world_lane_state"] = snapshot(get_manager()._solver)
                return metadata
            except BaseException as error:
                entry.update(check_pass=False, error=repr(error))
                raise
            finally:
                save()

        harness._model_meta = observed

    module.install_boundary_check = install
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
