# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check existing sticky flags at the unchanged Lab harness's two metadata boundaries.

The optional MJWarp line-search correction changes the algorithm, never its
iteration/tolerance budgets. Without that option this adds no physics kernels
or work inside the timed windows. A clean result
establishes only that the supported Newton narrow-phase and FPGS row/contact
flags or MJWarp warning/overflow flags were zero; it is not a convergence proof
or a substitute for full collision-buffer demand calibration. No flag is cleared.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import inspect
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType

HARNESS = Path("scripts/benchmarks/fpgs_profile/run_profiled.py")
COMPAT_PATH = Path(__file__).resolve().with_name("mjwarp_linesearch_compat.py")
FPGS_FLAGS = {"dense", "matrix_free", "propagation", "contacts"}
MJ_CONTACT_FLAGS = {"negative_count", "source_contacts", "mjwarp_contacts"}
NARROW_FLAGS = {
    "broad_phase",
    "split_query",
    "gjk",
    "split_gjk",
    "split_manifold",
    "mesh",
    "triangle",
    "mesh_plane",
    "mesh_mesh",
    "sdf_sdf",
    "contacts",
    "reduction_hash_load",
    "reduction_hash_insert",
}


def sha(path: Path) -> str:
    """Hash the exact source bytes used by the capture."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fused_contact_metadata(solver) -> dict:
    """Read current experimental ownership only at an existing untimed boundary."""
    fusion = getattr(solver, "_fused_contact_solve", None)
    active = getattr(solver, "_fused_contact_solve_active", None)
    result = {"configured": fusion is not None, "active": active is not None}
    if active is None:
        return result
    if active is not fusion:
        raise RuntimeError("Current fused contact owner differs from its configured owner")
    arrays = (fusion.owner, solver.constraint_count, fusion.fallback_counts)
    if any(array.shape != (solver.world_count,) or array.device.is_capturing for array in arrays):
        raise RuntimeError("Fused ownership must have one value per world and be read outside capture")
    owners, counts, fallback = (array.numpy().tolist() for array in arrays)
    if (
        any(type(value) is not int or value not in (0, 1) for value in owners)
        or any(type(value) is not int or not 0 <= value <= solver.dense_max_constraints for value in counts)
        or any(type(value) is not int for value in fallback)
        or fallback != [0 if owner else count for owner, count in zip(owners, counts, strict=True)]
    ):
        raise RuntimeError("Invalid current fused/fallback ownership or canonical row counts")
    fused_counts = [count for owner, count in zip(owners, counts, strict=True) if owner]
    fallback_counts = [count for owner, count in zip(owners, counts, strict=True) if not owner]
    result.update(
        scope="Current solver call only; not whole-run admission or a physical-quality check",
        worlds=solver.world_count,
        fused_worlds=len(fused_counts),
        fallback_worlds=len(fallback_counts),
        fused_nonempty_worlds=sum(count > 0 for count in fused_counts),
        fallback_nonempty_worlds=sum(count > 0 for count in fallback_counts),
        fused_rows=sum(fused_counts),
        fallback_rows=sum(fallback_counts),
        fused_row_max=max(fused_counts, default=0),
        fallback_row_max=max(fallback_counts, default=0),
    )
    return result


def joint_world_metadata(solver) -> dict:
    """Check actual current joint-world outputs outside both timed windows."""
    owner = getattr(solver, "_joint_world", None)
    result = {"configured": owner is not None, "whole_run_admission": False}
    if owner is None:
        return result
    if owner.solver is not solver:
        raise RuntimeError("Joint-world owner is bound to another solver")
    worlds, output = solver.world_count, owner.output

    def read(array, size):
        if array.device.is_capturing or array.shape != (size,):
            raise RuntimeError("Read correctly shaped joint-world outputs outside capture")
        values = array.numpy().tolist()
        if any(type(value) is not int for value in values):
            raise RuntimeError("Joint-world outputs must contain integer indices")
        return values

    count = read(output.active_count, 1)[0]
    resolved = read(output.resolved, worlds)
    active = read(output.active_worlds, worlds)
    fallback = read(output.predictor_fallback, worlds)
    invalid = read(owner.buckets.data.invalid, 1)[0]
    if (
        not 0 <= count <= worlds
        or any(value not in (0, 1) for value in resolved + fallback)
        or sorted(active[:count]) != [world for world, zero in enumerate(resolved) if not zero]
        or any(zero and original for zero, original in zip(resolved, fallback, strict=True))
        or invalid != 0
    ):
        raise RuntimeError("Invalid current joint-world partition, predictor fallback or raw prefix")
    result.update(
        scope="Most recent output buffers only; not whole-run admission or a numerical-convergence check",
        worlds=worlds,
        zero_worlds=sum(resolved),
        active_worlds=count,
        original_predictor_worlds=sum(fallback),
        raw_index_storage_bytes=owner.buckets.storage_bytes,
    )
    return result


def early_publication_metadata(solver) -> dict:
    """Read current disjoint publication lists only at the existing untimed boundary."""
    result = {}
    for name in ("franka", "kuka"):
        owner = getattr(solver, "_early_" + name, None)
        entry = result[name] = {"configured": owner is not None, "whole_run_admission": False}
        if owner is None:
            continue
        if owner.solver is not solver:
            raise RuntimeError("Early publication owner is bound to another solver")
        data = owner.data
        arts, worlds = solver.model.articulation_count, solver.world_count

        def read(array, size=None):
            if array.device.is_capturing or len(array.shape) != 1 or (size is not None and array.shape != (size,)):
                raise RuntimeError("Read correctly shaped early publication arrays outside capture")
            values = array.numpy().tolist()
            if any(type(value) is not int for value in values):
                raise RuntimeError("Early publication maps must contain integer indices")
            return values

        counts = read(data.counts, 2)
        mask = read(data.art_mask, arts)
        early, late = read(data.early_arts), read(data.late_arts)
        mapping = read(solver.art_to_world, arts)
        if (
            min(counts) < 0
            or sum(counts) != arts
            or counts[0] > len(early)
            or counts[1] > len(late)
            or any(value not in (0, 1) for value in mask)
            or any(value < -1 or value >= worlds for value in mapping)
        ):
            raise RuntimeError("Invalid early publication counts or ownership mask")
        early, late = early[: counts[0]], late[: counts[1]]
        if sorted(early + late) != list(range(arts)) or set(early) != {i for i, value in enumerate(mask) if value}:
            raise RuntimeError("Early/late publication lists must be disjoint and exhaustive")
        if any(mapping[art] < 0 for art in early):
            raise RuntimeError("Early publication cannot own an unmapped articulation")
        invalid = getattr(data, "invalid_raw", None)
        if invalid is not None and read(invalid, 1) != [0]:
            raise RuntimeError("Early publication observed an invalid raw contact prefix")
        entry.update(
            scope="Most recent cohort buffer only; not sustained admission, timing or numerical quality",
            articulations=arts,
            early_articulations=counts[0],
            late_articulations=counts[1],
            worlds=worlds,
            worlds_with_early_articulations=len({mapping[art] for art in early}),
            disjoint_exhaustive=True,
        )
    return result


def check_solver(physics: str, solver, entry: dict) -> None:
    """Publish flag details before raising; never clear or change solver state."""
    entry["solver_class"] = type(solver).__name__
    if physics == "feather_pgs":
        if type(solver).__name__ != "SolverFeatherPGS":
            raise RuntimeError("Checked FPGS capture resolved an unexpected solver")
        read = getattr(solver, "constraint_capacity_status", None)
        check = getattr(solver, "check_constraint_capacity", None)
        if not callable(read) or not callable(check):
            raise RuntimeError("This FPGS revision lacks the sticky capacity-check API")
        flags = read()
        if set(flags) != FPGS_FLAGS or any(type(value) is not bool for value in flags.values()):
            raise RuntimeError("Unrecognized FPGS capacity-status contract")
        entry["flags"] = flags
        entry["capacities"] = {
            name: int(getattr(solver, name))
            for name in ("dense_max_constraints", "mf_max_constraints", "propagation_max_constraints")
        }
        # Execute the public checker, not merely a shadow interpretation of it.
        check()
        if any(flags.values()):
            raise RuntimeError("FPGS capacity flags were nonzero despite a successful checker")
    elif physics == "newton_mjwarp":
        if type(solver).__name__ != "SolverMuJoCo" or solver.use_mujoco_cpu:
            raise RuntimeError("Checked MJWarp capture requires the GPU SolverMuJoCo path")
        if getattr(solver, "enable_sleeping", False):
            raise RuntimeError("Sleeping resets can clear MJ overflow history; unsupported by boundary-only checking")
        data, model = solver.mjw_data, solver.mjw_model
        if data is None or model is None or not model.opt.warn_overflow:
            raise RuntimeError("MJWarp warning recording must remain available and enabled")
        if getattr(data.overflow.device, "is_capturing", False):
            raise RuntimeError("Read MJWarp status outside graph capture only")
        if data.overflow.shape != (int(data.nworld),):
            raise RuntimeError("Unexpected MJWarp overflow shape")
        values = data.overflow.numpy().tolist()
        if any(type(value) is not int or value < -(2**31) or value >= 2**31 for value in values):
            raise RuntimeError("Unexpected MJWarp int32 overflow values")
        masks = [value & 0xFFFFFFFF for value in values]
        combined = 0
        for value in masks:
            combined |= value
        from mujoco_warp import OverflowType

        names = {int(flag): flag.name for flag in OverflowType}
        entry["flags"] = {
            "mask": combined,
            "worlds_nonzero": sum(value != 0 for value in masks),
            "worlds_by_bit": {
                names.get(1 << bit, f"UNKNOWN_BIT_{bit}"): sum(bool(value & (1 << bit)) for value in masks)
                for bit in range(32)
                if combined & (1 << bit)
            },
        }
        entry["capacities"] = {name: int(getattr(data, name)) for name in ("nworld", "njmax", "naconmax", "njmax_nnz")}
        conversion = entry["external_contact_conversion"] = {"status": "checking", "check_pass": False}
        read = getattr(solver, "contact_capacity_status", None)
        check = getattr(solver, "check_contact_capacity", None)
        if not callable(read) or not callable(check):
            raise RuntimeError("This MJWarp solver revision lacks the external contact capacity-check API")
        native = getattr(solver, "_use_mujoco_contacts", None)
        if type(native) is not bool:
            raise RuntimeError("Unrecognized MJWarp contact ownership contract")
        contact_flags = read()
        if set(contact_flags) != MJ_CONTACT_FLAGS or any(type(value) is not bool for value in contact_flags.values()):
            raise RuntimeError("Unrecognized MJWarp external contact capacity-status contract")
        conversion["flags"] = contact_flags
        conversion["scope"] = "native_mujoco_contacts" if native else "external_newton_prefix"
        check()
        if any(contact_flags.values()):
            raise RuntimeError("External contact capacity flags were nonzero despite a successful checker")
        conversion.update(status="not_applicable" if native else "checked", check_pass=True)
        if combined:
            raise RuntimeError(f"MJWarp sticky warning/overflow flags are nonzero: {entry['flags']}")
    else:
        raise RuntimeError(f"Unsupported checked backend: {physics}")
    entry["check_pass"] = True


def check_collision(physics: str, manager, entry: dict) -> None:
    """Check the actual rigid collision owner, not snapshots of reset-each-call counters."""
    result = entry["collision"] = {"check_pass": False, "status": "checking"}
    try:
        pipeline, solver = manager._collision_pipeline, manager._solver
        if pipeline is None:
            if (
                physics == "newton_mjwarp"
                and solver._use_mujoco_contacts
                and solver.mjw_model.opt.run_collision_detection
            ):
                result.update(
                    check_pass=True, status="not_applicable", reason="MJWarp owns collision; no Newton pipeline"
                )
                return
            raise RuntimeError("Missing Newton collision pipeline on an external-collision backend")
        narrow = pipeline.narrow_phase
        unsupported = {
            "soft_particles": int(pipeline.model.particle_count) != 0,
            "hydroelastic": narrow.hydroelastic_sdf is not None,
            "body_pair_reduction": pipeline._body_pair_reducer is not None
            or pipeline.contact_reduction_config.body_pairs,
            "contact_matching": pipeline._contact_matcher is not None or pipeline.contact_matching != "disabled",
            "post_narrow_sort": pipeline._contact_sorter is not None or pipeline.deterministic,
        }
        result["unsupported_owners"] = [name for name, enabled in unsupported.items() if enabled]
        if result["unsupported_owners"]:
            raise RuntimeError(
                f"Collision owners outside narrow-phase capacity coverage: {result['unsupported_owners']}"
            )
        read, check = getattr(narrow, "buffer_capacity_status", None), getattr(narrow, "check_buffer_capacity", None)
        if not callable(read) or not callable(check):
            raise RuntimeError("This Newton narrow phase lacks the sticky capacity-check API")
        if not narrow.verify_buffers:
            raise RuntimeError("Checked collision requires verify_buffers=True throughout the run")
        flags = read()
        result["flags"] = flags
        if set(flags) != NARROW_FLAGS or any(type(value) is not bool for value in flags.values()):
            raise RuntimeError("Unrecognized narrow-phase capacity-status contract")
        check()
        if any(flags.values()):
            raise RuntimeError("Narrow-phase capacity flags were nonzero despite a successful checker")
        result.update(check_pass=True, status="checked")
    except BaseException as error:
        result.update(check_pass=False, status="failed", error=repr(error))
        raise


def install_boundary_check(harness: ModuleType, report: dict, save, get_manager, compat=None, compact=None) -> None:
    """Wrap only the two existing post-warmup/post-profile metadata observations."""
    original = harness._model_meta

    def checked(physics):
        entry = {"boundary": len(report["boundaries"]), "physics": physics, "check_pass": False}
        report["boundaries"].append(entry)
        report["boundary_count"] = len(report["boundaries"])
        try:
            if report["boundary_count"] > 2:
                raise RuntimeError("Unexpected extra metadata boundary in checked capture")
            metadata = original(physics)
            manager = get_manager()
            if compact is not None:
                entry["allegro_capacity"] = compact.snapshot(manager)
            if compat is not None:
                supported = physics == "newton_mjwarp" and bool(compat.supports_model(manager._solver.mjw_model))
                entry["mjwarp_linesearch_model_supported"] = supported
                if not supported:
                    raise RuntimeError("Requested MJWarp line-search fix does not support the actual model")
            check_solver(physics, manager._solver, entry)
            check_collision(physics, manager, entry)
            if physics == "feather_pgs":
                entry["fused_contact_solve"] = fused_contact_metadata(manager._solver)
                entry["early_publication"] = early_publication_metadata(manager._solver)
                entry["joint_world"] = joint_world_metadata(manager._solver)
            if not isinstance(metadata, dict) or "error" in metadata or metadata.get("state_finite") is not True:
                raise RuntimeError(f"Invalid Lab state/model metadata: {metadata!r}")
            metadata["overflow_check"] = entry
            return metadata
        except BaseException as error:
            entry.update(check_pass=False, error=repr(error))
            raise
        finally:
            save()

    harness._model_meta = checked


def install_broad_phase_limit(pipeline_type, limit: int, report: dict, save):
    """Override only Newton's constructor kwarg, preserving its signature and validation."""
    original = pipeline_type.__init__
    if "broad_phase_output_max" not in inspect.signature(original).parameters:
        raise RuntimeError("Selected Newton lacks the broad-phase output-capacity option")
    records = report["collision_capacity"] = {"requested": limit, "pipelines": []}

    @functools.wraps(original)
    def configured(pipeline, *args, **kwargs):
        entry = {"construction_pass": False}
        records["pipelines"].append(entry)
        try:
            existing = kwargs.get("broad_phase_output_max")
            if existing is not None and existing != limit:
                raise ValueError(f"Conflicting broad_phase_output_max: {existing} versus checked request {limit}")
            kwargs["broad_phase_output_max"] = limit
            original(pipeline, *args, **kwargs)
            entry.update(
                full_input_pairs=len(pipeline.shape_pairs_filtered),
                effective=int(pipeline.shape_pairs_max),
                broad_phase_mode=pipeline.broad_phase_mode,
                construction_pass=True,
            )
            narrow = pipeline.narrow_phase
            entry["narrow_phase"] = {
                name: bool(getattr(narrow, name)) for name in ("split_gjk_mpr", "sparse_gjk_pairs")
            }
            entry["narrow_phase"].update(
                {name: int(getattr(narrow, name)) for name in ("max_candidate_pairs", "block_dim", "total_num_threads")}
            )
        except BaseException as error:
            entry.update(construction_pass=False, error=repr(error))
            raise
        finally:
            save()

    pipeline_type.__init__ = configured
    return original


def install_linesearch_fix(report: dict):
    """Execute the exact guarded helper bytes before any model or graph construction."""
    source = COMPAT_PATH.read_bytes()
    record = report["mjwarp_linesearch_compat"] = {
        "helper_sha256": hashlib.sha256(source).hexdigest(),
        "helper_path": str(COMPAT_PATH),
    }
    module = ModuleType("_newton_checked_mjwarp_linesearch")
    module.__file__ = str(COMPAT_PATH)
    sys.modules[module.__name__] = module
    exec(compile(source, str(COMPAT_PATH), "exec"), module.__dict__)
    record["installation"] = module.install()
    if not isinstance(record["installation"], dict):
        raise RuntimeError("Unexpected MJWarp compatibility installation metadata")
    if any(
        record["installation"].get(name) is not False
        for name in ("installed_package_modified", "gradient_tolerance_changed", "iteration_budget_changed")
    ):
        raise RuntimeError("MJWarp compatibility helper changed a protected budget or installed package")
    json.dumps(record, allow_nan=False)
    return module


def main(argv: list[str] | None = None) -> int:
    """Load the pinned original harness and persist checked success or failure."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--isaaclab", type=Path, required=True)
    parser.add_argument("--newton", type=Path, required=True)
    parser.add_argument("--expected-run-sha256", required=True)
    parser.add_argument("--checks-output", type=Path, required=True)
    parser.add_argument("--broad-phase-output-max", type=int)
    parser.add_argument("--mjwarp-linesearch-fix", action="store_true")
    parser.add_argument("--allegro-compact-capacity", action="store_true")
    options, forwarded = parser.parse_known_args(argv)
    if options.broad_phase_output_max is not None and not 0 < options.broad_phase_output_max < 2**31:
        parser.error("--broad-phase-output-max must be a positive int32 capacity")
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if options.mjwarp_linesearch_fix and (
        forwarded.count("--physics") != 1
        or forwarded.index("--physics") + 1 >= len(forwarded)
        or forwarded[forwarded.index("--physics") + 1] != "newton_mjwarp"
    ):
        parser.error("--mjwarp-linesearch-fix requires the newton_mjwarp backend")
    if "--trace-stats" in forwarded:
        parser.error("--trace-stats adds metadata reads inside timing; incompatible with boundary-only checks")
    if options.allegro_compact_capacity:
        required = {"--physics": "feather_pgs", "--num-envs": "16384", "--task": "Isaac-Reorient-Cube-Allegro"}
        if options.broad_phase_output_max != 524288 or any(
            forwarded.count(key) != 1
            or forwarded.index(key) + 1 >= len(forwarded)
            or forwarded[forwarded.index(key) + 1] != value
            for key, value in required.items()
        ):
            parser.error("Compact capacity requires the exact checked 16K Allegro FPGS recipe and broad cap 524288")
    if forwarded.count("--output") != 1:
        parser.error("Forward exactly one explicit original --output")
    output_index = forwarded.index("--output") + 1
    if output_index == len(forwarded) or forwarded[output_index].startswith("--"):
        parser.error("Original --output requires a path")
    output = Path(forwarded[output_index]).expanduser().resolve()
    checks = options.checks_output.expanduser().resolve()
    lab, newton = options.isaaclab.expanduser().resolve(), options.newton.expanduser().resolve()
    roots = (lab, newton, Path(__file__).resolve().parents[2])
    if output == checks or any(
        path.exists() or any(path.is_relative_to(root) for root in roots) for path in (output, checks)
    ):
        parser.error("Use distinct fresh capture/checks files outside all source trees")
    report = {
        "complete": False,
        "check_pass": False,
        "boundary_count": 0,
        "boundaries": [],
        "scope": "Available sticky flags only; no convergence proof or complete collision-demand calibration",
        "checks_output": str(checks),
        "capture_output": str(output),
        "mjwarp_linesearch_fix": options.mjwarp_linesearch_fix,
        "allegro_compact_capacity": options.allegro_compact_capacity,
        "physics_work_modified": options.mjwarp_linesearch_fix,
        "physics_budgets_modified": False,
    }
    checks.parent.mkdir(parents=True, exist_ok=True)

    def save():
        checks.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    previous_argv = sys.argv
    pipeline_type, original_constructor = None, None
    stack = ExitStack()
    compact = None
    save()
    try:
        path = lab / HARNESS
        source = path.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        report["run_profiled_sha256"] = digest
        report["checked_capture_sha256"] = sha(Path(__file__))
        if digest != options.expected_run_sha256:
            raise RuntimeError("Selected Lab run_profiled.py differs from the parent-pinned source")
        harness = ModuleType("_newton_checked_lab_capture")
        harness.__file__ = str(path)
        exec(compile(source, str(path), "exec"), harness.__dict__)
        if not callable(getattr(harness, "main", None)) or not callable(getattr(harness, "_model_meta", None)):
            raise RuntimeError("Selected Lab harness lacks the required capture boundary")
        runtime_newton = importlib.import_module("newton")
        if Path(runtime_newton.__file__).resolve() != newton / "newton/__init__.py":
            raise RuntimeError("Capture imported the wrong Newton checkout")
        compat = install_linesearch_fix(report) if options.mjwarp_linesearch_fix else None

        def get_manager():
            return importlib.import_module("isaaclab_newton.physics").NewtonManager

        if options.allegro_compact_capacity:
            helper = Path(__file__).resolve().with_name("allegro_capacity.py")
            helper_source = helper.read_bytes()
            compact = ModuleType("_checked_allegro_capacity")
            compact.__file__ = str(helper)
            exec(compile(helper_source, str(helper), "exec"), compact.__dict__)
            info = report["allegro_capacity"] = {"helper_sha256": hashlib.sha256(helper_source).hexdigest()}
            compact.verify_sources(lab, compact.LAB_PINS)
            compact.require_constructor(runtime_newton.CollisionPipeline, newton, "newton/_src/sim/collide.py")
            stack.enter_context(
                compact.scope(
                    importlib.import_module("newton.solvers").SolverFeatherPGS,
                    importlib.import_module("newton.geometry").NarrowPhase,
                    newton,
                    info,
                    get_manager,
                )
            )
        if options.broad_phase_output_max is not None:
            pipeline_type = runtime_newton.CollisionPipeline
            original_constructor = install_broad_phase_limit(
                pipeline_type, options.broad_phase_output_max, report, save
            )
            configured_constructor = pipeline_type.__init__
        install_boundary_check(harness, report, save, get_manager, compat, compact)
        sys.argv = [str(path), *forwarded]
        harness.main()
        if report["boundary_count"] != 2 or not all(entry["check_pass"] for entry in report["boundaries"]):
            raise RuntimeError("Checked capture did not complete both required boundaries")
        if options.broad_phase_output_max is not None and (
            not report["collision_capacity"]["pipelines"]
            or not all(entry["construction_pass"] for entry in report["collision_capacity"]["pipelines"])
        ):
            raise RuntimeError("Requested broad-phase capacity was not successfully applied")
        if sha(path) != digest or sha(Path(__file__)) != report["checked_capture_sha256"]:
            raise RuntimeError("Capture source changed during execution")
        if compat is not None and sha(COMPAT_PATH) != report["mjwarp_linesearch_compat"]["helper_sha256"]:
            raise RuntimeError("MJWarp compatibility helper changed during execution")
        if not output.is_file():
            raise RuntimeError("Original harness did not write its capture result")
        if compact is not None:
            compact.verify_sources(newton)
            compact.verify_sources(lab, compact.LAB_PINS)
            if sha(helper) != info["helper_sha256"]:
                raise RuntimeError("Compact Allegro helper changed during capture")
        if original_constructor is not None:
            if pipeline_type.__init__ is not configured_constructor:
                raise RuntimeError("Broad-phase constructor ownership changed")
            pipeline_type.__init__ = original_constructor
            original_constructor = None
        stack.close()
        report.update(complete=True, check_pass=True)
        return 0
    except BaseException as error:
        report.update(complete=False, check_pass=False, error=repr(error))
        raise
    finally:
        if original_constructor is not None:
            if pipeline_type.__init__ is configured_constructor:
                pipeline_type.__init__ = original_constructor
            else:
                report.update(complete=False, check_pass=False, error="Broad-phase constructor ownership changed")
        sys.argv = previous_argv
        try:
            stack.close()
        finally:
            save()


if __name__ == "__main__":
    raise SystemExit(main())
