# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check existing sticky flags at the unchanged Lab harness's two metadata boundaries.

The optional MJWarp line-search correction changes the algorithm, never its
iteration/tolerance budgets. Without that option this adds no physics kernels
or work inside the timed windows. A clean result
establishes only that the supported Newton narrow-phase and FPGS row/contact
flags or MJWarp warning/overflow flags were zero; it is not a convergence proof
or a substitute for full collision-buffer demand calibration. No flag is cleared.
The opt-in legacy65ab survey bridge has weaker collision coverage: retained
device warnings plus boundary counter snapshots, never a no-drop certificate.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

HARNESS = Path("scripts/benchmarks/fpgs_profile/run_profiled.py")
REQUIRED_PACKAGES = ("newton", "isaaclab", "isaaclab_newton", "isaaclab_tasks", "isaaclab_assets")
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
LEGACY_NEWTON_COMMIT = "65ab4b7973fafa81f6e17615600d2aae708b94e6"


def validate_legacy_source(newton: Path) -> None:
    """Admit only the audited unmodified baseline to the weaker survey protocol."""
    head = subprocess.check_output(["git", "-C", str(newton), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(newton), "status", "--porcelain"], text=True).strip()
    if head != LEGACY_NEWTON_COMMIT or status:
        raise RuntimeError("Legacy capacity observations require clean Newton65ab4b7973")


def install_legacy_replication(builder_type, report: dict):
    """Route Lab's per-world label prefixes through the existing baseline merger."""
    original = builder_type.replicate
    if "label_prefixes" in inspect.signature(original).parameters:
        raise RuntimeError("Legacy replication adapter encountered an unexpected native prefix API")

    def replicate(self, builder, world_count, spacing=(0.0, 0.0, 0.0), *, xforms=None, label_prefixes=None):
        if label_prefixes is None:
            return original(self, builder, world_count, spacing, xforms=xforms)
        if world_count <= 0:
            return None
        if self.current_world != -1:
            raise RuntimeError("Cannot begin a new world: already in world context; call end_world() first")
        if xforms is None:
            wp = original.__globals__["wp"]
            offsets = original.__globals__["compute_world_offsets"](world_count, spacing, self.up_axis)
            xforms = [wp.transform(offset, wp.quat_identity()) for offset in offsets]
        elif len(xforms) != world_count:
            raise ValueError(f"xforms must contain {world_count} entries, got {len(xforms)}")
        if len(label_prefixes) != world_count:
            raise ValueError(f"label_prefixes must contain {world_count} entries, got {len(label_prefixes)}")
        worlds = list(range(self.world_count, self.world_count + world_count))
        self._merge_builder_copies(builder, worlds, xforms, label_prefixes)
        self.world_gravity.extend(builder._gravity_as_vector() for _ in range(world_count))
        self.world_count += world_count
        return None

    builder_type.replicate = replicate
    report["legacy_replication_adapter"] = {
        "installed": True,
        "scope": "Construction-only label-prefix forwarding to the existing baseline _merge_builder_copies",
        "solver_kernels_modified": False,
        "labels_discarded": False,
    }
    return original


def install_collision_storage_overrides(harness, pipeline_type, report: dict, save) -> None:
    """Apply explicit storage overrides after parsing tasks whose collision config is None."""
    original = getattr(harness, "parse_env_cfg", None)
    if not callable(original):
        return
    names = {"env.sim.physics.collision_cfg." + name: name for name in ("rigid_contact_max", "max_triangle_pairs")}

    @functools.wraps(original)
    def parse(*args, **kwargs):
        requested, retained = {}, []
        for override in kwargs.get("overrides", ()):
            key, equals, value = override.partition("=")
            if key not in names:
                retained.append(override)
                continue
            name = names[key]
            if not equals or not value.isdecimal() or not 0 < int(value) < 2**31 or name in requested:
                raise ValueError(f"Invalid or repeated explicit collision storage override: {override}")
            requested[name] = int(value)
        if not requested:
            return original(*args, **kwargs)
        cfg = original(*args, **{**kwargs, "overrides": retained})
        physics = cfg.sim.physics
        if physics.solver_cfg.solver_type != "feather_pgs":
            raise RuntimeError("Storage setup bridge is restricted to the FPGS survey")
        materialized = physics.collision_cfg is None
        if materialized:
            module = importlib.import_module("isaaclab_newton.physics.newton_collision_cfg")
            collision_cfg = module.NewtonCollisionPipelineCfg()
            defaults = inspect.signature(pipeline_type.__init__).parameters
            for name, value in collision_cfg.to_pipeline_args().items():
                # NewtonManager already selects explicit when collision_cfg is None.
                expected = "explicit" if name == "broad_phase" else defaults[name].default
                if value != expected:
                    raise RuntimeError(f"Collision config default differs from the selected Newton: {name}")
            physics.collision_cfg = collision_cfg
        for name, value in requested.items():
            setattr(physics.collision_cfg, name, value)
        report.setdefault("collision_storage_setup", []).append(
            {"materialized_default_config": materialized, "requested": requested, "physics_budgets_modified": False}
        )
        save()
        return cfg

    harness.parse_env_cfg = parse


def legacy_row_status(solver, entry: dict) -> None:
    """Read the baseline's existing sticky row warnings without adding kernels."""
    flags = solver._row_overflow_warning_emitted
    if not solver.warn_constraint_overflow or not solver._track_row_capacity:
        raise RuntimeError("Legacy row warning recording must remain enabled")
    if flags is None or flags.shape != (3,) or flags.device.is_capturing:
        raise RuntimeError("Unexpected legacy sticky row warning storage")
    values = flags.numpy().tolist()
    if any(type(value) is not int or value not in (0, 1) for value in values):
        raise RuntimeError("Invalid legacy sticky row warning values")
    entry["flags"] = dict(zip(("dense", "matrix_free", "propagation"), map(bool, values), strict=True))
    entry["capacities"] = {
        name: int(getattr(solver, name))
        for name in ("dense_max_constraints", "mf_max_constraints", "propagation_max_constraints")
    }
    entry["capacity_coverage"] = "Existing sticky row warnings only; contacts checked separately at boundaries"
    entry["contact_law"] = {
        "friction_anchor_beta": float(solver.friction_anchor_beta),
        "contact_compliance": bool(solver.contact_compliance),
        "contact_torsion_radius": float(solver.contact_torsion_radius),
    }
    if any(values):
        raise RuntimeError(f"Legacy sticky row warning flags are nonzero: {entry['flags']}")
    entry["check_pass"] = True


def legacy_collision_status(manager, result: dict) -> None:
    """Take honest current-counter snapshots, retaining existing per-call warnings."""
    pipeline, solver, contacts = manager._collision_pipeline, manager._solver, manager._contacts
    narrow = pipeline.narrow_phase
    result.update(
        status="legacy_warning_screened_boundary_snapshots",
        whole_run_collision_capacity_verified=False,
        log_warning_screen_required=True,
        limitations=[
            "Collision counters reset each call; these are boundary snapshots, not whole-run maxima.",
            "Parent must reject collision capacity warnings in the complete process log.",
            "Reducer contact reservations can roll back on exhaustion without a sticky flag; no no-drop certificate.",
            "No convergence or contact-law parity claim.",
        ],
        snapshots={},
    )
    if not narrow.verify_buffers:
        raise RuntimeError("Legacy survey requires verify_buffers=True throughout the run")

    def snapshot(name, counter, capacity, index=0):
        if counter is None or counter.device.is_capturing:
            raise RuntimeError(f"Missing or captured legacy counter: {name}")
        count = counter.numpy().tolist()[index]
        result["snapshots"][name] = {"count": count, "capacity": int(capacity)}
        if type(count) is not int or not 0 <= count <= capacity:
            raise RuntimeError(f"Legacy boundary capacity failure: {name}={count}, capacity={capacity}")

    snapshot("broad_phase", pipeline.broad_phase_pair_count, pipeline.broad_phase_shape_pairs.shape[0])
    snapshot("gjk", narrow.gjk_candidate_pairs_count, narrow.gjk_candidate_pairs.shape[0])
    for name in (
        "shape_pairs_mesh",
        "triangle_pairs",
        "shape_pairs_mesh_plane",
        "shape_pairs_mesh_mesh",
        "shape_pairs_sdf_sdf",
    ):
        buffer, counter = getattr(narrow, name), getattr(narrow, name + "_count")
        if buffer is not None:
            snapshot(name, counter, buffer.shape[0])
    if narrow.split_gjk_mpr:
        query_count = pipeline.broad_phase_pair_count if narrow.sparse_gjk_pairs else narrow.gjk_candidate_pairs_count
        snapshot("split_query", query_count, narrow.split_query_results.shape[0])
        snapshot("split_gjk", narrow.split_gjk_work_count, narrow.split_gjk_work_items.shape[0])
        snapshot("split_manifold", narrow.split_manifold_work_count, narrow.split_manifold_work_items.shape[0])
    snapshot("contacts", contacts.rigid_contact_count, contacts.rigid_contact_max)
    result["contact_storage"] = {
        "contact_pool": int(contacts.rigid_contact_max),
        "solver_scratch": int(solver._max_contacts_alloc),
        "friction_patch": int(solver._friction_patches.capacity),
    }
    if min(solver._max_contacts_alloc, solver._friction_patches.capacity) < contacts.rigid_contact_max:
        raise RuntimeError("Legacy contact pool exceeds solver or friction-patch storage")
    reducer = narrow.global_contact_reducer
    if reducer is not None:
        snapshot("reducer_contacts", reducer.contact_count, reducer.capacity)
        snapshot(
            "reduction_hash_entries",
            reducer.hashtable.active_slots,
            reducer.hashtable.capacity,
            reducer.hashtable.capacity,
        )
        snapshot("reduction_hash_insert_failures", reducer.ht_insert_failures, 0)
        result["reduction_hash_warning_load_percent"] = 80  # Audited constant at the admitted source pin.
        if result["snapshots"]["reduction_hash_entries"]["count"] * 100 >= reducer.hashtable.capacity * 80:
            raise RuntimeError("Legacy boundary reducer hash load reaches its warning threshold")
    result["check_pass"] = True


def sha(path: Path) -> str:
    """Hash the exact source bytes used by the capture."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_packages(lab: Path, newton: Path) -> dict:
    """Describe the selected source packages without importing simulator modules."""
    lab, newton = lab.resolve(), newton.resolve()
    roots = {"newton": newton / "newton"}
    for directory in sorted((lab / "source").glob("isaaclab*")):
        package = directory / directory.name
        if (package / "__init__.py").is_file():
            if not package.resolve().is_relative_to(lab):
                raise RuntimeError(f"Selected Lab package escapes its checkout: {package}")
            roots[directory.name] = package.resolve()
    if set(REQUIRED_PACKAGES) - roots.keys() or not (roots["newton"] / "__init__.py").is_file():
        raise RuntimeError("Selected checkouts lack required Newton/Lab source packages")
    return {name: {"root": str(root), "init_sha256": sha(root / "__init__.py")} for name, root in roots.items()}


def select_imports(lab: Path, newton: Path) -> dict:
    """Select checkout packages explicitly without rewriting shared editable installs."""
    selected = source_packages(lab, newton)
    # A package imported by sitecustomize or an editable finder cannot be fixed
    # by changing sys.path. Fail rather than mix old and selected package state.
    import_provenance(selected, resolve=False)
    prefixes = list(dict.fromkeys(str(Path(source["root"]).parent) for source in selected.values()))
    sys.path[:] = prefixes + [entry for entry in sys.path if entry not in prefixes]
    importlib.invalidate_caches()
    import_provenance(selected)
    return selected


def import_provenance(selected: dict, *, required=(), previous: dict | None = None, resolve=True) -> dict:
    """Verify selected origins and all loaded Newton/Lab modules outside timing."""
    loaded = {}
    for name, module in list(sys.modules.items()):
        package = name.partition(".")[0]
        if package != "newton" and package != "isaaclab" and not package.startswith("isaaclab_"):
            continue
        if package not in selected:
            raise RuntimeError(f"Loaded an unselected Lab package: {name}")
        if module is None:
            raise RuntimeError(f"Unresolved selected module: {name}")
        root = Path(selected[package]["root"])
        # Read the namespace directly: getattr can execute a lazy module loader.
        filename = vars(module).get("__file__")
        paths = [Path(value).resolve() for value in vars(module).get("__path__", ())]
        generated_by = None
        if filename is None and not paths:
            owner_name = name
            while "." in owner_name:
                parent_name, _, child = owner_name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is None or vars(parent).get(child) is not sys.modules[owner_name]:
                    break
                filename = vars(parent).get("__file__")
                if filename is not None:
                    generated_by = parent_name
                    break
                owner_name = parent_name
        if (
            (filename is None and not paths)
            or any(not path.is_relative_to(root) for path in paths)
            or (filename is not None and not Path(filename).resolve().is_relative_to(root))
        ):
            raise RuntimeError(f"Selected import resolved outside its checkout: {name}")
        record = {"paths": [str(path) for path in paths]}
        if generated_by is not None:
            record["generated_by"] = generated_by
        if filename is not None:
            path = Path(filename).resolve()
            record.update(file=str(path), sha256=sha(path))
        loaded[name] = record
    if set(required) - loaded.keys():
        raise RuntimeError(f"Required selected imports were not loaded: {sorted(set(required) - loaded.keys())}")
    origins = {}
    for name, source in selected.items():
        path = Path(source["root"]) / "__init__.py"
        if sha(path) != source["init_sha256"]:
            raise RuntimeError(f"Selected import source changed: {name}")
        if resolve:
            module = sys.modules.get(name)
            spec = None if module is not None else importlib.util.find_spec(name)
            origin = getattr(module, "__file__", None) if module is not None else getattr(spec, "origin", None)
            if origin is None or Path(origin).resolve() != path:
                raise RuntimeError(f"Selected package resolver points outside its checkout: {name}")
            origins[name] = str(path)
    result = {"selected": selected, "resolved": origins, "loaded": loaded}
    if previous is not None and (
        previous["selected"] != selected
        or previous["resolved"] != origins
        or any(
            loaded.get(name) != record for name, record in previous["loaded"].items() if "generated_by" not in record
        )
    ):
        raise RuntimeError("Selected import provenance changed during capture")
    return result


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


def check_solver(physics: str, solver, entry: dict) -> None:
    """Publish flag details before raising; never clear or change solver state."""
    entry["solver_class"] = type(solver).__name__
    if physics == "feather_pgs":
        if type(solver).__name__ != "SolverFeatherPGS":
            raise RuntimeError("Checked FPGS capture resolved an unexpected solver")
        entry["solver_recipe"] = {
            name: getattr(solver, name, None)
            for name in (
                "grouped_dynamics",
                "lazy_kinematics",
                "pgs_mode",
                "mf_gs_parallel_rows",
                "mf_gs_parallel_matrix_free",
                "mf_gs_parallel_sweeps",
            )
        }
        entry["solver_paths"] = {
            "fused_k1": bool(getattr(solver, "_fused_k1", False)),
            "grouped_topology": getattr(solver, "_grouped_topology", None) is not None,
            "sparse_factor": getattr(solver, "_sparse_factor", None) is not None,
            "g1_kinetic_state": getattr(solver, "_g1_kinetic_state", None) is not None,
            "sparse_diagonal_contact_solve": bool(getattr(solver, "_sparse_diagonal_contact_solve", False)),
        }
        read = getattr(solver, "constraint_capacity_status", None)
        check = getattr(solver, "check_constraint_capacity", None)
        if not callable(read) or not callable(check):
            if os.environ.get("FPGS_BENCH_LEGACY_CAPACITY") == "1":
                legacy_row_status(solver, entry)
                return
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
        entry["solver_options"] = {
            name: int(getattr(model.opt, name))
            for name in ("solver", "integrator", "cone", "iterations", "ls_iterations")
        }
        entry["solver_options"].update(
            native_contacts=native,
            is_sparse=bool(model.is_sparse),
            enable_sleeping=bool(solver.enable_sleeping),
        )
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
            if physics == "feather_pgs" and os.environ.get("FPGS_BENCH_LEGACY_CAPACITY") == "1":
                legacy_collision_status(manager, result)
                return
            raise RuntimeError("This Newton narrow phase lacks the sticky capacity-check API")
        if not narrow.verify_buffers:
            raise RuntimeError("Checked collision requires verify_buffers=True throughout the run")
        flags = read()
        result["flags"] = flags
        expected_flags = NARROW_FLAGS | (
            {"heightfield_pair_csr_raw_or_membership"} if getattr(narrow, "_heightfield_pair_csr", False) else set()
        )
        if set(flags) != expected_flags or any(type(value) is not bool for value in flags.values()):
            raise RuntimeError("Unrecognized narrow-phase capacity-status contract")
        check()
        if any(flags.values()):
            raise RuntimeError("Narrow-phase capacity flags were nonzero despite a successful checker")
        result.update(check_pass=True, status="checked")
    except BaseException as error:
        result.update(check_pass=False, status="failed", error=repr(error))
        raise


def install_boundary_check(harness: ModuleType, report: dict, save, get_manager, compat=None) -> None:
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
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_initialized():
                free, total = torch.cuda.mem_get_info()
                entry["device_memory_snapshot"] = {
                    "free_bytes": int(free),
                    "total_bytes": int(total),
                    "scope": "Device-wide at this untimed boundary; not process allocation, task-only usage or peak",
                }
            if compat is not None:
                supported = physics == "newton_mjwarp" and bool(compat.supports_model(manager._solver.mjw_model))
                entry["mjwarp_linesearch_model_supported"] = supported
                if not supported:
                    raise RuntimeError("Requested MJWarp line-search fix does not support the actual model")
            check_solver(physics, manager._solver, entry)
            check_collision(physics, manager, entry)
            if physics == "feather_pgs":
                entry["fused_contact_solve"] = fused_contact_metadata(manager._solver)
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
        "physics_work_modified": options.mjwarp_linesearch_fix,
        "physics_budgets_modified": False,
    }
    if os.environ.get("FPGS_BENCH_LEGACY_CAPACITY") == "1":
        report.update(
            legacy_capacity_mode=True,
            legacy_newton_commit=LEGACY_NEWTON_COMMIT,
            scope="Legacy sticky row warnings and collision boundary snapshots; requires parent log screening; no no-drop certificate",
        )
    checks.parent.mkdir(parents=True, exist_ok=True)

    def save():
        checks.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    previous_argv, previous_path = sys.argv, sys.path.copy()
    pipeline_type, original_constructor = None, None
    builder_type, original_replication = None, None
    save()
    try:
        if report.get("legacy_capacity_mode"):
            validate_legacy_source(newton)
        path = lab / HARNESS
        source = path.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        report["run_profiled_sha256"] = digest
        report["checked_capture_sha256"] = sha(Path(__file__))
        if digest != options.expected_run_sha256:
            raise RuntimeError("Selected Lab run_profiled.py differs from the parent-pinned source")
        selected = select_imports(lab, newton)
        for name in ("newton", "isaaclab", "isaaclab_newton"):
            importlib.import_module(name)
        harness = ModuleType("_newton_checked_lab_capture")
        harness.__file__ = str(path)
        exec(compile(source, str(path), "exec"), harness.__dict__)
        if not callable(getattr(harness, "main", None)) or not callable(getattr(harness, "_model_meta", None)):
            raise RuntimeError("Selected Lab harness lacks the required capture boundary")
        runtime_newton = importlib.import_module("newton")
        if Path(runtime_newton.__file__).resolve() != newton / "newton/__init__.py":
            raise RuntimeError("Capture imported the wrong Newton checkout")
        if report.get("legacy_capacity_mode"):
            builder_type = runtime_newton.ModelBuilder
            original_replication = install_legacy_replication(builder_type, report)
        capture_physics = forwarded[forwarded.index("--physics") + 1] if "--physics" in forwarded else "feather_pgs"
        if capture_physics == "feather_pgs" and callable(getattr(harness, "parse_env_cfg", None)):
            install_collision_storage_overrides(harness, runtime_newton.CollisionPipeline, report, save)
        for name in REQUIRED_PACKAGES:
            importlib.import_module(name)
        report["import_provenance_before"] = import_provenance(selected, required=REQUIRED_PACKAGES)
        save()
        compat = install_linesearch_fix(report) if options.mjwarp_linesearch_fix else None
        if options.broad_phase_output_max is not None:
            pipeline_type = runtime_newton.CollisionPipeline
            original_constructor = install_broad_phase_limit(
                pipeline_type, options.broad_phase_output_max, report, save
            )
        install_boundary_check(
            harness, report, save, lambda: importlib.import_module("isaaclab_newton.physics").NewtonManager, compat
        )
        sys.argv = [str(path), *forwarded]
        harness.main()
        report["import_provenance_after"] = import_provenance(
            selected, required=REQUIRED_PACKAGES, previous=report["import_provenance_before"]
        )
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
        report.update(complete=True, check_pass=True)
        return 0
    except BaseException as error:
        report.update(complete=False, check_pass=False, error=repr(error))
        raise
    finally:
        sys.path[:] = previous_path
        if original_constructor is not None:
            pipeline_type.__init__ = original_constructor
        if original_replication is not None:
            builder_type.replicate = original_replication
        sys.argv = previous_argv
        save()


if __name__ == "__main__":
    raise SystemExit(main())
