# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Constructor-only reuse of the validated 16K Allegro C capacity recipe.

No diagnostic kernels, per-step hooks, installed edits, or Lab configuration
changes. The original sparse GJK routing is preserved explicitly before JIT.
"""

import functools
import hashlib
import inspect
import os
from contextlib import contextmanager
from pathlib import Path

PUBLIC = 286720
BROAD = 524288
PINS = {
    "newton/_src/solvers/feather_pgs/solver_feather_pgs.py": "4e6fff0ba5f94ee0b610360844aa578c328a2bb057137b162b6bff653d56704f",
    "newton/_src/geometry/narrow_phase.py": "f2e6f566d0b38c761eda38c89bfad1a398f6817daac48e9c2e7fdc5c60ccab39",
    "newton/_src/sim/collide.py": "4f5f20510e3c3fe0b9e60eefb09b3c90bd6fdf584b9fdbb6d810709da4066ad5",
    "newton/_src/geometry/coherent_convex.py": "467c5f3a580ee6cbe75ab27e6c2bb590a171e7f97e8a85bb70495d9b7f2ed338",
    "newton/_src/geometry/coherent_convex_rejection.py": "d26defca3d86d5f2267616ffb5207905fce58731b97bef5700dccbaad14a8abf",
}
LAB_PINS = {
    "source/isaaclab_newton/isaaclab_newton/physics/feather_pgs_manager.py": "31bea42769931529365a1f067049fa577b91719923795214dda69e819e8125c3",
    "source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py": "218fd3c4ff0216fa9bd6911d08b739d12bba47cc9c90d3a834687717e545707c",
}


def verify_sources(root, pins=PINS):
    """Fail closed on any unsupported source revision before construction."""
    for relative, expected in pins.items():
        if hashlib.sha256((Path(root) / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Unsupported compact Allegro source: {relative}")


def require_constructor(cls, root, relative):
    """Reject foreign/prewrapped constructors, including transparent wrappers."""
    original = cls.__init__
    if (
        inspect.unwrap(original) is not original
        or Path(original.__code__.co_filename).resolve() != Path(root) / relative
    ):
        raise RuntimeError(f"Unexpected constructor owner: {cls.__name__}")


@contextmanager
def scope(solver_type, narrow_type, root, report, get_manager):
    """Change two original constructor arguments and restore only owned wrappers."""
    verify_sources(root)
    require_constructor(solver_type, root, "newton/_src/solvers/feather_pgs/solver_feather_pgs.py")
    require_constructor(narrow_type, root, "newton/_src/geometry/narrow_phase.py")
    originals = solver_type.__init__, narrow_type.__init__
    signatures = tuple(inspect.signature(value) for value in originals)
    records = report["constructors"] = [{"calls": 0, "complete": False, "restored": False} for _ in originals]

    @functools.wraps(originals[0])
    def solver(self, *args, **kwargs):
        record = records[0]
        if record["calls"] or get_manager()._collision_cfg is not None:
            raise RuntimeError("Expected one original default-None Allegro solver constructor")
        bound = signatures[0].bind(self, *args, **kwargs)
        model = bound.arguments["model"]
        record.update(calls=1, prior_model_capacity=int(model.rigid_contact_max))
        model.rigid_contact_max = PUBLIC
        result = originals[0](*bound.args, **bound.kwargs)
        if (
            get_manager()._collision_cfg is not None
            or model.rigid_contact_max != PUBLIC
            or self._max_contacts_alloc != PUBLIC
        ):
            raise RuntimeError("Compact Allegro public capacity was not applied before allocation")
        record["complete"] = True
        return result

    @functools.wraps(originals[1])
    def narrow(self, *args, **kwargs):
        record = records[1]
        bound = signatures[1].bind(self, *args, **kwargs)
        if (
            record["calls"]
            or bound.arguments.get("sparse_gjk_pairs") is not None
            or bound.arguments["max_candidate_pairs"] != BROAD
        ):
            raise RuntimeError("Conflicting compact Allegro narrow-phase constructor")
        bound.arguments["sparse_gjk_pairs"] = True
        record["calls"] = 1
        result = originals[1](*bound.args, **bound.kwargs)
        if self.sparse_gjk_pairs is not True or self.max_candidate_pairs != BROAD:
            raise RuntimeError("Sparse routing was not preserved before construction")
        record["complete"] = True
        return result

    solver_type.__init__, narrow_type.__init__ = solver, narrow
    try:
        yield
    finally:
        errors = []
        for cls, wrapper, original, record in reversed(
            list(zip((solver_type, narrow_type), (solver, narrow), originals, records, strict=True))
        ):
            if cls.__init__ is wrapper:
                cls.__init__ = original
                record["restored"] = True
            else:
                errors.append(cls.__name__)
        if errors:
            raise RuntimeError(f"Foreign constructor ownership retained: {errors}")


def validate_snapshot(value):
    """Use the same strict scalar allocation contract in child and parent."""
    expected = {
        "worlds": 16384,
        "public": PUBLIC,
        "pipeline_public": PUBLIC,
        "solver_public": PUBLIC,
        "full_pairs": 2834432,
        "cache_keys": 2523136,
        "broad": BROAD,
        "query": BROAD,
        "sparse": True,
        "split": True,
        "verify": True,
        "collision_cfg_none": True,
        "block": 128,
        "worker_multiplier": 4,
        "iterations": 12,
        "sweeps": 24,
        "dense_rows": 192,
        "mf_rows": 64,
        "propagation_rows": 192,
        "grouped": True,
        "lazy": True,
        "parallel_rows": 128,
        "matrix_free": True,
    }
    if not isinstance(value, dict) or any(
        type(value.get(key)) is not type(item) or value[key] != item for key, item in expected.items()
    ):
        raise RuntimeError("Compact Allegro allocation/budget contract changed")
    if value.get("queues") != [BROAD] * 7 or value.get("route_sizes") != [PUBLIC] * 6:
        raise RuntimeError("Compact Allegro dependent buffer allocation changed")
    sm_count = value.get("sm_count")
    if (
        type(sm_count) is not int
        or sm_count <= 0
        or value.get("workers") != 128 * max(256, min((2834432 + 127) // 128, sm_count * 4)) * 4
    ):
        raise RuntimeError("Original sparse GJK worker grid was not preserved")


def snapshot(manager):
    """Read existing scalar metadata at the two original untimed boundaries only."""
    s, p, c = manager._solver, manager._collision_pipeline, manager._contacts
    n = p.narrow_phase
    cache = n._coherent_cache
    if (
        cache is None
        or os.environ.get("NEWTON_NARROW_PHASE_COHERENT_CONVEX") != "reject_only"
        or os.environ.get("NEWTON_NARROW_PHASE_THREADS_X") != "4"
    ):
        raise RuntimeError("Compact Allegro requires original C rejection-only / worker multiplier")
    value = {
        "worlds": int(s.world_count),
        "public": int(c.rigid_contact_max),
        "pipeline_public": int(p.rigid_contact_max),
        "solver_public": int(s._max_contacts_alloc),
        "full_pairs": len(p.shape_pairs_filtered),
        "cache_keys": cache.data.keys.shape[0],
        "broad": p.broad_phase_shape_pairs.shape[0],
        "query": n.max_candidate_pairs,
        "sparse": bool(n.sparse_gjk_pairs),
        "split": bool(n.split_gjk_mpr),
        "verify": bool(n.verify_buffers),
        "collision_cfg_none": manager._collision_cfg is None,
        "block": int(n.block_dim),
        "worker_multiplier": 4,
        "workers": int(n.total_num_threads),
        "sm_count": int(p.device.sm_count),
        "iterations": int(s.pgs_iterations),
        "sweeps": int(s.mf_gs_parallel_sweeps),
        "dense_rows": int(s.dense_max_constraints),
        "mf_rows": int(s.mf_max_constraints),
        "propagation_rows": int(s.propagation_max_constraints),
        "grouped": bool(s.grouped_dynamics),
        "lazy": bool(s.lazy_kinematics),
        "parallel_rows": int(s.mf_gs_parallel_rows),
        "matrix_free": bool(s.mf_gs_parallel_matrix_free),
        "queues": [
            array.shape[0]
            for array in (
                n.gjk_candidate_pairs,
                n.split_query_results,
                n.split_gjk_work_items,
                n.split_manifold_work_items,
                cache.data.unresolved_work_items,
                cache.data.query_slot,
                cache.data.query_done,
            )
        ],
        "route_sizes": [
            getattr(s, name).size
            for name in (
                "contact_world",
                "contact_slot",
                "contact_slots_needed",
                "contact_art_a",
                "contact_art_b",
                "contact_path",
            )
        ],
    }
    validate_snapshot(value)
    return value
