# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone Newton clone-plan replication.

This module imports no Isaac Lab packages. It mirrors the Newton builder
portion of ``isaaclab_newton.cloner.newton_replicate`` and replays captured
site requests so the exported bundle can rebuild the same Newton model.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np
import warp as wp
from clone_plan import SiteRequest
from newton import ModelBuilder, solvers
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd

BUILTIN_LABEL_TYPES: tuple[str, ...] = (
    "body",
    "joint",
    "shape",
    "articulation",
    "constraint_mimic",
    "equality_constraint",
)


def _inject_sites(
    main_builder: ModelBuilder,
    proto_builders: Mapping[str, ModelBuilder],
    site_requests: Sequence[SiteRequest],
) -> tuple[dict[str, int], dict[int, dict[str, list[int]]]]:
    global_sites: dict[str, int] = {}
    proto_sites: dict[int, dict[str, list[int]]] = {}

    for site in site_requests:
        vals = tuple(float(v) for v in site.xform)
        if len(vals) != 7:
            raise ValueError(f"Site xform must have 7 values, got {len(vals)}")
        xform = wp.transform(wp.vec3(vals[0], vals[1], vals[2]), wp.quat(vals[3], vals[4], vals[5], vals[6]))
        if site.body_pattern is None:
            global_sites[site.label] = main_builder.add_site(body=-1, xform=xform, label=site.label)
            continue

        any_matched = False
        for proto in proto_builders.values():
            matches = [(i, n) for i, n in enumerate(proto.body_label) if re.fullmatch(site.body_pattern, n)]
            if not matches:
                continue
            any_matched = True
            proto_sites.setdefault(id(proto), {})[site.label] = [
                proto.add_site(body=body_idx, xform=xform, label=f"{body_name}/{site.label}")
                for body_idx, body_name in matches
            ]

        if not any_matched:
            raise ValueError(
                f"Site {site.label!r} with body_pattern {site.body_pattern!r} matched no prototype bodies "
                f"across {len(proto_builders)} prototype(s)."
            )

    return global_sites, proto_sites


def rename_builder_labels(
    builder: ModelBuilder,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: Sequence[int] | np.ndarray,
    mapping: np.ndarray,
) -> None:
    """Rename source-root labels to destination-root labels in-place."""
    env_ids_arr = np.asarray(env_ids, dtype=np.int64)
    mapping_arr = np.asarray(mapping, dtype=np.bool_)

    for i, src_path in enumerate(sources):
        src_root = src_path.rstrip("/")
        world_cols = np.nonzero(mapping_arr[i])[0].tolist()
        world_roots = {int(env_ids_arr[c]): destinations[i].format(int(env_ids_arr[c])) for c in world_cols}

        def _rename_pair(values, worlds):
            if len(values) != len(worlds):
                raise ValueError(f"label/world column length mismatch: {len(values)} vs {len(worlds)}")
            for k in range(len(values)):
                value = values[k]
                if not isinstance(value, str):
                    continue
                world_id = int(worlds[k])
                if world_id not in world_roots:
                    continue
                if not value.startswith(src_root):
                    continue
                suffix = value[len(src_root) :]
                if suffix and not suffix.startswith("/"):
                    continue
                values[k] = world_roots[world_id] + suffix

        for label_type in BUILTIN_LABEL_TYPES:
            labels = getattr(builder, f"{label_type}_label", None)
            if labels is None:
                labels = getattr(builder, f"{label_type}_key", None)
            worlds = getattr(builder, f"{label_type}_world", None)
            if labels is None or worlds is None:
                continue
            _rename_pair(labels, worlds)

        custom = builder.custom_attributes
        world_by_freq: dict[str, ModelBuilder.CustomAttribute] = {}
        for attr in custom.values():
            if getattr(attr, "references", None) == "world":
                world_by_freq[attr.frequency] = attr
        for attr in custom.values():
            if attr.dtype is not str:
                continue
            world_attr = world_by_freq.get(attr.frequency)
            if world_attr is None:
                continue
            if not attr.values or not world_attr.values:
                continue
            _rename_pair(attr.values, world_attr.values)


def build_and_label(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: Sequence[int] | np.ndarray,
    mapping: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray | None = None,
    up_axis: str = "Z",
    simplify_meshes: bool = True,
    default_shape_cfg: Mapping | None = None,
    site_requests: Sequence[SiteRequest] = (),
) -> tuple[ModelBuilder, object]:
    """Build and label a Newton :class:`ModelBuilder` from a captured clone plan."""
    mapping_arr = np.asarray(mapping, dtype=np.bool_)
    env_ids_arr = np.asarray(env_ids, dtype=np.int64)
    positions_arr = np.asarray(positions, dtype=np.float32)
    if quaternions is None:
        quaternions_arr = np.zeros((mapping_arr.shape[1], 4), dtype=np.float32)
        quaternions_arr[:, 3] = 1.0
    else:
        quaternions_arr = np.asarray(quaternions, dtype=np.float32)

    if mapping_arr.shape[0] != len(sources):
        raise ValueError(f"mapping has {mapping_arr.shape[0]} rows but {len(sources)} sources were provided")
    if len(destinations) != len(sources):
        raise ValueError(f"Expected one destination per source, got {len(destinations)} and {len(sources)}")
    if mapping_arr.shape[1] != len(env_ids_arr):
        raise ValueError(f"mapping has {mapping_arr.shape[1]} columns but {len(env_ids_arr)} env ids were provided")

    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
    shape_cfg_overrides = dict(default_shape_cfg or {})

    builder = ModelBuilder(up_axis=up_axis)
    for key, value in shape_cfg_overrides.items():
        if hasattr(builder.default_shape_cfg, key):
            setattr(builder.default_shape_cfg, key, value)
    stage_info = builder.add_usd(stage, ignore_paths=["/World/envs", *sources], schema_resolvers=schema_resolvers)

    env0_pos = positions_arr[0]
    protos: dict[str, ModelBuilder] = {}
    for src_path in sources:
        proto = ModelBuilder(up_axis=up_axis)
        for key, value in shape_cfg_overrides.items():
            if hasattr(proto.default_shape_cfg, key):
                setattr(proto.default_shape_cfg, key, value)
        solvers.SolverMuJoCo.register_custom_attributes(proto)
        proto.add_usd(
            stage,
            root_path=src_path,
            load_visual_shapes=True,
            skip_mesh_approximation=True,
            schema_resolvers=schema_resolvers,
        )
        if simplify_meshes:
            proto.approximate_meshes("convex_hull", keep_visual_shapes=True)
        protos[src_path] = proto

    global_sites, proto_sites = _inject_sites(builder, protos, site_requests)
    global_site_map: dict[str, tuple[int, None]] = {label: (idx, None) for label, idx in global_sites.items()}
    local_site_map: dict[str, list[list[int]]] = {}

    for col, _env_id in enumerate(env_ids_arr.tolist()):
        builder.begin_world()
        delta_pos = (positions_arr[col] - env0_pos).tolist()
        for row in np.nonzero(mapping_arr[:, col])[0].tolist():
            proto = protos[sources[row]]
            offset = builder.shape_count
            builder.add_builder(proto, xform=wp.transform(delta_pos, quaternions_arr[col].tolist()))
            for label, proto_shape_indices in proto_sites.get(id(proto), {}).items():
                if label not in local_site_map:
                    local_site_map[label] = [[] for _ in range(mapping_arr.shape[1])]
                for proto_shape_idx in proto_shape_indices:
                    local_site_map[label][col].append(offset + proto_shape_idx)
        builder.end_world()

    builder.newton_repro_site_index_map = {
        **global_site_map,
        **{label: (None, per_world) for label, per_world in local_site_map.items()},
    }

    rename_builder_labels(builder, sources, destinations, env_ids_arr, mapping_arr)
    return builder, stage_info
