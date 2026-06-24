# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone Newton clone-plan replication.

This module imports no Isaac Lab packages. It mirrors the Newton builder portion
of :func:`isaaclab_newton.cloner.newton_replicate` (specifically
``_build_newton_builder_from_mapping``) and replays captured site requests so the
exported bundle can rebuild the same Newton model. The per-source builder
construction, per-world replication, and label renaming live in
:mod:`newton_clone_utils`; this module only assembles them and injects sites.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np
import warp as wp
from clone_plan import SiteRequest
from newton import ModelBuilder
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
from newton_clone_utils import build_source_builders, rename_builder_labels, replicate_builder_mapping
from newton_model_utils import replace_newton_builder_shape_colors

from pxr import Usd


def _resolve_matching_names(pattern: str, names: Sequence[str]) -> list[tuple[int, str]]:
    """Return ``(index, name)`` pairs whose label fully matches *pattern*, in label order.

    Standalone stand-in for :func:`isaaclab.utils.string.resolve_matching_names` for the
    single-pattern site-injection case: matches each body label with :func:`re.fullmatch`
    and preserves the order of *names* (``preserve_order=False`` upstream behavior).
    """
    compiled = re.compile(pattern)
    return [(index, name) for index, name in enumerate(names) if compiled.fullmatch(name)]


def _inject_sites(
    main_builder: ModelBuilder,
    source_builders: Mapping[str, ModelBuilder],
    site_requests: Sequence[SiteRequest],
) -> tuple[dict[str, int], dict[int, dict[str, list[int]]], dict[str, wp.transform]]:
    """Inject captured site requests, mirroring :meth:`NewtonManager._cl_inject_sites`.

    Returns ``(global_site_indices, source_site_indices, env_root_sites)`` where
    *global_site_indices* maps ``{label: main_builder_shape_idx}``, *source_site_indices*
    maps ``{id(source_builder): {label: [source_local_shape_idx, ...]}}``, and
    *env_root_sites* maps ``{label: env_root_relative_transform}`` for per-world sites.
    """
    global_site_indices: dict[str, int] = {}
    source_site_indices: dict[int, dict[str, list[int]]] = {}
    env_root_sites: dict[str, wp.transform] = {}

    for site in site_requests:
        vals = tuple(float(v) for v in site.xform)
        if len(vals) != 7:
            raise ValueError(f"Site xform must have 7 values, got {len(vals)}")
        xform = wp.transform(wp.vec3(vals[0], vals[1], vals[2]), wp.quat(vals[3], vals[4], vals[5], vals[6]))

        if site.per_world:
            env_root_sites[site.label] = xform
            continue
        if site.body_pattern is None:
            global_site_indices[site.label] = main_builder.add_site(body=-1, xform=xform, label=site.label)
            continue

        any_matched = False
        for source_builder in source_builders.values():
            matches = _resolve_matching_names(site.body_pattern, list(source_builder.body_label))
            if not matches:
                continue
            any_matched = True
            source_site_indices.setdefault(id(source_builder), {})[site.label] = [
                source_builder.add_site(body=body_idx, xform=xform, label=f"{body_name}/{site.label}")
                for body_idx, body_name in matches
            ]

        if not any_matched:
            raise ValueError(
                f"Site {site.label!r} with body_pattern {site.body_pattern!r} matched no source-builder bodies "
                f"across {len(source_builders)} source builder(s)."
            )

    return global_site_indices, source_site_indices, env_root_sites


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
    """Build and label a Newton :class:`ModelBuilder` from a captured clone plan.

    Mirrors ``isaaclab_newton.cloner.replicate._build_newton_builder_from_mapping``
    followed by :func:`isaaclab_newton.cloner.rename_builder_labels`, returning the
    populated builder and the main builder's USD stage info.
    """
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

    def _new_builder() -> ModelBuilder:
        builder = ModelBuilder(up_axis=up_axis)
        for key, value in shape_cfg_overrides.items():
            if hasattr(builder.default_shape_cfg, key):
                setattr(builder.default_shape_cfg, key, value)
        return builder

    builder = _new_builder()
    stage_info = builder.add_usd(stage, ignore_paths=["/World/envs", *sources], schema_resolvers=schema_resolvers)
    replace_newton_builder_shape_colors(builder, stage)

    source_builders = build_source_builders(
        stage,
        sources,
        _new_builder,
        schema_resolvers,
        simplify_meshes=simplify_meshes,
    )

    _global_sites, source_sites, env_root_sites = _inject_sites(builder, source_builders, site_requests)

    # ``replicate_builder_mapping`` populates per-world copies into ``builder`` as a side effect;
    # its returned site/transform maps are not consumed downstream.
    replicate_builder_mapping(
        builder,
        sources,
        mapping_arr,
        positions_arr,
        quaternions_arr,
        source_builders,
        source_site_indices=source_sites,
        env_root_sites=env_root_sites,
    )

    rename_builder_labels(builder, sources, destinations, env_ids_arr, mapping_arr)
    return builder, stage_info
