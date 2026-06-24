# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone port of ``isaaclab_newton.cloner.newton_clone_utils``.

Mirrors the per-source builder construction, per-world replication, and label
renaming used by the live :func:`isaaclab_newton.cloner.newton_replicate`, so the
exported bundle rebuilds a byte-identical Newton model. This module imports no
Isaac Lab packages: it operates on NumPy clone-plan arrays (instead of the live
torch tensors) and inlines :func:`isaaclab.cloner.cloner_utils.replace_path_prefix`.

Source: ``source/isaaclab_newton/isaaclab_newton/cloner/newton_clone_utils.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import warp as wp
from newton import ModelBuilder, solvers
from newton_model_utils import replace_newton_builder_shape_colors

from pxr import Usd

_BUILTIN_LABEL_TYPES: tuple[tuple[str, str], ...] = (
    ("body_label", "body_world"),
    ("joint_label", "joint_world"),
    ("shape_label", "shape_world"),
    ("articulation_label", "articulation_world"),
    ("constraint_mimic_label", "constraint_mimic_world"),
)


def replace_path_prefix(path: str, source_prefix: str, destination_prefix: str) -> str:
    """Replace ``source_prefix`` in ``path`` with ``destination_prefix`` on a path boundary.

    Inlined from :func:`isaaclab.cloner.cloner_utils.replace_path_prefix` to keep this
    module Isaac-Lab-free.
    """
    source_prefix = source_prefix.rstrip("/") or "/"
    destination_prefix = destination_prefix.rstrip("/") or "/"
    if not path.startswith(source_prefix):
        return path
    suffix = path[len(source_prefix) :]
    if suffix and not suffix.startswith("/"):
        return path
    return destination_prefix + suffix


def build_source_builders(
    stage: Usd.Stage,
    sources: Sequence[str],
    create_builder: Callable[[], ModelBuilder],
    schema_resolvers: Sequence[Any],
    *,
    ignore_paths: Sequence[str] | None = None,
    simplify_meshes: bool = True,
) -> dict[str, ModelBuilder]:
    """Build one Newton builder for each clone source prim path."""
    builders: dict[str, ModelBuilder] = {}
    for source in sources:
        builder = create_builder()
        solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(
            stage,
            root_path=source,
            load_visual_shapes=True,
            skip_mesh_approximation=True,
            schema_resolvers=schema_resolvers,
            ignore_paths=ignore_paths,
        )
        if simplify_meshes:
            builder.approximate_meshes("convex_hull", keep_visual_shapes=True)
        replace_newton_builder_shape_colors(builder, stage)
        builders[source] = builder
    return builders


def replicate_builder_mapping(
    builder: ModelBuilder,
    sources: Sequence[str],
    mapping: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    source_builders: dict[str, ModelBuilder],
    *,
    source_site_indices: dict[int, dict[str, list[int]]] | None = None,
    env_root_sites: dict[str, wp.transform] | None = None,
    per_world_builder_hooks: Sequence[Callable[[ModelBuilder, int, list[float], list[float]], None]] = (),
    post_replicate_hooks: Sequence[Callable[[ModelBuilder], None]] = (),
) -> tuple[dict[str, list[list[int]]], list[wp.transform]]:
    """Replicate source builders into per-env Newton worlds.

    The per-world placement composes the destination world transform with the
    inverse of the source world transform (``world_xform * source_xform^-1``),
    matching :func:`isaaclab_newton.cloner.replicate_builder_mapping`. This is
    only equivalent to a bare positional delta when every source sits at its own
    world origin with identity orientation, so the full compose is required for
    rotated worlds or sources placed away from env 0.
    """
    source_site_indices = source_site_indices or {}
    env_root_sites = env_root_sites or {}
    num_worlds = int(mapping.shape[1])
    local_site_map: dict[str, list[list[int]]] = {}
    world_xforms: list[wp.transform] = []
    source_world_indices = mapping.astype(np.int64).argmax(axis=1)

    for col in range(num_worlds):
        builder.begin_world()
        world_xform = wp.transform(positions[col].tolist(), quaternions[col].tolist())
        world_xforms.append(world_xform)

        for label, xform in env_root_sites.items():
            site_idx = builder.add_site(body=-1, xform=wp.transform_multiply(world_xform, xform), label=label)
            local_site_map.setdefault(label, [[] for _ in range(num_worlds)])[col].append(site_idx)

        for row in np.nonzero(mapping[:, col])[0].tolist():
            source_builder = source_builders[sources[int(row)]]
            offset = builder.shape_count
            source_col = int(source_world_indices[int(row)])
            source_xform = wp.transform(positions[source_col].tolist(), quaternions[source_col].tolist())
            builder.add_builder(
                source_builder, xform=wp.transform_multiply(world_xform, wp.transform_inverse(source_xform))
            )

            for label, source_shape_indices in source_site_indices.get(id(source_builder), {}).items():
                local_indices = local_site_map.setdefault(label, [[] for _ in range(num_worlds)])[col]
                local_indices.extend(offset + shape_idx for shape_idx in source_shape_indices)

        for hook in per_world_builder_hooks:
            hook(builder, col, positions[col].tolist(), quaternions[col].tolist())
        builder.end_world()

    for hook in post_replicate_hooks:
        hook(builder)
    return local_site_map, world_xforms


def rename_builder_labels(
    builder: ModelBuilder,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: Sequence[int] | np.ndarray,
    mapping: np.ndarray,
) -> None:
    """Rewrite source-root labels to per-env destination roots in-place.

    Mirrors :func:`isaaclab_newton.cloner.rename_builder_labels` but drops the
    Fabric body-binding bookkeeping, which the standalone replay does not use.
    """
    env_ids_arr = np.asarray(env_ids, dtype=np.int64)
    mapping_arr = np.asarray(mapping, dtype=np.bool_)

    for source_index, source in enumerate(sources):
        source_root = source.rstrip("/")
        world_cols = np.nonzero(mapping_arr[source_index])[0].tolist()
        world_roots = {
            int(env_ids_arr[col]): destinations[source_index].format(int(env_ids_arr[col])) for col in world_cols
        }

        def _rename_pair(values, worlds, world_roots=world_roots, source_root=source_root):
            for index, (value, world) in enumerate(zip(values, worlds, strict=True)):
                if world is None:
                    continue
                world_root = world_roots.get(int(world))
                if isinstance(value, str) and world_root is not None:
                    renamed_value = replace_path_prefix(value, source_root, world_root)
                    if renamed_value != value:
                        values[index] = renamed_value

        for labels_attr, worlds_attr in _BUILTIN_LABEL_TYPES:
            labels = getattr(builder, labels_attr, None)
            worlds = getattr(builder, worlds_attr, None)
            if labels is not None and worlds is not None:
                _rename_pair(labels, worlds)

        if "mujoco:equality_constraint_label" not in builder.custom_attributes:
            labels = getattr(builder, "equality_constraint_label", None)
            worlds = getattr(builder, "equality_constraint_world", None)
            if labels is not None and worlds is not None:
                _rename_pair(labels, worlds)

        custom_attrs = builder.custom_attributes.values()
        worlds_by_freq = {
            attr.frequency: attr.values for attr in custom_attrs if getattr(attr, "references", None) == "world"
        }
        for attr in custom_attrs:
            if attr.dtype is str and attr.values:
                worlds = worlds_by_freq.get(attr.frequency)
                if worlds:
                    _rename_pair(attr.values, worlds)
