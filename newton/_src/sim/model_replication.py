# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device replication of a finalized rigid world, owned by :class:`Model`."""

import operator
from functools import cache
from typing import Any

import numpy as np
import warp as wp

from .model import Model


@wp.kernel(module="unique", enable_backward=False)
def _repeat_values(source: wp.array[Any], stride: int, target: wp.array[Any]):
    i = wp.tid()
    target[i] = source[i % stride]


@cache
def _repeat_value_kernel(dtype):
    """Reuse code specialization without retaining a model or population storage."""
    return wp.overload(_repeat_values, [wp.array[dtype], int, wp.array[dtype]])


@wp.struct
class _RepeatWords:
    source: wp.array[wp.uint32]
    count: int
    repeat: int


@wp.struct
class _RepeatBytes:
    source: wp.array[wp.uint8]
    count: int
    repeat: int


@wp.kernel(module="unique", enable_backward=False)
def _repeat_offsets(fields: wp.array[Any], worlds: int, unit: int, offsets: wp.array[wp.int64]):
    offset = wp.int64(0)
    for i in range(fields.shape[0]):
        offsets[i] = offset
        count = wp.int64(fields[i].count) * wp.int64(wp.where(fields[i].repeat != 0, worlds, 1))
        offset += ((count * wp.int64(unit) + wp.int64(255)) // wp.int64(256)) * wp.int64(256 // unit)
    offsets[fields.shape[0]] = offset


@wp.kernel(module="unique", enable_backward=False)
def _repeat_words(
    fields: wp.array[_RepeatWords], offsets: wp.array[wp.int64], worlds: int, target: wp.array[wp.vec4ui]
):
    thread = wp.tid()
    i = wp.int64(thread) * wp.int64(4)
    lo, hi = int(0), fields.shape[0]
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if i < offsets[mid]:
            hi = mid
        else:
            lo = mid
    field = fields[lo]
    local = int(i - offsets[lo])
    limit = field.count * wp.where(field.repeat != 0, worlds, 1)
    value = wp.vec4ui(0)
    for component in range(4):
        if local + component < limit:
            value[component] = field.source[(local + component) % field.count]
    target[thread] = value


@wp.kernel(module="unique", enable_backward=False)
def _repeat_bytes(fields: wp.array[_RepeatBytes], offsets: wp.array[wp.int64], worlds: int, target: wp.array[wp.uint8]):
    i = wp.int64(wp.tid())
    lo, hi = int(0), fields.shape[0]
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if i < offsets[mid]:
            hi = mid
        else:
            lo = mid
    field = fields[lo]
    local = int(i - offsets[lo])
    if local < field.count * wp.where(field.repeat != 0, worlds, 1):
        target[i] = field.source[local % field.count]
    else:
        target[i] = wp.uint8(0)


class _PlainCopyPlan:
    """Prototype-owned descriptors for canonical plain arrays, independent of population size.

    Descriptors retain live source buffers, not a model or frozen values. Each replica
    retains its plan and copy offsets; its public array views independently retain their
    arena through a deleter closure. Keeping one extracted field keeps that arena alive.
    """

    def __init__(self, fields, signature):
        self.signature = signature
        self.groups = []
        for unit, dtype, struct in ((4, wp.uint32, _RepeatWords), (1, wp.uint8, _RepeatBytes)):
            records, descriptors = [], []
            for name, value, repeat, size, field_unit in fields:
                if field_unit != unit:
                    continue
                descriptor = struct()
                descriptor.source = wp.array(ptr=value.ptr, shape=size // unit, dtype=dtype, device=value.device)
                descriptor.count, descriptor.repeat = size // unit, int(repeat)
                records.append((name, value, repeat, size))
                descriptors.append(descriptor)
            if records:
                host = wp.array(descriptors, dtype=struct, device="cpu")
                device = host.to(records[0][1].device)
                kernel = wp.overload(_repeat_offsets, [wp.array[struct], int, int, wp.array[wp.int64]])
                self.groups.append((unit, dtype, records, device, host, kernel))

    def copy(self, worlds):
        arrays, storage = {}, [self]
        for unit, dtype, records, fields, _host, offset_kernel in self.groups:
            offsets, total = [], 0
            for _, _, repeat, size in records:
                offsets.append(total)
                total += ((size * (worlds if repeat else 1) + 255) // 256) * 256
            arena = wp.empty(total // unit, dtype=dtype, device=fields.device)

            def retain_arena(_ptr, _size, owner=arena):
                pass

            for offset, (name, value, repeat, size) in zip(offsets, records, strict=True):
                count = worlds if repeat else 1
                arrays[name] = wp.array(
                    ptr=arena.ptr + offset,
                    shape=(value.shape[0] * count, *value.shape[1:]),
                    dtype=value.dtype,
                    device=fields.device,
                    capacity=size * count,
                    deleter=retain_arena,
                )
            device_offsets = wp.empty(len(records) + 1, dtype=wp.int64, device=fields.device)
            storage.append(device_offsets)
            wp.launch(offset_kernel, 1, [fields, worlds, unit, device_offsets], device=fields.device)
            target = arena.reshape((-1, 4)).view(wp.vec4ui) if unit == 4 else arena
            kernel = _repeat_words if unit == 4 else _repeat_bytes
            wp.launch(kernel, target.size, [fields, device_offsets, worlds, target], device=fields.device)
        return arrays, tuple(storage)


def _packed_plain_arrays(source, specs, counts, worlds):
    """Derive a copy plan from AttributeSpec; unsupported raw layouts use normal repetition."""
    if not source.device.is_cuda:
        vars(source).pop("_replication_copy_plan", None)
        return {}, ()
    fields, signature, totals = [], [], {1: 0, 4: 0}
    for name, spec in specs:
        if spec.references is not None or spec.compaction_policy != "generic":
            continue
        owner, field = source, name
        if ":" in name:
            namespace, field = name.split(":", 1)
            owner = getattr(source, namespace, None)
        value = getattr(owner, field, None)
        if not isinstance(value, wp.array) or not value.size:
            continue
        if counts[spec.frequency] == 0 and spec.requires_empty_sentinel:
            continue
        if value.device != source.device or not value.is_contiguous or value.requires_grad:
            vars(source).pop("_replication_copy_plan", None)
            return {}, ()
        repeat = spec.frequency != Model.AttributeFrequency.ONCE
        size = value.size * value.strides[-1]
        unit = 4 if size % 4 == 0 and value.ptr % 4 == 0 else 1
        count = size // unit * (worlds if repeat else 1)
        # Per-field indices and launch bounds are int32; arena byte offsets are int64.
        # Check in Python before narrowing, including each field's alignment padding.
        if count >= 2**31 - 256:
            vars(source).pop("_replication_copy_plan", None)
            return {}, ()
        totals[unit] += ((count * unit + 255) // 256) * (256 // unit)
        fields.append((name, value, repeat, size, unit))
        signature.append((name, spec, id(value), value.ptr, value.shape, value.strides, value.dtype))
    if not fields or totals[4] // 4 >= 2**31 or totals[1] >= 2**31:
        vars(source).pop("_replication_copy_plan", None)
        return {}, ()
    signature = tuple(signature)
    plan = getattr(source, "_replication_copy_plan", None)
    if plan is None or plan.signature != signature:
        plan = source._replication_copy_plan = _PlainCopyPlan(fields, signature)
    return plan.copy(worlds)


@wp.kernel(enable_backward=False)
def _repeat_references(source: wp.array[wp.int32], stride: int, offset: int, target: wp.array[wp.int32]):
    i = wp.tid()
    value = source[i % stride]
    if value >= 0:
        value += (i // stride) * offset
    target[i] = value


@wp.kernel(enable_backward=False)
def _repeat_starts(source: wp.array[wp.int32], stride: int, offset: int, target: wp.array[wp.int32]):
    i = wp.tid()
    target[i] = source[i % stride] + (i // stride) * offset


@wp.kernel(enable_backward=False)
def _world_starts(count: int, worlds: int, target: wp.array[wp.int32]):
    i = wp.tid()
    target[i] = wp.min(i, worlds) * count


@wp.kernel(enable_backward=False)
def _gravity(source: wp.array[wp.vec3], worlds: int, target: wp.array[wp.vec3]):
    i = wp.tid()
    target[i] = source[wp.int32(i == worlds)]


@wp.kernel(module="unique", enable_backward=False)
def _copy_world_values(
    source: wp.array[Any],
    source_ids: wp.array[int],
    target_ids: wp.array[int],
    width: int,
    status: wp.array[int],
    target: wp.array[Any],
):
    pair, column = wp.tid()
    if status[0] == 0:
        target[target_ids[pair] * width + column] = source[source_ids[pair] * width + column]


@cache
def _copy_world_value_kernel(dtype):
    return wp.overload(
        _copy_world_values, [wp.array[dtype], wp.array[int], wp.array[int], int, wp.array[int], wp.array[dtype]]
    )


@wp.kernel(enable_backward=False)
def _copy_world_references(
    source: wp.array[int],
    source_ids: wp.array[int],
    target_ids: wp.array[int],
    width: int,
    offset: int,
    status: wp.array[int],
    target: wp.array[int],
):
    pair, column = wp.tid()
    if status[0] == 0:
        value = source[source_ids[pair] * width + column]
        if value >= 0:
            value += (target_ids[pair] - source_ids[pair]) * offset
        target[target_ids[pair] * width + column] = value


def _world_copy_arrays(source: Model, target: Model, states=(), controls=()):
    """Resolve model/state/control storage from the model's canonical metadata."""
    prototype = source._replication_source
    if prototype is None or target._replication_source is not prototype:
        raise ValueError("World transfer requires populations from the same prepared model")
    result = []
    frequency = Model.AttributeFrequency

    def append(name, src, dst, spec):
        if src is None and dst is None:
            return
        if not isinstance(src, wp.array) or not isinstance(dst, wp.array):
            raise ValueError(f"Incompatible world-transfer array {name}")
        if spec is None or spec.frequency == frequency.ONCE or spec.compaction_policy != "generic":
            raise ValueError(f"World transfer lacks per-world metadata for {name}")
        count = prototype._attribute_frequency_count(spec.frequency)
        if not src.size and not dst.size:
            return
        if count == 0 and spec.requires_empty_sentinel and src.shape == dst.shape:
            return
        if not src.is_contiguous or not dst.is_contiguous:
            raise ValueError(f"World-transfer storage must be contiguous: {name}")
        if (
            src.dtype != dst.dtype
            or src.device != dst.device
            or src.shape[1:] != dst.shape[1:]
            or src.shape[0] != count * source.world_count * spec.row_width
            or dst.shape[0] != count * target.world_count * spec.row_width
        ):
            raise ValueError(f"Incompatible world-transfer layout for {name}")
        if src.ptr == dst.ptr:
            raise ValueError(f"World-transfer arrays must not alias: {name}")
        offset = None if spec.references is None else prototype._attribute_frequency_count(spec.references)
        result.append((src, dst, offset, src.size // source.world_count))

    for name, spec in source._iter_attribute_specs():
        if spec.frequency == frequency.ONCE or spec.compaction_policy != "generic":
            continue  # Destination-owned immutable topology and host labels.
        owner, destination, field = source, target, name
        if ":" in name:
            namespace, field = name.split(":", 1)
            owner, destination = getattr(source, namespace, None), getattr(target, namespace, None)
        src, dst = getattr(owner, field, None), getattr(destination, field, None)
        if isinstance(src, wp.array) or isinstance(dst, wp.array):
            append(name, src, dst, spec)

    def object_arrays(src, dst, prefix=""):
        for field in vars(src).keys() | vars(dst).keys():
            a, b = getattr(src, field, None), getattr(dst, field, None)
            name = prefix + field
            if isinstance(a, wp.array) or isinstance(b, wp.array):
                append(name, a, b, source._attribute_spec(name))
            elif isinstance(a, Model.AttributeNamespace) or isinstance(b, Model.AttributeNamespace):
                if not isinstance(a, Model.AttributeNamespace) or not isinstance(b, Model.AttributeNamespace):
                    raise ValueError(f"Incompatible world-transfer namespace {name}")
                object_arrays(a, b, name + ":")

    for src, dst in (*states, *controls):
        object_arrays(src, dst)
    # Gravity includes a global-world sentinel; only local world rows move.
    result.append((source.gravity, target.gravity, None, 1))
    return result


@wp.struct
class _WorldCopyField:
    source: wp.uint64
    target: wp.uint64
    width: int
    start: wp.int64
    unit: int


@wp.kernel(module="unique", enable_backward=False)
def _copy_world_packed(
    fields: wp.array[_WorldCopyField], source_ids: wp.array[int], target_ids: wp.array[int], status: wp.array[int]
):
    actor, block = wp.tid()
    if status[0] != 0:
        return
    column = wp.int64(block) * wp.int64(4)
    lo, hi = int(0), fields.shape[0]
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if column < fields[mid].start:
            hi = mid
        else:
            lo = mid
    field = fields[lo]
    local = int(column - field.start)
    rowbytes = wp.uint64(field.width) * wp.uint64(field.unit)
    src = field.source + wp.uint64(source_ids[actor]) * rowbytes
    dst = field.target + wp.uint64(target_ids[actor]) * rowbytes
    if field.unit == 4:
        source_words = wp.array(ptr=src, shape=(field.width,), dtype=wp.uint32)
        target_words = wp.array(ptr=dst, shape=(field.width,), dtype=wp.uint32)
        for component in range(4):
            if local + component < field.width:
                target_words[local + component] = source_words[local + component]
    else:
        source_bytes = wp.array(ptr=src, shape=(field.width,), dtype=wp.uint8)
        target_bytes = wp.array(ptr=dst, shape=(field.width,), dtype=wp.uint8)
        for component in range(4):
            if local + component < field.width:
                target_bytes[local + component] = source_bytes[local + component]


def _copy_world_arrays(arrays, source_ids, target_ids, status):
    """Copy previously validated rows; failed solver transfer leaves them untouched."""
    if not source_ids.size:
        return
    # Preserve read/write dependencies across aliased fields; otherwise group canonical plain rows.
    spans = sorted(
        (value.ptr, value.ptr + value.size * value.strides[-1], write)
        for src, dst, _, _ in arrays
        for write, value in ((False, src), (True, dst))
        if value.size
    )
    read_end, write_end, overlap = 0, 0, False
    for start, end, write in spans:
        if start < write_end or (write and start < read_end):
            overlap = True
            break
        if write:
            write_end = max(write_end, end)
        else:
            read_end = max(read_end, end)
    records, ordinary, descriptors, total = [], [], [], 0
    for src, dst, offset, width in arrays:
        if not src.size:
            continue
        rowbytes = width * src.strides[-1]
        unit = 4 if rowbytes % 4 == 0 and src.ptr % 4 == 0 and dst.ptr % 4 == 0 else 1
        if (
            overlap
            or offset is not None
            or any(
                value.device != status.device or not value.is_contiguous or value.requires_grad for value in (src, dst)
            )
            or rowbytes // unit >= 2**31 - 4
        ):
            ordinary.append((src, dst, offset, width))
        else:
            descriptors.append((src.ptr, dst.ptr, rowbytes // unit, total, unit))
            total += ((rowbytes // unit + 3) // 4) * 4
            records.append((src, dst, offset, width))
    if source_ids.size * (total // 4) >= 2**31:
        ordinary.extend(records)
    elif records:
        # Public struct layout and kernel-local views avoid per-field Python array wrappers.
        host = wp.array(np.array(descriptors, dtype=_WorldCopyField.numpy_dtype()), dtype=_WorldCopyField, device="cpu")
        fields = host.to(status.device)
        wp.launch(
            _copy_world_packed,
            (source_ids.size, total // 4),
            [fields, source_ids, target_ids, status],
            device=status.device,
        )
        # Ordinary graphs do not retain raw pointers or upload storage; status does.
        status._newton_world_copy_storage = (host, fields, source_ids, target_ids, tuple(records))
    kernels = {
        src.dtype: _copy_world_value_kernel(src.dtype) for src, _, offset, _ in ordinary if offset is None and src.size
    }
    for src, dst, offset, width in ordinary:
        if not src.size:
            continue
        if offset is None:
            wp.launch(
                kernels[src.dtype],
                (source_ids.size, width),
                [src.flatten(), source_ids, target_ids, width, status, dst.flatten()],
                device=src.device,
            )
        else:
            if getattr(src.dtype, "_wp_scalar_type_", src.dtype) != wp.int32:
                raise ValueError("World transfer requires int32 entity references")
            src_int, dst_int = src.view(wp.int32).flatten(), dst.view(wp.int32).flatten()
            int_width = src_int.size // (src.size // width)
            wp.launch(
                _copy_world_references,
                (source_ids.size, int_width),
                [src_int, source_ids, target_ids, int_width, offset, status, dst_int],
                device=src.device,
            )


def _repeat(array: wp.array | None, count: int, *, offset: int | None = None, start: bool = False):
    if array is None:
        return None
    if array.size == 0 or (start and array.size == 1):
        return wp.clone(array)
    shape = ((array.shape[0] - int(start)) * count + int(start), *array.shape[1:])
    result = wp.empty(shape, dtype=array.dtype, device=array.device)
    if offset is None:
        wp.launch(
            _repeat_value_kernel(array.dtype),
            result.size,
            [array.flatten(), array.size, result.flatten()],
            device=array.device,
        )
    else:
        if getattr(array.dtype, "_wp_scalar_type_", array.dtype) != wp.int32:
            raise ValueError("Model.replicate requires int32 entity references")
        source_flat, target_flat = array.view(wp.int32).flatten(), result.view(wp.int32).flatten()
        kernel = _repeat_starts if start else _repeat_references
        wp.launch(
            kernel,
            target_flat.size,
            [source_flat, source_flat.size - int(start), offset, target_flat],
            device=array.device,
        )
    return result


def replicate_model(source: Model, world_count: int) -> Model:
    """Implement Model.replicate without builder reconstruction or device readback."""
    if isinstance(world_count, bool) or operator.index(world_count) < 1:
        raise ValueError("Model.replicate requires a positive integer world_count")
    world_count = operator.index(world_count)
    if source.world_count != 1 or source._has_global_entities is not False:
        raise ValueError("Model.replicate requires one finalized explicit local world without global entities")
    for name in (
        "particle_count",
        "tri_count",
        "tet_count",
        "edge_count",
        "spring_count",
        "muscle_count",
        "gaussians_count",
        "heightfield_count",
    ):
        if getattr(source, name):
            raise ValueError(f"Model.replicate does not support nonzero {name}")
    if source.requires_grad or source._has_rod_joints or source.actuators:
        raise ValueError("Model.replicate does not support gradients, rod joints, or Newton actuators")
    if any(source.custom_frequency_counts.values()):
        raise ValueError("Model.replicate does not support nonempty custom frequencies")
    if source.device.is_cuda and source.device.is_capturing:
        raise RuntimeError("Model.replicate allocates a fresh model outside CUDA graph capture")

    result = Model(source.device)
    frequency = Model.AttributeFrequency
    counts = {key: source._attribute_frequency_count(key) for key in source._ATTRIBUTE_FREQUENCY_COUNT_ATTRS}
    counts.update(source.custom_frequency_counts)
    counts[frequency.ONCE] = 1
    if max(counts.values(), default=0) * world_count > np.iinfo(np.int32).max:
        raise ValueError("Model.replicate entity counts exceed int32 indexing")
    specs = tuple(source._iter_attribute_specs())
    packed, result._replication_storage = _packed_plain_arrays(source, specs, counts, world_count)

    # Existing attribute metadata is the sole schema for per-entity storage.
    # Resource tables below are shared geometry, never population-sized fields.
    resources = {
        "_convex_support_lut",
        "_convex_support_vertex_offsets",
        "_convex_support_neighbors",
        "heightfield_data",
        "heightfield_elevations",
        "heightfield_meshes",
        "_mesh_keep_alive",
        "mesh_edge_indices",
        "mesh_edge_centers",
        "mesh_edge_halves",
        "_texture_sdf_data",
        "_texture_sdf_coarse_textures",
        "_texture_sdf_subgrid_textures",
        "_texture_sdf_subgrid_start_slots",
        "_generated_sdf_edge_meshes",
        "soft_mesh_adjacency",
        "soft_mesh_adjacency_device",
    }
    handled = set()
    namespaces = {}
    for name, value in source.__dict__.items():
        if isinstance(value, Model.AttributeNamespace):
            if value._deprecated_aliases:
                raise ValueError(f"Model.replicate does not support deprecated namespace aliases in {name}")
            namespace = Model.AttributeNamespace(value._name)
            setattr(result, name, namespace)
            namespaces[name] = namespace
            handled.add(name)

    for name, spec in specs:
        owner, target, field = source, result, name
        if ":" in name:
            namespace, field = name.split(":", 1)
            if namespace not in namespaces:
                continue
            owner, target = getattr(source, namespace), namespaces[namespace]
        if spec.compaction_policy == "passthrough" or not hasattr(owner, field):
            continue
        value = getattr(owner, field)
        handled.add(name)
        count = counts[spec.frequency]
        repeats = 1 if spec.frequency == frequency.ONCE else world_count
        offset = None if spec.references is None else counts[spec.references]
        policy = spec.compaction_policy
        if policy == "world_start":
            value = wp.empty(world_count + 2, dtype=wp.int32, device=source.device)
            wp.launch(_world_starts, value.size, [count, world_count, value], device=source.device)
        elif policy == "color_groups":
            value = [_repeat(group, repeats, offset=count) for group in value]
        elif isinstance(value, wp.array):
            if name in packed:
                value = packed[name]
            elif count == 0 and spec.requires_empty_sentinel:
                value = wp.clone(value)
            else:
                value = _repeat(value, repeats, offset=offset, start=policy == "start")
        elif isinstance(value, list):
            value = value * repeats
        elif value is not None:
            raise ValueError(f"Model.replicate does not support attribute storage {name}")
        setattr(target, field, value)

    for name in resources:
        if hasattr(source, name):
            value = getattr(source, name)
            setattr(result, name, list(value) if isinstance(value, list) else value)
            handled.add(name)

    # Host topology views remain lazy; numeric collision topology stays on device.
    result._replication_source = source
    del result.body_shapes
    result._shape_collision_filter_pairs = source._shape_collision_filter_pairs.repeat(world_count, source.shape_count)
    result.shape_contact_pairs = _repeat(source.shape_contact_pairs, world_count, offset=source.shape_count)
    result.shape_contact_pair_count = source.shape_contact_pair_count * world_count
    handled.update(("body_shapes", "_shape_collision_filter_pairs", "shape_contact_pairs", "shape_contact_pair_count"))

    # Preserve FK level schedules rather than rebuilding articulation traversal.
    if source._fk_articulation_level_start is not None:
        levels = source._fk_level_joint_start.size - 1
        result._fk_articulation_level_start = _repeat(
            source._fk_articulation_level_start, world_count, offset=levels, start=True
        )
        result._fk_level_joint_start = _repeat(
            source._fk_level_joint_start, world_count, offset=source._fk_level_joints.size, start=True
        )
        result._fk_level_joints = _repeat(source._fk_level_joints, world_count, offset=source.joint_count)
        result._fk_level_parent_pos = _repeat(source._fk_level_parent_pos, world_count)
    handled.update(
        ("_fk_articulation_level_start", "_fk_level_joint_start", "_fk_level_joints", "_fk_level_parent_pos")
    )

    result.gravity = wp.empty(world_count + 1, dtype=wp.vec3, device=source.device)
    wp.launch(_gravity, result.gravity.size, [source.gravity, world_count, result.gravity], device=source.device)
    result.custom_frequency_articulation = {
        name: wp.clone(value) for name, value in source.custom_frequency_articulation.items()
    }
    handled.update(("gravity", "custom_frequency_articulation"))
    # Empty provider-specific frequency boundaries still follow world_count.
    for name in source.custom_frequency_counts:
        owner, target, field = source, result, name
        if ":" in name:
            namespace, field = name.split(":", 1)
            owner, target = getattr(source, namespace), namespaces[namespace]
        boundary = field + "_world_start"
        if hasattr(owner, boundary):
            setattr(target, boundary, wp.zeros(world_count + 2, dtype=wp.int32, device=source.device))
            handled.add(name + "_world_start")

    # Copy scalar options and metadata containers, rejecting unknown nonempty
    # arrays rather than silently sharing stale extension-owned model state.
    reset = {
        "_collision_pipeline",
        "_replication_source",
        "_replication_copy_plan",
        "_replication_storage",
        "bvh_shapes",
        "bvh_shapes_group_roots",
        "bvh_shape_enabled",
        "bvh_shape_bounds",
        "bvh_shape_world_transforms",
    }
    for owner, target, prefix in [
        (source, result, ""),
        *((getattr(source, name), ns, name + ":") for name, ns in namespaces.items()),
    ]:
        for field, original in owner.__dict__.items():
            name = prefix + field
            if name in handled or name in reset:
                continue
            value = original
            if isinstance(value, wp.array):
                if value.size and field != "muscle_start":
                    raise ValueError(f"Model.replicate lacks metadata for array {name}")
                value = wp.clone(value)
            elif isinstance(value, (dict, list, set)):
                if value and field not in {
                    "attribute_specs",
                    "attribute_frequency",
                    "attribute_assignment",
                    "custom_frequency_counts",
                    "custom_frequency_label_attributes",
                    "_requested_state_attributes",
                    "_requested_contact_attributes",
                }:
                    raise ValueError(f"Model.replicate lacks metadata for container {name}")
                value = value.copy()
            elif value is not None and not isinstance(value, (bool, int, float, str, type(source.device))):
                raise ValueError(f"Model.replicate does not support unregistered field {name}")
            setattr(target, field, value)

    for key, name in source._ATTRIBUTE_FREQUENCY_COUNT_ATTRS.items():
        setattr(result, name, counts[key] * world_count)
    if source.shape_count and source.bvh_shape_bounds is not None:
        result.bvh_shape_bounds = _repeat(source.bvh_shape_bounds, world_count)
        enabled = source.bvh_shape_count_enabled
        result.bvh_shape_count_enabled = enabled * world_count
        result.bvh_shape_enabled = _repeat(
            source.bvh_shape_enabled[:enabled].view(wp.int32), world_count, offset=source.shape_count
        ).view(wp.uint32)
        result._build_shape_bvh(result)
    return result
