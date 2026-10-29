# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Declarative device fields: one declaration per field, from allocation to the Warp structs kernels take.

A :class:`Field` names a device array (or a scalar) once: its element type,
its shape as dimension names, its fill, whether it takes a gradient and a debug
mirror, and where a binding finds it. Everything else derives from it:

- :class:`FieldGroup` holds fields that share dimensions and are allocated
  together; :meth:`SolverArrays.allocate` allocates a group on its owner, each
  field as the owner's attribute of the field's name, with the owner's device,
  gradient policy and debug mirrors;
- :func:`struct` generates the Warp struct of a set of fields, so a kernel
  takes one argument where it listed every array;
- :func:`bind` fills an instance of such a struct from the objects that hold
  the fields (the solver, the model, a state, the contacts), each field found
  by its owner role and attribute.

Shapes are tuples of dimension names (``("world", "row")``) or fixed lengths,
resolved by the sizes an allocation passes; a feature that is off allocates its
group at the sizes of its placeholder (for example one row per world).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import warp as wp

__all__ = ["Field", "FieldGroup", "SolverArrays", "bind", "fields_of", "replace", "struct", "upload"]

MIRROR_PREFIX = "_debug_position_"
"""Attribute prefix of a field's debug mirror (the solver's position-problem snapshot)."""


@dataclasses.dataclass(frozen=True)
class Field:
    """One device field (or scalar), declared once."""

    name: str
    """The field's name: its owner's attribute and its struct member."""
    dtype: Any
    """Warp element type (or scalar type, for a scalar)."""
    dims: tuple = ()
    """Shape as dimension names, resolved at allocation, or fixed lengths; ``()`` is a scalar."""
    fill: Any = 0
    """Initial value: 0 zeros the array, another value fills it, ``None`` leaves it uninitialized."""
    grad: bool = True
    """Allocate a gradient when the allocation asks for gradients."""
    mirror: bool = False
    """Keep a debug mirror, ``_debug_position_<name>``, when the allocation keeps mirrors."""
    owner: str = "solver"
    """Role of the object a :func:`bind` reads the field from (a keyword of the call)."""
    attr: str | None = None
    """Attribute of that object, when it is not :attr:`name`."""

    @property
    def annotation(self):
        """The field's type in a Warp struct or kernel signature."""
        if not self.dims:
            return self.dtype
        return wp.array(dtype=self.dtype, ndim=len(self.dims))

    def shape(self, sizes: dict[str, int]) -> tuple[int, ...]:
        """The field's shape for the given dimension sizes."""
        return tuple(int(sizes[dim]) if isinstance(dim, str) else int(dim) for dim in self.dims)


class FieldGroup:
    """Fields that are allocated together: one table instead of one allocation call per field."""

    def __init__(self, *fields: Field):
        names = [field.name for field in fields]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate field names in {names}")
        self.fields = tuple(fields)
        self._by_name = {field.name: field for field in fields}

    def __iter__(self):
        return iter(self.fields)

    def __getitem__(self, name: str) -> Field:
        return self._by_name[name]

    def select(self, *names: str) -> tuple[Field, ...]:
        """The fields of ``names``, in that order."""
        return tuple(self._by_name[name] for name in names)

    def owned(self, role: str) -> FieldGroup:
        """The group of the fields whose owner role is ``role`` (for example the ones an owner allocates)."""
        return FieldGroup(*(field for field in self.fields if field.owner == role))

    @property
    def mirrored(self) -> tuple[Field, ...]:
        """The fields with a debug mirror."""
        return tuple(field for field in self.fields if field.mirror)


class SolverArrays:
    """Allocates field groups on an owner with one device, gradient policy and debug-mirror policy.

    Args:
        owner: The object whose attributes the fields become.
        device: Device of every array.
        requires_grad: Allocate gradients for the fields that take them.
        mirrors: Allocate the debug mirrors of mirrored fields (otherwise they are ``None``).
    """

    def __init__(self, owner, device, *, requires_grad: bool = False, mirrors: bool = False):
        self.owner = owner
        self.device = device
        self.requires_grad = bool(requires_grad)
        self.mirrors = bool(mirrors)
        self._placeholders: dict[tuple, wp.array] = {}

    def array(self, field: Field, sizes: dict[str, int], requires_grad: bool | None = None) -> wp.array:
        """Allocate one field at the given sizes (``requires_grad`` overrides the allocator's policy)."""
        shape = field.shape(sizes)
        grad = (self.requires_grad if requires_grad is None else requires_grad) and field.grad
        if field.fill is None:
            return wp.empty(shape, dtype=field.dtype, device=self.device, requires_grad=grad)
        if isinstance(field.fill, (int, float)) and field.fill == 0:
            return wp.zeros(shape, dtype=field.dtype, device=self.device, requires_grad=grad)
        return wp.full(shape, field.fill, dtype=field.dtype, device=self.device, requires_grad=grad)

    def allocate(self, group, /, *, owner=None, requires_grad: bool | None = None, **sizes: int) -> dict[str, wp.array]:
        """Allocate every field of ``group`` at ``sizes`` and give it to its owner, with its debug mirror.

        Args:
            group: The fields.
            owner: Where the arrays go instead of the allocator's owner: an object (attributes) or a dict (items,
                without mirrors).
            requires_grad: Overrides the allocator's gradient policy for this group.
            sizes: The size of every dimension name the fields use.

        Returns:
            The arrays by field name.
        """
        owner = self.owner if owner is None else owner
        arrays = {}
        for field in group:
            array = self.array(field, sizes, requires_grad)
            arrays[field.name] = array
            if isinstance(owner, dict):
                owner[field.name] = array
                continue
            setattr(owner, field.name, array)
            if field.mirror:
                mirror = self.array(field, sizes, requires_grad) if self.mirrors else None
                setattr(owner, MIRROR_PREFIX + field.name, mirror)
        return arrays

    def allocate_words(self, group, /) -> wp.array:
        """Allocate a group of one-word fields of one type as views of one zeroed array, and return that array.

        Each field is the owner's attribute of its name, a one-element view at its position in ``group``, so kernels
        take it as before while one readback of the returned array reads every field.
        """
        dtypes = {field.dtype for field in group}
        if len(dtypes) != 1 or any(field.dims != (1,) for field in group):
            raise ValueError("allocate_words takes one-word fields of one type")
        words = wp.zeros(len(group.fields), dtype=dtypes.pop(), device=self.device)
        for index, field in enumerate(group):
            setattr(self.owner, field.name, words[index : index + 1])
        return words

    def upload(self, group, /, *, owner=None, **values) -> dict[str, wp.array]:
        """Copy host values to the device, one per field of ``group`` (by field name), as the fields' arrays.

        Args:
            group: The fields; every one needs a value.
            owner: Where the arrays go instead of the allocator's owner (an object or a dict).
            values: Host arrays (or sequences) by field name.

        Returns:
            The arrays by field name.
        """
        owner = self.owner if owner is None else owner
        arrays = {}
        for field in group:
            grad = self.requires_grad and field.grad
            array = wp.array(values[field.name], dtype=field.dtype, device=self.device, requires_grad=grad)
            arrays[field.name] = array
            if isinstance(owner, dict):
                owner[field.name] = array
            else:
                setattr(owner, field.name, array)
        return arrays

    def mirror(self, group) -> None:
        """Allocate the debug mirrors of ``group``'s fields at the shapes of the owner's current arrays.

        For fields the owner holds as views of other storage (an alias, a placeholder).
        """
        for field in group.mirrored:
            array = getattr(self.owner, field.name)
            mirror = None
            if self.mirrors:
                grad = self.requires_grad and field.grad
                mirror = wp.zeros(array.shape, dtype=field.dtype, device=self.device, requires_grad=grad)
            setattr(self.owner, MIRROR_PREFIX + field.name, mirror)

    def release(self, group) -> None:
        """Set every field of ``group`` and its mirror to ``None`` on the owner: the feature is off."""
        for field in group:
            setattr(self.owner, field.name, None)
            if field.mirror:
                setattr(self.owner, MIRROR_PREFIX + field.name, None)

    def placeholder(self, dtype, shape: tuple = (1,), fill=0) -> wp.array:
        """The shared array of ``dtype``, ``shape`` and ``fill`` for operands of a feature that is off.

        One array per type, shape and fill serves every such operand. A placeholder always holds its fill:
        kernels may read it (a count of zero, a parent of -1) and write only that value into it.
        """
        key = (dtype, tuple(shape), fill)
        array = self._placeholders.get(key)
        if array is None:
            array = self.array(Field("placeholder", dtype, tuple(shape), fill=fill, grad=False), {})
            self._placeholders[key] = array
        return array

    def snapshot(self, group) -> None:
        """Copy every mirrored field of ``group`` into its mirror (no-op without mirrors)."""
        if not self.mirrors:
            return
        for field in group.mirrored:
            wp.copy(getattr(self.owner, MIRROR_PREFIX + field.name), getattr(self.owner, field.name))


_STRUCT_FIELDS: dict[Any, tuple[Field, ...]] = {}


def struct(name: str, fields, module: str):
    """Generate the Warp struct of ``fields``, one member per field in order.

    Args:
        name: The struct's name.
        fields: The fields, from any groups.
        module: The Python module the struct belongs to (``__name__`` of the declaring module).
    """
    fields = tuple(fields)
    annotations = {field.name: field.annotation for field in fields}
    if len(annotations) != len(fields):
        raise ValueError(f"duplicate field names in struct {name}")
    cls = type(name, (), {"__annotations__": annotations, "__module__": module, "__qualname__": name})
    struct_type = wp.struct(cls)
    _STRUCT_FIELDS[struct_type] = fields
    return struct_type


def fields_of(struct_type) -> tuple[Field, ...]:
    """The fields a :func:`struct` was generated from."""
    return _STRUCT_FIELDS[struct_type]


def bind(struct_type, instance=None, /, **owners):
    """Fill an instance of a generated struct from the objects that hold its fields.

    Each field is read from ``owners[field.owner]`` at its attribute (a field's
    ``owner`` keyword); a keyword named after a field gives that field's value
    directly. Scalars are converted to their declared type.

    Args:
        struct_type: A struct generated by :func:`struct`.
        instance: An existing instance to refill, or ``None`` for a new one.
        owners: The owner objects by role, and direct field values by field name.
    """
    data = struct_type() if instance is None else instance
    for field in _STRUCT_FIELDS[struct_type]:
        if field.name in owners:
            value = owners[field.name]
        else:
            value = getattr(owners[field.owner], field.attr or field.name)
        if not field.dims:
            value = _scalar(field.dtype, value)
        setattr(data, field.name, value)
    return data


def replace(instance, /, **values):
    """A new instance of ``instance``'s generated struct, with its field values except those ``values`` gives."""
    fields = _STRUCT_FIELDS[instance._cls]
    unknown = set(values) - {field.name for field in fields}
    if unknown:
        raise KeyError(f"{instance._cls.key} has no fields {sorted(unknown)}")
    return bind(
        instance._cls, **{field.name: values.get(field.name, getattr(instance, field.name)) for field in fields}
    )


def upload(struct_type, device, source=None, /, **values):
    """Fill a new instance of a generated struct from host data: arrays are copied to ``device``.

    Each field takes ``values[field.name]``, else ``source``'s attribute (a host plan),
    else stays empty: a struct that several owners share may leave fields an owner
    does not use unset. A field that reads a solver's private array derived from a model
    array (attribute ``_name``, such as the effective row data the solver publishes every
    step) is never taken from the raw ``name`` a model source holds: the upload raises
    instead, so a caller binds from the solver or passes the field.
    """
    data = struct_type()
    for field in _STRUCT_FIELDS[struct_type]:
        attr = field.attr or field.name
        if field.name in values:
            value = values[field.name]
        elif source is not None and hasattr(source, attr):
            value = getattr(source, attr)
        elif source is not None and attr.startswith("_") and hasattr(source, attr[1:]):
            raise ValueError(
                f"{struct_type.key}.{field.name} reads the solver's {attr}, not the source's {attr[1:]}: bind from the "
                f"solver or pass {field.name}"
            )
        else:
            continue
        if not field.dims:
            value = _scalar(field.dtype, value)
        elif value is not None and not isinstance(value, wp.array):
            value = wp.array(value, dtype=field.dtype, device=device)
        setattr(data, field.name, value)
    return data


def _scalar(dtype, value):
    if dtype in (int, wp.int32, wp.int64, wp.uint32):
        return int(value)
    if dtype in (float, wp.float32, wp.float64):
        return float(value)
    return value
