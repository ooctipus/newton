# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared packed dense and matrix-free row metadata contracts, and the row record the sweeps stream."""

from .row_kernels import PGS_CONSTRAINT_TYPE_COUNT

_DENSE_META_ROW_TYPE_BITS = 3
_DENSE_META_ROW_TYPE_MASK = (1 << _DENSE_META_ROW_TYPE_BITS) - 1
_DENSE_META_MAX_PARENT = ((2**31 - 1) >> _DENSE_META_ROW_TYPE_BITS) - 1
_MF_META_MAX_CAPACITY = 65535  # Parent values 0..65534; 0xffff encodes no parent.


def _validate_dense_metadata_encoding(max_constraints: int) -> None:
    """Validate packed dense-row metadata against its signed 32-bit layout."""
    if PGS_CONSTRAINT_TYPE_COUNT > _DENSE_META_ROW_TYPE_MASK + 1:
        raise RuntimeError("Dense row metadata does not have enough row-type bits")
    if max_constraints - 1 > _DENSE_META_MAX_PARENT:
        raise ValueError(
            f"dense_max_constraints={max_constraints} exceeds packed parent capacity {_DENSE_META_MAX_PARENT + 1}"
        )


def _validate_mf_metadata_encoding(max_constraints: int) -> None:
    """Reject capacities that alias the packed MF parent sentinel."""
    if not 0 <= max_constraints <= _MF_META_MAX_CAPACITY:
        raise ValueError(f"mf_max_constraints must be in [0, {_MF_META_MAX_CAPACITY}] for packed MF metadata")
    if PGS_CONSTRAINT_TYPE_COUNT > 65536:
        raise RuntimeError("MF metadata does not have enough row-type bits")


def _mf_parent_expression(packed: str, max_constraints: int) -> str:
    """Decode unsigned 16-bit parents plus -1, preserving smaller-capacity kernels."""
    _validate_mf_metadata_encoding(max_constraints)
    if max_constraints <= 32768:
        return f"{packed} >> 16"
    # Adding one modulo 2^16 maps the 0xffff sentinel to zero before signed
    # subtraction; valid high parent indices never undergo sign extension.
    return f"static_cast<int>(((static_cast<unsigned int>({packed}) >> 16) + 1u) & 0xffffu) - 1"


def pack_mf_header(word: str, dof_a: str, dof_b: str, inverse: str, rhs: str, kind: str, parent: str, indent: str):
    """Declare ``int4 word``, a free-body row's packed header word as :func:`unpack_mf_header` decodes it.

    ``x``: the bodies' first world coordinates, ``dof_a`` in the high and ``dof_b`` in the low 16 bits (-1 for
    none); ``y`` and ``z``: the bits of the stored inverse diagonal and of the right-hand side; ``w``: the row
    type in the low and the parent (0xFFFF for none) in the high 16 bits.
    """
    return (
        f"{indent}int4 {word};\n"
        f"{indent}{word}.x = ({dof_a} << 16) | ({dof_b} & 0xFFFF);\n"
        f"{indent}{word}.y = __float_as_int({inverse});\n"
        f"{indent}{word}.z = __float_as_int({rhs});\n"
        f"{indent}{word}.w = static_cast<int>(static_cast<unsigned int>({kind}) | "
        f"(static_cast<unsigned int>({parent}) << 16));"
    )


def unpack_mf_header(word: str, indent: str) -> str:
    """Declare a free-body row's fields from its packed header word ``word`` (:func:`pack_mf_header`): ``dof_a``,
    ``dof_b``, ``mf_diag`` (the stored inverse diagonal), ``packed_tp`` and ``mf_rt`` (the row type)."""
    return (
        f"{indent}int packed_dofs = {word}.x;\n"
        f"{indent}int dof_a = packed_dofs >> 16;\n"
        f"{indent}int dof_b = (packed_dofs << 16) >> 16;\n"
        f"{indent}float mf_diag = __int_as_float({word}.y);\n"
        f"{indent}int packed_tp = {word}.w;\n"
        f"{indent}int mf_rt = packed_tp & 0xFFFF;"
    )


RECORD_WORDS = 8
"""Words of a row's record (:func:`pack_row_record`): two ``int4``, which a row loop loads a row ahead."""


def pack_row_record(
    record: str,
    dof_a: str,
    dof_b: str,
    n_a: str,
    n_b: str,
    diagonal: str,
    rhs: str,
    kind: str,
    parent: str,
    mu: str,
    indent: str,
) -> str:
    """Store a row's record at the ``int`` pointer ``record``: everything a row loop reads of the row but its ``Z``
    and its impulse, in two ``int4`` (:data:`RECORD_WORDS`).

    The first is the header (:func:`pack_mf_header`): the first world coordinates of the row's sides ``dof_a`` and
    ``dof_b`` (-1 for no side), the bits of its stored diagonal (the class's form: the free bodies' inverse, the dense
    rows' ``A_ii``), of its right-hand side (the law's plus its incident velocity: its residual at ``du = 0``), and its
    kind and parent (0xFFFF for none). The second holds the sides' widths ``n_a`` and ``n_b`` (two 16-bit halves), the
    bits of the row's friction coefficient and two zero words.
    """
    return (
        pack_mf_header("record_a", dof_a, dof_b, diagonal, rhs, kind, parent, indent) + f"\n{indent}int4 record_b;\n"
        f"{indent}record_b.x = {n_a} | ({n_b} << 16);\n"
        f"{indent}record_b.y = __float_as_int({mu});\n"
        f"{indent}record_b.z = 0;\n"
        f"{indent}record_b.w = 0;\n"
        f"{indent}*reinterpret_cast<int4*>({record}) = record_a;\n"
        f"{indent}*reinterpret_cast<int4*>({record} + 4) = record_b;"
    )


def record_word(row: str, word: int, base: str) -> str:
    """Word ``word`` of row ``row``'s record, the records of its class from ``base`` (a native ``int`` offset)."""
    return f"row_record.data[{base} + ({row}) * {RECORD_WORDS} + {word}]"


def load_row_record(a: str, b: str, row: str, base: str, indent: str) -> str:
    """Load row ``row``'s record into the ``int4`` locals ``a`` and ``b`` (the class's records from ``base``)."""
    at = f"&row_record.data[{base} + ({row}) * {RECORD_WORDS}]"
    return (
        f"{indent}{a} = *reinterpret_cast<const int4*>({at});\n{indent}{b} = *reinterpret_cast<const int4*>({at} + 4);"
    )
