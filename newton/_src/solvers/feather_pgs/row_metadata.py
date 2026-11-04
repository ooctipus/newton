# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared packed dense and matrix-free row metadata contracts."""

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
