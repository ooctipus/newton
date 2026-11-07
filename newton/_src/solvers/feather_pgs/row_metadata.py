# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared packed dense and matrix-free row metadata contracts, and the row record the sweeps stream."""

from .row_laws import ROW_KINDS

_RECORD_MAX_ROWS = 65535  # Parent values 0..65534; 0xffff encodes no parent.


def validate_record_rows(rows: int) -> None:
    """Reject a line of ``rows`` rows whose parents a record's 16-bit half cannot hold (0xFFFF is none)."""
    if not 0 <= rows <= _RECORD_MAX_ROWS:
        raise ValueError(f"mf_max_constraints or dense_max_constraints past a record's {_RECORD_MAX_ROWS} rows: {rows}")
    if ROW_KINDS > 65536:
        raise RuntimeError("a row's record does not have enough row-type bits")


def record_parent(word: str, rows: int) -> str:
    """A row's parent (-1 for none) from its record's kind and parent word ``word``, in a line of ``rows`` rows."""
    if rows <= 32768:
        return f"{word} >> 16"
    # Adding one modulo 2^16 maps the 0xffff sentinel to zero before signed
    # subtraction; valid high parent indices never undergo sign extension.
    return f"static_cast<int>(((static_cast<unsigned int>({word}) >> 16) + 1u) & 0xffffu) - 1"


def record_kind(kind: str, parent: str) -> str:
    """A record's kind and parent word: the row type in the low and the parent (0xFFFF for none) in the high half."""
    return f"static_cast<int>(static_cast<unsigned int>({kind}) | (static_cast<unsigned int>({parent}) << 16))"


def unpack_mf_header(word: str, indent: str) -> str:
    """Declare a free-body row's fields from its record's header ``word`` (:func:`pack_row_record`): ``dof_a``,
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
"""Words of a dense row's record (:func:`pack_row_record`): two ``int4``, which a row loop loads a row ahead."""

HEADER_WORDS = 4
"""Words of a free row's record: its header alone (:func:`pack_row_record`), one ``int4``: its sides' widths are its
class's, its friction coefficient the line's (``row_mu``)."""


def record_words(free: bool) -> int:
    """Words of a row's record by its class: :data:`HEADER_WORDS` a free row, :data:`RECORD_WORDS` a dense one."""
    return HEADER_WORDS if free else RECORD_WORDS


def record_layout(first: int, base: int, pitch: int) -> tuple[int, int]:
    """A line's records from its row ``first`` (the dense rows' first where their metadata streams, ``base`` where only
    the free rows' stream, ``pitch`` where none): the free rows' first word (the dense rows' records before them,
    :data:`RECORD_WORDS` each) and the line's words (then the free rows' records, :data:`HEADER_WORDS` each)."""
    first, base, pitch = int(first), int(base), int(pitch)
    dense = max(min(base, pitch) - first, 0)
    free = max(pitch - max(first, base), 0)
    validate_record_rows(max(base if dense else 0, free))
    return dense * RECORD_WORDS, dense * RECORD_WORDS + free * HEADER_WORDS


def record_views(records, free_first: int):
    """A host array of lines' records (``[lines, words]``) as the dense rows' (``[lines, rows, RECORD_WORDS]``) and the
    free rows' (``[lines, rows, HEADER_WORDS]``) from the free rows' first word ``free_first`` (:func:`record_layout`)."""
    lines = records.shape[0]
    dense = records[:, :free_first].reshape(lines, -1, RECORD_WORDS)
    free = records[:, free_first:].reshape(lines, -1, HEADER_WORDS)
    return dense, free


def validate_record_coordinates(max_world_dofs: int) -> None:
    """Reject worlds whose coordinates a record's 16-bit halves cannot hold (its sides' first world coordinates)."""
    if int(max_world_dofs) > 32767:
        raise ValueError(f"a row's record holds world coordinates below 32768; a world holds {max_world_dofs}")


def pack_row_record(
    record: str,
    words: int,
    *,
    dof_a: str = "dof_a",
    dof_b: str = "dof_b",
    diagonal: str = "diagonal",
    rhs: str = "rhs",
    kind: str = "kind",
    parent: str = "parent",
    n_a: str = "n_a",
    n_b: str = "n_b",
    mu: str = "mu",
    indent: str = "    ",
) -> str:
    """Store a row's record of ``words`` words (:func:`record_words`) at the ``int`` pointer ``record``: its header, its
    sides' first world coordinates ``dof_a`` and ``dof_b`` (16-bit halves, -1 for none), the bits of its stored
    ``diagonal`` (a free row's inverse) and of its ``rhs`` at ``du = 0``, and its kind and parent word
    (:func:`record_kind`); a dense row's then its sides' widths ``n_a`` and ``n_b`` (two 16-bit halves), its friction
    coefficient ``mu`` and two zero words. Each field is a native expression, by default the local of its name."""
    if words not in (HEADER_WORDS, RECORD_WORDS):
        raise ValueError(f"a record holds {HEADER_WORDS} or {RECORD_WORDS} words, not {words}")
    text = (
        f"{indent}int4 record_a;\n"
        f"{indent}record_a.x = ({dof_a} << 16) | ({dof_b} & 0xFFFF);\n"
        f"{indent}record_a.y = __float_as_int({diagonal});\n"
        f"{indent}record_a.z = __float_as_int({rhs});\n"
        f"{indent}record_a.w = {record_kind(kind, parent)};"
    )
    if words == HEADER_WORDS:
        return text + f"\n{indent}*reinterpret_cast<int4*>({record}) = record_a;"
    return text + (
        f"\n{indent}int4 record_b;\n"
        f"{indent}record_b.x = {n_a} | ({n_b} << 16);\n"
        f"{indent}record_b.y = __float_as_int({mu});\n"
        f"{indent}record_b.z = 0;\n"
        f"{indent}record_b.w = 0;\n"
        f"{indent}*reinterpret_cast<int4*>({record}) = record_a;\n"
        f"{indent}*reinterpret_cast<int4*>({record} + 4) = record_b;"
    )


def record_word(row: str, word: int, base: str, words: int = RECORD_WORDS) -> str:
    """Word ``word`` of row ``row``'s record, the records of its class (``words`` each) from ``base`` (a native ``int``
    offset)."""
    if not 0 <= word < words:
        raise ValueError(f"a record of {words} words has no word {word}")
    return f"row_record.data[{base} + ({row}) * {words} + {word}]"


def load_row_record(a: str, b: str | None, row: str, base: str, indent: str, words: int = RECORD_WORDS) -> str:
    """Load row ``row``'s record into the ``int4`` locals ``a`` and ``b`` (the class's records, ``words`` each, from
    ``base``): a header's (``b`` None) into ``a`` alone."""
    at = f"&row_record.data[{base} + ({row}) * {words}]"
    text = f"{indent}{a} = *reinterpret_cast<const int4*>({at});"
    if b is None:
        if words != HEADER_WORDS:
            raise ValueError("a record past its header loads both its words")
        return text
    return text + f"\n{indent}{b} = *reinterpret_cast<const int4*>({at} + 4);"
