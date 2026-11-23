# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared packed dense and matrix-free row metadata contracts, and the row record the sweeps stream."""

from .row_laws import ROW_KINDS

_RECORD_MAX_ROWS = 65535  # Parent values 0..65534; 0xffff encodes no parent.


_RECORD_MAX_DOFS = 32767  # A side's first world coordinate in a signed 16-bit half.


def split_records(world_dofs: int, rows: int) -> bool:
    """Whether a line's records are split (:func:`pack_row_record`): each side's first world coordinate and the row's
    parent in a word of their own, where a world holds more than ``_RECORD_MAX_DOFS`` coordinates (``world_dofs``) or a
    class more than ``_RECORD_MAX_ROWS`` rows (``rows``), which the packed record's 16-bit halves cannot hold. The
    data sizes the record: a line that fits keeps the packed record."""
    return int(world_dofs) > _RECORD_MAX_DOFS or int(rows) > _RECORD_MAX_ROWS


def validate_record_rows(rows: int, split: bool = False) -> None:
    """Reject a line of ``rows`` rows whose parents a packed record's 16-bit half cannot hold (0xFFFF is none); a split
    record holds its parent in a word."""
    if not 0 <= rows <= (2**31 - 1 if split else _RECORD_MAX_ROWS):
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


def unpack_mf_header(word: str, indent: str, last: str | None = None) -> str:
    """Declare a free-body row's fields from its record's header ``word`` (:func:`pack_row_record`): ``dof_a``,
    ``dof_b``, ``mf_diag`` (the stored inverse diagonal), ``packed_tp`` and ``mf_rt`` (the row type); a split
    record's ``dof_b`` from its last quad ``last``."""
    sides = (
        f"{indent}int dof_a = {word}.x;\n{indent}int dof_b = {last}.x;\n"
        if last
        else (
            f"{indent}int packed_dofs = {word}.x;\n"
            f"{indent}int dof_a = packed_dofs >> 16;\n"
            f"{indent}int dof_b = (packed_dofs << 16) >> 16;\n"
        )
    )
    return (
        sides + f"{indent}float mf_diag = __int_as_float({word}.y);\n"
        f"{indent}int packed_tp = {word}.w;\n"
        f"{indent}int mf_rt = packed_tp & 0xFFFF;"
    )


RECORD_WORDS = 8
"""Words of a dense row's packed record (:func:`pack_row_record`): two ``int4``, which a row loop loads a row ahead."""

HEADER_WORDS = 4
"""Words of a free row's packed record: its header alone (:func:`pack_row_record`), one ``int4``: its sides' widths are
its class's, its friction coefficient the line's (``row_mu``)."""

SPLIT_WORDS = 4
"""Words a split record (:func:`split_records`) adds after the packed one's: one ``int4``, its second side's first world
coordinate and its parent (:func:`pack_row_record`)."""


def record_words(free: bool, split: bool = False) -> int:
    """Words of a row's record by its class: :data:`HEADER_WORDS` a free row, :data:`RECORD_WORDS` a dense one, and a
    split record's :data:`SPLIT_WORDS` more."""
    return (HEADER_WORDS if free else RECORD_WORDS) + (SPLIT_WORDS if split else 0)


def split_word(words: int, word: int) -> int:
    """Word ``word`` of a split record's last quad (0: the second side's first coordinate, 1: the parent) in a record
    of ``words`` words."""
    return int(words) - SPLIT_WORDS + int(word)


def record_layout(first: int, base: int, pitch: int, split: bool = False) -> tuple[int, int]:
    """A line's records from its row ``first`` (the dense rows' first where their metadata streams, ``base`` where only
    the free rows' stream, ``pitch`` where none): the free rows' first word (the dense rows' records before them,
    :func:`record_words` each) and the line's words (then the free rows' records); ``split`` (:func:`split_records`)."""
    first, base, pitch = int(first), int(base), int(pitch)
    dense = max(min(base, pitch) - first, 0)
    free = max(pitch - max(first, base), 0)
    validate_record_rows(max(base if dense else 0, free), split)
    dense_words = dense * record_words(False, split)
    return dense_words, dense_words + free * record_words(True, split)


def record_views(records, free_first: int, split: bool = False):
    """A host array of lines' records (``[lines, words]``) as the dense rows' (``[lines, rows, words]``) and the free
    rows' from the free rows' first word ``free_first`` (:func:`record_layout`; :func:`record_words` each)."""
    lines = records.shape[0]
    dense = records[:, :free_first].reshape(lines, -1, record_words(False, split))
    free = records[:, free_first:].reshape(lines, -1, record_words(True, split))
    return dense, free


def pack_row_record(
    record: str,
    words: int | str = "words",
    *,
    split: bool = False,
    dof_a: str = "dof_a",
    dof_b: str = "dof_b",
    diagonal: str = "diagonal",
    rhs: str = "rhs",
    kind: str = "kind",
    parent: str = "parent",
    n_a: str = "n_a",
    n_b: str = "n_b",
    mu: str = "mu",
    key: str = "key",
    root: str = "root",
    indent: str = "    ",
) -> str:
    """Store a row's record of ``words`` words (:func:`record_words`) at the ``int`` pointer ``record``: its header, its
    sides' first world coordinates ``dof_a`` and ``dof_b`` (16-bit halves, -1 for none), the bits of its stored
    ``diagonal`` (a free row's inverse) and of its ``rhs`` at ``du = 0``, and its kind and parent word
    (:func:`record_kind`); a dense row's then its sides' widths ``n_a`` and ``n_b`` (two 16-bit halves), its friction
    coefficient ``mu``, its idle ``key`` and its idle ``root`` (:func:`.projection_laws.idle_key`, ``idle_root``: the
    bounds the live order passes idle rows by, formed once with the record). Each field is a native expression, by default the local of its name, and
    ``words`` a class's count or, a native expression, the row's.

    A ``split`` record (:func:`split_records`, the line's) holds ``dof_a`` whole in the header's first word and no
    parent in its kind word, then a last quad of ``dof_b`` and ``parent`` (:func:`split_word`)."""
    if words not in (HEADER_WORDS, RECORD_WORDS) and not isinstance(words, str) and not split:
        raise ValueError(f"a record holds {HEADER_WORDS} or {RECORD_WORDS} words, not {words}")
    firsts, kind_parent = (dof_a, "-1") if split else (f"({dof_a} << 16) | ({dof_b} & 0xFFFF)", parent)
    text = (
        f"{indent}int4 record_a;\n"
        f"{indent}record_a.x = {firsts};\n"
        f"{indent}record_a.y = __float_as_int({diagonal});\n"
        f"{indent}record_a.z = __float_as_int({rhs});\n"
        f"{indent}record_a.w = {record_kind(kind, kind_parent)};"
    )
    text += f"\n{indent}*reinterpret_cast<int4*>({record}) = record_a;"
    tail = f"*reinterpret_cast<int4*>({record} + 4) = make_int4({n_a} | ({n_b} << 16), __float_as_int({mu}), __float_as_int({key}), __float_as_int({root}));"
    extra = SPLIT_WORDS if split else 0
    if isinstance(words, str):
        text += f"\n{indent}if ({words}{f' - {extra}' if extra else ''} > {HEADER_WORDS}) " + tail
    elif words - extra != HEADER_WORDS:
        text += f"\n{indent}" + tail
    if split:
        text += f"\n{indent}*reinterpret_cast<int4*>({record} + ({words}) - {SPLIT_WORDS}) = make_int4({dof_b}, {parent}, 0, 0);"
    return text


def record_word(row: str, word: int, base: str, words: int = RECORD_WORDS) -> str:
    """Word ``word`` of row ``row``'s record, the records of its class (``words`` each) from ``base`` (a native ``int``
    offset)."""
    if not 0 <= word < words:
        raise ValueError(f"a record of {words} words has no word {word}")
    return f"row_record.data[{base} + ({row}) * {words} + {word}]"


def load_row_record(names: tuple[str, ...], row: str, base: str, indent: str, words: int = RECORD_WORDS) -> str:
    """Load row ``row``'s record into the ``int4`` locals ``names``, a quad each (the class's records, ``words`` each,
    from ``base``): a packed free row's header into one, a packed dense record into two, a split one's into one more."""
    if len(names) * 4 != words:
        raise ValueError(f"a record of {words} words loads into {words // 4} quads, not {len(names)}")
    at = f"&row_record.data[{base} + ({row}) * {words}]"
    loads = [
        f"{indent}{name} = *reinterpret_cast<const int4*>({at}{f' + {4 * q}' if q else ''});"
        for q, name in enumerate(names)
    ]
    return "\n".join(loads)
