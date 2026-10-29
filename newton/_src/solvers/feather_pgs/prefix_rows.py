# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The prefix rows: one native primitive for a world's mimic, joint-limit and joint velocity-limit rows.

A world's dense rows open with its prefix: the position rows (mimic equalities and joint limits), then the joint
velocity limits (``dense_phase_bounds`` holds both ends). :func:`prefix_rows` is the candidate loop of every native
prefix producer: a lane group walks an articulation's candidates in order, takes each one's law from
:data:`.row_laws.ROW_LAW`, places the active rows in candidate order and writes their records. The producers differ
in their families, in who owns the world's row counter and in the sink their consumer reads: the general route's
joint limits (the per-size Jacobian) and the sparse factor's (packed ``L^-1`` columns) reserve on the shared counter;
local row packets (mimic and limit rows into sparse coefficients) and keyed rows (limit and velocity-limit rows into
row keys) own their worlds' counters.
"""

import warp as wp

from . import field_group
from .dense_rows import RECORDS
from .device_preludes import substitute
from .field_group import Field

LAW = (
    Field("limit_q_index", wp.int32, ("dof",), attr="_joint_limit_q_index"),
    Field("lower", wp.float32, ("dof",), owner="model", attr="joint_limit_lower"),
    Field("upper", wp.float32, ("dof",), owner="model", attr="joint_limit_upper"),
    Field("q", wp.float32, ("coord",), owner="state_in", attr="joint_q"),
    Field("activation_gap", wp.float32, attr="joint_limit_activation_gap"),
    Field("beta", wp.float32, attr="pgs_beta"),
    Field("cfm", wp.float32, attr="pgs_cfm"),
    Field("dt", wp.float32, owner="step"),
    Field("slot_counter", wp.int32, ("world",)),
    Field("bounds", wp.int32, ("world", 2), attr="dense_phase_bounds"),
    *(field for field in RECORDS if field.name != "row_w"),
)
"""The prefix law's inputs, the world's row counter and phase bounds, and the dense row records."""

PrefixLaw = field_group.struct("PrefixLaw", LAW, __name__)

VELOCITY_LAW = (
    Field("velocity_limit", wp.float32, ("dof",), owner="model", attr="joint_velocity_limit"),
    Field("vhat", wp.float32, ("dof",), attr="v_hat"),
    Field("activation", wp.float32, attr="velocity_limit_activation_fraction"),
)
"""The joint velocity-limit family's inputs."""


def _family(coordinates, name: str, evaluation: str) -> tuple:
    """A coordinate family: both sides of each coordinate, ``coordinates`` a tuple or a native count from 0.

    A dynamically indexed local array survives CUDA compilation even when its values form an interval, so an
    interval becomes arithmetic.
    """
    declaration, coordinate = "", "index / 2"
    if isinstance(coordinates, str):
        count = f"2 * {coordinates}"
    elif coordinates and coordinates == tuple(range(coordinates[0], coordinates[-1] + 1)):
        count = 2 * len(coordinates)
        coordinate += f" + {coordinates[0]}" if coordinates[0] else ""
    else:
        count = 2 * len(coordinates)
        declaration = f"    const int {name}[] = {{{', '.join(map(str, coordinates)) or '0'}}};\n"
        coordinate = f"{name}[index / 2]"
    return count, declaration, f"const int d = {coordinate};\n{evaluation}"


def limits(coordinates, dof: str, law: str) -> tuple:
    """The joint-limit family; ``dof`` is coordinate ``d``'s joint DOF, ``law`` the accessor of :data:`LAW`."""
    return _family(
        coordinates,
        "limit_coordinates",
        f"""            dof = {dof};
            side = index & 1;
            const int qi = {law}limit_q_index.data[dof];
            if (qi >= 0) {{
                const auto limit = RowLaw::joint_limit_side({law}q.data[qi], {law}lower.data[dof],
                                                            {law}upper.data[dof], side, {law}activation_gap);
                active = limit.v0;
                phi = limit.v1;
                beta = {law}beta;
                d0 = d;
                w0 = limit.v2;
            }}""",
    )


def velocity_limits(coordinates, dof: str, law: str) -> tuple:
    """The joint velocity-limit family: a row pair with target ``-limit`` and no bias (:data:`VELOCITY_LAW`)."""
    return _family(
        coordinates,
        "velocity_coordinates",
        f"""            dof = {dof};
            side = index & 1;
            const float limit = {law}velocity_limit.data[dof];
            active = RowLaw::velocity_limit_active(limit, {law}vhat.data[dof], {law}activation);
            type = 4;
            target = -limit;
            d0 = d;
            w0 = side == 0 ? 1.0f : -1.0f;""",
    )


_PHASE = """    for (int chunk = 0; chunk < $CANDIDATES; chunk += STRIDE) {
        const int candidate = chunk + LANE;
        int active = 0, type = 3, dof = 0, side = 0, d0 = -1, d1 = -1;
        float phi = 0.0f, target = 0.0f, beta = 0.0f, w0 = 0.0f, w1 = 0.0f;
$LOCALS$EVALUATE        $SLOT
        if (active && slot < LAW::row_type.shape[1]) {
            const int at = world * LAW::row_type.shape[1] + slot;
            LAW::row_type.data[at] = type;
            LAW::row_parent.data[at] = -1;
            LAW::row_mu.data[at] = 0.0f;
            LAW::row_beta.data[at] = beta;
            LAW::row_cfm.data[at] = LAW::cfm;
            LAW::phi.data[at] = phi;
            LAW::target.data[at] = target;
$SIDES$RHS$SINK
        }
$AFTER    }
"""
"""One phase's candidate loop (``LAW::`` stands for the accessor of :data:`LAW`'s fields)."""

_RHS = """            LAW::restitution.data[at] = 0.0f;
            // The position pass's row law (bias scale 1): a limit's Baumgarte or speculative term, a mimic's
            // Baumgarte term, a velocity limit's target.
            LAW::rhs.data[at] = RowLaw::row_rhs(type, phi, beta, target, 1.0f / LAW::dt, 1.0f, 0.0f, 1.0f, 1.0f).v0;
"""


def prefix_rows(
    phases,
    *,
    law: str,
    ordered: bool,
    rhs: bool,
    sink: str = "",
    after: str = "",
    locals_: str = "",
    side: str = "",
) -> str:
    """The prefix producer's candidate loop over one articulation, as native code inside its lane group.

    Args:
        phases: The position families, then the velocity families: per phase a tuple of families
            ``(candidates, declaration, evaluation)``. An evaluation of ``index`` (the candidate within its family)
            sets ``active``, ``type`` (3, a limit, by default), ``phi``, ``target``, ``beta``, the coordinates and
            coefficients ``d0, w0`` and ``d1, w1``, and a coordinate family's ``dof`` and ``side``.
        law: The accessor of :data:`LAW`'s fields (``"data."``, say), which ``world`` indexes.
        ordered: The lane group owns the world's counter: rows take slots in candidate order from 0, and the group
            writes the phase bounds and the count. Otherwise the rows reserve on the shared counter (``votes``).
        rhs: Write each row's restitution and position-pass ``rhs`` too.
        sink: Native code that publishes an active row's coefficients at ``slot`` (record ``at``).
        after: Native code the whole lane group runs after each chunk of candidates (a cooperative sink).
        locals_: Native declarations of the families' and the sinks' own per-candidate variables.
        side: The native articulation the rows touch, recorded as each row's sides (:func:`.dense_rows.row_sides`)
            for the packed placement (:mod:`.packed_rows`); empty: a sink that keeps its rows elsewhere.
    """
    slot = (
        "const int slot = ORDERED(base, active != 0);"
        if ordered
        else "const unsigned votes = BALLOT(active != 0);\n"
        f"        const int slot = RESERVE(&{law}slot_counter.data[world], COUNT(votes)) + BELOW(votes);"
    )
    source = "    int base = 0;\n" if ordered else ""
    for phase, families in enumerate(phases):
        start, evaluate = 0, ""
        for count, declaration, evaluation in families:
            if start == 0:
                end, low, index = count, "", "candidate"
            else:
                end = start + count if isinstance(start, int) and isinstance(count, int) else f"{start} + {count}"
                low, index = f"candidate >= {start} && ", f"candidate - {start}"
            source += declaration
            evaluate += f"""        if ({low}candidate < {end}) {{
            const int index = {index};
            {evaluation}
        }}
"""
            start = end
        if start != 0:
            source += substitute(
                _PHASE,
                CANDIDATES=start,
                LOCALS=f"        {locals_}\n" if locals_ else "",
                EVALUATE=evaluate,
                SLOT=slot,
                SIDES=f"            LAW::sides.data[at] = wp::vec2i({side}, -1);\n" if side else "",
                RHS=_RHS if rhs else "",
                SINK=sink,
                AFTER=after,
            ).replace("LAW::", law)
        if ordered:
            source += f"    if (LANE == 0) {law}bounds.data[world * 2 + {phase}] = base;\n"
    if ordered:
        source += f"    if (LANE == 0) {law}slot_counter.data[world] = base;\n"
    return source
