# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One experimental simultaneous limit prefix; ordered contacts stay unchanged.

FP64 CPU screen of a fixed map, not a literal GPU emulator. The scale-aware
negative-energy guard uses a 64*u32 margin. It certifies neither geometry nor
every FP32 operation; independent endpoint physics is the qualification gate.
No damping/line search, matrix factor, new reference, or extra outer pass.
"""

import hashlib
import inspect
from pathlib import Path

import numpy as np

from tools.fpgs_bench import spectral_gs_control as ordered

BASE_SOURCE_SHA = hashlib.sha256(Path(ordered.__file__).read_bytes()).hexdigest()
assert BASE_SOURCE_SHA == "a39eb83e4f5b240a24416f96683df0cc5e7453c8f180cbf539f9d752141256c7"
GUARD_RELATIVE_MARGIN = 64.0 * 2.0**-24


def prefix_proposal(z, old, residual, diagonal):
    """Scratch-only proposal; caller commits only on accepted=True."""
    old = np.asarray(old, float)
    residual = np.asarray(residual, float)
    diagonal = np.asarray(diagonal, float)
    info = {
        "zero_delta": False,
        "energy": None,
        "linear": None,
        "kinetic_norm2": None,
        "margin": None,
        "scaled_energy": None,
        "reason": "nonfinite_or_ineligible",
    }
    if not all(np.isfinite(x).all() for x in (z, old, residual, diagonal)) or np.any(diagonal <= 0) or np.any(old < 0):
        return old.copy(), np.zeros_like(old), False, info
    value = np.maximum(old - residual / diagonal, 0.0)
    delta = value - old
    if not np.isfinite(delta).all():
        return old.copy(), np.zeros_like(old), False, info
    if not np.any(delta):
        info.update(zero_delta=True, reason="zero", energy=0.0, linear=0.0, kinetic_norm2=0.0, margin=0.0)
        return value, delta, True, info
    kinetic = np.asarray(z, float).T @ delta
    linear = float(residual @ delta)
    norm2 = float(kinetic @ kinetic)
    energy = linear + 0.5 * norm2
    scale = abs(linear) + 0.5 * norm2
    margin = GUARD_RELATIVE_MARGIN * scale
    finite = bool(np.isfinite(kinetic).all() and np.isfinite([linear, norm2, energy, scale, margin]).all())
    accepted = finite and energy <= -margin
    info.update(
        energy=energy,
        linear=linear,
        kinetic_norm2=norm2,
        margin=margin,
        scaled_energy=energy / scale if finite and scale else None,
        reason="accept" if accepted else "energy_or_nonfinite",
    )
    return value, delta, accepted, info


def _build():
    source = inspect.getsource(ordered.solve)
    anchor = "    for iteration in range(iterations):\n        changed = False\n"
    assert source.count(anchor) == 1
    source = source.replace(
        anchor,
        """    limit_rows=np.flatnonzero(types==3)
    if not np.array_equal(limit_rows,np.arange(len(limit_rows))):
        raise ValueError('Screen requires the existing contiguous limit prefix')
    work.update(limit_attempts=0,limit_accepts=0,limit_zero=0,limit_fallbacks=0,
        limit_proposal_row_dots=0,limit_guard_scalar_reductions=0,
        limit_guard_products=0,limit_candidate_response_row_actions=0,
        limit_rejected_response_row_actions=0,limit_serial_fallback_rows=0,
        limit_curves=[],limit_count=int(len(limit_rows)))
    for iteration in range(iterations):
        changed = False
        prefix_handled=False
        if len(limit_rows):
            r_limit=J[limit_rows]@velocity+rhs[limit_rows]
            old_limit=impulse[limit_rows].copy()
            proposal,change,accepted,guard=prefix_proposal(Z[limit_rows],old_limit,r_limit,diagonal[limit_rows])
            work['limit_attempts']+=1
            work['limit_proposal_row_dots']+=len(limit_rows)
            changed_rows=int(np.count_nonzero(change))
            if not guard['zero_delta']:
                work['limit_guard_scalar_reductions']+=2
                work['limit_guard_products']+=len(limit_rows)+dofs
                work['limit_candidate_response_row_actions']+=changed_rows
                if not accepted: work['limit_rejected_response_row_actions']+=changed_rows
            work['limit_curves'].append(dict(sweep=iteration+1,changed_rows=changed_rows,accepted=bool(accepted),**guard))
            if accepted:
                prefix_handled=True
                work['limit_accepts']+=1
                work['limit_zero']+=int(guard['zero_delta'])
                if changed_rows:
                    velocity+=Y[limit_rows].T@change
                    changed=True
                impulse[limit_rows]=proposal
            else:
                work['limit_fallbacks']+=1
                work['limit_serial_fallback_rows']+=len(limit_rows)
""",
    )
    anchor = "        for row in np.flatnonzero(types != 2):"
    assert source.count(anchor) == 1
    source = source.replace(
        anchor, "        for row in np.flatnonzero((types != 2) & ((types != 3) | (not prefix_handled))):"
    )
    namespace = dict(ordered.__dict__, prefix_proposal=prefix_proposal)
    exec(compile(source, "<limit_prefix_only_cpu_screen>", "exec"), namespace)
    return namespace["solve"]


solve = _build()
