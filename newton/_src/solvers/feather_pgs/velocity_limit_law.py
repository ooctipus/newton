# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The velocity-limit row law that the serial sweeps' code generators share.

A joint velocity-limit row is unilateral, ``r = J v + b >= 0``, with ``J`` signed
by the side of the limit it bounds, ``b`` set by the speed limit and ``d`` the
row's diagonal. A free body's velocity-limit row is a speed row: it bounds the
magnitude of the body's linear or angular velocity, ``r = b - |v| >= 0`` with
``b`` the limit, and its correction acts along ``v``. Two laws decide what a
sweep does with a row:

* ``"stateless"`` is PhysX's clamp. Each visit applies only the impulse that
  removes the current overshoot, ``max(-r / d, 0)``, and stores that correction
  instead of a total. An impulse that later rows make unnecessary is never
  released, so the converged velocity depends on the row order, and the limit
  brakes harder than the limit itself requires.
* ``"accumulated"`` keeps the row's impulse as a total, like a contact normal's
  or a position limit's: ``new = max(old - r / d, 0)``, unrelaxed. Sweeps then
  converge to the complementarity solution, which is independent of the row
  order and is the fixed point the parallel matrix-free tiers solve.

Split mode has no joint velocity-limit rows and applies neither law: its
free-body sweep applies the stateless clamp outside these generators, as
``routes.velocity_limits.free_body_sweep_law`` records.

Both laws are one law of :mod:`.projection_laws`,
:func:`~.projection_laws.velocity_limit_update` (and its free-body form
:func:`~.projection_laws.speed_row`), whose last argument selects the
accumulated total; every sweep generator calls it at each visit of a
velocity-limit row with the law this module checks.
"""

VELOCITY_LIMIT_LAWS = ("stateless", "accumulated")


def check_velocity_limit_law(law: str) -> str:
    """Return ``law`` when it names a velocity-limit law.

    Raises:
        ValueError: If ``law`` is not one of :data:`VELOCITY_LIMIT_LAWS`.
    """
    if law not in VELOCITY_LIMIT_LAWS:
        raise ValueError(f"velocity_limit_law must be one of {VELOCITY_LIMIT_LAWS}, got {law!r}")
    return law


def velocity_limit_law_suffix(law: str) -> str:
    """Kernel-name suffix of a law: empty for the stateless law, whose kernels keep their names."""
    return "" if check_velocity_limit_law(law) == "stateless" else "_vacc"
