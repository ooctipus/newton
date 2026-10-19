# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The velocity-limit row law that the serial sweeps' code generators share.

A joint or free-body velocity-limit row is unilateral, ``r = J v + b >= 0``,
with ``b`` the negated speed limit and ``d`` the row's diagonal. Two laws decide
what a sweep does with it:

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

Every generator emits its accumulated sites through :func:`accumulated_update`
(or :func:`accumulated_impulse` where a site computes the change itself), so the
law is written once. The stateless sites keep their existing text, which keeps
the generated sources of the stateless law byte-identical.
"""

VELOCITY_LIMIT_LAWS = ("stateless", "accumulated")


def check_velocity_limit_law(law: str, *, source: str = "velocity_limit_law") -> str:
    """Return ``law`` when it names a velocity-limit law.

    Args:
        law: The requested law.
        source: The argument or environment variable that supplied ``law``, named in the error.

    Raises:
        ValueError: If ``law`` is not one of :data:`VELOCITY_LIMIT_LAWS`.
    """
    if law not in VELOCITY_LIMIT_LAWS:
        raise ValueError(f"{source} must be one of {VELOCITY_LIMIT_LAWS}, got {law!r}")
    return law


def velocity_limit_law_suffix(law: str) -> str:
    """Kernel-name suffix of a law: empty for the stateless law, whose kernels keep their names."""
    return "" if check_velocity_limit_law(law) == "stateless" else "_vacc"


def accumulated_impulse(old: str, step: str) -> str:
    """CUDA expression of an accumulated velocity-limit impulse, ``max(old - step, 0)`` with ``step = r / d``."""
    return f"fmaxf({old} - {step}, 0.0f)"


def accumulated_update(old: str, step: str, *, new: str, delta: str, indent: str, declare: str = "") -> str:
    """CUDA statements that set ``new`` to the accumulated impulse and ``delta`` to its change ``new - old``.

    Args:
        old: Expression of the row's stored impulse.
        step: Expression of the unrelaxed step ``r / d`` (or ``r * inverse_diagonal``).
        new: Variable that receives the new impulse.
        delta: Variable that receives the applied change.
        indent: Leading whitespace of both statements.
        declare: Type prefix of both statements, such as ``"float "``; empty assigns existing variables.
    """
    return f"{indent}{declare}{new} = {accumulated_impulse(old, step)};\n{indent}{declare}{delta} = {new} - {old};\n"
