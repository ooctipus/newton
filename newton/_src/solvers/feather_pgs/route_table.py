# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The route table's laws axis: the row laws a solver's worlds solve, and the laws each placement implements.

The row laws (:class:`Laws`) are read once from the physics options and the model, never from a route. Each
placement and schedule declares once, in :data:`CAPABILITIES`, the laws it implements, and :func:`capable` is the one
test of them that the owners' admissions read: an owner then tests only its own topology fit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = ["CAPABILITIES", "Laws", "capable"]


@dataclass(frozen=True)
class Laws:
    """Axis 2: the row laws, from the physics options and the model (never from a route)."""

    contact: str
    """``"patch"`` (persistent friction patches, ``friction_anchor_beta > 0``) or ``"point"`` friction."""
    regularized: bool
    """Contact rows step toward the hard law with a float32 weight below one (``pgs_contact_regularization``)."""
    torsion: bool
    """Contact torsion appends a spin row per touching group."""
    compliance: bool
    """Compliant contacts replace the hard normal law."""
    limits: bool
    """Joint position limits take rows (``enable_joint_limits``)."""
    velocity_limits: bool
    """Joint velocity limits take rows (``enable_joint_velocity_limits``)."""
    warm_start: bool
    """Contact impulses carry from the previous step."""
    debug_rows: bool
    """``pgs_debug`` keeps row snapshots and a host-driven sweep."""
    preelimination: bool
    """Bilateral rows are pre-eliminated through a Schur complement."""
    mimic: bool
    """The model has mimic rows."""
    connect: bool
    """The model has connect rows."""
    loops: bool
    """The model has loop-closing joints."""
    iterations: bool
    """The position solve has a positive iteration budget."""
    forward_cuda: bool
    """CUDA without gradient recording: native kernels without adjoints may run."""

    PLAIN = "no warm start, debug rows or bilateral pre-elimination"

    @property
    def plain(self) -> bool:
        """The sweep policy the specialized placements assume (:attr:`PLAIN`)."""
        return not (self.warm_start or self.debug_rows or self.preelimination)

    @property
    def agnostic(self) -> bool:
        """The contact rows follow the point or the patch row law, which every row writer implements (no torsion
        row, no compliant normal)."""
        return not (self.torsion or self.compliance)

    @property
    def point(self) -> bool:
        """Point-friction rows of the point row law."""
        return self.agnostic and self.contact == "point"

    @property
    def patch(self) -> bool:
        """Patch-friction rows of the patch row law."""
        return self.agnostic and self.contact == "patch"

    @classmethod
    def build(cls, laws, options, model, *, preelimination: bool) -> Laws:
        """Read the laws from the row laws' tables (:class:`.rows.RowLaws`: friction, regularization, torsion, mimic
        and connect plans), the options and the model, with the bilateral pre-elimination decision."""
        return cls(
            contact="patch" if laws.friction_anchors_enabled else "point",
            regularized=bool(laws.regularization_enabled),
            torsion=bool(laws.contact_torsion_enabled),
            compliance=bool(options.contact_compliance),
            limits=bool(options.enable_joint_limits),
            velocity_limits=bool(options.enable_joint_velocity_limits),
            warm_start=bool(options.pgs_warmstart),
            debug_rows=bool(options.pgs_debug),
            preelimination=bool(preelimination),
            mimic=bool(laws.mimic_count),
            connect=bool(laws.connect_count),
            loops=bool(laws.forest.has_loop_joints),
            iterations=options.pgs_iterations > 0,
            forward_cuda=bool(model.device.is_cuda and not model.requires_grad),
        )

    def record(self) -> dict:
        """The laws as a JSON-compliant record, with the derived ones."""
        return {**asdict(self), "plain": self.plain, "agnostic": self.agnostic}


# Each requirement a placement or schedule may hold: (test of the laws, what it requires).
_REQUIREMENTS = {
    "forward_cuda": (lambda laws: laws.forward_cuda, "CUDA without gradient recording"),
    "plain": (lambda laws: laws.plain, Laws.PLAIN),
    "agnostic": (lambda laws: laws.agnostic, "the point or patch row law (no torsion, no compliance)"),
    "unregularized": (lambda laws: not laws.regularized, "the unregularized contact law"),
    "limits": (lambda laws: laws.limits, "joint-limit rows (enable_joint_limits)"),
    "no_velocity_limits": (lambda laws: not laws.velocity_limits, "no joint velocity-limit rows"),
    "no_torsion": (lambda laws: not laws.torsion, "no torsion row"),
    "no_compliance": (lambda laws: not laws.compliance, "no compliant contacts"),
    "no_mimic": (lambda laws: not laws.mimic, "no mimic rows"),
    "no_connect": (lambda laws: not laws.connect, "no connect rows"),
    "no_loops": (lambda laws: not laws.loops, "no loop-closing joints"),
    "iterations": (lambda laws: laws.iterations, "a positive iteration budget"),
    "no_debug_rows": (lambda laws: not laws.debug_rows, "no debug row snapshots"),
    "no_preelimination": (lambda laws: not laws.preelimination, "no bilateral pre-elimination"),
}

CAPABILITIES = {
    # The held tree's evaluation of the dynamics (its topology fit is the held groups' own proof).
    "held": ("forward_cuda", "no_connect", "no_loops", "no_debug_rows"),
    # The packed placement of a single tree: native sparse factors and their sweep.
    "packed": (
        "forward_cuda",
        "iterations",
        "no_debug_rows",
        "unregularized",
        "no_preelimination",
        "no_torsion",
        "no_compliance",
        "no_connect",
        "no_loops",
    ),
    # On-chip blocks of a world's rows (local owners), claimed per world at run time.
    "local": ("forward_cuda", "plain", "no_velocity_limits", "no_torsion"),
    # The local owners' row producer: limit prefix and contact packets built in place.
    "row_packets": ("agnostic", "limits", "unregularized"),
    # The paired factor's sweep in factor coordinates (one articulation and one free body per world).
    "paired_factor": ("plain", "unregularized", "iterations", "no_velocity_limits", "no_torsion"),
    # Its runtime claim: independent articulation and free-body components per world.
    "components": ("forward_cuda", "plain", "unregularized", "no_compliance", "no_mimic", "no_connect"),
    # The keyed in-kernel row producer of the tiers.
    "keyed_rows": ("forward_cuda", "plain", "limits", "iterations", "no_loops", "no_mimic", "no_connect"),
    # The compact boundary's row producer of a star beside a six-coordinate arm.
    "compact": ("agnostic",),
}
"""The laws each placement or schedule implements, by owner: what its admission reads through :func:`capable`."""

_LABELS = {
    "held": "The held tree needs",
    "packed": "Native sparse factors need",
    "local": "Local owners need",
    "row_packets": "Local packets need",
    "paired_factor": "The factor-coordinate sweep needs",
    "components": "Components need",
    "keyed_rows": "Keyed rows need",
    "compact": "Compact contacts need",
}


def capable(owner: str, laws: Laws) -> str | None:
    """Why ``owner`` does not implement the solver's ``laws``, or ``None`` when it does (:data:`CAPABILITIES`)."""
    for requirement in CAPABILITIES[owner]:
        holds, phrase = _REQUIREMENTS[requirement]
        if not holds(laws):
            return f"{_LABELS[owner]} {phrase}."
    return None
