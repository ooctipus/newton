# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The route table: the row laws a solver's worlds solve and the placement case.

The row laws (:class:`Laws`) are read once from the physics options and the model, never from a route; every
placement's writer and sweep call each law's function, so no placement declines a law and an owner tests only its own
fit (topology, sizes, the device's budget). The placement axis (:class:`RouteTable`) gives the class to one case,
decided once in the precedence :data:`CASES`: an owner reads the case instead of the other owners' decisions. The data
that keep every class general are read once, here: a test pin, the device, and the bilateral pre-elimination.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = ["CASES", "Laws", "RouteRow", "RouteTable"]


@dataclass(frozen=True)
class Laws:
    """Axis 2: the row laws, from the physics options and the model (never from a route)."""

    regularized: bool
    """Contact rows step toward the hard law with a float32 weight below one (``pgs_contact_regularization``)."""
    torsion: bool
    """Spin rows may follow friction pairs: on a patch region with a footprint (one wrench a region), or on every
    friction contact under ``contact_torsion``; the sweeps read each contact's rows from the data."""
    compliance: bool
    """Material contacts take the row law's compliance (``contact_compliance``)."""
    limits: bool
    """Joint position limits take rows (``enable_joint_limits``)."""
    velocity_limits: bool
    """Joint velocity limits take rows (``enable_joint_velocity_limits``)."""
    warm_start: bool
    """Contact impulses carry from the previous step."""
    preelimination: bool
    """Bilateral rows are pre-eliminated through a Schur complement."""
    mimic: bool
    """The model has mimic rows."""

    @property
    def weighted(self) -> bool:
        """Contact rows step by their weights: the relative regularization's, or a material contact's from its
        compliance (every placement's sweep reads them)."""
        return self.regularized or self.compliance

    @classmethod
    def build(cls, laws, options, model, *, preelimination: bool) -> Laws:
        """Read the laws from the row laws' tables (:class:`.rows.RowLaws`: friction, regularization, torsion, mimic
        and link plans), the options and the model, with the bilateral pre-elimination decision."""
        return cls(
            regularized=bool(laws.regularization_enabled),
            torsion=bool(laws.spin_rows),
            compliance=bool(options.contact_compliance),
            limits=bool(options.enable_joint_limits),
            velocity_limits=bool(options.enable_joint_velocity_limits),
            warm_start=bool(options.pgs_warmstart),
            preelimination=bool(preelimination),
            mimic=bool(laws.mimic_count),
        )

    def record(self) -> dict:
        """The laws as a JSON-compliant record, with the derived ones."""
        return {**asdict(self), "weighted": self.weighted}


CASES = {
    "K2": "the sparse diagonal of a tree beside a star",
    "K1": "native sparse factors of a single tree",
    "K5": "factor coordinates of a tree beside one free body",
    "K4": "the Jacobi tiers of dense trees beside a free body",
}
"""The placement axis's specialized cases (placement and producer by class shape), in precedence order: the first
whose owner fits the class claims it. A class no case claims keeps the general placement, case K6."""


class RouteTable:
    """The placement axis: the case of the solver's class, decided once in :data:`CASES` order, and the route row.

    Construction asks the owners in that order. Each tests only its own fit (topology, sizes, capability, the device's
    budget) after :meth:`taken`, which says why an earlier case holds the class, :meth:`claim` gives the class to
    the first that fits, and each writes its case's reason. The producers and the runtime claims read :attr:`case`;
    once they have decided, :meth:`finish` returns the route, which the step's spec holds (:class:`.step_plan.StepSpec`).
    Three data keep every class general, read once here: the test pin ``placement="general"``; a device without the
    cases' native bodies (``native``: they are written for CUDA, so the CPU runs the general placement); and the
    bilateral pre-elimination (``eliminates``), whose blocks the one writer's block phase closes and whose wide rows
    only the general sweep's exit reads (the cases' writers and sweeps take neither yet).
    """

    def __init__(self, pins, *, native: bool, eliminates: bool):
        self.case = "K6"
        """The case holding the class: a key of :data:`CASES`, or ``"K6"`` (general) while none claims it."""
        self.reasons: dict[str, str | None] = {}
        """Why each asked case holds the class or does not, written by its owner."""
        general = (
            (pins.get("placement") == "general", "A test pin keeps the general placement (placement='general')."),
            (not native, "The device runs the general placement: the cases' native bodies are written for CUDA."),
            (eliminates, "Bilateral pre-elimination closes its blocks in the general row's block phase."),
        )
        self._general = next((reason for holds, reason in general if holds), None)
        """Why every class keeps the general placement, or ``None``."""
        self._asked = -1

    def taken(self, case: str) -> str | None:
        """Why ``case`` may not claim the class (the general pin, or an earlier case holds it), or ``None``."""
        order = list(CASES).index(case)
        if order < self._asked:
            raise RuntimeError(f"Case {case} is asked after a later case; owners are asked in CASES order.")
        self._asked = order
        if self._general:
            return self._general
        if self.case != "K6":
            return f"The class is case {self.case}: {CASES[self.case]}."
        return None

    def claim(self, case: str, fits: bool = True) -> bool:
        """Give the class to ``case`` (the case :meth:`taken` last asked about) when its own fit holds; return
        whether it holds the class."""
        if fits:
            if list(CASES).index(case) != self._asked or self.case != "K6" or self._general:
                raise RuntimeError(f"Case {case} claims a class it was not offered.")
            self.case = case
        return fits

    def finish(self, laws: Laws, **axes) -> RouteRow:
        """The route of the solver's class from the laws, the decided case and its reasons, and the other ``axes``
        (:class:`RouteRow`'s fields)."""
        return RouteRow(laws=laws, case=self.case, reasons=dict(self.reasons), **axes)


@dataclass(frozen=True)
class RouteRow:
    """The solver's route: the row of the route table its class takes, fixed once every owner has decided.

    The step's spec holds it (:attr:`.step_plan.StepSpec.route`); the step plan and the route record read it there,
    not the owners' state.
    """

    laws: Laws
    """Axis 2: the row laws."""
    case: str
    """Axis 3: the placement case (:data:`CASES`, or ``"K6"`` general)."""
    producer: str
    """Axis 3: the producer of the contact rows, decided once at construction: ``walk`` (the one writer), ``keyed_rows``
    (the tiers' in-kernel rows) or ``none`` (a class without a responding coordinate admits no contact)."""
    packed: bool
    """Axis 3: the matrix-free sweeps read the dense rows packed by support, so no size builds a response."""
    tier_rows: int
    """Axis 4: the dense rows of the worlds the Jacobi tiers own (case K4; 0 without tiers)."""
    sizes: dict[str, int | None]
    """The case owners' sizes: ``"packed"`` (K1's factored size), ``"diagonal"`` and ``"dense"`` (K2's pair)."""
    reasons: dict[str, str | None]
    """Why each case holds the class or does not (:attr:`RouteTable.reasons`)."""

    def record(self) -> dict:
        """The row's axes and each case's reason as a JSON-compliant record (``routes.row``; the laws are
        ``routes.laws``)."""
        return {name: value for name, value in asdict(self).items() if name != "laws"}
