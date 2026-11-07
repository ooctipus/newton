# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The route table: the row laws a solver's worlds solve and the route its class takes.

The row laws (:class:`Laws`) are read once from the physics options and the model, never from a route; every
placement's writer and sweep call each law's function, so no placement declines a law. The class's placement case is
decided by rules over data before its owners (:mod:`.class_plan`), each rule saying why the class does not fit by the
first check that fails (:func:`first_failed`); a test fixes a route by its pins (:class:`Pins`). The route
(:class:`RouteRow`) records the case with what the owners built from it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

__all__ = ["Laws", "Pins", "RouteRow", "first_failed"]


def first_failed(checks) -> str | None:
    """The reason of the first ``(holds, reason)`` check that fails, or ``None`` when every check holds."""
    return next((reason for holds, reason in checks if not holds), None)


@dataclass(frozen=True)
class Pins:
    """Test-only route pins: an axis value a test fixes where the rules would choose another, so that every route the
    data can select runs on the test scenes. Only a test passes them (``SolverFeatherPGS(..., pins=...)``); the spec
    holds them and the route record lists those in force (``routes.spec.pins``)."""

    storage: str | None = None
    """``"dense"``: no packed (native sparse) factor."""
    grouping: str | None = None
    """``"forest"`` or ``"per_size"``: the held families."""
    star: str | None = None
    """``"held"``: the compact diagonal family in a held tree group, never the one-coordinate class."""
    producer: str | None = None
    """``"walk"``: no keyed rows."""
    claims: tuple[str, ...] = ("resolved_worlds",)
    """The runtime claims kept, of ``("resolved_worlds",)``."""
    active_rows: bool = True
    """``False``: full-order sparse sweeps."""
    contact_units: str | None = "triplets"
    """``None``: unfused sparse contact triplets."""
    mf_storage: str | None = None
    """``"streamed"``: streamed matrix-free impulses, the free bodies'; ``"line"``: every row's, the line's too."""
    predicted_twists: bool = True
    """``False``: the held tree's predictor without the twists' publication."""
    row_support: bool = True
    """``False``: sparse rows on explicit indices."""
    placement: str | None = None
    """``"general"``: every class keeps the general placement, case K6 (:func:`.class_plan.general_reason`)."""
    tiers: int | None = None
    """The Jacobi tiers' row bound, on any class whose response they form (0: no tiers)."""
    tier_sweeps: int | None = None
    """Every Jacobi tier's sweep budget, in place of the equal-accuracy rule."""

    def record(self) -> dict:
        """The pins in force (those off their defaults) as a JSON-compliant record."""
        return {
            pin.name: list(value) if isinstance(value, tuple) else value
            for pin in fields(self)
            if (value := getattr(self, pin.name)) != pin.default
        }


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


@dataclass(frozen=True)
class RouteRow:
    """The solver's route: the row of the route table its class takes, fixed once every owner has built.

    The step's plan holds it (:attr:`.step_plan.StepPlan.route`); the stages and the route record read it there, not
    the owners' state.
    """

    laws: Laws
    """Axis 2: the row laws."""
    case: str
    """Axis 3: the placement case (:data:`.class_plan.RULES`, or ``"K6"`` general)."""
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
    """Why each case holds the class or does not (:attr:`.class_plan.ClassPlan.reasons`)."""

    def record(self) -> dict:
        """The row's axes and each case's reason as a JSON-compliant record (``routes.row``; the laws are
        ``routes.laws``)."""
        return {name: value for name, value in asdict(self).items() if name != "laws"}
