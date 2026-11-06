# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The route table: the row laws a solver's worlds solve, the laws each placement implements, and the placement case.

The row laws (:class:`Laws`) are read once from the physics options and the model, never from a route. Each
placement and schedule declares once, in :data:`CAPABILITIES`, the laws it implements, and :func:`capable` is the one
test of them that the owners' admissions read: an owner then tests only its own topology fit. The placement axis
(:class:`RouteTable`) gives the class to one case, decided once in the precedence :data:`CASES`: an owner reads the
case instead of the other owners' decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = ["CAPABILITIES", "CASES", "Laws", "RouteRow", "RouteTable", "capable"]


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
    iterations: bool
    """The position solve has a positive iteration budget."""
    cuda: bool
    """The device is CUDA, which runs the native bodies written for CUDA only (``cuda_native``)."""

    PLAIN = "no warm start or bilateral pre-elimination"

    @property
    def plain(self) -> bool:
        """The sweep policy the specialized placements assume (:attr:`PLAIN`)."""
        return not (self.warm_start or self.preelimination)

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
            iterations=options.pgs_iterations > 0,
            cuda=bool(model.device.is_cuda),
        )

    def record(self) -> dict:
        """The laws as a JSON-compliant record, with the derived ones."""
        return {**asdict(self), "plain": self.plain, "weighted": self.weighted}


# Each requirement a placement or schedule may hold: (test of the laws, what it requires).
_REQUIREMENTS = {
    "cuda": (lambda laws: laws.cuda, "a CUDA device (their native bodies are written for CUDA only)"),
    "plain": (lambda laws: laws.plain, Laws.PLAIN),
    "limits": (lambda laws: laws.limits, "joint-limit rows (enable_joint_limits)"),
    "no_mimic": (lambda laws: not laws.mimic, "no mimic rows"),
    "iterations": (lambda laws: laws.iterations, "a positive iteration budget"),
    "no_preelimination": (lambda laws: not laws.preelimination, "no bilateral pre-elimination"),
}

CAPABILITIES = {
    # The packed placement of a single tree: native sparse factors and their sweep.
    "packed": (
        "cuda",
        "iterations",
        "no_preelimination",
    ),
    # The paired factor's sweep in factor coordinates (one articulation and one free body per world).
    "paired_factor": ("cuda", "no_preelimination", "iterations"),
    # Its runtime claim: independent articulation and free-body components per world.
    "components": ("cuda", "no_preelimination", "no_mimic"),
    # The Jacobi tiers of dense trees beside a free body (case K4), whose sweep budget a build fixes.
    "tiers": ("cuda", "iterations"),
    # The keyed in-kernel row producer of the tiers.
    "keyed_rows": ("cuda", "plain", "limits", "iterations", "no_mimic"),
    # The colored mapping of the free-body rows' contact units.
    "colored": ("cuda",),
}
"""The laws each placement or schedule implements, by owner: what its admission reads through :func:`capable`."""

_LABELS = {
    "packed": "Native sparse factors need",
    "paired_factor": "The factor-coordinate sweep needs",
    "components": "Components need",
    "tiers": "The Jacobi tiers need",
    "keyed_rows": "Keyed rows need",
    "colored": "Colored units need",
}


def capable(owner: str, laws: Laws) -> str | None:
    """Why ``owner`` does not implement the solver's ``laws``, or ``None`` when it does (:data:`CAPABILITIES`)."""
    for requirement in CAPABILITIES[owner]:
        holds, phrase = _REQUIREMENTS[requirement]
        if not holds(laws):
            return f"{_LABELS[owner]} {phrase}."
    return None


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
    The test pin ``placement="general"`` keeps every class general.
    """

    def __init__(self, pins):
        self.case = "K6"
        """The case holding the class: a key of :data:`CASES`, or ``"K6"`` (general) while none claims it."""
        self.reasons: dict[str, str | None] = {}
        """Why each asked case holds the class or does not, written by its owner."""
        self._general = pins.get("placement") == "general"
        self._asked = -1

    def taken(self, case: str) -> str | None:
        """Why ``case`` may not claim the class (the general pin, or an earlier case holds it), or ``None``."""
        order = list(CASES).index(case)
        if order < self._asked:
            raise RuntimeError(f"Case {case} is asked after a later case; owners are asked in CASES order.")
        self._asked = order
        if self._general:
            return "A test pin keeps the general placement (placement='general')."
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
