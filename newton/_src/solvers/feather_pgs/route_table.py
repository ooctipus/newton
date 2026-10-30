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

    contact: str
    """``"patch"`` (persistent friction patches, ``friction_anchor_beta > 0``) or ``"point"`` friction."""
    regularized: bool
    """Contact rows step toward the hard law with a float32 weight below one (``pgs_contact_regularization``)."""
    torsion: bool
    """Contact torsion gives every friction contact a spin row after its pair (``contact_torsion``)."""
    compliance: bool
    """Material contacts take the row law's compliance (``contact_compliance``)."""
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
    cuda: bool
    """The device is CUDA, which runs the native bodies written for CUDA only (``cuda_native``)."""

    PLAIN = "no warm start, debug rows or bilateral pre-elimination"

    @property
    def plain(self) -> bool:
        """The sweep policy the specialized placements assume (:attr:`PLAIN`)."""
        return not (self.warm_start or self.debug_rows or self.preelimination)

    @property
    def weighted(self) -> bool:
        """Contact rows step by their weights: the relative regularization's, or a material contact's from its
        compliance (every placement's sweep reads them)."""
        return self.regularized or self.compliance

    @property
    def agnostic(self) -> bool:
        """The contact rows follow the point or the patch row law, which every row writer implements (no torsion
        row; a material normal is the row law's, with its compliance)."""
        return not self.torsion

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
            cuda=bool(model.device.is_cuda),
        )

    def record(self) -> dict:
        """The laws as a JSON-compliant record, with the derived ones."""
        return {**asdict(self), "plain": self.plain, "agnostic": self.agnostic, "weighted": self.weighted}


# Each requirement a placement or schedule may hold: (test of the laws, what it requires).
_REQUIREMENTS = {
    "cuda": (lambda laws: laws.cuda, "a CUDA device (their native bodies are written for CUDA only)"),
    "plain": (lambda laws: laws.plain, Laws.PLAIN),
    "agnostic": (lambda laws: laws.agnostic, "the point or patch row law (no torsion)"),
    "limits": (lambda laws: laws.limits, "joint-limit rows (enable_joint_limits)"),
    "no_velocity_limits": (lambda laws: not laws.velocity_limits, "no joint velocity-limit rows"),
    "no_torsion": (lambda laws: not laws.torsion, "no torsion row"),
    "no_mimic": (lambda laws: not laws.mimic, "no mimic rows"),
    "no_connect": (lambda laws: not laws.connect, "no connect rows"),
    "no_loops": (lambda laws: not laws.loops, "no loop-closing joints"),
    "iterations": (lambda laws: laws.iterations, "a positive iteration budget"),
    "no_debug_rows": (lambda laws: not laws.debug_rows, "no debug row snapshots"),
    "no_preelimination": (lambda laws: not laws.preelimination, "no bilateral pre-elimination"),
}

CAPABILITIES = {
    # The held tree's evaluation of the dynamics, on either device (its topology fit is the held groups' own proof).
    "held": ("no_connect", "no_loops", "no_debug_rows"),
    # The packed placement of a single tree: native sparse factors and their sweep.
    "packed": (
        "cuda",
        "iterations",
        "no_debug_rows",
        "no_preelimination",
        "no_torsion",
        "no_connect",
        "no_loops",
    ),
    # On-chip blocks of a world's rows (local owners), claimed per world at run time.
    "local": ("cuda", "plain", "no_velocity_limits"),
    # The local owners' row producer: limit prefix and contact packets built in place.
    "row_packets": ("limits",),
    # The paired factor's sweep in factor coordinates (one articulation and one free body per world).
    "paired_factor": ("cuda", "plain", "iterations", "no_velocity_limits"),
    # Its runtime claim: independent articulation and free-body components per world.
    "components": ("cuda", "plain", "no_mimic", "no_connect"),
    # The Jacobi tiers of dense trees beside a free body (case K4).
    "tiers": ("cuda",),
    # The keyed in-kernel row producer of the tiers.
    "keyed_rows": ("cuda", "plain", "limits", "iterations", "no_loops", "no_mimic", "no_connect", "no_torsion"),
    # The compact boundary's row producer of a star beside a six-coordinate arm.
    "compact": ("agnostic",),
    # The colored mapping of the free-body rows' contact units.
    "colored": ("cuda", "no_torsion"),
}
"""The laws each placement or schedule implements, by owner: what its admission reads through :func:`capable`."""

_LABELS = {
    "held": "The held tree needs",
    "packed": "Native sparse factors need",
    "local": "Local owners need",
    "row_packets": "Local packets need",
    "paired_factor": "The factor-coordinate sweep needs",
    "components": "Components need",
    "tiers": "The Jacobi tiers need",
    "keyed_rows": "Keyed rows need",
    "compact": "Compact contacts need",
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
    "K3": "the local owners of a small tree beside a body",
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
    once they have decided, :meth:`finish` sets :attr:`row`, the route the step plan and the record read. The test
    pin ``placement="general"`` keeps every class general.
    """

    def __init__(self, pins):
        self.case = "K6"
        """The case holding the class: a key of :data:`CASES`, or ``"K6"`` (general) while none claims it."""
        self.reasons: dict[str, str | None] = {}
        """Why each asked case holds the class or does not, written by its owner."""
        self.row: RouteRow | None = None
        """The route of the class, once :meth:`finish` has set it."""
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
        """Set the route of the solver's class (:attr:`row`) from the laws, the decided case and its reasons, and
        the other ``axes`` (:class:`RouteRow`'s fields), and return it."""
        self.row = RouteRow(laws=laws, case=self.case, reasons=dict(self.reasons), **axes)
        return self.row


@dataclass(frozen=True)
class RouteRow:
    """The solver's route: the row of the route table its class takes, fixed once every owner has decided.

    The step plan (:meth:`.step_plan.StepPlan.build`) and the route record read the row, not the owners' state.
    """

    laws: Laws
    """Axis 2: the row laws."""
    evaluator: str
    """Axis 1: ``"held"`` (the held tree evaluates the dynamics) or ``"ordinary"``."""
    case: str
    """Axis 3: the placement case (:data:`CASES`, or ``"K6"`` general)."""
    producer: str
    """Axis 3: the producer of the contact rows (:func:`.contact_row_plan.contact_row_plan`)."""
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
