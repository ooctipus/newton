# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Metadata of the per-slot arrays of actuators and their components."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..sim.model import Model

SLOT_FREQUENCY = "actuator_slot"
"""Frequency of the arrays with one row per slot of an :class:`~newton.actuators.Actuator`."""


def slot_attribute_specs(
    parameters: Iterable[str], references: Mapping[str, Model.AttributeFrequency] | None = None
) -> dict[str, Model.AttributeSpec]:
    """Return the specs of per-slot arrays: ``parameters`` hold values, ``references`` index the given domains."""
    from ..sim.model import Model  # noqa: PLC0415

    specs = {name: Model.AttributeSpec(SLOT_FREQUENCY) for name in parameters}
    specs.update(
        (name, Model.AttributeSpec(SLOT_FREQUENCY, references=domain)) for name, domain in (references or {}).items()
    )
    return specs
