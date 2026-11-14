# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental coupled-solver framework.

.. experimental::
"""

import importlib

from .interface import CouplingInterface

__all__ = [
    "CouplingInterface",
    "ModelView",
    "SolverCoupled",
    "SolverCoupledADMM",
    "SolverCoupledProxy",
]

# Every solver implements CouplingInterface and imports this package for it; the coupled solvers load on first use.
_MODULES = {
    "ModelView": "model_view",
    "SolverCoupled": "solver_coupled",
    "SolverCoupledADMM": "solver_coupled_admm",
    "SolverCoupledProxy": "solver_coupled_proxy",
}


def __getattr__(name: str):
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
