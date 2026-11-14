# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# ==================================================================================
# core
# ==================================================================================
from ._src.core import (
    MAXVAL,
    Axis,
    AxisType,
)
from ._src.exceptions import NewtonDeprecationWarning, NewtonWarning
from ._version import __version__

__all__ = [
    "MAXVAL",
    "Axis",
    "AxisType",
    "NewtonDeprecationWarning",
    "NewtonWarning",
    "__version__",
]

# ==================================================================================
# geometry
# ==================================================================================
from ._src.geometry import (
    SDF,
    Gaussian,
    GeoType,
    Heightfield,
    Mesh,
    ParticleFlags,
    ShapeFlags,
    TetMesh,
    intersect_ray,
)

__all__ += [
    "SDF",
    "Gaussian",
    "GeoType",
    "Heightfield",
    "Mesh",
    "ParticleFlags",
    "ShapeFlags",
    "TetMesh",
    "intersect_ray",
]

# ==================================================================================
# sim
# ==================================================================================
from ._src.sim import (  # noqa: E402
    BodyFlags,
    CollisionPipeline,
    CollisionSubstepScheduler,
    Contacts,
    Control,
    JointTargetMode,
    JointType,
    Model,
    ModelBuilder,
    ModelFlags,
    Rod,
    State,
    StateFlags,
    TendonGuide,
    TendonGuideFlags,
    TendonGuideType,
    eval_fk,
    eval_ik,
    eval_inverse_dynamics_force,
    eval_inverse_dynamics_passive,
    eval_jacobian,
    eval_mass_matrix,
    eval_mimic,
    eval_rigid_contact_kinematics,
)

__all__ += [
    "BodyFlags",
    "CollisionPipeline",
    "CollisionSubstepScheduler",
    "Contacts",
    "Control",
    "JointTargetMode",
    "JointType",
    "Model",
    "ModelBuilder",
    "ModelFlags",
    "Rod",
    "State",
    "StateFlags",
    "TendonGuide",
    "TendonGuideFlags",
    "TendonGuideType",
    "eval_fk",
    "eval_ik",
    "eval_inverse_dynamics_force",
    "eval_inverse_dynamics_passive",
    "eval_jacobian",
    "eval_mass_matrix",
    "eval_mimic",
    "eval_rigid_contact_kinematics",
]

# ==================================================================================
# submodule APIs
# ==================================================================================
from . import (  # noqa: E402
    actuators,
    config,
    controllers,
    geometry,
    ik,
    math,
    selection,
    sensors,
    solvers,
    usd,
    utils,
)

__all__ += [
    "actuators",
    "config",
    "controllers",
    "geometry",
    "ik",
    "math",
    "selection",
    "sensors",
    "solvers",
    "usd",
    "utils",
    "viewer",
]



def __getattr__(name: str):
    # The viewers (OpenGL, USD, Rerun, Viser and their kernels) load on first use: about 0.3 s that a headless
    # simulation never needs.
    if name == "viewer":
        import importlib  # noqa: PLC0415

        return importlib.import_module(".viewer", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from ._src.diagnostics import install_console_fallback as _install_console_fallback  # noqa: E402

_install_console_fallback()
del _install_console_fallback
